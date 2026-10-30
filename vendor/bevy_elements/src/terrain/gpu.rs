//! Render-world side of the terrain: the generator and the per-change pack, bounds and readback
//! kernels.
//!
//! The kernels run in one compute pass, on the first frame with a ready pipeline (generator,
//! then everything below) and afterwards on any frame in which something bumped
//! `TerrainGpu::changes` (hydrology and erosion do, once per frame they ran). Each run bumps
//! `TerrainState::version`, which the main world's readback compares against.

use std::sync::atomic::Ordering;

use bevy::asset::load_embedded_asset;
use bevy::core_pipeline::schedule::camera_driver;
use bevy::prelude::*;
use bevy::render::diagnostic::RecordDiagnostics;
use bevy::render::extract_resource::{ExtractResource, ExtractResourcePlugin};
use bevy::render::render_asset::RenderAssets;
use bevy::render::render_resource::binding_types::{
    storage_buffer_read_only_sized, storage_buffer_sized, texture_2d, uniform_buffer,
};
use bevy::render::render_resource::{
    BindGroup, BindGroupEntries, BindGroupLayoutDescriptor, BindGroupLayoutEntries,
    ComputePassDescriptor, PipelineCache, ShaderStages, TextureSampleType, UniformBuffer,
};
use bevy::render::renderer::{RenderContext, RenderDevice, RenderGraph, RenderQueue};
use bevy::render::storage::GpuShaderBuffer;
use bevy::render::texture::GpuImage;
use bevy::render::{Render, RenderApp, RenderStartup, RenderSystems};
use bevy::shader::ShaderDefVal;

use super::model::LEAF_CELLS;
use super::{GenParams, GenerateSource, HeightmapParams, TerrainGpu, TerrainParamsGpu};
use crate::gpu::kernel::{Kernel, groups, queue_kernel, ready};

impl ExtractResource<RenderApp> for TerrainGpu {
    type Source = TerrainGpu;

    fn extract_resource(source: &Self::Source) -> Self {
        source.clone()
    }
}

/// The terrain's kernels. Anything that reads `terrain_cells` in the same frame orders
/// itself `.after(TerrainDispatch)`; anything that writes the layers `.before` it.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct TerrainDispatch;

#[derive(Resource)]
struct TerrainPipelines {
    generate: Kernel,
    heightmap: [Kernel; 2],
    pack: Kernel,
    readback: Kernel,
    minmax: Kernel,
    flow: Kernel,
}

#[derive(Resource, Default)]
struct TerrainUniforms {
    generate: UniformBuffer<GenParams>,
    heightmap: UniformBuffer<HeightmapParams>,
    params: UniformBuffer<TerrainParamsGpu>,
}

#[derive(Resource)]
struct TerrainBindGroups {
    generate: Option<BindGroup>,
    main: BindGroup,
    flow: BindGroup,
}

pub(super) fn plugin(app: &mut App) {
    app.add_plugins(ExtractResourcePlugin::<TerrainGpu>::default());
    let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
        return;
    };
    render_app
        .init_resource::<TerrainUniforms>()
        .add_systems(RenderStartup, init_pipelines)
        .add_systems(Render, prepare.in_set(RenderSystems::PrepareBindGroups))
        .add_systems(
            RenderGraph,
            dispatch.in_set(TerrainDispatch).before(camera_driver),
        );
}

fn generate_layout() -> BindGroupLayoutDescriptor {
    BindGroupLayoutDescriptor::new(
        "elements_terrain_generate",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                uniform_buffer::<GenParams>(false),
                storage_buffer_sized(false, None),
                storage_buffer_sized(false, None),
            ),
        ),
    )
}

fn heightmap_layout(uint: bool) -> BindGroupLayoutDescriptor {
    let sample = if uint {
        TextureSampleType::Uint
    } else {
        TextureSampleType::Float { filterable: false }
    };
    BindGroupLayoutDescriptor::new(
        "elements_terrain_heightmap",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                uniform_buffer::<HeightmapParams>(false),
                storage_buffer_sized(false, None),
                storage_buffer_sized(false, None),
                texture_2d(sample),
            ),
        ),
    )
}

fn main_layout() -> BindGroupLayoutDescriptor {
    BindGroupLayoutDescriptor::new(
        "elements_terrain_pack",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                uniform_buffer::<TerrainParamsGpu>(false),
                storage_buffer_read_only_sized(false, None),
                storage_buffer_read_only_sized(false, None),
                storage_buffer_read_only_sized(false, None),
                storage_buffer_sized(false, None),
                storage_buffer_sized(false, None),
                storage_buffer_sized(false, None),
            ),
        ),
    )
}

fn flow_layout() -> BindGroupLayoutDescriptor {
    BindGroupLayoutDescriptor::new(
        "elements_terrain_flow",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                uniform_buffer::<TerrainParamsGpu>(false),
                storage_buffer_read_only_sized(false, None),
                storage_buffer_read_only_sized(false, None),
                storage_buffer_read_only_sized(false, None),
                storage_buffer_sized(false, None),
            ),
        ),
    )
}

fn init_pipelines(mut commands: Commands, assets: Res<AssetServer>, cache: Res<PipelineCache>) {
    let generate = load_embedded_asset!(assets.as_ref(), "generate.wesl");
    let heightmap = load_embedded_asset!(assets.as_ref(), "heightmap.wesl");
    let pack = load_embedded_asset!(assets.as_ref(), "pack.wesl");
    let flow = load_embedded_asset!(assets.as_ref(), "flow.wesl");
    let main = main_layout();
    let kernel = |shader: &Handle<Shader>, entry, layout: &BindGroupLayoutDescriptor| {
        queue_kernel(&cache, shader.clone(), entry, layout.clone(), Vec::new())
    };
    let heightmap_kernel = |uint: bool| {
        let defs = if uint {
            vec![ShaderDefVal::Bool("UINT_SOURCE".into(), true)]
        } else {
            Vec::new()
        };
        queue_kernel(
            &cache,
            heightmap.clone(),
            "terrain_heightmap",
            heightmap_layout(uint),
            defs,
        )
    };
    commands.insert_resource(TerrainPipelines {
        generate: kernel(&generate, "terrain_generate", &generate_layout()),
        heightmap: [heightmap_kernel(false), heightmap_kernel(true)],
        pack: kernel(&pack, "terrain_pack", &main),
        readback: kernel(&pack, "terrain_readback", &main),
        minmax: kernel(&pack, "terrain_minmax", &main),
        flow: kernel(&flow, "flow_pack", &flow_layout()),
    });
}

#[allow(clippy::too_many_arguments)]
fn prepare(
    mut commands: Commands,
    gpu: Option<Res<TerrainGpu>>,
    pipelines: Option<Res<TerrainPipelines>>,
    mut uniforms: ResMut<TerrainUniforms>,
    buffers: Res<RenderAssets<GpuShaderBuffer>>,
    images: Res<RenderAssets<GpuImage>>,
    cache: Res<PipelineCache>,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
) {
    let (Some(gpu), Some(pipelines)) = (gpu, pipelines) else {
        commands.remove_resource::<TerrainBindGroups>();
        return;
    };
    let generated = gpu.generated.load(Ordering::Relaxed);
    if generated && gpu.changes <= gpu.packed.load(Ordering::Relaxed) {
        commands.remove_resource::<TerrainBindGroups>();
        return;
    }
    let s = &gpu.state;
    let get = |h: &Handle<bevy::render::storage::ShaderBuffer>| buffers.get(h).map(|b| &b.buffer);
    let (
        Some(bedrock),
        Some(sediment),
        Some(water),
        Some(cells),
        Some(readback),
        Some(leaves),
        Some(velocity),
        Some(suspended),
        Some(flow_cells),
    ) = (
        get(&s.bedrock),
        get(&s.sediment),
        get(&s.water),
        get(&s.cells),
        get(&s.height_readback),
        get(&s.leaf_bounds),
        get(&s.velocity),
        get(&s.suspended),
        get(&s.flow_cells),
    )
    else {
        commands.remove_resource::<TerrainBindGroups>();
        return;
    };
    uniforms.params.set(gpu.params);
    uniforms.params.write_buffer(&device, &queue);
    let generate = if generated {
        None
    } else {
        match &gpu.source {
            GenerateSource::Procedural(params) => {
                uniforms.generate.set(*params);
                uniforms.generate.write_buffer(&device, &queue);
                Some(device.create_bind_group(
                    "elements_terrain_generate",
                    &cache.get_bind_group_layout(&pipelines.generate.layout),
                    &BindGroupEntries::sequential((
                        &uniforms.generate,
                        bedrock.as_entire_buffer_binding(),
                        sediment.as_entire_buffer_binding(),
                    )),
                ))
            }
            GenerateSource::Heightmap {
                image,
                uint,
                params,
            } => {
                let Some(image) = images.get(image) else {
                    commands.remove_resource::<TerrainBindGroups>();
                    return;
                };
                uniforms.heightmap.set(*params);
                uniforms.heightmap.write_buffer(&device, &queue);
                let kernel = &pipelines.heightmap[usize::from(*uint)];
                Some(device.create_bind_group(
                    "elements_terrain_heightmap",
                    &cache.get_bind_group_layout(&kernel.layout),
                    &BindGroupEntries::sequential((
                        &uniforms.heightmap,
                        bedrock.as_entire_buffer_binding(),
                        sediment.as_entire_buffer_binding(),
                        &image.texture_view,
                    )),
                ))
            }
        }
    };
    let main = device.create_bind_group(
        "elements_terrain_pack",
        &cache.get_bind_group_layout(&pipelines.pack.layout),
        &BindGroupEntries::sequential((
            &uniforms.params,
            bedrock.as_entire_buffer_binding(),
            sediment.as_entire_buffer_binding(),
            water.as_entire_buffer_binding(),
            cells.as_entire_buffer_binding(),
            readback.as_entire_buffer_binding(),
            leaves.as_entire_buffer_binding(),
        )),
    );
    let flow = device.create_bind_group(
        "elements_terrain_flow",
        &cache.get_bind_group_layout(&pipelines.flow.layout),
        &BindGroupEntries::sequential((
            &uniforms.params,
            velocity.as_entire_buffer_binding(),
            suspended.as_entire_buffer_binding(),
            water.as_entire_buffer_binding(),
            flow_cells.as_entire_buffer_binding(),
        )),
    );
    commands.insert_resource(TerrainBindGroups {
        generate,
        main,
        flow,
    });
}

fn dispatch(
    mut ctx: RenderContext,
    cache: Res<PipelineCache>,
    gpu: Option<Res<TerrainGpu>>,
    pipelines: Option<Res<TerrainPipelines>>,
    bind_groups: Option<Res<TerrainBindGroups>>,
) {
    let (Some(gpu), Some(pipelines), Some(bind_groups)) = (gpu, pipelines, bind_groups) else {
        return;
    };
    let generated = gpu.generated.load(Ordering::Relaxed);
    if generated && gpu.changes <= gpu.packed.load(Ordering::Relaxed) {
        return;
    }
    let generator = match &gpu.source {
        GenerateSource::Procedural(_) => &pipelines.generate,
        GenerateSource::Heightmap { uint, .. } => &pipelines.heightmap[usize::from(*uint)],
    };
    let (Some(generator), Some(pack), Some(readback), Some(minmax), Some(flow)) = (
        ready(&cache, generator),
        ready(&cache, &pipelines.pack),
        ready(&cache, &pipelines.readback),
        ready(&cache, &pipelines.minmax),
        ready(&cache, &pipelines.flow),
    ) else {
        return;
    };
    if !generated && bind_groups.generate.is_none() {
        return;
    }
    let grid = &gpu.state.grid;
    let (cx, cy) = (groups(grid.size.x, 8), groups(grid.size.y, 8));
    let rb = gpu.state.readback_size;
    let diagnostics = ctx.diagnostic_recorder();
    let diagnostics = diagnostics.as_deref();
    let mut pass = ctx
        .command_encoder()
        .begin_compute_pass(&ComputePassDescriptor {
            label: Some("elements_terrain"),
            ..default()
        });
    let span = diagnostics.pass_span(&mut pass, "elements_terrain");
    if let Some(group) = bind_groups.generate.as_ref().filter(|_| !generated) {
        pass.set_pipeline(generator);
        pass.set_bind_group(0, group, &[]);
        pass.dispatch_workgroups(cx, cy, 1);
    }
    pass.set_pipeline(pack);
    pass.set_bind_group(0, &bind_groups.main, &[]);
    pass.dispatch_workgroups(cx, cy, 1);
    pass.set_pipeline(flow);
    pass.set_bind_group(0, &bind_groups.flow, &[]);
    pass.dispatch_workgroups(cx, cy, 1);
    pass.set_pipeline(minmax);
    pass.set_bind_group(0, &bind_groups.main, &[]);
    pass.dispatch_workgroups(grid.size.x / LEAF_CELLS, grid.size.y / LEAF_CELLS, 1);
    pass.set_pipeline(readback);
    pass.set_bind_group(0, &bind_groups.main, &[]);
    pass.dispatch_workgroups(groups(rb.x, 8), groups(rb.y, 8), 1);
    span.end(&mut pass);
    drop(pass);
    gpu.packed.store(gpu.changes, Ordering::Relaxed);
    gpu.generated.store(true, Ordering::Relaxed);
    gpu.state.version.fetch_add(1, Ordering::Relaxed);
}
