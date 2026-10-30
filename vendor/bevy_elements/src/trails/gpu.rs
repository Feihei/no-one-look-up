//! Render-world side of the trails: pipelines, the bind groups, and one compute pass a frame
//! (`elements_trails`: stamp, merge, block occupancy, relax, and on a readback tick the readback
//! cells) after the terrain's pack kernels, so the maps it reads are this frame's.
//!
//! The render world remembers the window's origin of the last frame it ran, which is what the
//! field holds: the main world cannot know it under pipelined rendering.

use std::sync::atomic::Ordering;

use bevy::asset::load_embedded_asset;
use bevy::core_pipeline::schedule::camera_driver;
use bevy::prelude::*;
use bevy::render::diagnostic::RecordDiagnostics;
use bevy::render::extract_resource::{ExtractResource, ExtractResourcePlugin};
use bevy::render::render_asset::RenderAssets;
use bevy::render::render_resource::binding_types::{
    storage_buffer_read_only_sized, storage_buffer_sized, texture_storage_2d, uniform_buffer,
};
use bevy::render::render_resource::{
    BindGroup, BindGroupEntries, BindGroupLayoutDescriptor, BindGroupLayoutEntries,
    ComputePassDescriptor, PipelineCache, ShaderStages, StorageBuffer, StorageTextureAccess,
    TextureFormat, UniformBuffer,
};
use bevy::render::renderer::{RenderContext, RenderDevice, RenderGraph, RenderQueue};
use bevy::render::storage::{GpuShaderBuffer, ShaderBuffer};
use bevy::render::texture::GpuImage;
use bevy::render::{Render, RenderApp, RenderStartup, RenderSystems};

use super::model::ImprintGpu;
use super::{TrailGpu, TrailParams};
use crate::gpu::kernel::{Kernel, groups, queue_kernel, ready};
use crate::terrain::TerrainDispatch;

impl ExtractResource<RenderApp> for TrailGpu {
    type Source = TrailGpu;

    fn extract_resource(source: &Self::Source) -> Self {
        source.clone()
    }
}

#[derive(Resource)]
struct TrailPipelines {
    stamp: Kernel,
    merge: Kernel,
    blocks: Kernel,
    relax: Kernel,
    readback: Kernel,
}

#[derive(Resource, Default)]
struct TrailUniforms {
    params: UniformBuffer<TrailParams>,
    imprints: StorageBuffer<Vec<ImprintGpu>>,
}

#[derive(Resource)]
struct TrailBindGroup {
    main: BindGroup,
    readback: BindGroup,
}

/// What the field holds: the state buffer it was last run on and that run's window origin.
#[derive(Resource, Default)]
struct TrailMemory {
    buffer: Option<AssetId<ShaderBuffer>>,
    origin: Option<IVec2>,
    /// This frame's uniform, as prepared (with `prev_origin` and the clear flag filled in).
    params: Option<TrailParams>,
}

pub(super) fn plugin(app: &mut App) {
    app.add_plugins(ExtractResourcePlugin::<TrailGpu>::default());
    let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
        return;
    };
    render_app
        .init_resource::<TrailUniforms>()
        .init_resource::<TrailMemory>()
        .add_systems(RenderStartup, init_pipelines)
        .add_systems(Render, prepare.in_set(RenderSystems::PrepareBindGroups))
        .add_systems(
            RenderGraph,
            dispatch.after(TerrainDispatch).before(camera_driver),
        );
}

fn layout() -> BindGroupLayoutDescriptor {
    BindGroupLayoutDescriptor::new(
        "elements_trails",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                uniform_buffer::<TrailParams>(false),
                storage_buffer_sized(false, None),
                storage_buffer_sized(false, None),
                storage_buffer_sized(false, None),
                storage_buffer_read_only_sized(false, None),
                storage_buffer_read_only_sized(false, None),
                storage_buffer_read_only_sized(false, None),
                texture_storage_2d(TextureFormat::Rgba16Float, StorageTextureAccess::WriteOnly),
                storage_buffer_sized(false, None),
            ),
        ),
    )
}

/// The readback kernel's own layout (the uniform, the state, the readback cells), so the other
/// four keep seven storage buffers.
fn readback_layout() -> BindGroupLayoutDescriptor {
    BindGroupLayoutDescriptor::new(
        "elements_trails_readback",
        &BindGroupLayoutEntries::with_indices(
            ShaderStages::COMPUTE,
            (
                (0, uniform_buffer::<TrailParams>(false)),
                (1, storage_buffer_sized(false, None)),
                (9, storage_buffer_sized(false, None)),
            ),
        ),
    )
}

fn init_pipelines(mut commands: Commands, assets: Res<AssetServer>, cache: Res<PipelineCache>) {
    let shader = load_embedded_asset!(assets.as_ref(), "trails.wesl");
    let kernel = |entry| queue_kernel(&cache, shader.clone(), entry, layout(), Vec::new());
    commands.insert_resource(TrailPipelines {
        stamp: kernel("trail_stamp"),
        merge: kernel("trail_merge"),
        blocks: kernel("trail_blocks"),
        relax: kernel("trail_relax"),
        readback: queue_kernel(
            &cache,
            shader.clone(),
            "trail_readback",
            readback_layout(),
            Vec::new(),
        ),
    });
}

#[allow(clippy::too_many_arguments)]
fn prepare(
    mut commands: Commands,
    gpu: Option<Res<TrailGpu>>,
    pipelines: Option<Res<TrailPipelines>>,
    mut uniforms: ResMut<TrailUniforms>,
    mut memory: ResMut<TrailMemory>,
    buffers: Res<RenderAssets<GpuShaderBuffer>>,
    images: Res<RenderAssets<GpuImage>>,
    cache: Res<PipelineCache>,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
) {
    commands.remove_resource::<TrailBindGroup>();
    memory.params = None;
    let (Some(gpu), Some(pipelines)) = (gpu, pipelines) else {
        return;
    };
    let get = |h: &Handle<ShaderBuffer>| buffers.get(h).map(|b| &b.buffer);
    let (
        Some(a),
        Some(b),
        Some(stamp),
        Some(cells),
        Some(cover_cells),
        Some(field),
        Some(blocks),
        Some(readback),
    ) = (
        get(&gpu.state_a),
        get(&gpu.state_b),
        get(&gpu.stamp),
        get(&gpu.cells),
        get(&gpu.cover),
        images.get(&gpu.field),
        get(&gpu.blocks),
        get(&gpu.readback),
    )
    else {
        gpu.ready.store(false, Ordering::Relaxed);
        return;
    };
    let compiled = ready(&cache, &pipelines.stamp).is_some()
        && ready(&cache, &pipelines.merge).is_some()
        && ready(&cache, &pipelines.blocks).is_some()
        && ready(&cache, &pipelines.relax).is_some()
        && ready(&cache, &pipelines.readback).is_some();
    gpu.ready.store(compiled, Ordering::Relaxed);
    if !compiled || !gpu.run {
        return;
    }
    // A new state buffer holds nothing: clear the whole window on its first run.
    if memory.buffer != Some(gpu.state_a.id()) {
        memory.buffer = Some(gpu.state_a.id());
        memory.origin = None;
    }
    let mut params = gpu.params;
    params.prev_origin = memory.origin.unwrap_or(params.origin);
    params.counts.z = u32::from(memory.origin.is_none());
    memory.params = Some(params);

    uniforms.params.set(params);
    uniforms.params.write_buffer(&device, &queue);
    uniforms.imprints.set(gpu.imprints.clone());
    uniforms.imprints.write_buffer(&device, &queue);
    let Some(imprints) = uniforms.imprints.binding() else {
        return;
    };
    let main = device.create_bind_group(
        "elements_trails",
        &cache.get_bind_group_layout(&pipelines.stamp.layout),
        &BindGroupEntries::sequential((
            &uniforms.params,
            a.as_entire_buffer_binding(),
            b.as_entire_buffer_binding(),
            stamp.as_entire_buffer_binding(),
            imprints,
            cells.as_entire_buffer_binding(),
            cover_cells.as_entire_buffer_binding(),
            &field.texture_view,
            blocks.as_entire_buffer_binding(),
        )),
    );
    let readback = device.create_bind_group(
        "elements_trails_readback",
        &cache.get_bind_group_layout(&pipelines.readback.layout),
        &BindGroupEntries::with_indices((
            (0, &uniforms.params),
            (1, a.as_entire_buffer_binding()),
            (9, readback.as_entire_buffer_binding()),
        )),
    );
    commands.insert_resource(TrailBindGroup { main, readback });
}

fn dispatch(
    mut ctx: RenderContext,
    cache: Res<PipelineCache>,
    gpu: Option<Res<TrailGpu>>,
    pipelines: Option<Res<TrailPipelines>>,
    group: Option<Res<TrailBindGroup>>,
    mut memory: ResMut<TrailMemory>,
) {
    let (Some(gpu), Some(pipelines), Some(group), Some(params)) =
        (gpu, pipelines, group, memory.params)
    else {
        return;
    };
    let (Some(stamp), Some(merge), Some(blocks), Some(relax), Some(readback)) = (
        ready(&cache, &pipelines.stamp),
        ready(&cache, &pipelines.merge),
        ready(&cache, &pipelines.blocks),
        ready(&cache, &pipelines.relax),
        ready(&cache, &pipelines.readback),
    ) else {
        return;
    };
    let size = params.counts.x;
    let diagnostics = ctx.diagnostic_recorder();
    let diagnostics = diagnostics.as_deref();
    let mut pass = ctx
        .command_encoder()
        .begin_compute_pass(&ComputePassDescriptor {
            label: Some("elements_trails"),
            ..default()
        });
    let span = diagnostics.pass_span(&mut pass, "elements_trails");
    pass.set_bind_group(0, &group.main, &[]);
    if params.counts.y > 0 && params.counts.w > 0 {
        let tiles = groups(params.counts.w, 16);
        pass.set_pipeline(stamp);
        pass.dispatch_workgroups(tiles, tiles, params.counts.y);
    }
    pass.set_pipeline(merge);
    pass.dispatch_workgroups(groups(size, 8), groups(size, 8), 1);
    pass.set_pipeline(blocks);
    pass.dispatch_workgroups(groups(size / 8, 8), groups(size / 8, 8), 1);
    pass.set_pipeline(relax);
    pass.dispatch_workgroups(groups(size, 8), groups(size, 8), 1);
    span.end(&mut pass);
    drop(pass);
    // On a readback tick, the deepest point per cell for `Environment::trail_depth`, timed on its
    // own (`elements_trails_readback`).
    if gpu.read {
        let cells = size / params.mode.y.max(1);
        let mut pass = ctx
            .command_encoder()
            .begin_compute_pass(&ComputePassDescriptor {
                label: Some("elements_trails_readback"),
                ..default()
            });
        let span = diagnostics.pass_span(&mut pass, "elements_trails_readback");
        pass.set_bind_group(0, &group.readback, &[]);
        pass.set_pipeline(readback);
        pass.dispatch_workgroups(groups(cells, 8), groups(cells, 8), 1);
        span.end(&mut pass);
    }
    memory.origin = Some(params.origin);
    gpu.executed.fetch_add(1, Ordering::Relaxed);
}
