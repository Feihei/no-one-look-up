//! Render-world side of the wind field: pipelines, bind groups and the per-frame dispatch.

use bevy::asset::load_embedded_asset;
use bevy::core_pipeline::schedule::camera_driver;
use bevy::prelude::*;
use bevy::render::diagnostic::RecordDiagnostics;
use bevy::render::extract_resource::{ExtractResource, ExtractResourcePlugin};
use bevy::render::render_asset::RenderAssets;
use bevy::render::render_resource::binding_types::{
    storage_buffer_read_only_sized, storage_buffer_sized, texture_storage_3d, uniform_buffer,
};
use bevy::render::render_resource::{
    BindGroup, BindGroupEntries, BindGroupLayoutDescriptor, BindGroupLayoutEntries,
    ComputePassDescriptor, PipelineCache, ShaderStages, StorageTextureAccess, TextureFormat,
    UniformBuffer,
};
use bevy::render::renderer::{RenderContext, RenderDevice, RenderGraph, RenderQueue};
use bevy::render::storage::GpuShaderBuffer;
use bevy::render::texture::GpuImage;
use bevy::render::{Render, RenderApp, RenderStartup, RenderSystems};

use super::{WindGpu, WindParams};
use crate::gpu::kernel::{Kernel, WG3, groups, queue_kernel, ready};

impl ExtractResource<RenderApp> for WindGpu {
    type Source = WindGpu;

    fn extract_resource(source: &Self::Source) -> Self {
        source.clone()
    }
}

/// Kernel entry points in `sim.wesl`, in dispatch order.
const FORCING: usize = 0;
const ADVECT: usize = 1;
const DIVERGENCE: usize = 2;
const JACOBI: usize = 3;
const PROJECT: usize = 4;
const PUBLISH: usize = 5;
const ENTRIES: [&str; 6] = [
    "forcing",
    "advect",
    "divergence",
    "jacobi",
    "project",
    "publish",
];

#[derive(Resource)]
struct WindPipelines {
    kernels: Vec<Kernel>,
}

#[derive(Resource, Default)]
struct WindUniform(UniformBuffer<WindParams>);

/// `[velocity direction][pressure direction]`, direction 0 = A→B, 1 = B→A.
#[derive(Resource)]
struct WindBindGroups([[BindGroup; 2]; 2]);

pub(super) fn plugin(app: &mut App) {
    app.add_plugins(ExtractResourcePlugin::<WindGpu>::default());
    let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
        return;
    };
    render_app
        .init_resource::<WindUniform>()
        .add_systems(RenderStartup, init_pipelines)
        .add_systems(Render, prepare.in_set(RenderSystems::PrepareBindGroups))
        .add_systems(
            RenderGraph,
            dispatch
                .after(crate::terrain::TerrainDispatch)
                .before(camera_driver),
        );
}

fn layout() -> BindGroupLayoutDescriptor {
    BindGroupLayoutDescriptor::new(
        "elements_wind",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                uniform_buffer::<WindParams>(false),
                storage_buffer_read_only_sized(false, None),
                storage_buffer_sized(false, None),
                storage_buffer_sized(false, None),
                storage_buffer_sized(false, None),
                storage_buffer_read_only_sized(false, None),
                storage_buffer_sized(false, None),
                texture_storage_3d(TextureFormat::Rgba16Float, StorageTextureAccess::WriteOnly),
                storage_buffer_read_only_sized(false, None),
            ),
        ),
    )
}

fn init_pipelines(mut commands: Commands, assets: Res<AssetServer>, cache: Res<PipelineCache>) {
    let shader = load_embedded_asset!(assets.as_ref(), "sim.wesl");
    let layout = layout();
    let kernels = ENTRIES
        .iter()
        .map(|entry| queue_kernel(&cache, shader.clone(), entry, layout.clone(), Vec::new()))
        .collect();
    commands.insert_resource(WindPipelines { kernels });
}

#[allow(clippy::too_many_arguments)]
fn prepare(
    mut commands: Commands,
    grid: Option<Res<WindGpu>>,
    pipelines: Option<Res<WindPipelines>>,
    mut uniform: ResMut<WindUniform>,
    buffers: Res<RenderAssets<GpuShaderBuffer>>,
    images: Res<RenderAssets<GpuImage>>,
    cache: Res<PipelineCache>,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
) {
    let (Some(grid), Some(pipelines)) = (grid, pipelines) else {
        return;
    };
    let Some(kernel) = pipelines.kernels.first() else {
        return;
    };
    let get = |h: &Handle<bevy::render::storage::ShaderBuffer>| buffers.get(h).map(|b| &b.buffer);
    let (
        Some(va),
        Some(vb),
        Some(pa),
        Some(pb),
        Some(solid),
        Some(div),
        Some(field),
        Some(terrain),
    ) = (
        get(&grid.velocity[0]),
        get(&grid.velocity[1]),
        get(&grid.pressure[0]),
        get(&grid.pressure[1]),
        get(&grid.solid),
        get(&grid.divergence),
        images.get(&grid.field),
        get(&grid.terrain),
    )
    else {
        commands.remove_resource::<WindBindGroups>();
        return;
    };
    let mut params = grid.params;
    // Until the kernels have run once the buffers hold nothing: keep initialising every cell to
    // the forcing target, whichever frame the pipelines first become ready on.
    if !grid.live.load(std::sync::atomic::Ordering::Relaxed) {
        params.origin.w = 1;
    }
    uniform.0.set(params);
    uniform.0.write_buffer(&device, &queue);
    let layout = cache.get_bind_group_layout(&kernel.layout);
    let make = |v_in: &bevy::render::render_resource::Buffer,
                v_out: &bevy::render::render_resource::Buffer,
                p_in: &bevy::render::render_resource::Buffer,
                p_out: &bevy::render::render_resource::Buffer| {
        device.create_bind_group(
            "elements_wind",
            &layout,
            &BindGroupEntries::sequential((
                &uniform.0,
                v_in.as_entire_buffer_binding(),
                v_out.as_entire_buffer_binding(),
                solid.as_entire_buffer_binding(),
                div.as_entire_buffer_binding(),
                p_in.as_entire_buffer_binding(),
                p_out.as_entire_buffer_binding(),
                &field.texture_view,
                terrain.as_entire_buffer_binding(),
            )),
        )
    };
    commands.insert_resource(WindBindGroups([
        [make(va, vb, pa, pb), make(va, vb, pb, pa)],
        [make(vb, va, pa, pb), make(vb, va, pb, pa)],
    ]));
}

fn dispatch(
    mut ctx: RenderContext,
    cache: Res<PipelineCache>,
    grid: Option<Res<WindGpu>>,
    pipelines: Option<Res<WindPipelines>>,
    bind_groups: Option<Res<WindBindGroups>>,
) {
    let (Some(grid), Some(pipelines), Some(bind_groups)) = (grid, pipelines, bind_groups) else {
        return;
    };
    if !grid.active {
        return;
    }
    let Some(pipes) = pipelines
        .kernels
        .iter()
        .map(|k| ready(&cache, k))
        .collect::<Option<Vec<_>>>()
    else {
        return;
    };
    let n = grid.size;
    let (x, y, z) = (groups(n.x, WG3), groups(n.y, WG3), groups(n.z, WG3));
    let diagnostics = ctx.diagnostic_recorder();
    let diagnostics = diagnostics.as_deref();
    let mut pass = ctx
        .command_encoder()
        .begin_compute_pass(&ComputePassDescriptor {
            label: Some("elements_wind"),
            ..default()
        });
    let span = diagnostics.pass_span(&mut pass, "elements_wind");
    let bg = &bind_groups.0;
    let mut run = |kernel: usize, group: &BindGroup| {
        pass.set_pipeline(pipes[kernel]);
        pass.set_bind_group(0, group, &[]);
        pass.dispatch_workgroups(x, y, z);
    };
    run(FORCING, &bg[0][0]);
    run(ADVECT, &bg[1][0]);
    run(DIVERGENCE, &bg[0][1]);
    for i in 0..grid.iterations {
        run(JACOBI, &bg[0][(i % 2) as usize]);
    }
    run(PROJECT, &bg[0][0]);
    run(PUBLISH, &bg[1][0]);
    span.end(&mut pass);
    grid.live.store(true, std::sync::atomic::Ordering::Relaxed);
}

#[cfg(test)]
mod tests {
    use bevy::render::render_resource::{BindingType, BufferBindingType};

    /// Terrain validation 9: the wind layout stays within WebGPU's default of 8 storage buffers
    /// per stage (7 with the terrain).
    #[test]
    fn the_wind_layout_binds_seven_storage_buffers() {
        let storage = super::layout()
            .entries
            .iter()
            .filter(|e| {
                matches!(
                    e.ty,
                    BindingType::Buffer {
                        ty: BufferBindingType::Storage { .. },
                        ..
                    }
                )
            })
            .count();
        assert_eq!(storage, 7);
    }
}
