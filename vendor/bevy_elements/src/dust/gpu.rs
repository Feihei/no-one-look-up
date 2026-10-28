//! Render-world side of the aeolian transport: the three kernels, and the dispatch of hydrology's
//! thermal pair for the avalanching.
//!
//! Four compute passes a step, each with its own timestamp span: `elements_aeolian_capacity`,
//! `elements_aeolian_flux`, `elements_aeolian_move` and `elements_aeolian_avalanche`. They run
//! after hydrology's pass and before the terrain's pack, so the pack publishes the new bed in the
//! same frame.

use std::sync::atomic::Ordering;

use bevy::asset::load_embedded_asset;
use bevy::core_pipeline::schedule::camera_driver;
use bevy::prelude::*;
use bevy::render::diagnostic::RecordDiagnostics;
use bevy::render::extract_resource::{ExtractResource, ExtractResourcePlugin};
use bevy::render::render_asset::RenderAssets;
use bevy::render::render_resource::binding_types::{storage_buffer_sized, uniform_buffer};
use bevy::render::render_resource::{
    BindGroup, BindGroupEntries, BindGroupLayoutDescriptor, BindGroupLayoutEntries,
    ComputePassDescriptor, PipelineCache, ShaderStages, UniformBuffer,
};
use bevy::render::renderer::{RenderContext, RenderDevice, RenderGraph, RenderQueue};
use bevy::render::storage::{GpuShaderBuffer, ShaderBuffer};
use bevy::render::{Render, RenderApp, RenderStartup, RenderSystems};

use super::{AeolianParams, DustGpu};
use crate::gpu::kernel::{Kernel, groups, queue_kernel, ready};
use crate::hydrology::{HydroParams, HydrologyDispatch, HydrologyPipelines};
use crate::terrain::{TerrainDispatch, TerrainGpu};

impl ExtractResource<RenderApp> for DustGpu {
    type Source = DustGpu;

    fn extract_resource(source: &Self::Source) -> Self {
        source.clone()
    }
}

#[derive(Resource)]
struct DustPipelines {
    capacity: Kernel,
    flux: Kernel,
    mov: Kernel,
}

#[derive(Resource, Default)]
struct DustUniforms {
    aeolian: UniformBuffer<AeolianParams>,
    avalanche: UniformBuffer<HydroParams>,
}

#[derive(Resource)]
struct DustBindGroups {
    aeolian: BindGroup,
    avalanche: BindGroup,
}

pub(super) fn plugin(app: &mut App) {
    app.add_plugins(ExtractResourcePlugin::<DustGpu>::default());
    let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
        return;
    };
    render_app
        .init_resource::<DustUniforms>()
        .add_systems(RenderStartup, init_pipelines)
        .add_systems(Render, prepare.in_set(RenderSystems::PrepareBindGroups))
        .add_systems(
            RenderGraph,
            dispatch
                .after(HydrologyDispatch)
                .before(TerrainDispatch)
                .before(camera_driver),
        );
}

fn layout() -> BindGroupLayoutDescriptor {
    BindGroupLayoutDescriptor::new(
        "elements_aeolian",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                uniform_buffer::<AeolianParams>(false),
                storage_buffer_sized(false, None),
                storage_buffer_sized(false, None),
                storage_buffer_sized(false, None),
                storage_buffer_sized(false, None),
                storage_buffer_sized(false, None),
            ),
        ),
    )
}

fn init_pipelines(mut commands: Commands, assets: Res<AssetServer>, cache: Res<PipelineCache>) {
    let shader = load_embedded_asset!(assets.as_ref(), "aeolian.wesl");
    let kernel = |entry| queue_kernel(&cache, shader.clone(), entry, layout(), Vec::new());
    commands.insert_resource(DustPipelines {
        capacity: kernel("aeolian_capacity"),
        flux: kernel("aeolian_flux"),
        mov: kernel("aeolian_move"),
    });
}

#[allow(clippy::too_many_arguments)]
fn prepare(
    mut commands: Commands,
    gpu: Option<Res<DustGpu>>,
    terrain: Option<Res<TerrainGpu>>,
    pipelines: Option<Res<DustPipelines>>,
    thermal: Option<Res<HydrologyPipelines>>,
    mut uniforms: ResMut<DustUniforms>,
    buffers: Res<RenderAssets<GpuShaderBuffer>>,
    cache: Res<PipelineCache>,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
) {
    let (Some(gpu), Some(terrain), Some(pipelines), Some(thermal)) =
        (gpu, terrain, pipelines, thermal)
    else {
        commands.remove_resource::<DustBindGroups>();
        return;
    };
    let get = |h: &Handle<ShaderBuffer>| buffers.get(h).map(|b| &b.buffer);
    let s = &terrain.state;
    let (
        Some(bedrock),
        Some(sediment),
        Some(transport),
        Some(thermal_a),
        Some(thermal_b),
        Some(cap),
        Some(q_x),
        Some(q_z),
    ) = (
        get(&s.bedrock),
        get(&s.sediment),
        get(&s.transport),
        get(&s.thermal_flux[0]),
        get(&s.thermal_flux[1]),
        get(&gpu.buffers.cap),
        get(&gpu.buffers.q[0]),
        get(&gpu.buffers.q[1]),
    )
    else {
        gpu.ready.store(false, Ordering::Relaxed);
        commands.remove_resource::<DustBindGroups>();
        return;
    };
    let compiled = ready(&cache, &pipelines.capacity).is_some()
        && ready(&cache, &pipelines.flux).is_some()
        && ready(&cache, &pipelines.mov).is_some()
        && ready(&cache, &thermal.thermal_flux).is_some()
        && ready(&cache, &thermal.thermal_apply).is_some();
    gpu.ready.store(compiled, Ordering::Relaxed);
    if !compiled || !gpu.run {
        commands.remove_resource::<DustBindGroups>();
        return;
    }
    uniforms.aeolian.set(gpu.params);
    uniforms.aeolian.write_buffer(&device, &queue);
    uniforms.avalanche.set(gpu.avalanche);
    uniforms.avalanche.write_buffer(&device, &queue);

    let aeolian = device.create_bind_group(
        "elements_aeolian",
        &cache.get_bind_group_layout(&pipelines.capacity.layout),
        &BindGroupEntries::sequential((
            &uniforms.aeolian,
            bedrock.as_entire_buffer_binding(),
            sediment.as_entire_buffer_binding(),
            cap.as_entire_buffer_binding(),
            q_x.as_entire_buffer_binding(),
            q_z.as_entire_buffer_binding(),
        )),
    );
    let avalanche = device.create_bind_group(
        "elements_aeolian_avalanche",
        &cache.get_bind_group_layout(&thermal.thermal_flux.layout),
        &BindGroupEntries::sequential((
            &uniforms.avalanche,
            bedrock.as_entire_buffer_binding(),
            sediment.as_entire_buffer_binding(),
            thermal_a.as_entire_buffer_binding(),
            thermal_b.as_entire_buffer_binding(),
            transport.as_entire_buffer_binding(),
        )),
    );
    commands.insert_resource(DustBindGroups { aeolian, avalanche });
}

fn dispatch(
    mut ctx: RenderContext,
    cache: Res<PipelineCache>,
    gpu: Option<Res<DustGpu>>,
    terrain: Option<Res<TerrainGpu>>,
    pipelines: Option<Res<DustPipelines>>,
    thermal: Option<Res<HydrologyPipelines>>,
    bind_groups: Option<Res<DustBindGroups>>,
) {
    let (Some(gpu), Some(terrain), Some(pipelines), Some(thermal), Some(bind_groups)) =
        (gpu, terrain, pipelines, thermal, bind_groups)
    else {
        return;
    };
    if !gpu.run || !terrain.generated.load(Ordering::Relaxed) {
        return;
    }
    let (Some(capacity), Some(flux), Some(mov), Some(avalanche_flux), Some(avalanche_apply)) = (
        ready(&cache, &pipelines.capacity),
        ready(&cache, &pipelines.flux),
        ready(&cache, &pipelines.mov),
        ready(&cache, &thermal.thermal_flux),
        ready(&cache, &thermal.thermal_apply),
    ) else {
        return;
    };
    let size = gpu.params.size;
    let (cx, cy) = (groups(size.x, 8), groups(size.y, 8));
    let group = &bind_groups.aeolian;
    let diagnostics = ctx.diagnostic_recorder();
    let diagnostics = diagnostics.as_deref();
    let mut run =
        |label: &'static str,
         record: &mut dyn FnMut(&mut bevy::render::render_resource::ComputePass)| {
            let mut pass = ctx
                .command_encoder()
                .begin_compute_pass(&ComputePassDescriptor {
                    label: Some(label),
                    ..default()
                });
            let span = diagnostics.pass_span(&mut pass, label);
            record(&mut pass);
            span.end(&mut pass);
        };
    run("elements_aeolian_capacity", &mut |pass| {
        pass.set_pipeline(capacity);
        pass.set_bind_group(0, group, &[]);
        pass.dispatch_workgroups(cx, cy, 1);
    });
    run("elements_aeolian_flux", &mut |pass| {
        pass.set_pipeline(flux);
        pass.set_bind_group(0, group, &[]);
        // One thread per row and per column.
        pass.dispatch_workgroups(groups(size.x + size.y, 64), 1, 1);
    });
    run("elements_aeolian_move", &mut |pass| {
        pass.set_pipeline(mov);
        pass.set_bind_group(0, group, &[]);
        pass.dispatch_workgroups(cx, cy, 1);
    });
    run("elements_aeolian_avalanche", &mut |pass| {
        pass.set_bind_group(0, &bind_groups.avalanche, &[]);
        for _ in 0..gpu.avalanche_passes {
            pass.set_pipeline(avalanche_flux);
            pass.dispatch_workgroups(cx, cy, 1);
            pass.set_pipeline(avalanche_apply);
            pass.dispatch_workgroups(cx, cy, 1);
        }
    });
    gpu.executed.fetch_add(1, Ordering::Relaxed);
}
