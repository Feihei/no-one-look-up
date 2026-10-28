//! Render-world side of hydrology and erosion: pipelines, bind groups and the per-frame dispatch.
//!
//! One compute pass per frame: the source map when it is stale, then per substep `flux_step` and
//! `depth_erode` (and the two thermal kernels on every 4th substep), then the readback kernels on
//! a readback tick. The pass runs before the terrain's pack kernels, which publish the result to
//! the packed render buffer in the same frame.

use std::sync::atomic::Ordering;

use bevy::asset::load_embedded_asset;
use bevy::core_pipeline::schedule::camera_driver;
use bevy::prelude::*;
use bevy::render::diagnostic::RecordDiagnostics;
use bevy::render::extract_resource::{ExtractResource, ExtractResourcePlugin};
use bevy::render::render_asset::RenderAssets;
use bevy::render::render_resource::binding_types::{
    storage_buffer_read_only_sized, storage_buffer_sized, uniform_buffer,
};
use bevy::render::render_resource::{
    BindGroup, BindGroupEntries, BindGroupLayoutDescriptor, BindGroupLayoutEntries,
    ComputePassDescriptor, PipelineCache, ShaderStages, StorageBuffer, UniformBuffer,
};
use bevy::render::renderer::{RenderContext, RenderDevice, RenderGraph, RenderQueue};
use bevy::render::storage::GpuShaderBuffer;
use bevy::render::{Render, RenderApp, RenderStartup, RenderSystems};

use super::{HydroParams, HydrologyGpu};
use crate::gpu::kernel::{Kernel, groups, queue_kernel, ready};
use crate::precipitation::BlockerGpu;
use crate::terrain::{TerrainDispatch, TerrainGpu};

impl ExtractResource<RenderApp> for HydrologyGpu {
    type Source = HydrologyGpu;

    fn extract_resource(source: &Self::Source) -> Self {
        source.clone()
    }
}

/// The hydrology compute pass. Passes that read or write the layers after the substeps (`dust`)
/// order themselves `.after(HydrologyDispatch)`.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct HydrologyDispatch;

/// The hydrology kernels. The thermal pair is shared with `dust`, which dispatches it with its
/// own uniform (the avalanche at the sand's repose angle).
#[derive(Resource)]
pub(crate) struct HydrologyPipelines {
    flux: Kernel,
    depth_erode: Kernel,
    pub(crate) thermal_flux: Kernel,
    pub(crate) thermal_apply: Kernel,
    source_map: Kernel,
    depth_readback: Kernel,
    velocity_readback: Kernel,
}

impl HydrologyPipelines {
    /// Every compiled pipeline, or `None` while any compiles or after one failed.
    fn all_ready<'a>(&self, cache: &'a PipelineCache) -> Option<Ready<'a>> {
        Some(Ready {
            flux: ready(cache, &self.flux)?,
            depth_erode: ready(cache, &self.depth_erode)?,
            thermal_flux: ready(cache, &self.thermal_flux)?,
            thermal_apply: ready(cache, &self.thermal_apply)?,
            source_map: ready(cache, &self.source_map)?,
            depth_readback: ready(cache, &self.depth_readback)?,
            velocity_readback: ready(cache, &self.velocity_readback)?,
        })
    }
}

struct Ready<'a> {
    flux: &'a bevy::render::render_resource::ComputePipeline,
    depth_erode: &'a bevy::render::render_resource::ComputePipeline,
    thermal_flux: &'a bevy::render::render_resource::ComputePipeline,
    thermal_apply: &'a bevy::render::render_resource::ComputePipeline,
    source_map: &'a bevy::render::render_resource::ComputePipeline,
    depth_readback: &'a bevy::render::render_resource::ComputePipeline,
    velocity_readback: &'a bevy::render::render_resource::ComputePipeline,
}

#[derive(Resource, Default)]
struct HydrologyUniforms {
    /// The frame's uniform with the rain and evaporation quanta (the first substep).
    first: UniformBuffer<HydroParams>,
    /// The same with both quanta zero (every other substep, the thermal and readback kernels).
    rest: UniformBuffer<HydroParams>,
    sources: StorageBuffer<Vec<Vec4>>,
    blockers: StorageBuffer<Vec<BlockerGpu>>,
}

#[derive(Resource)]
struct HydrologyBindGroups {
    /// `[first substep, later substeps]`.
    substep: [BindGroup; 2],
    thermal: BindGroup,
    readback: BindGroup,
    /// Only while the source map is stale.
    source_map: Option<BindGroup>,
}

pub(super) fn plugin(app: &mut App) {
    app.add_plugins(ExtractResourcePlugin::<HydrologyGpu>::default());
    let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
        return;
    };
    render_app
        .init_resource::<HydrologyUniforms>()
        .add_systems(RenderStartup, init_pipelines)
        .add_systems(Render, prepare.in_set(RenderSystems::PrepareBindGroups))
        .add_systems(
            RenderGraph,
            dispatch
                .in_set(HydrologyDispatch)
                .before(TerrainDispatch)
                .before(camera_driver),
        );
}

fn substep_layout() -> BindGroupLayoutDescriptor {
    BindGroupLayoutDescriptor::new(
        "elements_hydrology_substep",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                uniform_buffer::<HydroParams>(false),
                storage_buffer_sized(false, None),
                storage_buffer_sized(false, None),
                storage_buffer_sized(false, None),
                storage_buffer_sized(false, None),
                storage_buffer_sized(false, None),
                storage_buffer_sized(false, None),
                storage_buffer_sized(false, None),
                storage_buffer_sized(false, None),
            ),
        ),
    )
}

fn thermal_layout() -> BindGroupLayoutDescriptor {
    BindGroupLayoutDescriptor::new(
        "elements_hydrology_thermal",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                uniform_buffer::<HydroParams>(false),
                storage_buffer_sized(false, None),
                storage_buffer_sized(false, None),
                storage_buffer_sized(false, None),
                storage_buffer_sized(false, None),
                storage_buffer_sized(false, None),
            ),
        ),
    )
}

fn source_map_layout() -> BindGroupLayoutDescriptor {
    BindGroupLayoutDescriptor::new(
        "elements_hydrology_source_map",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                uniform_buffer::<HydroParams>(false),
                storage_buffer_read_only_sized(false, None),
                storage_buffer_read_only_sized(false, None),
                storage_buffer_read_only_sized(false, None),
                storage_buffer_read_only_sized(false, None),
                storage_buffer_sized(false, None),
            ),
        ),
    )
}

fn readback_layout() -> BindGroupLayoutDescriptor {
    BindGroupLayoutDescriptor::new(
        "elements_hydrology_readback",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                uniform_buffer::<HydroParams>(false),
                storage_buffer_sized(false, None),
                storage_buffer_sized(false, None),
                storage_buffer_sized(false, None),
                storage_buffer_sized(false, None),
            ),
        ),
    )
}

fn init_pipelines(mut commands: Commands, assets: Res<AssetServer>, cache: Res<PipelineCache>) {
    let substep = load_embedded_asset!(assets.as_ref(), "substep.wesl");
    let thermal = load_embedded_asset!(assets.as_ref(), "thermal.wesl");
    let source_map = load_embedded_asset!(assets.as_ref(), "source_map.wesl");
    let readback = load_embedded_asset!(assets.as_ref(), "readback.wesl");
    let kernel = |shader: &Handle<Shader>, entry, layout: BindGroupLayoutDescriptor| {
        queue_kernel(&cache, shader.clone(), entry, layout, Vec::new())
    };
    commands.insert_resource(HydrologyPipelines {
        flux: kernel(&substep, "flux_step", substep_layout()),
        depth_erode: kernel(&substep, "depth_erode", substep_layout()),
        thermal_flux: kernel(&thermal, "thermal_flux", thermal_layout()),
        thermal_apply: kernel(&thermal, "thermal_apply", thermal_layout()),
        source_map: kernel(&source_map, "source_map_build", source_map_layout()),
        depth_readback: kernel(&readback, "downsample_depth", readback_layout()),
        velocity_readback: kernel(&readback, "downsample_velocity", readback_layout()),
    });
}

#[allow(clippy::too_many_arguments)]
fn prepare(
    mut commands: Commands,
    gpu: Option<Res<HydrologyGpu>>,
    terrain: Option<Res<TerrainGpu>>,
    pipelines: Option<Res<HydrologyPipelines>>,
    mut uniforms: ResMut<HydrologyUniforms>,
    buffers: Res<RenderAssets<GpuShaderBuffer>>,
    cache: Res<PipelineCache>,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
) {
    let (Some(gpu), Some(terrain), Some(pipelines)) = (gpu, terrain, pipelines) else {
        commands.remove_resource::<HydrologyBindGroups>();
        return;
    };
    let get = |h: &Handle<bevy::render::storage::ShaderBuffer>| buffers.get(h).map(|b| &b.buffer);
    let s = &terrain.state;
    let (
        Some(bedrock),
        Some(sediment),
        Some(water),
        Some(source_map),
        Some(flux),
        Some(velocity),
        Some(suspended),
        Some(transport),
        Some(thermal_a),
        Some(thermal_b),
        Some(depth_rb),
        Some(velocity_rb),
    ) = (
        get(&s.bedrock),
        get(&s.sediment),
        get(&s.water),
        get(&s.source_map),
        get(&s.flux),
        get(&s.velocity),
        get(&s.suspended),
        get(&s.transport),
        get(&s.thermal_flux[0]),
        get(&s.thermal_flux[1]),
        get(&gpu.buffers.depth),
        get(&gpu.buffers.velocity),
    )
    else {
        gpu.ready.store(false, Ordering::Relaxed);
        commands.remove_resource::<HydrologyBindGroups>();
        return;
    };
    let compiled = pipelines.all_ready(&cache).is_some();
    gpu.ready.store(
        compiled && terrain.generated.load(Ordering::Relaxed),
        Ordering::Relaxed,
    );
    let rebuild = gpu.source_built.load(Ordering::Relaxed) < gpu.source_epoch;
    if !compiled || (gpu.substeps == 0 && !rebuild && !gpu.downsample) {
        commands.remove_resource::<HydrologyBindGroups>();
        return;
    }

    let mut rest = gpu.params;
    rest.weather.x = 0.0;
    rest.weather.y = 0.0;
    uniforms.first.set(gpu.params);
    uniforms.first.write_buffer(&device, &queue);
    uniforms.rest.set(rest);
    uniforms.rest.write_buffer(&device, &queue);

    let substep_layout = cache.get_bind_group_layout(&pipelines.flux.layout);
    let substep = |params: &UniformBuffer<HydroParams>| {
        device.create_bind_group(
            "elements_hydrology_substep",
            &substep_layout,
            &BindGroupEntries::sequential((
                params,
                bedrock.as_entire_buffer_binding(),
                sediment.as_entire_buffer_binding(),
                water.as_entire_buffer_binding(),
                source_map.as_entire_buffer_binding(),
                flux.as_entire_buffer_binding(),
                velocity.as_entire_buffer_binding(),
                suspended.as_entire_buffer_binding(),
                transport.as_entire_buffer_binding(),
            )),
        )
    };
    let thermal = device.create_bind_group(
        "elements_hydrology_thermal",
        &cache.get_bind_group_layout(&pipelines.thermal_flux.layout),
        &BindGroupEntries::sequential((
            &uniforms.rest,
            bedrock.as_entire_buffer_binding(),
            sediment.as_entire_buffer_binding(),
            thermal_a.as_entire_buffer_binding(),
            thermal_b.as_entire_buffer_binding(),
            transport.as_entire_buffer_binding(),
        )),
    );
    let readback = device.create_bind_group(
        "elements_hydrology_readback",
        &cache.get_bind_group_layout(&pipelines.depth_readback.layout),
        &BindGroupEntries::sequential((
            &uniforms.rest,
            water.as_entire_buffer_binding(),
            velocity.as_entire_buffer_binding(),
            depth_rb.as_entire_buffer_binding(),
            velocity_rb.as_entire_buffer_binding(),
        )),
    );
    let source_group = if rebuild {
        uniforms.sources.set(gpu.sources.clone());
        uniforms.sources.write_buffer(&device, &queue);
        uniforms.blockers.set(gpu.blockers.clone());
        uniforms.blockers.write_buffer(&device, &queue);
        match (uniforms.sources.binding(), uniforms.blockers.binding()) {
            (Some(sources), Some(blockers)) => Some(device.create_bind_group(
                "elements_hydrology_source_map",
                &cache.get_bind_group_layout(&pipelines.source_map.layout),
                &BindGroupEntries::sequential((
                    &uniforms.rest,
                    bedrock.as_entire_buffer_binding(),
                    sediment.as_entire_buffer_binding(),
                    sources,
                    blockers,
                    source_map.as_entire_buffer_binding(),
                )),
            )),
            _ => None,
        }
    } else {
        None
    };
    commands.insert_resource(HydrologyBindGroups {
        substep: [substep(&uniforms.first), substep(&uniforms.rest)],
        thermal,
        readback,
        source_map: source_group,
    });
}

fn dispatch(
    mut ctx: RenderContext,
    cache: Res<PipelineCache>,
    gpu: Option<Res<HydrologyGpu>>,
    terrain: Option<Res<TerrainGpu>>,
    pipelines: Option<Res<HydrologyPipelines>>,
    bind_groups: Option<Res<HydrologyBindGroups>>,
) {
    let (Some(gpu), Some(terrain), Some(pipelines)) = (gpu, terrain, pipelines) else {
        return;
    };
    let substeps = u64::from(gpu.substeps);
    let work = (substeps > 0 || gpu.downsample)
        || gpu.source_built.load(Ordering::Relaxed) < gpu.source_epoch;
    if !work || !terrain.generated.load(Ordering::Relaxed) {
        gpu.dropped.fetch_add(substeps, Ordering::Relaxed);
        return;
    }
    let rebuild = gpu.source_built.load(Ordering::Relaxed) < gpu.source_epoch;
    let (Some(bind_groups), Some(pipes)) = (bind_groups, pipelines.all_ready(&cache)) else {
        gpu.dropped.fetch_add(substeps, Ordering::Relaxed);
        return;
    };
    if rebuild && bind_groups.source_map.is_none() {
        gpu.dropped.fetch_add(substeps, Ordering::Relaxed);
        return;
    }
    let size = gpu.params.size;
    let (cx, cy) = (groups(size.x, 8), groups(size.y, 8));
    let diagnostics = ctx.diagnostic_recorder();
    let diagnostics = diagnostics.as_deref();
    let mut pass = ctx
        .command_encoder()
        .begin_compute_pass(&ComputePassDescriptor {
            label: Some("elements_hydrology"),
            ..default()
        });
    let span = diagnostics.pass_span(&mut pass, "elements_hydrology");
    if let (true, Some(group)) = (rebuild, bind_groups.source_map.as_ref()) {
        pass.set_pipeline(pipes.source_map);
        pass.set_bind_group(0, group, &[]);
        pass.dispatch_workgroups(cx, cy, 1);
    }
    for k in 0..gpu.substeps {
        pass.set_bind_group(0, &bind_groups.substep[usize::from(k > 0)], &[]);
        pass.set_pipeline(pipes.flux);
        pass.dispatch_workgroups(cx, cy, 1);
        pass.set_pipeline(pipes.depth_erode);
        pass.dispatch_workgroups(cx, cy, 1);
        if gpu.thermal && (gpu.thermal_phase + k + 1).is_multiple_of(super::THERMAL_PERIOD as u32) {
            pass.set_bind_group(0, &bind_groups.thermal, &[]);
            pass.set_pipeline(pipes.thermal_flux);
            pass.dispatch_workgroups(cx, cy, 1);
            pass.set_pipeline(pipes.thermal_apply);
            pass.dispatch_workgroups(cx, cy, 1);
        }
    }
    if gpu.downsample {
        let ds = gpu.buffers.depth_ds;
        pass.set_bind_group(0, &bind_groups.readback, &[]);
        pass.set_pipeline(pipes.depth_readback);
        pass.dispatch_workgroups(groups(size.x / ds, 8), groups(size.y / ds, 8), 1);
        pass.set_pipeline(pipes.velocity_readback);
        pass.dispatch_workgroups(groups(size.x / 2, 8), groups(size.y / 2, 8), 1);
    }
    span.end(&mut pass);
    drop(pass);
    if rebuild {
        gpu.source_built.store(gpu.source_epoch, Ordering::Relaxed);
    }
    gpu.executed.fetch_add(substeps, Ordering::Relaxed);
    if gpu.downsample {
        // The readback buffers hold the water now; a readback may be requested.
        gpu.live.store(true, Ordering::Relaxed);
    }
}
