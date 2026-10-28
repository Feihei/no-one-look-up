//! Render-world side of the ground cover: pipelines, bind groups and the per-frame dispatch.
//!
//! Three compute passes a frame, each with its own diagnostic span: `elements_shelter` (the
//! shelter mask, only when the blockers or the fall directions changed), `elements_snow` (per
//! step: source, drift flux and apply when the wind can move snow, two diffusion passes) and
//! `elements_wetness` (the wetness update, the render pack, and the readback kernel on a readback
//! tick). The passes run after hydrology's substeps and before the terrain's pack kernels, which
//! publish the melt water to the packed render buffer in the same frame.

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
    ComputePassDescriptor, ComputePipeline, PipelineCache, ShaderStages, StorageBuffer,
    UniformBuffer,
};
use bevy::render::renderer::{RenderContext, RenderDevice, RenderGraph, RenderQueue};
use bevy::render::storage::{GpuShaderBuffer, ShaderBuffer};
use bevy::render::{Render, RenderApp, RenderStartup, RenderSystems};

use super::{CoverGpu, CoverParams};
use crate::gpu::kernel::{Kernel, groups, queue_kernel, ready};
use crate::hydrology::HydrologyDispatch;
use crate::precipitation::BlockerGpu;
use crate::terrain::{TerrainDispatch, TerrainGpu};

impl ExtractResource<RenderApp> for CoverGpu {
    type Source = CoverGpu;

    fn extract_resource(source: &Self::Source) -> Self {
        source.clone()
    }
}

#[derive(Resource)]
struct CoverPipelines {
    source: Kernel,
    drift_flux: Kernel,
    drift_apply: Kernel,
    diffuse_a: Kernel,
    diffuse_b: Kernel,
    wetness: Kernel,
    pack: Kernel,
    shelter: Kernel,
    readback: Kernel,
}

struct Ready<'a> {
    source: &'a ComputePipeline,
    drift_flux: &'a ComputePipeline,
    drift_apply: &'a ComputePipeline,
    diffuse_a: &'a ComputePipeline,
    diffuse_b: &'a ComputePipeline,
    wetness: &'a ComputePipeline,
    pack: &'a ComputePipeline,
    shelter: &'a ComputePipeline,
    readback: &'a ComputePipeline,
}

impl CoverPipelines {
    /// Every compiled pipeline, or `None` while any compiles or after one failed.
    fn all_ready<'a>(&self, cache: &'a PipelineCache) -> Option<Ready<'a>> {
        Some(Ready {
            source: ready(cache, &self.source)?,
            drift_flux: ready(cache, &self.drift_flux)?,
            drift_apply: ready(cache, &self.drift_apply)?,
            diffuse_a: ready(cache, &self.diffuse_a)?,
            diffuse_b: ready(cache, &self.diffuse_b)?,
            wetness: ready(cache, &self.wetness)?,
            pack: ready(cache, &self.pack)?,
            shelter: ready(cache, &self.shelter)?,
            readback: ready(cache, &self.readback)?,
        })
    }
}

#[derive(Resource, Default)]
struct CoverUniforms {
    params: UniformBuffer<CoverParams>,
    blockers: StorageBuffer<Vec<BlockerGpu>>,
}

#[derive(Resource)]
struct CoverBindGroups {
    maps: BindGroup,
    readback: BindGroup,
    /// Only while the shelter mask is stale.
    shelter: Option<BindGroup>,
}

pub(super) fn plugin(app: &mut App) {
    app.add_plugins(ExtractResourcePlugin::<CoverGpu>::default());
    let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
        return;
    };
    render_app
        .init_resource::<CoverUniforms>()
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

fn maps_layout() -> BindGroupLayoutDescriptor {
    BindGroupLayoutDescriptor::new(
        "elements_cover_maps",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                uniform_buffer::<CoverParams>(false),
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

fn shelter_layout() -> BindGroupLayoutDescriptor {
    BindGroupLayoutDescriptor::new(
        "elements_cover_shelter",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                uniform_buffer::<CoverParams>(false),
                storage_buffer_read_only_sized(false, None),
                storage_buffer_sized(false, None),
                storage_buffer_sized(false, None),
            ),
        ),
    )
}

fn readback_layout() -> BindGroupLayoutDescriptor {
    BindGroupLayoutDescriptor::new(
        "elements_cover_readback",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                uniform_buffer::<CoverParams>(false),
                storage_buffer_sized(false, None),
                storage_buffer_sized(false, None),
            ),
        ),
    )
}

fn init_pipelines(mut commands: Commands, assets: Res<AssetServer>, cache: Res<PipelineCache>) {
    let snow = load_embedded_asset!(assets.as_ref(), "snow.wesl");
    let shelter = load_embedded_asset!(assets.as_ref(), "shelter.wesl");
    let readback = load_embedded_asset!(assets.as_ref(), "readback.wesl");
    let kernel = |shader: &Handle<Shader>, entry, layout: BindGroupLayoutDescriptor| {
        queue_kernel(&cache, shader.clone(), entry, layout, Vec::new())
    };
    commands.insert_resource(CoverPipelines {
        source: kernel(&snow, "snow_source", maps_layout()),
        drift_flux: kernel(&snow, "snow_drift_flux", maps_layout()),
        drift_apply: kernel(&snow, "snow_drift_apply", maps_layout()),
        diffuse_a: kernel(&snow, "snow_diffuse_a", maps_layout()),
        diffuse_b: kernel(&snow, "snow_diffuse_b", maps_layout()),
        wetness: kernel(&snow, "wetness_update", maps_layout()),
        pack: kernel(&snow, "cover_pack", maps_layout()),
        shelter: kernel(&shelter, "shelter_mask", shelter_layout()),
        readback: kernel(&readback, "cover_readback", readback_layout()),
    });
}

#[allow(clippy::too_many_arguments)]
fn prepare(
    mut commands: Commands,
    gpu: Option<Res<CoverGpu>>,
    terrain: Option<Res<TerrainGpu>>,
    pipelines: Option<Res<CoverPipelines>>,
    mut uniforms: ResMut<CoverUniforms>,
    buffers: Res<RenderAssets<GpuShaderBuffer>>,
    cache: Res<PipelineCache>,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
) {
    let (Some(gpu), Some(terrain), Some(pipelines)) = (gpu, terrain, pipelines) else {
        commands.remove_resource::<CoverBindGroups>();
        return;
    };
    let get = |h: &Handle<ShaderBuffer>| buffers.get(h).map(|b| &b.buffer);
    let (
        Some(snow),
        Some(drift),
        Some(wetness),
        Some(shelter),
        Some(cover_cells),
        Some(readback),
        Some(cells),
        Some(water),
    ) = (
        get(&gpu.snow),
        get(&gpu.drift),
        get(&gpu.wetness),
        get(&gpu.shelter),
        get(&gpu.cover_cells),
        get(&gpu.readback),
        get(&terrain.state.cells),
        get(&terrain.state.water),
    )
    else {
        gpu.ready.store(false, Ordering::Relaxed);
        commands.remove_resource::<CoverBindGroups>();
        return;
    };
    let compiled = pipelines.all_ready(&cache).is_some();
    gpu.ready.store(
        compiled && terrain.generated.load(Ordering::Relaxed),
        Ordering::Relaxed,
    );
    let stale = gpu.shelter_built.load(Ordering::Relaxed) < gpu.shelter_epoch;
    if !compiled || (gpu.steps == 0 && !stale && !gpu.downsample) {
        commands.remove_resource::<CoverBindGroups>();
        return;
    }

    uniforms.params.set(gpu.params);
    uniforms.params.write_buffer(&device, &queue);
    let maps = device.create_bind_group(
        "elements_cover_maps",
        &cache.get_bind_group_layout(&pipelines.source.layout),
        &BindGroupEntries::sequential((
            &uniforms.params,
            snow.as_entire_buffer_binding(),
            drift.as_entire_buffer_binding(),
            wetness.as_entire_buffer_binding(),
            shelter.as_entire_buffer_binding(),
            cells.as_entire_buffer_binding(),
            water.as_entire_buffer_binding(),
            cover_cells.as_entire_buffer_binding(),
        )),
    );
    let readback = device.create_bind_group(
        "elements_cover_readback",
        &cache.get_bind_group_layout(&pipelines.readback.layout),
        &BindGroupEntries::sequential((
            &uniforms.params,
            cover_cells.as_entire_buffer_binding(),
            readback.as_entire_buffer_binding(),
        )),
    );
    let shelter_group = if stale {
        uniforms.blockers.set(gpu.blockers.clone());
        uniforms.blockers.write_buffer(&device, &queue);
        uniforms.blockers.binding().map(|blockers| {
            device.create_bind_group(
                "elements_cover_shelter",
                &cache.get_bind_group_layout(&pipelines.shelter.layout),
                &BindGroupEntries::sequential((
                    &uniforms.params,
                    blockers,
                    cells.as_entire_buffer_binding(),
                    shelter.as_entire_buffer_binding(),
                )),
            )
        })
    } else {
        None
    };
    commands.insert_resource(CoverBindGroups {
        maps,
        readback,
        shelter: shelter_group,
    });
}

fn dispatch(
    mut ctx: RenderContext,
    cache: Res<PipelineCache>,
    gpu: Option<Res<CoverGpu>>,
    terrain: Option<Res<TerrainGpu>>,
    pipelines: Option<Res<CoverPipelines>>,
    bind_groups: Option<Res<CoverBindGroups>>,
) {
    let (Some(gpu), Some(terrain), Some(pipelines)) = (gpu, terrain, pipelines) else {
        return;
    };
    let steps = u64::from(gpu.steps);
    let stale = gpu.shelter_built.load(Ordering::Relaxed) < gpu.shelter_epoch;
    if steps == 0 && !stale && !gpu.downsample {
        return;
    }
    if !terrain.generated.load(Ordering::Relaxed) {
        gpu.dropped.fetch_add(steps, Ordering::Relaxed);
        return;
    }
    let (Some(bind_groups), Some(pipes)) = (bind_groups, pipelines.all_ready(&cache)) else {
        gpu.dropped.fetch_add(steps, Ordering::Relaxed);
        return;
    };
    if stale && bind_groups.shelter.is_none() {
        gpu.dropped.fetch_add(steps, Ordering::Relaxed);
        return;
    }
    let size = gpu.params.size;
    let (cx, cy) = (groups(size.x, 8), groups(size.y, 8));
    let diagnostics = ctx.diagnostic_recorder();
    let diagnostics = diagnostics.as_deref();

    if let (true, Some(group)) = (stale, bind_groups.shelter.as_ref()) {
        let mut pass = ctx
            .command_encoder()
            .begin_compute_pass(&ComputePassDescriptor {
                label: Some("elements_shelter"),
                ..default()
            });
        let span = diagnostics.pass_span(&mut pass, "elements_shelter");
        pass.set_pipeline(pipes.shelter);
        pass.set_bind_group(0, group, &[]);
        pass.dispatch_workgroups(cx, cy, 1);
        span.end(&mut pass);
    }
    if steps > 0 {
        let mut pass = ctx
            .command_encoder()
            .begin_compute_pass(&ComputePassDescriptor {
                label: Some("elements_snow"),
                ..default()
            });
        let span = diagnostics.pass_span(&mut pass, "elements_snow");
        pass.set_bind_group(0, &bind_groups.maps, &[]);
        for _ in 0..gpu.steps {
            pass.set_pipeline(pipes.source);
            pass.dispatch_workgroups(cx, cy, 1);
            if gpu.drift_on {
                pass.set_pipeline(pipes.drift_flux);
                pass.dispatch_workgroups(cx, cy, 1);
                pass.set_pipeline(pipes.drift_apply);
                pass.dispatch_workgroups(cx, cy, 1);
            }
            pass.set_pipeline(pipes.diffuse_a);
            pass.dispatch_workgroups(cx, cy, 1);
            pass.set_pipeline(pipes.diffuse_b);
            pass.dispatch_workgroups(cx, cy, 1);
        }
        span.end(&mut pass);
    }
    if steps > 0 || gpu.downsample {
        let mut pass = ctx
            .command_encoder()
            .begin_compute_pass(&ComputePassDescriptor {
                label: Some("elements_wetness"),
                ..default()
            });
        let span = diagnostics.pass_span(&mut pass, "elements_wetness");
        pass.set_bind_group(0, &bind_groups.maps, &[]);
        if steps > 0 {
            pass.set_pipeline(pipes.wetness);
            pass.dispatch_workgroups(cx, cy, 1);
        }
        pass.set_pipeline(pipes.pack);
        pass.dispatch_workgroups(cx, cy, 1);
        if gpu.downsample {
            let ds = gpu.readback_ds;
            pass.set_bind_group(0, &bind_groups.readback, &[]);
            pass.set_pipeline(pipes.readback);
            pass.dispatch_workgroups(groups(size.x / ds, 8), groups(size.y / ds, 8), 1);
        }
        span.end(&mut pass);
    }
    if stale {
        gpu.shelter_built
            .store(gpu.shelter_epoch, Ordering::Relaxed);
    }
    if steps > 0 {
        gpu.executed.fetch_add(steps, Ordering::Relaxed);
        let seconds = f64::from(gpu.params.terrain.w);
        gpu.executed_us
            .fetch_add((seconds * 1e6) as u64, Ordering::Relaxed);
    }
    if gpu.downsample {
        // The readback buffer holds the maps now; a readback may be requested.
        gpu.live.store(true, Ordering::Relaxed);
    }
}
