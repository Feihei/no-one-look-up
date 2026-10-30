//! Render-world side of the waves: pipelines, bind groups, and one compute pass a frame
//! (`elements_waves`: the bodies' head, the substeps, the published field) after the terrain's
//! pack kernels, so the water depth and velocity it reads are this frame's.
//!
//! The render world remembers which state buffer holds the latest substep and the windows'
//! origins of the last frame it ran: the main world cannot know either under pipelined
//! rendering.

use std::sync::atomic::Ordering;

use bevy::asset::load_embedded_asset;
use bevy::core_pipeline::schedule::camera_driver;
use bevy::prelude::*;
use bevy::render::diagnostic::RecordDiagnostics;
use bevy::render::extract_resource::{ExtractResource, ExtractResourcePlugin};
use bevy::render::render_asset::RenderAssets;
use bevy::render::render_resource::binding_types::{
    storage_buffer_read_only_sized, storage_buffer_sized, texture_storage_2d_array, uniform_buffer,
};
use bevy::render::render_resource::{
    BindGroup, BindGroupEntries, BindGroupLayoutDescriptor, BindGroupLayoutEntries, Buffer,
    ComputePassDescriptor, PipelineCache, ShaderStages, StorageBuffer, StorageTextureAccess,
    TextureFormat, UniformBuffer,
};
use bevy::render::renderer::{RenderContext, RenderDevice, RenderGraph, RenderQueue};
use bevy::render::storage::{GpuShaderBuffer, ShaderBuffer};
use bevy::render::texture::GpuImage;
use bevy::render::{Render, RenderApp, RenderStartup, RenderSystems};

use super::model::{CASCADES, ContactGpu};
use super::{WaveGpu, WaveParams};
use crate::gpu::kernel::{Kernel, groups, queue_kernel, ready};
use crate::terrain::{TerrainDispatch, TerrainGpu};

impl ExtractResource<RenderApp> for WaveGpu {
    type Source = WaveGpu;

    fn extract_resource(source: &Self::Source) -> Self {
        source.clone()
    }
}

#[derive(Resource)]
struct WavePipelines {
    head: Kernel,
    force: Kernel,
    step: Kernel,
    publish: Kernel,
}

#[derive(Resource, Default)]
struct WaveUniforms {
    /// The first substep of a frame: last frame's origins, so texels entering a window start flat.
    first: UniformBuffer<WaveParams>,
    /// Every later substep: the windows have not moved since the first.
    rest: UniformBuffer<WaveParams>,
    contacts: StorageBuffer<Vec<ContactGpu>>,
}

/// This frame's bind groups, in dispatch order: the first substep's, then alternating.
#[derive(Resource)]
struct WaveBindGroups {
    first: BindGroup,
    /// From the buffer the first substep wrote into the other one, and back.
    forth: BindGroup,
    back: BindGroup,
}

/// What the field holds: the state buffers it was last run on, which of them holds the latest
/// substep, and that frame's window origins.
#[derive(Resource, Default)]
struct WaveMemory {
    buffer: Option<AssetId<ShaderBuffer>>,
    /// The latest substep is in `state_b`.
    latest_b: bool,
    origins: Option<[IVec2; CASCADES]>,
    /// This frame's uniform, as prepared.
    params: Option<WaveParams>,
}

pub(super) fn plugin(app: &mut App) {
    app.add_plugins(ExtractResourcePlugin::<WaveGpu>::default());
    let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
        return;
    };
    render_app
        .init_resource::<WaveUniforms>()
        .init_resource::<WaveMemory>()
        .add_systems(RenderStartup, init_pipelines)
        .add_systems(Render, prepare.in_set(RenderSystems::PrepareBindGroups))
        .add_systems(
            RenderGraph,
            dispatch.after(TerrainDispatch).before(camera_driver),
        );
}

fn layout() -> BindGroupLayoutDescriptor {
    BindGroupLayoutDescriptor::new(
        "elements_waves",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                uniform_buffer::<WaveParams>(false),
                storage_buffer_read_only_sized(false, None),
                storage_buffer_sized(false, None),
                storage_buffer_sized(false, None),
                storage_buffer_read_only_sized(false, None),
                storage_buffer_read_only_sized(false, None),
                storage_buffer_read_only_sized(false, None),
                texture_storage_2d_array(
                    TextureFormat::Rgba16Float,
                    StorageTextureAccess::WriteOnly,
                ),
                storage_buffer_sized(false, None),
            ),
        ),
    )
}

fn init_pipelines(mut commands: Commands, assets: Res<AssetServer>, cache: Res<PipelineCache>) {
    let shader = load_embedded_asset!(assets.as_ref(), "waves.wesl");
    let kernel = |entry| queue_kernel(&cache, shader.clone(), entry, layout(), Vec::new());
    commands.insert_resource(WavePipelines {
        head: kernel("wave_head"),
        force: kernel("wave_force"),
        step: kernel("wave_step"),
        publish: kernel("wave_publish"),
    });
}

#[allow(clippy::too_many_arguments)]
fn prepare(
    mut commands: Commands,
    gpu: Option<Res<WaveGpu>>,
    terrain: Option<Res<TerrainGpu>>,
    pipelines: Option<Res<WavePipelines>>,
    mut uniforms: ResMut<WaveUniforms>,
    mut memory: ResMut<WaveMemory>,
    buffers: Res<RenderAssets<GpuShaderBuffer>>,
    images: Res<RenderAssets<GpuImage>>,
    cache: Res<PipelineCache>,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
) {
    commands.remove_resource::<WaveBindGroups>();
    memory.params = None;
    let (Some(gpu), Some(terrain), Some(pipelines)) = (gpu, terrain, pipelines) else {
        return;
    };
    let get = |h: &Handle<ShaderBuffer>| buffers.get(h).map(|b| &b.buffer);
    let (Some(a), Some(b), Some(head), Some(force), Some(cells), Some(flow), Some(field)) = (
        get(&gpu.state_a),
        get(&gpu.state_b),
        get(&gpu.head),
        get(&gpu.force),
        get(&terrain.state.cells),
        get(&terrain.state.flow_cells),
        images.get(&gpu.field),
    ) else {
        gpu.ready.store(false, Ordering::Relaxed);
        return;
    };
    let compiled = ready(&cache, &pipelines.head).is_some()
        && ready(&cache, &pipelines.force).is_some()
        && ready(&cache, &pipelines.step).is_some()
        && ready(&cache, &pipelines.publish).is_some();
    gpu.ready.store(compiled, Ordering::Relaxed);
    if !compiled || gpu.substeps == 0 {
        return;
    }
    // New state buffers hold nothing: clear the whole field on their first run.
    if memory.buffer != Some(gpu.state_a.id()) {
        memory.buffer = Some(gpu.state_a.id());
        memory.latest_b = false;
        memory.origins = None;
    }
    let mut first = gpu.params;
    let mut rest = gpu.params;
    for c in 0..CASCADES {
        let now = first.origins[c].xy();
        let before = memory.origins.map_or(now, |o| o[c]);
        first.origins[c] = IVec4::new(now.x, now.y, before.x, before.y);
        rest.origins[c] = IVec4::new(now.x, now.y, now.x, now.y);
    }
    first.counts.z = u32::from(memory.origins.is_none());
    rest.counts.z = 0;
    memory.params = Some(first);

    uniforms.first.set(first);
    uniforms.first.write_buffer(&device, &queue);
    uniforms.rest.set(rest);
    uniforms.rest.write_buffer(&device, &queue);
    uniforms.contacts.set(gpu.contacts.clone());
    uniforms.contacts.write_buffer(&device, &queue);
    let Some(contacts) = uniforms.contacts.binding() else {
        return;
    };
    let layout = cache.get_bind_group_layout(&pipelines.step.layout);
    let group = |params: &UniformBuffer<WaveParams>, from: &Buffer, to: &Buffer| {
        device.create_bind_group(
            "elements_waves",
            &layout,
            &BindGroupEntries::sequential((
                params,
                from.as_entire_buffer_binding(),
                to.as_entire_buffer_binding(),
                head.as_entire_buffer_binding(),
                contacts.clone(),
                cells.as_entire_buffer_binding(),
                flow.as_entire_buffer_binding(),
                &field.texture_view,
                force.as_entire_buffer_binding(),
            )),
        )
    };
    let (latest, other) = if memory.latest_b { (b, a) } else { (a, b) };
    commands.insert_resource(WaveBindGroups {
        first: group(&uniforms.first, latest, other),
        forth: group(&uniforms.rest, other, latest),
        back: group(&uniforms.rest, latest, other),
    });
}

fn dispatch(
    mut ctx: RenderContext,
    cache: Res<PipelineCache>,
    gpu: Option<Res<WaveGpu>>,
    pipelines: Option<Res<WavePipelines>>,
    groups_: Option<Res<WaveBindGroups>>,
    mut memory: ResMut<WaveMemory>,
) {
    let (Some(gpu), Some(pipelines), Some(bind), Some(params)) =
        (gpu, pipelines, groups_, memory.params)
    else {
        return;
    };
    let (Some(head), Some(force), Some(step), Some(publish)) = (
        ready(&cache, &pipelines.head),
        ready(&cache, &pipelines.force),
        ready(&cache, &pipelines.step),
        ready(&cache, &pipelines.publish),
    ) else {
        return;
    };
    let size = params.counts.x;
    let substeps = gpu.substeps;
    let diagnostics = ctx.diagnostic_recorder();
    let diagnostics = diagnostics.as_deref();
    // Three passes, each timed: the bodies (head and forcing), the substeps, the publish.
    let mut pass = ctx
        .command_encoder()
        .begin_compute_pass(&ComputePassDescriptor {
            label: Some("elements_waves_force"),
            ..default()
        });
    let span = diagnostics.pass_span(&mut pass, "elements_waves_force");
    pass.set_bind_group(0, &bind.first, &[]);
    if params.counts.y > 0 && params.counts.w > 0 {
        let tiles = groups(params.counts.w, 8);
        pass.set_pipeline(head);
        pass.dispatch_workgroups(tiles, tiles, params.counts.y * (2 * CASCADES as u32 - 1));
    }
    // Always: with no bodies the forcing must return to zero.
    pass.set_pipeline(force);
    pass.dispatch_workgroups(groups(size, 8), groups(size, 8), CASCADES as u32);
    span.end(&mut pass);
    drop(pass);

    let mut pass = ctx
        .command_encoder()
        .begin_compute_pass(&ComputePassDescriptor {
            label: Some("elements_waves_step"),
            ..default()
        });
    let span = diagnostics.pass_span(&mut pass, "elements_waves_step");
    pass.set_pipeline(step);
    let mut last = &bind.first;
    for i in 0..substeps {
        last = match i {
            0 => &bind.first,
            i if i % 2 == 1 => &bind.forth,
            _ => &bind.back,
        };
        pass.set_bind_group(0, last, &[]);
        // A workgroup steps a 32 × 32 block (`waves.wesl` `wave_step`).
        pass.dispatch_workgroups(groups(size, 32), groups(size, 32), CASCADES as u32);
    }
    span.end(&mut pass);
    drop(pass);

    let mut pass = ctx
        .command_encoder()
        .begin_compute_pass(&ComputePassDescriptor {
            label: Some("elements_waves_publish"),
            ..default()
        });
    let span = diagnostics.pass_span(&mut pass, "elements_waves_publish");
    // The publish kernel reads the last substep's output binding.
    pass.set_bind_group(0, last, &[]);
    pass.set_pipeline(publish);
    pass.dispatch_workgroups(groups(size, 8), groups(size, 8), CASCADES as u32);
    span.end(&mut pass);
    drop(pass);
    if substeps % 2 == 1 {
        memory.latest_b = !memory.latest_b;
    }
    memory.origins = Some(std::array::from_fn(|c| params.origins[c].xy()));
    gpu.executed.fetch_add(1, Ordering::Relaxed);
}
