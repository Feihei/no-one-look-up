//! Render-world side of the precipitation: the particle kernel, and the draw pass that blends
//! streaks, flakes and splash rings into the HDR main texture.

use bevy::asset::load_embedded_asset;
use bevy::camera::{MainPassResolutionOverride, Viewport};
use bevy::core_pipeline::Core3dSystems;
use bevy::core_pipeline::prepass::ViewPrepassTextures;
use bevy::core_pipeline::schedule::{Core3d, camera_driver};
use bevy::prelude::*;
use bevy::render::camera::ExtractedCamera;
use bevy::render::diagnostic::RecordDiagnostics;
use bevy::render::extract_component::ExtractComponentPlugin;
use bevy::render::extract_resource::{ExtractResource, ExtractResourcePlugin};
use bevy::render::render_asset::RenderAssets;
use bevy::render::render_resource::binding_types::{
    sampler, storage_buffer_read_only_sized, storage_buffer_sized, texture_3d, texture_depth_2d,
    uniform_buffer,
};
use bevy::render::render_resource::{
    BindGroup, BindGroupEntries, BindGroupLayoutDescriptor, BindGroupLayoutEntries, BlendState,
    CachedRenderPipelineId, ColorTargetState, ColorWrites, ComputePassDescriptor, FragmentState,
    LoadOp, Operations, PipelineCache, RenderPassColorAttachment, RenderPassDescriptor,
    RenderPipelineDescriptor, SamplerBindingType, ShaderStages, StorageBuffer, StoreOp,
    TextureFormat, TextureSampleType, UniformBuffer, VertexState,
};
use bevy::render::renderer::{RenderContext, RenderDevice, RenderGraph, RenderQueue, ViewQuery};
use bevy::render::storage::GpuShaderBuffer;
use bevy::render::texture::GpuImage;
use bevy::render::view::{Msaa, ViewTarget, ViewUniform, ViewUniformOffset, ViewUniforms};
use bevy::render::{Render, RenderApp, RenderStartup, RenderSystems};

use super::{
    BlockerGpu, DrawParams, PrecipParams, PrecipitationGpu, PrecipitationView, SPLASH_RING,
};
use crate::gpu::kernel::{Kernel, groups, queue_kernel, ready};

/// Threads per workgroup of the particle kernel (`@workgroup_size(64, 1, 1)`).
const WORKGROUP: u32 = 64;

impl ExtractResource<RenderApp> for PrecipitationGpu {
    type Source = PrecipitationGpu;

    fn extract_resource(source: &Self::Source) -> Self {
        source.clone()
    }
}

#[derive(Resource)]
struct PrecipitationPipelines {
    sim: Kernel,
    draw_layout: BindGroupLayoutDescriptor,
    particles: CachedRenderPipelineId,
    splashes: CachedRenderPipelineId,
}

#[derive(Resource, Default)]
struct PrecipitationUniforms {
    sim: UniformBuffer<PrecipParams>,
    draw: UniformBuffer<DrawParams>,
    blockers: StorageBuffer<Vec<BlockerGpu>>,
}

#[derive(Resource)]
struct SimBindGroup(BindGroup);

pub(super) fn plugin(app: &mut App) {
    app.add_plugins((
        ExtractResourcePlugin::<PrecipitationGpu>::default(),
        ExtractComponentPlugin::<PrecipitationView>::default(),
    ));
    let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
        return;
    };
    render_app
        .init_resource::<PrecipitationUniforms>()
        .add_systems(RenderStartup, init_pipelines)
        .add_systems(Render, prepare.in_set(RenderSystems::PrepareBindGroups))
        .add_systems(
            RenderGraph,
            simulate
                .after(crate::terrain::TerrainDispatch)
                .before(camera_driver),
        )
        .add_systems(Core3d, draw.in_set(Core3dSystems::EarlyPostProcess));
}

fn sim_layout() -> BindGroupLayoutDescriptor {
    BindGroupLayoutDescriptor::new(
        "elements_precip_sim",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                uniform_buffer::<PrecipParams>(false),
                storage_buffer_sized(false, None),
                storage_buffer_sized(false, None),
                storage_buffer_read_only_sized(false, None),
                texture_3d(TextureSampleType::Float { filterable: true }),
                sampler(SamplerBindingType::Filtering),
                storage_buffer_read_only_sized(false, None),
            ),
        ),
    )
}

fn draw_layout() -> BindGroupLayoutDescriptor {
    BindGroupLayoutDescriptor::new(
        "elements_precip_draw",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::VERTEX_FRAGMENT,
            (
                uniform_buffer::<ViewUniform>(true),
                uniform_buffer::<DrawParams>(false),
                storage_buffer_read_only_sized(false, None).visibility(ShaderStages::VERTEX),
                storage_buffer_read_only_sized(false, None).visibility(ShaderStages::VERTEX),
                texture_depth_2d().visibility(ShaderStages::FRAGMENT),
            ),
        ),
    )
}

fn init_pipelines(mut commands: Commands, assets: Res<AssetServer>, cache: Res<PipelineCache>) {
    let sim_shader = load_embedded_asset!(assets.as_ref(), "precip.wesl");
    let sim = queue_kernel(
        &cache,
        sim_shader,
        "elements_precip_sim",
        sim_layout(),
        Vec::new(),
    );

    let draw_shader = load_embedded_asset!(assets.as_ref(), "draw.wesl");
    let layout = draw_layout();
    let queue = |label: &'static str, vertex: &'static str| {
        cache.queue_render_pipeline(RenderPipelineDescriptor {
            label: Some(label.into()),
            layout: vec![layout.clone()],
            vertex: VertexState {
                shader: draw_shader.clone(),
                entry_point: Some(vertex.into()),
                ..default()
            },
            fragment: Some(FragmentState {
                shader: draw_shader.clone(),
                entry_point: Some("particle_fragment".into()),
                targets: vec![Some(ColorTargetState {
                    format: TextureFormat::Rgba16Float,
                    blend: Some(BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                    write_mask: ColorWrites::ALL,
                })],
                ..default()
            }),
            ..default()
        })
    };
    let particles = queue("elements_precip_particles", "particle_vertex");
    let splashes = queue("elements_precip_splashes", "splash_vertex");
    commands.insert_resource(PrecipitationPipelines {
        sim,
        draw_layout: layout,
        particles,
        splashes,
    });
}

#[allow(clippy::too_many_arguments)]
fn prepare(
    mut commands: Commands,
    gpu: Option<Res<PrecipitationGpu>>,
    pipelines: Option<Res<PrecipitationPipelines>>,
    mut uniforms: ResMut<PrecipitationUniforms>,
    buffers: Res<RenderAssets<GpuShaderBuffer>>,
    images: Res<RenderAssets<GpuImage>>,
    cache: Res<PipelineCache>,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
) {
    let (Some(gpu), Some(pipelines)) = (gpu, pipelines) else {
        return;
    };
    if !gpu.active {
        commands.remove_resource::<SimBindGroup>();
        return;
    }
    let (Some(particles), Some(splashes), Some(wind), Some(terrain)) = (
        buffers.get(&gpu.particles),
        buffers.get(&gpu.splashes),
        images.get(&gpu.wind_field),
        buffers.get(&gpu.terrain),
    ) else {
        commands.remove_resource::<SimBindGroup>();
        return;
    };
    uniforms.sim.set(gpu.params);
    uniforms.sim.write_buffer(&device, &queue);
    uniforms.draw.set(gpu.draw);
    uniforms.draw.write_buffer(&device, &queue);
    uniforms.blockers.set(gpu.blockers.clone());
    uniforms.blockers.write_buffer(&device, &queue);
    let Some(blockers) = uniforms.blockers.binding() else {
        commands.remove_resource::<SimBindGroup>();
        return;
    };
    let group = device.create_bind_group(
        "elements_precip_sim",
        &cache.get_bind_group_layout(&pipelines.sim.layout),
        &BindGroupEntries::sequential((
            &uniforms.sim,
            particles.buffer.as_entire_buffer_binding(),
            splashes.buffer.as_entire_buffer_binding(),
            blockers,
            &wind.texture_view,
            &wind.sampler,
            terrain.buffer.as_entire_buffer_binding(),
        )),
    );
    commands.insert_resource(SimBindGroup(group));
}

fn simulate(
    mut ctx: RenderContext,
    cache: Res<PipelineCache>,
    gpu: Option<Res<PrecipitationGpu>>,
    pipelines: Option<Res<PrecipitationPipelines>>,
    group: Option<Res<SimBindGroup>>,
) {
    let (Some(gpu), Some(pipelines), Some(group)) = (gpu, pipelines, group) else {
        return;
    };
    if !gpu.active {
        return;
    }
    let Some(kernel) = ready(&cache, &pipelines.sim) else {
        return;
    };
    let diagnostics = ctx.diagnostic_recorder();
    let diagnostics = diagnostics.as_deref();
    let mut pass = ctx
        .command_encoder()
        .begin_compute_pass(&ComputePassDescriptor {
            label: Some("elements_precip_sim"),
            ..default()
        });
    let span = diagnostics.pass_span(&mut pass, "elements_precip_sim");
    pass.set_pipeline(kernel);
    pass.set_bind_group(0, &group.0, &[]);
    pass.dispatch_workgroups(groups(gpu.capacity, WORKGROUP), 1, 1);
    span.end(&mut pass);
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn draw(
    view: ViewQuery<(
        &ViewTarget,
        &ViewUniformOffset,
        &ViewPrepassTextures,
        &ExtractedCamera,
        Option<&MainPassResolutionOverride>,
        &Msaa,
        &PrecipitationView,
    )>,
    cache: Res<PipelineCache>,
    gpu: Option<Res<PrecipitationGpu>>,
    pipelines: Option<Res<PrecipitationPipelines>>,
    uniforms: Res<PrecipitationUniforms>,
    view_uniforms: Res<ViewUniforms>,
    buffers: Res<RenderAssets<GpuShaderBuffer>>,
    mut ctx: RenderContext,
    mut warned: Local<bool>,
) {
    let (Some(gpu), Some(pipelines)) = (gpu, pipelines) else {
        return;
    };
    if !gpu.active || gpu.drawn == 0 {
        return;
    }
    let (target, view_offset, prepass, camera, resolution_override, msaa, _) = view.into_inner();
    if *msaa != Msaa::Off || target.main_texture_format() != TextureFormat::Rgba16Float {
        if !*warned {
            warn!(
                "precipitation is not drawn on a view with {msaa:?} or a {:?} target; it needs Msaa::Off and an HDR camera",
                target.main_texture_format()
            );
            *warned = true;
        }
        return;
    }
    let (Some(particle_pipeline), Some(splash_pipeline), Some(view_binding), Some(depth)) = (
        cache.get_render_pipeline(pipelines.particles),
        cache.get_render_pipeline(pipelines.splashes),
        view_uniforms.uniforms.binding(),
        prepass.depth_only_view(),
    ) else {
        return;
    };
    let (Some(particles), Some(splashes), Some(draw_params)) = (
        buffers.get(&gpu.particles),
        buffers.get(&gpu.splashes),
        uniforms.draw.binding(),
    ) else {
        return;
    };
    let group = ctx.render_device().create_bind_group(
        "elements_precip_draw",
        &cache.get_bind_group_layout(&pipelines.draw_layout),
        &BindGroupEntries::sequential((
            view_binding,
            draw_params,
            particles.buffer.as_entire_buffer_binding(),
            splashes.buffer.as_entire_buffer_binding(),
            depth,
        )),
    );

    let diagnostics = ctx.diagnostic_recorder();
    let diagnostics = diagnostics.as_deref();
    let mut pass = ctx
        .command_encoder()
        .begin_render_pass(&RenderPassDescriptor {
            label: Some("elements_precip_draw"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view: target.main_texture_view(),
                depth_slice: None,
                resolve_target: None,
                ops: Operations {
                    load: LoadOp::Load,
                    store: StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
    let span = diagnostics.pass_span(&mut pass, "elements_precip_draw");
    if let Some(viewport) =
        Viewport::from_viewport_and_override(camera.viewport.as_ref(), resolution_override)
    {
        pass.set_viewport(
            viewport.physical_position.x as f32,
            viewport.physical_position.y as f32,
            viewport.physical_size.x as f32,
            viewport.physical_size.y as f32,
            viewport.depth.start,
            viewport.depth.end,
        );
    }
    pass.set_bind_group(0, &group, &[view_offset.offset]);
    pass.set_pipeline(particle_pipeline);
    pass.draw(0..6, 0..gpu.drawn);
    if gpu.params.counts.x > 0 {
        pass.set_pipeline(splash_pipeline);
        pass.draw(0..6, 0..SPLASH_RING as u32);
    }
    span.end(&mut pass);
}

#[cfg(test)]
mod tests {
    use bevy::render::render_resource::{BindingType, BufferBindingType};

    /// Terrain validation 9: the particle kernel binds four storage buffers of WebGPU's eight
    /// (particles, splashes, blockers, terrain).
    #[test]
    fn the_particle_layout_binds_four_storage_buffers() {
        let storage = super::sim_layout()
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
        assert_eq!(storage, 4);
    }
}
