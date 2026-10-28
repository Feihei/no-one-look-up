//! Render-world side of the clouds: pipelines, bind groups and the dispatches.
//!
//! View-independent kernels (noise, weather map, shadow map) run in the `RenderGraph` schedule
//! before any camera draws. The per-view march and resolve run in `Core3d` before the opaque
//! pass (the depth prepass and Bevy's atmosphere LUTs, both finished before `MainPass`, are
//! ready), and the composite draws after the transparent pass.

use std::sync::atomic::Ordering;

use bevy::asset::load_embedded_asset;
use bevy::core_pipeline::core_3d::{main_opaque_pass_3d, main_transparent_pass_3d};
use bevy::core_pipeline::prepass::{
    PreviousViewData, PreviousViewUniformOffset, PreviousViewUniforms, ViewPrepassTextures,
};
use bevy::core_pipeline::schedule::camera_driver;
use bevy::core_pipeline::{Core3d, Core3dSystems, FullscreenShader};
use bevy::pbr::resources::AtmosphereTextures;
use bevy::prelude::*;
use bevy::render::diagnostic::RecordDiagnostics;
use bevy::render::extract_component::ExtractComponentPlugin;
use bevy::render::extract_resource::{ExtractResource, ExtractResourcePlugin};
use bevy::render::render_asset::RenderAssets;
use bevy::render::render_resource::binding_types::{
    sampler, storage_buffer_sized, texture_2d, texture_3d, texture_depth_2d, texture_storage_2d,
    texture_storage_3d, uniform_buffer,
};
use bevy::render::render_resource::{
    AddressMode, BindGroup, BindGroupEntries, BindGroupLayoutDescriptor, BindGroupLayoutEntries,
    BlendComponent, BlendFactor, BlendOperation, BlendState, CachedRenderPipelineId,
    ColorTargetState, ColorWrites, ComputePassDescriptor, FilterMode, FragmentState, LoadOp,
    MipmapFilterMode, Operations, PipelineCache, RenderPassColorAttachment, RenderPassDescriptor,
    RenderPipelineDescriptor, Sampler, SamplerBindingType, SamplerDescriptor, ShaderStages,
    StorageTextureAccess, StoreOp, TextureFormat, TextureSampleType, UniformBuffer,
};
use bevy::render::renderer::{RenderContext, RenderDevice, RenderGraph, RenderQueue, ViewQuery};
use bevy::render::storage::GpuShaderBuffer;
use bevy::render::texture::GpuImage;
use bevy::render::view::{ViewTarget, ViewUniform, ViewUniformOffset, ViewUniforms};
use bevy::render::{Render, RenderApp, RenderStartup, RenderSystems};
use bevy::shader::Shader;

use super::{CloudGpu, CloudParams, CloudView, NOISE_BASE_TEXELS, NOISE_DETAIL_TEXELS};
use crate::gpu::kernel::{Kernel, WG3, groups, queue_kernel, ready};

impl ExtractResource<RenderApp> for CloudGpu {
    type Source = CloudGpu;

    fn extract_resource(source: &Self::Source) -> Self {
        source.clone()
    }
}

/// The base noise is produced in this many z slabs, one per frame.
pub(crate) const NOISE_PASSES: u32 = 4;
/// Slices of the base noise in one slab.
const NOISE_SLAB: u32 = NOISE_BASE_TEXELS / NOISE_PASSES;
/// Workgroup edge of the 2D kernels.
const WG2: u32 = 8;

#[derive(Resource)]
struct CloudPipelines {
    noise_base: Kernel,
    noise_detail: Kernel,
    histogram_clear: Kernel,
    field: Kernel,
    threshold: Kernel,
    weather: Kernel,
    shadow: Kernel,
    shadow_camera: Kernel,
    march: Kernel,
    resolve: Kernel,
    composite: CachedRenderPipelineId,
    composite_layout: BindGroupLayoutDescriptor,
    repeat: Sampler,
    clamp: Sampler,
}

/// How many noise slabs exist.
#[derive(Resource, Default)]
struct CloudProgress {
    noise_passes: u32,
}

#[derive(Resource, Default)]
struct CloudUniform(UniformBuffer<CloudParams>);

#[derive(Resource)]
struct CloudGlobalGroups {
    noise: BindGroup,
    weather: BindGroup,
    shadow: BindGroup,
}

/// Bind groups of the per-view passes, rebuilt every frame (the atmosphere LUTs and the view
/// uniform buffers can be re-created between frames).
#[derive(Component)]
struct CloudViewGroups {
    march: BindGroup,
    resolve: BindGroup,
    composite: BindGroup,
}

pub(super) fn plugin(app: &mut App) {
    app.add_plugins((
        ExtractResourcePlugin::<CloudGpu>::default(),
        ExtractComponentPlugin::<CloudView>::default(),
    ));
    let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
        return;
    };
    render_app
        .init_resource::<CloudUniform>()
        .init_resource::<CloudProgress>()
        .add_systems(RenderStartup, init_pipelines)
        .add_systems(
            Render,
            (prepare_global, prepare_views)
                .chain()
                .in_set(RenderSystems::PrepareBindGroups),
        )
        .add_systems(RenderGraph, dispatch_global.before(camera_driver))
        .add_systems(
            Core3d,
            (
                (cloud_march, cloud_resolve)
                    .chain()
                    .in_set(Core3dSystems::MainPass)
                    .before(main_opaque_pass_3d),
                cloud_composite
                    .in_set(Core3dSystems::MainPass)
                    .after(main_transparent_pass_3d),
            ),
        );
}

macro_rules! float_2d {
    () => {
        texture_2d(TextureSampleType::Float { filterable: true })
    };
}

macro_rules! float_3d {
    () => {
        texture_3d(TextureSampleType::Float { filterable: true })
    };
}

fn init_pipelines(
    mut commands: Commands,
    assets: Res<AssetServer>,
    cache: Res<PipelineCache>,
    device: Res<RenderDevice>,
    fullscreen: Res<FullscreenShader>,
) {
    let write = StorageTextureAccess::WriteOnly;
    let compute = ShaderStages::COMPUTE;
    let noise_layout = BindGroupLayoutDescriptor::new(
        "elements_clouds_noise",
        &BindGroupLayoutEntries::sequential(
            compute,
            (
                uniform_buffer::<CloudParams>(false),
                texture_storage_3d(TextureFormat::Rgba8Unorm, write),
                texture_storage_3d(TextureFormat::Rgba8Unorm, write),
            ),
        ),
    );
    let weather_layout = BindGroupLayoutDescriptor::new(
        "elements_clouds_weather",
        &BindGroupLayoutEntries::sequential(
            compute,
            (
                uniform_buffer::<CloudParams>(false),
                storage_buffer_sized(false, None),
                storage_buffer_sized(false, None),
                texture_storage_2d(TextureFormat::Rgba8Unorm, write),
            ),
        ),
    );
    let shadow_layout = BindGroupLayoutDescriptor::new(
        "elements_clouds_shadow",
        &BindGroupLayoutEntries::sequential(
            compute,
            (
                uniform_buffer::<CloudParams>(false),
                float_2d!(),
                float_3d!(),
                sampler(SamplerBindingType::Filtering),
                texture_storage_2d(TextureFormat::Rgba16Float, write),
                storage_buffer_sized(false, None),
                storage_buffer_sized(false, None),
            ),
        ),
    );
    let march_layout = BindGroupLayoutDescriptor::new(
        "elements_clouds_march",
        &BindGroupLayoutEntries::sequential(
            compute,
            (
                uniform_buffer::<ViewUniform>(true),
                uniform_buffer::<CloudParams>(false),
                float_3d!(),
                float_3d!(),
                float_2d!(),
                float_2d!(),
                float_3d!(),
                texture_depth_2d(),
                sampler(SamplerBindingType::Filtering),
                sampler(SamplerBindingType::Filtering),
                texture_storage_2d(TextureFormat::Rgba16Float, write),
            ),
        ),
    );
    let resolve_layout = BindGroupLayoutDescriptor::new(
        "elements_clouds_resolve",
        &BindGroupLayoutEntries::sequential(
            compute,
            (
                uniform_buffer::<ViewUniform>(true),
                uniform_buffer::<PreviousViewData>(true),
                uniform_buffer::<CloudParams>(false),
                float_2d!(),
                float_2d!(),
                sampler(SamplerBindingType::Filtering),
                texture_storage_2d(TextureFormat::Rgba16Float, write),
            ),
        ),
    );
    let composite_layout = BindGroupLayoutDescriptor::new(
        "elements_clouds_composite",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::FRAGMENT,
            (
                uniform_buffer::<ViewUniform>(true),
                uniform_buffer::<CloudParams>(false),
                float_2d!(),
                texture_depth_2d(),
            ),
        ),
    );

    let kernel = |file: Handle<Shader>, entry: &'static str, layout: &BindGroupLayoutDescriptor| {
        queue_kernel(&cache, file, entry, layout.clone(), Vec::new())
    };
    let generate = load_embedded_asset!(assets.as_ref(), "generate.wesl");
    let weather = load_embedded_asset!(assets.as_ref(), "weather.wesl");
    let shadow = load_embedded_asset!(assets.as_ref(), "shadow.wesl");
    let march = load_embedded_asset!(assets.as_ref(), "march.wesl");
    let resolve = load_embedded_asset!(assets.as_ref(), "resolve.wesl");
    let composite_shader = load_embedded_asset!(assets.as_ref(), "composite.wesl");

    // `scene · T + L`: the shader outputs (L, T); colour `src + dst · src.a`, alpha untouched.
    let blend = BlendState {
        color: BlendComponent {
            src_factor: BlendFactor::One,
            dst_factor: BlendFactor::SrcAlpha,
            operation: BlendOperation::Add,
        },
        alpha: BlendComponent {
            src_factor: BlendFactor::Zero,
            dst_factor: BlendFactor::One,
            operation: BlendOperation::Add,
        },
    };
    let composite = cache.queue_render_pipeline(RenderPipelineDescriptor {
        label: Some("elements_clouds_composite".into()),
        layout: vec![composite_layout.clone()],
        vertex: fullscreen.to_vertex_state(),
        fragment: Some(FragmentState {
            shader: composite_shader,
            entry_point: Some("cloud_composite".into()),
            targets: vec![Some(ColorTargetState {
                // `ElementsCamera` requires `Hdr`.
                format: TextureFormat::Rgba16Float,
                blend: Some(blend),
                write_mask: ColorWrites::ALL,
            })],
            ..default()
        }),
        ..default()
    });

    let make_sampler = |address: AddressMode| {
        device.create_sampler(&SamplerDescriptor {
            address_mode_u: address,
            address_mode_v: address,
            address_mode_w: address,
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            mipmap_filter: MipmapFilterMode::Linear,
            ..default()
        })
    };
    commands.insert_resource(CloudPipelines {
        noise_base: kernel(generate.clone(), "cloud_noise_base", &noise_layout),
        noise_detail: kernel(generate, "cloud_noise_detail", &noise_layout),
        histogram_clear: kernel(weather.clone(), "cloud_histogram_clear", &weather_layout),
        field: kernel(weather.clone(), "cloud_field", &weather_layout),
        threshold: kernel(weather.clone(), "cloud_threshold", &weather_layout),
        weather: kernel(weather, "cloud_weather", &weather_layout),
        shadow: kernel(shadow.clone(), "cloud_shadow", &shadow_layout),
        shadow_camera: kernel(shadow, "cloud_shadow_camera", &shadow_layout),
        march: kernel(march, "cloud_march", &march_layout),
        resolve: kernel(resolve, "cloud_resolve", &resolve_layout),
        composite,
        composite_layout,
        repeat: make_sampler(AddressMode::Repeat),
        clamp: make_sampler(AddressMode::ClampToEdge),
    });
}

#[allow(clippy::too_many_arguments)]
fn prepare_global(
    mut commands: Commands,
    cloud: Option<Res<CloudGpu>>,
    pipelines: Option<Res<CloudPipelines>>,
    progress: Res<CloudProgress>,
    mut uniform: ResMut<CloudUniform>,
    images: Res<RenderAssets<GpuImage>>,
    buffers: Res<RenderAssets<GpuShaderBuffer>>,
    cache: Res<PipelineCache>,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
) {
    let (Some(cloud), Some(pipelines)) = (cloud, pipelines) else {
        return;
    };
    let mut params = cloud.params;
    params.tail.x = progress.noise_passes.min(NOISE_PASSES - 1) * NOISE_SLAB;
    uniform.0.set(params);
    uniform.0.write_buffer(&device, &queue);

    let buffer = |h| buffers.get(h).map(|b| b.buffer.as_entire_buffer_binding());
    let (Some(base), Some(detail), Some(weather), Some(shadow)) = (
        images.get(&cloud.base),
        images.get(&cloud.detail),
        images.get(&cloud.weather),
        images.get(&cloud.shadow),
    ) else {
        commands.remove_resource::<CloudGlobalGroups>();
        return;
    };
    let (Some(histogram), Some(threshold), Some(camera_t), Some(shadow_coarse)) = (
        buffer(&cloud.histogram),
        buffer(&cloud.threshold),
        buffer(&cloud.camera_t),
        buffer(&cloud.shadow_coarse),
    ) else {
        commands.remove_resource::<CloudGlobalGroups>();
        return;
    };
    let noise = device.create_bind_group(
        "elements_clouds_noise",
        &cache.get_bind_group_layout(&pipelines.noise_base.layout),
        &BindGroupEntries::sequential((&uniform.0, &base.texture_view, &detail.texture_view)),
    );
    let weather_group = device.create_bind_group(
        "elements_clouds_weather",
        &cache.get_bind_group_layout(&pipelines.field.layout),
        &BindGroupEntries::sequential((&uniform.0, histogram, threshold, &weather.texture_view)),
    );
    let shadow_group = device.create_bind_group(
        "elements_clouds_shadow",
        &cache.get_bind_group_layout(&pipelines.shadow.layout),
        &BindGroupEntries::sequential((
            &uniform.0,
            &weather.texture_view,
            &base.texture_view,
            &pipelines.repeat,
            &shadow.texture_view,
            camera_t,
            shadow_coarse,
        )),
    );
    commands.insert_resource(CloudGlobalGroups {
        noise,
        weather: weather_group,
        shadow: shadow_group,
    });
}

#[allow(clippy::too_many_arguments)]
fn prepare_views(
    mut commands: Commands,
    cloud: Option<Res<CloudGpu>>,
    pipelines: Option<Res<CloudPipelines>>,
    uniform: Res<CloudUniform>,
    view_uniforms: Res<ViewUniforms>,
    previous_uniforms: Res<PreviousViewUniforms>,
    images: Res<RenderAssets<GpuImage>>,
    cache: Res<PipelineCache>,
    device: Res<RenderDevice>,
    views: Query<(Entity, &ViewPrepassTextures, &AtmosphereTextures), With<CloudView>>,
) {
    let (Some(cloud), Some(pipelines)) = (cloud, pipelines) else {
        return;
    };
    let (Some(images_view), Some(view_binding), Some(previous_binding)) = (
        cloud.view.as_ref(),
        view_uniforms.uniforms.binding(),
        previous_uniforms.uniforms.binding(),
    ) else {
        return;
    };
    let (Some(base), Some(detail), Some(weather), Some(march), Some(history_a), Some(history_b)) = (
        images.get(&cloud.base),
        images.get(&cloud.detail),
        images.get(&cloud.weather),
        images.get(&images_view.march),
        images.get(&images_view.history[0]),
        images.get(&images_view.history[1]),
    ) else {
        return;
    };
    // The resolve writes history[frame % 2] and reads the other; the composite shows the one
    // just written.
    let (write, read) = if cloud.params.counts.x.is_multiple_of(2) {
        (history_a, history_b)
    } else {
        (history_b, history_a)
    };
    for (entity, prepass, atmosphere) in &views {
        let Some(depth) = prepass.depth_only_view() else {
            continue;
        };
        let march_group = device.create_bind_group(
            "elements_clouds_march",
            &cache.get_bind_group_layout(&pipelines.march.layout),
            &BindGroupEntries::sequential((
                view_binding.clone(),
                &uniform.0,
                &base.texture_view,
                &detail.texture_view,
                &weather.texture_view,
                &atmosphere.transmittance_lut.default_view,
                &atmosphere.aerial_view_lut.default_view,
                depth,
                &pipelines.repeat,
                &pipelines.clamp,
                &march.texture_view,
            )),
        );
        let resolve_group = device.create_bind_group(
            "elements_clouds_resolve",
            &cache.get_bind_group_layout(&pipelines.resolve.layout),
            &BindGroupEntries::sequential((
                view_binding.clone(),
                previous_binding.clone(),
                &uniform.0,
                &march.texture_view,
                &read.texture_view,
                &pipelines.clamp,
                &write.texture_view,
            )),
        );
        let composite_group = device.create_bind_group(
            "elements_clouds_composite",
            &cache.get_bind_group_layout(&pipelines.composite_layout),
            &BindGroupEntries::sequential((
                view_binding.clone(),
                &uniform.0,
                &write.texture_view,
                depth,
            )),
        );
        commands.entity(entity).insert(CloudViewGroups {
            march: march_group,
            resolve: resolve_group,
            composite: composite_group,
        });
    }
}

fn dispatch_global(
    mut ctx: RenderContext,
    cache: Res<PipelineCache>,
    cloud: Option<Res<CloudGpu>>,
    pipelines: Option<Res<CloudPipelines>>,
    bind: Option<Res<CloudGlobalGroups>>,
    mut progress: ResMut<CloudProgress>,
) {
    let (Some(cloud), Some(pipelines), Some(bind)) = (cloud, pipelines, bind) else {
        return;
    };
    if !cloud.active {
        return;
    }
    let diagnostics = ctx.diagnostic_recorder();
    let diagnostics = diagnostics.as_deref();

    if progress.noise_passes < NOISE_PASSES {
        let (Some(base), Some(detail)) = (
            ready(&cache, &pipelines.noise_base),
            ready(&cache, &pipelines.noise_detail),
        ) else {
            return;
        };
        let mut pass = ctx
            .command_encoder()
            .begin_compute_pass(&ComputePassDescriptor {
                label: Some("elements_clouds_noise"),
                ..default()
            });
        let span = diagnostics.pass_span(&mut pass, "elements_clouds_noise");
        pass.set_bind_group(0, &bind.noise, &[]);
        pass.set_pipeline(base);
        pass.dispatch_workgroups(
            groups(NOISE_BASE_TEXELS, WG3),
            groups(NOISE_BASE_TEXELS, WG3),
            groups(NOISE_SLAB, WG3),
        );
        if progress.noise_passes == 0 {
            pass.set_pipeline(detail);
            let n = groups(NOISE_DETAIL_TEXELS, WG3);
            pass.dispatch_workgroups(n, n, n);
        }
        span.end(&mut pass);
        progress.noise_passes += 1;
        return;
    }

    let (Some(clear), Some(field), Some(threshold), Some(weather)) = (
        ready(&cache, &pipelines.histogram_clear),
        ready(&cache, &pipelines.field),
        ready(&cache, &pipelines.threshold),
        ready(&cache, &pipelines.weather),
    ) else {
        return;
    };
    let (Some(shadow), Some(shadow_camera)) = (
        ready(&cache, &pipelines.shadow),
        ready(&cache, &pipelines.shadow_camera),
    ) else {
        return;
    };
    {
        let mut pass = ctx
            .command_encoder()
            .begin_compute_pass(&ComputePassDescriptor {
                label: Some("elements_clouds_weather"),
                ..default()
            });
        let span = diagnostics.pass_span(&mut pass, "elements_clouds_weather");
        pass.set_bind_group(0, &bind.weather, &[]);
        let n = groups(super::WEATHER_TEXELS, WG2);
        pass.set_pipeline(clear);
        pass.dispatch_workgroups(1, 1, 1);
        pass.set_pipeline(field);
        pass.dispatch_workgroups(n, n, 1);
        pass.set_pipeline(threshold);
        pass.dispatch_workgroups(1, 1, 1);
        pass.set_pipeline(weather);
        pass.dispatch_workgroups(n, n, 1);
        span.end(&mut pass);
    }
    {
        let mut pass = ctx
            .command_encoder()
            .begin_compute_pass(&ComputePassDescriptor {
                label: Some("elements_clouds_shadow"),
                ..default()
            });
        let span = diagnostics.pass_span(&mut pass, "elements_clouds_shadow");
        pass.set_bind_group(0, &bind.shadow, &[]);
        let n = groups(super::SHADOW_TEXELS, WG2);
        pass.set_pipeline(shadow);
        pass.dispatch_workgroups(n, n, 1);
        pass.set_pipeline(shadow_camera);
        pass.dispatch_workgroups(1, 1, 1);
        span.end(&mut pass);
    }
    cloud.live.store(true, Ordering::Relaxed);
}

fn noise_ready(progress: &CloudProgress) -> bool {
    progress.noise_passes >= NOISE_PASSES
}

fn cloud_march(
    view: ViewQuery<(&ViewUniformOffset, &CloudViewGroups)>,
    cloud: Option<Res<CloudGpu>>,
    pipelines: Option<Res<CloudPipelines>>,
    progress: Res<CloudProgress>,
    cache: Res<PipelineCache>,
    mut ctx: RenderContext,
) {
    let (Some(cloud), Some(pipelines)) = (cloud, pipelines) else {
        return;
    };
    if !cloud.active || !noise_ready(&progress) {
        return;
    }
    let (offset, bind) = view.into_inner();
    let Some(pipeline) = ready(&cache, &pipelines.march) else {
        return;
    };
    let low = cloud.params.sizes.zw();
    let diagnostics = ctx.diagnostic_recorder();
    let diagnostics = diagnostics.as_deref();
    let mut pass = ctx
        .command_encoder()
        .begin_compute_pass(&ComputePassDescriptor {
            label: Some("elements_clouds_march"),
            ..default()
        });
    let span = diagnostics.pass_span(&mut pass, "elements_clouds_march");
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, &bind.march, &[offset.offset]);
    pass.dispatch_workgroups(groups(low.x, WG2), groups(low.y, WG2), 1);
    span.end(&mut pass);
}

fn cloud_resolve(
    view: ViewQuery<(
        &ViewUniformOffset,
        &PreviousViewUniformOffset,
        &CloudViewGroups,
    )>,
    cloud: Option<Res<CloudGpu>>,
    pipelines: Option<Res<CloudPipelines>>,
    progress: Res<CloudProgress>,
    cache: Res<PipelineCache>,
    mut ctx: RenderContext,
) {
    let (Some(cloud), Some(pipelines)) = (cloud, pipelines) else {
        return;
    };
    if !cloud.active || !noise_ready(&progress) {
        return;
    }
    let (offset, previous, bind) = view.into_inner();
    let Some(pipeline) = ready(&cache, &pipelines.resolve) else {
        return;
    };
    let size = cloud.params.sizes.xy();
    let diagnostics = ctx.diagnostic_recorder();
    let diagnostics = diagnostics.as_deref();
    let mut pass = ctx
        .command_encoder()
        .begin_compute_pass(&ComputePassDescriptor {
            label: Some("elements_clouds_resolve"),
            ..default()
        });
    let span = diagnostics.pass_span(&mut pass, "elements_clouds_resolve");
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, &bind.resolve, &[offset.offset, previous.offset]);
    pass.dispatch_workgroups(groups(size.x, WG2), groups(size.y, WG2), 1);
    span.end(&mut pass);
}

fn cloud_composite(
    view: ViewQuery<(&ViewTarget, &ViewUniformOffset, &CloudViewGroups)>,
    cloud: Option<Res<CloudGpu>>,
    pipelines: Option<Res<CloudPipelines>>,
    progress: Res<CloudProgress>,
    cache: Res<PipelineCache>,
    mut ctx: RenderContext,
) {
    let (Some(cloud), Some(pipelines)) = (cloud, pipelines) else {
        return;
    };
    if !cloud.active || !noise_ready(&progress) {
        return;
    }
    let (target, offset, bind) = view.into_inner();
    let Some(pipeline) = cache.get_render_pipeline(pipelines.composite) else {
        return;
    };
    let diagnostics = ctx.diagnostic_recorder();
    let diagnostics = diagnostics.as_deref();
    let mut pass = ctx
        .command_encoder()
        .begin_render_pass(&RenderPassDescriptor {
            label: Some("elements_clouds_composite"),
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
    let span = diagnostics.pass_span(&mut pass, "elements_clouds_composite");
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, &bind.composite, &[offset.offset]);
    pass.draw(0..3, 0..1);
    span.end(&mut pass);
}
