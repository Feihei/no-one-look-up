//! Render-world side of the ground-fog density kernel.

use bevy::asset::load_embedded_asset;
use bevy::core_pipeline::schedule::camera_driver;
use bevy::prelude::*;
use bevy::render::diagnostic::RecordDiagnostics;
use bevy::render::extract_resource::{ExtractResource, ExtractResourcePlugin};
use bevy::render::render_asset::RenderAssets;
use bevy::render::render_resource::binding_types::{
    sampler, texture_2d, texture_storage_3d, uniform_buffer,
};
use bevy::render::render_resource::{
    BindGroup, BindGroupEntries, BindGroupLayoutDescriptor, BindGroupLayoutEntries,
    ComputePassDescriptor, PipelineCache, SamplerBindingType, ShaderStages, StorageTextureAccess,
    TextureFormat, TextureSampleType, UniformBuffer,
};
use bevy::render::renderer::{RenderContext, RenderDevice, RenderGraph, RenderQueue};
use bevy::render::texture::GpuImage;
use bevy::render::{Render, RenderApp, RenderStartup, RenderSystems};

use super::{FogGpu, FogParams};
use crate::gpu::kernel::{Kernel, WG3, groups, queue_kernel, ready};

impl ExtractResource<RenderApp> for FogGpu {
    type Source = FogGpu;

    fn extract_resource(source: &Self::Source) -> Self {
        source.clone()
    }
}

#[derive(Resource)]
struct FogPipeline(Kernel);

#[derive(Resource, Default)]
struct FogUniform(UniformBuffer<FogParams>);

#[derive(Resource)]
struct FogBindGroup(BindGroup);

pub(super) fn plugin(app: &mut App) {
    app.add_plugins(ExtractResourcePlugin::<FogGpu>::default());
    let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
        return;
    };
    render_app
        .init_resource::<FogUniform>()
        .add_systems(RenderStartup, init_pipeline)
        .add_systems(Render, prepare.in_set(RenderSystems::PrepareBindGroups))
        .add_systems(RenderGraph, dispatch.before(camera_driver));
}

fn init_pipeline(mut commands: Commands, assets: Res<AssetServer>, cache: Res<PipelineCache>) {
    let layout = BindGroupLayoutDescriptor::new(
        "elements_fog",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                uniform_buffer::<FogParams>(false),
                texture_storage_3d(TextureFormat::Rgba16Float, StorageTextureAccess::WriteOnly),
                texture_2d(TextureSampleType::Float { filterable: true }),
                sampler(SamplerBindingType::Filtering),
            ),
        ),
    );
    let shader = load_embedded_asset!(assets.as_ref(), "density.wesl");
    commands.insert_resource(FogPipeline(queue_kernel(
        &cache,
        shader,
        "fog_density",
        layout,
        Vec::new(),
    )));
}

#[allow(clippy::too_many_arguments)]
fn prepare(
    mut commands: Commands,
    fog: Option<Res<FogGpu>>,
    pipeline: Option<Res<FogPipeline>>,
    mut uniform: ResMut<FogUniform>,
    images: Res<RenderAssets<GpuImage>>,
    cache: Res<PipelineCache>,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
) {
    let (Some(fog), Some(pipeline)) = (fog, pipeline) else {
        return;
    };
    let (Some(texture), Some(map)) = (images.get(&fog.texture), images.get(&fog.weather_map))
    else {
        commands.remove_resource::<FogBindGroup>();
        return;
    };
    uniform.0.set(fog.params);
    uniform.0.write_buffer(&device, &queue);
    let group = device.create_bind_group(
        "elements_fog",
        &cache.get_bind_group_layout(&pipeline.0.layout),
        &BindGroupEntries::sequential((
            &uniform.0,
            &texture.texture_view,
            &map.texture_view,
            &map.sampler,
        )),
    );
    commands.insert_resource(FogBindGroup(group));
}

fn dispatch(
    mut ctx: RenderContext,
    cache: Res<PipelineCache>,
    fog: Option<Res<FogGpu>>,
    pipeline: Option<Res<FogPipeline>>,
    group: Option<Res<FogBindGroup>>,
) {
    let (Some(fog), Some(pipeline), Some(group)) = (fog, pipeline, group) else {
        return;
    };
    if !fog.active {
        return;
    }
    let Some(compute) = ready(&cache, &pipeline.0) else {
        return;
    };
    let n = super::GROUND_FOG_TEXELS;
    let diagnostics = ctx.diagnostic_recorder();
    let diagnostics = diagnostics.as_deref();
    let mut pass = ctx
        .command_encoder()
        .begin_compute_pass(&ComputePassDescriptor {
            label: Some("elements_fog"),
            ..default()
        });
    let span = diagnostics.pass_span(&mut pass, "elements_fog");
    pass.set_pipeline(compute);
    pass.set_bind_group(0, &group.0, &[]);
    pass.dispatch_workgroups(groups(n.x, WG3), groups(n.y, WG3), groups(n.z, WG3));
    span.end(&mut pass);
}
