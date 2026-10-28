//! Render-world side of the water: the `water_normals` kernel, dispatched once, that writes every
//! mip of the tileable normal map.

use std::sync::atomic::Ordering;

use bevy::asset::load_embedded_asset;
use bevy::core_pipeline::schedule::camera_driver;
use bevy::math::{UVec4, Vec4};
use bevy::prelude::*;
use bevy::render::diagnostic::RecordDiagnostics;
use bevy::render::extract_resource::{ExtractResource, ExtractResourcePlugin};
use bevy::render::render_asset::RenderAssets;
use bevy::render::render_resource::binding_types::{texture_storage_2d, uniform_buffer};
use bevy::render::render_resource::{
    BindGroup, BindGroupEntries, BindGroupLayoutDescriptor, BindGroupLayoutEntries,
    ComputePassDescriptor, PipelineCache, ShaderStages, ShaderType, StorageTextureAccess,
    TextureFormat, TextureViewDescriptor, UniformBuffer,
};
use bevy::render::renderer::{RenderContext, RenderDevice, RenderGraph, RenderQueue};
use bevy::render::texture::GpuImage;
use bevy::render::{Render, RenderApp, RenderStartup, RenderSystems};

use super::water::{
    FOAM_OCTAVES, NORMAL_MIPS, NORMAL_OCTAVES, NORMAL_SLOPE_GAIN, NORMAL_TEXELS, WaterState,
};
use crate::gpu::kernel::{Kernel, groups, queue_kernel, ready};

impl ExtractResource<RenderApp> for WaterState {
    type Source = WaterState;

    fn extract_resource(source: &Self::Source) -> Self {
        source.clone()
    }
}

/// Uniform of `water_normals.wesl`.
#[derive(Clone, Copy, Debug, Default, PartialEq, ShaderType)]
pub(crate) struct NormalParams {
    /// x: mip level. y: its edge in texels. z: octaves of the height kept. w: octaves of the foam
    /// fbm kept.
    level: UVec4,
    /// x: the slope gain.
    gain: Vec4,
}

/// The octaves of the height field and of the foam fbm a mip level keeps: those whose wavelength
/// is at least two texels of the level. Octave `o` of the height has `64 / 2^o` texels, octave `j`
/// of the foam fbm `128 / 2^j`. The coarsest mips keep none: the 1-texel mip is the tile's mean
/// normal, flat, with a foam alpha of 0.5.
pub(crate) fn octaves_kept(level: u32) -> (u32, u32) {
    let two = 2u32 << level;
    let height = (0..NORMAL_OCTAVES).filter(|o| (64u32 >> o) >= two).count() as u32;
    let foam = (0..FOAM_OCTAVES).filter(|j| (128u32 >> j) >= two).count() as u32;
    (height, foam)
}

#[derive(Resource)]
struct WaterPipelines {
    normals: Kernel,
}

/// One bind group per mip level; the uniform buffers must live as long as the groups.
#[derive(Resource)]
struct NormalGroups {
    groups: Vec<BindGroup>,
    _uniforms: Vec<UniformBuffer<NormalParams>>,
}

pub(super) fn plugin(app: &mut App) {
    app.add_plugins(ExtractResourcePlugin::<WaterState>::default());
    let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
        return;
    };
    render_app
        .add_systems(RenderStartup, init_pipelines)
        .add_systems(Render, prepare.in_set(RenderSystems::PrepareBindGroups))
        .add_systems(RenderGraph, dispatch.before(camera_driver));
}

fn normals_layout() -> BindGroupLayoutDescriptor {
    BindGroupLayoutDescriptor::new(
        "elements_water_normals",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                uniform_buffer::<NormalParams>(false),
                texture_storage_2d(TextureFormat::Rgba8Unorm, StorageTextureAccess::WriteOnly),
            ),
        ),
    )
}

fn init_pipelines(mut commands: Commands, assets: Res<AssetServer>, cache: Res<PipelineCache>) {
    let shader = load_embedded_asset!(assets.as_ref(), "water_normals.wesl");
    commands.insert_resource(WaterPipelines {
        normals: queue_kernel(
            &cache,
            shader,
            "water_normals",
            normals_layout(),
            Vec::new(),
        ),
    });
}

#[allow(clippy::too_many_arguments)]
fn prepare(
    mut commands: Commands,
    state: Option<Res<WaterState>>,
    pipelines: Option<Res<WaterPipelines>>,
    existing: Option<Res<NormalGroups>>,
    images: Res<RenderAssets<GpuImage>>,
    cache: Res<PipelineCache>,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
) {
    let (Some(state), Some(pipelines)) = (state, pipelines) else {
        return;
    };
    if existing.is_some() || state.normals_ready.load(Ordering::Relaxed) {
        return;
    }
    let Some(image) = images.get(&state.normals) else {
        return;
    };
    let layout = cache.get_bind_group_layout(&pipelines.normals.layout);
    let mut uniforms = Vec::with_capacity(NORMAL_MIPS as usize);
    let mut bind_groups = Vec::with_capacity(NORMAL_MIPS as usize);
    for level in 0..NORMAL_MIPS {
        let (height, foam) = octaves_kept(level);
        let mut uniform = UniformBuffer::from(NormalParams {
            level: UVec4::new(level, NORMAL_TEXELS >> level, height, foam),
            gain: Vec4::new(NORMAL_SLOPE_GAIN, 0.0, 0.0, 0.0),
        });
        uniform.write_buffer(&device, &queue);
        let view = image.texture.create_view(&TextureViewDescriptor {
            label: Some("elements_water_normals_mip"),
            base_mip_level: level,
            mip_level_count: Some(1),
            ..default()
        });
        bind_groups.push(device.create_bind_group(
            "elements_water_normals",
            &layout,
            &BindGroupEntries::sequential((&uniform, &view)),
        ));
        uniforms.push(uniform);
    }
    commands.insert_resource(NormalGroups {
        groups: bind_groups,
        _uniforms: uniforms,
    });
}

fn dispatch(
    mut ctx: RenderContext,
    cache: Res<PipelineCache>,
    state: Option<Res<WaterState>>,
    pipelines: Option<Res<WaterPipelines>>,
    bind_groups: Option<Res<NormalGroups>>,
) {
    let (Some(state), Some(pipelines), Some(bind_groups)) = (state, pipelines, bind_groups) else {
        return;
    };
    if state.normals_ready.load(Ordering::Relaxed) {
        return;
    }
    let Some(kernel) = ready(&cache, &pipelines.normals) else {
        return;
    };
    let diagnostics = ctx.diagnostic_recorder();
    let diagnostics = diagnostics.as_deref();
    let mut pass = ctx
        .command_encoder()
        .begin_compute_pass(&ComputePassDescriptor {
            label: Some("elements_water_normals"),
            ..default()
        });
    let span = diagnostics.pass_span(&mut pass, "elements_water_normals");
    pass.set_pipeline(kernel);
    for (level, group) in bind_groups.groups.iter().enumerate() {
        let edge = NORMAL_TEXELS >> level;
        pass.set_bind_group(0, group, &[]);
        pass.dispatch_workgroups(groups(edge, 8), groups(edge, 8), 1);
    }
    span.end(&mut pass);
    drop(pass);
    state.normals_ready.store(true, Ordering::Relaxed);
}

#[cfg(test)]
mod tests {
    use super::octaves_kept;

    /// The octaves a mip keeps: all five down to level 1 (the finest wavelength is 4 texels), two
    /// fewer at level 3, none from level 6.
    #[test]
    fn mips_drop_the_octaves_they_cannot_resolve() {
        assert_eq!(octaves_kept(0).0, 5);
        assert_eq!(octaves_kept(1).0, 5);
        assert_eq!(octaves_kept(2).0, 4);
        assert_eq!(octaves_kept(3).0, 3);
        assert_eq!(octaves_kept(5).0, 1);
        assert_eq!(octaves_kept(9), (0, 0));
        assert_eq!(octaves_kept(0).1, 4);
    }
}
