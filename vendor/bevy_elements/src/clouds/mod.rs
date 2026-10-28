//! Volumetric clouds: a weather-driven cloud layer ray-marched in compute, accumulated over a few
//! frames, composited after the transparent pass, plus a cloud shadow map for the sun's light.
//! See `docs/research/clouds.md`.

mod gpu;
#[cfg(test)]
mod gpu_tests;
#[cfg(test)]
mod model;

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use bevy::asset::{RenderAssetUsages, embedded_asset};
use bevy::camera::Exposure;
use bevy::image::{ImageAddressMode, ImageFilterMode, ImageSampler, ImageSamplerDescriptor};
use bevy::light::Atmosphere;
use bevy::light::light_consts::lux;
use bevy::math::DVec2;
use bevy::pbr::{DistanceFog, FogFalloff};
use bevy::prelude::*;
use bevy::render::RenderApp;
use bevy::render::extract_component::ExtractComponent;
use bevy::render::render_resource::{
    Extent3d, ShaderType, TextureDimension, TextureFormat, TextureUsages,
};
use bevy::render::storage::ShaderBuffer;
use bevy::shader::load_shader_library;

use crate::celestial::{
    CelestialFrame, CloudLight, MOON_TINT, SunState, beam_transmittance, cloud_ambient,
    scene_illuminance, sky_gain,
};
use crate::climate::Weather;
use crate::flash::LightningFlash;
use crate::gpu::grid::storage_buffer;
use crate::gpu::readback::{Readbacks, decode};
use crate::plugin::{ElementsQuality, ElementsSystems, ElementsView, ensure_gpu_core};
use crate::query::EnvironmentSnapshot;
use crate::terrain::TerrainGrid;

/// Smoothstep, `S(a, b, x)` of the research note.
fn smoothstep(a: f32, b: f32, x: f32) -> f32 {
    let t = ((x - a) / (b - a)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Where the cloud layer is: a spherical shell whose base and top are altitudes above the
/// planet surface (`y = 0`).
///
/// Derived from [`Weather`] every frame ([`CloudLayer::from_weather`]); the lightning plugin
/// reads it to put a bolt's top at the cloud base. Read-only for everyone else.
#[derive(Resource, Clone, Copy, Debug, PartialEq, Reflect)]
pub struct CloudLayer {
    /// Altitude of the cloud base, m above `y = 0`. 600–1500.
    pub base_m: f32,
    /// Altitude of the cloud top, m above `y = 0`. 1400–9500.
    pub top_m: f32,
}

impl Default for CloudLayer {
    fn default() -> Self {
        Self {
            base_m: 1500.0,
            top_m: 3300.0,
        }
    }
}

impl CloudLayer {
    /// The layer for a weather state: base 1500 m, lowered toward 600 m in fog and drizzle;
    /// thickness 800 m (stratus), 1800 m (cumulus) or 8000 m (cumulonimbus) by
    /// [`kind_from_weather`](Self::kind_from_weather).
    pub fn from_weather(weather: &Weather) -> Self {
        let rh = weather.relative_humidity;
        let rain = weather.precipitation_mm_h;
        let low = (smoothstep(0.9, 0.98, rh) * (1.0 - smoothstep(0.1, 0.5, rain)))
            .max(smoothstep(0.1, 0.3, rain) * (1.0 - smoothstep(0.8, 1.5, rain)));
        let base_m = 1500.0 + (600.0 - 1500.0) * low;
        let kind = Self::kind_from_weather(weather);
        let thickness = if kind <= 0.5 {
            800.0 + 1000.0 * (kind / 0.5).min(1.0)
        } else {
            1800.0 + 6200.0 * (kind - 0.5) / 0.5
        };
        Self {
            base_m,
            top_m: base_m + thickness,
        }
    }

    /// The cloud type of a weather state, the weather map's blue channel: 0 stratus, 0.5
    /// cumulus, 1 cumulonimbus, continuous in every [`Weather`] field.
    pub fn kind_from_weather(weather: &Weather) -> f32 {
        let c = weather.cloud_cover;
        let stratus = smoothstep(0.5, 0.9, c).max(smoothstep(0.9, 0.98, weather.relative_humidity));
        let towering = (weather.storminess / 6.0)
            .clamp(0.0, 1.0)
            .max(smoothstep(12.0, 16.0, weather.wind_speed_ms) * smoothstep(0.9, 1.0, c));
        0.5 * (1.0 - stratus) * (1.0 - towering) + towering
    }

    /// Precipitation darkening `min(R / 25, 1)` of a weather state (the weather map's green
    /// channel before it is masked by coverage).
    pub fn precipitation_from_weather(weather: &Weather) -> f32 {
        (weather.precipitation_mm_h / 25.0).clamp(0.0, 1.0)
    }
}

/// The 512² `rgba8unorm` weather map the clouds are built from, covering 32 km around the
/// camera: R coverage, G precipitation (never above `precipitation_mm_h / 25`, zero where there
/// is no cloud), B cloud type. Sample it with a bilinear, repeating sampler at
/// `uv = (world_xz − origin) / extent`.
#[derive(Resource, Clone, Debug)]
pub struct CloudWeatherMap {
    /// The map texture; written by a compute kernel every frame.
    pub map: Handle<Image>,
    /// World XZ of the map's minimum corner, m (snapped to a whole texel).
    pub origin: Vec2,
    /// Side length of the map, m.
    pub extent: f32,
}

/// The cloud shadow map: sun (or moon) transmittance through the clouds for the plane `y = 0`,
/// 1024² `rgba16float` over an 8 km square centred on the camera. R is the direct transmittance
/// `T`, G the optical depth (clamped to 60), B the precipitation darkening of the weather map.
///
/// A consumer at world position `P` samples with `uv = (P.xz − light_direction.xz · P.y /
/// light_direction.y − origin) / extent` and `Clamp` addressing; outside the map `T = 1`.
#[derive(Resource, Clone, Debug)]
pub struct CloudShadows {
    /// The map texture.
    pub map: Handle<Image>,
    /// World XZ of the map's minimum corner, m (snapped to a whole texel).
    pub origin: Vec2,
    /// Side length of the map, m.
    pub extent: f32,
    /// Unit vector toward the light the map was generated for: the sun by day, the moon when
    /// the sun is lower than −4°.
    pub light_direction: Vec3,
    /// Mean direct transmittance over about 70 m around the camera, 0–1, smoothed over 1 s of
    /// real time. Before the first GPU readback it is `1 − cover^3.4`, the cover-only estimate of
    /// the weather. The sun's and moon's `DirectionalLight` carry `T̄` of their clear-sky
    /// illuminance and the camera's `AmbientLight` the diffuse remainder `0.25 · (1 − T̄)`.
    pub camera_transmittance: f32,
}

/// Lowest sine of the light's elevation the shadow map is generated for: arcsin 0.1 = 5.74°. A
/// lower sun would put the cloud layer beyond the weather map; it is dim and red anyway.
pub(crate) const SHADOW_MIN_LIGHT_SINE: f32 = 0.1;

impl CloudShadows {
    /// The unit direction toward the light the map was actually generated for: `light_direction`
    /// with its elevation raised to 5.74° when lower.
    pub fn generation_direction(&self) -> Vec3 {
        let l = self.light_direction;
        if l.y >= SHADOW_MIN_LIGHT_SINE {
            return l;
        }
        let flat = l.xz().normalize_or(Vec2::X);
        let c = (1.0 - SHADOW_MIN_LIGHT_SINE * SHADOW_MIN_LIGHT_SINE).sqrt();
        Vec3::new(flat.x * c, SHADOW_MIN_LIGHT_SINE, flat.y * c)
    }

    /// Texture coordinate (0–1 over the map) of the shadow cast on world position `position`:
    /// `(P.xz − l.xz · P.y / l.y − origin) / extent`, the point where the ray from `P` toward
    /// the light would cross the plane `y = 0` the map is generated for. Sample with `Clamp`
    /// addressing; a coordinate outside 0–1 is outside the map and unshadowed (`T = 1`).
    pub fn uv(&self, position: Vec3) -> Vec2 {
        let l = self.generation_direction();
        (position.xz() - l.xz() * (position.y / l.y) - self.origin) / self.extent
    }
}

/// Cloud rendering cost per [`ElementsQuality`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CloudQuality {
    /// Edge, in pixels, of the block one march texel stands for: 4 (a quarter of the resolution
    /// per axis) on Low and Medium, 2 on High and Ultra.
    pub block: u32,
    /// Primary ray-march steps.
    pub steps: u32,
    /// Samples along the path to the light.
    pub light_samples: u32,
    /// Frames until every pixel has been marched once (`block²`).
    pub window: u32,
}

impl CloudQuality {
    /// The settings of a quality level.
    pub fn of(quality: ElementsQuality) -> Self {
        let (block, steps) = match quality {
            ElementsQuality::Low => (4, 32),
            ElementsQuality::Medium => (4, 48),
            ElementsQuality::High => (2, 64),
            ElementsQuality::Ultra => (2, 96),
        };
        Self {
            block,
            steps,
            light_samples: 6,
            window: block * block,
        }
    }
}

/// Texels per side of the weather map and the world size it covers, m.
pub(crate) const WEATHER_TEXELS: u32 = 512;
pub(crate) const WEATHER_EXTENT_M: f32 = 32_000.0;
/// Texels per side of the shadow map and the world size it covers, m.
pub(crate) const SHADOW_TEXELS: u32 = 1024;
pub(crate) const SHADOW_EXTENT_M: f32 = 8_000.0;
/// Shadow-map texels per cell edge of the gameplay query grid ([`CloudShadowSnapshot`]).
pub(crate) const SHADOW_COARSE_STRIDE: u32 = 4;
/// Cells per side of the gameplay query grid.
pub(crate) const SHADOW_COARSE_CELLS: u32 = SHADOW_TEXELS / SHADOW_COARSE_STRIDE;
/// Texels per side of the base and detail noise volumes.
pub(crate) const NOISE_BASE_TEXELS: u32 = 128;
pub(crate) const NOISE_DETAIL_TEXELS: u32 = 32;
/// World size of the noise tiles, m; the cloud drift is wrapped to each so f32 keeps its
/// precision. The weather field is periodic in its 32 km extent.
const BASE_TILE_M: f64 = 8192.0;
const DETAIL_TILE_M: f64 = 1024.0;
/// The drift accumulates in f64 modulo a common multiple of the three periods (2²² · 5³).
const DRIFT_WRAP_M: f64 = 524_288_000.0;
/// Cloud drift is this multiple of the 10 m wind: the wind at cloud height (log-law 1.86 at
/// 1.5 km, 1.94 at the layer top) rounded up as the plan says.
const DRIFT_WIND_FACTOR: f64 = 2.0;
/// Sun altitude, degrees, below which the clouds are lit by the moon; the sun's light fades
/// out over the degree above it (the layer keeps the sun to −3.1° at 9.5 km, the dip).
const SUN_HANDOVER_DEG: f32 = -4.0;
/// Albedo of the ground the diffuse light reflects up into the air, for the in-scatter source.
const GROUND_ALBEDO: f32 = 0.3;
/// In-scatter source of the air per unit radiance of the ambient sky: isotropic scattering lights
/// the air with the scalar irradiance over 4π, `E (1 + ρ) / 4π` for the downward irradiance
/// `E = π L` of the sky and the ground albedo `ρ`, which is `L (1 + ρ) / 4`.
const AIRLIGHT_PER_SKY_RADIANCE: f32 = (1.0 + GROUND_ALBEDO) / 4.0;
/// Smoothing time of the camera transmittance, s of real time.
const CAMERA_TRANSMITTANCE_TAU_S: f32 = 1.0;

/// Uniform of every cloud kernel (`common.wesl`, `CloudParams`).
#[derive(Clone, Copy, Debug, Default, ShaderType)]
pub(crate) struct CloudParams {
    /// xyz: camera, m. w: planet radius, m.
    pub camera: Vec4,
    /// x: atmosphere outer radius, m. y: aerial LUT range, m. z: aerial LUT slices.
    pub atmo: Vec4,
    /// x: gain on the sky pixels of the composite (`celestial::sky_gain`). yzw: radiance of the
    /// air lit by the cloud-scattered diffuse light (`celestial::cloud_ambient`), cd/m².
    pub sky: Vec4,
    /// x: base altitude, m. y: top altitude, m. z: cloud type. w: cover.
    pub layer: Vec4,
    /// xyz: unit direction toward the light.
    pub light: Vec4,
    /// xyz: light illuminance outside the atmosphere, lx. w: ambient source, cd/m².
    pub light_e: Vec4,
    /// xy: drift of the weather field, m. zw: of the base noise, m.
    pub drift: Vec4,
    /// xy: drift of the detail noise, m. zw: unit wind direction.
    pub drift_detail: Vec4,
    /// xy: weather map origin. z: extent, m. w: precipitation darkening.
    pub weather: Vec4,
    /// xy: shadow map origin. z: extent, m.
    pub shadow: Vec4,
    /// xyz: flash position, m. w: intensity, cd.
    pub flash: Vec4,
    /// x: frame. y: steps. z: block edge. w: light samples.
    pub counts: UVec4,
    /// xy: full resolution. zw: march resolution.
    pub sizes: UVec4,
    /// x: previous/current exposure. y: debug mode. z: debug extinction. w: history reset.
    pub misc: Vec4,
    /// x: first noise slice of this frame.
    pub tail: UVec4,
    /// xyz: the camera's `DistanceFog` colour (linear, after exposure). w: its extinction, m⁻¹
    /// (0 without distance fog).
    pub haze: Vec4,
    /// x: extinction of the weather's air above clear air, m⁻¹ (`fog::sky_air_extinction`).
    /// The composite fogs sky pixels with it, so the sky converges on `haze.xyz` as distant
    /// ground does; 0 leaves the sky to the atmosphere alone. `clouds.md` Deviation 29.
    pub air: Vec4,
}

/// The per-view images: this frame's march result and the two history textures.
#[derive(Clone, Debug)]
pub(crate) struct CloudViewImages {
    pub size: UVec2,
    pub march: Handle<Image>,
    pub history: [Handle<Image>; 2],
}

/// Everything the render world needs, extracted every frame.
#[derive(Resource, Clone)]
pub(crate) struct CloudGpu {
    pub base: Handle<Image>,
    pub detail: Handle<Image>,
    pub weather: Handle<Image>,
    pub shadow: Handle<Image>,
    pub histogram: Handle<ShaderBuffer>,
    pub threshold: Handle<ShaderBuffer>,
    pub camera_t: Handle<ShaderBuffer>,
    /// Every fourth shadow-map texel's transmittance, `SHADOW_COARSE_CELLS²` row-major.
    pub shadow_coarse: Handle<ShaderBuffer>,
    pub view: Option<CloudViewImages>,
    pub params: CloudParams,
    /// Followed camera present: the weather and shadow kernels run.
    pub active: bool,
    /// Set by the render world after the weather and shadow kernels first ran.
    pub live: Arc<AtomicBool>,
}

/// Marks the camera the clouds are drawn for; extracted to the render world.
#[derive(Component, Clone, Copy, Debug, Default, ExtractComponent)]
#[extract_app(RenderApp)]
pub(crate) struct CloudView;

/// The system that moves [`CloudWeatherMap`] and [`CloudShadows`] and fills the GPU uniforms;
/// consumers of the maps' origins order themselves after it.
#[derive(SystemSet, Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct CloudGpuUpdate;

/// Main-world state of the cloud simulation.
#[derive(Resource, Default)]
struct CloudRuntime {
    /// Cloud drift, m, wrapped at `DRIFT_WRAP_M`.
    drift: DVec2,
    frame: u32,
    quality: Option<ElementsQuality>,
    /// Frames left on which the history is flagged invalid.
    reset_frames: u32,
    exposure: f32,
}

/// Test-only overrides of the kernels' inputs.
#[cfg(test)]
#[derive(Resource, Clone, Debug, Default)]
pub(crate) struct CloudDebug {
    /// 1: uniform slab of extinction `sigma`; 2: output the light transmittance at the entry.
    pub mode: f32,
    pub sigma: f32,
    pub layer: Option<(f32, f32)>,
    pub light: Option<Vec3>,
    pub light_e: Option<Vec3>,
    pub kind: Option<f32>,
    pub cover: Option<f32>,
    pub drift: Option<Vec2>,
    /// The weather's air extinction, m⁻¹: 0 is clear air and leaves the sky to the atmosphere.
    pub air: Option<f32>,
}

/// Volumetric clouds.
///
/// Creates the noise volumes, the weather map and the shadow map; each frame it derives the
/// [`CloudLayer`] from the weather, marches the clouds for the followed
/// [`ElementsCamera`](crate::ElementsCamera) and composites them after the transparent pass, and keeps
/// [`CloudShadows`] current. It dims the sun and moon lights by the cloud transmittance over the
/// camera. Needs `DepthPrepass`, `Msaa::Off` and an HDR target, which `ElementsCamera` sets up.
pub struct CloudPlugin;

impl Plugin for CloudPlugin {
    fn build(&self, app: &mut App) {
        ensure_gpu_core(app);
        load_shader_library!(app, "common.wesl");
        embedded_asset!(app, "generate.wesl");
        embedded_asset!(app, "weather.wesl");
        embedded_asset!(app, "shadow.wesl");
        embedded_asset!(app, "march.wesl");
        embedded_asset!(app, "resolve.wesl");
        embedded_asset!(app, "composite.wesl");
        app.init_resource::<LightningFlash>()
            .init_resource::<CloudLayer>()
            .init_resource::<CloudRuntime>()
            .add_systems(Startup, create_resources)
            .add_systems(
                Update,
                (
                    smooth_camera_transmittance
                        .in_set(ElementsSystems::Celestial)
                        .before(crate::celestial::update_sky),
                    (
                        mark_view,
                        update_layer,
                        advance_drift,
                        update_views,
                        // The march reads this frame's `LightningFlash`.
                        update_gpu
                            .after(crate::lightning::write_flash)
                            .in_set(CloudGpuUpdate),
                    )
                        .chain()
                        .in_set(ElementsSystems::Simulation),
                    read_shadows.in_set(ElementsSystems::Readback),
                ),
            );
        #[cfg(test)]
        app.add_systems(
            Update,
            apply_debug
                .in_set(ElementsSystems::Simulation)
                .after(update_gpu),
        );
        gpu::plugin(app);
    }
}

/// A storage texture the kernels write and shaders sample with a linear sampler.
fn texture(
    images: &mut Assets<Image>,
    size: Extent3d,
    dimension: TextureDimension,
    format: TextureFormat,
    pixel: &[u8],
    address: ImageAddressMode,
) -> Handle<Image> {
    let mut image = Image::new_fill(
        size,
        dimension,
        pixel,
        format,
        RenderAssetUsages::RENDER_WORLD,
    );
    // COPY_SRC lets the tests read any of them back.
    image.texture_descriptor.usage = TextureUsages::STORAGE_BINDING
        | TextureUsages::TEXTURE_BINDING
        | TextureUsages::COPY_SRC
        | TextureUsages::COPY_DST;
    image.sampler = ImageSampler::Descriptor(ImageSamplerDescriptor {
        address_mode_u: address,
        address_mode_v: address,
        address_mode_w: address,
        mag_filter: ImageFilterMode::Linear,
        min_filter: ImageFilterMode::Linear,
        mipmap_filter: ImageFilterMode::Linear,
        ..default()
    });
    images.add(image)
}

/// Half-precision 1.0, little-endian.
const HALF_ONE: [u8; 2] = [0x00, 0x3C];

fn create_resources(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    mut buffers: ResMut<Assets<ShaderBuffer>>,
) {
    let volume = |n: u32| Extent3d {
        width: n,
        height: n,
        depth_or_array_layers: n,
    };
    let plane = |n: u32| Extent3d {
        width: n,
        height: n,
        depth_or_array_layers: 1,
    };
    let base = texture(
        &mut images,
        volume(NOISE_BASE_TEXELS),
        TextureDimension::D3,
        TextureFormat::Rgba8Unorm,
        &[0; 4],
        ImageAddressMode::Repeat,
    );
    let detail = texture(
        &mut images,
        volume(NOISE_DETAIL_TEXELS),
        TextureDimension::D3,
        TextureFormat::Rgba8Unorm,
        &[0; 4],
        ImageAddressMode::Repeat,
    );
    let weather = texture(
        &mut images,
        plane(WEATHER_TEXELS),
        TextureDimension::D2,
        TextureFormat::Rgba8Unorm,
        &[0; 4],
        ImageAddressMode::Repeat,
    );
    // Nothing shadowed until the first dispatch: T = 1.
    let shadow = texture(
        &mut images,
        plane(SHADOW_TEXELS),
        TextureDimension::D2,
        TextureFormat::Rgba16Float,
        &[HALF_ONE[0], HALF_ONE[1], 0, 0, 0, 0, 0, 0],
        ImageAddressMode::ClampToEdge,
    );
    commands.insert_resource(CloudWeatherMap {
        map: weather.clone(),
        origin: Vec2::ZERO,
        extent: WEATHER_EXTENT_M,
    });
    commands.insert_resource(CloudShadows {
        map: shadow.clone(),
        origin: Vec2::ZERO,
        extent: SHADOW_EXTENT_M,
        light_direction: Vec3::Y,
        camera_transmittance: 1.0,
    });
    commands.insert_resource(CloudGpu {
        base,
        detail,
        weather,
        shadow,
        histogram: storage_buffer(&mut buffers, "elements_clouds_histogram", vec![0u32; 256]),
        threshold: storage_buffer(&mut buffers, "elements_clouds_threshold", vec![0f32; 4]),
        camera_t: storage_buffer(&mut buffers, "elements_clouds_camera_t", vec![0f32; 4]),
        // Nothing shadowed until the first dispatch: T = 1.
        shadow_coarse: storage_buffer(
            &mut buffers,
            "elements_clouds_shadow_coarse",
            vec![1f32; (SHADOW_COARSE_CELLS * SHADOW_COARSE_CELLS) as usize],
        ),
        view: None,
        params: CloudParams::default(),
        active: false,
        live: Arc::new(AtomicBool::new(false)),
    });
}

/// `CloudView` on the followed camera and no other.
fn mark_view(
    mut commands: Commands,
    view: Res<ElementsView>,
    marked: Query<Entity, With<CloudView>>,
) {
    for entity in &marked {
        if Some(entity) != view.entity {
            commands.entity(entity).remove::<CloudView>();
        }
    }
    if let Some(entity) = view.entity
        && !marked.contains(entity)
    {
        commands.entity(entity).insert(CloudView);
    }
}

fn update_layer(weather: Res<Weather>, mut layer: ResMut<CloudLayer>) {
    let next = CloudLayer::from_weather(&weather);
    if *layer != next {
        *layer = next;
    }
}

/// The clouds drift with the wind at cloud height, `2 · u₁₀` (`Time<Virtual>`, like wind and
/// fog, so the world clock's time scale does not move them).
fn advance_drift(
    time: Res<Time<Virtual>>,
    weather: Res<Weather>,
    mut runtime: ResMut<CloudRuntime>,
) {
    let wind = weather.wind_direction.as_dvec2() * f64::from(weather.wind_speed_ms);
    let next = runtime.drift + wind * DRIFT_WIND_FACTOR * time.delta_secs_f64();
    runtime.drift = next.rem_euclid(DVec2::splat(DRIFT_WRAP_M));
}

/// March, history and result textures sized for the followed camera and the quality; rebuilt
/// when either changes (the weather map, shadow map and noise are kept).
fn update_views(
    view: Res<ElementsView>,
    quality: Res<ElementsQuality>,
    cameras: Query<&Camera>,
    mut images: ResMut<Assets<Image>>,
    mut gpu: ResMut<CloudGpu>,
    mut runtime: ResMut<CloudRuntime>,
) {
    let size = view
        .entity
        .and_then(|e| cameras.get(e).ok())
        .and_then(Camera::physical_target_size)
        .filter(|s| s.x > 0 && s.y > 0);
    let Some(size) = size else {
        if gpu.view.is_some() {
            gpu.view = None;
        }
        return;
    };
    let up_to_date =
        gpu.view.as_ref().is_some_and(|v| v.size == size) && runtime.quality == Some(*quality);
    if up_to_date {
        return;
    }
    let block = CloudQuality::of(*quality).block;
    let low = (size + UVec2::splat(block - 1)) / block;
    let plane = |s: UVec2| Extent3d {
        width: s.x,
        height: s.y,
        depth_or_array_layers: 1,
    };
    // Clear sky (no radiance, full transmittance) until the first resolve.
    let clear = [0, 0, 0, 0, 0, 0, HALF_ONE[0], HALF_ONE[1]];
    let mut make = |s: UVec2| {
        texture(
            &mut images,
            plane(s),
            TextureDimension::D2,
            TextureFormat::Rgba16Float,
            &clear,
            ImageAddressMode::ClampToEdge,
        )
    };
    gpu.view = Some(CloudViewImages {
        size,
        march: make(low),
        history: [make(size), make(size)],
    });
    runtime.quality = Some(*quality);
    runtime.reset_frames = 2;
}

/// The light the clouds are lit by: the sun, fading out over the degree above the handover
/// altitude, and below it the moon.
fn pick_light(sun: &SunState, frame: &CelestialFrame) -> (Vec3, Vec3) {
    if sun.altitude_deg > SUN_HANDOVER_DEG {
        let weight = smoothstep(SUN_HANDOVER_DEG, SUN_HANDOVER_DEG + 1.0, sun.altitude_deg);
        (sun.direction, Vec3::splat(lux::RAW_SUNLIGHT * weight))
    } else {
        (sun.moon_direction, MOON_TINT * frame.moon_lux)
    }
}

/// Round `value` to a whole multiple of `step`.
fn snap(value: Vec2, step: f32) -> Vec2 {
    (value / step).round() * step
}

#[allow(clippy::too_many_arguments)]
fn update_gpu(
    view: Res<ElementsView>,
    weather: Res<Weather>,
    sun: Res<SunState>,
    frame: Res<CelestialFrame>,
    flash: Res<LightningFlash>,
    quality: Res<ElementsQuality>,
    layer: Res<CloudLayer>,
    atmospheres: Query<&Atmosphere>,
    exposures: Query<&Exposure>,
    fogs: Query<&DistanceFog>,
    mut runtime: ResMut<CloudRuntime>,
    mut gpu: ResMut<CloudGpu>,
    mut map: ResMut<CloudWeatherMap>,
    mut shadows: ResMut<CloudShadows>,
) {
    let Some(entity) = view.entity else {
        gpu.active = false;
        return;
    };
    let camera = view.position;
    let (inner, outer) = atmospheres
        .iter()
        .next()
        .map_or((6_360_000.0, 6_460_000.0), |a| {
            (a.inner_radius, a.outer_radius)
        });
    let (light_dir, light_e) = pick_light(&sun, &frame);
    // Diffuse sky light: the metered illuminance before cloud dimming, all of it diffuse in
    // twilight and 15 % of it at high sun.
    let sky_share = 1.0 + (0.15 - 1.0) * smoothstep(-2.0, 10.0, sun.altitude_deg);
    let ambient = sky_share * scene_illuminance(&sun, frame.moon_lux, CloudLight::CLEAR)
        / (2.0 * std::f32::consts::PI);

    let weather_texel = WEATHER_EXTENT_M / WEATHER_TEXELS as f32;
    let shadow_texel = SHADOW_EXTENT_M / SHADOW_TEXELS as f32;
    map.origin = snap(
        camera.xz() - Vec2::splat(WEATHER_EXTENT_M * 0.5),
        weather_texel,
    );
    shadows.origin = snap(
        camera.xz() - Vec2::splat(SHADOW_EXTENT_M * 0.5),
        shadow_texel,
    );
    shadows.light_direction = light_dir;

    let exposure = exposures.get(entity).map_or(1.0, Exposure::exposure);
    let exposure_ratio = if runtime.exposure > 0.0 && exposure > 0.0 {
        runtime.exposure / exposure
    } else {
        1.0
    };
    runtime.exposure = exposure;
    runtime.frame = runtime.frame.wrapping_add(1);
    let reset = runtime.reset_frames > 0;
    runtime.reset_frames = runtime.reset_frames.saturating_sub(1);

    let q = CloudQuality::of(*quality);
    let size = gpu.view.as_ref().map_or(UVec2::ONE, |v| v.size);
    let low = (size + UVec2::splat(q.block - 1)) / q.block;
    let drift = runtime.drift;
    let wrap = |period: f64| {
        Vec2::new(
            drift.x.rem_euclid(period) as f32,
            drift.y.rem_euclid(period) as f32,
        )
    };
    let field = wrap(f64::from(WEATHER_EXTENT_M));
    let base = wrap(BASE_TILE_M);
    let detail = wrap(DETAIL_TILE_M);
    let wind = weather.wind_direction.normalize_or(Vec2::X);

    gpu.active = true;
    let cloud = CloudLight::from_beam(shadows.camera_transmittance);
    let (diffuse_colour, diffuse_brightness) = cloud_ambient(&sun, frame.moon_lux, cloud);
    let airlight = diffuse_colour * (diffuse_brightness * AIRLIGHT_PER_SKY_RADIANCE);
    gpu.params = CloudParams {
        camera: camera.extend(inner),
        atmo: Vec4::new(outer, 32_000.0, 32.0, 0.0),
        sky: Vec4::new(sky_gain(&sun, cloud), airlight.x, airlight.y, airlight.z),
        layer: Vec4::new(
            layer.base_m,
            layer.top_m,
            CloudLayer::kind_from_weather(&weather),
            weather.cloud_cover.clamp(0.0, 1.0),
        ),
        light: light_dir.extend(0.0),
        light_e: light_e.extend(ambient),
        drift: Vec4::new(field.x, field.y, base.x, base.y),
        drift_detail: Vec4::new(detail.x, detail.y, wind.x, wind.y),
        weather: Vec4::new(
            map.origin.x,
            map.origin.y,
            WEATHER_EXTENT_M,
            CloudLayer::precipitation_from_weather(&weather),
        ),
        shadow: Vec4::new(shadows.origin.x, shadows.origin.y, SHADOW_EXTENT_M, 0.0),
        flash: flash.position.extend(flash.intensity_cd.max(0.0)),
        counts: UVec4::new(runtime.frame, q.steps, q.block, q.light_samples),
        sizes: UVec4::new(size.x, size.y, low.x, low.y),
        misc: Vec4::new(exposure_ratio, 0.0, 0.0, if reset { 1.0 } else { 0.0 }),
        tail: UVec4::ZERO,
        haze: fogs.get(entity).map_or(Vec4::ZERO, haze),
        // The sky's air comes from the camera's own `DistanceFog`, so the sky and the meshes
        // cannot disagree about the fog even if a game replaces it.
        air: Vec4::new(
            fogs.get(entity).map_or(0.0, |fog| {
                crate::fog::sky_air_extinction(fog_extinction(fog))
            }),
            0.0,
            0.0,
            0.0,
        ),
    };
}

/// The distance fog as the clouds' ground haze: its linear colour and its grey extinction
/// (`fog::distance_fog_falloff` sets extinction and in-scattering equal).
fn haze(fog: &DistanceFog) -> Vec4 {
    let c = fog.color.to_linear();
    Vec4::new(c.red, c.green, c.blue, fog_extinction(fog))
}

/// Extinction of a camera's distance fog, m⁻¹; 0 for a falloff this crate never writes
/// (`FogFalloff::Linear` or `ExponentialSquared`, which a game that replaces the component may).
fn fog_extinction(fog: &DistanceFog) -> f32 {
    match fog.falloff {
        FogFalloff::Atmospheric { extinction, .. } => extinction.x,
        FogFalloff::Exponential { density } => density,
        FogFalloff::ExponentialSquared { .. } | FogFalloff::Linear { .. } => 0.0,
    }
}

#[cfg(test)]
fn apply_debug(debug: Option<Res<CloudDebug>>, mut gpu: ResMut<CloudGpu>) {
    let Some(debug) = debug else {
        return;
    };
    let p = &mut gpu.params;
    p.misc.y = debug.mode;
    p.misc.z = debug.sigma;
    if let Some((base, top)) = debug.layer {
        p.layer.x = base;
        p.layer.y = top;
    }
    if let Some(kind) = debug.kind {
        p.layer.z = kind;
    }
    if let Some(cover) = debug.cover {
        p.layer.w = cover;
    }
    if let Some(air) = debug.air {
        p.air.x = air;
    }
    if let Some(light) = debug.light {
        p.light = light.extend(0.0);
    }
    if let Some(e) = debug.light_e {
        p.light_e = e.extend(p.light_e.w);
    }
    if let Some(drift) = debug.drift {
        p.drift.x = drift.x.rem_euclid(WEATHER_EXTENT_M);
        p.drift.y = drift.y.rem_euclid(WEATHER_EXTENT_M);
        p.drift.z = drift.x.rem_euclid(BASE_TILE_M as f32);
        p.drift.w = drift.y.rem_euclid(BASE_TILE_M as f32);
        p.drift_detail.x = drift.x.rem_euclid(DETAIL_TILE_M as f32);
        p.drift_detail.y = drift.y.rem_euclid(DETAIL_TILE_M as f32);
    }
}

/// The camera transmittance the lights are dimmed by: the readback, smoothed over a second;
/// until the first readback `1 − cover^3.4`, the cover-only beam.
fn smooth_camera_transmittance(
    time: Res<Time<Real>>,
    weather: Res<Weather>,
    snapshot: Res<EnvironmentSnapshot>,
    shadows: Option<ResMut<CloudShadows>>,
) {
    let Some(mut shadows) = shadows else {
        return;
    };
    match snapshot.cloud_transmittance {
        None => {
            shadows.camera_transmittance = beam_transmittance(weather.cloud_cover);
        }
        Some(measured) => {
            let k = 1.0 - (-time.delta_secs() / CAMERA_TRANSMITTANCE_TAU_S).exp();
            let now = shadows.camera_transmittance;
            shadows.camera_transmittance = now + (measured.clamp(0.0, 1.0) - now) * k;
        }
    }
}

fn read_shadows(
    gpu: Option<Res<CloudGpu>>,
    shadows: Option<Res<CloudShadows>>,
    mut readbacks: Readbacks,
) {
    let (Some(gpu), Some(shadows)) = (gpu, shadows) else {
        return;
    };
    if !gpu.active || !gpu.live.load(Ordering::Relaxed) {
        return;
    }
    readbacks.request("cloud_camera_t", gpu.camera_t.clone(), |bytes, snapshot| {
        if let Some(values) = decode::<f32>(bytes)
            && let Some(&t) = values.first()
            && t.is_finite()
        {
            snapshot.cloud_transmittance = Some(t);
        }
    });
    let grid = CloudShadowSnapshot::grid(shadows.origin);
    let light = shadows.generation_direction();
    readbacks.request(
        "cloud_shadow",
        gpu.shadow_coarse.clone(),
        move |bytes, snapshot| {
            let cells = (SHADOW_COARSE_CELLS * SHADOW_COARSE_CELLS) as usize;
            if let Some(values) = decode::<f32>(bytes)
                && values.len() >= cells
            {
                snapshot.cloud_shadow = Some(CloudShadowSnapshot {
                    grid,
                    light,
                    transmittance: values[..cells].into(),
                });
            }
        },
    );
}

/// CPU copy of the cloud shadow map's query grid: the direct transmittance toward the light
/// every 31.25 m over the map's 8 km square (`docs/research/cloud-shadow-query.md`).
#[derive(Clone, Debug)]
pub(crate) struct CloudShadowSnapshot {
    /// The query grid: cell `k` is centred on shadow-map texel `4k + 2`.
    pub grid: TerrainGrid,
    /// Unit direction toward the light the map was generated for (elevation ≥ 5.74°).
    pub light: Vec3,
    /// Row-major transmittance, 0–1.
    pub transmittance: Arc<[f32]>,
}

impl CloudShadowSnapshot {
    /// The query grid of a shadow map whose minimum corner is `origin`.
    pub(crate) fn grid(origin: Vec2) -> TerrainGrid {
        let texel = SHADOW_EXTENT_M / SHADOW_TEXELS as f32;
        TerrainGrid {
            // Cell k's centre, origin + (k + 0.5)·4t, is texel 4k + 2's centre, origin + (4k + 2.5)·t.
            origin: origin + Vec2::splat(0.5 * texel),
            cell_m: texel * SHADOW_COARSE_STRIDE as f32,
            size: UVec2::splat(SHADOW_COARSE_CELLS),
            y0: 0.0,
        }
    }

    /// Direct transmittance toward the light at a world position: the position projected along
    /// the light onto the map's plane `y = 0`, bilinear; 1 outside the map.
    pub(crate) fn transmittance_at(&self, position: Vec3) -> f32 {
        let q = position.xz() - self.light.xz() * (position.y / self.light.y);
        if !self.grid.contains(q) {
            return 1.0;
        }
        self.grid.sample(&self.transmittance, q).clamp(0.0, 1.0)
    }
}
