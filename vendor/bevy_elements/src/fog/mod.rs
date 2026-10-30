//! Fog: distance fog from the weather's visibility, and a camera-following ground-fog volume
//! whose density a compute kernel writes every frame. See `docs/research/fog.md`.

mod gpu;

use bevy::asset::RenderAssetUsages;
use bevy::asset::embedded_asset;
use bevy::camera::Exposure;
use bevy::light::FogVolume;
use bevy::pbr::{DistanceFog, FogFalloff};
use bevy::prelude::*;
use bevy::render::render_resource::{Extent3d, ShaderType, TextureDimension, TextureFormat};

use crate::celestial::{CelestialFrame, CloudLight, SunState, cloud_light, dusk_tint, scene_illuminance};
use crate::climate::{Weather, WeatherDirector, WeatherPreset};
use crate::clouds::{CloudShadows, CloudWeatherMap};
use crate::gpu::grid::storage_volume;
use crate::plugin::{ElementsCamera, ElementsSystems, ElementsUnits, ElementsView, ensure_gpu_core};
use crate::precipitation::model::{
    marshall_palmer_lambda, precipitation_split, rain_extinction, snow_extinction,
    water_weighted_fall_speed,
};
use crate::query::EnvironmentSnapshot;

/// Koschmieder's constant for a 2 % contrast threshold: extinction = 3.912 / visibility.
pub const KOSCHMIEDER: f32 = 3.912;
/// Visibility of the `Clear` preset, m: the air this crate calls clear. A weather's visibility
/// is measured against it, not against a vacuum (see [`sky_air_extinction`]).
pub const CLEAR_VISIBILITY_M: f32 = 40_000.0;
/// Extinction of clear air, m⁻¹: `KOSCHMIEDER / CLEAR_VISIBILITY_M`, the floor of every
/// visibility a weather can carry.
pub const CLEAR_AIR_EXTINCTION: f32 = KOSCHMIEDER / CLEAR_VISIBILITY_M;
/// Size of the ground-fog volume, m: the near field, around the camera; distance fog carries
/// the far field. Kept small because Bevy attenuates the light reaching every fog sample by the
/// volume's bounding radius (see [`volume_light_compensation`]).
pub const GROUND_FOG_EXTENT: Vec3 = Vec3::new(240.0, 120.0, 240.0);
/// Texels of the ground-fog density texture.
pub const GROUND_FOG_TEXELS: UVec3 = UVec3::new(128, 32, 128);
/// Fog colour in full light, linear RGB.
const FOG_COLOUR: Vec3 = Vec3::new(0.75, 0.80, 0.85);
/// Colour of airborne dust, linear RGB.
pub(crate) const DUST_ALBEDO: Vec3 = Vec3::new(0.76, 0.60, 0.42);
/// Ground-fog scale height in the Fog preset and otherwise, m.
const FOG_SCALE_HEIGHT_M: f32 = 25.0;
const HAZE_SCALE_HEIGHT_M: f32 = 60.0;
/// Game seconds over which the scale height follows a preset change.
const SCALE_HEIGHT_TAU_S: f32 = 600.0;
/// Single-scattering albedo of fog droplets and of dust.
const FOG_ALBEDO: f32 = 0.95;
const DUST_SINGLE_SCATTER: f32 = 0.9;
/// Largest |n| of the 2-octave `fbm3`, measured over 10⁵ points by
/// `the_kernel_normalisation_constants_match_the_noise`; maps the noise onto f ∈ [0, 1].
pub(crate) const FBM2_MAX: f32 = 0.522;
/// Mean of the density modulation 0.6 + 0.4 f with E[f] = 0.5.
const MEAN_NOISE_FACTOR: f32 = 0.8;
/// Lowest sine of the sun's altitude used for the slant path through the fog layer.
const MIN_SUN_SINE: f32 = 0.05;
/// Noise offsets wrap at this many metres to keep f32 precision.
const OFFSET_WRAP_M: f32 = 80.0 * 4096.0;

/// Marks the camera-following ground-fog `FogVolume` this crate spawns.
#[derive(Component, Clone, Copy, Debug, Default, Reflect)]
pub struct GroundFog;

/// Extinction coefficient for a visibility, m-¹ (Koschmieder, 2 % contrast).
pub fn extinction_for_visibility(visibility_m: f32) -> f32 {
    KOSCHMIEDER / visibility_m.max(1.0)
}

/// Extinction of the air the *sky* must carry above clear air, m⁻¹, for a camera whose
/// `DistanceFog` has extinction `fog_extinction_m`: `fog_extinction − CLEAR_AIR_EXTINCTION`,
/// floored at zero. Every mesh is fogged by the camera's distance fog, but a sky pixel already
/// has Bevy's atmosphere (which draws the clear air), so the sky only adds the excess the
/// weather's visibility implies over the `Clear` preset's. Taking the camera's own extinction
/// keeps the sky and the ground on the same fog at all times; 0 without distance fog, and 0 at
/// and above 40 km, so a clear sky is untouched. `docs/research/clouds.md` Deviation 29.
pub fn sky_air_extinction(fog_extinction_m: f32) -> f32 {
    (fog_extinction_m - CLEAR_AIR_EXTINCTION).max(0.0)
}

/// Distance fog for a visibility: grey extinction σ_V in every channel (fog, haze and dust
/// extinguish independently of wavelength), so every colour reaches 2 % transmittance at exactly
/// `visibility_m`; in-scattering at the same rate, so scene and fog weights sum to 1 and a fully
/// fogged pixel takes the fog colour.
pub fn distance_fog_falloff(visibility_m: f32) -> FogFalloff {
    let k = Vec3::splat(extinction_for_visibility(visibility_m));
    FogFalloff::Atmospheric {
        extinction: k,
        inscattering: k,
    }
}

/// Distance fog for an orthographic camera: `near_depth_m` is the depth, along the view axis, of
/// the nearest ground the view shows (`orthographic_near_depth`). Bevy measures fog from the
/// camera's translation, which an isometric camera puts far behind what it shows, so an
/// exponential would fog the whole frame by the empty stand-off. The fog starts at the bottom of
/// the view and rises with the exponential's initial slope `σ_V`: `Linear { start, end: start +
/// 1/σ_V }`, exact while the scene's depth is short of the visibility and fully fogged at a
/// quarter of it (`docs/research/view-anchor.md` Deviation 1).
pub fn orthographic_fog_falloff(visibility_m: f32, near_depth_m: f32) -> FogFalloff {
    let start = near_depth_m.max(0.0);
    FogFalloff::Linear {
        start,
        end: start + 1.0 / extinction_for_visibility(visibility_m),
    }
}

/// A falloff computed in metres as Bevy's world-unit falloff, `units` world units to the metre:
/// distances × `units`, extinction and in-scattering per world unit ÷ `units`.
pub fn falloff_in_world_units(falloff: FogFalloff, units: f32) -> FogFalloff {
    match falloff {
        FogFalloff::Linear { start, end } => FogFalloff::Linear {
            start: start * units,
            end: end * units,
        },
        FogFalloff::Atmospheric {
            extinction,
            inscattering,
        } => FogFalloff::Atmospheric {
            extinction: extinction / units,
            inscattering: inscattering / units,
        },
        other => other,
    }
}

/// Depth along the view axis of the nearest ground an orthographic view shows, m: the ground
/// through `focus` seen at the bottom edge of a view `view_height_m` high, looking down along the
/// unit `forward`. A screen offset `v` lands on the ground `v · cot θ` nearer than the focus, with
/// `θ` the look-down angle (at least 10°); a straight-down view sees all its ground at the focus.
pub fn orthographic_near_depth(eye: Vec3, forward: Vec3, focus: Vec3, view_height_m: f32) -> f32 {
    let down = (-forward.y).clamp(10f32.to_radians().sin(), 1.0);
    let cot = (1.0 - down * down).sqrt() / down;
    (focus - eye).dot(forward) - 0.5 * view_height_m * cot
}

/// Linear-RGB fog colour before brightness: the fog hue, tinted toward sunset orange as the sun
/// drops below 10° (`celestial::dusk_tint`); and toward dust as `dust` rises.
pub fn fog_hue(sun_altitude_deg: f32, dust: f32) -> Vec3 {
    let tinted = FOG_COLOUR * dusk_tint(sun_altitude_deg);
    tinted.lerp(DUST_ALBEDO, dust.clamp(0.0, 1.0))
}

/// Single-scattering albedo of the haze: fog droplets' 0.95, falling to dust's 0.9 as `dust` goes
/// from 0 to 1. Dust adds colour and a little absorption, not extinction: the weather's visibility
/// is already the dust's extinction (`docs/research/dust.md` "Fog consistency").
pub(crate) fn haze_albedo(dust: f32) -> f32 {
    FOG_ALBEDO + (DUST_SINGLE_SCATTER - FOG_ALBEDO) * dust.clamp(0.0, 1.0)
}

/// `FogVolume::light_intensity` that makes the volume's sunlit in-scattering right at the camera.
///
/// Bevy attenuates the light reaching each fog sample by `exp(−density · R)`, with `R` the
/// radius of the volume's bounding sphere (`bevy_pbr` `volumetric_fog.wesl`, the
/// `light_attenuation` term), whatever the sun's direction. In a ground-fog layer the sunlight
/// actually crosses the layer above the sample: optical depth `σ_V · 0.8 · H · e^(−h/H) / sin α`
/// from height `h` to the top along the sun. The returned factor replaces Bevy's attenuation by
/// that one at the camera's height, where most of the visible in-scattering happens, clamped to
/// 1–100.
pub fn volume_light_compensation(
    visibility_m: f32,
    scale_height_m: f32,
    camera_height_m: f32,
    sun_altitude_deg: f32,
    volume_extent: Vec3,
) -> f32 {
    let sigma_v = extinction_for_visibility(visibility_m);
    let falloff = (-camera_height_m.max(0.0) / scale_height_m).exp();
    let sigma_camera = sigma_v * falloff * MEAN_NOISE_FACTOR;
    let radius = (volume_extent * 0.5).length();
    let sine = sun_altitude_deg.to_radians().sin().max(MIN_SUN_SINE);
    let tau_layer = sigma_v * MEAN_NOISE_FACTOR * scale_height_m * falloff / sine;
    ((sigma_camera * radius - tau_layer).exp()).clamp(1.0, MAX_VOLUME_LIGHT_GAIN)
}

/// The share of the volume's extinction that it can actually re-emit: the beam `T̄` over the
/// scene's total light, 0–1.
///
/// Bevy lights a `FogVolume` from the `DirectionalLight`s (`volumetric_fog.wesl`, the
/// in-scattering loop), and under cloud those carry only the beam `T̄`: at full cover the sun's
/// illuminance is literally zero (measured `beam = 0.0000` at cover 1.0), so the volume's
/// in-scattering is zero however large `light_intensity` is. The cloud-scattered diffuse light
/// reaches the camera's `AmbientLight` and never the volume, and Bevy's `VolumetricFog.ambient_*`
/// term is unusable here: it is added once for the whole ray with no density texture and is not
/// multiplied by exposure (see the note). A volume that cannot return light must not take it:
/// without this the ground fog removes the sky and returns nothing, which under a short
/// `Weather::visibility_m` turns the whole sky black — measured at noon looking up, the zenith
/// read `(46, 47, 50)` in Blizzard against `(122, 124, 129)` with the density zeroed, while
/// Snow's 1500 m read `(172, 167, 160)` at the same cover. A blizzard is a whiteout, not a dark
/// hole.
///
/// Scaling the density by this share fades the volume out where Bevy cannot light it, and the
/// volume's radiance over its opacity stays the scene's `total` light whatever the share (see
/// [`volume_light_gain`]), so what remains of the volume looks unchanged while it fades.
pub(crate) fn volume_density_share(cloud: CloudLight) -> f32 {
    let total = cloud.total();
    if total <= 0.0 {
        return 0.0;
    }
    (cloud.direct / total).clamp(0.0, 1.0)
}

/// The factor on `light_intensity` that makes the volume scatter the scene's *total* light: the
/// reciprocal of [`volume_density_share`], the light Bevy does not give the volume.
///
/// The emitted radiance of a step is `density · illuminance · light_intensity`, with the
/// illuminance the beam `T̄` of `lux::RAW_SUNLIGHT`; times a gain of `total / T̄` it carries
/// `total`, the same light the sky and the distance fog carry. Together with the density share it
/// also holds the radiance-to-opacity ratio: `L ∝ share · total` against `τ ∝ share`, so the
/// volume keeps its colour balance as it fades. 1 in clear air, bounded by 100 where the beam is
/// nearly gone (the illuminance there is nearly zero, so the emitted light is too).
pub(crate) fn volume_light_gain(cloud: CloudLight) -> f32 {
    let total = cloud.total();
    if cloud.direct <= 0.0 || total <= 0.0 {
        return 1.0;
    }
    (total / cloud.direct).clamp(1.0, MAX_VOLUME_LIGHT_GAIN)
}

/// Bound on [`volume_light_gain`], shared with the light compensation's own bound.
pub(crate) const MAX_VOLUME_LIGHT_GAIN: f32 = 100.0;

/// Uniform of `density.wesl`.
#[derive(Clone, Copy, Debug, Default, ShaderType)]
pub(crate) struct FogParams {
    origin: Vec4,
    extent: Vec4,
    shape: Vec4,
    size: UVec4,
    /// x: precipitation extinction σ_p, m⁻¹. y: shaft slant per m/s of wind, s/m. z: weather map
    /// extent, m. w: map precipitation at full coverage (`g_nom`).
    precip: Vec4,
    /// xy: wind at 10 m, m/s. zw: weather map origin, world XZ, m.
    slant: Vec4,
}

/// The density texture and this frame's kernel parameters; extracted every frame.
#[derive(Resource, Clone)]
pub(crate) struct FogGpu {
    pub texture: Handle<Image>,
    /// The precipitation map the shaft term samples (`precipitation_map`).
    pub weather_map: Handle<Image>,
    pub params: FogParams,
    pub active: bool,
}

#[derive(Resource)]
struct FogRuntime {
    scale_height: f32,
    wind_offset: Vec2,
    /// A 1 × 1 map with full precipitation everywhere: the shaft term without clouds.
    neutral_map: Handle<Image>,
}

impl Default for FogRuntime {
    fn default() -> Self {
        Self {
            scale_height: HAZE_SCALE_HEIGHT_M,
            wind_offset: Vec2::ZERO,
            neutral_map: Handle::default(),
        }
    }
}

/// Distance fog on every [`ElementsCamera`] and a ground-fog volume around the followed one.
pub struct FogPlugin;

impl Plugin for FogPlugin {
    fn build(&self, app: &mut App) {
        ensure_gpu_core(app);
        embedded_asset!(app, "density.wesl");
        app.init_resource::<FogRuntime>()
            .init_resource::<WeatherDirector>()
            .init_resource::<CelestialFrame>()
            .add_systems(Startup, spawn_volume)
            .add_systems(
                Update,
                (
                    // The clouds move the weather map's origin there; the fog kernel samples it.
                    update_volume.after(crate::clouds::CloudGpuUpdate),
                    // The composite reads the camera's fog colour; write it first, or the sky
                    // lags the ground by a frame (invisible except on a weather change).
                    update_distance_fog.before(crate::clouds::CloudGpuUpdate),
                )
                    .in_set(ElementsSystems::Simulation),
            );
        gpu::plugin(app);
    }
}

fn spawn_volume(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    mut runtime: ResMut<FogRuntime>,
) {
    let texture = storage_volume(&mut images, GROUND_FOG_TEXELS);
    runtime.neutral_map = images.add(Image::new_fill(
        Extent3d::default(),
        TextureDimension::D2,
        &[0, 255, 0, 255],
        TextureFormat::Rgba8Unorm,
        RenderAssetUsages::RENDER_WORLD,
    ));
    commands.spawn((
        Name::new("Ground fog"),
        GroundFog,
        FogVolume {
            density_factor: 0.0,
            density_texture: Some(texture.clone()),
            absorption: 1.0 - FOG_ALBEDO,
            scattering: FOG_ALBEDO,
            ..default()
        },
        Transform::from_scale(GROUND_FOG_EXTENT),
    ));
    commands.insert_resource(FogGpu {
        texture,
        weather_map: Handle::default(),
        params: FogParams::default(),
        active: false,
    });
}

/// Centre of the ground-fog volume: camera XZ snapped to whole texels, the bottom on the
/// texel plane at or below `ground`.
fn volume_centre(camera: Vec3, ground: f32) -> Vec3 {
    let texel = GROUND_FOG_EXTENT / GROUND_FOG_TEXELS.as_vec3();
    let xz = (camera.xz() / texel.xz()).round() * texel.xz();
    let floor = (ground / texel.y).floor() * texel.y;
    Vec3::new(xz.x, floor + GROUND_FOG_EXTENT.y * 0.5, xz.y)
}

#[allow(clippy::too_many_arguments)]
fn update_volume(
    time: Res<Time<Virtual>>,
    view: Res<ElementsView>,
    sun: Res<SunState>,
    weather: Res<Weather>,
    director: Res<WeatherDirector>,
    clock: Res<crate::celestial::WorldClock>,
    shadows: Option<Res<CloudShadows>>,
    mut runtime: ResMut<FogRuntime>,
    gpu: Option<ResMut<FogGpu>>,
    cloud_map: Option<Res<CloudWeatherMap>>,
    mut volumes: Query<(&mut FogVolume, &mut Transform), With<GroundFog>>,
    cameras: Query<&Projection>,
    snapshot: Res<EnvironmentSnapshot>,
    units: Res<ElementsUnits>,
) {
    let Some(mut gpu) = gpu else {
        return;
    };
    let Ok((mut volume, mut transform)) = volumes.single_mut() else {
        return;
    };
    // Bevy extracts every FogVolume regardless of `Visibility`, so a volume is switched off by
    // its density.
    let Some(camera) = view.entity.map(|_| view.position) else {
        gpu.active = false;
        volume.density_factor = 0.0;
        return;
    };

    let dt = time.delta_secs();
    let game_dt = dt * clock.time_scale;
    let goal = if director.current() == WeatherPreset::Fog {
        FOG_SCALE_HEIGHT_M
    } else {
        HAZE_SCALE_HEIGHT_M
    };
    runtime.scale_height +=
        (goal - runtime.scale_height) * (1.0 - (-game_dt / SCALE_HEIGHT_TAU_S).exp());
    let wind = weather.wind_direction * weather.wind_speed_ms;
    runtime.wind_offset = (runtime.wind_offset + wind * dt).rem_euclid(Vec2::splat(OFFSET_WRAP_M));

    let ground = snapshot.ground_height(camera.xz());
    let centre = volume_centre(camera, ground);
    let u = units.per_metre();
    // Bevy's volume is in world units: its transform × u (the density kernel stays in metres).
    *transform = Transform::from_translation(centre * u).with_scale(GROUND_FOG_EXTENT * u);
    // An orthographic view looks down through the whole layer at every pixel along one slant: a
    // uniform veil set by the look-down angle and the layer's height, not by the scene, opaque
    // under `Fog`. Its distance fog, counted from the focus plane, carries the weather's whole
    // extinction instead, and the particles the precipitation (`docs/research/view-anchor.md`
    // Deviation 4).
    if view
        .entity
        .and_then(|e| cameras.get(e).ok())
        .is_some_and(|p| matches!(p, Projection::Orthographic(_)))
    {
        gpu.active = false;
        volume.density_factor = 0.0;
        return;
    }

    let dust = weather.dust.clamp(0.0, 1.0);
    // Precipitation takes its share of the visibility's extinction as shafts (see
    // `docs/research/precipitation.md`); water and ice do not absorb, so that share scatters
    // with albedo 1.
    let snow_fraction = weather.snow_fraction();
    let split = precipitation_split(
        weather.visibility_m,
        weather.precipitation_mm_h,
        snow_fraction,
    );
    let precipitation_share =
        split.precipitation / (MEAN_NOISE_FACTOR * split.haze + split.precipitation).max(1e-9);
    let haze_albedo = haze_albedo(dust);
    let albedo = haze_albedo + (1.0 - haze_albedo) * precipitation_share;
    volume.absorption = 1.0 - albedo;
    volume.scattering = albedo;
    // A volume Bevy cannot light must not occlude, and what it can light must carry the scene's
    // total light: the density is faded by the share its own light source carries and the
    // intensity raised by the reciprocal (see `volume_density_share` and `volume_light_gain`).
    let cloud = cloud_light(weather.cloud_cover, shadows.as_deref());
    // Bevy's density is per world unit: the metre density ÷ u keeps the optical depth.
    volume.density_factor = volume_density_share(cloud) / u;
    volume.light_intensity = volume_light_gain(cloud)
        * volume_light_compensation(
            weather.visibility_m,
            runtime.scale_height,
            camera.y - ground,
            sun.altitude_deg,
            GROUND_FOG_EXTENT,
        );
    volume.fog_color = Color::linear_rgb(1.0, 1.0, 1.0).mix(
        &Color::linear_rgb(DUST_ALBEDO.x, DUST_ALBEDO.y, DUST_ALBEDO.z),
        dust,
    );

    // Rain falls slanted by the wind: per m of height the shaft shifts by `slant · wind`, with
    // `slant` the extinction-weighted mean of 1 / fall speed (snow falls at 1 m/s).
    let snow_rate = weather.precipitation_mm_h * snow_fraction;
    let rain_rate = weather.precipitation_mm_h - snow_rate;
    let (rain_sigma, snow_sigma) = (rain_extinction(rain_rate), snow_extinction(snow_rate));
    let slant = if split.precipitation > 0.0 {
        let rain_speed = water_weighted_fall_speed(marshall_palmer_lambda(rain_rate));
        (rain_sigma / rain_speed + snow_sigma) / (rain_sigma + snow_sigma)
    } else {
        0.0
    };
    let (map, map_origin, map_extent) =
        precipitation_map(cloud_map.as_deref(), &runtime.neutral_map);
    gpu.weather_map = map;
    gpu.params = FogParams {
        origin: (centre - GROUND_FOG_EXTENT * 0.5).extend(1.0 / FBM2_MAX),
        extent: GROUND_FOG_EXTENT.extend(split.haze),
        shape: Vec4::new(
            runtime.scale_height,
            ground,
            runtime.wind_offset.x,
            runtime.wind_offset.y,
        ),
        size: GROUND_FOG_TEXELS.extend(0),
        precip: Vec4::new(
            split.precipitation,
            slant,
            map_extent,
            (weather.precipitation_mm_h / 25.0).clamp(0.0, 1.0),
        ),
        slant: Vec4::new(wind.x, wind.y, map_origin.x, map_origin.y),
    };
    gpu.active = true;
}

/// The map whose green channel is the precipitation at full coverage under cloud, with the world
/// XZ of its minimum corner and its extent in m: the clouds' weather map, so rain shafts fall
/// only under precipitating cloud. With `CloudPlugin` disabled it is the neutral 1 × 1 map
/// (green = 1 everywhere), which reduces the shaft term to its noise.
fn precipitation_map(
    clouds: Option<&CloudWeatherMap>,
    neutral: &Handle<Image>,
) -> (Handle<Image>, Vec2, f32) {
    match clouds {
        Some(map) => (map.map.clone(), map.origin, map.extent),
        None => (neutral.clone(), Vec2::ZERO, 1.0),
    }
}

/// The cameras whose distance fog the crate may drive, with what fixes their falloff.
type FogCameras<'w, 's> = Query<
    'w,
    's,
    (
        Entity,
        &'static ElementsCamera,
        &'static mut DistanceFog,
        Option<&'static Exposure>,
        Option<&'static Projection>,
        &'static GlobalTransform,
    ),
>;

fn update_distance_fog(
    weather: Res<Weather>,
    sun: Res<SunState>,
    frame: Res<CelestialFrame>,
    view: Res<ElementsView>,
    units: Res<ElementsUnits>,
    shadows: Option<Res<CloudShadows>>,
    mut cameras: FogCameras,
) {
    // Fog colour is added after exposure (bevy_pbr fog.wesl), so it carries the scene's
    // luminance times the camera's exposure: lit like everything else at a fixed exposure,
    // steady under auto exposure.
    let luminance = scene_illuminance(
        &sun,
        frame.moon_lux,
        cloud_light(weather.cloud_cover, shadows.as_deref()),
    ) / std::f32::consts::PI;
    let hue = fog_hue(sun.altitude_deg, weather.dust);
    for (entity, dressing, mut fog, exposure, projection, transform) in &mut cameras {
        if !dressing.distance_fog {
            continue;
        }
        let exposure = exposure.copied().unwrap_or_default().exposure();
        let brightness = luminance * exposure;
        let c = hue * brightness;
        fog.color = Color::linear_rgb(c.x, c.y, c.z);
        fog.falloff = match projection {
            Some(Projection::Orthographic(o)) => {
                // The followed camera's focus; another orthographic camera fogs from its plane.
                let eye = units.point_to_metres(transform.translation());
                let focus = if view.entity == Some(entity) {
                    view.position
                } else {
                    eye
                };
                let near = orthographic_near_depth(
                    eye,
                    *transform.forward(),
                    focus,
                    units.metres(o.area.height()),
                );
                falloff_in_world_units(
                    orthographic_fog_falloff(weather.visibility_m, near),
                    units.per_metre(),
                )
            }
            _ => falloff_in_world_units(
                distance_fog_falloff(weather.visibility_m),
                units.per_metre(),
            ),
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A world in feet: Bevy's distances are the metre ones times the units and its extinction
    /// per world unit the metre one over them, so the fog reaches the same metres (issue #8);
    /// the default 1.0 changes nothing.
    #[test]
    fn a_falloff_in_feet_fogs_the_same_metres() {
        let feet = 3.2808_f32;
        let metres = distance_fog_falloff(300.0);
        assert_eq!(
            format!("{:?}", falloff_in_world_units(metres.clone(), 1.0)),
            format!("{metres:?}")
        );
        let (
            FogFalloff::Atmospheric {
                extinction: m,
                inscattering: mi,
            },
            FogFalloff::Atmospheric {
                extinction: f,
                inscattering: fi,
            },
        ) = (metres, falloff_in_world_units(distance_fog_falloff(300.0), feet))
        else {
            unreachable!("distance fog is atmospheric");
        };
        // 300 m is 984.24 ft; the transmittance over it is the metre one.
        let (tm, tf) = ((-m * 300.0).exp(), (-f * 300.0 * feet).exp());
        assert!((tm - tf).abs().max_element() < 1e-6, "{tm} {tf}");
        assert!((fi - mi / feet).abs().max_element() < 1e-9);

        // An orthographic falloff starts and ends at the same metres.
        let FogFalloff::Linear { start, end } =
            falloff_in_world_units(orthographic_fog_falloff(300.0, 1500.0), feet)
        else {
            unreachable!("an orthographic view fogs linearly");
        };
        assert!((start / feet - 1500.0).abs() < 1e-2);
        assert!((end / feet - (1500.0 + 300.0 / KOSCHMIEDER)).abs() < 1e-2);
    }

    /// The nearest ground a view 40 m high shows, looking down 30° at a focus 1500 m away, is
    /// 20 · cot 30° = 34.6 m in front of the focus; straight down, all of it is at the focus; a
    /// grazing view is held at 10°.
    #[test]
    fn an_orthographic_view_fogs_from_the_bottom_of_its_ground() {
        let focus = Vec3::new(3.0, 0.0, -4.0);
        let at = |down_deg: f32| {
            let d = down_deg.to_radians();
            let forward = Vec3::new(0.0, -d.sin(), -d.cos());
            orthographic_near_depth(focus - forward * 1500.0, forward, focus, 40.0)
        };
        assert!(
            (at(30.0) - (1500.0 - 20.0 * 3f32.sqrt())).abs() < 1e-2,
            "{}",
            at(30.0)
        );
        assert!((at(90.0) - 1500.0).abs() < 1e-2, "{}", at(90.0));
        assert!((at(2.0) - at(10.0)).abs() < 1e-2);
    }

    #[test]
    fn the_fog_volume_stands_on_the_ground() {
        let texel = GROUND_FOG_EXTENT / GROUND_FOG_TEXELS.as_vec3();
        let camera = Vec3::new(13.7, 500.0, -2.2);
        for ground in [412.3, -37.0, 0.0] {
            let centre = volume_centre(camera, ground);
            let bottom = centre.y - GROUND_FOG_EXTENT.y * 0.5;
            assert!(
                bottom <= ground && bottom > ground - texel.y,
                "bottom {bottom} for ground {ground}"
            );
            let cells = centre.xz() / texel.xz();
            assert!(
                (cells - cells.round()).abs().max_element() < 1e-3,
                "centre {centre} is not on a texel"
            );
        }
    }

    #[test]
    fn the_skys_air_is_the_cameras_fog_extinction_above_clear_air() {
        // The camera's extinction, not a visibility: whatever `DistanceFog` the meshes carry.
        let air = |visibility: f32| sky_air_extinction(extinction_for_visibility(visibility));
        // Clear air adds nothing: the atmosphere is the whole sky.
        assert_eq!(air(CLEAR_VISIBILITY_M), 0.0);
        assert_eq!(air(80_000.0), 0.0);
        // The presets: 3.912/V minus the 40 km floor.
        assert!((air(200.0) - 1.946_22e-2).abs() < 1e-9);
        assert!((air(300.0) - 1.294_22e-2).abs() < 1e-9);
        assert!((air(1_500.0) - 2.510_2e-3).abs() < 1e-9);
        assert!((air(20_000.0) - 9.78e-5).abs() < 1e-9);
        // Monotone in visibility: longer sight never adds more air.
        let mut last = f32::MAX;
        for visibility in [200.0, 300.0, 1_500.0, 5_000.0, 20_000.0, 40_000.0, 80_000.0] {
            let a = air(visibility);
            assert!(a <= last, "{a} > {last} at {visibility} m");
            last = a;
        }
    }

    #[test]
    fn every_channel_reaches_two_percent_at_the_visibility_and_energy_is_conserved() {
        let FogFalloff::Atmospheric {
            extinction,
            inscattering,
        } = distance_fog_falloff(300.0)
        else {
            unreachable!("distance fog is always atmospheric");
        };
        assert!((extinction - Vec3::splat(0.013_040)).abs().max_element() < 1e-6);
        assert_eq!(extinction, inscattering);
        let t = (-extinction * 300.0).exp();
        assert!((t - Vec3::splat(0.0200)).abs().max_element() < 2e-4);
        for d in [0.0, 50.0, 300.0, 5000.0] {
            let scene = (-extinction * d).exp();
            let fog = Vec3::ONE - (-inscattering * d).exp();
            assert!(((scene + fog) - Vec3::ONE).abs().max_element() < 1e-6);
        }
    }

    #[test]
    fn the_volume_light_matches_the_layer_attenuation_at_the_camera() {
        // Fog preset, camera 2 m up, sun at 48°: Bevy's exp(−σR) over the 180 m radius is
        // undone and the slant layer depth σ_V·0.8·H·e^(−h/H)/sin α applied instead.
        let k = volume_light_compensation(300.0, 25.0, 2.0, 48.0, GROUND_FOG_EXTENT);
        let sigma_v = 3.912_f32 / 300.0;
        let falloff = (-2.0_f32 / 25.0).exp();
        let radius = (GROUND_FOG_EXTENT * 0.5).length();
        let expected = (sigma_v * falloff * 0.8 * radius
            - sigma_v * 0.8 * 25.0 * falloff / 48f32.to_radians().sin())
        .exp();
        assert!((k - expected).abs() < 1e-3 * expected, "{k} vs {expected}");
        // Clear air needs next to no help.
        let clear = volume_light_compensation(40_000.0, 60.0, 2.0, 48.0, GROUND_FOG_EXTENT);
        assert!((1.0..1.01).contains(&clear), "{clear}");
    }

    #[test]
    fn the_volume_goes_out_where_bevy_cannot_light_it() {
        // Clear sky: the beam is the whole light, so the volume keeps its full density.
        assert_eq!(volume_density_share(CloudLight::CLEAR), 1.0);
        // Full cover: Bevy's light for the volume (the beam) is zero — measured, the sun's
        // illuminance is literally 0 — so the volume must not occlude anything: a blizzard is a
        // whiteout, not a dark hole. This is the dark sky the fix replaces: the volume took the
        // sky's light and returned none of it.
        assert_eq!(volume_density_share(CloudLight::from_beam(0.0)), 0.0);
        // Half cover: the share is the beam over the beam-plus-diffuse total.
        let half = volume_density_share(CloudLight::from_beam(0.5));
        assert!((half - 0.5 / 0.625).abs() < 1e-6, "{half}");
        // Monotone in the beam, so a transition fades the volume in as the cloud breaks.
        let mut last = -1.0;
        for i in 0..=20 {
            let share = volume_density_share(CloudLight::from_beam(i as f32 / 20.0));
            assert!(share >= last, "not monotone at {i}");
            last = share;
        }
    }

    #[test]
    fn the_volumes_light_and_its_density_are_reciprocal() {
        // The density fade and the light boost are exact reciprocals, so the volume's radiance
        // over its opacity is the scene's total light — the same light the sky and the distance
        // fog carry — at every cover. Without that the ground fog would dim as the cloud thickens
        // instead of fading out evenly.
        for i in 0..=20 {
            let beam = 0.05 + i as f32 / 20.0 * 0.95;
            let cloud = CloudLight::from_beam(beam);
            let product = volume_density_share(cloud) * volume_light_gain(cloud);
            assert!((product - 1.0).abs() < 1e-3, "beam {beam}: {product}");
        }
        // A bright beam keeps the gain at 1; a nearly gone one is bounded.
        assert_eq!(volume_light_gain(CloudLight::CLEAR), 1.0);
        assert_eq!(
            volume_light_gain(CloudLight::from_beam(1e-6)),
            MAX_VOLUME_LIGHT_GAIN
        );
    }

    #[test]
    fn the_hue_tints_at_dusk_and_in_dust() {
        assert_eq!(fog_hue(90.0, 0.0), FOG_COLOUR);
        let dusk = fog_hue(5.0, 0.0) / FOG_COLOUR;
        assert!((dusk - Vec3::new(1.0, 0.775, 0.65)).length() < 1e-5);
        assert!((fog_hue(45.0, 1.0) - DUST_ALBEDO).length() < 1e-6);
        assert_eq!(fog_hue(0.0, 0.0), FOG_COLOUR * crate::celestial::DUSK_TINT);
        assert_eq!(fog_hue(-20.0, 0.0), FOG_COLOUR);
    }

    /// `dust.md` Validation 9: the dust storm's haze is the preset's visibility and nothing more,
    /// with albedo and colour moved toward the dust's.
    #[test]
    fn a_dust_storm_changes_the_haze_colour_and_albedo_not_its_extinction() {
        use crate::precipitation::model::precipitation_split;
        // Preset: visibility 800 m, no precipitation, dust 0.9.
        let sigma_v = extinction_for_visibility(800.0);
        assert!((sigma_v - 4.890e-3).abs() < 1e-6, "{sigma_v}");
        let split = precipitation_split(800.0, 0.0, 0.0);
        // The density kernel's haze amplitude is `split.haze`, with or without dust: no dust input.
        assert_eq!(split.haze, sigma_v);
        assert_eq!(split.precipitation, 0.0);
        // Albedo 0.95 → 0.9 with dust (0.905 at the preset's 0.9), colour toward the dust's.
        assert_eq!(haze_albedo(0.0), FOG_ALBEDO);
        assert!(
            (haze_albedo(0.9) - 0.9).abs() < 0.006,
            "{}",
            haze_albedo(0.9)
        );
        assert!((haze_albedo(1.0) - DUST_SINGLE_SCATTER).abs() < 1e-6);
        let colour = Vec3::ONE.lerp(DUST_ALBEDO, 0.9);
        assert!(
            (colour - Vec3::new(0.784, 0.64, 0.478)).length() < 1e-3,
            "{colour}"
        );
        // The camera's distance fog takes the same hue at the preset's dust.
        let hue = fog_hue(45.0, 0.9);
        assert!(hue.x > hue.y && hue.y > hue.z, "{hue}");
    }

    /// CPU transliteration of `fog_density` in `density.wesl` with the neutral weather map
    /// (g = 1): the haze term and the shaft term at `p`, with the noise passed through `fbm`.
    fn kernel(params: &FogParams, p: Vec3, wind_offset: Vec2) -> (f32, f32) {
        use crate::gpu::noise_mirror::fbm3;
        let unit = |n: f32| (0.5 + 0.5 * n * params.origin.w).clamp(0.0, 1.0);
        let h = (p.y - params.shape.y).max(0.0);
        let q = (p - Vec3::new(wind_offset.x, 0.0, wind_offset.y)) / 80.0;
        let haze = params.extent.w * (-h / params.shape.x).exp() * (0.6 + 0.4 * unit(fbm3(q, 2)));
        let y_ref = params.origin.y + 0.5 * params.extent.y;
        let x_ref = p.xz() + params.slant.xy() * params.precip.y * (p.y - y_ref);
        let coverage = (1.0 / params.precip.w.max(1e-4)).clamp(0.0, 1.0);
        let qs = Vec3::new(
            (x_ref.x - wind_offset.x) / 120.0,
            0.37,
            (x_ref.y - wind_offset.y) / 120.0,
        );
        let s = coverage * (0.4 + 1.2 * unit(fbm3(qs, 2)));
        (haze, params.precip.x * s)
    }

    /// Validation 8 of `docs/research/precipitation.md`: forced Rain at 4 mm/h with the neutral
    /// map. The slab mean of the density 60 m above the ground is
    /// `e^−1 · 0.8 · σ_haze + σ_p` = 7.68e-4; the shaft factor has mean 1; the shaft optical depth
    /// over the volume's 120 m height is σ_p · 120 = 0.065.
    #[test]
    fn the_kernel_with_the_shaft_term_matches_the_note() {
        use crate::gpu::noise_mirror::samples;
        let split = precipitation_split(3000.0, 4.0, 0.0);
        let params = FogParams {
            origin: Vec3::new(-120.0, 0.0, -120.0).extend(1.0 / FBM2_MAX),
            extent: GROUND_FOG_EXTENT.extend(split.haze),
            shape: Vec4::new(HAZE_SCALE_HEIGHT_M, 0.0, 0.0, 0.0),
            size: GROUND_FOG_TEXELS.extend(0),
            precip: Vec4::new(split.precipitation, 0.0, 1.0, 4.0 / 25.0),
            slant: Vec4::ZERO,
        };
        let n = 100_000;
        let (mut total, mut shaft_factor) = (0.0_f64, 0.0_f64);
        for p in samples(n, 20_000.0) {
            let p = Vec3::new(p.x, 60.0, p.z);
            let (haze, shafts) = kernel(&params, p, Vec2::ZERO);
            total += f64::from(haze + shafts);
            shaft_factor += f64::from(shafts / split.precipitation);
        }
        let (total, shaft_factor) = (total / n as f64, shaft_factor / n as f64);
        let expected = (-1.0_f64).exp() * 0.8 * 7.598e-4 + 5.442e-4;
        println!(
            "slab mean {total:.4e} (expected {expected:.4e}), shaft factor mean {shaft_factor:.4}"
        );
        assert!((expected - 7.68e-4).abs() < 1e-6, "{expected}");
        assert!(
            (total / expected - 1.0).abs() < 0.03,
            "{total:e} vs {expected:e}"
        );
        assert!(
            (shaft_factor - 1.0).abs() < 0.02,
            "shaft factor mean {shaft_factor}"
        );
        let optical_depth = f64::from(split.precipitation) * 120.0;
        assert!((optical_depth - 0.065).abs() < 0.0005, "{optical_depth}");
    }
}
