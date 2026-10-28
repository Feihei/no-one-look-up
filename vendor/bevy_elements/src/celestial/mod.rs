//! Sun, moon and stars from the real sky: a world clock, Meeus-accuracy ephemerides, the sun and
//! moon as directional lights for Bevy's atmosphere, and exposure that follows them.

pub mod astro;
mod clock;
mod moon;
mod stars;

use bevy::camera::Exposure;
use bevy::light::light_consts::lux;
use bevy::light::{CascadeShadowConfigBuilder, SunDisk, VolumetricLight};
use bevy::prelude::*;

pub(crate) use clock::plugin as clock_plugin;
pub use clock::{CivilDate, GeoLocation, WorldClock};

use crate::climate::Weather;
use crate::clouds::CloudShadows;
use crate::plugin::{ElementsCamera, ElementsSystems, ensure_gpu_core};

/// Marks the sun's `DirectionalLight`.
#[derive(Component, Clone, Copy, Debug, Default, Reflect)]
pub struct Sun;

/// Marks the moon's `DirectionalLight`.
#[derive(Component, Clone, Copy, Debug, Default, Reflect)]
pub struct Moon;

/// Where the sun and moon are and how much of the moon is lit.
#[derive(Resource, Clone, Copy, Debug, Reflect)]
pub struct SunState {
    /// Unit world vector toward the sun (refracted), +Y up, east +X, north −Z.
    pub direction: Vec3,
    /// True (unrefracted) altitude of the sun's centre, degrees.
    pub altitude_deg: f32,
    /// Azimuth of the sun from north through east, degrees.
    pub azimuth_deg: f32,
    /// Unit world vector toward the moon (topocentric, refracted).
    pub moon_direction: Vec3,
    /// True topocentric altitude of the moon's centre, degrees.
    pub moon_altitude_deg: f32,
    /// Place in the synodic month: 0 new, 0.25 first quarter, 0.5 full, 0.75 last quarter.
    pub moon_phase: f32,
    /// Fraction of the moon's disc that is lit, 0–1.
    pub moon_illuminated_fraction: f32,
}

impl Default for SunState {
    fn default() -> Self {
        Self {
            direction: Vec3::Y,
            altitude_deg: 90.0,
            azimuth_deg: 180.0,
            moon_direction: Vec3::NEG_Y,
            moon_altitude_deg: -90.0,
            moon_phase: 0.0,
            moon_illuminated_fraction: 0.0,
        }
    }
}

/// Sky geometry for the star field and the moon disc, recomputed every frame.
#[derive(Resource, Clone, Copy, Debug)]
pub(crate) struct CelestialFrame {
    /// Takes a J2000 equatorial unit vector `(cos δ cos α, cos δ sin α, sin δ)` to world space
    /// (precession to date, Earth rotation, latitude).
    pub j2000_to_world: Mat3,
    /// Unit world vector toward the north ecliptic pole of date.
    pub ecliptic_north: Vec3,
    /// Angular diameter of the moon's disc as seen by the observer, degrees.
    pub moon_angular_diameter_deg: f32,
    /// Illuminance of the moon outside the atmosphere, lx.
    pub moon_lux: f32,
}

impl Default for CelestialFrame {
    fn default() -> Self {
        Self {
            j2000_to_world: Mat3::IDENTITY,
            ecliptic_north: Vec3::Y,
            moon_angular_diameter_deg: 0.5,
            moon_lux: 0.0,
        }
    }
}

/// Sun, moon and exposure.
///
/// Spawns one [`Sun`] and one [`Moon`] `DirectionalLight` at startup and moves them every frame
/// in [`ElementsSystems::Celestial`]; keeps [`SunState`] current; drives `Exposure` on every
/// [`ElementsCamera`] with `auto_exposure` set.
pub struct CelestialPlugin;

impl Plugin for CelestialPlugin {
    fn build(&self, app: &mut App) {
        ensure_gpu_core(app);
        stars::plugin(app);
        moon::plugin(app);
        app.init_resource::<CelestialFrame>()
            .add_systems(Startup, spawn_lights)
            .add_systems(
                Update,
                (update_sky, update_ambient, auto_exposure)
                    .chain()
                    .in_set(ElementsSystems::Celestial),
            );
    }
}

/// Atmospheric pressure and temperature the refraction correction assumes, mbar and °C
/// (Saemundsson's reference state).
const REFRACTION_PRESSURE_MBAR: f64 = 1010.0;
const REFRACTION_TEMP_C: f64 = 10.0;
/// Below this true altitude a light is switched off, degrees.
const SUN_OFF_DEG: f32 = -2.0;
const MOON_OFF_DEG: f32 = -0.5;
/// Moon shadows are drawn only when the sun is lower than this, degrees (civil twilight).
const MOON_SHADOWS_BELOW_SUN_DEG: f32 = -6.0;

/// The moon light's colour, the tint of the moon's share of the sky's diffuse light and of the
/// moonlit cloud.
pub(crate) const MOON_TINT: Vec3 = Vec3::new(0.88, 0.92, 1.0);

/// How far the sun's shadow cascades reach, m. Volumetric fog takes its sunlight through the
/// shadow map, so beyond the last cascade fog gets no light at all and turns into a dark band
/// toward the horizon; the cascades cover the ground-fog volume (±400 m) with margin.
const SUN_SHADOW_DISTANCE_M: f32 = 600.0;

fn spawn_lights(mut commands: Commands) {
    commands.spawn((
        Name::new("Sun"),
        Sun,
        DirectionalLight {
            illuminance: lux::RAW_SUNLIGHT,
            shadow_maps_enabled: true,
            ..default()
        },
        CascadeShadowConfigBuilder {
            maximum_distance: SUN_SHADOW_DISTANCE_M,
            first_cascade_far_bound: 15.0,
            ..default()
        }
        .build(),
        VolumetricLight,
        Transform::default(),
    ));
    commands.spawn((
        Name::new("Moon"),
        Moon,
        DirectionalLight {
            illuminance: 0.0,
            shadow_maps_enabled: false,
            color: Color::srgb(MOON_TINT.x, MOON_TINT.y, MOON_TINT.z),
            ..default()
        },
        SunDisk::OFF,
        Transform::default(),
    ));
}

/// East-north-up to world: +X east, +Y up, −Z north.
fn enu_to_world(v: [f64; 3]) -> Vec3 {
    Vec3::new(v[0] as f32, v[2] as f32, -v[1] as f32)
}

/// Fraction of the clear-sky global irradiance that remains under complete overcast: Kasten &
/// Czeplak (1980, Solar Energy 24: 177) `G = G_clear · (1 − 0.75 · N^3.4)` at `N = 1`. With the
/// beam gone, all of it is diffuse (`D/G = 0.3 + 0.7 N² = 1` at `N = 1`).
pub(crate) const OVERCAST_DIFFUSE: f32 = 0.25;

/// Mean direct transmittance of the sun's beam when only the cover is known: the cloud shadow
/// map's mean at the camera before its first readback is `1 − cover^3.4`, the exponent of
/// Kasten & Czeplak's global-irradiance fit.
pub(crate) fn beam_transmittance(cover: f32) -> f32 {
    1.0 - cover.clamp(0.0, 1.0).powf(3.4)
}

/// How cloud divides the light of the sun and the moon, as fractions of their clear-sky
/// illuminance.
///
/// A point of the ground is either lit by the beam or shut off from it by cloud, so the mean beam
/// is the transmittance `T̄` of the cloud shadow map; where the beam is cut the ground has the
/// overcast remainder [`OVERCAST_DIFFUSE`], all of it light the cloud scatters. Hence
/// `direct = T̄` and `diffuse = 0.25 · (1 − T̄)`: `total = 0.25 + 0.75 · T̄`, which is Kasten &
/// Czeplak's `1 − 0.75 · N^3.4` when `T̄ = 1 − N^3.4`. `direct` goes to the `DirectionalLight`s
/// (which cast shadows and feed Bevy's atmosphere, so the sky light follows it too) and
/// `diffuse` to the camera's `AmbientLight`. `total` is what the exposure meters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct CloudLight {
    /// Share of the clear-sky illuminance that still arrives as a beam, 0–1.
    pub direct: f32,
    /// Share of it that cloud scatters into the sky's diffuse light, 0–0.25.
    pub diffuse: f32,
}

impl CloudLight {
    /// No cloud: everything direct.
    pub(crate) const CLEAR: Self = Self {
        direct: 1.0,
        diffuse: 0.0,
    };

    /// The split for a mean beam transmittance `beam` (clamped to 0–1).
    pub(crate) fn from_beam(beam: f32) -> Self {
        let direct = beam.clamp(0.0, 1.0);
        Self {
            direct,
            diffuse: OVERCAST_DIFFUSE * (1.0 - direct),
        }
    }

    /// Direct plus diffuse: the factor on the clear-sky illuminance.
    pub(crate) fn total(self) -> f32 {
        self.direct + self.diffuse
    }
}

/// The light split under the current cloud: with the cloud shadow map, its mean transmittance at
/// the camera (measured); without [`CloudShadows`], the cover-only [`beam_transmittance`].
pub(crate) fn cloud_light(cover: f32, shadows: Option<&CloudShadows>) -> CloudLight {
    CloudLight::from_beam(shadows.map_or_else(
        || beam_transmittance(cover),
        |shadows| shadows.camera_transmittance,
    ))
}

/// Smallest beam fraction [`sky_gain`] divides by.
const MIN_SKY_BEAM: f32 = 0.001;

/// The factor that keeps the sky's brightness where it was. Bevy's atmosphere is lit by the
/// sun's `DirectionalLight`, which under cloud carries only the beam `T̄`, so the sky dome (and
/// the sun disc in a gap) would dim to `T̄` of its clear value with the shadow the camera stands
/// in, and go black under full cloud. The clouds compose the sky pixels with `total / T̄`, which
/// restores the dome to the total factor `0.25 + 0.75 T̄` it has always had (a limit of the
/// change, not physics: the sky above a cloud layer is not dimmed by it at all [INFERENCE]).
/// 1 when the sun's light is off (the moon's sky and the stars are not rescaled).
/// [PATCH heads-up] How much of the gain survives at low sun altitude. The
/// crate assumes the sky above a cloud deck is not dimmed by it, which holds
/// when the sun is high. At dusk the sun sits within a few degrees of the
/// horizon, so a thick deck really does stand between the viewer and the lit
/// part of the sky, and the full gain (1.58 at cover 0.90) reads as the sky
/// getting *brighter* the more cloud there is. Fading the gain out as the sun
/// approaches the horizon removes that inversion without touching the physics
/// at high sun.
///
/// 1 above `SKY_GAIN_SUN_HIGH_DEG`, 0 below `SKY_GAIN_SUN_LOW_DEG`.
const SKY_GAIN_SUN_HIGH_DEG: f32 = 25.0;
const SKY_GAIN_SUN_LOW_DEG: f32 = 2.0;

fn dusk_gain_fade(altitude_deg: f32) -> f32 {
    let t = ((altitude_deg - SKY_GAIN_SUN_LOW_DEG)
        / (SKY_GAIN_SUN_HIGH_DEG - SKY_GAIN_SUN_LOW_DEG))
        .clamp(0.0, 1.0);
    // Smoothstep, so the transition has no visible kink as the sun moves.
    t * t * (3.0 - 2.0 * t)
}

pub(crate) fn sky_gain(state: &SunState, cloud: CloudLight) -> f32 {
    if state.altitude_deg > SUN_OFF_DEG {
        let gain = cloud.total() / cloud.direct.max(MIN_SKY_BEAM);
        // [PATCH heads-up] blend the gain away toward the horizon.
        1.0 + (gain - 1.0) * dusk_gain_fade(state.altitude_deg)
    } else {
        1.0
    }
}

fn light_transform(toward: Vec3) -> Transform {
    let up = if toward.y.abs() > 0.999 {
        Vec3::Z
    } else {
        Vec3::Y
    };
    Transform::default().looking_to(-toward, up)
}

pub(crate) fn update_sky(
    clock: Res<WorldClock>,
    location: Res<GeoLocation>,
    weather: Res<Weather>,
    shadows: Option<Res<CloudShadows>>,
    mut state: ResMut<SunState>,
    mut frame: ResMut<CelestialFrame>,
    mut lights: Query<(&mut Transform, &mut DirectionalLight, Has<Sun>, Has<Moon>)>,
) {
    use astro::*;
    let jd = clock.julian_day(location.utc_offset_hours);
    let year = f64::from(clock.date.year) + (f64::from(clock.date.month) - 0.5) / 12.0;
    let t = centuries(jd + delta_t_seconds(year) / 86_400.0);
    let nut = nutation(t);
    let lat = location.latitude_deg;
    let lst = local_sidereal_deg(jd, &nut, location.longitude_deg);

    let (sun, sun_lambda) = sun_apparent(t);
    let (sun_alt, sun_az) = horizontal(sun.ra, sun.dec, lat, lst);
    let sun_app = sun_alt + refraction_deg(sun_alt, REFRACTION_PRESSURE_MBAR, REFRACTION_TEMP_C);
    let sun_dir = enu_to_world(enu(sun_app, sun_az)).normalize_or(Vec3::Y);

    let (moon, moon_ecl) = moon_apparent(t, &nut);
    let (geo_alt, geo_az) = horizontal(moon.ra, moon.dec, lat, lst);
    let (moon_alt, moon_az) = topocentric(geo_alt, geo_az, moon.distance_km);
    let moon_app = moon_alt + refraction_deg(moon_alt, REFRACTION_PRESSURE_MBAR, REFRACTION_TEMP_C);
    let moon_dir = enu_to_world(enu(moon_app, moon_az)).normalize_or(Vec3::NEG_Y);
    let i = phase_angle(&sun, &moon);
    let lit = illuminated_fraction(i);
    let synodic = normalize_deg(moon_ecl.longitude + nut.dpsi - sun_lambda) / 360.0;
    let observer_distance = moon.distance_km - EARTH_RADIUS_KM * (moon_alt.to_radians()).sin();

    *state = SunState {
        direction: sun_dir,
        altitude_deg: sun_alt as f32,
        azimuth_deg: sun_az as f32,
        moon_direction: moon_dir,
        moon_altitude_deg: moon_alt as f32,
        moon_phase: synodic as f32,
        moon_illuminated_fraction: lit as f32,
    };

    let to_enu = mat_mul(equatorial_to_enu(lat, lst), precession_matrix(t));
    let col = |j: usize| enu_to_world([to_enu[0][j], to_enu[1][j], to_enu[2][j]]);
    let eps = nut.eps().to_radians();
    let pole = [0.0, -eps.sin(), eps.cos()];
    let pole_enu = equatorial_to_enu(lat, lst);
    let pole_world = enu_to_world([
        (0..3).map(|k| pole_enu[0][k] * pole[k]).sum(),
        (0..3).map(|k| pole_enu[1][k] * pole[k]).sum(),
        (0..3).map(|k| pole_enu[2][k] * pole[k]).sum(),
    ]);
    let moon_lux = (FULL_MOON_LUX * moon_phase_factor(i)) as f32;
    *frame = CelestialFrame {
        j2000_to_world: Mat3::from_cols(col(0), col(1), col(2)),
        ecliptic_north: pole_world.normalize_or(Vec3::Y),
        moon_angular_diameter_deg: moon_angular_diameter(observer_distance) as f32,
        moon_lux,
    };

    let cloud = cloud_light(weather.cloud_cover, shadows.as_deref());
    for (mut transform, mut light, is_sun, is_moon) in &mut lights {
        if is_sun {
            *transform = light_transform(sun_dir);
            let up = state.altitude_deg > SUN_OFF_DEG;
            light.illuminance = if up {
                lux::RAW_SUNLIGHT * cloud.direct
            } else {
                0.0
            };
            set_shadows(&mut light, up);
        } else if is_moon {
            *transform = light_transform(moon_dir);
            let up = state.moon_altitude_deg > MOON_OFF_DEG;
            light.illuminance = if up { moon_lux * cloud.direct } else { 0.0 };
            set_shadows(
                &mut light,
                up && state.altitude_deg < MOON_SHADOWS_BELOW_SUN_DEG,
            );
        }
    }
}

/// Write the shadow flag only when it changes, so the light is not re-flagged as modified.
fn set_shadows(light: &mut Mut<DirectionalLight>, on: bool) {
    if light.shadow_maps_enabled != on {
        light.shadow_maps_enabled = on;
    }
}

/// Real-time constant of the exposure adaptation, s.
const EXPOSURE_TAU_S: f32 = 1.5;
/// Illuminance floor so a moonless night still has a finite exposure, lx.
const ILLUMINANCE_FLOOR_LUX: f32 = 0.0005;

/// Clear-sky horizontal illuminance of the sun and of the moon, direct plus sky, lx: each body's
/// illuminance outside the atmosphere times the USNO total-illuminance factor at its refracted
/// altitude (which carries twilight).
fn clear_sky_illuminance(state: &SunState, moon_lux: f32) -> (f32, f32) {
    let factor = |true_altitude: f32| {
        let h = f64::from(true_altitude);
        let apparent = h + astro::refraction_deg(h, REFRACTION_PRESSURE_MBAR, REFRACTION_TEMP_C);
        astro::horizontal_illuminance_factor(apparent) as f32
    };
    (
        lux::RAW_SUNLIGHT * factor(state.altitude_deg),
        moon_lux * factor(state.moon_altitude_deg),
    )
}

/// Horizontal illuminance from the sun and the moon, direct plus sky, lx: the clear-sky
/// illuminance of [`clear_sky_illuminance`] times the cloud's total ([`CloudLight::total`]: the
/// beam and its diffuse remainder together), plus a floor.
pub(crate) fn scene_illuminance(state: &SunState, moon_lux: f32, cloud: CloudLight) -> f32 {
    let (sun, moon) = clear_sky_illuminance(state, moon_lux);
    (sun + moon) * cloud.total() + ILLUMINANCE_FLOOR_LUX
}

/// Colour of the sun's light toward the end of the day: white above `DUSK_ONSET_DEG`, tinted
/// toward sunset orange as the sun drops, fully at the horizon, fading out by the end of civil
/// twilight (−6°) so moonlit air is not orange.
pub(crate) fn dusk_tint(sun_altitude_deg: f32) -> Vec3 {
    let dusk = if sun_altitude_deg >= 0.0 {
        (1.0 - sun_altitude_deg / DUSK_ONSET_DEG).clamp(0.0, 1.0)
    } else {
        (1.0 + sun_altitude_deg / CIVIL_TWILIGHT_DEG).clamp(0.0, 1.0)
    };
    Vec3::ONE.lerp(DUSK_TINT, dusk)
}

/// Tint the light takes as the sun nears the horizon.
pub(crate) const DUSK_TINT: Vec3 = Vec3::new(1.0, 0.55, 0.3);
/// Sun altitude below which the dusk tint starts, degrees.
const DUSK_ONSET_DEG: f32 = 10.0;
/// Sun depression at which the dusk tint has faded out, degrees (end of civil twilight).
const CIVIL_TWILIGHT_DEG: f32 = 6.0;

/// The cloud-scattered share of the light as sky radiance: the colour (luminance 1) and the
/// brightness in cd/m² of a uniform sky whose horizontal illuminance is the sun's and the
/// moon's `diffuse` share of their clear-sky illuminance (`E = π · L`). Only a body whose
/// `DirectionalLight` is on contributes, so a night with no moon stays dark whatever the cloud.
/// The colour is the light of the two bodies, the sun's reddening toward the horizon
/// ([`dusk_tint`]) and the moon's tint, weighted by their illuminance.
pub(crate) fn cloud_ambient(state: &SunState, moon_lux: f32, cloud: CloudLight) -> (Vec3, f32) {
    let (sun, moon) = clear_sky_illuminance(state, moon_lux);
    let sun = if state.altitude_deg > SUN_OFF_DEG {
        sun * cloud.diffuse
    } else {
        0.0
    };
    let moon = if state.moon_altitude_deg > MOON_OFF_DEG {
        moon * cloud.diffuse
    } else {
        0.0
    };
    let lux = sun + moon;
    if lux <= 0.0 {
        return (Vec3::ONE, 0.0);
    }
    let mixed = (dusk_tint(state.altitude_deg) * sun + MOON_TINT * moon) / lux;
    let luminance = mixed.dot(LUMINANCE_WEIGHTS);
    (mixed / luminance, lux / std::f32::consts::PI)
}

/// Rec. 709 luminance weights of linear RGB.
const LUMINANCE_WEIGHTS: Vec3 = Vec3::new(0.2126, 0.7152, 0.0722);

/// What Bevy's `ambient_light()` makes of a uniform radiance `L`: a diffuse surface shows
/// `albedo · L · s` with `s = F_ab.x` of `F_AB(1.0, NdotV)` (`EnvBRDFApprox(diffuse_color, ..)`,
/// `bevy_pbr-0.20.0/src/render/pbr_ambient.wesl:18`, `pbr_lighting.wesl:527-544`). At perceptual
/// roughness 1 the polynomial is `r = (0, 0.015, 0.468, −0.018)`, `a004 = 0.015` for every
/// `NdotV`, `s = 0.468 − 1.04 · 0.015 = 0.452`, not the Lambertian 1. The ambient light is
/// scaled up by `1/s` so a horizontal surface receives the illuminance `π L` that
/// [`cloud_ambient`] computed.
const BEVY_AMBIENT_DIFFUSE_SCALE: f32 = 0.452;

/// Give every [`ElementsCamera`] the cloud-scattered light as its `AmbientLight`: Bevy's
/// per-camera ambient is a uniform sky radiance in cd/m² that casts no shadow, which is what
/// the diffuse remainder under cloud is.
fn update_ambient(
    weather: Res<Weather>,
    state: Res<SunState>,
    frame: Res<CelestialFrame>,
    shadows: Option<Res<CloudShadows>>,
    mut cameras: Query<&mut AmbientLight, With<ElementsCamera>>,
) {
    let cloud = cloud_light(weather.cloud_cover, shadows.as_deref());
    let (colour, radiance) = cloud_ambient(&state, frame.moon_lux, cloud);
    let brightness = radiance / BEVY_AMBIENT_DIFFUSE_SCALE;
    let colour = Color::linear_rgb(colour.x, colour.y, colour.z);
    for mut ambient in &mut cameras {
        if ambient.brightness != brightness || ambient.color != colour {
            ambient.brightness = brightness;
            ambient.color = colour;
        }
    }
}

/// Peak luminance of the display the exposure targets, cd/m² (an SDR monitor).
const DISPLAY_MAX_NITS: f32 = 100.0;
/// Reflectance of the surface the eye is assumed to adapt to (mid grey).
const ADAPTING_REFLECTANCE: f32 = 0.18;

/// Base-10 log of the just-noticeable luminance difference at adaptation luminance `la`, cd/m²
/// (Larson, Rushmeier & Piatko 1997, Table 1: rods and cones, after Ferwerda et al. 1996).
pub(crate) fn log10_threshold(la: f32) -> f32 {
    let x = la.max(1e-12).log10();
    if x < -3.94 {
        -2.86
    } else if x < -1.44 {
        (0.405 * x + 1.6).powf(2.18) - 2.86
    } else if x < -0.0184 {
        x - 0.395
    } else if x < 1.9 {
        (0.249 * x + 0.65).powf(2.7) - 0.72
    } else {
        x - 1.255
    }
}

/// Exposure for a scene lit by `lux` of horizontal illuminance.
///
/// Ward's contrast-based scale factor (Graphics Gems IV, 1994) with the threshold function of
/// Larson et al. 1997: world luminance is scaled by the ratio of the display's and the world's
/// just-noticeable differences, `ΔL(L_d,max/2) / ΔL(0.18·E/π)`, so a contrast that is just
/// visible in the world is just visible on screen. A moonlit scene therefore stays dark instead
/// of being exposed up to daylight. Bevy multiplies scene luminance by `2^−EV₁₀₀ / 1.2` and
/// treats 1 as display peak, so `EV₁₀₀ = −log₂(1.2 · m / L_d,max)`.
pub(crate) fn ev100_for(lux: f32) -> f32 {
    -(1.2 * exposure_for_adaptation(ADAPTING_REFLECTANCE * lux / std::f32::consts::PI)).log2()
}

/// The factor scene luminance is multiplied by for an eye adapted to luminance `la`, cd/m²
/// (`m / L_d,max` of [`ev100_for`]; Bevy's `Exposure::exposure()` for the matching EV).
pub(crate) fn exposure_for_adaptation(la: f32) -> f32 {
    let display = DISPLAY_MAX_NITS * 0.5;
    10f32.powf(log10_threshold(display) - log10_threshold(la)) / DISPLAY_MAX_NITS
}

fn auto_exposure(
    time: Res<Time<Real>>,
    state: Res<SunState>,
    frame: Res<CelestialFrame>,
    weather: Res<Weather>,
    shadows: Option<Res<CloudShadows>>,
    mut cameras: Query<(Entity, &ElementsCamera, &mut Exposure)>,
    mut adapted: Local<Vec<Entity>>,
) {
    let target = ev100_for(scene_illuminance(
        &state,
        frame.moon_lux,
        cloud_light(weather.cloud_cover, shadows.as_deref()),
    ));
    let k = 1.0 - (-time.delta_secs() / EXPOSURE_TAU_S).exp();
    for (entity, camera, mut exposure) in &mut cameras {
        if !camera.auto_exposure {
            continue;
        }
        if adapted.contains(&entity) {
            exposure.ev100 += (target - exposure.ev100) * k;
        } else {
            exposure.ev100 = target;
            adapted.push(entity);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::ecs::system::RunSystemOnce;

    #[test]
    fn exposure_targets_match_the_research_note() {
        let zenith_sun = SunState {
            altitude_deg: 90.0,
            moon_altitude_deg: -10.0,
            ..default()
        };
        assert!(
            (ev100_for(scene_illuminance(&zenith_sun, 0.0, CloudLight::CLEAR)) - 13.3360).abs()
                < 1e-3
        );
        let full_moon = SunState {
            altitude_deg: -30.0,
            moon_altitude_deg: 90.0,
            ..default()
        };
        let ev = ev100_for(scene_illuminance(
            &full_moon,
            astro::FULL_MOON_LUX as f32,
            CloudLight::CLEAR,
        ));
        assert!((ev - -2.2353).abs() < 1e-3, "{ev}");
        let dark = SunState {
            altitude_deg: -30.0,
            moon_altitude_deg: -30.0,
            ..default()
        };
        assert!(
            (ev100_for(scene_illuminance(&dark, 0.0, CloudLight::CLEAR)) - -4.7465).abs() < 1e-3
        );
        // Sunrise and civil twilight keep the sky's light (USNO), so dawn is not metered as night.
        let sunrise = SunState {
            altitude_deg: 0.0,
            moon_altitude_deg: -30.0,
            ..default()
        };
        assert!(
            (ev100_for(scene_illuminance(&sunrise, 0.0, CloudLight::CLEAR)) - 6.4780).abs() < 1e-3
        );
        let civil = SunState {
            altitude_deg: -6.0,
            moon_altitude_deg: -30.0,
            ..default()
        };
        assert!(
            (ev100_for(scene_illuminance(&civil, 0.0, CloudLight::CLEAR)) - 0.8567).abs() < 2e-3
        );
    }

    #[test]
    fn overcast_passes_a_quarter_of_the_sun_all_of_it_diffuse() {
        // Kasten & Czeplak: 1 − 0.75 N^3.4 is 1 for a clear sky and 0.25 under complete overcast.
        let clear = cloud_light(0.0, None);
        assert_eq!(clear, CloudLight::CLEAR);
        assert_eq!(clear.total(), 1.0);
        let overcast = cloud_light(1.0, None);
        assert_eq!(overcast.direct, 0.0);
        assert!((overcast.diffuse - 0.25).abs() < 1e-6);
        assert!((overcast.total() - 0.25).abs() < 1e-6);
        for c in [0.05f32, 0.3, 0.5, 0.8, 0.95] {
            let kc = 1.0 - 0.75 * c.powf(3.4);
            assert!(
                (cloud_light(c, None).total() - kc).abs() < 1e-6,
                "cover {c}"
            );
        }
    }

    #[test]
    fn the_total_follows_the_beam_linearly_and_the_diffuse_share_is_what_the_beam_lost() {
        for t in [0.0f32, 0.1, 0.5, 0.9, 1.0] {
            let light = CloudLight::from_beam(t);
            assert_eq!(light.direct, t);
            assert!((light.total() - (0.25 + 0.75 * t)).abs() < 1e-6);
            assert!((light.diffuse - 0.25 * (1.0 - t)).abs() < 1e-6);
        }
        // Out-of-range transmittances are clamped.
        assert_eq!(CloudLight::from_beam(-1.0), CloudLight::from_beam(0.0));
        assert_eq!(CloudLight::from_beam(2.0), CloudLight::CLEAR);
    }

    fn shadows(camera_transmittance: f32) -> CloudShadows {
        CloudShadows {
            map: Handle::default(),
            origin: Vec2::ZERO,
            extent: 8000.0,
            light_direction: Vec3::Y,
            camera_transmittance,
        }
    }

    /// Run `update_sky` once at `hour` on 2026-06-21 and return the state, the frame, the sun's
    /// and the moon's `DirectionalLight::illuminance`.
    fn run_sky(
        hour: f64,
        cover: f32,
        shadows: Option<CloudShadows>,
    ) -> (SunState, CelestialFrame, f32, f32) {
        let mut world = World::new();
        world.insert_resource(WorldClock {
            date: CivilDate::new(2026, 6, 21),
            hour,
            ..default()
        });
        world.insert_resource(GeoLocation::default());
        world.insert_resource(Weather {
            cloud_cover: cover,
            ..default()
        });
        world.insert_resource(SunState::default());
        world.insert_resource(CelestialFrame::default());
        if let Some(shadows) = shadows {
            world.insert_resource(shadows);
        }
        let sun = world
            .spawn((Sun, DirectionalLight::default(), Transform::default()))
            .id();
        let moon = world
            .spawn((Moon, DirectionalLight::default(), Transform::default()))
            .id();
        assert!(world.run_system_once(update_sky).is_ok());
        let state = *world.resource::<SunState>();
        let frame = *world.resource::<CelestialFrame>();
        let lux_of = |entity| {
            world
                .get::<DirectionalLight>(entity)
                .map_or(f32::NAN, |light| light.illuminance)
        };
        let (sun_lux, moon_lux) = (lux_of(sun), lux_of(moon));
        (state, frame, sun_lux, moon_lux)
    }

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() <= 1e-3 * b.abs().max(1e-9)
    }

    #[test]
    fn cloud_shadow_transmittance_dims_the_sun_and_moon() {
        let cover = 0.5;
        // A moon-up hour, found with the same ephemeris the system uses.
        let moon_hour = (0..24)
            .map(f64::from)
            .find(|&h| run_sky(h, cover, None).0.moon_altitude_deg > MOON_OFF_DEG);
        assert!(
            moon_hour.is_some(),
            "the moon is up at some hour of the day"
        );
        let moon_hour = moon_hour.unwrap_or(0.0);

        for (t, direct, total) in [(1.0, 1.0, 1.0), (0.5, 0.5, 0.625), (0.0, 0.0, 0.25)] {
            let (state, _, sun_lux, _) = run_sky(12.0, cover, Some(shadows(t)));
            assert!(state.altitude_deg > SUN_OFF_DEG);
            assert!(
                close(sun_lux, lux::RAW_SUNLIGHT * direct),
                "sun T={t}: {sun_lux}"
            );

            let (state, frame, _, moon_lux) = run_sky(moon_hour, cover, Some(shadows(t)));
            assert!(state.moon_altitude_deg > MOON_OFF_DEG);
            assert!(frame.moon_lux > 0.0);
            assert!(
                close(moon_lux, frame.moon_lux * direct),
                "moon T={t}: {moon_lux}"
            );

            // The metered total is what it was before the split: clear × (0.25 + 0.75 T̄).
            let dimmed = scene_illuminance(
                &state,
                frame.moon_lux,
                cloud_light(cover, Some(&shadows(t))),
            );
            let open = scene_illuminance(&state, frame.moon_lux, CloudLight::CLEAR)
                - ILLUMINANCE_FLOOR_LUX;
            assert!(close(dimmed, open * total + ILLUMINANCE_FLOOR_LUX));
        }

        // Without the cloud shadow map the cover-only beam stays.
        let (_, _, sun_lux, _) = run_sky(12.0, cover, None);
        assert!(close(
            sun_lux,
            lux::RAW_SUNLIGHT * beam_transmittance(cover)
        ));
        let (_, frame, _, moon_lux) = run_sky(moon_hour, cover, None);
        assert!(close(moon_lux, frame.moon_lux * beam_transmittance(cover)));

        // Before the first readback the camera transmittance is `1 − cover^3.4`: same split.
        for c in [0.0f32, 0.3, 0.7, 1.0] {
            let s = shadows(beam_transmittance(c));
            assert_eq!(cloud_light(c, Some(&s)), cloud_light(c, None));
        }
    }

    #[test]
    fn the_ambient_carries_the_diffuse_share_and_nothing_at_night() {
        let noon = SunState {
            altitude_deg: 90.0,
            moon_altitude_deg: -30.0,
            ..default()
        };
        let clear_noon = scene_illuminance(&noon, 0.0, CloudLight::CLEAR) - ILLUMINANCE_FLOOR_LUX;
        // Cover 0: all direct, no ambient.
        let (_, brightness) = cloud_ambient(&noon, 0.0, CloudLight::CLEAR);
        assert_eq!(brightness, 0.0);
        // Beam gone: no direct light, a quarter of the clear illuminance as uniform sky radiance
        // (E = π L), and the total the exposure meters is what it was.
        let storm = CloudLight::from_beam(0.0);
        let (colour, brightness) = cloud_ambient(&noon, 0.0, storm);
        assert!(close(brightness * std::f32::consts::PI, 0.25 * clear_noon));
        assert!((colour.dot(LUMINANCE_WEIGHTS) - 1.0).abs() < 1e-5);
        let direct = lux::RAW_SUNLIGHT * storm.direct;
        assert_eq!(direct, 0.0);
        let total = scene_illuminance(&noon, 0.0, storm);
        assert!(close(total, 0.25 * clear_noon + ILLUMINANCE_FLOOR_LUX));
        // Halfway: the ambient is a quarter of the lost beam.
        let half = CloudLight::from_beam(0.5);
        let (_, brightness) = cloud_ambient(&noon, 0.0, half);
        assert!(close(brightness * std::f32::consts::PI, 0.125 * clear_noon));

        // A night with no moon stays dark under any cloud; the moon's diffuse share is lit.
        let dark = SunState {
            altitude_deg: -30.0,
            moon_altitude_deg: -30.0,
            ..default()
        };
        for t in [0.0, 0.5, 1.0] {
            let (_, brightness) = cloud_ambient(&dark, 0.0, CloudLight::from_beam(t));
            assert_eq!(brightness, 0.0, "T = {t}");
            let (_, brightness) = cloud_ambient(&dark, 0.3, CloudLight::from_beam(t));
            assert_eq!(brightness, 0.0, "a moon below the horizon, T = {t}");
        }
        let moonlit = SunState {
            altitude_deg: -30.0,
            moon_altitude_deg: 60.0,
            ..default()
        };
        let (colour, brightness) = cloud_ambient(&moonlit, 0.3, storm);
        assert!(brightness > 0.0);
        assert!(colour.z > colour.x, "moonlight is blue: {colour}");
        // The sun on the horizon reddens the ambient.
        let sunrise = SunState {
            altitude_deg: 0.0,
            moon_altitude_deg: -30.0,
            ..default()
        };
        let (colour, _) = cloud_ambient(&sunrise, 0.0, storm);
        assert!(colour.x > colour.y && colour.y > colour.z, "{colour}");
    }

    #[test]
    fn the_ambient_system_drives_every_elements_camera() {
        let mut world = World::new();
        world.insert_resource(Weather {
            cloud_cover: 1.0,
            ..default()
        });
        world.insert_resource(SunState {
            altitude_deg: 90.0,
            moon_altitude_deg: -30.0,
            ..default()
        });
        world.insert_resource(CelestialFrame::default());
        world.insert_resource(shadows(0.0));
        let camera = world
            .spawn((
                ElementsCamera::default(),
                AmbientLight {
                    brightness: 0.0,
                    ..default()
                },
            ))
            .id();
        let other = world
            .spawn(AmbientLight {
                brightness: 0.0,
                ..default()
            })
            .id();
        assert!(world.run_system_once(update_ambient).is_ok());
        let brightness_of = |entity| {
            world
                .get::<AmbientLight>(entity)
                .map_or(f32::NAN, |ambient| ambient.brightness)
        };
        assert!(brightness_of(camera) > 1000.0, "{}", brightness_of(camera));
        assert_eq!(brightness_of(other), 0.0);
    }

    #[test]
    fn the_sky_gain_restores_the_dome_to_the_total_factor_by_day_only() {
        let noon = SunState {
            altitude_deg: 60.0,
            moon_altitude_deg: -30.0,
            ..default()
        };
        assert_eq!(sky_gain(&noon, CloudLight::CLEAR), 1.0);
        // The dome is lit by the beam `T̄`; times the gain it is the total `0.25 + 0.75 T̄`.
        for t in [0.01f32, 0.3, 0.5, 0.9] {
            let light = CloudLight::from_beam(t);
            assert!(
                (t * sky_gain(&noon, light) - light.total()).abs() < 1e-6,
                "{t}"
            );
        }
        // Bounded where the beam is gone.
        assert_eq!(
            sky_gain(&noon, CloudLight::from_beam(0.0)),
            0.25 / MIN_SKY_BEAM
        );
        // With the sun's light off the moon's sky and the stars are not rescaled.
        let night = SunState {
            altitude_deg: -30.0,
            moon_altitude_deg: 60.0,
            ..default()
        };
        assert_eq!(sky_gain(&night, CloudLight::from_beam(0.2)), 1.0);
    }
}
