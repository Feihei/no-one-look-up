//! Sun, moon and stars from the real sky: a world clock, Meeus-accuracy ephemerides, the sun and
//! moon as directional lights for Bevy's atmosphere, and exposure that follows them.

pub mod astro;
mod clock;
pub mod light;
mod moon;
mod stars;

use bevy::camera::Exposure;
use bevy::ecs::system::SystemParam;
use bevy::light::light_consts::lux;
use bevy::light::{CascadeShadowConfig, CascadeShadowConfigBuilder, SunDisk, VolumetricLight};
use bevy::prelude::*;

pub(crate) use clock::plugin as clock_plugin;
pub use clock::{CivilDate, GeoLocation, WorldClock};

use crate::climate::Weather;
use crate::clouds::{CloudShadows, SkyProbe};
use crate::plugin::{
    ElementsCamera, ElementsSystems, ElementsUnits, ElementsView, SkyBackground, ensure_gpu_core,
};

/// The camera's exposure, as the crate meters it, published every frame.
///
/// The crate's exposure is **illuminance-based**: it is computed from the scene's illuminance
/// (the sun, the moon and the cloud's diffuse share) and not from the frame's content, so a
/// game's own light does not move it (Patry 2021: an "illuminance-based exposure system… takes
/// albedo out of the equation altogether", `docs/research/game-lights.md`). A game places a
/// hand-held light, a street lamp or a window and the camera's exposure stays where the sky
/// put it.
///
/// Read it through [`Environment::exposure`](crate::Environment::exposure) and
/// [`Environment::display_value`](crate::Environment::display_value), or from this resource
/// directly.
#[derive(Resource, Clone, Copy, Debug, Reflect)]
pub struct ElementsExposure {
    /// The camera's EV100, the exposure value the crate meters: `log₂` of the scene's
    /// luminance over the reference white point's. The same number Bevy's `Exposure::ev100`
    /// carries.
    pub ev100: f32,
    /// The exposure factor the scene's luminance is multiplied by for the display:
    /// `2^(−ev100) / 1.2`, the same number `bevy::camera::Exposure::exposure()` returns.
    pub exposure: f32,
}

impl Default for ElementsExposure {
    fn default() -> Self {
        // Bevy's own default (the "Blender" calibration, EV100 9.7), so a world without the
        // celestial plugin still has a sane exposure.
        let ev100 = bevy::camera::Exposure::EV100_BLENDER;
        Self {
            ev100,
            exposure: bevy::camera::Exposure { ev100 }.exposure(),
        }
    }
}

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
    /// Illuminance the sun's light carries (the clear light through twilight and the cloud's
    /// mean beam), lx: what the crate's `Sun` light is set to, computed with or without it.
    pub sun_light_lux: f32,
    /// Illuminance the moon's light carries, lx: what the crate's `Moon` light is set to.
    pub moon_light_lux: f32,
}

impl Default for CelestialFrame {
    fn default() -> Self {
        Self {
            j2000_to_world: Mat3::IDENTITY,
            ecliptic_north: Vec3::Y,
            moon_angular_diameter_deg: 0.5,
            moon_lux: 0.0,
            sun_light_lux: 0.0,
            moon_light_lux: 0.0,
        }
    }
}

/// What [`CelestialPlugin`] puts in the world. Insert it first to override; both default on.
///
/// [`SunState`] (the sun and moon, their altitudes, the moon's phase) is kept current either way,
/// so a game that lights its scene itself still has the sun for `SurfaceWeathering`'s drying and
/// `Environment::is_daylight` (`docs/research/integration.md`).
#[derive(Resource, Clone, Copy, Debug, Reflect)]
pub struct CelestialSettings {
    /// Spawn the [`Sun`] and [`Moon`] `DirectionalLight`s and drive every light carrying those
    /// markers (direction, illuminance, shadows, cascades). Off: no light is spawned and none is
    /// touched, a game's own `Sun`-marked light included.
    pub lights: bool,
    /// Spawn the moon disc and the star field.
    pub sky: bool,
}

impl Default for CelestialSettings {
    fn default() -> Self {
        Self {
            lights: true,
            sky: true,
        }
    }
}

/// Sun, moon and exposure.
///
/// Spawns one [`Sun`] and one [`Moon`] `DirectionalLight` at startup and moves them every frame
/// in [`ElementsSystems::Celestial`]; keeps [`SunState`] current; drives `Exposure` on every
/// [`ElementsCamera`] with `auto_exposure` set. The sun keeps lighting the atmosphere below the
/// horizon through twilight, to −18°; a camera that kept Bevy's default clear colour is cleared
/// to the night sky's faint background instead. [`CelestialSettings`] turns the lights and the
/// moon disc and stars off.
pub struct CelestialPlugin;

impl Plugin for CelestialPlugin {
    fn build(&self, app: &mut App) {
        ensure_gpu_core(app);
        app.init_resource::<CelestialSettings>();
        stars::plugin(app);
        moon::plugin(app);
        app.init_resource::<CelestialFrame>()
            .init_resource::<ElementsExposure>()
            .add_systems(Startup, spawn_lights)
            .add_systems(
                Update,
                (
                    update_sky,
                    drive_lights,
                    update_ambient,
                    auto_exposure,
                    publish_exposure,
                    sky_background,
                    fit_shadow_cascades,
                )
                    .chain()
                    .in_set(ElementsSystems::Celestial),
            );
    }
}

/// Atmospheric pressure and temperature the refraction correction assumes, mbar and °C
/// (Saemundsson's reference state).
const REFRACTION_PRESSURE_MBAR: f64 = 1010.0;
const REFRACTION_TEMP_C: f64 = 10.0;
/// Below this true altitude the sun no longer lights the atmosphere, degrees: astronomical
/// twilight ends (Haber et al. 2005, `docs/research/twilight.md`).
const SUN_OFF_DEG: f32 = -18.0;
/// Above this true altitude the sun's disc is up and its shadow maps are drawn, degrees; below
/// it Bevy's visible-disc factor already keeps the sun off every surface.
const SUN_SHADOWS_ABOVE_DEG: f32 = -0.5;
/// Below the horizon the sun's light falls as `exp(k · h)` (h in degrees) down to the knee, so
/// Bevy's zenith twilight sky follows the brightness Hulburt (1953) measured
/// (`docs/research/twilight.md`).
const TWILIGHT_FALL_PER_DEG: f32 = 0.44;
/// Below this true altitude Bevy's zenith is its multiple-scattering floor, degrees.
const TWILIGHT_KNEE_DEG: f32 = -6.0;
/// The fall below the knee, per degree: Hulburt's 0.427 decades per degree (`ln 10 · 0.427`),
/// which a sky held up by a constant floor only follows if the light itself falls that fast.
const TWILIGHT_TAIL_PER_DEG: f32 = 0.983;
/// Below this true altitude the moon's light is switched off, degrees.
const MOON_OFF_DEG: f32 = -0.5;
/// Moon shadows are drawn only when the sun is lower than this, degrees (civil twilight).
const MOON_SHADOWS_BELOW_SUN_DEG: f32 = -6.0;

/// The factor on the sun's light at true altitude `altitude_deg`: 1 with the sun up, the
/// twilight fall below the horizon (steeper past the knee), 0 once astronomical twilight has
/// ended, by when it is under 10⁻⁶.
pub(crate) fn twilight_factor(altitude_deg: f32) -> f32 {
    if altitude_deg >= 0.0 {
        1.0
    } else if altitude_deg >= TWILIGHT_KNEE_DEG {
        (TWILIGHT_FALL_PER_DEG * altitude_deg).exp()
    } else if altitude_deg > SUN_OFF_DEG {
        (TWILIGHT_FALL_PER_DEG * TWILIGHT_KNEE_DEG
            + TWILIGHT_TAIL_PER_DEG * (altitude_deg - TWILIGHT_KNEE_DEG))
            .exp()
    } else {
        0.0
    }
}

/// Fraction of the sun's disc above the horizon at true altitude `altitude_deg`: the share of
/// its light that reaches the ground directly. Bevy applies the same factor to every surface
/// (`calculate_visible_sun_ratio`); what lights without Bevy (the precipitation draw) takes it
/// from here.
pub(crate) fn sun_disc_visible(altitude_deg: f32) -> f32 {
    let radius = SunDisk::EARTH.angular_size.to_degrees() * 0.5;
    ((altitude_deg + radius) / (2.0 * radius)).clamp(0.0, 1.0)
}

/// The moon light's colour, the tint of the moon's share of the sky's diffuse light and of the
/// moonlit cloud.
pub(crate) const MOON_TINT: Vec3 = Vec3::new(0.88, 0.92, 1.0);

/// How far the sun's shadow cascades reach, m. Volumetric fog takes its sunlight through the
/// shadow map, so beyond the last cascade fog gets no light at all and turns into a dark band
/// toward the horizon; the cascades cover the ground-fog volume (±400 m) with margin.
const SUN_SHADOW_DISTANCE_M: f32 = 600.0;

/// The sun's shadow cascades for a perspective view: four, out to [`SUN_SHADOW_DISTANCE_M`],
/// as world distances (`u` world units to the metre).
fn sun_cascades(u: f32) -> CascadeShadowConfig {
    CascadeShadowConfigBuilder {
        maximum_distance: SUN_SHADOW_DISTANCE_M * u,
        first_cascade_far_bound: 15.0 * u,
        ..default()
    }
    .build()
}

/// Shadow cascades for an orthographic view whose focus lies `focus_depth_m` in front of the
/// camera. Bevy splits cascades by depth from the camera, which an isometric camera puts far
/// behind what it shows, so the default cascades end before the scene begins. An orthographic
/// pixel is the same size at every depth, so one cascade, [`SUN_SHADOW_DISTANCE_M`] deep and
/// centred on the focus, serves the whole view evenly (`docs/research/view-anchor.md`
/// Deviation 2).
fn orthographic_cascades(focus_depth_m: f32, u: f32) -> CascadeShadowConfig {
    let near = (focus_depth_m - 0.5 * SUN_SHADOW_DISTANCE_M).max(0.0);
    CascadeShadowConfigBuilder {
        num_cascades: 1,
        minimum_distance: near * u,
        first_cascade_far_bound: (near + SUN_SHADOW_DISTANCE_M) * u,
        maximum_distance: (near + SUN_SHADOW_DISTANCE_M) * u,
        ..default()
    }
    .build()
}

fn spawn_lights(
    mut commands: Commands,
    settings: Res<CelestialSettings>,
    units: Res<ElementsUnits>,
) {
    if !settings.lights {
        return;
    }
    commands.spawn((
        Name::new("Sun"),
        Sun,
        DirectionalLight {
            illuminance: lux::RAW_SUNLIGHT,
            shadow_maps_enabled: true,
            ..default()
        },
        sun_cascades(units.per_metre()),
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

/// The sun's and the moon's shadow cascades, and whether each is the sun's.
type CelestialCascades<'w, 's> =
    Query<'w, 's, (&'static mut CascadeShadowConfig, Has<Sun>), Or<(With<Sun>, With<Moon>)>>;

/// Give the sun and the moon cascades that reach the scene: the perspective ones, or under an
/// orthographic followed camera one cascade around its focus.
fn fit_shadow_cascades(
    settings: Res<CelestialSettings>,
    view: Res<ElementsView>,
    units: Res<ElementsUnits>,
    cameras: Query<(&Projection, &GlobalTransform)>,
    mut lights: CelestialCascades,
) {
    if !settings.lights {
        return;
    }
    let orthographic =
        view.entity
            .and_then(|e| cameras.get(e).ok())
            .and_then(|(projection, transform)| match projection {
                Projection::Orthographic(_) => Some(orthographic_cascades(
                    (view.position - units.point_to_metres(transform.translation()))
                        .dot(*transform.forward()),
                    units.per_metre(),
                )),
                _ => None,
            });
    for (mut config, is_sun) in &mut lights {
        let want = orthographic.clone().unwrap_or_else(|| {
            if is_sun {
                sun_cascades(units.per_metre())
            } else {
                CascadeShadowConfig::default()
            }
        });
        if config.bounds != want.bounds || config.minimum_distance != want.minimum_distance {
            *config = want;
        }
    }
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

/// The light split under the current cloud: with the cloud shadow map, its mean transmittance
/// over the map (measured); without [`CloudShadows`], the cover-only [`beam_transmittance`].
pub(crate) fn cloud_light(cover: f32, shadows: Option<&CloudShadows>) -> CloudLight {
    CloudLight::from_beam(shadows.map_or_else(
        || beam_transmittance(cover),
        |shadows| shadows.mean_transmittance,
    ))
}

/// Smallest beam fraction [`sky_gain`] divides by.
const MIN_SKY_BEAM: f32 = 0.001;

/// The cloud factor on the sun's `DirectionalLight`: the beam `T̄` while its disc is up (the
/// light then lights the ground, and its shadows fade with the beam), blending over the disc to
/// the dome's total factor `0.25 + 0.75 T̄` once it has set, when the light lights only the sky
/// (`docs/research/twilight.md`, Model 3). A sun below the horizon has no beam to the camera,
/// and the cloud shadow over the camera says nothing about the sky it lights.
pub(crate) fn sun_cloud_factor(state: &SunState, cloud: CloudLight) -> f32 {
    let up = sun_disc_visible(state.altitude_deg);
    cloud.total() + (cloud.direct - cloud.total()) * up
}

/// The factor that keeps the sky's brightness where it was. Bevy's atmosphere is lit by the
/// sun's `DirectionalLight`, which under cloud carries only the beam `T̄` while the sun is up
/// ([`sun_cloud_factor`]), so the sky dome (and the sun disc in a gap) would dim to `T̄` of its
/// clear value with the shadow the camera stands in, and go black under full cloud. The clouds
/// compose the sky pixels with `total / factor`, which restores the dome to the total factor
/// `0.25 + 0.75 T̄` it has always had (a limit of the change, not physics: the sky above a cloud
/// layer is not dimmed by it at all [INFERENCE]). Exactly 1 once the sun has set (its light then
/// carries the total already) and when its light is off, so the night background, the moon's
/// sky and the stars are never rescaled.
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
        let gain = cloud.total() * sky_clear_gain(state, cloud);
        // [PATCH heads-up] blend the gain away toward the horizon.
        1.0 + (gain - 1.0) * dusk_gain_fade(state.altitude_deg)
    } else {
        1.0
    }
}

/// The factor that undoes the cloud dimming of Bevy's sky: the sun's `DirectionalLight` carries
/// [`sun_cloud_factor`] of the clear sun, so the atmosphere (and its environment map) shows that
/// fraction of the clear sky; `1 / factor` gives the clear sky, the sky above a cloud layer,
/// which the layer does not dim. 1 when the sun's light is off.
pub(crate) fn sky_clear_gain(state: &SunState, cloud: CloudLight) -> f32 {
    if state.altitude_deg > SUN_OFF_DEG {
        1.0 / sun_cloud_factor(state, cloud).max(MIN_SKY_BEAM)
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
    let cloud = cloud_light(weather.cloud_cover, shadows.as_deref());
    let altitude = sun_alt as f32;
    let sun_light_lux =
        lux::RAW_SUNLIGHT * twilight_factor(altitude) * sun_cloud_factor(&state, cloud);
    let moon_light_lux = if state.moon_altitude_deg > MOON_OFF_DEG {
        moon_lux * cloud.direct
    } else {
        0.0
    };
    *frame = CelestialFrame {
        j2000_to_world: Mat3::from_cols(col(0), col(1), col(2)),
        ecliptic_north: pole_world.normalize_or(Vec3::Y),
        moon_angular_diameter_deg: moon_angular_diameter(observer_distance) as f32,
        moon_lux,
        sun_light_lux,
        moon_light_lux,
    };
}

/// Point the [`Sun`] and [`Moon`] lights at their bodies and give them [`update_sky`]'s
/// illuminance and shadows; nothing with [`CelestialSettings::lights`] off.
fn drive_lights(
    settings: Res<CelestialSettings>,
    state: Res<SunState>,
    frame: Res<CelestialFrame>,
    mut lights: Query<(&mut Transform, &mut DirectionalLight, Has<Sun>, Has<Moon>)>,
) {
    if !settings.lights {
        return;
    }
    for (mut transform, mut light, is_sun, is_moon) in &mut lights {
        if is_sun {
            *transform = light_transform(state.direction);
            light.illuminance = frame.sun_light_lux;
            set_shadows(&mut light, state.altitude_deg > SUN_SHADOWS_ABOVE_DEG);
        } else if is_moon {
            *transform = light_transform(state.moon_direction);
            light.illuminance = frame.moon_light_lux;
            set_shadows(
                &mut light,
                state.moon_altitude_deg > MOON_OFF_DEG
                    && state.altitude_deg < MOON_SHADOWS_BELOW_SUN_DEG,
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

/// Display value (1 = display peak, before tonemapping) the sun-side horizon glow may reach: the
/// glow at a sun of 27° under the illuminance metering, so the afternoon is metered as before and
/// only the low sun is capped.
const GLOW_DISPLAY_MAX: f32 = 3.2;

/// Cloud-free luminance of the horizon sky toward the sun, 5° above the horizon, cd/m², against
/// the sun's true altitude: Bevy's atmosphere lit by the twilight-scaled sun, measured on a
/// linear target (`docs/research/twilight.md`, Parameters). Below the first row the glow is
/// far under any exposure the floor allows.
// −6.28° is a measured sun altitude, not τ.
#[allow(clippy::approx_constant)]
const GLOW_TABLE: [(f32, f32); 17] = [
    (-13.91, 4.648e-5),
    (-12.04, 1.079e-3),
    (-10.70, 1.057e-2),
    (-9.29, 1.054e-1),
    (-8.19, 5.721e-1),
    (-7.05, 3.063),
    (-6.28, 9.431),
    (-5.49, 2.301e1),
    (-3.87, 9.334e1),
    (-2.20, 3.878e2),
    (-0.49, 1.981e3),
    (2.18, 4.476e3),
    (6.81, 8.167e3),
    (11.65, 1.069e4),
    (27.02, 1.432e4),
    (65.40, 1.524e4),
    (90.0, 1.524e4),
];

/// Log-linear interpolation of [`GLOW_TABLE`], clamped at its ends.
pub(crate) fn horizon_glow(altitude_deg: f32) -> f32 {
    let first = GLOW_TABLE[0];
    if altitude_deg <= first.0 {
        return first.1;
    }
    for pair in GLOW_TABLE.windows(2) {
        let (a, b) = (pair[0], pair[1]);
        if altitude_deg <= b.0 {
            let t = (altitude_deg - a.0) / (b.0 - a.0);
            return (a.1.ln() + (b.1.ln() - a.1.ln()) * t).exp();
        }
    }
    GLOW_TABLE[GLOW_TABLE.len() - 1].1
}

/// The lowest EV₁₀₀ at which the sun-side glow stays at or under [`GLOW_DISPLAY_MAX`]. The glow
/// is light scattered along tens of kilometres of air toward the sun, so it is dimmed by the
/// cover's mean beam ([`beam_transmittance`]), not by the cloud shadow over the camera (which a
/// low sun's slant ray almost always finds).
pub(crate) fn ev100_for_glow(altitude_deg: f32, cover: f32) -> f32 {
    let glow = horizon_glow(altitude_deg) * beam_transmittance(cover);
    if glow <= 0.0 {
        return f32::NEG_INFINITY;
    }
    (glow / (1.2 * GLOW_DISPLAY_MAX)).log2()
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
///
/// The **night sky's own light** is added to it (`docs/research/night-sky-light.md`): the
/// airglow, zodiacal light and starlight that lie *behind* the atmosphere and so are not dimmed
/// by the cloud. Without it a moonless night has no ambient at all and the ground is lit by
/// nothing — the isometric town rendered at 1 DN, the sky, the ground and the houses' windows
/// all black. Jensen et al. 2001 §6: "These effects are especially important on moonless
/// nights, when a small change in illumination can determine whether an object in the scene is
/// visible or invisible."
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
    // The night sky's own radiance, as a colour (luminance 1) and a brightness in cd/m². It is
    // behind the atmosphere, so the cloud does not dim it, and it is *added* to the body's
    // light: the two have different origins and no common factor.
    let night = NIGHT_SKY_LUMINANCE;
    let total = lux + night * std::f32::consts::PI;
    if total <= 0.0 {
        return (Vec3::ONE, 0.0);
    }
    // The colour: the bodies' light, and the night sky's own. The night sky's colour is the
    // *scotopic* one: Jensen et al. 2001 §7 on the blue shift — "the subjective impression
    // that night scenes exhibit a bluish tint", supported by psychophysical data — and the
    // rod's curve peaks at 507 nm, which is where the airglow's 557.7 nm O-I line is read.
    let body = if lux > 0.0 {
        (dusk_tint(state.altitude_deg) * sun + MOON_TINT * moon) / lux
    } else {
        Vec3::ONE
    };
    let mixed = (body * lux + NIGHT_SKY_COLOUR * night * std::f32::consts::PI) / total;
    let luminance = mixed.dot(LUMINANCE_WEIGHTS);
    (mixed / luminance, total / std::f32::consts::PI)
}

/// The night sky's own colour, as a multiplier on its radiance: the *scotopic* one, a cool
/// blue-green. Jensen et al. 2001 §7 on the blue shift — night scenes "exhibit a bluish tint" —
/// and the rod's luminous efficiency, whose peak is at 507 nm. The airglow's own lines are
/// green (557.7 nm O-I) and red (630 nm O-I), but the scotopic eye reads 630 nm very weakly and
/// 507 nm at its peak, so the sky's glow reads blue-green.
pub(crate) const NIGHT_SKY_COLOUR: Vec3 = Vec3::new(0.72, 0.95, 1.28);

/// Rec. 709 luminance weights of linear RGB.
const LUMINANCE_WEIGHTS: Vec3 = Vec3::new(0.2126, 0.7152, 0.0722);

/// What Bevy's `ambient_light()` makes of a uniform radiance `L`: a diffuse surface shows
/// `albedo · L · s` with `s = F_ab.x` of `F_AB(1.0, NdotV)` (`EnvBRDFApprox(diffuse_color, ..)`,
/// `bevy_pbr-0.20.0/src/render/pbr_ambient.wesl:18`, `pbr_lighting.wesl:527-544`). At perceptual
/// roughness 1 the polynomial is `r = (0, 0.015, 0.468, −0.018)`, `a004 = 0.015` for every
/// `NdotV`, `s = 0.468 − 1.04 · 0.015 = 0.452`, not the Lambertian 1. The ambient light is
/// scaled up by `1/s` so a horizontal surface receives the illuminance `π L` that
/// [`cloud_ambient`] computed.
pub(crate) const BEVY_AMBIENT_DIFFUSE_SCALE: f32 = 0.452;

/// Give every [`ElementsCamera`] the cloud-scattered light as its `AmbientLight`: Bevy's
/// per-camera ambient is a uniform sky radiance in cd/m² that casts no shadow, which is what
/// the diffuse remainder under cloud is. A camera whose environment map the clouds' sky probe
/// has taken over gets that light from the cube, graded over the dome and reflected as the sky
/// shows it, and no ambient (`docs/research/sky-light.md`, Deviation 1).
fn update_ambient(
    weather: Res<Weather>,
    state: Res<SunState>,
    frame: Res<CelestialFrame>,
    shadows: Option<Res<CloudShadows>>,
    mut cameras: Query<(&ElementsCamera, &mut AmbientLight, Option<&SkyProbe>)>,
) {
    let cloud = cloud_light(weather.cloud_cover, shadows.as_deref());
    let (colour, radiance) = cloud_ambient(&state, frame.moon_lux, cloud);
    let colour = Color::linear_rgb(colour.x, colour.y, colour.z);
    for (dressing, mut ambient, probe) in &mut cameras {
        if !dressing.ambient {
            continue;
        }
        let brightness = if probe.is_some_and(|p| p.lit) {
            0.0
        } else {
            radiance / BEVY_AMBIENT_DIFFUSE_SCALE
        };
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

/// The sky's state the exposure is metered from.
#[derive(SystemParam)]
pub(crate) struct ExposureSky<'w> {
    state: Res<'w, SunState>,
    frame: Res<'w, CelestialFrame>,
    weather: Res<'w, Weather>,
    shadows: Option<Res<'w, CloudShadows>>,
}

fn auto_exposure(
    time: Res<Time<Real>>,
    sky: ExposureSky,
    mut published: ResMut<ElementsExposure>,
    mut cameras: Query<(Entity, Option<&ElementsCamera>, &mut Exposure)>,
    mut adapted: Local<Vec<Entity>>,
) {
    // Illuminance-based, with the sky toward the sun as the highlight that caps it (Patry 2021;
    // `docs/research/twilight.md`).
    let cloud = cloud_light(sky.weather.cloud_cover, sky.shadows.as_deref());
    let target = ev100_for(scene_illuminance(&sky.state, sky.frame.moon_lux, cloud))
        .max(ev100_for_glow(sky.state.altitude_deg, sky.weather.cloud_cover));
    let k = 1.0 - (-time.delta_secs() / EXPOSURE_TAU_S).exp();
    for (entity, camera, mut exposure) in &mut cameras {
        let Some(camera) = camera else {
            continue;
        };
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
    // The crate's metered target, which is what a camera gets when there is none.
    published.ev100 = target;
    published.exposure = bevy::camera::Exposure { ev100: target }.exposure();
}

/// Publish the exposure the camera really has, so a game that lights its own night can balance
/// its own light against it.
///
/// What the camera has, whether the crate metered it or the game drove it (`auto_exposure`
/// off): a game needs the exposure the camera is really using, not the one the crate would
/// have chosen. With no camera at all (a headless world, or one whose camera is added later)
/// the crate's own metered target is what a camera would get, so that is what it publishes.
fn publish_exposure(
    view: Res<ElementsView>,
    mut published: ResMut<ElementsExposure>,
    cameras: Query<(Entity, &Exposure)>,
) {
    let mut followed: Option<f32> = None;
    let mut any: Option<f32> = None;
    for (entity, exposure) in &cameras {
        let ev = exposure.ev100;
        if any.is_none() {
            any = Some(ev);
        }
        if Some(entity) == view.entity {
            followed = Some(ev);
        }
    }
    let Some(ev100) = followed.or(any) else {
        return;
    };
    published.ev100 = ev100;
    published.exposure = bevy::camera::Exposure { ev100 }.exposure();
}

/// Luminance of the moonless night sky behind the atmosphere (airglow, zodiacal light,
/// starlight; Jensen et al. 2001), cd/m²: the uniform sky that gives the metered illuminance
/// floor, `E / π`.
pub(crate) const NIGHT_SKY_LUMINANCE: f32 = ILLUMINANCE_FLOOR_LUX / std::f32::consts::PI;

/// The atmosphere's transmittance at the zenith, per channel: what the sky behind it comes
/// through. Measured from the capture against the probe's neutral clear colour
/// (`docs/research/night-sky-colour.md`, Deviation 1): (0.904, 0.640, 0.353). It is Rayleigh
/// extinction — the blue is scattered out of the line of sight most — and it is *warm*, so a
/// neutral clear colour behind it is a brown sky.
pub(crate) const ZENITH_TRANSMITTANCE: Vec3 = Vec3::new(0.904, 0.640, 0.353);

/// Clear every [`SkyBackground`] camera to the night sky's background radiance at its exposure.
/// The clear colour is written to the target before the sky and is never exposed, so it is
/// given in display units; Bevy's atmosphere then draws over it with its transmittance.
///
/// The clear colour is what lies **behind** the atmosphere, and the atmosphere's transmittance is
/// warm — it takes the blue out of the line of sight most — so a neutral clear colour comes
/// through as a sepia sky. The clear colour is therefore the night sky's *scotopic* colour
/// divided by that transmittance: the sky that comes out is the cool blue-black the rod's eye
/// expects (Jensen et al. 2001 §7 on the blue shift, and the scotopic curve, whose peak is at
/// 507 nm), and the horizon, whose path is longer, is warmer — as it is in the capture
/// (red/green 1.41 at the zenith, 2.12 at the low sky).
fn sky_background(mut cameras: Query<(&mut Camera, Option<&Exposure>), With<SkyBackground>>) {
    for (mut camera, exposure) in &mut cameras {
        let v = NIGHT_SKY_LUMINANCE * exposure.copied().unwrap_or_default().exposure();
        let colour = Color::linear_rgb(
            v * NIGHT_SKY_COLOUR.x / ZENITH_TRANSMITTANCE.x,
            v * NIGHT_SKY_COLOUR.y / ZENITH_TRANSMITTANCE.y,
            v * NIGHT_SKY_COLOUR.z / ZENITH_TRANSMITTANCE.z,
        );
        if !matches!(camera.clear_color, ClearColorConfig::Custom(c) if c == colour) {
            camera.clear_color = ClearColorConfig::Custom(colour);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::ecs::system::RunSystemOnce;

    /// A world in feet: the cascades reach the same metres, so their world distances are the
    /// metre ones times the units (issue #8). The default 1.0 is the metre scene itself.
    #[test]
    fn the_shadow_cascades_reach_the_same_metres_in_a_world_of_feet() {
        let feet = 3.2808_f32;
        let (m, f) = (sun_cascades(1.0), sun_cascades(feet));
        assert_eq!(m.bounds.len(), f.bounds.len());
        for (a, b) in m.bounds.iter().zip(&f.bounds) {
            assert!((a * feet - b).abs() <= 1e-3 * b, "{a} {b}");
        }
        assert!(
            (m.bounds.last().copied().unwrap_or_default() - SUN_SHADOW_DISTANCE_M).abs() < 1e-2
        );
        let (m, f) = (
            orthographic_cascades(1500.0, 1.0),
            orthographic_cascades(1500.0, feet),
        );
        assert_eq!(m.bounds.len(), 1);
        assert!((m.minimum_distance * feet - f.minimum_distance).abs() < 1e-2);
        assert!((m.bounds[0] * feet - f.bounds[0]).abs() < 1e-2);
        assert_eq!(m.minimum_distance, 1500.0 - 0.5 * SUN_SHADOW_DISTANCE_M);
    }

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

    fn shadows(mean_transmittance: f32) -> CloudShadows {
        CloudShadows {
            map: Handle::default(),
            origin: Vec2::ZERO,
            extent: 8000.0,
            light_direction: Vec3::Y,
            mean_transmittance,
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
        world.insert_resource(CelestialSettings::default());
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
        assert!(world.run_system_once(drive_lights).is_ok());
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

        // Before the first readback the mean transmittance is `1 − cover^3.4`: same split.
        for c in [0.0f32, 0.3, 0.7, 1.0] {
            let s = shadows(beam_transmittance(c));
            assert_eq!(cloud_light(c, Some(&s)), cloud_light(c, None));
        }
    }

    #[test]
    fn the_ambient_carries_the_diffuse_share_and_the_night_sky_at_night() {
        let noon = SunState {
            altitude_deg: 90.0,
            moon_altitude_deg: -30.0,
            ..default()
        };
        let clear_noon = scene_illuminance(&noon, 0.0, CloudLight::CLEAR) - ILLUMINANCE_FLOOR_LUX;
        // Cover 0: all direct, no ambient but the night sky's own glow, which is a millionth
        // of the noon and so does not move the day.
        let (_, brightness) = cloud_ambient(&noon, 0.0, CloudLight::CLEAR);
        assert!(
            brightness < NIGHT_SKY_LUMINANCE * 1.001,
            "the day carries the night sky's glow: {brightness}"
        );
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

        // A night with no moon is lit by the **night sky's own glow** — the airglow, zodiacal
        // light and starlight that lie behind the atmosphere — and not by the cloud, which is
        // between the sky and the ground but not between the sky's *source* and the ground.
        // The test used to assert the opposite ("a night with no moon stays dark whatever the
        // cloud"), and that is what made the isometric town render at 1 DN: the sky was lit
        // and the world on it was not (`docs/research/night-sky-light.md`).
        let dark = SunState {
            altitude_deg: -30.0,
            moon_altitude_deg: -30.0,
            ..default()
        };
        for t in [0.0, 0.5, 1.0] {
            let (colour, brightness) = cloud_ambient(&dark, 0.0, CloudLight::from_beam(t));
            assert!(
                close(brightness, NIGHT_SKY_LUMINANCE),
                "the night sky's radiance, T = {t}: {brightness}"
            );
            // The colour is the *scotopic* one, a cool blue-green: Jensen et al. 2001 §7 on
            // the blue shift, and the rod's curve, whose peak is at 507 nm.
            assert!(colour.z > colour.y && colour.y > colour.x, "the night is blue: {colour}");
        }
        // The cloud does not dim it: the night sky is behind the atmosphere.
        let (colour, clear) = cloud_ambient(&dark, 0.0, CloudLight::from_beam(0.0));
        let (_, overcast) = cloud_ambient(&dark, 0.0, CloudLight::from_beam(1.0));
        assert!(close(clear, overcast), "the cloud dimmed the night sky");
        assert!(colour.z > colour.x, "the night is blue: {colour}");
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

    /// The clear colour behind the atmosphere must be blue enough to survive the atmosphere's
    /// warm transmittance. A *neutral* clear colour comes through as a sepia sky: the night sky
    /// measured R/G 1.52 and B/G 0.51 before this, and the crate's own `twilight.md` says the
    /// night should be "Nothing brown" (`docs/research/night-sky-colour.md`).
    ///
    /// The test checks the sky's *result* rather than the clear colour: the ratio of the
    /// clear colour to the zenith's transmittance must be the scotopic colour.
    #[test]
    fn the_night_sky_behind_the_atmosphere_is_blue() {
        // The clear colour is the scotopic colour divided by the transmittance, so what comes
        // through is the scotopic colour.
        let clear = NIGHT_SKY_COLOUR / ZENITH_TRANSMITTANCE;
        let through = clear * ZENITH_TRANSMITTANCE;
        assert!(
            through.z > through.x,
            "the sky behind the atmosphere is not blue: {through:?}"
        );
        for c in 0..3 {
            assert!(
                (through[c] - NIGHT_SKY_COLOUR[c]).abs() < 1e-4,
                "the round trip is not the scotopic colour: {through:?}"
            );
        }
        // And the clear colour is *more* blue than the scotopic colour, not less: that is what
        // the division buys, and it is only worth doing because the transmittance is warm.
        assert!(
            clear.z / clear.x > NIGHT_SKY_COLOUR.z / NIGHT_SKY_COLOUR.x,
            "the clear colour is not blue enough: {clear:?}"
        );
        // The transmittance must be warm — the blue is scattered out of the line of sight
        // most — or the division would be unnecessary.
        let warm = ZENITH_TRANSMITTANCE.x - ZENITH_TRANSMITTANCE.z;
        assert!(warm > 0.0, "the transmittance is not warm: {ZENITH_TRANSMITTANCE:?}");
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

    #[test]
    fn the_sky_gain_never_amplifies_the_sky_once_the_sun_has_set() {
        let at = |altitude_deg: f32| SunState {
            altitude_deg,
            ..default()
        };
        for t in [0.0f32, 0.01, 0.5, 1.0] {
            let light = CloudLight::from_beam(t);
            // Up: the beam, restored to the dome's total.
            assert!((sun_cloud_factor(&at(5.0), light) - t).abs() < 1e-6, "{t}");
            // Set: the dome's total on the light itself, and no gain on the night behind it.
            for altitude in [-0.3, -3.0, -10.0, -17.0, -30.0] {
                assert_eq!(sun_cloud_factor(&at(altitude), light), light.total());
                assert!(
                    (sky_gain(&at(altitude), light) - 1.0).abs() < 1e-6,
                    "{altitude} {t}"
                );
            }
        }
    }

    #[test]
    fn the_sun_lights_the_sky_through_twilight_and_the_ground_only_while_its_disc_is_up() {
        // Continuous at sunset, falling through twilight, off at astronomical night.
        assert_eq!(twilight_factor(0.0), 1.0);
        assert!((twilight_factor(-0.01) - 1.0).abs() < 0.01);
        let mut last = 1.0;
        for tenth in 1..180 {
            let g = twilight_factor(-0.1 * tenth as f32);
            assert!(g < last && g > 0.0, "{tenth}: {g}");
            last = g;
        }
        assert_eq!(twilight_factor(-18.0), 0.0);
        // Off where it no longer shows: Bevy's sky keeps ~4 cd/m² per full sun from its
        // multiple-scattering floor; at the switch-off that is under 2 % of the 1.6e-4 cd/m² night
        // background (about a fifth of a code value at the night exposure).
        assert!(4.1 * twilight_factor(-17.99) < 0.02 * NIGHT_SKY_LUMINANCE);
        // The disc: whole above its radius, gone below it, half on the horizon.
        assert_eq!(sun_disc_visible(0.3), 1.0);
        assert_eq!(sun_disc_visible(-0.3), 0.0);
        assert!((sun_disc_visible(0.0) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn the_glow_caps_the_exposure_only_for_a_low_sun() {
        let target = |altitude_deg: f32, cover: f32| {
            let state = SunState {
                altitude_deg,
                moon_altitude_deg: -30.0,
                ..default()
            };
            let cloud = CloudLight::from_beam(beam_transmittance(cover));
            let illuminance = ev100_for(scene_illuminance(&state, 0.0, cloud));
            (illuminance, ev100_for_glow(altitude_deg, cover))
        };
        // The afternoon is metered on illuminance alone.
        for altitude in [27.0, 45.0, 68.0, 90.0] {
            let (lux, glow) = target(altitude, 0.05);
            assert!(glow <= lux + 0.1, "{altitude}°: {glow} over {lux}");
        }
        // Around sunset the glow wins, by more than a stop at the horizon.
        for altitude in [8.0, 2.0, -0.5, -2.0, -4.0] {
            let (lux, glow) = target(altitude, 0.05);
            assert!(glow > lux, "{altitude}°: {glow} under {lux}");
        }
        let (lux, glow) = target(-0.5, 0.05);
        assert!(glow - lux > 1.0, "{glow} against {lux}");
        // Under overcast the glow is hidden and the illuminance meters again.
        let (lux, glow) = target(-0.5, 1.0);
        assert!(glow < lux, "{glow} against {lux}");
    }
}

