//! Precipitation: a GPU particle set in a 60 m box around the camera for the streaks, flakes and
//! ground splashes that can be told apart, drawn after the main pass and soft-faded against the
//! depth prepass; and the far-field shafts and haze, which the ground-fog density kernel
//! (`fog`) renders from the same extinction formulas in [`model`]. See
//! `docs/research/precipitation.md`.
//!
//! The particles read the wind field, so [`WindPlugin`](crate::WindPlugin) must be present;
//! without it (or without an [`ElementsCamera`](crate::ElementsCamera)) nothing is simulated or
//! drawn. Ground is the terrain's surface plus the water standing on it (the plane y = 0 without a
//! terrain). A rain impact rings on standing water at least
//! [`WetnessSettings::puddle_min_m`](crate::WetnessSettings) deep and on a blocker's top face, not
//! on the terrain's dry or merely wet ground (the plane without a terrain rings everywhere);
//! `docs/research/wet-ground.md`.

mod gpu;
mod look;
pub(crate) mod model;

pub use look::PrecipitationLook;

use bevy::asset::embedded_asset;
use bevy::color::Luminance;
use bevy::ecs::system::SystemParam;
use bevy::light::DirectionalLight;
use bevy::prelude::*;
use bevy::render::RenderApp;
use bevy::render::extract_component::ExtractComponent;
use bevy::render::render_resource::ShaderType;
use bevy::render::storage::ShaderBuffer;
use bevy::shader::load_shader_library;

use crate::LightningFlash;
use crate::celestial::{
    BEVY_AMBIENT_DIFFUSE_SCALE, CelestialFrame, CelestialSettings, Moon, Sun, SunState,
    cloud_light, scene_illuminance, sun_disc_visible,
};
use crate::climate::Weather;
use crate::clouds::CloudShadows;
use crate::cover::WetnessSettings;
use crate::fog::extinction_for_visibility;
use crate::gpu::grid::storage_buffer;
use crate::plugin::{
    ElementsCamera, ElementsQuality, ElementsSystems, ElementsUnits, ElementsView, ensure_gpu_core,
    needs_depth_prepass,
};
use crate::terrain::{NO_TERRAIN, TerrainParamsGpu, TerrainState};
use crate::wind::{WindFieldParams, WindState};
use model::{
    DUST_FALL_SPEED_MS, DUST_HEIGHT_SCALE_M, DUST_SPRITE_DRAW_M, marshall_palmer_lambda,
    particle_counts,
};

/// Edge of the camera-centred particle box, m.
pub const PRECIPITATION_BOX_M: f32 = 60.0;
/// Most blockers uploaded to the GPU per frame; the nearest to the camera win.
pub const MAX_PRECIPITATION_BLOCKERS: usize = 256;
/// Entries of the splash ring buffer.
pub(crate) const SPLASH_RING: usize = 4096;
/// Rain impacts closer than this horizontally to the camera spawn a splash, m.
const SPLASH_RADIUS_M: f32 = 20.0;
/// Height of the ground plane without a terrain, m.
const GROUND_M: f32 = 0.0;
/// The shaders' clock wraps at this many seconds so f32 keeps its resolution.
const TIME_WRAP_S: f32 = 1024.0;
/// Snow flutter: lateral speed amplitude m/s, fall speed m/s, frequency range Hz.
const SNOW_FLUTTER: Vec4 = Vec4::new(0.3, 1.0, 0.5, 1.5);
/// Rain streak opacity before the coverage law (`α₀`, plan 2.2).
const STREAK_ALPHA: f32 = 1.5;
/// Streak exposure, s (Garg & Nayar 2006).
const STREAK_EXPOSURE_S: f32 = 1.0 / 60.0;
/// Albedo of a snowflake.
const SNOW_ALBEDO: f32 = 0.85;
/// Splash ring radius at the end of its life, m, and its lifetime, s.
const SPLASH_RADIUS_END_M: f32 = 0.15;
const SPLASH_LIFE_S: f32 = 0.25;
/// Soft-particle fade against scene depth for streaks and flakes, and for splashes, m.
const SOFT_FADE_M: f32 = 0.3;
const SPLASH_SOFT_FADE_M: f32 = 0.05;
/// Width of the fade-out at the sides of the box so its edges do not show, m.
const EDGE_FADE_M: f32 = 8.0;
/// `Particle::kind` of a parked (inactive) particle. The kind is in the low two bits: 0 rain,
/// 1 snow, 2 dust, 3 parked; the upper bits count respawns.
const KIND_PARKED: u32 = 3;

/// A box that stops precipitation: a roof, an awning, a tent. The box is `half_extents` in the
/// entity's local space, placed by its `GlobalTransform` (rotation and scale included), so any
/// oriented box works. Drops and flakes that reach it are respawned at the top of the box (rain
/// splashes on its top face); the nearest [`MAX_PRECIPITATION_BLOCKERS`] to the camera are
/// tested.
#[derive(Component, Clone, Copy, Debug, Default, PartialEq, Reflect)]
#[require(Transform)]
pub struct PrecipitationBlocker {
    /// Half the box's size along each local axis, m, before the transform's scale. A component
    /// that is zero or negative disables the blocker.
    pub half_extents: Vec3,
}

/// The GPU particle and splash buffers, for tests and custom effects that want to read them.
///
/// `particles` holds `capacity` 32-byte records `{pos: vec3<f32>, kind: u32, vel: vec3<f32>,
/// life: f32}`; `kind & 3` is 0 for rain, 1 for snow, 2 for dust, 3 for a parked (inactive)
/// particle, and the upper bits count respawns. The first `rain` records are rain, the next
/// `snow` are snow, the next `dust` are dust, every other record is parked. `splashes` is
/// `{counter: u32, 3 × u32 padding, ring: 4096 ×
/// vec4<f32>}` where a ring entry is the splash position and the (wrapped) birth time, or a
/// negative birth time when never written.
#[derive(Resource, Clone, Debug)]
pub struct PrecipitationState {
    /// The particle buffer.
    pub particles: Handle<ShaderBuffer>,
    /// The splash ring buffer.
    pub splashes: Handle<ShaderBuffer>,
    /// Records in `particles`: the quality maximum.
    pub capacity: u32,
    /// Active rain particles this frame.
    pub rain: u32,
    /// Active snow particles this frame.
    pub snow: u32,
    /// Active dust particles this frame.
    pub dust: u32,
}

/// Marks a camera the precipitation draws into; added to every [`ElementsCamera`].
#[derive(Component, Clone, Copy, Default, ExtractComponent)]
#[extract_app(RenderApp)]
pub(crate) struct PrecipitationView;

/// Uniform of `precip.wesl`.
#[derive(Clone, Copy, Debug, Default, ShaderType)]
pub(crate) struct PrecipParams {
    /// xyz: followed camera, m. w: Δt, s.
    camera: Vec4,
    /// x: shader clock, s. y: box half edge, m. z: splash radius, m. w: ground height, m.
    motion: Vec4,
    /// x: rain count. y: snow count. z: buffer capacity. w: blocker count.
    counts: UVec4,
    /// x: Marshall–Palmer Λ, mm⁻¹. y: the standing water an impact on a live terrain needs to
    /// ring, m.
    rain: Vec4,
    /// x: flutter speed m/s. y: snow fall speed m/s. zw: flutter frequency range, Hz.
    snow: Vec4,
    /// x: dust fall speed, m/s. y: dust height scale, m.
    dust: Vec4,
    /// x: dust count.
    dust_counts: UVec4,
    wind: WindFieldParams,
    terrain: TerrainParamsGpu,
}

/// Uniform of `draw.wesl`.
#[derive(Clone, Copy, Debug, Default, ShaderType)]
pub(crate) struct DrawParams {
    /// xyz: unit vector toward the sun. w: its direct illuminance, lx.
    sun: Vec4,
    /// xyz: unit vector toward the moon. w: its direct illuminance, lx.
    moon: Vec4,
    /// x: sky illuminance, lx. y: extinction σ_V, m⁻¹. z: soft fade, m. w: splash soft fade, m.
    ambient: Vec4,
    /// xyz: lightning flash position, m (metre frame). w: its intensity, cd; 0 when none.
    flash: Vec4,
    /// x: Marshall–Palmer Λ, mm⁻¹. y: streak alpha. z: exposure, s. w: edge fade, m.
    rain: Vec4,
    /// x: snow albedo. y: ring radius, m. z: ring life, s. w: shader clock, s.
    snow: Vec4,
    /// rgb: dust albedo. w: dust sprite diameter, m.
    dust: Vec4,
    /// xyz: camera velocity, m/s. w: box half edge, m.
    camera: Vec4,
    /// xyz: the box centre (the view's focus), m. w: under an orthographic view, the depth at
    /// which its distance fog starts (`fog::orthographic_near_depth`), m.
    centre: Vec4,
    /// x: world units per metre ([`crate::ElementsUnits`]). The particles are simulated in the
    /// metre frame; the shader draws them at world position × this, and takes the view's world
    /// distances and depths back to metres. yzw: unused.
    units: Vec4,
    /// [`PrecipitationLook`]: x rain fall scale, y rain wind scale, z flake size spread σ,
    /// w smallest flake half width, px.
    look: Vec4,
    /// [`PrecipitationLook::rain_tint`]: rgb on the streak's radiance, a on its opacity.
    tint: Vec4,
}

/// One blocker as the kernel reads it (80 bytes): the rows of the inverse of the box's affine
/// matrix, its centre and its half extents.
#[derive(Clone, Copy, Debug, Default, PartialEq, ShaderType)]
pub(crate) struct BlockerGpu {
    rot: [Vec4; 3],
    centre: Vec3,
    extent: Vec3,
}

impl BlockerGpu {
    /// The blocker for a box of `half_extents` placed by `transform`; `None` when degenerate.
    pub(crate) fn new(half_extents: Vec3, transform: &GlobalTransform) -> Option<Self> {
        if half_extents.min_element() <= 0.0 || !half_extents.is_finite() {
            return None;
        }
        let affine = transform.affine();
        let matrix = Mat3::from(affine.matrix3);
        if matrix.determinant().abs() < 1e-9 {
            return None;
        }
        let inverse = matrix.inverse();
        let row = |i: usize| inverse.row(i).extend(0.0);
        Some(Self {
            rot: [row(0), row(1), row(2)],
            centre: Vec3::from(affine.translation),
            extent: half_extents,
        })
    }
}

/// The nearest [`MAX_PRECIPITATION_BLOCKERS`] of `items` to `camera`, by squared distance of
/// the centre, ties by entity, so the set is deterministic.
pub(crate) fn nearest_blockers(
    camera: Vec3,
    items: impl Iterator<Item = (Entity, BlockerGpu)>,
) -> Vec<BlockerGpu> {
    let mut keyed: Vec<(f32, Entity, BlockerGpu)> = items
        .map(|(entity, b)| (b.centre.distance_squared(camera), entity, b))
        .collect();
    let order = |a: &(f32, Entity, BlockerGpu), b: &(f32, Entity, BlockerGpu)| {
        a.0.total_cmp(&b.0).then(a.1.cmp(&b.1))
    };
    if keyed.len() > MAX_PRECIPITATION_BLOCKERS {
        keyed.select_nth_unstable_by(MAX_PRECIPITATION_BLOCKERS, order);
        keyed.truncate(MAX_PRECIPITATION_BLOCKERS);
    }
    keyed.into_iter().map(|(_, _, b)| b).collect()
}

/// The particle buffers and this frame's parameters; extracted to the render world every frame.
#[derive(Resource, Clone)]
pub(crate) struct PrecipitationGpu {
    pub particles: Handle<ShaderBuffer>,
    pub splashes: Handle<ShaderBuffer>,
    pub wind_field: Handle<Image>,
    /// The terrain's `terrain_cells`, or the one-cell placeholder without a live terrain.
    pub terrain: Handle<ShaderBuffer>,
    pub capacity: u32,
    pub params: PrecipParams,
    pub draw: DrawParams,
    pub blockers: Vec<BlockerGpu>,
    /// Particles to draw: the active rain, snow and dust, which are the first records.
    pub drawn: u32,
    /// Whether the kernel runs this frame.
    pub active: bool,
}

#[derive(Resource, Default)]
struct PrecipitationRuntime {
    time: f32,
}

/// Rain and snow around the followed [`ElementsCamera`].
///
/// Needs [`WindPlugin`](crate::WindPlugin) (the particles ride the wind field) and
/// [`CelestialPlugin`](crate::CelestialPlugin) for the light they are drawn with; the
/// far-field shafts live in [`FogPlugin`](crate::FogPlugin).
pub struct PrecipitationPlugin;

impl Plugin for PrecipitationPlugin {
    fn build(&self, app: &mut App) {
        ensure_gpu_core(app);
        needs_depth_prepass(app);
        load_shader_library!(app, "common.wesl");
        embedded_asset!(app, "precip.wesl");
        embedded_asset!(app, "draw.wesl");
        app.init_resource::<PrecipitationRuntime>()
            .init_resource::<PrecipitationLook>()
            .init_resource::<CelestialFrame>()
            .init_resource::<LightningFlash>()
            .add_observer(mark_camera)
            .add_systems(
                Update,
                // The draw reads this frame's `LightningFlash`.
                (
                    ensure_buffers,
                    update_precipitation.after(crate::lightning::write_flash),
                )
                    .chain()
                    .in_set(ElementsSystems::Simulation),
            );
        gpu::plugin(app);
    }
}

fn mark_camera(add: On<Add<ElementsCamera>>, mut commands: Commands) {
    commands.entity(add.entity).insert(PrecipitationView);
}

fn ensure_buffers(
    mut commands: Commands,
    quality: Res<ElementsQuality>,
    state: Option<Res<PrecipitationState>>,
    wind: Res<WindState>,
    mut buffers: ResMut<Assets<ShaderBuffer>>,
) {
    let capacity = quality.precipitation_particles();
    if state.is_some_and(|s| s.capacity == capacity) {
        return;
    }
    // Every particle starts parked (kind 3): a zeroed record would be live rain.
    let parked = [0, 0, 0, KIND_PARKED, 0, 0, 0, 0];
    let particles = storage_buffer(
        &mut buffers,
        "elements_precip_particles",
        vec![parked; capacity as usize],
    );
    // A negative birth marks a ring entry that was never written.
    let mut ring = vec![[0.0, 0.0, 0.0, -1.0]; SPLASH_RING + 1];
    ring[0] = [0.0; 4];
    let splashes = storage_buffer(&mut buffers, "elements_precip_splashes", ring);
    commands.insert_resource(PrecipitationState {
        particles: particles.clone(),
        splashes: splashes.clone(),
        capacity,
        rain: 0,
        snow: 0,
        dust: 0,
    });
    commands.insert_resource(PrecipitationGpu {
        particles,
        splashes,
        wind_field: wind.field.clone(),
        terrain: NO_TERRAIN,
        capacity,
        params: PrecipParams::default(),
        draw: DrawParams::default(),
        blockers: Vec::new(),
        drawn: 0,
        active: false,
    });
}

/// Largest half edge the box grows to under an orthographic view, m: beyond it the drops would
/// be spread too thin to read.
const MAX_ORTHOGRAPHIC_HALF_EDGE_M: f32 = 120.0;

/// The box's half edge, m: [`PRECIPITATION_BOX_M`]` / 2`, or under an orthographic followed
/// camera enough to cover the ground it shows, half the diagonal of the view's footprint (its
/// width by its height over the sine of the look-down angle), up to
/// [`MAX_ORTHOGRAPHIC_HALF_EDGE_M`]. A perspective view sees the rain near the eye; an
/// orthographic one sees all of it at one size, so the container covers what the view shows
/// (Puig-Centelles et al. 2009, §4.3; `docs/research/view-anchor.md` Deviation 4). The view's
/// `area` is in world units, `units` converts it to metres.
pub(crate) fn box_half_edge(
    view: Option<(&Projection, &GlobalTransform)>,
    units: &ElementsUnits,
) -> f32 {
    let base = PRECIPITATION_BOX_M * 0.5;
    let Some((Projection::Orthographic(o), transform)) = view else {
        return base;
    };
    let down = (-transform.forward().y).clamp(0.2, 1.0);
    let footprint = units
        .metres(o.area.width())
        .hypot(units.metres(o.area.height()) / down);
    (0.5 * footprint).clamp(base, MAX_ORTHOGRAPHIC_HALF_EDGE_M)
}

/// The lights marked `Sun` or `Moon`, each with whether it is the sun.
type MarkedLights<'w, 's> = Query<
    'w,
    's,
    (
        &'static DirectionalLight,
        &'static GlobalTransform,
        Has<Sun>,
    ),
    Or<(With<Sun>, With<Moon>)>,
>;

/// What lights the particles: the sun and moon, the lights marked `Sun` and `Moon`, and the
/// followed camera's own ambient.
#[derive(SystemParam)]
struct ParticleLight<'w, 's> {
    sun: Res<'w, SunState>,
    frame: Res<'w, CelestialFrame>,
    celestial: Option<Res<'w, CelestialSettings>>,
    lights: MarkedLights<'w, 's>,
    dressings: Query<'w, 's, (&'static ElementsCamera, Option<&'static AmbientLight>)>,
}

/// The particles' light: the sun's direction and illuminance, the moon's, and the sky's.
struct DrawLight {
    sun_dir: Vec3,
    e_sun: f32,
    moon_dir: Vec3,
    e_moon: f32,
    e_sky: f32,
}

impl ParticleLight<'_, '_> {
    /// Light the particles with what the scene is lit with. The direct part is the light marked
    /// `Sun` (and `Moon`): the crate's, which carries the beam `E · T̄` (only the visible part of
    /// the sun's disc lights anything on the ground, though the sun lights the sky through
    /// twilight), or a game's own (`CelestialSettings::lights` off), in its direction and at its
    /// illuminance; without one, the light `update_sky` computes. The sky part is the rest of the
    /// scene illuminance (the sky and the cloud-scattered diffuse light) or, on a camera whose
    /// ambient the game keeps (`ElementsCamera::ambient` off), that `AmbientLight`
    /// (`docs/research/integration.md`).
    fn draw_light(
        &self,
        view: &ElementsView,
        weather: &Weather,
        shadows: Option<&CloudShadows>,
    ) -> DrawLight {
        let (sun, frame) = (&*self.sun, &*self.frame);
        let crate_lights = self.celestial.as_ref().is_none_or(|c| c.lights);
        let marked = |want_sun: bool| {
            self.lights
                .iter()
                .find(|(_, _, is_sun)| *is_sun == want_sun)
                .map(|(light, transform, _)| (*transform.back(), light.illuminance))
        };
        let disc = sun_disc_visible(sun.altitude_deg);
        let (sun_dir, e_sun) = match marked(true) {
            Some((dir, lux)) if crate_lights => (dir, lux * disc),
            Some(own) => own,
            None => (sun.direction, frame.sun_light_lux * disc),
        };
        let (moon_dir, e_moon) =
            marked(false).unwrap_or((sun.moon_direction, frame.moon_light_lux));
        let game_ambient = view
            .entity
            .and_then(|e| self.dressings.get(e).ok())
            .and_then(|(dressing, ambient)| ambient.filter(|_| !dressing.ambient));
        let e_sky = match game_ambient {
            Some(ambient) => {
                std::f32::consts::PI
                    * BEVY_AMBIENT_DIFFUSE_SCALE
                    * ambient.brightness
                    * ambient.color.luminance()
            }
            None => {
                scene_illuminance(
                    sun,
                    frame.moon_lux,
                    cloud_light(weather.cloud_cover, shadows),
                ) - e_sun * sun_dir.y.max(0.0)
                    - e_moon * moon_dir.y.max(0.0)
            }
        }
        .max(0.0005);
        DrawLight {
            sun_dir,
            e_sun,
            moon_dir,
            e_moon,
            e_sky,
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn update_precipitation(
    time: Res<Time<Virtual>>,
    view: Res<ElementsView>,
    units: Res<ElementsUnits>,
    cameras: Query<(&Projection, &GlobalTransform)>,
    weather: Res<Weather>,
    wind: Res<WindState>,
    light: ParticleLight,
    blockers: Query<(Entity, &PrecipitationBlocker, &GlobalTransform)>,
    mut runtime: ResMut<PrecipitationRuntime>,
    gpu: Option<ResMut<PrecipitationGpu>>,
    state: Option<ResMut<PrecipitationState>>,
    flash: Res<LightningFlash>,
    look: Res<PrecipitationLook>,
    shadows: Option<Res<CloudShadows>>,
    (terrain, wetness): (Option<Res<TerrainState>>, Option<Res<WetnessSettings>>),
) {
    let (Some(mut gpu), Some(mut state)) = (gpu, state) else {
        return;
    };
    let dt = time.delta_secs();
    runtime.time = (runtime.time + dt) % TIME_WRAP_S;
    gpu.wind_field = wind.field.clone();

    let snow_fraction = weather.snow_fraction();
    let counts = if view.entity.is_some() {
        particle_counts(
            gpu.capacity,
            weather.precipitation_mm_h,
            snow_fraction,
            weather.dust,
        )
    } else {
        model::ParticleCounts::default()
    };
    state.rain = counts.rain;
    state.snow = counts.snow;
    state.dust = counts.dust;
    gpu.drawn = counts.total();
    // A paused clock (dt = 0) still dispatches: the particles hold still and stay drawn, as the
    // clouds do.
    gpu.active = counts.total() > 0;
    if !gpu.active {
        return;
    }
    let camera = view.position;
    let followed = view.entity.and_then(|e| cameras.get(e).ok());
    let half_edge = box_half_edge(followed, &units);
    let fog_start = match followed {
        Some((Projection::Orthographic(o), t)) => crate::fog::orthographic_near_depth(
            units.point_to_metres(t.translation()),
            *t.forward(),
            view.position,
            units.metres(o.area.height()),
        )
        .max(0.0),
        _ => 0.0,
    };
    let live = terrain.as_ref().filter(|t| t.is_live());

    let selected = nearest_blockers(
        camera,
        blockers.iter().filter_map(|(entity, blocker, transform)| {
            BlockerGpu::new(blocker.half_extents, &units.to_metres(transform)).map(|b| (entity, b))
        }),
    );
    gpu.params = PrecipParams {
        camera: camera.extend(dt),
        motion: Vec4::new(
            runtime.time,
            half_edge,
            SPLASH_RADIUS_M * half_edge / (PRECIPITATION_BOX_M * 0.5),
            GROUND_M,
        ),
        counts: UVec4::new(
            counts.rain,
            counts.snow,
            gpu.capacity,
            selected.len() as u32,
        ),
        rain: Vec4::new(
            marshall_palmer_lambda(weather.precipitation_mm_h * (1.0 - snow_fraction)),
            wetness.map_or_else(
                || WetnessSettings::default().puddle_min_m,
                |w| w.puddle_min_m,
            ),
            0.0,
            0.0,
        ),
        snow: SNOW_FLUTTER,
        dust: Vec4::new(DUST_FALL_SPEED_MS, DUST_HEIGHT_SCALE_M, 0.0, 0.0),
        dust_counts: UVec4::new(counts.dust, 0, 0, 0),
        wind: wind.field_params(),
        terrain: live.map_or_else(TerrainParamsGpu::default, |t| {
            TerrainParamsGpu::new(&t.grid, 1, true)
        }),
    };
    gpu.terrain = live.map_or(NO_TERRAIN, |t| t.cells.clone());
    // Never an empty binding: the count says how many records are real.
    gpu.blockers = if selected.is_empty() {
        vec![BlockerGpu::default()]
    } else {
        selected
    };

    let lit = light.draw_light(&view, &weather, shadows.as_deref());
    let (style, tint) = look.gpu();
    gpu.draw = DrawParams {
        sun: lit.sun_dir.extend(lit.e_sun),
        moon: lit.moon_dir.extend(lit.e_moon),
        ambient: Vec4::new(
            lit.e_sky,
            extinction_for_visibility(weather.visibility_m),
            SOFT_FADE_M,
            SPLASH_SOFT_FADE_M,
        ),
        flash: units
            .point_to_metres(flash.position)
            .extend(flash.intensity_cd),
        rain: Vec4::new(
            marshall_palmer_lambda(weather.precipitation_mm_h * (1.0 - snow_fraction)),
            STREAK_ALPHA,
            STREAK_EXPOSURE_S,
            EDGE_FADE_M,
        ),
        snow: Vec4::new(
            SNOW_ALBEDO,
            SPLASH_RADIUS_END_M,
            SPLASH_LIFE_S,
            runtime.time,
        ),
        dust: crate::fog::DUST_ALBEDO.extend(DUST_SPRITE_DRAW_M),
        camera: view.velocity.extend(half_edge),
        centre: view.position.extend(fog_start),
        units: Vec4::new(units.per_metre(), 0.0, 0.0, 0.0),
        look: style,
        tint,
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    fn blocker_at(centre: Vec3) -> BlockerGpu {
        let transform = GlobalTransform::from(Transform::from_translation(centre));
        BlockerGpu::new(Vec3::ONE, &transform).expect("a unit box is valid")
    }

    /// CPU mirror of the kernel's point-in-box test.
    fn contains(b: &BlockerGpu, p: Vec3) -> bool {
        let d = p - b.centre;
        let q = Vec3::new(
            b.rot[0].truncate().dot(d),
            b.rot[1].truncate().dot(d),
            b.rot[2].truncate().dot(d),
        );
        q.abs().cmple(b.extent).all()
    }

    /// Validation 12: 300 blockers at distinct distances upload the 256 nearest, 80 bytes each.
    #[test]
    fn the_nearest_256_blockers_are_uploaded() {
        use bevy::render::render_resource::ShaderType;
        assert_eq!(BlockerGpu::min_size().get(), 80);
        let mut world = World::new();
        let camera = Vec3::new(5.0, 2.0, -3.0);
        // Spawn in an order unrelated to distance; distances 1..=300 m along x.
        let items: Vec<(Entity, BlockerGpu)> = (0..300)
            .map(|i| {
                let d = ((i * 7919) % 300 + 1) as f32;
                (world.spawn_empty().id(), blocker_at(camera + Vec3::X * d))
            })
            .collect();
        let chosen = nearest_blockers(camera, items.into_iter());
        assert_eq!(chosen.len(), MAX_PRECIPITATION_BLOCKERS);
        let mut distances: Vec<f32> = chosen.iter().map(|b| b.centre.distance(camera)).collect();
        distances.sort_by(f32::total_cmp);
        assert!((distances[0] - 1.0).abs() < 1e-4);
        assert!((distances[255] - 256.0).abs() < 1e-3, "{}", distances[255]);
        // Fewer than the cap are all kept.
        assert_eq!(nearest_blockers(camera, std::iter::empty()).len(), 0);
    }

    /// Validation 12: a rotated 10 × 1 × 10 m box classifies a point inside and a point 1 cm
    /// outside correctly, scale and rotation included.
    #[test]
    fn the_blocker_test_follows_rotation_and_scale() {
        let rotation = Quat::from_rotation_y(30f32.to_radians());
        let centre = Vec3::new(12.0, 7.0, -4.0);
        let transform = GlobalTransform::from(Transform {
            translation: centre,
            rotation,
            scale: Vec3::new(1.0, 1.0, 1.0),
        });
        let b = BlockerGpu::new(Vec3::new(5.0, 0.5, 5.0), &transform).expect("valid");
        let local = |x: f32, y: f32, z: f32| centre + rotation * Vec3::new(x, y, z);
        assert!(contains(&b, local(4.99, 0.49, -4.99)));
        assert!(contains(&b, centre));
        assert!(!contains(&b, local(5.01, 0.0, 0.0)));
        assert!(!contains(&b, local(0.0, 0.51, 0.0)));
        assert!(!contains(&b, local(0.0, 0.0, -5.01)));
        // Unrotated, the same points sit at different world positions: rotation matters.
        assert!(!contains(&b, centre + Vec3::new(5.0, 0.0, 5.0)));
        // Non-uniform scale stretches the box: half extents 5 × 0.5 × 5 scaled by 2 along x.
        let scaled = GlobalTransform::from(Transform {
            translation: centre,
            rotation,
            scale: Vec3::new(2.0, 1.0, 1.0),
        });
        let s = BlockerGpu::new(Vec3::new(5.0, 0.5, 5.0), &scaled).expect("valid");
        assert!(contains(&s, local(9.9, 0.0, 0.0)));
        assert!(!contains(&s, local(10.1, 0.0, 0.0)));
        // Degenerate boxes are dropped.
        assert!(BlockerGpu::new(Vec3::new(5.0, 0.0, 5.0), &transform).is_none());
        assert!(
            BlockerGpu::new(
                Vec3::ONE,
                &GlobalTransform::from(Transform::from_scale(Vec3::ZERO))
            )
            .is_none()
        );
    }

    /// Issue #8: the rain box keeps its 60 m in a world of feet, 196.8 units wide, and a
    /// blocker placed in feet is the same box in metres.
    #[test]
    fn the_rain_box_is_sixty_metres_in_a_world_of_feet() {
        let feet = ElementsUnits {
            units_per_metre: 3.2808,
        };
        let width = feet.world(2.0 * box_half_edge(None, &feet));
        assert!((width - 196.848).abs() < 1e-3, "{width}");
        let roof = GlobalTransform::from(
            Transform::from_xyz(32.808, 16.404, 0.0).with_scale(Vec3::splat(3.2808)),
        );
        let b = BlockerGpu::new(Vec3::ONE, &feet.to_metres(&roof)).expect("a valid box");
        assert!(contains(&b, Vec3::new(10.0, 5.0, 0.0) + Vec3::splat(0.9)));
        assert!(!contains(
            &b,
            Vec3::new(10.0, 5.0, 0.0) + Vec3::new(1.1, 0.0, 0.0)
        ));
    }
}
