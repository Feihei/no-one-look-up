//! Ground cover: snow and wetness maps on the terrain grid. Snow accumulates at each cell's own
//! rain–snow split and shelter, densifies, drifts with the wind, smooths, and melts into
//! hydrology's water layer; wetness follows rain and evaporation. The terrain and ground
//! materials shade the result (snow, wet darkening and gloss, puddles with ripples), and
//! [`SurfaceWeathering`](crate::SurfaceWeathering) becomes the map's mean around the camera. See
//! `docs/research/snow-cover.md` and `docs/research/wetness.md`.
//!
//! The simulation runs on game time ([`WorldClock`](crate::WorldClock)), at most
//! [`SnowSettings::max_rate_hz`] times a second of real time, after hydrology's substeps each
//! frame. It idles while there is no terrain. Gameplay reads the maps through
//! [`Environment::snow_depth`](crate::Environment::snow_depth) and
//! [`Environment::wetness`](crate::Environment::wetness); both are up to one readback interval
//! old.

mod gpu;
pub(crate) mod model;

#[cfg(test)]
mod gpu_tests;
#[cfg(test)]
mod mirror;

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Duration;

use bevy::asset::embedded_asset;
use bevy::math::UVec4;
use bevy::prelude::*;
use bevy::render::render_resource::ShaderType;
use bevy::render::storage::ShaderBuffer;
use bevy::shader::load_shader_library;

use crate::celestial::{SunState, WorldClock};
use crate::climate::weathering::update_weathering;
use crate::climate::{Climate, LAPSE_RATE_C_PER_M, SurfaceWeathering, Weather};
use crate::gpu::grid::zeroed_buffer;
use crate::gpu::readback::{ReadbackTick, Readbacks, decode};
use crate::plugin::{ElementsSystems, ElementsView, ensure_gpu_core};
use crate::precipitation::model::{marshall_palmer_lambda, water_weighted_fall_speed};
use crate::precipitation::{BlockerGpu, PrecipitationBlocker, nearest_blockers};
use crate::query::EnvironmentSnapshot;
use crate::terrain::{TerrainGpu, TerrainGrid, TerrainState};
use model::{
    MAX_PENDING_S, READBACK_MAX_CELLS, SURFACE_WEATHERING_RADIUS_M, plan_steps, quantise,
    snow_coverage,
};

/// The fall speed of snowflakes used for the shelter ray, m/s (`docs/research/wetness.md`).
const SNOW_FALL_SPEED_MS: f32 = 1.0;

/// How snow behaves: its density, how fast it settles and melts, when the wind moves it and which
/// slopes shed it. Insert it before the plugin to override the defaults.
#[derive(Resource, Clone, Debug, PartialEq, Reflect)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SnowSettings {
    /// Density of fresh snow, kg/m³. Default 100.
    pub new_density: f32,
    /// Density an old pack settles toward, kg/m³. Default 300.
    pub settled_density: f32,
    /// Time constant of the settling, days of game time. Default 3.
    pub settle_days: f32,
    /// Degree-day melt factor, mm of water equivalent per °C above freezing per day. Default 4.
    pub degree_day_factor: f32,
    /// The 10 m wind above which snow drifts, m/s. Default 5.
    pub drift_threshold_ms: f32,
    /// The drift flux coefficient `k_d`, kg s² m⁻⁴: the flux is `k_d/ρ_w · (u − u_t)³` m²/s of
    /// water equivalent. Default 3.7·10⁻⁵ (Pomeroy & Gray 1990 magnitudes).
    pub drift_coefficient: f32,
    /// Whether the wind moves snow. Default `true`.
    pub drift: bool,
    /// Slopes between these angles, degrees, keep less and less of the snowfall; at the upper
    /// one none stays. Default 35 to 55.
    pub shed_slope_deg: [f32; 2],
    /// The most times a second of real time the cover maps are stepped (snow and wetness), 0 for
    /// every frame. Each step integrates the game time since the last. Default 20.
    pub max_rate_hz: f32,
}

impl Default for SnowSettings {
    fn default() -> Self {
        Self {
            new_density: 100.0,
            settled_density: 300.0,
            settle_days: 3.0,
            degree_day_factor: 4.0,
            drift_threshold_ms: 5.0,
            drift_coefficient: 3.7e-5,
            drift: true,
            shed_slope_deg: [35.0, 55.0],
            max_rate_hz: 20.0,
        }
    }
}

/// How the ground gets wet and dries, and when water becomes a puddle. Insert it before the
/// plugin to override the defaults.
#[derive(Resource, Clone, Debug, PartialEq, Reflect)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct WetnessSettings {
    /// Wetness gained per mm of rain: 1 is a film of `1 / wet_per_mm` mm. Default 0.5.
    pub wet_per_mm: f32,
    /// Wetness lost per hour at the reference evaporation of 0.1 mm/h (scaled by the wind, the
    /// temperature, the humidity and the sun). Default 0.05.
    pub dry_rate_h: f32,
    /// Water depth where a puddle begins, m: the depth of a full film. Default 0.002.
    pub puddle_min_m: f32,
    /// Water depth of a full puddle, m. Default 0.02.
    pub puddle_full_m: f32,
}

impl Default for WetnessSettings {
    fn default() -> Self {
        Self {
            wet_per_mm: 0.5,
            dry_rate_h: 0.05,
            puddle_min_m: 0.002,
            puddle_full_m: 0.02,
        }
    }
}

/// What the cover simulation has done, for tests and examples that wait on game time. Reset when
/// the terrain is rebuilt.
#[derive(Resource, Clone, Debug, Default, PartialEq)]
pub struct CoverStats {
    /// Snow steps the GPU has recorded.
    pub steps: u64,
    /// Game time the recorded steps integrated, s.
    pub simulated_s: f64,
    /// Steps that were scheduled but could not be recorded (a pipeline or buffer was not ready on
    /// the render side that frame). Zero in normal running.
    pub dropped: u64,
}

/// The cover layers of the active terrain. Present while a terrain and the plugin exist.
///
/// The layers are on the GPU only; the handles let materials bind them and tests replace their
/// contents (the same pattern as [`TerrainState`]).
#[derive(Resource, Clone, Debug)]
pub struct CoverState {
    /// Snow, `array<vec4<f32>>`: depth m, water equivalent m, pending melt m, spare.
    pub snow: Handle<ShaderBuffer>,
    /// Wetness, `array<f32>`, 0–1.
    pub wetness: Handle<ShaderBuffer>,
    /// The render pack, `array<vec4<f32>>`: snow depth m, wetness, water equivalent m, spare.
    pub cover_cells: Handle<ShaderBuffer>,
}

/// The cover readback: `Environment::snow_depth` and `wetness` answer from it.
#[derive(Clone, Debug)]
pub(crate) struct CoverSnapshot {
    /// The readback grid (the simulation grid box-averaged by 2 above 1024²).
    pub grid: TerrainGrid,
    pub depth: Arc<[f32]>,
    pub wetness: Arc<[f32]>,
}

impl CoverSnapshot {
    pub(crate) fn depth_at(&self, p: Vec2) -> Option<f32> {
        self.grid
            .contains(p)
            .then(|| self.grid.sample(&self.depth, p).max(0.0))
    }

    pub(crate) fn wetness_at(&self, p: Vec2) -> Option<f32> {
        self.grid
            .contains(p)
            .then(|| self.grid.sample(&self.wetness, p).clamp(0.0, 1.0))
    }

    /// Mean snow depth (m), mean snow coverage and mean wetness of the cells whose centres lie
    /// within `radius_m` of `centre` (world XZ); `None` when there are none.
    pub(crate) fn mean_around(&self, centre: Vec2, radius_m: f32) -> Option<(f32, f32, f32)> {
        let g = &self.grid;
        let lo = ((centre - radius_m - g.origin) / g.cell_m - 0.5)
            .floor()
            .as_ivec2();
        let hi = ((centre + radius_m - g.origin) / g.cell_m - 0.5)
            .ceil()
            .as_ivec2();
        let size = g.size.as_ivec2();
        let (mut depth, mut cover, mut wet, mut n) = (0.0_f64, 0.0_f64, 0.0_f64, 0u32);
        for y in lo.y.max(0)..=hi.y.min(size.y - 1) {
            for x in lo.x.max(0)..=hi.x.min(size.x - 1) {
                let at = g.origin + (Vec2::new(x as f32, y as f32) + 0.5) * g.cell_m;
                if at.distance(centre) > radius_m {
                    continue;
                }
                let k = (y * size.x + x) as usize;
                depth += f64::from(self.depth[k]);
                cover += f64::from(snow_coverage(self.depth[k]));
                wet += f64::from(self.wetness[k]);
                n += 1;
            }
        }
        (n > 0).then(|| {
            let n = f64::from(n);
            ((depth / n) as f32, (cover / n) as f32, (wet / n) as f32)
        })
    }
}

/// Uniform of the cover kernels. Mirrors `CoverParams` of `common.wesl`.
#[derive(Clone, Copy, Debug, Default, PartialEq, ShaderType)]
pub(crate) struct CoverParams {
    origin: Vec2,
    cell: f32,
    pad0: f32,
    size: UVec2,
    pad1: UVec2,
    /// x: Δt of a step, s. y: precipitation, mm/h. z: air temperature at the reference
    /// elevation, °C. w: lapse, °C/m.
    sim: Vec4,
    /// x: terrain y0 − reference elevation, m. y: terrain y0. z: 10 m wind speed, m/s. w: Δt of
    /// the wetness update, s.
    terrain: Vec4,
    /// x: new-snow density. y: settled density. z: settling time constant, h. w: degree-day
    /// factor.
    snow: Vec4,
    /// x, y: shed slope range, rad. z: drift threshold, m/s. w: drift coefficient / ρ_w.
    shed: Vec4,
    /// xy: unit XZ direction the wind blows toward.
    wind: Vec4,
    /// x: wetness per mm. y: dry rate, 1/h. z: relative humidity. w: sun factor.
    wet: Vec4,
    /// x: wind speed for evaporation, m/s.
    air: Vec4,
    /// xy: horizontal velocity of falling rain (the wind), m/s. w: its fall speed, m/s.
    rain: Vec4,
    /// As `rain`, for snow.
    snow_ray: Vec4,
    /// x: blockers uploaded. z: readback downsample factor.
    flags: UVec4,
}

/// This frame's work and the GPU's progress; extracted to the render world every frame.
#[derive(Resource, Clone)]
pub(crate) struct CoverGpu {
    pub snow: Handle<ShaderBuffer>,
    pub drift: Handle<ShaderBuffer>,
    pub wetness: Handle<ShaderBuffer>,
    pub shelter: Handle<ShaderBuffer>,
    pub cover_cells: Handle<ShaderBuffer>,
    /// Box-averaged (depth, wetness), `array<vec2<f32>>`.
    pub readback: Handle<ShaderBuffer>,
    pub readback_ds: u32,
    pub params: CoverParams,
    /// Snow steps to run this frame, and whether they move snow with the wind.
    pub steps: u32,
    pub drift_on: bool,
    pub blockers: Vec<BlockerGpu>,
    /// The shelter mask is current when `shelter_built` has caught up with this.
    pub shelter_epoch: u64,
    pub shelter_built: Arc<AtomicU64>,
    /// Run the readback kernel this frame.
    pub downsample: bool,
    /// Steps recorded, game time they integrated (µs), and steps asked for that could not be.
    pub executed: Arc<AtomicU64>,
    pub executed_us: Arc<AtomicU64>,
    pub dropped: Arc<AtomicU64>,
    /// The render world has every pipeline and buffer it needs.
    pub ready: Arc<AtomicBool>,
    /// A first readback kernel has run: the readback buffer holds the maps.
    pub live: Arc<AtomicBool>,
}

/// What the shelter mask was built for; it is rebuilt when this changes.
#[derive(Clone, Debug, Default, PartialEq)]
struct ShelterKey {
    grid: Option<(Vec2, f32, UVec2)>,
    /// The wind and the rain fall speed, quantised.
    rays: [i32; 3],
}

/// Main-world bookkeeping.
#[derive(Resource, Default)]
struct CoverRuntime {
    /// The water buffer the current GPU resources were built for.
    built_for: Option<AssetId<ShaderBuffer>>,
    last_hours: Option<f64>,
    /// Game time not yet handed to the GPU, s.
    pending_s: f64,
    /// Real time since the last step.
    since_step: Duration,
    shelter_epoch: u64,
    shelter_key: ShelterKey,
    blockers: Vec<BlockerGpu>,
}

/// Ground cover: snow and wetness maps, their kernels and their readback.
///
/// Needs a [`Terrain`](crate::Terrain); idles without one. Inserts, if absent,
/// [`SnowSettings`], [`WetnessSettings`] and [`CoverStats`]. Add it with
/// [`ElementsPlugins`](crate::ElementsPlugins).
pub struct GroundCoverPlugin;

impl Plugin for GroundCoverPlugin {
    fn build(&self, app: &mut App) {
        ensure_gpu_core(app);
        load_shader_library!(app, "common.wesl");
        embedded_asset!(app, "snow.wesl");
        embedded_asset!(app, "shelter.wesl");
        embedded_asset!(app, "readback.wesl");
        app.init_resource::<SnowSettings>()
            .init_resource::<WetnessSettings>()
            .init_resource::<CoverStats>()
            .init_resource::<Climate>()
            .init_resource::<WorldClock>()
            .init_resource::<CoverRuntime>()
            .add_systems(
                Update,
                (ensure_cover, run_cover)
                    .chain()
                    .in_set(ElementsSystems::Simulation),
            )
            .add_systems(Update, read_cover.in_set(ElementsSystems::Readback))
            .add_systems(
                Update,
                weathering_from_map
                    .after(update_weathering)
                    .in_set(ElementsSystems::Weather),
            );
        gpu::plugin(app);
    }
}

/// Readback sizes of a grid: the downsample factor.
fn readback_downsample(size: UVec2) -> u32 {
    if size.max_element() > READBACK_MAX_CELLS {
        2
    } else {
        1
    }
}

/// Create the cover layers when a terrain appears or is rebuilt, and drop everything when it
/// goes.
fn ensure_cover(
    mut commands: Commands,
    terrain: Option<Res<TerrainState>>,
    gpu: Option<Res<CoverGpu>>,
    mut runtime: ResMut<CoverRuntime>,
    mut stats: ResMut<CoverStats>,
    mut snapshot: ResMut<EnvironmentSnapshot>,
    mut buffers: ResMut<Assets<ShaderBuffer>>,
) {
    let Some(terrain) = terrain else {
        if gpu.is_some() {
            commands.remove_resource::<CoverGpu>();
            commands.remove_resource::<CoverState>();
            *runtime = CoverRuntime::default();
            *stats = CoverStats::default();
            snapshot.cover = None;
        }
        return;
    };
    if runtime.built_for == Some(terrain.water.id()) && gpu.is_some() {
        return;
    }
    let size = terrain.grid.size;
    let cells = u64::from(size.x) * u64::from(size.y);
    let ds = readback_downsample(size);
    let readback_cells = u64::from(size.x / ds) * u64::from(size.y / ds);
    let snow = zeroed_buffer(&mut buffers, "elements_cover_snow", cells * 16);
    let wetness = zeroed_buffer(&mut buffers, "elements_cover_wetness", cells * 4);
    let cover_cells = zeroed_buffer(&mut buffers, "elements_cover_cells", cells * 16);
    *runtime = CoverRuntime {
        built_for: Some(terrain.water.id()),
        shelter_epoch: 1,
        ..CoverRuntime::default()
    };
    *stats = CoverStats::default();
    snapshot.cover = None;
    commands.insert_resource(CoverState {
        snow: snow.clone(),
        wetness: wetness.clone(),
        cover_cells: cover_cells.clone(),
    });
    commands.insert_resource(CoverGpu {
        snow,
        drift: zeroed_buffer(&mut buffers, "elements_cover_drift", cells * 16),
        wetness,
        shelter: zeroed_buffer(&mut buffers, "elements_cover_shelter", cells * 8),
        cover_cells,
        readback: zeroed_buffer(&mut buffers, "elements_cover_readback", readback_cells * 8),
        readback_ds: ds,
        params: CoverParams::default(),
        steps: 0,
        drift_on: false,
        blockers: Vec::new(),
        shelter_epoch: 1,
        shelter_built: Arc::new(AtomicU64::new(0)),
        downsample: false,
        executed: Arc::new(AtomicU64::new(0)),
        executed_us: Arc::new(AtomicU64::new(0)),
        dropped: Arc::new(AtomicU64::new(0)),
        ready: Arc::new(AtomicBool::new(false)),
        live: Arc::new(AtomicBool::new(false)),
    });
}

/// Schedule this frame's steps and fill the uniform.
#[allow(clippy::too_many_arguments)]
fn run_cover(
    snow: Res<SnowSettings>,
    wetness: Res<WetnessSettings>,
    weather: Res<Weather>,
    climate: Res<Climate>,
    sun: Res<SunState>,
    clock: Res<WorldClock>,
    view: Res<ElementsView>,
    time: Res<Time<Real>>,
    tick: Res<ReadbackTick>,
    terrain: Option<Res<TerrainState>>,
    terrain_gpu: Option<Res<TerrainGpu>>,
    gpu: Option<ResMut<CoverGpu>>,
    mut runtime: ResMut<CoverRuntime>,
    mut stats: ResMut<CoverStats>,
    blockers: Query<(Entity, &PrecipitationBlocker, &GlobalTransform)>,
) {
    let (Some(terrain), Some(terrain_gpu), Some(mut gpu)) = (terrain, terrain_gpu, gpu) else {
        return;
    };
    let grid = terrain.grid;
    stats.steps = gpu.executed.load(Ordering::Relaxed);
    stats.simulated_s = gpu.executed_us.load(Ordering::Relaxed) as f64 * 1e-6;
    stats.dropped = gpu.dropped.load(Ordering::Relaxed);

    // Game time since the last frame; a paused or rewound clock adds none.
    let now = clock.hours_since_2000();
    if let Some(last) = runtime.last_hours {
        runtime.pending_s = (runtime.pending_s + (now - last).max(0.0) * 3600.0).min(MAX_PENDING_S);
    }
    runtime.last_hours = Some(now);
    runtime.since_step += time.delta();

    // The shelter mask: the blockers and the fall directions of rain and snow.
    let wind = weather.wind_direction * weather.wind_speed_ms;
    let rain_mm_h = weather.precipitation_mm_h * (1.0 - weather.snow_fraction());
    let rain_fall = water_weighted_fall_speed(marshall_palmer_lambda(rain_mm_h.max(1.0)));
    let key = ShelterKey {
        grid: Some((grid.origin, grid.cell_m, grid.size)),
        rays: [
            quantise(wind.x, 0.25),
            quantise(wind.y, 0.25),
            quantise(rain_fall, 0.1),
        ],
    };
    if runtime.shelter_key != key {
        runtime.shelter_key = key;
        runtime.shelter_epoch += 1;
    }
    let selected = nearest_blockers(
        view.position,
        blockers.iter().filter_map(|(entity, blocker, transform)| {
            BlockerGpu::new(blocker.half_extents, transform).map(|b| (entity, b))
        }),
    );
    if runtime.blockers != selected {
        runtime.blockers = selected;
        runtime.shelter_epoch += 1;
    }

    // Steps: when the GPU is ready and the rate cap allows.
    let ready = gpu.ready.load(Ordering::Relaxed)
        && terrain.is_live()
        && terrain_gpu.generated.load(Ordering::Relaxed);
    let due = snow.max_rate_hz <= 0.0 || runtime.since_step.as_secs_f32() * snow.max_rate_hz >= 1.0;
    let mut steps = 0;
    let mut dt_s = 0.0;
    if ready
        && due
        && let Some((n, dt)) = plan_steps(runtime.pending_s)
    {
        steps = n;
        dt_s = dt;
        runtime.pending_s = 0.0;
        runtime.since_step = Duration::ZERO;
    }

    let direction = if weather.wind_direction.length_squared() > 1e-6 {
        weather.wind_direction.normalize()
    } else {
        Vec2::X
    };
    let sun_factor = sun.altitude_deg.to_radians().sin().max(0.0);
    let ray = |velocity: Vec2, fall: f32| Vec4::new(-velocity.x, -velocity.y, 0.0, fall);
    gpu.params = CoverParams {
        origin: grid.origin,
        cell: grid.cell_m,
        pad0: 0.0,
        size: grid.size,
        pad1: UVec2::ZERO,
        sim: Vec4::new(
            dt_s,
            weather.precipitation_mm_h.max(0.0),
            weather.temperature_c,
            LAPSE_RATE_C_PER_M,
        ),
        terrain: Vec4::new(
            grid.y0 - climate.reference_elevation_m,
            grid.y0,
            weather.wind_speed_ms,
            dt_s * steps as f32,
        ),
        snow: Vec4::new(
            snow.new_density,
            snow.settled_density,
            snow.settle_days * 24.0,
            snow.degree_day_factor,
        ),
        shed: Vec4::new(
            snow.shed_slope_deg[0].to_radians(),
            snow.shed_slope_deg[1].to_radians(),
            snow.drift_threshold_ms,
            snow.drift_coefficient / 1000.0,
        ),
        wind: Vec4::new(direction.x, direction.y, 0.0, 0.0),
        wet: Vec4::new(
            wetness.wet_per_mm,
            wetness.dry_rate_h,
            weather.relative_humidity,
            sun_factor,
        ),
        air: Vec4::new(weather.wind_speed_ms, 0.0, 0.0, 0.0),
        rain: ray(wind, rain_fall),
        snow_ray: ray(wind, SNOW_FALL_SPEED_MS),
        flags: UVec4::new(runtime.blockers.len() as u32, 0, gpu.readback_ds, 0),
    };
    gpu.steps = steps;
    gpu.drift_on = snow.drift && weather.wind_speed_ms * 2.0 > snow.drift_threshold_ms;
    // Never an empty binding: the count says how many entries are real.
    gpu.blockers = if runtime.blockers.is_empty() {
        vec![BlockerGpu::default()]
    } else {
        runtime.blockers.clone()
    };
    gpu.shelter_epoch = runtime.shelter_epoch;
    gpu.downsample = tick.is_due() || !gpu.live.load(Ordering::Relaxed);
    if steps > 0 {
        // The terrain pack publishes the melt to the render buffer's water depth.
        terrain_gpu.dirty.store(true, Ordering::Relaxed);
    }
}

/// Request the cover readback once the kernels have produced state.
fn read_cover(
    terrain: Option<Res<TerrainState>>,
    gpu: Option<Res<CoverGpu>>,
    tick: Res<ReadbackTick>,
    mut readbacks: Readbacks,
) {
    let (Some(terrain), Some(gpu)) = (terrain, gpu) else {
        return;
    };
    if !gpu.live.load(Ordering::Relaxed) || !tick.is_due() {
        return;
    }
    let ds = gpu.readback_ds;
    let grid = terrain.grid;
    let read_grid = TerrainGrid {
        cell_m: grid.cell_m * ds as f32,
        size: grid.size / ds,
        ..grid
    };
    let expected = (read_grid.size.x * read_grid.size.y) as usize;
    readbacks.request(
        "cover_maps",
        gpu.readback.clone(),
        move |bytes, snapshot| {
            let Some(cells) = decode::<[f32; 2]>(bytes) else {
                return;
            };
            if cells.len() < expected {
                return;
            }
            let cells = &cells[..expected];
            snapshot.cover = Some(CoverSnapshot {
                grid: read_grid,
                depth: cells.iter().map(|c| c[0]).collect(),
                wetness: cells.iter().map(|c| c[1]).collect(),
            });
        },
    );
}

/// `SurfaceWeathering` from the map within 50 m of the camera, when a terrain has been read
/// back; otherwise the point model's values stand.
fn weathering_from_map(
    snapshot: Res<EnvironmentSnapshot>,
    view: Res<ElementsView>,
    mut ground: ResMut<SurfaceWeathering>,
) {
    let Some(cover) = snapshot.cover.as_ref() else {
        return;
    };
    let centre = Vec2::new(view.position.x, view.position.z);
    if let Some((depth, coverage, wetness)) = cover.mean_around(centre, SURFACE_WEATHERING_RADIUS_M)
    {
        ground.snow_depth_m = depth;
        ground.snow_cover = coverage;
        ground.wetness = wetness;
    }
}
