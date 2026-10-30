//! Hydrology and erosion: flooding, rivers and lakes as one shallow-water field on the terrain
//! grid (Mei et al. 2007 virtual pipes with a depth-dependent pipe area), hydraulic erosion with
//! flux-form sediment transport, and thermal erosion. See `docs/research/hydrology.md` and
//! `docs/research/erosion.md`.
//!
//! The simulation runs on the [`Terrain`](crate::Terrain)'s grid and writes its `water`, `flux`,
//! `velocity`, `suspended`, `bedrock` and `sediment` layers. It idles while there is no terrain.
//! Each frame runs a whole number of fixed substeps ([`HydrologySettings::step_s`]) scheduled
//! from real time; [`ErodeTerrain`] bakes erosion at the quality's full rate instead.
//!
//! Gameplay reads the water through
//! [`Environment::water_depth`](crate::Environment::water_depth) and
//! [`Environment::water_velocity`](crate::Environment::water_velocity); both are up to one
//! readback interval old.

mod gpu;
pub(crate) use gpu::{HydrologyDispatch, HydrologyPipelines};
pub(crate) mod model;
mod water;
mod water_gpu;

pub use water::{
    WaterExtension, WaterMaterial, WaterParams, WaterPatch, WaterPlugin, WaterSettings, WaterState,
};

use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use bevy::asset::embedded_asset;
use bevy::ecs::system::SystemParam;
use bevy::math::UVec4;
use bevy::prelude::*;
use bevy::render::render_resource::ShaderType;
use bevy::render::storage::ShaderBuffer;

use crate::climate::{Climate, LAPSE_RATE_C_PER_M, Weather};
use crate::gpu::grid::zeroed_buffer;
use crate::gpu::readback::{ReadbackTick, Readbacks, decode};
use crate::plugin::{
    ElementsQuality, ElementsSystems, ElementsUnits, ElementsView, ensure_gpu_core,
};
use crate::precipitation::{BlockerGpu, PrecipitationBlocker, nearest_blockers};
use crate::query::EnvironmentSnapshot;
use crate::terrain::{TerrainGpu, TerrainGrid, TerrainState};
pub use model::{MAX_WATER_SOURCES, evaporation_mm_h};
use model::{
    MM_H_TO_M_S, QuantumAccumulator, Scheduler, THERMAL_PERIOD, depth_cap_m, source_entry,
    speed_cap_ms, step_fraction, wet_depth_m,
};

/// Largest water readback edge: a larger simulation grid is box-averaged down by 2 first.
const DEPTH_READBACK_MAX_CELLS: u32 = 1024;

/// What happens at the edge of the terrain.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Reflect)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum HydrologyBoundary {
    /// Water leaves over the edge as a free overfall at the edge cell's own bed level, and
    /// nothing enters. Sediment leaves with it.
    #[default]
    Open,
    /// The edge is a wall: no water or sediment crosses it.
    Closed,
}

/// The water simulation's settings. Insert it before the plugin to override the defaults.
#[derive(Resource, Clone, Debug, PartialEq, Reflect)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct HydrologySettings {
    /// The simulation step, s. Default 0.02.
    pub step_s: f32,
    /// Simulated seconds per real second, independent of the `WorldClock`, `Time<Virtual>` and
    /// pausing. The substeps per frame are capped by the quality level (2, 4, 4 and 16 from
    /// `Low` to `Ultra`, [`ElementsQuality::hydrology_substeps`]), so a large scale makes the
    /// simulation run slower than asked rather than stall the frame. Default 1.
    pub time_scale: f32,
    /// The terrain's edge. Default [`HydrologyBoundary::Open`].
    pub boundary: HydrologyBoundary,
    /// Whether open water evaporates, at [`evaporation_mm_h`]'s rate for the weather. Default
    /// `true`.
    pub evaporation: bool,
    /// Manning's roughness `n` of the ground, s m⁻¹ᐟ³ (natural channels 0.03–0.05). Default
    /// 0.035.
    pub manning_n: f32,
}

impl Default for HydrologySettings {
    fn default() -> Self {
        Self {
            step_s: 0.02,
            time_scale: 1.0,
            boundary: HydrologyBoundary::Open,
            evaporation: true,
            manning_n: 0.035,
        }
    }
}

/// Hydraulic and thermal erosion. The water flows whether or not erosion is enabled.
#[derive(Resource, Clone, Debug, PartialEq, Reflect)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ErosionSettings {
    /// Whether the flow dissolves, carries and deposits sediment and the slopes relax. Default
    /// `true`.
    pub enabled: bool,
    /// Capacity constant `K_c`, s: the water can carry `K_c · max(sin α, min_tilt) · |v|` metres
    /// of solids per cell area. Default 0.6.
    pub capacity: f32,
    /// Dissolve fraction `K_s` per 20 ms step of the difference to capacity; scaled by
    /// `step_s / 0.02` and capped at 1 per step. Default 0.03.
    pub dissolve: f32,
    /// Deposit fraction `K_d`, as `dissolve`. Default 0.03.
    pub deposit: f32,
    /// The smallest slope sine used for the capacity, so flat water still carries a little.
    /// Default 0.05 (2.9°).
    pub min_tilt: f32,
    /// The angle of repose, degrees: thermal erosion moves loose sediment from slopes steeper
    /// than this. Bedrock moves only above this plus 10°, and only during an [`ErodeTerrain`]
    /// bake: a rock face stands at any angle on game time. Default 35.
    pub talus_deg: f32,
    /// The fraction of the largest stable transfer that one thermal pass (every 4th substep)
    /// moves. Default 0.15.
    pub thermal_rate: f32,
    /// Multiplies `dissolve`, `deposit` and `thermal_rate` (each capped at 1). Default 1.
    pub strength: f32,
}

/// Present while a voxel ground owns the terrain (`VoxelGround`, feature `isomesh`): erosion is
/// off whatever [`ErosionSettings::enabled`] says, because the volume's field is the truth and
/// erosion would rewrite only the heightfield under it.
#[derive(Resource, Clone, Copy, Debug, Default)]
#[cfg_attr(not(feature = "isomesh"), allow(dead_code))]
pub(crate) struct ErosionLocked;

/// Whether the hydrology kernels erode: the setting, unless a voxel ground locks it off.
pub(crate) fn erosion_enabled(erosion: &ErosionSettings, locked: bool) -> bool {
    erosion.enabled && !locked
}

impl Default for ErosionSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            capacity: 0.6,
            dissolve: 0.03,
            deposit: 0.03,
            min_tilt: 0.05,
            talus_deg: 35.0,
            thermal_rate: 0.15,
            strength: 1.0,
        }
    }
}

/// Pours water into the terrain: `rate_m3_s` spread evenly over the cells within `radius_m` of
/// the entity's position (at least the cell under it). A source off the terrain does nothing. At
/// most [`crate::MAX_WATER_SOURCES`] sources and as many [`WaterSink`]s are used (lowest
/// `Entity` first).
#[derive(Component, Clone, Copy, Debug, PartialEq, Reflect)]
#[require(Transform)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct WaterSource {
    /// Volume per second, m³/s.
    pub rate_m3_s: f32,
    /// Radius of the footprint, m.
    pub radius_m: f32,
}

/// Drains water from the terrain: up to `rate_m3_s` from the cells within `radius_m` of the
/// entity's position, never more than is there. See [`WaterSource`].
#[derive(Component, Clone, Copy, Debug, PartialEq, Reflect)]
#[require(Transform)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct WaterSink {
    /// Volume per second, m³/s.
    pub rate_m3_s: f32,
    /// Radius of the footprint, m.
    pub radius_m: f32,
}

/// Start an erosion bake: hydrology runs at the quality's full substep rate with `rain_mm_h` of
/// uniform liquid rain (the weather's rain is ignored) until `duration_s` of simulated time has
/// passed, then [`TerrainBaked`] is written. A bake requested during a bake starts when the
/// first one ends; a bake interrupted by the terrain being rebuilt or removed restarts in full
/// on the next terrain.
#[derive(Message, Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ErodeTerrain {
    /// Simulated time to bake, s.
    pub duration_s: f32,
    /// Uniform rain over the whole terrain, mm/h.
    pub rain_mm_h: f32,
}

/// A bake finished: its `duration_s` of simulated time has run.
#[derive(Message, Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TerrainBaked {
    /// The simulated time that was baked, s.
    pub duration_s: f32,
}

/// What the simulation has done, for tests and examples that wait on simulated time. Reset when
/// the terrain is rebuilt.
#[derive(Resource, Clone, Debug, Default, PartialEq)]
pub struct HydrologyStats {
    /// Substeps the GPU has been asked to run and has recorded.
    pub substeps: u64,
    /// `substeps · step_s`, s.
    pub simulated_s: f64,
    /// Water on the terrain at the last depth readback, m³.
    pub volume_m3: f64,
    /// An estimate of the water that left through an open boundary or a sink, m³: the volume at
    /// the first readback plus the nominal input since (rain over the whole extent at the
    /// reference elevation's phase, sources at their rate, sinks at their rate, evaporation over
    /// the cells that were wet at the last readback) minus `volume_m3`. It is up to one readback
    /// interval old and does not know the per-cell snow split or the blockers' shade.
    pub lost_m3: f64,
    /// A bake is running.
    pub baking: bool,
    /// Substeps that were scheduled but could not be recorded (a pipeline or buffer was not
    /// ready on the render side that frame). Zero in normal running; their rain and evaporation
    /// are lost with them.
    pub dropped: u64,
}

/// The depth readback: `Environment::water_depth` answers from it.
#[derive(Clone, Debug)]
pub(crate) struct WaterDepthSnapshot {
    /// The readback grid (the simulation grid box-averaged by 2 above 1024²).
    pub grid: TerrainGrid,
    pub depth: Arc<[f32]>,
    /// `Σ depth · cell²` over the readback, m³.
    pub volume_m3: f64,
    /// Cells deeper than the wet threshold.
    pub wet_cells: u32,
    /// The request this was read for, and the nominal input at that request.
    pub request: u64,
    pub input_m3: f64,
}

/// The velocity readback at half resolution, depth-weighted.
#[derive(Clone, Debug)]
pub(crate) struct WaterVelocitySnapshot {
    pub grid: TerrainGrid,
    pub u: Arc<[f32]>,
    pub v: Arc<[f32]>,
}

impl WaterDepthSnapshot {
    pub(crate) fn depth_at(&self, p: Vec2) -> Option<f32> {
        self.grid
            .contains(p)
            .then(|| self.grid.sample(&self.depth, p).max(0.0))
    }
}

impl WaterVelocitySnapshot {
    pub(crate) fn velocity_at(&self, p: Vec2) -> Option<Vec2> {
        self.grid
            .contains(p)
            .then(|| Vec2::new(self.grid.sample(&self.u, p), self.grid.sample(&self.v, p)))
    }
}

/// Uniform of the hydrology kernels. Mirrors `HydroParams` of `substep.wesl`.
#[derive(Clone, Copy, Debug, Default, PartialEq, ShaderType)]
pub(crate) struct HydroParams {
    origin: Vec2,
    cell: f32,
    pad0: f32,
    size: UVec2,
    pad1: UVec2,
    /// x: Δt. y: Manning n. z: pipe depth cap, m. w: wet threshold, m.
    sim: Vec4,
    /// x: rain quantum, m. y: evaporation quantum, m. z: terrain y0 − reference elevation, m.
    /// w: air temperature at the reference elevation, °C.
    weather: Vec4,
    /// x: lapse, °C/m. y: speed clamp, m/s. z: 1 in a bake. w: `Weather::snow_possible`.
    misc: Vec4,
    /// x: closed boundary. y: erosion on. z: sources uploaded. w: blockers uploaded.
    flags: UVec4,
    /// x: K_c. y: min tilt. z: dissolve per step. w: deposit per step.
    erosion: Vec4,
    /// x: tan talus. y: tan (talus + 10°). z: thermal fraction per pass.
    thermal: Vec4,
    /// x: terrain y0. y: depth readback downsample.
    extra: Vec4,
}

impl HydroParams {
    /// The uniform of a thermal pass that is not hydrology's: the grid, the talus and the
    /// thermal fraction. Bedrock moves only above `talus_deg + bedrock_extra_deg`, and an
    /// excess that reaches 90° pins the threshold out of reach (no tangent), so the pass moves
    /// loose sediment only.
    pub(crate) fn avalanche(
        grid: &TerrainGrid,
        talus_deg: f32,
        bedrock_extra_deg: f32,
        rate: f32,
    ) -> Self {
        let bedrock_angle = talus_deg + bedrock_extra_deg;
        let tan_bedrock = if bedrock_angle >= 89.9 {
            model::UNREACHABLE_TAN
        } else {
            bedrock_angle.to_radians().tan()
        };
        Self {
            origin: grid.origin,
            cell: grid.cell_m,
            size: grid.size,
            thermal: Vec4::new(
                talus_deg.to_radians().tan(),
                tan_bedrock,
                rate.clamp(0.0, 1.0),
                0.0,
            ),
            ..Self::default()
        }
    }
}

/// The readback targets of the active terrain.
#[derive(Clone, Debug)]
pub(crate) struct ReadbackBuffers {
    /// Box-averaged water depth, `array<f32>`.
    pub depth: Handle<ShaderBuffer>,
    /// Depth-weighted velocity at half resolution, `array<vec2<f32>>`.
    pub velocity: Handle<ShaderBuffer>,
    /// Depth readback downsample factor.
    pub depth_ds: u32,
}

/// This frame's work and the GPU's progress; extracted to the render world every frame.
#[derive(Resource, Clone)]
pub(crate) struct HydrologyGpu {
    pub buffers: ReadbackBuffers,
    pub params: HydroParams,
    /// Substeps to run this frame.
    pub substeps: u32,
    /// Substeps run since the start (so the thermal passes keep their phase across frames).
    pub thermal_phase: u32,
    pub thermal: bool,
    /// Source entries (xy centre, z radius, w depth rate) and blockers for `source_map_build`.
    pub sources: Vec<Vec4>,
    pub blockers: Vec<BlockerGpu>,
    /// The source map is current when `source_built` has caught up with this.
    pub source_epoch: u64,
    pub source_built: Arc<AtomicU64>,
    /// Run the readback kernels this frame.
    pub downsample: bool,
    /// Substeps recorded, and substeps asked for that could not be.
    pub executed: Arc<AtomicU64>,
    pub dropped: Arc<AtomicU64>,
    /// The render world has every pipeline and buffer it needs.
    pub ready: Arc<AtomicBool>,
    /// A first dispatch has run: the buffers hold simulation state.
    pub live: Arc<AtomicBool>,
}

/// A running bake.
#[derive(Clone, Copy, Debug)]
struct Bake {
    request: ErodeTerrain,
    /// Substeps not yet scheduled.
    remaining: u64,
    /// `executed + dropped` at which the last substep has been recorded.
    done_at: u64,
}

/// What the sources, sinks and blockers looked like when their footprints were last rasterised.
#[derive(Clone, Debug, Default, PartialEq)]
struct SourceInputs {
    grid: Option<(Vec2, f32, UVec2)>,
    /// (world XZ, radius, signed rate) of every source and sink.
    entries: Vec<(Vec2, f32, f32)>,
}

/// Main-world bookkeeping of the simulation.
#[derive(Resource, Default)]
struct HydrologyRuntime {
    /// The water buffer the current GPU resources were built for.
    built_for: Option<AssetId<ShaderBuffer>>,
    scheduler: Scheduler,
    rain: QuantumAccumulator,
    evaporation: QuantumAccumulator,
    bake: Option<Bake>,
    queue: VecDeque<ErodeTerrain>,
    /// Substeps scheduled since the terrain was built.
    scheduled: u64,
    source_epoch: u64,
    inputs: SourceInputs,
    sources: Vec<Vec4>,
    blockers: Vec<BlockerGpu>,
    warned_cap: bool,
    /// Nominal water input since the start, m³ (see `HydrologyStats::lost_m3`).
    input_m3: f64,
    /// The readback requests issued, and the first processed one's volume and input.
    requests: u64,
    processed: u64,
    baseline: Option<(f64, f64)>,
    /// Wet cells at the last readback.
    wet_area_m2: f64,
}

impl HydrologyRuntime {
    /// The bake requests to keep across a rebuild: the running bake first (it restarts in full on
    /// the new layers), then the queue.
    fn take_requests(&mut self) -> VecDeque<ErodeTerrain> {
        let mut queue = std::mem::take(&mut self.queue);
        if let Some(bake) = self.bake.take() {
            queue.push_front(bake.request);
        }
        queue
    }
}

/// Hydrology, erosion and the water readback.
///
/// Needs a [`Terrain`](crate::Terrain); idles without one. Inserts, if absent,
/// [`HydrologySettings`], [`ErosionSettings`] and [`HydrologyStats`].
pub struct HydrologyPlugin;

impl Plugin for HydrologyPlugin {
    fn build(&self, app: &mut App) {
        ensure_gpu_core(app);
        embedded_asset!(app, "substep.wesl");
        embedded_asset!(app, "thermal.wesl");
        embedded_asset!(app, "source_map.wesl");
        embedded_asset!(app, "readback.wesl");
        app.init_resource::<HydrologySettings>()
            .init_resource::<ErosionSettings>()
            .init_resource::<HydrologyStats>()
            .init_resource::<Climate>()
            .init_resource::<HydrologyRuntime>()
            .add_message::<ErodeTerrain>()
            .add_message::<TerrainBaked>()
            .add_systems(
                Update,
                (ensure_hydrology, run_hydrology)
                    .chain()
                    .in_set(ElementsSystems::Simulation),
            )
            .add_systems(
                Update,
                (read_water, update_stats)
                    .chain()
                    .in_set(ElementsSystems::Readback),
            );
        gpu::plugin(app);
    }
}

/// Readback sizes of a grid: the depth downsample factor.
fn depth_downsample(size: UVec2) -> u32 {
    if size.max_element() > DEPTH_READBACK_MAX_CELLS {
        2
    } else {
        1
    }
}

/// Create the readback targets when a terrain appears or is rebuilt, and drop everything when it
/// goes.
#[allow(clippy::too_many_arguments)]
fn ensure_hydrology(
    mut commands: Commands,
    terrain: Option<Res<TerrainState>>,
    gpu: Option<Res<HydrologyGpu>>,
    mut runtime: ResMut<HydrologyRuntime>,
    mut stats: ResMut<HydrologyStats>,
    mut snapshot: ResMut<EnvironmentSnapshot>,
    mut buffers: ResMut<Assets<ShaderBuffer>>,
) {
    let Some(terrain) = terrain else {
        if gpu.is_some() {
            commands.remove_resource::<HydrologyGpu>();
            // A bake asked for before the terrain exists (or while it is rebuilt) waits for it.
            let queue = runtime.take_requests();
            *runtime = HydrologyRuntime {
                queue,
                ..HydrologyRuntime::default()
            };
            *stats = HydrologyStats::default();
            snapshot.water_depth = None;
            snapshot.water_velocity = None;
        }
        return;
    };
    if runtime.built_for == Some(terrain.water.id()) && gpu.is_some() {
        return;
    }
    let size = terrain.grid.size;
    let ds = depth_downsample(size);
    let depth_cells = u64::from(size.x / ds) * u64::from(size.y / ds);
    let velocity_cells = u64::from(size.x / 2) * u64::from(size.y / 2);
    let readback = ReadbackBuffers {
        depth: zeroed_buffer(
            &mut buffers,
            "elements_hydrology_depth_readback",
            depth_cells * 4,
        ),
        velocity: zeroed_buffer(
            &mut buffers,
            "elements_hydrology_velocity_readback",
            velocity_cells * 8,
        ),
        depth_ds: ds,
    };
    *runtime = HydrologyRuntime {
        built_for: Some(terrain.water.id()),
        source_epoch: 1,
        queue: runtime.take_requests(),
        ..HydrologyRuntime::default()
    };
    *stats = HydrologyStats::default();
    snapshot.water_depth = None;
    snapshot.water_velocity = None;
    commands.insert_resource(HydrologyGpu {
        buffers: readback,
        params: HydroParams::default(),
        substeps: 0,
        thermal_phase: 0,
        thermal: false,
        sources: Vec::new(),
        blockers: Vec::new(),
        source_epoch: 1,
        source_built: Arc::new(AtomicU64::new(0)),
        downsample: false,
        executed: Arc::new(AtomicU64::new(0)),
        dropped: Arc::new(AtomicU64::new(0)),
        ready: Arc::new(AtomicBool::new(false)),
        live: Arc::new(AtomicBool::new(false)),
    });
}

/// The settings and environment the schedule reads.
#[derive(SystemParam)]
struct Inputs<'w> {
    settings: Res<'w, HydrologySettings>,
    erosion: Res<'w, ErosionSettings>,
    quality: Res<'w, ElementsQuality>,
    weather: Res<'w, Weather>,
    climate: Res<'w, Climate>,
    view: Res<'w, ElementsView>,
    time: Res<'w, Time<Real>>,
    units: Res<'w, ElementsUnits>,
    erosion_locked: Option<Res<'w, ErosionLocked>>,
}

/// The raw descriptions of the sources and sinks (positions in the metre frame, from their world
/// translations through `units`), in a deterministic order, and whether the cap cut any off.
fn collect_sources(
    units: &ElementsUnits,
    sources: &Query<(Entity, &WaterSource, &GlobalTransform)>,
    sinks: &Query<(Entity, &WaterSink, &GlobalTransform)>,
) -> (Vec<(Vec2, f32, f32)>, bool) {
    fn take<'a>(
        items: impl Iterator<Item = (Entity, Vec2, f32, f32)> + 'a,
    ) -> (Vec<(Vec2, f32, f32)>, bool) {
        let mut all: Vec<_> = items.collect();
        all.sort_by_key(|item| item.0);
        let cut = all.len() > MAX_WATER_SOURCES;
        all.truncate(MAX_WATER_SOURCES);
        (all.into_iter().map(|(_, p, r, q)| (p, r, q)).collect(), cut)
    }
    let (mut entries, cut_sources) = take(sources.iter().map(|(e, s, t)| {
        (
            e,
            units.point_to_metres(t.translation()).xz(),
            s.radius_m,
            s.rate_m3_s,
        )
    }));
    let (sink_entries, cut_sinks) = take(sinks.iter().map(|(e, s, t)| {
        (
            e,
            units.point_to_metres(t.translation()).xz(),
            s.radius_m,
            -s.rate_m3_s,
        )
    }));
    entries.extend(sink_entries);
    (entries, cut_sources || cut_sinks)
}

/// Schedule this frame's substeps and fill the uniform, the sources and the bake bookkeeping.
#[allow(clippy::too_many_arguments)]
fn run_hydrology(
    inputs: Inputs,
    terrain: Option<Res<TerrainState>>,
    terrain_gpu: Option<ResMut<TerrainGpu>>,
    gpu: Option<ResMut<HydrologyGpu>>,
    mut runtime: ResMut<HydrologyRuntime>,
    mut stats: ResMut<HydrologyStats>,
    sources: Query<(Entity, &WaterSource, &GlobalTransform)>,
    sinks: Query<(Entity, &WaterSink, &GlobalTransform)>,
    blockers: Query<(Entity, &PrecipitationBlocker, &GlobalTransform)>,
    tick: Res<ReadbackTick>,
    mut requests: MessageReader<ErodeTerrain>,
    mut baked: MessageWriter<TerrainBaked>,
) {
    runtime.queue.extend(requests.read().copied());
    let (Some(terrain), Some(mut terrain_gpu), Some(mut gpu)) = (terrain, terrain_gpu, gpu) else {
        return;
    };
    let Inputs {
        settings,
        erosion,
        quality,
        weather,
        climate,
        view,
        time,
        erosion_locked,
        units,
    } = inputs;
    let erosion_on = erosion_enabled(&erosion, erosion_locked.is_some());
    let grid = terrain.grid;
    let step_s = settings.step_s.max(1e-4);

    let executed = gpu.executed.load(Ordering::Relaxed);
    let dropped = gpu.dropped.load(Ordering::Relaxed);
    stats.substeps = executed;
    stats.simulated_s = executed as f64 * f64::from(step_s);
    stats.dropped = dropped;

    // Finish a bake whose last substep has been recorded; start the next one.
    if let Some(bake) = runtime.bake
        && bake.remaining == 0
        && executed + dropped >= bake.done_at
    {
        baked.write(TerrainBaked {
            duration_s: bake.request.duration_s,
        });
        runtime.bake = None;
        runtime.scheduler.reset();
    }
    if runtime.bake.is_none()
        && let Some(request) = runtime.queue.pop_front()
    {
        // `step_s` is an f32, so 600 s / 0.02 s is 30 000.0007: allow a thousandth of a step.
        let steps = f64::from(request.duration_s.max(0.0)) / f64::from(step_s);
        let total = (steps - 1e-3).ceil().max(0.0) as u64;
        let done_at = runtime.scheduled + total;
        if total == 0 {
            baked.write(TerrainBaked {
                duration_s: request.duration_s,
            });
        } else {
            runtime.bake = Some(Bake {
                request,
                remaining: total,
                done_at,
            });
        }
    }
    stats.baking = runtime.bake.is_some();

    // Footprints of the sources and sinks, and the blockers; the map is rebuilt when they change.
    let (entries, cut) = collect_sources(&units, &sources, &sinks);
    if cut && !runtime.warned_cap {
        warn!(
            "more than {MAX_WATER_SOURCES} WaterSources or WaterSinks exist; the extra ones (highest Entity) are ignored"
        );
        runtime.warned_cap = true;
    }
    let source_inputs = SourceInputs {
        grid: Some((grid.origin, grid.cell_m, grid.size)),
        entries,
    };
    if runtime.inputs != source_inputs {
        runtime.sources = source_inputs
            .entries
            .iter()
            .filter_map(|&(p, radius, rate)| {
                source_entry(grid.origin, grid.cell_m, grid.size, p, radius, rate)
            })
            .collect();
        runtime.inputs = source_inputs;
        runtime.source_epoch += 1;
    }
    let selected = nearest_blockers(
        view.position,
        blockers.iter().filter_map(|(entity, blocker, transform)| {
            BlockerGpu::new(blocker.half_extents, &units.to_metres(transform)).map(|b| (entity, b))
        }),
    );
    if runtime.blockers != selected {
        runtime.blockers = selected;
        runtime.source_epoch += 1;
    }

    // This frame's substeps: the bake's, or the real-time schedule's.
    let ready = gpu.ready.load(Ordering::Relaxed)
        && terrain.is_live()
        && terrain_gpu.generated.load(Ordering::Relaxed);
    let max = quality.hydrology_substeps();
    let n = if !ready {
        0
    } else if let Some(bake) = runtime.bake.as_mut() {
        let n = bake.remaining.min(u64::from(max));
        bake.remaining -= n;
        n as u32
    } else {
        let dt = time.delta_secs_f64();
        runtime.scheduler.step(dt, settings.time_scale, step_s, max)
    };

    let baking = runtime.bake.is_some();
    let rain_mm_h = match runtime.bake {
        Some(bake) => bake.request.rain_mm_h,
        None => weather.precipitation_mm_h,
    };
    let (rain_quantum, evaporation_quantum) = if n > 0 {
        let rain_rate = f64::from(rain_mm_h.max(0.0)) * MM_H_TO_M_S;
        let evap_rate = if settings.evaporation {
            f64::from(evaporation_mm_h(
                weather.wind_speed_ms,
                weather.temperature_c,
                weather.relative_humidity,
            )) * MM_H_TO_M_S
        } else {
            0.0
        };
        let quanta = (runtime.rain.take(), runtime.evaporation.take());
        runtime.rain.owe(rain_rate, step_s, n);
        runtime.evaporation.owe(evap_rate, step_s, n);
        // The nominal water input of these substeps (see `HydrologyStats::lost_m3`).
        let seconds = f64::from(step_s) * f64::from(n);
        let area = f64::from(grid.extent().x) * f64::from(grid.extent().y);
        let liquid = if baking {
            1.0
        } else {
            f64::from(1.0 - weather.snow_fraction())
        };
        let source_rate: f64 = runtime.inputs.entries.iter().map(|e| f64::from(e.2)).sum();
        runtime.input_m3 +=
            seconds * (rain_rate * liquid * area - evap_rate * runtime.wet_area_m2 + source_rate);
        quanta
    } else {
        (0.0, 0.0)
    };

    let cell = grid.cell_m;
    let tan_talus = erosion.talus_deg.to_radians().tan();
    // Rock faces collapse on geological time, so only a bake moves bedrock (erosion.md
    // Deviation 15); on game time thermal erosion is the slumping of loose sediment.
    let tan_bedrock = if runtime.bake.is_some() {
        (erosion.talus_deg + model::BEDROCK_EXTRA_DEG)
            .to_radians()
            .tan()
    } else {
        model::UNREACHABLE_TAN
    };
    gpu.params = HydroParams {
        origin: grid.origin,
        cell,
        pad0: 0.0,
        size: grid.size,
        pad1: UVec2::ZERO,
        sim: Vec4::new(
            step_s,
            settings.manning_n.max(0.0),
            depth_cap_m(cell, step_s),
            wet_depth_m(cell),
        ),
        weather: Vec4::new(
            rain_quantum,
            evaporation_quantum,
            grid.y0 - climate.reference_elevation_m,
            weather.temperature_c,
        ),
        misc: Vec4::new(
            LAPSE_RATE_C_PER_M,
            speed_cap_ms(cell, step_s),
            f32::from(u8::from(runtime.bake.is_some())),
            weather.snow_possible,
        ),
        flags: UVec4::new(
            u32::from(settings.boundary == HydrologyBoundary::Closed),
            u32::from(erosion_on),
            runtime.sources.len() as u32,
            runtime.blockers.len() as u32,
        ),
        erosion: Vec4::new(
            erosion.capacity,
            erosion.min_tilt,
            step_fraction(erosion.dissolve, erosion.strength, step_s),
            step_fraction(erosion.deposit, erosion.strength, step_s),
        ),
        thermal: Vec4::new(
            tan_talus,
            tan_bedrock,
            (erosion.thermal_rate * erosion.strength).clamp(0.0, 1.0),
            0.0,
        ),
        extra: Vec4::new(grid.y0, gpu.buffers.depth_ds as f32, 0.0, 0.0),
    };
    gpu.substeps = n;
    gpu.thermal_phase = (runtime.scheduled % THERMAL_PERIOD) as u32;
    gpu.thermal = erosion_on;
    // Never an empty binding: the counts say how many entries are real.
    gpu.sources = if runtime.sources.is_empty() {
        vec![Vec4::ZERO]
    } else {
        runtime.sources.clone()
    };
    gpu.blockers = if runtime.blockers.is_empty() {
        vec![BlockerGpu::default()]
    } else {
        runtime.blockers.clone()
    };
    gpu.source_epoch = runtime.source_epoch;
    // Until a readback kernel has run once the readback buffers hold nothing: keep running them.
    gpu.downsample = tick.is_due() || !gpu.live.load(Ordering::Relaxed);
    runtime.scheduled += u64::from(n);
    if n > 0 {
        // The pack, bounds and height readback run once this frame, after the substeps.
        terrain_gpu.changes += 1;
    }
}

/// Request the water readbacks once the kernels have produced state.
fn read_water(
    terrain: Option<Res<TerrainState>>,
    gpu: Option<Res<HydrologyGpu>>,
    mut runtime: ResMut<HydrologyRuntime>,
    tick: Res<ReadbackTick>,
    mut readbacks: Readbacks,
) {
    let (Some(terrain), Some(gpu)) = (terrain, gpu) else {
        return;
    };
    if !gpu.live.load(Ordering::Relaxed) || !tick.is_due() {
        return;
    }
    let ds = gpu.buffers.depth_ds;
    let grid = terrain.grid;
    let wet = wet_depth_m(grid.cell_m);
    let depth_grid = TerrainGrid {
        cell_m: grid.cell_m * ds as f32,
        size: grid.size / ds,
        ..grid
    };
    let expected = (depth_grid.size.x * depth_grid.size.y) as usize;
    runtime.requests += 1;
    let (request, input_m3) = (runtime.requests, runtime.input_m3);
    readbacks.request(
        "water_depth",
        gpu.buffers.depth.clone(),
        move |bytes, snapshot| {
            let Some(depth) = decode::<f32>(bytes) else {
                return;
            };
            if depth.len() < expected {
                return;
            }
            let depth = &depth[..expected];
            let area = f64::from(depth_grid.cell_m) * f64::from(depth_grid.cell_m);
            snapshot.water_depth = Some(WaterDepthSnapshot {
                grid: depth_grid,
                volume_m3: depth.iter().map(|d| f64::from(*d)).sum::<f64>() * area,
                wet_cells: depth.iter().filter(|d| **d > wet).count() as u32,
                depth: depth.into(),
                request,
                input_m3,
            });
        },
    );
    let velocity_grid = TerrainGrid {
        cell_m: grid.cell_m * 2.0,
        size: grid.size / 2,
        ..grid
    };
    let expected = (velocity_grid.size.x * velocity_grid.size.y) as usize;
    readbacks.request(
        "water_velocity",
        gpu.buffers.velocity.clone(),
        move |bytes, snapshot| {
            let Some(velocity) = decode::<[f32; 2]>(bytes) else {
                return;
            };
            if velocity.len() < expected {
                return;
            }
            let velocity = &velocity[..expected];
            snapshot.water_velocity = Some(WaterVelocitySnapshot {
                grid: velocity_grid,
                u: velocity.iter().map(|v| v[0]).collect(),
                v: velocity.iter().map(|v| v[1]).collect(),
            });
        },
    );
}

/// Fold a new depth readback into [`HydrologyStats`].
fn update_stats(
    snapshot: Res<EnvironmentSnapshot>,
    mut runtime: ResMut<HydrologyRuntime>,
    mut stats: ResMut<HydrologyStats>,
) {
    let Some(water) = snapshot.water_depth.as_ref() else {
        return;
    };
    if water.request == runtime.processed {
        return;
    }
    runtime.processed = water.request;
    let (volume0, input0) = *runtime
        .baseline
        .get_or_insert((water.volume_m3, water.input_m3));
    let cell = f64::from(water.grid.cell_m);
    runtime.wet_area_m2 = f64::from(water.wet_cells) * cell * cell;
    stats.volume_m3 = water.volume_m3;
    stats.lost_m3 = volume0 + (water.input_m3 - input0) - water.volume_m3;
}

#[cfg(test)]
mod tests {
    use bevy::ecs::system::RunSystemOnce;

    use super::*;

    #[test]
    fn an_interrupted_bake_restarts_before_the_queue() {
        let a = ErodeTerrain {
            duration_s: 60.0,
            rain_mm_h: 10.0,
        };
        let b = ErodeTerrain {
            duration_s: 30.0,
            rain_mm_h: 5.0,
        };
        let mut runtime = HydrologyRuntime {
            bake: Some(Bake {
                request: a,
                remaining: 5,
                done_at: 9,
            }),
            queue: VecDeque::from([b]),
            ..default()
        };
        assert_eq!(runtime.take_requests(), VecDeque::from([a, b]));
        assert!(runtime.bake.is_none());
        assert!(runtime.queue.is_empty());
    }

    /// A source and a sink placed in feet land at the metre-frame cell the same scene in metres
    /// has them in; their radius and rate stay in m and m³/s.
    #[test]
    fn a_source_in_feet_is_read_at_its_position_in_metres() {
        let mut world = World::new();
        world.insert_resource(ElementsUnits {
            units_per_metre: 3.2808,
        });
        let at = |x: f32, z: f32| GlobalTransform::from_xyz(x * 3.2808, 7.0, z * 3.2808);
        world.spawn((
            WaterSource {
                rate_m3_s: 0.5,
                radius_m: 2.0,
            },
            at(10.0, 20.0),
        ));
        world.spawn((
            WaterSink {
                rate_m3_s: 0.25,
                radius_m: 1.0,
            },
            at(-4.0, 6.0),
        ));
        let (entries, cut) = world
            .run_system_once(
                |units: Res<ElementsUnits>,
                 sources: Query<(Entity, &WaterSource, &GlobalTransform)>,
                 sinks: Query<(Entity, &WaterSink, &GlobalTransform)>| {
                    collect_sources(&units, &sources, &sinks)
                },
            )
            .expect("the system runs");
        assert!(!cut);
        assert_eq!(entries.len(), 2);
        let (source, sink) = (entries[0], entries[1]);
        assert!(
            source.0.distance(Vec2::new(10.0, 20.0)) < 1e-3,
            "{source:?}"
        );
        assert_eq!((source.1, source.2), (2.0, 0.5));
        assert!(sink.0.distance(Vec2::new(-4.0, 6.0)) < 1e-3, "{sink:?}");
        assert_eq!((sink.1, sink.2), (1.0, -0.25));
    }
}
