//! Dust storms: wind-blown sand. Where the wind at the ground exceeds a threshold it picks up the
//! terrain's loose `sediment` layer, carries it downwind and drops it where the wind can no longer
//! carry it (the lee of every rise), so dunes migrate; steep slopes then avalanche at the sand's
//! angle of repose. The dust that hangs in the air is drawn by the precipitation particles
//! (`kind = 2`) and the ground fog. See `docs/research/dust.md`.
//!
//! The transport (`aeolian.wesl`: a capacity kernel, a flux scan along the wind and a bed update) runs outside hydrology's substep loop, at most
//! [`DustSettings::step_hz`] times a second of real time and only while the wind can exceed the
//! threshold, on the terrain's `sediment` layer; hydrology's thermal kernels do the avalanching.
//! The bed advances by the virtual time since the last step times
//! [`WorldClock::time_scale`](crate::WorldClock): dune evolution follows game time. (A paused
//! `WorldClock` freezes the sky, not the sand; pause `Time<Virtual>` to stop both.) Needs a
//! [`Terrain`](crate::Terrain) and [`HydrologyPlugin`](crate::HydrologyPlugin) (the thermal
//! kernels and their scratch buffers); idles without either. The edge is
//! [`HydrologySettings::boundary`](crate::HydrologySettings).

mod gpu;
#[cfg(test)]
mod gpu_tests;
pub(crate) mod model;

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use bevy::asset::embedded_asset;
use bevy::prelude::*;
use bevy::render::render_resource::ShaderType;
use bevy::render::storage::ShaderBuffer;
use bevy::shader::load_shader_library;

use crate::celestial::WorldClock;
use crate::climate::Weather;
use crate::gpu::grid::zeroed_buffer;
use crate::hydrology::{HydroParams, HydrologyBoundary, HydrologySettings};
use crate::plugin::{ElementsSystems, ensure_gpu_core};
use crate::terrain::{TerrainGpu, TerrainState};
use model::{StepClock, can_transport};

/// The aeolian transport's settings. Insert it before the plugin to override the defaults.
#[derive(Resource, Clone, Debug, PartialEq, Reflect)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DustSettings {
    /// Whether the wind moves sand. Default `true`.
    pub enabled: bool,
    /// The 10 m wind below which sand does not move, m/s (plan 4.3; Bagnold's fluid threshold
    /// of 250 µm sand is 8.14). Default 8.
    pub threshold_ms: f32,
    /// The flux coefficient `k` of `q = k u² (u − u_t)`, m² s⁻¹ (m/s)⁻³: bulk sand volume per
    /// unit width. The plan's 2·10⁻⁶ is 167× the physical flux, a morphological acceleration
    /// (`docs/research/dust.md` Deviation 1). Default 2·10⁻⁶.
    pub flux_coefficient: f32,
    /// The distance over which the flux relaxes toward saturation, m (Kok et al. 2012: 1–20).
    /// Default 10.
    pub saturation_length_m: f32,
    /// The angle of repose the avalanching holds, degrees. Default 34.
    pub talus_deg: f32,
    /// Avalanche passes per transport step (hydrology's thermal kernels). Default 2.
    pub avalanche_passes: u32,
    /// Most transport steps per second of real time. Default 20.
    pub step_hz: f32,
}

impl Default for DustSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            threshold_ms: model::THRESHOLD_MS,
            flux_coefficient: model::FLUX_COEFFICIENT,
            saturation_length_m: model::SATURATION_LENGTH_M,
            talus_deg: model::TALUS_DEG,
            avalanche_passes: model::AVALANCHE_PASSES,
            step_hz: model::STEP_HZ,
        }
    }
}

/// What the transport has done, for tests and examples that wait on it. Reset when the terrain
/// is rebuilt.
#[derive(Resource, Clone, Debug, Default, PartialEq)]
pub struct DustStats {
    /// Transport steps the GPU has recorded.
    pub steps: u64,
    /// Bed time the steps covered, s (real time × the clock's time scale).
    pub bed_time_s: f64,
}

/// Uniform of `aeolian.wesl`.
#[derive(Clone, Copy, Debug, Default, PartialEq, ShaderType)]
pub(crate) struct AeolianParams {
    origin: Vec2,
    cell: f32,
    pad0: f32,
    size: UVec2,
    pad1: UVec2,
    /// xy: unit wind direction (x, z). z: the 10 m wind, m/s. w: 1 when the boundary is closed.
    wind: Vec4,
    /// x: flux coefficient. y: transport threshold, m/s.
    law: Vec4,
    /// x: saturation length, m. y: the bed's time step, s.
    step: Vec4,
}

/// The scratch of the transport: the saturation flux and the face fluxes along x and z.
#[derive(Clone, Debug)]
pub(crate) struct AeolianBuffers {
    /// `aeolian_cap`, `array<f32>`, m²/s.
    pub cap: Handle<ShaderBuffer>,
    /// `aeolian_q_x` and `aeolian_q_z`, the flux through each cell's downwind x and z face,
    /// `array<f32>`, m²/s.
    pub q: [Handle<ShaderBuffer>; 2],
}

/// This frame's work and the GPU's progress; extracted to the render world every frame.
#[derive(Resource, Clone)]
pub(crate) struct DustGpu {
    pub buffers: AeolianBuffers,
    pub params: AeolianParams,
    /// The uniform of the avalanche passes (hydrology's thermal kernels).
    pub avalanche: HydroParams,
    pub avalanche_passes: u32,
    /// Run a transport step this frame.
    pub run: bool,
    /// Steps recorded.
    pub executed: Arc<AtomicU64>,
    /// The render world has every pipeline and buffer it needs.
    pub ready: Arc<AtomicBool>,
}

/// Main-world bookkeeping.
#[derive(Resource, Default)]
struct DustRuntime {
    /// The water buffer the current GPU resources were built for (it identifies the terrain).
    built_for: Option<AssetId<ShaderBuffer>>,
    clock: StepClock,
    scheduled: u64,
    bed_time_s: f64,
}

/// Aeolian transport and the dust particles' and haze's partner on the ground.
///
/// Inserts, if absent, [`DustSettings`] and [`DustStats`].
pub struct DustPlugin;

impl Plugin for DustPlugin {
    fn build(&self, app: &mut App) {
        ensure_gpu_core(app);
        load_shader_library!(app, "surface_wind.wesl");
        embedded_asset!(app, "aeolian.wesl");
        app.init_resource::<DustSettings>()
            .init_resource::<DustStats>()
            .init_resource::<DustRuntime>()
            .add_systems(
                Update,
                (ensure_dust, run_dust)
                    .chain()
                    .in_set(ElementsSystems::Simulation),
            );
        gpu::plugin(app);
    }
}

/// Create the transport's buffers when a terrain appears or is rebuilt, and drop them when it
/// goes.
fn ensure_dust(
    mut commands: Commands,
    terrain: Option<Res<TerrainState>>,
    gpu: Option<Res<DustGpu>>,
    mut runtime: ResMut<DustRuntime>,
    mut stats: ResMut<DustStats>,
    mut buffers: ResMut<Assets<ShaderBuffer>>,
) {
    let Some(terrain) = terrain else {
        if gpu.is_some() {
            commands.remove_resource::<DustGpu>();
            *runtime = DustRuntime::default();
            *stats = DustStats::default();
        }
        return;
    };
    if runtime.built_for == Some(terrain.water.id()) && gpu.is_some() {
        return;
    }
    let cells = u64::from(terrain.grid.size.x) * u64::from(terrain.grid.size.y);
    *runtime = DustRuntime {
        built_for: Some(terrain.water.id()),
        ..DustRuntime::default()
    };
    *stats = DustStats::default();
    commands.insert_resource(DustGpu {
        buffers: AeolianBuffers {
            cap: zeroed_buffer(&mut buffers, "elements_aeolian_cap", cells * 4),
            q: [
                zeroed_buffer(&mut buffers, "elements_aeolian_q_x", cells * 4),
                zeroed_buffer(&mut buffers, "elements_aeolian_q_z", cells * 4),
            ],
        },
        params: AeolianParams::default(),
        avalanche: HydroParams::default(),
        avalanche_passes: 0,
        run: false,
        executed: Arc::new(AtomicU64::new(0)),
        ready: Arc::new(AtomicBool::new(false)),
    });
}

/// Decide this frame's step and fill the uniforms.
#[allow(clippy::too_many_arguments)]
fn run_dust(
    settings: Res<DustSettings>,
    weather: Res<Weather>,
    clock: Res<WorldClock>,
    time: Res<Time<Virtual>>,
    hydrology: Option<Res<HydrologySettings>>,
    terrain: Option<Res<TerrainState>>,
    terrain_gpu: Option<Res<TerrainGpu>>,
    gpu: Option<ResMut<DustGpu>>,
    mut runtime: ResMut<DustRuntime>,
    mut stats: ResMut<DustStats>,
) {
    let (Some(terrain), Some(terrain_gpu), Some(mut gpu)) = (terrain, terrain_gpu, gpu) else {
        return;
    };
    gpu.run = false;
    stats.steps = gpu.executed.load(Ordering::Relaxed);
    let ready = gpu.ready.load(Ordering::Relaxed)
        && terrain.is_live()
        && terrain_gpu.generated.load(Ordering::Relaxed);
    let direction = weather.wind_direction.normalize_or_zero();
    let wind = weather.wind_speed_ms;
    if !ready
        || !settings.enabled
        || direction == Vec2::ZERO
        || !can_transport(wind, settings.threshold_ms)
    {
        runtime.clock.reset();
        return;
    }
    let Some(real_s) = runtime.clock.advance(time.delta_secs(), settings.step_hz) else {
        return;
    };
    let bed_s = real_s * clock.time_scale.clamp(0.0, model::MAX_TIME_SCALE);
    if bed_s <= 0.0 {
        return;
    }
    let grid = terrain.grid;
    let closed = hydrology.is_some_and(|h| h.boundary == HydrologyBoundary::Closed);
    gpu.params = AeolianParams {
        origin: grid.origin,
        cell: grid.cell_m,
        pad0: 0.0,
        size: grid.size,
        pad1: UVec2::ZERO,
        wind: Vec4::new(direction.x, direction.y, wind, f32::from(u8::from(closed))),
        law: Vec4::new(settings.flux_coefficient, settings.threshold_ms, 0.0, 0.0),
        step: Vec4::new(settings.saturation_length_m.max(1e-3), bed_s, 0.0, 0.0),
    };
    // The bedrock threshold is pinned out of reach: only loose sand avalanches.
    gpu.avalanche = HydroParams::avalanche(&grid, settings.talus_deg, 90.0, model::AVALANCHE_RATE);
    gpu.avalanche_passes = settings.avalanche_passes;
    gpu.run = true;
    runtime.scheduled += 1;
    runtime.bed_time_s += f64::from(bed_s);
    stats.bed_time_s = runtime.bed_time_s;
    // The pack, bounds and height readback follow the step, in the same frame.
    terrain_gpu.dirty.store(true, Ordering::Relaxed);
}
