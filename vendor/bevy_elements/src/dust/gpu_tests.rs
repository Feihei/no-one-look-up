//! GPU tests of the aeolian transport: the kernels against their CPU mirror, conservation, the
//! pile's migration, the avalanche bound, the threshold, and the dunes' migration distance
//! (`docs/research/dust.md` Validation 3–7). They read the terrain's layer buffers back.
//!
//! Needs a GPU: `cargo test --release --lib -- --ignored --test-threads=1 dust`.

#[allow(clippy::duplicate_mod)]
#[path = "../../tests/support/mod.rs"]
mod support;

use std::sync::{Arc, Mutex};
use std::time::Duration;

use bevy::app::AppExit;
use bevy::prelude::*;
use bevy::render::gpu_readback::{Readback, ReadbackComplete};
use bevy::render::storage::ShaderBuffer;
use bevy::time::TimeUpdateStrategy;

use super::model::mirror::{Boundary, Field};
use super::model::{FLUX_COEFFICIENT, SATURATION_LENGTH_M, THRESHOLD_MS};
use super::{DustRuntime, DustSettings, DustStats};
use crate::terrain::{ProceduralTerrain, Terrain, TerrainState};
use crate::{
    CivilDate, Climate, ElementsCamera, ElementsPlugins, ElementsQuality, ElementsSystems,
    HydrologyBoundary, HydrologySettings, Weather, WeatherDirector, WeatherPreset, WorldClock,
};

static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());

fn serial() -> std::sync::MutexGuard<'static, ()> {
    ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner())
}

const DT: f64 = 1.0 / 60.0;
/// The note's bed time per step: 1/20 s of real time at the example's time scale.
const TIME_SCALE: f32 = 600.0;

/// The wind every test pins after the climate has run, each frame.
#[derive(Resource, Clone, Copy)]
struct WindPin {
    speed: f32,
    direction: Vec2,
}

fn pin_wind(pin: Res<WindPin>, mut weather: ResMut<Weather>) {
    weather.wind_speed_ms = pin.speed;
    weather.wind_direction = pin.direction;
}

/// A running app with the terrain generated and the transport held.
struct Rig {
    app: App,
    terrain: TerrainState,
    n: usize,
    cell: f32,
}

impl Rig {
    /// A flat bedrock plane of `n × n` cells of `cell` m.
    fn flat(n: u32, cell: f32, boundary: HydrologyBoundary, wind_ms: f32) -> Self {
        let terrain = Terrain::flat(0.0)
            .with_extent(Vec2::splat(n as f32 * cell))
            .with_resolution(UVec2::splat(n));
        Self::new(terrain, boundary, wind_ms, ElementsQuality::Ultra)
    }

    fn new(
        terrain: Terrain,
        boundary: HydrologyBoundary,
        wind_ms: f32,
        quality: ElementsQuality,
    ) -> Self {
        let mut app = App::new();
        app.add_plugins((support::headless_plugins(), ElementsPlugins))
            .insert_resource(TimeUpdateStrategy::ManualDuration(Duration::from_secs_f64(
                DT,
            )))
            .insert_resource(quality)
            .insert_resource(HydrologySettings {
                boundary,
                evaporation: false,
                // Hydrology's own thermal passes would move the sand: hold it.
                time_scale: 0.0,
                ..default()
            })
            .insert_resource(DustSettings {
                // Held until the test has written its layers.
                enabled: false,
                ..default()
            })
            .insert_resource(
                WorldClock::new(CivilDate::new(2026, 4, 2))
                    .at_hour(12.0)
                    .with_time_scale(TIME_SCALE),
            )
            .insert_resource(WindPin {
                speed: wind_ms,
                direction: Vec2::X,
            })
            .insert_resource(Spawn(terrain))
            .add_systems(Startup, setup)
            .add_systems(
                Update,
                pin_wind
                    .after(ElementsSystems::Weather)
                    .before(ElementsSystems::Simulation),
            );
        support::finish(&mut app);
        let deadline = std::time::Instant::now() + Duration::from_secs(300);
        loop {
            app.update();
            if app
                .world()
                .get_resource::<TerrainState>()
                .is_some_and(|s| s.is_live())
            {
                break;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "timed out waiting for the terrain"
            );
        }
        let terrain = app.world().resource::<TerrainState>().clone();
        Self {
            app,
            n: terrain.grid.size.x as usize,
            cell: terrain.grid.cell_m,
            terrain,
        }
    }

    fn run_until(&mut self, what: &str, mut done: impl FnMut(&mut App) -> bool) {
        let deadline = std::time::Instant::now() + Duration::from_secs(300);
        while std::time::Instant::now() < deadline {
            self.app.update();
            if done(&mut self.app) {
                return;
            }
        }
        panic!("timed out waiting for {what}");
    }

    fn frames(&mut self, n: usize) {
        for _ in 0..n {
            self.app.update();
        }
    }

    fn enable(&mut self, on: bool) {
        self.app.world_mut().resource_mut::<DustSettings>().enabled = on;
    }

    fn settings(&mut self, f: impl FnOnce(&mut DustSettings)) {
        f(&mut self.app.world_mut().resource_mut::<DustSettings>());
    }

    fn wind(&mut self, speed: f32) {
        self.app.world_mut().resource_mut::<WindPin>().speed = speed;
    }

    fn scheduled(&self) -> u64 {
        self.app.world().resource::<DustRuntime>().scheduled
    }

    fn steps(&self) -> u64 {
        self.app.world().resource::<DustStats>().steps
    }

    /// Run exactly `n` transport steps: enable, wait until `n` are scheduled, hold, wait until the
    /// GPU has recorded them all.
    fn run_steps(&mut self, n: u64) {
        let target = self.scheduled() + n;
        self.enable(true);
        self.run_until("the steps to be scheduled", |app| {
            app.world().resource::<DustRuntime>().scheduled >= target
        });
        self.enable(false);
        self.run_until("the steps to be recorded", |app| {
            app.world().resource::<DustStats>().steps >= target
        });
        // The step that was scheduled last is recorded and its readback copy is in flight.
        self.frames(3);
    }

    /// Replace the contents of a layer; the buffer keeps its identity.
    fn write(&mut self, handle: &Handle<ShaderBuffer>, data: &[f32]) {
        {
            let mut buffers = self.app.world_mut().resource_mut::<Assets<ShaderBuffer>>();
            let mut buffer = buffers.get_mut(handle).expect("the layer exists");
            buffer.clear();
            buffer.extend_from_slice(data);
        }
        self.app.update();
    }

    fn read(&mut self, handle: &Handle<ShaderBuffer>) -> Vec<f32> {
        let slot: Arc<Mutex<Option<Vec<u8>>>> = Arc::default();
        let sink = slot.clone();
        let entity = self
            .app
            .world_mut()
            .spawn(Readback::buffer(handle.clone()))
            .observe(move |done: On<ReadbackComplete>| {
                let mut sink = sink.lock().expect("lock");
                if sink.is_none() {
                    *sink = Some(done.data.to_vec());
                }
            })
            .id();
        self.run_until("a buffer readback", |_| {
            slot.lock().expect("lock").is_some()
        });
        self.app.world_mut().despawn(entity);
        let bytes = slot
            .lock()
            .expect("lock")
            .take()
            .expect("the readback completed");
        bytemuck::pod_collect_to_vec::<u8, f32>(&bytes[..self.n * self.n * 4])
    }

    fn sediment(&mut self) -> Vec<f32> {
        let handle = self.terrain.sediment.clone();
        self.read(&handle)
    }

    fn bedrock(&mut self) -> Vec<f32> {
        let handle = self.terrain.bedrock.clone();
        self.read(&handle)
    }

    fn write_sediment(&mut self, data: &[f32]) {
        let handle = self.terrain.sediment.clone();
        self.write(&handle, data);
    }

    fn write_bedrock(&mut self, data: &[f32]) {
        let handle = self.terrain.bedrock.clone();
        self.write(&handle, data);
    }

    fn finish(mut self) {
        self.app.world_mut().write_message(AppExit::Success);
    }
}

#[derive(Resource, Clone)]
struct Spawn(Terrain);

fn setup(
    mut commands: Commands,
    spawn: Res<Spawn>,
    mut director: ResMut<WeatherDirector>,
    mut images: ResMut<Assets<Image>>,
) {
    commands.spawn((
        Camera3d::default(),
        support::image_target(&mut images),
        ElementsCamera::default(),
        Transform::from_xyz(0.0, 30.0, 0.0),
    ));
    commands.spawn(spawn.0.clone());
    director.force(WeatherPreset::Clear, Duration::ZERO);
}

fn volume(sediment: &[f32], cell: f32) -> f64 {
    sediment.iter().map(|h| f64::from(*h)).sum::<f64>() * f64::from(cell) * f64::from(cell)
}

/// Centre of mass of the sediment, m along x and z from the grid's minimum corner.
fn centre_of_mass(sediment: &[f32], n: usize, cell: f32) -> (f64, f64) {
    let (mut sum, mut sx, mut sz) = (0.0f64, 0.0f64, 0.0f64);
    for (i, h) in sediment.iter().enumerate() {
        let h = f64::from(*h);
        sum += h;
        sx += h * ((i % n) as f64 + 0.5);
        sz += h * ((i / n) as f64 + 0.5);
    }
    (sx / sum * f64::from(cell), sz / sum * f64::from(cell))
}

/// A Gaussian pile of sediment, `sigma` m wide and `height` m high, centred on cell
/// `(cx, cz)`, over `base` m of sand.
fn pile(n: usize, cell: f32, centre: (f32, f32), sigma: f32, height: f32, base: f32) -> Vec<f32> {
    let mut sediment = vec![base; n * n];
    for z in 0..n {
        for x in 0..n {
            let dx = (x as f32 + 0.5 - centre.0) * cell;
            let dz = (z as f32 + 0.5 - centre.1) * cell;
            sediment[z * n + x] += height * (-(dx * dx + dz * dz) / (2.0 * sigma * sigma)).exp();
        }
    }
    sediment
}

/// The surface slope of the sediment, degrees, at every cell with sand on it (central
/// differences over `bedrock + sediment`).
fn sand_slopes(bedrock: &[f32], sediment: &[f32], n: usize, cell: f32) -> Vec<f32> {
    let surface = |x: usize, z: usize| bedrock[z * n + x] + sediment[z * n + x];
    let mut out = Vec::new();
    for z in 1..n - 1 {
        for x in 1..n - 1 {
            if sediment[z * n + x] <= 0.0 {
                continue;
            }
            let gx = (surface(x + 1, z) - surface(x - 1, z)) / (2.0 * cell);
            let gz = (surface(x, z + 1) - surface(x, z - 1)) / (2.0 * cell);
            out.push(gx.hypot(gz).atan().to_degrees());
        }
    }
    out
}

fn percentile(values: &mut [f32], p: f32) -> f32 {
    values.sort_by(f32::total_cmp);
    values[((values.len() - 1) as f32 * p).round() as usize]
}

/// The transport kernels against the CPU mirror, on a rough bedrock under a pile, for one step
/// and for six, with the avalanching off (the mirror has no thermal pass).
#[test]
#[ignore = "needs a GPU"]
fn gpu_aeolian_matches_the_cpu_mirror() {
    let _lock = serial();
    let (n, cell) = (128usize, 2.0f32);
    let mut rig = Rig::flat(n as u32, cell, HydrologyBoundary::Closed, 16.0);
    rig.settings(|s| s.avalanche_passes = 0);
    // Low hills under the sand, so the shadow, the speed-up and the supply limit all act.
    let mut bedrock = vec![0.0f32; n * n];
    for z in 0..n {
        for x in 0..n {
            let (fx, fz) = (x as f32 * cell, z as f32 * cell);
            bedrock[z * n + x] = 2.5 * (fx * 0.06).sin() * (fz * 0.045 + 1.0).cos() + 2.5;
        }
    }
    let sediment = pile(n, cell, (50.0, 64.0), 14.0, 3.0, 0.4);
    rig.write_bedrock(&bedrock);
    rig.write_sediment(&sediment);
    // The first reading is the layers themselves.
    assert_eq!(rig.sediment(), sediment);

    let mut mirror = Field {
        n,
        cell,
        bedrock: bedrock.clone(),
        sediment: sediment.clone(),
        dir: [1.0, 0.0],
        u10: 16.0,
        boundary: Boundary::Closed,
        k: FLUX_COEFFICIENT,
        threshold: THRESHOLD_MS,
        l_sat: SATURATION_LENGTH_M,
        // The step is 1/20 s of real time at time scale 600.
        dt_bed: 30.0,
    };
    let mut done = 0u64;
    for steps in [1u64, 6] {
        rig.run_steps(steps - done);
        while done < steps {
            let cap = mirror.capacity();
            let (qx, qz) = mirror.flux_scan(&cap);
            mirror.sediment = mirror.move_bed(&qx, &qz);
            done += 1;
        }
        let gpu = rig.sediment();
        let worst = gpu
            .iter()
            .zip(&mirror.sediment)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let moved = gpu
            .iter()
            .zip(&sediment)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        println!(
            "{steps} steps: worst GPU-CPU difference {worst:.2e} m, largest change {moved:.2e} m"
        );
        assert!(moved > 1e-3, "the sand did not move: {moved}");
        assert!(
            worst < 1e-3 * moved.max(1e-2) + 2e-4,
            "{steps} steps: {worst} m apart"
        );
    }
    rig.finish();
}

/// Validation 3, 4 and 6: a 3 m slab of sand on flat bedrock, 16 m/s along +x, 600 frames at time
/// scale 600: sand is conserved, never negative, bedrock is bit-unchanged, the sand moves downwind
/// at the Bagnold rate, and no slope is steeper than the repose angle plus 3°.
#[test]
#[ignore = "needs a GPU"]
fn gpu_aeolian_conserves_sand_and_moves_the_pile_downwind() {
    let _lock = serial();
    let (n, cell) = (256usize, 2.0f32);
    let mut rig = Rig::flat(n as u32, cell, HydrologyBoundary::Closed, 16.0);
    // A slab of sand 3 m thick and 60 m long across the whole width, 120 m from the upwind wall:
    // every row is the same one-dimensional problem. (A gentle Gaussian pile would do nothing
    // like it: see the note's Deviation 12.) Its upwind edge is stripped back at the speed the
    // note estimates, `c = F q_s / H`.
    let mut sediment = vec![0.0f32; n * n];
    for z in 0..n {
        for x in 0..n {
            let at = (x as f32 + 0.5) * cell - 120.0;
            if (0.0..60.0).contains(&at) {
                sediment[z * n + x] = 3.0;
            }
        }
    }
    rig.write_sediment(&sediment);
    let bedrock0 = rig.bedrock();
    let before = volume(&sediment, cell);
    // The centre of mass of the sand in the field: what the wind carried past the last cells
    // and the closed edge holds is not the slab's.
    let in_field = |layer: &[f32]| {
        let mut layer = layer.to_vec();
        for z in 0..n {
            for x in n - 4..n {
                layer[z * n + x] = 0.0;
            }
        }
        centre_of_mass(&layer, n, cell)
    };
    let (x0, z0) = in_field(&sediment);

    // 600 frames of 1/60 s at 20 steps per second: 200 steps of 30 s of bed time.
    rig.enable(true);
    let start = rig.scheduled();
    rig.run_until("600 frames", |app| {
        app.world().resource::<DustRuntime>().scheduled >= start + 200
    });
    rig.enable(false);
    let target = rig.scheduled();
    rig.run_until("the steps to be recorded", |app| {
        app.world().resource::<DustStats>().steps >= target
    });
    rig.frames(3);
    let after_layer = rig.sediment();
    let after = volume(&after_layer, cell);
    let bedrock1 = rig.bedrock();
    let (x1, z1) = in_field(&after_layer);

    let drift = (after - before).abs() / before;
    println!("volume {before:.1} -> {after:.1} m³ (relative drift {drift:.2e})");
    assert!(drift < 1e-5, "relative volume drift {drift}");
    assert!(after_layer.iter().all(|h| *h >= 0.0), "negative thickness");
    assert_eq!(bedrock0, bedrock1, "bedrock changed");

    let (dx, dz) = (x1 - x0, z1 - z0);
    let bed_time = rig.app.world().resource::<DustStats>().bed_time_s;
    let real_s = bed_time / f64::from(TIME_SCALE);
    // Bagnold's crest estimate c = F q_s(16) / H of the note, per second of real time.
    let bagnold = f64::from(TIME_SCALE) * f64::from(FLUX_COEFFICIENT) * 256.0 * 8.0 / 3.0;
    let speed = dx / real_s;
    println!(
        "centre of mass moved {dx:.2} m downwind, {dz:.3} m across in {real_s:.2} s: {speed:.3} m/s vs Bagnold {bagnold:.3} m/s (ratio {:.2})",
        speed / bagnold
    );
    assert!(dx > 0.0, "the pile moved upwind: {dx}");
    assert!(dz.abs() < 0.05 * dx, "lateral shift {dz} of {dx}");
    assert!(
        (0.3..=1.2).contains(&(speed / bagnold)),
        "speed {speed} m/s against Bagnold {bagnold} m/s"
    );
    let mut slopes = sand_slopes(&bedrock1, &after_layer, n, cell);
    let p99 = percentile(&mut slopes, 0.99);
    println!("99th percentile sand slope {p99:.1}°");
    assert!(p99 <= 34.0 + 3.0, "99th percentile slope {p99}°");
    rig.finish();
}

/// Validation 4: sand piled steeper than its repose angle avalanches back to it, in a wind that
/// also moves it, without moving bedrock or losing any.
#[test]
#[ignore = "needs a GPU"]
fn gpu_aeolian_avalanches_a_steep_pile_to_the_repose_angle() {
    let _lock = serial();
    let (n, cell) = (128usize, 2.0f32);
    let mut rig = Rig::flat(n as u32, cell, HydrologyBoundary::Closed, 16.0);
    // A cone at 50°: 12 m high, radius 10 m, standing on a 2 m sheet.
    let mut sediment = vec![2.0f32; n * n];
    for z in 0..n {
        for x in 0..n {
            let r = (((x as f32 + 0.5) - 64.0).powi(2) + ((z as f32 + 0.5) - 64.0).powi(2)).sqrt()
                * cell;
            sediment[z * n + x] += (12.0 - r * 50f32.to_radians().tan()).max(0.0);
        }
    }
    rig.write_sediment(&sediment);
    let bedrock0 = rig.bedrock();
    let before = volume(&sediment, cell);
    let steepest = |layer: &[f32]| {
        sand_slopes(&bedrock0, layer, n, cell)
            .into_iter()
            .fold(0.0f32, f32::max)
    };
    let p99_before = steepest(&sediment);
    // 2 s of real time: 40 steps.
    rig.run_steps(40);
    let after_layer = rig.sediment();
    let p99_after = steepest(&after_layer);
    let drift = (volume(&after_layer, cell) - before).abs() / before;
    println!("steepest slope {p99_before:.1}° -> {p99_after:.1}°, volume drift {drift:.2e}");
    assert!(p99_before > 40.0, "the cone is not steep: {p99_before}");
    assert!(
        p99_after < p99_before - 3.0,
        "the pile did not relax: {p99_after}"
    );
    assert!(
        p99_after <= 37.0 + 3.0,
        "slope {p99_after}° after the avalanche"
    );
    assert!(drift < 1e-5, "volume drift {drift}");
    assert_eq!(rig.bedrock(), bedrock0, "bedrock changed");
    rig.finish();
}

/// Validation 5: at 7.9 m/s on flat sand nothing changes, bit for bit; below half the threshold
/// the step is not dispatched at all; at 8.1 m/s a pile changes.
#[test]
#[ignore = "needs a GPU"]
fn gpu_aeolian_does_not_move_sand_below_the_threshold() {
    let _lock = serial();
    let (n, cell) = (128usize, 2.0f32);
    let mut rig = Rig::flat(n as u32, cell, HydrologyBoundary::Closed, 7.9);
    let flat = vec![4.0f32; n * n];
    rig.write_sediment(&flat);
    rig.run_steps(100);
    assert!(rig.steps() >= 100, "the steps were skipped at 7.9 m/s");
    let after = rig.sediment();
    assert_eq!(after, flat, "flat sand moved at 7.9 m/s");

    // Calm: no dispatch, no repack.
    rig.wind(3.9);
    rig.frames(120);
    let steps = rig.steps();
    let scheduled = rig.scheduled();
    rig.enable(true);
    rig.frames(120);
    assert_eq!(rig.steps(), steps, "steps recorded in a calm");
    assert_eq!(rig.scheduled(), scheduled, "steps scheduled in a calm");
    rig.enable(false);

    // 8.1 m/s over a pile: the speed-up over the crest and the shadow behind it move sand.
    rig.wind(8.1);
    let sediment = pile(n, cell, (64.0, 64.0), 10.0, 6.0, 4.0);
    rig.write_sediment(&sediment);
    rig.run_steps(100);
    let moved = rig.sediment();
    let change = moved
        .iter()
        .zip(&sediment)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!("8.1 m/s: largest change {change:.4} m");
    assert!(change > 1e-3, "nothing moved at 8.1 m/s: {change}");
    rig.finish();
}

/// Strip a terrain's uniform sand sheet for 60 s of real time and measure how far the stripped
/// region's front moved downwind: `(predicted m, measured m, spread m, volume before, after)`.
/// The front of a row is the first cell with more than 0.5 m of sand (where the terrain material
/// turns to sand); the measure is the median over the rows, the spread their 25–75 % range.
fn front_migration(
    terrain: Terrain,
    quality: ElementsQuality,
    lay_sheet_m: Option<f32>,
) -> (f64, f64, f64, f64, f64) {
    let mut rig = Rig::new(terrain, HydrologyBoundary::Closed, 16.0, quality);
    let (n, cell) = (rig.n, rig.cell);
    assert_eq!(cell, 2.0);
    if let Some(depth) = lay_sheet_m {
        rig.write_sediment(&vec![depth; n * n]);
    }
    let bedrock = rig.bedrock();
    let sheet = rig.sediment();
    let depth = f64::from(sheet.iter().sum::<f32>() / sheet.len() as f32);
    let front = |layer: &[f32]| -> Vec<f64> {
        let margin = 40usize;
        (margin..n - margin)
            .map(|z| {
                let x = (0..n - 8).find(|x| layer[z * n + x] > 0.5).unwrap_or(n - 8);
                x as f64 * f64::from(cell)
            })
            .collect()
    };
    let median = |mut v: Vec<f64>, p: f64| {
        v.sort_by(f64::total_cmp);
        v[((v.len() - 1) as f64 * p).round() as usize]
    };
    // Two seconds for the flux to establish itself, then 60 s.
    rig.run_steps(40);
    let first = rig.sediment();
    rig.run_steps(1200);
    let second = rig.sediment();
    assert_eq!(rig.bedrock(), bedrock, "bedrock changed");
    let (f0, f1) = (front(&first), front(&second));
    let moved: Vec<f64> = f0.iter().zip(&f1).map(|(a, b)| b - a).collect();
    let (before, after) = (volume(&first, cell), volume(&second, cell));
    rig.finish();
    // `Q_eff · 60 s / H` with H the sheet's depth: the note's 36.9 m for 4 m.
    let q_eff = f64::from(FLUX_COEFFICIENT) * 8.0 * 256.0 * f64::from(TIME_SCALE);
    (
        q_eff * 60.0 / depth,
        median(moved.clone(), 0.5),
        median(moved.clone(), 0.75) - median(moved, 0.25),
        before,
        after,
    )
}

fn assert_front(label: &str, result: (f64, f64, f64, f64, f64)) {
    let (predicted, measured, spread, before, after) = result;
    println!(
        "{label}: predicted front shift {predicted:.1} m, measured {measured:.1} m (rows' 25-75 % spread {spread:.1} m); volume {before:.0} -> {after:.0} m³"
    );
    assert!(
        (0.4 * predicted..=1.6 * predicted).contains(&measured),
        "{label}: shift {measured} m outside [0.4, 1.6] × {predicted}"
    );
    assert!(
        measured >= 6.0,
        "{label}: shift {measured} m is under three cells"
    );
    assert!(
        (after - before).abs() / before < 1e-3,
        "{label}: volume {before} -> {after}"
    );
}

/// Validation 7 on level ground: the stripped front of a 4 m sand sheet moves by `Q_eff · 60 s / H`.
#[test]
#[ignore = "needs a GPU"]
fn gpu_aeolian_strips_a_sand_sheet_by_the_predicted_distance() {
    let _lock = serial();
    let terrain = Terrain::flat(0.0)
        .with_extent(Vec2::splat(1024.0))
        .with_resolution(UVec2::splat(512));
    // `flat` has no sand: lay the 4 m sheet the dunes preset has.
    assert_front(
        "flat",
        front_migration(terrain, ElementsQuality::High, Some(4.0)),
    );
}

/// Validation 7 on `ProceduralTerrain::dunes(7)`, the example's terrain and sand sheet.
#[test]
#[ignore = "needs a GPU"]
fn gpu_aeolian_strips_the_dunes_sheet_by_the_predicted_distance() {
    let _lock = serial();
    let terrain = Terrain::procedural(ProceduralTerrain::dunes(7))
        .with_extent(Vec2::splat(1024.0))
        .with_resolution(UVec2::splat(512));
    assert_front(
        "dunes(7)",
        front_migration(terrain, ElementsQuality::High, None),
    );
}

/// `ProceduralTerrain::dunes` is the preset the note names.
#[test]
fn the_dunes_preset_is_the_one_the_note_names() {
    let dunes = ProceduralTerrain::dunes(7);
    assert_eq!(
        (
            dunes.feature_size_m,
            dunes.relief_m,
            dunes.ridged,
            dunes.initial_sand_m
        ),
        (300.0, 40.0, 0.0, 4.0)
    );
    // A desert climate has dust storms, and the preset's wind is above the threshold.
    assert!(Climate::desert().dust_share > 0.0);
}
