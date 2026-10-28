//! Hydrology and erosion on a real GPU: water conservation, rain delivery, sources and sinks, the
//! open boundary, erosion mass balance, thermal erosion, the bake and the flood gameplay query,
//! each read back from the terrain's layer buffers. See `docs/research/hydrology.md` and
//! `docs/research/erosion.md` (Validation targets).
//!
//! Needs a GPU: `cargo test --release --test gpu_hydrology -- --ignored --test-threads=1`. The
//! tests share one process and each builds a renderer, so they take a lock and run one at a time.

mod support;

use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;

use bevy::app::AppExit;
use bevy::ecs::system::RunSystemOnce;
use bevy::prelude::*;
use bevy::render::gpu_readback::{Readback, ReadbackComplete};
use bevy::render::storage::ShaderBuffer;
use bevy::time::TimeUpdateStrategy;
use bevy_elements::*;

static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());

fn serial() -> MutexGuard<'static, ()> {
    ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner())
}

const DT: f64 = 1.0 / 60.0;
/// How far the delivered rain may lag the ideal: one quantum (5 µm) still owed, a frame's worth of
/// rain (16 substeps, 0.36 µm) and the f32 rounding of a 2 m depth (0.12 µm each way).
const RAIN_LAG_M: f64 = 5.6e-6;
const STEP_S: f32 = 0.02;

/// What the test pins the weather to after the climate has run, each frame.
#[derive(Resource, Clone, Copy)]
struct Pin {
    mm_h: f32,
    temperature_c: f32,
}

fn pin_weather(pin: Res<Pin>, mut weather: ResMut<Weather>) {
    weather.precipitation_mm_h = pin.mm_h;
    weather.temperature_c = pin.temperature_c;
}

/// Counts of the messages the tests wait for.
#[derive(Resource, Default)]
struct Baked(Vec<f32>);

fn record_baked(mut messages: MessageReader<TerrainBaked>, mut baked: ResMut<Baked>) {
    baked.0.extend(messages.read().map(|m| m.duration_s));
}

#[derive(Clone)]
struct Config {
    terrain: Terrain,
    hydrology: HydrologySettings,
    erosion: ErosionSettings,
    preset: WeatherPreset,
    mm_h: f32,
    temperature_c: f32,
    /// A source or sink spawned at startup: (world XZ, signed rate m³/s, radius m).
    sources: Vec<(Vec2, f32, f32)>,
    /// A precipitation blocker box (centre, half extents) spawned at startup.
    blockers: Vec<(Vec3, Vec3)>,
    /// A bake requested before the terrain exists.
    early_bake: Option<ErodeTerrain>,
}

impl Config {
    fn new(terrain: Terrain) -> Self {
        Self {
            terrain,
            hydrology: HydrologySettings {
                boundary: HydrologyBoundary::Closed,
                evaporation: false,
                // Held until the test has written its layers.
                time_scale: 0.0,
                ..default()
            },
            erosion: ErosionSettings::default(),
            preset: WeatherPreset::Clear,
            mm_h: 0.0,
            temperature_c: 15.0,
            sources: Vec::new(),
            blockers: Vec::new(),
            early_bake: None,
        }
    }
}

#[derive(Resource, Clone)]
struct Spawn(Config);

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
    commands.spawn(spawn.0.terrain.clone());
    for &(at, rate, radius) in &spawn.0.sources {
        let transform = Transform::from_xyz(at.x, 0.0, at.y);
        if rate >= 0.0 {
            commands.spawn((
                WaterSource {
                    rate_m3_s: rate,
                    radius_m: radius,
                },
                transform,
            ));
        } else {
            commands.spawn((
                WaterSink {
                    rate_m3_s: -rate,
                    radius_m: radius,
                },
                transform,
            ));
        }
    }
    for &(centre, half_extents) in &spawn.0.blockers {
        commands.spawn((
            PrecipitationBlocker { half_extents },
            Transform::from_translation(centre),
        ));
    }
    director.force(spawn.0.preset, Duration::ZERO);
}

/// Update until `done` holds (at most 240 s of wall time: a bake is 7500 frames).
fn run_until(app: &mut App, what: &str, mut done: impl FnMut(&mut App) -> bool) {
    let deadline = std::time::Instant::now() + Duration::from_secs(240);
    while std::time::Instant::now() < deadline {
        app.update();
        if done(app) {
            return;
        }
    }
    panic!("timed out waiting for {what}");
}

/// A running app with the terrain generated and the hydrology held.
struct Sim {
    app: App,
    terrain: TerrainState,
    /// Substeps per frame at the quality: the tests run at `Ultra` (16) so a thousand substeps
    /// take a hundred frames, which matters on a GPU shared with other work.
    per_frame: u64,
}

impl Sim {
    fn new(config: Config) -> Self {
        Self::at(config, ElementsQuality::Ultra)
    }

    fn at(config: Config, quality: ElementsQuality) -> Self {
        let mut app = App::new();
        app.add_plugins((support::headless_plugins(), ElementsPlugins))
            .insert_resource(TimeUpdateStrategy::ManualDuration(Duration::from_secs_f64(
                DT,
            )))
            .insert_resource(quality)
            .insert_resource(config.hydrology.clone())
            .insert_resource(config.erosion.clone())
            .insert_resource(Pin {
                mm_h: config.mm_h,
                temperature_c: config.temperature_c,
            })
            .insert_resource(Spawn(config.clone()))
            .init_resource::<Baked>()
            .add_systems(Startup, setup)
            .add_systems(
                Update,
                pin_weather
                    .after(ElementsSystems::Weather)
                    .before(ElementsSystems::Simulation),
            )
            .add_systems(Update, record_baked.after(ElementsSystems::Simulation));
        support::finish(&mut app);
        if let Some(bake) = config.early_bake {
            app.world_mut().write_message(bake);
        }
        run_until(&mut app, "the terrain", |app| {
            app.world()
                .get_resource::<TerrainState>()
                .is_some_and(|s| s.is_live())
        });
        let terrain = app.world().resource::<TerrainState>().clone();
        Self {
            app,
            terrain,
            per_frame: u64::from(quality.hydrology_substeps()),
        }
    }

    fn run_until(&mut self, what: &str, done: impl FnMut(&mut App) -> bool) {
        run_until(&mut self.app, what, done);
    }

    fn frames(&mut self, n: usize) {
        for _ in 0..n {
            self.app.update();
        }
    }

    fn stats(&self) -> HydrologyStats {
        self.app.world().resource::<HydrologyStats>().clone()
    }

    fn set_time_scale(&mut self, scale: f32) {
        self.app
            .world_mut()
            .resource_mut::<HydrologySettings>()
            .time_scale = scale;
    }

    /// Run at least `n` more substeps at the full per-frame rate, hold, and wait until every
    /// scheduled substep has been recorded. Returns the substeps run.
    fn run_substeps(&mut self, n: u64) -> u64 {
        let start = self.stats().substeps;
        // The time scale that runs exactly `per_frame` substeps in a 1/60 s frame.
        self.set_time_scale(self.per_frame as f32 * 1.2);
        self.run_until("the substeps", |app| {
            app.world().resource::<HydrologyStats>().substeps >= start + n
        });
        self.set_time_scale(0.0);
        self.settle();
        self.stats().substeps - start
    }

    /// Update until the substep counter has not moved for 20 frames.
    fn settle(&mut self) {
        let mut last = self.stats().substeps;
        let mut quiet = 0;
        while quiet < 20 {
            self.app.update();
            let now = self.stats().substeps;
            if now == last {
                quiet += 1;
            } else {
                quiet = 0;
                last = now;
            }
        }
    }

    /// Replace the contents of a layer. The buffer keeps its identity, so bind groups stay valid.
    fn write(&mut self, handle: &Handle<ShaderBuffer>, data: &[f32]) {
        {
            let mut buffers = self.app.world_mut().resource_mut::<Assets<ShaderBuffer>>();
            let mut buffer = buffers.get_mut(handle).expect("the layer exists");
            buffer.clear();
            buffer.extend_from_slice(data);
        }
        // One frame to upload it.
        self.app.update();
    }

    fn read_bytes(&mut self, handle: &Handle<ShaderBuffer>) -> Vec<u8> {
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
        slot.lock()
            .expect("lock")
            .take()
            .expect("the readback completed")
    }

    fn read_f32(&mut self, handle: &Handle<ShaderBuffer>, count: usize) -> Vec<f32> {
        let bytes = self.read_bytes(handle);
        bytemuck::pod_collect_to_vec::<u8, f32>(&bytes[..count * 4])
    }

    fn read_vec2(&mut self, handle: &Handle<ShaderBuffer>, count: usize) -> Vec<[f32; 2]> {
        let bytes = self.read_bytes(handle);
        bytemuck::pod_collect_to_vec::<u8, [f32; 2]>(&bytes[..count * 8])
    }

    fn water_depth(&mut self, p: Vec2) -> Option<f32> {
        self.app
            .world_mut()
            .run_system_once(move |env: Environment| env.water_depth(p))
            .expect("the query runs")
    }

    fn finish(mut self) {
        self.app.world_mut().write_message(AppExit::Success);
    }
}

fn tank(cells: u32, cell_m: f32) -> Terrain {
    Terrain::flat(0.0)
        .with_extent(Vec2::splat(cells as f32 * cell_m))
        .with_resolution(UVec2::splat(cells))
}

fn sum(values: &[f32]) -> f64 {
    values.iter().map(|v| f64::from(*v)).sum()
}

/// Validation 1: a closed bowl with a 1 m column conserves its water.
#[test]
#[ignore = "needs a GPU"]
fn gpu_hydrology_conserves_water() {
    let _lock = serial();
    let n = 256usize;
    let mut sim = Sim::new(Config::new(tank(256, 2.0)));
    let cells = n * n;
    // b = 6 m (r / 128)² around the grid centre, 1 m of water inside 8 cells.
    let mut bed = vec![0.0f32; cells];
    let mut water = vec![0.0f32; cells];
    for j in 0..n {
        for i in 0..n {
            let r = (((i as f32 - 127.5).powi(2)) + ((j as f32 - 127.5).powi(2))).sqrt();
            bed[j * n + i] = 6.0 * (r / 128.0).powi(2);
            if r <= 8.0 {
                water[j * n + i] = 1.0;
            }
        }
    }
    let v0 = sum(&water);
    assert_eq!(v0, 208.0, "the column covers 208 cells");
    let (bedrock, water_handle) = (sim.terrain.bedrock.clone(), sim.terrain.water.clone());
    sim.write(&bedrock, &bed);
    sim.write(&water_handle, &water);
    let ran = sim.run_substeps(2000);
    let after = sim.read_f32(&water_handle, cells);
    let v = sum(&after);
    let drift = (v / v0 - 1.0).abs();
    let min = after.iter().copied().fold(f32::INFINITY, f32::min);
    let max = after.iter().copied().fold(0.0, f32::max);
    let velocity = sim.read_vec2(&sim.terrain.velocity.clone(), cells);
    let fastest = velocity
        .iter()
        .map(|v| (v[0] * v[0] + v[1] * v[1]).sqrt())
        .fold(0.0, f32::max);
    println!(
        "conservation: {ran} substeps, V {v:.6} / {v0}, drift {drift:.2e}, min {min}, max {max:.5}, fastest {fastest:.2} m/s"
    );
    assert!(ran >= 2000);
    assert!(drift <= 1e-3, "the plan's 0.1 %: {drift}");
    assert!(drift <= 1e-5, "regression guard: {drift}");
    assert!(min >= 0.0, "negative depth {min}");
    assert!(max <= 1.001, "the column rose to {max} m");
    assert!(
        fastest <= 0.5 * 2.0 / STEP_S,
        "speed {fastest} above the clamp"
    );
    // The water has spread: the column is no longer a column.
    assert!(
        after.iter().filter(|d| **d > 1e-3).count() > 208,
        "the column did not spread"
    );
    sim.finish();
}

/// Run a 1 m column of radius 8 cells on a 128 m plane sloping `slope` toward −X, 3000 substeps
/// in chunks of 250: the volume after each chunk, the highest depth seen and the stats volume.
fn column_run(slope: f32, boundary: HydrologyBoundary) -> (Vec<f64>, f32, f64) {
    let n = 64usize;
    let mut config = Config::new(tank(64, 2.0));
    config.hydrology.boundary = boundary;
    let mut sim = Sim::new(config);
    let cells = n * n;
    let mut water = vec![0.0f32; cells];
    let mut bed = vec![0.0f32; cells];
    for j in 0..n {
        for i in 0..n {
            let r = (((i as f32 - 31.5).powi(2)) + ((j as f32 - 31.5).powi(2))).sqrt();
            bed[j * n + i] = slope * (i as f32 * 2.0 + 1.0);
            if r <= 8.0 {
                water[j * n + i] = 1.0;
            }
        }
    }
    let (bedrock, water_handle) = (sim.terrain.bedrock.clone(), sim.terrain.water.clone());
    sim.write(&bedrock, &bed);
    sim.write(&water_handle, &water);
    let mut volumes = vec![sum(&water)];
    let mut highest = 1.0f32;
    for _ in 0..12 {
        sim.run_substeps(250);
        let after = sim.read_f32(&water_handle, cells);
        highest = highest.max(after.iter().copied().fold(0.0, f32::max));
        volumes.push(sum(&after));
    }
    // The depth readback agrees with the buffer (m³ with 4 m² cells).
    sim.frames(60);
    let stats = sim.stats().volume_m3;
    sim.finish();
    (volumes, highest, stats)
}

/// Validation 2: the open boundary loses water and never gains it; closed conserves it.
#[test]
#[ignore = "needs a GPU"]
fn an_open_boundary_loses_water_and_never_gains_it() {
    let _lock = serial();
    // Flat bed, open: the column spreads, nothing rises above its initial depth and the volume
    // never grows.
    let (flat, highest, _) = column_run(0.0, HydrologyBoundary::Open);
    println!(
        "open flat: V {:.3} of {}, highest depth {highest}",
        flat[12], flat[0]
    );
    for pair in flat.windows(2) {
        assert!(
            pair[1] <= pair[0] + 1e-6 * flat[0],
            "the volume rose from {} to {}",
            pair[0],
            pair[1]
        );
    }
    // Inertia lets the collapsing column ring a little (the note's "never above the initial
    // depth" is the bowl's 1 mm; measured 1.0063 m here at a chunk boundary).
    assert!(highest <= 1.02, "a cell rose to {highest} m");
    // A 5 % slope toward −X carries the column to the open edge within the run (on a flat bed
    // the spreading sheet is too thin and slow to get there in 60 s).
    let (sloped, _, stats) = column_run(0.05, HydrologyBoundary::Open);
    println!(
        "open slope: V {:.3} of {}, stats volume {stats:.3} m³",
        sloped[12], sloped[0]
    );
    for pair in sloped.windows(2) {
        assert!(pair[1] <= pair[0] + 1e-6 * sloped[0]);
    }
    assert!(
        sloped[12] < 0.9 * sloped[0],
        "little water left through the open edge: {}",
        sloped[12]
    );
    assert!((stats - sloped[12] * 4.0).abs() < 0.01 * sloped[12] * 4.0);
    // The same slope, closed: everything stays.
    let (closed, _, _) = column_run(0.05, HydrologyBoundary::Closed);
    println!("closed slope: V {:.6} of {}", closed[12], closed[0]);
    assert!((closed[12] / closed[0] - 1.0).abs() < 1e-5);
}

/// Validation 3: rain on a deep tank arrives in full, in quanta; cold air makes it snow; a blocker
/// shelters the cells under it.
#[test]
#[ignore = "needs a GPU"]
fn rain_is_delivered_exactly_in_quanta() {
    let _lock = serial();
    let n = 64usize;
    let cells = n * n;
    let mean_gain = |after: &[f32], base: f64| (sum(after) - base * cells as f64) / cells as f64;
    let mut config = Config::new(tank(64, 2.0));
    config.preset = WeatherPreset::Rain;
    config.mm_h = 4.0;
    let mut sim = Sim::new(config);
    let water = sim.terrain.water.clone();
    sim.write(&water, &vec![2.0f32; cells]);
    let ran = sim.run_substeps(1800);
    assert_eq!(sim.stats().dropped, 0, "a dropped frame loses its rain");
    let after = sim.read_f32(&water, cells);
    let ideal = ran as f64 * f64::from(STEP_S) * 4.0 / 3.6e6;
    println!(
        "rain: {ran} substeps, mean gain {:.3e} (ideal {ideal:.3e})",
        mean_gain(&after, 2.0)
    );
    // Within one quantum of the ideal: the plain per-substep f32 sum gains exactly 0.
    assert!(
        mean_gain(&after, 2.0) >= ideal - RAIN_LAG_M && mean_gain(&after, 2.0) <= ideal + 1e-6,
        "gain {} vs ideal {ideal}",
        mean_gain(&after, 2.0)
    );
    sim.finish();

    // A roof over half of the tank (|x| < 32 m): no rain under it. Dry ground, so the rain stays
    // a thin film (flows between a 40 µm film and none are slow) and the volume is what counts:
    // half of the ideal. (In 2 m of water the level steps of a few µm drive flows far below the
    // f32 spacing of the depth, and the rounding of those creates water: hydrology.md Deviation
    // 21.)
    let mut roofed = Config::new(tank(64, 2.0));
    roofed.preset = WeatherPreset::Rain;
    roofed.mm_h = 4.0;
    roofed.blockers = vec![(Vec3::new(0.0, 5.0, 0.0), Vec3::new(32.0, 0.5, 64.0))];
    let mut sim = Sim::new(roofed);
    let water = sim.terrain.water.clone();
    let ran = sim.run_substeps(1800);
    assert_eq!(sim.stats().dropped, 0, "a dropped frame loses its rain");
    let map = sim.read_vec2(&sim.terrain.source_map.clone(), cells);
    let shaded = map.iter().filter(|m| m[1] == 0.0).count();
    let after = sim.read_f32(&water, cells);
    let ideal = 0.5 * ran as f64 * f64::from(STEP_S) * 4.0 / 3.6e6;
    println!(
        "roof: {shaded} shaded cells, mean gain {:.3e} (half the ideal {ideal:.3e})",
        mean_gain(&after, 0.0)
    );
    assert_eq!(shaded, 32 * 64, "the roof shades half of the cells");
    // Half of the quantum still owed, at most, and the quanta are 5 µm.
    assert!(
        mean_gain(&after, 0.0) >= ideal - 0.5 * RAIN_LAG_M
            && mean_gain(&after, 0.0) <= ideal + 1e-6,
        "gain {} vs half the ideal {ideal}",
        mean_gain(&after, 0.0)
    );
    sim.finish();

    // Below freezing the same rate falls as snow: the water layer gains nothing.
    let mut cold = Config::new(tank(64, 2.0));
    cold.preset = WeatherPreset::Rain;
    cold.mm_h = 4.0;
    cold.temperature_c = -5.0;
    let mut sim = Sim::new(cold);
    let water = sim.terrain.water.clone();
    sim.write(&water, &vec![2.0f32; cells]);
    sim.run_substeps(1800);
    let after = sim.read_f32(&water, cells);
    assert!(
        after.iter().all(|d| *d == 2.0),
        "snow fell as rain: {}",
        after[0]
    );
    sim.finish();
}

/// A sink removes its rate from deep water, and only what is there.
#[test]
#[ignore = "needs a GPU"]
fn a_sink_removes_its_rate_and_a_source_adds_it() {
    let _lock = serial();
    let n = 64usize;
    let cells = n * n;
    let mut config = Config::new(tank(64, 2.0));
    config.sources = vec![(Vec2::new(-30.0, 0.0), -10.0, 4.0)];
    let mut sim = Sim::new(config);
    let water = sim.terrain.water.clone();
    sim.write(&water, &vec![1.0f32; cells]);
    let v0 = sum(&vec![1.0f32; cells]) * 4.0;
    let ran = sim.run_substeps(500);
    let v = sum(&sim.read_f32(&water, cells)) * 4.0;
    let seconds = ran as f64 * f64::from(STEP_S);
    println!(
        "sink: {ran} substeps ({seconds} s), removed {:.2} m³ (10 m³/s → {:.2})",
        v0 - v,
        10.0 * seconds
    );
    assert!(
        ((v0 - v) - 10.0 * seconds).abs() < 0.03 * 10.0 * seconds,
        "removed {}",
        v0 - v
    );
    sim.finish();

    // A source on dry ground fills it at its rate.
    let mut config = Config::new(tank(64, 2.0));
    config.sources = vec![(Vec2::new(0.0, 0.0), 5.0, 3.0)];
    let mut sim = Sim::new(config);
    let ran = sim.run_substeps(500);
    let water = sim.terrain.water.clone();
    let v = sum(&sim.read_f32(&water, cells)) * 4.0;
    let seconds = ran as f64 * f64::from(STEP_S);
    println!("source: added {v:.2} m³ (5 m³/s → {:.2})", 5.0 * seconds);
    assert!(
        (v - 5.0 * seconds).abs() < 0.01 * 5.0 * seconds,
        "added {v}"
    );
    // A sink on dry ground removes nothing.
    sim.finish();
    let mut config = Config::new(tank(64, 2.0));
    config.sources = vec![(Vec2::new(0.0, 0.0), -10.0, 3.0)];
    let mut sim = Sim::new(config);
    sim.run_substeps(200);
    let water = sim.terrain.water.clone();
    assert_eq!(sum(&sim.read_f32(&water, cells)), 0.0);
    sim.finish();
}

/// A smooth 40 m-relief height field with a few scales, deterministic.
fn smooth_field(n: usize) -> Vec<f32> {
    let tau = std::f32::consts::TAU;
    let mut field = Vec::with_capacity(n * n);
    for j in 0..n {
        for i in 0..n {
            let (x, y) = (i as f32, j as f32);
            let h = 20.0
                + 12.0 * (tau * x / 128.0).sin()
                + 8.0 * (tau * y / 96.0 + 1.0).cos()
                + 5.0 * (tau * (x + y) / 64.0).sin();
            field.push(h);
        }
    }
    field
}

/// Wait for the first `TerrainBaked`, counting the frames; returns them.
fn bake_frames(sim: &mut Sim) -> usize {
    let mut frames = 0;
    sim.run_until("the bake", |app| {
        frames += 1;
        !app.world().resource::<Baked>().0.is_empty()
    });
    frames
}

/// Erosion validation 1: a closed bake keeps every unit of solid.
#[test]
#[ignore = "needs a GPU"]
fn gpu_erosion_conserves_mass() {
    let _lock = serial();
    let n = 256usize;
    let cells = n * n;
    let l2 = 4.0f64;
    let mut sim = Sim::new(Config::new(tank(256, 2.0)));
    let bed0 = smooth_field(n);
    let sed0 = vec![0.5f32; cells];
    let (bedrock, sediment, suspended, water) = (
        sim.terrain.bedrock.clone(),
        sim.terrain.sediment.clone(),
        sim.terrain.suspended.clone(),
        sim.terrain.water.clone(),
    );
    sim.write(&bedrock, &bed0);
    sim.write(&sediment, &sed0);
    let m0 = (sum(&bed0) + sum(&sed0)) * l2;
    sim.app.world_mut().write_message(ErodeTerrain {
        duration_s: 100.0,
        rain_mm_h: 20.0,
    });
    // Halfway through, the water carries something.
    sim.run_until("2500 substeps", |app| {
        app.world().resource::<HydrologyStats>().substeps >= 2500
    });
    let mid = sim.read_f32(&suspended, cells);
    let mid_max = mid.iter().copied().fold(0.0, f32::max);
    let frames = bake_frames(&mut sim);
    sim.settle();
    let bed = sim.read_f32(&bedrock, cells);
    let sed = sim.read_f32(&sediment, cells);
    let susp = sim.read_f32(&suspended, cells);
    let wat = sim.read_f32(&water, cells);
    let m1 = (sum(&bed) + sum(&sed) + sum(&susp)) * l2;
    let moved: f64 = bed
        .iter()
        .zip(&sed)
        .zip(bed0.iter().zip(&sed0))
        .map(|((b, s), (b0, s0))| f64::from((b + s) - (b0 + s0)).abs())
        .sum::<f64>()
        * l2
        / 2.0;
    let rel = (m1 / m0 - 1.0).abs();
    let datum_free = (m1 - m0).abs();
    let min_susp = susp.iter().copied().fold(f32::INFINITY, f32::min);
    let min_sed = sed.iter().copied().fold(f32::INFINITY, f32::min);
    let min_wat = wat.iter().copied().fold(f32::INFINITY, f32::min);
    let max_susp = susp.iter().copied().fold(0.0, f32::max);
    println!(
        "erosion mass: {} substeps, M {m1:.4} / {m0:.4} (rel {rel:.2e}), |ΔM| {datum_free:.3e} m³ of {moved:.1} m³ moved ({:.2e}), min s {min_susp:e}, min Bs {min_sed:e}, min d {min_wat:e}, max s {mid_max:.4}/{max_susp:.4}",
        sim.stats().substeps,
        datum_free / moved
    );
    let _ = frames;
    assert!(rel <= 5e-3, "the plan's 0.5 %: {rel}");
    assert!(moved > 100.0, "nothing eroded: {moved} m³ moved");
    assert!(
        datum_free <= 1e-3 * moved,
        "|ΔM| {datum_free} above 0.1 % of the moved mass {moved}"
    );
    assert!(min_susp >= 0.0 && min_sed >= 0.0 && min_wat >= 0.0);
    assert!(mid_max > 0.0, "the water never carried any sediment");
    sim.finish();
}

/// The steepest of the eight downhill neighbour slopes of a cell, degrees.
fn steepest_deg(surface: &[f32], n: usize, i: usize, j: usize, l: f32) -> f32 {
    let mut best = 0.0f32;
    for dj in -1i32..=1 {
        for di in -1i32..=1 {
            if di == 0 && dj == 0 {
                continue;
            }
            let (x, y) = (i as i32 + di, j as i32 + dj);
            if x < 0 || y < 0 || x >= n as i32 || y >= n as i32 {
                continue;
            }
            let dist = l * ((di * di + dj * dj) as f32).sqrt();
            let drop = surface[j * n + i] - surface[y as usize * n + x as usize];
            best = best.max((drop / dist).atan().to_degrees());
        }
    }
    best
}

/// The central-difference slope of the terrain shader's normal, degrees.
fn central_deg(surface: &[f32], n: usize, i: usize, j: usize, l: f32) -> f32 {
    let at = |x: usize, y: usize| surface[y.min(n - 1) * n + x.min(n - 1)];
    let gx = (at(i + 1, j) - at(i.saturating_sub(1), j)) / (2.0 * l);
    let gz = (at(i, j + 1) - at(i, j.saturating_sub(1))) / (2.0 * l);
    (gx * gx + gz * gz).sqrt().atan().to_degrees()
}

fn percentile(values: &mut [f32], p: f64) -> f32 {
    values.sort_by(f32::total_cmp);
    values[((values.len() - 1) as f64 * p).round() as usize]
}

/// A cone of base radius 12 cells and slope tan 60° (41.6 m high) at the grid centre.
fn cone(n: usize) -> Vec<f32> {
    let mut field = vec![0.0f32; n * n];
    for j in 0..n {
        for i in 0..n {
            let r = (((i as f32 - 127.5).powi(2)) + ((j as f32 - 127.5).powi(2))).sqrt() * 2.0;
            field[j * n + i] = (24.0 - r).max(0.0) * 60.0f32.to_radians().tan();
        }
    }
    field
}

/// Erosion validation 2: a sediment cone relaxes to the talus angle, a bedrock cone to 45°.
#[test]
#[ignore = "needs a GPU"]
fn gpu_thermal_respects_talus() {
    let _lock = serial();
    let n = 256usize;
    let cells = n * n;
    let heap = cone(n);
    let zero = vec![0.0f32; cells];
    for sediment_cone in [true, false] {
        let mut sim = Sim::new(Config::new(tank(256, 2.0)));
        let (bedrock, sediment) = (sim.terrain.bedrock.clone(), sim.terrain.sediment.clone());
        if sediment_cone {
            sim.write(&bedrock, &zero);
            sim.write(&sediment, &heap);
        } else {
            sim.write(&bedrock, &heap);
            sim.write(&sediment, &zero);
        }
        let total0 = sum(&heap);
        // No rain, no sources: 4000 substeps are 1000 thermal passes.
        sim.app.world_mut().write_message(ErodeTerrain {
            duration_s: 80.0,
            rain_mm_h: 0.0,
        });
        bake_frames(&mut sim);
        sim.settle();
        let bed = sim.read_f32(&bedrock, cells);
        let sed = sim.read_f32(&sediment, cells);
        let surface: Vec<f32> = bed.iter().zip(&sed).map(|(b, s)| b + s).collect();
        let drift = (sum(&surface) / total0 - 1.0).abs();
        let footprint: Vec<(usize, usize)> = (0..n)
            .flat_map(|j| (0..n).map(move |i| (i, j)))
            .filter(|&(i, j)| surface[j * n + i] > 0.01)
            .collect();
        let mut steepest: Vec<f32> = footprint
            .iter()
            .map(|&(i, j)| steepest_deg(&surface, n, i, j, 2.0))
            .collect();
        let mut central: Vec<f32> = footprint
            .iter()
            .map(|&(i, j)| central_deg(&surface, n, i, j, 2.0))
            .collect();
        let max_steep = steepest.iter().copied().fold(0.0, f32::max);
        let p99_steep = percentile(&mut steepest, 0.99);
        let p99_central = percentile(&mut central, 0.99);
        println!(
            "thermal ({}): steepest p99 {p99_steep:.2}° max {max_steep:.2}°, central p99 {p99_central:.2}°, drift {drift:.2e}, footprint {} cells, {} substeps",
            if sediment_cone { "sediment" } else { "bedrock" },
            footprint.len(),
            sim.stats().substeps
        );
        assert!(drift <= 1e-5, "solid drift {drift}");
        if sediment_cone {
            assert!(p99_steep <= 37.0, "p99 {p99_steep}° above talus + 2°");
            assert!(max_steep <= 37.5, "max {max_steep}°");
        } else {
            assert!(
                (44.0..=47.0).contains(&p99_central),
                "a bedrock cone ends near 45°: {p99_central}°"
            );
        }
        sim.finish();
    }
}

/// Bake `duration_s` on a 256² tank under a forced thunderstorm at `quality` with `rain_mm_h` 0:
/// the frames until `TerrainBaked`, and the stats and the water volume after.
fn bake_case(quality: ElementsQuality, duration_s: f32) -> (usize, HydrologyStats, f64, Sim) {
    let n = 256usize;
    let mut config = Config::new(tank(256, 2.0));
    // A thunderstorm's 25 mm/h would fill the tank if the bake did not replace it.
    config.preset = WeatherPreset::Thunderstorm;
    config.mm_h = 25.0;
    config.erosion.enabled = false;
    let mut sim = Sim::at(config, quality);
    sim.app.world_mut().write_message(ErodeTerrain {
        duration_s,
        rain_mm_h: 0.0,
    });
    let frames = bake_frames(&mut sim);
    sim.settle();
    let stats = sim.stats();
    let water = sim.terrain.water.clone();
    let volume = sum(&sim.read_f32(&water, n * n));
    println!(
        "bake at {quality:?}: {frames} frames, {} substeps ({} dropped), {:.2} s simulated, water {volume:e} m³",
        stats.substeps, stats.dropped, stats.simulated_s
    );
    (frames, stats, volume, sim)
}

/// Erosion validation 6: a bake issues duration / (substeps · 0.02 s) frames, ignores the
/// weather's rain, and then says so. 600 s is 1875 frames at Ultra's 16 substeps and 7500 at
/// High's 4 (see Deviation 20 of the hydrology note); High is run for 100 s (1250 frames).
#[test]
#[ignore = "needs a GPU"]
fn a_bake_runs_its_duration_and_ignores_the_weather() {
    let _lock = serial();
    let (frames, stats, volume, sim) = bake_case(ElementsQuality::High, 100.0);
    assert!(
        (1249..=1258).contains(&frames),
        "{frames} frames for 1250 of 4 substeps"
    );
    assert_eq!(stats.substeps, 5_000);
    assert!(
        (stats.simulated_s - 100.0).abs() < 0.021,
        "{}",
        stats.simulated_s
    );
    assert_eq!(volume, 0.0, "weather rain fell during the bake");
    assert!(!stats.baking);
    sim.finish();

    let (frames, stats, volume, mut sim) = bake_case(ElementsQuality::Ultra, 600.0);
    assert!(
        (1874..=1883).contains(&frames),
        "{frames} frames for 1875 of 16 substeps"
    );
    assert_eq!(stats.substeps, 30_000);
    assert!(
        (stats.simulated_s - 600.0).abs() < 0.021,
        "{}",
        stats.simulated_s
    );
    assert_eq!(volume, 0.0, "weather rain fell during the bake");
    // Two more bakes queued behind each other run in order.
    sim.app.world_mut().write_message(ErodeTerrain {
        duration_s: 1.0,
        rain_mm_h: 20.0,
    });
    sim.app.world_mut().write_message(ErodeTerrain {
        duration_s: 2.0,
        rain_mm_h: 20.0,
    });
    sim.run_until("two more bakes", |app| {
        app.world().resource::<Baked>().0.len() == 3
    });
    let baked = sim.app.world().resource::<Baked>().0.clone();
    assert_eq!(baked, vec![600.0, 1.0, 2.0], "queued bakes run in order");
    sim.finish();
}

/// The valley of `examples/flood.rs`: V-shaped, running along +X at a 1 % bed slope.
fn valley(n: usize, cell: f32) -> Vec<f32> {
    let mut bed = Vec::with_capacity(n * n);
    for j in 0..n {
        for i in 0..n {
            let x = (i as f32 + 0.5 - n as f32 * 0.5) * cell;
            let z = (j as f32 + 0.5 - n as f32 * 0.5) * cell;
            bed.push(0.01 * (n as f32 * cell * 0.5 - x) + 0.2 * z.abs());
        }
    }
    bed
}

/// Hydrology validation 6 and 12: the `flood` expectation and the readback.
#[test]
#[ignore = "needs a GPU"]
fn a_flood_reaches_the_marker_and_stays() {
    let _lock = serial();
    let n = 256usize;
    let cells = n * n;
    let mut config = Config::new(tank(256, 2.0));
    // The default settings, held until the valley is in the bedrock layer.
    config.hydrology = HydrologySettings {
        time_scale: 0.0,
        ..default()
    };
    // The hydrology note's flood is water only: with erosion on, the river carves its bare
    // bedrock by up to 8 m in these 120 s and the front arrives after 63 s instead of 36 s.
    config.erosion.enabled = false;
    config.preset = WeatherPreset::Rain;
    config.mm_h = 4.0;
    config.sources = vec![(Vec2::new(-200.0, 0.0), 50.0, 5.0)];
    let mut sim = Sim::at(config, ElementsQuality::High);
    let bedrock = sim.terrain.bedrock.clone();
    sim.write(&bedrock, &valley(n, 2.0));
    sim.set_time_scale(1.0);
    let marker = Vec2::new(-100.0, 0.0);
    let mut samples: Vec<Option<f32>> = Vec::new();
    for frame in 0..(120 * 60) {
        sim.app.update();
        if frame % 60 == 0 {
            samples.push(sim.water_depth(marker));
        }
    }
    let line: Vec<String> = samples
        .iter()
        .map(|d| d.map_or("-".into(), |d| format!("{d:.2}")))
        .collect();
    println!("flood depth at the marker per second: {}", line.join(" "));
    let bed_now = sim.read_f32(&sim.terrain.bedrock.clone(), cells);
    let bed_then = valley(n, 2.0);
    let (mut carved, mut raised) = (0.0f32, 0.0f32);
    for (now, then) in bed_now.iter().zip(&bed_then) {
        carved = carved.max(then - now);
        raised = raised.max(now - then);
    }
    println!("flood bed change: carved at most {carved:.3} m, raised at most {raised:.3} m");
    assert!(
        carved == 0.0 && raised == 0.0,
        "the bed changed with erosion off"
    );
    assert_eq!(samples[0], None, "a depth before the first readback");
    let first_some = samples
        .iter()
        .position(Option::is_some)
        .expect("a readback");
    assert!(first_some <= 1, "first Some at {first_some} s");
    let first_wet = samples
        .iter()
        .position(|d| d.is_some_and(|d| d > 0.5))
        .expect("the flood never reached 0.5 m");
    println!("flood: the first sample above 0.5 m is at {first_wet} s");
    assert!(
        first_wet as f32 <= 100.0 / 2.5 + 2.0,
        "arrival at {first_wet} s"
    );
    // The front's ramp may show in the two samples before the crossing: each sample reads the
    // asynchronous readback, which lags the simulation by a load-dependent number of frames, so on
    // a busy GPU two consecutive samples can be less than a simulated second apart.
    for (t, d) in samples.iter().enumerate().take(first_wet) {
        assert!(
            d.is_none_or(|d| d < 1e-3 || t + 2 >= first_wet),
            "water at {t} s before the front"
        );
    }
    for (t, d) in samples.iter().enumerate().skip(first_wet) {
        assert!(
            d.is_some_and(|d| d >= 0.5),
            "depth {d:?} at {t} s after the crossing"
        );
    }

    // Readback: hold, let two intervals pass, compare with the buffer.
    sim.set_time_scale(0.0);
    sim.settle();
    sim.frames(60);
    let water = sim.terrain.water.clone();
    let depth = sim.read_f32(&water, cells);
    let grid = sim.app.world().resource::<TerrainState>().grid;
    let snapshot = sim.water_depth(marker).expect("a readback");
    let g = (marker - grid.origin) / grid.cell_m - 0.5;
    let (i0, j0) = (g.x.floor() as usize, g.y.floor() as usize);
    let (wx, wy) = (g.x - g.x.floor(), g.y - g.y.floor());
    let at = |i: usize, j: usize| depth[j * n + i];
    let want = (at(i0, j0) * (1.0 - wx) + at(i0 + 1, j0) * wx) * (1.0 - wy)
        + (at(i0, j0 + 1) * (1.0 - wx) + at(i0 + 1, j0 + 1) * wx) * wy;
    println!("readback: snapshot {snapshot:.4} m, buffer {want:.4} m");
    assert!((snapshot - want).abs() < 1e-3, "{snapshot} vs {want}");

    // Velocity at the marker against Q / A_wet of the same section.
    let velocity = sim
        .app
        .world_mut()
        .run_system_once(move |env: Environment| env.water_velocity(marker))
        .expect("the query runs")
        .expect("a velocity readback");
    let area: f64 = [i0, i0 + 1]
        .iter()
        .map(|i| (0..n).map(|j| f64::from(at(*i, j))).sum::<f64>() * 2.0)
        .sum::<f64>()
        / 2.0;
    let expected = 50.0 / area;
    println!(
        "velocity at the marker {velocity:?}, Q/A = {expected:.2} m/s over {area:.1} m² (stats lost {:.0} m³)",
        sim.stats().lost_m3
    );
    assert!(
        (velocity.x as f64 - expected).abs() <= 0.35 * expected,
        "{} vs {expected}",
        velocity.x
    );
    assert!(velocity.y.abs() < 0.2 * velocity.x.abs());
    sim.finish();
}

/// A bake requested before the terrain exists waits for it and then runs.
#[test]
#[ignore = "needs a GPU"]
fn a_bake_asked_for_before_the_terrain_exists_still_runs() {
    let _lock = serial();
    let mut config = Config::new(tank(64, 2.0));
    config.early_bake = Some(ErodeTerrain {
        duration_s: 4.0,
        rain_mm_h: 20.0,
    });
    let mut sim = Sim::new(config);
    bake_frames(&mut sim);
    sim.settle();
    let baked = sim.app.world().resource::<Baked>().0.clone();
    assert_eq!(baked, vec![4.0]);
    assert_eq!(sim.stats().substeps, 200);
    sim.finish();
}
