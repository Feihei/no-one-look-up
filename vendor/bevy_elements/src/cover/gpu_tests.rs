//! GPU tests of the ground cover: snow accumulation and melt into hydrology's water layer, drift,
//! the shelter mask, wetness, and the map as `SurfaceWeathering` and as `Environment` answers.
//! See `docs/research/snow-cover.md` and `docs/research/wetness.md` (Validation targets).
//!
//! Needs a GPU: `cargo test --release --lib -- --ignored --test-threads=1 cover::`. The tests
//! share one process and each builds a renderer, so they take a lock and run one at a time.

#[allow(clippy::duplicate_mod)]
#[path = "../../tests/support/mod.rs"]
mod support;

use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;

use bevy::app::AppExit;
use bevy::ecs::system::RunSystemOnce;
use bevy::prelude::*;
use bevy::render::gpu_readback::{Readback, ReadbackComplete};
use bevy::render::storage::ShaderBuffer;
use bevy::time::TimeUpdateStrategy;

use super::mirror::{SnowCell, sheltered};
use super::{CoverGpu, CoverState, CoverStats, SnowSettings};
use crate::celestial::SunState;
use crate::climate::SurfaceWeathering;
use crate::terrain::{Terrain, TerrainState};
use crate::{
    CivilDate, Climate, ElementsCamera, ElementsPlugins, ElementsQuality, ElementsReadback,
    ElementsSystems, Environment, GeoLocation, HydrologyBoundary, HydrologySettings,
    PrecipitationBlocker, Weather, WorldClock,
};

static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());

fn serial() -> MutexGuard<'static, ()> {
    ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner())
}

const DT: f64 = 1.0 / 60.0;

/// What the test pins the weather to after the climate has run, each frame.
#[derive(Resource, Clone, Copy)]
struct Pin {
    mm_h: f32,
    temperature_c: f32,
    wind_ms: f32,
    relative_humidity: f32,
}

impl Pin {
    fn new(mm_h: f32, temperature_c: f32) -> Self {
        Self {
            mm_h,
            temperature_c,
            wind_ms: 0.0,
            relative_humidity: 0.5,
        }
    }

    fn wind(mut self, wind_ms: f32) -> Self {
        self.wind_ms = wind_ms;
        self
    }
}

fn pin_weather(pin: Res<Pin>, mut weather: ResMut<Weather>) {
    weather.precipitation_mm_h = pin.mm_h;
    weather.temperature_c = pin.temperature_c;
    weather.wind_speed_ms = pin.wind_ms;
    weather.wind_direction = Vec2::X;
    weather.relative_humidity = pin.relative_humidity;
}

#[derive(Clone)]
struct Config {
    terrain: Terrain,
    snow: SnowSettings,
    pin: Pin,
    /// Precipitation blocker boxes (centre, half extents).
    blockers: Vec<(Vec3, Vec3)>,
    /// Observer latitude: 85 on 21 December keeps the sun below the horizon all day.
    latitude_deg: f64,
}

impl Config {
    fn new(terrain: Terrain, pin: Pin) -> Self {
        Self {
            terrain,
            snow: SnowSettings {
                max_rate_hz: 0.0,
                ..default()
            },
            pin,
            blockers: Vec::new(),
            latitude_deg: 85.0,
        }
    }
}

#[derive(Resource, Clone)]
struct Spawn(Config);

fn setup(mut commands: Commands, spawn: Res<Spawn>, mut images: ResMut<Assets<Image>>) {
    commands.spawn((
        Camera3d::default(),
        support::image_target(&mut images),
        ElementsCamera::default(),
        Transform::from_xyz(0.0, 30.0, 0.0),
    ));
    commands.spawn(spawn.0.terrain.clone());
    for &(centre, half_extents) in &spawn.0.blockers {
        commands.spawn((
            PrecipitationBlocker { half_extents },
            Transform::from_translation(centre),
        ));
    }
}

fn tank(cells: u32, cell_m: f32) -> Terrain {
    Terrain::flat(0.0)
        .with_extent(Vec2::splat(cells as f32 * cell_m))
        .with_resolution(UVec2::splat(cells))
}

/// A running app with the terrain generated, the cover kernels live and the game clock held.
struct Sim {
    app: App,
    terrain: TerrainState,
    cover: CoverState,
    n: usize,
}

impl Sim {
    fn new(config: Config) -> Self {
        let mut app = App::new();
        app.add_plugins((support::headless_plugins(), ElementsPlugins))
            .insert_resource(TimeUpdateStrategy::ManualDuration(Duration::from_secs_f64(
                DT,
            )))
            .insert_resource(ElementsQuality::High)
            .insert_resource(HydrologySettings {
                boundary: HydrologyBoundary::Closed,
                evaporation: false,
                // The water layer only collects what the snow melts.
                time_scale: 0.0,
                ..default()
            })
            .insert_resource(ElementsReadback {
                interval: Duration::from_millis(1),
            })
            .insert_resource(config.snow.clone())
            .insert_resource(Climate {
                reference_elevation_m: 0.0,
                ..default()
            })
            .insert_resource(GeoLocation {
                latitude_deg: config.latitude_deg,
                ..default()
            })
            .insert_resource(WorldClock::new(CivilDate::new(2026, 12, 21)).with_time_scale(0.0))
            .insert_resource(config.pin)
            .insert_resource(Spawn(config))
            .add_systems(Startup, setup)
            .add_systems(
                Update,
                pin_weather
                    .after(ElementsSystems::Weather)
                    .before(ElementsSystems::Simulation),
            );
        support::finish(&mut app);
        run_until(&mut app, "the cover kernels", |app| {
            let world = app.world();
            world
                .get_resource::<TerrainState>()
                .is_some_and(|s| s.is_live())
                && world.get_resource::<CoverGpu>().is_some_and(|g| {
                    g.live.load(std::sync::atomic::Ordering::Relaxed)
                        && g.shelter_built.load(std::sync::atomic::Ordering::Relaxed) > 0
                })
        });
        let terrain = app.world().resource::<TerrainState>().clone();
        let cover = app.world().resource::<CoverState>().clone();
        let n = (terrain.grid.size.x * terrain.grid.size.y) as usize;
        Self {
            app,
            terrain,
            cover,
            n,
        }
    }

    fn stats(&self) -> CoverStats {
        self.app.world().resource::<CoverStats>().clone()
    }

    fn set_pin(&mut self, pin: Pin) {
        *self.app.world_mut().resource_mut::<Pin>() = pin;
    }

    fn set_scale(&mut self, scale: f32) {
        self.app.world_mut().resource_mut::<WorldClock>().time_scale = scale;
    }

    fn frames(&mut self, n: usize) {
        for _ in 0..n {
            self.app.update();
        }
    }

    /// Hold the clock and update until the step counter has not moved for 20 frames.
    fn hold(&mut self) {
        self.set_scale(0.0);
        let mut last = self.stats().steps;
        let mut quiet = 0;
        while quiet < 20 {
            self.app.update();
            let now = self.stats().steps;
            if now == last {
                quiet += 1;
            } else {
                quiet = 0;
                last = now;
            }
        }
    }

    /// Run the clock at `scale` until `simulated_s` of game time has been integrated, then hold.
    fn advance_to(&mut self, scale: f32, simulated_s: f64) {
        self.set_scale(scale);
        run_until(&mut self.app, "the game time", |app| {
            app.world().resource::<CoverStats>().simulated_s >= simulated_s
        });
        self.hold();
    }

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
        run_until(&mut self.app, "a buffer readback", |_| {
            slot.lock().expect("lock").is_some()
        });
        self.app.world_mut().despawn(entity);
        slot.lock()
            .expect("lock")
            .take()
            .expect("the readback completed")
    }

    fn read_f32(&mut self, handle: &Handle<ShaderBuffer>) -> Vec<f32> {
        let n = self.n;
        let bytes = self.read_bytes(handle);
        bytemuck::pod_collect_to_vec::<u8, f32>(&bytes[..n * 4])
    }

    fn read_vec4(&mut self, handle: &Handle<ShaderBuffer>) -> Vec<[f32; 4]> {
        let n = self.n;
        let bytes = self.read_bytes(handle);
        bytemuck::pod_collect_to_vec::<u8, [f32; 4]>(&bytes[..n * 16])
    }

    fn read_vec2(&mut self, handle: &Handle<ShaderBuffer>) -> Vec<[f32; 2]> {
        let n = self.n;
        let bytes = self.read_bytes(handle);
        bytemuck::pod_collect_to_vec::<u8, [f32; 2]>(&bytes[..n * 8])
    }

    fn snow(&mut self) -> Vec<[f32; 4]> {
        let handle = self.cover.snow.clone();
        self.read_vec4(&handle)
    }

    fn wetness(&mut self) -> Vec<f32> {
        let handle = self.cover.wetness.clone();
        self.read_f32(&handle)
    }

    fn water(&mut self) -> Vec<f32> {
        let handle = self.terrain.water.clone();
        self.read_f32(&handle)
    }

    fn finish(mut self) {
        self.app.world_mut().write_message(AppExit::Success);
    }
}

/// Update until `done` holds (at most 240 s of wall time).
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

fn mean(values: impl Iterator<Item = f32>) -> f32 {
    let (sum, n) = values.fold((0.0_f64, 0usize), |(s, n), v| (s + f64::from(v), n + 1));
    (sum / n as f64) as f32
}

fn close(got: f32, want: f32, tolerance: f32, what: &str) {
    assert!(
        (got - want).abs() <= tolerance * want.abs().max(1e-9),
        "{what}: got {got}, want {want} ± {}%",
        tolerance * 100.0
    );
}

/// Snow validation 1: ten hours of 1.5 mm/h snow lay 15 mm of water (0.1330 m deep after
/// settling), a 10 °C day melts it at 1.667 mm/h, and the melt arrives in hydrology's water layer.
#[test]
#[ignore = "needs a GPU"]
fn gpu_snow_accumulates_and_melts() {
    let _lock = serial();
    let cells = 64;
    let mut sim = Sim::new(Config::new(tank(cells, 2.0), Pin::new(1.5, -5.0)));
    let area_per_cell = 4.0_f64;

    // Phase 1: snowfall. 10 game hours.
    sim.advance_to(600.0, 36_000.0);
    let hours = sim.stats().simulated_s / 3600.0;
    let snow = sim.snow();
    let swe_mm = 1.5 * hours as f32;
    for (i, c) in snow.iter().enumerate() {
        assert!(
            (c[1] * 1000.0 - swe_mm).abs() <= 1e-3 * swe_mm,
            "cell {i}: SWE {} mm, want {swe_mm}",
            c[1] * 1000.0
        );
    }
    let depth = mean(snow.iter().map(|c| c[0]));
    let density = mean(snow.iter().map(|c| 1000.0 * c[1] / c[0]));
    println!(
        "RESULT snow 10 h: {hours:.4} h, SWE {:.4} mm, depth {depth:.4} m, density {density:.1} kg/m3",
        swe_mm
    );
    close(depth, 0.1330, 0.02, "depth after 10 h");
    assert!((0.1275..=0.1725).contains(&depth), "plan band: {depth}");
    close(density, 112.7, 0.01, "bulk density");
    let swe0 = mean(snow.iter().map(|c| c[1]));
    assert!(sim.water().iter().all(|w| *w == 0.0), "no melt yet");

    // Phase 2: a 10 °C clear day. Melt 4 mm/°C/day · 10 °C = 1.667 mm/h.
    sim.set_pin(Pin::new(0.0, 10.0));
    let base = sim.stats().simulated_s;
    for hours in [2.0, 4.5, 8.0] {
        sim.advance_to(600.0, base + hours * 3600.0);
        let elapsed_h = (sim.stats().simulated_s - base) / 3600.0;
        let want = swe0 - 1.666_667e-3 * elapsed_h as f32;
        let swe = mean(sim.snow().iter().map(|c| c[1]));
        println!(
            "RESULT melt {elapsed_h:.3} h: SWE {:.4} mm, formula {:.4} mm",
            swe * 1000.0,
            want * 1000.0
        );
        close(
            swe,
            want,
            0.02,
            &format!("SWE after {elapsed_h:.2} h of melt"),
        );
    }
    let depth_4_5 = {
        sim.advance_to(600.0, base + 8.0 * 3600.0);
        mean(sim.snow().iter().map(|c| c[0]))
    };
    assert!(depth_4_5 > 0.0);
    sim.advance_to(600.0, base + 9.01 * 3600.0);
    let snow = sim.snow();
    assert!(
        snow.iter()
            .all(|c| c[0] == 0.0 && c[1] == 0.0 && c[2] == 0.0),
        "melted out at 9.01 h"
    );
    sim.advance_to(2400.0, base + 48.0 * 3600.0);
    assert!(sim.snow().iter().all(|c| c[0] == 0.0 && c[1] == 0.0));

    // Every metre of melt water is in the water layer.
    let water = sim.water();
    let total_m3: f64 = water.iter().map(|w| f64::from(*w)).sum::<f64>() * area_per_cell;
    let want_m3 = f64::from(swe0) * area_per_cell * f64::from(sim.n as u32);
    assert!(
        (total_m3 / want_m3 - 1.0).abs() < 0.02,
        "water {total_m3} m³, melt {want_m3} m³"
    );
    let stats = sim.stats();
    println!(
        "RESULT melt: water {total_m3:.4} m3 of {want_m3:.4} m3 melted ({:.4} %), {} steps, {} dropped",
        100.0 * (total_m3 / want_m3 - 1.0),
        stats.steps,
        stats.dropped
    );
    assert_eq!(stats.dropped, 0);
    sim.finish();
}

/// Densification on the GPU follows the CPU cell model (snow validation 3): a layer filled in
/// one pass and left alone for three days.
#[test]
#[ignore = "needs a GPU"]
fn gpu_snow_settles_like_the_cell_model() {
    let _lock = serial();
    let mut sim = Sim::new(Config::new(tank(32, 2.0), Pin::new(0.0, -5.0)));
    let n = sim.n;
    let mut data = Vec::with_capacity(n * 4);
    for _ in 0..n {
        data.extend_from_slice(&[0.15, 0.015, 0.0, 0.0]);
    }
    let handle = sim.cover.snow.clone();
    sim.write(&handle, &data);
    let start = sim.stats().simulated_s;
    sim.advance_to(3600.0, start + 3.0 * 86_400.0);
    let seconds = sim.stats().simulated_s - start;
    let snow = sim.snow();
    let settings = SnowSettings::default();
    let mut cell = SnowCell {
        depth: 0.15,
        swe: 0.015,
        pending: 0.0,
    };
    let steps = 3 * 24 * 60;
    for _ in 0..steps {
        cell.step(
            &settings,
            (seconds / steps as f64 / 3600.0) as f32,
            0.0,
            -5.0,
            1.0,
        );
    }
    println!(
        "RESULT settle 3 d: GPU depth {:.5} m density {:.2}, CPU depth {:.5} m",
        snow[0][0],
        1000.0 * snow[0][1] / snow[0][0],
        cell.depth
    );
    close(snow[0][0], cell.depth, 0.005, "depth");
    close(
        1000.0 * snow[0][1] / snow[0][0],
        226.4,
        0.01,
        "density after 3 days",
    );
    sim.finish();
}

/// Snow validation 4a and 4c: no drift below the threshold; a Gaussian patch in 12 m/s wind
/// keeps its snow, loses none, and moves downwind.
#[test]
#[ignore = "needs a GPU"]
fn gpu_snow_drift_conserves_and_moves_downwind() {
    let _lock = serial();
    let cells = 128;
    let mut config = Config::new(tank(cells, 2.0), Pin::new(0.0, -5.0).wind(5.0));
    config.snow = SnowSettings {
        new_density: 200.0,
        settled_density: 200.0,
        max_rate_hz: 0.0,
        ..default()
    };
    let mut sim = Sim::new(config);
    let n = sim.n;
    let grid = sim.terrain.grid;
    let at = |i: usize| {
        let (x, z) = (i % cells as usize, i / cells as usize);
        grid.origin + (Vec2::new(x as f32, z as f32) + 0.5) * grid.cell_m
    };

    // (a) A uniform pack in 5 m/s wind, and in 12 m/s wind over flat ground, does not move.
    let uniform: Vec<f32> = (0..n).flat_map(|_| [0.1, 0.02, 0.0, 0.0]).collect();
    let handle = sim.cover.snow.clone();
    sim.write(&handle, &uniform);
    let start = sim.stats().steps;
    sim.set_scale(1800.0);
    run_until(&mut sim.app, "100 steps", |app| {
        app.world().resource::<CoverStats>().steps >= start + 100
    });
    sim.hold();
    // Below the threshold nothing moves: the pack is the one written, to the rounding of its
    // density (200.0000x kg/m³) each step.
    for (a, b) in sim.snow().iter().zip(uniform.chunks(4)) {
        close(a[0], b[0], 1e-5, "depth below the threshold");
        assert_eq!(a[1], b[1], "SWE below the threshold");
    }
    sim.set_pin(Pin::new(0.0, -5.0).wind(12.0));
    let start = sim.stats().steps;
    sim.set_scale(1800.0);
    run_until(&mut sim.app, "100 steps", |app| {
        app.world().resource::<CoverStats>().steps >= start + 100
    });
    sim.hold();
    // The edges are walls: the wind erodes the first column and piles the snow against the last,
    // and diffusion smooths both a little (the first and last ten columns are skipped);
    // every other cell keeps its snow to the rounding of 100 subtractions and additions.
    for (i, c) in sim.snow().iter().enumerate() {
        if (10..cells as usize - 10).contains(&(i % cells as usize)) {
            close(c[1], 0.02, 1e-4, "uniform pack in uniform wind");
        }
    }

    // (c) A Gaussian patch, sigma 20 m, peak 0.3 m, centred 40 m upwind of the middle.
    let patch: Vec<f32> = (0..n)
        .flat_map(|i| {
            let p = at(i) - Vec2::new(-40.0, 0.0);
            let d = 0.3 * (-p.length_squared() / (2.0 * 20.0 * 20.0)).exp();
            [d, 0.2 * d, 0.0, 0.0]
        })
        .collect();
    sim.write(&handle, &patch);
    let centre = |snow: &[[f32; 4]]| {
        let total: f64 = snow.iter().map(|c| f64::from(c[1])).sum();
        let (mut x, mut z) = (0.0_f64, 0.0_f64);
        for (i, c) in snow.iter().enumerate() {
            x += f64::from(c[1]) * f64::from(at(i).x);
            z += f64::from(c[1]) * f64::from(at(i).y);
        }
        (
            total,
            x / total,
            z / total,
            snow.iter().map(|c| f64::from(c[0])).sum::<f64>(),
        )
    };
    let before = centre(&sim.snow());
    let start = sim.stats().steps;
    sim.set_scale(1800.0);
    run_until(&mut sim.app, "1000 steps", |app| {
        app.world().resource::<CoverStats>().steps >= start + 1000
    });
    sim.hold();
    let snow = sim.snow();
    let after = centre(&snow);
    println!(
        "RESULT drift: SWE {:.5} -> {:.5} ({:.2e}), depth {:.5} -> {:.5} ({:.2e}), shift {:.2} m downwind, {:.3} m lateral",
        before.0,
        after.0,
        after.0 / before.0 - 1.0,
        before.3,
        after.3,
        after.3 / before.3 - 1.0,
        after.1 - before.1,
        after.2 - before.2
    );
    assert!(
        (after.0 / before.0 - 1.0).abs() < 1e-5,
        "SWE {} -> {}",
        before.0,
        after.0
    );
    assert!(
        (after.3 / before.3 - 1.0).abs() < 1e-5,
        "depth {} -> {}",
        before.3,
        after.3
    );
    let shift = after.1 - before.1;
    assert!(shift > 0.5, "the patch moved {shift} m downwind");
    assert!(
        (after.2 - before.2).abs() < 0.05 * shift,
        "lateral shift {}",
        after.2 - before.2
    );
    assert!(snow.iter().all(|c| c[0] >= 0.0 && c[1] >= 0.0));
    sim.finish();
}

/// Snow validation 4d: behind a 4 m drop in the surface the wind deposits snow in the lee and
/// takes it from the plateau; the total stays.
#[test]
#[ignore = "needs a GPU"]
fn gpu_snow_deposits_in_the_lee_of_a_step() {
    let _lock = serial();
    let cells = 128usize;
    let mut config = Config::new(tank(cells as u32, 2.0), Pin::new(0.0, -5.0));
    config.snow = SnowSettings {
        new_density: 200.0,
        settled_density: 200.0,
        max_rate_hz: 0.0,
        ..default()
    };
    let mut sim = Sim::new(config);
    let n = sim.n;
    let grid = sim.terrain.grid;
    let x_of = |i: usize| grid.origin.x + ((i % cells) as f32 + 0.5) * grid.cell_m;
    // The plateau (x < 0) is 4 m high; the wind blows toward +x over its edge.
    let bedrock: Vec<f32> = (0..n)
        .map(|i| if x_of(i) < 0.0 { 4.0 } else { 0.0 })
        .collect();
    let bedrock_handle = sim.terrain.bedrock.clone();
    sim.write(&bedrock_handle, &bedrock);
    let pack: Vec<f32> = (0..n)
        .flat_map(|i| {
            let x = x_of(i);
            if (-60.0..40.0).contains(&x) {
                [0.1, 0.02, 0.0, 0.0]
            } else {
                [0.0; 4]
            }
        })
        .collect();
    let handle = sim.cover.snow.clone();
    sim.write(&handle, &pack);
    // A few calm steps so the terrain pack holds the step.
    sim.set_scale(1800.0);
    sim.frames(10);
    sim.hold();
    let zone = |snow: &[[f32; 4]], lo: f32, hi: f32| -> f64 {
        snow.iter()
            .enumerate()
            .filter(|(i, _)| (lo..hi).contains(&x_of(*i)))
            .map(|(_, c)| f64::from(c[1]))
            .sum()
    };
    let before = sim.snow();
    sim.set_pin(Pin::new(0.0, -5.0).wind(10.0));
    let start = sim.stats().steps;
    sim.set_scale(1800.0);
    run_until(&mut sim.app, "100 steps", |app| {
        app.world().resource::<CoverStats>().steps >= start + 100
    });
    sim.hold();
    let after = sim.snow();
    let lee = zone(&after, 0.0, 10.0) - zone(&before, 0.0, 10.0);
    let plateau = zone(&after, -60.0, 0.0) - zone(&before, -60.0, 0.0);
    let total = zone(&after, -200.0, 200.0) / zone(&before, -200.0, 200.0) - 1.0;
    println!(
        "RESULT lee: lee {lee:+.5}, plateau {plateau:+.5}, total {total:+.2e} (SWE m summed over cells)"
    );
    assert!(lee > 0.0, "the lee gained {lee}");
    assert!(plateau < 0.0, "the plateau lost {plateau}");
    assert!(total.abs() < 1e-5, "total changed by {total}");
    sim.finish();
}

/// Snow validation 8 and wetness validation 5: the shelter mask equals the CPU slab test; no snow
/// or wetness under the roof, the flat value outside it.
#[test]
#[ignore = "needs a GPU"]
fn gpu_roofs_shelter_the_ground_from_snow_and_rain() {
    let _lock = serial();
    let roof = (Vec3::new(0.0, 5.0, 0.0), Vec3::new(5.0, 0.5, 5.0));
    // Snow, calm: 2 mm/h for an hour.
    let mut config = Config::new(tank(64, 1.0), Pin::new(2.0, -5.0));
    config.blockers = vec![roof];
    let mut sim = Sim::new(config);
    let grid = sim.terrain.grid;
    sim.advance_to(600.0, 3600.0);
    let hours = sim.stats().simulated_s / 3600.0;
    let snow = sim.snow();
    let flat = 2.0e-3 * hours as f32;
    let cells = grid.size.x as usize;
    for (i, c) in snow.iter().enumerate() {
        let p =
            grid.origin + (Vec2::new((i % cells) as f32, (i / cells) as f32) + 0.5) * grid.cell_m;
        let under = p.x.abs() <= 5.0 && p.y.abs() <= 5.0;
        if under {
            assert_eq!(c[1], 0.0, "snow under the roof at {p}");
        } else if p.x.abs() > 6.0 || p.y.abs() > 6.0 {
            close(c[1], flat, 0.01, "snow outside the roof");
        }
    }
    sim.finish();

    // Rain in 7 m/s wind: the mask follows the slab test along the slanted fall, and wetness
    // is 0 inside the band and 1 outside it.
    let mut config = Config::new(tank(64, 1.0), Pin::new(4.0, 15.0).wind(7.0));
    config.blockers = vec![roof];
    let mut sim = Sim::new(config);
    sim.advance_to(600.0, 3600.0);
    let gpu_shelter = {
        let handle = sim.app.world().resource::<CoverGpu>().shelter.clone();
        sim.read_vec2(&handle)
    };
    let wetness = sim.wetness();
    let rain_fall = crate::precipitation::model::water_weighted_fall_speed(
        crate::precipitation::model::marshall_palmer_lambda(4.0),
    );
    assert!((rain_fall - 4.39).abs() < 0.02, "{rain_fall}");
    let (mut sheltered_cells, mut mismatches) = (0, 0);
    for (i, s) in gpu_shelter.iter().enumerate() {
        let p =
            grid.origin + (Vec2::new((i % cells) as f32, (i / cells) as f32) + 0.5) * grid.cell_m;
        let cpu = sheltered(
            Vec3::new(p.x, 0.0, p.y),
            Vec2::new(7.0, 0.0),
            rain_fall,
            &[roof],
        );
        sheltered_cells += usize::from(s[0] < 0.5);
        mismatches += usize::from((s[0] < 0.5) != cpu);
        // A cell well inside or outside the band: the GPU's fall speed may differ from the CPU's
        // by a quantum, so only cells on the band's edge may disagree.
    }
    println!(
        "RESULT shelter: {sheltered_cells} sheltered cells, {mismatches} differ from the CPU slab test"
    );
    assert!(sheltered_cells > 100, "{sheltered_cells} sheltered cells");
    assert!(
        mismatches <= 6,
        "{mismatches} cells differ from the CPU slab test"
    );
    for (i, w) in wetness.iter().enumerate() {
        let p =
            grid.origin + (Vec2::new((i % cells) as f32, (i / cells) as f32) + 0.5) * grid.cell_m;
        if p.y.abs() < 4.0 && (4.0..12.0).contains(&p.x) {
            assert!(*w < 0.05, "wet at {p} under the slanted roof shadow: {w}");
        }
        if p.y.abs() < 4.0 && (p.x < 0.0 || p.x > 16.0) {
            assert!(*w > 0.95, "dry at {p} in the open: {w}");
        }
    }
    sim.finish();
}

/// Wetness validation 1: an hour of 4 mm/h rain soaks the ground; a clear 20 °C night at RH 0.5
/// and 3 m/s dries it from 1 to 0.1 in 11.25 h; `Environment::wetness` answers.
#[test]
#[ignore = "needs a GPU"]
fn gpu_wetness_soaks_and_dries_on_the_formula() {
    let _lock = serial();
    let mut rain = Pin::new(4.0, 20.0).wind(3.0);
    rain.relative_humidity = 0.5;
    let mut sim = Sim::new(Config::new(tank(32, 2.0), rain));
    sim.advance_to(600.0, 3600.0);
    for w in sim.wetness() {
        assert!((w - 1.0).abs() < 1e-3, "soaked: {w}");
    }
    sim.set_pin(Pin::new(0.0, 20.0).wind(3.0));
    let base = sim.stats().simulated_s;
    // Dry rate 0.05 · (1 + 3/5) · (1 + 20/20) · (1 − 0.5) = 0.08 per hour, constant.
    for hours in [10.0, 11.25, 12.5] {
        sim.advance_to(600.0, base + hours * 3600.0);
        let elapsed_h = ((sim.stats().simulated_s - base) / 3600.0) as f32;
        let want = (1.0 - 0.08 * elapsed_h).max(0.0);
        let got = mean(sim.wetness().into_iter());
        assert!(
            (got - want).abs() < 0.005,
            "wetness {got} after {elapsed_h} h, formula {want}"
        );
        println!("RESULT wetness {elapsed_h:.3} h of drying: {got:.4}, formula {want:.4}");
        match hours as u32 {
            10 => assert!(got > 0.1),
            12 => assert!(got < 0.1),
            _ => assert!((got - 0.1).abs() < 0.01),
        }
    }
    let probe = sim
        .app
        .world_mut()
        .run_system_once(|env: Environment| {
            (
                env.wetness(Vec2::ZERO),
                env.wetness(Vec2::splat(1e5)),
                env.snow_depth(Vec2::ZERO),
            )
        })
        .expect("the query runs");
    assert!(probe.0.is_some_and(|w| w < 0.1), "{probe:?}");
    assert_eq!(probe.1, None);
    assert_eq!(probe.2, Some(0.0));
    sim.finish();
}

/// Snow validation 7, wetness validation 8 and the queries: after the same weather the map's
/// mean around the camera is the point model, hour by hour.
#[test]
#[ignore = "needs a GPU"]
fn gpu_surface_weathering_follows_the_map() {
    let _lock = serial();
    let mut sim = Sim::new(Config::new(tank(64, 2.0), Pin::new(1.5, -5.0)));
    let mut point = SurfaceWeathering::default();
    let scale = 1200.0_f32;
    sim.set_scale(scale);
    let mut last_hours = sim.app.world().resource::<WorldClock>().hours_since_2000();
    let mut compared = 0;
    let mut worst = (0.0_f32, 0.0_f32, 0.0_f32);
    let schedule = [
        (10.0, Pin::new(1.5, -5.0)),
        (12.0, Pin::new(0.0, 10.0)),
        (1.0, Pin::new(4.0, 15.0)),
        (3.0, Pin::new(0.0, 20.0).wind(0.0)),
    ];
    let mut elapsed = 0.0;
    for (hours, pin) in schedule {
        sim.set_pin(pin);
        let end = elapsed + hours;
        let mut next_mark = (elapsed + 1.0_f64).floor();
        while elapsed < end {
            sim.app.update();
            let now = sim.app.world().resource::<WorldClock>().hours_since_2000();
            let dt = now - last_hours;
            last_hours = now;
            elapsed += dt;
            let weather = *sim.app.world().resource::<Weather>();
            let sun = sim
                .app
                .world()
                .resource::<SunState>()
                .altitude_deg
                .to_radians()
                .sin()
                .max(0.0);
            let pinned = Weather {
                precipitation_mm_h: pin.mm_h,
                temperature_c: pin.temperature_c,
                wind_speed_ms: pin.wind_ms,
                relative_humidity: pin.relative_humidity,
                ..weather
            };
            point.step(&pinned, sun, dt as f32);
            if elapsed >= next_mark {
                next_mark += 1.0;
                let ground = *sim.app.world().resource::<SurfaceWeathering>();
                worst.0 = worst
                    .0
                    .max((ground.snow_depth_m - point.snow_depth_m).abs());
                worst.1 = worst.1.max((ground.snow_cover - point.snow_cover).abs());
                worst.2 = worst.2.max((ground.wetness - point.wetness).abs());
                let depth_ok = (ground.snow_depth_m - point.snow_depth_m).abs()
                    <= 0.03 * point.snow_depth_m + 5e-4;
                assert!(
                    depth_ok,
                    "hour {elapsed:.2}: depth {} vs point {}",
                    ground.snow_depth_m, point.snow_depth_m
                );
                assert!(
                    (ground.snow_cover - point.snow_cover).abs() <= 0.02,
                    "hour {elapsed:.2}: cover {} vs {}",
                    ground.snow_cover,
                    point.snow_cover
                );
                assert!(
                    (ground.wetness - point.wetness).abs() <= 0.02,
                    "hour {elapsed:.2}: wetness {} vs {}",
                    ground.wetness,
                    point.wetness
                );
                compared += 1;
            }
        }
    }
    println!(
        "RESULT map vs point model over {compared} hours: depth {:.4} m (max abs {:.2e} m), cover (max abs {:.4}), wetness (max abs {:.4})",
        point.snow_depth_m, worst.0, worst.1, worst.2
    );
    assert!(compared >= 25, "{compared} hours compared");
    // The map is gone with the snow, and the query agrees with the resource.
    let ground = *sim.app.world().resource::<SurfaceWeathering>();
    assert_eq!(ground.snow_depth_m, 0.0);
    let depth = sim
        .app
        .world_mut()
        .run_system_once(|env: Environment| env.snow_depth(Vec2::new(3.0, 3.0)))
        .expect("the query runs");
    assert_eq!(depth, Some(0.0));
    sim.finish();
}
