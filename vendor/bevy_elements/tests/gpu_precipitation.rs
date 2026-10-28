//! Precipitation on a real GPU: particle counts, sheltering under a blocker, splashes, wind
//! coupling and the splash rate, read back from the particle and splash buffers.
//!
//! Needs a GPU: `cargo test --test gpu_precipitation -- --ignored`. The tests share one process
//! and each builds a renderer, so they take a lock and run one at a time.

mod support;

use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;

use bevy::app::AppExit;
use bevy::prelude::*;
use bevy::render::gpu_readback::{ReadbackComplete, ReadbackOnce};
use bevy::render::storage::ShaderBuffer;
use bevy::time::TimeUpdateStrategy;
use bevy_elements::*;

static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());

fn serial() -> MutexGuard<'static, ()> {
    ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner())
}

const DT: f64 = 1.0 / 60.0;

/// What the test pins the weather to after the climate has run, each frame.
#[derive(Resource, Clone, Copy)]
struct Pin {
    preset: WeatherPreset,
    mm_h: f32,
    temperature_c: f32,
}

#[derive(Resource, Clone, Copy)]
struct Scene {
    camera_y: f32,
    /// A blocker of half extents (5, 0.5, 5) this far above the camera.
    roof_above: Option<f32>,
}

struct Particle {
    pos: Vec3,
    kind: u32,
    vel: Vec3,
}

fn pin_weather(pin: Res<Pin>, mut weather: ResMut<Weather>) {
    weather.precipitation_mm_h = pin.mm_h;
    weather.temperature_c = pin.temperature_c;
}

fn setup(
    mut commands: Commands,
    pin: Res<Pin>,
    scene: Res<Scene>,
    mut director: ResMut<WeatherDirector>,
    mut images: ResMut<Assets<Image>>,
) {
    commands.spawn((
        Camera3d::default(),
        support::image_target(&mut images),
        ElementsCamera::default(),
        Transform::from_xyz(0.0, scene.camera_y, 0.0),
    ));
    if let Some(above) = scene.roof_above {
        commands.spawn((
            PrecipitationBlocker {
                half_extents: Vec3::new(5.0, 0.5, 5.0),
            },
            Transform::from_xyz(0.0, scene.camera_y + above, 0.0),
        ));
    }
    director.force(pin.preset, Duration::ZERO);
}

fn app_with(quality: ElementsQuality, climate: Climate, pin: Pin, scene: Scene) -> App {
    let mut app = App::new();
    app.add_plugins((support::headless_plugins(), ElementsPlugins))
        .init_resource::<PendingReadback>()
        .add_systems(Update, issue_readback)
        .insert_resource(TimeUpdateStrategy::ManualDuration(Duration::from_secs_f64(
            DT,
        )))
        .insert_resource(quality)
        .insert_resource(climate)
        .insert_resource(pin)
        .insert_resource(scene)
        .add_systems(Startup, setup)
        .add_systems(
            Update,
            pin_weather
                .after(ElementsSystems::Weather)
                .before(ElementsSystems::Simulation),
        );
    support::finish(&mut app);
    app
}

fn finish(mut app: App) {
    app.world_mut().write_message(AppExit::Success);
}

type Slot = Arc<Mutex<Option<Vec<u8>>>>;

/// A readback waiting to be issued: `ReadbackOnce` lives for one frame (Bevy removes it in
/// `First`), so it has to be spawned from a system that runs after `First`, not between updates.
#[derive(Resource, Default)]
struct PendingReadback(Mutex<Option<(Handle<ShaderBuffer>, Slot)>>);

fn issue_readback(mut commands: Commands, pending: Res<PendingReadback>) {
    let Some((handle, slot)) = pending.0.lock().unwrap_or_else(|e| e.into_inner()).take() else {
        return;
    };
    commands
        .spawn(ReadbackOnce::buffer(handle))
        .observe(move |done: On<ReadbackComplete>| {
            *slot.lock().unwrap_or_else(|e| e.into_inner()) = Some(done.data.to_vec());
        });
}

/// Read a whole buffer back, updating the app until it arrives.
fn read_buffer(app: &mut App, handle: Handle<ShaderBuffer>) -> Vec<u8> {
    let slot: Slot = Arc::default();
    *app.world()
        .resource::<PendingReadback>()
        .0
        .lock()
        .unwrap_or_else(|e| e.into_inner()) = Some((handle, slot.clone()));
    for _ in 0..240 {
        app.update();
        if let Some(bytes) = slot.lock().unwrap_or_else(|e| e.into_inner()).take() {
            return bytes;
        }
    }
    panic!("the readback never completed");
}

fn particles(app: &mut App) -> Vec<Particle> {
    let (handle, capacity) = {
        let state = app.world().resource::<PrecipitationState>();
        (state.particles.clone(), state.capacity as usize)
    };
    // Bevy's readback pool can hand back a larger buffer than was asked for.
    let bytes = read_buffer(app, handle);
    bytes[..capacity * 32]
        .as_chunks::<32>()
        .0
        .iter()
        .map(|c| {
            let f = |i: usize| f32::from_le_bytes(c[i * 4..i * 4 + 4].try_into().unwrap());
            Particle {
                pos: Vec3::new(f(0), f(1), f(2)),
                kind: u32::from_le_bytes(c[12..16].try_into().unwrap()),
                vel: Vec3::new(f(4), f(5), f(6)),
            }
        })
        .collect()
}

/// The splash counter and the ring's entries.
fn splashes(app: &mut App) -> (u32, Vec<Vec4>) {
    let handle = app
        .world()
        .resource::<PrecipitationState>()
        .splashes
        .clone();
    let bytes = read_buffer(app, handle);
    let counter = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
    let ring = bytes[16..16 + 4096 * 16]
        .as_chunks::<16>()
        .0
        .iter()
        .map(|c| {
            let f = |i: usize| f32::from_le_bytes(c[i * 4..i * 4 + 4].try_into().unwrap());
            Vec4::new(f(0), f(1), f(2), f(3))
        })
        .collect();
    (counter, ring)
}

fn run(app: &mut App, frames: usize) {
    for _ in 0..frames {
        app.update();
    }
}

fn count_kind(all: &[Particle], kind: u32) -> usize {
    all.iter().filter(|p| p.kind & 3 == kind).count()
}

/// Validation 1: forced Rain at 10 mm/h fills half the buffer with rain at every quality, and
/// every other record is parked; active particles stay inside the box above the ground.
#[test]
#[ignore = "needs a GPU"]
fn rain_at_ten_mm_per_hour_fills_half_the_buffer_at_every_quality() {
    let _lock = serial();
    for quality in [
        ElementsQuality::Low,
        ElementsQuality::Medium,
        ElementsQuality::High,
        ElementsQuality::Ultra,
    ] {
        let pin = Pin {
            preset: WeatherPreset::Rain,
            mm_h: 10.0,
            temperature_c: 12.0,
        };
        let scene = Scene {
            camera_y: 1.7,
            roof_above: None,
        };
        let mut app = app_with(quality, Climate::temperate(), pin, scene);
        run(&mut app, 60);
        let all = particles(&mut app);
        let max = quality.precipitation_particles() as usize;
        let rain = count_kind(&all, 0);
        let snow = count_kind(&all, 1);
        let parked = count_kind(&all, 3);
        println!("{quality:?}: {max} slots, {rain} rain, {snow} snow, {parked} parked");
        assert_eq!(all.len(), max);
        assert!(
            rain.abs_diff(max / 2) <= 1,
            "{quality:?}: {rain} rain particles, expected {}",
            max / 2
        );
        assert_eq!(snow, 0);
        assert_eq!(rain + parked, max, "every other record is parked");
        let state = app.world().resource::<PrecipitationState>();
        assert_eq!(state.rain as usize, rain);
        // Inside the 60 m box around the camera, above ground (a frame of fall of slack).
        for p in all.iter().filter(|p| p.kind & 3 == 0) {
            assert!(
                (p.pos.x.abs() <= 30.01)
                    && (p.pos.z.abs() <= 30.01)
                    && ((p.pos.y - 1.7).abs() <= 30.01),
                "{:?} outside the box",
                p.pos
            );
            assert!(p.pos.y > -0.2, "{:?} below ground", p.pos);
        }
        finish(app);
    }
}

/// Validation 3: forced Snow at 1.5 mm/h is half the buffer in snowflakes and no rain.
#[test]
#[ignore = "needs a GPU"]
fn snow_at_one_and_a_half_mm_per_hour_fills_half_the_buffer() {
    let _lock = serial();
    let pin = Pin {
        preset: WeatherPreset::Snow,
        mm_h: 1.5,
        temperature_c: -5.0,
    };
    let scene = Scene {
        camera_y: 1.7,
        roof_above: None,
    };
    let mut app = app_with(ElementsQuality::High, Climate::temperate(), pin, scene);
    run(&mut app, 60);
    let all = particles(&mut app);
    let (rain, snow) = (count_kind(&all, 0), count_kind(&all, 1));
    println!("snow: {rain} rain, {snow} snow of {}", all.len());
    assert_eq!(rain, 0);
    assert!(snow.abs_diff(75_000) <= 1, "{snow} snowflakes");
    // Flakes fall slowly: |v_y| near 1 m/s (wind adds only vertical turbulence).
    let flakes: Vec<_> = all.iter().filter(|p| p.kind & 3 == 1).collect();
    let mean_vy = flakes.iter().map(|p| p.vel.y).sum::<f32>() / flakes.len() as f32;
    assert!((mean_vy + 1.0).abs() < 0.3, "mean flake v_y {mean_vy}");
    finish(app);
}

/// Validation 2: a 10 × 1 × 10 m blocker 5 m over the camera holds no particle, shelters the
/// column below it, and its top face collects the splashes.
#[test]
#[ignore = "needs a GPU"]
fn a_blocker_over_the_camera_holds_no_particles_and_collects_splashes() {
    let _lock = serial();
    let pin = Pin {
        preset: WeatherPreset::Rain,
        mm_h: 10.0,
        temperature_c: 12.0,
    };
    let scene = Scene {
        camera_y: 1.7,
        roof_above: Some(5.0),
    };
    // No wind: the shelter is a plain column.
    let climate = Climate {
        mean_wind_ms: 0.0,
        ..Climate::temperate()
    };
    let mut app = app_with(ElementsQuality::High, climate, pin, scene);
    run(&mut app, 1200);
    let all = particles(&mut app);
    let centre = Vec3::new(0.0, 6.7, 0.0);
    let half = Vec3::new(5.0, 0.5, 5.0);
    let active: Vec<_> = all.iter().filter(|p| p.kind & 3 == 0).collect();
    let inside = active
        .iter()
        .filter(|p| (p.pos - centre).abs().cmple(half).all())
        .count();
    let sheltered = active
        .iter()
        .filter(|p| p.pos.x.abs() < 5.0 && p.pos.z.abs() < 5.0 && p.pos.y > 0.0 && p.pos.y < 6.2)
        .count();
    // The same column outside the footprint, for contrast: it is full of rain.
    let open = active
        .iter()
        .filter(|p| p.pos.x > 10.0 && p.pos.x < 20.0 && p.pos.z.abs() < 5.0 && p.pos.y < 6.2)
        .count();
    let (counter, ring) = splashes(&mut app);
    let on_roof = ring
        .iter()
        .filter(|s| s.w >= 0.0 && (s.y - 7.22).abs() < 0.005)
        .count();
    let on_ground = ring
        .iter()
        .filter(|s| s.w >= 0.0 && (s.y - 0.02).abs() < 0.005)
        .count();
    println!(
        "{} active, {inside} inside the blocker, {sheltered} in the sheltered column, {open} in an open column of equal size, splash counter {counter}, ring: {on_roof} on the roof, {on_ground} on the ground",
        active.len()
    );
    assert_eq!(inside, 0, "particles inside the blocker");
    assert_eq!(sheltered, 0, "rain under the roof");
    assert!(open > 100, "the open column has {open} particles");
    assert!(counter > 0, "no splashes");
    assert!(on_roof > 0, "no splash on the roof's top face");
    finish(app);
}

/// Validation 10: with 7 m/s wind, rain particles ride the wind field and fall at the
/// water-weighted speed.
#[test]
#[ignore = "needs a GPU"]
fn rain_rides_the_wind_and_falls_at_the_mean_speed() {
    let _lock = serial();
    let pin = Pin {
        preset: WeatherPreset::Rain,
        mm_h: 4.0,
        temperature_c: 12.0,
    };
    let scene = Scene {
        camera_y: 10.0,
        roof_above: None,
    };
    // Rain blows 7 m/s × mean/4: mean 4 m/s gives 7 m/s at 10 m.
    let climate = Climate {
        mean_wind_ms: 4.0,
        ..Climate::temperate()
    };
    let mut app = app_with(ElementsQuality::High, climate, pin, scene);
    run(&mut app, 240);
    let (speed, direction) = {
        let w = app.world().resource::<Weather>();
        (w.wind_speed_ms, w.wind_direction)
    };
    let all = particles(&mut app);
    let rain: Vec<_> = all.iter().filter(|p| p.kind & 3 == 0).collect();
    let n = rain.len() as f32;
    let along = rain
        .iter()
        .map(|p| p.vel.x * direction.x + p.vel.z * direction.y)
        .sum::<f32>()
        / n;
    let vy = rain.iter().map(|p| p.vel.y).sum::<f32>() / n;
    println!(
        "wind {speed} m/s along ({}, {}); mean along-wind velocity {along:.2}, mean v_y {vy:.2}",
        direction.x, direction.y
    );
    assert!((speed - 7.0).abs() < 1e-3, "preset wind {speed}");
    assert!(
        (along - 7.0).abs() < 1.5,
        "mean horizontal velocity {along}"
    );
    assert!((vy + 4.39).abs() < 0.3, "mean fall speed {vy}");
    finish(app);
}

/// Validation 11 (informational, asserted loosely): at High and 25 mm/h the splash counter
/// advances at about 9.3 k/s, below the ring's capacity at its lifetime (16.4 k/s).
#[test]
#[ignore = "needs a GPU"]
fn the_splash_rate_at_saturation_matches_the_estimate() {
    let _lock = serial();
    let pin = Pin {
        preset: WeatherPreset::Thunderstorm,
        mm_h: 25.0,
        temperature_c: 15.0,
    };
    let scene = Scene {
        camera_y: 1.7,
        roof_above: None,
    };
    let mut app = app_with(ElementsQuality::High, Climate::temperate(), pin, scene);
    run(&mut app, 120);
    let (first, _) = splashes(&mut app);
    let frame_a = app.world().resource::<Time<Virtual>>().elapsed_secs_f64();
    run(&mut app, 240);
    let (second, ring) = splashes(&mut app);
    let frame_b = app.world().resource::<Time<Virtual>>().elapsed_secs_f64();
    let rate = f64::from(second - first) / (frame_b - frame_a);
    let live = ring.iter().filter(|s| s.w >= 0.0).count();
    println!(
        "splash rate {rate:.0} /s over {:.2} s, {live} ring entries ever written",
        frame_b - frame_a
    );
    assert!(
        (7_000.0..=11_700.0).contains(&rate),
        "splash rate {rate} per second"
    );
    assert!(rate < 16_400.0, "the ring would be overrun");
    finish(app);
}

/// `dust.md` Validation 8: the forced dust storm fills 40 % of the buffer with dust (kind 2),
/// and no rain or snow; the sprites ride the wind, fall at 0.05 m/s and follow the haze's
/// exponential profile in height above the ground.
#[test]
#[ignore = "needs a GPU"]
fn a_dust_storm_fills_forty_percent_of_the_buffer_with_wind_driven_sprites() {
    let _lock = serial();
    let pin = Pin {
        preset: WeatherPreset::DustStorm,
        mm_h: 0.0,
        temperature_c: 30.0,
    };
    let scene = Scene {
        camera_y: 1.7,
        roof_above: None,
    };
    let mut app = app_with(ElementsQuality::High, Climate::desert(), pin, scene);
    run(&mut app, 240);
    let (speed, direction, dust_level) = {
        let w = app.world().resource::<Weather>();
        (w.wind_speed_ms, w.wind_direction, w.dust)
    };
    let all = particles(&mut app);
    let (rain, snow, dust) = (
        count_kind(&all, 0),
        count_kind(&all, 1),
        count_kind(&all, 2),
    );
    let parked = count_kind(&all, 3);
    println!(
        "dust {dust_level}: {rain} rain, {snow} snow, {dust} dust, {parked} parked of {}",
        all.len()
    );
    assert_eq!((rain, snow), (0, 0));
    assert!((dust_level - 0.9).abs() < 1e-3, "weather dust {dust_level}");
    assert!(dust.abs_diff(60_000) <= 1, "{dust} dust sprites");
    assert_eq!(dust + parked, all.len(), "every other record is parked");
    assert_eq!(
        app.world().resource::<PrecipitationState>().dust as usize,
        dust
    );
    // The dust records are the first after rain and snow: indices 0..dust.
    assert!(all[..dust].iter().all(|p| p.kind & 3 == 2));

    let sprites: Vec<_> = all.iter().filter(|p| p.kind & 3 == 2).collect();
    let n = sprites.len() as f32;
    let along = sprites
        .iter()
        .map(|p| p.vel.x * direction.x + p.vel.z * direction.y)
        .sum::<f32>()
        / n;
    let vy = sprites.iter().map(|p| p.vel.y).sum::<f32>() / n;
    println!("wind {speed} m/s: mean along-wind velocity {along:.2}, mean v_y {vy:.3}");
    assert!(
        (along - speed).abs() < 0.25 * speed,
        "mean horizontal velocity {along}"
    );
    assert!(vy.abs() < 0.5, "mean vertical velocity {vy}");
    // Heights above the ground plane follow the exponential of scale 60 m truncated to the part
    // of the box above it: 0 to 31.7 m for a camera at 1.7 m.
    let top = 1.7_f32 + 30.0;
    let mean_expected = 60.0 - top * (-top / 60.0).exp() / (1.0 - (-top / 60.0).exp());
    let mean_height = sprites.iter().map(|p| p.pos.y).sum::<f32>() / n;
    println!("mean sprite height {mean_height:.2} m, expected {mean_expected:.2} m");
    // The wind's vertical turbulence mixes the sprites between
    // respawns, so the stationary mean is within 15 % of the sampler's, not equal to it (the
    // sampler itself is pinned to 0.5 m on 10⁶ draws by `model.rs`).
    assert!(
        (mean_height - mean_expected).abs() < 0.15 * mean_expected,
        "{mean_height} vs {mean_expected}"
    );
    assert!(
        sprites
            .iter()
            .all(|p| p.pos.y > -0.2 && p.pos.y < top + 0.01)
    );
    finish(app);
}
