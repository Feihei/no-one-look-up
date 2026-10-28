//! Shared example scaffolding: a free-fly camera, clock and weather keys, a status line, and the
//! `ELEMENTS_FRAMES` / `ELEMENTS_SHOT` switches used for unattended runs.
//!
//! - `ELEMENTS_FRAMES=N` exits after N frames at a fixed 1/60 s step, so every run simulates and
//!   captures the same frames (GPU results are not bit-deterministic, but the simulated time,
//!   weather, clock and camera are). Without `ELEMENTS_FRAMES` the examples run on wall-clock time.
//! - `ELEMENTS_SHOT=/tmp/x.png` saves a screenshot of the primary window on the last frame, then exits.
//! - `ELEMENTS_GROW_HOURS=N` integrates `N` game hours of the forced weather before the first
//!   frame (see [`grow_weather`]): a still of a snowfall or a rain shows accumulated snow, wet
//!   ground and streaks instead of the first instant of the preset.
//! - `ELEMENTS_SHOT_DIR=/tmp/frames` saves a frame sequence to that directory (created if
//!   missing) as `frame0000.png` and up, `ELEMENTS_SHOT_EVERY=N` capturing every Nth frame
//!   (default 1) from `ELEMENTS_SHOT_FROM` (default 1). With `ELEMENTS_FRAMES` the simulated time
//!   is the same every run, which is what temporal effects (cloud accumulation, rain, lightning,
//!   wind, water flow, snow melt) are reviewed on. A run with `ELEMENTS_SHOT_DIR` and no
//!   `ELEMENTS_FRAMES` captures until it is killed.
//!
//! Keys: WASD + Q/E move (Shift ×5), hold the right mouse button to look, `[`/`]` halve/double the
//! time scale, `T` +1 hour, `1`–`0` force a weather preset, `R` release to the scheduler.

#![allow(dead_code)]

mod ground;

use std::time::Duration;

use ground::GroundPlane;

use bevy::input::mouse::AccumulatedMouseMotion;
use bevy::prelude::*;
use bevy::render::view::window::screenshot::{Screenshot, ScreenshotCaptured, save_to_disk};
use bevy::time::TimeUpdateStrategy;
use bevy_elements::*;

/// Adds the harness to an example.
pub struct ExampleHarness;

/// The free-fly camera.
#[derive(Component, Default)]
pub struct FlyCamera {
    yaw: f32,
    pitch: f32,
}

impl FlyCamera {
    /// Start looking along yaw/pitch, in radians.
    pub fn looking(yaw: f32, pitch: f32) -> Self {
        Self { yaw, pitch }
    }

    /// Turn the camera to yaw/pitch, in radians (an aim system pointing it at a body).
    pub fn face(&mut self, yaw: f32, pitch: f32) {
        self.yaw = yaw;
        self.pitch = pitch.clamp(-1.54, 1.54);
    }
}

#[derive(Resource)]
struct RunConfig {
    frames: Option<u32>,
    shot: Option<String>,
    shot_dir: Option<String>,
    shot_every: u32,
    shot_from: u32,
    frame: u32,
    shot_taken: bool,
    /// `ELEMENTS_GROW_HOURS`: game hours of the held weather to integrate before the first
    /// captured frame.
    grow_hours: f64,
    /// Screenshots spawned but not yet written; the run does not end over one.
    pending: std::sync::Arc<std::sync::atomic::AtomicU32>,
}

#[derive(Component)]
struct StatusLine;

impl Plugin for ExampleHarness {
    fn build(&self, app: &mut App) {
        let env_u32 = |name: &str, default: u32| {
            std::env::var(name)
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(default)
        };
        let frames: Option<u32> = std::env::var("ELEMENTS_FRAMES")
            .ok()
            .and_then(|v| v.parse().ok());
        let shot = std::env::var("ELEMENTS_SHOT").ok();
        let shot_dir = std::env::var("ELEMENTS_SHOT_DIR").ok();
        // A frame sequence is only worth reviewing when the frames are the same every run, so
        // either capture switch runs the app on a fixed 1/60 s step.
        if frames.is_some() || shot_dir.is_some() {
            app.insert_resource(TimeUpdateStrategy::ManualDuration(Duration::from_secs_f64(
                1.0 / 60.0,
            )));
        }
        app.insert_resource(RunConfig {
            frames,
            shot,
            shot_dir,
            shot_every: env_u32("ELEMENTS_SHOT_EVERY", 1).max(1),
            shot_from: env_u32("ELEMENTS_SHOT_FROM", 1).max(1),
            frame: 0,
            shot_taken: false,
            grow_hours: std::env::var("ELEMENTS_GROW_HOURS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.0),
            pending: std::sync::Arc::default(),
        })
        .add_systems(Startup, spawn_status)
        .add_systems(
            Update,
            (grow_weather, fly, clock_keys, weather_keys).before(ElementsSystems::Clock),
        )
        .add_systems(
            Update,
            (settle_on_ground, status, capture_frames).after(ElementsSystems::Readback),
        )
        .add_systems(Update, run_limit.after(capture_frames));
        ground::plugin(app);
    }
}

/// The standard example camera: an `ElementsCamera` that flies.
pub fn camera(position: Vec3, look_at: Vec3) -> impl Bundle {
    let dir = (look_at - position).normalize_or(Vec3::NEG_Z);
    let yaw = (-dir.x).atan2(-dir.z);
    let pitch = dir.y.clamp(-1.0, 1.0).asin();
    (
        Camera3d::default(),
        ElementsCamera::default(),
        FlyCamera::looking(yaw, pitch),
        Transform::from_translation(position).with_rotation(Quat::from_euler(
            EulerRot::YXZ,
            yaw,
            pitch,
            0.0,
        )),
    )
}

/// Marks a camera that moves to `height` metres above the terrain at its XZ once the terrain's
/// heights have been read back (the heights are not known on the first frames).
#[derive(Component)]
pub struct StandOnGround {
    height: f32,
}

/// See [`StandOnGround`].
pub fn stand_on_ground(height: f32) -> StandOnGround {
    StandOnGround { height }
}

fn settle_on_ground(
    mut commands: Commands,
    env: Environment,
    mut cameras: Query<(Entity, &mut Transform, &StandOnGround)>,
) {
    for (entity, mut transform, stand) in &mut cameras {
        if let Some(ground) = env.terrain_height(transform.translation.xz()) {
            transform.translation.y = ground + stand.height;
            commands.entity(entity).remove::<StandOnGround>();
        }
    }
}

fn fly(
    keys: Res<ButtonInput<KeyCode>>,
    mouse: Res<ButtonInput<MouseButton>>,
    motion: Res<AccumulatedMouseMotion>,
    time: Res<Time<Real>>,
    mut cameras: Query<(&mut Transform, &mut FlyCamera)>,
) {
    let dt = time.delta_secs();
    for (mut transform, mut fly) in &mut cameras {
        if mouse.pressed(MouseButton::Right) {
            fly.yaw -= motion.delta.x * 0.003;
            fly.pitch = (fly.pitch - motion.delta.y * 0.003).clamp(-1.54, 1.54);
            transform.rotation = Quat::from_euler(EulerRot::YXZ, fly.yaw, fly.pitch, 0.0);
        }
        let mut wish = Vec3::ZERO;
        let forward = transform.forward().as_vec3();
        let right = transform.right().as_vec3();
        for (key, dir) in [
            (KeyCode::KeyW, forward),
            (KeyCode::KeyS, -forward),
            (KeyCode::KeyD, right),
            (KeyCode::KeyA, -right),
            (KeyCode::KeyE, Vec3::Y),
            (KeyCode::KeyQ, -Vec3::Y),
        ] {
            if keys.pressed(key) {
                wish += dir;
            }
        }
        let speed = if keys.pressed(KeyCode::ShiftLeft) {
            60.0
        } else {
            12.0
        };
        transform.translation += wish.normalize_or_zero() * speed * dt;
    }
}

/// `ELEMENTS_GROW_HOURS`: integrate the held weather's game time before the first frame.
///
/// Wetness, snow and the cover maps advance on *game* time, and a held clock (`ELEMENTS_HOUR`)
/// adds none — so a still captured at the first frame of a preset shows bare ground under a
/// snowfall whatever the material does. This system advances the clock one chunk per frame and
/// then rewinds by exactly what it advanced: the last frame's delta comes back and the capture's
/// hour, sun and sky are the example's own.
///
/// Two details make the growth exact. The consumers record the clock on their first frame and
/// integrate nothing from it (their `last_hours` is `None`), so the first chunk is lost to them
/// and an extra arming frame is spent; and a jump larger than 24 h is clamped by both consumers,
/// so the growth is split into chunks of at most an hour. The weather is forced before the first
/// frame, so the whole warm-up is under the held preset, and a rewound clock adds no game time.
/// The cover maps drain their queue over the following frames, so a capture a few frames after
/// the rewind still sees the growth. Only the warm-up frames are at a moved hour — a frame
/// sequence should start past them.
fn grow_weather(
    config: Res<RunConfig>,
    mut clock: ResMut<WorldClock>,
    mut plan: Local<Option<(f64, u32)>>,
    mut advanced: Local<f64>,
) {
    if config.grow_hours <= 0.0 {
        return;
    }
    let Some(&mut (chunk, frames)) = plan.as_mut() else {
        let hours = config.grow_hours.ceil();
        // `ceil` frames of growth plus one arming frame; `grow_hours` is at most `ceil` hours, so
        // the growth is integrated in chunks of at most one hour and the consumers' clamps are
        // never reached.
        *plan = Some((config.grow_hours / hours, hours as u32 + 1));
        return;
    };
    if frames > 0 {
        clock.advance_hours(chunk);
        *advanced += chunk;
        *plan = Some((chunk, frames - 1));
        return;
    }
    let back = *advanced;
    clock.advance_hours(-back);
    *advanced = 0.0;
    *plan = Some((0.0, 0));
}

fn clock_keys(keys: Res<ButtonInput<KeyCode>>, mut clock: ResMut<WorldClock>) {
    if keys.just_pressed(KeyCode::BracketLeft) {
        clock.time_scale *= 0.5;
    }
    if keys.just_pressed(KeyCode::BracketRight) {
        clock.time_scale *= 2.0;
    }
    if keys.just_pressed(KeyCode::KeyT) {
        clock.advance_hours(1.0);
    }
}

fn weather_keys(keys: Res<ButtonInput<KeyCode>>, mut director: ResMut<WeatherDirector>) {
    const DIGITS: [KeyCode; 10] = [
        KeyCode::Digit1,
        KeyCode::Digit2,
        KeyCode::Digit3,
        KeyCode::Digit4,
        KeyCode::Digit5,
        KeyCode::Digit6,
        KeyCode::Digit7,
        KeyCode::Digit8,
        KeyCode::Digit9,
        KeyCode::Digit0,
    ];
    for (key, preset) in DIGITS.into_iter().zip(WeatherPreset::ALL) {
        if keys.just_pressed(key) {
            director.force(preset, Duration::from_secs(120));
        }
    }
    if keys.just_pressed(KeyCode::KeyR) {
        director.release(Duration::from_secs(120));
    }
}

fn spawn_status(mut commands: Commands) {
    commands.spawn((
        StatusLine,
        Text::new(""),
        TextFont {
            font_size: FontSize::Px(16.0),
            ..default()
        },
        TextColor(Color::WHITE),
        BackgroundColor(Color::srgba(0.0, 0.0, 0.0, 0.55)),
        Node {
            position_type: PositionType::Absolute,
            left: Val::Px(10.0),
            top: Val::Px(10.0),
            padding: UiRect::all(Val::Px(4.0)),
            ..default()
        },
    ));
}

/// A 40 km ground plane — out past the horizon the sky's own haze covers, so the ground meets the
/// sky without the sliver of unlit below-horizon atmosphere a short plane leaves between its edge
/// and the horizon (measured: a 35 DN dark shelf at the horizon of the 2 km plane under Overcast
/// and PartlyCloudy, gone with this) — and a ring of boxes, columns and spheres to read shadows
/// and fog depth. The plane gets `WeatheredMaterial` (cloud shadows, wetness, rain streaks and
/// snow) from this harness.
pub fn test_scene(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
) {
    commands.spawn((
        GroundPlane,
        Mesh3d(meshes.add(Plane3d::default().mesh().size(40000.0, 40000.0))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.30, 0.36, 0.22),
            perceptual_roughness: 0.95,
            ..default()
        })),
    ));
    let stone = materials.add(StandardMaterial {
        base_color: Color::srgb(0.55, 0.53, 0.50),
        perceptual_roughness: 0.8,
        ..default()
    });
    let white = materials.add(StandardMaterial {
        base_color: Color::srgb(0.85, 0.85, 0.85),
        perceptual_roughness: 0.4,
        ..default()
    });
    let cube = meshes.add(Cuboid::new(6.0, 6.0, 6.0));
    let column = meshes.add(Cylinder::new(1.5, 24.0));
    let ball = meshes.add(Sphere::new(4.0));
    for i in 0..24 {
        let angle = i as f32 / 24.0 * std::f32::consts::TAU;
        let radius = 40.0 + (i % 4) as f32 * 60.0;
        let at = Vec3::new(angle.cos() * radius, 0.0, angle.sin() * radius);
        let (mesh, material, y) = match i % 3 {
            0 => (cube.clone(), stone.clone(), 3.0),
            1 => (column.clone(), stone.clone(), 12.0),
            _ => (ball.clone(), white.clone(), 4.0),
        };
        commands.spawn((
            Mesh3d(mesh),
            MeshMaterial3d(material),
            ground::ShowcaseProp,
            Transform::from_translation(at + Vec3::Y * y),
        ));
    }
}

fn status(
    env: Environment,
    director: Res<WeatherDirector>,
    mut text: Query<&mut Text, With<StatusLine>>,
) {
    let Ok(mut text) = text.single_mut() else {
        return;
    };
    let clock = env.clock();
    let weather = env.weather();
    let sun = env.sun();
    text.0 = format!(
        "{:04}-{:02}-{:02} {:05.2} h x{}  |  {:?}{}  |  {:.1} C  cover {:.2}  {:.1} mm/h  wind {:.1} m/s  vis {:.0} m  |  sun {:.1} deg  moon {:.1} deg ({:.0}%)",
        clock.date.year,
        clock.date.month,
        clock.date.day,
        clock.hour,
        clock.time_scale,
        director.current(),
        if director.is_forced() {
            " (forced)"
        } else {
            ""
        },
        env.temperature_c(),
        weather.cloud_cover,
        weather.precipitation_mm_h,
        weather.wind_speed_ms,
        weather.visibility_m,
        sun.altitude_deg,
        sun.moon_altitude_deg,
        sun.moon_illuminated_fraction * 100.0,
    );
}

/// Writes a frame sequence while the run lasts: one screenshot every `shot_every` frames from
/// `shot_from`, as `frameNNNN.png` in `shot_dir`. The counter is counted from `shot_from`, so a
/// sequence of frames 301..360 is `frame0000.png`..`frame0059.png` and two runs' sequences
/// compare frame for frame.
fn capture_frames(
    mut commands: Commands,
    config: Res<RunConfig>,
    mut seen: Local<u32>,
    mut index: Local<u32>,
) {
    let Some(dir) = config.shot_dir.clone() else {
        return;
    };
    *seen += 1;
    if *seen < config.shot_from || !(*seen - config.shot_from).is_multiple_of(config.shot_every) {
        return;
    }
    if config.frames.is_some_and(|limit| *seen > limit) {
        return;
    }
    if *index == 0
        && let Err(error) = std::fs::create_dir_all(&dir)
    {
        error!("cannot create frame directory {dir}: {error}");
        return;
    }
    let path = format!("{dir}/frame{:04}.png", *index);
    *index += 1;
    let pending = config.pending.clone();
    pending.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    commands
        .spawn(Screenshot::primary_window())
        .observe(move |captured: On<ScreenshotCaptured>| {
            save_to_disk(path.clone())(captured);
            pending.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
        });
}

fn run_limit(
    mut config: ResMut<RunConfig>,
    mut commands: Commands,
    mut exit: MessageWriter<AppExit>,
    mut waited: Local<u32>,
) {
    config.frame += 1;
    let Some(limit) = config.frames else {
        return;
    };
    if config.frame < limit {
        return;
    }
    // Every capture is written before the app exits; a screenshot takes a frame or two to
    // arrive. Give up after two seconds' worth of frames.
    let pending = config.pending.load(std::sync::atomic::Ordering::Relaxed);
    if pending > 0 {
        *waited += 1;
        if *waited < 120 {
            return;
        }
    }
    match config.shot.clone() {
        Some(path) if !config.shot_taken => {
            config.shot_taken = true;
            let mut save = save_to_disk(path);
            commands.spawn(Screenshot::primary_window()).observe(
                move |captured: On<ScreenshotCaptured>, mut exit: MessageWriter<AppExit>| {
                    exit.write(AppExit::Success);
                    save(captured);
                },
            );
            // Wait for the single shot too, rather than exiting over it.
            *waited = 0;
        }
        Some(_) if *waited < 120 => *waited += 1,
        _ => {
            exit.write(AppExit::Success);
        }
    }
}
