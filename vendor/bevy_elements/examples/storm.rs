//! A thunderstorm: the Thunderstorm preset forced, lightning striking around the camera, and a
//! line printed for every strike and every clap of thunder.
//!
//! `cargo run --release --example storm`
//!
//! - `ELEMENTS_HOUR=12` pauses the clock at that local hour (default: 23:00, paused).
//! - `ELEMENTS_STRIKE=1500` is for screenshots: time steps by exactly 1/60 s per frame and one
//!   scripted strike hits that many metres ahead of the camera on the last frame
//!   (`ELEMENTS_FRAMES`, default 600), so `ELEMENTS_SHOT` lands on the first stroke.
//!   `ELEMENTS_STRIKE=0` keeps the fixed time step and scripts no strike (a calm control).
//!   `ELEMENTS_STRIKE_LEAD=6` scripts the strike that many frames before the last one, so the
//!   shot lands inside the flash rather than on its first frame.

mod common;

use std::time::Duration;

use bevy::camera::PerspectiveProjection;
use bevy::prelude::*;
use bevy::time::TimeUpdateStrategy;
use bevy_elements::*;
use common::{ExampleHarness, camera, test_scene};

/// The camera, m: eye height above the test scene's ground.
const EYE: Vec3 = Vec3::new(0.0, 6.0, 0.0);

#[derive(Resource)]
struct ScriptedStrike {
    distance_m: f32,
    on_frame: u32,
    frame: u32,
}

fn main() {
    let hour: f64 = std::env::var("ELEMENTS_HOUR")
        .ok()
        .and_then(|h| h.parse().ok())
        .unwrap_or(23.0);
    let strike: Option<f32> = std::env::var("ELEMENTS_STRIKE")
        .ok()
        .and_then(|d| d.parse().ok());
    let frames: u32 = std::env::var("ELEMENTS_FRAMES")
        .ok()
        .and_then(|n| n.parse().ok())
        .unwrap_or(600);
    let lead: u32 = std::env::var("ELEMENTS_STRIKE_LEAD")
        .ok()
        .and_then(|n| n.parse().ok())
        .unwrap_or(0);
    let mut director = WeatherDirector::default();
    director.force(WeatherPreset::Thunderstorm, Duration::ZERO);
    let mut app = App::new();
    app.add_plugins((DefaultPlugins, ElementsPlugins, ExampleHarness))
        .insert_resource(WorldClock {
            paused: true,
            ..WorldClock::new(CivilDate::new(2026, 6, 21))
                .at_hour(hour)
                .with_time_scale(1.0)
        })
        .insert_resource(director)
        .add_systems(Startup, setup)
        .add_systems(Update, print.after(ElementsSystems::Simulation));
    if let Some(distance_m) = strike {
        app.insert_resource(TimeUpdateStrategy::ManualDuration(Duration::from_secs_f64(
            1.0 / 60.0,
        )))
        .insert_resource(ScriptedStrike {
            distance_m,
            on_frame: frames.saturating_sub(lead).max(1),
            frame: 0,
        })
        .add_systems(Update, script.before(ElementsSystems::Clock));
    }
    app.run();
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    // Looking along +x and 22 degrees up, so a bolt 1.5 km away shows from the ground to the
    // cloud base.
    commands.spawn((
        camera(EYE, EYE + Vec3::new(1000.0, 400.0, 0.0)),
        Projection::Perspective(PerspectiveProjection {
            fov: 80f32.to_radians(),
            ..default()
        }),
    ));
    test_scene(&mut commands, &mut meshes, &mut materials);
    // Ground out to the horizon, just under the test scene's 2 km plane.
    commands.spawn((
        Mesh3d(meshes.add(Plane3d::default().mesh().size(40_000.0, 40_000.0))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.22, 0.27, 0.16),
            perceptual_roughness: 0.95,
            ..default()
        })),
        Transform::from_xyz(0.0, -0.05, 0.0),
    ));
}

fn script(mut scripted: ResMut<ScriptedStrike>, mut strikes: MessageWriter<StrikeLightning>) {
    scripted.frame += 1;
    if scripted.frame == scripted.on_frame && scripted.distance_m > 0.0 {
        strikes.write(StrikeLightning {
            at: Vec3::new(scripted.distance_m, 0.0, 0.0),
        });
    }
}

fn print(
    mut strikes: MessageReader<LightningStrike>,
    mut thunder: MessageReader<Thunder>,
    time: Res<Time<Real>>,
) {
    let now = time.elapsed_secs();
    for strike in strikes.read() {
        println!(
            "[{now:7.2} s] lightning at ({:.0}, {:.0}, {:.0})",
            strike.position.x, strike.position.y, strike.position.z
        );
    }
    for clap in thunder.read() {
        println!(
            "[{now:7.2} s] thunder: {:.0} m away, loudness {:.2}",
            clap.distance_m, clap.loudness
        );
    }
}
