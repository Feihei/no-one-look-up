//! Heads-Up: a first-person endless strolling experience focused on the
//! golden-hour sky. Bevy 0.20 migration of the original Godot 4.7 project.
//!
//! R1: golden-hour sky (bevy_elements). World/plugins per milestone
//! (R2 world, R3 placement/seasons, R4 save/menu).

use bevy::camera::Exposure;
use bevy::prelude::*;

use heads_up_bevy::plugins::{GoldenHourCamera, HeadsUpPlugins, sky::HeadsUpSky};

fn main() -> AppExit {
    App::new()
        .add_plugins((
            DefaultPlugins.set(WindowPlugin {
                primary_window: Some(Window {
                    title: "Heads-Up".to_string(),
                    ..default()
                }),
                ..default()
            }),
            HeadsUpPlugins,
        ))
        .add_systems(Startup, (setup_scene, log_sky_state.after(setup_scene)))
        .add_systems(Update, report_sun_altitude)
        .run()
}

/// R1 verification scene: the sky camera looking up from street level plus a
/// ground plane for light/IBL context. Replaced by the world in R2.
fn setup_scene(
    mut commands: Commands,
    sky: Res<HeadsUpSky>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    // Look 50° up (inside the design band of 40–60° elevation, adr-0001) at a
    // heading roughly opposite the evening sun so the gradient is in frame.
    commands.spawn((
        Name::new("PlayerCamera"),
        Camera3d::default(),
        GoldenHourCamera {
            exposure: Exposure {
                ev100: sky.exposure_ev100,
            },
            ..default()
        },
        Transform::from_xyz(0.0, 1.6, 0.0).with_rotation(Quat::from_euler(
            EulerRot::YXZ,
            0.0,
            50f32.to_radians(),
            0.0,
        )),
    ));

    // Simple asphalt-toned ground plane for IBL/fog reference.
    commands.spawn((
        Name::new("GroundPlane"),
        Mesh3d(meshes.add(Plane3d::default().mesh().size(4000.0, 4000.0))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.08, 0.08, 0.09),
            perceptual_roughness: 0.95,
            ..default()
        })),
        Transform::from_xyz(0.0, 0.0, 0.0),
    ));
}

/// One-shot startup log: confirms the golden-hour lock took effect. The first
/// frame's `SunState` is still the default (update runs after Startup), so
/// `report_sun_altitude` shows the settled ephemeris.
fn log_sky_state(sky: Res<HeadsUpSky>) {
    info!(
        "golden hour locked: date={:?} hour={} loc=({:.4}, {:.4}, UTC+{})",
        sky.date,
        sky.golden_hour,
        sky.location.latitude_deg,
        sky.location.longitude_deg,
        sky.location.utc_offset_hours
    );
}

/// Continuous sun-altitude report during R1 tuning (rate-limited).
fn report_sun_altitude(
    sun: Res<bevy_elements::SunState>,
    time: Res<Time>,
    mut next_log: Local<f32>,
) {
    if time.elapsed_secs() >= *next_log {
        *next_log = time.elapsed_secs() + 2.0;
        info!(
            "sun_alt={:.2}° moon_alt={:.2}°",
            sun.altitude_deg, sun.moon_altitude_deg
        );
    }
}
