//! Heads-Up: a first-person endless strolling experience focused on the
//! golden-hour sky. Bevy 0.20 migration of the original Godot 4.7 project.
//!
//! R2: world generation + chunk streaming (ground tiles stream around the
//! camera; debug gizmos toggle with G). Sky per R1; placement/seasons (R3)
//! and save/menu (R4) per milestone.

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
        // PostUpdate, not Update: the aim must run AFTER bevy_elements'
        // `update_sun_state` (ElementsSystems::Celestial, in Update) so the
        // first aim reads the settled ephemeris instead of the default
        // overhead-sun state. Same pattern as the day_night example.
        .add_systems(PostUpdate, aim_at_sun_once)
        .run()
}

/// R2 scene: the golden-hour sky camera at street level over the streamed
/// world (chunk ground tiles spawn around the camera). The R1 verification
/// ground plane is gone — the world provides the floor. Replaced by the
/// player controller in R3.
fn setup_scene(mut commands: Commands, sky: Res<HeadsUpSky>) {
    // Spawn level at the world center; `aim_at_sun_once` turns the camera
    // toward the sun after the ephemeris settles.
    commands.spawn((
        Name::new("PlayerCamera"),
        Camera3d::default(),
        GoldenHourCamera {
            exposure: Exposure {
                ev100: sky.exposure_ev100,
            },
            ..default()
        },
        Transform::from_xyz(1000.0, 1.6, 1000.0),
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

/// Aims the camera at the sun once the ephemeris settles (first frame's
/// `SunState` is still the default), keeping the golden horizon band and sun
/// disc in frame. Replaced by the player controller in R3.
fn aim_at_sun_once(
    sun: Res<bevy_elements::SunState>,
    mut cameras: Query<&mut Transform, With<Camera3d>>,
    mut framed: Local<bool>,
) {
    if *framed {
        return;
    }
    let Ok(mut transform) = cameras.single_mut() else {
        return;
    };
    *framed = true;
    let dir = sun.direction.normalize_or(Vec3::NEG_Z);
    let yaw = (-dir.x).atan2(-dir.z);
    // Keep the horizon in frame: aim at the sun but never above 15°.
    let pitch = dir.y.clamp(-1.0, 1.0).asin().max(10f32.to_radians());
    *transform = Transform::from_xyz(1000.0, 1.6, 1000.0).with_rotation(Quat::from_euler(
        EulerRot::YXZ,
        yaw,
        pitch,
        0.0,
    ));
}
