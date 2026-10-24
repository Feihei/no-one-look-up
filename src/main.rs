//! Heads-Up: a first-person endless strolling experience focused on the
//! golden-hour sky. Bevy 0.20 migration of the original Godot 4.7 project.
//!
//! R2: world generation + chunk streaming (ground tiles stream around the
//! camera; debug gizmos toggle with G). R3.4: kinematic player controller
//! (A/D turn, W/S walk, mouse look clamped to -10°-60°, E/Q speed).
//! Placement/seasons (R3) and save/menu (R4) per milestone.

use bevy::camera::Exposure;
use bevy::prelude::*;

use heads_up_bevy::player::{EYE_HEIGHT, PlayerBody, PlayerPlugin};
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
            PlayerPlugin,
        ))
        .add_systems(Startup, (setup_scene, log_sky_state.after(setup_scene)))
        .run()
}

/// The player: one entity carrying both the camera and the kinematic body
/// (R3.4), spawned at the world centre on street level.
///
/// `PlayerBody` owns the camera's rotation from now on — the R1 `aim_at_sun`
/// helper is gone, because two systems writing one Transform is exactly the
/// implicit-ordering trap the project forbids. The controller starts pitched
/// at the top of the allowed band (60°) so the golden-hour sky fills the view
/// the way ADR-0001 describes.
fn setup_scene(mut commands: Commands, sky: Res<HeadsUpSky>) {
    commands.spawn((
        Name::new("Player"),
        Camera3d::default(),
        GoldenHourCamera {
            exposure: Exposure {
                ev100: sky.exposure_ev100,
            },
            ..default()
        },
        PlayerBody::default(),
        Transform::from_xyz(1000.0, EYE_HEIGHT, 1000.0),
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
