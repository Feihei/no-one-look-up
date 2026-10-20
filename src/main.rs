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
        // PostUpdate, not Update: the aim must run AFTER bevy_elements'
        // `update_sun_state` (ElementsSystems::Celestial, in Update) so the
        // first aim reads the settled ephemeris instead of the default
        // overhead-sun state. Same pattern as the day_night example.
        .add_systems(PostUpdate, aim_at_sun_once)
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
    // Spawn level; `aim_at_sun_once` turns the camera toward the sun after the
    // ephemeris settles. (A fixed 50° pitch only frames the high sky — the
    // horizon band needs a low sun-centered view.)
    commands.spawn((
        Name::new("PlayerCamera"),
        Camera3d::default(),
        GoldenHourCamera {
            exposure: Exposure {
                ev100: sky.exposure_ev100,
            },
            ..default()
        },
        Transform::from_xyz(0.0, 1.6, 0.0),
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
    *transform = Transform::from_xyz(0.0, 1.6, 0.0).with_rotation(Quat::from_euler(
        EulerRot::YXZ,
        yaw,
        pitch,
        0.0,
    ));
}
