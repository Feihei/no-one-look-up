//! Rain over the test scene, with a roof that shelters the ground under it.
//!
//! `cargo run --release --example rain`
//!
//! `ELEMENTS_PRESET=Thunderstorm` (or `Drizzle`, `Rain`) picks the weather; `ELEMENTS_HOUR=23`
//! pauses the clock at that local hour (for screenshots). The roof is a
//! [`PrecipitationBlocker`]: no rain falls under it and its top collects splashes.

mod common;

use std::time::Duration;

use bevy::prelude::*;
use bevy_elements::*;
use common::{ExampleHarness, camera, test_scene};

fn main() {
    let hour: Option<f64> = std::env::var("ELEMENTS_HOUR")
        .ok()
        .and_then(|h| h.parse().ok());
    let clock = WorldClock::new(CivilDate::new(2026, 4, 2)).at_hour(hour.unwrap_or(10.0));
    let clock = WorldClock {
        paused: hour.is_some(),
        time_scale: 1.0,
        ..clock
    };
    App::new()
        .add_plugins((DefaultPlugins, ElementsPlugins, ExampleHarness))
        .insert_resource(clock)
        .insert_resource(Climate::temperate())
        .add_systems(Startup, setup)
        .run();
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut director: ResMut<WeatherDirector>,
) {
    commands.spawn(camera(
        Vec3::new(0.0, 1.8, 0.0),
        Vec3::new(120.0, 6.0, 60.0),
    ));
    test_scene(&mut commands, &mut meshes, &mut materials);

    // A shelter 12 m ahead: four posts and a roof slab 3 m up.
    let wood = materials.add(StandardMaterial {
        base_color: Color::srgb(0.45, 0.32, 0.20),
        perceptual_roughness: 0.9,
        ..default()
    });
    let roof = Vec3::new(8.0, 3.0, 4.0);
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(6.0, 0.2, 6.0))),
        MeshMaterial3d(wood.clone()),
        Transform::from_translation(roof),
        PrecipitationBlocker {
            half_extents: Vec3::new(3.0, 0.1, 3.0),
        },
    ));
    let post = meshes.add(Cuboid::new(0.2, 3.0, 0.2));
    for (x, z) in [(-2.8, -2.8), (2.8, -2.8), (-2.8, 2.8), (2.8, 2.8)] {
        commands.spawn((
            Mesh3d(post.clone()),
            MeshMaterial3d(wood.clone()),
            Transform::from_translation(Vec3::new(roof.x + x, 1.5, roof.z + z)),
        ));
    }

    let preset = std::env::var("ELEMENTS_PRESET")
        .ok()
        .and_then(|name| {
            WeatherPreset::ALL
                .into_iter()
                .find(|p| format!("{p:?}").eq_ignore_ascii_case(&name))
        })
        .unwrap_or(WeatherPreset::Rain);
    director.force(preset, Duration::ZERO);
}
