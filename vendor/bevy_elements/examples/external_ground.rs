//! A consumer's own ground mesh shaded by the crate: a plain plane with `GroundExtension`, under a
//! terrain the crate only simulates (`Terrain::external()`). The plane gets cloud shadows by
//! world position, and darkens where the simulation puts water over it.
//!
//! `cargo run --release --example external_ground`

mod common;

use bevy::prelude::*;
use bevy_elements::*;
use common::{ExampleHarness, camera};

fn main() {
    App::new()
        .add_plugins((DefaultPlugins, ElementsPlugins, ExampleHarness))
        .insert_resource(
            WorldClock::new(CivilDate::new(2026, 6, 21))
                .at_hour(10.0)
                .with_time_scale(1.0),
        )
        .insert_resource(Climate::temperate())
        .add_systems(Startup, setup)
        .run();
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut standard: ResMut<Assets<StandardMaterial>>,
    mut grounds: ResMut<Assets<GroundMaterial>>,
    mut director: ResMut<WeatherDirector>,
) {
    director.force(WeatherPreset::PartlyCloudy, std::time::Duration::ZERO);
    // The crate simulates this flat terrain; the consumer draws the ground.
    commands.spawn(Terrain::flat(0.0).external());
    commands.spawn((
        Mesh3d(meshes.add(Plane3d::default().mesh().size(2048.0, 2048.0))),
        MeshMaterial3d(grounds.add(GroundMaterial {
            base: StandardMaterial {
                base_color: Color::srgb(0.30, 0.36, 0.22),
                perceptual_roughness: 0.95,
                ..default()
            },
            extension: GroundExtension::default(),
        })),
    ));
    let stone = standard.add(StandardMaterial {
        base_color: Color::srgb(0.55, 0.53, 0.50),
        perceptual_roughness: 0.8,
        ..default()
    });
    let cube = meshes.add(Cuboid::new(6.0, 6.0, 6.0));
    for i in 0..24 {
        let angle = i as f32 / 24.0 * std::f32::consts::TAU;
        let radius = 40.0 + (i % 4) as f32 * 60.0;
        commands.spawn((
            Mesh3d(cube.clone()),
            MeshMaterial3d(stone.clone()),
            Transform::from_xyz(angle.cos() * radius, 3.0, angle.sin() * radius),
        ));
    }
    commands.spawn(camera(
        Vec3::new(0.0, 30.0, 0.0),
        Vec3::new(150.0, 10.0, 80.0),
    ));
}
