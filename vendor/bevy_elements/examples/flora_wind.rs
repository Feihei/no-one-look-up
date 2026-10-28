//! `bevy_flora` trees moved by this crate's wind, gusts, season and wetness.
//!
//! `cargo run --release --example flora_wind --features flora`
//!
//! A windy autumn rain: trees lean downwind, gust bands sweep through, leaves are turning and the
//! bark darkens as it gets wet.

mod common;

use std::time::Duration;

use bevy::prelude::*;
use bevy_elements::*;
use bevy_flora::{FloraPlugin, ProceduralTree, TreePreset};
use common::{ExampleHarness, camera, test_scene};

fn main() {
    let mut director = WeatherDirector::default();
    director.force(WeatherPreset::Rain, Duration::ZERO);
    App::new()
        .add_plugins((
            DefaultPlugins,
            ElementsPlugins,
            FloraPlugin,
            ElementsFloraPlugin,
            ExampleHarness,
        ))
        .insert_resource(
            WorldClock::new(CivilDate::new(2026, 10, 10))
                .at_hour(11.0)
                .with_time_scale(60.0),
        )
        .insert_resource(Climate {
            mean_wind_ms: 9.0,
            ..Climate::temperate()
        })
        .insert_resource(director)
        .add_systems(Startup, setup)
        .run();
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    commands.spawn(camera(
        Vec3::new(-30.0, 8.0, -30.0),
        Vec3::new(10.0, 6.0, 10.0),
    ));
    test_scene(&mut commands, &mut meshes, &mut materials);
    for i in 0..16 {
        let at = Vec3::new(
            (i % 4) as f32 * 12.0 - 18.0,
            0.0,
            (i / 4) as f32 * 12.0 - 18.0,
        );
        let preset = if i % 2 == 0 {
            TreePreset::Oak
        } else {
            TreePreset::Birch
        };
        commands.spawn((
            ProceduralTree(preset.params()),
            Transform::from_translation(at),
        ));
    }
}
