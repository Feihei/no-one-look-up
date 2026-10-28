//! Snow falling over the test scene: flakes with a sideways flutter, drifting with the wind, and
//! the haze of the flakes too small to see.
//!
//! `cargo run --release --example snowfall`
//!
//! `ELEMENTS_PRESET=Blizzard` picks the heavier preset; `ELEMENTS_HOUR=23` pauses the clock at
//! that local hour (for screenshots).

mod common;

use std::time::Duration;

use bevy::prelude::*;
use bevy_elements::*;
use common::{ExampleHarness, camera, test_scene};

fn main() {
    let hour: Option<f64> = std::env::var("ELEMENTS_HOUR")
        .ok()
        .and_then(|h| h.parse().ok());
    let clock = WorldClock::new(CivilDate::new(2026, 1, 15)).at_hour(hour.unwrap_or(11.0));
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
    let preset = std::env::var("ELEMENTS_PRESET")
        .ok()
        .and_then(|name| {
            WeatherPreset::ALL
                .into_iter()
                .find(|p| format!("{p:?}").eq_ignore_ascii_case(&name))
        })
        .unwrap_or(WeatherPreset::Snow);
    director.force(preset, Duration::ZERO);
}
