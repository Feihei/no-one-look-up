//! The smallest complete scene: the plugin group, a clock, a climate, a camera and rolling hills.
//!
//! `cargo run --release --example quickstart`

mod common;

use bevy::prelude::*;
use bevy_elements::*;
use common::{ExampleHarness, camera, stand_on_ground};

fn main() {
    App::new()
        .add_plugins((DefaultPlugins, ElementsPlugins, ExampleHarness))
        .insert_resource(
            WorldClock::new(CivilDate::new(2026, 6, 21))
                .at_hour(7.0)
                .with_time_scale(120.0),
        )
        .insert_resource(Climate::temperate())
        .add_systems(Startup, setup)
        .run();
}

fn setup(mut commands: Commands) {
    commands.spawn(Terrain::procedural(ProceduralTerrain::hills(42)));
    // The camera is lifted to 6 m above the ground as soon as the terrain's heights are known.
    commands.spawn((
        camera(Vec3::new(0.0, 160.0, 0.0), Vec3::new(150.0, 100.0, 80.0)),
        stand_on_ground(6.0),
    ));
}
