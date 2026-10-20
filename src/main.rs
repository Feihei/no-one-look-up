//! Heads-Up: a first-person endless strolling experience focused on the
//! golden-hour sky. Bevy 0.20 migration of the original Godot 4.7 project.
//!
//! R0 skeleton: empty window launch with the plugin group wired. Feature
//! plugins are filled in per milestone (R1 sky, R2 world, R3 placement/
//! seasons, R4 save/menu).

mod core;
mod plugins;

use bevy::prelude::*;
use plugins::HeadsUpPlugins;

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
        .add_systems(Startup, setup_camera)
        .run()
}

fn setup_camera(mut commands: Commands) {
    commands.spawn(Camera3d::default());
}
