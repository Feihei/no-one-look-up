//! Trial gallery for bevy_flora's procedural trees (2026-10-30).
//!
//! The game's world trees are the in-house system (`world/tree_shape.rs` +
//! `world/tree_mesh.rs`, R3.9/R3.10: skeleton growth, leaf cards, three LOD
//! tiers, chunk streaming). This example spawns bevy_flora trees in a grid so
//! the two approaches can be compared by eye — crown shape, silhouette
//! density, species variety — before deciding whether the crate replaces or
//! complements the in-house trees. It is a trial, not world content: nothing
//! here feeds `chunks.rs`, and the example can be deleted without touching the
//! game.
//!
//! cargo run --release --example flora_gallery

use bevy::prelude::*;

use bevy_flora::{FloraPlugin, FloraWind, ProceduralTree, TreePreset};

/// The presets shown, one column each. Deciduous, so the silhouettes compare
/// against the game's own leaf-card trees (which are poplar / street /
/// broadleaf).
const PRESETS: [TreePreset; 4] = [
    TreePreset::Willow,
    TreePreset::Maple,
    TreePreset::Oak,
    TreePreset::Birch,
];

/// Seeds per preset (one row each). Deterministic: the same seed gives the
/// same tree, matching ADR-0002's determinism requirement.
const SEEDS: [i32; 4] = [7, 8, 9, 10];

/// Column/row spacing, m. Mature crowns reach ~10 m, so this keeps the
/// neighbouring trees from touching.
const SPACING: f32 = 14.0;

fn main() {
    App::new()
        .add_plugins((
            DefaultPlugins.set(WindowPlugin {
                primary_window: Some(Window {
                    title: "bevy_flora trial gallery".into(),
                    ..default()
                }),
                ..default()
            }),
            FloraPlugin,
        ))
        // Still air: this is a form reference, not a wind demo.
        .insert_resource(FloraWind {
            strength: 0.0,
            ..default()
        })
        .add_systems(Startup, setup)
        .run();
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    for (col, preset) in PRESETS.iter().enumerate() {
        for (row, seed) in SEEDS.iter().enumerate() {
            let mut params = preset.params();
            params.seed = *seed;
            let at = Vec3::new(col as f32 * SPACING, 0.0, row as f32 * SPACING);
            // No `Growing`: the default `TreeAge` is the mature tree, which is
            // what the world trees render as.
            commands.spawn((ProceduralTree(params), Transform::from_translation(at)));
            println!("{preset:?} seed {seed} at ({:.0}, {:.0})", at.x, at.z);
        }
    }

    // Ground, light and a camera at eye height, back from the first row and
    // centred on the grid.
    commands.spawn((
        Mesh3d(meshes.add(Plane3d::default().mesh().size(200.0, 200.0))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.25, 0.28, 0.22),
            ..default()
        })),
    ));
    commands.spawn((
        DirectionalLight {
            shadow_maps_enabled: true,
            illuminance: 10_000.0,
            ..default()
        },
        Transform::from_xyz(20.0, 40.0, 20.0).looking_at(Vec3::new(21.0, 0.0, 21.0), Vec3::Y),
    ));
    let centre = Vec3::new(SPACING * 1.5, 0.0, SPACING * 1.5);
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(centre.x, 1.65, centre.z - 30.0)
            .looking_at(centre + Vec3::Y * 4.0, Vec3::Y),
    ));
}
