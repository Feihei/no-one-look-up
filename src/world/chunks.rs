//! Chunk streaming (R2): 50 m chunks spawned/despawned around the player
//! (ADR-0002, second generation phase).
//!
//! The [`ChunkManager`] resource tracks the active set; every chunk's
//! entities are grouped under one root entity (children), so despawn removes
//! the whole group recursively (Bevy 0.20 `despawn` despawns `Children`).
//! Chunk content in R2 is a ground tile per chunk (seed-tinted asphalt) for
//! streaming feedback; buildings/trees arrive with R3 placement.
//!
//! The active set is an oriented footprint around the walk direction:
//! `CHUNK_AHEAD` chunks in front, `CHUNK_BEHIND` behind, `CHUNK_SIDE` to each
//! side — about 49 chunks at peak (see `desired_chunks`).

use std::collections::HashMap;

use bevy::prelude::*;

use crate::core::input::{MOVE_BACKWARD, MOVE_FORWARD, TURN_LEFT, TURN_RIGHT};
use crate::world::layout_graph::LayoutGraph;
use crate::world::path_network::{HeadsUpSeed, generate_world};
use crate::world::{CHUNK_AHEAD, CHUNK_BEHIND, CHUNK_SIDE, CHUNK_SIZE, GRID_CELLS};

/// Default seed until the R4 menu provides one (dev builds).
pub const DEFAULT_SEED: u64 = 0x4845_4144; // "HEAD"

/// Chunk grid coordinates. Chunk `(x, z)` covers the world square
/// `[x * CHUNK_SIZE, (x + 1) * CHUNK_SIZE) x [z * CHUNK_SIZE, (z + 1) *
/// CHUNK_SIZE)`; 40 x 40 chunks tile the 2 km world.
#[derive(Component, Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ChunkCoords {
    pub x: i32,
    pub z: i32,
}

impl ChunkCoords {
    /// Center of the chunk in world coordinates (ground plane, meters).
    pub fn center(self) -> Vec2 {
        Vec2::new(
            (self.x as f32 + 0.5) * CHUNK_SIZE,
            (self.z as f32 + 0.5) * CHUNK_SIZE,
        )
    }

    /// Chunk index for a world position.
    pub fn from_world(pos: Vec2) -> Self {
        Self {
            x: (pos.x / CHUNK_SIZE).floor() as i32,
            z: (pos.y / CHUNK_SIZE).floor() as i32,
        }
    }
}

/// Marker on a chunk group root: despawning the root removes every per-chunk
/// entity (ground tile, later placement content) recursively.
#[derive(Component, Debug, Default)]
pub struct ChunkRoot;

/// Tracks the active chunks and their group roots.
#[derive(Resource, Debug, Default)]
pub struct ChunkManager {
    /// Active chunk -> group root entity.
    pub active: HashMap<(i32, i32), Entity>,
}

impl ChunkManager {
    /// Chunks that should be active for a player at `pos` walking toward
    /// `forward` (world-plane direction).
    pub fn desired_chunks(pos: Vec2, forward: Vec2) -> Vec<ChunkCoords> {
        let forward = if forward.length_squared() > f32::EPSILON {
            forward.normalize()
        } else {
            Vec2::new(0.0, -1.0)
        };
        let right = Vec2::new(-forward.y, forward.x);
        let player = ChunkCoords::from_world(pos);

        // Chebyshev scan bound covering the rotated footprint (radius-4
        // ahead + 3-side footprint fits within 8 chunk indices).
        let radius = CHUNK_AHEAD + CHUNK_SIDE + 1;
        let mut out = Vec::new();
        for cx in (player.x - radius)..=(player.x + radius) {
            for cz in (player.z - radius)..=(player.z + radius) {
                // The world is bounded: no chunks outside the 40x40 grid.
                if cx < 0 || cz < 0 || cx >= GRID_CELLS as i32 || cz >= GRID_CELLS as i32 {
                    continue;
                }
                let center = Vec2::new(
                    (cx as f32 + 0.5) * CHUNK_SIZE,
                    (cz as f32 + 0.5) * CHUNK_SIZE,
                );
                let delta = center - pos;
                let along = delta.dot(forward);
                let cross = delta.dot(right);
                let behind_limit = -(CHUNK_BEHIND as f32 + 0.5) * CHUNK_SIZE;
                let ahead_limit = (CHUNK_AHEAD as f32 + 0.5) * CHUNK_SIZE;
                let side_limit = (CHUNK_SIDE as f32 + 0.5) * CHUNK_SIZE;
                if along >= behind_limit && along <= ahead_limit && cross.abs() <= side_limit {
                    out.push(ChunkCoords { x: cx, z: cz });
                }
            }
        }
        out
    }
}

/// Deterministic per-chunk asphalt tint (subtle luminance variation) so the
/// streaming feedback is visible in dev mode.
fn tile_color(coords: ChunkCoords) -> Color {
    let h = (coords.x as u64).rotate_left(32) ^ coords.z as u64;
    let h = h.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    let v = 0.085 + (h >> 56) as f32 / 255.0 * 0.02;
    Color::srgb(v, v, v * 1.05)
}

/// World plugin systems (ordering set, R2.8: every system chain is explicit).
#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub enum WorldSystems {
    /// Startup layout generation.
    Generation,
    /// Chunk streaming (spawn/despawn around the player).
    Streaming,
}

/// Generates the world layout at startup from the seed resource (or the
/// default seed until the R4 menu exists) and initializes the chunk manager.
fn generate_layout(mut commands: Commands, seed: Option<Res<HeadsUpSeed>>) {
    let seed_value = seed.map(|s| s.0).unwrap_or(DEFAULT_SEED);
    match generate_world(seed_value) {
        Ok((grid, graph)) => {
            info!(
                "world generated: seed={seed_value:#x} blocks={} street segments={} nodes={}",
                graph.blocks.len(),
                graph.edges.len(),
                graph.nodes.len()
            );
            commands.insert_resource(grid);
            commands.insert_resource(graph);
            commands.insert_resource(ChunkManager::default());
        }
        Err(err) => error!("world generation failed: {err}"),
    }
}

/// Updates the active chunk set from the camera position and facing, spawning
/// missing chunk groups and despawning the ones the player walked away from.
pub(crate) fn update_chunk_streaming(
    cameras: Query<&Transform, With<Camera3d>>,
    mut manager: ResMut<ChunkManager>,
    mut commands: Commands,
    mut meshes: Option<ResMut<Assets<Mesh>>>,
    mut materials: Option<ResMut<Assets<StandardMaterial>>>,
) {
    let Ok(camera) = cameras.single() else {
        return;
    };
    let pos = Vec2::new(camera.translation.x, camera.translation.z);
    let forward = Vec2::new(camera.forward().x, camera.forward().z);

    let desired = ChunkManager::desired_chunks(pos, forward);
    let mut desired_keys = Vec::with_capacity(desired.len());

    // Spawn missing chunks (group root + ground tile).
    for coords in desired {
        let key = (coords.x, coords.z);
        desired_keys.push(key);
        if manager.active.contains_key(&key) {
            continue;
        }
        let root = commands
            .spawn((
                Name::new(format!("Chunk({}, {})", coords.x, coords.z)),
                ChunkRoot,
                ChunkCoords {
                    x: coords.x,
                    z: coords.z,
                },
            ))
            .id();
        // Ground tile: real streaming feedback; R3 adds placement content.
        if let (Some(meshes), Some(materials)) = (meshes.as_deref_mut(), materials.as_deref_mut()) {
            let mesh = meshes.add(Plane3d::default().mesh().size(CHUNK_SIZE, CHUNK_SIZE));
            let material = materials.add(StandardMaterial {
                base_color: tile_color(coords),
                perceptual_roughness: 0.95,
                ..default()
            });
            let center = coords.center();
            commands.entity(root).with_children(|parent| {
                parent.spawn((
                    Name::new("ChunkGround"),
                    Mesh3d(mesh),
                    MeshMaterial3d(material),
                    Transform::from_xyz(center.x, 0.0, center.y),
                ));
            });
        }
        manager.active.insert(key, root);
    }

    // Despawn chunks the player walked away from (recursive group removal).
    let mut removed = Vec::new();
    for (&key, &root) in manager.active.iter() {
        if !desired_keys.contains(&key) {
            removed.push((key, root));
        }
    }
    for (key, root) in removed {
        manager.active.remove(&key);
        commands.entity(root).try_despawn();
    }
}

/// Temporary debug movement for R2 verification (no player controller yet;
/// R3.4 replaces this with the kinematic controller): W/S walk along the
/// facing direction, A/D turn, 1.5 m/s.
pub(crate) fn debug_walk_camera(
    input: Res<ButtonInput<KeyCode>>,
    time: Res<Time>,
    mut cameras: Query<&mut Transform, With<Camera3d>>,
) {
    let Ok(mut transform) = cameras.single_mut() else {
        return;
    };
    let turn = if input.pressed(TURN_LEFT) {
        1.0
    } else if input.pressed(TURN_RIGHT) {
        -1.0
    } else {
        0.0
    };
    if turn != 0.0 {
        transform.rotate_y(turn * time.delta_secs());
    }
    let forward: Vec3 = *transform.forward();
    let mut step = Vec3::ZERO;
    if input.pressed(MOVE_FORWARD) {
        step += forward;
    }
    if input.pressed(MOVE_BACKWARD) {
        step -= forward;
    }
    if step != Vec3::ZERO {
        transform.translation += step.normalize() * 1.5 * time.delta_secs();
    }
}

/// Registers the world generation and chunk streaming systems.
///
/// Ordering contract (R2.8): explicit chains only — the debug walk (writes
/// camera `Transform`) runs before streaming (reads it) so the active set is
/// computed from the settled position; `GlobalTransform` propagation then
/// happens in Bevy's transform systems.
pub struct WorldPlugin;

impl Plugin for WorldPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, generate_layout)
            .add_systems(
                Update,
                update_chunk_streaming
                    .in_set(WorldSystems::Streaming)
                    .after(debug_walk_camera),
            )
            .add_systems(Update, debug_walk_camera);
    }
}

/// Convenience accessor for tests and the debug visualization: the layout
/// graph resource generated at startup.
pub fn layout_resource(world: &World) -> Option<&LayoutGraph> {
    world.get_resource::<LayoutGraph>()
}

/// Sanity check for tests: the world fits the chunk grid.
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunk_grid_tiles_the_world() {
        use crate::world::WORLD_SIZE;
        assert_eq!(CHUNK_SIZE, 50.0);
        assert_eq!(
            (WORLD_SIZE / CHUNK_SIZE) as u32,
            GRID_CELLS,
            "40x40 chunks tile the 2 km world"
        );
    }

    #[test]
    fn desired_chunks_follow_the_walk_direction() {
        // Player at the world center facing -Z (north, toward row 0).
        let pos = Vec2::new(1000.0, 1000.0);
        let forward = Vec2::new(0.0, -1.0);
        let desired = ChunkManager::desired_chunks(pos, forward);
        // Center-based window: 8 x 8 = 64 at the worst sub-chunk offset.
        assert!(
            (49..=64).contains(&desired.len()),
            "unexpected footprint size: {}",
            desired.len()
        );

        let player = ChunkCoords::from_world(pos);
        for chunk in &desired {
            let center = chunk.center();
            let delta = center - pos;
            let along = delta.dot(forward);
            let cross = delta.dot(Vec2::new(-forward.y, forward.x));
            assert!(along <= (CHUNK_AHEAD as f32 + 0.5) * CHUNK_SIZE);
            assert!(along >= -(CHUNK_BEHIND as f32 + 0.5) * CHUNK_SIZE);
            assert!(cross.abs() <= (CHUNK_SIDE as f32 + 0.5) * CHUNK_SIZE);
        }
        assert_eq!(player, ChunkCoords { x: 20, z: 20 });
    }

    #[test]
    fn desired_chunks_shift_with_facing() {
        let pos = Vec2::new(1000.0, 1000.0);
        // Facing +X (east): chunks ahead in x.
        let desired = ChunkManager::desired_chunks(pos, Vec2::new(1.0, 0.0));
        let max_x = desired.iter().map(|c| c.x).max().expect("non-empty");
        let min_x = desired.iter().map(|c| c.x).min().expect("non-empty");
        assert!(max_x - 20 >= CHUNK_AHEAD - 1, "chunks reach ahead of +X");
        assert!(min_x >= 20 - CHUNK_BEHIND - 1);
    }

    #[test]
    fn tile_color_is_deterministic_per_chunk() {
        let a = tile_color(ChunkCoords { x: 3, z: 4 });
        let b = tile_color(ChunkCoords { x: 3, z: 4 });
        assert_eq!(a, b);
        assert_ne!(
            tile_color(ChunkCoords { x: 3, z: 4 }),
            tile_color(ChunkCoords { x: 4, z: 3 })
        );
    }

    /// Headless streaming test: chunk groups spawn/despawn as the camera
    /// moves (R2 verification: "移动时周围物体出现/消失").
    #[test]
    fn chunk_streaming_spawns_and_despawns() {
        let mut app = App::new();
        app.add_plugins((MinimalPlugins, super::WorldPlugin))
            .init_resource::<ButtonInput<KeyCode>>();

        // A camera entity (data only; no render plugins needed).
        app.world_mut().spawn((
            Camera3d::default(),
            Transform::from_xyz(1000.0, 1.6, 1000.0),
        ));

        app.update(); // Startup: layout generated; Update: chunks spawned.

        // No camera movement yet: chunk set computed from the spawn position.
        app.update();
        let manager = app.world().resource::<ChunkManager>();
        assert!(!manager.active.is_empty(), "chunks spawn at startup");

        // Teleport the camera 500 m east and update twice: the old chunks
        // despawn, new ones spawn (both positions inside the world).
        let mut cameras = app
            .world_mut()
            .query_filtered::<&mut Transform, With<Camera3d>>();
        for mut transform in cameras.iter_mut(app.world_mut()) {
            transform.translation = Vec3::new(1500.0, 1.6, 1000.0);
        }
        app.update();
        app.update();

        let manager = app.world().resource::<ChunkManager>();
        let all_x: Vec<i32> = manager.active.keys().map(|k| k.0).collect();
        assert!(
            all_x.iter().all(|x| *x >= 26),
            "chunks behind the teleport should have despawned: {all_x:?}"
        );
        // The active set matches the center-based footprint at (30, 20).
        assert!((49..=64).contains(&manager.active.len()));
    }

    #[test]
    fn layout_resource_is_available_after_generation() {
        use crate::world::WORLD_SIZE;
        let mut app = App::new();
        app.add_plugins((MinimalPlugins, super::WorldPlugin))
            .init_resource::<ButtonInput<KeyCode>>();
        app.world_mut().spawn((
            Camera3d::default(),
            Transform::from_xyz(1000.0, 1.6, 1000.0),
        ));
        app.update();
        let graph = layout_resource(app.world()).expect("layout graph resource");
        assert!(!graph.blocks.is_empty());
        assert!(!graph.edges.is_empty());
        assert!((graph.bounds.x - WORLD_SIZE).abs() < 1e-3);
    }
}
