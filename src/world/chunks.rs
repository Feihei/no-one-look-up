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

use std::collections::{HashMap, HashSet, VecDeque};

use bevy::prelude::*;

use crate::world::collision::CollisionIndex;
use crate::world::layout_graph::BlockType;
use crate::world::layout_graph::LayoutGraph;
use crate::world::path_network::{HeadsUpSeed, generate_world};
use crate::world::placement;
use crate::world::tree_mesh::{TreeLod, TreeMeshKey, bake_tree_mesh_lod};
use crate::world::tree_shape::TREE_PROFILES;
use crate::world::{CHUNK_AHEAD, CHUNK_BEHIND, CHUNK_SIZE, CHUNKS_PER_FRAME, GRID_CELLS};

/// Default seed until the R4 menu provides one (dev builds).
pub const DEFAULT_SEED: u64 = 0x4845_4144; // "HEAD"

/// Chunk grid coordinates. Chunk `(x, z)` covers the world square
/// `[x * CHUNK_SIZE, (x + 1) * CHUNK_SIZE) x [z * CHUNK_SIZE, (z + 1) *
/// CHUNK_SIZE)`; 16 x 16 chunks tile the 800 m world.
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
    /// Chunks waiting to be spawned, nearest-first (original load queue).
    pub pending_load: VecDeque<(i32, i32)>,
    /// Chunks waiting to be despawned (original unload queue).
    pub pending_unload: VecDeque<(i32, i32)>,
}

impl ChunkManager {
    /// Chunks that should be active for a player at `pos` walking toward
    /// `forward` (world-plane direction).
    pub fn desired_chunks(pos: Vec2, forward: Vec2) -> Vec<ChunkCoords> {
        // Port of `chunk_manager.gd::_compute_needed` (user-facing behaviour:
        // a square window with a smaller radius behind).
        let forward = if forward.length_squared() > f32::EPSILON {
            forward.normalize()
        } else {
            // Looking straight down: no horizontal facing, keep the defaults.
            Vec2::new(0.0, -1.0)
        };
        let player = ChunkCoords::from_world(pos);
        let max_r = CHUNK_AHEAD.max(CHUNK_BEHIND);

        let mut out = Vec::new();
        for dx in -max_r..=max_r {
            for dz in -max_r..=max_r {
                let x = player.x + dx;
                let z = player.z + dz;
                if x < 0 || z < 0 || x >= GRID_CELLS as i32 || z >= GRID_CELLS as i32 {
                    continue;
                }
                let coords = ChunkCoords { x, z };
                if dx == 0 && dz == 0 {
                    out.push(coords); // Player's own chunk: always loaded.
                    continue;
                }
                let center = coords.center();
                let to_chunk = center - pos;
                // Ahead (or exactly abeam) uses the front radius, behind the
                // back radius. Chebyshev keeps the window square, so the side
                // extent equals the front radius ahead of the player.
                let radius = if to_chunk.dot(forward) >= 0.0 {
                    CHUNK_AHEAD
                } else {
                    CHUNK_BEHIND
                };
                if dx.abs().max(dz.abs()) <= radius {
                    out.push(coords);
                }
            }
        }
        out
    }
}

/// Expands a Chebyshev key towards `pos` for stable load ordering.
fn chunk_distance(key: (i32, i32), pos: Vec2) -> f32 {
    let center = ChunkCoords { x: key.0, z: key.1 }.center();
    (center - pos).length_squared()
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
            commands.init_resource::<CollisionIndex>();
        }
        Err(err) => error!("world generation failed: {err}"),
    }
}

/// Updates the active chunk set from the camera position and facing, spawning
/// missing chunk groups and despawning the ones the player walked away from.
#[allow(clippy::too_many_arguments)]
pub(crate) fn update_chunk_streaming(
    cameras: Query<&Transform, With<Camera3d>>,
    mut manager: ResMut<ChunkManager>,
    mut collision: ResMut<CollisionIndex>,
    graph: Option<Res<LayoutGraph>>,
    assets: Option<Res<PlacementAssets>>,
    seed: Option<Res<HeadsUpSeed>>,
    mut commands: Commands,
    mut meshes: Option<ResMut<Assets<Mesh>>>,
    mut materials: Option<ResMut<Assets<StandardMaterial>>>,
) {
    let Ok(camera) = cameras.single() else {
        return;
    };
    let seed_value = seed.map(|s| s.0).unwrap_or(DEFAULT_SEED);
    let pos = Vec2::new(camera.translation.x, camera.translation.z);
    let forward = Vec2::new(camera.forward().x, camera.forward().z);

    let desired = ChunkManager::desired_chunks(pos, forward);
    let desired_keys: HashSet<(i32, i32)> = desired.iter().map(|c| (c.x, c.z)).collect();

    // Rebuild the load queue: everything wanted but not active and not
    // already queued, ordered nearest-first so the player's surroundings fill
    // in before the horizon.
    let active_keys: Vec<(i32, i32)> = manager.active.keys().copied().collect();
    manager
        .pending_load
        .retain(|key| desired_keys.contains(key) && !active_keys.contains(key));
    for coords in &desired {
        let key = (coords.x, coords.z);
        if !active_keys.contains(&key) && !manager.pending_load.contains(&key) {
            manager.pending_load.push_back(key);
        }
    }
    manager.pending_load.make_contiguous().sort_by(|a, b| {
        chunk_distance(*a, pos)
            .partial_cmp(&chunk_distance(*b, pos))
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Same for unloads: chunks outside the window leave, closest first.
    manager
        .pending_unload
        .retain(|key| active_keys.contains(key));
    for key in &active_keys {
        if !desired_keys.contains(key) && !manager.pending_unload.contains(key) {
            manager.pending_unload.push_back(*key);
        }
    }

    // Process at most CHUNKS_PER_FRAME of each per frame (original budget):
    // spawning builds a chunk's whole contents, so a teleport or sharp turn
    // must not pay for all of them in a single frame.
    for _ in 0..CHUNKS_PER_FRAME {
        let Some(key) = manager.pending_unload.pop_front() else {
            break;
        };
        if let Some(root) = manager.active.remove(&key) {
            collision.unregister_chunk(key);
            commands.entity(root).try_despawn();
        }
    }

    for _ in 0..CHUNKS_PER_FRAME {
        let Some(key) = manager.pending_load.pop_front() else {
            break;
        };
        if manager.active.contains_key(&key) {
            continue;
        }
        let coords = ChunkCoords { x: key.0, z: key.1 };
        let root = commands
            .spawn((
                Name::new(format!("Chunk({}, {})", coords.x, coords.z)),
                ChunkRoot,
                coords,
                // A parent WITHOUT `Transform`/`Visibility` breaks hierarchy
                // propagation for every child: the ground tile, the buildings
                // and the trees all lost their `GlobalTransform` and rendered
                // nothing (while the gizmo overlay, which does not go through
                // the hierarchy, kept drawing). Bevy warns B0004 once per
                // child; the log showed 1120 of them.
                Transform::default(),
                Visibility::default(),
            ))
            .id();
        // Ground tile: real streaming feedback; placement content follows.
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
        // Collision prisms travel with the chunk: R3.3 keeps the solid set in
        // step with what is actually spawned, so a walker can never slip
        // through a block whose collision was registered for another chunk.
        if let Some(graph) = graph.as_deref() {
            collision.register_chunk(graph, key);
            // R3 placement: plan this chunk's buildings/trees and spawn them
            // under the chunk root, so despawning the root removes them all.
            if assets.is_none() {
                static ONCE: std::sync::Once = std::sync::Once::new();
                ONCE.call_once(|| {
                    error!(
                        "chunk {key:?} streamed with NO PlacementAssets: no buildings, no trees"
                    );
                });
            }
            if let Some(assets) = assets.as_deref() {
                let plan = placement::plan_chunk(
                    graph,
                    seed_value,
                    key,
                    // Procedural trees: the planner samples a SPECIES index.
                    TREE_PROFILES.len() as u8,
                );
                // Trees/buildings use the shared handles in `assets`, so
                // streaming needs no `Assets<Mesh>` access at all.
                spawn_chunk_content(&mut commands, assets, root, &plan);
            }
        }
        manager.active.insert(key, root);
    }
}

/// Shared placeholder materials + tree scenes, built once (Draw Call
/// control, ADR-0004).
#[derive(Resource)]
pub(crate) struct PlacementAssets {
    /// `building_materials[type][bucket]`: one shared material per (style,
    /// wear bucket) pair, so per-building weathering costs no extra draw
    /// state beyond the bucket switch (R3.6).
    building_materials: [[Handle<StandardMaterial>; placement::WEAR_BUCKETS]; 3],
    plaza_material: Handle<StandardMaterial>,
    /// One baked mesh per `(lod, species, variant)` (R3.9): a whole tree in a
    /// single `Mesh3d`, so a chunk costs one entity per tree per LOD tier.
    /// Indices follow `lod * TreeMeshKey::VARIANTS * TREE_PROFILES.len() +
    /// species * TreeMeshKey::VARIANTS + variant`.
    tree_meshes: Vec<(Handle<Mesh>, Handle<Mesh>)>,
    /// Opaque silhouette material for the branches.
    tree_material: Handle<crate::plugins::season::TreeSeasonMaterial>,
    /// Alpha-tested material for the foliage cards, textured with a species
    /// leaf mask.
    tree_leaf_material: Handle<crate::plugins::season::TreeSeasonMaterial>,
    /// One unit cube reused by every building and plaza pad; each instance
    /// scales it. Buildings used to call `meshes.add(Cuboid)` per object,
    /// which allocated a new mesh every time a chunk streamed in.
    unit_cube: Handle<Mesh>,
}

impl PlacementAssets {
    /// Baked mesh for a placement tree instance at one LOD tier, or `None` if
    /// out of range.
    /// `(wood, leaves)` meshes for a placement tree instance at one LOD tier.
    fn tree_mesh(
        &self,
        asset: u8,
        wear: f32,
        lod: TreeLod,
    ) -> Option<&(Handle<Mesh>, Handle<Mesh>)> {
        let key = crate::world::tree_mesh::TreeMeshKey::from_placement(asset, wear);
        let per_lod = TREE_PROFILES.len() * TreeMeshKey::VARIANTS as usize;
        let tier = TreeLod::ALL.iter().position(|t| *t == lod)?;
        let index = tier * per_lod
            + key.species as usize * TreeMeshKey::VARIANTS as usize
            + key.variant as usize;
        self.tree_meshes.get(index)
    }
}

/// Builds the placement assets on first chunk spawn (lazy: `Mesh3d` handles
/// need the render `Assets` stores, which headless tests do not run).
fn build_placement_assets(
    mut commands: Commands,
    meshes: Option<ResMut<Assets<Mesh>>>,
    materials: Option<ResMut<Assets<StandardMaterial>>>,
    tree_materials: Option<ResMut<Assets<crate::plugins::season::TreeSeasonMaterial>>>,
    mut images: Option<ResMut<Assets<Image>>>,
    existing: Option<Res<PlacementAssets>>,
) {
    // Build ONCE. This system used to run every frame with no guard: it baked
    // 12 tree meshes, rebuilt every building material and replaced the
    // `PlacementAssets` resource each frame. Measured over 240 frames: meshes
    // 13 -> 3242, materials 15 -> 3411, all leaked — and every chunk kept
    // handles to a resource that was already gone.
    if existing.is_some() {
        return;
    }
    // Headless tests run without the render assets stores; placement content
    // stays disabled for them (plan_chunk still works).
    //
    // R3.9: no `AssetServer` needed any more. Trees are baked procedurally
    // from `tree_shape`, not loaded from glTF — which also means the world
    // gets trees in a headless test, so streaming is actually testable.
    //
    // The season material store is a separate requirement: it only exists once
    // `SeasonPlugin` has registered the `MaterialPlugin` (which headless tests
    // do add, so trees stay testable). Waiting for it — instead of returning
    // after the meshes have been allocated — is what keeps this system
    // idempotent; returning late would leak a fresh set of meshes every frame.
    let (Some(mut meshes), Some(mut materials), Some(mut tree_materials)) =
        (meshes, materials, tree_materials)
    else {
        return;
    };
    let palette = placement::StylePalette::default();
    let mut wear_color = |rgb: [f32; 3], bucket: usize| {
        // Wear grades: pristine keeps the style colour; higher buckets
        // darken + desaturate (soot/faded paint) and get slightly glossier
        // grime sheen. Same recipe per bucket so types stay comparable.
        let t = bucket as f32 / (placement::WEAR_BUCKETS - 1) as f32;
        let luma = rgb[0] * 0.3 + rgb[1] * 0.6 + rgb[2] * 0.1;
        let desat = 0.25 * t;
        let darken = 1.0 - 0.35 * t;
        let mixed = |c: f32| (luma * desat + c * (1.0 - desat)) * darken;
        materials.add(StandardMaterial {
            base_color: Color::srgb(mixed(rgb[0]), mixed(rgb[1]), mixed(rgb[2])),
            perceptual_roughness: 0.9 - 0.15 * t,
            ..default()
        })
    };
    let mut style_wear = |rgb: [f32; 3]| std::array::from_fn(|bucket| wear_color(rgb, bucket));
    let building_materials = [
        style_wear(palette.old_city),
        style_wear(palette.apartment),
        style_wear(palette.factory),
    ];
    let plaza_material = wear_color(palette.plaza, 1);
    // R3.9: trees are procedural, baked to one mesh per (species, variant).
    // 3 species x 4 variants = 12 meshes for the whole world, reused by every
    // chunk that spawns a tree.
    // Each entry is (wood, leaves): two meshes per tree, because the branches
    // stay opaque while the foliage cards are alpha-tested. See
    // `world::tree_mesh`'s header for why they cannot share a material.
    let per_lod = TREE_PROFILES.len() * TreeMeshKey::VARIANTS as usize;
    let mut tree_meshes: Vec<(Handle<Mesh>, Handle<Mesh>)> =
        Vec::with_capacity(per_lod * TreeLod::ALL.len());
    for lod in TreeLod::ALL {
        for species in 0..TREE_PROFILES.len() as u8 {
            for variant in 0..TreeMeshKey::VARIANTS {
                let parts = bake_tree_mesh_lod(TreeMeshKey { species, variant }, lod);
                tree_meshes.push((meshes.add(parts.wood), meshes.add(parts.leaves)));
            }
        }
    }
    // R3.5: the season-aware material replaces the plain silhouette. Every
    // tree in the world shares this one handle, which is what makes a season
    // switch a single 32-byte uniform write instead of a walk over every tree
    // entity (ADR-0004 §6 "no per-material traversal").
    let tree_material = tree_materials.add(crate::plugins::season::TreeSeasonMaterial {
        base: crate::plugins::season::tree_base_material(),
        extension: Default::default(),
    });
    // The foliage cards need a texture. Until the species atlas exists, one
    // shared placeholder keeps every tree drawing.
    // An `Image` store only exists once the render plugins are up; a headless
    // test has none. Falling back to a texture-less card keeps the world
    // building there, where nothing is drawn anyway -- and the test that counts
    // tree entities is about streaming, not about what the crown looks like.
    let leaf_texture = images.as_deref_mut().map(|images| {
        let profiles = crate::world::leaf_shape::leaf_profiles();
        let profile = crate::world::leaf_shape::find_profile("poplar")
            .or_else(|| profiles.first())
            .expect("the leaf table is never empty");
        let mask = crate::world::leaf_shape::bake_leaf_alpha(profile, 256);
        images.add(crate::world::leaf_shape::mask_to_image(&mask))
    });
    let tree_leaf_material =
        tree_materials.add(crate::plugins::season::tree_leaf_material(leaf_texture));
    let unit_cube = meshes.add(Cuboid::new(1.0, 1.0, 1.0));

    info!(
        "placement assets built: {} tree meshes ({} LODs), {} building materials",
        tree_meshes.len(),
        TreeLod::ALL.len(),
        building_materials.len()
    );
    commands.insert_resource(PlacementAssets {
        building_materials,
        plaza_material,
        tree_meshes,
        tree_material,
        tree_leaf_material,
        unit_cube,
    });
}

/// Spawns a chunk's planned content under its root: buildings as BoxMesh
/// placeholders (ADR-0004 §4; R3.6 adds wear materials), trees as glTF
/// scenes, the plaza pad as a thin box. Called inside the streaming system so
/// a chunk's contents exist in the same frame its collision is registered.
fn spawn_chunk_content(
    commands: &mut Commands,
    assets: &PlacementAssets,
    root: Entity,
    plan: &placement::ChunkPlan,
) {
    for object in &plan.objects {
        match object.kind {
            placement::PlacedKind::Building(kind) => {
                let style = match kind {
                    BlockType::OldCity => 0,
                    BlockType::Apartment => 1,
                    _ => 2,
                };
                let material =
                    &assets.building_materials[style][placement::wear_bucket(object.wear)];
                let mesh = assets.unit_cube.clone();
                let position = Vec3::new(object.position.x, object.size.y * 0.5, object.position.y);
                let rotation = Quat::from_rotation_y(object.yaw);
                commands.entity(root).with_children(|parent| {
                    parent.spawn((
                        Name::new(format!(
                            "Building@{:?} wear={:.2}",
                            object.position, object.wear
                        )),
                        Mesh3d(mesh),
                        MeshMaterial3d(material.clone()),
                        Transform::from_translation(position)
                            .with_rotation(rotation)
                            .with_scale(object.size),
                    ));
                });
            }
            placement::PlacedKind::PlazaPad => {
                let mesh = assets.unit_cube.clone();
                let position = Vec3::new(object.position.x, object.size.y * 0.5, object.position.y);
                commands.entity(root).with_children(|parent| {
                    parent.spawn((
                        Name::new("PlazaPad"),
                        Mesh3d(mesh),
                        MeshMaterial3d(assets.plaza_material.clone()),
                        Transform::from_translation(position).with_scale(object.size),
                    ));
                });
            }
            _ => {
                // Trees (street/forest/plaza): one baked mesh per LOD tier
                // (R3.9 step 4). All three sit at the same place; each carries
                // a `VisibilityRange` so exactly one is drawn at any camera
                // distance. Foliage is ~85-90% of a tree, so the tiers differ
                // mostly in how many leaf clusters survive: measured 2952
                // triangles near, 1238 mid (-58%), 183 far (-94%).
                let Some(tree) = object.tree else { continue };
                let transform = Transform::from_xyz(object.position.x, 0.0, object.position.y)
                    .with_rotation(Quat::from_rotation_y(tree.yaw))
                    .with_scale(Vec3::splat(tree.scale));

                commands.entity(root).with_children(|parent| {
                    for lod in TreeLod::ALL {
                        let Some((wood, leaves)) = assets.tree_mesh(tree.asset, object.wear, lod)
                        else {
                            continue;
                        };
                        // Two entities per tier: the branches keep the opaque
                        // silhouette material, the cards take the alpha-tested
                        // leaf material. Same transform, so they stay together.
                        let suffix = match lod {
                            TreeLod::Near => "",
                            TreeLod::Mid => "Mid",
                            TreeLod::Far => "Far",
                        };
                        // The wood entity carries the tier's canonical name
                        // (`Tree` / `TreeMid` / `TreeFar`) so the LOD and
                        // streaming tests keep working; the leaf entity is
                        // named alongside it.
                        for (part_name, mesh, material) in [
                            (format!("Tree{suffix}"), wood, &assets.tree_material),
                            (
                                format!("TreeLeaves{suffix}"),
                                leaves,
                                &assets.tree_leaf_material,
                            ),
                        ] {
                            let mut entity = parent.spawn((
                                Name::new(part_name),
                                Mesh3d(mesh.clone()),
                                MeshMaterial3d(material.clone()),
                                transform,
                            ));
                            // The near tier has nothing to fade from, so it
                            // needs no range component at all.
                            if lod != TreeLod::Near {
                                entity.insert(lod.range());
                            }
                        }
                    }
                });
            }
        }
    }
}

pub struct WorldPlugin;

impl Plugin for WorldPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<crate::world::explored::ExploredBitmap>()
            // The tree material store is registered here rather than by
            // `SeasonPlugin`: `Assets<T>` must exist before
            // `build_placement_assets` runs, and the headless tests build the
            // world without the render plugins. Registering the asset type
            // twice is a no-op, so the real app (which also has the
            // `MaterialPlugin` that needs it) is unaffected.
            .init_resource::<Assets<crate::plugins::season::TreeSeasonMaterial>>()
            .add_message::<crate::world::explored::PlayerChunkChanged>()
            .add_systems(Startup, generate_layout)
            .add_systems(PreUpdate, build_placement_assets)
            .add_systems(
                Update,
                (
                    update_chunk_streaming
                        .in_set(WorldSystems::Streaming)
                        // The controller writes the camera Transform; streaming
                        // reads it, so the window is computed from the settled
                        // position (explicit, per root AGENTS.md).
                        .after(crate::player::PlayerSystems::Move),
                    // Exploration follows the settled position too, after
                    // streaming (it reads the same camera transform).
                    crate::world::explored::track_player_chunk.after(WorldSystems::Streaming),
                ),
            );
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

    /// World centre, grounded on the actual constants under test.
    const CENTER: f32 = crate::world::WORLD_SIZE * 0.5;

    #[test]
    fn chunk_grid_tiles_the_world() {
        use crate::world::WORLD_SIZE;
        assert_eq!(CHUNK_SIZE, 50.0);
        assert_eq!(
            (WORLD_SIZE / CHUNK_SIZE) as u32,
            GRID_CELLS,
            "16x16 chunks tile the 800 m world"
        );
    }

    #[test]
    fn desired_chunks_follow_the_walk_direction() {
        // Player at the world center facing -Z (north): the window is the
        // original's square scan — 4 ahead, 2 behind (Chebyshev).
        let pos = Vec2::new(CENTER, CENTER);
        let forward = Vec2::new(0.0, -1.0);
        let desired = ChunkManager::desired_chunks(pos, forward);

        let player = ChunkCoords::from_world(pos);
        assert_eq!(player, ChunkCoords { x: 8, z: 8 });
        // Square scan: rows are 2 behind + own + 4 ahead. Columns whose
        // centre sits exactly abeam have dot == 0 and count as "ahead", so the
        // window is symmetric on both axes near the centre of a chunk.
        assert!(
            (49..=63).contains(&desired.len()),
            "unexpected footprint size: {}",
            desired.len()
        );

        for chunk in &desired {
            let dx = chunk.x - player.x;
            let dz = chunk.z - player.z;
            let ahead = (chunk.center() - pos).dot(forward) >= 0.0;
            let radius = if ahead { CHUNK_AHEAD } else { CHUNK_BEHIND };
            assert!(
                dx.abs().max(dz.abs()) <= radius,
                "chunk {chunk:?} outside the chebyshev radius {radius}"
            );
        }
        // The own chunk is always present.
        assert!(desired.contains(&player));
    }

    #[test]
    fn desired_chunks_shift_with_facing() {
        let pos = Vec2::new(CENTER, CENTER);
        let player = ChunkCoords::from_world(pos);
        // Facing +X (east): the window reaches CHUNK_AHEAD east, less west.
        let east = ChunkManager::desired_chunks(pos, Vec2::new(1.0, 0.0));
        assert!(east.iter().any(|c| c.x == player.x + CHUNK_AHEAD));
        assert!(!east.iter().any(|c| c.x == player.x + CHUNK_AHEAD + 1));
        assert!(!east.iter().any(|c| c.x == player.x - CHUNK_AHEAD));

        // Turning 180° mirrors it.
        let west = ChunkManager::desired_chunks(pos, Vec2::new(-1.0, 0.0));
        assert!(west.iter().any(|c| c.x == player.x - CHUNK_AHEAD));
        assert!(!west.iter().any(|c| c.x == player.x + CHUNK_AHEAD));
    }

    #[test]
    fn desired_chunks_never_leave_the_world() {
        // Player in the north-west corner: no chunk index may go negative or
        // past the 40x40 grid.
        let desired = ChunkManager::desired_chunks(Vec2::new(10.0, 10.0), Vec2::Y);
        assert!(!desired.is_empty());
        for chunk in &desired {
            assert!((0..GRID_CELLS as i32).contains(&chunk.x), "{chunk:?}");
            assert!((0..GRID_CELLS as i32).contains(&chunk.z), "{chunk:?}");
        }
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
            Transform::from_xyz(CENTER, 1.6, CENTER),
        ));

        app.update(); // Startup: layout generated; Update: chunks spawned.

        // No camera movement yet: chunk set computed from the spawn position.
        app.update();
        let manager = app.world().resource::<ChunkManager>();
        assert!(!manager.active.is_empty(), "chunks spawn at startup");

        // Teleport the camera 300 m east (still 6 chunks from the east edge)
        // and update twice: the old chunks despawn, new ones spawn.
        let mut cameras = app
            .world_mut()
            .query_filtered::<&mut Transform, With<Camera3d>>();
        for mut transform in cameras.iter_mut(app.world_mut()) {
            transform.translation = Vec3::new(CENTER + 300.0, 1.6, CENTER);
        }
        app.update();
        app.update();

        let manager = app.world().resource::<ChunkManager>();
        let all_x: Vec<i32> = manager.active.keys().map(|k| k.0).collect();
        assert!(
            all_x.iter().all(|x| *x >= 10),
            "chunks behind the teleport should have despawned: {all_x:?}"
        );
        // The active set matches the footprint once the per-frame budget has
        // drained the queue (CHUNKS_PER_FRAME = 1, so give it time).
        let expected =
            ChunkManager::desired_chunks(Vec2::new(CENTER + 300.0, CENTER), Vec2::new(0.0, -1.0))
                .len();
        for _ in 0..(expected + 8) {
            app.update();
        }
        let manager = app.world().resource::<ChunkManager>();
        assert_eq!(manager.active.len(), expected, "active set should converge");
    }

    #[test]
    fn chunk_root_can_propagate_transforms_to_its_children() {
        // Regression: `ChunkRoot` was spawned with only `Name`/`ChunkRoot`/
        // `coords`, so the hierarchy had a parent without `Transform`. Every
        // child (ground tile, buildings, trees) failed to receive a
        // `GlobalTransform` and rendered nothing — while the gizmo overlay,
        // which bypasses the hierarchy, kept drawing. Bevy logged B0004 once
        // per child: 1120 of them in a 25 s run.
        //
        // The whole rendered world hangs off these roots, so assert the parent
        // can actually carry a transform rather than just checking a count.
        let mut app = App::new();
        app.add_plugins((MinimalPlugins, super::WorldPlugin))
            .init_resource::<ButtonInput<KeyCode>>()
            .init_resource::<Assets<Mesh>>()
            .init_resource::<Assets<StandardMaterial>>()
            .init_resource::<Assets<crate::plugins::season::TreeSeasonMaterial>>();
        app.world_mut().spawn((
            Camera3d::default(),
            Transform::from_xyz(CENTER, 1.6, CENTER),
        ));
        for _ in 0..30 {
            app.update();
        }

        let mut roots = app
            .world_mut()
            .query_filtered::<Entity, (With<ChunkRoot>, Without<Transform>)>();
        let missing: Vec<Entity> = roots.iter(app.world()).collect();
        assert!(
            missing.is_empty(),
            "{} chunk roots have no Transform; their children cannot be placed              in the world and will not render",
            missing.len()
        );

        // And the children really do have a GlobalTransform after propagation.
        let mut children = app
            .world_mut()
            .query_filtered::<&Name, (Without<GlobalTransform>, With<ChildOf>)>();
        let unplaced: Vec<String> = children
            .iter(app.world())
            .map(|n| n.as_str().to_string())
            .collect();
        assert!(
            unplaced.is_empty(),
            "{} children never got a GlobalTransform, e.g. {:?}",
            unplaced.len(),
            &unplaced[..unplaced.len().min(5)]
        );
    }

    #[test]
    fn every_tree_spawns_all_lod_tiers_with_ranges() {
        // LOD is wired per entity, and a missing tier is invisible: the tree
        // simply vanishes between two boundaries. Assert the structure, not
        // that "a tree exists".
        let mut app = App::new();
        app.add_plugins((MinimalPlugins, super::WorldPlugin))
            .init_resource::<ButtonInput<KeyCode>>()
            .init_resource::<Assets<Mesh>>()
            .init_resource::<Assets<StandardMaterial>>()
            .init_resource::<Assets<crate::plugins::season::TreeSeasonMaterial>>();
        app.world_mut().spawn((
            Camera3d::default(),
            Transform::from_xyz(CENTER, 1.6, CENTER),
        ));
        for _ in 0..30 {
            app.update();
        }

        let count = |app: &mut App, name: &'static str| -> usize {
            let mut query = app.world_mut().query::<&Name>();
            query
                .iter(app.world())
                .filter(|n| n.as_str() == name)
                .count()
        };
        let near = count(&mut app, "Tree");
        let mid = count(&mut app, "TreeMid");
        let far = count(&mut app, "TreeFar");
        assert!(near > 0, "no trees spawned");
        assert_eq!(near, mid, "every tree needs a mid tier");
        assert_eq!(near, far, "every tree needs a far tier");

        // The two distant tiers must carry ranges, and they must be the ones
        // the LOD table defines.
        for (name, expected) in [("TreeMid", TreeLod::Mid), ("TreeFar", TreeLod::Far)] {
            let mut query = app
                .world_mut()
                .query::<(&Name, Option<&bevy::camera::visibility::VisibilityRange>)>();
            let mut checked = 0usize;
            for (entity_name, range) in query.iter(app.world()) {
                if entity_name.as_str() != name {
                    continue;
                }
                let range = range.unwrap_or_else(|| panic!("{name} has no VisibilityRange"));
                // `VisibilityRange` has no Debug impl, so compare the parts.
                let want = expected.range();
                assert!(
                    range.start_margin.start == want.start_margin.start
                        && range.end_margin.end == want.end_margin.end,
                    "{name} carries the wrong visibility range ({:?}-{:?}, expected {:?}-{:?})",
                    range.start_margin.start,
                    range.end_margin.end,
                    want.start_margin.start,
                    want.end_margin.end
                );
                checked += 1;
            }
            assert_eq!(checked, near, "{name} is missing on some trees");
        }

        // The near tier deliberately has no range (it is visible from 0 m and
        // `VisibilityRange` cannot express an open start).
        let mut query = app
            .world_mut()
            .query::<(&Name, Option<&bevy::camera::visibility::VisibilityRange>)>();
        for (entity_name, range) in query.iter(app.world()) {
            if entity_name.as_str() == "Tree" {
                assert!(range.is_none(), "the near tier must not be range-gated");
            }
        }
    }

    #[test]
    fn placement_assets_and_meshes_are_allocated_once() {
        // Regression: `build_placement_assets` ran every frame with no guard.
        // It re-baked 12 tree meshes, rebuilt every building material and
        // replaced the `PlacementAssets` resource each frame — measured over
        // 240 frames: meshes 13 -> 3242, materials 15 -> 3411. Buildings also
        // allocated a fresh `Cuboid` per object on every chunk load.
        let mut app = App::new();
        app.add_plugins((MinimalPlugins, super::WorldPlugin))
            .init_resource::<ButtonInput<KeyCode>>()
            .init_resource::<Assets<Mesh>>()
            .init_resource::<Assets<StandardMaterial>>()
            .init_resource::<Assets<crate::plugins::season::TreeSeasonMaterial>>();
        app.world_mut().spawn((
            Camera3d::default(),
            Transform::from_xyz(CENTER, 1.6, CENTER),
        ));

        for _ in 0..60 {
            app.update();
        }
        let meshes_after_warmup = app.world().resource::<Assets<Mesh>>().len();
        let materials_after_warmup = app.world().resource::<Assets<StandardMaterial>>().len();

        // Run a lot more frames with no world change: nothing may be
        // allocated.
        for _ in 0..120 {
            app.update();
        }
        assert_eq!(
            app.world().resource::<Assets<Mesh>>().len(),
            meshes_after_warmup,
            "mesh assets grow every frame; the placement assets are being rebuilt"
        );
        assert_eq!(
            app.world().resource::<Assets<StandardMaterial>>().len(),
            materials_after_warmup,
            "material assets grow every frame"
        );
    }

    #[test]
    fn procedural_trees_stream_with_their_chunks() {
        // R3.9 regression: the first tree implementation was a one-shot debug
        // gallery pinned at a fixed spot, so walking away left the world
        // treeless until the player toggled it off and on again. Trees must
        // be part of chunk content and despawn with their chunk root.
        use bevy::mesh::Indices;
        use bevy::prelude::Mesh3d;

        let mut app = App::new();
        app.add_plugins((MinimalPlugins, super::WorldPlugin))
            .init_resource::<ButtonInput<KeyCode>>()
            .init_resource::<Assets<Mesh>>()
            .init_resource::<Assets<StandardMaterial>>()
            .init_resource::<Assets<crate::plugins::season::TreeSeasonMaterial>>();
        app.world_mut().spawn((
            Camera3d::default(),
            Transform::from_xyz(CENTER, 1.6, CENTER),
        ));

        // `build_placement_assets` runs in PreUpdate; the first update brings
        // up the layout, later ones stream chunks in.
        for _ in 0..12 {
            app.update();
        }

        let trees_near = count_tree_entities(&mut app);
        assert!(
            trees_near > 0,
            "no procedural trees spawned around the spawn point"
        );

        // Every tree must be a single baked mesh, not one entity per branch.
        let single_mesh_trees = {
            let mut query = app.world_mut().query::<(&Mesh3d, &Name)>();
            query
                .iter(app.world())
                .filter(|(_, name)| name.as_str() == "Tree")
                .count()
        };
        assert_eq!(
            single_mesh_trees, trees_near,
            "each tree should be exactly one Mesh3d entity"
        );

        // Baked trees must be real geometry (a shared handle could still be
        // an empty mesh if the bake silently failed).
        {
            let handle = {
                let mut query = app.world_mut().query::<&Mesh3d>();
                query
                    .iter(app.world())
                    .next()
                    .map(|m| m.0.clone())
                    .expect("a tree mesh handle")
            };
            let meshes = app.world().resource::<Assets<Mesh>>();
            let mesh = meshes.get(&handle).expect("tree mesh exists");
            let tris = match mesh.indices() {
                Some(Indices::U32(i)) => i.len() / 3,
                Some(Indices::U16(i)) => i.len() / 3,
                None => 0,
            };
            assert!(tris > 0, "baked tree mesh is empty");
        }

        // Teleport far enough that the original chunks leave the window.
        let mut cameras = app
            .world_mut()
            .query_filtered::<&mut Transform, With<Camera3d>>();
        for mut transform in cameras.iter_mut(app.world_mut()) {
            transform.translation = Vec3::new(CENTER + 300.0, 1.6, CENTER);
        }
        for _ in 0..80 {
            app.update();
        }

        // The world stays treed at the new location, and no tree survives at
        // the old one: the tree set follows the player rather than sitting
        // still like the old gallery.
        let trees_far = count_tree_entities(&mut app);
        assert!(
            trees_far > 0,
            "no trees after walking away; trees are not streaming with chunks"
        );

        // Every tree must belong to an ACTIVE chunk root. Counting trees that
        // merely have a `ChildOf` misses orphans entirely — the earlier
        // version of this check passed even with trees spawned unparented,
        // which is exactly the "pinned gallery" failure mode.
        let manager = app.world().resource::<ChunkManager>();
        let live_roots: std::collections::HashSet<Entity> =
            manager.active.values().copied().collect();

        let mut query = app
            .world_mut()
            .query::<(&Transform, &Name, Option<&ChildOf>)>();
        let mut orphaned = 0usize;
        let mut behind = 0usize;
        for (transform, name, parent) in query.iter(app.world()) {
            if name.as_str() != "Tree" {
                continue;
            }
            match parent {
                Some(parent) if live_roots.contains(&parent.parent()) => {}
                _ => orphaned += 1,
            }
            // Trees left standing in the abandoned half of the map mean the
            // set is not following the player.
            if transform.translation.x <= CENTER {
                behind += 1;
            }
        }
        assert_eq!(
            orphaned, 0,
            "{orphaned} trees are not attached to an active chunk root"
        );
        assert_eq!(
            behind, 0,
            "{behind} trees are still standing in the abandoned area"
        );
    }

    fn count_tree_entities(app: &mut App) -> usize {
        let mut query = app.world_mut().query::<&Name>();
        query
            .iter(app.world())
            .filter(|name| name.as_str() == "Tree")
            .count()
    }

    #[test]
    fn player_chunk_changes_are_tracked_and_explored() {
        use crate::world::explored::{ExploredBitmap, PlayerChunkChanged};
        let mut app = App::new();
        app.add_plugins((MinimalPlugins, super::WorldPlugin))
            .init_resource::<ButtonInput<KeyCode>>();
        app.world_mut().spawn((
            Camera3d::default(),
            Transform::from_xyz(CENTER, 1.6, CENTER),
        ));

        app.update();
        app.update();

        let bitmap = app.world().resource::<ExploredBitmap>();
        assert!(
            bitmap.is_explored((8, 8)),
            "the starting chunk counts as explored"
        );
        assert!(!bitmap.is_empty());

        // Teleport two chunks east: the change message fires with the old
        // chunk recorded.
        let mut cameras = app
            .world_mut()
            .query_filtered::<&mut Transform, With<Camera3d>>();
        for mut transform in cameras.iter_mut(app.world_mut()) {
            transform.translation = Vec3::new(CENTER + 120.0, 1.6, CENTER);
        }
        app.update();
        app.update();

        let messages: Vec<_> = app
            .world()
            .resource::<bevy::ecs::message::Messages<PlayerChunkChanged>>()
            .iter_current_update_messages()
            .copied()
            .collect();
        let last = messages.last().expect("a chunk change was sent");
        assert_eq!(last.new_chunk, (10, 8));
        assert_eq!(last.old_chunk, Some((8, 8)));

        let bitmap = app.world().resource::<ExploredBitmap>();
        assert!(bitmap.is_explored((10, 8)), "the new chunk is explored");
        assert!(bitmap.is_explored((8, 8)), "the old chunk stays explored");
    }

    #[test]
    fn layout_resource_is_available_after_generation() {
        use crate::world::WORLD_SIZE;
        let mut app = App::new();
        app.add_plugins((MinimalPlugins, super::WorldPlugin))
            .init_resource::<ButtonInput<KeyCode>>();
        app.world_mut().spawn((
            Camera3d::default(),
            Transform::from_xyz(CENTER, 1.6, CENTER),
        ));
        app.update();
        let graph = layout_resource(app.world()).expect("layout graph resource");
        assert!(!graph.blocks.is_empty());
        assert!(!graph.edges.is_empty());
        assert!((graph.bounds.x - WORLD_SIZE).abs() < 1e-3);
    }
}
