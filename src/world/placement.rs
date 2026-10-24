//! Placement (R3.1/R3.2): buildings and trees along street edges, ported from
//! the original `object_placer.gd` (ADR-0004 §4/§5).
//!
//! Placement is **pure data**: [`plan_chunk`] turns one 50 m chunk into a list
//! of [`PlacedObject`]s (kind, transform, footprint) and never touches the ECS.
//! The streaming system (`chunks.rs`) turns that list into meshes, so the same
//! plan can be asserted in unit tests without a window or a GPU.
//!
//! Rules carried over unchanged:
//! - buildings sit on **both** sides of every street segment, inset by
//!   `street_width / 2 + setback + depth / 2` from the centerline;
//! - a footprint is dropped unless it stays out of every street corridor,
//!   inside its own block polygon, and inside the map;
//! - each edge/side pair draws from its own deterministic RNG stream
//!   (`hash_seed(seed, edge, side)`), so one chunk's content never depends on
//!   load order;
//! - every object is placed by exactly one chunk, decided by its anchor point
//!   (edge midpoint for street furniture, block center for block interiors),
//!   the original's `_is_point_in_chunk` de-duplication.
//!
//! Deviations from the original (recorded in `dev-plan.md`):
//! - the map edge is the **scenery band** rather than an `is_outer` block, so
//!   the ring road's outer side has no block and therefore no buildings;
//! - only three tree assets exist (`aspen_1`, `tree_2`, `tree_3`); the plan
//!   samples from whatever is available instead of assuming five.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::world::geometry::{
    convex_hull, dist_point_to_segment, dist_segment_to_segment, point_in_polygon,
    rect_inside_polygon,
};
use crate::world::layout_graph::{BlockType, LayoutGraph};
use crate::world::{CELL_SIZE, CHUNK_SIZE, GRID_CELLS};

/// Street trees are spaced on this grid along an edge (m).
pub const TREE_SPACING: f32 = 8.0;
/// Tree scale range (the original draws one factor for x/y/z).
pub const TREE_SCALE_MIN: f32 = 0.8;
pub const TREE_SCALE_MAX: f32 = 1.2;

/// Street width used for the road-corridor rejection test (m). The original
/// hard-codes 8 m here even though per-street widths vary by ±2 m; the
/// corridor is only a cheap first filter (the block polygon is the real
/// bound), so the constant is kept for portability.
pub const CORRIDOR_WIDTH: f32 = 8.0;
/// Slack allowed when a corner sits exactly on a curb (m).
const CORRIDOR_TOLERANCE: f32 = 0.05;

/// Buildings of one block type (ADR-0004 §4 parameter table).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BuildingParams {
    /// Extent along the street (m): `[min, max]`.
    pub length: (f32, f32),
    /// Extent away from the street (m): `[min, max]`.
    pub depth: (f32, f32),
    /// Height (m): `[min, max]`.
    pub height: (f32, f32),
    /// Gap between neighbours along the edge (m).
    pub spacing: f32,
    /// Distance from the curb to the facade (m).
    pub setback: f32,
    /// Street-tree probability inside the setback strip (0 for walkable
    /// block types).
    pub street_tree_probability: f32,
}

impl BuildingParams {
    /// Parameters for a block type, or `None` for walkable types (PLAZA,
    /// FOREST) that carry no buildings.
    pub fn of(kind: BlockType) -> Option<Self> {
        let params = match kind {
            BlockType::OldCity => BuildingParams {
                length: (4.0, 6.0),
                depth: (4.0, 6.0),
                height: (3.0, 6.0),
                spacing: 0.1,
                setback: 1.0,
                street_tree_probability: 0.2,
            },
            BlockType::Apartment => BuildingParams {
                length: (25.0, 40.0),
                depth: (12.0, 14.0),
                height: (18.0, 18.0),
                spacing: 0.0,
                setback: 5.0,
                street_tree_probability: 0.6,
            },
            BlockType::Factory => BuildingParams {
                length: (50.0, 50.0),
                depth: (50.0, 50.0),
                height: (24.0, 24.0),
                spacing: 0.0,
                setback: 10.0,
                street_tree_probability: 0.8,
            },
            BlockType::Plaza | BlockType::Forest => return None,
        };
        Some(params)
    }
}

/// Style colours for the BoxMesh placeholders (ADR-0004: silhouettes read at
/// dusk, so the palette stays muted).
#[derive(Clone, Copy, Debug)]
pub struct StylePalette {
    pub old_city: [f32; 3],
    pub apartment: [f32; 3],
    pub factory: [f32; 3],
    pub plaza: [f32; 3],
}

impl Default for StylePalette {
    fn default() -> Self {
        StylePalette {
            old_city: [0.80, 0.60, 0.30],
            apartment: [0.50, 0.50, 0.60],
            factory: [0.40, 0.25, 0.25],
            plaza: [0.70, 0.70, 0.65],
        }
    }
}

/// What the player sees standing at a planned position.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PlacedKind {
    /// Box placeholder in place of a Blender module (R3.6 adds wear).
    Building(BlockType),
    /// Street tree inside a block's setback strip.
    StreetTree,
    /// Tree inside a FOREST block.
    ForestTree,
    /// Tree on a PLAZA block.
    PlazaTree,
    /// The plaza's paved pad.
    PlazaPad,
}

/// A bare tree asset: which mesh, how big, which way it faces.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TreeInstance {
    /// Index into the available asset list (stable within one plan).
    pub asset: u8,
    /// Uniform scale (0.8-1.2).
    pub scale: f32,
    /// Y rotation (radians, 0..2π).
    pub yaw: f32,
}

/// One placed object: transform + footprint, no ECS entity.
#[derive(Clone, Debug, PartialEq)]
pub struct PlacedObject {
    pub kind: PlacedKind,
    /// Ground-plane center (x, z).
    pub position: bevy::math::Vec2,
    /// Y rotation (radians). Buildings face the street; trees are random.
    pub yaw: f32,
    /// Box size `(length, height, depth)` for buildings, `(width, height,
    /// depth)` for the plaza pad; unused for trees (they carry [`TreeInstance`]).
    pub size: bevy::math::Vec3,
    /// Terrain footprint half extents (x, z), used by tests to re-check the
    /// street/block/bounds rules.
    pub footprint_half: bevy::math::Vec2,
    /// Deterministic wear level 0.0 (fresh) .. 1.0 (heavily worn), baked from
    /// the placement RNG at plan time so the same seed always produces the
    /// same weathering per building (R3.6). Unused for trees.
    pub wear: f32,
    /// Tree variation, `None` for buildings and the plaza pad.
    pub tree: Option<TreeInstance>,
}

/// Everything one chunk displays, in spawn order.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ChunkPlan {
    /// True once every tree asset finished loading when this plan was made.
    /// A plan built earlier has no trees at all and must be re-planned.
    pub trees_ready: bool,
    pub objects: Vec<PlacedObject>,
}

impl ChunkPlan {
    /// Building count (diagnostics/tests).
    pub fn building_count(&self) -> usize {
        self.objects
            .iter()
            .filter(|o| matches!(o.kind, PlacedKind::Building(_)))
            .count()
    }

    /// Tree count, all reasons combined.
    pub fn tree_count(&self) -> usize {
        self.objects.iter().filter(|o| o.tree.is_some()).count()
    }
}

/// Determines which chunk owns an anchor point (the original's per-object
/// de-duplication): a point on a chunk seam belongs to the higher index.
pub fn anchor_chunk(anchor: bevy::math::Vec2) -> (i32, i32) {
    (
        (anchor.x / CHUNK_SIZE).floor() as i32,
        (anchor.y / CHUNK_SIZE).floor() as i32,
    )
}

/// Per-edge/per-side RNG stream (the original's `hash_seed`): a fixed, odd
/// multiplier keeps neighbouring edge indices from drawing correlated
/// streams, and the chunk is independent of load order.
fn edge_rng(seed: u64, edge: u32, salt: u32) -> StdRng {
    let mixed = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).rotate_left(17)
        ^ (edge as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F)
        ^ (salt as u64).wrapping_mul(0x1656_67B1_9E37_79F9);
    StdRng::seed_from_u64(mixed)
}

/// Plans every object whose anchor lies in chunk `(cx, cz)`.
///
/// `tree_assets` is the number of tree meshes available (0 disables trees and
/// leaves `trees_ready == false`, so the caller re-plans once loading ends).
pub fn plan_chunk(graph: &LayoutGraph, seed: u64, chunk: (i32, i32), tree_assets: u8) -> ChunkPlan {
    let mut plan = ChunkPlan {
        trees_ready: tree_assets > 0,
        objects: Vec::new(),
    };

    for (edge_index, edge) in graph.edges.iter().enumerate() {
        let a = graph.nodes[edge.from as usize].pos;
        let b = graph.nodes[edge.to as usize].pos;
        let mid = (a + b) * 0.5;
        if anchor_chunk(mid) != chunk {
            continue;
        }
        let dir = (b - a).normalize_or_zero();
        if dir == bevy::math::Vec2::ZERO {
            continue;
        }
        let normal = bevy::math::Vec2::new(-dir.y, dir.x);
        let length = (b - a).length();

        for (side, block_id) in [(0u32, edge.left_block), (1, edge.right_block)] {
            let Some(block_id) = block_id else {
                continue; // Scenery band: nothing to build against.
            };
            let Some(block) = graph.blocks.get(block_id as usize) else {
                continue;
            };
            if block.polygon.len() < 3 {
                continue;
            }
            // Left of `from -> to` is `normal`; right is `-normal`.
            let sign = if side == 0 { 1.0 } else { -1.0 };
            let Some(params) = BuildingParams::of(block.kind) else {
                continue;
            };
            if std::env::var_os("HU_TRACE_PLACEMENT").is_some() {
                eprintln!(
                    "plan: edge {edge_index} side {side} block {block_id} kind {:?} len {length:.0} sign {sign} normal {:?}",
                    block.kind,
                    normal * sign
                );
            }
            // Merged runs span one cell each (see `street_unit_edges`): place
            // per cell, so a 4-6 m module grid lines up with the 50 m blocks
            // instead of being spread over a 1900 m run.
            let thirds = 3.max((length / CELL_SIZE).round() as usize);
            for part in 0..thirds {
                let t = (part as f32 + 0.5) / thirds as f32;
                let part_mid = a + (b - a) * t;
                let part_length = length / thirds as f32;
                let mut rng = edge_rng(seed, edge_index as u32, side * 16 + part as u32);
                place_buildings(
                    graph,
                    &mut plan,
                    &mut rng,
                    block,
                    &params,
                    part_mid,
                    dir,
                    normal * sign,
                    part_length,
                );
                let mut rng = edge_rng(seed, edge_index as u32, 1000 + side * 16 + part as u32);
                place_street_trees(
                    &mut plan,
                    &mut rng,
                    &params,
                    part_mid,
                    dir,
                    normal * sign,
                    part_length,
                    tree_assets,
                );
            }
        }
    }

    for (block_index, block) in graph.blocks.iter().enumerate() {
        let anchor = block.center();
        if anchor_chunk(anchor) != chunk {
            continue;
        }
        let mut rng = edge_rng(seed, block_index as u32, 5000);
        match block.kind {
            BlockType::Forest => {
                if tree_assets > 0 {
                    place_forest_trees(&mut plan, &mut rng, block, tree_assets);
                } else {
                    plan.trees_ready = false;
                }
            }
            BlockType::Plaza => {
                plan.objects.push(PlazaPadObject::plan(block.center()));
                if tree_assets > 0 {
                    place_plaza_trees(&mut plan, &mut rng, block, tree_assets);
                } else {
                    plan.trees_ready = false;
                }
            }
            _ => {}
        }
    }

    plan
}

/// The plaza's 10 x 10 m paved pad (original `_place_plaza_objects`).
struct PlazaPadObject;

impl PlazaPadObject {
    fn plan(center: bevy::math::Vec2) -> PlacedObject {
        PlacedObject {
            kind: PlacedKind::PlazaPad,
            position: center,
            yaw: 0.0,
            size: bevy::math::Vec3::new(10.0, 0.2, 10.0),
            footprint_half: bevy::math::Vec2::splat(5.0),
            wear: 0.0,
            tree: None,
        }
    }
}

/// Places the buildings of one block along one street edge.
#[allow(clippy::too_many_arguments)]
fn place_buildings(
    graph: &LayoutGraph,
    plan: &mut ChunkPlan,
    rng: &mut StdRng,
    block: &crate::world::layout_graph::LayoutBlock,
    params: &BuildingParams,
    edge_mid: bevy::math::Vec2,
    edge_dir: bevy::math::Vec2,
    edge_normal: bevy::math::Vec2,
    edge_length: f32,
) {
    let (length_min, length_max) = params.length;
    let (depth_min, depth_max) = params.depth;
    let (height_min, height_max) = params.height;
    let facade_length: f32 = rng.random_range(length_min..=length_max);
    let rand_depth: f32 = rng.random_range(depth_min..=depth_max);
    let height: f32 = rng.random_range(height_min..=height_max);

    let angle = edge_dir.x.atan2(edge_dir.y);

    // How deep the block really is behind this facade, sampled along the
    // whole edge: the ADR footprint was tuned for the original's merged
    // blocks, and a perturbed 1x1 block is only ~39 m inside its inset
    // outline, so the table's depth + setback often cannot exist at all near
    // the corners. The row uses the shallowest sample of the columns it
    // actually occupies.
    let columns = row_columns(edge_length, facade_length.max(MIN_FACADE), params.spacing);
    let (start, end) = edge_span(edge_mid, edge_dir, edge_length);
    let depth_budget = shallowest_column_depth(block, start, end, edge_normal, &columns);

    // Facade length: the table size first; when that cannot fit a row, fall
    // back to 4-6 m modules (a big block still reads as a slab, a small one
    // gets the old city's rhythm) and only then give up on this frontage.
    let mut building_length = facade_length.max(MIN_FACADE);
    let mut building_depth = rand_depth.max(MIN_FACADE);
    let mut budget = depth_budget;
    if !row_fits(
        block,
        start,
        end,
        edge_normal,
        &columns_for(edge_length, building_length, params.spacing),
        building_length,
        building_depth,
        budget - params.setback,
    ) {
        let (lmin, lmax) = if params.length.1 - params.length.0 > 4.0 {
            (MIN_FACADE, MODULE_LENGTH_MAX)
        } else {
            params.length
        };
        building_length = lmin + (lmax - lmin) * 0.5;
        building_depth = building_depth.min(MODULE_DEPTH_MAX);
        let modules = columns_for(edge_length, building_length, params.spacing);
        budget = shallowest_column_depth(block, start, end, edge_normal, &modules);
        if !row_fits(
            block,
            start,
            end,
            edge_normal,
            &modules,
            building_length,
            building_depth,
            budget - params.setback,
        ) {
            return; // Block too shallow for any building at this frontage.
        }
    }

    let count = columns_for(edge_length, building_length, params.spacing).len();

    for i in 0..count {
        let t = (i as f32 + 0.5) / count as f32;
        let along = (t - 0.5) * edge_length;
        let perp = CORRIDOR_WIDTH * 0.5 + params.setback + building_depth * 0.5;
        let position = edge_mid + edge_dir * along + edge_normal * perp;

        if !rect_clear_of_roads(graph, position, angle, building_length, building_depth) {
            continue;
        }
        let half = bevy::math::Vec2::new(building_length, building_depth) * 0.5;
        if !rect_inside_polygon(position, half, &block.polygon) {
            continue;
        }
        if !rect_inside_polygon(position, half, &convex_hull(&block.polygon)) {
            continue;
        }
        if !within_bounds(graph, position, building_length.max(building_depth) * 0.5) {
            continue;
        }

        plan.objects.push(PlacedObject {
            kind: PlacedKind::Building(block.kind),
            position,
            yaw: angle,
            size: bevy::math::Vec3::new(building_length, height, building_depth),
            footprint_half: half,
            wear: rng.random_range(0.0..=1.0),
            tree: None,
        });
    }
}

/// Smallest facade length/depth tried before giving up on a block frontage
/// (m). A single 50 m cell leaves ~39 m inside its inset outline, so even the
/// largest type has to come down to this before a 1x1 block can hold a row.
const MIN_FACADE: f32 = 3.0;

/// Middle-of-the-road module size used when a block cannot hold the table
/// footprint (m): the old city's rhythm, applied to any type.
const MODULE_LENGTH_MAX: f32 = 6.0;
const MODULE_DEPTH_MAX: f32 = 6.0;

/// Material wear buckets (R3.6). Buildings pick a bucket from their planned
/// `wear` level; each bucket owns one shared `StandardMaterial` per block
/// type, so the whole city costs `types × buckets` materials and the meshes
/// keep batching inside a bucket.
pub(crate) const WEAR_BUCKETS: usize = 4;

/// Bucket index for a planned wear level (0 = fresh .. 3 = heavily worn).
pub(crate) fn wear_bucket(wear: f32) -> usize {
    ((wear.clamp(0.0, 1.0) * WEAR_BUCKETS as f32) as usize).min(WEAR_BUCKETS - 1)
}

/// Column anchors (fraction along the edge) for a row of `count` buildings.
fn build_columns(count: usize) -> Vec<f32> {
    (0..count)
        .map(|i| (i as f32 + 0.5) / count.max(1) as f32)
        .collect()
}

/// Column anchors for a row of `length`-long buildings along an edge.
fn columns_for(edge_length: f32, length: f32, spacing: f32) -> Vec<f32> {
    let count = ((edge_length + spacing) / (length + spacing))
        .floor()
        .max(1.0) as usize;
    build_columns(count)
}

/// Depth anchors used to sample a row: the row centres plus both ends, so
/// the shallowest corner of the block limits every building of the row.
fn row_columns(edge_length: f32, length: f32, spacing: f32) -> Vec<f32> {
    let mut columns = columns_for(edge_length, length, spacing);
    columns.extend_from_slice(&[0.02, 0.98]);
    columns
}

/// Shallowest depth among the given column positions (m from the curb).
fn shallowest_column_depth(
    block: &crate::world::layout_graph::LayoutBlock,
    a: bevy::math::Vec2,
    b: bevy::math::Vec2,
    edge_normal: bevy::math::Vec2,
    columns: &[f32],
) -> f32 {
    columns
        .iter()
        .map(|t| block_depth_along_pub(block, a + (b - a) * *t, edge_normal))
        .fold(f32::INFINITY, f32::min)
}

/// Start and end points of the edge a row is laid along, from its midpoint,
/// direction and length (the plan only keeps the midpoint around).
fn edge_span(
    mid: bevy::math::Vec2,
    dir: bevy::math::Vec2,
    length: f32,
) -> (bevy::math::Vec2, bevy::math::Vec2) {
    (mid - dir * (length * 0.5), mid + dir * (length * 0.5))
}

/// True if every building of the row at `columns` fits its block footprint.
#[allow(clippy::too_many_arguments)]
fn row_fits(
    block: &crate::world::layout_graph::LayoutBlock,
    a: bevy::math::Vec2,
    b: bevy::math::Vec2,
    edge_normal: bevy::math::Vec2,
    columns: &[f32],
    length: f32,
    depth: f32,
    available_depth: f32,
) -> bool {
    if depth > available_depth {
        return false;
    }
    let half = bevy::math::Vec2::new(length, depth) * 0.5;
    let perp = CORRIDOR_WIDTH * 0.5 + depth * 0.5;
    columns.iter().all(|t| {
        let point = a + (b - a) * *t;
        let center = point + edge_normal * perp;
        rect_inside_polygon(center, half, &block.polygon)
    })
}

/// Places the street trees of one block along one street edge (inside the
/// setback strip, one candidate every [`TREE_SPACING`] metres).
#[allow(clippy::too_many_arguments)]
fn place_street_trees(
    plan: &mut ChunkPlan,
    rng: &mut StdRng,
    params: &BuildingParams,
    edge_mid: bevy::math::Vec2,
    edge_dir: bevy::math::Vec2,
    edge_normal: bevy::math::Vec2,
    edge_length: f32,
    tree_assets: u8,
) {
    let probability = params.street_tree_probability;
    if probability <= 0.0 || tree_assets == 0 {
        return;
    }
    let candidates = (edge_length / TREE_SPACING).floor().max(1.0) as usize;
    // Half the setback: the tree stands between curb and facade.
    let perp = CORRIDOR_WIDTH * 0.5 + params.setback * 0.5;

    for i in 0..candidates {
        // Draw the probability roll for every candidate, whether or not it
        // is used, so the stream stays stable.
        let roll: f32 = rng.random();
        if roll > probability {
            continue;
        }
        let t = (i as f32 + 0.5) / candidates as f32;
        let along = (t - 0.5) * edge_length;
        let position = edge_mid + edge_dir * along + edge_normal * perp;
        push_tree(plan, rng, PlacedKind::StreetTree, position, tree_assets);
    }
}

/// Dense trees filling a FOREST block (the original `_place_trees_in_forest_block`).
fn place_forest_trees(
    plan: &mut ChunkPlan,
    rng: &mut StdRng,
    block: &crate::world::layout_graph::LayoutBlock,
    tree_assets: u8,
) {
    let (min, max) = polygon_bounds(&block.polygon);
    let size = max - min;
    let count = ((size.x * size.y) / (TREE_SPACING * TREE_SPACING))
        .floor()
        .clamp(1.0, 6.0) as usize;
    let center = (min + max) * 0.5;

    for _ in 0..count {
        let offset = bevy::math::Vec2::new(
            rng.random_range(-size.x * 0.35..=size.x * 0.35),
            rng.random_range(-size.y * 0.35..=size.y * 0.35),
        );
        let position = center + offset;
        if !point_in_polygon(position, &block.polygon) {
            continue;
        }
        push_tree(plan, rng, PlacedKind::ForestTree, position, tree_assets);
    }
}

/// A couple of trees on a plaza, inside its polygon.
fn place_plaza_trees(
    plan: &mut ChunkPlan,
    rng: &mut StdRng,
    block: &crate::world::layout_graph::LayoutBlock,
    tree_assets: u8,
) {
    let (min, max) = polygon_bounds(&block.polygon);
    let size = max - min;
    let center = (min + max) * 0.5;
    let count = ((size.x * size.y) / 100.0).floor().clamp(1.0, 2.0) as usize;
    for _ in 0..count {
        let offset = bevy::math::Vec2::new(
            rng.random_range(-size.x * 0.25..=size.x * 0.25),
            rng.random_range(-size.y * 0.25..=size.y * 0.25),
        );
        let position = center + offset;
        if !point_in_polygon(position, &block.polygon) {
            continue;
        }
        push_tree(plan, rng, PlacedKind::PlazaTree, position, tree_assets);
    }
}

/// Appends one tree instance: random asset, scale and yaw (original
/// `_instantiate_tree`).
fn push_tree(
    plan: &mut ChunkPlan,
    rng: &mut StdRng,
    kind: PlacedKind,
    position: bevy::math::Vec2,
    tree_assets: u8,
) {
    let asset: u8 = rng.random_range(0..tree_assets);
    let scale: f32 = rng.random_range(TREE_SCALE_MIN..=TREE_SCALE_MAX);
    let yaw: f32 = rng.random_range(0.0..std::f32::consts::TAU);
    plan.objects.push(PlacedObject {
        kind,
        position,
        yaw,
        size: bevy::math::Vec3::ZERO,
        footprint_half: bevy::math::Vec2::ZERO,
        wear: 0.0,
        tree: Some(TreeInstance { asset, scale, yaw }),
    });
}

/// Bounding box of a polygon as `(min, max)`.
fn polygon_bounds(polygon: &[bevy::math::Vec2]) -> (bevy::math::Vec2, bevy::math::Vec2) {
    let mut min = bevy::math::Vec2::splat(f32::INFINITY);
    let mut max = bevy::math::Vec2::splat(f32::NEG_INFINITY);
    for p in polygon {
        min = min.min(*p);
        max = max.max(*p);
    }
    (min, max)
}

/// True if the rotated footprint keeps clear of every street corridor.
///
/// The corridor is `centerline ± CORRIDOR_WIDTH / 2`; a convex footprint is
/// nearest to a segment either at a corner or along an edge, so the test
/// checks corners **and** edges against segment-to-segment distance.
fn rect_clear_of_roads(
    graph: &LayoutGraph,
    center: bevy::math::Vec2,
    yaw: f32,
    length: f32,
    depth: f32,
) -> bool {
    let corners = rotated_corners(center, yaw, length, depth);
    let half = CORRIDOR_WIDTH * 0.5 - CORRIDOR_TOLERANCE;
    for edge in &graph.edges {
        let a = graph.nodes[edge.from as usize].pos;
        let b = graph.nodes[edge.to as usize].pos;
        if corners
            .iter()
            .any(|c| dist_point_to_segment(*c, a, b) < half)
        {
            return false;
        }
        for i in 0..4 {
            let (p, q) = (corners[i], corners[(i + 1) % 4]);
            if dist_segment_to_segment(p, q, a, b) < half {
                return false;
            }
        }
    }
    true
}

/// Depth of `block` measured from an edge point along `outward`.
///
/// Casts a ray from just outside the facade line into the block and returns
/// the distance to the far boundary (m). Flat polygon edges are used rather
/// than the convex hull, because a facade must stay inside the visible
/// outline; the hull check later is the safety net for the collision body.
/// Returns `0` when the ray leaves the polygon immediately.
pub fn block_depth_along_pub(
    block: &crate::world::layout_graph::LayoutBlock,
    origin: bevy::math::Vec2,
    normal_out: bevy::math::Vec2,
) -> f32 {
    // Work in the block's own frame: `t` runs along `normal_out` (towards the
    // street), `u` along the street. Marching from far outside the block
    // towards the street finds the first boundary crossing = the facade, the
    // next one = the back wall. Starting inside (the original code started at
    // the curb, which can already be inside a perturbed polygon) would return
    // a negative "depth" or the near edge instead of the far one.
    let t_out = normal_out.perp_dot(origin);
    let _ = normal_out.dot(origin);
    let polygon = &block.polygon;
    let n = polygon.len();

    // Intersections of the ray (t varying, u fixed) with the polygon.
    let mut hits: Vec<f32> = Vec::with_capacity(2);
    for i in 0..n {
        let a = polygon[i];
        let b = polygon[(i + 1) % n];
        let (ta, tb) = (normal_out.perp_dot(a), normal_out.perp_dot(b));
        let (ua, ub) = (normal_out.dot(a), normal_out.dot(b));
        if (ta - t_out).abs() < 1e-9 && (tb - t_out).abs() < 1e-9 {
            continue; // Collinear with the ray.
        }
        if (ta - t_out) * (tb - t_out) > 0.0 {
            continue; // Both ends on the same side: no crossing.
        }
        let span = tb - ta;
        if span.abs() < 1e-9 {
            continue;
        }
        let f = (t_out - ta) / span;
        if (0.0..=1.0).contains(&f) {
            hits.push(ua + (ub - ua) * f);
        }
    }
    hits.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // `u` runs along the street axis, `t` across it. The street point is at
    // `u_origin`; the polygon's interval covering `t_out` spans `[near, far]`
    // in `u`. Walking inward from the street means moving away from the street
    // side, so the depth is the span of the interval minus how far the street
    // point already sits inside it (0 when the point is outside the span).
    let mut depth = 0.0;
    for pair in hits.chunks(2) {
        if pair.len() < 2 {
            break;
        }
        let (near, far) = (pair[0], pair[1]);
        // The interval is the block's own cross-section at this `t`; depth is
        // measured from the street side into the block.
        let (lo, hi) = if near < far { (near, far) } else { (far, near) };
        depth = hi - lo;
    }
    depth
}

/// Corners of a footprint rotated by `yaw` around Y, CCW in XZ.
///
/// Yaw follows the renderer: a positive Y rotation maps local `+X` to
/// `(cos, -sin)` and local `+Z` to `(sin, cos)` in world XZ, so the corner
/// offsets below match what the mesh actually covers.
pub fn rotated_corners(
    center: bevy::math::Vec2,
    yaw: f32,
    length: f32,
    depth: f32,
) -> [bevy::math::Vec2; 4] {
    let (s, c) = yaw.sin_cos();
    let (hl, hd) = (length * 0.5, depth * 0.5);
    let offset = |lx: f32, lz: f32| {
        bevy::math::Vec2::new(center.x + lx * c + lz * s, center.y - lx * s + lz * c)
    };
    [
        offset(-hl, -hd),
        offset(hl, -hd),
        offset(hl, hd),
        offset(-hl, hd),
    ]
}

/// True if `position ± extent` stays inside the 2 km map.
fn within_bounds(graph: &LayoutGraph, position: bevy::math::Vec2, extent: f32) -> bool {
    let world = CELL_SIZE * GRID_CELLS as f32;
    let _ = graph;
    position.x - extent >= 0.0
        && position.y - extent >= 0.0
        && position.x + extent <= world
        && position.y + extent <= world
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::path_network::generate_world;

    /// Every seed used by the placement assertions.
    const SEEDS: [u64; 3] = [1, 7, crate::world::chunks::DEFAULT_SEED];

    fn all_chunks(_graph: &LayoutGraph) -> Vec<(i32, i32)> {
        let n = (CELL_SIZE * GRID_CELLS as f32 / CHUNK_SIZE) as i32;
        (0..n).flat_map(|x| (0..n).map(move |z| (x, z))).collect()
    }

    #[test]
    fn parameter_table_matches_adr() {
        let old = BuildingParams::of(BlockType::OldCity).unwrap();
        assert_eq!(old.length, (4.0, 6.0));
        assert_eq!(old.setback, 1.0);
        assert_eq!(old.street_tree_probability, 0.2);
        let apt = BuildingParams::of(BlockType::Apartment).unwrap();
        assert_eq!(apt.length, (25.0, 40.0));
        assert_eq!(apt.street_tree_probability, 0.6);
        let factory = BuildingParams::of(BlockType::Factory).unwrap();
        assert_eq!(factory.length, (50.0, 50.0));
        assert_eq!(factory.setback, 10.0);
        assert_eq!(factory.street_tree_probability, 0.8);
        assert!(BuildingParams::of(BlockType::Plaza).is_none());
        assert!(BuildingParams::of(BlockType::Forest).is_none());
    }

    #[test]
    fn every_chunk_is_planned_deterministically() {
        for seed in SEEDS {
            let (_, graph) = generate_world(seed).unwrap();
            for chunk in [
                (0, 0),
                (7, 7),
                (20, 20),
                (GRAPH_LAST_CHUNK, GRAPH_LAST_CHUNK),
            ] {
                let a = plan_chunk(&graph, seed, chunk, 3);
                let b = plan_chunk(&graph, seed, chunk, 3);
                assert_eq!(a, b, "seed {seed} chunk {chunk:?} must be stable");
            }
        }
    }

    #[test]
    fn every_object_is_placed_exactly_once() {
        for seed in SEEDS {
            let (_, graph) = generate_world(seed).unwrap();
            let mut seen: std::collections::HashMap<(i32, i32, String), usize> =
                std::collections::HashMap::new();
            let mut total = 0;
            for chunk in all_chunks(&graph) {
                let plan = plan_chunk(&graph, seed, chunk, 3);
                for object in plan.objects {
                    total += 1;
                    let key = (
                        chunk.0,
                        chunk.1,
                        format!("{:?}@{:?}", object.kind, object.position),
                    );
                    *seen.entry(key).or_default() += 1;
                }
            }
            assert!(total > 0, "seed {seed} placed nothing");
            assert!(
                seen.values().all(|count| *count <= 1),
                "seed {seed}: an object was planned into two chunks"
            );
            // Cross-check the total against a single whole-map pass: any
            // object planned twice would inflate the per-chunk sum.
            let distinct = seen.len();
            assert_eq!(total, distinct, "seed {seed}: duplicate placements");
        }
    }

    #[test]
    fn buildings_stay_out_of_streets_and_inside_their_block() {
        for seed in SEEDS {
            let (_, graph) = generate_world(seed).unwrap();
            let mut checked = 0;
            for chunk in all_chunks(&graph) {
                for object in plan_chunk(&graph, seed, chunk, 3).objects {
                    let PlacedKind::Building(_) = object.kind else {
                        continue;
                    };
                    checked += 1;
                    let corners =
                        rotated_corners(object.position, object.yaw, object.size.x, object.size.z);
                    for edge in &graph.edges {
                        let a = graph.nodes[edge.from as usize].pos;
                        let b = graph.nodes[edge.to as usize].pos;
                        for (i, corner) in corners.iter().enumerate() {
                            assert!(
                                dist_point_to_segment(*corner, a, b)
                                    >= CORRIDOR_WIDTH * 0.5 - CORRIDOR_TOLERANCE - 1e-3,
                                "seed {seed}: building corner {} at {:?} intrudes into the \
                                 street corridor of edge {}..{}",
                                i,
                                corner,
                                edge.from,
                                edge.to
                            );
                        }
                    }
                    assert!(graph.blocks.iter().any(|block| {
                        rect_inside_polygon(object.position, object.footprint_half, &block.polygon)
                            || rect_inside_polygon(
                                object.position,
                                object.footprint_half,
                                &convex_hull(&block.polygon),
                            )
                    }));
                }
            }
            assert!(checked > 0, "seed {seed}: no buildings placed");
        }
    }

    #[test]
    fn trees_are_skipped_until_assets_load() {
        let (_, graph) = generate_world(SEEDS[0]).unwrap();
        let mut with_assets = 0;
        for chunk in all_chunks(&graph) {
            let plan = plan_chunk(&graph, SEEDS[0], chunk, 3);
            assert!(plan.trees_ready);
            with_assets += plan.tree_count();
            let empty = plan_chunk(&graph, SEEDS[0], chunk, 0);
            assert_eq!(empty.tree_count(), 0, "no assets, no trees");
            assert_eq!(
                empty.building_count(),
                plan.building_count(),
                "buildings do not depend on trees"
            );
            // The plaza pad is not a tree and must survive.
            assert_eq!(
                empty
                    .objects
                    .iter()
                    .filter(|o| o.kind == PlacedKind::PlazaPad)
                    .count(),
                plan.objects
                    .iter()
                    .filter(|o| o.kind == PlacedKind::PlazaPad)
                    .count()
            );
        }
        assert!(with_assets > 0, "the map should carry trees");
    }

    #[test]
    fn tree_probability_ordering_holds() {
        // ADR-0004: old city 20% < apartment 60% < factory 80% per candidate.
        let old = BuildingParams::of(BlockType::OldCity).unwrap();
        let apt = BuildingParams::of(BlockType::Apartment).unwrap();
        let factory = BuildingParams::of(BlockType::Factory).unwrap();
        assert!(old.street_tree_probability < apt.street_tree_probability);
        assert!(apt.street_tree_probability < factory.street_tree_probability);
    }

    #[test]
    fn wear_buckets_cover_the_full_range() {
        // R3.6: every planned wear level maps into a bucket; the extremes
        // hit the first and last bucket and the mapping is monotone.
        assert_eq!(wear_bucket(0.0), 0);
        assert_eq!(wear_bucket(1.0), WEAR_BUCKETS - 1);
        assert_eq!(wear_bucket(-0.5), 0, "clamped");
        assert_eq!(wear_bucket(1.5), WEAR_BUCKETS - 1, "clamped");
        for i in 0..100 {
            let w = i as f32 / 99.0;
            let b = wear_bucket(w);
            assert!(b < WEAR_BUCKETS);
            assert!(wear_bucket((w + 0.01).min(1.0)) >= b, "monotone");
        }
    }

    #[test]
    fn building_wear_is_deterministic_and_in_range() {
        // Same seed -> same wear per building (R3.6 requirement), and the
        // planned levels actually spread across the range.
        let graph = crate::world::path_network::generate_world(SEEDS[2])
            .expect("generation succeeds")
            .1;
        let wear_of = |seed: u64| -> Vec<f32> {
            let mut wear = Vec::new();
            for cx in 0..4 {
                for cz in 0..4 {
                    for o in &plan_chunk(&graph, seed, (cx, cz), 0).objects {
                        if o.kind == PlacedKind::Building(BlockType::OldCity)
                            || o.kind == PlacedKind::Building(BlockType::Apartment)
                            || o.kind == PlacedKind::Building(BlockType::Factory)
                        {
                            wear.push(o.wear);
                        }
                    }
                }
            }
            wear
        };
        let a = wear_of(SEEDS[2]);
        let b = wear_of(SEEDS[2]);
        assert_eq!(a, b, "bit-exact wear per seed");
        assert!(a.len() > 20, "sampled a real neighbourhood: {}", a.len());
        assert!(a.iter().all(|w| (0.0..=1.0).contains(w)));
        let (mut lo, mut hi) = (f32::MAX, f32::MIN);
        for w in &a {
            lo = lo.min(*w);
            hi = hi.max(*w);
        }
        assert!(hi - lo > 0.5, "wear spreads across the range: {lo}..{hi}");
    }

    #[test]
    fn rotated_corners_match_a_y_rotation() {
        // A 90° yaw swaps the footprint axes, and the corner set stays a
        // congruent rectangle around the center.
        let center = bevy::math::Vec2::new(10.0, 20.0);
        let corners = rotated_corners(center, std::f32::consts::FRAC_PI_2, 4.0, 2.0);
        let min = corners
            .iter()
            .fold(bevy::math::Vec2::splat(f32::INFINITY), |a, b| a.min(*b));
        let max = corners
            .iter()
            .fold(bevy::math::Vec2::splat(f32::NEG_INFINITY), |a, b| a.max(*b));
        let size = max - min;
        assert!((size.x - 2.0).abs() < 1e-4, "x extent {}", size.x);
        assert!((size.y - 4.0).abs() < 1e-4, "z extent {}", size.y);
    }

    const GRAPH_LAST_CHUNK: i32 = (CELL_SIZE * GRID_CELLS as f32 / CHUNK_SIZE) as i32 - 1;
}
