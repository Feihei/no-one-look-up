//! Baking a [`TreeSkeleton`] into the two meshes a tree needs, plus the
//! per-vertex season data the foliage shader reads (R3.5, R3.10).
//!
//! This module merges a tree into **one pair of meshes per species-and-variant
//! key**, so a chunk spawns two `Mesh3d` entities per tree and every chunk
//! sharing a key reuses the same handles (and therefore the same draw state).
//!
//! # Two meshes per tree: wood and leaves
//!
//! [`TreeParts::wood`] holds the branches (opaque silhouette material) and
//! [`TreeParts::leaves`] the foliage cards (alpha-tested leaf texture). They
//! were one mesh while the foliage was opaque icosphere blobs. Cards need
//! `AlphaMode::Mask`, and a merged mesh would force that alpha mode onto the
//! branches too, where a texture-less alpha test erases the trunk.
//!
//! # Leaves are cards, not blobs
//!
//! A 1.2 m sphere is a solid ball at any distance, which is what "a clump of
//! leaves" looked like in game. The silhouette the design wants — sky cut into
//! pieces by real leaf outlines — cannot come from a sphere at any subdivision.
//! Each foliage cluster instead becomes a handful of crossed quads textured
//! with a baked leaf alpha mask ([`crate::world::leaf_shape`]).
//!
//! A card costs 2 triangles against a blob's 20, so the switch also cuts the
//! foliage triangle budget by roughly 90% while making the outline correct.
//!
//! # Season data on the vertex stream
//!
//! | Channel | Meaning |
//! |---|---|
//! | `COLOR.r` | sway weight (ADR-0004: "the vertex-colour R channel controls each vertex's sway strength") |
//! | `COLOR.g` | foliage mask, 0 = branch |
//! | `COLOR.b` | per-card sway phase, so neighbouring cards do not move in lockstep |
//! | `UV_0` | leaf texture coordinates (leaves only) |
//!
//! Leaf density no longer collapses geometry: that was the opaque-blob era,
//! when there was no alpha to work with. It is the material's base-colour alpha
//! multiplied into the texture alpha, which the depth prepass honours (see
//! `plugins::season`). `UV_1`, which used to carry the collapse offset, is gone
//! with it.
//!
//! Silhouette material (ADR-0001): near-black, fully rough, no specular, so the
//! tree reads as a shape against the dusk sky rather than as shaded geometry.
//! The foliage tint is a *multiplier* on that base colour, not a replacement,
//! so all four seasons stay inside the silhouette budget.

use bevy::asset::RenderAssetUsages;
use bevy::camera::visibility::VisibilityRange;
use bevy::mesh::{ConicalFrustumMeshBuilder, Indices, Mesh, MeshBuilder, PrimitiveTopology};
use bevy::prelude::*;

use crate::world::tree_shape::{TREE_PROFILES, TreeProfile, TreeSkeleton, grow_tree};

/// Branches are tapered 5-sided tubes. At silhouette distance a 5-gon is
/// indistinguishable from a 6-gon, and branches are ~40% of the triangle budget,
/// so the odd side is worth having.
const BRANCH_SIDES: u32 = 5;
/// Cards per foliage cluster near the camera.
///
/// The first version used four cards the SIZE OF THE CLUSTER, which is a 2.4 m
/// plane carrying a 10 cm leaf: at that scale the outline is a giant rectangle
/// no texture can hide. The card count and the card size have to move together
/// -- many small cards, not a few huge ones -- which is also what makes the sky
/// get cut into pieces rather than occluded in slabs.
///
/// 22 cards of ~0.26 of the cluster radius each: 44 triangles against a
/// 20-face blob, still cheaper than the sphere it replaced, and small enough
/// that the gaps between cards are leaves-sized rather than cluster-sized.
const CARDS_PER_CLUSTER: usize = 22;
/// Cards per cluster at the far tier, where individual leaves are sub-pixel and
/// only the broken-up outline matters.
const FAR_CARDS_PER_CLUSTER: usize = 6;
/// A card's half-extent as a fraction of the cluster radius.
///
/// Small enough that several cards overlap to form one cluster, large enough
/// that a leaf unit is still legible at a few metres.
const CARD_REACH_FRACTION: f32 = 0.26;
/// Branches thinner than this (m) are dropped from the bake: at any walking
/// distance they are sub-pixel, and the finest twigs are ~40% of all parts.
const MIN_BAKED_RADIUS: f32 = 0.035;

/// Sway weight written into `COLOR.r` for foliage vertices.
///
/// Leaves carry the motion because they are what catches the wind; the branch
/// tube they hang from is wood and barely moves. A trunk that swayed like the
/// crown would read as rubber.
pub const SWAY_WEIGHT_FOLIAGE: f32 = 1.0;
/// Sway weight for branch vertices. Not zero: a completely rigid trunk under a
/// moving crown reads as two disconnected objects at the join.
pub const SWAY_WEIGHT_BRANCH: f32 = 0.12;

/// How strongly height modulates the sway, per unit of normalised height.
///
/// Wind moves a crown more than the underside of the same crown; a uniform
/// offset would slide the whole thing sideways and read as a floating object.
pub const SWAY_HEIGHT_GAIN: f32 = 0.35;

/// How much detail a baked mesh keeps.
///
/// Foliage dominates the triangle count, so the tiers differ mostly in how many
/// leaf cards survive. A dusk silhouette is a flat dark shape, so dropping
/// cards and detail costs little visually while cutting the frame cost of the
/// far field sharply.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TreeLod {
    /// Full detail: every branch and every cluster's cards.
    Near,
    /// Coarser foliage: the crown keeps its shape but fewer cards.
    Mid,
    /// A couple of big cards: at this distance the crown is a smudge.
    Far,
}

impl TreeLod {
    /// Tiers in draw order, nearest first.
    pub const ALL: [TreeLod; 3] = [TreeLod::Near, TreeLod::Mid, TreeLod::Far];

    /// Where this tier takes over from the previous one, in metres from the
    /// camera. The last entry is where trees disappear entirely.
    ///
    /// 40 m / 110 m / 220 m: `CHUNK_AHEAD` is 4 chunks (200 m), so the far tier
    /// only ever covers the last stretch the player can see. Close enough to the
    /// walk that the crown still reads, far enough that a 1.2-2.0 m/s walker
    /// sees the swap rarely.
    pub const BOUNDARIES: [f32; 4] = [0.0, 40.0, 110.0, 220.0];

    /// The visibility range for this tier, or `None` for `Near` (visible from
    /// 0 m, which `VisibilityRange` expresses as an open start).
    pub fn range(self) -> VisibilityRange {
        let index = Self::ALL.iter().position(|t| *t == self).unwrap_or(0);
        VisibilityRange::abrupt(Self::BOUNDARIES[index], Self::BOUNDARIES[index + 1])
    }

    /// Smallest branch radius kept at this tier (m).
    fn min_branch_radius(self) -> f32 {
        match self {
            TreeLod::Near => MIN_BAKED_RADIUS,
            // Mid drops twigs entirely: at 40 m+ a 3.5 cm twig is under a pixel
            // and only carried the crown's inner structure.
            TreeLod::Mid => 0.06,
            TreeLod::Far => f32::INFINITY,
        }
    }

    /// Keep every `n`th cluster (1 = all).
    fn blob_stride(self) -> usize {
        match self {
            TreeLod::Near => 1,
            TreeLod::Mid => 2,
            TreeLod::Far => 4,
        }
    }

    /// Clusters below this radius (m) are dropped; keeps the big crown masses
    /// and discards the small filler.
    fn min_blob_radius(self) -> f32 {
        match self {
            TreeLod::Near => 0.0,
            TreeLod::Mid => 0.30,
            TreeLod::Far => 0.0,
        }
    }

    /// Fatten the surviving clusters so the crown still fills its silhouette
    /// once most of them are gone.
    fn blob_scale(self) -> f32 {
        match self {
            TreeLod::Near => 1.0,
            TreeLod::Mid => 1.18,
            TreeLod::Far => 1.55,
        }
    }

    /// Cards per cluster at this tier.
    fn cards_per_cluster(self) -> usize {
        match self {
            TreeLod::Near | TreeLod::Mid => CARDS_PER_CLUSTER,
            TreeLod::Far => FAR_CARDS_PER_CLUSTER,
        }
    }
}

/// Which skeleton to bake: species (index into [`TREE_PROFILES`]) and a
/// per-species variation id.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TreeMeshKey {
    pub species: u8,
    pub variant: u8,
}

impl TreeMeshKey {
    /// Number of distinct variants baked per species.
    pub const VARIANTS: u8 = 4;

    /// Maps the placement planner's `(asset, wear)` onto a bake key.
    ///
    /// `asset` selects the species (the planner samples `0..tree_asset_count`)
    /// and `wear` — which trees also carry, unused for anything else — picks the
    /// variant. That gives neighbouring trees of one species different shapes
    /// without extra RNG plumbing through placement.
    pub fn from_placement(asset: u8, wear: f32) -> Self {
        let variant = (wear * Self::VARIANTS as f32)
            .floor()
            .clamp(0.0, (Self::VARIANTS - 1) as f32) as u8;
        Self {
            species: asset % TREE_PROFILES.len() as u8,
            variant,
        }
    }
}

/// World-scale silhouette values for a procedural tree.
///
/// Split out from the material so the numbers can be asserted headless: the
/// foliage shader multiplies the base colour by a per-season tint, and the crown
/// must stay this dark in every season.
pub const SILHOUETTE_BASE_COLOR_SRGB: [f32; 3] = [0.030, 0.038, 0.032];

/// The shipped silhouette as a plain `StandardMaterial`.
///
/// Only used by the gallery examples; the world builds the season-aware
/// `ExtendedMaterial` in [`crate::plugins::season`].
pub fn silhouette_material() -> StandardMaterial {
    StandardMaterial {
        base_color: Color::srgb(
            SILHOUETTE_BASE_COLOR_SRGB[0],
            SILHOUETTE_BASE_COLOR_SRGB[1],
            SILHOUETTE_BASE_COLOR_SRGB[2],
        ),
        perceptual_roughness: 1.0,
        metallic: 0.0,
        // No specular at all: a rim highlight would read as wet plastic and
        // break the cut-out look the whole design rests on.
        reflectance: 0.0,
        ..default()
    }
}

/// One tree baked into the two meshes it needs (see the module header).
#[derive(Clone, Debug)]
pub struct TreeParts {
    /// Branches: opaque silhouette material.
    pub wood: Mesh,
    /// Foliage cards: alpha-masked leaf texture.
    pub leaves: Mesh,
}

impl TreeParts {
    /// Total triangles across both meshes, for the budgets the tests assert.
    pub fn triangle_count(&self) -> usize {
        triangle_count(&self.wood) + triangle_count(&self.leaves)
    }
}

/// Grows the skeleton behind `key` and bakes it.
pub fn bake_tree_mesh(key: TreeMeshKey) -> TreeParts {
    bake_tree_mesh_lod(key, TreeLod::Near)
}

/// [`bake_tree_mesh`] at a specific detail tier.
pub fn bake_tree_mesh_lod(key: TreeMeshKey, lod: TreeLod) -> TreeParts {
    let profile = TREE_PROFILES
        .get(key.species as usize)
        .copied()
        .unwrap_or(TREE_PROFILES[0]);
    // Deliberately a fixed gallery seed: a tree's SHAPE must not depend on world
    // position, or the mesh cache would miss on every chunk. It also keeps every
    // LOD of one tree the same tree.
    let skeleton = grow_tree(
        crate::world::tree_shape::TREE_SHAPE_SEED,
        key.species,
        key.variant as u64,
    );
    bake_skeleton_lod(&skeleton, &profile, lod)
}

/// Bakes every branch and leaf of `skeleton`.
pub fn bake_skeleton(skeleton: &TreeSkeleton, profile: &TreeProfile) -> TreeParts {
    bake_skeleton_lod(skeleton, profile, TreeLod::Near)
}

/// [`bake_skeleton`] at a specific detail tier.
///
/// Besides the geometry this writes the season attributes documented at the top
/// of the module.
pub fn bake_skeleton_lod(
    skeleton: &TreeSkeleton,
    profile: &TreeProfile,
    lod: TreeLod,
) -> TreeParts {
    let mut wood_parts: Vec<Mesh> = Vec::new();
    let mut leaf_parts: Vec<Mesh> = Vec::new();
    let min_branch = lod.min_branch_radius();
    let top = trunk_top(skeleton);

    for branch in &skeleton.branches {
        let start = Vec3::from_array(branch.start);
        let end = Vec3::from_array(branch.end);
        let delta = end - start;
        let length = delta.length();
        // Sub-millimetre twigs contribute nothing at any viewing distance.
        if length < 1e-3 {
            continue;
        }
        if branch.radius < min_branch {
            continue;
        }
        let radius = branch.radius;
        // A frustum whose top is ~(radius_ratio) of its bottom gives the branch
        // its taper without a second mesh.
        let taper = profile.radius_ratio.clamp(0.25, 0.95);
        let unit = ConicalFrustumMeshBuilder::new(taper, 1.0, 1.0, BRANCH_SIDES)
            .build()
            // Unit tube runs +Y from its base; move the base to the origin before
            // orienting so `start` lands where it should.
            .translated_by(Vec3::new(0.0, 0.5, 0.0));
        let rotation = Quat::from_rotation_arc(Vec3::Y, delta / length);
        let placed = unit.transformed_by(
            Transform::from_translation(start)
                .with_rotation(rotation)
                .with_scale(Vec3::new(radius, length, radius)),
        );
        wood_parts.push(with_branch_season_data(placed, top));
    }

    // Foliage: keep every `stride`th cluster and drop the small filler. The
    // stride walk is deterministic (index order), so each tier stays a stable
    // subset of the same crown rather than a re-randomised one.
    let stride = lod.blob_stride();
    let min_blob = lod.min_blob_radius();
    let blob_scale = lod.blob_scale();
    let cards = lod.cards_per_cluster();

    for (index, foliage) in skeleton.foliage.iter().enumerate() {
        if index % stride != 0 {
            continue;
        }
        if foliage.radius < min_blob {
            continue;
        }
        let center = Vec3::from_array(foliage.center);
        // The cluster's radius now sizes the CARDS rather than a ball: a card
        // spans the cluster's diameter, so a few of them cover the volume the
        // sphere used to.
        let cluster_radius = (foliage.radius * blob_scale).max(0.05);
        leaf_parts.push(leaf_cards(center, cluster_radius, index, cards, top));
    }

    TreeParts {
        wood: merge_mesh_parts(wood_parts),
        leaves: merge_mesh_parts(leaf_parts),
    }
}

/// Height of the tallest point of the tree (m), used to normalise the sway
/// height term.
fn trunk_top(skeleton: &TreeSkeleton) -> f32 {
    skeleton
        .branches
        .iter()
        .map(|b| b.end[1])
        .chain(skeleton.foliage.iter().map(|f| f.center[1] + f.radius))
        .fold(0.0_f32, f32::max)
}

/// Writes `COLOR` for one branch part.
///
/// No UVs: branches are texture-less. They used to carry an offset pack in
/// `UV_0`/`UV_1` for the density collapse, which is gone now that density is the
/// leaf texture's alpha.
fn with_branch_season_data(mut mesh: Mesh, top: f32) -> Mesh {
    let count = mesh.count_vertices();
    let Some(positions) = mesh.attribute(Mesh::ATTRIBUTE_POSITION).cloned() else {
        return mesh;
    };
    let bevy::mesh::VertexAttributeValues::Float32x3(positions) = positions else {
        return mesh;
    };

    let mut colors = Vec::with_capacity(count);
    for p in &positions {
        colors.push([
            SWAY_WEIGHT_BRANCH * (1.0 + SWAY_HEIGHT_GAIN * height_term(p[1], top)),
            0.0,
            0.0,
            1.0,
        ]);
    }
    mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, colors);
    mesh
}

/// Builds `cards` crossed quads spanning a sphere of `reach` metres at `center`.
///
/// Cards are rotated about the vertical axis at even intervals plus a
/// per-cluster offset, with a slow tilt, so a crown has cards facing every
/// direction instead of a fan of parallel planes.
fn leaf_cards(center: Vec3, cluster_radius: f32, cluster: usize, cards: usize, top: f32) -> Mesh {
    let mut positions: Vec<[f32; 3]> = Vec::with_capacity(cards * 4);
    let mut normals: Vec<[f32; 3]> = Vec::with_capacity(cards * 4);
    let mut uvs: Vec<[f32; 2]> = Vec::with_capacity(cards * 4);
    let mut colors: Vec<[f32; 4]> = Vec::with_capacity(cards * 4);
    let mut indices: Vec<u32> = Vec::with_capacity(cards * 6);

    // Per-cluster sway phase, so neighbouring cards do not move in lockstep.
    // Lockstep motion across a crown reads as the whole tree being dragged
    // rather than as wind moving through it.
    let phase = ((cluster as f32 * 0.618_034).fract()).clamp(0.0, 1.0);
    // Cluster-level yaw offset, so two clusters never stack their cards at the
    // same angles.
    let yaw0 = (cluster as f32 * 2.399_963) % std::f32::consts::TAU;
    let sway = SWAY_WEIGHT_FOLIAGE * (1.0 + SWAY_HEIGHT_GAIN * height_term(center.y, top));

    // Deterministic scatter, so the same tree always bakes the same cards.
    let mut hash = (cluster as u32).wrapping_mul(0x9E37_79B9).wrapping_add(1);
    let mut next_unit = || {
        hash ^= hash << 13;
        hash ^= hash >> 17;
        hash ^= hash << 5;
        (hash & 0xFFFF) as f32 / 65535.0
    };

    for card in 0..cards {
        let yaw = yaw0 + std::f32::consts::PI * card as f32 / cards as f32;
        // A slow tilt keeps the cards from all being vertical slices; a card
        // that is perfectly edge-on would disappear from an overhead view.
        let tilt = (card as f32 * 0.7).sin() * 0.35;
        let rotation = Quat::from_euler(EulerRot::YXZ, yaw, tilt, 0.0);

        // Cards sit on a SHELL, not throughout the volume.
        //
        // Scattering them through the volume was the second attempt and it did
        // not fix the "solid clump" reading: cards near the middle are hidden by
        // cards in front of them, so they cost triangles without ever showing.
        // Real foliage does the same thing for the same reason -- leaves grow at
        // the outside of a crown where light is -- so the shell is both cheaper
        // and more truthful: the middle stays open, and a sightline through the
        // crown goes: leaves, gap, leaves.
        //
        // The radius varies a little so the shell is a fuzzy ball rather than a
        // paper lantern.
        let dir = Vec3::new(
            next_unit() * 2.0 - 1.0,
            next_unit() * 2.0 - 1.0,
            next_unit() * 2.0 - 1.0,
        );
        let dir = if dir.length_squared() < 1e-4 {
            Vec3::Y
        } else {
            dir.normalize()
        };
        let shell = cluster_radius * (0.72 + 0.28 * next_unit());
        let card_center = center + dir * shell;

        // A smaller card than the cluster, so several of them compose a clump.
        // Slightly taller than wide: a leaf unit (one blade or one pinnate leaf)
        // is longer than it is wide, and the texture's own aspect does the rest.
        let half_w = cluster_radius * CARD_REACH_FRACTION;
        let half_h = half_w * 1.15;
        let corners = [
            [-half_w, -half_h],
            [half_w, -half_h],
            [half_w, half_h],
            [-half_w, half_h],
        ];
        let base = positions.len() as u32;
        for (corner, [u, v]) in corners
            .iter()
            .zip([[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]])
        {
            let local = rotation * Vec3::new(corner[0], corner[1], 0.0);
            positions.push((card_center + local).to_array());
            // Cards are drawn double-sided, so one normal per quad is enough;
            // the PBR shader flips it for back faces.
            normals.push((rotation * Vec3::Z).to_array());
            uvs.push([u, v]);
            colors.push([sway, 1.0, phase, 1.0]);
        }
        indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }

    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
    mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, colors);
    mesh.insert_indices(Indices::U32(indices));
    mesh
}

/// Normalised height of a vertex within the tree, clamped to 0..1.
fn height_term(y: f32, top: f32) -> f32 {
    if top <= 1e-3 {
        return 0.0;
    }
    (y / top).clamp(0.0, 1.0)
}

/// Merges parts that share an attribute set.
///
/// `Mesh::merge` requires identical attributes, and each group above does carry
/// one consistent set. A merge failure keeps what has been merged so far rather
/// than panicking — this runs inside chunk streaming, where a panic would take
/// the frame down.
fn merge_mesh_parts(parts: Vec<Mesh>) -> Mesh {
    let mut merged: Option<Mesh> = None;
    for part in parts {
        match &mut merged {
            None => merged = Some(part),
            Some(mesh) => {
                if let Err(err) = mesh.merge(&part) {
                    warn!("tree mesh merge failed: {err}");
                }
            }
        }
    }
    let mut merged = merged.unwrap_or_else(empty_mesh);
    merged.asset_usage = RenderAssetUsages::default();
    merged
}

/// Triangle count of a mesh, for the budgets the tests assert.
fn triangle_count(mesh: &Mesh) -> usize {
    match mesh.indices() {
        Some(Indices::U32(i)) => i.len() / 3,
        Some(Indices::U16(i)) => i.len() / 3,
        None => 0,
    }
}

/// An empty mesh with the attributes the merge path expects.
fn empty_mesh() -> Mesh {
    Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, Vec::<[f32; 3]>::new())
    .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, Vec::<[f32; 3]>::new())
    .with_inserted_indices(Indices::U32(Vec::new()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn colors(m: &Mesh) -> Vec<[f32; 4]> {
        match m.attribute(Mesh::ATTRIBUTE_COLOR).expect("colour") {
            bevy::mesh::VertexAttributeValues::Float32x4(v) => v.clone(),
            other => panic!("unexpected colour layout: {other:?}"),
        }
    }

    fn positions(m: &Mesh) -> Vec<[f32; 3]> {
        match m.attribute(Mesh::ATTRIBUTE_POSITION).expect("positions") {
            bevy::mesh::VertexAttributeValues::Float32x3(v) => v.clone(),
            other => panic!("unexpected position layout: {other:?}"),
        }
    }

    #[test]
    fn both_meshes_have_geometry() {
        // The tree ships as two meshes; an empty either way means a silent hole
        // in the crown, or a floating crown with no trunk.
        for species in 0..TREE_PROFILES.len() as u8 {
            let parts = bake_tree_mesh(TreeMeshKey {
                species,
                variant: 0,
            });
            assert!(
                triangle_count(&parts.wood) > 0,
                "species {species}: no branches"
            );
            assert!(
                triangle_count(&parts.leaves) > 0,
                "species {species}: no leaf cards"
            );
        }
    }

    #[test]
    fn one_tree_stays_inside_the_triangle_budget() {
        // Cards are 2 triangles against a blob's 20, so the switch should keep a
        // tree cheap. If the card path regressed to spheres, this fails.
        for species in 0..TREE_PROFILES.len() as u8 {
            let parts = bake_tree_mesh(TreeMeshKey {
                species,
                variant: 0,
            });
            let total = parts.triangle_count();
            assert!(total < 20_000, "species {species}: {total} triangles");
        }
    }

    #[test]
    fn lod_tiers_actually_get_cheaper() {
        // A "LOD" that loses 5% is three copies of the same tree. Measured on
        // cards, each tier must be noticeably cheaper than the one before.
        for species in 0..TREE_PROFILES.len() as u8 {
            let key = TreeMeshKey {
                species,
                variant: 0,
            };
            let near = bake_tree_mesh_lod(key, TreeLod::Near).triangle_count();
            let mid = bake_tree_mesh_lod(key, TreeLod::Mid).triangle_count();
            let far = bake_tree_mesh_lod(key, TreeLod::Far).triangle_count();
            assert!(
                near > 0 && mid > 0 && far > 0,
                "species {species}: empty LOD"
            );
            assert!(
                mid < near * 3 / 4,
                "species {species}: mid keeps {mid}/{near}, not a real reduction"
            );
            assert!(far < mid / 2, "species {species}: far keeps {far}/{mid}");
        }
    }

    #[test]
    fn leaf_mesh_is_quads_with_uvs() {
        // The leaf material samples a texture, so every leaf vertex needs a UV.
        // Without one the crown would sample a single texel and read as a solid
        // rectangle, which is exactly the look the cards exist to remove.
        let parts = bake_tree_mesh(TreeMeshKey {
            species: 0,
            variant: 0,
        });
        let positions = positions(&parts.leaves);
        let uvs = match parts.leaves.attribute(Mesh::ATTRIBUTE_UV_0).expect("uv0") {
            bevy::mesh::VertexAttributeValues::Float32x2(v) => v.clone(),
            other => panic!("unexpected uv layout: {other:?}"),
        };
        assert_eq!(uvs.len(), positions.len(), "every leaf vertex needs a UV");
        assert_eq!(positions.len() % 4, 0, "the leaf mesh is not made of quads");
        let max_u = uvs.iter().map(|uv| uv[0]).fold(0.0_f32, f32::max);
        let max_v = uvs.iter().map(|uv| uv[1]).fold(0.0_f32, f32::max);
        assert!(
            max_u >= 0.99 && max_v >= 0.99,
            "UVs do not span the texture"
        );
    }

    #[test]
    fn wood_carries_no_texture_and_no_foliage_mask() {
        // Branches use the opaque material, which has no texture. Their UV set
        // went away with the density collapse; if one came back it would need a
        // meaning.
        let parts = bake_tree_mesh(TreeMeshKey {
            species: 1,
            variant: 0,
        });
        for color in colors(&parts.wood) {
            assert_eq!(color[1], 0.0, "a branch vertex is flagged as foliage");
        }
        // The frustum builder emits its own UVs, which the opaque material
        // simply ignores. What matters is that they are not mistaken for the
        // old collapse pack: deleting that pack is what freed UV_0 for the leaf
        // texture, and a branch quad carrying leaf-like UVs would mean the two
        // paths got crossed again.
        if let Some(uvs) = parts.wood.attribute(Mesh::ATTRIBUTE_UV_0) {
            let bevy::mesh::VertexAttributeValues::Float32x2(uvs) = uvs else {
                panic!("unexpected uv layout");
            };
            assert!(
                uvs.iter().all(|uv| (0.0..=1.0).contains(&uv[0])),
                "branch UVs escaped the unit range"
            );
        }
    }

    #[test]
    fn leaves_are_flagged_so_the_shader_can_tint_them() {
        // The season tint is mixed in by the foliage mask in COLOR.g; without it
        // the crown keeps the bare silhouette in every season.
        let parts = bake_tree_mesh(TreeMeshKey {
            species: 2,
            variant: 0,
        });
        let mask = colors(&parts.leaves);
        assert!(!mask.is_empty());
        for color in &mask {
            assert_eq!(color[1], 1.0, "a leaf vertex is not flagged as foliage");
        }
        // The per-card phase must vary, or the cards sway in lockstep.
        let phases: std::collections::HashSet<u32> =
            mask.iter().map(|c| (c[2] * 1000.0) as u32).collect();
        assert!(phases.len() > 1, "every card shares one sway phase");
    }

    #[test]
    fn cards_cross_so_the_crown_has_volume() {
        // All cards at one angle would make the crown vanish edge-on.
        let parts = bake_tree_mesh(TreeMeshKey {
            species: 0,
            variant: 0,
        });
        let normals = match parts
            .leaves
            .attribute(Mesh::ATTRIBUTE_NORMAL)
            .expect("normals")
        {
            bevy::mesh::VertexAttributeValues::Float32x3(v) => v.clone(),
            other => panic!("unexpected normal layout: {other:?}"),
        };
        let mut sectors = std::collections::HashSet::new();
        for n in &normals {
            let angle = n[2].atan2(n[0]);
            let sector = ((angle / std::f32::consts::TAU) * 8.0).round() as i32;
            sectors.insert(sector.rem_euclid(8));
        }
        assert!(
            sectors.len() >= 4,
            "cards only face {} of 8 directions; the crown would disappear from some angles",
            sectors.len()
        );
    }

    #[test]
    fn sway_weight_grows_with_height() {
        // Wind moves the crown more than the trunk. A weight that ignored height
        // would slide the whole tree sideways as one rigid piece.
        let parts = bake_tree_mesh(TreeMeshKey {
            species: 1,
            variant: 0,
        });
        let all: Vec<([f32; 4], [f32; 3])> = colors(&parts.wood)
            .into_iter()
            .zip(positions(&parts.wood))
            .chain(
                colors(&parts.leaves)
                    .into_iter()
                    .zip(positions(&parts.leaves)),
            )
            .collect();
        let mut low = f32::INFINITY;
        let mut high: f32 = 0.0;
        let mut low_y = f32::INFINITY;
        let mut high_y: f32 = 0.0;
        for (color, p) in &all {
            if p[1] < low_y {
                low_y = p[1];
                low = color[0];
            }
            if p[1] > high_y {
                high_y = p[1];
                high = color[0];
            }
        }
        assert!(high_y > low_y, "tree has no vertical extent");
        assert!(
            high > low,
            "sway weight does not grow with height ({low} at {low_y} m, {high} at {high_y} m)"
        );
        for (color, _) in &all {
            assert!(
                (0.0..=2.5).contains(&color[0]),
                "sway weight {} out of range",
                color[0]
            );
        }
    }

    /// Fraction of the tree's crown bounding box that the leaf cards leave
    /// open, measured by projecting every card onto the XZ plane and testing
    /// coverage on a grid.
    ///
    /// This is the number the design actually cares about -- "sky cut into
    /// pieces by leaves" is a coverage question -- and it cannot be judged from
    /// triangle counts. A solid ball scores 0.0; a lacy crown scores high.
    fn xz_openness(parts: &TreeParts, mask: &crate::world::leaf_shape::AlphaMask) -> f32 {
        openness_along(parts, mask, true)
    }

    /// [`xz_openness`] but measured along the X axis instead: the SIDE view,
    /// which is what a player walking past actually sees. The top-down number
    /// says how much sky is visible from underneath, which answers a different
    /// question.
    fn side_openness(parts: &TreeParts, mask: &crate::world::leaf_shape::AlphaMask) -> f32 {
        openness_along(parts, mask, false)
    }

    fn openness_along(
        parts: &TreeParts,
        mask: &crate::world::leaf_shape::AlphaMask,
        top_down: bool,
    ) -> f32 {
        let positions = positions(&parts.leaves);
        let uvs = match parts.leaves.attribute(Mesh::ATTRIBUTE_UV_0).expect("uv0") {
            bevy::mesh::VertexAttributeValues::Float32x2(v) => v.clone(),
            other => panic!("unexpected uv layout: {other:?}"),
        };
        if positions.is_empty() {
            return 1.0;
        }
        let (mut min_x, mut max_x) = (f32::INFINITY, f32::NEG_INFINITY);
        let (mut min_z, mut max_z) = (f32::INFINITY, f32::NEG_INFINITY);
        // `top_down` projects onto XZ; otherwise onto YZ (a side elevation).
        let (ax, ay) = if top_down {
            (0usize, 2usize)
        } else {
            (2usize, 1usize)
        };
        for p in &positions {
            min_x = min_x.min(p[ax]);
            max_x = max_x.max(p[ax]);
            min_z = min_z.min(p[ay]);
            max_z = max_z.max(p[ay]);
        }
        const N: usize = 96;
        let mut covered = vec![false; N * N];
        for (quad, chunk) in positions.chunks(4).enumerate() {
            let quad_uv = &uvs[quad * 4..quad * 4 + 4];
            // Two triangles per quad.
            for tri in [[0usize, 1, 2], [0, 2, 3]] {
                let p0 = Vec3::from_array(chunk[tri[0]]);
                let p1 = Vec3::from_array(chunk[tri[1]]);
                let p2 = Vec3::from_array(chunk[tri[2]]);
                // Project onto the two chosen axes.
                let (p0, p1, p2) = (
                    (p0.to_array()[ax], p0.to_array()[ay]),
                    (p1.to_array()[ax], p1.to_array()[ay]),
                    (p2.to_array()[ax], p2.to_array()[ay]),
                );
                let area = (p1.0 - p0.0) * (p2.1 - p0.1) - (p1.1 - p0.1) * (p2.0 - p0.0);
                if area.abs() < 1e-9 {
                    continue;
                }
                for gz in 0..N {
                    for gx in 0..N {
                        let x = min_x + (gx as f32 + 0.5) / N as f32 * (max_x - min_x);
                        let z = min_z + (gz as f32 + 0.5) / N as f32 * (max_z - min_z);
                        let w0 = ((p1.0 - x) * (p2.1 - z) - (p1.1 - z) * (p2.0 - x)) / area;
                        let w1 = ((p2.0 - x) * (p0.1 - z) - (p2.1 - z) * (p0.0 - x)) / area;
                        let w2 = 1.0 - w0 - w1;
                        if w0 < 0.0 || w1 < 0.0 || w2 < 0.0 {
                            continue;
                        }
                        let u = w0 * quad_uv[tri[0]][0]
                            + w1 * quad_uv[tri[1]][0]
                            + w2 * quad_uv[tri[2]][0];
                        let v = w0 * quad_uv[tri[0]][1]
                            + w1 * quad_uv[tri[1]][1]
                            + w2 * quad_uv[tri[2]][1];
                        let tx = (u * mask.width as f32).clamp(0.0, mask.width as f32 - 1.0) as u32;
                        let ty =
                            (v * mask.height as f32).clamp(0.0, mask.height as f32 - 1.0) as u32;
                        if mask.get(tx, ty) >= 128 {
                            covered[gz * N + gx] = true;
                        }
                    }
                }
            }
        }
        let open = covered.iter().filter(|c| !**c).count();
        open as f32 / covered.len() as f32
    }

    #[test]
    fn the_crown_lets_sky_through() {
        // The design's whole reason for cards: looking up, the sky must be cut
        // into pieces rather than blocked by a mass. A sphere scores 0 here; the
        // first card version measured under 20% and read as a solid clump.
        let profile = crate::world::leaf_shape::find_profile("poplar")
            .copied()
            .unwrap_or(crate::world::leaf_shape::leaf_profiles()[0]);
        let mask = crate::world::leaf_shape::bake_leaf_alpha(&profile, 256);
        for species in 0..TREE_PROFILES.len() as u8 {
            let parts = bake_tree_mesh(TreeMeshKey {
                species,
                variant: 0,
            });
            let openness = xz_openness(&parts, &mask);
            let side = side_openness(&parts, &mask);
            assert!(
                openness > 0.35,
                "species {species}: only {:.0}% of the crown leaves sky visible; it reads as a solid clump",
                openness * 100.0
            );
            // The side elevation is the view that matters on the street: a
            // crown can be open from underneath and still block the whole
            // horizon band when seen from a walking eye height.
            assert!(
                side > 0.30,
                "species {species}: only {:.0}% of the side elevation shows through",
                side * 100.0
            );
        }
    }

    #[test]
    fn lod_ranges_tile_the_distances_without_gaps_or_overlap() {
        for pair in TreeLod::ALL.windows(2) {
            let (lower, upper) = (pair[0], pair[1]);
            assert_eq!(
                lower.range().end_margin.end,
                upper.range().start_margin.start,
                "{lower:?} and {upper:?} leave a gap or overlap"
            );
        }
        for distance in [0.0, 39.9, 40.0, 109.9, 110.0, 219.9] {
            let visible = TreeLod::ALL
                .iter()
                .filter(|lod| !lod.range().is_culled(distance))
                .count();
            assert_eq!(visible, 1, "{distance} m: {visible} tiers visible at once");
        }
    }

    #[test]
    fn keys_vary_by_species_and_variant() {
        let mut seen = std::collections::HashSet::new();
        for species in 0..TREE_PROFILES.len() as u8 {
            for variant in 0..TreeMeshKey::VARIANTS {
                seen.insert((species, variant));
            }
        }
        assert_eq!(
            seen.len(),
            TREE_PROFILES.len() * TreeMeshKey::VARIANTS as usize
        );
        for wear in [0.0, 0.5, 0.99, 1.0] {
            assert!(TreeMeshKey::from_placement(9, wear).variant < TreeMeshKey::VARIANTS);
        }
    }

    #[test]
    fn same_key_bakes_the_same_mesh() {
        let key = TreeMeshKey {
            species: 1,
            variant: 2,
        };
        let a = bake_tree_mesh(key);
        let b = bake_tree_mesh(key);
        assert_eq!(
            format!("{:?}", positions(&a.leaves)),
            format!("{:?}", positions(&b.leaves)),
            "bake must be deterministic"
        );
    }
}
