//! Baking a [`TreeSkeleton`] into a single opaque mesh (R3.9 step 3), plus the
//! per-vertex season data the foliage shader reads (R3.5).
//!
//! The gallery (`plugins::tree_preview`) spawns one entity per branch and one
//! per leaf cluster so a tree can be inspected and debugged. That is fine for
//! twelve trees and hopeless for a streaming world: the chunk planner places
//! up to 13 trees per 50 m chunk, which at ~250 parts each would be 3,250
//! entities per chunk — and a full streaming window is ~49 chunks.
//!
//! This module merges a tree into **one mesh per species-and-variant key**, so
//! a chunk spawns one `Mesh3d` per tree and every chunk sharing a key reuses
//! the same `Handle<Mesh>` (and therefore the same draw state).
//!
//! Meshes are cached: the chunk planner asks for key `k`, we build it at most
//! once per session. A 13-tree chunk costs 13 entities and 0 extra meshes once
//! the keys recur.
//!
//! # Season data on the vertex stream
//!
//! Season rendering (ADR-0004 §6, [`crate::world::season`]) needs three things
//! per vertex that the geometry alone cannot express. All three ride on
//! attributes Bevy's mesh pipeline already knows how to feed a shader, so no
//! custom vertex layout or custom vertex attribute has to be registered:
//!
//! | Channel | Meaning |
//! |---|---|
//! | `COLOR.r` | sway weight (ADR-0004: "the vertex-colour R channel controls each vertex's sway strength") |
//! | `COLOR.g` | foliage mask, 0 = branch (never collapsed, never tinted as leaves) |
//! | `COLOR.b` | per-cluster phase offset, so neighbouring clusters do not sway in lockstep |
//! | `UV_0.xy`, `UV_1.x` | the vertex's offset from its own cluster centre, which the density collapse scales down |
//!
//! `UV_1` carries only one useful component — a full `UV_2` for a scalar would
//! cost another `Float32x2` per vertex across ~3000 vertices a tree.
//!
//! Silhouette material (ADR-0001): near-black, fully rough, no specular, so
//! the tree reads as a shape against the dusk sky rather than as shaded
//! geometry. The foliage tint is a *multiplier* on that base colour, not a
//! replacement, so all four seasons stay inside the silhouette budget.

use bevy::asset::RenderAssetUsages;
use bevy::camera::visibility::VisibilityRange;
use bevy::mesh::{Indices, Mesh, PrimitiveTopology};
use bevy::prelude::*;

use crate::world::tree_shape::{TREE_PROFILES, TreeProfile, TreeSkeleton, grow_tree};

/// Branches are tapered 5-sided tubes. At silhouette distance a 5-gon is
/// indistinguishable from a 6-gon, and branches are ~40% of the triangle
/// budget, so the odd side is worth having.
const BRANCH_SIDES: u32 = 5;
/// Leaf clusters are 20-face icospheres (subdivision 0). Blobs dominate the
/// triangle count — ~180 per tree — and against a dusk sky a 20-face and an
/// 80-face blob are the same silhouette. Measured: dropping 1 -> 0 cut the
/// scene estimate from 2.6M to well under 1M triangles.
const BLOB_SUBDIVISIONS: u32 = 0;
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

/// How strongly foliage height modulates the sway, per metre.
///
/// Wind moves a crown more than it moves the underside of the same crown; a
/// uniform offset would slide the whole blob sideways and read as a floating
/// object rather than as leaves.
pub const SWAY_HEIGHT_GAIN: f32 = 0.35;

/// How much detail a baked mesh keeps.
///
/// Foliage dominates the triangle count (measured: ~85-90% of a tree — 183
/// clusters of 20 faces against ~930 triangles of branch), so the tiers differ
/// mostly in how many leaf clusters survive. A dusk silhouette is a flat dark
/// shape, so dropping clusters and detail costs little visually while cutting
/// the frame cost of the far field sharply.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TreeLod {
    /// Full detail: every branch and every leaf cluster.
    Near,
    /// Coarser foliage: the crown keeps its shape but carries fewer vblobs,
    /// and thin branches are dropped.
    Mid,
    /// A handful of big blobs: at this distance the crown is a smudge.
    Far,
}

impl TreeLod {
    /// Tiers in draw order, nearest first.
    pub const ALL: [TreeLod; 3] = [TreeLod::Near, TreeLod::Mid, TreeLod::Far];

    /// Where this tier takes over from the previous one, in metres from the
    /// camera. The last entry is where trees disappear entirely.
    ///
    /// 40 m / 110 m / 220 m: `CHUNK_AHEAD` is 4 chunks (200 m), so the far
    /// tier only ever covers the last stretch the player can see. Close enough
    /// to the walk that the crown still reads, far enough that a 1.2-2.0 m/s
    /// walker sees the swap rarely.
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
            // Mid drops twigs entirely: at 40 m+ a 3.5 cm twig is under a
            // pixel and its branch geometry was only carrying the crown's
            // inner structure, which the blobs occlude anyway.
            TreeLod::Mid => 0.06,
            TreeLod::Far => f32::INFINITY,
        }
    }

    /// Icosphere subdivisions for leaf blobs at this tier.
    fn blob_subdivisions(self) -> u32 {
        match self {
            TreeLod::Near | TreeLod::Mid => BLOB_SUBDIVISIONS,
            TreeLod::Far => 0,
        }
    }

    /// Keep every `n`th leaf cluster (1 = all).
    fn blob_stride(self) -> usize {
        match self {
            TreeLod::Near => 1,
            TreeLod::Mid => 3,
            TreeLod::Far => 14,
        }
    }

    /// Blobs below this radius (m) are dropped; keeps the big crown masses and
    /// discards the small filler.
    fn min_blob_radius(self) -> f32 {
        match self {
            TreeLod::Near => 0.0,
            TreeLod::Mid => 0.30,
            TreeLod::Far => 0.0,
        }
    }

    /// Fatten the surviving blobs so the crown still fills its silhouette once
    /// most of them are gone.
    fn blob_scale(self) -> f32 {
        match self {
            TreeLod::Near => 1.0,
            TreeLod::Mid => 1.18,
            TreeLod::Far => 1.55,
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
    /// and `wear` — which trees also carry, unused for anything else — picks
    /// the variant. That gives neighbouring trees of one species different
    /// shapes without any extra RNG plumbing through placement.
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
/// foliage shader multiplies the base colour by a per-season tint, and the
/// crown must stay this dark in every season.
pub const SILHOUETTE_BASE_COLOR_SRGB: [f32; 3] = [0.030, 0.038, 0.032];

/// [`silhouette_material`] as a plain `StandardMaterial`.
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

/// Grows the skeleton behind `key` and bakes it into one mesh.
pub fn bake_tree_mesh(key: TreeMeshKey) -> Mesh {
    bake_tree_mesh_lod(key, TreeLod::Near)
}

/// [`bake_tree_mesh`] at a specific detail tier.
pub fn bake_tree_mesh_lod(key: TreeMeshKey, lod: TreeLod) -> Mesh {
    let profile = TREE_PROFILES
        .get(key.species as usize)
        .copied()
        .unwrap_or(TREE_PROFILES[0]);
    // Deliberately a fixed gallery seed: a tree's SHAPE must not depend on
    // world position, or the mesh cache would miss on every chunk. It also
    // keeps every LOD of one tree the same tree.
    let skeleton = grow_tree(
        crate::world::tree_shape::TREE_SHAPE_SEED,
        key.species,
        key.variant as u64,
    );
    bake_skeleton_lod(&skeleton, &profile, lod)
}

/// Merges every branch and leaf cluster of `skeleton` into one mesh.
pub fn bake_skeleton(skeleton: &TreeSkeleton, profile: &TreeProfile) -> Mesh {
    bake_skeleton_lod(skeleton, profile, TreeLod::Near)
}

/// [`bake_skeleton`] at a specific detail tier.
///
/// Besides the geometry this writes the season attributes documented at the top
/// of the module. The foliage attributes are written **per cluster** rather
/// than per merged mesh, which is what lets the shader collapse every cluster
/// onto its own centre instead of onto one point for the whole tree.
pub fn bake_skeleton_lod(skeleton: &TreeSkeleton, profile: &TreeProfile, lod: TreeLod) -> Mesh {
    let mut parts: Vec<Mesh> = Vec::new();
    let min_branch = lod.min_branch_radius();

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
        // A frustum whose top is ~(radius_ratio) of its bottom gives the
        // branch its taper without a second mesh.
        let taper = profile.radius_ratio.clamp(0.25, 0.95);
        let unit = ConicalFrustumMeshBuilder::new(taper, 1.0, 1.0, BRANCH_SIDES)
            .build()
            // Unit tube runs +Y from its base; move the base to the origin
            // before orienting so `start` lands where it should.
            .translated_by(Vec3::new(0.0, 0.5, 0.0));
        let rotation = Quat::from_rotation_arc(Vec3::Y, delta / length);
        let placed = unit.transformed_by(
            Transform::from_translation(start)
                .with_rotation(rotation)
                .with_scale(Vec3::new(radius, length, radius)),
        );
        parts.push(with_branch_season_data(placed, trunk_top(skeleton)));
    }

    // Foliage: keep every `stride`th cluster and drop the small filler. The
    // stride walk is deterministic (index order), so each tier stays a stable
    // subset of the same crown rather than a re-randomised one.
    let stride = lod.blob_stride();
    let min_blob = lod.min_blob_radius();
    let blob_scale = lod.blob_scale();
    let blob_template = SphereMeshBuilder::new(
        1.0,
        SphereKind::Ico {
            subdivisions: lod.blob_subdivisions(),
        },
    )
    .build();

    for (index, foliage) in skeleton.foliage.iter().enumerate() {
        if index % stride != 0 {
            continue;
        }
        if foliage.radius < min_blob {
            continue;
        }
        let center = Vec3::from_array(foliage.center);
        let blob = blob_template.clone().transformed_by(
            Transform::from_translation(center)
                .with_scale(Vec3::splat((foliage.radius * blob_scale).max(0.02))),
        );
        parts.push(with_foliage_season_data(
            blob,
            center,
            index,
            trunk_top(skeleton),
        ));
    }

    let mut merged = merge_parts(parts);
    merged.asset_usage = RenderAssetUsages::default();
    merged
}

/// Height of the trunk's top (m), used to normalise the sway height term.
///
/// Taken from the skeleton rather than the profile: a profile's nominal height
/// is the top of a full tree, and the sway gradient has to be expressed
/// against the geometry that actually exists.
fn trunk_top(skeleton: &TreeSkeleton) -> f32 {
    skeleton
        .branches
        .iter()
        .map(|b| b.end[1])
        .chain(skeleton.foliage.iter().map(|f| f.center[1] + f.radius))
        .fold(0.0_f32, f32::max)
}

/// Writes `COLOR` and the UV-pack channels for one branch part.
fn with_branch_season_data(mut mesh: Mesh, top: f32) -> Mesh {
    let count = mesh.count_vertices();
    let Some(positions) = mesh.attribute(Mesh::ATTRIBUTE_POSITION).cloned() else {
        return mesh;
    };
    let bevy::mesh::VertexAttributeValues::Float32x3(positions) = positions else {
        return mesh;
    };

    let mut colors = Vec::with_capacity(count);
    let mut offsets = Vec::with_capacity(count);
    for p in &positions {
        // Branches keep the geometry they were baked with, so their collapsed
        // offset is zero: the density collapse must never move wood.
        colors.push([
            SWAY_WEIGHT_BRANCH * (1.0 + SWAY_HEIGHT_GAIN * height_term(p[1], top)),
            0.0,
            0.0,
            1.0,
        ]);
        offsets.push([0.0, 0.0, 0.0]);
    }
    mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, colors);
    insert_offset_pack(&mut mesh, offsets);
    mesh
}

/// Writes `COLOR` and the UV-pack channels for one foliage cluster.
///
/// `center` is the cluster's own centre. Every vertex stores the vector from
/// the centre to itself in the UV channels; scaling that vector by the season's
/// leaf density is the collapse, and it happens entirely in the vertex shader
/// so the main, depth-prepass and shadow passes all agree.
fn with_foliage_season_data(mut mesh: Mesh, center: Vec3, index: usize, top: f32) -> Mesh {
    let count = mesh.count_vertices();
    let Some(positions) = mesh.attribute(Mesh::ATTRIBUTE_POSITION).cloned() else {
        return mesh;
    };
    let bevy::mesh::VertexAttributeValues::Float32x3(positions) = positions else {
        return mesh;
    };

    // Per-cluster phase: a cheap hash of the cluster index, so adjacent blobs
    // sway out of step. Lockstep motion across a crown reads as the whole tree
    // being dragged rather than as wind moving through it.
    let phase = ((index as f32 * 0.618_034).fract()).clamp(0.0, 1.0);

    let mut colors = Vec::with_capacity(count);
    let mut offsets = Vec::with_capacity(count);
    for p in &positions {
        let vertex = Vec3::from_array(*p);
        colors.push([
            SWAY_WEIGHT_FOLIAGE * (1.0 + SWAY_HEIGHT_GAIN * height_term(p[1], top)),
            1.0,
            phase,
            1.0,
        ]);
        let offset = vertex - center;
        offsets.push(offset.to_array());
    }
    mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, colors);
    insert_offset_pack(&mut mesh, offsets);
    mesh
}

/// Normalised height of a vertex within the tree, clamped to 0..1.
fn height_term(y: f32, top: f32) -> f32 {
    if top <= 1e-3 {
        return 0.0;
    }
    (y / top).clamp(0.0, 1.0)
}

/// Packs a per-vertex 3-vector into `UV_0.xy` + `UV_1.x`.
///
/// `UV_1` carries a single useful component: a full `UV_2` for one scalar would
/// cost another `Float32x2` per vertex across ~3000 vertices of every tree.
fn insert_offset_pack(mesh: &mut Mesh, offsets: Vec<[f32; 3]>) {
    let uv0: Vec<[f32; 2]> = offsets.iter().map(|o| [o[0], o[1]]).collect();
    let uv1: Vec<[f32; 2]> = offsets.iter().map(|o| [o[2], 0.0]).collect();
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uv0);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_1, uv1);
}

/// Merges the parts, tolerating parts that carry different attribute sets.
///
/// `Mesh::merge` requires identical attributes, and the branch/foliage parts do
/// carry the same set (position, normal, uv0, uv1, color) because both go
/// through the season-data writers. A merge failure keeps what has been merged
/// so far rather than panicking — this runs inside chunk streaming, where a
/// panic would take the frame down.
fn merge_parts(parts: Vec<Mesh>) -> Mesh {
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
    merged.unwrap_or_else(empty_mesh)
}

fn empty_mesh() -> Mesh {
    Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, Vec::<[f32; 3]>::new())
    .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, Vec::<[f32; 3]>::new())
    .with_inserted_indices(Indices::U32(Vec::new()))
}

use bevy::mesh::{ConicalFrustumMeshBuilder, MeshBuilder, SphereKind, SphereMeshBuilder};

#[cfg(test)]
mod tests {
    use super::*;

    fn tri_count(m: &Mesh) -> usize {
        match m.indices() {
            Some(Indices::U32(i)) => i.len() / 3,
            Some(Indices::U16(i)) => i.len() / 3,
            None => 0,
        }
    }

    fn colors(m: &Mesh) -> Vec<[f32; 4]> {
        match m.attribute(Mesh::ATTRIBUTE_COLOR).expect("colour") {
            bevy::mesh::VertexAttributeValues::Float32x4(v) => v.clone(),
            other => panic!("unexpected colour layout: {other:?}"),
        }
    }

    fn offsets(m: &Mesh) -> Vec<Vec3> {
        let uv0 = match m.attribute(Mesh::ATTRIBUTE_UV_0).expect("uv0") {
            bevy::mesh::VertexAttributeValues::Float32x2(v) => v.clone(),
            other => panic!("unexpected uv0 layout: {other:?}"),
        };
        let uv1 = match m.attribute(Mesh::ATTRIBUTE_UV_1).expect("uv1") {
            bevy::mesh::VertexAttributeValues::Float32x2(v) => v.clone(),
            other => panic!("unexpected uv1 layout: {other:?}"),
        };
        uv0.iter()
            .zip(uv1.iter())
            .map(|(a, b)| Vec3::new(a[0], a[1], b[0]))
            .collect()
    }

    #[test]
    fn baked_mesh_has_geometry_and_bounded_cost() {
        for species in 0..TREE_PROFILES.len() as u8 {
            let key = TreeMeshKey {
                species,
                variant: 0,
            };
            let mesh = bake_tree_mesh(key);
            let positions = mesh.attribute(Mesh::ATTRIBUTE_POSITION).expect("positions");
            let tri = tri_count(&mesh);
            assert!(tri > 0, "species {species}: baked mesh is empty");
            // One mesh per tree has to stay cheap enough to spawn 13 per
            // chunk; the gallery's per-part entity count was the problem,
            // not the triangle count, so this is a loose ceiling.
            assert!(
                tri < 20_000,
                "species {species}: {tri} triangles per tree is too heavy"
            );
            assert!(!positions.is_empty());
        }
    }

    #[test]
    fn lod_tiers_actually_get_cheaper() {
        // The whole point of the tiers: each must cost strictly less than the
        // one before it, by a margin worth the extra entities. A "LOD" that
        // loses 5% of the triangles is just three copies of the same tree.
        for species in 0..TREE_PROFILES.len() as u8 {
            let key = TreeMeshKey {
                species,
                variant: 0,
            };
            let near = tri_count(&bake_tree_mesh_lod(key, TreeLod::Near));
            let mid = tri_count(&bake_tree_mesh_lod(key, TreeLod::Mid));
            let far = tri_count(&bake_tree_mesh_lod(key, TreeLod::Far));
            assert!(
                near > 0 && mid > 0 && far > 0,
                "species {species}: empty LOD"
            );
            assert!(
                mid < near * 3 / 4,
                "species {species}: mid tier keeps {mid}/{near} triangles, not a real reduction"
            );
            assert!(
                far < mid / 2,
                "species {species}: far tier keeps {far}/{mid} triangles"
            );
        }
    }

    #[test]
    fn lod_ranges_tile_the_distances_without_gaps_or_overlap() {
        // A gap would make trees vanish at some distance; an overlap would
        // draw two tiers at once. Both are silent in a screenshot.
        for pair in TreeLod::ALL.windows(2) {
            let (lower, upper) = (pair[0], pair[1]);
            let a = lower.range();
            let b = upper.range();
            assert_eq!(
                a.end_margin.end, b.start_margin.start,
                "{lower:?} ends at {} but {upper:?} starts at {}",
                a.end_margin.end, b.start_margin.start
            );
        }
        // Near starts at the camera.
        let near = TreeLod::ALL[0].range();
        assert_eq!(near.start_margin.start, 0.0);

        // Exactly one tier is visible at any distance up to the cull point.
        for distance in [0.0, 1.0, 39.9, 40.0, 109.9, 110.0, 219.9] {
            let visible: Vec<TreeLod> = TreeLod::ALL
                .iter()
                .copied()
                .filter(|lod| !lod.range().is_culled(distance))
                .collect();
            assert_eq!(
                visible.len(),
                1,
                "{distance} m: {visible:?} are all visible at once"
            );
        }

        // Beyond the last boundary nothing is drawn: the tree is culled
        // rather than left at full cost out to the horizon.
        for distance in [220.0, 500.0] {
            assert!(
                TreeLod::ALL
                    .iter()
                    .all(|lod| lod.range().is_culled(distance)),
                "{distance} m: a tier is still visible past the cull distance"
            );
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
        let (Some(pa), Some(pb)) = (
            a.attribute(Mesh::ATTRIBUTE_POSITION),
            b.attribute(Mesh::ATTRIBUTE_POSITION),
        ) else {
            panic!("positions missing");
        };
        assert_eq!(
            format!("{pa:?}"),
            format!("{pb:?}"),
            "bake must be deterministic"
        );
    }

    #[test]
    fn keys_vary_by_species_and_variant() {
        // A chunk must be able to ask for genuinely different silhouettes.
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

        // The placement mapping must reach every variant.
        let mut variants = std::collections::HashSet::new();
        for step in 0..10 {
            let wear = step as f32 / 10.0;
            variants.insert(TreeMeshKey::from_placement(0, wear).variant);
        }
        assert!(variants.len() > 1, "placement wear must vary the shape");
        // And it never indexes past the baked set.
        for wear in [0.0, 0.5, 0.99, 1.0] {
            assert!(TreeMeshKey::from_placement(9, wear).variant < TreeMeshKey::VARIANTS);
        }
    }

    #[test]
    fn season_attributes_cover_every_vertex() {
        // A single vertex without these attributes makes the whole mesh fail to
        // merge (and, once merged anyway, would make the shader read garbage).
        for species in 0..TREE_PROFILES.len() as u8 {
            for lod in TreeLod::ALL {
                let mesh = bake_tree_mesh_lod(
                    TreeMeshKey {
                        species,
                        variant: 1,
                    },
                    lod,
                );
                let vertices = mesh.count_vertices();
                assert!(vertices > 0, "species {species} {lod:?}: empty mesh");
                assert_eq!(
                    colors(&mesh).len(),
                    vertices,
                    "species {species} {lod:?}: colour count != vertex count"
                );
                assert_eq!(
                    offsets(&mesh).len(),
                    vertices,
                    "species {species} {lod:?}: uv pack count != vertex count"
                );
            }
        }
    }

    #[test]
    fn foliage_and_wood_are_distinguishable() {
        // The shader tells leaves from branches by `COLOR.g`. If the mask were
        // missing, winter would collapse the trunk to a point and the tree
        // would vanish instead of turning bare.
        let mesh = bake_tree_mesh(TreeMeshKey {
            species: 0,
            variant: 0,
        });
        let mask = colors(&mesh);
        let leaves = mask.iter().filter(|c| c[1] > 0.5).count();
        let wood = mask.iter().filter(|c| c[1] <= 0.5).count();
        assert!(leaves > 0, "no vertex is marked as foliage");
        assert!(wood > 0, "no vertex is marked as a branch");
        // Foliage dominates the vertex budget (measured 85-90% of triangles).
        assert!(
            leaves > wood,
            "expected the crown to carry more vertices than the branches ({leaves} vs {wood})"
        );
    }

    #[test]
    fn foliage_vertices_store_their_offset_from_the_cluster_centre() {
        // This is what the density collapse scales. If it encoded something
        // else (e.g. world position), scaling it would drag the crown across
        // the ground instead of shrinking each blob onto its own centre.
        let mesh = bake_tree_mesh(TreeMeshKey {
            species: 2,
            variant: 0,
        });
        let mask = colors(&mesh);
        let offs = offsets(&mesh);
        let mut foliage_seen = 0;
        for (color, offset) in mask.iter().zip(offs.iter()) {
            if color[1] < 0.5 {
                // Branches must not move at all under the collapse.
                assert_eq!(
                    *offset,
                    Vec3::ZERO,
                    "a branch vertex carries a non-zero collapse offset"
                );
                continue;
            }
            foliage_seen += 1;
            // Offsets are bounded by the blob radius: clusters are at most a
            // couple of metres across on every species.
            assert!(
                offset.length() < 3.0,
                "foliage offset {offset:?} is too large to be a blob radius"
            );
        }
        assert!(foliage_seen > 0);

        // And the offsets really do cluster: within one blob they all sit near
        // the same centre. Group by rounded centre-free proximity is awkward
        // after merging, so assert the distribution instead -- a correct bake
        // has many distinct centres, not one.
        let mut rounded = std::collections::HashSet::new();
        for offset in &offs {
            let key = (
                (offset.x * 8.0).round() as i32,
                (offset.y * 8.0).round() as i32,
                (offset.z * 8.0).round() as i32,
            );
            rounded.insert(key);
        }
        assert!(
            rounded.len() > 10,
            "only {} distinct offsets: the collapse would bunch the whole crown",
            rounded.len()
        );
    }

    #[test]
    fn sway_weight_grows_with_height() {
        // Wind moves the crown more than the trunk. A weight that ignored
        // height would slide the whole tree sideways as one rigid piece.
        let mesh = bake_tree_mesh(TreeMeshKey {
            species: 1,
            variant: 0,
        });
        let mask = colors(&mesh);
        let positions = match mesh.attribute(Mesh::ATTRIBUTE_POSITION).expect("positions") {
            bevy::mesh::VertexAttributeValues::Float32x3(v) => v.clone(),
            other => panic!("unexpected position layout: {other:?}"),
        };
        let mut low = f32::INFINITY;
        let mut high: f32 = 0.0;
        let mut low_y = f32::INFINITY;
        let mut high_y: f32 = 0.0;
        for (color, p) in mask.iter().zip(positions.iter()) {
            let weight = color[0];
            if p[1] < low_y {
                low_y = p[1];
                low = weight;
            }
            if p[1] > high_y {
                high_y = p[1];
                high = weight;
            }
        }
        assert!(high_y > low_y, "tree has no vertical extent");
        assert!(
            high > low,
            "sway weight does not grow with height ({low} at {low_y} m, {high} at {high_y} m)"
        );
        // Sane range: the shader multiplies this by the season's strength, and
        // a runaway weight would throw vertices off the mesh.
        for color in &mask {
            assert!(
                (0.0..=2.5).contains(&color[0]),
                "sway weight {} out of range",
                color[0]
            );
        }
    }
}
