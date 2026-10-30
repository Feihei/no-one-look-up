//! CDLOD patch selection (Strugar 2009): a quadtree over the terrain's 32 × 32-cell leaves, a
//! doubling range per level, and the min/max heights that bound every node. Pure functions: the
//! pool and the shaders are in `mod.rs` and `terrain_render.wesl`. See `docs/research/terrain.md`
//! ("CDLOD selection").
//!
//! All positions here are terrain-local: x and z from the grid's minimum corner, y relative to
//! the terrain transform.

use bevy::math::{UVec2, Vec2, Vec3};

use super::model::{AABB_PAD_M, LEAF_CELLS};

/// Most quadtree levels: 2⁷ leaves reach 8 km at 64 m leaves, and the leaf grows with the cell.
pub(crate) const MAX_LEVELS: usize = 16;

/// First range as a multiple of the leaf edge: `r_0 = 2 S_0`.
const FIRST_RANGE_LEAVES: f32 = 2.0;
/// Morph starts at this fraction of a range unless the continuity bound asks for later.
pub(crate) const MORPH_START: f32 = 0.7;
/// The morph never covers less than this fraction of a range (Strugar: 15–30 %).
const MIN_MORPH_FRACTION: f32 = 0.15;

/// The geometry of the quadtree and its ranges.
///
/// A node of level `k` is subdivided when its box is within `r_{k−1}` of the camera. A vertex of
/// level `k` morphs onto the level `k+1` grid between `morph_start[k]` and `ranges[k]`. For the
/// morph to meet its neighbours without a crack or a pop, a node that is about to subdivide must
/// be unmorphed everywhere: `morph_start[k] ≥ r_{k−1} + diag_k`, the diagonal of the node's
/// box (`terrain.md` Deviation 12). The ranges are derived from that bound ([`derive`]).
///
/// [`derive`]: LodConfig::derive
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct LodConfig {
    /// World size of the grid, m.
    pub extent: Vec2,
    /// Edge of a leaf patch, `S_0 = 32 c`, m.
    pub leaf_m: f32,
    /// Levels, `K = ⌈log₂(max extent / S_0)⌉ + 1`.
    pub levels: u32,
    ranges: [f32; MAX_LEVELS],
    morph_start: [f32; MAX_LEVELS],
}

impl LodConfig {
    /// The ranges of a flat terrain (only the box padding gives the nodes any height).
    pub fn new(extent: Vec2, cell_m: f32) -> Self {
        let leaf_m = LEAF_CELLS as f32 * cell_m;
        let ratio = (extent.max_element() / leaf_m).max(1.0);
        let levels = (ratio.log2().ceil() as u32 + 1).min(MAX_LEVELS as u32);
        Self {
            extent,
            leaf_m,
            levels,
            ranges: [0.0; MAX_LEVELS],
            morph_start: [0.0; MAX_LEVELS],
        }
        .derive(None)
    }

    /// The ranges for the node heights of `tree` (a flat terrain without one).
    pub fn derive(mut self, tree: Option<&BoundsTree>) -> Self {
        let pad = 2.0 * AABB_PAD_M;
        for k in 0..self.levels as usize {
            let s = self.node_m(k as u32);
            if k == 0 {
                self.ranges[0] = FIRST_RANGE_LEAVES * s;
                self.morph_start[0] = MORPH_START * self.ranges[0];
                continue;
            }
            let h = tree.map_or(0.0, |t| t.height_range(k as u32)) + pad;
            let start = self.ranges[k - 1] + (2.0 * s * s + h * h).sqrt();
            self.ranges[k] = (2.0 * self.ranges[k - 1]).max(start / (1.0 - MIN_MORPH_FRACTION));
            self.morph_start[k] = start.max(MORPH_START * self.ranges[k]);
        }
        self
    }

    /// Edge of a level's node, `S_k = S_0 2^k`, m.
    pub fn node_m(&self, level: u32) -> f32 {
        self.leaf_m * (1u32 << level) as f32
    }

    /// Range of a level, `r_k`, m: where its morph completes.
    pub fn range_m(&self, level: u32) -> f32 {
        self.ranges[level as usize]
    }

    /// Distance where a level's morph begins, m.
    pub fn morph_start_m(&self, level: u32) -> f32 {
        self.morph_start[level as usize]
    }

    /// Nodes of a level that overlap the grid, per axis.
    pub fn nodes(&self, level: u32) -> UVec2 {
        (self.extent / self.node_m(level))
            .ceil()
            .as_uvec2()
            .max(UVec2::ONE)
    }

    /// Offset of every level's vertex lattice from the grid's corner on each axis, m: half a
    /// cell, so the vertices sit on the height samples, the cell centres
    /// (`docs/research/orthographic-lod.md` Model 3).
    pub fn sample_offset(&self) -> f32 {
        0.5 * self.leaf_m / LEAF_CELLS as f32
    }

    /// The last sample on each axis, terrain-local m: a vertex past it is clamped onto it.
    pub fn last_sample(&self) -> Vec2 {
        self.extent - self.sample_offset()
    }
}

/// One selected patch: a node of the quadtree.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) struct Patch {
    pub level: u32,
    pub ix: u32,
    pub iz: u32,
}

/// Min and max surface height of every node of every level: level 0 is the leaves.
#[derive(Clone, Debug)]
pub(crate) struct BoundsTree {
    levels: Vec<(UVec2, Vec<(f32, f32)>)>,
}

impl BoundsTree {
    /// One bound for the whole terrain (before the first readback).
    pub fn uniform(config: &LodConfig, min: f32, max: f32) -> Self {
        Self {
            levels: (0..config.levels)
                .map(|k| {
                    let dims = config.nodes(k);
                    (dims, vec![(min, max); (dims.x * dims.y) as usize])
                })
                .collect(),
        }
    }

    /// The tree over the readback's per-leaf `(min, max)`, built by max/min upward.
    pub fn from_leaves(
        config: &LodConfig,
        dims: UVec2,
        leaves: impl Iterator<Item = (f32, f32)>,
    ) -> Self {
        let mut levels = vec![(dims, leaves.collect::<Vec<_>>())];
        for k in 1..config.levels {
            let (pd, below) = &levels[(k - 1) as usize];
            let d = config.nodes(k);
            let mut level = vec![(f32::INFINITY, f32::NEG_INFINITY); (d.x * d.y) as usize];
            for z in 0..pd.y {
                for x in 0..pd.x {
                    let (lo, hi) = below[(z * pd.x + x) as usize];
                    let slot = &mut level[((z / 2) * d.x + x / 2) as usize];
                    slot.0 = slot.0.min(lo);
                    slot.1 = slot.1.max(hi);
                }
            }
            levels.push((d, level));
        }
        Self { levels }
    }

    /// The tallest unpadded node of a level, m.
    pub fn height_range(&self, level: u32) -> f32 {
        self.levels[level as usize]
            .1
            .iter()
            .map(|(lo, hi)| hi - lo)
            .fold(0.0, f32::max)
    }

    /// The unpadded `(min, max)` of a node.
    pub fn get(&self, level: u32, ix: u32, iz: u32) -> (f32, f32) {
        let (dims, bounds) = &self.levels[level as usize];
        bounds[(iz.min(dims.y - 1) * dims.x + ix.min(dims.x - 1)) as usize]
    }

    /// The lowest and highest surface of the whole terrain, m (unpadded).
    pub fn extremes(&self) -> (f32, f32) {
        self.levels.last().map_or((0.0, 0.0), |(_, roots)| {
            roots
                .iter()
                .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), &(a, b)| {
                    (lo.min(a), hi.max(b))
                })
        })
    }
}

/// Distance from `p` to the box `[lo, hi]`.
fn box_distance(lo: Vec3, hi: Vec3, p: Vec3) -> f32 {
    (lo - p).max(p - hi).max(Vec3::ZERO).length()
}

/// The 3D box of a node, with the height padding, in terrain-local space: the node's tile on
/// the vertex lattice (half a cell in from the grid's corner), clipped to the last sample.
pub(crate) fn node_box(config: &LodConfig, tree: &BoundsTree, node: Patch) -> (Vec3, Vec3) {
    let s = config.node_m(node.level);
    let lo = Vec2::new(node.ix as f32, node.iz as f32) * s + config.sample_offset();
    let hi = (lo + s).min(config.last_sample());
    let (y0, y1) = tree.get(node.level, node.ix, node.iz);
    (
        Vec3::new(lo.x, y0 - AABB_PAD_M, lo.y),
        Vec3::new(hi.x, y1 + AABB_PAD_M, hi.y),
    )
}

/// The patches that draw the terrain for a camera at `camera` (terrain-local).
///
/// A node is drawn whole when it is a leaf or lies entirely outside the sphere of the next finer
/// range; otherwise its four children are visited (Strugar's `LODSelect`). A node that sticks out
/// of the grid is always subdivided, so the patches tile the grid exactly. The coarsest level has
/// no range of its own and is always considered.
pub(crate) fn select_patches(config: &LodConfig, tree: &BoundsTree, camera: Vec3) -> Vec<Patch> {
    let top = config.levels - 1;
    let roots = config.nodes(top);
    let mut out = Vec::new();
    for iz in 0..roots.y {
        for ix in 0..roots.x {
            visit(config, tree, camera, Patch { level: top, ix, iz }, &mut out);
        }
    }
    out
}

fn visit(config: &LodConfig, tree: &BoundsTree, camera: Vec3, node: Patch, out: &mut Vec<Patch>) {
    let (lo, hi) = node_box(config, tree, node);
    if node.level > 0 {
        let sticks_out = {
            let s = config.node_m(node.level);
            (Vec2::new(node.ix as f32, node.iz as f32) + 1.0) * s - config.extent
        }
        .max_element()
            > 1e-3;
        let near = box_distance(lo, hi, camera) <= config.range_m(node.level - 1);
        if near || sticks_out {
            for dz in 0..2 {
                for dx in 0..2 {
                    let child = Patch {
                        level: node.level - 1,
                        ix: node.ix * 2 + dx,
                        iz: node.iz * 2 + dz,
                    };
                    // Children that start beyond the grid do not exist.
                    if (Vec2::new(child.ix as f32, child.iz as f32) * config.node_m(child.level))
                        .cmplt(config.extent)
                        .all()
                    {
                        visit(config, tree, camera, child, out);
                    }
                }
            }
            return;
        }
    }
    out.push(node);
}

#[cfg(test)]
/// `morphK` of the vertex stage: 0 up to `0.7 r_k`, 1 from `r_k`, linear between; the coarsest
/// level never morphs.
pub(crate) fn morph_k(config: &LodConfig, level: u32, distance: f32) -> f32 {
    if level + 1 >= config.levels {
        return 0.0;
    }
    let (start, end) = (config.morph_start_m(level), config.range_m(level));
    ((distance - start) / (end - start)).clamp(0.0, 1.0)
}

#[cfg(test)]
/// The vertex stage of `render.wesl` for grid vertex `gi` (0–32 on each axis) of a patch,
/// as a function of `height`, the terrain height at a local XZ, morphed about `centre`: the
/// morphed local position.
pub(crate) fn vertex_position(
    config: &LodConfig,
    patch: Patch,
    gi: UVec2,
    centre: Vec3,
    height: impl Fn(Vec2) -> f32,
) -> Vec3 {
    let step = config.node_m(patch.level) / LEAF_CELLS as f32;
    let origin = Vec2::new(patch.ix as f32, patch.iz as f32) * config.node_m(patch.level)
        + config.sample_offset();
    let unmorphed = origin + gi.as_vec2() * step;
    let d = Vec3::new(unmorphed.x, height(unmorphed), unmorphed.y).distance(centre);
    let k = morph_k(config, patch.level, d);
    let odd = Vec2::new((gi.x % 2) as f32, (gi.y % 2) as f32);
    let morphed = (unmorphed - odd * step * k).min(config.last_sample());
    Vec3::new(morphed.x, height(morphed), morphed.y)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::terrain::TerrainGrid;
    use bevy::math::{IVec2, Vec3Swizzles};

    /// Deterministic pseudo-random numbers in `[0, 1)`.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> f32 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (self.0 >> 40) as f32 / 16_777_216.0
        }
    }

    fn high() -> LodConfig {
        LodConfig::new(Vec2::splat(2048.0), 2.0)
    }

    fn flat(config: &LodConfig) -> BoundsTree {
        BoundsTree::uniform(config, 0.0, 0.0)
    }

    fn counts(patches: &[Patch], levels: u32) -> Vec<usize> {
        (0..levels)
            .map(|k| patches.iter().filter(|p| p.level == k).count())
            .collect()
    }

    /// A tree with a random min/max per leaf, up to `tall` m of height range.
    fn random_tree(config: &LodConfig, rng: &mut Rng, tall: f32) -> BoundsTree {
        let dims = config.nodes(0);
        BoundsTree::from_leaves(
            config,
            dims,
            (0..dims.x * dims.y).map(|_| {
                let lo = rng.next() * tall;
                (lo, lo + rng.next() * tall * 0.5)
            }),
        )
    }

    #[test]
    fn the_levels_and_ranges_follow_the_quality_levels() {
        // (cell m, levels, r_0 m): High, Ultra, Medium, Low at a 2048 m extent.
        for (cell, levels, r0) in [
            (2.0, 6, 128.0),
            (1.0, 7, 64.0),
            (4.0, 5, 256.0),
            (8.0, 4, 512.0),
        ] {
            let c = LodConfig::new(Vec2::splat(2048.0), cell);
            assert_eq!(c.levels, levels, "cell {cell}");
            assert_eq!(c.range_m(0), r0);
            assert_eq!(c.nodes(c.levels - 1), UVec2::ONE);
            for k in 1..levels {
                // At least doubling, and the morph starts after the node is no longer
                // subdividable and ends at the range.
                assert!(c.range_m(k) >= 2.0 * c.range_m(k - 1));
                assert!(c.morph_start_m(k) >= c.range_m(k - 1) + c.node_m(k));
                assert!(c.morph_start_m(k) <= 0.85 * c.range_m(k) + 1e-3);
            }
        }
    }

    /// Validation 3 (d) and (e): the patch counts of a flat terrain, for a camera over a corner
    /// of the vertex lattice's central nodes (half a cell off the grid's centre).
    #[test]
    fn a_flat_terrain_selects_the_measured_patch_counts() {
        let at = |cell: f32, y: f32| {
            let c = LodConfig::new(Vec2::splat(2048.0), cell);
            let centre = 1024.0 + c.sample_offset();
            let p = select_patches(&c, &flat(&c), Vec3::new(centre, y, centre));
            (counts(&p, c.levels), p.len())
        };
        assert_eq!(at(2.0, 30.0), (vec![16, 60, 48, 0, 0, 0], 124));
        assert_eq!(at(1.0, 30.0), (vec![16, 44, 52, 48, 0, 0, 0], 160));
        assert_eq!(at(4.0, 30.0), (vec![16, 60, 0, 0, 0], 76));
        assert_eq!(at(8.0, 30.0), (vec![16, 12, 0, 0], 28));
        // 300 m up there is no level 0.
        assert_eq!(at(2.0, 300.0), (vec![0, 16, 60, 0, 0, 0], 76));
        // 1500 m up: sixteen level-3 patches.
        assert_eq!(at(2.0, 1500.0), (vec![0, 0, 0, 16, 0, 0], 16));
    }

    /// Validation 3 (a), (b), (c): the patches tile the grid exactly once, each is at least
    /// `r_{k−1}` from the camera, and neighbours differ by at most one level.
    #[test]
    fn the_selection_tiles_the_grid_and_neighbours_differ_by_one_level() {
        let config = high();
        let mut rng = Rng(7);
        let mut worst_gap = 0;
        for _ in 0..1000 {
            let tree = random_tree(&config, &mut rng, 200.0);
            let config = config.derive(Some(&tree));
            let cam = Vec3::new(
                rng.next() * 2048.0,
                2.0 + rng.next() * 600.0,
                rng.next() * 2048.0,
            );
            let patches = select_patches(&config, &tree, cam);
            let area: f32 = patches.iter().map(|p| config.node_m(p.level).powi(2)).sum();
            assert_eq!(area, 2048.0 * 2048.0, "camera {cam}");
            for p in &patches {
                if p.level > 0 {
                    let (lo, hi) = node_box(&config, &tree, *p);
                    let d = box_distance(lo, hi, cam);
                    assert!(d > config.range_m(p.level - 1), "{p:?} at {d} from {cam}");
                }
            }
            // Adjacency: two patches share an edge when their tiles touch along a segment.
            for (i, a) in patches.iter().enumerate() {
                for b in &patches[i + 1..] {
                    let (sa, sb) = (config.node_m(a.level), config.node_m(b.level));
                    let (pa, pb) = (
                        Vec2::new(a.ix as f32, a.iz as f32) * sa,
                        Vec2::new(b.ix as f32, b.iz as f32) * sb,
                    );
                    let overlap_x = (pa.x + sa).min(pb.x + sb) - pa.x.max(pb.x);
                    let overlap_z = (pa.y + sa).min(pb.y + sb) - pa.y.max(pb.y);
                    let touching = (overlap_x == 0.0 && overlap_z > 0.0)
                        || (overlap_z == 0.0 && overlap_x > 0.0);
                    if touching {
                        worst_gap = worst_gap.max(a.level.abs_diff(b.level));
                    }
                }
            }
        }
        assert_eq!(worst_gap, 1, "adjacent patches two levels apart");
    }

    /// A generated terrain over 2048 m: its quadtree, the leaf bounds the readback would give
    /// (each leaf's cells and a one-cell halo, as `terrain_minmax` reduces them) and the grid's
    /// surface.
    fn generated(
        preset: &super::super::ProceduralTerrain,
        cell: f32,
    ) -> (LodConfig, BoundsTree, TerrainGrid, Vec<f32>) {
        use crate::terrain::model::mirror::generate_surface;
        let size = UVec2::splat((2048.0 / cell) as u32);
        let surface = generate_surface(preset, size, cell);
        let grid = TerrainGrid {
            origin: Vec2::ZERO,
            cell_m: cell,
            size,
            y0: 0.0,
        };
        let leaves = size / LEAF_CELLS;
        let mut bounds = Vec::new();
        for lz in 0..leaves.y as i32 {
            for lx in 0..leaves.x as i32 {
                let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
                let edge = LEAF_CELLS as i32;
                for z in -1..=edge {
                    for x in -1..=edge {
                        let h = surface[grid.index(IVec2::new(lx * edge + x, lz * edge + z))];
                        lo = lo.min(h);
                        hi = hi.max(h);
                    }
                }
                bounds.push((lo, hi));
            }
        }
        let config = LodConfig::new(Vec2::splat(2048.0), cell);
        let tree = BoundsTree::from_leaves(&config, leaves, bounds.into_iter());
        (config.derive(Some(&tree)), tree, grid, surface)
    }

    /// Validation 3 (f): the pool capacity is never exceeded, on flat ground and on the
    /// generated hills and mountains (whose steep nodes lengthen the ranges).
    #[test]
    fn the_pool_capacity_holds_for_random_cameras() {
        use crate::terrain::ProceduralTerrain;
        let mut rng = Rng(99);
        let mut worst = 0;
        let mut cases = vec![];
        for cell in [1.0, 2.0, 4.0, 8.0] {
            if cell == 1.0 {
                // Ultra: flat only (generating 2048² on the CPU takes too long).
                let flat =
                    BoundsTree::uniform(&LodConfig::new(Vec2::splat(2048.0), cell), 0.0, 0.0);
                cases.push((LodConfig::new(Vec2::splat(2048.0), cell), flat));
                continue;
            }
            for preset in [
                ProceduralTerrain::hills(42),
                ProceduralTerrain::mountains(42),
            ] {
                let (config, tree, _, _) = generated(&preset, cell);
                cases.push((config, tree));
            }
            let flat = BoundsTree::uniform(&LodConfig::new(Vec2::splat(2048.0), cell), 0.0, 0.0);
            cases.push((LodConfig::new(Vec2::splat(2048.0), cell), flat));
        }
        for (config, tree) in &cases {
            let mut most = 0;
            for _ in 0..1500 {
                let cam = Vec3::new(
                    rng.next() * 2048.0,
                    2.0 + rng.next() * 300.0,
                    rng.next() * 2048.0,
                );
                most = most.max(select_patches(config, tree, cam).len());
            }
            println!("leaf {} m: most {most} patches", config.leaf_m);
            worst = worst.max(most);
        }
        assert!(worst <= super::super::POOL_CAPACITY, "worst {worst}");
    }

    /// A grid whose leaf count is not a power of two still tiles exactly.
    #[test]
    fn a_non_power_of_two_grid_tiles_exactly() {
        let config = LodConfig::new(Vec2::new(2048.0, 1536.0), 2.0);
        let tree = flat(&config);
        let mut rng = Rng(5);
        for _ in 0..200 {
            let cam = Vec3::new(
                rng.next() * 2048.0,
                5.0 + rng.next() * 500.0,
                rng.next() * 1536.0,
            );
            let area: f32 = select_patches(&config, &tree, cam)
                .iter()
                .map(|p| config.node_m(p.level).powi(2))
                .sum();
            assert_eq!(area, 2048.0 * 1536.0, "camera {cam}");
        }
    }

    #[test]
    fn morph_is_zero_before_its_start_and_one_at_the_range() {
        let c = high();
        for k in 0..c.levels - 1 {
            let (start, r) = (c.morph_start_m(k), c.range_m(k));
            assert_eq!(morph_k(&c, k, 0.0), 0.0);
            assert_eq!(morph_k(&c, k, start), 0.0);
            assert!((morph_k(&c, k, 0.5 * (start + r)) - 0.5).abs() < 1e-6);
            assert_eq!(morph_k(&c, k, r), 1.0);
            assert_eq!(morph_k(&c, k, 10.0 * r), 1.0);
        }
        // The coarsest level has nothing to morph to.
        assert_eq!(morph_k(&c, c.levels - 1, 1e6), 0.0);
    }

    /// A smooth stand-in for the terrain height.
    fn hill(p: Vec2) -> f32 {
        30.0 * (p.x * 0.01).sin() * (p.y * 0.013).cos() + 0.02 * p.x
    }

    /// The selection about `cam` has no crack: the vertices of two adjacent patches on their
    /// shared vertical edge are bit-identical (the horizontal edge is the same with the axes
    /// swapped).
    fn assert_no_cracks(
        config: &LodConfig,
        tree: &BoundsTree,
        cam: Vec3,
        height: &impl Fn(Vec2) -> f32,
    ) {
        let patches = select_patches(config, tree, cam);
        for (i, a) in patches.iter().enumerate() {
            for b in &patches[i + 1..] {
                let (sa, sb) = (config.node_m(a.level), config.node_m(b.level));
                let (pa, pb) = (
                    Vec2::new(a.ix as f32, a.iz as f32) * sa,
                    Vec2::new(b.ix as f32, b.iz as f32) * sb,
                );
                if pa.x + sa != pb.x {
                    continue;
                }
                // The coarser patch's vertices on the shared edge are the finer one's even
                // vertices; compare them where both have one.
                let z0 = pa.y.max(pb.y);
                let z1 = (pa.y + sa).min(pb.y + sb);
                if z1 <= z0 {
                    continue;
                }
                let (stepa, stepb) = (sa / 32.0, sb / 32.0);
                let mut z = z0;
                while z <= z1 {
                    // Vertex index of this z in each patch; both exist when z is a multiple of
                    // the coarser step.
                    let coarse_step = stepa.max(stepb);
                    if (z - z0) % coarse_step == 0.0 {
                        let ga = UVec2::new(32, ((z - pa.y) / stepa) as u32);
                        let gb = UVec2::new(0, ((z - pb.y) / stepb) as u32);
                        let va = vertex_position(config, *a, ga, cam, height);
                        let vb = vertex_position(config, *b, gb, cam, height);
                        assert_eq!(va, vb, "{a:?} {b:?} at z {z}, camera {cam}");
                    }
                    z += stepa.min(stepb);
                }
            }
        }
    }

    /// Validation 4: shared-edge vertices of adjacent patches are bit-identical, and a patch
    /// whose morph is 1 everywhere has its odd vertices on the level-k+1 grid.
    #[test]
    fn adjacent_patches_meet_without_cracks() {
        let tree = BoundsTree::uniform(&high(), -50.0, 80.0);
        let config = high().derive(Some(&tree));
        let mut rng = Rng(11);
        for _ in 0..40 {
            let cam = Vec3::new(
                rng.next() * 2048.0,
                5.0 + rng.next() * 150.0,
                rng.next() * 2048.0,
            );
            assert_no_cracks(&config, &tree, cam, &hill);
        }
        // A fully morphed patch sits on the next level's grid (error < 1e-4 m).
        let patch = Patch {
            level: 0,
            ix: 3,
            iz: 4,
        };
        let far = Vec3::new(3000.0, 0.0, 3000.0);
        for gz in 0..=32 {
            for gx in 0..=32 {
                let v = vertex_position(&config, patch, UVec2::new(gx, gz), far, hill);
                let coarse = config.node_m(1) / 32.0;
                let origin = Vec2::new(3.0, 4.0) * config.node_m(0) + config.sample_offset();
                let on_grid =
                    |p: f32, o: f32| (((p - o) / coarse).round() * coarse - (p - o)).abs();
                assert!(
                    on_grid(v.x, origin.x) < 1e-4 && on_grid(v.z, origin.y) < 1e-4,
                    "{gx},{gz}: {v}"
                );
            }
        }
    }

    /// The unmorphed position of grid vertex `gi` of a patch, on the vertex lattice and clamped
    /// to the last sample.
    fn grid_position(
        config: &LodConfig,
        patch: Patch,
        gi: UVec2,
        height: impl Fn(Vec2) -> f32,
    ) -> Vec3 {
        let step = config.node_m(patch.level) / LEAF_CELLS as f32;
        let p = (Vec2::new(patch.ix as f32, patch.iz as f32) * config.node_m(patch.level)
            + config.sample_offset()
            + gi.as_vec2() * step)
            .min(config.last_sample());
        Vec3::new(p.x, height(p), p.y)
    }

    /// No pop for a centre moving from `a` to `b`: a patch either selection drops for its four
    /// children is unmorphed everywhere (it looks like its own grid) and the children's odd
    /// vertices are fully morphed (they look like the patch's grid), up to the morph the move
    /// itself buys, so the surface does not jump at the swap. Returns the swaps exercised.
    fn swaps_barely_move(
        config: &LodConfig,
        tree: &BoundsTree,
        a: Vec3,
        b: Vec3,
        height: &impl Fn(Vec2) -> f32,
    ) -> usize {
        let mut swaps = 0;
        let shift = a.distance(b);
        let (sel_a, sel_b) = (
            select_patches(config, tree, a),
            select_patches(config, tree, b),
        );
        for (before, after, cam_before, cam_after) in
            [(&sel_a, &sel_b, a, b), (&sel_b, &sel_a, b, a)]
        {
            for parent in before.iter().filter(|p| p.level > 0 && !after.contains(p)) {
                let children: Vec<&Patch> = after
                    .iter()
                    .filter(|c| {
                        c.level + 1 == parent.level
                            && c.ix / 2 == parent.ix
                            && c.iz / 2 == parent.iz
                    })
                    .collect();
                if children.len() != 4 {
                    continue;
                }
                swaps += 1;
                // The centre moved `shift` m since the swap boundary and the morph is
                // continuous, so the jump is at most the morph that distance buys.
                let slack = |level: u32| {
                    let step = config.node_m(level) / LEAF_CELLS as f32;
                    2.0 * step * shift / (config.range_m(level) - config.morph_start_m(level))
                        + 1e-3
                };
                for gz in 0..=32 {
                    for gx in 0..=32 {
                        let gi = UVec2::new(gx, gz);
                        let moved = vertex_position(config, *parent, gi, cam_before, height);
                        let want = grid_position(config, *parent, gi, height);
                        assert!(
                            (moved - want).length() <= slack(parent.level),
                            "{parent:?} is morphed at {gi}: {moved} vs {want}"
                        );
                    }
                }
                for child in children {
                    for gz in 0..=32 {
                        for gx in 0..=32 {
                            let gi = UVec2::new(gx, gz);
                            let moved = vertex_position(config, *child, gi, cam_after, height);
                            let snapped = UVec2::new(gx & !1, gz & !1);
                            let want = grid_position(config, *child, snapped, height);
                            assert!(
                                (moved - want).length() <= slack(child.level),
                                "{child:?} vertex {gi}: {moved} vs {want} (shift {shift})"
                            );
                        }
                    }
                }
            }
        }
        swaps
    }

    /// No pop: a camera move that swaps a patch for its four children barely moves the surface.
    #[test]
    fn a_patch_swapped_for_its_children_barely_moves() {
        let tree = BoundsTree::uniform(&high(), -50.0, 80.0);
        let config = high().derive(Some(&tree));
        let mut rng = Rng(21);
        let mut swaps = 0;
        for _ in 0..400 {
            let a = Vec3::new(
                rng.next() * 2048.0,
                5.0 + rng.next() * 200.0,
                rng.next() * 2048.0,
            );
            let b = a + Vec3::new(rng.next() - 0.5, rng.next() - 0.5, rng.next() - 0.5) * 6.0;
            swaps += swaps_barely_move(&config, &tree, a, b, &hill);
        }
        assert!(swaps > 20, "only {swaps} swaps were exercised");
    }

    /// The view direction of an isometric orthographic camera like Sanctuary's (issue #36):
    /// pitched 35.26° down, heading 45°.
    fn isometric_forward() -> Vec3 {
        let (pitch, heading) = (35.26f32.to_radians(), 45f32.to_radians());
        Vec3::new(
            heading.sin() * pitch.cos(),
            -pitch.sin(),
            -heading.cos() * pitch.cos(),
        )
    }

    /// Issue #36, acceptance 1 (`orthographic-lod.md` Validation 1). About an orthographic
    /// camera's focus on the ground, every patch within 30 m of it is level 0, its vertices there
    /// are unmorphed and sit on the height samples, and between them the patch mesh's triangles
    /// are within a quarter of their quad's twist of the bilinear; about the eye 457 m back (the
    /// old centre) some of that ground is drawn coarser.
    #[test]
    fn an_orthographic_focus_is_drawn_at_level_0_on_the_samples() {
        use crate::terrain::ProceduralTerrain;
        let (config, tree, grid, surface) = generated(&ProceduralTerrain::hills(42), 4.0);
        let height = |p: Vec2| grid.sample(&surface, p);
        let cell = grid.cell_m;
        // A vertex on a sample, at the sample's own height.
        let on_samples = |v: Vec3| {
            let index = (v.xz() - 0.5 * cell) / cell;
            (index - index.round()).abs().max_element() < 1e-3
                && (v.y - surface[grid.index(index.round().as_ivec2())]).abs() < 1e-3
        };
        let mut rng = Rng(36);
        let mut quads = 0;
        for _ in 0..12 {
            let xz = Vec2::new(200.0 + rng.next() * 1648.0, 200.0 + rng.next() * 1648.0);
            let focus = Vec3::new(xz.x, height(xz), xz.y);
            let eye = focus - isometric_forward() * 457.0;
            let near = |p: &Patch| {
                let (lo, hi) = node_box(&config, &tree, *p);
                box_distance(lo, hi, focus) <= 30.0
            };
            // The vertices within 30 m of the focus that do not draw the map at its own
            // resolution, for patches selected and morphed about `centre`: on a coarser level,
            // morphed toward one, or off the samples.
            let not_the_map = |centre: Vec3| {
                select_patches(&config, &tree, centre)
                    .iter()
                    .filter(|p| near(p))
                    .flat_map(|p| {
                        (0..=32u32).flat_map(move |gz| (0..=32u32).map(move |gx| (*p, gx, gz)))
                    })
                    .filter(|&(p, gx, gz)| {
                        let gi = UVec2::new(gx, gz);
                        let v = vertex_position(&config, p, gi, centre, height);
                        let unmorphed = grid_position(&config, p, gi, height);
                        v.distance(focus) <= 30.0
                            && (p.level > 0 || v.distance(unmorphed) > 1e-4 || !on_samples(v))
                    })
                    .count()
            };
            assert!(
                not_the_map(eye) > 0,
                "about the eye the ground at {focus} is the map"
            );
            assert_eq!(not_the_map(focus), 0, "about the focus {focus}");
            let patches: Vec<Patch> = select_patches(&config, &tree, focus)
                .into_iter()
                .filter(near)
                .collect();
            assert!(!patches.is_empty());
            for patch in &patches {
                assert_eq!(patch.level, 0, "{patch:?} near the focus {focus}");
                for gz in 0..32 {
                    for gx in 0..32 {
                        let v = |dx: u32, dz: u32| {
                            vertex_position(
                                &config,
                                *patch,
                                UVec2::new(gx + dx, gz + dz),
                                focus,
                                height,
                            )
                        };
                        let (v00, v10, v01, v11) = (v(0, 0), v(1, 0), v(0, 1), v(1, 1));
                        if v00.distance(focus) > 30.0 || v10.x == v00.x || v01.z == v00.z {
                            continue;
                        }
                        quads += 1;
                        // Between the vertices, the triangles (split along (1, 0)–(0, 1)).
                        let twist = (v00.y + v11.y - v10.y - v01.y).abs() * 0.25;
                        for j in 0..=4 {
                            for i in 0..=4 {
                                let (u, w) = (i as f32 / 4.0, j as f32 / 4.0);
                                let drawn = if u + w <= 1.0 {
                                    v00.y + u * (v10.y - v00.y) + w * (v01.y - v00.y)
                                } else {
                                    v11.y
                                        + (1.0 - u) * (v01.y - v11.y)
                                        + (1.0 - w) * (v10.y - v11.y)
                                };
                                let p = v00.xz() + Vec2::new(u, w) * cell;
                                assert!(
                                    (drawn - height(p)).abs() <= twist + 1e-4,
                                    "at {p}: drawn {drawn}, bilinear {}, twist/4 {twist}",
                                    height(p)
                                );
                            }
                        }
                    }
                }
            }
        }
        assert!(quads > 1000, "only {quads} quads within 30 m");
    }

    /// Issue #36, acceptance 3 (`orthographic-lod.md` Validation 3): walking an orthographic
    /// camera's focus 100 m over the hills in 1 m steps, the patches it swaps barely move, and
    /// no edge between patches cracks.
    #[test]
    fn walking_the_focus_neither_pops_nor_cracks() {
        use crate::terrain::ProceduralTerrain;
        let (config, tree, grid, surface) = generated(&ProceduralTerrain::hills(42), 4.0);
        let height = |p: Vec2| grid.sample(&surface, p);
        let direction = isometric_forward().xz().normalize();
        let mut swaps = 0;
        // Twelve walks across the terrain.
        let starts = (0..12).map(|i| {
            let (row, col) = ((i / 4) as f32, (i % 4) as f32);
            Vec2::new(250.0 + col * 450.0, 450.0 + row * 550.0)
        });
        for start in starts {
            let at = |s: f32| {
                let xz = start + direction * s;
                Vec3::new(xz.x, height(xz), xz.y)
            };
            for step in 0..100 {
                swaps += swaps_barely_move(
                    &config,
                    &tree,
                    at(step as f32),
                    at(step as f32 + 1.0),
                    &height,
                );
                if step % 25 == 0 {
                    assert_no_cracks(&config, &tree, at(step as f32), &height);
                }
            }
        }
        assert!(swaps > 10, "only {swaps} swaps were exercised");
    }
}
