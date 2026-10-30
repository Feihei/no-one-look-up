//! The terrain's CPU side that is not a system: the grid geometry, the height snapshot gameplay
//! queries sample, and the generator's constants. The generator and layer-blend mirrors that the
//! tests compare the kernels against are in `mirror`.

use std::sync::Arc;

use bevy::math::{UVec2, Vec2, Vec3, Vec3Swizzles};

/// Cells per leaf patch edge (`terrain_minmax` reduces one leaf per workgroup).
pub(crate) const LEAF_CELLS: u32 = 32;

/// Lacunarity of the generator's octaves.
pub(crate) const LACUNARITY: f32 = 2.0;
/// Gain of the generator's octaves (Musgrave `ω = λ^{-H}`, H = 1).
pub(crate) const GAIN: f32 = 0.5;
/// Octave rotation: the golden angle, rad (137.5°).
pub(crate) const OCTAVE_ROTATION: f32 = 2.399_96;
/// Half range of the fBm sum, mapped to `[0, 1]` as `0.5 + f / (2 · F_HALF)`. Measured on this
/// crate's noise over 64 seeds (`terrain.md` Deviation 11), not the note's numpy port.
pub(crate) const F_HALF: f32 = 0.33;
/// Low end and span of the ridged sum, mapped to `[0, 1]` as `(r − R_LOW) / R_SPAN`.
pub(crate) const R_LOW: f32 = 0.21;
pub(crate) const R_SPAN: f32 = 0.71;
/// Leaf patches' bounding boxes are padded by this many metres.
pub(crate) const AABB_PAD_M: f32 = 2.0;

/// Placement of the terrain's grid in the metre frame (world ÷ `ElementsUnits::units_per_metre`):
/// row-major `size.x × size.y` cells, cell `(0, 0)` centred at `origin + 0.5 · cell_m`, the row
/// index running along +Z.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TerrainGrid {
    /// XZ of the minimum corner in the metre frame, m.
    pub origin: Vec2,
    /// Cell edge, m (cells are square).
    pub cell_m: f32,
    /// Cells per axis.
    pub size: UVec2,
    /// Height of a stored height of 0 in the metre frame, m (the terrain transform's `y` ÷ units).
    pub y0: f32,
}

impl TerrainGrid {
    /// World size of the grid, m.
    pub fn extent(&self) -> Vec2 {
        self.size.as_vec2() * self.cell_m
    }

    /// Whether `p` (metre-frame XZ) lies inside the grid.
    pub fn contains(&self, p: Vec2) -> bool {
        let local = p - self.origin;
        local.cmpge(Vec2::ZERO).all() && local.cmple(self.extent()).all()
    }

    /// Row-major index of a cell coordinate clamped into the grid.
    pub fn index(&self, c: bevy::math::IVec2) -> usize {
        let m = self.size.as_ivec2() - 1;
        let k = c.clamp(bevy::math::IVec2::ZERO, m).as_uvec2();
        (k.y * self.size.x + k.x) as usize
    }

    /// Bilinear sample of a row-major layer at world XZ `p`, with the same tap rule as
    /// `grid_taps` in `grid.wesl` (cell centres at +0.5; taps clamped at the edge).
    pub fn sample(&self, layer: &[f32], p: Vec2) -> f32 {
        let g = (p - self.origin) / self.cell_m - 0.5;
        let base = g.floor();
        let w = g - base;
        let b = base.as_ivec2();
        let at = |dx: i32, dz: i32| layer[self.index(b + bevy::math::IVec2::new(dx, dz))];
        let row0 = at(0, 0) * (1.0 - w.x) + at(1, 0) * w.x;
        let row1 = at(0, 1) * (1.0 - w.x) + at(1, 1) * w.x;
        row0 * (1.0 - w.y) + row1 * w.y
    }
}

/// The grid the quality level and the requested extent and resolution give, and whether the
/// constructor had to adjust the request.
///
/// `resolution` is clamped per axis to `max` and rounded down to a power of two (at least one
/// leaf, 32 cells). Cells must be square: the cell `c` is the larger of `extent.x / res.x` and
/// `extent.y / res.y` (the coarser axis binds), and if they differ both axes are re-derived as
/// `round(extent / c / 32) · 32`, which keeps each within its clamped resolution.
pub(crate) fn resolve_resolution(extent: Vec2, resolution: UVec2, max: u32) -> (UVec2, f32, bool) {
    let snap = |r: u32| {
        let r = r.clamp(LEAF_CELLS, max.max(LEAF_CELLS));
        1 << r.ilog2()
    };
    let res = UVec2::new(snap(resolution.x), snap(resolution.y));
    let cells = extent / res.as_vec2();
    let cell = cells.max_element();
    let adjusted = (cells - cell).abs().max_element() > cell * 1e-4;
    if !adjusted {
        return (res, cell, false);
    }
    let derive = |e: f32| ((e / cell / LEAF_CELLS as f32).round().max(1.0) as u32) * LEAF_CELLS;
    (UVec2::new(derive(extent.x), derive(extent.y)), cell, true)
}

/// Octaves the generator uses: the request, clamped so the finest octave's wavelength stays
/// above two cells (`⌊log₂(F / 2c)⌋ + 1`), and at least one.
pub(crate) fn octave_count(requested: u32, feature_m: f32, cell_m: f32) -> u32 {
    let nyquist = (feature_m / (2.0 * cell_m)).max(1.0).log2().floor() as u32 + 1;
    requested.min(nyquist).max(1)
}

/// The seed's contribution to the octave lattice offsets: `(seed · 2654435761) mod 4096`.
pub(crate) fn seed_mix(seed: u64) -> u32 {
    (seed.wrapping_mul(2_654_435_761) % 4096) as u32
}

/// The terrain surface heights at the last readback: what `Environment::terrain_height` and the
/// wiring in other domains sample. Cloning is cheap (the heights are shared).
#[derive(Clone, Debug)]
pub struct TerrainSnapshot {
    /// The sampled grid: its cell and size are those of the readback (2× coarser than the
    /// simulation above 1024²).
    pub grid: TerrainGrid,
    /// Surface height (bedrock + sediment) per cell, m, relative to `grid.y0`.
    pub surface: Arc<[f32]>,
    /// Sediment thickness per cell, m (the loose layer over the bedrock: soil, sand, deposits).
    pub sediment: Arc<[f32]>,
    /// The GPU's surface version this was read at (changes when terrain, erosion or an edit
    /// changed the heights).
    pub version: u32,
}

impl TerrainSnapshot {
    /// Height of the ground at `p` (metre-frame XZ), m; `None` outside the grid.
    pub fn height(&self, p: Vec2) -> Option<f32> {
        self.grid.contains(p).then(|| self.height_clamped(p))
    }

    /// Height of the ground at `p` (metre-frame XZ), m, with the edge cells extended outward:
    /// defined everywhere, for callers (such as `bevy_flora`'s ground) that cannot take `None`.
    pub fn height_clamped(&self, p: Vec2) -> f32 {
        self.grid.y0 + self.grid.sample(&self.surface, p)
    }

    /// Sediment thickness at `p` (metre-frame XZ), m; `None` outside the grid.
    pub fn sediment(&self, p: Vec2) -> Option<f32> {
        self.grid
            .contains(p)
            .then(|| self.grid.sample(&self.sediment, p))
    }

    /// Where the ray from `origin` along the unit `dir` (metre frame) first meets the ground,
    /// searched between the heights `low` and `high` (m, the ground's lowest and highest
    /// surface) every half cell and refined by bisection. `None` for a ray that does not
    /// descend or meets no ground inside the grid.
    pub(crate) fn ray_hit(&self, origin: Vec3, dir: Vec3, low: f32, high: f32) -> Option<Vec3> {
        /// The most samples along the slab before the step grows.
        const MAX_STEPS: f32 = 4096.0;
        if dir.y >= 0.0 {
            return None;
        }
        let at = |t: f32| origin + dir * t;
        // Above the ground (or off the grid) is positive.
        let above = |t: f32| {
            let p = at(t);
            self.height(p.xz()).map_or(f32::INFINITY, |h| p.y - h)
        };
        let (t0, t1) = (
            ((origin.y - high) / -dir.y).max(0.0),
            (origin.y - low) / -dir.y,
        );
        if t1 < t0 {
            return None;
        }
        let step = (0.5 * self.grid.cell_m).max((t1 - t0) / MAX_STEPS);
        let mut prev = t0;
        if above(prev) <= 0.0 {
            return Some(at(prev));
        }
        loop {
            let t = (prev + step).min(t1);
            if above(t) <= 0.0 {
                let (mut a, mut b) = (prev, t);
                for _ in 0..16 {
                    let mid = 0.5 * (a + b);
                    if above(mid) <= 0.0 {
                        b = mid;
                    } else {
                        a = mid;
                    }
                }
                return Some(at(b));
            }
            if t >= t1 {
                return None;
            }
            prev = t;
        }
    }
}

#[cfg(test)]
pub(crate) mod mirror;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_resolution_is_clamped_and_rounded_to_a_power_of_two() {
        let e = Vec2::splat(2048.0);
        assert_eq!(
            resolve_resolution(e, UVec2::splat(1024), 1024),
            (UVec2::splat(1024), 2.0, false)
        );
        assert_eq!(
            resolve_resolution(e, UVec2::splat(4096), 2048).0,
            UVec2::splat(2048)
        );
        assert_eq!(
            resolve_resolution(e, UVec2::splat(1024), 256).0,
            UVec2::splat(256)
        );
        assert_eq!(
            resolve_resolution(e, UVec2::splat(1000), 2048).0,
            UVec2::splat(512)
        );
        assert_eq!(
            resolve_resolution(e, UVec2::splat(1), 2048).0,
            UVec2::splat(32)
        );
    }

    #[test]
    fn square_cells_are_kept_by_rederiving_the_rows() {
        let (res, cell, adjusted) =
            resolve_resolution(Vec2::new(2048.0, 1024.0), UVec2::splat(1024), 1024);
        // 2048 × 1024 at 1024²: cells would be 2 × 1, so rows become 1024 / 2 = 512.
        assert_eq!(res, UVec2::new(1024, 512));
        assert_eq!(cell, 2.0);
        assert!(adjusted);
        // 1000 × 1500 at 256²: the rows bind (5.86 m cells), so the columns become 160.
        let (res, cell, adjusted) =
            resolve_resolution(Vec2::new(1000.0, 1500.0), UVec2::splat(256), 256);
        assert!(adjusted);
        assert_eq!(res, UVec2::new(160, 256));
        assert_eq!(cell, 1500.0 / 256.0);
        // A long strip never exceeds the cap on its long axis: 4096 m of rows at 512 cells.
        assert_eq!(
            resolve_resolution(Vec2::new(256.0, 4096.0), UVec2::splat(1024), 512),
            (UVec2::new(32, 512), 8.0, true)
        );
    }

    #[test]
    fn octaves_stop_at_the_nyquist_limit() {
        assert_eq!(octave_count(8, 600.0, 2.0), 8);
        assert_eq!(octave_count(8, 300.0, 2.0), 7);
        assert_eq!(octave_count(8, 1500.0, 2.0), 8);
        assert_eq!(octave_count(8, 800.0, 2.0), 8);
        assert_eq!(octave_count(8, 600.0, 8.0), 6);
        assert_eq!(octave_count(0, 600.0, 2.0), 1);
    }

    #[test]
    fn sampling_is_bilinear_with_cell_centres_at_half() {
        let grid = TerrainGrid {
            origin: Vec2::new(-10.0, -10.0),
            cell_m: 2.0,
            size: UVec2::new(4, 4),
            y0: 5.0,
        };
        // Surface z = x·1 + row·10 at cell centres.
        let layer: Vec<f32> = (0..16)
            .map(|i| (i % 4) as f32 + (i / 4) as f32 * 10.0)
            .collect();
        // Cell (1, 2) centre = (-10 + 3, -10 + 5): the stored value.
        assert!((grid.sample(&layer, Vec2::new(-7.0, -5.0)) - 21.0).abs() < 1e-5);
        // Half way between cell (1, 2) and (2, 2).
        assert!((grid.sample(&layer, Vec2::new(-6.0, -5.0)) - 21.5).abs() < 1e-5);
        let snap = TerrainSnapshot {
            grid,
            surface: layer.into(),
            sediment: vec![0.25; 16].into(),
            version: 1,
        };
        assert_eq!(snap.height(Vec2::new(-7.0, -5.0)), Some(26.0));
        assert_eq!(snap.sediment(Vec2::new(-7.0, -5.0)), Some(0.25));
        assert_eq!(snap.height(Vec2::new(-10.1, 0.0)), None);
        // The far corner of the extent is inside; a hair beyond it is not.
        assert!(snap.height(Vec2::new(-2.0, -2.0)).is_some());
        assert_eq!(snap.height(Vec2::new(-1.9, -2.0)), None);
    }

    /// A 512 m square of 2 m cells centred on the origin with the surface `h(x, z)`, 3 m above
    /// the metre frame's zero.
    fn snapshot(h: impl Fn(f32, f32) -> f32) -> TerrainSnapshot {
        let (n, cell) = (256u32, 2.0);
        let origin = -0.5 * n as f32 * cell;
        let surface: Vec<f32> = (0..n * n)
            .map(|i| {
                let (x, z) = ((i % n) as f32 + 0.5, (i / n) as f32 + 0.5);
                h(origin + x * cell, origin + z * cell)
            })
            .collect();
        TerrainSnapshot {
            grid: TerrainGrid {
                origin: Vec2::splat(origin),
                cell_m: cell,
                size: UVec2::splat(n),
                y0: 3.0,
            },
            sediment: Arc::from(vec![0.0; surface.len()]),
            surface: Arc::from(surface),
            version: 1,
        }
    }

    /// `orthographic-lod.md` Model 1: an orthographic camera without a focus centres its patches
    /// where its forward ray first meets the ground.
    #[test]
    fn a_view_ray_meets_the_ground_where_it_first_crosses_it() {
        // Pitched 35.26° down along +X+Z.
        let forward = Vec3::new(0.5, -0.5, 0.5).normalize();
        // The stored 0 is 3 m up (`y0`): the ray from 457 m back lands on the analytic point.
        let flat = snapshot(|_, _| 0.0);
        let target = Vec3::new(20.0, 3.0, -30.0);
        let hit = flat
            .ray_hit(target - forward * 457.0, forward, 3.0, 3.0)
            .expect("the ray meets the plane");
        assert!(hit.distance(target) < 1e-2, "{hit} vs {target}");
        // A ridge 40 m high across the ray's path, short of where it would meet the plane (at
        // x = z = -153): the ray stops on the ridge's near face, about 18 m up, not behind it.
        let ridge = snapshot(|x, z| {
            if (x + z + 320.0).abs() < 15.0 {
                40.0
            } else {
                0.0
            }
        });
        let eye = Vec3::new(-150.0, 0.0, -150.0) - forward * 300.0;
        let hit = ridge
            .ray_hit(eye, forward, 3.0, 43.0)
            .expect("the ray meets the ridge");
        let ground = ridge.height(hit.xz()).expect("inside the grid");
        assert!(
            (hit.y - ground).abs() < 0.05,
            "{hit} is not on the ground ({ground})"
        );
        assert!(hit.y > 10.0, "{hit} passed the ridge");
        // A level or rising ray, and one that leaves the grid before it comes down, meet none.
        assert_eq!(flat.ray_hit(target, Vec3::X, 3.0, 3.0), None);
        assert_eq!(
            flat.ray_hit(target, Vec3::new(0.0, 0.6, 0.8), 3.0, 3.0),
            None
        );
        let shallow = Vec3::new(1.0, -0.01, 0.0).normalize();
        assert_eq!(
            flat.ray_hit(Vec3::new(0.0, 10.0, 0.0), shallow, 3.0, 3.0),
            None
        );
    }
}
