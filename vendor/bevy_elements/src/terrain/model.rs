//! The terrain's CPU side that is not a system: the grid geometry, the height snapshot gameplay
//! queries sample, and the generator's constants. The generator and layer-blend mirrors that the
//! tests compare the kernels against are in `mirror`.

use std::sync::Arc;

use bevy::math::{UVec2, Vec2};

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

/// Placement of the terrain's grid in the world: row-major `size.x × size.y` cells, cell `(0, 0)`
/// centred at `origin + 0.5 · cell_m`, the row index running along world +Z.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TerrainGrid {
    /// World XZ of the minimum corner, m.
    pub origin: Vec2,
    /// Cell edge, m (cells are square).
    pub cell_m: f32,
    /// Cells per axis.
    pub size: UVec2,
    /// World height of a stored height of 0, m (the terrain transform's `y`).
    pub y0: f32,
}

impl TerrainGrid {
    /// World size of the grid, m.
    pub fn extent(&self) -> Vec2 {
        self.size.as_vec2() * self.cell_m
    }

    /// Whether `p` (world XZ) lies inside the grid.
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
/// leaf, 32 cells). Cells must be square: `c = extent.x / res.x`; if `extent.y / res.y` differs,
/// `res.y` is re-derived as `round(extent.y / c / 32) · 32`.
pub(crate) fn resolve_resolution(extent: Vec2, resolution: UVec2, max: u32) -> (UVec2, f32, bool) {
    let snap = |r: u32| {
        let r = r.clamp(LEAF_CELLS, max.max(LEAF_CELLS));
        1 << r.ilog2()
    };
    let res = UVec2::new(snap(resolution.x), snap(resolution.y));
    let cell = extent.x / res.x as f32;
    let adjusted = ((extent.y / res.y as f32) - cell).abs() > cell * 1e-4;
    if !adjusted {
        return (res, cell, false);
    }
    let rows = ((extent.y / cell / LEAF_CELLS as f32).round().max(1.0) as u32) * LEAF_CELLS;
    (UVec2::new(res.x, rows), cell, true)
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
    /// The GPU's surface version this was read at (changes when terrain, erosion or an edit
    /// changed the heights).
    pub version: u32,
}

impl TerrainSnapshot {
    /// World height of the ground at `p` (world XZ), `None` outside the grid.
    pub fn height(&self, p: Vec2) -> Option<f32> {
        self.grid.contains(p).then(|| self.height_clamped(p))
    }

    /// World height of the ground at `p`, with the edge cells extended outward: defined
    /// everywhere, for callers (such as `bevy_flora`'s ground) that cannot take `None`.
    pub fn height_clamped(&self, p: Vec2) -> f32 {
        self.grid.y0 + self.grid.sample(&self.surface, p)
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
        let (res, _, adjusted) =
            resolve_resolution(Vec2::new(1000.0, 1500.0), UVec2::splat(256), 256);
        assert!(adjusted);
        assert_eq!(res.y % LEAF_CELLS, 0);
        assert_eq!(res.y, 384);
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
            version: 1,
        };
        assert_eq!(snap.height(Vec2::new(-7.0, -5.0)), Some(26.0));
        assert_eq!(snap.height(Vec2::new(-10.1, 0.0)), None);
        // The far corner of the extent is inside; a hair beyond it is not.
        assert!(snap.height(Vec2::new(-2.0, -2.0)).is_some());
        assert_eq!(snap.height(Vec2::new(-1.9, -2.0)), None);
    }
}
