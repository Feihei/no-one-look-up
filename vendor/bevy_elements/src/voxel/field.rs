//! `SnowyField`: a voxel field with a slab of snow over its column tops.

use std::fmt;
use std::sync::Arc;

use bevy::math::Vec2;
use bevy_isomesh::VolumeField;
use isomesh::Sdf;

use crate::terrain::TerrainGrid;

/// A voxel field with snow on it: the union of `base` and a slab that exists only over each
/// column's top.
///
/// For a world point `p`, with `c` the voxel cell size, `t(x, z)` the column top and `s(x, z)`
/// the snow depth (both bilinear in their grids, edge clamped, the tap rule of the terrain):
///
/// ```text
/// sample(p) = min( base(p),  max( p.y − (t + s),  (t − c) − p.y ) )
/// ```
///
/// The `max` is the signed distance to the slab `t − c < y < t + s`, negative strictly inside
/// it. The slab's bottom is one cell below the column top so that it overlaps the solid it sits
/// on and the union has no gap at the join. `min` is the union, so the base is never carved, and
/// at every point at least `c` below the top the slab term is `≥ 0`: **the sign there, and so the
/// mesh, is the base's, exactly.** A tunnel under a roof thicker than `c` never gets snow.
///
/// The value is a signed distance only inside the slab's own faces; like the reference fields of
/// `isomesh` it is a valid solid but not an exact distance, and `gradient` is the default central
/// difference.
///
/// [`VoxelGround`](crate::VoxelGround) builds and swaps these for a volume; the type is public so
/// a game or a test can build one by hand.
#[derive(Clone)]
pub struct SnowyField {
    base: Arc<dyn VolumeField>,
    cell_size: f32,
    top: Arc<[f32]>,
    top_grid: TerrainGrid,
    snow: Arc<[f32]>,
    snow_grid: TerrainGrid,
}

impl SnowyField {
    /// Wrap `base`. `cell_size` is the voxel cell size `c`; `top` holds the column tops (world
    /// height, m) on `top_grid` and `snow` the snow depths (m) on `snow_grid`, row-major, the
    /// layout of a [`TerrainGrid`] (`grid.y0` is not added: both are world heights).
    ///
    /// # Panics
    ///
    /// If a grid's cell count is not the length of its layer: sampling would read out of range.
    pub fn new(
        base: Arc<dyn VolumeField>,
        cell_size: f32,
        top: Arc<[f32]>,
        top_grid: TerrainGrid,
        snow: Arc<[f32]>,
        snow_grid: TerrainGrid,
    ) -> Self {
        let cells = |g: &TerrainGrid| g.size.x as usize * g.size.y as usize;
        assert_eq!(
            top.len(),
            cells(&top_grid),
            "the column tops must cover their grid"
        );
        assert_eq!(
            snow.len(),
            cells(&snow_grid),
            "the snow depths must cover their grid"
        );
        Self {
            base,
            cell_size,
            top,
            top_grid,
            snow,
            snow_grid,
        }
    }

    /// The field this one adds snow to.
    pub fn base(&self) -> &Arc<dyn VolumeField> {
        &self.base
    }

    /// The snow depths (m) this field holds, on [`snow_grid`](Self::snow_grid).
    pub fn snow(&self) -> &Arc<[f32]> {
        &self.snow
    }

    /// The grid of [`snow`](Self::snow).
    pub fn snow_grid(&self) -> &TerrainGrid {
        &self.snow_grid
    }

    /// The column tops (world height, m) this field holds, on [`top_grid`](Self::top_grid).
    pub fn top(&self) -> &Arc<[f32]> {
        &self.top
    }

    /// The grid of [`top`](Self::top).
    pub fn top_grid(&self) -> &TerrainGrid {
        &self.top_grid
    }
}

impl fmt::Debug for SnowyField {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SnowyField")
            .field("cell_size", &self.cell_size)
            .field("top_grid", &self.top_grid)
            .field("snow_grid", &self.snow_grid)
            .finish_non_exhaustive()
    }
}

impl Sdf for SnowyField {
    type Scalar = f32;

    #[inline]
    fn sample(&self, p: [f32; 3]) -> f32 {
        let xz = Vec2::new(p[0], p[2]);
        let top = self.top_grid.sample(&self.top, xz);
        let snow = self.snow_grid.sample(&self.snow, xz);
        let slab = (p[1] - (top + snow)).max((top - self.cell_size) - p[1]);
        self.base.sample(p).min(slab)
    }
}

#[cfg(test)]
mod tests {
    use bevy::math::UVec2;
    use isomesh::fields::Sphere;

    use super::*;

    /// An `n × n` grid of `cell`-metre cells centred on the origin.
    pub(crate) fn centred_grid(n: u32, cell: f32) -> TerrainGrid {
        TerrainGrid {
            origin: Vec2::splat(-(n as f32) * cell * 0.5),
            cell_m: cell,
            size: UVec2::splat(n),
            y0: 0.0,
        }
    }

    fn constant(grid: &TerrainGrid, value: f32) -> Arc<[f32]> {
        vec![value; (grid.size.x * grid.size.y) as usize].into()
    }

    fn sphere_slab(radius: f32, snow: f32) -> SnowyField {
        let grid = centred_grid(8, 1.0);
        SnowyField::new(
            Arc::new(Sphere {
                center: [0.0; 3],
                radius,
            }),
            0.5,
            constant(&grid, radius),
            grid,
            constant(&grid, snow),
            grid,
        )
    }

    /// Validation 1: the slab sits on the sphere top and ends at `t + s`.
    #[test]
    fn snowy_field_sphere_slab() {
        let r = 2.0;
        let field = sphere_slab(r, 0.2);
        let inside_slab = field.sample([0.0, r + 0.1, 0.0]);
        assert!(inside_slab < 0.0, "inside the slab: {inside_slab}");
        assert!((inside_slab + 0.1).abs() < 1e-6, "{inside_slab}");
        let above = field.sample([0.0, r + 0.3, 0.0]);
        assert!(above > 0.0, "above the snow: {above}");
        assert!((above - 0.1).abs() < 1e-6, "{above}");
        // Deep in the solid the slab term is positive and the base's value stands.
        let base = Sphere {
            center: [0.0; 3],
            radius: r,
        };
        let deep = field.sample([0.0, r - 0.7, 0.0]);
        assert_eq!(deep, base.sample([0.0, r - 0.7, 0.0]));
        assert!((deep + 0.7).abs() < 1e-6, "{deep}");
        // The snow surface is at t + s, to a millimetre.
        assert!(field.sample([0.0, r + 0.2 + 1e-3, 0.0]) > 0.0);
        assert!(field.sample([0.0, r + 0.2 - 1e-3, 0.0]) < 0.0);
    }

    /// The bilinear column top at the origin of a sphere's cell-centre heights is
    /// `√(r² − 0.5)`, so the assertion above is about the slab, not about the interpolation.
    #[test]
    fn snowy_field_interpolates_the_top_bilinearly() {
        let r = 2.0_f32;
        let grid = centred_grid(8, 1.0);
        let tops: Arc<[f32]> = (0..grid.size.y)
            .flat_map(|j| (0..grid.size.x).map(move |i| (i, j)))
            .map(|(i, j)| {
                let x = grid.origin.x + (i as f32 + 0.5) * grid.cell_m;
                let z = grid.origin.y + (j as f32 + 0.5) * grid.cell_m;
                (r * r - x * x - z * z).max(0.0).sqrt()
            })
            .collect();
        let at_origin = grid.sample(&tops, Vec2::ZERO);
        let want = (r * r - 0.5).sqrt();
        assert!((at_origin - want).abs() < 1e-4, "{at_origin} vs {want}");
        assert!((want - 1.8708).abs() < 1e-4);
    }

    /// Validation 2: a tunnel under the top is the base's, bit for bit; the slab is on top.
    #[test]
    fn snowy_field_tunnel_is_snow_free() {
        struct Tunnel;
        impl Sdf for Tunnel {
            type Scalar = f32;
            fn sample(&self, p: [f32; 3]) -> f32 {
                (p[1] - 10.0).max(1.0 - (p[1] - 5.0).abs())
            }
        }
        let grid = centred_grid(8, 1.0);
        let field = SnowyField::new(
            Arc::new(Tunnel),
            0.5,
            constant(&grid, 10.0),
            grid,
            constant(&grid, 0.5),
            grid,
        );
        assert_eq!(field.sample([0.0, 5.0, 0.0]), 1.0);
        let mut y = 4.0_f32;
        while y <= 6.0 {
            assert_eq!(
                field.sample([0.0, y, 0.0]),
                Tunnel.sample([0.0, y, 0.0]),
                "y = {y}"
            );
            y += 0.0625;
        }
        assert_eq!(field.sample([0.0, 9.0, 0.0]), -1.0);
        assert!((field.sample([0.0, 10.25, 0.0]) + 0.25).abs() < 1e-6);
        assert!((field.sample([0.0, 10.75, 0.0]) - 0.25).abs() < 1e-6);
    }

    /// Validation 3: the slab's top is a horizontal plane, so the gradient points up.
    #[test]
    fn gradient_points_up_on_the_slab() {
        struct Tunnel;
        impl Sdf for Tunnel {
            type Scalar = f32;
            fn sample(&self, p: [f32; 3]) -> f32 {
                (p[1] - 10.0).max(1.0 - (p[1] - 5.0).abs())
            }
        }
        let grid = centred_grid(8, 1.0);
        let field = SnowyField::new(
            Arc::new(Tunnel),
            0.5,
            constant(&grid, 10.0),
            grid,
            constant(&grid, 0.5),
            grid,
        );
        let g = field.gradient([0.0, 10.25, 0.0]);
        assert!(
            g[0].abs() < 1e-2 && (g[1] - 1.0).abs() < 1e-2 && g[2].abs() < 1e-2,
            "{g:?}"
        );
    }
}
