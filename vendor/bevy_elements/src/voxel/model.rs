//! The voxel ground's CPU model: the column-top scan, the chunks a change of columns can reach,
//! and the per-column sample-and-hold of the snow depth. Plain functions over plain data; the
//! systems in `mod.rs` call them. See `docs/research/voxel-snow.md`.

use std::collections::{HashMap, HashSet};

use bevy::math::{UVec2, Vec2};
use bevy::prelude::Entity;
use bevy_isomesh::VoxelChunk;
use isomesh::Sdf;
use isomesh::chunk::{ChunkId, ChunkLayout};

use crate::terrain::TerrainGrid;

/// Bisection steps that refine a top once the scan has bracketed it: the error is
/// `c / 2⁹` (the midpoint of a bracket `c / 2⁸` wide), 1 mm at `c = 0.5`.
pub(crate) const BISECTIONS: u32 = 8;

/// A top that moved less than this, m, is the same top (below the bisection error and above the
/// `f32` noise of a position near 16 m).
pub(crate) const TOP_EPSILON_M: f32 = 1e-3;

/// A column's snow is swapped when its depth is further than this many voxel cells from the held
/// value (the plan's `0.5 c`).
pub(crate) const SWAP_CELLS: f32 = 0.5;

/// Where a volume's columns are: the terrain's grid over the volume's XZ extent, and the
/// vertical range the columns are scanned over.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Columns {
    /// The terrain grid (`y0` = 0: a stored top is a world height).
    pub grid: TerrainGrid,
    /// Bottom and top plane of the volume's chunks, m.
    pub y_lo: f32,
    pub y_hi: f32,
    /// The voxel cell size `c`, m.
    pub cell: f32,
}

impl Columns {
    /// World XZ of the centre of column `index`.
    pub fn centre(&self, index: usize) -> Vec2 {
        let w = self.grid.size.x as usize;
        let at = Vec2::new((index % w) as f32, (index / w) as f32);
        self.grid.origin + (at + 0.5) * self.grid.cell_m
    }

    /// The top stored for a column with no solid: two cells under the volume, so no sample ever
    /// sits inside its slab.
    pub fn no_top(&self) -> f32 {
        self.y_lo - 2.0 * self.cell
    }

    /// Whether a stored top is a real surface.
    pub fn has_top(&self, top: f32) -> bool {
        top > self.y_lo - self.cell
    }

    /// The column count.
    pub fn len(&self) -> usize {
        self.grid.size.x as usize * self.grid.size.y as usize
    }

    /// The columns whose centre lies in the XZ box of a chunk column (closed on both sides: a
    /// column on the shared plane belongs to both chunks that touch it), as inclusive
    /// `(i0, i1, j0, j1)`, or `None` when the box misses the grid.
    pub fn columns_in(
        &self,
        layout: &ChunkLayout<f32>,
        chunk: ChunkId,
    ) -> Option<(u32, u32, u32, u32)> {
        let n = layout.cells() as i64;
        let lo = layout.world_of_sample(layout.base_sample(chunk));
        let hi_sample = layout.base_sample(chunk).map(|s| s + n);
        let hi = layout.world_of_sample(hi_sample);
        let axis = |lo: f32, hi: f32, origin: f32, count: u32| {
            let first = ((lo - origin) / self.grid.cell_m - 0.5).ceil().max(0.0);
            let last = ((hi - origin) / self.grid.cell_m - 0.5).floor();
            (last >= first && first < count as f32)
                .then(|| (first as u32, (last as u32).min(count - 1)))
        };
        let (i0, i1) = axis(lo[0], hi[0], self.grid.origin.x, self.grid.size.x)?;
        let (j0, j1) = axis(lo[2], hi[2], self.grid.origin.y, self.grid.size.y)?;
        Some((i0, i1, j0, j1))
    }
}

/// The volume's columns from its chunk entities: the XZ box (as the extent and centre of the
/// terrain) and the vertical range. `None` while no chunk exists.
pub(crate) fn volume_box<'a>(
    layout: &ChunkLayout<f32>,
    chunks: impl IntoIterator<Item = &'a VoxelChunk>,
) -> Option<([f32; 3], [f32; 3])> {
    let n = layout.cells() as i64;
    let mut bounds: Option<([i64; 3], [i64; 3])> = None;
    for chunk in chunks {
        let lo = layout.base_sample(chunk.id);
        let hi = lo.map(|s| s + n);
        bounds = Some(match bounds {
            None => (lo, hi),
            Some((min, max)) => (
                std::array::from_fn(|a| lo[a].min(min[a])),
                std::array::from_fn(|a| hi[a].max(max[a])),
            ),
        });
    }
    let (min, max) = bounds?;
    Some((layout.world_of_sample(min), layout.world_of_sample(max)))
}

/// The highest zero crossing of `base` along the vertical through `(x, z)`: the first
/// `f ≥ 0 → f < 0` transition from `y_hi` down in steps of `c`, refined by [`BISECTIONS`]
/// bisections. `None` when the column has no solid. A column that is solid at `y_hi` has its
/// top there: the volume has no chunk above it.
pub(crate) fn column_top<S: Sdf<Scalar = f32> + ?Sized>(
    base: &S,
    x: f32,
    z: f32,
    y_hi: f32,
    y_lo: f32,
    c: f32,
) -> Option<f32> {
    let mut above = y_hi;
    if base.sample([x, above, z]) < 0.0 {
        return Some(y_hi);
    }
    // The lowest sample plane is `y_lo` itself: a chunk samples one plane beyond its cells.
    let steps = ((y_hi - y_lo) / c).round() as u32;
    for k in 1..=steps {
        let y = y_hi - k as f32 * c;
        let f = base.sample([x, y, z]);
        if f < 0.0 {
            let (mut lo, mut hi) = (y, above);
            for _ in 0..BISECTIONS {
                let mid = 0.5 * (lo + hi);
                if base.sample([x, mid, z]) >= 0.0 {
                    hi = mid;
                } else {
                    lo = mid;
                }
            }
            return Some(0.5 * (lo + hi));
        }
        above = y;
    }
    None
}

/// The top of every column in `columns` (flat indices), written into `tops`, with the no-top
/// value for a column with no solid. Returns the number of columns scanned.
pub(crate) fn scan_columns<S: Sdf<Scalar = f32> + ?Sized>(
    base: &S,
    columns: &Columns,
    indices: impl IntoIterator<Item = usize>,
    tops: &mut [f32],
) -> u64 {
    let mut scanned = 0;
    for index in indices {
        let at = columns.centre(index);
        tops[index] = column_top(base, at.x, at.y, columns.y_hi, columns.y_lo, columns.cell)
            .unwrap_or_else(|| columns.no_top());
        scanned += 1;
    }
    scanned
}

/// The flat indices of the columns in an inclusive box.
pub(crate) fn box_indices(
    size: UVec2,
    (i0, i1, j0, j1): (u32, u32, u32, u32),
) -> impl Iterator<Item = usize> {
    (j0..=j1).flat_map(move |j| (i0..=i1).map(move |i| (j * size.x + i) as usize))
}

/// The Y interval whose chunks a change of a column's slab can reach, m, as `(low, high)`.
/// Each slab is `[t − c, t + s]`; the interval is the union of the old and the new one (a slab
/// exists only over a real top) with a cell of halo.
pub(crate) fn slab_span(columns: &Columns, old: (f32, f32), new: (f32, f32)) -> Option<(f32, f32)> {
    let c = columns.cell;
    let mut span: Option<(f32, f32)> = None;
    for (top, snow) in [old, new] {
        if !columns.has_top(top) {
            continue;
        }
        let (lo, hi) = (top - c - c, top + snow + c);
        span = Some(match span {
            None => (lo, hi),
            Some((a, b)) => (a.min(lo), b.max(hi)),
        });
    }
    span
}

/// The Y interval whose chunks a change of a column's snow depth alone can reach, m: the mesh
/// moves where the old and the new slab surface lie, with a cell of halo. A column with no top
/// has no slab.
pub(crate) fn snow_span(columns: &Columns, top: f32, old: f32, new: f32) -> Option<(f32, f32)> {
    columns.has_top(top).then(|| {
        (
            top + old.min(new) - columns.cell,
            top + old.max(new) + columns.cell,
        )
    })
}

/// The chunks that sample column `index`'s bilinear footprint over `[y0, y1]`: the footprint is
/// `± cell` about the column centre, widened by one voxel cell `c` (a chunk samples the plane
/// beyond the cells it owns, and ownership is half-open).
pub(crate) fn chunks_of_column(
    layout: &ChunkLayout<f32>,
    columns: &Columns,
    index: usize,
    (y0, y1): (f32, f32),
) -> ([i32; 3], [i32; 3]) {
    let at = columns.centre(index);
    let reach = columns.grid.cell_m + columns.cell;
    let lo = layout.chunk_of([at.x - reach, y0, at.y - reach]).coords;
    let hi = layout.chunk_of([at.x + reach, y1, at.y + reach]).coords;
    (lo, hi)
}

/// Accumulates the chunk ids of changed columns without re-inserting the box of the column
/// before it (adjacent columns mostly share one).
#[derive(Default)]
pub(crate) struct DirtyChunks {
    ids: HashSet<ChunkId>,
    last: Option<([i32; 3], [i32; 3])>,
}

impl DirtyChunks {
    /// Add the chunks of one changed column over a Y interval.
    pub fn add(
        &mut self,
        layout: &ChunkLayout<f32>,
        columns: &Columns,
        index: usize,
        span: (f32, f32),
    ) {
        let boxed = chunks_of_column(layout, columns, index, span);
        if self.last == Some(boxed) {
            return;
        }
        self.last = Some(boxed);
        let (lo, hi) = boxed;
        for x in lo[0]..=hi[0] {
            for y in lo[1]..=hi[1] {
                for z in lo[2]..=hi[2] {
                    self.ids.insert(ChunkId::new([x, y, z]));
                }
            }
        }
    }

    /// The chunk ids collected.
    pub fn into_ids(self) -> HashSet<ChunkId> {
        self.ids
    }
}

/// The chunk entities of one volume by id.
pub(crate) fn chunk_entities<'a>(
    volume: Entity,
    chunks: impl IntoIterator<Item = (Entity, &'a VoxelChunk)>,
) -> HashMap<ChunkId, Entity> {
    chunks
        .into_iter()
        .filter(|(_, chunk)| chunk.volume == volume)
        .map(|(entity, chunk)| (chunk.id, entity))
        .collect()
}

/// The result of comparing a fresh snow map with the held one.
#[derive(Debug)]
pub(crate) struct Swap {
    /// The new held depths.
    pub snow: Vec<f32>,
    /// Columns that took a fresh value: `(index, old, new)`.
    pub changed: Vec<(usize, f32, f32)>,
}

/// Per-column sample-and-hold: a column whose fresh depth is further than `0.5 c` from the held
/// one takes the fresh value, every other column keeps what it held, a column with no top holds
/// 0. `fresh(i)` is the fresh depth of column `i` (m). `None` when no column is over the
/// threshold: nothing is allocated.
pub(crate) fn hold_swap(
    columns: &Columns,
    tops: &[f32],
    held: &[f32],
    fresh: impl Fn(usize) -> f32,
) -> Option<Swap> {
    let threshold = SWAP_CELLS * columns.cell;
    let over =
        |i: usize| columns.has_top(tops[i]) && (fresh(i).max(0.0) - held[i]).abs() > threshold;
    if !(0..held.len()).any(over) {
        return None;
    }
    let mut snow = held.to_vec();
    let mut changed = Vec::new();
    for i in 0..held.len() {
        if over(i) {
            let new = fresh(i).max(0.0);
            changed.push((i, held[i], new));
            snow[i] = new;
        }
    }
    Some(Swap { snow, changed })
}

/// The columns of a [`VoxelSnow`](super::VoxelSnow) volume over its field-space box `(min, max)`:
/// one column per voxel, the column width doubled until the grid fits in `max_cells` a side. The
/// grid is in the volume's field space (`y0` = 0).
pub(crate) fn snow_columns(min: [f32; 3], max: [f32; 3], cell: f32, max_cells: u32) -> Columns {
    let extent = Vec2::new(max[0] - min[0], max[2] - min[2]);
    let mut cell_m = cell;
    while (extent / cell_m).max_element() > max_cells as f32 {
        cell_m *= 2.0;
    }
    let size = (extent / cell_m).ceil().max(Vec2::ONE).as_uvec2();
    Columns {
        grid: TerrainGrid {
            origin: Vec2::new(min[0], min[2]),
            cell_m,
            size,
            y0: 0.0,
        },
        y_lo: min[1],
        y_hi: max[1],
        cell,
    }
}

/// The share of the snowfall the top of column `index` keeps on its slope:
/// `1 − smoothstep(α₀, α₁, α)` with `α` the slope (degrees) from the neighbouring tops (central
/// differences, one-sided at an edge, beside a column with no top or across a cliff: a
/// neighbour more than a voxel higher or lower is a wall, not a slope, as in [`SnowyField`]'s
/// cliff rule), `shed_deg = [α₀, α₁]` (`docs/research/snow-cover.md`). A column with no top
/// keeps nothing.
///
/// [`SnowyField`]: super::SnowyField
pub(crate) fn top_exposure(
    columns: &Columns,
    tops: &[f32],
    index: usize,
    shed_deg: [f32; 2],
) -> f32 {
    let top = tops[index];
    if !columns.has_top(top) {
        return 0.0;
    }
    let (w, h) = (columns.grid.size.x as usize, columns.grid.size.y as usize);
    let (i, j) = (index % w, index / w);
    let neighbour = |ni: usize, nj: usize| {
        let t = tops[nj * w + ni];
        (columns.has_top(t) && (t - top).abs() <= columns.cell).then_some(t)
    };
    let slope = |lo: Option<(f32, f32)>, hi: Option<(f32, f32)>| match (lo, hi) {
        (Some((tl, dl)), Some((th, dh))) => (th - tl) / (dh + dl),
        (Some((tl, dl)), None) => (top - tl) / dl,
        (None, Some((th, dh))) => (th - top) / dh,
        (None, None) => 0.0,
    };
    let c = columns.grid.cell_m;
    let dx = slope(
        (i > 0)
            .then(|| neighbour(i - 1, j))
            .flatten()
            .map(|t| (t, c)),
        (i + 1 < w)
            .then(|| neighbour(i + 1, j))
            .flatten()
            .map(|t| (t, c)),
    );
    let dz = slope(
        (j > 0)
            .then(|| neighbour(i, j - 1))
            .flatten()
            .map(|t| (t, c)),
        (j + 1 < h)
            .then(|| neighbour(i, j + 1))
            .flatten()
            .map(|t| (t, c)),
    );
    let alpha = Vec2::new(dx, dz).length().atan().to_degrees();
    1.0 - crate::cover::model::smoothstep(shed_deg[0], shed_deg[1], alpha)
}

/// The uniform scale of a transform whose vertical stays vertical (any translation, any turn
/// about the vertical, a uniform scale), or `None` for a tilted or unevenly scaled one: snow
/// falls down the world's vertical, which must be the field's.
pub(crate) fn upright_scale(transform: &bevy::prelude::GlobalTransform) -> Option<f32> {
    let (scale, rotation, _) = transform.to_scale_rotation_translation();
    let up = rotation * bevy::math::Vec3::Y;
    let even = (scale.x - scale.y).abs() <= 1e-3 * scale.y.abs()
        && (scale.z - scale.y).abs() <= 1e-3 * scale.y.abs();
    (up.y > 0.9999 && even && scale.y > 0.0).then_some(scale.y)
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use isomesh::fields::Sphere;

    use super::*;

    fn layout() -> ChunkLayout<f32> {
        ChunkLayout::new(16, 0.5, [0.0; 3]).expect("a valid layout")
    }

    fn columns(layout: &ChunkLayout<f32>, n: u32, y: (f32, f32)) -> Columns {
        Columns {
            grid: TerrainGrid {
                origin: Vec2::ZERO,
                cell_m: 0.5,
                size: UVec2::splat(n),
                y0: 0.0,
            },
            y_lo: y.0,
            y_hi: y.1,
            cell: layout.cell_size(),
        }
    }

    /// Counts the field evaluations of a scan.
    struct Counting<S>(S, Cell<u32>);

    impl<S: Sdf<Scalar = f32>> Sdf for Counting<S> {
        type Scalar = f32;
        fn sample(&self, p: [f32; 3]) -> f32 {
            self.1.set(self.1.get() + 1);
            self.0.sample(p)
        }
    }

    struct Layers<F: Fn([f32; 3]) -> f32>(F);

    impl<F: Fn([f32; 3]) -> f32> Sdf for Layers<F> {
        type Scalar = f32;
        fn sample(&self, p: [f32; 3]) -> f32 {
            (self.0)(p)
        }
    }

    /// Validation 4: the top of a sphere column is found to the bisection error in the
    /// evaluations the scan allows, a column outside has no top, a roof over a cave is the top,
    /// and a pit open to the sky has its floor as the top.
    #[test]
    fn column_top_extraction() {
        let sphere = Counting(
            Sphere {
                center: [0.0; 3],
                radius: 2.0,
            },
            Cell::new(0),
        );
        let top = column_top(&sphere, 0.0, 0.0, 4.0, -4.0, 0.5).expect("a top");
        assert!((top - 2.0).abs() <= 2e-3, "{top}");
        // `⌊(y_hi − t)/c⌋ + 2` scan samples (the two that bracket it) and the bisections. (The
        // note counts one fewer scan sample: Deviation 11.)
        assert!(
            sphere.1.get() <= 6 + BISECTIONS,
            "{} evaluations",
            sphere.1.get()
        );
        assert!(column_top(&sphere, 3.0, 3.0, 4.0, -4.0, 0.5).is_none());

        let roof = Layers(|p: [f32; 3]| {
            // Solid below 4 and in [8, 10]; air between.
            let below = p[1] - 4.0;
            let roof = (p[1] - 10.0).max(8.0 - p[1]);
            below.min(roof)
        });
        let top = column_top(&roof, 0.0, 0.0, 12.0, 0.0, 0.5).expect("a top");
        assert!((top - 10.0).abs() <= 2e-3, "a roof over a cave: {top}");

        let pit = Layers(|p: [f32; 3]| {
            let in_pit = p[0].abs().max(p[2].abs()) < 1.0;
            if in_pit { p[1] - 4.0 } else { p[1] - 10.0 }
        });
        let pit_top = column_top(&pit, 0.0, 0.0, 12.0, 0.0, 0.5).expect("a top");
        assert!((pit_top - 4.0).abs() <= 2e-3, "the pit floor: {pit_top}");
        let rim = column_top(&pit, 3.0, 0.0, 12.0, 0.0, 0.5).expect("a top");
        assert!((rim - 10.0).abs() <= 2e-3, "{rim}");
        // Solid at the top plane: the volume has nothing above.
        assert_eq!(column_top(&pit, 3.0, 0.0, 8.0, 0.0, 0.5), Some(8.0));
    }

    /// Validation 4, the no-top value of `scan_columns`.
    #[test]
    fn scan_columns_marks_no_top() {
        let layout = layout();
        let columns = columns(&layout, 8, (-4.0, 4.0));
        let sphere = Sphere {
            center: [2.0, 0.0, 2.0],
            radius: 2.0,
        };
        let mut tops = vec![0.0; columns.len()];
        scan_columns(&sphere, &columns, 0..columns.len(), &mut tops);
        // Column (7, 7) is centred at (3.75, 3.75): outside the sphere.
        assert_eq!(tops[7 * 8 + 7], -5.0);
        assert!(!columns.has_top(tops[7 * 8 + 7]));
        // Column (4, 4) at (2.25, 2.25) is under the dome.
        assert!(columns.has_top(tops[4 * 8 + 4]));
    }

    /// Validation 5: the chunks a changed column can reach.
    #[test]
    fn dirty_chunks_from_columns() {
        let layout = layout();
        let columns = columns(&layout, 32, (-8.0, 16.0));
        let at = |i: usize, j: usize| j * 32 + i;
        let dirty = |index: usize, t: f32, s0: f32, s1: f32| {
            let span = snow_span(&columns, t, s0, s1).expect("a top");
            let mut set = DirtyChunks::default();
            set.add(&layout, &columns, index, span);
            set.into_ids()
        };
        // Column 15 (centre 7.75), z column 7 (centre 3.75) inside one chunk column.
        let set = dirty(at(15, 7), 3.0, 0.0, 0.3);
        let xs: Vec<i32> = {
            let mut v: Vec<_> = set.iter().map(|id| id.coords[0]).collect();
            v.sort_unstable();
            v.dedup();
            v
        };
        assert_eq!(xs, vec![0, 1], "x chunks");
        assert!(set.iter().all(|id| id.coords[2] == 0), "one z chunk");
        // t = 3: the span [2.0, 3.8] is the layer y in [0, 8) of an origin-at-zero layout; the
        // volume bottom is at −8 but chunk ids are not clipped here.
        assert_eq!(set.len(), 2);
        // An interior column marks one chunk.
        assert_eq!(dirty(at(7, 7), 3.0, 0.0, 0.3).len(), 1);
        // A corner column of a chunk marks eight when its Y span crosses a plane.
        let set = dirty(at(15, 15), 7.5, 0.0, 0.4);
        assert_eq!(set.len(), 8, "{set:?}");
    }

    /// The note's span of a snow change alone: `[t + min(s) − c, t + max(s) + c]`.
    #[test]
    fn a_new_slab_spans_its_halo() {
        let layout = layout();
        let columns = columns(&layout, 32, (-8.0, 16.0));
        let (lo, hi) = slab_span(&columns, (3.0, 0.0), (3.0, 0.3)).expect("a top");
        assert!(
            (lo - 2.0).abs() < 1e-6 && (hi - 3.8).abs() < 1e-6,
            "{lo} {hi}"
        );
        // A column with no top before has only the new slab.
        let none = columns.no_top();
        let (lo, hi) = slab_span(&columns, (none, 0.0), (3.0, 0.3)).expect("a top");
        assert!(
            (lo - 2.0).abs() < 1e-6 && (hi - 3.8).abs() < 1e-6,
            "{lo} {hi}"
        );
        assert!(slab_span(&columns, (none, 0.0), (none, 0.0)).is_none());
    }

    #[test]
    fn no_changed_column_marks_nothing() {
        let set = DirtyChunks::default().into_ids();
        assert!(set.is_empty());
    }

    /// Validation 6, the hold rule: under the threshold nothing is allocated; over it only the
    /// over-threshold columns change; a column that swapped needs another `0.5 c`.
    #[test]
    fn hold_swap_is_per_column() {
        let layout = layout();
        let columns = columns(&layout, 4, (-8.0, 16.0));
        let tops = vec![3.0; columns.len()];
        let held = vec![0.0; columns.len()];
        assert!(hold_swap(&columns, &tops, &held, |_| 0.24).is_none());
        let swap = hold_swap(&columns, &tops, &held, |i| if i == 5 { 0.26 } else { 0.10 })
            .expect("a swap");
        assert_eq!(swap.changed, vec![(5, 0.0, 0.26)]);
        assert_eq!(swap.snow[5], 0.26);
        assert!(
            swap.snow
                .iter()
                .enumerate()
                .all(|(i, s)| i == 5 || *s == 0.0)
        );
        // The same readback again: no swap.
        assert!(
            hold_swap(&columns, &tops, &swap.snow, |i| if i == 5 {
                0.26
            } else {
                0.10
            })
            .is_none()
        );
        // 0.26 to 0.52 swaps (more than 0.25 since the last swap), 0.26 to 0.50 does not.
        assert!(
            hold_swap(&columns, &tops, &swap.snow, |_| 0.50).is_some_and(|s| s.changed.len() == 15)
        );
        let only_5 = |i: usize| if i == 5 { 0.50 } else { 0.10 };
        assert!(hold_swap(&columns, &tops, &swap.snow, only_5).is_none());
        let only_5 = |i: usize| if i == 5 { 0.52 } else { 0.10 };
        assert_eq!(
            hold_swap(&columns, &tops, &swap.snow, only_5).map(|s| s.changed),
            Some(vec![(5, 0.26, 0.52)])
        );
        // A column with no top never takes snow.
        let mut tops = tops;
        tops[3] = columns.no_top();
        let swap = hold_swap(&columns, &tops, &held, |_| 0.8).expect("a swap");
        assert_eq!(swap.snow[3], 0.0);
        assert_eq!(swap.changed.len(), 15);
    }

    #[test]
    fn a_chunk_box_selects_its_columns() {
        let layout = layout();
        let columns = columns(&layout, 32, (-8.0, 16.0));
        // Chunk (1, _, 0) spans x in [8, 16]: centres at 8.25 ..= 15.75 are columns 16 ..= 31.
        let (i0, i1, j0, j1) = columns
            .columns_in(&layout, ChunkId::new([1, 0, 0]))
            .expect("inside");
        assert_eq!((i0, i1, j0, j1), (16, 31, 0, 15));
        assert!(
            columns
                .columns_in(&layout, ChunkId::new([2, 0, 0]))
                .is_none()
        );
    }

    #[test]
    fn snow_columns_are_one_per_voxel_until_the_cap() {
        let (min, max) = ([0.0, -8.0, 0.0], [16.0, 16.0, 8.0]);
        let fine = snow_columns(min, max, 0.5, 1024);
        assert_eq!(
            (fine.grid.size, fine.grid.cell_m),
            (UVec2::new(32, 16), 0.5)
        );
        assert_eq!(
            (fine.grid.origin, fine.y_lo, fine.y_hi),
            (Vec2::ZERO, -8.0, 16.0)
        );
        // Past the cap the columns widen by doubling; the voxel cell stays.
        let capped = snow_columns(min, max, 0.5, 16);
        assert_eq!(
            (capped.grid.size, capped.grid.cell_m, capped.cell),
            (UVec2::new(16, 8), 1.0, 0.5)
        );
    }

    #[test]
    fn a_top_sheds_by_its_slope() {
        let columns = snow_columns([0.0; 3], [4.0, 8.0, 4.0], 1.0, 64);
        let flat = vec![2.0_f32; 16];
        assert_eq!(top_exposure(&columns, &flat, 5, [35.0, 55.0]), 1.0);
        // A 45° ramp along x sits halfway through the 35°–55° shed band.
        let ramp: Vec<f32> = (0..16).map(|i| (i % 4) as f32).collect();
        let half = top_exposure(&columns, &ramp, 5, [35.0, 55.0]);
        assert!((half - 0.5).abs() < 1e-5, "{half}");
        // A column with no top keeps nothing, and its neighbours read one-sided slopes.
        let mut holed = flat.clone();
        holed[6] = columns.no_top();
        assert_eq!(top_exposure(&columns, &holed, 6, [35.0, 55.0]), 0.0);
        assert_eq!(top_exposure(&columns, &holed, 5, [35.0, 55.0]), 1.0);
        // A flat walk beside a merlon 2 m higher keeps its snow: the step is a wall, not a slope.
        let mut merlon = flat.clone();
        merlon[6] = 4.0;
        assert_eq!(top_exposure(&columns, &merlon, 5, [35.0, 55.0]), 1.0);
        assert_eq!(top_exposure(&columns, &merlon, 6, [35.0, 55.0]), 1.0);
    }

    #[test]
    fn only_an_upright_evenly_scaled_volume_takes_snow() {
        use bevy::prelude::{GlobalTransform, Quat, Transform, Vec3};
        let upright = |t: Transform| upright_scale(&GlobalTransform::from(t));
        assert_eq!(upright(Transform::IDENTITY), Some(1.0));
        let turned = Transform::from_xyz(40.0, 2.0, -10.0)
            .with_rotation(Quat::from_rotation_y(0.7))
            .with_scale(Vec3::splat(2.0));
        let s = upright(turned).expect("turned about the vertical");
        assert!((s - 2.0).abs() < 1e-5, "{s}");
        assert_eq!(
            upright(Transform::from_rotation(Quat::from_rotation_x(0.2))),
            None
        );
        assert_eq!(
            upright(Transform::from_scale(Vec3::new(1.0, 2.0, 1.0))),
            None
        );
    }
}
