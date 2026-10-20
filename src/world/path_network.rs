//! Block-first world generation (ADR-0002/0004, ported from the original
//! M2/M3 algorithms).
//!
//! Algorithm: the map is partitioned into rectangular blocks (merged cell
//! groups, block-first — blocks are the positive shape, streets are the
//! negative space between them). Street unit edges are the grid lines between
//! blocks, minus the lines between adjacent outer-band cells (the enclosure
//! band is never crossed by streets). Full partitioning plus the band rule
//! guarantees every street forms a cycle: with a complete rectangle
//! partition, an interior lattice vertex cannot have street degree 1, and
//! border vertices only see band-internal (excluded) edges — so there are no
//! dead ends. Both properties are asserted by unit tests.
//!
//! Determinism: a single `rand_chacha::StdRng::seed_from_u64(seed)` drives
//! every decision in a fixed cell-scan order; `thread_rng` never enters the
//! generation path. Same seed → bit-exact world (unit tested).

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use bevy::ecs::prelude::Resource;

use crate::core::error::{HeadsUpError, HeadsUpResult};
use crate::world::geometry::{cell_rect_polygon, inset_polygon_clip};
use crate::world::layout_graph::{
    BlockType, LayoutBlock, LayoutGraph, NodeKind, NodeVertex, StreetUnitEdge,
};
use crate::world::world_grid::WorldGrid;
use crate::world::{
    FOREST_EDGE_MAX_DISTANCE_CELLS, GRID_CELLS, OUTER_BAND_CELLS, STREET_WIDTH_MAX,
    STREET_WIDTH_MIN,
};

/// The save seed driving generation (ADR-0002: one save = one seed = one
/// map). Inserted before adding the world plugin (R4 menu provides it);
/// defaults to [`DEFAULT_SEED`] in dev builds until then.
#[derive(Resource, Clone, Copy, Debug, PartialEq, Eq)]
pub struct HeadsUpSeed(pub u64);

/// Type selection weights (ADR-0001: 老城区多数, 公寓少量, 厂房极少,
/// 广场/树林少量). Cumulative thresholds over a single `[0, 1)` roll.
const P_OLD_CITY: f32 = 0.50;
const P_APARTMENT: f32 = 0.68;
const P_FACTORY: f32 = 0.74;
const P_PLAZA: f32 = 0.87;

/// Generates the complete world layout for `seed`.
///
/// Runs at startup in well under 1 s (pure CPU over a 40x40 grid, no mesh
/// data). Returns the serde-friendly [`WorldGrid`] plus the [`LayoutGraph`].
pub fn generate_world(seed: u64) -> HeadsUpResult<(WorldGrid, LayoutGraph)> {
    let mut rng = StdRng::seed_from_u64(seed);
    let (blocks, mut grid) = partition_blocks(&mut rng)?;
    let unit_edges = street_unit_edges(&grid, &blocks);
    let node_vertices = node_vertices(&unit_edges);
    mark_grid_flags(&mut grid, &unit_edges, &node_vertices)?;
    let graph = LayoutGraph::from_generation(blocks, &unit_edges, &node_vertices);
    Ok((grid, graph))
}

/// Partitions the grid into rectangular blocks (block-first merging).
///
/// Returns the block records and the grid with block ids assigned. Cells are
/// scanned row-major; the outermost band is consumed first (row-major reaches
/// it before any interior carve), so blocks carved from interior cells never
/// touch the map border.
fn partition_blocks(rng: &mut StdRng) -> HeadsUpResult<(Vec<LayoutBlock>, WorldGrid)> {
    let width = GRID_CELLS;
    let height = GRID_CELLS;
    let mut grid = WorldGrid::filled(width, height, u32::MAX);
    let mut assigned: Vec<Option<u32>> = vec![None; (width * height) as usize];
    let mut blocks: Vec<LayoutBlock> = Vec::new();

    for row in 0..height {
        for col in 0..width {
            if assigned[(row * width + col) as usize].is_some() {
                continue;
            }
            let distance_to_border = row.min(col).min(height - 1 - row).min(width - 1 - col);
            let is_outer = distance_to_border < OUTER_BAND_CELLS;

            // Type roll: one [0,1) draw for interior cells, one bool for the
            // band (50/50 OLD_CITY/APARTMENT per ADR-0004). Every draw stays
            // deterministic in the fixed scan order.
            let (kind, merge) = if is_outer {
                let kind = if rng.random_bool(0.5) {
                    BlockType::OldCity
                } else {
                    BlockType::Apartment
                };
                (kind, 1)
            } else {
                let roll: f32 = rng.random();
                let mut kind = if roll < P_OLD_CITY {
                    BlockType::OldCity
                } else if roll < P_APARTMENT {
                    BlockType::Apartment
                } else if roll < P_FACTORY {
                    BlockType::Factory
                } else if roll < P_PLAZA {
                    BlockType::Plaza
                } else {
                    BlockType::Forest
                };
                // FOREST is the edge transition zone (ADR-0004): interior
                // rolls far from the border fall back to OLD_CITY.
                if kind == BlockType::Forest && distance_to_border > FOREST_EDGE_MAX_DISTANCE_CELLS
                {
                    kind = BlockType::OldCity;
                }
                (kind, merge_size(rng, kind))
            };

            // Interior merges must fit entirely inside the non-band region;
            // otherwise fall back to a 1x1 OLD_CITY (scattered small blocks
            // near the band-adjacent ring, keeps the band a clean enclosure).
            let merge_fits = is_outer
                || (row + merge <= height - OUTER_BAND_CELLS
                    && col + merge <= width - OUTER_BAND_CELLS);
            let (kind, merge) = if merge_fits {
                (kind, merge)
            } else {
                (BlockType::OldCity, 1)
            };

            // Carve the (clamped) rectangle and record the block. All target
            // cells must be free: earlier carves can reach into this
            // rectangle; if so, fall back to a 1x1 OLD_CITY.
            let rows = row..(row + merge).min(height);
            let cols = col..(col + merge).min(width);
            let all_free = rows.clone().all(|r| {
                cols.clone()
                    .all(|c| assigned[(r * width + c) as usize].is_none())
            });
            let (kind, rows, cols) = if all_free {
                (kind, rows, cols)
            } else {
                (BlockType::OldCity, row..row + 1, col..col + 1)
            };
            let block_id = blocks.len() as u32;
            let street_width = rng.random_range(STREET_WIDTH_MIN..=STREET_WIDTH_MAX);
            let mut cells = Vec::with_capacity(rows.len() * cols.len());
            for r in rows.clone() {
                for c in cols.clone() {
                    assigned[(r * width + c) as usize] = Some(block_id);
                    cells.push((r, c));
                }
            }
            let rect = cell_rect_polygon(rows, cols);
            let polygon = inset_polygon_clip(&rect, street_width * 0.5).ok_or_else(|| {
                HeadsUpError::Generation(format!(
                    "block {block_id} inset collapsed (width {street_width})"
                ))
            })?;
            blocks.push(LayoutBlock {
                id: block_id,
                kind,
                cells,
                street_width,
                is_outer,
                polygon,
            });
        }
    }

    for row in 0..height {
        for col in 0..width {
            let cell = grid.get_cell_mut(row, col)?;
            cell.block_id =
                assigned[(row * width + col) as usize].expect("full partition assigns every cell");
        }
    }
    Ok((blocks, grid))
}

/// Merge size for a block type (ADR-0004 table): OLD_CITY/PLAZA/FOREST 1x1,
/// APARTMENT 2x2 or 3x3, FACTORY 4x4 or 5x5 (random).
fn merge_size(rng: &mut StdRng, kind: BlockType) -> u32 {
    match kind {
        BlockType::Apartment => {
            if rng.random_bool(0.5) {
                2
            } else {
                3
            }
        }
        BlockType::Factory => {
            if rng.random_bool(0.5) {
                4
            } else {
                5
            }
        }
        _ => 1,
    }
}

/// Street unit edges: grid lines between cells of different blocks, excluding
/// lines between two outer-band cells (the enclosure band has no internal
/// streets — the band rule that guarantees no dead ends).
///
/// Edge width is the average of the flanking blocks' street widths (each
/// block insets by width / 2 on its own side of the shared centerline).
fn street_unit_edges(grid: &WorldGrid, blocks: &[LayoutBlock]) -> Vec<StreetUnitEdge> {
    let width = grid.width;
    let height = grid.height;
    let block_of =
        |row: u32, col: u32| grid.get_cell(row, col).expect("in bounds").block_id as usize;
    let is_band = |row: u32, col: u32| {
        row.min(col).min(height - 1 - row).min(width - 1 - col) < OUTER_BAND_CELLS
    };
    let mut edges = Vec::new();
    for row in 0..height {
        for col in 0..width {
            for (nr, nc) in [
                (row, col + 1), // right neighbor
                (row + 1, col), // bottom neighbor
            ] {
                if nr >= height || nc >= width {
                    continue;
                }
                let a = block_of(row, col);
                let b = block_of(nr, nc);
                if a == b {
                    continue; // Internal road deleted (merged block interior).
                }
                if is_band(row, col) && is_band(nr, nc) {
                    continue; // Enclosure band: no internal streets.
                }
                let wa = blocks[a].street_width;
                let wb = blocks[b].street_width;
                let vertices = if nr == row {
                    // Side-by-side cells: vertical line at col = max.
                    let c = col.max(nc);
                    ((row, c), (row + 1, c))
                } else {
                    // Stacked cells: horizontal line at row = max.
                    let r = row.max(nr);
                    ((r, col), (r, col + 1))
                };
                edges.push(StreetUnitEdge {
                    vertices,
                    width: (wa + wb) * 0.5,
                });
            }
        }
    }
    edges
}

/// Street degree per lattice vertex, from the unit edges.
fn vertex_degrees(unit_edges: &[StreetUnitEdge]) -> Vec<Vec<u8>> {
    let mut degrees = vec![vec![0u8; GRID_CELLS as usize + 1]; GRID_CELLS as usize + 1];
    for edge in unit_edges {
        let ((r0, c0), (r1, c1)) = edge.vertices;
        degrees[r0 as usize][c0 as usize] += 1;
        degrees[r1 as usize][c1 as usize] += 1;
    }
    degrees
}

/// Lattice vertices hosting graph nodes: junctions (degree >= 3) and corners
/// (degree 2 with perpendicular edges). Degree-2 straight vertices are
/// mid-run (no node); degree 0/1 are impossible in a valid partition.
fn node_vertices(unit_edges: &[StreetUnitEdge]) -> Vec<NodeVertex> {
    let degrees = vertex_degrees(unit_edges);
    let mut nodes = Vec::new();
    for (row, row_degrees) in degrees.iter().enumerate() {
        for (col, degree) in row_degrees.iter().enumerate() {
            let kind = match *degree {
                3.. => NodeKind::Junction,
                2 => {
                    // Corner when the two edges are perpendicular.
                    let r = row as u32;
                    let c = col as u32;
                    let vertical = unit_edges.iter().any(|e| {
                        let ((er0, ec0), (er1, ec1)) = e.vertices;
                        er0 == er1 && (ec0 == c || ec1 == c) && (er0 == r || er1 == r)
                    });
                    let horizontal = unit_edges.iter().any(|e| {
                        let ((er0, ec0), (er1, ec1)) = e.vertices;
                        ec0 == ec1 && (er0 == r || er1 == r) && (ec0 == c || ec1 == c)
                    });
                    if vertical && horizontal {
                        NodeKind::Corner
                    } else {
                        continue; // Straight mid-run vertex: no node.
                    }
                }
                _ => continue,
            };
            nodes.push(NodeVertex {
                row: row as u32,
                col: col as u32,
                kind,
            });
        }
    }
    nodes
}

/// Fills the derived `is_road`/`is_junction` flags on the grid.
///
/// - `is_road`: the cell flanks at least one street unit edge (street is the
///   negative space between blocks; flagged cells border walkable street).
/// - `is_junction`: a corner vertex of the cell hosts a graph node.
fn mark_grid_flags(
    grid: &mut WorldGrid,
    unit_edges: &[StreetUnitEdge],
    node_vertices: &[NodeVertex],
) -> HeadsUpResult<()> {
    for edge in unit_edges {
        let ((r0, c0), (r1, c1)) = edge.vertices;
        // Vertical unit edge ((r, c)..(r+1, c)): the line between columns
        // c-1 and c, spanning rows r..r+1. Flanking cells: (r, c-1), (r, c).
        if c0 == c1 && r0 != r1 {
            let col = c0;
            if col > 0 {
                grid.get_cell_mut(r0, col - 1)?.is_road = true;
            }
            if col < grid.width {
                grid.get_cell_mut(r0, col)?.is_road = true;
            }
        }
        // Horizontal unit edge ((r, c)..(r, c+1)): the line between rows
        // r-1 and r, spanning cols c..c+1. Flanking cells: (r-1, c), (r, c).
        if r0 == r1 && c0 != c1 {
            let row = r0;
            if row > 0 {
                grid.get_cell_mut(row - 1, c0)?.is_road = true;
            }
            if row < grid.height {
                grid.get_cell_mut(row, c0)?.is_road = true;
            }
        }
    }
    for node in node_vertices {
        // Any cell with this vertex as a corner.
        for r in [node.row.saturating_sub(1), node.row] {
            for c in [node.col.saturating_sub(1), node.col] {
                if r < grid.height && c < grid.width {
                    grid.get_cell_mut(r, c)?.is_junction = true;
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference world for the shared assertions.
    fn world(seed: u64) -> (WorldGrid, LayoutGraph) {
        generate_world(seed).expect("generation succeeds")
    }

    #[test]
    fn same_seed_is_bit_exact() {
        let (grid_a, graph_a) = world(0x4845_4144);
        let (grid_b, graph_b) = world(0x4845_4144);
        assert_eq!(grid_a, grid_b, "same seed must regenerate the exact grid");
        assert_eq!(
            graph_a, graph_b,
            "same seed must regenerate the exact graph"
        );
    }

    #[test]
    fn different_seeds_produce_different_maps() {
        let (grid_a, _) = world(1);
        let (grid_b, _) = world(2);
        assert_ne!(grid_a, grid_b);
    }

    #[test]
    fn partition_is_complete_and_valid() {
        let (grid, graph) = world(7);
        assert_eq!(grid.width, GRID_CELLS);
        assert_eq!(grid.cells.len(), (GRID_CELLS * GRID_CELLS) as usize);
        // Every cell references a valid block.
        for cell in &grid.cells {
            assert!(cell.block_id < graph.blocks.len() as u32);
        }
        // Blocks own disjoint, contiguous cell sets.
        for block in &graph.blocks {
            assert!(!block.cells.is_empty());
            let rows: Vec<u32> = block.cells.iter().map(|c| c.0).collect();
            let cols: Vec<u32> = block.cells.iter().map(|c| c.1).collect();
            let (r0, r1) = (rows.iter().min(), rows.iter().max());
            let (c0, c1) = (cols.iter().min(), cols.iter().max());
            let area = ((r1.unwrap() - r0.unwrap() + 1) * (c1.unwrap() - c0.unwrap() + 1)) as usize;
            assert_eq!(area, block.cells.len(), "block {} is a rectangle", block.id);
        }
    }

    #[test]
    fn streets_have_no_dead_ends() {
        for seed in [1u64, 7, 0x4845_4144, 123_456_789] {
            let (grid, graph) = world(seed);
            let unit_edges = street_unit_edges(&grid, &graph.blocks);
            let degrees = vertex_degrees(&unit_edges);
            for (r, row) in degrees.iter().enumerate() {
                for (c, degree) in row.iter().enumerate() {
                    assert_ne!(
                        *degree, 1,
                        "dead end at lattice vertex ({r}, {c}), seed {seed}"
                    );
                }
            }
        }
    }

    #[test]
    fn outer_band_encloses_the_world() {
        let (grid, graph) = world(7);
        let width = grid.width;
        let height = grid.height;
        // is_outer must exactly match the outermost band.
        for row in 0..height {
            for col in 0..width {
                let is_band = row.min(col).min(height - 1 - row).min(width - 1 - col) == 0;
                let cell = grid.get_cell(row, col).expect("in bounds");
                let block = &graph.blocks[cell.block_id as usize];
                assert_eq!(
                    block.is_outer, is_band,
                    "is_outer must match the band at ({row}, {col})"
                );
            }
        }
        // Streets never touch the map border: every border lattice vertex
        // has street degree 0 (the enclosure band is not crossed by streets).
        let unit_edges = street_unit_edges(&grid, &graph.blocks);
        let degrees = vertex_degrees(&unit_edges);
        let n = GRID_CELLS as usize;
        for (c, col) in degrees[0].iter().enumerate().take(n + 1) {
            assert_eq!(*col, 0, "street touches north border at col {c}");
            assert_eq!(degrees[n][c], 0, "street touches south border at col {c}");
        }
        for (r, row) in degrees.iter().enumerate().take(n + 1) {
            assert_eq!(row[0], 0, "street touches west border at row {r}");
            assert_eq!(row[n], 0, "street touches east border at row {r}");
        }
    }

    #[test]
    fn inset_polygons_never_overlap() {
        // OVERLAPS=0: every pair of block polygons that could touch must not
        // intersect (ADR-0004 geometric guarantee, kept as a unit test).
        for seed in [1u64, 7, 0x4845_4144, 123_456_789] {
            let (_, graph) = world(seed);
            for i in 0..graph.blocks.len() {
                for j in (i + 1)..graph.blocks.len() {
                    let (a, b) = (&graph.blocks[i], &graph.blocks[j]);
                    if !blocks_adjacent(a, b) {
                        continue;
                    }
                    assert!(
                        !crate::world::geometry::convex_polygons_overlap(&a.polygon, &b.polygon),
                        "blocks {} and {} overlap (seed {seed})",
                        a.id,
                        b.id
                    );
                }
            }
        }
    }

    /// True if the two blocks own adjacent cells (they could touch across a
    /// street or, for band blocks, sit flush together).
    fn blocks_adjacent(a: &LayoutBlock, b: &LayoutBlock) -> bool {
        let owned: std::collections::HashSet<(u32, u32)> = a.cells.iter().copied().collect();
        for &(r, c) in &b.cells {
            for n in [
                (r.wrapping_sub(1), c),
                (r + 1, c),
                (r, c.wrapping_sub(1)),
                (r, c + 1),
            ] {
                if owned.contains(&n) {
                    return true;
                }
            }
        }
        false
    }

    #[test]
    fn street_widths_stay_in_range() {
        let (_, graph) = world(7);
        for block in &graph.blocks {
            assert!(
                (STREET_WIDTH_MIN..=STREET_WIDTH_MAX).contains(&block.street_width),
                "block {} width {} out of range",
                block.id,
                block.street_width
            );
        }
        for edge in &graph.edges {
            assert!(
                (STREET_WIDTH_MIN..=STREET_WIDTH_MAX).contains(&edge.width),
                "street edge width {} out of range",
                edge.width
            );
        }
    }

    #[test]
    fn block_types_match_the_merge_table() {
        let (_, graph) = world(7);
        for block in &graph.blocks {
            let span = block.cells.len();
            match block.kind {
                BlockType::OldCity | BlockType::Plaza | BlockType::Forest => {
                    assert_eq!(span, 1, "{:?} blocks merge 1x1", block.kind);
                }
                BlockType::Apartment => {
                    // Interior: 2x2/3x3 (random). Band cells are forced 1x1
                    // (the enclosure band is one cell wide).
                    assert!(
                        block.is_outer || (4..=9).contains(&span),
                        "apartment merges 2x2/3x3, got {span}"
                    );
                }
                BlockType::Factory => {
                    // 4x4/5x5 (random); clamped near the opposite border can
                    // yield non-square merges (e.g. 3x4).
                    assert!(
                        (12..=25).contains(&span),
                        "factory merges 4x4/5x5, got {span}"
                    );
                }
            }
        }
        // The world contains a realistic mix (老城区多数, 厂房极少).
        let kinds: Vec<BlockType> = graph.blocks.iter().map(|b| b.kind).collect();
        assert!(kinds.contains(&BlockType::Apartment));
        assert!(kinds.contains(&BlockType::Plaza));
        assert!(kinds.contains(&BlockType::Forest));
        let old_city = kinds.iter().filter(|k| **k == BlockType::OldCity).count();
        let factory = kinds.iter().filter(|k| **k == BlockType::Factory).count();
        assert!(
            old_city > factory * 2,
            "old city blocks must dominate: {old_city} vs {factory} factories"
        );
    }

    #[test]
    fn road_flags_mark_street_adjacent_cells() {
        let (grid, graph) = world(7);
        // Every cell adjacent to a street must be flagged; band cells and
        // merged block interiors are not.
        let unit_edges = street_unit_edges(&grid, &graph.blocks);
        let mut expected = vec![false; grid.cells.len()];
        for edge in &unit_edges {
            let ((r0, c0), (r1, c1)) = edge.vertices;
            if c0 == c1 && r0 != r1 {
                // Vertical unit edge: flanking cells to the left/right.
                if c0 > 0 {
                    expected[(r0 * grid.width + c0 - 1) as usize] = true;
                }
                if c0 < grid.width {
                    expected[(r0 * grid.width + c0) as usize] = true;
                }
            } else {
                // Horizontal unit edge: flanking cells above/below.
                if r0 > 0 {
                    expected[((r0 - 1) * grid.width + c0) as usize] = true;
                }
                if r0 < grid.height {
                    expected[(r0 * grid.width + c0) as usize] = true;
                }
            }
            let _ = (r1, c1);
        }
        for (idx, expected_road) in expected.iter().enumerate() {
            assert_eq!(
                grid.cells[idx].is_road, *expected_road,
                "is_road mismatch at cell {idx}"
            );
        }
        let roads = grid.cells.iter().filter(|c| c.is_road).count();
        assert!(roads > 0, "world must have streets");
        let junctions = graph
            .nodes
            .iter()
            .filter(|n| n.kind == NodeKind::Junction)
            .count();
        assert!(junctions > 0, "street network must have junctions");
    }
}
