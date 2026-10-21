//! `WorldGrid`: the 2D cell grid produced by generation (ADR-0002/0004).
//!
//! Every cell of the `GRID_CELLS x GRID_CELLS` world records its owning block
//! plus two derived flags used for fast gameplay queries and save
//! serialization:
//!
//! - `is_road`: the cell borders at least one street edge (blocks are solid;
//!   the negative space between blocks is the walkable street).
//! - `is_junction`: at least one corner vertex of the cell hosts a graph node
//!   (junction or corner; see `layout_graph::NodeKind`).
//!
//! The grid is serde-serializable: saves (R4) store the grid rather than the
//! polygon-heavy layout graph, at a fraction of the size.

use bevy::ecs::prelude::Resource;
use bevy::math::Vec2;
use serde::{Deserialize, Serialize};

use crate::core::error::{HeadsUpError, HeadsUpResult};

/// Per-cell record. `block_id` indexes into `LayoutGraph::blocks`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CellInfo {
    pub block_id: u32,
    pub is_road: bool,
    pub is_junction: bool,
}

/// Row-major 2D grid of [`CellInfo`]s.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Resource)]
pub struct WorldGrid {
    pub width: u32,
    pub height: u32,
    /// Row-major: index = `row * width + col`. Row 0 is the northernmost band.
    pub cells: Vec<CellInfo>,
    /// Perturbed lattice node positions (R2.9), row-major over
    /// `(width + 1) x (height + 1)` nodes: index = `row * (width + 1) + col`.
    /// These are the street centerline vertices every block polygon is built
    /// from, so placement and collision can reuse them directly.
    #[serde(default)]
    pub node_positions: Vec<Vec2>,
}

impl WorldGrid {
    /// Position of a lattice node (`0..=height`, `0..=width`).
    pub fn node(&self, row: u32, col: u32) -> Option<Vec2> {
        if row > self.height || col > self.width || self.node_positions.is_empty() {
            return None;
        }
        self.node_positions
            .get((row * (self.width + 1) + col) as usize)
            .copied()
    }
}

impl WorldGrid {
    /// Creates the grid with every cell assigned to `block_id` and no flags.
    pub fn filled(width: u32, height: u32, block_id: u32) -> Self {
        Self {
            width,
            height,
            cells: vec![
                CellInfo {
                    block_id,
                    is_road: false,
                    is_junction: false,
                };
                (width * height) as usize
            ],
            node_positions: Vec::new(),
        }
    }

    /// Fast cell lookup. Row-major indexing; row 0 is the northernmost band.
    pub fn get_cell(&self, row: u32, col: u32) -> HeadsUpResult<&CellInfo> {
        if row >= self.height || col >= self.width {
            return Err(HeadsUpError::Generation(format!(
                "cell ({row}, {col}) outside the {}x{} grid",
                self.width, self.height
            )));
        }
        Ok(&self.cells[(row * self.width + col) as usize])
    }

    /// Mutable cell access used by the generator to fill derived flags.
    pub fn get_cell_mut(&mut self, row: u32, col: u32) -> HeadsUpResult<&mut CellInfo> {
        if row >= self.height || col >= self.width {
            return Err(HeadsUpError::Generation(format!(
                "cell ({row}, {col}) outside the {}x{} grid",
                self.width, self.height
            )));
        }
        Ok(&mut self.cells[(row * self.width + col) as usize])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn filled_grid_has_uniform_block_ids() {
        let grid = WorldGrid::filled(4, 3, 7);
        assert_eq!(grid.cells.len(), 12);
        for row in 0..3u32 {
            for col in 0..4u32 {
                let cell = grid.get_cell(row, col).expect("in bounds");
                assert_eq!(cell.block_id, 7);
                assert!(!cell.is_road && !cell.is_junction);
            }
        }
    }

    #[test]
    fn out_of_bounds_lookup_is_a_generation_error() {
        let grid = WorldGrid::filled(2, 2, 0);
        let err = grid.get_cell(2, 0).unwrap_err().to_string();
        assert!(err.contains("outside"), "unexpected error: {err}");
    }

    #[test]
    fn grid_round_trips_through_serde() {
        let mut grid = WorldGrid::filled(3, 3, 0);
        grid.get_cell_mut(1, 1).expect("in bounds").is_road = true;
        let json = serde_json::to_string(&grid).expect("serialize");
        let restored: WorldGrid = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(grid, restored);
        assert!(restored.get_cell(1, 1).expect("in bounds").is_road);
    }
}
