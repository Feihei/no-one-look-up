//! World generation plugin (R2).
//!
//! Layout graph + world grid + block-first path network + chunk streaming,
//! ported from the original M2/M3 algorithms (ADR-0002/0004). Deterministic
//! generation via `rand_chacha` seeded from the save seed. Filled in during
//! milestone R2.

use bevy::prelude::*;

pub struct WorldPlugin;

impl Plugin for WorldPlugin {
    fn build(&self, _app: &mut App) {
        // R2: layout_graph, world_grid, path_network, chunks.
    }
}
