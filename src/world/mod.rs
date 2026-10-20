//! World generation (R2): layout graph, block-first path network, and chunk
//! streaming, ported from the original M2/M3 algorithms (ADR-0002/0004).
//!
//! Pipeline (ADR-0002):
//!
//! ```text
//! seed -> path_network::generate_world -> (WorldGrid, LayoutGraph)
//!              (startup, <1s, no 3D meshes)
//! player position -> chunks::ChunkManager -> spawn/despawn chunk groups
//! ```

pub mod chunks;
pub mod debug_viz;
pub mod geometry;
pub mod layout_graph;
pub mod path_network;
pub mod world_grid;

/// Base grid cell edge, in meters. The world is `GRID_CELLS` x `GRID_CELLS`
/// cells of this size; blocks are merged cell groups and streets run along the
/// grid lines between blocks (ADR-0004: block-first, streets are negative
/// space).
pub const CELL_SIZE: f32 = 50.0;

/// Grid cells per axis. 40 x 40 cells = 2 km x 2 km world.
pub const GRID_CELLS: u32 = 40;

/// Total world extent in meters (`CELL_SIZE * GRID_CELLS`).
pub const WORLD_SIZE: f32 = CELL_SIZE * GRID_CELLS as f32;

/// Outermost ring(s) of cells forced to be solid blocks (OLD_CITY/APARTMENT),
/// forming the enclosure wall around the world (ADR-0004).
pub const OUTER_BAND_CELLS: u32 = 1;

/// Per-block street half-width range. Each block stores a random street width
/// in `[MIN, MAX]` and insets by `width / 2`; adjacent blocks share the street
/// centerline, so the full street width between two blocks is
/// `(width_a + width_b) / 2` (ADR-0004).
pub const STREET_WIDTH_MIN: f32 = 6.0;
pub const STREET_WIDTH_MAX: f32 = 10.0;

/// Height of the invisible solid collision prism (R3). Defined here because
/// the layout graph is its input.
pub const BLOCK_SOLID_HEIGHT: f32 = 2.0;

/// FOREST blocks are edge transition zones (ADR-0004): only cells within this
/// many cells of the map border may roll the FOREST type.
pub const FOREST_EDGE_MAX_DISTANCE_CELLS: u32 = 8;

/// Chunk edge, in meters. Chunks align 1:1 with base grid cells
/// (2000 / 50 = 40 x 40 = 1600 chunks, ADR-0002).
pub const CHUNK_SIZE: f32 = 50.0;

/// Streaming radius: chunks loaded ahead of the walk direction (ADR-0002,
/// first-person view optimization).
pub const CHUNK_AHEAD: i32 = 4;
/// Streaming radius: chunks kept behind the walk direction (turn safety).
pub const CHUNK_BEHIND: i32 = 2;
/// Streaming radius: chunks kept to each side.
///
/// Active window: `(AHEAD + 0.5)` chunks ahead, `BEHIND + 0.5` behind,
/// `SIDE + 0.5` to each side (chunk-center based). Worst case (player at a
/// sub-chunk offset) the window covers 8 x 8 = 64 chunks; the ADR-0002
/// estimate was ~55 at peak — in range, and ground tiles are cheap for the
/// 2-4 GB VRAM budget (placement content is what R5's LOD budget targets).
pub const CHUNK_SIDE: i32 = 3;

#[cfg(test)]
mod tests {
    //! R2.8: scheduling property testing — the world generation + chunk
    //! streaming + debug visualization systems are registered in explicit
    //! `.before()`/`.after()` chains, so a schedule built in ANY registration
    //! order must initialize without scheduling ambiguities.

    use bevy::ecs::schedule::{LogLevel, Schedule, ScheduleBuildSettings};
    use bevy::prelude::*;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    use crate::world::chunks::{WorldSystems, debug_walk_camera, update_chunk_streaming};
    use crate::world::debug_viz::{
        DebugVizEnabled, draw_debug_viz, spawn_block_labels, toggle_debug_viz,
    };

    #[test]
    fn world_systems_have_no_scheduling_ambiguities_in_any_order() {
        for seed in 0..16u64 {
            // Fisher–Yates shuffle of the registration order, seeded.
            let mut order = [0usize, 1, 2, 3, 4];
            let mut rng = StdRng::seed_from_u64(seed);
            for i in (1..order.len()).rev() {
                let j = rng.random_range(..=i);
                order.swap(i, j);
            }

            let mut schedule = Schedule::default();
            schedule.set_build_settings(ScheduleBuildSettings {
                ambiguity_detection: LogLevel::Error,
                ..default()
            });
            // Register the systems in the shuffled order, with the explicit
            // ordering contract attached to the systems themselves:
            // spawn_block_labels -> toggle_debug_viz -> draw_debug_viz and
            // debug_walk_camera -> update_chunk_streaming -> draw_debug_viz.
            for index in order {
                match index {
                    0 => schedule.add_systems(debug_walk_camera),
                    1 => schedule.add_systems(
                        update_chunk_streaming
                            .in_set(WorldSystems::Streaming)
                            .after(debug_walk_camera),
                    ),
                    2 => schedule.add_systems(spawn_block_labels),
                    3 => schedule.add_systems(toggle_debug_viz.after(spawn_block_labels)),
                    4 => schedule.add_systems(
                        draw_debug_viz
                            .after(toggle_debug_viz)
                            .after(WorldSystems::Streaming),
                    ),
                    _ => unreachable!(),
                };
            }
            let mut world = World::new();
            world.init_resource::<ButtonInput<KeyCode>>();
            world.init_resource::<DebugVizEnabled>();
            if let Err(err) = schedule.initialize(&mut world) {
                panic!("seed {seed}: schedule order {order:?} failed to build: {err:?}");
            }
        }
    }
}
