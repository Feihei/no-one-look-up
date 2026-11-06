//! Dev tool: walk the generated world on a straight line and report where the
//! collision resolver stops the walker.
//!
//! This exercises the same code path as the in-game debug walk (R3.3): sample
//! the collision index along a ray and print the first blocked point. Used to
//! confirm that streets are walkable and blocks are not, without a window.
//!
//! ```text
//! cargo run --example collision_probe
//! ```
use bevy::math::Vec2;

use no_one_looks_up::world::collision::{CollisionIndex, PLAYER_RADIUS, is_solid};
use no_one_looks_up::world::layout_graph::LayoutGraph;
use no_one_looks_up::world::path_network::generate_world;
use no_one_looks_up::world::world_grid::WorldGrid;

fn index_world(graph: &LayoutGraph, grid: &WorldGrid) -> CollisionIndex {
    let mut index = CollisionIndex::default();
    for row in 0..grid.height {
        for col in 0..grid.width {
            index.register_chunk(graph, (col as i32, row as i32));
        }
    }
    index
}

fn main() {
    let (grid, graph) = generate_world(0x4845_4144).expect("generation");
    let index = index_world(&graph, &grid);

    let solid = graph
        .blocks
        .iter()
        .filter(|b| is_solid(b.kind, b.is_outer))
        .count();
    println!(
        "blocks {} (solid {solid}), indexed prisms {}",
        graph.blocks.len(),
        index.prism_count()
    );

    // Sample a grid of points: how much of the map is walkable?
    let step = 5.0f32;
    let world = no_one_looks_up::world::WORLD_SIZE;
    let (mut free, mut blocked) = (0u32, 0u32);
    let mut y = 0.0;
    while y < world {
        let mut x = 0.0;
        while x < world {
            if index.is_blocked(Vec2::new(x, y)) {
                blocked += 1;
            } else {
                free += 1;
            }
            x += step;
        }
        y += step;
    }
    let total = free + blocked;
    println!(
        "walkable {:.1}% of sampled points ({free} free / {total})",
        free as f32 / total as f32 * 100.0
    );

    // Walk a straight line along a street: start at the middle of the map and
    // walk east until the resolver stops advancing.
    let mut pos = Vec2::splat(no_one_looks_up::world::world_center());
    // Nudge onto the nearest street first: if the centre is inside a block,
    // resolve() pushes it out.
    pos = index.resolve(pos, PLAYER_RADIUS);
    println!("start {pos:?}");
    let dir = Vec2::new(1.0, 0.0);
    let mut travelled = 0.0f32;
    let mut blocked_at = None;
    for _ in 0..4000 {
        let next = pos + dir * 0.25;
        let resolved = index.resolve(next, PLAYER_RADIUS);
        // Progress only counts if the resolver let us move forward.
        if (resolved - pos).dot(dir) < 0.05 {
            blocked_at = Some(pos);
            break;
        }
        travelled += (resolved - pos).length();
        pos = resolved;
    }
    match blocked_at {
        Some(p) => println!(
            "walker stopped after {travelled:.1} m at ({:.1}, {:.1})",
            p.x, p.y
        ),
        None => println!("walker travelled {travelled:.1} m without stopping"),
    }
}
