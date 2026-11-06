//! Dev tool: simulate the player controller against the real generated world
//! and report whether walking can ever end up inside a solid block.
//!
//! Mirrors `player::walk_and_collide`'s motion + substep resolve (no Bevy App,
//! no window), so it runs in CI and on machines without a GPU.
//!
//! ```text
//! cargo run --example controller_probe
//! ```
use bevy::math::Vec2;

use no_one_looks_up::player::{ACCELERATION, MAX_WALK_SPEED};
use no_one_looks_up::world::collision::{CollisionIndex, PLAYER_RADIUS, is_solid};
use no_one_looks_up::world::path_network::generate_world;

/// Same fixed-step motion loop as the controller, returning the final point.
fn walk(index: &CollisionIndex, start: Vec2, yaw: f32, speed: f32, frames: u32) -> Vec2 {
    let dt = 1.0 / 60.0;
    let dir = Vec2::new(-yaw.sin(), -yaw.cos());
    let mut pos = start;
    let mut forward_speed = 0.0f32;
    for _ in 0..frames {
        // Accelerate towards the walking speed (controller: ACCELERATION).
        let target = speed;
        let delta = target - forward_speed;
        let max_delta = ACCELERATION * dt;
        forward_speed = if delta.abs() <= max_delta {
            target
        } else {
            forward_speed + delta.signum() * max_delta
        };

        let travel = forward_speed * dt;
        let steps = ((travel.abs() / (PLAYER_RADIUS * 0.5)).ceil() as u32).max(1);
        for _ in 0..steps {
            pos = index.resolve(pos + dir * (travel / steps as f32), PLAYER_RADIUS);
        }
    }
    pos
}

fn main() {
    let (grid, graph) = generate_world(0x4845_4144).expect("generation");
    let mut index = CollisionIndex::default();
    for row in 0..grid.height {
        for col in 0..grid.width {
            index.register_chunk(&graph, (col as i32, row as i32));
        }
    }

    let solid: Vec<_> = graph
        .blocks
        .iter()
        .filter(|b| is_solid(b.kind, b.is_outer))
        .collect();
    println!("{} solid blocks of {}", solid.len(), graph.blocks.len());

    // Sample starts on free street points near each solid block, then walk
    // straight at the block for 2 seconds at max speed.
    let mut tested = 0u32;
    let mut inside = 0u32;
    let mut farthest = 0.0f32;
    for block in &solid {
        let center = block.center();
        // Find the nearest free point by spiralling out.
        let mut start = None;
        'search: for ring in 1..120 {
            for k in 0..24 {
                let angle = k as f32 * std::f32::consts::TAU / 24.0;
                let p = center + Vec2::new(angle.cos(), angle.sin()) * (ring as f32 * 0.5);
                if !index.is_blocked(p) {
                    start = Some(p);
                    break 'search;
                }
            }
        }
        let Some(start) = start else { continue };
        let to_center = (center - start).normalize_or_zero();
        // Walk direction that would take the walker into the block: the body
        // yaw whose forward (-sin, -cos) equals `to_center`.
        let yaw = (-to_center.x).atan2(-to_center.y);
        let end = walk(&index, start, yaw, MAX_WALK_SPEED, 120);
        tested += 1;
        if index.is_blocked(end) {
            inside += 1;
            farthest = farthest.max((end - center).length());
        }
    }
    println!(
        "walked into {tested} blocks at {MAX_WALK_SPEED} m/s for 2 s: ended inside {inside} (worst {farthest:.2} m from centre)"
    );
    assert_eq!(inside, 0, "controller let the player end up inside a block");
}
