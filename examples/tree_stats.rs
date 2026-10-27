//! Diagnostic: procedural tree skeleton stats (R3.9 step 1).
//!
//! Run with `cargo run --example tree_stats`. Prints, per species, the
//! measured crown height/width, branch and leaf-cluster counts and the
//! instancing cost if every branch becomes one low-poly cylinder and every
//! cluster one 20-face icosphere. Numbers, not vibes: a profile that drifts
//! wide or explodes in branch count shows up here before anyone runs the game.

use heads_up_bevy::world::tree_shape::{TREE_PROFILES, grow_tree};

/// Faces per branch proxy at 6 sides x 1 ring.
const BRANCH_FACES: usize = 12;
/// Faces per 20-face icosphere.
const BLOB_FACES: usize = 20;

fn main() {
    println!(
        "{:<8} {:>2} {:>7} {:>7} {:>7} {:>6} {:>6} {:>8}",
        "species", "id", "height", "width", "branches", "leaves", "tris", "lvl"
    );
    for (species, profile) in TREE_PROFILES.iter().enumerate() {
        let mut max_level = 0u8;
        let mut totals = (0usize, 0usize, 0usize);
        for variant in 0..4u64 {
            let tree = grow_tree(0x4845_4144, species as u8, variant);
            let tris = tree.branches.len() * BRANCH_FACES + tree.foliage.len() * BLOB_FACES;
            for b in &tree.branches {
                max_level = max_level.max(b.level);
            }
            totals.0 += tree.branches.len();
            totals.1 += tree.foliage.len();
            totals.2 += tris;
            println!(
                "{:<8} {:>2} {:>6.1}m {:>6.1}m {:>7} {:>6} {:>8} {:>6}",
                if variant == 0 { "profile" } else { "" },
                species,
                tree.crown_height(),
                tree.crown_width(),
                tree.branches.len(),
                tree.foliage.len(),
                tris,
                max_level
            );
        }
        println!(
            "{:<8} {:>2} {:>6} {:>6} {:>7} {:>6} {:>8}   target {:.1}m x {:.1}m, <=400 br",
            "avg",
            species,
            "",
            "",
            totals.0 / 4,
            totals.1 / 4,
            totals.2 / 4,
            profile.height,
            profile.width
        );
    }
    println!("\nproxy tris = (branches x {BRANCH_FACES}) + (clusters x {BLOB_FACES})");
}
