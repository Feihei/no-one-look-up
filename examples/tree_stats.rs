//! Diagnostic: procedural tree skeleton + baked-foliage stats (R3.9/R3.10).
//!
//! Run with `cargo run --example tree_stats`. Prints, per species, the
//! measured crown height/width, branch and leaf-cluster counts, and — for the
//! BAKED foliage — the card count, mean card size, the crown's sky-opening
//! rate (a side-view alpha-tested rasterisation, i.e. how much sky a crown
//! lets through) and the baked triangle count.
//!
//! Numbers, not vibes: the fragmentation direction (the locust's compound
//! leaves read as the finest silhouette) is judged by the opening rate and
//! card counts here, not by eyeballing a shrunk gallery sheet.

use heads_up_bevy::world::leaf_shape::{ATLAS_HEIGHT, bake_species_atlas};
use heads_up_bevy::world::tree_mesh::{TreeLod, TreeMeshKey, bake_tree_mesh_lod};
use heads_up_bevy::world::tree_shape::{TREE_PROFILES, grow_tree};

/// Seed variants per species averaged.
const VARIANTS: u64 = 4;
/// Side-view rasterisation resolution (crown bbox mapped onto this width).
const RASTER: u32 = 480;

fn main() {
    println!(
        "{:<8} {:>2} {:>7} {:>7} {:>7} {:>6} {:>8}",
        "species", "id", "height", "width", "branches", "clusters", "proxy tris"
    );
    for (species, profile) in TREE_PROFILES.iter().enumerate() {
        let mut totals = (0usize, 0usize, 0usize);
        for variant in 0..VARIANTS {
            let tree = grow_tree(0x4845_4144, species as u8, variant);
            let tris = tree.branches.len() * 12 + tree.foliage.len() * 20;
            totals.0 += tree.branches.len();
            totals.1 += tree.foliage.len();
            totals.2 += tris;
            if variant == 0 {
                println!(
                    "{:<8} {:>2} {:>6.1}m {:>6.1}m {:>7} {:>6} {:>8}",
                    "profile",
                    species,
                    tree.crown_height(),
                    tree.crown_width(),
                    tree.branches.len(),
                    tree.foliage.len(),
                    tris
                );
            }
        }
        println!(
            "{:<8} {:>2} {:>6} {:>6} {:>7} {:>6} {:>8}   target {:.1}m x {:.1}m",
            "avg",
            species,
            "",
            "",
            totals.0 / VARIANTS as usize,
            totals.1 / VARIANTS as usize,
            totals.2 / VARIANTS as usize,
            profile.height,
            profile.width
        );
    }

    println!("\nbaked foliage (variant 0, Near LOD):");
    println!(
        "{:<8} {:>8} {:>10} {:>10} {:>10} {:>8}",
        "species", "cards", "card w (m)", "card h (m)", "opening", "tris"
    );
    let atlas = bake_species_atlas(ATLAS_HEIGHT);
    for species in 0..TREE_PROFILES.len() as u8 {
        let parts = bake_tree_mesh_lod(
            TreeMeshKey {
                species,
                variant: 0,
            },
            TreeLod::Near,
        );
        let positions = attribute_positions(&parts.leaves);
        let Some(uvs) = attribute_uvs(&parts.leaves) else {
            println!("species {species}: no leaf UVs");
            continue;
        };
        let cards = positions.len() / 4;
        let (mean_w, mean_h) = quad_sizes(&positions);
        let opening = sky_opening(&positions, &uvs, &atlas);
        let tris = parts.wood.count_vertices() / 3 + parts.leaves.count_vertices() / 3;
        println!(
            "{:<8} {:>8} {:>10.2} {:>10.2} {:>10.3} {:>8}",
            if species == 0 {
                "aspen"
            } else if species == 1 {
                "street"
            } else {
                "broadleaf"
            },
            cards,
            mean_w,
            mean_h,
            opening,
            tris
        );
    }
    println!(
        "\nopening = fraction of the crown's side-view bbox no leaf covers\n\
         proxy tris = wood vertices/3 + leaf vertices/3 (quads under-count)"
    );
}

fn attribute_positions(mesh: &bevy::mesh::Mesh) -> Vec<[f32; 3]> {
    match mesh.attribute(bevy::mesh::Mesh::ATTRIBUTE_POSITION) {
        Some(bevy::mesh::VertexAttributeValues::Float32x3(v)) => v.clone(),
        _ => Vec::new(),
    }
}

fn attribute_uvs(mesh: &bevy::mesh::Mesh) -> Option<Vec<[f32; 2]>> {
    match mesh.attribute(bevy::mesh::Mesh::ATTRIBUTE_UV_0) {
        Some(bevy::mesh::VertexAttributeValues::Float32x2(v)) => Some(v.clone()),
        _ => None,
    }
}

/// Mean width/height of the leaf quads, in metres, straight from the corners.
fn quad_sizes(positions: &[[f32; 3]]) -> (f32, f32) {
    let mut sum = (0.0f32, 0.0f32);
    let mut n = 0usize;
    for quad in positions.chunks(4) {
        if quad.len() < 4 {
            continue;
        }
        let d = |a: [f32; 3], b: [f32; 3]| {
            ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
        };
        sum.0 += d(quad[0], quad[1]);
        sum.1 += d(quad[0], quad[3]);
        n += 1;
    }
    if n == 0 {
        return (0.0, 0.0);
    }
    (sum.0 / n.max(1) as f32, sum.1 / n.max(1) as f32)
}

/// Rasterises the leaf quads from the side (+Z, slight yaw like the gallery)
/// and reports the fraction of the crown's projected bbox no leaf covers.
fn sky_opening(
    positions: &[[f32; 3]],
    uvs: &[[f32; 2]],
    atlas: &heads_up_bevy::world::leaf_shape::AlphaMask,
) -> f32 {
    if positions.is_empty() || uvs.len() != positions.len() {
        return 0.0;
    }
    // Project: x horizontal, y vertical (side view).
    let xs: Vec<f32> = positions.iter().map(|p| p[0]).collect();
    let ys: Vec<f32> = positions.iter().map(|p| p[1]).collect();
    let (x_min, x_max) = (
        xs.iter().cloned().fold(f32::MAX, f32::min),
        xs.iter().cloned().fold(f32::MIN, f32::max),
    );
    let (y_min, y_max) = (
        ys.iter().cloned().fold(f32::MAX, f32::min),
        ys.iter().cloned().fold(f32::MIN, f32::max),
    );
    let w = (x_max - x_min).max(1e-3);
    let h = (y_max - y_min).max(1e-3);
    let scale = RASTER as f32 / w;
    let gw = RASTER;
    let gh = ((h * scale).ceil() as u32).clamp(8, 4 * RASTER);
    let mut covered = vec![false; (gw * gh) as usize];

    for (quad, quad_uv) in positions.chunks(4).zip(uvs.chunks(4)) {
        if quad.len() < 4 || quad_uv.len() < 4 {
            continue;
        }
        let project = |p: [f32; 3]| ((p[0] - x_min) * scale, (y_max - p[1]) * scale);
        let corners: Vec<(f32, f32)> = quad.iter().map(|p| project(*p)).collect();
        for tri in [[0usize, 1, 2], [0, 2, 3]] {
            let (a, b, c) = (corners[tri[0]], corners[tri[1]], corners[tri[2]]);
            let area = (b.0 - a.0) * (c.1 - a.1) - (b.1 - a.1) * (c.0 - a.0);
            if area.abs() < 1e-6 {
                continue;
            }
            let min_x = a.0.min(b.0).min(c.0).floor().max(0.0) as u32;
            let max_x = a.0.max(b.0).max(c.0).ceil().min(gw as f32 - 1.0) as u32;
            let min_y = a.1.min(b.1).min(c.1).floor().max(0.0) as u32;
            let max_y = a.1.max(b.1).max(c.1).ceil().min(gh as f32 - 1.0) as u32;
            for py in min_y..=max_y {
                for px in min_x..=max_x {
                    let p = (px as f32 + 0.5, py as f32 + 0.5);
                    let w0 = ((b.0 - a.0) * (p.1 - a.1) - (b.1 - a.1) * (p.0 - a.0)) / area;
                    let w1 = ((c.0 - b.0) * (p.1 - b.1) - (c.1 - b.1) * (p.0 - b.0)) / area;
                    let w2 = ((a.0 - c.0) * (p.1 - c.1) - (a.1 - c.1) * (p.0 - c.0)) / area;
                    if w0 < 0.0 || w1 < 0.0 || w2 < 0.0 {
                        continue;
                    }
                    let u =
                        w0 * quad_uv[tri[0]][0] + w1 * quad_uv[tri[1]][0] + w2 * quad_uv[tri[2]][0];
                    let v =
                        w0 * quad_uv[tri[0]][1] + w1 * quad_uv[tri[1]][1] + w2 * quad_uv[tri[2]][1];
                    let tx = (u * atlas.width as f32).clamp(0.0, atlas.width as f32 - 1.0) as u32;
                    let ty = (v * atlas.height as f32).clamp(0.0, atlas.height as f32 - 1.0) as u32;
                    if atlas.get(tx, ty) >= 128 {
                        covered[(py * gw + px) as usize] = true;
                    }
                }
            }
        }
    }

    let inside = covered.iter().filter(|c| **c).count();
    1.0 - inside as f32 / covered.len() as f32
}
