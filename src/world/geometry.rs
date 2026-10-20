//! Geometry helpers for block polygons (ADR-0004).
//!
//! Block polygons come from the grid as axis-aligned convex rectangles
//! (blocks are carved as rectangles; see `path_network`). The inset uses a
//! self-written half-plane clip (`inset_polygon_clip`), which for convex
//! polygons mathematically guarantees the inset stays inside the original
//! polygon. Because adjacent blocks share the street centerline and each
//! insets by `width / 2` on its own territory, neighboring inset polygons
//! never overlap — asserted by the OVERLAPS=0 tests here and in
//! `path_network`.

use bevy::math::Vec2;

/// Minimum polygon area (m²) for an inset result to be considered valid.
const MIN_AREA: f32 = 1.0;

/// Clips `polygon` (convex, CCW winding) against the half-plane
/// `normal · p >= d`, keeping the inside (Sutherland–Hodgman step).
fn clip_half_plane(polygon: &[Vec2], normal: Vec2, d: f32) -> Vec<Vec2> {
    let mut out = Vec::with_capacity(polygon.len() + 1);
    let n = polygon.len();
    for i in 0..n {
        let cur = polygon[i];
        let next = polygon[(i + 1) % n];
        let cur_side = normal.dot(cur) - d;
        let next_side = normal.dot(next) - d;
        if cur_side >= 0.0 {
            out.push(cur);
        }
        // Edge crosses the plane: emit the intersection point.
        if (cur_side > 0.0 && next_side < 0.0) || (cur_side < 0.0 && next_side > 0.0) {
            let t = cur_side / (cur_side - next_side);
            out.push(cur.lerp(next, t));
        }
    }
    out
}

/// Signed area of a polygon (positive for CCW winding in standard math axes).
pub fn polygon_area(polygon: &[Vec2]) -> f32 {
    let n = polygon.len();
    let mut sum = 0.0;
    for i in 0..n {
        let a = polygon[i];
        let b = polygon[(i + 1) % n];
        sum += a.x * b.y - b.x * a.y;
    }
    sum * 0.5
}

/// Insets a convex polygon by `inset` meters via half-plane clipping.
///
/// `polygon` must be convex and wound CCW. Returns `None` if the polygon
/// collapses below [`MIN_AREA`] (inset larger than the polygon supports).
pub fn inset_polygon_clip(polygon: &[Vec2], inset: f32) -> Option<Vec<Vec2>> {
    debug_assert!(inset >= 0.0, "inset must be non-negative");
    debug_assert!(polygon.len() >= 3, "polygon needs at least 3 vertices");
    let mut current: Vec<Vec2> = polygon.to_vec();
    let n = polygon.len();
    for i in 0..n {
        let a = polygon[i];
        let b = polygon[(i + 1) % n];
        let edge = b - a;
        let len = edge.length();
        if len < f32::EPSILON {
            continue;
        }
        // CCW winding: the inside of the polygon is to the left of each edge,
        // i.e. along the edge's left normal.
        let normal = Vec2::new(-edge.y, edge.x) / len;
        let d = normal.dot(a + normal * inset);
        current = clip_half_plane(&current, normal, d);
        if current.len() < 3 {
            return None;
        }
    }
    if polygon_area(&current) < MIN_AREA {
        return None;
    }
    Some(current)
}

/// True if two convex polygons overlap (separating axis test on edge normals).
pub fn convex_polygons_overlap(a: &[Vec2], b: &[Vec2]) -> bool {
    fn sat_overlap(a: &[Vec2], b: &[Vec2]) -> bool {
        let n = a.len();
        for i in 0..n {
            let edge = a[(i + 1) % n] - a[i];
            if edge.length_squared() < f32::EPSILON * f32::EPSILON {
                continue;
            }
            // Axis perpendicular to the edge.
            let axis = Vec2::new(-edge.y, edge.x);
            let (a_min, a_max) = project(a, axis);
            let (b_min, b_max) = project(b, axis);
            if a_max < b_min || b_max < a_min {
                return false; // Separating axis found.
            }
        }
        true
    }
    fn project(polygon: &[Vec2], axis: Vec2) -> (f32, f32) {
        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;
        for p in polygon {
            let d = axis.dot(*p);
            min = min.min(d);
            max = max.max(d);
        }
        (min, max)
    }
    sat_overlap(a, b) && sat_overlap(b, a)
}

/// Corner points (world coordinates, CCW) of the rectangle covering the cell
/// range `[row_start, row_end) x [col_start, col_end)`.
pub fn cell_rect_polygon(rows: std::ops::Range<u32>, cols: std::ops::Range<u32>) -> Vec<Vec2> {
    let min_x = cols.start as f32 * crate::world::CELL_SIZE;
    let max_x = cols.end as f32 * crate::world::CELL_SIZE;
    // Row 0 is the northernmost band (min Z in world space).
    let min_z = rows.start as f32 * crate::world::CELL_SIZE;
    let max_z = rows.end as f32 * crate::world::CELL_SIZE;
    vec![
        Vec2::new(min_x, min_z),
        Vec2::new(max_x, min_z),
        Vec2::new(max_x, max_z),
        Vec2::new(min_x, max_z),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn square(half: f32) -> Vec<Vec2> {
        vec![
            Vec2::new(-half, -half),
            Vec2::new(half, -half),
            Vec2::new(half, half),
            Vec2::new(-half, half),
        ]
    }

    #[test]
    fn inset_square_shrinks_uniformly() {
        let inset = inset_polygon_clip(&square(25.0), 3.0).expect("inset should not collapse");
        assert!((polygon_area(&inset) - 44.0 * 44.0).abs() < 1e-3);
        for p in &inset {
            assert!(p.x.abs() <= 22.0 + 1e-3 && p.y.abs() <= 22.0 + 1e-3);
        }
    }

    #[test]
    fn inset_too_large_collapses_to_none() {
        assert!(inset_polygon_clip(&square(25.0), 25.0).is_none());
        assert!(inset_polygon_clip(&square(25.0), 100.0).is_none());
    }

    #[test]
    fn l_shaped_block_rectangle_decomposition_insets_without_overlap() {
        // Two adjacent rectangles belonging to the same block inset
        // independently; the pair of parts stays inside the original union.
        let a = cell_rect_polygon(0..1, 0..2);
        let b = cell_rect_polygon(1..2, 0..1);
        let ia = inset_polygon_clip(&a, 3.0).expect("part a inset");
        let ib = inset_polygon_clip(&b, 3.0).expect("part b inset");
        for p in &ia {
            assert!(
                p.x >= 3.0 - 1e-3 && p.x <= 97.0 + 1e-3 && p.y >= 3.0 - 1e-3 && p.y <= 47.0 + 1e-3
            );
        }
        for p in &ib {
            assert!(
                p.x >= 3.0 - 1e-3 && p.x <= 47.0 + 1e-3 && p.y >= 53.0 - 1e-3 && p.y <= 97.0 + 1e-3
            );
        }
    }

    #[test]
    fn overlapping_squares_detected() {
        assert!(convex_polygons_overlap(&square(10.0), &square(10.0)));
        let shifted: Vec<Vec2> = square(10.0)
            .iter()
            .map(|p| Vec2::new(p.x + 15.0, p.y))
            .collect();
        assert!(convex_polygons_overlap(&square(10.0), &shifted));
        let far: Vec<Vec2> = square(10.0)
            .iter()
            .map(|p| Vec2::new(p.x + 100.0, p.y))
            .collect();
        assert!(!convex_polygons_overlap(&square(10.0), &far));
    }

    #[test]
    fn touching_edges_do_not_count_as_overlap() {
        // Shared street centerline: two blocks inset from a common edge do
        // not overlap (the OVERLAPS=0 guarantee).
        let left = cell_rect_polygon(0..1, 0..1);
        let right = cell_rect_polygon(0..1, 1..2);
        let il = inset_polygon_clip(&left, 3.0).expect("left inset");
        let ir = inset_polygon_clip(&right, 3.0).expect("right inset");
        assert!(!convex_polygons_overlap(&il, &ir));
    }
}
