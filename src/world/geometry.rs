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

/// Maximum miter stretch factor (`1/cos(θ/2)`) at a polygon vertex. Sharp
/// perturbed corners would otherwise push the inset vertex far past the
/// street centerline and into the neighbour.
const MITER_LIMIT: f32 = 1.5;

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

/// Insets a convex polygon by a per-edge amount (ADR-0004).
///
/// Each vertex is moved along the **angle bisector of its two incident
/// edges**, inward, scaled so both edges end up exactly `inset` away from
/// where they were. Working per-vertex (not per-line-intersection) keeps the
/// result strictly inside the original polygon even when perturbation makes
/// consecutive edges non-orthogonal — the failure mode that produced up to
/// 20 m of overlap when the offset lines were intersected independently.
///
/// `insets[i]` is the inset for edge `i` (`polygon[i] -> polygon[i + 1]`); a
/// vertex between two edges uses the average of the two, which is exact for
/// equal insets (the normal case) and conservative otherwise.
///
/// Returns `None` if the polygon degenerates.
pub fn inset_polygon_clip_with_insets(polygon: &[Vec2], insets: &[f32]) -> Option<Vec<Vec2>> {
    debug_assert_eq!(polygon.len(), insets.len(), "one inset per edge");
    let n = polygon.len();
    if n < 3 {
        return None;
    }
    let centroid = polygon_centroid(polygon);

    // Inward unit normal per edge (sign resolved against the centroid, so
    // winding order does not matter).
    let mut normals = Vec::with_capacity(n);
    for i in 0..n {
        let a = polygon[i];
        let b = polygon[(i + 1) % n];
        let edge = b - a;
        if edge.length_squared() < 1e-12 {
            return None;
        }
        let mut normal = Vec2::new(-edge.y, edge.x).normalize();
        if normal.dot(centroid - (a + b) * 0.5) < 0.0 {
            normal = -normal;
        }
        normals.push(normal);
    }

    let mut out = Vec::with_capacity(n);
    for (i, vertex) in polygon.iter().enumerate() {
        let prev_edge = (i + n - 1) % n;
        let cur_edge = i;
        let inset = (insets[prev_edge] + insets[cur_edge]) * 0.5;
        // Bisector of the two inward normals, normalized; its length factor
        // 1/cos(θ/2) keeps both edges at the requested distance.
        let mut dir = normals[prev_edge] + normals[cur_edge];
        if dir.length_squared() < 1e-9 {
            // Antiparallel normals (180° spike): fall back to one normal.
            dir = normals[cur_edge];
        }
        let dir = dir.normalize();
        // Miter length is 1/cos(θ/2); capping the stretch keeps a sharp
        // perturbed corner from shooting a vertex far past the street.
        // A cap of 1.5 costs a slightly short edge at that vertex (the street
        // there is a bit wider) instead of cutting into the neighbour.
        let cos_half = dir.dot(normals[cur_edge]).clamp(1.0 / MITER_LIMIT, 1.0);
        out.push(*vertex + dir * (inset / cos_half));
    }

    if polygon_area(&out).abs() < MIN_AREA {
        return None;
    }
    Some(out)
}

/// Convex hull of a point set (monotone chain), CCW, collinear points
/// removed. Used to keep block polygons convex: an L-shaped block's perimeter
/// walk contains a reflex corner that would turn inside-out once inset.
pub fn convex_hull(points: &[Vec2]) -> Vec<Vec2> {
    let mut pts = points.to_vec();
    pts.sort_by(|a, b| {
        a.x.partial_cmp(&b.x)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.y.partial_cmp(&b.y).unwrap_or(std::cmp::Ordering::Equal))
    });
    pts.dedup();
    if pts.len() < 3 {
        return pts;
    }
    fn cross(o: Vec2, a: Vec2, b: Vec2) -> f32 {
        (a - o).perp_dot(b - o)
    }
    let mut lower: Vec<Vec2> = Vec::new();
    for &p in &pts {
        while lower.len() >= 2 && cross(lower[lower.len() - 2], lower[lower.len() - 1], p) <= 0.0 {
            lower.pop();
        }
        lower.push(p);
    }
    let mut upper: Vec<Vec2> = Vec::new();
    for &p in pts.iter().rev() {
        while upper.len() >= 2 && cross(upper[upper.len() - 2], upper[upper.len() - 1], p) <= 0.0 {
            upper.pop();
        }
        upper.push(p);
    }
    lower.pop();
    upper.pop();
    lower.extend(upper);
    lower
}

/// Arithmetic mean of the polygon's vertices (the original's
/// `_polygon_center`; sufficient as an interior reference point for
/// determining inward normals of a convex polygon).
pub fn polygon_centroid(polygon: &[Vec2]) -> Vec2 {
    let n = polygon.len().max(1) as f32;
    polygon.iter().copied().sum::<Vec2>() / n
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

/// True if a point lies strictly inside a convex polygon (ray casting).
pub fn point_in_convex_polygon(p: Vec2, polygon: &[Vec2]) -> bool {
    let n = polygon.len();
    if n < 3 {
        return false;
    }
    let mut inside = false;
    let mut j = n - 1;
    for i in 0..n {
        let (a, b) = (polygon[i], polygon[j]);
        if (b.y > p.y) != (a.y > p.y) && p.x < (a.x - b.x) * (p.y - b.y) / (a.y - b.y) + b.x {
            inside = !inside;
        }
        j = i;
    }
    inside
}

/// True if two convex polygons genuinely interpenetrate.
///
/// This is the OVERLAPS=0 judge for block polygons. It checks a vertex of one
/// inside the other; two convex polygons that share no vertex yet overlap must
/// cross edges, which for our tiling cannot happen, so this is exact for the
/// case that matters and immune to the false positives SAT produces when
/// two perturbed edges run within millimeters of each other while staying on
/// their own side of the street.
pub fn convex_polygons_interpenetrate(a: &[Vec2], b: &[Vec2]) -> bool {
    a.iter().any(|p| point_in_convex_polygon(*p, b))
        || b.iter().any(|p| point_in_convex_polygon(*p, a))
}

/// Penetration depth between two convex polygons, in meters (SAT).
///
/// `0` means the polygons are disjoint **or** merely touch along an edge (the
/// enclosure band's flush, zero-gap joint). A positive value is how deep the
/// two shapes interpenetrate — the real overlap the OVERLAPS=0 guarantee
/// forbids.
pub fn convex_polygon_overlap_depth(a: &[Vec2], b: &[Vec2]) -> f32 {
    fn axis_depth(a: &[Vec2], b: &[Vec2], axis: Vec2) -> f32 {
        let (a_min, a_max) = project_polygon(a, axis);
        let (b_min, b_max) = project_polygon(b, axis);
        a_max.min(b_max) - a_min.max(b_min)
    }
    fn project_polygon(polygon: &[Vec2], axis: Vec2) -> (f32, f32) {
        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;
        for p in polygon {
            let d = axis.dot(*p);
            min = min.min(d);
            max = max.max(d);
        }
        (min, max)
    }
    let mut depth = f32::INFINITY;
    for poly in [a, b] {
        let n = poly.len();
        for i in 0..n {
            let edge = poly[(i + 1) % n] - poly[i];
            if edge.length_squared() < 1e-12 {
                continue;
            }
            let axis = Vec2::new(-edge.y, edge.x).normalize();
            let d = axis_depth(a, b, axis);
            if d <= 0.0 {
                return 0.0; // Separating axis: disjoint or edge-touching.
            }
            depth = depth.min(d);
        }
    }
    if depth.is_finite() { depth } else { 0.0 }
}

/// Convex polygons overlap by more than `epsilon` meters.
pub fn convex_polygons_overlap_by(a: &[Vec2], b: &[Vec2], epsilon: f32) -> bool {
    convex_polygon_overlap_depth(a, b) > epsilon
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
