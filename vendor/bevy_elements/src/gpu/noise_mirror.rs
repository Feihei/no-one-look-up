//! A CPU transliteration of `src/noise.wesl` (f32, wrapping u32 arithmetic), used by tests to
//! check the analytic derivative and to measure the normalisation constants the kernels use.

use bevy::math::{IVec2, IVec3, UVec3, Vec2, Vec3, Vec4};

const NOISE_PERIOD: i32 = 4096;

/// How far a cell network's feature point is drawn *towards* its cell's centre, 0 to 1
/// (`noise.wesl`'s `CELL_INSET`): the cells are unequal, and a face of unequal cells is a
/// scatter of fragments and not a mosaic.
const CELL_INSET: f32 = 0.0;

fn pcg3d(v_in: UVec3) -> UVec3 {
    let mul = |a: u32, b: u32| a.wrapping_mul(b);
    let mut v = UVec3::new(
        mul(v_in.x, 1_664_525).wrapping_add(1_013_904_223),
        mul(v_in.y, 1_664_525).wrapping_add(1_013_904_223),
        mul(v_in.z, 1_664_525).wrapping_add(1_013_904_223),
    );
    v.x = v.x.wrapping_add(mul(v.y, v.z));
    v.y = v.y.wrapping_add(mul(v.z, v.x));
    v.z = v.z.wrapping_add(mul(v.x, v.y));
    v ^= v >> 16;
    v.x = v.x.wrapping_add(mul(v.y, v.z));
    v.y = v.y.wrapping_add(mul(v.z, v.x));
    v.z = v.z.wrapping_add(mul(v.x, v.y));
    v
}

pub(crate) fn hash33(p: IVec3) -> Vec3 {
    let w = p & IVec3::splat(NOISE_PERIOD - 1);
    let h: UVec3 = pcg3d(w.as_uvec3()) >> 8u32;
    h.as_vec3() * (1.0 / 16_777_216.0)
}

/// Hash of an integer lattice point to one float in [0, 1), the mirror of `hash31` in
/// `noise.wesl`.
pub(crate) fn hash31(p: IVec3) -> f32 {
    hash33(p).x
}

fn lattice_gradient(cell: IVec3) -> Vec3 {
    (hash33(cell) * 2.0 - 1.0 + Vec3::splat(1e-6)).normalize()
}

/// Mirror of `perlin3_d`: (value, ∂/∂x, ∂/∂y, ∂/∂z).
pub(crate) fn perlin3_d(p: Vec3) -> Vec4 {
    let i = p.floor().as_ivec3();
    let f = p - p.floor();
    let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    let du = 30.0 * f * f * (f * (f - 2.0) + 1.0);
    let g = |o: IVec3| lattice_gradient(i + o);
    let (ga, gb, gc, gd) = (
        g(IVec3::ZERO),
        g(IVec3::X),
        g(IVec3::Y),
        g(IVec3::new(1, 1, 0)),
    );
    let (ge, gf, gg, gh) = (
        g(IVec3::Z),
        g(IVec3::new(1, 0, 1)),
        g(IVec3::new(0, 1, 1)),
        g(IVec3::ONE),
    );
    let va = ga.dot(f);
    let vb = gb.dot(f - Vec3::X);
    let vc = gc.dot(f - Vec3::Y);
    let vd = gd.dot(f - Vec3::new(1.0, 1.0, 0.0));
    let ve = ge.dot(f - Vec3::Z);
    let vf = gf.dot(f - Vec3::new(1.0, 0.0, 1.0));
    let vg = gg.dot(f - Vec3::new(0.0, 1.0, 1.0));
    let vh = gh.dot(f - Vec3::ONE);
    let (k1, k2, k3) = (vb - va, vc - va, ve - va);
    let (k4, k5, k6) = (va - vb - vc + vd, va - vc - ve + vg, va - vb - ve + vf);
    let k7 = -va + vb + vc - vd + ve - vf - vg + vh;
    let value = va
        + u.x * k1
        + u.y * k2
        + u.z * k3
        + u.x * u.y * k4
        + u.y * u.z * k5
        + u.z * u.x * k6
        + u.x * u.y * u.z * k7;
    let yzx = Vec3::new(u.y, u.z, u.x);
    let zxy = Vec3::new(u.z, u.x, u.y);
    let grad = ga
        + u.x * (gb - ga)
        + u.y * (gc - ga)
        + u.z * (ge - ga)
        + u.x * u.y * (ga - gb - gc + gd)
        + u.y * u.z * (ga - gc - ge + gg)
        + u.z * u.x * (ga - gb - ge + gf)
        + u.x * u.y * u.z * (-ga + gb + gc - gd + ge - gf - gg + gh)
        + du * (Vec3::new(k1, k2, k3)
            + yzx * Vec3::new(k4, k5, k6)
            + zxy * Vec3::new(k6, k4, k5)
            + yzx * zxy * k7);
    Vec4::new(value, grad.x, grad.y, grad.z) * 1.1547
}

pub(crate) fn seed_offset(k: u32) -> Vec3 {
    Vec3::new(
        (k * 1031 % 4096) as f32,
        (k * 2053 % 4096) as f32,
        (k * 3079 % 4096) as f32,
    )
}

/// Mirror of `fbm3`.
/// Mirror of `perlin3` in `noise.wesl`: the **value only**, with the derivative omitted.
///
/// The same eight lattice gradients and the same interpolation as `perlin3_d`, with only the
/// derivative's terms left out, so the value is **bit-identical** to `perlin3_d(p).x`.
pub(crate) fn perlin3(p: Vec3) -> f32 {
    perlin3_d(p).x
}

pub(crate) fn fbm3(p: Vec3, octaves: u32) -> f32 {
    let (mut sum, mut amplitude, mut norm, mut q) = (0.0, 0.5, 0.0, p);
    for o in 0..octaves {
        sum += amplitude * perlin3_d(q + seed_offset(o + 1)).x;
        norm += amplitude;
        amplitude *= 0.5;
        q *= 2.0;
    }
    sum / f32::max(norm, 1e-6)
}

/// Mirror of `curl3`.
pub(crate) fn curl3(p: Vec3) -> Vec3 {
    let a = perlin3_d(p + seed_offset(11));
    let b = perlin3_d(p + seed_offset(12));
    let c = perlin3_d(p + seed_offset(13));
    Vec3::new(c.z - b.w, a.w - c.y, b.y - a.z)
}

/// A deterministic spread of sample points over `[0, extent)³`.
pub(crate) fn samples(n: usize, extent: f32) -> impl Iterator<Item = Vec3> {
    (0..n).map(move |k| {
        let h = hash33(IVec3::new(k as i32, 7, 1_234));
        let j = hash33(IVec3::new(k as i32, 99, 4_321));
        Vec3::new(h.x, h.y, j.z) * extent
    })
}

/// Mirror of `perlin3_periodic`: Perlin noise on a lattice that wraps every `period` cells.
pub(crate) fn perlin3_periodic(p: Vec3, period: i32) -> f32 {
    let i = p.floor().as_ivec3();
    let f = p - p.floor();
    let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    let m = IVec3::splat(period - 1);
    let dot_at = |cell: IVec3, corner: Vec3| lattice_gradient((i + cell) & m).dot(f - corner);
    let va = dot_at(IVec3::ZERO, Vec3::ZERO);
    let vb = dot_at(IVec3::X, Vec3::X);
    let vc = dot_at(IVec3::Y, Vec3::Y);
    let vd = dot_at(IVec3::new(1, 1, 0), Vec3::new(1.0, 1.0, 0.0));
    let ve = dot_at(IVec3::Z, Vec3::Z);
    let vf = dot_at(IVec3::new(1, 0, 1), Vec3::new(1.0, 0.0, 1.0));
    let vg = dot_at(IVec3::new(0, 1, 1), Vec3::new(0.0, 1.0, 1.0));
    let vh = dot_at(IVec3::ONE, Vec3::ONE);
    let mix = |a: f32, b: f32, t: f32| a * (1.0 - t) + b * t;
    let x0 = mix(mix(va, vb, u.x), mix(vc, vd, u.x), u.y);
    let x1 = mix(mix(ve, vf, u.x), mix(vg, vh, u.x), u.y);
    mix(x0, x1, u.z) * 1.1547
}

/// Mirror of `fbm3_periodic`: `octaves` periodic octaves, tiling with `period` lattice units at
/// the base octave, divided by the sum of the amplitudes.
pub(crate) fn fbm3_periodic(p: Vec3, period: i32, octaves: u32) -> f32 {
    let (mut sum, mut amplitude, mut norm, mut q, mut n) = (0.0, 0.5, 0.0, p, period);
    for o in 0..octaves {
        sum += amplitude * perlin3_periodic(q + seed_offset(o + 1), n);
        norm += amplitude;
        amplitude *= 0.5;
        q *= 2.0;
        n *= 2;
    }
    sum / f32::max(norm, 1e-6)
}

/// Mirror of `worley3` with a periodic lattice (`period > 0`): distance to the nearest feature
/// point, roughly in [0, 1].
pub(crate) fn worley3(p: Vec3, period: i32) -> f32 {
    let i = p.floor().as_ivec3();
    let f = p - p.floor();
    let mut best = 1e9_f32;
    for z in -1..=1 {
        for y in -1..=1 {
            for x in -1..=1 {
                let o = IVec3::new(x, y, z);
                let mut c = i + o;
                if period > 0 {
                    c = IVec3::new(
                        c.x.rem_euclid(period),
                        c.y.rem_euclid(period),
                        c.z.rem_euclid(period),
                    );
                }
                let d = o.as_vec3() + hash33(c) - f;
                best = best.min(d.dot(d));
            }
        }
    }
    best.sqrt()
}

/// A cell of the 2D cellular network, the mirror of `cell2` in `noise.wesl`: the distance to the
/// nearest feature point, the point itself, and the cell's own random factor.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Cell2 {
    /// The distance to the feature point, in lattice units.
    pub distance: f32,
    /// The feature point's position, in lattice units.
    pub point: Vec2,
    /// The cell's own random factor, in [0, 1).
    pub factor: f32,
}

/// Mirror of `cell2` in `noise.wesl`: the 2D cellular network's nearest feature point, its
/// distance and its cell's random factor.
pub(crate) fn cell2(p: Vec2) -> Cell2 {
    let i = p.floor().as_ivec2();
    let f = p - p.floor();
    let mut best = 1e9_f32;
    let mut best_point = Vec2::ZERO;
    let mut best_factor = 0.0;
    for y in -1..=1 {
        for x in -1..=1 {
            let o = IVec2::new(x, y);
            let c = (i + o) & IVec2::splat(NOISE_PERIOD - 1);
            let h = hash33(IVec3::new(c.x, c.y, 0));
            let point = o.as_vec2() + h.truncate();
            let d = point - f;
            let d2 = d.dot(d);
            if d2 < best {
                best = d2;
                best_point = point;
                best_factor = h.z;
            }
        }
    }
    Cell2 {
        distance: best.sqrt(),
        point: best_point,
        factor: best_factor,
    }
}

/// A cell of the 3D cellular network, the mirror of `cell3` in `noise.wesl`: the distance to the
/// nearest feature point, the distance to the second nearest (the cell's *boundary*), the point
/// itself, and the cell's two random factors.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Cell3 {
    /// The feature point's position, in lattice units.
    #[allow(dead_code)]
    pub point: Vec3,
    /// The cell's own random factor, in [0, 1).
    #[allow(dead_code)]
    pub factor: f32,
    /// A **second** per-cell draw, in [0, 1): an independent hash of the cell, for the rock's
    /// palette (`noise.wesl`'s `factor2`).
    #[allow(dead_code)]
    pub factor2: f32,
}

/// Mirror of `cell3` in `noise.wesl`: the 3D cellular network's nearest feature point, the
/// second nearest point's distance and the cell's two random factors.
pub(crate) fn cell3(p: Vec3) -> Cell3 {
    let i = p.floor().as_ivec3();
    let f = p - p.floor();
    let mut best = 1e9_f32;
    let mut best_point = Vec3::ZERO;
    let mut best_factor = 0.0;
    let mut best_factor2 = 0.0;
    // The cells are visited in order of increasing distance, so the prune's bound tightens
    // quickly. The bound is exact, so the result is the same as the full search.
    for ring in 0..3 {
        for z in -1i32..=1 {
            for y in -1i32..=1 {
                for x in -1i32..=1 {
                    // The ring the cell is in, from the point's own cell outwards.
                    if x.abs().max(y.abs()).max(z.abs()) != ring {
                        continue;
                    }
                    let o = IVec3::new(x, y, z);
                // The cell is pruned by the distance to its nearest corner: a feature point
                // lies inside its cell, so a cell whose corner is further than the best
                // distance so far cannot hold a closer one. The bound is exact, so the result
                // is the same as the full search.
                let lo = (o.as_vec3() - f).abs().max((f - o.as_vec3() - 1.0).abs());
                if lo.dot(lo) >= best {
                    continue;
                }
                let c = (i + o) & IVec3::splat(NOISE_PERIOD - 1);
                let h = hash33(c);
                // The point's offset inside its cell is drawn towards the cell's centre, so
                // the cells are **unequal**: a point near its cell's corner leaves a big cell
                // behind it and a point at the centre a small one. A cell network's cells are
                // the same size when the points sit on a lattice, and a face of equal cells is
                // a mosaic (`noise.wesl`'s `CELL_INSET`).
                let inset = h.lerp(Vec3::splat(0.5), CELL_INSET);
                let point = o.as_vec3() + inset;
                let d = point - f;
                let d2 = d.dot(d);
                    if d2 < best {
                        best = d2;
                        best_point = point;
                        best_factor = h.z;
                        best_factor2 = hash31(c + IVec3::new(7919, 104729, 1299709));
                    }
                }
            }
        }
    }
    Cell3 {
        point: best_point,
        factor: best_factor,
        factor2: best_factor2,
    }
}

/// `PERLIN_FBM_MAX` of `generate.wesl`: the largest |fbm| of the 3-octave periodic Perlin sum.
pub(crate) const PERLIN_FBM_MAX: f32 = 0.55;
/// `FIELD_MAX` of `weather.wesl`: the same for the coverage field's slice (z = 0.31, period 4),
/// whose largest |fbm| is lower than the 3D noise's because a 2D slice of Perlin noise seldom
/// reaches the corner of the cube (10⁵ points: 0.374; a 2000² grid over the period: 0.383).
pub(crate) const FIELD_MAX: f32 = 0.385;

/// `inverted_worley` of `generate.wesl`: Worley noise with `k` cells across the tile, inverted.
fn inverted_worley(x: Vec3, k: i32) -> f32 {
    1.0 - worley3(x * k as f32, k).min(1.0)
}

/// The base cloud noise (RGBA, 0..1) at a continuous tile coordinate `uvw` in [0, 1)³: what
/// `cloud_noise_base` stores at the texel whose centre is `uvw`.
pub(crate) fn cloud_base_at(uvw: Vec3) -> [f32; 4] {
    let w8 = inverted_worley(uvw, 8);
    let w16 = inverted_worley(uvw, 16);
    let w32 = inverted_worley(uvw, 32);
    let w56 = inverted_worley(uvw, 56);
    let w64 = inverted_worley(uvw, 64);
    let perlin = (0.5 + 0.5 * fbm3_periodic(uvw * 8.0, 8, 3) / PERLIN_FBM_MAX).clamp(0.0, 1.0);
    let dilated = 0.625 * w8 + 0.25 * w32 + 0.125 * w56;
    let r = dilated + perlin * (1.0 - dilated);
    let g = 0.625 * w8 + 0.25 * w16 + 0.125 * w32;
    let b = 0.625 * w16 + 0.25 * w32 + 0.125 * w64;
    let a = 0.75 * w32 + 0.25 * w64;
    [r, g, b, a]
}

/// `cloud_noise_base` for texel `(x, y, z)` of the 128³ volume, before the 8-bit quantisation of
/// `textureStore`.
pub(crate) fn cloud_base_texel(x: u32, y: u32, z: u32) -> [f32; 4] {
    cloud_base_at((UVec3::new(x, y, z).as_vec3() + 0.5) / 128.0)
}

/// The detail cloud noise (RGBA, 0..1) at a continuous tile coordinate `uvw` in [0, 1)³: what
/// `cloud_noise_detail` stores at the texel whose centre is `uvw`.
pub(crate) fn cloud_detail_at(uvw: Vec3) -> [f32; 4] {
    let w2 = inverted_worley(uvw, 2);
    let w4 = inverted_worley(uvw, 4);
    let w8 = inverted_worley(uvw, 8);
    let w16 = inverted_worley(uvw, 16);
    let r = 0.625 * w2 + 0.25 * w4 + 0.125 * w8;
    let g = 0.625 * w4 + 0.25 * w8 + 0.125 * w16;
    let b = 0.75 * w8 + 0.25 * w16;
    [r, g, b, 1.0]
}

/// `cloud_noise_detail` for texel `(x, y, z)` of the 32³ volume, before the 8-bit quantisation.
pub(crate) fn cloud_detail_texel(x: u32, y: u32, z: u32) -> [f32; 4] {
    cloud_detail_at((UVec3::new(x, y, z).as_vec3() + 0.5) / 32.0)
}

/// Mirror of `coverage_field` of `weather.wesl` at a world position, `[0, 1]`: 3 octaves of 2D
/// Perlin with 8, 4 and 2 km wavelengths on a lattice that wraps every 32 km, at the position
/// minus the cloud `drift`.
pub(crate) fn cloud_coverage_field(world_xz: [f32; 2], drift: [f32; 2]) -> f32 {
    let qx = (world_xz[0] - drift[0]) / 8000.0;
    let qz = (world_xz[1] - drift[1]) / 8000.0;
    let n = fbm3_periodic(Vec3::new(qx, qz, 0.31), 4, 3);
    (0.5 + 0.5 * n / FIELD_MAX).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_analytic_derivative_matches_finite_differences() {
        let h = 1e-3;
        let mut worst: f32 = 0.0;
        for p in samples(2_000, 4.0) {
            let d = perlin3_d(p);
            for (axis, analytic) in [(Vec3::X, d.y), (Vec3::Y, d.z), (Vec3::Z, d.w)] {
                let fd = (perlin3_d(p + axis * h).x - perlin3_d(p - axis * h).x) / (2.0 * h);
                worst = worst.max((fd - analytic).abs());
            }
        }
        assert!(worst < 2e-3, "worst derivative error {worst}");
    }

    /// The value-only noise and the derivative-carrying one agree **exactly**.
    ///
    /// The value-only noise omits the derivative's terms, so a call site can switch to it
    /// without changing a pixel. That is the property the crate's shaders rely on: five
    /// evaluations per rock pixel use the value alone, and the derivative was computed for
    /// them and thrown away (`docs/research/rock-cost.md`).
    #[test]
    fn the_value_only_noise_is_the_same_value() {
        for i in 0..500 {
            let p = Vec3::new(
                i as f32 * 0.137,
                (i as f32) * 0.071 + 0.5,
                (i as f32) * 0.211 - 1.5,
            );
            assert_eq!(perlin3(p), perlin3_d(p).x, "the value differs at {p:?}");
        }
    }

    #[test]
    fn the_lattice_is_periodic() {
        let p = Vec3::new(3.3, 7.7, 11.1);
        let q = p + Vec3::new(4096.0, -4096.0, 8192.0);
        assert!((perlin3_d(p) - perlin3_d(q)).abs().max_element() < 2e-3);
    }

    #[test]
    fn the_kernel_normalisation_constants_match_the_noise() {
        let n = 100_000;
        let rms = (samples(n, 256.0)
            .map(|p| curl3(p).length_squared())
            .sum::<f32>()
            / n as f32)
            .sqrt();
        let fbm_max = samples(n, 256.0)
            .map(|p| fbm3(p, 2).abs())
            .fold(0.0_f32, f32::max);
        println!("curl rms {rms}, 2-octave fbm max |n| {fbm_max}");
        assert!(
            (rms * crate::wind::CURL_NORMALISE - 1.0).abs() < 0.02,
            "curl rms {rms}"
        );
        assert!(
            (fbm_max - crate::fog::FBM2_MAX).abs() < 0.02,
            "fbm max {fbm_max}"
        );
    }

    /// Largest |fbm| of the 3-octave periodic Perlin sums over 10⁵ deterministic points: the base
    /// noise's Perlin-Worley input (`fbm3_periodic(uvw·8, 8, 3)`) and the coverage field's slice
    /// (z = 0.31, period 4). `generate.wesl` and `weather.wesl` divide by 0.55.
    #[test]
    fn the_noise_normalisation_constants_match_the_noise() {
        let n = 100_000;
        let base_max = samples(n, 1.0)
            .map(|p| fbm3_periodic(p * 8.0, 8, 3).abs())
            .fold(0.0_f32, f32::max);
        let field_max = samples(n, 4.0)
            .map(|p| fbm3_periodic(Vec3::new(p.x, p.y, 0.31), 4, 3).abs())
            .fold(0.0_f32, f32::max);
        println!("base fbm max |n| {base_max}, field fbm max |n| {field_max}");
        let base_ratio = base_max / PERLIN_FBM_MAX;
        let field_ratio = field_max / FIELD_MAX;
        assert!(
            (0.9..=1.0).contains(&base_ratio),
            "base fbm max {base_max} against PERLIN_FBM_MAX {PERLIN_FBM_MAX}"
        );
        assert!(
            (0.9..=1.0).contains(&field_ratio),
            "field fbm max {field_max} against FIELD_MAX {FIELD_MAX}"
        );
    }

    /// Mean absolute difference per channel between the texel planes `i` and `j` of `axis` of an
    /// `n`³ volume, sampled every fourth texel.
    fn plane_difference(
        texel: fn(u32, u32, u32) -> [f32; 4],
        n: u32,
        axis: usize,
        i: u32,
        j: u32,
    ) -> [f32; 4] {
        let at = |plane: u32, a: u32, b: u32| match axis {
            0 => texel(plane, a, b),
            1 => texel(a, plane, b),
            _ => texel(a, b, plane),
        };
        let mut sum = [0.0_f32; 4];
        let mut count = 0.0;
        for a in (1..n).step_by(4) {
            for b in (1..n).step_by(4) {
                let (p, q) = (at(i, a, b), at(j, a, b));
                for (s, (p, q)) in sum.iter_mut().zip(p.into_iter().zip(q)) {
                    *s += (p - q).abs();
                }
                count += 1.0;
            }
        }
        sum.map(|s| s / count)
    }

    /// Every axis of a generated volume tiles: the difference between its first and last texel
    /// plane is no larger than 1.5 times that between interior neighbours.
    fn assert_volume_tiles(name: &str, texel: fn(u32, u32, u32) -> [f32; 4], n: u32) {
        for axis in 0..3 {
            let wrap = plane_difference(texel, n, axis, 0, n - 1);
            let mut interior = [0.0_f32; 4];
            let ks = [n / 8, n / 4, 3 * n / 8, n / 2, 5 * n / 8, 3 * n / 4];
            for k in ks {
                let d = plane_difference(texel, n, axis, k, k + 1);
                for (m, d) in interior.iter_mut().zip(d) {
                    *m += d / ks.len() as f32;
                }
            }
            for (c, (w, i)) in wrap.into_iter().zip(interior).enumerate() {
                assert!(
                    w <= 1.5 * i + 1e-6,
                    "{name} axis {axis} channel {c}: wrap difference {w} against interior {i}"
                );
            }
        }
    }

    #[test]
    fn the_cloud_noise_volumes_tile_without_a_seam() {
        assert_volume_tiles("base", cloud_base_texel, 128);
        assert_volume_tiles("detail", cloud_detail_texel, 32);
    }

    /// Channel means of the generated noise, measured with this mirror: base A (the finest Worley
    /// mix) over every fourth texel of the 128³ volume, and detail B over all of the 32³ volume.
    /// The note does not predict them; they tell the shader's `lf` and `hf` blends their centre.
    /// Measured: base A 0.4987, detail B 0.4764.
    const BASE_A_MEAN: f32 = 0.4987;
    const DETAIL_B_MEAN: f32 = 0.4764;

    #[test]
    fn the_cloud_noise_channel_means_are_the_recorded_ones() {
        let mut base = 0.0;
        let mut count = 0.0;
        for z in (1..128).step_by(4) {
            for y in (1..128).step_by(4) {
                for x in (1..128).step_by(4) {
                    base += cloud_base_texel(x, y, z)[3];
                    count += 1.0;
                }
            }
        }
        let base = base / count;
        let mut detail = 0.0;
        for z in 0..32 {
            for y in 0..32 {
                for x in 0..32 {
                    detail += cloud_detail_texel(x, y, z)[2];
                }
            }
        }
        let detail = detail / (32.0 * 32.0 * 32.0);
        println!("base A mean {base}, detail B mean {detail}");
        assert!((base - BASE_A_MEAN).abs() <= 0.05, "base A mean {base}");
        assert!(
            (detail - DETAIL_B_MEAN).abs() <= 0.05,
            "detail B mean {detail}"
        );
    }

    #[test]
    fn the_cloud_noise_stays_in_the_unit_range() {
        for p in samples(2_000, 1.0) {
            for v in cloud_base_at(p).into_iter().chain(cloud_detail_at(p)) {
                assert!((0.0..=1.0).contains(&v), "noise value {v} at {p:?}");
            }
        }
    }
}
