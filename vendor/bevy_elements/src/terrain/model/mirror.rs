//! CPU transliterations of the terrain kernels that the tests compare against: the procedural
//! generator of `generate.wesl` and the layer weights of `terrain.wesl`.

use bevy::math::{UVec2, Vec2, Vec3};

use super::{F_HALF, GAIN, LACUNARITY, OCTAVE_ROTATION, R_LOW, R_SPAN, octave_count, seed_mix};
use crate::gpu::noise_mirror::perlin3_d;
use crate::terrain::ProceduralTerrain;

/// Both noise sums of a sample point: `(f, r)` of the research note.
pub(crate) fn noise_sums(local_xz: Vec2, feature_m: f32, octaves: u32, seed: u64) -> (f32, f32) {
    let p = local_xz / feature_m;
    let mix = seed_mix(seed);
    let (mut f_sum, mut r_sum, mut norm, mut gain) = (0.0, 0.0, 0.0, 1.0);
    let mut weight = 1.0_f32;
    for k in 0..octaves {
        let (s, c) = (OCTAVE_ROTATION * k as f32).sin_cos();
        let q = Vec2::new(c * p.x - s * p.y, s * p.x + c * p.y) * LACUNARITY.powi(k as i32);
        let v = ((mix + k * 1031) % 4096) as f32;
        let n = perlin3_d(Vec3::new(q.x + v, q.y + v, 0.5 + k as f32)).x;
        let ridge = (1.0 - n.abs()).max(0.0);
        let s_k = weight * ridge * ridge;
        f_sum += gain * n;
        r_sum += gain * s_k;
        norm += gain;
        weight = (2.0 * s_k).clamp(0.0, 1.0);
        gain *= GAIN;
    }
    (f_sum / norm, r_sum / norm)
}

/// The unit terrain value `(1 − q) u_f + q u_r` of the two sums.
pub(crate) fn unit_height(f: f32, r: f32, ridged: f32) -> f32 {
    let u_f = (0.5 + f / (2.0 * F_HALF)).clamp(0.0, 1.0);
    let u_r = ((r - R_LOW) / R_SPAN).clamp(0.0, 1.0);
    (1.0 - ridged) * u_f + ridged * u_r
}

/// The surface height at a point `local_xz` metres from the terrain's minimum corner, relative
/// to the terrain transform: the base height plus the relief times the unit value.
pub(crate) fn surface_height(p: &ProceduralTerrain, octaves: u32, local_xz: Vec2) -> f32 {
    let (f, r) = noise_sums(local_xz, p.feature_size_m, octaves, p.seed);
    p.base_height_m + p.relief_m * unit_height(f, r, p.ridged.clamp(0.0, 1.0))
}

/// The generated surface at every cell of a `size` grid of `cell_m` cells, row-major.
pub(crate) fn generate_surface(p: &ProceduralTerrain, size: UVec2, cell_m: f32) -> Vec<f32> {
    let octaves = octave_count(p.octaves, p.feature_size_m, cell_m);
    let mut out = Vec::with_capacity((size.x * size.y) as usize);
    for j in 0..size.y {
        for i in 0..size.x {
            let local = (Vec2::new(i as f32, j as f32) + 0.5) * cell_m;
            out.push(surface_height(p, octaves, local));
        }
    }
    out
}

/// The slope angle of a surface at a cell by central differences over ±1 cell (one-sided at the
/// edge), degrees.
pub(crate) fn slope_deg(surface: &[f32], size: UVec2, cell_m: f32, i: u32, j: u32) -> f32 {
    let at = |x: u32, z: u32| surface[(z * size.x + x) as usize];
    let (xl, xh) = (i.saturating_sub(1), (i + 1).min(size.x - 1));
    let (zl, zh) = (j.saturating_sub(1), (j + 1).min(size.y - 1));
    let gx = (at(xh, j) - at(xl, j)) / ((xh - xl) as f32 * cell_m);
    let gz = (at(i, zh) - at(i, zl)) / ((zh - zl) as f32 * cell_m);
    (gx * gx + gz * gz).sqrt().atan().to_degrees()
}

/// Percentile `q` (0–1) of `values` (sorted in place).
pub(crate) fn percentile(values: &mut [f32], q: f32) -> f32 {
    values.sort_by(f32::total_cmp);
    values[((values.len() - 1) as f32 * q).round() as usize]
}

/// `smoothstep(a, b, x)` of WGSL.
pub(crate) fn smoothstep(a: f32, b: f32, x: f32) -> f32 {
    let t = ((x - a) / (b - a)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Andersson 2007 §5.2.3, Listing 2: the Overlay blend of `detail` onto `mask` at full opacity.
pub(crate) fn overlay(mask: f32, detail: f32) -> f32 {
    if mask < 0.5 {
        2.0 * mask * detail
    } else {
        1.0 - 2.0 * (1.0 - mask) * (1.0 - detail)
    }
}

/// The four layer weights `(rock, sand, soil, grass)` of `render.wesl` for a slope angle
/// (degrees), a sediment depth (m) and a detail value in [0, 1] (0.5: none).
pub(crate) fn layer_weights(slope_deg: f32, sediment_m: f32, detail: f32) -> [f32; 4] {
    let rock = overlay(smoothstep(35.0, 45.0, slope_deg), detail);
    let sand = overlay(smoothstep(0.3, 0.7, sediment_m), detail)
        * (1.0 - smoothstep(26.0, 30.0, slope_deg))
        * (1.0 - rock);
    let band = overlay(smoothstep(26.0, 36.0, slope_deg), detail);
    let rest = (1.0 - rock) * (1.0 - sand);
    [rock, sand, rest * band, rest * (1.0 - band)]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// FNV-1a over the bytes of `values`.
    fn fnv1a(bytes: &[u8]) -> u64 {
        bytes.iter().fold(0xcbf2_9ce4_8422_2325, |h, b| {
            (h ^ u64::from(*b)).wrapping_mul(0x0000_0100_0000_01b3)
        })
    }

    /// The statistics of one preset at 1024² cells of 2 m: (spread of `h − h_base` between the
    /// 0.5 and 99.5 percentiles, slope p50, p95, share above 40° in percent).
    fn stats(p: &ProceduralTerrain) -> (f32, f32, f32, f32) {
        let size = UVec2::splat(1024);
        let surface = generate_surface(p, size, 2.0);
        let mut heights: Vec<f32> = surface.iter().map(|h| h - p.base_height_m).collect();
        let spread = percentile(&mut heights, 0.995) - percentile(&mut heights, 0.005);
        let mut slopes: Vec<f32> = (0..size.y)
            .flat_map(|j| (0..size.x).map(move |i| (i, j)))
            .map(|(i, j)| slope_deg(&surface, size, 2.0, i, j))
            .collect();
        let steep = slopes.iter().filter(|s| **s > 40.0).count() as f32 / slopes.len() as f32;
        (
            spread,
            percentile(&mut slopes, 0.5),
            percentile(&mut slopes, 0.95),
            steep * 100.0,
        )
    }

    /// Validation 1: the two noise sums on this crate's noise land on the constants the unit
    /// maps assume. The percentiles are the stationary ones (pooled over seeds): one seed's
    /// 2048 m tile of a 600 m feature has only 3.4 base-octave cells, so its own percentiles
    /// scatter by ±0.08 (seed 42: f −0.347 / 0.256).
    #[test]
    fn the_noise_sums_fill_their_unit_ranges() {
        let (mut f, mut r) = (Vec::new(), Vec::new());
        for seed in 0..32u64 {
            for j in 0..96 {
                for i in 0..96 {
                    let p = Vec2::new(i as f32, j as f32) * 0.31 + Vec2::splat(0.1);
                    let (a, b) = noise_sums(p, 1.0, 8, seed * 7 + 1);
                    f.push(a);
                    r.push(b);
                }
            }
        }
        let f_lo = percentile(&mut f, 0.005);
        let f_hi = percentile(&mut f, 0.995);
        let r_lo = percentile(&mut r, 0.005);
        let r_hi = percentile(&mut r, 0.995);
        println!("f {f_lo:.3} {f_hi:.3}  r {r_lo:.3} {r_hi:.3}");
        assert!(
            (f_lo + F_HALF).abs() < 0.03 && (f_hi - F_HALF).abs() < 0.03,
            "f {f_lo} {f_hi}"
        );
        assert!(
            (r_lo - R_LOW).abs() < 0.03 && (r_hi - (R_LOW + R_SPAN)).abs() < 0.03,
            "r {r_lo} {r_hi}"
        );
    }

    /// Validation 1: per preset, the percentile spread of the heights, the slope percentiles and
    /// the share of rock-steep ground, re-blessed once on this crate's noise (Deviation 11; the
    /// note's numpy port, a 2D gradient noise, is about 30 % steeper).
    #[test]
    fn the_presets_have_the_measured_statistics() {
        // (preset, spread m, slope p50, slope p95, share above 40° in %), seed 42, 2 m cells.
        let table = [
            (ProceduralTerrain::hills(42), 91.8, 12.4, 25.5, 0.0),
            (ProceduralTerrain::mountains(42), 549.5, 41.1, 61.9, 52.6),
            (ProceduralTerrain::dunes(42), 39.9, 9.1, 18.6, 0.0),
            (ProceduralTerrain::plains(42), 13.7, 1.4, 2.9, 0.0),
        ];
        let mut misses = Vec::new();
        for (preset, spread, p50, p95, steep) in table {
            let (s, m, h, e) = stats(&preset);
            println!(
                "R {} F {}: spread {s:.1} (want {spread}) slope p50 {m:.1} ({p50}) p95 {h:.1} ({p95}) >40° {e:.2} % ({steep})",
                preset.relief_m, preset.feature_size_m
            );
            if (s / spread - 1.0).abs() >= 0.10
                || (m / p50 - 1.0).abs() >= 0.15
                || (h / p95 - 1.0).abs() >= 0.15
                || (e - steep).abs() >= 3.0
            {
                misses.push(format!("R {} F {}", preset.relief_m, preset.feature_size_m));
            }
        }
        assert!(misses.is_empty(), "{misses:?}");
    }

    /// Validation 2: the generator is frozen on the CPU mirror.
    #[test]
    fn the_terrain_is_frozen() {
        let surface = generate_surface(&ProceduralTerrain::hills(42), UVec2::splat(256), 8.0);
        let mm: Vec<i32> = surface
            .iter()
            .map(|h| (h * 1000.0).round() as i32)
            .collect();
        let hash = fnv1a(bytemuck::cast_slice(&mm));
        println!("the_terrain_is_frozen: {hash:#018x}");
        assert_eq!(
            hash, GOLDEN_HILLS_256,
            "the generator changed: re-bless once, deliberately"
        );
    }

    const GOLDEN_HILLS_256: u64 = 0x6f04_892c_64bc_b5eb;

    /// Validation 7: the layer weights sum to one for any detail, pick the plan's layers without
    /// detail, and the detail never moves a layer that is fully on or off.
    #[test]
    fn the_layer_weights_sum_to_one_and_pick_the_right_layer() {
        for a in (0..90).step_by(1) {
            for s in [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0, 4.0] {
                for d in [0.0, 0.25, 0.5, 0.75, 1.0] {
                    let w = layer_weights(a as f32, s, d);
                    assert!(
                        (w.iter().sum::<f32>() - 1.0).abs() < 1e-6,
                        "{a}° {s} m detail {d}: {w:?}"
                    );
                    assert!(w.iter().all(|x| (0.0..=1.0).contains(x)));
                }
            }
        }
        assert!(layer_weights(45.0, 0.0, 0.5)[0] >= 0.99);
        assert!(layer_weights(20.0, 1.0, 0.5)[1] >= 0.99);
        assert_eq!(layer_weights(10.0, 0.0, 0.5)[3], 1.0);
        let soil = layer_weights(31.0, 0.0, 0.5)[2];
        assert!((0.3..0.7).contains(&soil), "soil at 31°: {soil}");
        // Fully on or off stays put under any detail: rock at 50°, grass at 10°.
        for d in [0.0, 0.3, 0.7, 1.0] {
            assert_eq!(layer_weights(50.0, 0.0, d)[0], 1.0);
            assert_eq!(layer_weights(10.0, 0.0, d)[3], 1.0);
        }
        // Inside a transition the detail spans the whole range: at 40° rock goes 0 → 1.
        assert_eq!(layer_weights(40.0, 0.0, 0.0)[0], 0.0);
        assert_eq!(layer_weights(40.0, 0.0, 1.0)[0], 1.0);
    }

    /// Validation 5: on the plane `h = 0.5 x` the central-difference slope is atan(0.5) at every
    /// cell, edges included, and a flat field has slope 0.
    #[test]
    fn the_slope_of_a_plane_is_exact_up_to_the_edge() {
        let size = UVec2::new(16, 8);
        let plane: Vec<f32> = (0..size.y)
            .flat_map(|_| (0..size.x).map(|i| 0.5 * (i as f32 + 0.5) * 2.0))
            .collect();
        for j in 0..size.y {
            for i in 0..size.x {
                let a = slope_deg(&plane, size, 2.0, i, j);
                assert!((a - 26.565).abs() < 0.01, "cell {i},{j}: {a}");
            }
        }
        assert_eq!(slope_deg(&vec![3.0; 128], size, 2.0, 0, 0), 0.0);
    }
}
