//! CPU mirrors of the water shaders' math and the note's validation targets 1–10
//! (`docs/research/water-rendering.md`). The GPU ones (7 on the real kernel, 11, 12) are in
//! `tests/gpu_water.rs`.

use bevy::math::{IVec2, Vec2, Vec3};
use bevy::transform::components::Transform;

use crate::gpu::noise_mirror::{fbm3_periodic, perlin3_periodic};
use crate::hydrology::water::{
    NORMAL_OCTAVES, NORMAL_RMS_SLOPE, NORMAL_SLOPE_GAIN, NORMAL_TEXELS, WATER_POOL_CAPACITY,
    WaterSettings, ripple_slope,
};
use crate::hydrology::water_gpu::octaves_kept;

const F0: f32 = 0.02;
const FOAM_SPEED_RAMP: f32 = 0.1;
const FOAM_THICKNESS_RAMP: f32 = 0.03;
const DISCARD_DEPTH: f32 = 0.003;
const FADE_DEPTH: f32 = 0.03;

/// `beer_lambert` of `water.wesl`.
fn beer_lambert(sigma: Vec3, t: f32) -> Vec3 {
    Vec3::new(
        (-sigma.x * t).exp(),
        (-sigma.y * t).exp(),
        (-sigma.z * t).exp(),
    )
}

/// `fresnel` of `water.wesl`: Schlick's approximation with `F₀ = 0.02`.
fn fresnel(cos_v: f32) -> f32 {
    F0 + (1.0 - F0) * (1.0 - cos_v.clamp(0.0, 1.0)).powi(5)
}

/// `refracted_thickness` of `water.wesl`.
fn refracted_thickness(t_ray: f32, cos_v: f32, n: f32) -> f32 {
    let cos_r = (1.0 - (1.0 - cos_v * cos_v) / (n * n)).max(1e-4).sqrt();
    t_ray * cos_v / cos_r
}

/// Bevy's `F₀ = 0.16 r²` (`pbr_functions.wesl`).
fn bevy_f0(reflectance: f32) -> f32 {
    0.16 * reflectance * reflectance
}

/// `depth_ndc_to_view_z` of Bevy for a reverse-Z infinite perspective projection.
fn depth_ndc_to_view_z(ndc: f32, near: f32) -> f32 {
    -near / ndc
}

/// The thickness along the view ray the fragment shader computes from the two view-space depths
/// and the ray's z.
fn thickness(z_fragment: f32, z_scene: f32, ray_z: f32) -> f32 {
    ((z_fragment - z_scene) / (-ray_z).max(1e-3)).clamp(0.0, 1000.0)
}

fn smoothstep(lo: f32, hi: f32, x: f32) -> f32 {
    let t = ((x - lo) / (hi - lo)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn phase_weight(p: f32) -> f32 {
    1.0 - (2.0 * p - 1.0).abs()
}

/// The two layers' phases and weights for time `t` (s), phase noise `psi` and cycle `period`.
fn flow_phases(t: f32, psi: f32, period: f32) -> ([f32; 2], [f32; 2]) {
    let phase = t / period + psi;
    let p = [phase.fract(), (phase + 0.5).fract()];
    (p, [phase_weight(p[0]), phase_weight(p[1])])
}

/// The flow offset per unit phase in tiles, its length limited to `limit`.
fn flow_offset(velocity: Vec2, period: f32, tile: f32, limit: f32) -> Vec2 {
    (velocity * period / tile).clamp_length_max(limit)
}

fn foam_from_speed(speed: f32, threshold: f32) -> f32 {
    smoothstep(
        threshold * (1.0 - FOAM_SPEED_RAMP),
        threshold * (1.0 + FOAM_SPEED_RAMP),
        speed,
    )
}

fn foam_from_thickness(t: f32, threshold: f32) -> f32 {
    1.0 - smoothstep(
        threshold - FOAM_THICKNESS_RAMP,
        threshold + FOAM_THICKNESS_RAMP,
        t,
    )
}

fn edge_alpha(depth: f32) -> f32 {
    smoothstep(DISCARD_DEPTH, FADE_DEPTH, depth)
}

/// A small deterministic generator for the property tests.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 40) as f32 / (1u64 << 24) as f32
    }

    fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.next()
    }
}

fn close(a: f32, b: f32, tolerance: f32) -> bool {
    (a - b).abs() <= tolerance
}

/// Validation 1: Beer–Lambert with the plan's σ, and the reason for two passes.
#[test]
fn beer_lambert_matches_the_computed_table() {
    let sigma = WaterSettings::default().absorption;
    let table = [
        (0.5, [0.799, 0.970, 0.985]),
        (1.0, [0.638, 0.942, 0.970]),
        (2.0, [0.407, 0.887, 0.942]),
        (5.0, [0.105, 0.741, 0.861]),
    ];
    for (t, expected) in table {
        let got = beer_lambert(sigma, t);
        for (channel, (g, e)) in [got.x, got.y, got.z].into_iter().zip(expected).enumerate() {
            assert!(close(g, e, 1e-3), "T({t})[{channel}] = {g}, expected {e}");
        }
    }
    let two = beer_lambert(sigma, 2.0);
    assert!(close(two.x / two.z, 0.432, 1e-3), "red/blue at 2 m");
    // A single blend scales every channel by one scalar: with the luminance-weighted mean σ̄ the
    // ratio is exactly 1 whatever the depth.
    let mean = 0.2126 * sigma.x + 0.7152 * sigma.y + 0.0722 * sigma.z;
    assert!(close(mean, 0.141, 1e-3), "σ̄ = {mean}");
    let scalar = beer_lambert(Vec3::splat(mean), 2.0);
    assert_eq!(scalar.x / scalar.z, 1.0);
}

/// Validation 2: the defaults against the measured spectra.
#[test]
fn absorption_defaults_exceed_the_pure_water_bound() {
    let sigma = WaterSettings::default().absorption;
    // Pure water in display primaries (the note's table, Pope & Fry 1997 weighted by the CIE
    // matching functions): the lower bound.
    let pure = Vec3::new(0.30, 0.05, 0.004);
    assert!(sigma.x > pure.x && sigma.y > pure.y && sigma.z > pure.z);
    // Pope & Fry (1997) via `omlc.org/spectra/water/data/pope97.txt`, in m⁻¹: (nm, absorption).
    let pope = [
        (450.0, 0.0092),
        (465.0, 0.0101),
        (500.0, 0.0204),
        (550.0, 0.0565),
        (600.0, 0.2224),
        (620.0, 0.2755),
        (650.0, 0.3400),
        (680.0, 0.4650),
        (700.0, 0.6240),
    ];
    let minimum = (417.5, 0.00442);
    assert!(pope.windows(2).all(|w| w[1].1 > w[0].1), "rises with λ");
    assert!(pope.iter().all(|&(_, a)| a > minimum.1));
    // The plan's red lies inside the table's 650–700 nm span, its green is the 550 nm sample
    // and its blue is above the 450 nm sample (natural water absorbs more blue than pure water).
    assert!((0.34..=0.624).contains(&sigma.x));
    assert!(close(sigma.y, 0.0565, 0.0565 * 0.1));
    assert!(sigma.z > pope[0].1);
}

/// Validation 3: Fresnel, reflectance and roughness.
#[test]
fn fresnel_and_reflectance_follow_the_table() {
    assert!(close(bevy_f0(super::WATER_REFLECTANCE), 0.0200, 1e-4));
    assert!(close(fresnel(1.0), 0.0200, 5e-4));
    assert!(close(fresnel(60f32.to_radians().cos()), 0.0506, 5e-4));
    assert!(close(fresnel(80f32.to_radians().cos()), 0.398, 5e-4));
    // The stored roughness is the one Bevy's lighting clamps to, so it is what renders.
    assert!(close(super::WATER_ROUGHNESS, 0.089, 1e-6));
}

/// Validation 4: the thickness reconstruction on a flat bed 2 m under a flat surface.
#[test]
fn thickness_is_reconstructed_from_the_depth_prepass() {
    let near = 0.1;
    let n = 1.33;
    let depth = 2.0;
    for (angle_deg, expected_ray) in [(0.0f32, 2.0), (60.0, 4.0)] {
        let angle = angle_deg.to_radians();
        // The camera looks along the pixel's ray (the centre pixel): the view-space z is the
        // distance along it. The surface is hit at `s`, the bed `depth / cos` further.
        let s = 37.0;
        let s_bed = s + depth / angle.cos();
        // Through the reverse-Z reconstruction a prepass stores.
        let z_f = depth_ndc_to_view_z(near / s, near);
        let z_s = depth_ndc_to_view_z(near / s_bed, near);
        let t_ray = thickness(z_f, z_s, -1.0);
        assert!(
            close(t_ray, expected_ray, expected_ray * 0.01),
            "{angle_deg}°: t_ray {t_ray}"
        );
        let t_eff = refracted_thickness(t_ray, angle.cos(), n);
        let expected_eff = if angle_deg == 0.0 { 2.0 } else { 2.635 };
        assert!(
            close(t_eff, expected_eff, expected_eff * 0.01),
            "{angle_deg}°: t_eff {t_eff}"
        );
    }
    // Off the optical axis the thickness is along the ray, not along the view z.
    let off_axis = 20f32.to_radians();
    let ray_z = -off_axis.cos();
    let t = thickness(
        -30.0 * off_axis.cos(),
        -(30.0 + 4.0) * off_axis.cos(),
        ray_z,
    );
    assert!(close(t, 4.0, 0.04), "off axis {t}");
    for (deg, ratio) in [(0.0f32, 1.000), (30.0, 0.935), (60.0, 0.659), (75.0, 0.377)] {
        let got = refracted_thickness(1.0, deg.to_radians().cos(), 1.33);
        assert!(close(got, ratio, 1e-3), "{deg}°: {got}");
    }
}

/// Validation 5: the absorb and surface passes composite to the radiance model.
#[test]
fn absorb_and_surface_composite_to_the_radiance_model() {
    let mut rng = Lcg(5);
    for _ in 0..1000 {
        let f = rng.range(0.02, 0.6);
        let t = Vec3::new(rng.next(), rng.next(), rng.next());
        let foam = rng.next();
        let a = rng.next();
        let dst = Vec3::new(
            rng.range(0.0, 5.0),
            rng.range(0.0, 5.0),
            rng.range(0.0, 5.0),
        );
        let s = Vec3::new(rng.next(), rng.next(), rng.next());
        // The absorb shader writes (X·a, a): `out = dst·src.rgb + (1 − src.a)·dst`.
        let x = (1.0 - f) * t * (1.0 - foam);
        let multiply = |dst: Vec3| dst * (x * a) + (1.0 - a) * dst;
        // The surface shader writes (rgb·a, 0) with `Add`: `out = dst + src.rgb`.
        let composite = multiply(dst) + a * s;
        // The model with edge alpha a: a·model + (1 − a)·dst.
        let model = dst * (1.0 - foam) * (1.0 - f) * t + s;
        let full = a * model + (1.0 - a) * dst;
        assert!((composite - full).abs().max_element() < 1e-4);
        if a == 1.0 {
            assert!((composite - model).abs().max_element() < 1e-4);
        }
        // Full foam hides the bed completely.
        let covered = dst * (x * 0.0) + (1.0 - 1.0) * dst;
        assert_eq!(covered, Vec3::ZERO);
        let foamed = (1.0 - f) * t * (1.0 - 1.0);
        assert_eq!(foamed, Vec3::ZERO);
    }
    // a = 0 leaves the destination untouched; a = 1 with no foam is the plain filter.
    let dst = Vec3::new(1.0, 2.0, 3.0);
    let x = Vec3::new(0.2, 0.5, 0.9);
    assert_eq!(dst * (x * 0.0) + (1.0 - 0.0) * dst, dst);
    assert_eq!(dst * (x * 1.0) + (1.0 - 1.0) * dst, dst * x);
}

/// Validation 6: the flow phases.
#[test]
fn flow_phases_hide_the_restart() {
    let mut rng = Lcg(6);
    let settings = WaterSettings::default();
    let period = settings.flow_cycle_s;
    for _ in 0..10_000 {
        let (t, psi) = (rng.range(0.0, 10.0), rng.next());
        let (p, w) = flow_phases(t, psi, period);
        assert!(close(w[0] + w[1], 1.0, 1e-6), "weights sum {}", w[0] + w[1]);
        for i in 0..2 {
            assert!(
                p[i] != 0.0 || w[i] == 0.0,
                "layer {i} restarts with weight {}",
                w[i]
            );
        }
    }
    // At the restart of a layer its weight is exactly zero.
    let (_, w) = flow_phases(period * 3.0, 0.0, period);
    assert_eq!(w[0], 0.0);
    let (_, w) = flow_phases(period * 3.5, 0.0, period);
    assert_eq!(w[1], 0.0);
    // The offset saturates at D tiles for any speed (at v_max = D L / T = 1.78 m/s).
    let (tile, limit) = (settings.tile_m, settings.flow_distortion);
    for speed in [0.0f32, 0.5, 1.0, 1.7, 1.8, 3.0, 50.0] {
        let o = flow_offset(Vec2::new(speed, 0.0), period, tile, limit);
        assert!(o.length() <= limit + 1e-6, "{speed} m/s: {}", o.length());
    }
    assert!(close(limit * tile / period, 1.78, 0.01));
    assert!(close(
        flow_offset(Vec2::new(1.78, 0.0), period, tile, limit).length(),
        limit,
        0.003
    ));
    // The layer's coordinate `uv − o·p` moves along +v as the phase advances.
    let v = Vec2::new(1.0, -0.5);
    let o = flow_offset(v, period, tile, limit);
    let uv = Vec2::new(0.3, 0.7);
    let earlier = uv - o * 0.2;
    let later = uv - o * 0.4;
    assert!(
        (later - earlier).dot(v) < 0.0,
        "the coordinate moves against +v"
    );
    // The pattern sampled at the coordinate therefore moves with +v: a feature at `f` shows at
    // the position `x` with `x/L − o·p = f`, i.e. `x = L (f + o·p)`.
    assert!((o * 0.4 - o * 0.2).dot(v) > 0.0);
    // No flow, a static normal: the offset is zero and the layer coordinate never changes.
    assert_eq!(flow_offset(Vec2::ZERO, period, tile, limit), Vec2::ZERO);
}

/// Height amplitude ratio per octave of the normal map: 0.5, so every octave's slope is the
/// same (a flat slope spectrum; `docs/research/water-surface.md`, Model 1). Mirrors
/// `OCTAVE_GAIN` in `water_normals.wesl`.
const OCTAVE_GAIN: f32 = 0.5;
/// Lattice cells of the base octave across the tile: 16 (0.25 m wavelength at the 8 m tile).
/// Mirrors `BASE_CELLS` in `water_normals.wesl`.
const BASE_CELLS: f32 = 16.0;

/// Height of the normal map at integer texel `t` of the full-resolution lattice (mirror of
/// `height` in `water_normals.wesl`).
fn map_height(t: IVec2, octaves: u32) -> f32 {
    let n = NORMAL_TEXELS as i32;
    let w = IVec2::new(t.x & (n - 1), t.y & (n - 1));
    let u = (w.as_vec2() + 0.5) / NORMAL_TEXELS as f32;
    let mut h = 0.0;
    let mut amplitude = 1.0;
    for o in 0..octaves {
        let cells = BASE_CELLS * (1u32 << o) as f32;
        h += amplitude
            * perlin3_periodic(
                Vec3::new(u.x * cells, u.y * cells, 0.5 + o as f32),
                cells as i32,
            );
        amplitude *= OCTAVE_GAIN;
    }
    h
}

/// The slope `(-K ∂h/∂u, -K ∂h/∂v)` the kernel stores for texel `id` of mip `level`.
fn map_slope(id: IVec2, level: u32, gain: f32) -> Vec2 {
    let edge = NORMAL_TEXELS >> level;
    let stride = (NORMAL_TEXELS / edge) as i32;
    let centre = id * stride + IVec2::splat(stride / 2);
    let octaves = octaves_kept(level).0;
    let hx = map_height(centre + IVec2::X, octaves) - map_height(centre - IVec2::X, octaves);
    let hy = map_height(centre + IVec2::Y, octaves) - map_height(centre - IVec2::Y, octaves);
    Vec2::new(-gain * hx, -gain * hy)
}

/// rms of the slope over the texels of a mip with the given sample stride in texels.
fn rms_slope(level: u32, gain: f32, every: u32) -> f32 {
    let edge = (NORMAL_TEXELS >> level) as i32;
    let (mut sum, mut count) = (0.0f64, 0u32);
    let mut y = 0;
    while y < edge {
        let mut x = 0;
        while x < edge {
            sum += f64::from(map_slope(IVec2::new(x, y), level, gain).length_squared());
            count += 1;
            x += every as i32;
        }
        y += every as i32;
    }
    (sum / f64::from(count)).sqrt() as f32
}

/// The constant `NORMAL_SLOPE_GAIN` gives mip 0 the Cox & Munk rms slope.
#[test]
fn normal_gain_gives_the_cox_munk_slope() {
    let unit = rms_slope(0, 1.0, 7);
    println!("rms slope of mip 0 at gain 1: {unit}");
    let got = NORMAL_SLOPE_GAIN * unit;
    assert!(
        close(got, NORMAL_RMS_SLOPE, 0.004),
        "rms {got}, wanted {NORMAL_RMS_SLOPE} (gain {})",
        NORMAL_RMS_SLOPE / unit
    );
}

/// Validation 7 (CPU half): the slope of a mip level is the kept octaves' share; the 1-texel mip
/// is flat. The seam and the real kernel are checked on the GPU in `tests/gpu_water.rs`.
#[test]
fn mips_keep_the_expected_share_of_the_slope() {
    let k = NORMAL_SLOPE_GAIN;
    let mip0 = rms_slope(0, k, 7);
    let mip3 = rms_slope(3, k, 1);
    let ratio = mip3 / mip0;
    println!("mip 3 / mip 0 rms slope: {ratio}");
    // Mip 3 keeps one octave of the three (16, 8 and 4 texels over the 8 m tile; a mip 3 texel
    // is 8 of the full-resolution ones, so the 4-texel band is gone). The slope spectrum is
    // flat, so the kept octave carries 1/3 of the slope energy
    // (`docs/research/water-surface.md`, Model 1); the mip's own central difference runs over
    // 8 texels, which carries a little more than the octave's own share.
    assert!(close(ratio, 0.594, 0.02), "mip 3 / mip 0 = {ratio}");
    // The mip's own central difference spans 8 of the full-resolution texels, so it is a
    // low-pass of the kept band: it keeps less of it than the band's own energy share
    // (0.76 for one octave of three, a rms of 0.87) and lands at 0.59. The closed form of the
    // band share is the bound the measurement must sit under.
    let weights = |n: u32| (0..n).map(|o| 0.25f32.powi(o as i32)).sum::<f32>();
    let share = (weights(1) / weights(3)).sqrt();
    assert!(ratio < share, "the mip keeps more than its band: {ratio} against {share}");
    assert!(ratio > 0.5, "the mip lost its band: {ratio}");
    let flat = map_slope(IVec2::ZERO, 9, k);
    assert!(flat.length() < 0.01, "mip 9 slope {flat}");
    // The foam alpha is centred.
    let mean = (0..4096)
        .map(|i| {
            let u = Vec2::new((i % 64) as f32, (i / 64) as f32 + 0.5) / 64.0;
            0.5 + 0.5
                * (2.5 * fbm3_periodic(Vec3::new(u.x * 4.0, u.y * 4.0, 7.3), 4, 4)).clamp(-1.0, 1.0)
        })
        .sum::<f32>()
        / 4096.0;
    assert!((0.3..=0.7).contains(&mean), "foam alpha mean {mean}");
    assert_eq!(NORMAL_OCTAVES, 3);
}

/// Validation 7: the height field tiles, so the map has no seam.
#[test]
fn the_normal_map_tiles_without_a_seam() {
    let n = NORMAL_TEXELS as i32;
    let k = NORMAL_SLOPE_GAIN;
    let mean_diff = |a: &dyn Fn(i32) -> Vec2, b: &dyn Fn(i32) -> Vec2| {
        (0..n)
            .map(|i| (a(i) - b(i)).abs().element_sum())
            .sum::<f32>()
            / n as f32
    };
    let slope = |x: i32, y: i32| map_slope(IVec2::new(x, y), 0, k);
    // Across the seam (column 511 to 0) against across the interior (columns 255 to 256).
    let seam_x = mean_diff(&|y| slope(n - 1, y), &|y| slope(0, y));
    let interior_x = mean_diff(&|y| slope(255, y), &|y| slope(256, y));
    let seam_y = mean_diff(&|x| slope(x, n - 1), &|x| slope(x, 0));
    let interior_y = mean_diff(&|x| slope(x, 255), &|x| slope(x, 256));
    assert!(
        seam_x <= 1.5 * interior_x,
        "columns: {seam_x} vs {interior_x}"
    );
    assert!(seam_y <= 1.5 * interior_y, "rows: {seam_y} vs {interior_y}");
}

/// Validation 8: the foam thresholds.
#[test]
fn foam_follows_the_plan_thresholds() {
    let s = WaterSettings::default();
    assert_eq!(foam_from_speed(0.5, s.foam_speed_ms), 0.0);
    assert!(close(foam_from_speed(1.5, s.foam_speed_ms), 0.5, 1e-6));
    assert_eq!(foam_from_speed(1.8, s.foam_speed_ms), 1.0);
    assert_eq!(foam_from_thickness(0.02, s.foam_thickness_m), 1.0);
    assert!(close(
        foam_from_thickness(0.05, s.foam_thickness_m),
        0.5,
        1e-6
    ));
    assert_eq!(foam_from_thickness(0.08, s.foam_thickness_m), 0.0);
    // Froude 1 at 1.5 m/s: v²/g.
    assert!(close(1.5f32 * 1.5 / 9.81, 0.229, 5e-4));
}

/// Validation 9: the depth fade.
#[test]
fn the_depth_fade_is_a_smoothstep() {
    assert_eq!(edge_alpha(0.003), 0.0);
    assert!(close(edge_alpha(0.0165), 0.5, 1e-6));
    assert_eq!(edge_alpha(0.03), 1.0);
    assert_eq!(edge_alpha(1.0), 1.0);
    // The shader discards below the lowest depth: whatever fade value lies there is never shown.
    assert_eq!(edge_alpha(0.002), 0.0);
}

/// Validation 10: the surface item of a patch sorts after its absorb item, in Bevy's own
/// rangefinder.
#[test]
fn the_surface_pass_sorts_after_its_absorb_pass() {
    use bevy::render::render_phase::ViewRangefinder3d;

    let mut rng = Lcg(10);
    // `Transparent3d` sorts ascending by `distance + depth_bias`; the surface material's
    // `depth_bias` is 1.
    let key = |range: &ViewRangefinder3d, centre: Vec3, bias: f32| range.distance(&centre) + bias;
    for _ in 0..10_000 {
        let eye = Vec3::new(
            rng.range(-500.0, 500.0),
            rng.range(1.0, 300.0),
            rng.range(-500.0, 500.0),
        );
        let target = Vec3::new(rng.range(-500.0, 500.0), 0.0, rng.range(-500.0, 500.0));
        let world_from_view = Transform::from_translation(eye)
            .looking_at(target, Vec3::Y)
            .compute_affine();
        let range = ViewRangefinder3d::from_world_from_view(&world_from_view);
        let a = Vec3::new(
            rng.range(-500.0, 500.0),
            rng.range(0.0, 50.0),
            rng.range(-500.0, 500.0),
        );
        let absorb = key(&range, a, 0.0);
        let surface = key(&range, a, 1.0);
        assert!(
            surface > absorb,
            "the surface item sorts after its absorb item"
        );
        // Another patch whose centre is more than the bias away in view depth is not
        // interleaved with this one.
        let b = Vec3::new(
            rng.range(-500.0, 500.0),
            rng.range(0.0, 50.0),
            rng.range(-500.0, 500.0),
        );
        let (b_absorb, b_surface) = (key(&range, b, 0.0), key(&range, b, 1.0));
        if (b_absorb - absorb).abs() > 1.0 {
            let a_first = absorb < b_absorb;
            let (first_end, second_start) = if a_first {
                (surface, b_absorb)
            } else {
                (b_surface, absorb)
            };
            assert!(first_end < second_start, "patches interleave");
        }
    }
}

/// Validation 13 (CPU half): a patch is drawn when and only when its deepest cell holds at least
/// 3 mm, one pair of entities per patch, and the pool's capacity is the note's 192.
#[test]
fn only_patches_with_water_are_drawn() {
    use crate::terrain::lod::{BoundsTree, LodConfig, Patch, select_patches};
    let lod = LodConfig::new(Vec2::splat(1024.0), 4.0);
    let dims = (1024.0 / (32.0 * 4.0)) as u32;
    // A lake in the leaf (2, 3) and a film of 2 mm in (5, 5).
    let depth = |x: u32, z: u32| match (x, z) {
        (2, 3) => 1.5,
        (5, 5) => 0.002,
        _ => 0.0,
    };
    let leaves = (0..dims * dims).map(|i| {
        let d = depth(i % dims, i / dims);
        (d, d)
    });
    let tree = BoundsTree::from_leaves(&lod, bevy::math::UVec2::splat(dims), leaves);
    let flat = BoundsTree::uniform(&lod, 0.0, 0.0);
    let selected = select_patches(&lod, &flat, Vec3::new(160.0, 5.0, 224.0));
    let wet = super::wet_patches(&selected, &tree);
    assert!(!wet.is_empty() && wet.len() < selected.len());
    let leaf = |p: &Patch| p.level == 0 && p.ix == 2 && p.iz == 3;
    assert!(wet.iter().any(leaf), "the lake's leaf is drawn");
    assert!(
        wet.iter().all(|p| {
            let size = 1u32 << p.level;
            (p.ix * size..(p.ix + 1) * size).contains(&2)
                && (p.iz * size..(p.iz + 1) * size).contains(&3)
        }),
        "only the patches over the lake are drawn; the 2 mm film is not"
    );
    assert_eq!(WATER_POOL_CAPACITY, 192);
    // The ripple slope follows Cox & Munk.
    assert!(close(ripple_slope(4.0), 0.153, 1e-3));
    assert!(close(ripple_slope(10.0), 0.233, 1e-3));
}
