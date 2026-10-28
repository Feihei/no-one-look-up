// bevy_elements::dust::surface_wind — the terrain-modified wind shared by aeolian transport and
// snow drift (docs/research/dust.md "Surface wind", docs/research/snow-cover.md "Drift").
//
//     import bevy_elements::dust::surface_wind::{venturi_speedup, shadow_factor};
//
// The surface wind of a cell is `u10 · venturi_speedup(z − mean of the 50 m ring) ·
// shadow_factor(steepest upwind slope)`. WGSL cannot pass a storage buffer to a function, so the
// caller does the surface taps (the ring of radius `RING_M`, the upwind march of `SHADOW_MARCH_M`)
// and hands the two scalars in. `src/dust/model.rs` mirrors every function here in its tests.

/// Venturi coefficient k_W, 1/m (Paris et al. 2019 §4.2).
const VENTURI_K: f32 = 0.005;
/// The speed-up is held to this range.
const SPEEDUP_MIN: f32 = 0.5;
const SPEEDUP_MAX: f32 = 2.0;
/// Radius of the ring whose mean surface height the speed-up is measured from, m.
const RING_M: f32 = 50.0;
/// How far upwind the shadow march looks, m (Paris et al. 2019 §4.3).
const SHADOW_MARCH_M: f32 = 10.0;
/// The upwind surface's elevation angle at which the wind is fully sheltered, and at which
/// shelter starts: 15° and 10°.
const SHADOW_FULL: f32 = 0.2617994;
const SHADOW_NONE: f32 = 0.17453293;

/// Speed-up of the wind over a surface `above_mean` metres above the mean of the ring around it.
fn venturi_speedup(above_mean: f32) -> f32 {
    return clamp(1.0 + VENTURI_K * above_mean, SPEEDUP_MIN, SPEEDUP_MAX);
}

/// Share of the wind that reaches a cell whose steepest upwind rise (a tangent, over the march
/// radius) is `slope`: 1 below 10°, 0 above 15°, linear in the angle between.
fn shadow_factor(slope: f32) -> f32 {
    let beta = atan(slope);
    return clamp((SHADOW_FULL - beta) / (SHADOW_FULL - SHADOW_NONE), 0.0, 1.0);
}

/// Bulk-volume sand flux per unit width, m²/s, that a surface wind `u` carries at saturation:
/// `k u² (u − u_t)`, zero at and below the threshold.
fn flux_capacity(u: f32, k: f32, threshold: f32) -> f32 {
    return k * u * u * max(u - threshold, 0.0);
}
