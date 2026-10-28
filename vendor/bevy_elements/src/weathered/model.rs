//! The CPU side of the weathered shading: what is computed once per frame and handed to the
//! shaders in `ElementsGlobals`. The per-fragment functions are in `surface_cover.wesl` and
//! transliterated in `mirror.rs` for the tests.

use crate::climate::Weather;

/// Liquid precipitation at which the ripple and streak intensity reaches 1, mm/h: the rate at
/// which the rain particle count saturates (`precipitation/model.rs`, `min(1, R / 20)`).
pub(crate) const FULL_RAIN_MM_H: f32 = 20.0;

/// Lagarde (2013), "Water drop 2b": the four ripple layers' clock multipliers and offsets, and
/// the 1.6 that turns the product into a period of 0.625 s.
pub(crate) const RIPPLE_TIME_MUL: [f64; 4] = [1.0, 0.85, 0.93, 1.13];
pub(crate) const RIPPLE_TIME_ADD: [f64; 4] = [0.0, 0.2, 0.45, 0.7];
pub(crate) const RIPPLE_RATE: f64 = 1.6;

/// The streak phase wraps here, s. The streak noise drifts 0.5 lattice units per second and its
/// lattice repeats every 4096 units (`noise.wesl`), so 8192 s is exactly one lattice period and
/// the wrap is invisible.
pub(crate) const STREAK_WRAP_S: f64 = 8192.0;

/// Rain intensity `f = min(1, R_liquid / 20)`, 0 when only snow falls.
pub(crate) fn rain_intensity(weather: &Weather) -> f32 {
    let liquid = weather.precipitation_mm_h.max(0.0) * (1.0 - weather.snow_fraction());
    (liquid / FULL_RAIN_MM_H).min(1.0)
}

/// Phase of the four ripple layers at `seconds` of the virtual clock:
/// `frac((t · TimeMul_k + TimeAdd_k) · 1.6)`, in f64 so it stays continuous however long the
/// game runs (a wrapped f32 clock jumps, because the multipliers are not integers).
pub(crate) fn ripple_phases(seconds: f64) -> [f32; 4] {
    std::array::from_fn(|k| {
        let x = (seconds * RIPPLE_TIME_MUL[k] + RIPPLE_TIME_ADD[k]) * RIPPLE_RATE;
        x.rem_euclid(1.0) as f32
    })
}

/// The streak phase at `seconds` of the virtual clock, in `[0, STREAK_WRAP_S)`.
pub(crate) fn streak_phase(seconds: f64) -> f32 {
    seconds.rem_euclid(STREAK_WRAP_S) as f32
}
