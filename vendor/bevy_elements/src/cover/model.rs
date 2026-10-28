//! The ground-cover model's shared numbers and the scheduling that turns clock time into kernel
//! steps. The kernels' equations are in `snow.wesl` and `shelter.wesl`; `mirror.rs` is their CPU
//! transliteration used by the tests.

/// Snow depth at which the ground reads as fully covered, m.
pub(crate) const FULL_COVER_DEPTH_M: f32 = 0.05;
/// The longest game time one kernel step integrates, s.
pub(crate) const MAX_STEP_S: f64 = 600.0;
/// The most steps one frame runs; a clock jump that needs more is cut.
pub(crate) const MAX_STEPS_PER_FRAME: u32 = 144;
/// The longest game time kept waiting for the GPU, s (`weathering.rs` clamps to 24 h too).
pub(crate) const MAX_PENDING_S: f64 = 86_400.0;
/// Largest cover readback edge: a larger simulation grid is box-averaged down by 2 first.
pub(crate) const READBACK_MAX_CELLS: u32 = 1024;
/// How far from the camera the map is averaged into `SurfaceWeathering`, m.
pub(crate) const SURFACE_WEATHERING_RADIUS_M: f32 = 50.0;

/// Hermite smoothstep.
pub(crate) fn smoothstep(lo: f32, hi: f32, x: f32) -> f32 {
    let t = ((x - lo) / (hi - lo)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Fraction of the ground under snow of `depth_m`: `smoothstep(0, 0.05, d)`.
pub(crate) fn snow_coverage(depth_m: f32) -> f32 {
    smoothstep(0.0, FULL_COVER_DEPTH_M, depth_m)
}

/// How many kernel steps `pending_s` of game time needs and how long each is: steps of at most
/// [`MAX_STEP_S`], never more than [`MAX_STEPS_PER_FRAME`]. `None` for no time.
pub(crate) fn plan_steps(pending_s: f64) -> Option<(u32, f32)> {
    if pending_s <= 0.0 {
        return None;
    }
    let n = (pending_s / MAX_STEP_S)
        .ceil()
        .clamp(1.0, f64::from(MAX_STEPS_PER_FRAME));
    Some((n as u32, (pending_s.min(n * MAX_STEP_S) / n) as f32))
}

/// A quantised value for change detection: the shelter mask is rebuilt only when the fall
/// directions move by more than a step.
pub(crate) fn quantise(value: f32, step: f32) -> i32 {
    (value / step).round() as i32
}
