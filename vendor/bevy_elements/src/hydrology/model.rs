//! The hydrology and erosion model's CPU side: the substep scheduler, the quantised delivery of
//! rain and evaporation, the source footprints, the evaporation formula and the CPU mirrors of
//! the kernels' per-cell rules that the tests compare the shader against.
//!
//! Every number is `docs/research/hydrology.md` or `docs/research/erosion.md`; this module only
//! computes them.

use bevy::math::{IVec2, UVec2, Vec2, Vec4};

/// Gravity, m/s².
pub(crate) const GRAVITY: f32 = 9.81;
/// A cell with less water than `WET_FRACTION · l` is dry (Chentanez §2.1.4, ε = 10⁻⁴ Δx).
pub(crate) const WET_FRACTION: f32 = 1e-4;
/// Pipe depth cap factor: `h_cap = 0.16 l² / (g Δt²)` keeps `c Δt / l` under the stability limit
/// at any step and cell size.
pub(crate) const DEPTH_CAP_FACTOR: f32 = 0.16;
/// Velocity clamp: `|v| ≤ α l / Δt` (Chentanez §2.1.5).
pub(crate) const SPEED_CLAMP_ALPHA: f32 = 0.5;
/// Smallest rain, evaporation or melt quantum a cell receives, m. An f32 depth of 0.5–1 m
/// cannot hold less than half an ulp (6×10⁻⁸ m).
pub(crate) const QUANTUM_M: f64 = 5e-6;
/// Thermal erosion runs after every this many substeps.
pub(crate) const THERMAL_PERIOD: u64 = 4;
/// Bedrock moves only above `talus + BEDROCK_EXTRA_DEG`.
pub(crate) const BEDROCK_EXTRA_DEG: f32 = 10.0;
/// The tangent that stands for a bedrock threshold nothing reaches (a pass that moves loose
/// sediment only): far above any slope, and small enough that `cell · tan` stays finite.
pub(crate) const UNREACHABLE_TAN: f32 = 1e20;
/// Most `WaterSource`s and most `WaterSink`s uploaded.
pub const MAX_WATER_SOURCES: usize = 64;
/// The step the dissolve and deposit fractions are specified at, s.
pub(crate) const REFERENCE_STEP_S: f32 = 0.02;
/// Rain delivered per second per mm/h, m/s.
pub(crate) const MM_H_TO_M_S: f64 = 1.0 / 3.6e6;

/// Evaporation of an open water surface, mm/h: `0.1 (1 + u/5) (1 + max(T, 0)/20) (1 − RH)`.
///
/// `wind_ms` is the 10 m wind, `temperature_c` the air temperature, `relative_humidity` 0–1.
/// Checked against FAO-56 Penman–Monteith in `docs/research/hydrology.md`.
pub fn evaporation_mm_h(wind_ms: f32, temperature_c: f32, relative_humidity: f32) -> f32 {
    0.1 * (1.0 + wind_ms / 5.0)
        * (1.0 + temperature_c.max(0.0) / 20.0)
        * (1.0 - relative_humidity).max(0.0)
}

/// The deepest water a pipe sees, m: `0.16 l² / (g Δt²)`.
pub(crate) fn depth_cap_m(cell_m: f32, step_s: f32) -> f32 {
    DEPTH_CAP_FACTOR * cell_m * cell_m / (GRAVITY * step_s * step_s)
}

/// The deepest water for which the uncapped model is stable at `step_s`, m:
/// `(l / 2Δt)² / g`.
#[cfg(test)]
pub(crate) fn stable_depth_m(cell_m: f32, step_s: f32) -> f32 {
    let r = cell_m / (2.0 * step_s);
    r * r / GRAVITY
}

/// The longest stable step at depth `depth_m`, s: `l / (2 √(g h))`.
#[cfg(test)]
pub(crate) fn stable_step_s(cell_m: f32, depth_m: f32) -> f32 {
    cell_m / (2.0 * (GRAVITY * depth_m).sqrt())
}

/// The speed clamp, m/s: `0.5 l / Δt`.
pub(crate) fn speed_cap_ms(cell_m: f32, step_s: f32) -> f32 {
    SPEED_CLAMP_ALPHA * cell_m / step_s
}

/// Wet threshold, m.
pub(crate) fn wet_depth_m(cell_m: f32) -> f32 {
    WET_FRACTION * cell_m
}

/// Substeps per frame from real time: a signed remainder carries between frames so the long-run
/// rate is exact, and the backlog is capped at one maximal frame.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct Scheduler {
    backlog_s: f64,
}

impl Scheduler {
    /// Advance by `real_dt_s · time_scale` and return how many substeps of `step_s` this frame
    /// runs, at most `max`.
    pub(crate) fn step(&mut self, real_dt_s: f64, time_scale: f32, step_s: f32, max: u32) -> u32 {
        let step = f64::from(step_s);
        self.backlog_s += real_dt_s * f64::from(time_scale);
        let n = (self.backlog_s / step + 0.5)
            .floor()
            .clamp(0.0, f64::from(max));
        self.backlog_s -= n * step;
        self.backlog_s = self.backlog_s.min(f64::from(max) * step);
        n as u32
    }

    /// Forget the carried time (a scheduling restart).
    pub(crate) fn reset(&mut self) {
        self.backlog_s = 0.0;
    }

    /// The carried time, s.
    #[cfg(test)]
    pub(crate) fn backlog_s(&self) -> f64 {
        self.backlog_s
    }
}

/// Water depth owed to the cells and not yet delivered, m (f64). Delivered as a quantum of at
/// least [`QUANTUM_M`] on the first substep of a frame, so an f32 depth never has to hold an
/// increment below its rounding.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct QuantumAccumulator {
    pending_m: f64,
}

impl QuantumAccumulator {
    /// The quantum to deliver on this frame's first substep, m: everything owed if at least
    /// [`QUANTUM_M`], else 0. The f32 value actually uploaded is what leaves the account.
    pub(crate) fn take(&mut self) -> f32 {
        if self.pending_m >= QUANTUM_M {
            let q = self.pending_m as f32;
            self.pending_m -= f64::from(q);
            q
        } else {
            0.0
        }
    }

    /// Owe `rate_m_s` for `substeps` substeps of `step_s`.
    pub(crate) fn owe(&mut self, rate_m_s: f64, step_s: f32, substeps: u32) {
        self.pending_m += rate_m_s * f64::from(step_s) * f64::from(substeps);
    }

    /// Owed and not yet delivered, m.
    #[cfg(test)]
    pub(crate) fn pending_m(&self) -> f64 {
        self.pending_m
    }
}

/// Whether the cell `c` of a grid is inside the footprint of a source at world XZ `centre` with
/// `radius_m`: its centre lies within the radius, or it contains the source centre. This is the
/// expression of the `source_map_build` kernel.
pub(crate) fn covers(origin: Vec2, cell_m: f32, c: IVec2, centre: Vec2, radius_m: f32) -> bool {
    let cell_centre = origin + (c.as_vec2() + 0.5) * cell_m;
    let d = cell_centre - centre;
    d.dot(d) <= radius_m * radius_m || ((centre - origin) / cell_m).floor().as_ivec2() == c
}

/// Cells of the grid a source covers; 0 when its centre is off the grid (it delivers nothing).
pub(crate) fn source_cells(
    origin: Vec2,
    cell_m: f32,
    size: UVec2,
    centre: Vec2,
    radius_m: f32,
) -> u32 {
    let home = ((centre - origin) / cell_m).floor().as_ivec2();
    if home.cmplt(IVec2::ZERO).any() || home.cmpge(size.as_ivec2()).any() {
        return 0;
    }
    let reach = (radius_m.max(0.0) / cell_m).ceil() as i32 + 1;
    let lo = (home - reach).max(IVec2::ZERO);
    let hi = (home + reach).min(size.as_ivec2() - 1);
    let mut count = 0;
    for y in lo.y..=hi.y {
        for x in lo.x..=hi.x {
            if covers(origin, cell_m, IVec2::new(x, y), centre, radius_m) {
                count += 1;
            }
        }
    }
    count
}

/// One uploaded source entry: world XZ, radius and the depth rate per covered cell, m/s
/// (positive for a source, negative for a sink).
pub(crate) fn source_entry(
    origin: Vec2,
    cell_m: f32,
    size: UVec2,
    centre: Vec2,
    radius_m: f32,
    signed_rate_m3_s: f32,
) -> Option<Vec4> {
    let n = source_cells(origin, cell_m, size, centre, radius_m);
    (n > 0).then(|| {
        let rho = signed_rate_m3_s / (n as f32 * cell_m * cell_m);
        Vec4::new(centre.x, centre.y, radius_m, rho)
    })
}

/// `sin α` of a surface gradient: `|∇S| / √(1 + |∇S|²)`.
#[cfg(test)]
pub(crate) fn sin_alpha(gradient: Vec2) -> f32 {
    let g = gradient.length();
    g / (1.0 + g * g).sqrt()
}

/// Sediment transport capacity, m: `K_c max(sin α, s_min) |v|`.
#[cfg(test)]
pub(crate) fn capacity_m(kc: f32, min_tilt: f32, sin_alpha: f32, speed_ms: f32) -> f32 {
    kc * sin_alpha.max(min_tilt) * speed_ms
}

/// The per-step dissolve or deposit fraction: `min(1, K · strength · Δt / 0.02)`.
pub(crate) fn step_fraction(rate: f32, strength: f32, step_s: f32) -> f32 {
    (rate * strength * step_s / REFERENCE_STEP_S).clamp(0.0, 1.0)
}

/// The layers and suspended load of one cell.
#[cfg(test)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct CellSolids {
    pub bedrock: f32,
    pub sediment: f32,
    pub suspended: f32,
}

/// Dissolve or deposit one cell toward capacity `capacity_m` (the kernel's rule): sediment is
/// taken first, then bedrock; deposits land in the sediment layer; every transfer is the amount
/// actually applied after f32 rounding.
#[cfg(test)]
pub(crate) fn exchange(cell: CellSolids, capacity_m: f32, k_s: f32, k_d: f32) -> CellSolids {
    let CellSolids {
        mut bedrock,
        mut sediment,
        mut suspended,
    } = cell;
    if capacity_m > suspended {
        let a = k_s * (capacity_m - suspended);
        let from_sediment = a.min(sediment);
        let sediment_after = sediment - from_sediment;
        let mut applied = sediment - sediment_after;
        sediment = sediment_after;
        if from_sediment < a {
            let bedrock_after = bedrock - (a - from_sediment);
            applied += bedrock - bedrock_after;
            bedrock = bedrock_after;
        }
        suspended += applied;
    } else {
        let a = k_d * (suspended - capacity_m);
        let sediment_after = sediment + a;
        let applied = sediment_after - sediment;
        sediment = sediment_after;
        suspended -= applied;
    }
    CellSolids {
        bedrock,
        sediment,
        suspended,
    }
}

/// What one thermal pass moves out of a cell with surface drops `drops` to its neighbours at
/// distances `distances` (the kernel's rule): `(from sediment, from bedrock)`, and the share each
/// neighbour receives.
#[cfg(test)]
pub(crate) fn thermal_moves(
    drops: &[f32],
    distances: &[f32],
    sediment: f32,
    talus_deg: f32,
    rate: f32,
) -> (f32, f32, Vec<f32>) {
    let tan_t = talus_deg.to_radians().tan();
    let tan_b = (talus_deg + BEDROCK_EXTRA_DEG).to_radians().tan();
    let e: Vec<f32> = drops
        .iter()
        .zip(distances)
        .map(|(h, d)| (h - d * tan_t).max(0.0))
        .collect();
    let eb: Vec<f32> = drops
        .iter()
        .zip(distances)
        .map(|(h, d)| (h - d * tan_b).max(0.0))
        .collect();
    let max_e = e.iter().copied().fold(0.0, f32::max);
    let max_eb = eb.iter().copied().fold(0.0, f32::max);
    let from_sediment = (0.5 * rate * max_e).min(sediment);
    let from_bedrock = (0.5 * rate * max_e - from_sediment).min(0.5 * rate * max_eb);
    let sum_e: f32 = e.iter().sum();
    let sum_eb: f32 = eb.iter().sum();
    let q = e
        .iter()
        .zip(&eb)
        .map(|(e, eb)| {
            (if sum_e > 0.0 {
                from_sediment * e / sum_e
            } else {
                0.0
            }) + (if sum_eb > 0.0 {
                from_bedrock * eb / sum_eb
            } else {
                0.0
            })
        })
        .collect();
    (from_sediment, from_bedrock, q)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn substeps_in(fps: f64, seconds: f64, time_scale: f32, max: u32) -> (u64, f64) {
        let mut scheduler = Scheduler::default();
        let frames = (fps * seconds).round() as u64;
        let mut total = 0u64;
        let mut worst_backlog = 0.0_f64;
        for _ in 0..frames {
            total += u64::from(scheduler.step(1.0 / fps, time_scale, 0.02, max));
            worst_backlog = worst_backlog.max(scheduler.backlog_s());
        }
        (total, worst_backlog)
    }

    /// Validation 8: 50 substeps per second at any frame rate; the cap and the backlog bound.
    #[test]
    fn the_scheduler_runs_fifty_substeps_per_second_at_any_frame_rate() {
        for fps in [60.0, 144.0, 30.0, 15.0] {
            let (total, _) = substeps_in(fps, 10.0, 1.0, 8);
            assert!(
                (499..=501).contains(&total),
                "{fps} fps gave {total} substeps in 10 s"
            );
        }
        // Time scale 10 at 60 fps: 480 per second capped at 8 a frame, 120 at 2 a frame.
        assert_eq!(substeps_in(60.0, 10.0, 10.0, 8).0, 4800);
        assert_eq!(substeps_in(60.0, 10.0, 10.0, 2).0, 1200);
        let (_, backlog) = substeps_in(60.0, 10.0, 10.0, 8);
        assert!(backlog <= 8.0 * 0.02 + 1e-12, "backlog {backlog}");
        // A hitch of a whole second runs one maximal frame, not fifty substeps.
        let mut scheduler = Scheduler::default();
        assert_eq!(scheduler.step(1.0, 1.0, 0.02, 8), 8);
        assert!(scheduler.backlog_s() <= 8.0 * 0.02 + 1e-12);
        // A paused clock runs nothing and a zero time scale runs nothing.
        assert_eq!(Scheduler::default().step(0.0, 1.0, 0.02, 8), 0);
        assert_eq!(Scheduler::default().step(1.0 / 60.0, 0.0, 0.02, 8), 0);
    }

    /// Validation 5 (the CPU half): the stable depth, the cap and the longest stable step.
    #[test]
    fn the_stability_cap_matches_the_derived_table() {
        // Table of hydrology.md "Stability": (l, h_stable at 0.02 s, h_cap, speed cap).
        for (l, stable, cap, speed) in [
            (8.0, 4077.0, 2610.0, 200.0),
            (4.0, 1019.0, 652.0, 100.0),
            (2.0, 255.0, 163.0, 50.0),
            (1.0, 64.0, 41.0, 25.0),
        ] {
            let close = |got: f32, want: f32| (got - want).abs() <= want * 0.01;
            assert!(close(stable_depth_m(l, 0.02), stable), "stable at l={l}");
            assert!(close(depth_cap_m(l, 0.02), cap), "cap at l={l}");
            assert!(close(speed_cap_ms(l, 0.02), speed), "speed at l={l}");
        }
        // Δt_max at l = 2 m: 0.32 s at 1 m, 0.10 s at 10 m, 0.045 s at 50 m.
        for (h, dt) in [(1.0, 0.32), (10.0, 0.10), (50.0, 0.045)] {
            let got = stable_step_s(2.0, h);
            assert!((got - dt).abs() < dt * 0.03, "h={h}: {got} vs {dt}");
        }
        // With the cap on, the pipe depth never exceeds what is stable at any step.
        for dt in [0.02_f32, 0.2, 0.5] {
            let cap = depth_cap_m(2.0, dt);
            assert!(
                stable_step_s(2.0, cap) >= dt * 0.99,
                "step {dt}: cap {cap} m is not stable"
            );
        }
        assert!((wet_depth_m(2.0) - 2e-4).abs() < 1e-9);
    }

    /// Validation 10.
    #[test]
    fn the_evaporation_formula_matches_the_oracle() {
        let close = |got: f32, want: f32| (got - want).abs() < want.abs().max(1.0) * 1e-5;
        assert!(close(evaporation_mm_h(5.0, 20.0, 0.5), 0.200));
        assert_eq!(evaporation_mm_h(0.0, 0.0, 1.0), 0.0);
        assert!(close(evaporation_mm_h(10.0, -10.0, 0.5), 0.150));
        // The dust-storm preset: 16 m/s, 35 °C, RH 0.15 → 0.98 mm/h = 23.6 mm/day.
        let dust = evaporation_mm_h(16.0, 35.0, 0.15);
        assert!((dust - 0.98).abs() < 0.01, "{dust}");
        assert!((dust * 24.0 - 23.6).abs() < 0.2, "{}", dust * 24.0);
        // Saturated or over-saturated air never condenses.
        assert_eq!(evaporation_mm_h(3.0, 10.0, 1.2), 0.0);
    }

    /// Validation 7: the accumulator.
    #[test]
    fn rain_is_delivered_in_quanta_and_never_lost() {
        let rate = 4.0 * MM_H_TO_M_S;
        let per_substep = rate * 0.02;
        assert!((per_substep - 2.22e-8).abs() < 1e-10);
        let mut acc = QuantumAccumulator::default();
        let mut delivered = 0.0_f64;
        let mut first_delivery = None;
        // One substep per frame, so a frame boundary follows every substep.
        for substep in 1..=1_000_000u64 {
            let q = acc.take();
            if q > 0.0 && first_delivery.is_none() {
                first_delivery = Some(substep);
            }
            delivered += f64::from(q);
            acc.owe(rate, 0.02, 1);
        }
        // 5e-6 / 2.22e-8 = 225.0 substeps: the quantum goes out at the next frame boundary.
        assert!(
            matches!(first_delivery, Some(226..=227)),
            "first quantum at {first_delivery:?}"
        );
        // The step is an f32 setting, so the account is exact for that value of it.
        let ideal = rate * f64::from(0.02_f32) * 1_000_000.0;
        assert!(
            (delivered + acc.pending_m() - ideal).abs() < 1e-12,
            "delivered {delivered} + pending {} vs {ideal}",
            acc.pending_m()
        );
        assert!(acc.pending_m() < QUANTUM_M * 1.01);
    }

    /// Validation 9: a 50 m³/s source covers a handful of 2 m cells and delivers its rate.
    #[test]
    fn a_source_delivers_its_rate_over_its_footprint() {
        let size = UVec2::splat(256);
        let origin = Vec2::splat(-256.0);
        let l = 2.0;
        let centre = Vec2::new(0.3, 0.2);
        let n = source_cells(origin, l, size, centre, 5.0);
        // Brute force over the whole grid, same predicate.
        let brute = (0..256)
            .flat_map(|y| (0..256).map(move |x| IVec2::new(x, y)))
            .filter(|c| covers(origin, l, *c, centre, 5.0))
            .count() as u32;
        assert_eq!(n, brute);
        assert!((16..=28).contains(&n), "{n} cells");
        let entry = source_entry(origin, l, size, centre, 5.0, 50.0).expect("on the grid");
        let delivered = entry.w * n as f32 * l * l;
        assert!((delivered - 50.0).abs() < 1e-3, "{delivered}");
        assert!((entry.w - 50.0 / (n as f32 * 4.0)).abs() < 1e-6);
        // A source at a cell centre covers 21 cells, at a cell corner 16 (even and odd lattices).
        assert_eq!(source_cells(origin, l, size, Vec2::new(1.0, 1.0), 5.0), 21);
        assert_eq!(source_cells(origin, l, size, Vec2::new(0.0, 0.0), 5.0), 16);
        // Never fewer than the cell holding the centre.
        assert_eq!(source_cells(origin, l, size, Vec2::new(0.3, 0.2), 0.1), 1);
        // Off the grid delivers nothing.
        assert_eq!(source_cells(origin, l, size, Vec2::new(900.0, 0.0), 5.0), 0);
        assert!(source_entry(origin, l, size, Vec2::new(900.0, 0.0), 5.0, 50.0).is_none());
        // A sink has the opposite sign.
        let sink = source_entry(origin, l, size, centre, 5.0, -10.0).expect("on the grid");
        assert!(sink.w < 0.0);
    }

    /// Erosion validation 4: capacity.
    #[test]
    fn capacity_follows_the_slope_and_the_minimum_tilt() {
        assert!((sin_alpha(Vec2::new(0.3, 0.0)) - 0.2873).abs() < 1e-4);
        let c = capacity_m(0.6, 0.05, sin_alpha(Vec2::new(0.3, 0.0)), 1.0);
        assert!((c - 0.1724).abs() < 1e-4, "{c}");
        // Gradient 0.02 is below the minimum tilt: C = 0.6 · 0.05 · |v|.
        let flat = capacity_m(0.6, 0.05, sin_alpha(Vec2::new(0.02, 0.0)), 2.0);
        assert!((flat - 0.6 * 0.05 * 2.0).abs() < 1e-6, "{flat}");
    }

    /// Erosion validation 4: dissolve and deposit per step, and their scaling with `step_s`.
    #[test]
    fn dissolve_and_deposit_move_a_fixed_fraction_per_step() {
        let k = step_fraction(0.03, 1.0, 0.02);
        assert!((k - 0.03).abs() < 1e-9);
        let cell = CellSolids {
            bedrock: 10.0,
            sediment: 0.5,
            suspended: 0.0,
        };
        let c = 0.1724;
        let after = exchange(cell, c, k, k);
        assert!((after.suspended - 0.03 * c).abs() < 1e-6, "{after:?}");
        assert!((cell.sediment - after.sediment - after.suspended).abs() < 1e-6);
        assert_eq!(
            after.bedrock, 10.0,
            "bedrock untouched while sediment lasts"
        );
        // Sediment exhausted: the rest comes from bedrock.
        let bare = exchange(
            CellSolids {
                bedrock: 10.0,
                sediment: 0.001,
                suspended: 0.0,
            },
            c,
            k,
            k,
        );
        assert_eq!(bare.sediment, 0.0);
        assert!(bare.bedrock < 10.0);
        // Deposit: s = 0.2, C = 0 → 6e-3.
        let dep = exchange(
            CellSolids {
                bedrock: 10.0,
                sediment: 0.0,
                suspended: 0.2,
            },
            0.0,
            k,
            k,
        );
        assert!((dep.sediment - 6e-3).abs() < 1e-7 && (dep.suspended - 0.194).abs() < 1e-6);
        // step_s = 0.04 moves 0.06 of the difference, and the fraction is capped at 1.
        assert!((step_fraction(0.03, 1.0, 0.04) - 0.06).abs() < 1e-9);
        assert_eq!(step_fraction(0.03, 100.0, 0.04), 1.0);
        assert!((step_fraction(0.03, 2.0, 0.02) - 0.06).abs() < 1e-9);
    }

    /// Erosion validation 5: nothing leaks when the amount is below an ulp of the layer.
    #[test]
    fn a_deposit_below_an_ulp_does_not_leak() {
        let cell = CellSolids {
            bedrock: 10.0,
            sediment: 1.0,
            suspended: 3.0e-8 / 0.03, // k_d · (s − C) = 3e-8 with C = 0
        };
        let after = exchange(cell, 0.0, 0.03, 0.03);
        let before_total = f64::from(cell.sediment) + f64::from(cell.suspended);
        let after_total = f64::from(after.sediment) + f64::from(after.suspended);
        assert!(
            (after_total - before_total).abs() < 1e-7,
            "{before_total} → {after_total}"
        );
        // The naive form would have lost the whole deposit from `suspended`.
        let sediment_after = cell.sediment + 3.0e-8;
        assert_eq!(sediment_after, cell.sediment, "the layer cannot hold it");
        assert_eq!(after.suspended, cell.suspended, "so the load keeps it");
    }

    /// Erosion validation 3: the transfer rule of one thermal pass.
    #[test]
    fn a_thermal_pass_moves_the_excess_over_the_talus() {
        // Two cells, sediment only, l = 1, a 40° slope.
        let drop = 40.0_f32.to_radians().tan();
        let (ms, mb, q) = thermal_moves(&[drop], &[1.0], 5.0, 35.0, 0.15);
        let e = drop - 35.0_f32.to_radians().tan();
        assert!((e - 0.1388).abs() < 1e-3, "{e}");
        assert!((ms - 0.5 * 0.15 * e).abs() < 1e-7);
        assert!((ms - 0.01041).abs() < 1e-4, "{ms}");
        assert_eq!(mb, 0.0, "bedrock stays under 45°");
        assert!((q[0] - ms).abs() < 1e-7, "mass moved equals mass gained");
        // Bedrock exposed under a 46° slope moves 0.00266 l; at 44° it does not.
        let steep = 46.0_f32.to_radians().tan();
        let (ms, mb, q) = thermal_moves(&[steep], &[1.0], 0.0, 35.0, 0.15);
        assert_eq!(ms, 0.0);
        assert!((mb - 0.00266).abs() < 1e-4, "{mb}");
        assert!((q[0] - mb).abs() < 1e-7);
        let gentle = 44.0_f32.to_radians().tan();
        assert_eq!(thermal_moves(&[gentle], &[1.0], 0.0, 35.0, 0.15).1, 0.0);
        // Under the talus nothing moves.
        let (ms, mb, _) = thermal_moves(&[0.6], &[1.0], 5.0, 35.0, 0.15);
        assert_eq!((ms, mb), (0.0, 0.0));
        // Two lower neighbours split the move by their excess.
        let (_, _, q) = thermal_moves(&[1.0, 0.8], &[1.0, 1.0], 5.0, 35.0, 0.15);
        assert!(q[0] > q[1] && q[1] > 0.0);
    }
}
