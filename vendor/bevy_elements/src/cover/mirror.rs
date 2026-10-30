//! The ground-cover kernels on the CPU: the equations of `snow.wesl` and `shelter.wesl` in the
//! same operation order, used by the unit tests to check the numbers of
//! `docs/research/snow-cover.md` and `docs/research/wetness.md` and by the GPU tests as the
//! oracle.

use bevy::math::{Vec2, Vec3};

use super::model::{MAX_STEPS_PER_FRAME, plan_steps, smoothstep, snow_coverage};
use super::{SnowSettings, WetnessSettings};
use crate::climate::Weather;

/// A pending melt below this stays in the snow until it grows past it or the cell empties
/// (`hydrology::model::QuantumAccumulator`: an f32 water depth cannot hold less).
pub(crate) const RELEASE_QUANTUM_M: f32 = 5e-6;
/// The weight of one diffusion pass at the longest step, and that step, s.
pub(crate) const DIFFUSE_LAMBDA: f32 = 0.125;
pub(crate) const DIFFUSE_REFERENCE_STEP_S: f32 = 600.0;

/// The weight of one diffusion pass for a step of `dt_s`: proportional to the step, so the
/// smoothing is a rate in game time (`diffuse_lambda` of `snow.wesl`).
pub(crate) fn diffuse_lambda(dt_s: f32) -> f32 {
    DIFFUSE_LAMBDA * (dt_s / DIFFUSE_REFERENCE_STEP_S).min(1.0)
}
/// Highest a blocker still shelters the ground, m (`shelter.wesl`).
pub(crate) const SHELTER_HEIGHT_M: f32 = 500.0;

/// Fraction of precipitation that falls as snow at `t_c` (`snow.wesl` in `climate`), in the snow
/// season (`possible` 1).
pub(crate) fn snow_fraction_at(t_c: f32) -> f32 {
    Weather {
        temperature_c: t_c,
        ..Weather::default()
    }
    .snow_fraction()
}

/// Share of the snowfall a slope of `slope_deg` keeps: `1 − smoothstep(α0, α1, α)`.
pub(crate) fn snow_exposure(slope_deg: f32, shed_deg: [f32; 2]) -> f32 {
    1.0 - smoothstep(shed_deg[0], shed_deg[1], slope_deg)
}

/// The puddle mask of a water depth: `smoothstep(d_min, d_full, d_w)`.
pub(crate) fn puddle_mask(depth_m: f32, wetness: &WetnessSettings) -> f32 {
    smoothstep(wetness.puddle_min_m, wetness.puddle_full_m, depth_m)
}

/// Potential drift flux of snow water equivalent, m²/s: `k_d/ρ_w · (u − u_t)₊³`.
pub(crate) fn drift_flux(speed_ms: f32, snow: &SnowSettings) -> f32 {
    let excess = (speed_ms - snow.drift_threshold_ms).max(0.0);
    snow.drift_coefficient / 1000.0 * excess * excess * excess
}

/// Air temperature of a ground point `surface_above_reference_m` above the climate's reference
/// elevation, with the lapse rate in °C/m (negative).
pub(crate) fn local_temperature(
    t_ref_c: f32,
    lapse_c_per_m: f32,
    surface_above_reference_m: f32,
) -> f32 {
    t_ref_c + lapse_c_per_m * surface_above_reference_m
}

/// One cell of the snow layer, as `snow_source` updates it.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct SnowCell {
    /// Depth, m.
    pub depth: f32,
    /// Water equivalent, m.
    pub swe: f32,
    /// Melt not yet handed to hydrology, m of water.
    pub pending: f32,
}

impl SnowCell {
    /// Bulk density, kg/m³ (new-snow density when empty).
    pub(crate) fn density(&self, snow: &SnowSettings) -> f32 {
        if self.depth > 0.0 {
            1000.0 * self.swe / self.depth
        } else {
            snow.new_density
        }
    }

    /// Advance by `dt_h` hours: accumulate, densify, melt and release. `exposure` is the share
    /// of the snowfall the cell keeps. Returns the water handed to hydrology, m.
    pub(crate) fn step(
        &mut self,
        snow: &SnowSettings,
        dt_h: f32,
        precipitation_mm_h: f32,
        t_loc_c: f32,
        exposure: f32,
    ) -> f32 {
        let swe_in = 1e-3 * precipitation_mm_h * snow_fraction_at(t_loc_c) * exposure * dt_h;
        self.depth += swe_in * 1000.0 / snow.new_density;
        self.swe += swe_in;
        if self.swe > 0.0 {
            let mut rho = 1000.0 * self.swe / self.depth.max(1e-9);
            rho += (snow.settled_density - rho).max(0.0)
                * (1.0 - (-dt_h / (snow.settle_days * 24.0)).exp());
            let melt = self
                .swe
                .min(snow.degree_day_factor * 1e-3 * t_loc_c.max(0.0) * dt_h / 24.0);
            self.swe -= melt;
            self.pending += melt;
            self.depth = 1000.0 * self.swe / rho;
        }
        if self.swe <= 0.0 {
            self.depth = 0.0;
            self.swe = 0.0;
        }
        if self.pending >= RELEASE_QUANTUM_M || (self.swe <= 0.0 && self.pending > 0.0) {
            let released = self.pending;
            self.pending = 0.0;
            return released;
        }
        0.0
    }
}

/// Wetness after `dt_h` hours, as `wetness_update`: rain wets `k_w` per mm, evaporation dries
/// `k_d · E/E₀ · (1 + sun)` per hour. `rain_mm_h` is the liquid rain that reaches the ground.
#[allow(clippy::too_many_arguments)]
pub(crate) fn wetness_step(
    wetness: f32,
    settings: &WetnessSettings,
    dt_h: f32,
    rain_mm_h: f32,
    wind_ms: f32,
    t_loc_c: f32,
    relative_humidity: f32,
    sun_factor: f32,
) -> f32 {
    let evaporation = (1.0 + wind_ms / 5.0)
        * (1.0 + t_loc_c.max(0.0) / 20.0)
        * (1.0 - relative_humidity).max(0.0);
    let drying = settings.dry_rate_h * evaporation * (1.0 + sun_factor);
    (wetness + (settings.wet_per_mm * rain_mm_h - drying) * dt_h).clamp(0.0, 1.0)
}

/// Rain that reaches a cell, mm/h: the liquid share, under the shelter, not on a snow cover.
pub(crate) fn ground_rain_mm_h(
    precipitation_mm_h: f32,
    t_loc_c: f32,
    rain_exposure: f32,
    snow_depth_m: f32,
) -> f32 {
    precipitation_mm_h
        * (1.0 - snow_fraction_at(t_loc_c))
        * rain_exposure
        * (1.0 - snow_coverage(snow_depth_m))
}

/// One diffusion pass over a row-major `(depth, swe)` grid with no-flux edges.
pub(crate) fn diffuse(cells: &[[f32; 2]], size: usize) -> Vec<[f32; 2]> {
    let at = |x: isize, y: isize, own: [f32; 2]| {
        if x < 0 || y < 0 || x >= size as isize || y >= size as isize {
            own
        } else {
            cells[y as usize * size + x as usize]
        }
    };
    let mut out = cells.to_vec();
    for y in 0..size {
        for x in 0..size {
            let own = cells[y * size + x];
            let (xi, yi) = (x as isize, y as isize);
            let n = [
                at(xi, yi - 1, own),
                at(xi + 1, yi, own),
                at(xi, yi + 1, own),
                at(xi - 1, yi, own),
            ];
            for c in 0..2 {
                let sum: f32 = n.iter().map(|v| v[c]).sum();
                out[y * size + x][c] = own[c] + DIFFUSE_LAMBDA * (sum - 4.0 * own[c]);
            }
        }
    }
    out
}

/// Whether the segment `p0 → p1` meets the axis-aligned box (`shelter.wesl`'s slab test with an
/// identity rotation).
pub(crate) fn segment_hits_box(centre: Vec3, half: Vec3, p0: Vec3, p1: Vec3) -> bool {
    let q0 = p0 - centre;
    let d = p1 - p0;
    let (mut t_in, mut t_out) = (0.0_f32, 1.0_f32);
    for a in 0..3 {
        if d[a].abs() < 1e-7 {
            if q0[a].abs() > half[a] {
                return false;
            }
        } else {
            let inv = 1.0 / d[a];
            let t_a = (-half[a] - q0[a]) * inv;
            let t_b = (half[a] - q0[a]) * inv;
            t_in = t_in.max(t_a.min(t_b));
            t_out = t_out.min(t_a.max(t_b));
        }
    }
    t_in <= t_out
}

/// Whether a ground point is sheltered from precipitation falling with horizontal velocity
/// `fall_xz` (the wind) and speed `fall_speed`: the ray toward where it came from,
/// `(−fall_xz, fall_speed)`, meets a box.
pub(crate) fn sheltered(
    ground: Vec3,
    wind_xz: Vec2,
    fall_speed: f32,
    boxes: &[(Vec3, Vec3)],
) -> bool {
    let up = Vec3::new(-wind_xz.x, fall_speed, -wind_xz.y) * (SHELTER_HEIGHT_M / fall_speed);
    boxes
        .iter()
        .any(|&(centre, half)| segment_hits_box(centre, half, ground, ground + up))
}

mod tests {
    use super::*;
    use crate::climate::SurfaceWeathering;

    fn snow() -> SnowSettings {
        SnowSettings::default()
    }

    /// Snow validation 2: the coverage curve.
    #[test]
    fn coverage_follows_the_five_centimetre_smoothstep() {
        for (depth, want) in [
            (0.0, 0.0),
            (0.0125, 0.156),
            (0.025, 0.5),
            (0.05, 1.0),
            (0.2, 1.0),
        ] {
            let got = snow_coverage(depth);
            assert!((got - want).abs() < 1e-3, "c({depth}) = {got}");
        }
    }

    /// Snow validation 3: a layer filled to 15 mm in one step densifies along the formula, and
    /// the step size does not matter.
    #[test]
    fn densification_follows_the_exponential_and_ignores_the_step() {
        let s = snow();
        for (days, want) in [(1.0, 156.7), (3.0, 226.4), (10.0, 292.9)] {
            let mut densities = Vec::new();
            for step_s in [10.0_f32, 60.0, 600.0] {
                let mut cell = SnowCell::default();
                // 15 mm in one step at 100 kg/m³: a 1 h snowfall of 15 mm/h at -5 °C, then still air.
                cell.step(&s, 1.0 / 3600.0, 15.0 * 3600.0, -5.0, 1.0);
                assert!((cell.swe - 0.015).abs() < 1e-7);
                assert!(
                    (cell.density(&s) - 100.0).abs() < 0.3,
                    "{}",
                    cell.density(&s)
                );
                let steps = (days * 86_400.0 / step_s) as usize;
                for _ in 0..steps {
                    cell.step(&s, step_s / 3600.0, 0.0, -5.0, 1.0);
                }
                let rho = cell.density(&s);
                assert!((rho - want).abs() < 0.2, "{days} d at {step_s} s: {rho}");
                assert!((cell.depth * rho - 1000.0 * cell.swe).abs() < 1e-5);
                densities.push(rho);
            }
            for r in &densities {
                assert!((r / densities[0] - 1.0).abs() < 1e-3);
            }
        }
    }

    /// Snow validation 6: the shed curve and the rain-snow line by elevation.
    #[test]
    fn exposure_sheds_steep_slopes_and_the_snow_line_follows_elevation() {
        let shed = snow().shed_slope_deg;
        for (slope, want) in [
            (30.0, 1.0),
            (40.0, 0.844),
            (45.0, 0.5),
            (50.0, 0.156),
            (55.0, 0.0),
        ] {
            let got = snow_exposure(slope, shed);
            assert!((got - want).abs() < 1e-3, "exposure({slope}°) = {got}");
        }
        let lapse = -crate::climate::LAPSE_RATE_C_PER_M.abs();
        let t_high = local_temperature(3.5, lapse, 1000.0);
        assert!((t_high + 3.0).abs() < 1e-5, "{t_high}");
        assert_eq!(snow_fraction_at(t_high), 1.0);
        assert_eq!(snow_fraction_at(3.5), 0.0);
    }

    /// Snow validation 4b: the flux law.
    #[test]
    fn the_drift_flux_is_cubic_above_the_threshold() {
        let s = snow();
        assert_eq!(drift_flux(5.0, &s), 0.0);
        assert!((drift_flux(10.0, &s) / drift_flux(7.5, &s) - 8.0).abs() < 0.08);
        assert!((drift_flux(8.0, &s) / drift_flux(6.0, &s) - 27.0).abs() < 0.27);
        assert!((drift_flux(10.0, &s) / 4.6e-6 - 1.0).abs() < 0.01 + 0.0125);
        assert!((drift_flux(18.0, &s) / 8.1e-5 - 1.0).abs() < 0.04);
    }

    /// Snow validation 5: two passes of the five-point stencil.
    #[test]
    fn two_diffusion_passes_have_the_stencil_weights_and_conserve_snow() {
        let n = 15;
        let mut cells = vec![[0.0_f32; 2]; n * n];
        cells[7 * n + 7] = [1.0, 1.0];
        let once = diffuse(&cells, n);
        let twice = diffuse(&once, n);
        let at = |c: &Vec<[f32; 2]>, x: usize, y: usize| c[y * n + x][1];
        // Weights of the second pass: centre (1 − 4λ)² + 4λ², edge neighbour 2λ(1 − 4λ), ...
        let l = DIFFUSE_LAMBDA;
        assert!((at(&twice, 7, 7) - ((1.0 - 4.0 * l).powi(2) + 4.0 * l * l)).abs() < 1e-6);
        assert!((at(&twice, 8, 7) - 2.0 * l * (1.0 - 4.0 * l)).abs() < 1e-6);
        assert!((at(&twice, 8, 8) - 2.0 * l * l).abs() < 1e-6);
        assert!((at(&twice, 9, 7) - l * l).abs() < 1e-6);
        let total: f32 = twice.iter().map(|c| c[1]).sum();
        assert!((total - 1.0).abs() < 1e-6);
        let second_moment: f32 = (0..n)
            .map(|x| {
                let dx = x as f32 - 7.0;
                dx * dx * (0..n).map(|y| at(&twice, x, y)).sum::<f32>()
            })
            .sum();
        assert!((second_moment - 4.0 * l).abs() < 1e-6, "{second_moment}");
        assert!(twice.iter().all(|c| c[0] >= 0.0 && c[1] >= 0.0));
    }

    /// Snow validation 1, CPU arithmetic: ten hours of 1.5 mm/h and then melt.
    #[test]
    fn ten_hours_of_snow_lay_15_mm_and_a_warm_day_melts_it_in_nine_hours() {
        let s = snow();
        let mut cell = SnowCell::default();
        let mut water = 0.0;
        let dt = 10.0 / 3600.0;
        for _ in 0..3600 {
            water += cell.step(&s, dt, 1.5, -5.0, 1.0);
        }
        assert!((cell.swe - 0.015).abs() < 1.5e-5, "{}", cell.swe);
        assert!(
            (cell.depth - 0.1330).abs() < 0.1330 * 0.02,
            "{}",
            cell.depth
        );
        assert_eq!(water, 0.0);
        let melt_start = cell;
        for _ in 0..(3600 * 9 / 10 + 2) {
            water += cell.step(&s, dt, 0.0, 10.0, 1.0);
        }
        assert_eq!(cell.depth, 0.0);
        assert_eq!(cell.swe, 0.0);
        assert!((f64::from(water) - 0.015).abs() < 1e-6, "{water}");
        assert!(melt_start.swe > 0.0);
    }

    /// Snow validation 7: the cell model is `SurfaceWeathering` on flat exposed terrain.
    #[test]
    fn the_cell_model_agrees_with_the_point_model_every_hour() {
        let s = snow();
        let mut cell = SnowCell::default();
        let mut point = SurfaceWeathering::default();
        let cold = Weather {
            precipitation_mm_h: 1.5,
            temperature_c: -5.0,
            relative_humidity: 0.5,
            wind_speed_ms: 0.0,
            ..Weather::default()
        };
        let warm = Weather {
            precipitation_mm_h: 0.0,
            temperature_c: 10.0,
            ..cold
        };
        let dt_h = 1.0 / 60.0;
        for (hours, weather) in [(10, &cold), (12, &warm)] {
            for _ in 0..hours {
                for _ in 0..60 {
                    cell.step(
                        &s,
                        dt_h,
                        weather.precipitation_mm_h,
                        weather.temperature_c,
                        1.0,
                    );
                    point.step(weather, 0.0, dt_h);
                }
                let swe_mm = cell.swe * 1000.0;
                let want = point.snow_water_mm();
                assert!(
                    (swe_mm - want).abs() <= 0.02 * want.max(0.05),
                    "swe {swe_mm} vs {want}"
                );
                let want = point.snow_depth_m;
                assert!(
                    (cell.depth - want).abs() <= 0.03 * want.max(1e-3),
                    "depth {} vs {want}",
                    cell.depth
                );
            }
        }
        assert_eq!(cell.depth, 0.0);
        assert_eq!(point.snow_depth_m, 0.0);
    }

    /// Wetness validation 1 (CPU): an hour of 4 mm/h soaks, then a clear night dries to 0.1 in
    /// 11.25 h.
    #[test]
    fn wetness_soaks_in_half_an_hour_and_dries_in_eleven_and_a_quarter() {
        let w = WetnessSettings::default();
        let mut wet = 0.0;
        let mut reached = None;
        for i in 0..3600 {
            wet = wetness_step(wet, &w, 1.0 / 3600.0, 4.0, 3.0, 20.0, 0.5, 0.0);
            if reached.is_none() && wet >= 1.0 {
                reached = Some(i as f32 / 3600.0);
            }
        }
        assert_eq!(wet, 1.0);
        // 4 mm/h wets 2.0 per hour and the evaporation of this weather dries 0.08 of it back.
        assert!((reached.expect("soaked") - 1.0 / 1.92).abs() < 0.002);
        let mut hours = 0.0_f32;
        let mut at_ten = 0.0_f32;
        while wet > 0.1 {
            wet = wetness_step(wet, &w, 1.0 / 360.0, 0.0, 3.0, 20.0, 0.5, 0.0);
            hours += 1.0 / 360.0;
            if (hours - 10.0).abs() < 1.0 / 720.0 {
                at_ten = wet;
            }
        }
        assert!(at_ten > 0.1, "wetter than 0.1 at 10 h");
        assert!((hours - 11.25).abs() < 0.05, "{hours}");
    }

    /// Wetness validations 2 and 3: the rain that fills the film, the identity of film and
    /// free-water threshold, and no wetting under snow cover.
    #[test]
    fn two_millimetres_fill_the_film_and_snow_cover_keeps_the_ground_dry() {
        let w = WetnessSettings::default();
        let off = WetnessSettings {
            dry_rate_h: 0.0,
            ..w.clone()
        };
        let got = wetness_step(0.0, &off, 1.0, 2.0, 0.0, 15.0, 0.5, 0.0);
        assert!((got - 1.0).abs() < 1e-3);
        let got = wetness_step(0.0, &off, 1.0, 1.0, 0.0, 15.0, 0.5, 0.0);
        assert!((got - 0.5).abs() < 1e-6);
        assert!((w.puddle_min_m - 1e-3 / w.wet_per_mm).abs() < 1e-9);
        assert!((w.dry_rate_h - 0.1 / (1.0 / w.wet_per_mm)).abs() < 1e-9);
        assert_eq!(ground_rain_mm_h(4.0, 5.0, 1.0, 0.05), 0.0);
        assert_eq!(ground_rain_mm_h(4.0, 5.0, 1.0, 0.0), 4.0);
        assert_eq!(ground_rain_mm_h(4.0, 5.0, 0.0, 0.0), 0.0);
    }

    /// Wetness validation 4: the puddle mask.
    #[test]
    fn the_puddle_mask_is_a_smoothstep_of_the_water_depth() {
        let w = WetnessSettings::default();
        for (depth, want) in [
            (0.0, 0.0),
            (0.002, 0.0),
            (0.0065, 0.156),
            (0.011, 0.5),
            (0.02, 1.0),
        ] {
            let got = puddle_mask(depth, &w);
            assert!((got - want).abs() < 1e-3, "p({depth}) = {got}");
        }
    }

    /// Wetness validation 5: the shelter band of a roof in the wind (the slab test).
    #[test]
    fn a_roof_shelters_a_band_displaced_downwind_by_the_wind() {
        let roof = [(Vec3::new(0.0, 5.0, 0.0), Vec3::new(5.0, 0.5, 5.0))];
        let at = |x: f32, wind: Vec2, v: f32| sheltered(Vec3::new(x, 0.0, 0.0), wind, v, &roof);
        let wind = Vec2::new(7.0, 0.0);
        for (x, want) in [
            (2.0, false),
            (2.3, true),
            (13.6, true),
            (13.9, false),
            (-3.0, false),
        ] {
            assert_eq!(at(x, wind, 4.39), want, "rain at x = {x}");
        }
        for (x, want) in [(-5.0, true), (4.9, true), (5.2, false), (0.0, true)] {
            assert_eq!(at(x, Vec2::ZERO, 4.39), want, "calm at x = {x}");
        }
        for (x, want) in [(26.0, false), (27.0, true), (43.0, true), (44.0, false)] {
            assert_eq!(at(x, wind, 1.0), want, "snow at x = {x}");
        }
    }

    /// The smoothing does not depend on how many steps a second the frame rate allows: ten
    /// minutes of game time smooth the same in one 600 s step and in sixty 10 s steps.
    #[test]
    fn diffusion_is_a_rate_in_game_time() {
        assert_eq!(diffuse_lambda(600.0), DIFFUSE_LAMBDA);
        assert_eq!(diffuse_lambda(6000.0), DIFFUSE_LAMBDA);
        let variance = |dt: f32| (600.0 / dt) * 4.0 * diffuse_lambda(dt);
        assert!((variance(10.0) - variance(600.0)).abs() < 1e-5);
        assert!((variance(30.0) - 0.5).abs() < 1e-5);
    }

    #[test]
    fn a_clock_jump_is_cut_into_steps_of_at_most_ten_minutes() {
        assert_eq!(plan_steps(0.0), None);
        assert_eq!(plan_steps(30.0), Some((1, 30.0)));
        let (n, dt) = plan_steps(3600.0).expect("steps");
        assert_eq!(n, 6);
        assert!((dt - 600.0).abs() < 1e-3);
        let (n, dt) = plan_steps(10.0 * 86_400.0).expect("steps");
        assert_eq!(n, MAX_STEPS_PER_FRAME);
        assert_eq!(dt, 600.0);
    }
}
