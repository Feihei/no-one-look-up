//! The aeolian model's CPU side: the step clock, the calm test, the constants the kernels share
//! with `docs/research/dust.md`, and (in the tests) the CPU mirrors of the three kernels that the
//! GPU is compared against.
//!
//! The kernels in `aeolian.wesl` and `surface_wind.wesl` evaluate the same formulas; the tests
//! here pin them to the numbers of the note's Validation targets 1 and 2.

/// Transport threshold of the 10 m wind, m/s (plan 4.3; Bagnold's fluid threshold of 250 µm sand
/// is 8.14, Kok et al. 2012 Eq. 2.5, 2.21).
pub(crate) const THRESHOLD_MS: f32 = 8.0;
/// Flux coefficient `k`, m² s⁻¹ (m/s)⁻³ of bulk sand volume per unit width (plan 4.3: 167× the
/// physical flux, a morphological acceleration, `dust.md` Deviation 1).
pub(crate) const FLUX_COEFFICIENT: f32 = 2e-6;
/// Saturation length, m (Kok §2.3.2.2, 1–20 m).
pub(crate) const SATURATION_LENGTH_M: f32 = 10.0;
/// Angle of repose of dry sand, degrees (Kok §3.1.1).
pub(crate) const TALUS_DEG: f32 = 34.0;
/// Avalanche passes per transport step.
pub(crate) const AVALANCHE_PASSES: u32 = 2;
/// Fraction of the largest stable transfer one avalanche pass moves (`erosion.md`).
pub(crate) const AVALANCHE_RATE: f32 = 0.15;
/// Transport steps per second of real time at most (`dust.md` GPU mapping: the 20 Hz kernel cap).
pub(crate) const STEP_HZ: f32 = 20.0;
/// The longest bed step, s of real time: a frame hitch advances the bed by this much, not by
/// the hitch (the donor-cell update is explicit).
pub(crate) const MAX_STEP_S: f32 = 0.1;
/// The largest speed-up of the wind over a crest (`surface_wind.wesl`): the wind can exceed
/// the threshold at a 10 m wind of `threshold / SPEEDUP_MAX` and above.
pub(crate) const SPEEDUP_MAX: f32 = 2.0;
/// Boundary of the bed's time coupling: the clock's time scale is capped here (the bed would
/// jump whole dune widths in a step above it).
pub(crate) const MAX_TIME_SCALE: f32 = 10_000.0;

/// Whether the wind can move sand at all: the 10 m wind times the largest speed-up reaches the
/// threshold. Below it a step is skipped and calm weather costs nothing.
pub(crate) fn can_transport(wind_ms: f32, threshold_ms: f32) -> bool {
    wind_ms * SPEEDUP_MAX > threshold_ms
}

/// Real time that has accumulated toward the next transport step.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct StepClock {
    elapsed_s: f32,
}

impl StepClock {
    /// Add a frame's real time. Returns the real seconds the step covers once at least
    /// `1 / hz` have accumulated (capped at [`MAX_STEP_S`]), else `None`.
    pub(crate) fn advance(&mut self, dt_s: f32, hz: f32) -> Option<f32> {
        self.elapsed_s += dt_s.max(0.0);
        if hz <= 0.0 || self.elapsed_s < 1.0 / hz {
            return None;
        }
        let step = self.elapsed_s.min(MAX_STEP_S);
        self.elapsed_s = 0.0;
        Some(step)
    }

    /// Forget the accumulated time (the terrain was rebuilt or the wind went calm).
    pub(crate) fn reset(&mut self) {
        self.elapsed_s = 0.0;
    }
}

#[cfg(test)]
pub(crate) mod mirror {
    //! CPU mirrors of the WESL kernels, in the same f32 arithmetic.

    /// `dust.md` Model adopted: the flux law `k u² (u − u_t)+`.
    pub(crate) fn flux_capacity(u: f32, k: f32, threshold: f32) -> f32 {
        k * u * u * (u - threshold).max(0.0)
    }

    pub(crate) const VENTURI_K: f32 = 0.005;
    pub(crate) const RING_M: f32 = 50.0;
    pub(crate) const SHADOW_MARCH_M: f32 = 10.0;

    pub(crate) fn venturi_speedup(above_mean: f32) -> f32 {
        (1.0 + VENTURI_K * above_mean).clamp(0.5, 2.0)
    }

    pub(crate) fn shadow_factor(slope: f32) -> f32 {
        let beta = slope.atan();
        let (full, none) = (15f32.to_radians(), 10f32.to_radians());
        ((full - beta) / (full - none)).clamp(0.0, 1.0)
    }

    /// Bagnold's fluid threshold of the 10 m wind for grains of `d_m` and roughness
    /// `z0 = k_s / 30` with `k_s = ks_over_d · d_m` (Kok et al. 2012 Eq. 2.5 and 2.21).
    pub(crate) fn bagnold_threshold_10m(d_m: f64, ks_over_d: f64) -> f64 {
        let (rho_p, rho_a, g) = (2650.0, 1.2, 9.81);
        let u_star = 0.10 * ((rho_p - rho_a) * g * d_m / rho_a).sqrt();
        let z0 = ks_over_d * d_m / 30.0;
        u_star / 0.4 * (10.0 / z0).ln()
    }

    /// Whether a grid face lets sand through.
    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    pub(crate) enum Boundary {
        Open,
        Closed,
    }

    /// The layers and the wind of one transport step.
    #[derive(Clone, Debug)]
    pub(crate) struct Field {
        pub n: usize,
        pub cell: f32,
        pub bedrock: Vec<f32>,
        pub sediment: Vec<f32>,
        pub dir: [f32; 2],
        pub u10: f32,
        pub boundary: Boundary,
        pub k: f32,
        pub threshold: f32,
        pub l_sat: f32,
        pub dt_bed: f32,
    }

    impl Field {
        fn idx(&self, x: i32, y: i32) -> usize {
            y as usize * self.n + x as usize
        }

        fn inside(&self, x: i32, y: i32) -> bool {
            x >= 0 && y >= 0 && (x as usize) < self.n && (y as usize) < self.n
        }

        fn surface(&self, x: i32, y: i32) -> f32 {
            let m = self.n as i32 - 1;
            let k = self.idx(x.clamp(0, m), y.clamp(0, m));
            self.bedrock[k] + self.sediment[k]
        }

        fn sign(&self) -> (i32, i32) {
            (
                if self.dir[0] >= 0.0 { 1 } else { -1 },
                if self.dir[1] >= 0.0 { 1 } else { -1 },
            )
        }

        /// The surface wind of a cell (`aeolian_capacity`).
        pub(crate) fn surface_wind(&self, x: i32, y: i32) -> f32 {
            let z = self.surface(x, y);
            let taps = ((SHADOW_MARCH_M / self.cell + 1e-3).floor() as i32).max(1);
            let mut slope = -1e9f32;
            for i in 1..=taps {
                let ox = (self.dir[0] * i as f32).round_ties_even() as i32;
                let oy = (self.dir[1] * i as f32).round_ties_even() as i32;
                slope = slope.max((self.surface(x - ox, y - oy) - z) / (i as f32 * self.cell));
            }
            let ring = RING_M / self.cell;
            let s = std::f32::consts::FRAC_1_SQRT_2;
            let dirs = [
                [1.0, 0.0],
                [s, s],
                [0.0, 1.0],
                [-s, s],
                [-1.0, 0.0],
                [-s, -s],
                [0.0, -1.0],
                [s, -s],
            ];
            let mut mean = 0.0;
            for d in dirs {
                let ox = (d[0] * ring).round_ties_even() as i32;
                let oy = (d[1] * ring).round_ties_even() as i32;
                mean += self.surface(x + ox, y + oy);
            }
            mean *= 0.125;
            self.u10 * venturi_speedup(z - mean) * shadow_factor(slope)
        }

        pub(crate) fn capacity(&self) -> Vec<f32> {
            let mut cap = vec![0.0; self.n * self.n];
            for y in 0..self.n as i32 {
                for x in 0..self.n as i32 {
                    let u = self.surface_wind(x, y);
                    cap[self.idx(x, y)] = flux_capacity(u, self.k, self.threshold);
                }
            }
            cap
        }

        /// The scans (`aeolian_flux`): the flux through every cell's downwind x face and z face.
        pub(crate) fn flux_scan(&self, cap: &[f32]) -> (Vec<f32>, Vec<f32>) {
            let (sx, sy) = self.sign();
            let a = [self.dir[0].abs(), self.dir[1].abs()];
            let mut qx = vec![0.0f32; self.n * self.n];
            let mut qz = vec![0.0f32; self.n * self.n];
            let closed = self.boundary == Boundary::Closed;
            let n = self.n;
            let scan = |axis: usize, line: usize, out: &mut Vec<f32>| {
                let share = a[axis] / (a[0] + a[1]);
                let rho = 1.0 - (-self.cell / (self.l_sat * a[axis].max(1e-3))).exp();
                let forward = if axis == 0 { sx > 0 } else { sy > 0 };
                let mut arriving = 0.0f32;
                for m in 0..n {
                    let i = if forward { m } else { n - 1 - m };
                    let k = if axis == 0 {
                        line * n + i
                    } else {
                        i * n + line
                    };
                    let mut q = 0.0;
                    if a[axis] > 1e-6 {
                        let relaxed = arriving + (cap[k] * a[axis] - arriving) * rho;
                        let supply = arriving + self.sediment[k] * self.cell / self.dt_bed * share;
                        q = relaxed.min(supply).max(0.0);
                        if closed && m == n - 1 {
                            q = 0.0;
                        }
                    }
                    out[k] = q;
                    arriving = q;
                }
            };
            for line in 0..n {
                scan(0, line, &mut qx);
                scan(1, line, &mut qz);
            }
            (qx, qz)
        }

        /// The bed change (`aeolian_move`); returns the new sediment layer.
        pub(crate) fn move_bed(&self, qx: &[f32], qz: &[f32]) -> Vec<f32> {
            let (sx, sy) = self.sign();
            let mut out = self.sediment.clone();
            for y in 0..self.n as i32 {
                for x in 0..self.n as i32 {
                    let k = self.idx(x, y);
                    let mut arriving = 0.0;
                    if self.inside(x - sx, y) {
                        arriving += qx[self.idx(x - sx, y)];
                    }
                    if self.inside(x, y - sy) {
                        arriving += qz[self.idx(x, y - sy)];
                    }
                    let leaving = qx[k] + qz[k];
                    out[k] = (self.sediment[k] + self.dt_bed / self.cell * (arriving - leaving))
                        .max(0.0);
                }
            }
            out
        }

        /// Sediment volume, m³ (f64 sum).
        pub(crate) fn volume(&self) -> f64 {
            self.sediment.iter().map(|h| f64::from(*h)).sum::<f64>()
                * f64::from(self.cell)
                * f64::from(self.cell)
        }

        /// The sediment that has left through open faces in one step, m³, for the conservation
        /// check of an open boundary.
        pub(crate) fn exported(&self, qx: &[f32], qz: &[f32]) -> f64 {
            let (sx, sy) = self.sign();
            let mut lost = 0.0f64;
            for y in 0..self.n as i32 {
                for x in 0..self.n as i32 {
                    let k = self.idx(x, y);
                    if !self.inside(x + sx, y) {
                        lost += f64::from(qx[k]);
                    }
                    if !self.inside(x, y + sy) {
                        lost += f64::from(qz[k]);
                    }
                }
            }
            // q is m²/s through a face one cell wide: m³ over the step.
            lost * f64::from(self.cell) * f64::from(self.dt_bed)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::mirror::*;
    use super::*;

    const K: f32 = FLUX_COEFFICIENT;

    /// Validation 1: the flux law and its threshold.
    #[test]
    fn the_flux_law_matches_the_plan() {
        for u in [0.0, 3.0, 7.99, 8.0] {
            assert_eq!(flux_capacity(u, K, THRESHOLD_MS), 0.0, "u = {u}");
        }
        for (u, want) in [
            (10.0, 4.0e-4),
            (12.0, 1.15e-3),
            (16.0, 4.1e-3),
            (20.0, 9.6e-3),
        ] {
            let got = flux_capacity(u, K, THRESHOLD_MS);
            // 1.152e-3 and 4.096e-3 are the exact values; the note rounds them.
            assert!(
                (got - want).abs() <= want * 0.004,
                "u = {u}: {got} vs {want}"
            );
        }
        let ratio = flux_capacity(12.0, K, THRESHOLD_MS) / flux_capacity(10.0, K, THRESHOLD_MS);
        assert!((ratio - 2.88).abs() < 1e-4, "{ratio}");
    }

    /// Validation 1: the coefficient is 167 times the physical flux, and Bagnold's threshold of
    /// 250 µm sand lies where the note computed it.
    #[test]
    fn the_plan_flux_is_a_morphological_acceleration_of_the_physical_one() {
        let physical = 1.2e-8_f64;
        let factor = f64::from(FLUX_COEFFICIENT) / physical;
        assert!((factor - 167.0).abs() < 1.0, "{factor}");
        let fluid = bagnold_threshold_10m(2.5e-4, 1.0);
        assert!((fluid - 8.14).abs() < 0.05, "{fluid}");
        let double = bagnold_threshold_10m(2.5e-4, 2.0);
        assert!((double - 7.74).abs() < 0.05, "{double}");
        // The plan's 8 m/s is within 2 % of the fluid threshold.
        assert!((f64::from(THRESHOLD_MS) - fluid).abs() / fluid < 0.02);
    }

    /// Validation 2: the shadow of a 4 m cliff facing the wind, level ground and the 12.5° case.
    #[test]
    fn a_cliff_shadows_ten_metres_downwind_and_level_ground_does_not() {
        // Slope over the march: tangent of the steepest upwind rise; the cliff face is at x = 0
        // and the tap `d` metres upwind of a cell at `x` m downwind sits on the plateau when
        // d ≥ x (the plateau is upwind of the face), so the cliff is *behind* the cell otherwise.
        // Cliff: the surface is 4 m higher upwind of x = 0, level downwind of it.
        let slope_at = |x: f32| {
            let mut slope = f32::MIN;
            let mut d = 2.0;
            while d <= SHADOW_MARCH_M + 1e-3 {
                let up = x - d;
                let rise = if up <= 0.0 { 4.0 } else { 0.0 };
                slope = slope.max(rise / d);
                d += 2.0;
            }
            slope
        };
        for x in [3.0, 6.0, 9.0] {
            let slope = slope_at(x);
            assert!(
                slope.atan().to_degrees() >= 21.8 - 0.05,
                "x = {x}: {} deg",
                slope.atan().to_degrees()
            );
            assert_eq!(shadow_factor(slope), 0.0, "x = {x}");
        }
        // At 12 m the face is beyond the march radius, even at the 10 m tap.
        assert_eq!(shadow_factor(slope_at(12.0)), 1.0);
        assert_eq!(shadow_factor(0.0), 1.0);
        assert_eq!(shadow_factor(-0.5), 1.0);
        let half = 12.5f32.to_radians().tan();
        assert!((shadow_factor(half) - 0.5).abs() < 1e-5);
    }

    /// Validation 2: the speed-up.
    #[test]
    fn the_speedup_follows_the_height_above_the_ring_mean_and_is_clamped() {
        assert!((venturi_speedup(2.0) - 1.01).abs() < 1e-6);
        assert!((venturi_speedup(40.0) - 1.2).abs() < 1e-6);
        assert_eq!(venturi_speedup(1000.0), 2.0);
        assert_eq!(venturi_speedup(-1000.0), 0.5);
        assert_eq!(venturi_speedup(0.0), 1.0);
    }

    #[test]
    fn calm_weather_skips_the_step() {
        assert!(!can_transport(4.0, THRESHOLD_MS));
        assert!(!can_transport(3.0, THRESHOLD_MS));
        assert!(can_transport(4.1, THRESHOLD_MS));
        assert!(can_transport(16.0, THRESHOLD_MS));
    }

    #[test]
    fn the_step_clock_runs_at_most_twenty_times_a_second() {
        let mut clock = StepClock::default();
        let mut steps = 0;
        let mut covered = 0.0;
        for _ in 0..600 {
            if let Some(dt) = clock.advance(1.0 / 60.0, STEP_HZ) {
                steps += 1;
                covered += dt;
            }
        }
        assert!((199..=201).contains(&steps), "{steps}");
        assert!((covered - 10.0).abs() < 0.1, "{covered}");
        // A hitch covers the cap, not the hitch.
        assert_eq!(StepClock::default().advance(2.0, STEP_HZ), Some(MAX_STEP_S));
        assert_eq!(StepClock::default().advance(0.01, STEP_HZ), None);
        assert_eq!(StepClock::default().advance(1.0, 0.0), None);
    }

    /// A pile of sand on a bedrock plane, 16 m/s along +x.
    fn pile(n: usize, cell: f32, boundary: Boundary) -> Field {
        let mut sediment = vec![0.0f32; n * n];
        let c = n as f32 * 0.5;
        for y in 0..n {
            for x in 0..n {
                let r2 =
                    ((x as f32 + 0.5 - c).powi(2) + (y as f32 + 0.5 - c).powi(2)) * cell * cell;
                sediment[y * n + x] = 3.0 * (-r2 / (2.0 * 20.0 * 20.0)).exp();
            }
        }
        Field {
            n,
            cell,
            bedrock: vec![0.0; n * n],
            sediment,
            dir: [1.0, 0.0],
            u10: 16.0,
            boundary,
            k: K,
            threshold: THRESHOLD_MS,
            l_sat: SATURATION_LENGTH_M,
            dt_bed: 0.05 * 600.0,
        }
    }

    fn run(field: &mut Field, steps: usize) {
        for _ in 0..steps {
            let cap = field.capacity();
            let (qx, qz) = field.flux_scan(&cap);
            field.sediment = field.move_bed(&qx, &qz);
        }
    }

    /// Validation 3 (CPU mirror): a closed pile loses nothing, an open one only what crosses the
    /// edge; no negative thickness; and the pile moves downwind.
    #[test]
    fn the_cpu_scheme_conserves_sand_and_moves_the_pile_downwind() {
        let mut field = pile(128, 2.0, Boundary::Closed);
        let before = field.volume();
        let com = |f: &Field| {
            let mut sum = 0.0f64;
            let mut x_sum = 0.0f64;
            for (i, h) in f.sediment.iter().enumerate() {
                sum += f64::from(*h);
                x_sum += f64::from(*h) * ((i % f.n) as f64 + 0.5);
            }
            x_sum / sum * f64::from(f.cell)
        };
        let x0 = com(&field);
        run(&mut field, 200);
        let after = field.volume();
        assert!(
            (after - before).abs() / before < 1e-5,
            "{before} -> {after}"
        );
        assert!(field.sediment.iter().all(|h| *h >= 0.0));
        assert!(com(&field) > x0 + 5.0, "{x0} -> {}", com(&field));
        // Open boundary: what is gone left through the downwind edge.
        let mut open = pile(128, 2.0, Boundary::Open);
        let before = open.volume();
        let mut gone = 0.0;
        for _ in 0..400 {
            let cap = open.capacity();
            let (qx, qz) = open.flux_scan(&cap);
            gone += open.exported(&qx, &qz);
            open.sediment = open.move_bed(&qx, &qz);
        }
        assert!(gone > 0.0, "the pile never reached the edge in this run");
        let after = open.volume();
        assert!(
            (before - after - gone).abs() / before < 1e-4,
            "{before} - {after} vs exported {gone}"
        );
    }

    /// Validation 5 (CPU mirror): below the threshold on flat ground nothing moves, bit for bit.
    #[test]
    fn flat_ground_below_the_threshold_is_bit_unchanged() {
        let mut field = pile(64, 2.0, Boundary::Closed);
        field.sediment = vec![4.0; 64 * 64];
        field.u10 = 7.9;
        let before = field.sediment.clone();
        run(&mut field, 100);
        assert_eq!(field.sediment, before);
        // At 8.1 m/s the open edge erodes: the flux has nothing to leave with but a flat plane
        // saturates, so use the pile.
        let mut moving = pile(64, 2.0, Boundary::Closed);
        moving.u10 = 8.1;
        let before = moving.sediment.clone();
        run(&mut moving, 100);
        assert_ne!(moving.sediment, before);
    }

    /// Validation 4 (oracle): the avalanche rule at talus 34° moves `½ r_t E` per pass from the
    /// top of a column and conserves its sum.
    #[test]
    fn the_avalanche_rule_relaxes_a_steep_column_monotonically() {
        // Cell 2 m; the left cell stands 40° above its neighbour: drop = 2 tan 40°.
        let drop = 2.0 * 40f32.to_radians().tan();
        let mut heights = [drop, 0.0];
        let mut last_slope = f32::MAX;
        let total: f32 = heights.iter().sum();
        for _ in 0..50 {
            let (moved, _, share) = crate::hydrology::model::thermal_moves(
                &[heights[0] - heights[1]],
                &[2.0],
                heights[0].max(0.0) + 100.0,
                TALUS_DEG,
                AVALANCHE_RATE,
            );
            assert!(moved >= 0.0 && share[0] <= moved + 1e-9);
            heights[0] -= moved;
            heights[1] += moved;
            let slope = ((heights[0] - heights[1]) / 2.0).atan().to_degrees();
            assert!(slope <= last_slope + 1e-4, "{slope} after {last_slope}");
            last_slope = slope;
        }
        assert!((heights.iter().sum::<f32>() - total).abs() < 1e-6);
        assert!((34.0 - 1e-3..40.0).contains(&last_slope), "{last_slope}");
        // The first pass moves ½ · 0.15 · (2 tan 40° − 2 tan 34°).
        let e = 2.0 * 40f32.to_radians().tan() - 2.0 * 34f32.to_radians().tan();
        let (first, _, _) = crate::hydrology::model::thermal_moves(
            &[2.0 * 40f32.to_radians().tan()],
            &[2.0],
            100.0,
            TALUS_DEG,
            AVALANCHE_RATE,
        );
        assert!((first - 0.5 * AVALANCHE_RATE * e).abs() < 1e-6);
    }
}
