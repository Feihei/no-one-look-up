//! The precipitation model that has a closed form: rain microphysics, particle counts, the
//! far-field extinction and its split from the weather's visibility. The GPU kernels in
//! `common.wesl`, `precip.wesl` and `draw.wesl` and the fog kernel evaluate the same formulas;
//! the tests here pin them to the numbers in `docs/research/precipitation.md`.

use crate::fog::{CLEAR_AIR_EXTINCTION, extinction_for_visibility};

/// Marshall–Palmer slope `Λ = 4.1 R^−0.21` in mm⁻¹ for a liquid-equivalent rate `R` in mm/h
/// (Uijlenhoet 2001, Eq. 11). Rates below 0.01 mm/h are held there so Λ stays finite.
pub(crate) fn marshall_palmer_lambda(rate_mm_h: f32) -> f32 {
    4.1 * rate_mm_h.max(0.01).powf(-0.21)
}

/// Terminal velocity of a raindrop of diameter `d_mm` in m/s: the Atlas & Ulbrich power law
/// `3.778 D^0.67` (Uijlenhoet 2001, Eq. 5), capped at the 9 m/s plateau.
pub(crate) fn drop_fall_speed(d_mm: f32) -> f32 {
    (3.778 * d_mm.max(0.0).powf(0.67)).min(9.0)
}

/// Water-weighted mean fall speed in m/s for slope `lambda` (mm⁻¹): the mean of
/// [`drop_fall_speed`] over the diameters weighted by the water they carry, `D³ e^(−ΛD)`. The
/// speed that makes a rain rate come out right, and the slant of a rain shaft.
pub(crate) fn water_weighted_fall_speed(lambda: f32) -> f32 {
    const STEPS: usize = 256;
    let top = 24.0 / lambda;
    let step = top / STEPS as f32;
    let (mut weight, mut weighted) = (0.0_f64, 0.0_f64);
    for i in 0..STEPS {
        let d = (i as f32 + 0.5) * step;
        let w = f64::from(d * d * d * (-lambda * d).exp());
        weight += w;
        weighted += w * f64::from(drop_fall_speed(d));
    }
    (weighted / weight) as f32
}

/// Rain extinction in m⁻¹ for a liquid rate in mm/h: `2.12×10⁻⁴ R^0.68` (Ulbrich & Atlas 1985
/// through Kanazawa & Uchida 2025).
pub(crate) fn rain_extinction(rate_mm_h: f32) -> f32 {
    2.12e-4 * rate_mm_h.max(0.0).powf(0.68)
}

/// Snow extinction in m⁻¹ for a water-equivalent rate in mm/h: `4.917×10⁻³ S`, linear in `S`
/// (Rasmussen et al. 1999, Eq. 13, dry aggregates: 3.912 m⁻¹ / 795.6 m).
pub(crate) fn snow_extinction(rate_mm_h: f32) -> f32 {
    4.917e-3 * rate_mm_h.max(0.0)
}

/// Extinction of clear air at the `Clear` preset's 40 km visibility, m⁻¹: the fog module's one
/// authority (`fog::CLEAR_AIR_EXTINCTION`).
pub(crate) const FLOOR_EXTINCTION: f32 = CLEAR_AIR_EXTINCTION;

/// The weather's extinction `σ_V`, split into the precipitation that is drawn as shafts and the
/// haze that remains.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct ExtinctionSplit {
    /// Total extinction `3.912 / visibility`, m⁻¹.
    pub total: f32,
    /// Extinction carried by precipitation, m⁻¹.
    pub precipitation: f32,
    /// Extinction left to the haze, m⁻¹; `total − precipitation`.
    pub haze: f32,
}

/// Split `visibility_m` between precipitation and haze: the precipitation's own extinction,
/// capped so the haze keeps at least the clear-air floor. `Weather::visibility_m` stays the one
/// authority on the total.
pub(crate) fn precipitation_split(
    visibility_m: f32,
    precipitation_mm_h: f32,
    snow_fraction: f32,
) -> ExtinctionSplit {
    let total = extinction_for_visibility(visibility_m);
    let snow = precipitation_mm_h * snow_fraction;
    let rain = precipitation_mm_h - snow;
    let wanted = rain_extinction(rain) + snow_extinction(snow);
    let precipitation = wanted.min((total - FLOOR_EXTINCTION).max(0.0));
    ExtinctionSplit {
        total,
        precipitation,
        haze: total - precipitation,
    }
}

/// Share of the particle buffer that is dust at the `DustStorm` preset's dust (plan 4.3).
pub(crate) const DUST_FRACTION: f32 = 0.4;
/// The `Weather::dust` at which the dust fraction reaches [`DUST_FRACTION`]: the `DustStorm`
/// preset's value. Weaker dust ramps the fraction down linearly (`dust.md` Deviation 5).
pub(crate) const DUST_FULL: f32 = 0.9;
/// Fall speed of a dust sprite, m/s (`dust.md` Parameters).
pub(crate) const DUST_FALL_SPEED_MS: f32 = 0.05;
/// Scale height of the dust sprites' respawn heights above the ground, m: the ground fog's
/// haze scale height (`fog.md`).
pub(crate) const DUST_HEIGHT_SCALE_M: f32 = 60.0;
/// Grain diameter of a dust sprite, m (plan 4.3): the size the sprite extinction accounts
/// (`dust_sprite_extinction`, `dust.md` Validation 8). The sprite is *drawn* at
/// `DUST_SPRITE_DRAW_M` — see `dust.md` Deviation 15.
#[cfg(test)]
pub(crate) const DUST_SIZE_M: f32 = 2e-3;
/// Diameter a dust sprite is drawn with, m: a visible puff cluster, not the 2 mm grain. At the
/// grain size a sprite spans 0.02–0.1 px anywhere in the 60 m box, so it fades to nothing under
/// the sub-pixel coverage rule and a dust storm reads as static fog (`dust.md` Deviation 15).
pub(crate) const DUST_SPRITE_DRAW_M: f32 = 0.35;

/// How many particles of each kind are active.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct ParticleCounts {
    pub rain: u32,
    pub snow: u32,
    pub dust: u32,
}

impl ParticleCounts {
    /// Active particles in total; they are the first `total()` of the buffer.
    pub fn total(self) -> u32 {
        self.rain + self.snow + self.dust
    }
}

/// The dust fraction of the buffer for `Weather::dust`: `0.4 · min(1, dust / 0.9)`.
pub(crate) fn dust_fraction(dust: f32) -> f32 {
    DUST_FRACTION * (dust.max(0.0) / DUST_FULL).min(1.0)
}

/// Active particles for a buffer of `max`: rain `min(1, R_liq/20)`, snow `min(1, S/3)` of it,
/// where `R_liq` and `S` are the liquid and snow shares of `precipitation_mm_h`, and dust
/// [`dust_fraction`]. When the fractions sum above 1 (mixed phase) all are divided by the sum.
pub(crate) fn particle_counts(
    max: u32,
    precipitation_mm_h: f32,
    snow_fraction: f32,
    dust: f32,
) -> ParticleCounts {
    let snow = precipitation_mm_h.max(0.0) * snow_fraction.clamp(0.0, 1.0);
    let rain = precipitation_mm_h.max(0.0) - snow;
    let mut f_rain = (rain / 20.0).min(1.0);
    let mut f_snow = (snow / 3.0).min(1.0);
    let mut f_dust = dust_fraction(dust);
    let sum = f_rain + f_snow + f_dust;
    if sum > 1.0 {
        f_rain /= sum;
        f_snow /= sum;
        f_dust /= sum;
    }
    let n_rain = ((max as f32 * f_rain).round() as u32).min(max);
    let n_snow = ((max as f32 * f_snow).round() as u32).min(max - n_rain);
    let n_dust = ((max as f32 * f_dust).round() as u32).min(max - n_rain - n_snow);
    ParticleCounts {
        rain: n_rain,
        snow: n_snow,
        dust: n_dust,
    }
}

/// Height above the ground of a respawning dust sprite, m: the exponential profile of scale
/// `scale` truncated to `[lo, hi]` (the part of the box above the ground), by inverse CDF of
/// `u ∈ (0, 1)`: `lo − H ln(1 − u (1 − e^{−(hi−lo)/H}))`. With `lo = 0`, `hi = 60` it is the
/// note's `−H ln(1 − u (1 − e^{−60/H}))`.
#[cfg(test)]
pub(crate) fn dust_height_m(u: f32, lo: f32, hi: f32, scale: f32) -> f32 {
    lo - scale * (1.0 - u * (1.0 - (-(hi - lo) / scale).exp())).ln()
}

/// Extinction of `count` dust sprites of diameter [`DUST_SIZE_M`] spread over a cube of edge
/// `box_m`, m⁻¹: `n π r²` (geometric cross-section, each sprite counted once).
#[cfg(test)]
pub(crate) fn dust_sprite_extinction(count: u32, box_m: f32) -> f32 {
    let radius = 0.5 * DUST_SIZE_M;
    count as f32 / box_m.powi(3) * std::f32::consts::PI * radius * radius
}

#[cfg(test)]
mod tests {
    use super::*;

    /// CPU transliteration of `pcg4d` and `particle_rand` in `common.wesl`.
    fn pcg4d(v: [u32; 4]) -> [u32; 4] {
        let mut v = v.map(|x| x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223));
        let mix = |v: &mut [u32; 4]| {
            v[0] = v[0].wrapping_add(v[1].wrapping_mul(v[3]));
            v[1] = v[1].wrapping_add(v[2].wrapping_mul(v[0]));
            v[2] = v[2].wrapping_add(v[0].wrapping_mul(v[1]));
            v[3] = v[3].wrapping_add(v[1].wrapping_mul(v[2]));
        };
        mix(&mut v);
        v = v.map(|x| x ^ (x >> 16));
        mix(&mut v);
        v
    }

    fn particle_rand(index: u32, epoch: u32, salt: u32) -> [f32; 4] {
        pcg4d([index, epoch, salt, 0x9E37_79B9]).map(|x| ((x >> 8) as f32 + 0.5) / 16_777_216.0)
    }

    fn drop_diameter_mm(u: [f32; 4], lambda: f32) -> f32 {
        -(u[0] * u[1] * u[2] * u[3]).ln() / lambda
    }

    #[test]
    fn lambda_and_fall_speed_match_the_note() {
        for (rate, lambda) in [
            (0.5, 4.742),
            (1.0, 4.100),
            (4.0, 3.064),
            (10.0, 2.528),
            (20.0, 2.186),
            (25.0, 2.086),
        ] {
            let got = marshall_palmer_lambda(rate);
            assert!((got - lambda).abs() < 2e-3, "Λ({rate}) = {got}");
        }
        for (d, v) in [
            (1.0, 3.78),
            (2.0, 6.01),
            (3.0, 7.89),
            (4.0, 9.0),
            (5.0, 9.0),
        ] {
            assert!((drop_fall_speed(d) - v).abs() < 0.01, "v({d})");
        }
        // Number density N_T = 8000/Λ at 10 mm/h.
        assert!((8000.0 / marshall_palmer_lambda(10.0) - 3165.0).abs() < 2.0);
    }

    /// Validation 4: the sampler the shaders use, over 10⁶ particle indices.
    #[test]
    fn the_water_weighted_sampler_has_the_documented_mean_size_and_speed() {
        for (rate, mean_d, mean_v) in [
            (1.0, 0.976, 3.62),
            (4.0, 1.305, 4.39),
            (10.0, 1.582, 4.98),
            (25.0, 1.918, 5.63),
        ] {
            let lambda = marshall_palmer_lambda(rate);
            let n = 1_000_000_u32;
            let (mut sum_d, mut sum_v, mut big) = (0.0_f64, 0.0_f64, 0u32);
            for i in 0..n {
                let d = drop_diameter_mm(particle_rand(i, 1, 0), lambda);
                sum_d += f64::from(d);
                sum_v += f64::from(drop_fall_speed(d));
                big += u32::from(d > 4.0);
            }
            let (d, v, f) = (
                sum_d / f64::from(n),
                sum_v / f64::from(n),
                f64::from(big) / f64::from(n),
            );
            assert!(
                (d / mean_d - 1.0).abs() < 0.005,
                "R {rate}: mean D {d} vs {mean_d}"
            );
            assert!(
                (v / mean_v - 1.0).abs() < 0.01,
                "R {rate}: mean v {v} vs {mean_v}"
            );
            // Share of drops over 4 mm: the gamma(4, Λ) tail, e^−x (1 + x + x²/2 + x³/6) with
            // x = 4Λ (the note's 0.0001 at 1 mm/h is 6.65e-5 to one figure). Tolerance: 10 %,
            // or four standard deviations of the count when few drops are that large.
            let x = f64::from(4.0 * lambda);
            let tail = (-x).exp() * (1.0 + x + x * x / 2.0 + x * x * x / 6.0);
            let tolerance = (4.0 / (f64::from(n) * tail).sqrt()).max(0.10);
            assert!(
                (f / tail - 1.0).abs() < tolerance,
                "R {rate}: share over 4 mm {f} vs {tail}"
            );
            // The numerical weighted mean agrees with the sampled one.
            let integral = f64::from(water_weighted_fall_speed(lambda));
            assert!(
                (integral / v - 1.0).abs() < 0.01,
                "R {rate}: {integral} vs {v}"
            );
        }
    }

    #[test]
    fn the_water_weighted_speed_matches_the_table() {
        for (rate, v) in [
            (1.0, 3.62),
            (4.0, 4.39),
            (10.0, 4.98),
            (20.0, 5.47),
            (25.0, 5.63),
            (100.0, 6.63),
        ] {
            let got = water_weighted_fall_speed(marshall_palmer_lambda(rate));
            assert!((got / v - 1.0).abs() < 0.01, "R {rate}: {got} vs {v}");
        }
    }

    /// Validation 5: the Marshall–Palmer + Atlas–Ulbrich pair overshoots the rate it was
    /// parametrised by: +16 % at 1 mm/h, +11 % at 10 mm/h, +9 % at 25 mm/h. The bound documents
    /// that so nobody tightens it (the note's "within 12 %" was wrong at 1 mm/h, Deviation 8).
    #[test]
    fn the_rain_rate_integral_overshoots_the_rate_by_at_most_seventeen_percent() {
        for (rate, expected) in [(1.0_f32, 1.16), (10.0, 11.08), (25.0, 27.2)] {
            let lambda = f64::from(marshall_palmer_lambda(rate));
            // 6π×10⁻⁴ ∫ D³ v(D) N(D) dD with the uncapped power law, D in mm, N0 = 8000.
            let (steps, top) = (20_000, 15.0_f64);
            let h = top / f64::from(steps);
            let mut integral = 0.0;
            for i in 0..steps {
                let d = (f64::from(i) + 0.5) * h;
                integral += d.powi(3) * 3.778 * d.powf(0.67) * 8000.0 * (-lambda * d).exp() * h;
            }
            let rate_back = 6.0 * std::f64::consts::PI * 1e-4 * integral;
            assert!(
                (rate_back / f64::from(rate) - 1.0).abs() < 0.17 && rate_back > f64::from(rate),
                "R {rate}: {rate_back}"
            );
            assert!(
                (rate_back - expected).abs() < 0.05 * expected,
                "{rate_back}"
            );
        }
    }

    /// Validation 6.
    #[test]
    fn extinction_and_visibility_match_the_literature_not_the_plan() {
        for (rate, sigma, visibility) in [
            (1.0, 2.120e-4, 18_453.0),
            (4.0, 5.442e-4, 7_189.0),
            (10.0, 1.015e-3, 3_855.0),
            (25.0, 1.892e-3, 2_068.0),
            (100.0, 4.857e-3, 805.0),
        ] {
            let got = rain_extinction(rate);
            assert!((got / sigma - 1.0).abs() < 0.005, "σ_rain({rate}) = {got}");
            assert!((3.912 / got / visibility - 1.0).abs() < 0.005);
        }
        let s1 = snow_extinction(1.0);
        assert!((s1 / 4.917e-3 - 1.0).abs() < 0.002);
        assert!((3.912 / s1 - 795.6).abs() < 1.0);
        assert!((snow_extinction(4.0) / 1.967e-2 - 1.0).abs() < 0.002);
        // The Blizzard preset's own 200 m visibility is reproduced.
        assert!((3.912 / snow_extinction(4.0) - 200.0).abs() < 2.0);
        // The plan's 0.0015 per mm/h would have given 261 m at 10 mm/h; the literature is 15× clearer.
        assert!(rain_extinction(10.0) < 2.5e-3);
        assert!(3.912 / rain_extinction(10.0) > 1500.0);
    }

    /// Validation 7, with the preset targets of `climate/weather.rs`.
    #[test]
    fn the_split_keeps_the_visibility_the_single_authority() {
        // (visibility m, mm/h, snow fraction, σ_p, σ_haze)
        let presets = [
            ("Drizzle", 5_000.0, 0.5, 0.0, 1.323e-4, 6.501e-4),
            ("Rain", 3_000.0, 4.0, 0.0, 5.442e-4, 7.598e-4),
            ("Thunderstorm", 1_500.0, 25.0, 0.0, 1.892e-3, 7.159e-4),
            ("Snow", 1_500.0, 1.5, 1.0, 2.510e-3, 9.780e-5),
            ("Blizzard", 200.0, 4.0, 1.0, 1.946e-2, 9.780e-5),
        ];
        for (name, visibility, rate, snow, sigma_p, haze) in presets {
            let s = precipitation_split(visibility, rate, snow);
            assert!(
                (s.precipitation + s.haze - s.total).abs() < 1e-9,
                "{name} sums"
            );
            assert!(s.haze >= FLOOR_EXTINCTION * 0.999_99, "{name} floor");
            assert!(
                s.precipitation <= s.total - FLOOR_EXTINCTION + 1e-9,
                "{name} cap"
            );
            assert!(
                (s.precipitation / sigma_p - 1.0).abs() < 0.005,
                "{name}: σ_p {}",
                s.precipitation
            );
            assert!(
                (s.haze / haze - 1.0).abs() < 0.005,
                "{name}: haze {}",
                s.haze
            );
        }
        // Dry weather has nothing to redistribute.
        let clear = precipitation_split(40_000.0, 0.0, 0.0);
        assert_eq!(clear.precipitation, 0.0);
        assert_eq!(clear.haze, clear.total);
    }

    /// Validation 1 and 3 (the counting half), and the mixed-phase normalisation.
    #[test]
    fn counts_follow_the_plan_and_mixed_phase_is_normalised() {
        let high = 150_000;
        assert_eq!(
            particle_counts(high, 10.0, 0.0, 0.0),
            ParticleCounts {
                rain: 75_000,
                snow: 0,
                dust: 0
            }
        );
        assert_eq!(
            particle_counts(high, 1.5, 1.0, 0.0),
            ParticleCounts {
                rain: 0,
                snow: 75_000,
                dust: 0
            }
        );
        // Saturation: 20 mm/h of rain, 3 mm/h of snow.
        assert_eq!(particle_counts(high, 25.0, 0.0, 0.0).rain, high);
        assert_eq!(particle_counts(high, 4.0, 1.0, 0.0).snow, high);
        // R = 10 mm/h at half snow: 0.25 + 1.0 unnormalised, 0.2 + 0.8 normalised.
        let mixed = particle_counts(high, 10.0, 0.5, 0.0);
        assert_eq!(
            mixed,
            ParticleCounts {
                rain: 30_000,
                snow: 120_000,
                dust: 0
            }
        );
        assert!(mixed.total() <= high);
        assert_eq!(particle_counts(high, 0.0, 0.0, 0.0).total(), 0);
        for max in [20_000, 60_000, 150_000, 300_000] {
            for (mm, sf, dust) in [
                (10.0, 0.0, 0.0),
                (1.5, 1.0, 0.0),
                (10.0, 0.5, 0.0),
                (30.0, 0.3, 0.0),
                (0.5, 0.0, 0.0),
                (10.0, 0.5, 0.9),
                (30.0, 0.0, 0.9),
                (0.0, 0.0, 1.0),
            ] {
                assert!(particle_counts(max, mm, sf, dust).total() <= max);
            }
            let half = particle_counts(max, 10.0, 0.0, 0.0).rain as i64;
            assert!((half - i64::from(max) / 2).abs() <= 1);
        }
    }

    /// `dust.md` Validation 8 (the counting half): 40 % of the maximum at the preset's dust, a
    /// linear ramp below it, and the shared normalisation with rain.
    #[test]
    fn dust_takes_forty_percent_at_the_preset_and_ramps_below() {
        let high = 150_000i64;
        let dust = |d: f32| i64::from(particle_counts(150_000, 0.0, 0.0, d).dust);
        assert!((dust(0.9) - 60_000).abs() <= 1);
        assert!((dust(0.45) - 30_000).abs() <= 1);
        assert_eq!(dust(0.0), 0);
        assert_eq!(
            dust(1.0),
            60_000,
            "above the preset the fraction stays at 0.4"
        );
        let storm = particle_counts(150_000, 0.0, 0.0, 0.9);
        assert_eq!(
            (storm.rain, storm.snow),
            (0, 0),
            "no rain or snow without precipitation"
        );
        // Dust and heavy rain together: 1.0 + 0.4 normalised to 0.714 + 0.286.
        let both = particle_counts(150_000, 30.0, 0.0, 0.9);
        assert!(i64::from(both.total()) <= high);
        assert!((i64::from(both.dust) - (high as f64 * 0.4 / 1.4).round() as i64).abs() <= 1);
        // The records are laid out rain, then snow, then dust.
        assert_eq!(both.rain + both.snow + both.dust, both.total());
    }

    /// `dust.md` Validation 8 (the sampler): the median respawn height of the exponential
    /// profile over the 60 m box is 22.8 m, the density at the top is `e⁻¹` of the bottom's.
    #[test]
    fn the_dust_respawn_heights_follow_the_haze_profile() {
        let h = |u| dust_height_m(u, 0.0, 60.0, DUST_HEIGHT_SCALE_M);
        let median = 60.0 * -(1.0 - 0.5 * (1.0 - (-1.0f32).exp())).ln();
        assert!((median - 22.8).abs() < 0.1, "{median}");
        assert!((h(0.5) - median).abs() < 1e-4);
        // 10⁶ stratified uniforms stand in for the shader's hash.
        let n = 1_000_000usize;
        let mut heights: Vec<f32> = (0..n).map(|i| h((i as f32 + 0.5) / n as f32)).collect();
        heights.sort_by(f32::total_cmp);
        assert!((heights[n / 2] - 22.8).abs() < 0.5, "{}", heights[n / 2]);
        assert!(heights[0] >= 0.0 && heights[n - 1] <= 60.0);
        // Density ratio top/bottom: histogram of the first and last 1 m.
        let bottom = heights.iter().filter(|h| **h < 1.0).count() as f64;
        let top = heights.iter().filter(|h| **h >= 59.0).count() as f64;
        assert!(
            (top / bottom - (-1.0f64).exp()).abs() < 0.01,
            "{}",
            top / bottom
        );
        // A box that only reaches 32 m above the camera's ground: the same profile, truncated.
        assert!((dust_height_m(1.0 - 1e-6, 0.0, 32.0, 60.0) - 32.0).abs() < 0.01);
        assert!((dust_height_m(0.0, 5.0, 32.0, 60.0) - 5.0).abs() < 1e-5);
    }

    /// `dust.md` Validation 8 and 9: the sprites carry 8.7·10⁻⁷ m⁻¹ of the 4.89·10⁻³ m⁻¹.
    #[test]
    fn the_sprites_carry_almost_none_of_the_dust_extinction() {
        let count = particle_counts(150_000, 0.0, 0.0, 0.9).dust;
        let sprites = dust_sprite_extinction(count, crate::precipitation::PRECIPITATION_BOX_M);
        assert!((sprites - 8.7e-7).abs() < 8.7e-7 * 0.05, "{sprites}");
        let sigma_v = extinction_for_visibility(800.0);
        assert!((sigma_v - 4.890e-3).abs() < 1e-6, "{sigma_v}");
        assert!(sprites / sigma_v < 3e-4);
    }

    /// CPU mirror of the streak setup in `draw.wesl`'s `particle_vertex`: the image length in
    /// pixels of a streak of relative speed `v_perp` across the view at depth `z`, and the
    /// coverage factor that scales its opacity.
    fn streak_pixels(v_perp: f32, z: f32, focal_px: f32) -> f32 {
        v_perp * (1.0 / 60.0) * focal_px / z
    }

    fn coverage(diameter_m: f32, z: f32, focal_px: f32) -> f32 {
        (diameter_m * focal_px / z).min(1.0)
    }

    /// Validation 9: the streak is `|v_rel| T` long in the world, `v T f / z` pixels in the
    /// image, and the opacity follows Garg & Nayar's coverage law: constant inside `z_m = f D`,
    /// falling as `1/z` beyond.
    #[test]
    fn the_streak_geometry_matches_the_note() {
        let t = 1.0_f32 / 60.0;
        assert!((9.0 * t - 0.150).abs() < 1e-3);
        assert!((4.98 * t - 0.083).abs() < 1e-3);
        assert!((1.0 * t - 0.0167).abs() < 1e-4);
        // 1080p, 60° vertical field of view: f = 540 / tan(30°) = 935 px.
        let f = 540.0 / 30f32.to_radians().tan();
        assert!((f - 935.3).abs() < 0.5);
        for (z, px) in [(2.0, 39.0), (5.0, 15.6), (10.0, 7.8), (30.0, 2.6)] {
            assert!((streak_pixels(5.0, z, f) - px).abs() < 0.2, "z {z}");
        }
        for (d_mm, z_m) in [(1.0, 0.94), (1.6, 1.50), (3.0, 2.81)] {
            let z_m_got = d_mm * 1e-3 * f;
            assert!((z_m_got - z_m).abs() < 0.01, "z_m({d_mm} mm) = {z_m_got}");
        }
        let d = 1.6e-3;
        let z_m = d * f;
        assert_eq!(coverage(d, z_m * 0.5, f), 1.0);
        assert!((coverage(d, z_m, f) - 1.0).abs() < 1e-6);
        assert!((coverage(d, 30.0, f) - z_m / 30.0).abs() < 1e-6);
        // The note's 0.25 at z_m gives 0.0125 at 30 m.
        assert!((0.25 * coverage(d, 30.0, f) - 0.0125).abs() < 5e-4);
    }

    /// CPU mirror of the snow stretch in `particle_vertex`: the world length of a flake's streak
    /// for the relative speed `v_rel`, capped at `SNOW_MAX_STRETCH` diameters.
    fn snow_streak_m(v_rel: f32, diameter_m: f32) -> f32 {
        const SNOW_MAX_STRETCH: f32 = 1.0;
        (v_rel / 60.0).min(SNOW_MAX_STRETCH * diameter_m)
    }

    /// Image of a flake at depth `z` moving across the view at `v_rel`: the capsule's total
    /// length over its width, both in pixels. The width is the flake's own, at least two pixels
    /// (`SNOW_MIN_HALF_WIDTH` = 1 px each side, as in the vertex shader).
    fn snow_quad_ratio(v_rel: f32, diameter_m: f32, z: f32, focal_px: f32) -> f32 {
        const SNOW_MIN_HALF_WIDTH: f32 = 1.0;
        let half_width = (0.5 * diameter_m * focal_px / z).max(SNOW_MIN_HALF_WIDTH);
        let length = snow_streak_m(v_rel, diameter_m) * focal_px / z;
        (length + 2.0 * half_width) / (2.0 * half_width)
    }

    /// Deviation 17: a flake is a soft, slightly elongated disc, whatever the wind. The note's
    /// streak `|v_rel| T` made a 6 mm flake in a 4 m/s wind 6.7 cm long (a 12:1 dash); the
    /// capsule is now at most twice as long as it is wide at 4 and 14 m/s, and a flake moving
    /// slower than the cap is stretched as the note says.
    #[test]
    fn a_snowflake_is_a_disc_of_bounded_elongation_in_any_wind() {
        let f = 540.0 / 30f32.to_radians().tan();
        for v_rel in [4.0_f32, 14.0] {
            for d in [0.004_f32, 0.006, 0.008] {
                // Without the cap: the streak the note's formula gives, in diameters.
                assert!(v_rel / 60.0 / d > 8.0, "{v_rel} m/s, {d} m");
                for z in [0.5_f32, 1.0, 2.0, 5.0, 10.0, 30.0] {
                    let ratio = snow_quad_ratio(v_rel, d, z, f);
                    assert!(
                        (1.0..=2.0 + 1e-4).contains(&ratio),
                        "{v_rel} m/s, d {d} m, z {z} m: length/width {ratio}"
                    );
                }
                // A flake wider than its minimum is at its cap: exactly 2:1.
                assert!((snow_quad_ratio(v_rel, d, 1.0, f) - 2.0).abs() < 1e-4);
            }
        }
        // Slower than the cap the note's length stands: 1.7 mm at 0.1 m/s, 4.5 mm flake.
        let slow = snow_streak_m(0.1, 0.0045);
        assert!((slow - 0.1 / 60.0).abs() < 1e-7, "{slow}");
        // Rain streaks are untouched (Validation 9): 9 m/s is 15 cm.
        assert!((9.0_f32 / 60.0 - 0.150).abs() < 1e-3);
    }

    /// Validation 10 (the CPU half): with 7 m/s wind and the 4.4 m/s mean fall speed a shaft
    /// leans atan(7 / 4.4) = 57.8° from vertical.
    #[test]
    fn the_shaft_lean_matches_the_note() {
        let lean = (7.0_f32 / 4.4).atan().to_degrees();
        assert!((lean - 57.8).abs() < 0.1, "{lean}");
        let v = water_weighted_fall_speed(marshall_palmer_lambda(4.0));
        assert!((v - 4.4).abs() < 0.05, "{v}");
    }
}
