//! How the rain and snow are drawn, as a game wants them to read: a draw-only style over the
//! simulated particles (`docs/research/precipitation-look.md`).

use bevy::prelude::*;

/// How rain and snow are drawn. Draw-only: the particles' simulation, the wetness and snow they
/// leave, the splashes, the fog and the gameplay queries never read it. The default is the
/// physical look; [`PrecipitationLook::readable`] is a stylised one for orthographic and strategy
/// cameras, where physically sized drops and flakes read as sleet and noise. Insert one before
/// the plugins to start with it, or change it at any time.
#[derive(Resource, Clone, Copy, Debug, PartialEq, Reflect)]
pub struct PrecipitationLook {
    /// Multiplier on the vertical part of a rain streak (its fall): longer, more upright rain.
    /// Default 1.
    pub rain_fall_scale: f32,
    /// Multiplier on the horizontal part of a rain streak (its wind drift): less slant.
    /// Default 1.
    pub rain_wind_scale: f32,
    /// Multiplier on a rain streak's colour (rgb) and opacity (alpha). Default white, opaque: the
    /// physically lit streak.
    pub rain_tint: LinearRgba,
    /// Log-normal spread σ of each flake's drawn size about its own (mean 1, coefficient of
    /// variation `√(e^{σ²} − 1)`: 0.53 at 0.5). Default 0: every flake its own physical size.
    pub flake_size_spread: f32,
    /// Smallest half width a flake is drawn at, px, before the spread. Default 1.
    pub flake_min_half_width_px: f32,
}

impl Default for PrecipitationLook {
    fn default() -> Self {
        Self {
            rain_fall_scale: 1.0,
            rain_wind_scale: 1.0,
            rain_tint: LinearRgba::WHITE,
            flake_size_spread: 0.0,
            flake_min_half_width_px: 1.0,
        }
    }
}

impl PrecipitationLook {
    /// A look that reads at a glance from far away (an orthographic or strategy camera): rain as
    /// long, nearly upright streaks a little bluer and stronger than the physical ones, snow as
    /// flakes of visibly different sizes. Chosen in the capture review of
    /// `docs/research/precipitation-look.md`.
    pub fn readable() -> Self {
        Self {
            rain_fall_scale: 4.0,
            rain_wind_scale: 0.5,
            rain_tint: LinearRgba::new(0.85, 0.9, 1.0, 0.7),
            flake_size_spread: 0.5,
            flake_min_half_width_px: 1.5,
        }
    }

    /// The draw uniform's `look` (fall scale, wind scale, size spread, minimum flake half width)
    /// and `tint`, with every factor kept finite and non-negative.
    pub(crate) fn gpu(&self) -> (Vec4, Vec4) {
        let sane = |v: f32, fallback: f32| if v.is_finite() { v.max(0.0) } else { fallback };
        let tint = self.rain_tint;
        (
            Vec4::new(
                sane(self.rain_fall_scale, 1.0),
                sane(self.rain_wind_scale, 1.0),
                sane(self.flake_size_spread, 0.0),
                sane(self.flake_min_half_width_px, 1.0),
            ),
            Vec4::new(
                sane(tint.red, 1.0),
                sane(tint.green, 1.0),
                sane(tint.blue, 1.0),
                sane(tint.alpha, 1.0),
            ),
        )
    }
}

/// A rain streak (metres, world axes) with its fall scaled by `fall` and its drift by `wind`:
/// the CPU mirror of `draw.wesl`'s styled streak.
#[cfg(test)]
pub(crate) fn styled_streak(streak: Vec3, fall: f32, wind: f32) -> Vec3 {
    Vec3::new(streak.x * wind, streak.y * fall, streak.z * wind)
}

/// A flake's size factor `exp(σ z − σ²/2)` for the standard normal `z` that Box–Muller makes of
/// the uniforms `u1`, `u2` in (0, 1), clamped to ±2 so no flake is drawn larger than e^{2σ} of its
/// own size: the CPU mirror of `draw.wesl`'s `flake_size_factor`.
#[cfg(test)]
pub(crate) fn flake_size_factor(u1: f32, u2: f32, sigma: f32) -> f32 {
    let z = ((-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()).clamp(-2.0, 2.0);
    (sigma * z - 0.5 * sigma * sigma).exp()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::precipitation::model::{drop_fall_speed, marshall_palmer_lambda};

    /// Deterministic uniforms in (0, 1) (a 64-bit LCG's top bits).
    struct Uniforms(u64);

    impl Uniforms {
        fn next(&mut self) -> f32 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((self.0 >> 40) as f32 + 0.5) / 16_777_216.0
        }
    }

    /// Mean on-screen angle from vertical, degrees, of the `Rain` preset's streaks (4 mm/h,
    /// water-weighted Marshall–Palmer drops at their Atlas–Ulbrich speeds, the 7 m/s wind blowing
    /// straight across the view, the worst case) under an orthographic camera pitched `pitch_deg`
    /// down. An orthographic projection is affine, so a streak's screen direction is its view-space
    /// x and y.
    fn mean_streak_angle(pitch_deg: f32, fall: f32, wind: f32) -> f32 {
        let camera = Transform::default().looking_to(
            Vec3::new(
                0.0,
                -pitch_deg.to_radians().sin(),
                -pitch_deg.to_radians().cos(),
            ),
            Vec3::Y,
        );
        let (right, up) = (*camera.right(), *camera.up());
        let lambda = marshall_palmer_lambda(4.0);
        let mut u = Uniforms(7);
        let n = 10_000;
        let mut sum = 0.0;
        for _ in 0..n {
            let d_mm = -(u.next() * u.next() * u.next() * u.next()).ln() / lambda;
            let velocity = Vec3::new(7.0, -drop_fall_speed(d_mm), 0.0);
            let s = styled_streak(velocity / 60.0, fall, wind);
            let (x, y) = (s.dot(right), s.dot(up));
            sum += x.abs().atan2(y.abs()).to_degrees();
        }
        sum / n as f32
    }

    /// Issue #25 (precipitation-look.md Validation 2): with the fall ×4 and the wind ×0.5 under an
    /// orthographic camera pitched 35°, the `Rain` preset's streaks stand within 25° of vertical
    /// on screen; at the defaults they lean more than 45° (the sleet the issue describes).
    #[test]
    fn a_readable_rain_stands_upright_under_an_orthographic_camera() {
        let styled = mean_streak_angle(35.0, 4.0, 0.5);
        let physical = mean_streak_angle(35.0, 1.0, 1.0);
        println!(
            "RESULT rain streak from vertical: physical {physical:.1}°, readable {styled:.1}°"
        );
        assert!(styled < 25.0, "styled rain leans {styled}°");
        assert!(physical > 45.0, "physical rain leans {physical}°");
        let readable = PrecipitationLook::readable();
        assert_eq!(
            mean_streak_angle(35.0, readable.rain_fall_scale, readable.rain_wind_scale),
            styled
        );
    }

    /// Issue #25 (Validation 3): at a spread of 0.5 the flakes' drawn sizes vary with a
    /// coefficient of variation of at least 0.3 about an unchanged mean; at 0 every factor is
    /// exactly 1, so the default draws every flake at its own size.
    #[test]
    fn a_size_spread_varies_the_flakes_about_their_own_size() {
        let mut u = Uniforms(11);
        let factors: Vec<f32> = (0..10_000)
            .map(|_| flake_size_factor(u.next(), u.next(), 0.5))
            .collect();
        let n = factors.len() as f32;
        let mean = factors.iter().sum::<f32>() / n;
        let sd = (factors.iter().map(|f| (f - mean).powi(2)).sum::<f32>() / n).sqrt();
        assert!(sd / mean >= 0.3, "coefficient of variation {}", sd / mean);
        assert!((mean - 1.0).abs() < 0.05, "mean factor {mean}");
        let mut u = Uniforms(13);
        for _ in 0..1000 {
            assert_eq!(flake_size_factor(u.next(), u.next(), 0.0), 1.0);
        }
    }

    /// The default look is the physical one: the draw uniform carries the identity factors.
    #[test]
    fn the_default_look_is_the_identity() {
        let (look, tint) = PrecipitationLook::default().gpu();
        assert_eq!(look, Vec4::new(1.0, 1.0, 0.0, 1.0));
        assert_eq!(tint, Vec4::ONE);
        let bad = PrecipitationLook {
            rain_fall_scale: f32::NAN,
            flake_size_spread: -1.0,
            ..default()
        };
        assert_eq!(bad.gpu().0, Vec4::new(1.0, 1.0, 0.0, 1.0));
    }
}
