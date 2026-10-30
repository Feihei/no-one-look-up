//! What a game's own light has to be, to read against the crate's exposure.
//!
//! The crate meters its exposure on the scene's *illuminance* and not on the frame's content,
//! so a game's own light does not move it (Patry 2021: an "illuminance-based exposure system…
//! takes albedo out of the equation altogether"). That is the stable behaviour, and it leaves a
//! game one job of its own: to give its light a *power*, in physical units, that reads against
//! the exposure the camera has.
//!
//! These are the crate's own numbers, made public. The crate's lightning writes its bolt's
//! `PointLight` in exactly these terms (`light.intensity = FLASH_LUMENS * active.mean * u * u`,
//! `src/lightning/visual.rs`), so a light the crate places and a light a game places are
//! balanced the same way.
//!
//! # The units
//!
//! - A **luminance** is cd/m²: how bright a surface is. The crate's night sky is 0.00016
//!   cd/m², the noon sun 5000.
//! - An **illuminance** is lx: how much light falls on a surface.
//! - A **luminous power** (Bevy's `PointLight::intensity` and `SpotLight::intensity`) is
//!   lumens: the light a source emits in all directions.
//!
//! Bevy maps a light's power to a luminous intensity of `Φ / 4π` (lumens per steradian,
//! `bevy_pbr-0.20.0/src/render/light.rs:574`) and the shader's illuminance at a distance `d` is
//! that over `d²` (`pbr_lighting.wesl`, `getDistanceAttenuation`), so the illuminance on a
//! surface is `Φ / 4πd²` and the surface's luminance is that times its albedo over π.

use bevy::prelude::*;

use crate::celestial::NIGHT_SKY_COLOUR;

/// The largest scene-linear value a surface may have, the ceiling of the crate's `Rgba16Float`
/// target's useful range: about sixteen crossed lightning layers can overlap, and the result
/// stays finite.
pub const DISPLAY_MAX: f32 = 4096.0;

/// The scene-linear value a scene luminance `luminance` (cd/m²) has under camera exposure
/// `exposure` (`2^(−EV₁₀₀) / 1.2`): the number the camera's target holds for it, before any
/// tonemapper.
///
/// The clamp keeps the value finite in `Rgba16Float`; it is the ceiling the crate's own
/// lightning layers use. A game that tonemaps reads this as the input to its curve, not as the
/// final pixel: Bevy's default `TonyMcMapface` maps 0.18 to mid-grey, so a scene-linear value
/// of 0.18 is the middle of the picture.
pub fn display_value(luminance: f32, exposure: f32) -> f32 {
    (luminance * exposure).min(DISPLAY_MAX)
}

/// The scene luminance (cd/m²) that reads at a given scene-linear `display` value under camera
/// exposure `exposure`: the inverse of [`display_value`].
///
/// The inverse is the direction a game authors in: "I want the ground under my lamp to read at
/// 0.18, so it must be lit to this many cd/m²." A `display` of 0 is the black point and any
/// positive value is reachable; a `display` above [`DISPLAY_MAX`] is clamped to it.
pub fn luminance_for_display(display: f32, exposure: f32) -> f32 {
    if exposure <= 0.0 {
        return 0.0;
    }
    (display.clamp(0.0, DISPLAY_MAX) / exposure).max(0.0)
}

/// The `PointLight::intensity` or `SpotLight::intensity` (Bevy's **luminous power in
/// lumens**) that lights a surface `distance` metres away to a scene luminance of `luminance`
/// cd/m², for a surface of albedo `albedo`.
///
/// A light spreads its power over the sphere, so the illuminance on the surface is `Φ / 4πd²`
/// and the surface's luminance is that times its albedo over π:
///
/// ```text
/// L = Φ · albedo / (4 π² d²)   ⇒   Φ = 4 π² d² L / albedo
/// ```
///
/// The `4 π²` is the `4π` of the luminous intensity and the `π` of the Lambertian exitance.
/// A `distance` of 0 gives 0, and an `albedo` of 0 gives no finite power: the surface reflects
/// nothing, so no light reads on it.
///
/// **The same number works for a point light and for a spot light.** Bevy maps *both* to `Φ / 4π`
/// ("we choose to use the same 4\*PI divisor in both cases so that toggling between point
/// light and spot light keeps lit areas lit equally",
/// `bevy_pbr-0.20.0/src/render/light.rs:711`), where Filament would use `π` for a spot. A
/// spot's cone attenuates to 1 inside its inner angle and 0 at its outer angle
/// (`pbr_lighting.wesl`), so the brightness on the axis is the same as a point light's and the
/// cone only decides *where* the light lands: a spot throws none of its power behind it.
pub fn light_intensity(luminance: f32, distance: f32, albedo: f32) -> f32 {
    if albedo <= 0.0 || distance <= 0.0 {
        return 0.0;
    }
    let four_pi_sq = 4.0 * std::f32::consts::PI * std::f32::consts::PI;
    four_pi_sq * distance * distance * luminance.max(0.0) / albedo
}

/// The illuminance (lx) a light of `intensity` (luminous power in lumens) puts on a surface
/// `distance` metres away: `Φ / 4πd²`.
///
/// The inverse of [`light_intensity`]'s falloff, without the surface's albedo: this is the
/// light *falling* on the surface, and the surface's luminance is this times its albedo over π.
pub fn light_illuminance(intensity: f32, distance: f32) -> f32 {
    if distance <= 0.0 {
        return 0.0;
    }
    intensity.max(0.0) / (4.0 * std::f32::consts::PI * distance * distance)
}

/// The colour a light reads as at night: the *scotopic* colour, the cool blue-green the rod's
/// eye expects when the cones have given way (Jensen et al. 2001 §7 on the blue shift, and the
/// scotopic curve, whose peak is at 507 nm).
///
/// It is the crate's own night sky's colour, normalised to a luminance of 1, so a game's lamp
/// and the crate's sky are *one* night and not two. A warm lamp in the crate's night reads as
/// a different scene laid over the sky.
pub fn night_light_colour() -> Vec3 {
    NIGHT_SKY_COLOUR
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The round trip: a scene-linear value and the luminance that reads as it.
    #[test]
    fn the_round_trip_holds() {
        for exposure in [0.01_f32, 0.5, 22.4, 4096.0] {
            for display in [0.001_f32, 0.05, 0.18, 0.9, 1.0] {
                let luminance = luminance_for_display(display, exposure);
                let back = display_value(luminance, exposure);
                assert!(
                    (back - display).abs() <= 1e-3 * display.max(1e-6),
                    "{display} at {exposure} came back as {back}"
                );
            }
        }
    }

    /// A light's power: the illuminance it puts on a surface at the distance, and the luminance
    /// of that surface.
    #[test]
    fn a_lights_power_is_right() {
        for (luminance, distance, albedo) in [
            (1.0_f32, 3.0_f32, 0.2_f32),
            (0.5, 10.0, 0.5),
            (12.0, 1.5, 0.8),
        ] {
            let intensity = light_intensity(luminance, distance, albedo);
            let illuminance = light_illuminance(intensity, distance);
            let got = illuminance * albedo / std::f32::consts::PI;
            assert!(
                (got - luminance).abs() <= 1e-4 * luminance,
                "{luminance} wanted, {got} got (intensity {intensity})"
            );
        }
    }

    /// Bevy's own calibration: a 1 000 000-lumen cinema light reads brightly at Bevy's default
    /// "very overcast day" exposure (EV100 12), and a surface of a plausible albedo at a
    /// plausible distance is a plausible scene-linear value.
    ///
    /// Bevy documents 1 000 000 lumens as "a very large 'cinema light' capable of registering
    /// brightly at Bevy's default 'very overcast day' exposure level"
    /// (`bevy_light-0.20.0/src/spot_light.rs:152`), which is EV100 12.
    #[test]
    fn the_units_match_bevys_own_calibration() {
        let exposure = bevy::camera::Exposure { ev100: 12.0 }.exposure();
        // Bevy's own cinema light: 1e6 lumens, on a 0.5-albedo surface 5 m away.
        let intensity = 1_000_000.0_f32;
        let illuminance = light_illuminance(intensity, 5.0);
        let luminance = illuminance * 0.5 / std::f32::consts::PI;
        let value = display_value(luminance, exposure);
        assert!(
            (0.001..0.5).contains(&value),
            "a 1e6 lm cinema light reads at {value} (the luminance {luminance})"
        );
        // And the inverse gets back to the same power.
        let back = light_intensity(luminance, 5.0, 0.5);
        assert!(
            (back - intensity).abs() <= 1.0,
            "the round trip gave {back} against {intensity}"
        );
    }

    /// A light of no power reads nothing, and a black surface reflects nothing.
    #[test]
    fn the_degenerate_cases_are_zero() {
        assert_eq!(light_intensity(1.0, 0.0, 0.5), 0.0);
        assert_eq!(light_intensity(1.0, 3.0, 0.0), 0.0);
        assert_eq!(light_illuminance(1.0, 0.0), 0.0);
        assert_eq!(light_intensity(-1.0, 3.0, 0.5), 0.0);
    }

    /// The crate's promise: a light balanced against the exposure reads the same at every hour,
    /// so a game never has to re-tune its light between noon and midnight.
    ///
    /// The exposure spans the crate's own range — the noon sun's 0.001 and the moonless night's
    /// 22.4 — and the same art direction (a surface of a given albedo reading at a given
    /// scene-linear value) gives the right power at every one of them.
    #[test]
    fn a_light_reads_the_same_at_every_hour() {
        for exposure in [0.0008_f32, 0.001, 0.05, 1.34, 6.0, 22.4] {
            for (display, distance, albedo) in [
                (0.18_f32, 1.92_f32, 0.14_f32),
                (0.06, 1.92, 0.14),
                (0.5, 4.0, 0.4),
            ] {
                let luminance = luminance_for_display(display, exposure);
                let intensity = light_intensity(luminance, distance, albedo);
                // The surface's luminance, and what it reads at under the same exposure.
                let got = light_illuminance(intensity, distance) * albedo / std::f32::consts::PI;
                let read = display_value(got, exposure);
                assert!(
                    (read - display).abs() <= 1e-3 * display,
                    "at the exposure {exposure} a surface asked to read at {display} reads at {read}"
                );
            }
        }
    }

    /// A light at the crate's night reads as the crate's night.
    #[test]
    fn the_night_colour_is_the_scotopic_one() {
        let c = night_light_colour();
        assert!(c.x > 0.0 && c.y > 0.0 && c.z > 0.0);
        // The scotopic eye's blue shift: the blue reads above the green.
        assert!(c.z > c.y, "the scotopic colour is cool: {c:?}");
        assert!(c.z > c.x, "the scotopic colour is cool: {c:?}");
    }
}
