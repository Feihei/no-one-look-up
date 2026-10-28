//! Ground wetness and snow cover at one point: exposed, flat ground at the reference elevation.
//!
//! The equations are the ground-cover model (snow accumulation, settling and degree-day melt;
//! rain wetting and evaporative drying) evaluated without spatial maps. Everything runs on game
//! time, so snow survives a time-lapse.

use bevy::prelude::*;

use super::weather::Weather;
use crate::celestial::{SunState, WorldClock};

/// Density of freshly fallen snow, kg/m³.
const NEW_SNOW_DENSITY: f32 = 100.0;
/// Density settled snow approaches, kg/m³.
const SETTLED_DENSITY: f32 = 300.0;
/// Settling time constant, days.
const SETTLE_DAYS: f32 = 3.0;
/// Degree-day melt factor, mm water equivalent per °C per day.
const DEGREE_DAY_FACTOR: f32 = 4.0;
/// Wetness gained per mm of rain.
const WET_PER_MM: f32 = 0.5;
/// Base drying rate, wetness per hour.
const DRY_RATE_H: f32 = 0.05;
/// Snow depth at which the ground reads as fully covered, m.
const FULL_COVER_DEPTH_M: f32 = 0.05;
/// The longest clock jump integrated in one step, game hours; larger jumps are clamped.
const MAX_STEP_H: f64 = 24.0;

/// How wet and snowy exposed ground is around the camera. Feeds foliage and any material that
/// wants to look rained or snowed on.
#[derive(Resource, Clone, Copy, Debug, Default, Reflect)]
pub struct SurfaceWeathering {
    /// Surface wetness, 0 dry – 1 soaked.
    pub wetness: f32,
    /// Fraction of the ground under snow, 0–1 (smoothstep of depth over 0–5 cm).
    pub snow_cover: f32,
    /// Snow depth, m.
    pub snow_depth_m: f32,
    /// Snow water equivalent, kg/m² (= mm of water).
    snow_water_mm: f32,
    /// Bulk density of the snowpack, kg/m³.
    snow_density: f32,
    last_hours: Option<f64>,
}

impl SurfaceWeathering {
    /// Snow water equivalent of the point model, mm.
    #[cfg(test)]
    pub(crate) fn snow_water_mm(&self) -> f32 {
        self.snow_water_mm
    }

    /// Advance by `hours` of game time under `weather`, with the sun `sun_factor` (0–1) above
    /// the horizon.
    pub(crate) fn step(&mut self, weather: &Weather, sun_factor: f32, hours: f32) {
        if hours <= 0.0 {
            return;
        }
        let snow_fraction = weather.snow_fraction();
        let t = weather.temperature_c;

        // Snow: accumulate, settle, melt.
        let new_swe = weather.precipitation_mm_h * snow_fraction * hours;
        let old_depth = self.snow_depth_m;
        self.snow_water_mm += new_swe;
        if self.snow_water_mm > 0.0 {
            let added_depth = new_swe / NEW_SNOW_DENSITY;
            let density = self.snow_water_mm / (old_depth + added_depth).max(1e-6);
            let k = 1.0 - (-hours / (SETTLE_DAYS * 24.0)).exp();
            self.snow_density = density + (SETTLED_DENSITY - density) * k;
            let melt = DEGREE_DAY_FACTOR * t.max(0.0) * hours / 24.0;
            self.snow_water_mm = (self.snow_water_mm - melt).max(0.0);
        }
        self.snow_depth_m = if self.snow_water_mm > 0.0 {
            self.snow_water_mm / self.snow_density.max(NEW_SNOW_DENSITY)
        } else {
            self.snow_density = NEW_SNOW_DENSITY;
            0.0
        };
        let c = (self.snow_depth_m / FULL_COVER_DEPTH_M).clamp(0.0, 1.0);
        self.snow_cover = c * c * (3.0 - 2.0 * c);

        // Wetness: rain wets, evaporation dries.
        let rain = weather.precipitation_mm_h * (1.0 - snow_fraction);
        let drying = DRY_RATE_H
            * (1.0 + weather.wind_speed_ms / 5.0)
            * (1.0 + t.max(0.0) / 20.0)
            * (1.0 - weather.relative_humidity).max(0.0)
            * (1.0 + sun_factor);
        self.wetness = (self.wetness + (rain * WET_PER_MM - drying) * hours).clamp(0.0, 1.0);
    }
}

pub(crate) fn update_weathering(
    clock: Res<WorldClock>,
    weather: Res<Weather>,
    sun: Res<SunState>,
    mut ground: ResMut<SurfaceWeathering>,
) {
    let now = clock.hours_since_2000();
    let hours = match ground.last_hours {
        Some(last) => (now - last).clamp(0.0, MAX_STEP_H) as f32,
        None => 0.0,
    };
    ground.last_hours = Some(now);
    let sun_factor = sun.altitude_deg.to_radians().sin().max(0.0);
    ground.step(&weather, sun_factor, hours);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn weather(mm_h: f32, t: f32) -> Weather {
        Weather {
            precipitation_mm_h: mm_h,
            temperature_c: t,
            relative_humidity: 0.5,
            wind_speed_ms: 0.0,
            ..Weather::default()
        }
    }

    #[test]
    fn ten_hours_of_snow_at_1_5_mm_h_lays_about_15_cm() {
        let mut g = SurfaceWeathering::default();
        for _ in 0..600 {
            g.step(&weather(1.5, -5.0), 0.0, 1.0 / 60.0);
        }
        assert!((g.snow_water_mm - 15.0).abs() < 1e-3);
        // Settling over 10 h already compacts it a little; undisturbed it would be 0.15 m.
        assert!(
            (0.12..=0.15).contains(&g.snow_depth_m),
            "{}",
            g.snow_depth_m
        );
        assert!(g.snow_cover > 0.99);
    }

    #[test]
    fn warm_clear_days_melt_the_pack_away() {
        let mut g = SurfaceWeathering::default();
        g.step(&weather(1.5, -5.0), 0.0, 10.0);
        // 15 mm at 4 mm/°C/day and 10 °C melts in 0.375 days.
        for _ in 0..48 {
            g.step(&weather(0.0, 10.0), 1.0, 0.25);
        }
        assert_eq!(g.snow_depth_m, 0.0);
        assert_eq!(g.snow_cover, 0.0);
    }

    #[test]
    fn an_hour_of_rain_soaks_and_a_warm_dry_afternoon_dries() {
        let mut g = SurfaceWeathering::default();
        g.step(&weather(4.0, 15.0), 0.0, 1.0);
        assert_eq!(g.wetness, 1.0);
        // 0.05/h × (1 + 0) × (1 + 20/20) × (1 − 0.5) × (1 + 1) = 0.1/h: 1.0 → 0.1 in 9 h.
        g.step(&weather(0.0, 20.0), 1.0, 9.0);
        assert!((g.wetness - 0.1).abs() < 1e-4, "{}", g.wetness);
    }
}
