//! The climate: what the weather of a place is like on average.

use bevy::prelude::*;

use crate::celestial::CivilDate;

/// Standard-atmosphere temperature lapse rate, °C per m (ISO 2533: −6.5 K/km).
pub const LAPSE_RATE_C_PER_M: f32 = -0.0065;

/// The dates between which snow may fall: a game's calendar over the physical phase
/// model (`docs/research/snow-season.md`).
///
/// Outside the season the scheduler never picks Snow or Blizzard and precipitation falls as rain
/// whatever the temperature ([`Weather::snow_possible`](crate::Weather::snow_possible) is 0);
/// inside it nothing changes. The change of phase ramps over the hour before the first day and
/// the hour after the last, outside the season. A forced Snow or Blizzard still snows, and snow
/// already lying when the season ends melts as usual.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Reflect)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SnowSeason {
    /// The first day snow may fall, `(month 1–12, day 1–31)` of the local calendar, inclusive.
    pub first: (u8, u8),
    /// The last day snow may fall, `(month, day)`, inclusive. Before `first` in the year, the
    /// season wraps the new year (14 November to 31 March). The dates are the game's calendar in
    /// either hemisphere.
    pub last: (u8, u8),
}

impl SnowSeason {
    /// Whether `date` is in the season.
    pub fn contains(&self, date: CivilDate) -> bool {
        let d = (date.month, date.day);
        if self.first <= self.last {
            self.first <= d && d <= self.last
        } else {
            d >= self.first || d <= self.last
        }
    }

    /// How far snow is possible at local `hour` (0–24) of `date`: 1 in the season, 0 outside,
    /// ramping linearly over the hour before the first day and the hour after the last.
    pub fn possible(&self, date: CivilDate, hour: f64) -> f32 {
        if self.contains(date) {
            return 1.0;
        }
        let before_first = if self.contains(date.add_days(1)) {
            (hour - 23.0).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let after_last = if self.contains(date.add_days(-1)) {
            (1.0 - hour).clamp(0.0, 1.0)
        } else {
            0.0
        };
        before_first.max(after_last) as f32
    }
}

/// The long-run character of a place. Drives the weather scheduler and the temperature model.
///
/// Temperature (°C) at local hour `h`, day of year `d` and elevation `z` m:
/// `mean + annual·cos(2π(d − warmest)/365.25) + diurnal·cos(2π(h − 15)/24) + weather offset − 0.0065·(z − reference)`.
/// In the southern hemisphere (`GeoLocation::latitude_deg < 0`) `warmest_day` is shifted by +182.
#[derive(Resource, Clone, Debug, Reflect)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Climate {
    /// Annual mean air temperature at the reference elevation, °C.
    pub mean_temperature_c: f32,
    /// Half the warmest-to-coldest swing of daily means over a year, °C.
    pub annual_amplitude_c: f32,
    /// Half the afternoon-to-dawn swing within a day, °C. The maximum falls at 15:00.
    pub diurnal_amplitude_c: f32,
    /// Day of year (1–365) of the warmest daily mean, northern hemisphere.
    pub warmest_day: u16,
    /// Mean relative humidity, 0–1.
    pub mean_humidity: f32,
    /// How readily the scheduler moves into wet presets, 0–1. 0.35 is a temperate baseline.
    pub wet_hour_probability: f32,
    /// Approximate fraction of hours in thunderstorms, 0–1 (summer-weighted).
    pub storm_share: f32,
    /// Approximate fraction of hours in fog, 0–1.
    pub fog_share: f32,
    /// Approximate fraction of hours in dust storms, 0–1. 0 disables them.
    pub dust_share: f32,
    /// Unit XZ direction the prevailing wind blows toward.
    pub prevailing_wind: Vec2,
    /// Mean wind speed at 10 m, m/s. Preset wind speeds are scaled by this / 4.
    pub mean_wind_ms: f32,
    /// Elevation at which `mean_temperature_c` holds, m.
    pub reference_elevation_m: f32,
    /// Seed of the weather scheduler.
    pub seed: u64,
    /// The dates snow may fall between; `None` (the default) lets the temperature alone decide.
    #[cfg_attr(feature = "serde", serde(default))]
    pub snow_season: Option<SnowSeason>,
}

impl Default for Climate {
    fn default() -> Self {
        Self::temperate()
    }
}

impl Climate {
    #[allow(clippy::too_many_arguments)]
    const fn preset(
        mean: f32,
        annual: f32,
        diurnal: f32,
        humidity: f32,
        wet: f32,
        storm: f32,
        fog: f32,
        dust: f32,
        wind: f32,
    ) -> Self {
        Self {
            mean_temperature_c: mean,
            annual_amplitude_c: annual,
            diurnal_amplitude_c: diurnal,
            warmest_day: 200,
            mean_humidity: humidity,
            wet_hour_probability: wet,
            storm_share: storm,
            fog_share: fog,
            dust_share: dust,
            prevailing_wind: Vec2::X,
            mean_wind_ms: wind,
            reference_elevation_m: 0.0,
            seed: 0,
            snow_season: None,
        }
    }

    /// Mid-latitude maritime: mild, often wet, occasional storms and fog.
    pub const fn temperate() -> Self {
        Self::preset(11.0, 9.0, 5.0, 0.75, 0.35, 0.05, 0.08, 0.0, 4.0)
    }

    /// Mountain: cold, windy, wet, snowy for much of the year.
    pub const fn alpine() -> Self {
        Self::preset(2.0, 10.0, 7.0, 0.70, 0.40, 0.04, 0.06, 0.0, 6.0)
    }

    /// Hot desert: dry, large daily swing, dust storms.
    pub const fn desert() -> Self {
        Self::preset(24.0, 9.0, 14.0, 0.25, 0.03, 0.01, 0.0, 0.08, 5.0)
    }

    /// Humid tropics: hot all year, frequent rain and thunderstorms.
    pub const fn tropical() -> Self {
        Self::preset(27.0, 2.0, 6.0, 0.85, 0.55, 0.20, 0.03, 0.0, 3.0)
    }

    /// Arctic tundra: long cold winters, windy, snow.
    pub const fn tundra() -> Self {
        Self::preset(-8.0, 18.0, 4.0, 0.80, 0.25, 0.0, 0.05, 0.0, 7.0)
    }

    /// The same climate with another scheduler seed.
    pub const fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// The same climate with snow only between the dates of `season`.
    pub const fn with_snow_season(mut self, season: SnowSeason) -> Self {
        self.snow_season = Some(season);
        self
    }

    /// How far snow is possible at local `hour` of `date`: 1 without a season.
    pub fn snow_possible(&self, date: CivilDate, hour: f64) -> f32 {
        self.snow_season.map_or(1.0, |s| s.possible(date, hour))
    }

    /// Day of year of the warmest daily mean, hemisphere-aware.
    pub fn warmest_day_for(&self, southern: bool) -> f32 {
        f32::from(self.warmest_day) + if southern { 182.0 } else { 0.0 }
    }

    /// Climatological temperature with no weather offset at the reference elevation, °C, for a
    /// local hour (0–24) and a day of year (1-based, may be fractional).
    pub fn base_temperature_c(&self, day_of_year: f32, hour: f32, southern: bool) -> f32 {
        use std::f32::consts::TAU;
        let warmest = self.warmest_day_for(southern);
        self.mean_temperature_c
            + self.annual_amplitude_c * (TAU * (day_of_year - warmest) / 365.25).cos()
            + self.diurnal_amplitude_c * (TAU * (hour - 15.0) / 24.0).cos()
    }

    /// Temperature correction for standing `elevation_m` instead of at the reference elevation,
    /// °C (negative above it).
    pub fn lapse_c(&self, elevation_m: f32) -> f32 {
        LAPSE_RATE_C_PER_M * (elevation_m - self.reference_elevation_m)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn temperature_harmonics_hit_the_research_targets() {
        let c = Climate::temperate();
        assert!((c.base_temperature_c(200.0, 15.0, false) - 25.0).abs() < 1e-4);
        assert!((c.base_temperature_c(200.0, 3.0, false) - 15.0).abs() < 1e-4);
        assert!((c.base_temperature_c(17.0, 3.0, false) - -3.0).abs() < 1e-3);
        assert!((c.lapse_c(1000.0) - -6.5).abs() < 1e-5);
    }

    #[test]
    fn the_southern_hemisphere_has_its_summer_in_january() {
        let c = Climate::temperate();
        assert!(c.base_temperature_c(17.0, 15.0, true) > c.base_temperature_c(200.0, 15.0, true));
    }

    #[test]
    fn a_snow_season_wraps_the_year_and_ramps_outside_it() {
        let d = |m, day| CivilDate::new(2026, m, day);
        let winter = SnowSeason {
            first: (11, 14),
            last: (3, 31),
        };
        for (date, inside) in [
            (d(11, 13), false),
            (d(11, 14), true),
            (d(12, 31), true),
            (d(1, 1), true),
            (d(3, 31), true),
            (d(4, 1), false),
            (d(7, 1), false),
        ] {
            assert_eq!(winter.contains(date), inside, "{date:?}");
        }
        // The hour after the last day ramps down, the hour before the first ramps up.
        assert_eq!(winter.possible(d(3, 31), 23.9), 1.0);
        assert!((winter.possible(d(4, 1), 0.25) - 0.75).abs() < 1e-6);
        assert_eq!(winter.possible(d(4, 1), 1.0), 0.0);
        assert!((winter.possible(d(11, 13), 23.5) - 0.5).abs() < 1e-6);
        assert_eq!(winter.possible(d(11, 13), 22.0), 0.0);

        let summer = SnowSeason {
            first: (6, 1),
            last: (8, 31),
        };
        assert!(summer.contains(d(7, 1)) && !summer.contains(d(12, 1)));
        assert_eq!(Climate::temperate().snow_possible(d(7, 1), 12.0), 1.0);
    }
}
