//! Calendar, world clock and observer location.

use bevy::prelude::*;

use crate::plugin::ElementsSystems;

/// A proleptic Gregorian calendar date.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Reflect)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CivilDate {
    /// Astronomical year (1 BC is 0).
    pub year: i32,
    /// Month, 1–12.
    pub month: u8,
    /// Day of month, 1–31.
    pub day: u8,
}

impl CivilDate {
    /// A date. Out-of-range months and days are clamped into range.
    pub const fn new(year: i32, month: u8, day: u8) -> Self {
        let month = if month < 1 {
            1
        } else if month > 12 {
            12
        } else {
            month
        };
        let max = days_in_month(year, month);
        let day = if day < 1 {
            1
        } else if day > max {
            max
        } else {
            day
        };
        Self { year, month, day }
    }

    /// Whether the year has 29 February.
    pub const fn is_leap_year(self) -> bool {
        is_leap(self.year)
    }

    /// Day of the year, 1 for 1 January.
    pub fn day_of_year(self) -> u16 {
        (self.days_since_2000() - Self::new(self.year, 1, 1).days_since_2000() + 1) as u16
    }

    /// Whole days from 2000-01-01 (negative before it).
    pub fn days_since_2000(self) -> i64 {
        // Hinnant, "chrono-Compatible Low-Level Date Algorithms", days_from_civil.
        let y = i64::from(self.year) - i64::from(self.month <= 2);
        let era = y.div_euclid(400);
        let yoe = y - era * 400;
        let m = i64::from(self.month);
        let mp = (m + 9) % 12;
        let doy = (153 * mp + 2) / 5 + i64::from(self.day) - 1;
        let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
        era * 146_097 + doe - 719_468 - 10_957
    }

    /// The date `days` whole days later (earlier when negative).
    pub fn add_days(self, days: i64) -> Self {
        // Hinnant, civil_from_days.
        let z = self.days_since_2000() + days + 10_957 + 719_468;
        let era = z.div_euclid(146_097);
        let doe = z - era * 146_097;
        let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
        let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
        let mp = (5 * doy + 2) / 153;
        let day = (doy - (153 * mp + 2) / 5 + 1) as u8;
        let month = if mp < 10 { mp + 3 } else { mp - 9 } as u8;
        let year = (yoe + era * 400 + i64::from(month <= 2)) as i32;
        Self { year, month, day }
    }
}

const fn is_leap(year: i32) -> bool {
    (year % 4 == 0 && year % 100 != 0) || year % 400 == 0
}

const fn days_in_month(year: i32, month: u8) -> u8 {
    match month {
        2 if is_leap(year) => 29,
        2 => 28,
        4 | 6 | 9 | 11 => 30,
        _ => 31,
    }
}

/// The game's calendar and time of day. Advances in [`ElementsSystems::Clock`] by
/// `Time<Virtual>` delta × `time_scale`.
#[derive(Resource, Clone, Debug, Reflect)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct WorldClock {
    /// Local calendar date.
    pub date: CivilDate,
    /// Local time of day, hours in [0, 24).
    pub hour: f64,
    /// Game seconds per real second. Default 60 (one game hour per real minute).
    pub time_scale: f32,
    /// When set the clock does not advance.
    pub paused: bool,
}

impl Default for WorldClock {
    fn default() -> Self {
        Self::new(CivilDate::new(2026, 6, 21))
    }
}

impl WorldClock {
    /// Noon on `date`, time scale 60, running.
    pub fn new(date: CivilDate) -> Self {
        Self {
            date,
            hour: 12.0,
            time_scale: 60.0,
            paused: false,
        }
    }

    /// Set the local time of day, hours. Values outside [0, 24) roll the date.
    pub fn at_hour(mut self, hour: f64) -> Self {
        self.hour = 0.0;
        self.advance_hours(hour);
        self
    }

    /// Set the time scale, game seconds per real second.
    pub fn with_time_scale(mut self, time_scale: f32) -> Self {
        self.time_scale = time_scale;
        self
    }

    /// Day of the year of the local date, 1 for 1 January.
    pub fn day_of_year(&self) -> u16 {
        self.date.day_of_year()
    }

    /// Julian Day of this instant (Meeus ch. 7), given the local offset from UTC in hours.
    pub fn julian_day(&self, utc_offset_hours: f64) -> f64 {
        2_451_544.5 + self.date.days_since_2000() as f64 + (self.hour - utc_offset_hours) / 24.0
    }

    /// Local hours since 2000-01-01 00:00 local time. The weather scheduler's hour index is the
    /// floor of this, so a seed and a date always draw the same hour.
    pub fn hours_since_2000(&self) -> f64 {
        self.date.days_since_2000() as f64 * 24.0 + self.hour
    }

    /// Move the clock by `hours` (negative moves back), rolling the date.
    pub fn advance_hours(&mut self, hours: f64) {
        let total = self.hour + hours;
        let days = (total / 24.0).floor();
        self.hour = total - days * 24.0;
        if self.hour >= 24.0 {
            self.hour = 0.0;
        }
        if days != 0.0 {
            self.date = self.date.add_days(days as i64);
        }
    }
}

/// Where the observer stands on Earth.
#[derive(Resource, Clone, Debug, Reflect)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GeoLocation {
    /// Geographic latitude, degrees north (negative south). Default 45.
    pub latitude_deg: f64,
    /// Geographic longitude, degrees east (negative west). Default 0.
    pub longitude_deg: f64,
    /// Local civil time minus UTC, hours. Default 0.
    pub utc_offset_hours: f64,
}

impl Default for GeoLocation {
    fn default() -> Self {
        Self {
            latitude_deg: 45.0,
            longitude_deg: 0.0,
            utc_offset_hours: 0.0,
        }
    }
}

impl GeoLocation {
    /// Whether the observer is in the southern hemisphere, where the seasons are shifted by half
    /// a year.
    pub fn is_southern(&self) -> bool {
        self.latitude_deg < 0.0
    }
}

pub(crate) fn advance_clock(time: Res<Time<Virtual>>, mut clock: ResMut<WorldClock>) {
    if clock.paused {
        return;
    }
    let hours = time.delta_secs_f64() * f64::from(clock.time_scale) / 3600.0;
    if hours != 0.0 {
        clock.advance_hours(hours);
    }
}

pub(crate) fn plugin(app: &mut App) {
    app.init_resource::<WorldClock>()
        .init_resource::<GeoLocation>()
        .add_systems(Update, advance_clock.in_set(ElementsSystems::Clock));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn julian_day_matches_meeus_example_7a() {
        // Meeus ex. 7.a: 1957 October 4.81 UT is JD 2436116.31.
        let clock = WorldClock::new(CivilDate::new(1957, 10, 4)).at_hour(0.81 * 24.0);
        assert!((clock.julian_day(0.0) - 2_436_116.31).abs() < 1e-6);
    }

    #[test]
    fn dates_round_trip_across_leap_years_and_centuries() {
        let start = CivilDate::new(1899, 12, 31);
        for n in 0..200_000 {
            let d = start.add_days(n);
            assert_eq!(d.days_since_2000() - start.days_since_2000(), n);
            assert_eq!(CivilDate::new(d.year, d.month, d.day), d);
        }
        assert_eq!(CivilDate::new(2024, 12, 31).day_of_year(), 366);
        assert_eq!(CivilDate::new(2026, 3, 1).day_of_year(), 60);
    }

    #[test]
    fn advancing_rolls_the_date_both_ways() {
        let mut clock = WorldClock::new(CivilDate::new(2026, 12, 31)).at_hour(23.5);
        clock.advance_hours(1.0);
        assert_eq!(clock.date, CivilDate::new(2027, 1, 1));
        assert!((clock.hour - 0.5).abs() < 1e-9);
        clock.advance_hours(-1.0);
        assert_eq!(clock.date, CivilDate::new(2026, 12, 31));
        assert!((clock.hour - 23.5).abs() < 1e-9);
    }
}
