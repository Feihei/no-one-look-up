//! A record of recent days: the daily temperatures, humidity and precipitation the weather
//! reported, so a game can ask what the last few days were like
//! (`docs/research/climate-history.md`).
//!
//! A day is the record climatology's daily indices are computed from (ETCCDI): the mean, minimum
//! and maximum temperature and the precipitation total, with the mean relative humidity and the
//! snowfall. The weather of a past hour cannot be recomputed (the scheduler's chain depends on
//! the hour it started), so the days are recorded as the clock runs, and the record is saved
//! with the weather.

use bevy::prelude::*;

use super::weather::Weather;
use crate::celestial::{CivilDate, WorldClock};

/// The least precipitation of a wet day, mm of liquid-water equivalent: the ETCCDI's wet day,
/// RR ≥ 1 mm.
pub const WET_DAY_MM: f32 = 1.0;

/// The temperature a frost day's minimum (ETCCDI FD) and an icing day's maximum (ETCCDI ID)
/// fall below, °C.
pub const FREEZING_C: f32 = 0.0;

/// Completed days kept by default: a month.
const DEFAULT_CAPACITY_DAYS: u32 = 31;
/// The longest clock step integrated, game hours, as the point snowpack's: a longer jump leaves
/// the days it skips unrecorded.
const MAX_STEP_H: f64 = 24.0;
/// The furthest a loaded clock may be from the saved one for the record to continue across the
/// gap, game hours.
const RESTORE_TOLERANCE_H: f64 = 1.0;

/// One local calendar day of weather at the climate's reference elevation, as recorded while the
/// clock ran through it.
#[derive(Clone, Copy, Debug, PartialEq, Reflect)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DayConditions {
    /// The local date.
    pub date: CivilDate,
    /// Game hours of the day that were recorded, 0–24: less on the day recording started and on
    /// the days either side of a clock jump longer than a day.
    pub observed_hours: f32,
    /// Mean air temperature over the observed hours (TG), °C.
    pub mean_temperature_c: f32,
    /// Lowest air temperature (TN), °C.
    pub min_temperature_c: f32,
    /// Highest air temperature (TX), °C.
    pub max_temperature_c: f32,
    /// Mean relative humidity over the observed hours, 0–1.
    pub mean_relative_humidity: f32,
    /// Precipitation total (RR), mm of liquid-water equivalent, snow included.
    pub precipitation_mm: f32,
    /// The part of the precipitation that fell as snow, mm of water equivalent.
    pub snowfall_mm: f32,
}

impl DayConditions {
    /// A wet day: at least [`WET_DAY_MM`] of precipitation.
    pub fn is_wet(&self) -> bool {
        self.precipitation_mm >= WET_DAY_MM
    }

    /// A frost day: the minimum temperature below [`FREEZING_C`].
    pub fn is_frost(&self) -> bool {
        self.min_temperature_c < FREEZING_C
    }

    /// An icing day: the maximum temperature below [`FREEZING_C`] (it froze all day).
    pub fn is_icing(&self) -> bool {
        self.max_temperature_c < FREEZING_C
    }
}

/// The days of a window summed up: the counts of the daily indices, the means weighted by the
/// observed hours, the extremes and the totals.
#[derive(Clone, Copy, Debug, PartialEq, Reflect)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ClimateSummary {
    /// Recorded days in the window.
    pub days: u32,
    /// Game hours recorded over them.
    pub observed_hours: f32,
    /// Days with at least [`WET_DAY_MM`] of precipitation.
    pub wet_days: u32,
    /// Days whose minimum temperature fell below [`FREEZING_C`].
    pub frost_days: u32,
    /// Days whose maximum temperature stayed below [`FREEZING_C`].
    pub icing_days: u32,
    /// Mean air temperature over the observed hours, °C.
    pub mean_temperature_c: f32,
    /// Lowest air temperature, °C.
    pub min_temperature_c: f32,
    /// Highest air temperature, °C.
    pub max_temperature_c: f32,
    /// Mean relative humidity over the observed hours, 0–1.
    pub mean_relative_humidity: f32,
    /// Precipitation total, mm of liquid-water equivalent.
    pub precipitation_mm: f32,
    /// Snowfall total, mm of water equivalent.
    pub snowfall_mm: f32,
}

/// The day being recorded, its integrals over game hours in f64 (a day at time scale 60 and 60
/// frames a second is 86 400 frames).
#[derive(Clone, Copy, Debug, PartialEq, Reflect)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
struct OpenDay {
    date: CivilDate,
    hours: f64,
    temperature: f64,
    humidity: f64,
    precipitation_mm: f64,
    snowfall_mm: f64,
    min_c: f32,
    max_c: f32,
}

impl OpenDay {
    fn new(date: CivilDate) -> Self {
        Self {
            date,
            hours: 0.0,
            temperature: 0.0,
            humidity: 0.0,
            precipitation_mm: 0.0,
            snowfall_mm: 0.0,
            min_c: 0.0,
            max_c: 0.0,
        }
    }

    /// `hours` of `weather`.
    fn add(&mut self, weather: &Weather, hours: f64) {
        if hours <= 0.0 {
            return;
        }
        let t = weather.temperature_c;
        if self.hours == 0.0 {
            self.min_c = t;
            self.max_c = t;
        }
        self.min_c = self.min_c.min(t);
        self.max_c = self.max_c.max(t);
        self.hours += hours;
        self.temperature += f64::from(t) * hours;
        self.humidity += f64::from(weather.relative_humidity) * hours;
        let mm = f64::from(weather.precipitation_mm_h) * hours;
        self.precipitation_mm += mm;
        self.snowfall_mm += mm * f64::from(weather.snow_fraction());
    }

    /// The day so far; `None` before any of it was observed.
    fn conditions(&self) -> Option<DayConditions> {
        (self.hours > 0.0).then(|| DayConditions {
            date: self.date,
            observed_hours: self.hours as f32,
            mean_temperature_c: (self.temperature / self.hours) as f32,
            min_temperature_c: self.min_c,
            max_temperature_c: self.max_c,
            mean_relative_humidity: (self.humidity / self.hours) as f32,
            precipitation_mm: self.precipitation_mm as f32,
            snowfall_mm: self.snowfall_mm as f32,
        })
    }
}

/// The weather of the last days, recorded from [`Weather`] as the [`WorldClock`] runs: one
/// [`DayConditions`] per local calendar day, so a game can ask what the last few days were like
/// ([`recent`](Self::recent), [`summary`](Self::summary), or the same through
/// [`Environment`](crate::Environment)).
///
/// Each frame adds the game time since the last one with that frame's weather, split at
/// midnight. A step of up to a day is recorded (as the point snowpack integrates it); a longer
/// jump leaves the days it skips unrecorded, and a clock moved back forgets the days from its
/// new date on. The days are never stale: the record reads [`Weather`], not the GPU readback.
///
/// It keeps the last 31 completed days; insert [`ClimateHistory::with_capacity`] before
/// [`ClimatePlugin`](crate::ClimatePlugin) to keep more. With the `serde` feature it is part of
/// a save, with the weather's ([`WeatherDirector`](crate::WeatherDirector), [`WorldClock`],
/// [`Climate`](crate::Climate), [`GeoLocation`](crate::GeoLocation)): a load onto a clock within
/// an hour of the saved one continues the same days; onto another clock it starts recording
/// there. A save without it loads with an empty record.
#[derive(Resource, Clone, Debug, PartialEq, Reflect)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ClimateHistory {
    capacity: u32,
    days: Vec<DayConditions>,
    today: Option<OpenDay>,
    last_hours: Option<f64>,
    /// Read back from a save and not yet stepped: never written, true when deserialized.
    #[cfg_attr(feature = "serde", serde(skip_serializing, default = "restored"))]
    restored: bool,
}

#[cfg(feature = "serde")]
fn restored() -> bool {
    true
}

impl Default for ClimateHistory {
    fn default() -> Self {
        Self::with_capacity(DEFAULT_CAPACITY_DAYS)
    }
}

impl ClimateHistory {
    /// An empty record that keeps the last `days` completed days (at least one).
    pub fn with_capacity(days: u32) -> Self {
        Self {
            capacity: days.max(1),
            days: Vec::new(),
            today: None,
            last_hours: None,
            restored: false,
        }
    }

    /// How many completed days the record keeps.
    pub fn capacity(&self) -> u32 {
        self.capacity
    }

    /// Every completed day kept, oldest first. Days a clock jump skipped are absent.
    pub fn days(&self) -> &[DayConditions] {
        &self.days
    }

    /// The day in progress so far; `None` before the first frame has advanced the clock.
    pub fn today(&self) -> Option<DayConditions> {
        self.today.and_then(|day| day.conditions())
    }

    /// The completed days of the last `window_days` calendar days, from today − `window_days`
    /// to yesterday, oldest first. Today is not included ([`today`](Self::today)). A window of
    /// 0 is empty; a window longer than the record is the record.
    pub fn recent(&self, window_days: u32) -> &[DayConditions] {
        let Some(today) = self.today else {
            return &[];
        };
        let first = today.date.add_days(-i64::from(window_days));
        &self.days[self.days.partition_point(|day| day.date < first)..]
    }

    /// The [`recent`](Self::recent) days summed up; `None` when the window holds no recorded
    /// day.
    pub fn summary(&self, window_days: u32) -> Option<ClimateSummary> {
        summarise(self.recent(window_days))
    }

    /// Record the game time from the last step to `now` (hours since 2000-01-01 00:00 local)
    /// with `weather`. The day's date is derived from `now` itself, not from the clock's date:
    /// at 2.3×10⁵ hours an hour of 23.999999999999996 sums to the next midnight in f64, and a
    /// date taken from the clock would then label the new day with the old one's date.
    pub(crate) fn step(&mut self, now: f64, weather: &Weather) {
        let date = date_of(now);
        let restored = std::mem::take(&mut self.restored);
        let Some(last) = self.last_hours.replace(now) else {
            self.today = Some(OpenDay::new(date));
            return;
        };
        let hours = now - last;
        if hours < 0.0 {
            // The clock moved back: the days from its new date on are the future now.
            self.days
                .truncate(self.days.partition_point(|day| day.date < date));
            self.today = Some(OpenDay::new(date));
        } else if hours > MAX_STEP_H || (restored && hours > RESTORE_TOLERANCE_H) {
            self.close();
            self.today = Some(OpenDay::new(date));
        } else {
            let midnight = (last / 24.0).floor() * 24.0 + 24.0;
            if now >= midnight {
                if let Some(day) = &mut self.today {
                    day.add(weather, midnight - last);
                }
                self.close();
                let mut day = OpenDay::new(date);
                day.add(weather, now - midnight);
                self.today = Some(day);
            } else if let Some(day) = &mut self.today {
                day.add(weather, hours);
            }
        }
    }

    /// Complete the day in progress, if any of it was observed, and drop the oldest days past
    /// the capacity.
    fn close(&mut self) {
        if let Some(day) = self.today.take().and_then(|day| day.conditions()) {
            self.days.push(day);
            let excess = self.days.len().saturating_sub(self.capacity as usize);
            self.days.drain(..excess);
        }
    }
}

fn summarise(days: &[DayConditions]) -> Option<ClimateSummary> {
    let first = days.first()?;
    let mut summary = ClimateSummary {
        days: 0,
        observed_hours: 0.0,
        wet_days: 0,
        frost_days: 0,
        icing_days: 0,
        mean_temperature_c: 0.0,
        min_temperature_c: first.min_temperature_c,
        max_temperature_c: first.max_temperature_c,
        mean_relative_humidity: 0.0,
        precipitation_mm: 0.0,
        snowfall_mm: 0.0,
    };
    let (mut hours, mut temperature, mut humidity) = (0.0_f64, 0.0_f64, 0.0_f64);
    let (mut precipitation, mut snowfall) = (0.0_f64, 0.0_f64);
    for day in days {
        let h = f64::from(day.observed_hours);
        summary.days += 1;
        summary.wet_days += u32::from(day.is_wet());
        summary.frost_days += u32::from(day.is_frost());
        summary.icing_days += u32::from(day.is_icing());
        summary.min_temperature_c = summary.min_temperature_c.min(day.min_temperature_c);
        summary.max_temperature_c = summary.max_temperature_c.max(day.max_temperature_c);
        hours += h;
        temperature += f64::from(day.mean_temperature_c) * h;
        humidity += f64::from(day.mean_relative_humidity) * h;
        precipitation += f64::from(day.precipitation_mm);
        snowfall += f64::from(day.snowfall_mm);
    }
    summary.observed_hours = hours as f32;
    summary.mean_temperature_c = (temperature / hours) as f32;
    summary.mean_relative_humidity = (humidity / hours) as f32;
    summary.precipitation_mm = precipitation as f32;
    summary.snowfall_mm = snowfall as f32;
    Some(summary)
}

pub(crate) fn record_history(
    clock: Res<WorldClock>,
    weather: Res<Weather>,
    mut history: ResMut<ClimateHistory>,
) {
    history.step(clock.hours_since_2000(), &weather);
}

/// The local date `hours` (since 2000-01-01 00:00) falls on.
fn date_of(hours: f64) -> CivilDate {
    CivilDate::new(2000, 1, 1).add_days((hours / 24.0).floor() as i64)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn weather(temperature_c: f32, rh: f32, mm_h: f32) -> Weather {
        Weather {
            temperature_c,
            relative_humidity: rh,
            precipitation_mm_h: mm_h,
            ..Weather::default()
        }
    }

    const MARCH_1: CivilDate = CivilDate::new(2026, 3, 1);

    /// Hours since 2000 at `hour` of `days` after 1 March 2026.
    fn at(days: i64, hour: f64) -> f64 {
        MARCH_1.add_days(days).days_since_2000() as f64 * 24.0 + hour
    }

    /// Record `days` whole days from midnight, a frame a game minute, day `d`'s weather from
    /// `w(d)`.
    fn record(history: &mut ClimateHistory, days: i64, w: impl Fn(i64) -> Weather) {
        history.step(at(0, 0.0), &w(0));
        for minute in 1..=days * 1440 {
            let day = (minute - 1) / 1440;
            let now = at(minute / 1440, (minute % 1440) as f64 / 60.0);
            history.step(now, &w(day));
        }
    }

    #[test]
    fn a_day_of_steady_weather_is_its_own_mean_and_its_total_is_the_rate_for_a_day() {
        let mut history = ClimateHistory::default();
        record(&mut history, 1, |_| weather(5.0, 0.8, 0.5));
        let [day] = history.days() else {
            panic!("one completed day: {:?}", history.days());
        };
        assert_eq!(day.date, MARCH_1);
        assert!((day.observed_hours - 24.0).abs() < 1e-4);
        assert!((day.mean_temperature_c - 5.0).abs() < 1e-5);
        assert_eq!((day.min_temperature_c, day.max_temperature_c), (5.0, 5.0));
        assert!((day.mean_relative_humidity - 0.8).abs() < 1e-5);
        assert!((day.precipitation_mm - 12.0).abs() < 1e-3);
        assert_eq!(day.snowfall_mm, 0.0, "5 °C rain is not snow");
        assert_eq!(
            history.today(),
            None,
            "midnight: nothing of 2 March observed yet"
        );
        assert_eq!(history.recent(1), history.days(), "1 March is yesterday");
    }

    #[test]
    fn a_frame_across_midnight_is_split_between_the_days() {
        let mut history = ClimateHistory::default();
        history.step(at(0, 23.5), &weather(0.0, 0.5, 0.0));
        history.step(at(1, 0.25), &weather(-4.0, 0.9, 2.0));
        let [day] = history.days() else {
            panic!("one completed day");
        };
        assert!((day.observed_hours - 0.5).abs() < 1e-6);
        assert!((day.precipitation_mm - 1.0).abs() < 1e-6);
        assert!((day.snowfall_mm - 1.0).abs() < 1e-6, "−4 °C falls as snow");
        let today = history.today().expect("today");
        assert!((today.observed_hours - 0.25).abs() < 1e-6);
        assert!((today.precipitation_mm - 0.5).abs() < 1e-6);
    }

    #[test]
    fn a_jump_over_days_leaves_them_unrecorded_and_a_rewind_forgets_the_future() {
        let mut history = ClimateHistory::default();
        record(&mut history, 3, |d| weather(d as f32, 0.5, 0.0));
        assert_eq!(history.days().len(), 3);
        // Three days and a half later: the open day (4 March, 0 h observed) is not a day.
        history.step(at(6, 12.0), &weather(20.0, 0.5, 0.0));
        let dates: Vec<_> = history.days().iter().map(|d| d.date).collect();
        assert_eq!(dates, [0, 1, 2].map(|d| MARCH_1.add_days(d)));
        history.step(at(7, 12.0), &weather(20.0, 0.5, 0.0));
        assert_eq!(
            history.days().last().map(|d| d.date),
            Some(MARCH_1.add_days(6))
        );
        assert!((history.days()[3].observed_hours - 12.0).abs() < 1e-6);
        // Back to 2 March: 2 March onwards is the future.
        history.step(at(1, 8.0), &weather(20.0, 0.5, 0.0));
        assert_eq!(
            history.days().iter().map(|d| d.date).collect::<Vec<_>>(),
            [MARCH_1]
        );
        assert_eq!(history.today().map(|d| d.observed_hours), None);
    }

    #[test]
    fn the_window_counts_back_from_today_and_clamps_to_the_record() {
        let mut history = ClimateHistory::default();
        assert!(history.recent(7).is_empty() && history.summary(7).is_none());
        record(&mut history, 40, |d| weather(d as f32, 0.5, 0.0));
        assert_eq!(history.days().len(), 31, "the capacity");
        assert_eq!(history.days()[0].date, MARCH_1.add_days(9));
        assert!(history.recent(0).is_empty());
        assert_eq!(history.summary(0), None);
        let week = history.recent(7);
        assert_eq!(week.len(), 7);
        assert_eq!(week[0].date, MARCH_1.add_days(33));
        assert_eq!(week[6].date, MARCH_1.add_days(39), "yesterday");
        assert_eq!(history.recent(10_000).len(), 31);
        let mut longer = ClimateHistory::with_capacity(90);
        record(&mut longer, 40, |d| weather(d as f32, 0.5, 0.0));
        assert_eq!(longer.recent(10_000).len(), 40);
    }

    #[test]
    fn wet_frost_and_icing_days_are_the_etccdi_thresholds() {
        let day = |min, max, mm| DayConditions {
            date: MARCH_1,
            observed_hours: 24.0,
            mean_temperature_c: 0.5 * (min + max),
            min_temperature_c: min,
            max_temperature_c: max,
            mean_relative_humidity: 0.7,
            precipitation_mm: mm,
            snowfall_mm: 0.0,
        };
        assert!(day(2.0, 9.0, 1.0).is_wet());
        assert!(!day(2.0, 9.0, 0.99).is_wet());
        assert!(day(-0.01, 9.0, 0.0).is_frost());
        assert!(!day(0.0, 9.0, 0.0).is_frost());
        assert!(day(-8.0, -0.01, 0.0).is_icing());
        assert!(!day(-8.0, 0.0, 0.0).is_icing());
    }

    #[test]
    fn the_summary_weights_by_the_observed_hours_and_counts_the_indices() {
        let mut history = ClimateHistory::default();
        // Half of 1 March at 10 °C, rain; 2 March whole at 0 °C under a −3 °C night, dry.
        history.step(at(0, 12.0), &weather(10.0, 0.9, 0.25));
        history.step(at(1, 0.0), &weather(10.0, 0.9, 0.25));
        for hour in 1..=24 {
            let t = if hour <= 6 { -3.0 } else { 0.0 };
            history.step(at(hour / 24 + 1, (hour % 24) as f64), &weather(t, 0.6, 0.0));
        }
        let s = history.summary(7).expect("two days");
        assert_eq!(
            (s.days, s.wet_days, s.frost_days, s.icing_days),
            (2, 1, 1, 0)
        );
        assert!((s.observed_hours - 36.0).abs() < 1e-5);
        let mean = (12.0 * 10.0 + 6.0 * -3.0) / 36.0;
        assert!((s.mean_temperature_c - mean).abs() < 1e-5, "{s:?}");
        assert!((s.mean_relative_humidity - (12.0 * 0.9 + 24.0 * 0.6) / 36.0).abs() < 1e-5);
        assert!((s.precipitation_mm - 3.0).abs() < 1e-5);
        assert_eq!((s.min_temperature_c, s.max_temperature_c), (-3.0, 10.0));
    }
}
