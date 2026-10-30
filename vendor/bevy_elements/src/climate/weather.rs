//! Weather: the presets, the blended state the rest of the crate reads, the director that forces
//! and releases it, and the seeded hourly scheduler.

use std::time::Duration;

use bevy::prelude::*;

use super::model::Climate;
use super::season::Season;
use crate::celestial::{CivilDate, GeoLocation, WorldClock};

/// A named weather state the scheduler moves between.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Reflect)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum WeatherPreset {
    /// Clear sky.
    #[default]
    Clear,
    /// Broken cumulus.
    PartlyCloudy,
    /// Full stratus cover, dry.
    Overcast,
    /// Radiation or advection fog, 300 m visibility.
    Fog,
    /// Light rain, 0.5 mm/h.
    Drizzle,
    /// Steady rain, 4 mm/h.
    Rain,
    /// Convective storm: 25 mm/h, lightning 6 strikes per minute within 8 km.
    Thunderstorm,
    /// Steady snowfall, 1.5 mm/h water equivalent.
    Snow,
    /// Wind-driven snow, 4 mm/h water equivalent, 200 m visibility.
    Blizzard,
    /// Wind-raised dust, 800 m visibility.
    DustStorm,
}

/// The target values a preset blends toward. Wind speed here is before scaling by
/// `Climate::mean_wind_ms / 4`.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Target {
    cover: f32,
    mm_h: f32,
    rh: f32,
    wind: f32,
    gust: f32,
    visibility: f32,
    storm: f32,
    dust: f32,
    delta_t: f32,
}

#[rustfmt::skip]
fn target(preset: WeatherPreset) -> Target {
    use WeatherPreset::*;
    let t = |cover, mm_h, rh, wind, gust, visibility, storm, dust, delta_t| Target { cover, mm_h, rh, wind, gust, visibility, storm, dust, delta_t };
    match preset {
        Clear =>        t(0.05, 0.0,  0.50, 3.0,  0.20, 40_000.0, 0.0, 0.0,  2.0),
        PartlyCloudy => t(0.40, 0.0,  0.60, 4.0,  0.30, 30_000.0, 0.0, 0.0,  1.0),
        Overcast =>     t(0.95, 0.0,  0.75, 5.0,  0.30, 20_000.0, 0.0, 0.0, -1.0),
        Fog =>          t(0.60, 0.0,  0.99, 1.0,  0.05,    300.0, 0.0, 0.0, -1.0),
        Drizzle =>      t(0.95, 0.5,  0.92, 4.0,  0.30,  5_000.0, 0.0, 0.0, -2.0),
        Rain =>         t(1.00, 4.0,  0.95, 7.0,  0.50,  3_000.0, 0.0, 0.0, -3.0),
        Thunderstorm => t(1.00, 25.0, 0.97, 14.0, 0.90,  1_500.0, 6.0, 0.0, -5.0),
        Snow =>         t(1.00, 1.5,  0.90, 4.0,  0.30,  1_500.0, 0.0, 0.0, -4.0),
        Blizzard =>     t(1.00, 4.0,  0.90, 18.0, 0.90,    200.0, 0.0, 0.0, -6.0),
        DustStorm =>    t(0.30, 0.0,  0.15, 16.0, 0.80,    800.0, 0.0, 0.9,  3.0),
    }
}

impl WeatherPreset {
    /// Every preset, in declaration order.
    pub const ALL: [Self; 10] = [
        Self::Clear,
        Self::PartlyCloudy,
        Self::Overcast,
        Self::Fog,
        Self::Drizzle,
        Self::Rain,
        Self::Thunderstorm,
        Self::Snow,
        Self::Blizzard,
        Self::DustStorm,
    ];

    /// Whether the preset precipitates.
    pub fn is_wet(self) -> bool {
        target(self).mm_h > 0.0
    }

    /// Whether the preset only exists in snow (forcing it caps the temperature at −2 °C).
    pub fn is_snow(self) -> bool {
        matches!(self, Self::Snow | Self::Blizzard)
    }

    fn neighbours(self) -> &'static [Self] {
        use WeatherPreset::*;
        match self {
            Clear => &[PartlyCloudy, DustStorm, Fog],
            PartlyCloudy => &[Clear, Overcast, DustStorm, Fog],
            Overcast => &[PartlyCloudy, Drizzle, Rain, Snow, Fog],
            Fog => &[Overcast, Clear, PartlyCloudy],
            Drizzle => &[Overcast],
            Rain => &[Overcast, Thunderstorm],
            Thunderstorm => &[Rain],
            Snow => &[Overcast, Blizzard],
            Blizzard => &[Snow],
            DustStorm => &[Clear, PartlyCloudy],
        }
    }
}

/// The weather now, blended between presets. Read by every domain.
#[derive(Resource, Clone, Copy, Debug, PartialEq, Reflect)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Weather {
    /// Fraction of the sky covered by cloud, 0–1.
    pub cloud_cover: f32,
    /// Precipitation rate, liquid-water equivalent, mm/h.
    pub precipitation_mm_h: f32,
    /// Air temperature at the climate's reference elevation, °C.
    pub temperature_c: f32,
    /// Relative humidity, 0–1.
    pub relative_humidity: f32,
    /// Mean wind speed at 10 m above ground, m/s.
    pub wind_speed_ms: f32,
    /// Unit XZ direction the wind blows toward.
    pub wind_direction: Vec2,
    /// Gust and turbulence intensity, 0–1.
    pub gustiness: f32,
    /// Meteorological visibility (2 % contrast), m.
    pub visibility_m: f32,
    /// Lightning strikes per minute within 8 km.
    pub storminess: f32,
    /// Airborne dust, 0–1.
    pub dust: f32,
    /// How far precipitation may fall as snow, 0–1: 0 outside the climate's
    /// [`SnowSeason`](crate::SnowSeason) (raised by a forced Snow or Blizzard), 1 otherwise.
    /// [`snow_fraction`](Self::snow_fraction) is scaled by it.
    #[cfg_attr(feature = "serde", serde(default = "one"))]
    pub snow_possible: f32,
}

#[cfg(feature = "serde")]
fn one() -> f32 {
    1.0
}

impl Default for Weather {
    fn default() -> Self {
        let t = target(WeatherPreset::Clear);
        Self {
            cloud_cover: t.cover,
            precipitation_mm_h: t.mm_h,
            temperature_c: 15.0,
            relative_humidity: t.rh,
            wind_speed_ms: t.wind,
            wind_direction: Vec2::X,
            gustiness: t.gust,
            visibility_m: t.visibility,
            storminess: t.storm,
            dust: t.dust,
            snow_possible: 1.0,
        }
    }
}

impl Weather {
    /// Fraction of precipitation falling as snow: 1 at ≤ −1 °C, 0 at ≥ 2 °C, linear between,
    /// times [`snow_possible`](Self::snow_possible).
    pub fn snow_fraction(&self) -> f32 {
        ((2.0 - self.temperature_c) / 3.0).clamp(0.0, 1.0) * self.snow_possible
    }
}

/// A weather transition started: by the scheduler, by [`WeatherDirector::force`] or by
/// [`WeatherDirector::release`].
#[derive(Message, Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct WeatherChanged {
    /// The preset being left.
    pub from: WeatherPreset,
    /// The preset being blended toward.
    pub to: WeatherPreset,
    /// Whether a [`WeatherDirector::force`] caused it.
    pub forced: bool,
}

/// Every value a transition blends.
#[derive(Clone, Copy, Debug, PartialEq, Reflect)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
struct Blend {
    cover: f32,
    mm_h: f32,
    rh: f32,
    wind: f32,
    direction: Vec2,
    gust: f32,
    visibility: f32,
    storm: f32,
    dust: f32,
    delta_t: f32,
    /// How far the forced-snow −2 °C cap is applied, 0–1.
    snow_cap: f32,
}

impl Blend {
    fn of(preset: WeatherPreset, climate: &Climate, direction: Vec2, forced: bool) -> Self {
        let t = target(preset);
        Self {
            cover: t.cover,
            mm_h: t.mm_h,
            rh: t.rh,
            wind: t.wind * climate.mean_wind_ms / 4.0,
            direction,
            gust: t.gust,
            visibility: t.visibility,
            storm: t.storm,
            dust: t.dust,
            delta_t: t.delta_t,
            snow_cap: if forced && preset.is_snow() { 1.0 } else { 0.0 },
        }
    }

    fn lerp(self, to: Self, s: f32) -> Self {
        let l = |a: f32, b: f32| a + (b - a) * s;
        Self {
            cover: l(self.cover, to.cover),
            mm_h: l(self.mm_h, to.mm_h),
            rh: l(self.rh, to.rh),
            wind: l(self.wind, to.wind),
            direction: self
                .direction
                .lerp(to.direction, s)
                .normalize_or(to.direction),
            gust: l(self.gust, to.gust),
            visibility: l(self.visibility, to.visibility),
            storm: l(self.storm, to.storm),
            dust: l(self.dust, to.dust),
            delta_t: l(self.delta_t, to.delta_t),
            snow_cap: l(self.snow_cap, to.snow_cap),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Reflect)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
struct Transition {
    from: Blend,
    to: Blend,
    preset: WeatherPreset,
    start_h: f64,
    duration_h: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Reflect)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
enum Request {
    Force(WeatherPreset, Duration),
    Release(Duration),
}

/// Who decides the weather: the seeded scheduler, or the game through [`force`](Self::force).
///
/// The scheduler keeps stepping while weather is forced, so [`release`](Self::release) blends to
/// the preset an unforced run would show at that hour. Transition durations are game time.
///
/// This is the weather's save state, with [`WorldClock`], [`Climate`] and [`GeoLocation`]:
/// insert a saved copy (through `serde` with the `serde` feature, or a reflected scene) and the
/// scheduler continues the same chain, forced preset and blend; a fresh director would restart
/// at `Clear`. [`Weather`] is recomputed from it each frame and need not be saved.
#[derive(Resource, Clone, Debug, Default, PartialEq, Reflect)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct WeatherDirector {
    scheduled: WeatherPreset,
    scheduled_direction: Option<Vec2>,
    forced: Option<WeatherPreset>,
    pending: Option<Request>,
    transition: Option<Transition>,
    current: Option<Blend>,
    last_hour: Option<i64>,
}

impl WeatherDirector {
    /// Override the scheduler: blend to `preset` over `transition` of game time and hold it until
    /// [`release`](Self::release). While Snow or Blizzard is forced the temperature is capped at
    /// −2 °C, so a forced snowfall always falls as snow.
    pub fn force(&mut self, preset: WeatherPreset, transition: Duration) {
        self.forced = Some(preset);
        self.pending = Some(Request::Force(preset, transition));
    }

    /// Hand control back to the scheduler, blending to its current preset over `transition` of
    /// game time.
    pub fn release(&mut self, transition: Duration) {
        self.forced = None;
        self.pending = Some(Request::Release(transition));
    }

    /// The preset the weather is at or blending toward.
    pub fn current(&self) -> WeatherPreset {
        match (self.forced, self.transition) {
            (Some(p), _) => p,
            (None, Some(t)) => t.preset,
            (None, None) => self.scheduled,
        }
    }

    /// Whether [`force`](Self::force) is in effect.
    pub fn is_forced(&self) -> bool {
        self.forced.is_some()
    }

    fn start(&mut self, preset: WeatherPreset, climate: &Climate, now_h: f64, duration_h: f64) {
        let direction = self.scheduled_direction.unwrap_or(climate.prevailing_wind);
        let to = Blend::of(preset, climate, direction, self.forced.is_some());
        let from = self.current.unwrap_or(to);
        self.transition = Some(Transition {
            from,
            to,
            preset,
            start_h: now_h,
            duration_h,
        });
    }
}

/// SplitMix64 finaliser (Steele, Lea & Flood 2014).
pub(crate) fn mix64(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// The scheduler's hash of one game hour.
pub(crate) fn hash64(seed: u64, index: u64) -> u64 {
    mix64(seed ^ index.wrapping_mul(0x9E37_79B9_7F4A_7C15))
}

fn unit24(bits: u64) -> f32 {
    (bits & 0xFF_FFFF) as f32 / 16_777_216.0
}

/// One hourly decision of the scheduler.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Step {
    pub preset: WeatherPreset,
    /// Game minutes the blend to `preset` takes, 20–60.
    pub transition_min: f32,
    /// Rotation of the prevailing wind for this preset, degrees, ±30.
    pub wind_turn_deg: f32,
}

/// Probability of keeping the current preset at an hour boundary.
const STAY: f32 = 0.85;
/// Base neighbour weight of Fog (every other base weight is 1), calibrated so the preset
/// climates spend roughly `fog_share` of their hours in fog.
const FOG_BASE_WEIGHT: f32 = 0.3;
/// Base neighbour weight of DustStorm, calibrated so `Climate::desert()` spends roughly
/// `dust_share` of its hours in dust storms.
const DUST_BASE_WEIGHT: f32 = 0.2;

/// Decide the preset for absolute hour `index` (local hours since 2000-01-01 00:00) following
/// `prev`.
pub(crate) fn schedule(climate: &Climate, southern: bool, prev: WeatherPreset, index: i64) -> Step {
    let date = CivilDate::new(2000, 1, 1).add_days(index.div_euclid(24));
    let hour = index.rem_euclid(24) as f32;
    let doy = f32::from(date.day_of_year()) + hour / 24.0;
    let temperature = climate.base_temperature_c(doy, hour, southern);
    let summer = Season::of_month(date.month, southern) == Season::Summer;
    // Out of the snow season (its whole day, the ramps included) the snow presets are gated as
    // above 1 °C.
    let snowless = climate.snow_season.is_some_and(|s| !s.contains(date));

    let gated = |p: WeatherPreset| match p {
        WeatherPreset::Snow | WeatherPreset::Blizzard => temperature > 1.0 || snowless,
        WeatherPreset::Rain | WeatherPreset::Drizzle | WeatherPreset::Thunderstorm => {
            temperature < -1.0
        }
        WeatherPreset::DustStorm => climate.dust_share <= 0.0,
        _ => false,
    };
    let weight = |p: WeatherPreset| -> f32 {
        if gated(p) {
            return 0.0;
        }
        let mut w = 1.0;
        if p.is_wet() {
            w *= climate.wet_hour_probability / 0.35;
        }
        match p {
            WeatherPreset::Thunderstorm => {
                w *= climate.storm_share / 0.05 * if summer { 1.0 } else { 0.3 };
            }
            WeatherPreset::Fog => {
                w *= FOG_BASE_WEIGHT * climate.fog_share / 0.08;
                if (4.0..=9.0).contains(&hour) && target(prev).rh > 0.85 {
                    w *= 3.0;
                }
            }
            WeatherPreset::DustStorm => w *= DUST_BASE_WEIGHT * climate.dust_share / 0.08,
            _ => {}
        }
        w.max(0.0)
    };

    let h = hash64(climate.seed, index as u64);
    let u_stay = unit24(h >> 40);
    let u_pick = unit24(h >> 16);
    let u_duration = (h & 0xFFFF) as f32 / 65_536.0;
    let u_turn = unit24(mix64(h) >> 40);

    let prev_gated = gated(prev);
    let preset = if !prev_gated && u_stay < STAY {
        prev
    } else {
        let neighbours = prev.neighbours();
        let total: f32 = neighbours.iter().map(|&p| weight(p)).sum();
        if total <= 0.0 {
            if prev_gated {
                WeatherPreset::Overcast
            } else {
                prev
            }
        } else {
            let mut pick = u_pick * total;
            let mut chosen = prev;
            for &p in neighbours {
                let w = weight(p);
                if w <= 0.0 {
                    continue;
                }
                chosen = p;
                if pick < w {
                    break;
                }
                pick -= w;
            }
            chosen
        }
    };
    Step {
        preset,
        transition_min: 20.0 + 40.0 * u_duration,
        wind_turn_deg: (u_turn * 2.0 - 1.0) * 30.0,
    }
}

fn rotate(v: Vec2, degrees: f32) -> Vec2 {
    let (s, c) = degrees.to_radians().sin_cos();
    Vec2::new(v.x * c - v.y * s, v.x * s + v.y * c)
}

/// The most hours one frame steps the scheduler through; a larger clock jump starts the chain
/// afresh at the new hour.
const MAX_STEPS_PER_FRAME: i64 = 24 * 366;

pub(crate) fn run_weather(
    clock: Res<WorldClock>,
    location: Res<GeoLocation>,
    climate: Res<Climate>,
    mut director: ResMut<WeatherDirector>,
    mut weather: ResMut<Weather>,
    mut changed: MessageWriter<WeatherChanged>,
) {
    let now_h = clock.hours_since_2000();
    let index = now_h.floor() as i64;
    let southern = location.is_southern();
    let prevailing = climate.prevailing_wind.normalize_or(Vec2::X);

    match director.last_hour {
        None => {
            // First frame: the chain starts Clear at this hour, already settled.
            director.scheduled = WeatherPreset::Clear;
            director.scheduled_direction = Some(prevailing);
            if director.forced.is_none() {
                director.start(WeatherPreset::Clear, &climate, now_h, 0.0);
            }
            director.last_hour = Some(index);
        }
        Some(last) if index < last || index - last > MAX_STEPS_PER_FRAME => {
            director.last_hour = Some(index);
        }
        Some(last) => {
            for hour in last + 1..=index {
                let step = schedule(&climate, southern, director.scheduled, hour);
                if step.preset == director.scheduled {
                    continue;
                }
                let from = director.current();
                director.scheduled = step.preset;
                director.scheduled_direction = Some(rotate(prevailing, step.wind_turn_deg));
                if director.forced.is_none() {
                    director.start(
                        step.preset,
                        &climate,
                        hour as f64,
                        f64::from(step.transition_min) / 60.0,
                    );
                    changed.write(WeatherChanged {
                        from,
                        to: step.preset,
                        forced: false,
                    });
                }
            }
            director.last_hour = Some(index);
        }
    }

    if let Some(request) = director.pending.take() {
        let from = director.current_target_preset();
        let (to, duration, forced) = match request {
            Request::Force(preset, d) => (preset, d, true),
            Request::Release(d) => (director.scheduled, d, false),
        };
        director.start(to, &climate, now_h, duration.as_secs_f64() / 3600.0);
        if from != to {
            changed.write(WeatherChanged { from, to, forced });
        }
    }

    let Some(t) = director.transition else {
        return;
    };
    let s = if t.duration_h <= 0.0 || now_h < t.start_h {
        1.0
    } else {
        let x = ((now_h - t.start_h) / t.duration_h).clamp(0.0, 1.0) as f32;
        x * x * (3.0 - 2.0 * x)
    };
    let blend = t.from.lerp(t.to, s);
    director.current = Some(blend);

    let doy = f32::from(clock.day_of_year()) + clock.hour as f32 / 24.0;
    let mut temperature =
        climate.base_temperature_c(doy, clock.hour as f32, southern) + blend.delta_t;
    temperature += (temperature.min(-2.0) - temperature) * blend.snow_cap;
    *weather = Weather {
        cloud_cover: blend.cover,
        precipitation_mm_h: blend.mm_h,
        temperature_c: temperature,
        relative_humidity: blend.rh,
        wind_speed_ms: blend.wind,
        wind_direction: blend.direction,
        gustiness: blend.gust,
        visibility_m: blend.visibility,
        storminess: blend.storm,
        dust: blend.dust,
        snow_possible: climate
            .snow_possible(clock.date, clock.hour)
            .max(blend.snow_cap),
    };
}

impl WeatherDirector {
    /// The preset of the transition in progress, before this frame's request is applied.
    fn current_target_preset(&self) -> WeatherPreset {
        self.transition.map_or(self.scheduled, |t| t.preset)
    }

    /// The scheduler's coming weather: the preset it will have chosen for each of the `hours`
    /// game hours after `clock`'s hour, element 0 for the next hour. It runs the seeded chain
    /// forward from where this director is, exactly as the hourly steps will (a director that has
    /// not stepped yet starts its chain Clear at `clock`'s hour), so with the same `climate` and
    /// `location` the weather turns out as forecast. A forced preset does not enter it: the chain
    /// keeps stepping underneath a force, and [`release`](Self::release) blends back to it. Pure:
    /// the director is not changed.
    pub fn forecast(
        &self,
        climate: &Climate,
        location: &GeoLocation,
        clock: &WorldClock,
        hours: u32,
    ) -> Vec<WeatherPreset> {
        let index = clock.hours_since_2000().floor() as i64;
        let southern = location.is_southern();
        let (mut preset, from) = match self.last_hour {
            None => (WeatherPreset::Clear, index),
            Some(last) if index < last || index - last > MAX_STEPS_PER_FRAME => {
                (self.scheduled, index)
            }
            Some(last) => (self.scheduled, last),
        };
        for hour in from + 1..=index {
            preset = schedule(climate, southern, preset, hour).preset;
        }
        (index + 1..=index + i64::from(hours))
            .map(|hour| {
                preset = schedule(climate, southern, preset, hour).preset;
                preset
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_hash_matches_splitmix64() {
        assert_eq!(mix64(0x9E37_79B9_7F4A_7C15), 0xE220_A839_7B1D_CDAF);
        assert_eq!(hash64(0, 1), 0xE220_A839_7B1D_CDAF);
        assert_eq!(hash64(42, 0), 0xA759_EA27_D472_7622);
        assert_eq!(hash64(42, 1), 0xBDD7_3226_2FEB_6E95);
        assert_eq!(hash64(42, 8759), 0xAA5E_C7C5_CF51_7539);
    }

    #[test]
    fn the_clear_preset_is_the_visibility_the_sky_air_is_measured_against() {
        // `fog::CLEAR_AIR_EXTINCTION` treats the `Clear` preset's visibility as clear air
        // (and `precipitation_split`'s floor follows); if either moves, the sky's fog moves
        // with it, so pin them together.
        assert_eq!(target(WeatherPreset::Clear).visibility, crate::fog::CLEAR_VISIBILITY_M);
    }

    #[test]
    fn snow_fraction_ramps_from_minus_one_to_two() {
        let at = |t| {
            Weather {
                temperature_c: t,
                ..Weather::default()
            }
            .snow_fraction()
        };
        assert_eq!(at(-10.0), 1.0);
        assert_eq!(at(-1.0), 1.0);
        assert!((at(0.5) - 0.5).abs() < 1e-6);
        assert_eq!(at(2.0), 0.0);
        assert_eq!(at(30.0), 0.0);
    }

    fn year_from(climate: &Climate, start: CivilDate, hours: i64) -> Vec<WeatherPreset> {
        let first = start.days_since_2000() * 24;
        let mut preset = WeatherPreset::Clear;
        (0..hours)
            .map(|h| {
                preset = schedule(climate, false, preset, first + h).preset;
                preset
            })
            .collect()
    }

    /// The scheduler's output for a temperate year, seed 42, from 2026-01-01 00:00, folded with
    /// FNV-1a over the preset discriminants.
    ///
    /// **A red result means the scheduler model moved:** a weight, the hash, the graph, the gates
    /// or the draw order changed, and every seeded world now has different weather. Re-bless only
    /// when that is intended, in the same commit as the change.
    #[test]
    fn the_weather_schedule_is_frozen() {
        let climate = Climate::temperate().with_seed(42);
        let presets = year_from(&climate, CivilDate::new(2026, 1, 1), 8760);
        let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
        for p in &presets {
            let byte = WeatherPreset::ALL.iter().position(|q| q == p).unwrap_or(0) as u64;
            hash ^= byte;
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
        assert_eq!(
            hash, SCHEDULE_GOLDEN,
            "weather schedule hash is {hash:#018x}"
        );
    }

    const SCHEDULE_GOLDEN: u64 = 0x025f6c432fa413f4;

    fn hour_share(climate: &Climate, preset: WeatherPreset) -> f32 {
        let mut hits = 0usize;
        for seed in 1..=8u64 {
            let year = year_from(
                &climate.clone().with_seed(seed),
                CivilDate::new(2026, 1, 1),
                8760,
            );
            hits += year.iter().filter(|p| **p == preset).count();
        }
        hits as f32 / (8.0 * 8760.0)
    }

    /// `fog_share`, `dust_share` and `storm_share` read as fractions of hours, and the base
    /// weights are calibrated so the preset climates honour that within a few points.
    #[test]
    fn preset_climates_spend_about_their_share_in_fog_dust_and_storms() {
        let within = |x: f32, share: f32| (x - share).abs() < 0.035;
        let t = Climate::temperate();
        assert!(within(hour_share(&t, WeatherPreset::Fog), t.fog_share));
        assert!(within(
            hour_share(&t, WeatherPreset::Thunderstorm),
            t.storm_share
        ));
        let d = Climate::desert();
        assert!(within(
            hour_share(&d, WeatherPreset::DustStorm),
            d.dust_share
        ));
        let tr = Climate::tropical();
        assert!(within(
            hour_share(&tr, WeatherPreset::Thunderstorm),
            tr.storm_share
        ));
    }

    #[test]
    fn runs_last_about_as_long_as_the_stay_probability_implies() {
        let climate = Climate::temperate().with_seed(7);
        let presets = year_from(&climate, CivilDate::new(2026, 1, 1), 8760 * 4);
        let changes = presets.windows(2).filter(|w| w[0] != w[1]).count();
        let mean_run = presets.len() as f32 / (changes + 1) as f32;
        // 1 / (1 − 0.85) = 6.67 h, minus the hours gates force a move.
        assert!((5.0..=7.5).contains(&mean_run), "mean run {mean_run} h");
    }

    #[test]
    fn gates_keep_snow_out_of_summer_and_rain_out_of_deep_winter() {
        let presets = year_from(
            &Climate::temperate().with_seed(3),
            CivilDate::new(2026, 7, 1),
            24 * 31,
        );
        assert!(!presets.iter().any(|p| p.is_snow()));
        let tundra = Climate::tundra().with_seed(3);
        let presets = year_from(&tundra, CivilDate::new(2026, 1, 10), 24 * 20);
        assert!(!presets.iter().any(|p| matches!(
            p,
            WeatherPreset::Rain | WeatherPreset::Drizzle | WeatherPreset::Thunderstorm
        )));
        assert!(presets.iter().any(|p| p.is_snow()));
    }

    #[test]
    fn a_climate_without_dust_never_raises_a_dust_storm() {
        let presets = year_from(
            &Climate::temperate().with_seed(9),
            CivilDate::new(2026, 1, 1),
            8760,
        );
        assert!(!presets.contains(&WeatherPreset::DustStorm));
        let desert = year_from(
            &Climate::desert().with_seed(9),
            CivilDate::new(2026, 1, 1),
            8760,
        );
        assert!(desert.contains(&WeatherPreset::DustStorm));
    }

    /// Issue #27 (precipitation-look.md Validation 5): the forecast is the weather the director
    /// then schedules, hour by hour, whether a preset is forced meanwhile or not, from a director
    /// that has not stepped yet and from one mid-chain.
    #[test]
    fn the_forecast_is_the_weather_that_comes() {
        use crate::ClimatePlugin;
        use std::collections::BTreeMap;
        let app = |force: bool| {
            let mut app = App::new();
            app.add_plugins((MinimalPlugins, ClimatePlugin))
                .insert_resource(bevy::time::TimeUpdateStrategy::ManualDuration(
                    Duration::from_millis(100),
                ))
                .insert_resource(Climate::temperate().with_seed(42))
                .insert_resource(
                    WorldClock::new(CivilDate::new(2026, 3, 1))
                        .at_hour(6.5)
                        .with_time_scale(3600.0),
                );
            if force {
                app.world_mut()
                    .resource_mut::<WeatherDirector>()
                    .force(WeatherPreset::Thunderstorm, Duration::ZERO);
            }
            app
        };
        let forecast = |app: &App, hours| {
            let w = app.world();
            w.resource::<WeatherDirector>().forecast(
                w.resource::<Climate>(),
                w.resource::<GeoLocation>(),
                w.resource::<WorldClock>(),
                hours,
            )
        };
        // Scheduled preset per hour index, 0.1 game hours an update.
        let run = |app: &mut App, hours: usize| {
            let mut seen = BTreeMap::new();
            for _ in 0..hours * 10 + 5 {
                app.update();
                let w = app.world();
                let index = w.resource::<WorldClock>().hours_since_2000().floor() as i64;
                seen.insert(index, w.resource::<WeatherDirector>().scheduled);
            }
            seen
        };
        let start = |app: &App| app.world().resource::<WorldClock>().hours_since_2000().floor() as i64;

        // A director that has not stepped: the chain starts Clear at the clock's hour.
        let mut fresh = app(false);
        let first = forecast(&fresh, 72);
        let from = start(&fresh);
        let seen = run(&mut fresh, 72);
        let came: Vec<_> = (1..=72).map(|h| seen[&(from + h)]).collect();
        assert_eq!(first, came);
        assert!(came.windows(2).any(|w| w[0] != w[1]), "72 h without a change");

        // Mid-chain, with Thunderstorm forced underneath: the same chain.
        let mut forced = app(true);
        run(&mut forced, 30);
        let director = forced.world().resource::<WeatherDirector>().clone();
        let next = forecast(&forced, 48);
        assert_eq!(*forced.world().resource::<WeatherDirector>(), director);
        assert!(director.is_forced());
        let from = start(&forced);
        let seen = run(&mut forced, 48);
        let came: Vec<_> = (1..=48).map(|h| seen[&(from + h)]).collect();
        assert_eq!(next, came);
    }
}
