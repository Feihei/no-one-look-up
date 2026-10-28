//! Climate, seasons and weather: a seeded scheduler that picks hourly weather from a
//! [`Climate`], a director that lets the game force it, and the ground's response.

mod model;
mod season;
mod weather;
pub(crate) mod weathering;

use bevy::prelude::*;

pub use model::{Climate, LAPSE_RATE_C_PER_M};
pub use season::{Season, SeasonState};
pub use weather::{Weather, WeatherChanged, WeatherDirector, WeatherPreset};
pub(crate) use weather::{hash64, mix64};
pub use weathering::SurfaceWeathering;

use crate::celestial::{GeoLocation, WorldClock};
use crate::plugin::{ElementsSystems, ensure_core};

/// The season changed. Also written once on the first frame, so a reader can sync without
/// polling [`SeasonState`].
#[derive(Message, Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SeasonChanged {
    /// The season now.
    pub season: Season,
}

/// Climate, seasons, weather and surface weathering.
///
/// Inserts (if absent) [`Climate`], [`WeatherDirector`], [`SeasonState`] and
/// [`SurfaceWeathering`], and writes [`Weather`] every frame in [`ElementsSystems::Weather`].
pub struct ClimatePlugin;

impl Plugin for ClimatePlugin {
    fn build(&self, app: &mut App) {
        ensure_core(app);
        app.init_resource::<Climate>()
            .init_resource::<WeatherDirector>()
            .init_resource::<SeasonState>()
            .init_resource::<SurfaceWeathering>()
            .add_message::<WeatherChanged>()
            .add_message::<SeasonChanged>()
            .add_systems(
                Update,
                (
                    update_season,
                    weather::run_weather,
                    weathering::update_weathering,
                )
                    .chain()
                    .in_set(ElementsSystems::Weather),
            );
    }
}

fn update_season(
    clock: Res<WorldClock>,
    location: Res<GeoLocation>,
    mut state: ResMut<SeasonState>,
    mut changed: MessageWriter<SeasonChanged>,
    mut first: Local<bool>,
) {
    let days_in_year = if clock.date.is_leap_year() {
        366.0
    } else {
        365.0
    };
    let day = f32::from(clock.day_of_year()) + clock.hour as f32 / 24.0;
    let next = SeasonState::at(day, clock.date.month, location.is_southern(), days_in_year);
    if !*first || next.season != state.season {
        changed.write(SeasonChanged {
            season: next.season,
        });
        *first = true;
    }
    *state = next;
}
