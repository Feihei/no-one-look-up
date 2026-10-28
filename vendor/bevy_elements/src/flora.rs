//! Feed `bevy_flora` from this crate's wind, seasons and weathering (feature `flora`).
//!
//! `bevy_flora` owns tree visuals and their global wind, season and wetness resources; this
//! module only writes those resources from [`WindState`], [`WindGust`], [`SeasonState`],
//! [`Weather`] and [`SurfaceWeathering`]. Flora's shaders read one global wind, so the spatial
//! wind field is not passed on.

use bevy::prelude::*;
use bevy_flora::{FloraGround, FloraSeason, FloraWetness, FloraWind, GustFront};

use crate::climate::{SeasonState, SurfaceWeathering, Weather};
use crate::plugin::{ElementsSystems, ensure_core};
use crate::query::EnvironmentSnapshot;
use crate::wind::{WindGust, WindState};

/// Wind speed at 10 m, m/s, that `bevy_flora`'s `FloraWind::strength` of 1 ("a strong wind")
/// stands for. Gust strengths are converted on the same scale.
pub const FLORA_FULL_WIND_MS: f32 = 20.0;

/// Drives `bevy_flora`'s `FloraWind`, `FloraSeason` and `FloraWetness` from this crate.
///
/// Not part of [`ElementsPlugins`](crate::ElementsPlugins): add it next to `bevy_flora`'s
/// `FloraPlugin`. It runs in `Update` after [`ElementsSystems::Readback`] and leaves every
/// `FloraWind` field it does not map (speed, flutter, bend, frequencies) as the game set it.
///
/// | `bevy_flora` | from |
/// |---|---|
/// | `FloraWind::direction` | `WindState::prevailing` |
/// | `FloraWind::strength` | `WindState::speed_10m / 20`, clamped to 0–1 |
/// | `FloraWind::turbulence` | `Weather::gustiness` |
/// | `FloraWind::gust_strength` | `(WindState::gust_factor − 1) / 0.5` (= gustiness), 0–1 |
/// | `FloraWind::launch_gust` | each [`WindGust`], strength / 20 |
/// | `FloraSeason` | `SeasonState::{autumn, leaf_fall, blossom}`, `snow` = `SurfaceWeathering::snow_cover` |
/// | `FloraWetness` | `SurfaceWeathering::wetness` |
/// | `FloraGround` | the terrain's height snapshot ([`Environment::terrain_height`](crate::Environment::terrain_height)), edge cells extended outside the terrain; removed again when the terrain goes |
pub struct ElementsFloraPlugin;

impl Plugin for ElementsFloraPlugin {
    fn build(&self, app: &mut App) {
        ensure_core(app);
        app.add_systems(
            Update,
            (drive_wind, drive_season, drive_ground).after(ElementsSystems::Readback),
        );
    }
}

fn drive_wind(
    wind: Res<WindState>,
    weather: Res<Weather>,
    time: Res<Time>,
    mut gusts: MessageReader<WindGust>,
    flora: Option<ResMut<FloraWind>>,
) {
    let Some(mut flora) = flora else {
        gusts.clear();
        return;
    };
    let mut next = *flora;
    next.direction = wind.prevailing;
    next.strength = (wind.speed_10m / FLORA_FULL_WIND_MS).clamp(0.0, 1.0);
    next.turbulence = weather.gustiness;
    next.gust_strength = ((wind.gust_factor - 1.0) / 0.5).clamp(0.0, 1.0);
    for gust in gusts.read() {
        next.launch_gust(
            GustFront {
                origin: gust.origin,
                direction: gust.direction,
                speed: gust.speed,
                width: gust.width,
                strength: gust.strength / FLORA_FULL_WIND_MS,
                start: 0.0,
            },
            &time,
        );
    }
    flora.set_if_neq(next);
}

fn drive_season(
    season: Option<Res<SeasonState>>,
    ground: Option<Res<SurfaceWeathering>>,
    flora_season: Option<ResMut<FloraSeason>>,
    wetness: Option<ResMut<FloraWetness>>,
) {
    let snow = ground.as_ref().map_or(0.0, |g| g.snow_cover);
    if let (Some(season), Some(mut flora_season)) = (season, flora_season) {
        flora_season.set_if_neq(FloraSeason {
            autumn: season.autumn,
            leaf_fall: season.leaf_fall,
            snow,
            blossom: season.blossom,
        });
    }
    if let (Some(ground), Some(mut wetness)) = (ground, wetness) {
        wetness.set_if_neq(FloraWetness(ground.wetness));
    }
}

/// Insert `FloraGround` from the terrain snapshot whenever a new readback arrived. The closure
/// captures the snapshot (its heights are shared, not copied).
fn drive_ground(
    snapshot: Res<EnvironmentSnapshot>,
    mut commands: Commands,
    mut inserted: Local<Option<usize>>,
) {
    let key = snapshot
        .terrain
        .as_ref()
        .map(|t| std::sync::Arc::as_ptr(&t.surface).cast::<f32>() as usize);
    if key == *inserted {
        return;
    }
    match &snapshot.terrain {
        Some(terrain) => {
            let terrain = terrain.clone();
            commands.insert_resource(FloraGround::new(move |at: Vec2| terrain.height_clamped(at)));
        }
        None => commands.remove_resource::<FloraGround>(),
    }
    *inserted = key;
}
