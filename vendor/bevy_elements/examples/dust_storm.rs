//! A dust storm over a dune field: a forced `DustStorm` in a desert climate over
//! `ProceduralTerrain::dunes(7)`. The ground fog turns tan and thick, dust sprites stream past the
//! camera with the wind, and the wind carries the terrain's loose sand downwind, so the sand
//! pattern migrates (`DustPlugin`). Hydrology is held and the edge is closed, so the sand is only
//! moved, never lost.
//!
//! `cargo run --release --example dust_storm`
//!
//! The bed moves at the clock's time scale (here 600, so 60 s of real time are ten hours of dune
//! evolution at the plan's flux): the status line shows the migration the note predicts for a
//! 4 m feature, `Q_eff · 60 s / H`. Keys: `[` and `]` halve and double the time scale (the
//! harness's).
//!
//! For unattended runs set `ELEMENTS_FRAMES` (and `ELEMENTS_SHOT`) as in the other examples; the
//! frames then run at a fixed 1/60 s, so 3 600 frames are exactly 60 s. `ELEMENTS_FRAMES=600` and
//! `ELEMENTS_FRAMES=4200` are the two shots, 60 s apart.

mod common;

use std::time::Duration;

use bevy::prelude::*;
use bevy::time::TimeUpdateStrategy;
use bevy_elements::*;
use common::{ExampleHarness, camera};

/// Edge of the terrain, m: 2 m cells at the 512² a `High` terrain has.
const EXTENT_M: f32 = 1024.0;
/// Bed seconds per real second: the plan's flux at this scale moves a 4 m feature 37 m a minute.
const TIME_SCALE: f32 = 600.0;

fn main() {
    let hour = std::env::var("ELEMENTS_HOUR")
        .ok()
        .and_then(|h| h.parse().ok())
        .unwrap_or(10.0);
    let mut app = App::new();
    app.add_plugins((DefaultPlugins, ElementsPlugins, ExampleHarness))
        .insert_resource(Climate::desert())
        .insert_resource(WorldClock {
            paused: true,
            time_scale: TIME_SCALE,
            ..WorldClock::new(CivilDate::new(2026, 6, 21)).at_hour(hour)
        })
        // Water is not the subject: hold hydrology, and close the edge so no sand leaves.
        .insert_resource(HydrologySettings {
            time_scale: 0.0,
            boundary: HydrologyBoundary::Closed,
            evaporation: false,
            ..default()
        })
        .add_systems(Startup, setup)
        .add_systems(Update, report.after(ElementsSystems::Readback));
    if std::env::var("ELEMENTS_FRAMES").is_ok() {
        app.insert_resource(TimeUpdateStrategy::ManualDuration(Duration::from_secs_f64(
            1.0 / 60.0,
        )));
    }
    app.run();
}

/// `x,y,z` from an environment variable.
fn vec3_env(name: &str) -> Option<Vec3> {
    let value = std::env::var(name).ok()?;
    let mut parts = value.split(',').map(|p| p.trim().parse::<f32>());
    Some(Vec3::new(
        parts.next()?.ok()?,
        parts.next()?.ok()?,
        parts.next()?.ok()?,
    ))
}

fn setup(mut commands: Commands, mut director: ResMut<WeatherDirector>) {
    commands.spawn(
        Terrain::procedural(ProceduralTerrain::dunes(7))
            .with_extent(Vec2::splat(EXTENT_M))
            .with_resolution(UVec2::splat(512)),
    );
    let eye = vec3_env("ELEMENTS_EYE").unwrap_or(Vec3::new(-330.0, 70.0, -260.0));
    let look = vec3_env("ELEMENTS_LOOK").unwrap_or(Vec3::new(-60.0, 12.0, 0.0));
    commands.spawn(camera(eye, look));
    director.force(WeatherPreset::DustStorm, Duration::ZERO);
}

/// Print the transport's progress and the predicted migration once per second of real time.
fn report(
    weather: Res<Weather>,
    clock: Res<WorldClock>,
    stats: Res<DustStats>,
    settings: Res<DustSettings>,
    time: Res<Time<Real>>,
    precipitation: Option<Res<PrecipitationState>>,
    mut last: Local<u64>,
) {
    let second = time.elapsed_secs() as u64;
    if second == *last {
        return;
    }
    *last = second;
    let u = weather.wind_speed_ms;
    // The note's effective flux: k u² (u − u_t) at the bed's time scale, and the shift it makes
    // a feature of crest-to-base height H move in a minute of real time.
    let flux = settings.flux_coefficient * u * u * (u - settings.threshold_ms).max(0.0);
    let q_eff = flux * clock.time_scale;
    println!(
        "t = {:>4} s  wind {u:.1} m/s  dust {:.2}  {} sprites  {} transport steps ({:.0} s of bed time)  predicted shift per minute {:.1} m for H = 4 m",
        second,
        weather.dust,
        precipitation.map_or(0, |p| p.dust),
        stats.steps,
        stats.bed_time_s,
        q_eff * 60.0 / 4.0,
    );
}
