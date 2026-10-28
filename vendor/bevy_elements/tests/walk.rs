//! Headless walk through the climate and weather plugin. No window, no GPU: `MinimalPlugins` and
//! `ClimatePlugin` only, with time stepped by hand.

use std::time::Duration;

use bevy::prelude::*;
use bevy::time::TimeUpdateStrategy;
use bevy_elements::*;

#[derive(Resource, Default)]
struct Seen(Vec<WeatherChanged>);

fn collect(mut changed: MessageReader<WeatherChanged>, mut seen: ResMut<Seen>) {
    seen.0.extend(changed.read().copied());
}

/// A CPU-only app at noon on 2026-06-21, `time_scale` game seconds per real second, stepped by
/// `step` per update.
fn app(climate: Climate, time_scale: f32, step: Duration) -> App {
    let mut app = App::new();
    app.add_plugins((MinimalPlugins, ClimatePlugin))
        .insert_resource(TimeUpdateStrategy::ManualDuration(step))
        .insert_resource(climate)
        .insert_resource(
            WorldClock::new(CivilDate::new(2026, 6, 21))
                .at_hour(12.0)
                .with_time_scale(time_scale),
        )
        .init_resource::<Seen>()
        .add_systems(Update, collect.after(ElementsSystems::Weather));
    app.update();
    app
}

fn run_for(app: &mut App, updates: usize) {
    for _ in 0..updates {
        app.update();
    }
}

#[test]
fn a_forced_thunderstorm_arrives_on_time_and_is_announced_once() {
    let mut app = app(
        Climate::temperate().with_seed(42),
        1.0,
        Duration::from_millis(100),
    );
    app.world_mut()
        .resource_mut::<WeatherDirector>()
        .force(WeatherPreset::Thunderstorm, Duration::from_secs(60));
    // 61 s of game time at time scale 1.
    run_for(&mut app, 610);

    let weather = *app.world().resource::<Weather>();
    assert_eq!(weather.precipitation_mm_h, 25.0);
    let forced: Vec<_> = app
        .world()
        .resource::<Seen>()
        .0
        .iter()
        .filter(|c| c.to == WeatherPreset::Thunderstorm && c.forced)
        .copied()
        .collect();
    assert_eq!(
        forced.len(),
        1,
        "messages: {:?}",
        app.world().resource::<Seen>().0
    );
    assert!(app.world().resource::<WeatherDirector>().is_forced());
}

#[test]
fn release_hands_the_weather_back_to_the_scheduler() {
    let climate = Climate::temperate().with_seed(7);
    // One game hour per real second, 0.1 s per update: 10 updates per game hour.
    let step = Duration::from_millis(100);
    let mut free = app(climate.clone(), 3600.0, step);
    let mut forced = app(climate, 3600.0, step);

    forced
        .world_mut()
        .resource_mut::<WeatherDirector>()
        .force(WeatherPreset::Snow, Duration::from_secs(600));
    run_for(&mut free, 50);
    run_for(&mut forced, 50);
    forced
        .world_mut()
        .resource_mut::<WeatherDirector>()
        .release(Duration::ZERO);
    // Past the next hour boundary.
    run_for(&mut free, 15);
    run_for(&mut forced, 15);

    let after = forced.world().resource::<WeatherDirector>();
    assert!(!after.is_forced());
    assert_eq!(
        after.current(),
        free.world().resource::<WeatherDirector>().current()
    );
}

#[test]
fn forced_snow_falls_as_snow_even_in_the_tropics() {
    let mut app = app(Climate::tropical(), 60.0, Duration::from_millis(100));
    app.world_mut()
        .resource_mut::<WeatherDirector>()
        .force(WeatherPreset::Snow, Duration::from_secs(600));
    // 600 game seconds at time scale 60 is 10 real seconds.
    run_for(&mut app, 110);

    let weather = *app.world().resource::<Weather>();
    assert!(
        weather.temperature_c <= -2.0,
        "temperature {}",
        weather.temperature_c
    );
    assert_eq!(weather.snow_fraction(), 1.0);
    assert_eq!(weather.precipitation_mm_h, 1.5);
}
