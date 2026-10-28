//! Saving and restoring the weather through `serde`: a restored director continues the same
//! scheduler chain, forced preset and blend as the run it was saved from. Headless, no GPU.

use std::time::Duration;

use bevy::prelude::*;
use bevy::time::TimeUpdateStrategy;
use bevy_elements::*;

/// One game hour per real second, stepped 0.1 s per update: 0.1 game hours per update.
fn app() -> App {
    let mut app = App::new();
    app.add_plugins((MinimalPlugins, ClimatePlugin))
        .insert_resource(TimeUpdateStrategy::ManualDuration(Duration::from_millis(100)))
        .insert_resource(Climate::temperate().with_seed(42))
        .insert_resource(
            WorldClock::new(CivilDate::new(2026, 3, 1))
                .at_hour(6.0)
                .with_time_scale(3600.0),
        );
    app.update();
    app
}

fn run_for(app: &mut App, updates: usize) {
    for _ in 0..updates {
        app.update();
    }
}

/// The save file: everything the weather cannot re-derive.
fn save(app: &App) -> String {
    let w = app.world();
    serde_json::to_string(&(
        w.resource::<WorldClock>(),
        w.resource::<Climate>(),
        w.resource::<GeoLocation>(),
        w.resource::<WeatherDirector>(),
    ))
    .unwrap()
}

fn load(app: &mut App, file: &str) {
    let (clock, climate, location, director): (WorldClock, Climate, GeoLocation, WeatherDirector) =
        serde_json::from_str(file).unwrap();
    app.insert_resource(clock)
        .insert_resource(climate)
        .insert_resource(location)
        .insert_resource(director);
}

/// Steps both apps together and returns how many updates showed the same weather, and how many
/// of those updates the original changed preset or was mid-blend.
fn compare(a: &mut App, b: &mut App, updates: usize) -> (usize, usize) {
    let (mut same, mut moving) = (0, 0);
    let mut last = *a.world().resource::<Weather>();
    for _ in 0..updates {
        a.update();
        b.update();
        let wa = *a.world().resource::<Weather>();
        let wb = *b.world().resource::<Weather>();
        assert_eq!(
            a.world().resource::<WorldClock>().hours_since_2000(),
            b.world().resource::<WorldClock>().hours_since_2000()
        );
        same += usize::from(wa == wb);
        moving += usize::from(wa.cloud_cover != last.cloud_cover);
        last = wa;
    }
    (same, moving)
}

#[test]
fn a_restored_director_continues_the_same_weather() {
    let mut original = app();
    run_for(&mut original, 720);
    let file = save(&original);

    let mut restored = app();
    load(&mut restored, &file);
    let (same, moving) = compare(&mut original, &mut restored, 720);
    assert_eq!(same, 720, "restored weather diverged");
    assert!(moving > 0, "the 72 h after the save never changed the weather");

    // Without the director the chain restarts at Clear and drifts away.
    let mut original = app();
    run_for(&mut original, 720);
    let mut clock_only = app();
    clock_only.insert_resource(original.world().resource::<WorldClock>().clone());
    let (same, _) = compare(&mut original, &mut clock_only, 720);
    assert!(same < 720, "a fresh director matched the saved run by chance");
}

#[test]
fn a_director_saved_mid_force_restores_forced_at_the_same_blend() {
    let mut original = app();
    run_for(&mut original, 100);
    original
        .world_mut()
        .resource_mut::<WeatherDirector>()
        .force(WeatherPreset::Thunderstorm, Duration::from_secs(2 * 3600));
    // One game hour into a two-hour blend.
    run_for(&mut original, 10);
    let mid = *original.world().resource::<Weather>();
    assert!(mid.precipitation_mm_h > 0.0 && mid.precipitation_mm_h < 25.0);
    let file = save(&original);

    let mut restored = app();
    load(&mut restored, &file);
    let director = restored.world().resource::<WeatherDirector>();
    assert!(director.is_forced());
    assert_eq!(director.current(), WeatherPreset::Thunderstorm);
    let (same, _) = compare(&mut original, &mut restored, 200);
    assert_eq!(same, 200);
    assert_eq!(
        restored.world().resource::<Weather>().precipitation_mm_h,
        25.0
    );
}
