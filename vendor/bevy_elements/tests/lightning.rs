//! Headless lightning: the strike process, the messages, the thunder delay and the flash
//! resource. No window, no GPU: `MinimalPlugins` and `LightningPlugin` only, time stepped by hand.

use std::time::Duration;

use bevy::prelude::*;
use bevy::time::TimeUpdateStrategy;
use bevy_elements::*;

const FRAME: Duration = Duration::from_nanos(16_666_667);

#[derive(Resource, Default)]
struct Seen {
    /// Strikes with the `Time<Real>` second of the frame that announced them.
    strikes: Vec<(f64, LightningStrike)>,
    thunder: Vec<(f64, Thunder)>,
    /// `LightningFlash` as it stood after each frame.
    flash: Vec<LightningFlash>,
}

fn collect(
    mut strikes: MessageReader<LightningStrike>,
    mut thunder: MessageReader<Thunder>,
    real: Res<Time<Real>>,
    flash: Res<LightningFlash>,
    mut seen: ResMut<Seen>,
) {
    let now = real.elapsed_secs_f64();
    seen.strikes.extend(strikes.read().map(|s| (now, *s)));
    seen.thunder.extend(thunder.read().map(|t| (now, *t)));
    seen.flash.push(*flash);
}

fn storm(seed: u64, storminess: f32) -> App {
    let mut app = App::new();
    app.add_plugins((MinimalPlugins, LightningPlugin))
        .insert_resource(TimeUpdateStrategy::ManualDuration(FRAME))
        .insert_resource(Climate::temperate().with_seed(seed))
        .insert_resource(Weather {
            storminess,
            ..default()
        })
        .init_resource::<Seen>()
        .add_systems(Update, collect.after(ElementsSystems::Simulation));
    app.update();
    app
}

fn run(app: &mut App, updates: usize) {
    for _ in 0..updates {
        app.update();
    }
}

/// 600 s of `Time<Virtual>` at 60 updates per second.
const TEN_MINUTES: usize = 36_000;

fn strikes_in_ten_minutes(seed: u64) -> usize {
    let mut app = storm(seed, 6.0);
    run(&mut app, TEN_MINUTES);
    app.world().resource::<Seen>().strikes.len()
}

#[test]
fn a_ten_minute_storm_strikes_forty_to_eighty_times() {
    let count = strikes_in_ten_minutes(42);
    assert!((40..=80).contains(&count), "{count} strikes");
}

/// A red result means the strike model moved: the stream definition, the rate integrator or the
/// hash domain changed. The counts are the Python prototype's in `docs/research/lightning.md`.
#[test]
fn the_lightning_schedule_is_frozen() {
    let counts: Vec<usize> = [0, 1, 7, 42, 1234]
        .into_iter()
        .map(strikes_in_ten_minutes)
        .collect();
    assert_eq!(counts, [60, 62, 45, 64, 51]);
}

#[test]
fn every_thunder_arrives_distance_over_343_seconds_after_its_strike() {
    let mut app = storm(42, 6.0);
    app.world_mut().spawn((
        Camera::default(),
        ElementsCamera::default(),
        GlobalTransform::from_xyz(10_000.0, 3.0, -5_000.0),
    ));
    run(&mut app, TEN_MINUTES);
    let seen = app.world().resource::<Seen>();
    let end = app.world().resource::<Time<Real>>().elapsed_secs_f64();
    let frame = FRAME.as_secs_f64();
    let camera = Vec3::new(10_000.0, 3.0, -5_000.0);
    assert!(seen.strikes.len() >= 40);
    let mut matched = 0;
    for (struck_at, strike) in &seen.strikes {
        let horizontal = Vec2::new(strike.position.x - camera.x, strike.position.z - camera.z);
        assert!(horizontal.length() <= STRIKE_RADIUS_M + 0.01, "{strike:?}");
        let distance = strike.position.distance(camera);
        let due = struck_at + f64::from(distance / SPEED_OF_SOUND_MS);
        let thunders: Vec<_> = seen
            .thunder
            .iter()
            .filter(|(_, t)| t.position == strike.position)
            .collect();
        if due > end {
            // Still in the air when the run ended.
            assert!(thunders.is_empty(), "{strike:?}");
            continue;
        }
        assert_eq!(thunders.len(), 1, "one thunder per strike: {strike:?}");
        let (heard_at, thunder) = thunders[0];
        let lag = heard_at - due;
        assert!(
            (-1e-6..frame + 1e-6).contains(&lag),
            "thunder {lag} s late for {distance} m"
        );
        assert!((thunder.distance_m - distance).abs() < 1e-3);
        assert!((thunder.loudness - 1.0 / (1.0 + distance / 1000.0)).abs() < 1e-6);
        matched += 1;
    }
    assert!(matched >= 35, "{matched} thunders checked");
}

#[test]
fn a_scripted_strike_at_343_m_is_heard_one_second_later() {
    let mut app = storm(5, 0.0);
    let at = Vec3::new(343.0, 0.0, 0.0);
    app.world_mut().write_message(StrikeLightning { at });
    run(&mut app, 180);
    let seen = app.world().resource::<Seen>();
    assert_eq!(seen.strikes.len(), 1);
    assert_eq!(seen.strikes[0].1.position, at);
    assert_eq!(seen.thunder.len(), 1);
    let delay = seen.thunder[0].0 - seen.strikes[0].0;
    assert!(
        (1.0 - 1e-6..1.0 + FRAME.as_secs_f64() + 1e-6).contains(&delay),
        "{delay}"
    );
    assert_eq!(seen.thunder[0].1.distance_m, 343.0);
}

#[test]
fn the_same_seed_gives_the_same_storm() {
    let record = |seed| {
        let mut app = storm(seed, 6.0);
        run(&mut app, 6_000);
        app.world()
            .resource::<Seen>()
            .strikes
            .iter()
            .map(|(_, s)| *s)
            .collect::<Vec<_>>()
    };
    let (a, b, c) = (record(42), record(42), record(43));
    assert!(!a.is_empty());
    assert_eq!(a, b);
    assert_ne!(a, c);
}

#[test]
fn a_calm_sky_never_strikes() {
    let mut app = storm(42, 0.0);
    run(&mut app, 6_000);
    assert!(app.world().resource::<Seen>().strikes.is_empty());
}

#[test]
fn the_flash_follows_the_first_stroke_then_goes_dark() {
    let mut app = storm(9, 0.0);
    let at = Vec3::new(200.0, 0.0, -300.0);
    app.world_mut().write_message(StrikeLightning { at });
    app.update();
    let first = *app.world().resource::<LightningFlash>();
    // 4 pi cd per lumen at the peak, 0.830 of it averaged over a 1/60 s first frame.
    let peak = FLASH_LUMENS / (4.0 * std::f32::consts::PI);
    assert!(
        (first.intensity_cd / peak - 0.830).abs() < 0.01,
        "{} of peak",
        first.intensity_cd / peak
    );
    // The channel top is at the cloud base (1500 m in clear weather), within the wander.
    assert!(
        (1500.0..1520.0).contains(&first.position.y),
        "{:?}",
        first.position
    );
    assert!(Vec2::new(first.position.x - at.x, first.position.z - at.z).length() < 150.0);
    run(&mut app, 40);
    // The last stroke is 40-240 ms after the first and its tail ends within 240 ms of it.
    assert_eq!(app.world().resource::<LightningFlash>().intensity_cd, 0.0);
    let seen = app.world().resource::<Seen>();
    assert!(
        seen.flash.iter().filter(|f| f.intensity_cd > 0.0).count() >= 12,
        "the flash lasts at least 200 ms"
    );
    // Flicker: after the first peak the intensity falls, and a restrike brings it back up.
    let lit: Vec<f32> = seen.flash.iter().map(|f| f.intensity_cd).collect();
    assert!(
        lit.windows(2).skip(2).any(|w| w[1] > w[0]),
        "no restrike flicker in {lit:?}"
    );
}
