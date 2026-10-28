//! A day passing at latitude 45° on the June solstice: sunrise near 04:11, a 68° noon sun,
//! sunset near 19:49, then stars and a waxing moon. The camera follows what is up — the sun
//! while it is above the horizon, the moon once the sun is down — and returns to the evening
//! heading when neither is up, so every body is framed as a disc in its turn.
//!
//! `cargo run --release --example day_night`
//!
//! `ELEMENTS_HOUR=19.5` pauses the clock at that local hour (for screenshots) and aims once, so
//! the still is framed on the body; otherwise a day passes in two minutes. Hold the right mouse
//! button to look around; the aim resumes when you let go.
//! `ELEMENTS_DATE=2026-07-14` sets the calendar date (proleptic Gregorian, out-of-range months
//! and days clamped), which is how a moon-phase sweep is captured — the date fixes the moon's
//! phase and the status line prints it.

mod common;

use bevy::prelude::*;
use bevy_elements::*;
use common::{ExampleHarness, FlyCamera, camera, test_scene};

fn main() {
    let fixed: Option<f64> = std::env::var("ELEMENTS_HOUR")
        .ok()
        .and_then(|h| h.parse().ok());
    let date = std::env::var("ELEMENTS_DATE")
        .ok()
        .and_then(|d| {
            let mut part = d.split('-');
            let year: i32 = part.next()?.trim().parse().ok()?;
            let month: u8 = part.next()?.trim().parse().ok()?;
            let day: u8 = part.next()?.trim().parse().ok()?;
            Some(CivilDate::new(year, month, day))
        })
        .unwrap_or(CivilDate::new(2026, 6, 21));
    let clock = WorldClock::new(date).at_hour(fixed.unwrap_or(4.0));
    let clock = match fixed {
        Some(_) => WorldClock {
            paused: true,
            ..clock
        },
        None => clock.with_time_scale(720.0),
    };
    let mut director = WeatherDirector::default();
    director.force(WeatherPreset::Clear, std::time::Duration::ZERO);
    App::new()
        .add_plugins((DefaultPlugins, ElementsPlugins, ExampleHarness))
        .insert_resource(clock)
        .insert_resource(director)
        .insert_resource(GeoLocation {
            latitude_deg: 45.0,
            ..default()
        })
        .add_systems(Startup, setup)
        .add_systems(PostUpdate, aim_at_sky)
        .run();
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    // Spawned west-south-west — the heading the aim returns to when no body is up (stars over
    // the evening ground); `aim_at_sky` turns it toward whatever is up after this.
    commands.spawn(camera(
        Vec3::new(0.0, 8.0, 0.0),
        Vec3::new(-200.0, 30.0, 60.0),
    ));
    test_scene(&mut commands, &mut meshes, &mut materials);
}

/// Aim the camera at what the showcase is for: the sun while it is up, the moon when the sun is
/// down and the moon is up, otherwise the evening heading it was spawned with. While the clock
/// runs the aim follows the bodies across the day; paused (`ELEMENTS_HOUR`) it aims once, so a
/// still is framed on the disc and the view stays free to drag. Holding the right button always
/// releases the aim, which resumes when it is let go (once per pause state).
fn aim_at_sky(
    sun: Res<SunState>,
    clock: Res<WorldClock>,
    mouse: Res<ButtonInput<MouseButton>>,
    mut cameras: Query<(&mut Transform, &mut FlyCamera)>,
    mut framed: Local<bool>,
) {
    if std::env::var("DAYNIGHT_DEBUG").as_deref() == Ok("1") {
        eprintln!(
            "AIMENTRY paused={} mouse={} sun {:.2} moon {:.2}",
            clock.paused,
            mouse.pressed(MouseButton::Right),
            sun.altitude_deg,
            sun.moon_altitude_deg
        );
    }
    if mouse.pressed(MouseButton::Right) {
        *framed = false;
        return;
    }
    if clock.paused && *framed {
        return;
    }
    let dir = if sun.altitude_deg > 0.5 {
        Some(sun.direction)
    } else if sun.moon_altitude_deg > 0.5 {
        Some(sun.moon_direction)
    } else {
        None
    };
    if std::env::var("DAYNIGHT_DEBUG").as_deref() == Ok("1") {
        eprintln!(
            "AIM sun {:.2} moon {:.2} dir {:?} branch {}",
            sun.altitude_deg,
            sun.moon_altitude_deg,
            sun.moon_direction,
            dir.is_some()
        );
    }
    let Some(dir) = dir else {
        *framed = clock.paused;
        return;
    };
    let yaw = (-dir.x).atan2(-dir.z);
    let pitch = dir.y.clamp(-1.0, 1.0).asin();
    for (mut transform, mut fly) in &mut cameras {
        fly.face(yaw, pitch);
        transform.rotation = Quat::from_euler(EulerRot::YXZ, yaw, pitch, 0.0);
    }
    *framed = clock.paused;
}
