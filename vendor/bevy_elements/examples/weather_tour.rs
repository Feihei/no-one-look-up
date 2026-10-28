//! Every weather preset in turn, forced for 20 s each, with the preset named on screen.
//!
//! `cargo run --release --example weather_tour`
//!
//! `ELEMENTS_PRESET=Fog` holds one preset instead (for screenshots). `ELEMENTS_HOUR=19.5` pauses
//! the clock at that local hour, `ELEMENTS_PITCH=15` looks that many degrees above the horizon,
//! `ELEMENTS_HEADING=290` looks along that compass bearing.
//!
//! A frame sequence of the whole tour (the README's `docs/tour.gif`): `ELEMENTS_SHOT_DIR=/tmp/tour
//! ELEMENTS_FRAMES=12600 ELEMENTS_SHOT_EVERY=48 ELEMENTS_SHOT_FROM=36` — 210 s of game time, one
//! frame every 0.8 s, the first skipped past the shader compile. Do **not** set `ELEMENTS_HOUR`
//! for it: that pauses the clock, and with the tour's timer on `Time` the forced preset then never
//! advances (every captured frame is the first preset). Assemble with
//! `magick -delay 13 -loop 0 /tmp/tour/frame*.png -resize 512x -colors 200 -layers Optimize
//! docs/tour.gif`.

mod common;

use std::time::Duration;

use bevy::prelude::*;
use bevy_elements::*;
use common::{ExampleHarness, camera, test_scene};

#[derive(Resource)]
struct Tour {
    held: Option<WeatherPreset>,
    index: usize,
    timer: Timer,
}

#[derive(Component)]
struct PresetLabel;

fn main() {
    let held = std::env::var("ELEMENTS_PRESET").ok().and_then(|name| {
        WeatherPreset::ALL
            .into_iter()
            .find(|p| format!("{p:?}").eq_ignore_ascii_case(&name))
    });
    let hour: Option<f64> = std::env::var("ELEMENTS_HOUR")
        .ok()
        .and_then(|h| h.parse().ok());
    let clock = WorldClock::new(CivilDate::new(2026, 4, 20)).at_hour(hour.unwrap_or(10.0));
    let clock = WorldClock {
        paused: hour.is_some(),
        ..clock.with_time_scale(1.0)
    };
    App::new()
        .add_plugins((DefaultPlugins, ElementsPlugins, ExampleHarness))
        .insert_resource(clock)
        .insert_resource(Tour {
            held,
            index: 0,
            timer: Timer::from_seconds(20.0, TimerMode::Repeating),
        })
        .add_systems(Startup, setup)
        .add_systems(Update, tour.before(ElementsSystems::Clock))
        .run();
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut tour: ResMut<Tour>,
    mut director: ResMut<WeatherDirector>,
) {
    let pitch = std::env::var("ELEMENTS_PITCH")
        .ok()
        .and_then(|p| p.parse::<f32>().ok())
        .unwrap_or(3.5)
        .to_radians();
    // `ELEMENTS_HEADING` is a compass bearing in degrees: 0 north (−Z), 90 east (+X).
    let heading = std::env::var("ELEMENTS_HEADING")
        .ok()
        .and_then(|h| h.parse::<f32>().ok())
        .map_or(Vec2::new(120.0, 60.0).normalize(), |deg| {
            let a = deg.to_radians();
            Vec2::new(a.sin(), -a.cos())
        });
    let position = Vec3::new(0.0, 6.0, 0.0);
    commands.spawn(camera(
        position,
        position
            + Vec3::new(
                heading.x * pitch.cos(),
                pitch.sin(),
                heading.y * pitch.cos(),
            ) * 120.0,
    ));
    test_scene(&mut commands, &mut meshes, &mut materials);
    commands.spawn((
        PresetLabel,
        Text::new(""),
        TextFont {
            font_size: FontSize::Px(36.0),
            ..default()
        },
        BackgroundColor(Color::srgba(0.0, 0.0, 0.0, 0.55)),
        Node {
            position_type: PositionType::Absolute,
            right: Val::Px(20.0),
            bottom: Val::Px(20.0),
            ..default()
        },
    ));
    let first = tour.held.unwrap_or(WeatherPreset::ALL[0]);
    tour.index = 0;
    director.force(first, Duration::ZERO);
}

fn tour(
    time: Res<Time>,
    mut tour: ResMut<Tour>,
    mut director: ResMut<WeatherDirector>,
    mut label: Query<&mut Text, With<PresetLabel>>,
) {
    if tour.held.is_none() && tour.timer.tick(time.delta()).just_finished() {
        tour.index = (tour.index + 1) % WeatherPreset::ALL.len();
        // Transitions are game time; at time scale 1 this is 4 s of blending.
        director.force(WeatherPreset::ALL[tour.index], Duration::from_secs(4));
    }
    if let Ok(mut text) = label.single_mut() {
        text.0 = format!("{:?}", director.current());
    }
}
