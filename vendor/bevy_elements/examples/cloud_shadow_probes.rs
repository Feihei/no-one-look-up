//! Gameplay probes of the cloud shadow: a grid of unlit markers on the ground, each coloured by
//! `Environment::cloud_shadow_at` at its foot — yellow in the open, blue under a cloud's shadow —
//! so the query can be checked against the shadows the ground shows. The status line adds the
//! illuminance `Environment::illuminance_at` gives at the grid's centre.
//!
//! `cargo run --release --example cloud_shadow_probes`
//!
//! `ELEMENTS_PRESET=Clear` holds another preset (default `PartlyCloudy`), `ELEMENTS_HOUR=16`
//! pauses the clock at that local hour (default 11, running), `ELEMENTS_HEIGHT=6` puts the camera
//! that many metres up (default 120), `ELEMENTS_PITCH=-3` looks that many degrees below or above
//! the horizon (default −25), `ELEMENTS_HEADING=290` along that compass bearing (default 135),
//! `ELEMENTS_SPACING=150` spaces the markers that many metres apart (default a third of the
//! height, at least 40 m, so a high view spans the kilometres a cumulus shadow does).

mod common;

use std::time::Duration;

use bevy::prelude::*;
use bevy_elements::*;
use common::{ExampleHarness, camera, test_scene};

/// Markers per side of the grid.
const PROBES: i32 = 15;
/// Colour of an open probe and of a shadowed one.
const OPEN: Vec3 = Vec3::new(1.0, 0.85, 0.1);
const SHADOWED: Vec3 = Vec3::new(0.1, 0.3, 1.0);

#[derive(Component)]
struct Probe {
    foot: Vec3,
}

#[derive(Resource)]
struct Centre(Vec3);

#[derive(Component)]
struct Readout;

fn env_f32(name: &str) -> Option<f32> {
    std::env::var(name).ok().and_then(|v| v.parse().ok())
}

fn main() {
    let preset = std::env::var("ELEMENTS_PRESET")
        .ok()
        .and_then(|name| {
            WeatherPreset::ALL
                .into_iter()
                .find(|p| format!("{p:?}").eq_ignore_ascii_case(&name))
        })
        .unwrap_or(WeatherPreset::PartlyCloudy);
    let hour = env_f32("ELEMENTS_HOUR");
    let clock = WorldClock {
        paused: hour.is_some(),
        ..WorldClock::new(CivilDate::new(2026, 6, 21))
            .at_hour(f64::from(hour.unwrap_or(11.0)))
            .with_time_scale(1.0)
    };
    App::new()
        .add_plugins((DefaultPlugins, ElementsPlugins, ExampleHarness))
        .insert_resource(clock)
        .add_systems(
            Startup,
            move |mut commands: Commands,
                  meshes: ResMut<Assets<Mesh>>,
                  materials: ResMut<Assets<StandardMaterial>>,
                  mut director: ResMut<WeatherDirector>| {
                director.force(preset, Duration::ZERO);
                setup(&mut commands, meshes, materials);
            },
        )
        .add_systems(Update, colour_probes.after(ElementsSystems::Readback))
        .run();
}

fn setup(
    commands: &mut Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let height = env_f32("ELEMENTS_HEIGHT").unwrap_or(120.0);
    let pitch = env_f32("ELEMENTS_PITCH").unwrap_or(-25.0).to_radians();
    // A compass bearing: 0 north (−Z), 90 east (+X).
    let heading = env_f32("ELEMENTS_HEADING").unwrap_or(135.0).to_radians();
    let flat = Vec2::new(heading.sin(), -heading.cos());
    let forward = Vec3::new(flat.x * pitch.cos(), pitch.sin(), flat.y * pitch.cos());
    let position = Vec3::new(0.0, height, 0.0);
    commands.spawn(camera(position, position + forward * 120.0));
    test_scene(commands, &mut meshes, &mut materials);

    // The grid is centred where the view meets the ground, or 300 m ahead for a level view.
    let ahead = if pitch < -0.02 {
        height / -pitch.sin() * pitch.cos()
    } else {
        300.0
    };
    let centre = Vec3::new(flat.x * ahead, 0.0, flat.y * ahead);
    commands.insert_resource(Centre(centre));
    let spacing = env_f32("ELEMENTS_SPACING").unwrap_or((height / 3.0).max(40.0));
    let radius = (spacing / 15.0).clamp(0.6, 30.0);
    let marker = meshes.add(Sphere::new(radius));
    let half = (PROBES - 1) as f32 * 0.5;
    for i in 0..PROBES {
        for j in 0..PROBES {
            let foot = centre
                + Vec3::new(
                    (i as f32 - half) * spacing,
                    0.0,
                    (j as f32 - half) * spacing,
                );
            commands.spawn((
                Probe { foot },
                Mesh3d(marker.clone()),
                MeshMaterial3d(materials.add(StandardMaterial {
                    base_color: Color::srgb(0.5, 0.5, 0.5),
                    unlit: true,
                    ..default()
                })),
                Transform::from_translation(foot + Vec3::Y * radius),
            ));
        }
    }
    commands.spawn((
        Readout,
        Text::new(""),
        TextFont {
            font_size: FontSize::Px(28.0),
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
}

/// Colour every probe from the query at its foot: grey until the first readback. The readout is
/// also logged every 120 frames, for unattended runs.
fn colour_probes(
    env: Environment,
    centre: Option<Res<Centre>>,
    probes: Query<(&Probe, &MeshMaterial3d<StandardMaterial>)>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut readout: Query<&mut Text, With<Readout>>,
    mut frames: Local<u32>,
) {
    let mut shadowed = 0;
    for (probe, material) in &probes {
        let Some(t) = env.cloud_shadow_at(probe.foot) else {
            return;
        };
        shadowed += usize::from(t < 0.5);
        if let Some(mut material) = materials.get_mut(&material.0) {
            let c = SHADOWED.lerp(OPEN, t);
            material.base_color = Color::srgb(c.x, c.y, c.z);
        }
    }
    if let (Some(centre), Ok(mut text)) = (centre, readout.single_mut()) {
        text.0 = format!(
            "cloud shadow at centre {:.2}  |  {:.0} lx  |  {shadowed} of {} probes shadowed",
            env.cloud_shadow_at(centre.0).unwrap_or(1.0),
            env.illuminance_at(centre.0),
            PROBES * PROBES,
        );
        *frames += 1;
        if frames.is_multiple_of(120) {
            info!("{}", text.0);
        }
    }
}
