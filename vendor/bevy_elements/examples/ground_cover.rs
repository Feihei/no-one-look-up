//! Snow and wet ground on the quickstart hills: the cover maps accumulate, shade and (for snow)
//! drift and melt on the terrain.
//!
//! `cargo run --release --example ground_cover`
//!
//! `ELEMENTS_COVER=snow` (default) forces Snow at ten game minutes per real second, so the hills
//! are white within a minute, and after `ELEMENTS_SNOW_FRAMES` (default 300) frames the sky
//! clears so the snow can be seen without the fall's fog; `ELEMENTS_PRESET=Blizzard` picks the heavier, windier preset
//! whose drifts pile up in the lee of the ridges. `ELEMENTS_COVER=rain` bakes a few millimetres
//! of rain into the hollows (`ErodeTerrain`, erosion off, so the hills stay as they are) while
//! the Rain preset wets the ground: darker, glossier ground, and puddles with ripples where water
//! stands deeper than 2 mm. `ELEMENTS_QUALITY=ultra` runs the bake four times faster per frame.
//! `ELEMENTS_TERRAIN=mountains` swaps the hills for steep mountains, where the slopes shed their
//! snow and the wind piles it up. `ELEMENTS_AGL` sets the camera's height above the ground (default 6 m). `ELEMENTS_FRAMES` and
//! `ELEMENTS_SHOT` work as in the other examples.

mod common;

use std::time::Duration;

use bevy::prelude::*;
use bevy_elements::*;
use common::{ExampleHarness, camera, stand_on_ground};

fn main() {
    let rain = std::env::var("ELEMENTS_COVER").as_deref() == Ok("rain");
    let quality = match std::env::var("ELEMENTS_QUALITY").as_deref() {
        Ok("low") => ElementsQuality::Low,
        Ok("medium") => ElementsQuality::Medium,
        Ok("ultra") => ElementsQuality::Ultra,
        _ => ElementsQuality::High,
    };
    let clock = if rain {
        WorldClock::new(CivilDate::new(2026, 6, 21))
            .at_hour(13.0)
            .with_time_scale(120.0)
    } else {
        WorldClock::new(CivilDate::new(2026, 1, 15))
            .at_hour(7.5)
            .with_time_scale(600.0)
    };
    App::new()
        .add_plugins((DefaultPlugins, ElementsPlugins, ExampleHarness))
        .insert_resource(clock)
        .insert_resource(Climate::temperate())
        .insert_resource(quality)
        .insert_resource(Mode { rain })
        .insert_resource(ErosionSettings {
            enabled: false,
            ..default()
        })
        .add_systems(Startup, setup)
        .add_systems(
            Update,
            (clear_after_snow, report).after(ElementsSystems::Readback),
        )
        .run();
}

/// The snow stops after `ELEMENTS_SNOW_FRAMES` frames (default 300) and the sky clears, so the
/// snow can be seen without the fall's fog; a forced Blizzard or the rain mode keeps its weather.
fn clear_after_snow(mode: Res<Mode>, mut director: ResMut<WeatherDirector>, mut frame: Local<u32>) {
    *frame += 1;
    let stop = env_f32("ELEMENTS_SNOW_FRAMES", 300.0) as u32;
    if !mode.rain && *frame == stop {
        director.force(WeatherPreset::Clear, Duration::ZERO);
    }
}

/// Once a second: the snow depth and wetness under the camera, read from the maps.
fn report(
    env: Environment,
    cameras: Query<&Transform, With<ElementsCamera>>,
    mut frame: Local<u32>,
) {
    *frame += 1;
    if !(*frame).is_multiple_of(60) {
        return;
    }
    for transform in &cameras {
        let at = transform.translation.xz();
        println!(
            "under the camera: snow {:.1} mm deep, wetness {:.2}, water {:.1} mm",
            env.snow_depth(at).unwrap_or(f32::NAN) * 1000.0,
            env.wetness(at).unwrap_or(f32::NAN),
            env.water_depth(at).unwrap_or(f32::NAN) * 1000.0,
        );
    }
}

#[derive(Resource)]
struct Mode {
    rain: bool,
}

fn env_f32(name: &str, default: f32) -> f32 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn setup(
    mut commands: Commands,
    mode: Res<Mode>,
    mut director: ResMut<WeatherDirector>,
    mut baking: MessageWriter<ErodeTerrain>,
) {
    let mountains = std::env::var("ELEMENTS_TERRAIN").as_deref() == Ok("mountains");
    commands.spawn(Terrain::procedural(if mountains {
        ProceduralTerrain::mountains(7)
    } else {
        ProceduralTerrain::hills(42)
    }));
    commands.spawn((
        camera(Vec3::new(0.0, 160.0, 0.0), Vec3::new(150.0, 100.0, 80.0)),
        stand_on_ground(env_f32("ELEMENTS_AGL", 6.0)),
    ));
    let default = if mode.rain {
        WeatherPreset::Rain
    } else {
        WeatherPreset::Snow
    };
    let preset = std::env::var("ELEMENTS_PRESET")
        .ok()
        .and_then(|name| {
            WeatherPreset::ALL
                .into_iter()
                .find(|p| format!("{p:?}").eq_ignore_ascii_case(&name))
        })
        .unwrap_or(default);
    director.force(preset, Duration::ZERO);
    if mode.rain {
        baking.write(ErodeTerrain {
            duration_s: env_f32("ELEMENTS_BAKE_S", 240.0),
            rain_mm_h: env_f32("ELEMENTS_BAKE_MM_H", 120.0),
        });
    }
}
