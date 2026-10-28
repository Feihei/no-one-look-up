//! Bake erosion into mountains: `ErodeTerrain { duration_s: 600.0, rain_mm_h: 20.0 }`, shown
//! before and after.
//!
//! `cargo run --release --example erosion_bake`
//!
//! Press `B` to start the bake; the progress is printed every second and `TerrainBaked` ends it
//! (125 s at 60 fps with the default `High` quality: 4 substeps of 20 ms per frame, 7500 frames).
//! For unattended runs set `ELEMENTS_BAKE=1` to start it at once, `ELEMENTS_FRAMES` to the frame
//! to stop at and `ELEMENTS_SHOT` for the screenshot as in the other examples: a "before" shot is
//! `ELEMENTS_FRAMES=120`, an "after" shot is `ELEMENTS_BAKE=1 ELEMENTS_FRAMES=7700`.
//! `ELEMENTS_BAKE_S` and `ELEMENTS_BAKE_MM_H` override the duration and the rain.

mod common;

use bevy::prelude::*;
use bevy_elements::*;
use common::{ExampleHarness, camera};

fn main() {
    let clock = WorldClock {
        paused: true,
        ..WorldClock::new(CivilDate::new(2026, 6, 21)).at_hour(16.0)
    };
    App::new()
        .add_plugins((DefaultPlugins, ElementsPlugins, ExampleHarness))
        .insert_resource(clock)
        .insert_resource(Climate::temperate())
        .add_systems(Startup, setup)
        .add_systems(Update, (bake_key, report).after(ElementsSystems::Readback))
        .run();
}

fn env_f32(name: &str, default: f32) -> f32 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn bake_request() -> ErodeTerrain {
    ErodeTerrain {
        duration_s: env_f32("ELEMENTS_BAKE_S", 600.0),
        rain_mm_h: env_f32("ELEMENTS_BAKE_MM_H", 20.0),
    }
}

fn setup(mut commands: Commands, mut baking: MessageWriter<ErodeTerrain>) {
    commands.spawn(Terrain::procedural(ProceduralTerrain::mountains(7)));
    commands.spawn(camera(
        Vec3::new(0.0, 900.0, 1500.0),
        Vec3::new(0.0, 250.0, 0.0),
    ));
    if std::env::var("ELEMENTS_BAKE").is_ok() {
        baking.write(bake_request());
    }
}

fn bake_key(keys: Res<ButtonInput<KeyCode>>, mut baking: MessageWriter<ErodeTerrain>) {
    if keys.just_pressed(KeyCode::KeyB) {
        let request = bake_request();
        info!(
            "baking {} s of simulated erosion under {} mm/h",
            request.duration_s, request.rain_mm_h
        );
        baking.write(request);
    }
}

/// Heights sampled on a 48 × 48 grid over the middle of the terrain, for the before and after.
const SURVEY: i32 = 48;
const SURVEY_SPACING_M: f32 = 40.0;

/// The surface heights of the survey grid, `None` until every point has been read back.
fn survey(env: &Environment) -> Option<Vec<f32>> {
    (0..SURVEY * SURVEY)
        .map(|k| {
            let at = Vec2::new(
                (k % SURVEY - SURVEY / 2) as f32,
                (k / SURVEY - SURVEY / 2) as f32,
            ) * SURVEY_SPACING_M;
            env.terrain_height(at)
        })
        .collect()
}

#[derive(Default)]
struct Before {
    heights: Option<Vec<f32>>,
    /// Real time the bake finished at, until its numbers have been printed.
    baked_at: Option<f32>,
}

fn report(
    env: Environment,
    stats: Res<HydrologyStats>,
    mut baked: MessageReader<TerrainBaked>,
    time: Res<Time<Real>>,
    mut before: Local<Before>,
    mut last: Local<u64>,
) {
    let now = time.elapsed_secs();
    // The first survey that lands before the bake has run for five simulated seconds.
    if before.heights.is_none() && stats.simulated_s < 5.0 {
        before.heights = survey(&env);
    }
    for done in baked.read() {
        info!("terrain baked: {} s of erosion done", done.duration_s);
        before.baked_at = Some(now);
    }
    // A second after the bake the height readback has caught up: print what changed.
    if let (Some(at), Some(first)) = (before.baked_at, before.heights.as_ref())
        && now > at + 1.0
        && let Some(after) = survey(&env)
    {
        let mut change: Vec<f32> = after.iter().zip(first).map(|(a, b)| a - b).collect();
        let lowered = change.iter().copied().fold(0.0, f32::min);
        let raised = change.iter().copied().fold(0.0, f32::max);
        change.iter_mut().for_each(|c| *c = c.abs());
        change.sort_by(f32::total_cmp);
        println!(
            "surface change over {} points: mean |dz| {:.2} m, p99 {:.2} m, lowered at most {:.2} m, raised at most {:.2} m",
            change.len(),
            change.iter().sum::<f32>() / change.len() as f32,
            change[change.len() * 99 / 100],
            lowered.abs(),
            raised
        );
        before.baked_at = None;
    }
    let second = now as u64;
    if stats.baking && second != *last {
        *last = second;
        println!("baking: {:.0} s simulated", stats.simulated_s);
    }
}
