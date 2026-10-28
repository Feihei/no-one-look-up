//! A flood in a V-shaped valley: a `WaterSource` of 50 m³/s at the head, rain, and a marker 100 m
//! downstream whose water depth is printed every second from `Environment::water_depth`.
//!
//! `cargo run --release --example flood`
//!
//! The valley runs along +X (1 % bed slope, side walls 1:5). The depth reads `None` until the
//! first water readback, `0.000` while the front is on its way, and settles near 1.4 m. Keys: `G`
//! doubles the simulation speed (`HydrologySettings::time_scale`), `H` halves it. For unattended
//! runs set `ELEMENTS_FRAMES` (and `ELEMENTS_SHOT`) as in the other examples; the flood reaches
//! the marker after about 36 s (2200 frames). `ELEMENTS_PRESET=Clear` replaces the forced rain.
//!
//! The water is drawn by `WaterPlugin` (`docs/research/water-rendering.md`): absorbing and tinted
//! over the bed, ripples advected along the flow, and foam where the flow is faster than 1.5 m/s
//! (the whole flood channel here) and at the waterline.

mod common;

use std::time::Duration;

use bevy::asset::RenderAssetUsages;
use bevy::prelude::*;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};
use bevy_elements::*;
use common::{ExampleHarness, camera};

/// Cells per axis and cell edge of the valley's height map, m.
const CELLS: u32 = 256;
const CELL_M: f32 = 2.0;
/// Bed slope along the valley, and the walls' rise per metre across it.
const BED_SLOPE: f32 = 0.01;
const WALL_SLOPE: f32 = 0.2;
/// Height range the normalised map covers, m.
const RANGE_M: f32 = 60.0;
/// The source's and the marker's world X; both lie on the valley floor (z = 0).
const SOURCE_X: f32 = -200.0;
const MARKER_X: f32 = -100.0;

fn main() {
    App::new()
        .add_plugins((DefaultPlugins, ElementsPlugins, ExampleHarness))
        .insert_resource(WorldClock {
            paused: true,
            ..WorldClock::new(CivilDate::new(2026, 4, 2)).at_hour(15.0)
        })
        .insert_resource(Climate::temperate())
        // Water only: the river would carve its bare bedrock by up to 8 m in 120 s (the plan's
        // capacity constants), which is the `erosion_bake` example's subject.
        .insert_resource(ErosionSettings {
            enabled: false,
            ..default()
        })
        .add_systems(Startup, setup)
        .add_systems(
            Update,
            (speed_keys, print_depth)
                .chain()
                .after(ElementsSystems::Readback),
        )
        .run();
}

/// The valley bed height at world (x, z), m.
fn bed_m(x: f32, z: f32) -> f32 {
    BED_SLOPE * (CELLS as f32 * CELL_M * 0.5 - x) + WALL_SLOPE * z.abs()
}

fn valley_map() -> Image {
    let mut data = Vec::with_capacity((CELLS * CELLS) as usize * 4);
    for j in 0..CELLS {
        for i in 0..CELLS {
            let x = (i as f32 + 0.5 - CELLS as f32 * 0.5) * CELL_M;
            let z = (j as f32 + 0.5 - CELLS as f32 * 0.5) * CELL_M;
            data.extend_from_slice(&(bed_m(x, z) / RANGE_M).to_le_bytes());
        }
    }
    Image::new(
        Extent3d {
            width: CELLS,
            height: CELLS,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        data,
        TextureFormat::R32Float,
        RenderAssetUsages::default(),
    )
}

#[derive(Component)]
struct Marker;

fn setup(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut director: ResMut<WeatherDirector>,
) {
    let map = images.add(valley_map());
    commands.spawn(
        Terrain::from_heightmap(map, 0.0..RANGE_M)
            .with_extent(Vec2::splat(CELLS as f32 * CELL_M))
            .with_resolution(UVec2::splat(CELLS)),
    );
    commands.spawn((
        WaterSource {
            rate_m3_s: 50.0,
            radius_m: 5.0,
        },
        Transform::from_xyz(SOURCE_X, bed_m(SOURCE_X, 0.0), 0.0),
    ));
    commands.spawn((
        Marker,
        Mesh3d(meshes.add(Cylinder::new(0.6, 6.0))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.9, 0.15, 0.1),
            ..default()
        })),
        Transform::from_xyz(MARKER_X, bed_m(MARKER_X, 0.0) + 3.0, 0.0),
    ));
    commands.spawn(camera(
        Vec3::new(-165.0, 14.0, 30.0),
        Vec3::new(-150.0, 0.0, 0.0),
    ));
    // Rain by default; `ELEMENTS_PRESET=Clear` (any preset name) shows the water under another sky.
    let preset = std::env::var("ELEMENTS_PRESET")
        .ok()
        .and_then(|name| {
            WeatherPreset::ALL
                .into_iter()
                .find(|p| format!("{p:?}").eq_ignore_ascii_case(&name))
        })
        .unwrap_or(WeatherPreset::Rain);
    director.force(preset, Duration::ZERO);
}

fn speed_keys(keys: Res<ButtonInput<KeyCode>>, mut settings: ResMut<HydrologySettings>) {
    if keys.just_pressed(KeyCode::KeyG) {
        settings.time_scale *= 2.0;
    }
    if keys.just_pressed(KeyCode::KeyH) {
        settings.time_scale *= 0.5;
    }
}

/// Print the depth at the marker once per second of real time.
fn print_depth(
    env: Environment,
    stats: Res<HydrologyStats>,
    time: Res<Time<Real>>,
    mut last: Local<u64>,
) {
    let second = time.elapsed_secs() as u64;
    if second == *last && time.elapsed_secs() > 0.1 {
        return;
    }
    *last = second;
    let depth = env.water_depth(Vec2::new(MARKER_X, 0.0));
    let speed = env
        .water_velocity(Vec2::new(MARKER_X, 0.0))
        .map(Vec2::length);
    println!(
        "t = {:>4} s (simulated {:>6.1} s)  water_depth(marker) = {}  speed = {}  lost {:.0} m³",
        second,
        stats.simulated_s,
        depth.map_or("None".to_string(), |d| format!("Some({d:.3})")),
        speed.map_or("None".to_string(), |v| format!("{v:.2} m/s")),
        stats.lost_m3,
    );
}
