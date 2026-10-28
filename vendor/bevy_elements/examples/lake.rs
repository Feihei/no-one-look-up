//! A lake: a `WaterSource` of 60 m³/s fills a round basin in a plain, and the water draws as a
//! lake: blue-green over the depths, transparent and foamy at the shore, the bed seen through the
//! shallows.
//!
//! `cargo run --release --example lake`
//!
//! The basin is 4 m deep and 50 m across; at the example's time scale (`HydrologySettings::
//! time_scale`, 16 simulated seconds per second at `Ultra`) it holds water in a few seconds and
//! is full after about 5 s, when a system closes the source. Keys: `G` doubles the simulation speed, `H` halves it. For unattended
//! runs set `ELEMENTS_FRAMES` (and `ELEMENTS_SHOT`) as in the other examples; 1200 frames show a
//! full lake.

mod common;

use std::time::Duration;

use bevy::asset::RenderAssetUsages;
use bevy::prelude::*;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};
use bevy_elements::*;
use common::{ExampleHarness, camera};

/// Cells per axis and cell edge of the height map, m.
const CELLS: u32 = 256;
const CELL_M: f32 = 1.0;
/// Height range the normalised map covers, m.
const RANGE_M: f32 = 12.0;
/// The plain's height, the basin's radius and depth, m.
const PLAIN_M: f32 = 6.0;
const BASIN_RADIUS_M: f32 = 25.0;
const BASIN_DEPTH_M: f32 = 4.0;

fn main() {
    App::new()
        .add_plugins((DefaultPlugins, ElementsPlugins, ExampleHarness))
        .insert_resource(WorldClock {
            paused: true,
            ..WorldClock::new(CivilDate::new(2026, 6, 21)).at_hour(15.0)
        })
        .insert_resource(Climate::temperate())
        .insert_resource(ElementsQuality::Ultra)
        .insert_resource(HydrologySettings {
            time_scale: 16.0,
            boundary: HydrologyBoundary::Closed,
            evaporation: false,
            ..default()
        })
        .insert_resource(ErosionSettings {
            enabled: false,
            ..default()
        })
        .add_systems(Startup, setup)
        .add_systems(
            Update,
            (speed_keys, close_the_tap).after(ElementsSystems::Readback),
        )
        .run();
}

/// The ground height at world (x, z), m: a plain with a round basin, smooth at the rim.
fn ground_m(x: f32, z: f32) -> f32 {
    let r = (x * x + z * z).sqrt() / BASIN_RADIUS_M;
    // A cosine bowl: flat at the rim and at the centre, the steepest bank at half the radius.
    let bowl = if r < 1.0 {
        0.5 * (1.0 + (std::f32::consts::PI * r).cos())
    } else {
        0.0
    };
    PLAIN_M - BASIN_DEPTH_M * bowl
}

fn basin_map() -> Image {
    let mut data = Vec::with_capacity((CELLS * CELLS) as usize * 4);
    for j in 0..CELLS {
        for i in 0..CELLS {
            let x = (i as f32 + 0.5 - CELLS as f32 * 0.5) * CELL_M;
            let z = (j as f32 + 0.5 - CELLS as f32 * 0.5) * CELL_M;
            data.extend_from_slice(&(ground_m(x, z) / RANGE_M).to_le_bytes());
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

fn setup(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    mut director: ResMut<WeatherDirector>,
) {
    let map = images.add(basin_map());
    commands.spawn(
        Terrain::from_heightmap(map, 0.0..RANGE_M)
            .with_extent(Vec2::splat(CELLS as f32 * CELL_M))
            .with_resolution(UVec2::splat(CELLS)),
    );
    commands.spawn((
        WaterSource {
            rate_m3_s: 60.0,
            radius_m: 3.0,
        },
        Transform::from_xyz(0.0, ground_m(0.0, 0.0), 0.0),
    ));
    commands.spawn(camera(
        Vec3::new(-48.0, 17.0, 38.0),
        Vec3::new(0.0, 4.0, -4.0),
    ));
    director.force(WeatherPreset::Clear, Duration::ZERO);
}

/// Water depth at the basin's centre at which the tap is closed, m: just under the rim.
const FULL_DEPTH_M: f32 = 3.7;

/// Close the source once the lake is full, so it does not spill over the plain.
fn close_the_tap(env: Environment, mut sources: Query<&mut WaterSource>) {
    if env
        .water_depth(Vec2::ZERO)
        .is_some_and(|depth| depth >= FULL_DEPTH_M)
    {
        for mut source in &mut sources {
            if source.rate_m3_s != 0.0 {
                source.rate_m3_s = 0.0;
            }
        }
    }
}

fn speed_keys(keys: Res<ButtonInput<KeyCode>>, mut settings: ResMut<HydrologySettings>) {
    if keys.just_pressed(KeyCode::KeyG) {
        settings.time_scale *= 2.0;
    }
    if keys.just_pressed(KeyCode::KeyH) {
        settings.time_scale *= 0.5;
    }
}
