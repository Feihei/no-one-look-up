//! Snow on a voxel hill. A `bevy_isomesh` volume holds a gyroid-carved hill with two tunnels
//! through it; `VoxelGround` gives it a terrain the snow maps run on, and `SnowyField` grows a
//! cap of snow over the hill as it falls: the mesh is remeshed when the depth moves by half a
//! voxel, and the tunnels stay dark.
//!
//! `cargo run --release --features isomesh --example voxel_snow`
//!
//! Snow is forced at 3 600 game seconds per real second (`ELEMENTS_TIME_SCALE` changes it), so
//! the first half-metre slab is on the hill after 21 s and the second after 50 s. The status line
//! of the harness shows the weather; the console prints the depth at the crest once a second.
//! `ELEMENTS_EYE` / `ELEMENTS_LOOK` (`x,y,z`) place the camera, `ELEMENTS_HOUR` sets the start
//! hour (default 8). With `ELEMENTS_FRAMES` the frames run at a fixed 1/60 s: 1 500 frames are 25
//! s, a day of game time at the default scale.
//!
//! The ground material gates snow and wetness by depth below the column top
//! (`GroundExtension::with_column_gate_m`), so a tunnel under a snowy hill is not painted white.

mod common;

use std::time::Duration;

use bevy::prelude::*;
use bevy::time::TimeUpdateStrategy;
use bevy_elements::*;
use bevy_isomesh::{ChunkMesh, IsomeshPlugin, MeshBudget, NeedsRemesh, VoxelChunk, VoxelVolume};
use common::{ExampleHarness, camera};
use isomesh::Sdf;
use isomesh::brush::smooth_min;
use isomesh::chunk::{ChunkId, ChunkLayout};

/// Cells of a chunk edge.
const CHUNK_CELLS: u32 = 16;
/// Voxel edge, m (a power of two: isomesh chunk seams close only then).
const CELL_SIZE: f32 = 0.5;
/// Chunk columns per side: 8 × 16 × 0.5 m = 64 m.
const CHUNKS_XZ: i32 = 8;
/// Chunk layers: `y` in `[-8, 16)`.
const CHUNKS_Y: i32 = 3;
/// Bottom of the volume, m.
const Y_MIN: f32 = -8.0;
/// Gyroid wavenumber: a period of 12.6 m.
const K: f32 = 0.5;
/// Gyroid level: the cave is where the gyroid is above it.
const LEVEL: f32 = 0.35;
/// How far the rounding of the rock's rims reaches, m (above one voxel).
const BLEND: f32 = 0.7;
/// Rock above the gyroid caves, m: they never break through the surface.
const CRUST: f32 = 2.5;

/// A dome of rock with the gyroid's labyrinth carved out of it under a crust, and two straight
/// tunnels along x that open onto the flanks. The surface falls below the volume at its edge, so
/// the hill stands alone.
#[derive(Clone, Copy)]
struct HillWorld;

impl HillWorld {
    /// The ground height, m.
    fn height(x: f32, z: f32) -> f32 {
        let r2 = x * x + z * z;
        let dome = 10.0 * (-r2 / (2.0 * 14.0 * 14.0)).exp();
        let ripple = 0.5 * (x * 0.23 + 1.7).sin() + 0.4 * (z * 0.27 - 0.6).cos();
        let r = r2.sqrt();
        let t = ((r - 22.0) / 8.0).clamp(0.0, 1.0);
        let edge = t * t * (3.0 - 2.0 * t);
        dome + ripple - 14.0 * edge
    }
}

impl Sdf for HillWorld {
    type Scalar = f32;

    fn sample(&self, p: [f32; 3]) -> f32 {
        let [x, y, z] = p;
        let h = Self::height(x, z);
        let ground = y - h;
        let g = (K * x).sin() * (K * y).cos()
            + (K * y).sin() * (K * z).cos()
            + (K * z).sin() * (K * x).cos();
        // Air where the gyroid is above its level and the point is under the crust.
        let labyrinth = (LEVEL - g).max(y - (h - CRUST));
        // Two tunnels along x, 2 m in radius, whose roofs end where the flank falls to them.
        let tunnel = |zc: f32| ((y - 2.0).powi(2) + (z - zc).powi(2)).sqrt() - 2.0;
        let air = labyrinth.min(tunnel(-6.0)).min(tunnel(7.0));
        // Solid is the ground minus the air: a smooth intersection of `ground` and `-air`.
        -smooth_min(-ground, air, BLEND)
    }
}

/// The material every chunk mesh gets.
#[derive(Resource)]
struct Rock(Handle<GroundMaterial>);

fn env_f32(name: &str, default: f32) -> f32 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// `x,y,z` from an environment variable.
fn vec3_env(name: &str) -> Option<Vec3> {
    let value = std::env::var(name).ok()?;
    let mut parts = value.split(',').map(|p| p.trim().parse::<f32>());
    Some(Vec3::new(
        parts.next()?.ok()?,
        parts.next()?.ok()?,
        parts.next()?.ok()?,
    ))
}

fn main() {
    let hour = f64::from(env_f32("ELEMENTS_HOUR", 8.0));
    let mut app = App::new();
    app.add_plugins((
        DefaultPlugins,
        ElementsPlugins,
        IsomeshPlugin,
        ExampleHarness,
    ))
    .insert_resource(
        WorldClock::new(CivilDate::new(2026, 1, 15))
            .at_hour(hour)
            .with_time_scale(env_f32("ELEMENTS_TIME_SCALE", 3600.0)),
    )
    .insert_resource(Climate::temperate())
    // Uniform accumulation, so the slab is the same depth over the whole crest.
    .insert_resource(SnowSettings {
        drift: false,
        ..default()
    })
    // Meshing is budgeted as in isomesh's `game_showcase`.
    .insert_resource(MeshBudget {
        per_frame: Duration::from_millis(3),
        max_in_flight: 8,
    })
    .add_systems(Startup, setup)
    .add_systems(
        Update,
        (
            attach,
            clear_after_snow,
            report.after(ElementsSystems::Readback),
        ),
    );
    if std::env::var("ELEMENTS_FRAMES").is_ok() {
        app.insert_resource(TimeUpdateStrategy::ManualDuration(Duration::from_secs_f64(
            1.0 / 60.0,
        )));
    }
    app.run();
}

fn setup(
    mut commands: Commands,
    mut grounds: ResMut<Assets<GroundMaterial>>,
    mut director: ResMut<WeatherDirector>,
) {
    let layout = ChunkLayout::new(CHUNK_CELLS, CELL_SIZE, [-32.0, Y_MIN, -32.0])
        .expect("a valid chunk layout");
    // 64 m of 0.5 m voxels: one terrain cell is one voxel wide.
    let volume = commands
        .spawn((
            VoxelVolume::new(layout, HillWorld),
            VoxelGround::new(UVec2::splat(128)),
        ))
        .id();
    for x in 0..CHUNKS_XZ {
        for y in 0..CHUNKS_Y {
            for z in 0..CHUNKS_XZ {
                commands.spawn((
                    VoxelChunk {
                        id: ChunkId::new([x, y, z]),
                        volume,
                    },
                    NeedsRemesh,
                ));
            }
        }
    }
    commands.insert_resource(Rock(grounds.add(GroundMaterial {
        base: StandardMaterial {
            base_color: Color::srgb(0.46, 0.41, 0.36),
            perceptual_roughness: 0.92,
            // Tunnels are lit from one side, and the far wall of one is a back face.
            double_sided: true,
            cull_mode: None,
            ..default()
        },
        // Snow and wetness stop one voxel below the column top: nothing in the tunnels.
        extension: GroundExtension::default().with_column_gate_m(CELL_SIZE),
    })));
    let eye = vec3_env("ELEMENTS_EYE").unwrap_or(Vec3::new(-44.0, 9.0, 6.0));
    let look = vec3_env("ELEMENTS_LOOK").unwrap_or(Vec3::new(-8.0, 3.5, -3.0));
    commands.spawn(camera(eye, look));
    director.force(WeatherPreset::Snow, Duration::ZERO);
}

/// Give every chunk mesh, new or rebuilt, the ground material. A chunk that is all rock or all
/// air meshes to nothing and gets no `Mesh3d`.
fn attach(
    mut commands: Commands,
    rock: Option<Res<Rock>>,
    meshes: Res<Assets<Mesh>>,
    fresh: Query<(Entity, &ChunkMesh), Changed<ChunkMesh>>,
) {
    let Some(rock) = rock else {
        return;
    };
    for (entity, chunk) in &fresh {
        if meshes.get(&chunk.0).is_none_or(|m| m.count_vertices() == 0) {
            commands.entity(entity).try_remove::<Mesh3d>();
            continue;
        }
        commands.entity(entity).try_insert((
            Mesh3d(chunk.0.clone()),
            MeshMaterial3d(rock.0.clone()),
            Transform::default(),
        ));
    }
}

/// After `ELEMENTS_SNOW_FRAMES` frames (default: never) the snow stops, the sky clears and the
/// clock slows to a minute a second, so the cap can be seen under a clear sky.
fn clear_after_snow(
    mut director: ResMut<WeatherDirector>,
    mut clock: ResMut<WorldClock>,
    mut frame: Local<u32>,
) {
    *frame += 1;
    if *frame == env_f32("ELEMENTS_SNOW_FRAMES", f32::MAX) as u32 {
        director.force(WeatherPreset::Clear, Duration::ZERO);
        clock.time_scale = 60.0;
    }
}

/// Once a second: the snow depth at the crest, from the map.
fn report(env: Environment, mut last: Local<u64>, time: Res<Time<Real>>) {
    let second = time.elapsed_secs() as u64;
    if second == *last {
        return;
    }
    *last = second;
    println!(
        "crest: snow {:.0} mm, ground {:.2} m",
        env.snow_depth(Vec2::ZERO).unwrap_or(f32::NAN) * 1000.0,
        env.terrain_height(Vec2::ZERO).unwrap_or(f32::NAN),
    );
}
