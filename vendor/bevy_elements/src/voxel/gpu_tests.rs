//! GPU test of the voxel ground: the snow maps run on the terrain a `VoxelGround` volume owns,
//! the volume's field swaps in the snow as it falls, the tunnels stay clear, and erosion cannot
//! move the heights the mesh was made from. See `docs/research/voxel-snow.md` (Validation 8).
//!
//! Needs a GPU: `cargo test --release --features isomesh --lib -- --ignored --test-threads=1
//! voxel::`. Each test builds a renderer, so they take a lock and run one at a time.

#[allow(clippy::duplicate_mod)]
#[path = "../../tests/support/mod.rs"]
mod support;

use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;

use bevy::app::AppExit;
use bevy::prelude::*;
use bevy::render::gpu_readback::{Readback, ReadbackComplete};
use bevy::time::TimeUpdateStrategy;
use bevy_isomesh::{
    ChunkMesh, IsomeshPlugin, MeshBudget, NeedsRemesh, VolumeField, VoxelChunk, VoxelVolume,
};
use isomesh::Sdf;
use isomesh::brush::smooth_min;
use isomesh::chunk::{ChunkId, ChunkLayout};

use super::VoxelGround;
use super::systems::VoxelGroundState;
use crate::cover::{CoverGpu, CoverStats, SnowSettings};
use crate::hydrology::{HydrologyBoundary, HydrologyGpu, HydrologySettings};
use crate::terrain::TerrainState;
use crate::{
    CivilDate, Climate, ElementsCamera, ElementsPlugins, ElementsQuality, ElementsSystems,
    EnvironmentSnapshot, ErodeTerrain, GeoLocation, TerrainBaked, Weather, WorldClock,
};

static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());

fn serial() -> MutexGuard<'static, ()> {
    ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner())
}

const DT: f64 = 1.0 / 60.0;
const CELL: f32 = 0.5;

/// The example's hill without its island flank and its portals: a dome with the gyroid's
/// labyrinth carved out under a crust, and two straight tunnels along x under a roof of at
/// least 0.8 m (`examples/voxel_snow.rs`). Snow is the same depth on every column but a steep or
/// a shaft one: no ground is steeper than 31°, under the 35° where snow starts to shed, and
/// nothing is open to the sky below the top. A column that sheds, or that sits at the foot of a
/// cliff, accumulates at its own rate, so its depth crosses the swap threshold at its own time
/// and the field swaps at every readback until it is done.
#[derive(Clone, Copy)]
pub(super) struct HillWorld;

impl Sdf for HillWorld {
    type Scalar = f32;

    fn sample(&self, p: [f32; 3]) -> f32 {
        let [x, y, z] = p;
        let r2 = x * x + z * z;
        let dome = 10.0 * (-r2 / (2.0 * 14.0 * 14.0)).exp();
        let ripple = 0.5 * (x * 0.23 + 1.7).sin() + 0.4 * (z * 0.27 - 0.6).cos();
        let h = dome + ripple;
        let ground = y - h;
        const K: f32 = 0.5;
        let g = (K * x).sin() * (K * y).cos()
            + (K * y).sin() * (K * z).cos()
            + (K * z).sin() * (K * x).cos();
        let labyrinth = (0.35 - g).max(y - (h - 2.5));
        let tunnel = |zc: f32| ((y + 3.0).powi(2) + (z - zc).powi(2)).sqrt() - 2.0;
        let air = labyrinth.min(tunnel(-6.0)).min(tunnel(7.0));
        -smooth_min(-ground, air, 0.7)
    }
}

/// What the test pins the weather to after the climate has run, each frame.
#[derive(Resource, Clone, Copy)]
struct Pin {
    mm_h: f32,
    temperature_c: f32,
}

fn pin_weather(pin: Res<Pin>, mut weather: ResMut<Weather>) {
    weather.precipitation_mm_h = pin.mm_h;
    weather.temperature_c = pin.temperature_c;
    weather.wind_speed_ms = 0.0;
    weather.wind_direction = Vec2::X;
    weather.relative_humidity = 0.5;
}

fn setup(mut commands: Commands, mut images: ResMut<Assets<Image>>) {
    commands.spawn((
        Camera3d::default(),
        support::image_target(&mut images),
        ElementsCamera::default(),
        Transform::from_xyz(0.0, 40.0, 0.0),
    ));
    let layout = ChunkLayout::new(16, CELL, [-32.0, -8.0, -32.0]).expect("a valid layout");
    let volume = commands
        .spawn((
            VoxelVolume::new(layout, HillWorld),
            VoxelGround::new(UVec2::splat(128)),
        ))
        .id();
    for x in 0..8 {
        for y in 0..3 {
            for z in 0..8 {
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
}

/// Update until `done` holds (at most 240 s of wall time).
fn run_until(app: &mut App, what: &str, mut done: impl FnMut(&mut App) -> bool) {
    let deadline = std::time::Instant::now() + Duration::from_secs(240);
    while std::time::Instant::now() < deadline {
        app.update();
        if done(app) {
            return;
        }
    }
    panic!("timed out waiting for {what}");
}

fn volume_entity(app: &mut App) -> Entity {
    app.world_mut()
        .query_filtered::<Entity, With<VoxelGround>>()
        .single(app.world())
        .expect("one voxel volume")
}

fn read_f32(
    app: &mut App,
    handle: &Handle<bevy::render::storage::ShaderBuffer>,
    n: usize,
) -> Vec<f32> {
    let slot: Arc<Mutex<Option<Vec<u8>>>> = Arc::default();
    let sink = slot.clone();
    let entity = app
        .world_mut()
        .spawn(Readback::buffer(handle.clone()))
        .observe(move |done: On<ReadbackComplete>| {
            let mut sink = sink.lock().expect("lock");
            if sink.is_none() {
                *sink = Some(done.data.to_vec());
            }
        })
        .id();
    run_until(app, "a buffer readback", |_| {
        slot.lock().expect("lock").is_some()
    });
    app.world_mut().despawn(entity);
    let bytes = slot
        .lock()
        .expect("lock")
        .take()
        .expect("the readback completed");
    bytemuck::pod_collect_to_vec::<u8, f32>(&bytes[..n * 4])
}

/// Validation 8: forced snow for 60 game hours lays 0.572 m on the crest, the field swaps twice,
/// no tunnel sample changes sign, and an erosion bake leaves the heights alone.
#[test]
#[ignore = "needs a GPU"]
fn gpu_voxel_snow_covers_the_top() {
    let _lock = serial();
    let mut app = App::new();
    app.add_plugins((support::headless_plugins(), ElementsPlugins, IsomeshPlugin))
        .insert_resource(TimeUpdateStrategy::ManualDuration(Duration::from_secs_f64(
            DT,
        )))
        .insert_resource(ElementsQuality::High)
        // Water is held: the bake below runs at the quality's full rate regardless.
        .insert_resource(HydrologySettings {
            boundary: HydrologyBoundary::Closed,
            evaporation: false,
            time_scale: 0.0,
            ..default()
        })
        .insert_resource(MeshBudget {
            per_frame: Duration::from_millis(4),
            max_in_flight: 16,
        })
        .insert_resource(SnowSettings {
            drift: false,
            max_rate_hz: 0.0,
            ..default()
        })
        .insert_resource(Climate {
            reference_elevation_m: 0.0,
            ..default()
        })
        .insert_resource(GeoLocation {
            latitude_deg: 85.0,
            ..default()
        })
        .insert_resource(WorldClock::new(CivilDate::new(2026, 12, 21)).with_time_scale(0.0))
        .insert_resource(Pin {
            mm_h: 0.0,
            temperature_c: -5.0,
        })
        .add_systems(Startup, setup)
        .add_systems(
            Update,
            pin_weather
                .after(ElementsSystems::Weather)
                .before(ElementsSystems::Simulation),
        );
    // `ErosionSettings` stays at its default (on): the ground must ignore it.
    support::finish(&mut app);
    run_until(&mut app, "the voxel terrain and the cover kernels", |app| {
        let world = app.world();
        world
            .get_resource::<TerrainState>()
            .is_some_and(|s| s.is_live())
            && world.get_resource::<CoverGpu>().is_some_and(|g| {
                g.live.load(std::sync::atomic::Ordering::Relaxed)
                    && g.shelter_built.load(std::sync::atomic::Ordering::Relaxed) > 0
            })
    });
    let volume = volume_entity(&mut app);
    let terrain = app.world().resource::<TerrainState>().clone();
    let n = (terrain.grid.size.x * terrain.grid.size.y) as usize;
    assert_eq!(terrain.grid.size, UVec2::splat(128));
    assert!(
        app.world().resource::<HydrologyGpu>().thermal.eq(&false),
        "thermal erosion is locked off for voxel ground"
    );

    // The terrain's heights are the scanned tops, to the bit, and an erosion bake (rain on the
    // hill, erosion on by default) leaves them alone.
    let tops: Vec<f32> = app
        .world()
        .get::<VoxelGroundState>(volume)
        .expect("the ground was built")
        .top
        .to_vec();
    app.world_mut().write_message(ErodeTerrain {
        duration_s: 20.0,
        rain_mm_h: 120.0,
    });
    run_until(&mut app, "the erosion bake", |app| {
        let mut baked = app
            .world_mut()
            .resource_mut::<bevy::ecs::message::Messages<TerrainBaked>>();
        baked.drain().next().is_some()
    });
    let bedrock = read_f32(&mut app, &terrain.bedrock, n);
    assert_eq!(bedrock, tops, "erosion moved the bedrock of a voxel ground");
    println!("RESULT bedrock after a 20 s bake at 120 mm/h: identical to the {n} scanned tops");

    // Sixty game hours of 1.5 mm/h snow at 3 600 game seconds per real second.
    app.world_mut().resource_mut::<Pin>().mm_h = 1.5;
    let crest = (64 * 128 + 64) as usize;
    let crest_xz = {
        let grid = terrain.grid;
        grid.origin + Vec2::new(64.5, 64.5) * grid.cell_m
    };
    let start_s = app.world().resource::<CoverStats>().simulated_s;
    app.world_mut().resource_mut::<WorldClock>().time_scale = 3600.0;
    let mut swaps_at: Vec<f64> = Vec::new();
    let (mut frame_ms, mut swap_ms) = (Vec::new(), Vec::new());
    for _ in 0..3600 {
        let began = std::time::Instant::now();
        app.update();
        let ms = began.elapsed().as_secs_f64() * 1000.0;
        frame_ms.push(ms);
        let state = app
            .world()
            .get::<VoxelGroundState>(volume)
            .expect("the ground");
        if state.swaps as usize > swaps_at.len() {
            let hours = (app.world().resource::<CoverStats>().simulated_s - start_s) / 3600.0;
            println!(
                "RESULT swap {} at {hours:.2} game hours, frame {ms:.1} ms",
                state.swaps
            );
            swaps_at.push(hours);
            swap_ms.push(ms);
        }
    }
    frame_ms.sort_by(f64::total_cmp);
    println!(
        "RESULT frame time over 3 600 frames: median {:.2} ms, 99th {:.2} ms, max {:.2} ms; the swap frames {swap_ms:.1?} ms",
        frame_ms[frame_ms.len() / 2],
        frame_ms[frame_ms.len() * 99 / 100],
        frame_ms[frame_ms.len() - 1],
    );
    let hours = (app.world().resource::<CoverStats>().simulated_s - start_s) / 3600.0;
    app.world_mut().resource_mut::<WorldClock>().time_scale = 0.0;
    for _ in 0..40 {
        app.update();
    }
    let depth = app
        .world()
        .resource::<EnvironmentSnapshot>()
        .cover
        .as_ref()
        .and_then(|c| c.depth_at(crest_xz))
        .expect("a cover readback");
    println!("RESULT crest depth after {hours:.2} h: {depth:.4} m (want 0.572 +- 0.03)");
    assert!((depth - 0.572).abs() <= 0.03, "crest depth {depth}");
    let stats = app.world().resource::<CoverStats>().clone();
    assert_eq!(stats.dropped, 0);

    let state = app
        .world()
        .get::<VoxelGroundState>(volume)
        .expect("the ground");
    assert_eq!(state.swaps, 2, "swaps at {swaps_at:?} h");
    assert!(
        (swaps_at[0] - 20.7).abs() <= 2.0,
        "the first swap at {} h",
        swaps_at[0]
    );
    assert!(
        (swaps_at[1] - 50.0).abs() <= 2.0,
        "the second swap at {} h",
        swaps_at[1]
    );

    // The installed field at the crest: the held depth is about half a metre.
    let field = Arc::clone(
        &app.world()
            .get::<VoxelVolume>(volume)
            .expect("the volume")
            .field,
    );
    let t = state.top[crest];
    let held = state.snow[crest];
    println!("RESULT crest top {t:.3} m, held snow {held:.4} m");
    let (x, z) = (crest_xz.x, crest_xz.y);
    assert!(field.sample([x, t + 0.45, z]) < 0.0, "inside the slab");
    assert!(field.sample([x, t + 0.55, z]) > 0.0, "above the snow");

    // The tunnels: at every lattice point at least one cell under the column top, the sign of
    // the field is the base's.
    let base: Arc<dyn VolumeField> = Arc::clone(&state.base);
    let (grid, tops) = (state.columns.grid, Arc::clone(&state.top));
    let (mut checked, mut mismatches, mut in_rock) = (0_u64, 0_u64, 0_u64);
    for ix in 0..=128 {
        for iz in 0..=128 {
            for iy in 0..=48 {
                let q = [
                    -32.0 + CELL * ix as f32,
                    -8.0 + CELL * iy as f32,
                    -32.0 + CELL * iz as f32,
                ];
                if q[1] > grid.sample(&tops, Vec2::new(q[0], q[2])) - CELL {
                    continue;
                }
                checked += 1;
                let (with_snow, bare) = (field.sample(q) < 0.0, base.sample(q) < 0.0);
                in_rock += u64::from(bare);
                mismatches += u64::from(with_snow != bare);
            }
        }
    }
    println!(
        "RESULT tunnel check: {checked} lattice points at least one cell under the top, {in_rock} in rock, {mismatches} sign mismatches"
    );
    assert!(checked >= 100_000);
    assert_eq!(mismatches, 0);

    // The volume was meshed.
    let mut meshed = app.world_mut().query::<&ChunkMesh>();
    assert!(meshed.iter(app.world()).count() > 0);
    app.world_mut().write_message(AppExit::Success);
}
