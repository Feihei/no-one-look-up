//! GPU tests of the terrain kernels that need the crate's internals (the CPU mirrors in
//! `model/mirror.rs`, the layer buffers of `TerrainState`). Tests of the public behaviour across
//! domains are in `tests/`.
//!
//! Needs a GPU: `cargo test --release --lib -- --ignored --test-threads=1 terrain`.

#[allow(clippy::duplicate_mod)]
#[path = "../../tests/support/mod.rs"]
mod support;

use std::sync::{Arc, Mutex};
use std::time::Duration;

use bevy::app::AppExit;
use bevy::ecs::system::RunSystemOnce;
use bevy::prelude::*;
use bevy::render::gpu_readback::{Readback, ReadbackComplete};
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};
use bevy::render::storage::ShaderBuffer;
use bevy::time::TimeUpdateStrategy;

use super::model::mirror::{generate_surface, slope_deg};
use super::{ProceduralTerrain, Terrain, TerrainState};
use crate::{ElementsCamera, ElementsPlugins, ElementsQuality, Environment};

/// One test builds one renderer at a time.
static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());

pub(crate) fn serial() -> std::sync::MutexGuard<'static, ()> {
    ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner())
}

#[derive(Resource, Clone)]
struct Spawn(Terrain);

fn app(terrain: Terrain, quality: ElementsQuality) -> App {
    let mut app = App::new();
    app.add_plugins((support::headless_plugins(), ElementsPlugins))
        .insert_resource(TimeUpdateStrategy::ManualDuration(Duration::from_secs_f64(
            1.0 / 60.0,
        )))
        .insert_resource(quality)
        .insert_resource(Spawn(terrain))
        .add_systems(
            Startup,
            |mut commands: Commands, spawn: Res<Spawn>, mut images: ResMut<Assets<Image>>| {
                commands.spawn((
                    Camera3d::default(),
                    support::image_target(&mut images),
                    ElementsCamera::default(),
                    Transform::from_xyz(0.0, 30.0, 0.0),
                ));
                commands.spawn(spawn.0.clone());
            },
        );
    support::finish(&mut app);
    app
}

fn finish(app: &mut App) {
    app.world_mut().write_message(AppExit::Success);
}

/// Update until `done` holds (at most 60 s of wall time: a readback on a GPU shared with other
/// work can take over 20 s).
fn run_until(app: &mut App, what: &str, mut done: impl FnMut(&mut App) -> bool) {
    let deadline = std::time::Instant::now() + Duration::from_secs(60);
    while std::time::Instant::now() < deadline {
        app.update();
        if done(app) {
            return;
        }
    }
    panic!("timed out waiting for {what}");
}

fn state(app: &App) -> Option<TerrainState> {
    app.world().get_resource::<TerrainState>().cloned()
}

fn live(app: &mut App) -> bool {
    state(app).is_some_and(|s| s.is_live())
}

/// Read a storage buffer back; updates the app until the bytes arrive.
fn read_buffer(app: &mut App, buffer: Handle<ShaderBuffer>) -> Vec<u8> {
    let slot: Arc<Mutex<Option<Vec<u8>>>> = Arc::default();
    let sink = slot.clone();
    let entity = app
        .world_mut()
        .spawn(Readback::buffer(buffer))
        .observe(move |done: On<ReadbackComplete>| {
            let mut sink = sink.lock().expect("lock");
            if sink.is_none() {
                *sink = Some(done.data.clone());
            }
        })
        .id();
    run_until(app, "a buffer readback", |_| {
        slot.lock().expect("lock").is_some()
    });
    app.world_mut().despawn(entity);
    slot.lock()
        .expect("lock")
        .take()
        .expect("the readback completed")
}

fn floats(bytes: &[u8], count: usize) -> Vec<f32> {
    bytemuck::pod_collect_to_vec::<u8, f32>(&bytes[..count * 4])
}

fn height_at(app: &mut App, p: Vec2) -> Option<f32> {
    app.world_mut()
        .run_system_once(move |env: Environment| env.terrain_height(p))
        .expect("the query runs")
}

/// Validation 1 and 5: the GPU generator reproduces the CPU mirror, the packed buffer holds the
/// surface, the sediment and the slope, and `terrain_height` answers from the readback.
#[test]
#[ignore = "needs a GPU"]
fn the_gpu_generator_matches_the_cpu_mirror() {
    let _lock = serial();
    let hills = ProceduralTerrain::hills(42);
    let mut app = app(
        Terrain::procedural(hills).with_resolution(UVec2::splat(256)),
        ElementsQuality::High,
    );
    assert_eq!(
        height_at(&mut app, Vec2::ZERO),
        None,
        "no height before the first readback"
    );
    run_until(&mut app, "the generator", live);
    let state = state(&app).expect("a terrain state");
    assert_eq!(state.grid.size, UVec2::splat(256));
    assert_eq!(state.grid.cell_m, 8.0);
    assert_eq!(state.grid.origin, Vec2::splat(-1024.0));

    let n = 256 * 256;
    let cpu = generate_surface(&hills, UVec2::splat(256), 8.0);
    let cells = bytemuck::pod_collect_to_vec::<u8, [f32; 4]>(
        &read_buffer(&mut app, state.cells.clone())[..n * 16],
    );
    let mut diffs: Vec<f32> = cells
        .iter()
        .zip(&cpu)
        .map(|(c, h)| (c[0] - h).abs())
        .collect();
    diffs.sort_by(f32::total_cmp);
    println!(
        "generator: median diff {:.5} m, p99 {:.5} m, max {:.5} m",
        diffs[n / 2],
        diffs[n * 99 / 100],
        diffs[n - 1]
    );
    assert!(diffs[n / 2] < 0.01 && diffs[n * 99 / 100] < 0.1 && diffs[n - 1] < 1.0);

    // Sediment is zero, water is zero; the bedrock buffer is the surface.
    assert!(cells.iter().all(|c| c[1] == 0.0 && c[2] == 0.0));
    let bedrock = floats(&read_buffer(&mut app, state.bedrock.clone()), n);
    assert!(
        bedrock
            .iter()
            .zip(&cells)
            .all(|(b, c)| (b - c[0]).abs() < 1e-4)
    );

    // The packed gradient reproduces the slope of the stored surface.
    let surface: Vec<f32> = cells.iter().map(|c| c[0]).collect();
    let mut worst = 0.0_f32;
    for j in [0u32, 1, 100, 254, 255] {
        for i in 0..256u32 {
            let w = cells[(j * 256 + i) as usize][3].to_bits();
            let gx = support::half(w as u16);
            let gz = support::half((w >> 16) as u16);
            let angle = (gx * gx + gz * gz).sqrt().atan().to_degrees();
            worst = worst.max((angle - slope_deg(&surface, UVec2::splat(256), 8.0, i, j)).abs());
        }
    }
    println!("packed slope vs CPU slope: worst {worst:.4}°");
    assert!(worst < 0.1, "packed slope off by {worst}°");

    // The height snapshot arrives and agrees with the surface at a cell centre.
    run_until(&mut app, "the height readback", |app| {
        height_at(app, Vec2::ZERO).is_some()
    });
    let at = Vec2::new(-1024.0 + 100.5 * 8.0, -1024.0 + 60.5 * 8.0);
    let got = height_at(&mut app, at).expect("inside the terrain");
    assert!(
        (got - surface[60 * 256 + 100]).abs() < 1e-3,
        "{got} vs {}",
        surface[60 * 256 + 100]
    );
    assert_eq!(height_at(&mut app, Vec2::new(1100.0, 0.0)), None);
    finish(&mut app);
}

/// Validation 6: level ground reads back its height, and the leaf bounds say so.
#[test]
#[ignore = "needs a GPU"]
fn a_flat_terrain_reads_back_its_height() {
    let _lock = serial();
    let mut app = app(
        Terrain::flat(12.5).with_resolution(UVec2::splat(128)),
        ElementsQuality::High,
    );
    assert_eq!(height_at(&mut app, Vec2::ZERO), None);
    run_until(&mut app, "the height readback", |app| {
        height_at(app, Vec2::ZERO).is_some()
    });
    assert_eq!(height_at(&mut app, Vec2::ZERO), Some(12.5));
    assert_eq!(height_at(&mut app, Vec2::new(500.0, -300.0)), Some(12.5));
    assert_eq!(height_at(&mut app, Vec2::new(1030.0, 0.0)), None);
    // Leaf bounds: every one of the 4 × 4 leaves is flat at 12.5.
    let state = state(&app).expect("state");
    let bounds = bytemuck::pod_collect_to_vec::<u8, [f32; 4]>(
        &read_buffer(&mut app, state.leaf_bounds)[..16 * 16],
    );
    assert!(
        bounds
            .iter()
            .all(|b| b[0] == 12.5 && b[1] == 12.5 && b[2] == 12.5 && b[3] == 0.0),
        "{bounds:?}"
    );
    finish(&mut app);
}

/// Validation 6: a 256² `R16Uint` ramp resampled to 128².
#[test]
#[ignore = "needs a GPU"]
fn a_heightmap_is_resampled_by_box_average() {
    let _lock = serial();
    let mut data = Vec::new();
    for _ in 0..256 {
        for x in 0..256u32 {
            data.extend_from_slice(&((x * 257) as u16).to_le_bytes());
        }
    }
    let image = Image::new(
        Extent3d {
            width: 256,
            height: 256,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        data,
        TextureFormat::R16Uint,
        bevy::asset::RenderAssetUsages::default(),
    );
    let mut app = App::new();
    app.add_plugins((support::headless_plugins(), ElementsPlugins))
        .insert_resource(TimeUpdateStrategy::ManualDuration(Duration::from_secs_f64(
            1.0 / 60.0,
        )))
        .insert_resource(ElementsQuality::High)
        .add_systems(
            Startup,
            move |mut commands: Commands, mut images: ResMut<Assets<Image>>| {
                commands.spawn((
                    Camera3d::default(),
                    support::image_target(&mut images),
                    ElementsCamera::default(),
                ));
                let source = images.add(image.clone());
                commands.spawn(
                    Terrain::from_heightmap(source, 0.0..255.0).with_resolution(UVec2::splat(128)),
                );
            },
        );
    support::finish(&mut app);
    run_until(&mut app, "the heightmap import", live);
    let state = state(&app).expect("state");
    let bedrock = floats(&read_buffer(&mut app, state.bedrock.clone()), 128 * 128);
    let mut worst = 0.0_f32;
    for j in 0..128 {
        for i in 0..128 {
            let want = 2.0 * i as f32 + 0.5;
            worst = worst.max((bedrock[j * 128 + i] - want).abs());
            assert_eq!(
                bedrock[j * 128 + i],
                bedrock[i],
                "row {j} differs from row 0"
            );
        }
    }
    println!("heightmap ramp: worst error {worst:.5} m");
    assert!(worst < 0.01, "worst {worst}");
    finish(&mut app);
}

/// Validation 6: an sRGB heightmap is refused and the terrain stays unspawned.
#[test]
#[ignore = "needs a GPU"]
fn an_srgb_heightmap_leaves_the_terrain_unspawned() {
    let _lock = serial();
    let image = Image::new(
        Extent3d {
            width: 4,
            height: 4,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        vec![128u8; 4 * 4 * 4],
        TextureFormat::Rgba8UnormSrgb,
        bevy::asset::RenderAssetUsages::default(),
    );
    let mut a = App::new();
    a.add_plugins((support::headless_plugins(), ElementsPlugins))
        .insert_resource(TimeUpdateStrategy::ManualDuration(Duration::from_secs_f64(
            1.0 / 60.0,
        )))
        .add_systems(
            Startup,
            move |mut commands: Commands, mut images: ResMut<Assets<Image>>| {
                commands.spawn((
                    Camera3d::default(),
                    support::image_target(&mut images),
                    ElementsCamera::default(),
                ));
                let source = images.add(image.clone());
                commands.spawn(Terrain::from_heightmap(source, 0.0..100.0));
            },
        );
    support::finish(&mut a);
    for _ in 0..60 {
        a.update();
    }
    assert!(
        state(&a).is_none(),
        "an sRGB heightmap must not spawn a terrain"
    );
    finish(&mut a);
}

// ---------------------------------------------------------------------------------------------
// Wiring: wind, precipitation and the patch pool over a real terrain.
// ---------------------------------------------------------------------------------------------

use bevy::camera::primitives::Aabb;
use bevy::camera::visibility::NoAutoAabb;

use super::TerrainRuntime;
use crate::plugin::ElementsView;
use crate::query::EnvironmentSnapshot;
use crate::wind::{WIND_CELL_M, WindGpu, WindState};
use crate::{PrecipitationState, Weather, WeatherDirector, WeatherPreset};

/// Mean |divergence| of the wind field (1/s) over cells that are in the air with their whole
/// stencil, 12 m above the ground, and the number of such cells.
fn divergence_over_terrain(app: &mut App, above: f32) -> (f32, usize) {
    app.world_mut()
        .run_system_once(move |env: Environment, wind: Res<WindState>| {
            let h = WIND_CELL_M;
            let (lo, hi) = (
                wind.field_origin + 2.0 * h,
                wind.field_origin + wind.field_extent - 2.0 * h,
            );
            let centre = Vec2::new(
                wind.field_origin.x + 0.5 * wind.field_extent.x,
                wind.field_origin.z + 0.5 * wind.field_extent.z,
            );
            let (mut sum, mut n) = (0.0, 0usize);
            for ix in -20..20 {
                for iz in -20..20 {
                    let xz = centre + Vec2::new(ix as f32, iz as f32) * h;
                    let Some(ground) = env.terrain_height(xz) else {
                        continue;
                    };
                    let p = Vec3::new(xz.x, ground + above, xz.y);
                    // The whole stencil inside the window and clear of the ground.
                    let clear = [Vec2::X, -Vec2::X, Vec2::Y, -Vec2::Y].iter().all(|o| {
                        env.terrain_height(xz + *o * h)
                            .is_some_and(|g| g + 4.0 < p.y - h)
                    });
                    if !clear
                        || p.y + h > hi.y
                        || p.y - h < lo.y
                        || xz.x < lo.x
                        || xz.x > hi.x
                        || xz.y < lo.z
                        || xz.y > hi.z
                    {
                        continue;
                    }
                    let d = (env.wind_at(p + Vec3::X * h).x - env.wind_at(p - Vec3::X * h).x
                        + env.wind_at(p + Vec3::Y * h).y
                        - env.wind_at(p - Vec3::Y * h).y
                        + env.wind_at(p + Vec3::Z * h).z
                        - env.wind_at(p - Vec3::Z * h).z)
                        / (2.0 * h);
                    sum += d.abs();
                    n += 1;
                }
            }
            (if n > 0 { sum / n as f32 } else { f32::NAN }, n)
        })
        .expect("the query system runs")
}

/// Validation of the wind wiring and wind.md Deviation 1: over hills the projection still
/// leaves the field nearly divergence-free, and the Jacobi iteration count of the High grid
/// (12, warm-started) is as good as 30.
#[test]
#[ignore = "needs a GPU"]
fn wind_over_hills_stays_divergence_free_at_the_jacobi_counts() {
    let _lock = serial();
    // One renderer for all three counts: the same field, relaxed again after each change.
    let mut app = app(
        Terrain::procedural(ProceduralTerrain::hills(42)).with_resolution(UVec2::splat(256)),
        ElementsQuality::High,
    );
    app.update();
    run_until(&mut app, "the height readback", |app| {
        height_at(app, Vec2::ZERO).is_some()
    });
    let mut results = Vec::new();
    for iterations in [4u32, 12, 30] {
        app.world_mut().resource_mut::<WindGpu>().iterations = iterations;
        for _ in 0..300 {
            app.update();
        }
        let (mean, cells) = divergence_over_terrain(&mut app, 12.0);
        println!(
            "{iterations} Jacobi iterations over hills: mean |div| {mean:.5} 1/s over {cells} cells"
        );
        results.push((mean, cells));
    }
    finish(&mut app);
    for (mean, cells) in &results {
        assert!(*cells > 200, "too few air cells ({cells})");
        assert!(*mean < 1e-2, "mean |div| {mean}");
    }
    // 12 iterations are as good as 30 (within a factor of two), as on flat ground.
    assert!(results[1].0 < 2.0 * results[2].0 + 1e-4, "{results:?}");
}

/// Validation 10: a flat terrain at 10 m makes every wind cell whose centre is below 10 m solid
/// (bit-exact on the CPU model of the mask) and the window starts at `4·⌊(h − 8)/4⌋`.
#[test]
#[ignore = "needs a GPU"]
fn a_flat_terrain_makes_the_cells_below_it_solid() {
    let _lock = serial();
    let mut app = app(
        Terrain::flat(10.0).with_resolution(UVec2::splat(128)),
        ElementsQuality::High,
    );
    run_until(&mut app, "the height readback", |app| {
        height_at(app, Vec2::ZERO).is_some()
    });
    for _ in 0..60 {
        app.update();
    }
    let (origin, extent, handle, live) = {
        let w = app.world().resource::<WindState>();
        let g = app.world().resource::<WindGpu>();
        (
            w.field_origin,
            w.field_extent,
            g.solid.clone(),
            w.field_live,
        )
    };
    assert!(live);
    assert_eq!(origin.y, 0.0, "window starts at 4·⌊(10 − 8)/4⌋ = 0");
    let n = (extent / WIND_CELL_M).as_ivec3();
    let solid = bytemuck::pod_collect_to_vec::<u8, u32>(
        &read_buffer(&mut app, handle)[..(n.x * n.y * n.z) as usize * 4],
    );
    let origin_cell = (origin / WIND_CELL_M).as_ivec3();
    let mut below = 0;
    for gy in 0..n.y {
        for gz in 0..n.z {
            for gx in 0..n.x {
                let w = origin_cell + IVec3::new(gx, gy, gz);
                let s = IVec3::new(
                    w.x.rem_euclid(n.x),
                    w.y.rem_euclid(n.y),
                    w.z.rem_euclid(n.z),
                );
                let slot = (s.x + n.x * (s.y + n.y * s.z)) as usize;
                let centre_y = (w.y as f32 + 0.5) * WIND_CELL_M;
                let expected = u32::from(centre_y < 10.0);
                below += expected;
                assert_eq!(solid[slot], expected, "cell {w} centred at y = {centre_y}");
            }
        }
    }
    assert_eq!(
        below,
        2 * (n.x * n.z) as u32,
        "the two lowest layers (centres 2 and 6 m) are solid"
    );
    finish(&mut app);
}

/// Pin the weather to steady rain after the climate has run.
fn pin_rain(mut weather: ResMut<Weather>) {
    weather.precipitation_mm_h = 25.0;
    weather.temperature_c = 15.0;
}

/// Positions of the active rain particles at the next readback.
fn rain_heights(app: &mut App) -> Vec<f32> {
    let (particles, rain) = {
        let s = app.world().resource::<PrecipitationState>();
        (s.particles.clone(), s.rain as usize)
    };
    let bytes = read_buffer(app, particles);
    let records = bytemuck::pod_collect_to_vec::<u8, [f32; 8]>(&bytes[..rain * 32]);
    records.iter().map(|r| r[1]).collect()
}

/// Validation 10: rain over a flat terrain at 10 m never goes below it, and some drops reach
/// it; with 1 m of water standing the ground is 11 m.
#[test]
#[ignore = "needs a GPU"]
fn rain_lands_on_the_terrain_and_on_a_flood() {
    let _lock = serial();
    let mut app = app(
        Terrain::flat(10.0).with_resolution(UVec2::splat(128)),
        ElementsQuality::High,
    );
    app.add_systems(
        Update,
        pin_rain
            .after(crate::ElementsSystems::Weather)
            .before(crate::ElementsSystems::Simulation),
    );
    app.world_mut()
        .resource_mut::<WeatherDirector>()
        .force(WeatherPreset::Rain, Duration::ZERO);
    run_until(&mut app, "the generator", live);
    for _ in 0..180 {
        app.update();
    }
    let heights = rain_heights(&mut app);
    let lowest = heights.iter().copied().fold(f32::INFINITY, f32::min);
    println!(
        "{} rain particles over a 10 m plain: lowest y {lowest:.3}",
        heights.len()
    );
    assert!(heights.len() > 1000);
    assert!(
        lowest >= 10.0,
        "a drop fell through the ground to y = {lowest}"
    );
    assert!(
        lowest < 10.6,
        "no drop came near the ground (lowest {lowest})"
    );

    // A 1 m flood: write the water layer and ask for a repack.
    let (water, cells) = {
        let s = state(&app).expect("state");
        (s.water.clone(), (s.grid.size.x * s.grid.size.y) as usize)
    };
    {
        let mut buffers = app.world_mut().resource_mut::<Assets<ShaderBuffer>>();
        let mut buffer = buffers
            .get_mut(&water)
            .expect("the water layer is kept in the main world");
        *buffer = ShaderBuffer::new(
            vec![1.0f32; cells],
            bevy::asset::RenderAssetUsages::default(),
        );
        buffer.label = "elements_terrain_water".into();
        buffer.buffer_usage = bevy::render::render_resource::BufferUsages::STORAGE
            | bevy::render::render_resource::BufferUsages::COPY_SRC
            | bevy::render::render_resource::BufferUsages::COPY_DST;
    }
    app.world_mut().resource_mut::<super::TerrainGpu>().changes += 1;
    for _ in 0..180 {
        app.update();
    }
    let heights = rain_heights(&mut app);
    let lowest = heights.iter().copied().fold(f32::INFINITY, f32::min);
    println!("over a 1 m flood: lowest y {lowest:.3}");
    assert!(
        lowest >= 11.0,
        "a drop fell through the water surface to y = {lowest}"
    );
    assert!(
        lowest < 11.6,
        "no drop came near the water (lowest {lowest})"
    );
    finish(&mut app);
}

/// Validation 8 and the pool: patches are selected after a readback, every pool entity has
/// `NoAutoAabb`, every patch's `Aabb` contains all 1089 vertices the shader would displace, and
/// re-inserting `Mesh3d` leaves the `Aabb` alone.
#[test]
#[ignore = "needs a GPU"]
fn every_patch_bounds_its_displaced_vertices() {
    let _lock = serial();
    let mut app = app(
        Terrain::procedural(ProceduralTerrain::mountains(42)).with_resolution(UVec2::splat(512)),
        ElementsQuality::High,
    );
    run_until(&mut app, "the bounds readback", |app| {
        app.world().resource::<TerrainRuntime>().bounds_version > 0
    });
    for _ in 0..10 {
        app.update();
    }
    let (lod, pool, grid) = {
        let rt = app.world().resource::<TerrainRuntime>();
        (
            rt.lod.expect("a quadtree"),
            rt.pool
                .iter()
                .map(|s| (s.entity, s.patch))
                .collect::<Vec<_>>(),
            state(&app).expect("state").grid,
        )
    };
    let camera = app.world().resource::<ElementsView>().eye;
    let snapshot = app
        .world()
        .resource::<EnvironmentSnapshot>()
        .terrain
        .clone()
        .expect("a height snapshot");
    let local = Vec3::new(
        camera.x - grid.origin.x,
        camera.y - grid.y0,
        camera.z - grid.origin.y,
    );
    let height = |p: Vec2| snapshot.grid.sample(&snapshot.surface, p + grid.origin);
    let mut visible = 0;
    let mut before = Vec::new();
    for (entity, patch) in &pool {
        let e = app.world().entity(*entity);
        assert!(
            e.contains::<NoAutoAabb>(),
            "every pool entity carries NoAutoAabb"
        );
        let Some(patch) = patch else {
            assert_eq!(e.get::<Visibility>(), Some(&Visibility::Hidden));
            continue;
        };
        visible += 1;
        let aabb = *e.get::<Aabb>().expect("an Aabb");
        let (lo, hi) = (aabb.min().y, aabb.max().y);
        for gz in 0..=32 {
            for gx in 0..=32 {
                let v =
                    super::lod::vertex_position(&lod, *patch, UVec2::new(gx, gz), local, height);
                assert!(
                    v.y >= lo - 1e-3 && v.y <= hi + 1e-3,
                    "{patch:?} vertex {gx},{gz} at {} outside [{lo}, {hi}]",
                    v.y
                );
            }
        }
        before.push((*entity, aabb));
    }
    println!("{visible} patches selected over the mountains");
    assert!(visible > 20);
    // Re-inserting the mesh handle must not recompute the bounds.
    for (entity, _) in &before {
        let mesh = app
            .world()
            .entity(*entity)
            .get::<Mesh3d>()
            .expect("a mesh")
            .0
            .clone();
        app.world_mut().entity_mut(*entity).insert(Mesh3d(mesh));
    }
    for _ in 0..3 {
        app.update();
    }
    for (entity, aabb) in before {
        // The selection may have moved on, so compare only patches that still draw the same tile.
        if app
            .world()
            .resource::<TerrainRuntime>()
            .pool
            .iter()
            .any(|s| s.entity == entity && s.patch.is_some())
        {
            let now = *app.world().entity(entity).get::<Aabb>().expect("an Aabb");
            assert!(
                now.half_extents.x == aabb.half_extents.x,
                "the Aabb was recomputed from the flat mesh"
            );
        }
    }
    finish(&mut app);
}

/// Issue #35, acceptance 4 (`docs/research/terrain-look.md`): a `TerrainLook` written every frame
/// is a uniform write. Over 240 frames of a changing grade the render world's pipeline count does
/// not grow, and the terrain's material carries the last grade written.
#[test]
#[ignore = "needs a GPU"]
fn a_look_written_every_frame_compiles_nothing() {
    use bevy::render::render_resource::PipelineCache;
    use bevy::render::{Render, RenderApp};

    use super::{TerrainLook, TerrainMaterial};

    let _lock = serial();
    let counts: Arc<Mutex<Vec<usize>>> = Arc::default();
    let mut app = App::new();
    app.add_plugins((support::headless_plugins(), ElementsPlugins))
        .insert_resource(TimeUpdateStrategy::ManualDuration(Duration::from_secs_f64(
            1.0 / 60.0,
        )))
        .insert_resource(ElementsQuality::Low)
        .add_systems(
            Startup,
            |mut commands: Commands, mut images: ResMut<Assets<Image>>| {
                commands.spawn((
                    Camera3d::default(),
                    support::image_target(&mut images),
                    ElementsCamera::default(),
                    Transform::from_xyz(0.0, 60.0, 0.0)
                        .looking_at(Vec3::new(200.0, 0.0, 200.0), Vec3::Y),
                ));
                commands.spawn(Terrain::procedural(ProceduralTerrain::hills(42)));
            },
        );
    // The render world leaves the app at `finish`, so its counter is registered before.
    let sink = counts.clone();
    app.sub_app_mut(RenderApp)
        .add_systems(Render, move |cache: Res<PipelineCache>| {
            sink.lock()
                .unwrap_or_else(|e| e.into_inner())
                .push(cache.pipelines().count());
        });
    support::finish(&mut app);
    run_until(&mut app, "the terrain", live);
    // Everything the scene draws is compiled (synchronously) in its first frames.
    for _ in 0..60 {
        app.update();
    }
    let last = |counts: &Arc<Mutex<Vec<usize>>>| {
        *counts
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .last()
            .expect("the render world ran")
    };
    let before = last(&counts);
    let mut written = TerrainLook::default();
    for frame in 0..240 {
        let t = frame as f32 / 239.0;
        written = TerrainLook {
            tint: Vec3::new(1.0 - 0.5 * t, 1.0 - 0.4 * t, 1.0),
            saturation: 1.0 - 0.6 * t,
        };
        app.insert_resource(written);
        app.update();
    }
    // Two more frames: the render world runs one behind the main world.
    app.update();
    app.update();
    let after = last(&counts);
    assert!(before > 0, "the render world cached no pipeline");
    assert_eq!(
        after, before,
        "pipelines were compiled while the look changed"
    );
    let materials = app.world().resource::<Assets<TerrainMaterial>>();
    let (_, material) = materials.iter().next().expect("the terrain's material");
    assert_eq!(
        material.extension.params.look,
        written.tint.extend(written.saturation),
        "the material carries the last look"
    );
    finish(&mut app);
}

/// Issue #36, acceptance 1 and 4 (`docs/research/orthographic-lod.md` Validation 1 and 4): under
/// an orthographic `ElementsCamera` 457 m back from its `ElementsFocus`, pitched 35.26°, the
/// patches are selected about the focus (level 0 under it) and every terrain material morphs
/// about it; and a view 400 m high, which puts the morph band on screen, draws the terrain over
/// every pixel: no main-pass fragment fails the depth test against a prepass displaced about
/// another centre, which would leave the clear colour.
#[test]
#[ignore = "needs a GPU"]
fn an_orthographic_camera_selects_and_draws_about_its_focus() {
    use bevy::camera::{RenderTarget, ScalingMode};
    use bevy::core_pipeline::tonemapping::{DebandDither, Tonemapping};
    use bevy::render::render_resource::TextureUsages;

    use super::{TerrainMaterial, TerrainRuntime};
    use crate::ElementsFocus;

    const SIZE: u32 = 256;
    const BACK_M: f32 = 457.0;
    #[derive(Resource)]
    struct Target(Handle<Image>);

    let _lock = serial();
    let forward = Vec3::new(0.5, -0.5, 0.5).normalize();
    let mut app = App::new();
    app.add_plugins((support::headless_plugins(), ElementsPlugins))
        .insert_resource(TimeUpdateStrategy::ManualDuration(Duration::from_secs_f64(
            1.0 / 60.0,
        )))
        .insert_resource(ElementsQuality::High)
        .add_systems(
            Startup,
            move |mut commands: Commands, mut images: ResMut<Assets<Image>>| {
                let mut target =
                    Image::new_target_texture(SIZE, SIZE, TextureFormat::Rgba8UnormSrgb, None);
                target.texture_descriptor.usage |= TextureUsages::COPY_SRC;
                let target = images.add(target);
                commands.insert_resource(Target(target.clone()));
                let focus = Vec3::new(100.0, 0.0, -50.0);
                commands.spawn((
                    Camera3d::default(),
                    Camera {
                        clear_color: ClearColorConfig::Custom(Color::srgb(1.0, 0.0, 1.0)),
                        ..default()
                    },
                    RenderTarget::Image(target.into()),
                    // The plain clear colour behind the terrain: no sky, no fog; single-sampled
                    // like every crate-dressed camera.
                    ElementsCamera {
                        msaa_off: true,
                        ..ElementsCamera::bare()
                    },
                    Tonemapping::None,
                    DebandDither::Disabled,
                    Projection::from(OrthographicProjection {
                        scaling_mode: ScalingMode::FixedVertical {
                            viewport_height: 400.0,
                        },
                        far: 2.0 * BACK_M,
                        ..OrthographicProjection::default_3d()
                    }),
                    Transform::from_translation(focus - forward * BACK_M)
                        .looking_to(forward, Vec3::Y),
                    ElementsFocus::Point(focus),
                ));
                commands.spawn(Terrain::procedural(ProceduralTerrain::hills(42)));
            },
        );
    support::finish(&mut app);
    run_until(&mut app, "the bounds readback", |app| {
        app.world().resource::<TerrainRuntime>().bounds_version > 0
    });
    // Put the focus on the ground and the camera behind it.
    let xz = Vec2::new(100.0, -50.0);
    let ground = height_at(&mut app, xz).expect("the focus is on the terrain");
    let focus = Vec3::new(xz.x, ground, xz.y);
    let camera = app
        .world_mut()
        .query_filtered::<Entity, With<ElementsCamera>>()
        .single(app.world())
        .expect("one camera");
    app.world_mut().entity_mut(camera).insert((
        ElementsFocus::Point(focus),
        Transform::from_translation(focus - forward * BACK_M).looking_to(forward, Vec3::Y),
    ));
    for _ in 0..20 {
        app.update();
    }
    let (centre, lod, selected) = {
        let runtime = app.world().resource::<TerrainRuntime>();
        (
            runtime.lod_centre,
            runtime.lod.expect("a quadtree"),
            runtime.selected.clone(),
        )
    };
    assert_eq!(centre, Some(focus), "the patches' centre is the focus");
    // The patch drawn over the focus, on the vertex lattice (terrain-local XZ).
    let grid = state(&app).expect("state").grid;
    let local = focus.xz() - grid.origin - Vec2::splat(lod.sample_offset());
    let under = selected
        .iter()
        .find(|p| {
            let s = lod.node_m(p.level);
            let lo = Vec2::new(p.ix as f32, p.iz as f32) * s;
            local.cmpge(lo).all() && local.cmplt(lo + s).all()
        })
        .expect("a patch covers the focus");
    assert_eq!(under.level, 0, "the focus is drawn at {under:?}");
    for (_, material) in app.world().resource::<Assets<TerrainMaterial>>().iter() {
        assert_eq!(material.extension.params.lod_centre, focus.extend(1.0));
    }
    let target = app.world().resource::<Target>().0.clone();
    let pixels = support::read_texture(&mut app, target, SIZE as usize * 4, SIZE as usize);
    let clear = pixels
        .as_chunks::<4>()
        .0
        .iter()
        .filter(|p| p[0] >= 250 && p[1] <= 5 && p[2] >= 250)
        .count();
    assert_eq!(
        clear, 0,
        "{clear} pixels show the clear colour through the terrain"
    );
    finish(&mut app);
}
