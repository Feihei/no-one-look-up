//! Unit tests of the voxel ground's systems (no GPU): the first scan and the terrain it spawns,
//! the snow swap with its hold and thresholds, the edit pass after a carve, the cleanup, and the
//! erosion lock. See `docs/research/voxel-snow.md` (Validation targets 6, 7 and 11).

use std::collections::HashSet;
use std::sync::Arc;
use std::sync::atomic::AtomicU32;

use bevy::prelude::*;
use bevy::render::storage::ShaderBuffer;
use bevy_isomesh::{
    IsomeshPlugin, MeshBudget, MeshStats, NeedsRemesh, VolumeField, VoxelChunk, VoxelVolume,
};
use isomesh::Sdf;
use isomesh::chunk::{ChunkId, ChunkLayout};

use super::systems::{VoxelGroundState, should_warn_erosion};
use super::{VoxelGround, VoxelGroundSystems, VoxelSnow, add_systems, configure_sets};
use crate::climate::SurfaceWeathering;
use crate::cover::CoverSnapshot;
use crate::hydrology::{ErosionLocked, ErosionSettings, erosion_enabled};
use crate::plugin::ElementsQuality;
use crate::query::EnvironmentSnapshot;
use crate::terrain::model::resolve_resolution;
use crate::terrain::{
    GenParams, GenerateSource, Terrain, TerrainGpu, TerrainGrid, TerrainParamsGpu, TerrainRender,
    TerrainSource, TerrainState,
};

/// A flat top at `y = 6` over everything: solid below, air above.
#[derive(Clone, Copy)]
struct Plane;

impl Sdf for Plane {
    type Scalar = f32;
    fn sample(&self, p: [f32; 3]) -> f32 {
        p[1] - 6.0
    }
}

/// `Plane` with a box carved out of it: x and z in (2, 4), everything above `y = 2`.
#[derive(Clone, Copy)]
struct Carved;

impl Sdf for Carved {
    type Scalar = f32;
    fn sample(&self, p: [f32; 3]) -> f32 {
        let carve = ((p[0] - 3.0).abs() - 1.0)
            .max((p[2] - 3.0).abs() - 1.0)
            .max(2.0 - p[1]);
        Plane.sample(p).max(-carve)
    }
}

/// 2 × 2 chunk columns of 16 half-metre cells (16 m), three layers (`y` in `[−8, 16)`).
fn layout() -> ChunkLayout<f32> {
    ChunkLayout::new(16, 0.5, [0.0, -8.0, 0.0]).expect("a valid layout")
}

fn ground_app(meshing: bool) -> App {
    let mut app = App::new();
    app.add_plugins((MinimalPlugins, AssetPlugin::default()))
        .init_asset::<Image>()
        .init_asset::<ShaderBuffer>()
        .init_resource::<ElementsQuality>()
        .init_resource::<crate::ElementsUnits>()
        .init_resource::<EnvironmentSnapshot>()
        .init_resource::<ErosionSettings>();
    if meshing {
        app.add_plugins(IsomeshPlugin).insert_resource(MeshBudget {
            per_frame: std::time::Duration::from_millis(50),
            max_in_flight: 64,
        });
    }
    configure_sets(&mut app);
    add_systems(&mut app);
    app
}

/// A volume over `base` with every chunk of the layout spawned and marked; returns its entity.
fn spawn_volume(app: &mut App, base: impl VolumeField) -> Entity {
    let volume = app
        .world_mut()
        .spawn((
            VoxelVolume::new(layout(), base),
            VoxelGround::new(UVec2::splat(32)),
        ))
        .id();
    for x in 0..2 {
        for y in 0..3 {
            for z in 0..2 {
                app.world_mut().spawn((
                    VoxelChunk {
                        id: ChunkId::new([x, y, z]),
                        volume,
                    },
                    NeedsRemesh,
                ));
            }
        }
    }
    volume
}

fn state(app: &App, volume: Entity) -> &VoxelGroundState {
    app.world()
        .get::<VoxelGroundState>(volume)
        .expect("the ground was built")
}

fn field_ptr(app: &App, volume: Entity) -> *const () {
    let field = &app
        .world()
        .get::<VoxelVolume>(volume)
        .expect("the volume")
        .field;
    Arc::as_ptr(field).cast()
}

fn marked_ids(app: &mut App) -> HashSet<[i32; 3]> {
    let mut query = app
        .world_mut()
        .query_filtered::<&VoxelChunk, With<NeedsRemesh>>();
    query.iter(app.world()).map(|c| c.id.coords).collect()
}

fn clear_marks(app: &mut App) {
    let entities: Vec<Entity> = app
        .world_mut()
        .query_filtered::<Entity, With<NeedsRemesh>>()
        .iter(app.world())
        .collect();
    for e in entities {
        app.world_mut().entity_mut(e).remove::<NeedsRemesh>();
    }
}

/// A readback snapshot on the terrain grid of the 16 m test volume (32² cells of 0.5 m).
fn snapshot(depth: impl Fn(usize) -> f32) -> CoverSnapshot {
    let grid = TerrainGrid {
        origin: Vec2::ZERO,
        cell_m: 0.5,
        size: UVec2::splat(32),
        y0: 0.0,
    };
    let depth: Arc<[f32]> = (0..32 * 32).map(depth).collect();
    CoverSnapshot {
        grid,
        wetness: vec![0.0; 32 * 32].into(),
        depth,
    }
}

fn feed(app: &mut App, depth: impl Fn(usize) -> f32) {
    app.world_mut().resource_mut::<EnvironmentSnapshot>().cover = Some(snapshot(depth));
    app.update();
}

/// The first scan: the terrain over the volume's XZ box with the tops as its heights, the first
/// field wrapped around the user's.
#[test]
fn the_first_scan_spawns_the_terrain_and_wraps_the_field() {
    let mut app = ground_app(false);
    let volume = spawn_volume(&mut app, Plane);
    let base_ptr = field_ptr(&app, volume);
    app.update();

    let state = state(&app, volume);
    assert_eq!(state.columns.grid.size, UVec2::splat(32));
    assert!((state.columns.grid.cell_m - 0.5).abs() < 1e-6);
    assert_eq!(state.columns.grid.origin, Vec2::ZERO);
    assert_eq!((state.columns.y_lo, state.columns.y_hi), (-8.0, 16.0));
    assert!(state.top.iter().all(|t| (t - 6.0).abs() <= 2e-3));
    assert!(state.snow.iter().all(|s| *s == 0.0));
    assert_eq!(state.columns_rescanned, 32 * 32);
    assert_ne!(field_ptr(&app, volume), base_ptr, "the field is wrapped");
    assert!(Arc::ptr_eq(
        &app.world()
            .get::<VoxelVolume>(volume)
            .expect("volume")
            .field,
        &state.installed
    ));

    let terrain_entity = state.terrain.expect("a voxel ground owns a terrain");
    let terrain = app
        .world()
        .get::<Terrain>(terrain_entity)
        .expect("the terrain entity");
    assert_eq!(terrain.render, TerrainRender::External);
    assert_eq!(terrain.extent_m, Vec2::splat(16.0));
    assert_eq!(terrain.resolution, UVec2::splat(32));
    let transform = app
        .world()
        .get::<Transform>(terrain_entity)
        .expect("transform");
    assert_eq!(transform.translation, Vec3::new(8.0, 0.0, 8.0));
    let TerrainSource::Heightmap {
        image,
        height_range_m,
    } = &terrain.source
    else {
        panic!("the terrain imports a heightmap");
    };
    assert_eq!(*height_range_m, 0.0..1.0);
    let image = app
        .world()
        .resource::<Assets<Image>>()
        .get(image)
        .expect("the image");
    let data = image.data.as_ref().expect("the pixels");
    let heights: &[f32] = bytemuck::cast_slice(data);
    assert_eq!(
        heights,
        &state.top[..],
        "the image holds the tops to the bit"
    );
    // The resolved grid is the one the terrain will build.
    let (size, cell, adjusted) = resolve_resolution(Vec2::splat(16.0), UVec2::splat(32), 512);
    assert_eq!((size, cell, adjusted), (UVec2::splat(32), 0.5, false));
}

/// With two world units to the metre the same volume (16 world units wide, tops at 6) is 8 m
/// wide with tops at 3 m: the columns stay in field units, the terrain under them is in metres.
#[test]
fn the_terrain_under_a_ground_in_other_units_is_in_metres() {
    let mut app = ground_app(false);
    app.insert_resource(crate::ElementsUnits {
        units_per_metre: 2.0,
    });
    let volume = spawn_volume(&mut app, Plane);
    app.update();

    let state = state(&app, volume);
    assert!(
        (state.columns.grid.cell_m - 0.5).abs() < 1e-6,
        "world units"
    );
    assert!(state.top.iter().all(|t| (t - 6.0).abs() <= 2e-3));
    assert_eq!(state.extent_m, Vec2::splat(8.0));
    let terrain_entity = state.terrain.expect("a voxel ground owns a terrain");
    let terrain = app
        .world()
        .get::<Terrain>(terrain_entity)
        .expect("the terrain entity");
    assert_eq!(terrain.extent_m, Vec2::splat(8.0));
    let transform = app
        .world()
        .get::<Transform>(terrain_entity)
        .expect("transform");
    assert_eq!(
        transform.translation,
        Vec3::new(8.0, 0.0, 8.0),
        "world units"
    );
    let TerrainSource::Heightmap { image, .. } = &terrain.source else {
        panic!("the terrain imports a heightmap");
    };
    let image = app
        .world()
        .resource::<Assets<Image>>()
        .get(image)
        .expect("the image");
    let heights: &[f32] = bytemuck::cast_slice(image.data.as_ref().expect("the pixels"));
    assert!(heights.iter().all(|h| (h - 3.0).abs() <= 1e-3), "metres");
}

/// Validation 6: the hold rule at the system level.
#[test]
fn swap_threshold_and_hold() {
    let mut app = ground_app(false);
    let volume = spawn_volume(&mut app, Plane);
    app.update();
    clear_marks(&mut app);
    app.update();
    let base_ptr = field_ptr(&app, volume);
    let rescanned = state(&app, volume).columns_rescanned;

    // Under the threshold (0.5 c = 0.25 m): nothing.
    feed(&mut app, |_| 0.24);
    assert_eq!(state(&app, volume).swaps, 0);
    assert_eq!(field_ptr(&app, volume), base_ptr, "the field is untouched");
    assert!(marked_ids(&mut app).is_empty());

    // One column over (column 15 of row 7: centre (7.75, 3.75)), the rest under.
    let column = 7 * 32 + 15;
    feed(&mut app, |i| if i == column { 0.26 } else { 0.10 });
    let s = state(&app, volume);
    assert_eq!(s.swaps, 1);
    assert_eq!(s.snow[column], 0.26);
    assert!(
        s.snow
            .iter()
            .enumerate()
            .all(|(i, d)| i == column || *d == 0.0),
        "every other column holds what it held"
    );
    assert_ne!(field_ptr(&app, volume), base_ptr);
    // Its chunks: two in x (the footprint crosses the plane at 8), one in z, one layer.
    let want: HashSet<[i32; 3]> = [[0, 1, 0], [1, 1, 0]].into();
    assert_eq!(marked_ids(&mut app), want);

    // Our own marks are not a user edit: the next frame rescans nothing.
    app.update();
    assert_eq!(state(&app, volume).columns_rescanned, rescanned);

    // The same readback again (a new allocation, the same values): no swap.
    clear_marks(&mut app);
    feed(&mut app, |i| if i == column { 0.26 } else { 0.10 });
    assert_eq!(state(&app, volume).swaps, 1);
    assert!(marked_ids(&mut app).is_empty());
    // 0.26 to 0.50 is not more than 0.25 since the last swap; 0.26 to 0.52 is.
    feed(&mut app, |i| if i == column { 0.50 } else { 0.10 });
    assert_eq!(state(&app, volume).swaps, 1);
    feed(&mut app, |i| if i == column { 0.52 } else { 0.10 });
    assert_eq!(state(&app, volume).swaps, 2);
    assert_eq!(state(&app, volume).snow[column], 0.52);
}

/// Validation 7: carving the crest lowers the top, writes it to `bedrock`, keeps the snow, and
/// meshes every affected chunk once, in the frame of the edit.
#[test]
fn carve_the_top_lowers_the_top() {
    let mut app = ground_app(true);
    // The terrain as the render side would have it: live, with a bedrock buffer.
    let bedrock = app
        .world_mut()
        .resource_mut::<Assets<ShaderBuffer>>()
        .add(ShaderBuffer::new(
            vec![0.0_f32; 32 * 32],
            bevy::asset::RenderAssetUsages::default(),
        ));
    let volume = spawn_volume(&mut app, Plane);
    app.update();
    let terrain_entity = state(&app, volume)
        .terrain
        .expect("a voxel ground owns a terrain");
    let grid = state(&app, volume).columns.grid;
    let live_state = TerrainState {
        entity: terrain_entity,
        grid,
        bedrock: bedrock.clone(),
        sediment: Handle::default(),
        water: Handle::default(),
        flux: Handle::default(),
        velocity: Handle::default(),
        suspended: Handle::default(),
        transport: Handle::default(),
        source_map: Handle::default(),
        thermal_flux: [Handle::default(), Handle::default()],
        cells: Handle::default(),
        flow_cells: Handle::default(),
        height_readback: Handle::default(),
        leaf_bounds: Handle::default(),
        readback_size: UVec2::splat(32),
        version: Arc::new(AtomicU32::new(1)),
    };
    app.insert_resource(TerrainGpu {
        state: live_state.clone(),
        params: TerrainParamsGpu::default(),
        source: GenerateSource::Procedural(GenParams::default()),
        changes: 0,
        packed: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        generated: Arc::new(std::sync::atomic::AtomicBool::new(true)),
    })
    .insert_resource(live_state);

    // Every column holds 0.5 m of snow.
    feed(&mut app, |_| 0.5);
    assert_eq!(state(&app, volume).swaps, 1);
    settle(&mut app);

    // A user system, ordered before the ground, carves and marks the chunk it carved.
    app.add_systems(
        Update,
        (move |mut commands: Commands,
               mut volumes: Query<&mut VoxelVolume>,
               chunks: Query<(Entity, &VoxelChunk)>,
               mut done: Local<bool>| {
            if *done {
                return;
            }
            *done = true;
            for mut volume in &mut volumes {
                volume.replace_field(Carved);
            }
            for (entity, chunk) in &chunks {
                if chunk.id.coords == [0, 1, 0] {
                    commands.entity(entity).insert(NeedsRemesh);
                }
            }
        })
        .before(VoxelGroundSystems),
    );
    let before = app.world().resource::<TerrainGpu>().changes;
    app.update();

    let s = state(&app, volume);
    let at = |i: usize, j: usize| j * 32 + i;
    // Column (6, 6) is centred at (3.25, 3.25), inside the carve; (20, 20) is not.
    assert!((s.top[at(6, 6)] - 2.0).abs() <= 2e-3, "{}", s.top[at(6, 6)]);
    assert!((s.top[at(20, 20)] - 6.0).abs() <= 2e-3);
    assert_eq!(s.snow[at(6, 6)], 0.5, "the snow is carried down");
    let buffers = app.world().resource::<Assets<ShaderBuffer>>();
    let written: &[f32] = buffers
        .get(&bedrock)
        .expect("bedrock")
        .cast_slice()
        .expect("the buffer holds data");
    assert_eq!(written, &s.top[..], "bedrock was rewritten with the tops");
    assert!(
        app.world().resource::<TerrainGpu>().changes > before,
        "the bedrock write asks for a repack"
    );
    assert!(
        Arc::ptr_eq(
            &app.world()
                .get::<VoxelVolume>(volume)
                .expect("volume")
                .field,
            &s.installed
        ),
        "the field is the SnowyField, not the bare base"
    );
    // The user's chunk and the derived ones are one and the same: one chunk, meshed once.
    assert_eq!(app.world().resource::<MeshStats>().spawned, 1);
}

/// Without a `VoxelGround` the field is the user's again, the chunks are remeshed and the
/// terrain goes.
#[test]
fn removing_the_ground_restores_the_field() {
    let mut app = ground_app(false);
    let volume = spawn_volume(&mut app, Plane);
    let base_ptr = field_ptr(&app, volume);
    app.update();
    let terrain = state(&app, volume)
        .terrain
        .expect("a voxel ground owns a terrain");
    assert_ne!(field_ptr(&app, volume), base_ptr);
    clear_marks(&mut app);
    app.world_mut().entity_mut(volume).remove::<VoxelGround>();
    app.update();
    app.update();
    assert_eq!(field_ptr(&app, volume), base_ptr);
    assert!(app.world().get::<VoxelGroundState>(volume).is_none());
    assert!(
        app.world().get_entity(terrain).is_err(),
        "the terrain is gone"
    );
    assert_eq!(marked_ids(&mut app).len(), 12, "every chunk is remeshed");
}

/// A second terrain: the voxel ground steps aside with one warning.
#[test]
fn a_user_terrain_blocks_the_ground() {
    let mut app = ground_app(false);
    app.world_mut().spawn(Terrain::flat(0.0));
    let volume = spawn_volume(&mut app, Plane);
    app.update();
    app.update();
    assert!(app.world().get::<VoxelGroundState>(volume).is_none());
}

/// Validation 11: erosion is off whatever the setting says while a voxel ground exists, and the
/// warning is logged once.
#[test]
fn erosion_is_ignored_for_voxel_ground() {
    let on = ErosionSettings {
        enabled: true,
        ..default()
    };
    assert!(erosion_enabled(&on, false));
    assert!(!erosion_enabled(&on, true));
    let mut warned = false;
    let said = (0..100)
        .filter(|_| should_warn_erosion(true, &mut warned))
        .count();
    assert_eq!(said, 1, "one warning across 100 updates");
    let mut warned = false;
    assert!(!should_warn_erosion(false, &mut warned), "off says nothing");

    // The lock follows the component.
    let mut app = ground_app(false);
    let volume = spawn_volume(&mut app, Plane);
    app.update();
    app.update();
    assert!(app.world().get_resource::<ErosionLocked>().is_some());
    app.world_mut().entity_mut(volume).remove::<VoxelGround>();
    app.update();
    app.update();
    assert!(app.world().get_resource::<ErosionLocked>().is_none());
}

/// Update until every queued mesh has been built and applied: nothing in flight and no chunk
/// still marked (a chunk marked while its task runs waits for it without being counted).
fn settle(app: &mut App) {
    for _ in 0..2000 {
        app.update();
        let stats = *app.world().resource::<MeshStats>();
        if stats.in_flight == 0 && marked_ids(app).is_empty() {
            return;
        }
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    panic!("meshing did not settle");
}

/// Validation 9 (CPU rows): what the voxel ground costs at the example's size, 128² columns by
/// 48 cells. Timing, so ignored by default: `cargo test --release --features isomesh --lib --
/// --ignored --nocapture voxel_cost`.
#[test]
#[ignore = "timing: run in release"]
fn voxel_cost() {
    use std::time::Instant;

    use isomesh::MeshBuffer;
    use isomesh::chunk::dirty::{DirtySet, mark_edit};
    use isomesh::marching_cubes::MarchingCubes;

    use super::field::SnowyField;
    use super::gpu_tests::HillWorld;
    use super::model::{Columns, box_indices, scan_columns};

    let layout = ChunkLayout::new(16, 0.5, [-32.0, -8.0, -32.0]).expect("a valid layout");
    let grid = TerrainGrid {
        origin: Vec2::splat(-32.0),
        cell_m: 0.5,
        size: UVec2::splat(128),
        y0: 0.0,
    };
    let columns = Columns {
        grid,
        y_lo: -8.0,
        y_hi: 16.0,
        cell: 0.5,
    };
    // (a) the initial scan of every column.
    let mut tops = vec![0.0_f32; columns.len()];
    let began = Instant::now();
    scan_columns(&HillWorld, &columns, 0..columns.len(), &mut tops);
    let scan_ms = began.elapsed().as_secs_f64() * 1000.0;
    // (b) one chunk column's rescan.
    let range = columns
        .columns_in(&layout, ChunkId::new([4, 0, 4]))
        .expect("inside");
    let mut rescan_ms = f64::INFINITY;
    for _ in 0..20 {
        let began = Instant::now();
        let n = scan_columns(
            &HillWorld,
            &columns,
            box_indices(grid.size, range),
            &mut tops,
        );
        rescan_ms = rescan_ms.min(began.elapsed().as_secs_f64() * 1000.0);
        assert_eq!(n, 256, "16 × 16 columns under a chunk column");
    }
    // (c) one surface chunk on a worker: Marching Cubes of the crest chunk against the field
    // with half a metre of snow.
    let snow: Arc<[f32]> = vec![0.5_f32; columns.len()].into();
    let top: Arc<[f32]> = tops.clone().into();
    let field = SnowyField::new(Arc::new(HillWorld), 0.5, top, grid, snow, grid);
    let shape = layout.sample_shape().expect("a shape");
    let id = ChunkId::new([4, 2, 4]);
    let mut mesh_ms = f64::INFINITY;
    let mut triangles = 0;
    for _ in 0..20 {
        let mut out = MeshBuffer::<f32>::new();
        let began = Instant::now();
        MarchingCubes::<f32>::new()
            .extract(&field, &shape, layout.sample_origin(id), 0.5, &mut out)
            .expect("a mesh");
        mesh_ms = mesh_ms.min(began.elapsed().as_secs_f64() * 1000.0);
        triangles = out.triangle_count();
    }
    // (d) `mark_changed_chunks` over the whole volume, which the column rule replaces.
    let bare = SnowyField::new(
        Arc::new(HillWorld),
        0.5,
        field.top().clone(),
        grid,
        vec![0.0_f32; columns.len()].into(),
        grid,
    );
    let mut dirty = DirtySet::new();
    let began = Instant::now();
    mark_edit(
        &layout,
        &bare,
        &field,
        [0, 0, 0],
        [127, 47, 127],
        &mut dirty,
    )
    .expect("an edit");
    let mark_ms = began.elapsed().as_secs_f64() * 1000.0;
    println!(
        "RESULT cost: initial scan {scan_ms:.1} ms (16 384 columns), one chunk column rescan {rescan_ms:.3} ms (256 columns), one surface chunk {mesh_ms:.3} ms ({triangles} triangles), mark_changed_chunks over the volume {mark_ms:.1} ms ({} chunks)",
        dirty.iter().count()
    );
    assert!(scan_ms <= 150.0, "the initial scan: {scan_ms} ms");
    assert!(rescan_ms <= 2.0, "a chunk column rescan: {rescan_ms} ms");
    assert!(mesh_ms <= 2.0, "a surface chunk: {mesh_ms} ms");
}

/// A `VoxelSnow` volume over `Plane` (a flat top at `y = 6`) placed by `transform`, every chunk
/// spawned and marked; returns its entity. The test app has no transform propagation, so the
/// `GlobalTransform` is set directly.
fn spawn_snow_volume(app: &mut App, transform: Transform) -> Entity {
    let volume = app
        .world_mut()
        .spawn((
            VoxelVolume::new(layout(), Plane),
            VoxelSnow,
            GlobalTransform::from(transform),
        ))
        .id();
    for x in 0..2 {
        for y in 0..3 {
            for z in 0..2 {
                app.world_mut().spawn((
                    VoxelChunk {
                        id: ChunkId::new([x, y, z]),
                        volume,
                    },
                    NeedsRemesh,
                ));
            }
        }
    }
    volume
}

/// An app with a point snowpack of 120 mm of water at 200 kg/m³: 0.6 m on exposed flat ground,
/// above the half-voxel (0.25 m) swap threshold of the test layout even halved by a scale of 2.
fn snowy_app() -> App {
    let mut app = ground_app(false);
    app.insert_resource(SurfaceWeathering::with_snowpack(0.6, 200.0, 0.0));
    app
}

fn turned() -> Transform {
    Transform::from_xyz(40.0, 2.0, -10.0)
        .with_rotation(Quat::from_rotation_y(0.3))
        .with_scale(Vec3::splat(2.0))
}

fn held(app: &App, volume: Entity) -> Vec<f32> {
    state(app, volume).snow.to_vec()
}

/// Issue #13: two volumes, one moved, turned and scaled, each get a slab from the point pack and
/// no terrain exists.
#[test]
fn two_snow_volumes_get_slabs_and_no_terrain() {
    let mut app = snowy_app();
    let plain = spawn_snow_volume(&mut app, Transform::IDENTITY);
    let moved = spawn_snow_volume(&mut app, turned());
    app.update();
    app.update();
    let terrains = app
        .world_mut()
        .query::<&Terrain>()
        .iter(app.world())
        .count();
    assert_eq!(terrains, 0, "VoxelSnow spawns no terrain");
    // The pack's 0.6 m in field units: the volume scaled by 2 holds half of it.
    for (volume, want) in [(plain, 0.6), (moved, 0.3)] {
        assert!(state(&app, volume).terrain.is_none());
        let snow = held(&app, volume);
        assert!(
            snow.iter().all(|d| (d - want).abs() < 1e-5),
            "every flat column holds {want}: {:?}",
            &snow[..4]
        );
        let field = &app
            .world()
            .get::<VoxelVolume>(volume)
            .expect("the volume")
            .field;
        assert!(
            field.sample([8.0, 6.0 + 0.5 * want, 8.0]) < 0.0,
            "inside the slab"
        );
        assert!(
            field.sample([8.0, 6.0 + 1.5 * want, 8.0]) > 0.0,
            "above the slab"
        );
    }
    assert!(
        !marked_ids(&mut app).is_empty(),
        "the slab's chunks are remeshed"
    );
}

/// Where a cover map lies under a volume's world position its depth is used; elsewhere the pack.
#[test]
fn a_snow_volume_samples_the_cover_map_at_its_world_position() {
    let mut app = snowy_app();
    let plain = spawn_snow_volume(&mut app, Transform::IDENTITY);
    let moved = spawn_snow_volume(&mut app, turned());
    // 1 m of snow over x in [30, 158), z in [−60, 68): under the moved volume only.
    let grid = TerrainGrid {
        origin: Vec2::new(30.0, -60.0),
        cell_m: 1.0,
        size: UVec2::splat(128),
        y0: 0.0,
    };
    app.world_mut().resource_mut::<EnvironmentSnapshot>().cover = Some(CoverSnapshot {
        grid,
        depth: vec![1.0_f32; 128 * 128].into(),
        wetness: vec![0.0; 128 * 128].into(),
    });
    app.update();
    app.update();
    assert!(held(&app, plain).iter().all(|d| (d - 0.6).abs() < 1e-5));
    // 1 m of world snow on a volume scaled by 2 is 0.5 field units.
    assert!(held(&app, moved).iter().all(|d| (d - 0.5).abs() < 1e-5));
}

#[test]
fn a_tilted_snow_volume_gets_no_slab() {
    let mut app = snowy_app();
    let volume = spawn_snow_volume(
        &mut app,
        Transform::from_rotation(Quat::from_rotation_x(0.2)),
    );
    let base = field_ptr(&app, volume);
    app.update();
    app.update();
    assert!(app.world().get::<VoxelGroundState>(volume).is_none());
    assert_eq!(field_ptr(&app, volume), base);
}

#[test]
fn a_volume_with_both_components_is_a_voxel_ground() {
    let mut app = snowy_app();
    let volume = spawn_snow_volume(&mut app, Transform::IDENTITY);
    app.world_mut()
        .entity_mut(volume)
        .insert(VoxelGround::new(UVec2::splat(32)));
    app.update();
    app.update();
    assert!(state(&app, volume).terrain.is_some());
}

#[test]
fn removing_voxel_snow_restores_the_field() {
    let mut app = snowy_app();
    let volume = spawn_snow_volume(&mut app, Transform::IDENTITY);
    let base = field_ptr(&app, volume);
    app.update();
    app.update();
    assert_ne!(field_ptr(&app, volume), base);
    clear_marks(&mut app);
    app.world_mut().entity_mut(volume).remove::<VoxelSnow>();
    app.update();
    assert!(app.world().get::<VoxelGroundState>(volume).is_none());
    assert_eq!(field_ptr(&app, volume), base);
    assert_eq!(marked_ids(&mut app).len(), 12, "every chunk is remeshed");
}
