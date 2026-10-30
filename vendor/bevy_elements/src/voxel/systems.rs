//! The voxel ground's and voxel snow's systems and the state they keep on the volume entity. All
//! of them run in `Update`, in [`VoxelGroundSystems`](super::VoxelGroundSystems), before
//! bevy_isomesh's meshing systems: whatever they change is meshed in the same frame.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use bevy::asset::RenderAssetUsages;
use bevy::math::Vec2;
use bevy::prelude::*;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};
use bevy::render::storage::ShaderBuffer;
use bevy_isomesh::{NeedsRemesh, VolumeField, VoxelChunk, VoxelVolume};
use isomesh::chunk::ChunkId;

use super::field::SnowyField;
use super::model::{
    Columns, DirtyChunks, TOP_EPSILON_M, box_indices, chunk_entities, hold_swap, scan_columns,
    slab_span, snow_columns, snow_span, top_exposure, upright_scale, volume_box,
};
use super::{VoxelGround, VoxelSnow};
use crate::climate::SurfaceWeathering;
use crate::cover::SnowSettings;
use crate::hydrology::{ErosionLocked, ErosionSettings};
use crate::plugin::{ElementsQuality, ElementsUnits};
use crate::query::EnvironmentSnapshot;
use crate::terrain::model::resolve_resolution;
use crate::terrain::{Terrain, TerrainGpu, TerrainState};

/// What a snowy volume keeps, a [`VoxelGround`] or a [`VoxelSnow`]: the user's field, the tops
/// and the held snow the installed [`SnowyField`] was built from, and the bookkeeping of the edit
/// and swap passes. Inserted on the volume entity once the first column scan is done.
#[derive(Component)]
pub(crate) struct VoxelGroundState {
    /// The user's field: what the slab is added to.
    pub base: Arc<dyn VolumeField>,
    /// The `SnowyField` this module put in `VoxelVolume::field`; a different `Arc` there means
    /// the user replaced the field.
    pub installed: Arc<dyn VolumeField>,
    pub columns: Columns,
    /// The XZ extent (m) and the resolution the terrain was built for.
    pub extent_m: Vec2,
    /// Column tops (field units: the world's for a `VoxelGround`) and the held snow depths (field
    /// units), on `columns.grid`.
    pub top: Arc<[f32]>,
    pub snow: Arc<[f32]>,
    /// The terrain entity a `VoxelGround` volume owns; `None` for a `VoxelSnow` volume.
    pub terrain: Option<Entity>,
    /// The tops changed since they were last written to the `bedrock` buffer.
    pub tops_dirty: bool,
    /// Built this frame: the first edit pass has nothing to see (every chunk is new).
    pub fresh: bool,
    /// Chunks this module marked `NeedsRemesh` last frame: their change is not a user edit.
    pub own_marks: HashSet<Entity>,
    /// The last readback's depth map handled by the swap pass.
    pub last_depth: Option<Arc<[f32]>>,
    /// The point snowpack depth (m) a `VoxelSnow` volume last refreshed from.
    pub last_pack_m: f32,
    pub warned_outside: bool,
    /// Field swaps for new snow, and columns the edit passes rescanned (tests read these).
    pub swaps: u32,
    pub columns_rescanned: u64,
}

/// Marks the terrain entity a volume owns, so it can be cleaned up with the volume.
#[derive(Component)]
pub(crate) struct VoxelGroundTerrain {
    pub volume: Entity,
}

/// One-time warnings of the build systems.
#[derive(Default)]
pub(crate) struct BuildWarnings {
    other_terrain: bool,
    cell_size: bool,
    tilted: bool,
    both: bool,
}

/// Warn once when the voxel cell size is not a power of two.
fn check_cell_size(cell: f32, warnings: &mut BuildWarnings) {
    if (cell.log2() - cell.log2().round()).abs() > 1e-6 && !warnings.cell_size {
        warn!(
            "the voxel cell size {cell} is not a power of two: isomesh chunk seams close only at powers of two, and the snow slab inherits the cracks"
        );
        warnings.cell_size = true;
    }
}

/// Install a fresh `SnowyField` over the state's base, tops and held snow.
fn install(volume: &mut VoxelVolume, state: &mut VoxelGroundState) {
    let grid = state.columns.grid;
    volume.replace_field(SnowyField::new(
        Arc::clone(&state.base),
        state.columns.cell,
        Arc::clone(&state.top),
        grid,
        Arc::clone(&state.snow),
        grid,
    ));
    state.installed = Arc::clone(&volume.field);
}

/// `NeedsRemesh` on the chunks of `ids` that exist, remembering that the mark is ours.
fn mark(
    commands: &mut Commands,
    state: &mut VoxelGroundState,
    entities: &HashMap<ChunkId, Entity>,
    ids: HashSet<ChunkId>,
) {
    for id in ids {
        if let Some(&entity) = entities.get(&id) {
            commands.entity(entity).try_insert(NeedsRemesh);
            state.own_marks.insert(entity);
        }
    }
}

/// The component a state belongs to is gone, or a `VoxelSnow` state's volume also has a
/// `VoxelGround` now (it is rebuilt as one).
fn orphaned(state: &VoxelGroundState, ground: bool, snow: bool) -> bool {
    match state.terrain {
        Some(_) => !ground,
        None => !snow || ground,
    }
}

/// Every volume, with whatever snow state and components it has.
type MaintainedVolumes<'w, 's> = Query<
    'w,
    's,
    (
        Entity,
        &'static mut VoxelVolume,
        Option<&'static VoxelGroundState>,
        Option<&'static VoxelGround>,
        Has<VoxelSnow>,
    ),
>;

/// `VoxelSnow` volumes not scanned yet.
type UnbuiltSnowVolumes<'w, 's> = Query<
    'w,
    's,
    (
        Entity,
        &'static mut VoxelVolume,
        &'static GlobalTransform,
        Has<VoxelGround>,
    ),
    (With<VoxelSnow>, Without<VoxelGroundState>),
>;

/// Clean up what a volume no longer needs: a terrain whose volume is gone, and the state of a
/// volume that lost its `VoxelGround` or `VoxelSnow` (its field is the user's again and every
/// chunk is remeshed without snow).
pub(crate) fn voxel_ground_maintain(
    mut commands: Commands,
    terrains: Query<(Entity, &VoxelGroundTerrain)>,
    mut volumes: MaintainedVolumes,
    chunks: Query<(Entity, &VoxelChunk)>,
    quality: Res<ElementsQuality>,
) {
    // A user who changed `resolution` (or a quality level that clamps it differently) gets a
    // terrain, an image and a held-snow grid of the new size: tear down, and `build` runs again.
    let mut torn_down: HashSet<Entity> = HashSet::new();
    for (entity, _, state, ground, snow) in &volumes {
        let Some(state) = state else {
            continue;
        };
        let resized = ground.is_some_and(|ground| {
            let (size, _, _) = resolve_resolution(
                state.extent_m,
                ground.resolution,
                quality.terrain_max_resolution(),
            );
            state.terrain.is_some() && size != state.columns.grid.size
        });
        if resized || orphaned(state, ground.is_some(), snow) {
            torn_down.insert(entity);
        }
    }
    for entity in &torn_down {
        let Ok((_, mut volume, Some(state), _, _)) = volumes.get_mut(*entity) else {
            continue;
        };
        volume.field = Arc::clone(&state.base);
        for (chunk_entity, _) in chunks.iter().filter(|(_, c)| c.volume == *entity) {
            commands.entity(chunk_entity).try_insert(NeedsRemesh);
        }
        commands.entity(*entity).remove::<VoxelGroundState>();
    }
    for (terrain, marker) in &terrains {
        let owned = volumes
            .get(marker.volume)
            .is_ok_and(|(_, _, state, _, _)| state.is_some_and(|s| s.terrain == Some(terrain)));
        if !owned || torn_down.contains(&marker.volume) {
            commands.entity(terrain).despawn();
        }
    }
}

/// The first scan: for a `VoxelGround` volume that has chunks and no state yet, find every
/// column's top, spawn the terrain over the volume's XZ extent with the tops as its heights, and
/// put the first `SnowyField` (no snow yet) in the volume.
#[allow(clippy::too_many_arguments)]
pub(crate) fn voxel_ground_build(
    mut commands: Commands,
    quality: Res<ElementsQuality>,
    units: Res<ElementsUnits>,
    mut images: ResMut<Assets<Image>>,
    mut volumes: Query<(Entity, &mut VoxelVolume, &VoxelGround), Without<VoxelGroundState>>,
    built: Query<&VoxelGroundState>,
    chunks: Query<(Entity, &VoxelChunk)>,
    terrains: Query<(), With<Terrain>>,
    mut warnings: Local<BuildWarnings>,
) {
    let mut taken = built.iter().any(|s| s.terrain.is_some()) || !terrains.is_empty();
    for (entity, mut volume, ground) in &mut volumes {
        let Some((min, max)) = volume_box(
            &volume.layout,
            chunks
                .iter()
                .filter(|(_, c)| c.volume == entity)
                .map(|(_, c)| c),
        ) else {
            continue;
        };
        if taken {
            if !warnings.other_terrain {
                warn!(
                    "a VoxelGround needs the only Terrain: another terrain (or another VoxelGround volume) exists, so {entity} gets no ground"
                );
                warnings.other_terrain = true;
            }
            continue;
        }
        let cell = volume.layout.cell_size();
        check_cell_size(cell, &mut warnings);
        // The columns are in field space, which for a ground is world space (world units); the
        // terrain under it is in metres.
        let u = units.per_metre();
        let extent = Vec2::new(max[0] - min[0], max[2] - min[2]);
        let centre = Vec2::new(0.5 * (max[0] + min[0]), 0.5 * (max[2] + min[2]));
        let extent_m = extent / u;
        let (size, terrain_cell_m, _) = resolve_resolution(
            extent_m,
            ground.resolution,
            quality.terrain_max_resolution(),
        );
        let terrain_cell = terrain_cell_m * u;
        // The terrain centres its grid on its transform exactly this way.
        let grid = crate::terrain::TerrainGrid {
            origin: centre - size.as_vec2() * terrain_cell * 0.5,
            cell_m: terrain_cell,
            size,
            y0: 0.0,
        };
        let columns = Columns {
            grid,
            y_lo: min[1],
            y_hi: max[1],
            cell,
        };
        let mut tops = vec![0.0_f32; columns.len()];
        let base = Arc::clone(&volume.field);
        scan_columns(&*base, &columns, 0..columns.len(), &mut tops);

        let heights_m: Vec<f32> = tops.iter().map(|top| top / u).collect();
        let image = images.add(Image::new(
            Extent3d {
                width: size.x,
                height: size.y,
                depth_or_array_layers: 1,
            },
            TextureDimension::D2,
            bytemuck::cast_slice(&heights_m).to_vec(),
            TextureFormat::R32Float,
            RenderAssetUsages::default(),
        ));
        let terrain = commands
            .spawn((
                Name::new("Voxel ground"),
                Terrain::from_heightmap(image, 0.0..1.0)
                    .with_extent(extent_m)
                    .with_resolution(size)
                    .external(),
                Transform::from_xyz(centre.x, 0.0, centre.y),
                VoxelGroundTerrain { volume: entity },
            ))
            .id();

        let mut state = VoxelGroundState {
            base,
            installed: Arc::clone(&volume.field),
            columns,
            extent_m,
            top: tops.into(),
            snow: vec![0.0_f32; columns.len()].into(),
            terrain: Some(terrain),
            tops_dirty: false,
            fresh: true,
            own_marks: HashSet::new(),
            last_depth: None,
            last_pack_m: 0.0,
            warned_outside: false,
            swaps: 0,
            columns_rescanned: columns.len() as u64,
        };
        install(&mut volume, &mut state);
        commands.entity(entity).insert(state);
        taken = true;
    }
}

/// The first scan of a `VoxelSnow` volume: the column tops over its field-space box, one column
/// per voxel, and the first `SnowyField` (no snow yet). No terrain.
pub(crate) fn voxel_snow_build(
    mut commands: Commands,
    quality: Res<ElementsQuality>,
    mut volumes: UnbuiltSnowVolumes,
    chunks: Query<(Entity, &VoxelChunk)>,
    mut warnings: Local<BuildWarnings>,
) {
    for (entity, mut volume, transform, ground) in &mut volumes {
        if ground {
            if !warnings.both {
                warn!("{entity} has both VoxelGround and VoxelSnow: it is a voxel ground");
                warnings.both = true;
            }
            continue;
        }
        if upright_scale(transform).is_none() {
            if !warnings.tilted {
                warn!(
                    "VoxelSnow needs the volume's vertical to stay vertical (a translation, a turn about the vertical and a uniform scale): {entity} is tilted or unevenly scaled and gets no snow"
                );
                warnings.tilted = true;
            }
            continue;
        }
        let Some((min, max)) = volume_box(
            &volume.layout,
            chunks
                .iter()
                .filter(|(_, c)| c.volume == entity)
                .map(|(_, c)| c),
        ) else {
            continue;
        };
        let cell = volume.layout.cell_size();
        check_cell_size(cell, &mut warnings);
        let columns = snow_columns(min, max, cell, quality.terrain_max_resolution());
        let mut tops = vec![0.0_f32; columns.len()];
        let base = Arc::clone(&volume.field);
        scan_columns(&*base, &columns, 0..columns.len(), &mut tops);
        let mut state = VoxelGroundState {
            base,
            installed: Arc::clone(&volume.field),
            columns,
            extent_m: columns.grid.extent(),
            top: tops.into(),
            snow: vec![0.0_f32; columns.len()].into(),
            terrain: None,
            tops_dirty: false,
            fresh: true,
            own_marks: HashSet::new(),
            last_depth: None,
            last_pack_m: f32::NAN,
            warned_outside: false,
            swaps: 0,
            columns_rescanned: columns.len() as u64,
        };
        install(&mut volume, &mut state);
        commands.entity(entity).insert(state);
    }
}

/// User edits: adopt a field the user replaced, rescan the columns under the chunks marked
/// `NeedsRemesh` since the last frame, write the new tops, and mark the chunks the moved slabs
/// touch, all before this frame's meshing.
#[allow(clippy::too_many_arguments)]
pub(crate) fn voxel_ground_edits(
    mut commands: Commands,
    mut volumes: Query<(Entity, &mut VoxelVolume, &mut VoxelGroundState)>,
    marked: Query<(Entity, &VoxelChunk), Changed<NeedsRemesh>>,
    chunks: Query<(Entity, &VoxelChunk)>,
) {
    for (entity, mut volume, mut state) in &mut volumes {
        let own = std::mem::take(&mut state.own_marks);
        if state.fresh {
            state.fresh = false;
            continue;
        }
        let adopted = !Arc::ptr_eq(&volume.field, &state.installed);
        if adopted {
            state.base = Arc::clone(&volume.field);
        }
        // The chunk columns the user marked.
        let footprints: HashSet<(i32, i32)> = marked
            .iter()
            .filter(|(e, c)| c.volume == entity && !own.contains(e))
            .map(|(_, c)| (c.id.coords[0], c.id.coords[2]))
            .collect();

        if footprints.is_empty() && !adopted {
            continue;
        }
        let columns = state.columns;
        let mut tops = state.top.to_vec();
        let mut scratch: Option<Vec<f32>> = None;
        let mut snow: Option<Vec<f32>> = None;
        let mut dirty = DirtyChunks::default();
        let mut scanned = 0_u64;
        let mut moved = false;
        for (cx, cz) in footprints {
            let id = ChunkId::new([cx, 0, cz]);
            let Some(range) = columns.columns_in(&volume.layout, id) else {
                if !state.warned_outside {
                    warn!(
                        "a chunk of {entity} lies outside the extent the voxel ground was built for: it gets no snow (the extent is frozen at the first scan)"
                    );
                    state.warned_outside = true;
                }
                continue;
            };
            let indices: Vec<usize> = box_indices(columns.grid.size, range).collect();
            let fresh = scratch.get_or_insert_with(|| vec![0.0_f32; columns.len()]);
            scanned += scan_columns(&*state.base, &columns, indices.iter().copied(), fresh);
            for index in indices {
                let (old, new) = (tops[index], fresh[index]);
                if (old - new).abs() <= TOP_EPSILON_M {
                    continue;
                }
                let held = state.snow[index];
                tops[index] = new;
                moved = true;
                // A column that lost its top holds no snow.
                let kept = if columns.has_top(new) { held } else { 0.0 };
                if kept != held {
                    snow.get_or_insert_with(|| state.snow.to_vec())[index] = kept;
                }
                if let Some(span) = slab_span(&columns, (old, held), (new, kept)) {
                    dirty.add(&volume.layout, &columns, index, span);
                }
            }
        }
        state.columns_rescanned += scanned;
        if !adopted && !moved {
            continue;
        }
        if moved {
            state.top = tops.into();
            state.tops_dirty = true;
        }
        if let Some(snow) = snow {
            state.snow = snow.into();
        }
        install(&mut volume, &mut state);
        let entities = chunk_entities(entity, chunks.iter());
        mark(&mut commands, &mut state, &entities, dirty.into_ids());
    }
}

/// Swap the held snow when a readback differs from it by more than half a voxel in any column:
/// per-column sample-and-hold, a new field, and the chunks derived from the changed columns.
pub(crate) fn voxel_ground_snow(
    mut commands: Commands,
    snapshot: Res<EnvironmentSnapshot>,
    units: Res<ElementsUnits>,
    mut volumes: Query<(Entity, &mut VoxelVolume, &mut VoxelGroundState)>,
    chunks: Query<(Entity, &VoxelChunk)>,
) {
    let Some(cover) = snapshot.cover.as_ref() else {
        return;
    };
    for (entity, mut volume, mut state) in &mut volumes {
        // A `VoxelSnow` volume samples the map at its own columns (`voxel_snow_refresh`).
        if state.terrain.is_none() {
            continue;
        }
        if state
            .last_depth
            .as_ref()
            .is_some_and(|last| Arc::ptr_eq(last, &cover.depth))
        {
            continue;
        }
        let columns = state.columns;
        let grid = columns.grid;
        // The map is in the metre frame, the columns in field (= world) units.
        let u = units.per_metre();
        let (origin_m, cell_m, extent_m) = (grid.origin / u, grid.cell_m / u, grid.extent() / u);
        // A snapshot of another terrain (a rebuild in flight) is not ours.
        let same_box = cover.grid.origin.distance(origin_m) < 1e-3 * cell_m
            && cover.grid.extent().distance(extent_m) < 1e-3 * cell_m;
        if !same_box || cover.depth.len() != cover.grid.size.x as usize * cover.grid.size.y as usize
        {
            continue;
        }
        state.last_depth = Some(Arc::clone(&cover.depth));
        // Snow depth in metres from the map, in field units for the slab.
        let fresh = |i: usize| {
            u * if cover.grid.size == grid.size {
                cover.depth[i]
            } else {
                cover.grid.sample(&cover.depth, columns.centre(i) / u)
            }
        };
        let Some(swap) = hold_swap(&columns, &state.top, &state.snow, fresh) else {
            continue;
        };
        let mut dirty = DirtyChunks::default();
        for &(index, old, new) in &swap.changed {
            if let Some(span) = snow_span(&columns, state.top[index], old, new) {
                dirty.add(&volume.layout, &columns, index, span);
            }
        }
        state.snow = swap.snow.into();
        state.swaps += 1;
        install(&mut volume, &mut state);
        let entities = chunk_entities(entity, chunks.iter());
        mark(&mut commands, &mut state, &entities, dirty.into_ids());
    }
}

/// A `VoxelSnow` volume's snow: at each column's world position, the ground-cover map's depth
/// where a map covers it, the point snowpack of [`SurfaceWeathering`] elsewhere, times the top's
/// shed-slope exposure, in field units; per-column sample-and-hold as for a voxel ground.
/// Refreshed when a new map readback arrived or the pack moved by more than a quarter voxel.
pub(crate) fn voxel_snow_refresh(
    mut commands: Commands,
    snapshot: Res<EnvironmentSnapshot>,
    units: Res<ElementsUnits>,
    ground: Option<Res<SurfaceWeathering>>,
    settings: Option<Res<SnowSettings>>,
    mut volumes: Query<
        (
            Entity,
            &mut VoxelVolume,
            &mut VoxelGroundState,
            &GlobalTransform,
        ),
        With<VoxelSnow>,
    >,
    chunks: Query<(Entity, &VoxelChunk)>,
) {
    let u = units.per_metre();
    let pack_m = ground.map_or(0.0, |g| g.pack_seed().x);
    let shed_deg = settings.map_or(SnowSettings::default().shed_slope_deg, |s| s.shed_slope_deg);
    let cover = snapshot.cover.as_ref();
    for (entity, mut volume, mut state, transform) in &mut volumes {
        if state.terrain.is_some() {
            continue;
        }
        let Some(scale) = upright_scale(transform) else {
            continue;
        };
        let columns = state.columns;
        let new_map = match (cover, &state.last_depth) {
            (Some(c), Some(last)) => !Arc::ptr_eq(&c.depth, last),
            (Some(_), None) => true,
            (None, _) => false,
        };
        // `last_pack_m` is NaN until the first refresh.
        let pack_moved = state.last_pack_m.is_nan()
            || (pack_m - state.last_pack_m).abs() > 0.25 * columns.cell * scale / u;
        if !new_map && !pack_moved {
            continue;
        }
        state.last_depth = cover.map(|c| Arc::clone(&c.depth));
        state.last_pack_m = pack_m;
        let tops = Arc::clone(&state.top);
        let fresh = |i: usize| {
            let top = tops[i];
            if !columns.has_top(top) {
                return 0.0;
            }
            let local = columns.centre(i);
            let world = transform.transform_point(Vec3::new(local.x, top, local.y));
            let at_m = units.point_to_metres(world);
            let at = Vec2::new(at_m.x, at_m.z);
            let source_m = cover
                .filter(|c| c.grid.contains(at))
                .and_then(|c| c.depth_at(at))
                .unwrap_or(pack_m);
            // Metres of snow in the volume's field units: world units, then the volume's scale.
            source_m * top_exposure(&columns, &tops, i, shed_deg) * u / scale
        };
        let Some(swap) = hold_swap(&columns, &tops, &state.snow, fresh) else {
            continue;
        };
        let mut dirty = DirtyChunks::default();
        for &(index, old, new) in &swap.changed {
            if let Some(span) = snow_span(&columns, tops[index], old, new) {
                dirty.add(&volume.layout, &columns, index, span);
            }
        }
        state.snow = swap.snow.into();
        state.swaps += 1;
        install(&mut volume, &mut state);
        let entities = chunk_entities(entity, chunks.iter());
        mark(&mut commands, &mut state, &entities, dirty.into_ids());
    }
}

/// Write the column tops to the terrain's `bedrock` buffer once the terrain is live (the import
/// runs once, from the image, and would overwrite an earlier write). Nothing on the GPU writes
/// bedrock afterwards: erosion is off for voxel ground.
pub(crate) fn voxel_ground_sync(
    mut states: Query<&mut VoxelGroundState>,
    terrain: Option<Res<TerrainState>>,
    gpu: Option<ResMut<TerrainGpu>>,
    mut buffers: ResMut<Assets<ShaderBuffer>>,
    units: Res<ElementsUnits>,
) {
    let (Some(terrain), Some(mut gpu)) = (terrain, gpu) else {
        return;
    };
    for mut state in &mut states {
        if !state.tops_dirty
            || state.terrain != Some(terrain.entity)
            || terrain.grid.size != state.columns.grid.size
            || !terrain.is_live()
        {
            continue;
        }
        let Some(mut buffer) = buffers.get_mut(&terrain.bedrock) else {
            continue;
        };
        // The terrain's heights are metres; the tops are field (= world) units.
        let heights_m: Vec<f32> = state.top.iter().map(|top| units.metres(*top)).collect();
        buffer.clear();
        buffer.extend_from_slice(&heights_m);
        gpu.changes += 1;
        state.tops_dirty = false;
    }
}

/// Erosion is off for voxel ground: it would rewrite `bedrock` on the GPU while the volume field
/// (the truth) does not change. Says so once when `ErosionSettings::enabled` is on, and tells
/// hydrology to ignore it while a voxel ground exists.
pub(crate) fn voxel_ground_erosion_guard(
    mut commands: Commands,
    erosion: Res<ErosionSettings>,
    grounds: Query<(), With<VoxelGround>>,
    locked: Option<Res<ErosionLocked>>,
    mut warned: Local<bool>,
) {
    if grounds.is_empty() {
        if locked.is_some() {
            commands.remove_resource::<ErosionLocked>();
        }
        return;
    }
    if locked.is_none() {
        commands.insert_resource(ErosionLocked);
    }
    if should_warn_erosion(erosion.enabled, &mut warned) {
        warn!(
            "ErosionSettings is ignored for voxel ground: the volume field is the truth and erosion would move only the terrain under it"
        );
    }
}

/// Whether to log the erosion warning now: when erosion is on and it has not been said yet.
pub(crate) fn should_warn_erosion(erosion_on: bool, warned: &mut bool) -> bool {
    let warn = erosion_on && !*warned;
    *warned |= warn;
    warn
}
