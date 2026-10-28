//! The voxel ground's systems and the state they keep on the volume entity. All of them run in
//! `Update`, in [`VoxelGroundSystems`](super::VoxelGroundSystems), before bevy_isomesh's meshing
//! systems: whatever they change is meshed in the same frame.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use bevy::asset::RenderAssetUsages;
use bevy::math::Vec2;
use bevy::prelude::*;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};
use bevy::render::storage::ShaderBuffer;
use bevy_isomesh::{NeedsRemesh, VolumeField, VoxelChunk, VoxelVolume};
use isomesh::chunk::ChunkId;

use super::VoxelGround;
use super::field::SnowyField;
use super::model::{
    Columns, DirtyChunks, TOP_EPSILON_M, box_indices, chunk_entities, hold_swap, scan_columns,
    slab_span, snow_span, volume_box,
};
use crate::hydrology::{ErosionLocked, ErosionSettings};
use crate::plugin::ElementsQuality;
use crate::query::EnvironmentSnapshot;
use crate::terrain::model::resolve_resolution;
use crate::terrain::{Terrain, TerrainGpu, TerrainState};

/// What the voxel ground keeps for a volume: the user's field, the tops and the held snow the
/// installed [`SnowyField`] was built from, and the bookkeeping of the edit and swap passes.
/// Inserted on the volume entity once the first column scan is done.
#[derive(Component)]
pub(crate) struct VoxelGroundState {
    /// The user's field: what the slab is added to.
    pub base: Arc<dyn VolumeField>,
    /// The `SnowyField` this module put in `VoxelVolume::field`; a different `Arc` there means
    /// the user replaced the field.
    pub installed: Arc<dyn VolumeField>,
    pub columns: Columns,
    /// The XZ extent and the resolution the terrain was built for.
    pub extent_m: Vec2,
    /// Column tops (world metres) and the held snow depths, on `columns.grid`.
    pub top: Arc<[f32]>,
    pub snow: Arc<[f32]>,
    /// The terrain entity this volume owns.
    pub terrain: Entity,
    /// The tops changed since they were last written to the `bedrock` buffer.
    pub tops_dirty: bool,
    /// Built this frame: the first edit pass has nothing to see (every chunk is new).
    pub fresh: bool,
    /// Chunks this module marked `NeedsRemesh` last frame: their change is not a user edit.
    pub own_marks: HashSet<Entity>,
    /// The last readback's depth map handled by the swap pass.
    pub last_depth: Option<Arc<[f32]>>,
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

/// One-time warnings of the build system.
#[derive(Default)]
pub(crate) struct BuildWarnings {
    other_terrain: bool,
    cell_size: bool,
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

/// A volume that has state but lost its `VoxelGround`.
type WithoutGround = (With<VoxelVolume>, Without<VoxelGround>);

/// Clean up what a volume no longer needs: a terrain whose volume is gone, and the state of a
/// volume that lost its `VoxelGround` (its field is the user's again and every chunk is
/// remeshed without snow).
pub(crate) fn voxel_ground_maintain(
    mut commands: Commands,
    terrains: Query<(Entity, &VoxelGroundTerrain)>,
    mut volumes: Query<(Entity, &mut VoxelVolume, Option<&VoxelGroundState>)>,
    removed: Query<(Entity, &VoxelGroundState), WithoutGround>,
    chunks: Query<(Entity, &VoxelChunk)>,
    grounds: Query<&VoxelGround>,
    quality: Res<ElementsQuality>,
) {
    // A user who changed `resolution` (or a quality level that clamps it differently) gets a
    // terrain, an image and a held-snow grid of the new size: tear down, and `build` runs again.
    let mut stale: HashSet<Entity> = HashSet::new();
    for (entity, _, state) in &volumes {
        let (Some(state), Ok(ground)) = (state, grounds.get(entity)) else {
            continue;
        };
        let (size, _, _) = resolve_resolution(
            state.extent_m,
            ground.resolution,
            quality.terrain_max_resolution(),
        );
        if size != state.columns.grid.size {
            stale.insert(entity);
        }
    }
    let mut torn_down: HashSet<Entity> = removed.iter().map(|(e, _)| e).collect();
    torn_down.extend(stale);
    for entity in &torn_down {
        let Ok((_, mut volume, Some(state))) = volumes.get_mut(*entity) else {
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
            .is_ok_and(|(_, _, state)| state.is_some_and(|s| s.terrain == terrain));
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
    mut images: ResMut<Assets<Image>>,
    mut volumes: Query<(Entity, &mut VoxelVolume, &VoxelGround), Without<VoxelGroundState>>,
    built: Query<(), With<VoxelGroundState>>,
    chunks: Query<(Entity, &VoxelChunk)>,
    terrains: Query<(), With<Terrain>>,
    mut warnings: Local<BuildWarnings>,
) {
    let mut taken = !built.is_empty() || !terrains.is_empty();
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
        if (cell.log2() - cell.log2().round()).abs() > 1e-6 && !warnings.cell_size {
            warn!(
                "the voxel cell size {cell} is not a power of two: isomesh chunk seams close only at powers of two, and the snow slab inherits the cracks"
            );
            warnings.cell_size = true;
        }
        let extent = Vec2::new(max[0] - min[0], max[2] - min[2]);
        let centre = Vec2::new(0.5 * (max[0] + min[0]), 0.5 * (max[2] + min[2]));
        let (size, terrain_cell, _) =
            resolve_resolution(extent, ground.resolution, quality.terrain_max_resolution());
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

        let image = images.add(Image::new(
            Extent3d {
                width: size.x,
                height: size.y,
                depth_or_array_layers: 1,
            },
            TextureDimension::D2,
            bytemuck::cast_slice(&tops).to_vec(),
            TextureFormat::R32Float,
            RenderAssetUsages::default(),
        ));
        let terrain = commands
            .spawn((
                Name::new("Voxel ground"),
                Terrain::from_heightmap(image, 0.0..1.0)
                    .with_extent(extent)
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
            extent_m: extent,
            top: tops.into(),
            snow: vec![0.0_f32; columns.len()].into(),
            terrain,
            tops_dirty: false,
            fresh: true,
            own_marks: HashSet::new(),
            last_depth: None,
            warned_outside: false,
            swaps: 0,
            columns_rescanned: columns.len() as u64,
        };
        install(&mut volume, &mut state);
        commands.entity(entity).insert(state);
        taken = true;
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
    mut volumes: Query<(Entity, &mut VoxelVolume, &mut VoxelGroundState)>,
    chunks: Query<(Entity, &VoxelChunk)>,
) {
    let Some(cover) = snapshot.cover.as_ref() else {
        return;
    };
    for (entity, mut volume, mut state) in &mut volumes {
        if state
            .last_depth
            .as_ref()
            .is_some_and(|last| Arc::ptr_eq(last, &cover.depth))
        {
            continue;
        }
        let columns = state.columns;
        let grid = columns.grid;
        // A snapshot of another terrain (a rebuild in flight) is not ours.
        let same_box = cover.grid.origin.distance(grid.origin) < 1e-3 * grid.cell_m
            && cover.grid.extent().distance(grid.extent()) < 1e-3 * grid.cell_m;
        if !same_box || cover.depth.len() != cover.grid.size.x as usize * cover.grid.size.y as usize
        {
            continue;
        }
        state.last_depth = Some(Arc::clone(&cover.depth));
        let fresh = |i: usize| {
            if cover.grid.size == grid.size {
                cover.depth[i]
            } else {
                cover.grid.sample(&cover.depth, columns.centre(i))
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

/// Write the column tops to the terrain's `bedrock` buffer once the terrain is live (the import
/// runs once, from the image, and would overwrite an earlier write). Nothing on the GPU writes
/// bedrock afterwards: erosion is off for voxel ground.
pub(crate) fn voxel_ground_sync(
    mut states: Query<&mut VoxelGroundState>,
    terrain: Option<Res<TerrainState>>,
    gpu: Option<Res<TerrainGpu>>,
    mut buffers: ResMut<Assets<ShaderBuffer>>,
) {
    let (Some(terrain), Some(gpu)) = (terrain, gpu) else {
        return;
    };
    for mut state in &mut states {
        if !state.tops_dirty
            || terrain.entity != state.terrain
            || terrain.grid.size != state.columns.grid.size
            || !terrain.is_live()
        {
            continue;
        }
        let Some(mut buffer) = buffers.get_mut(&terrain.bedrock) else {
            continue;
        };
        buffer.clear();
        buffer.extend_from_slice(&state.top);
        gpu.dirty.store(true, std::sync::atomic::Ordering::Relaxed);
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
