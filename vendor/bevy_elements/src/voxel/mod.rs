//! Volumetric snow on [`bevy_isomesh`] voxel volumes (feature `isomesh`). The snow maps of the
//! ground cover run on a heightfield, so a voxel volume gets one of its own: [`VoxelGround`]
//! scans the top of every column of a `VoxelVolume` and spawns a crate-internal
//! [`Terrain`](crate::Terrain) with [`TerrainRender::External`](crate::TerrainRender::External)
//! over the volume's XZ extent whose heights are those tops. Hydrology, snow, wetness, wind and
//! rain then run on it unchanged, and [`SnowyField`] puts the snow depth back into the volume
//! as a slab over the column tops, so the mesh grows a snow cap and a tunnel under the hill
//! stays dark. See `docs/research/voxel-snow.md`.
//!
//! [`VoxelSnow`] puts the same slab on any number of volumes without a terrain, from the cover
//! map of a terrain under them or the point snowpack (`docs/research/voxel-volumes.md`).
//!
//! The frame: the user's edit systems, then [`VoxelGroundSystems`] (the first scan, the edit
//! pass, the snow swap), then bevy_isomesh's `IsomeshSystems`: anything the ground marks
//! `NeedsRemesh` is meshed in the same frame, once, against the new field.

mod field;
pub(crate) mod model;
mod systems;
#[cfg(test)]
mod tests;

#[cfg(test)]
mod gpu_tests;

use bevy::prelude::*;
use bevy_isomesh::IsomeshSystems;

pub use field::SnowyField;
use systems::{
    voxel_ground_build, voxel_ground_edits, voxel_ground_erosion_guard, voxel_ground_maintain,
    voxel_ground_snow, voxel_ground_sync, voxel_snow_build, voxel_snow_refresh,
};

use crate::cover::GroundCoverPlugin;
use crate::hydrology::HydrologyPlugin;
use crate::plugin::ElementsSystems;
use crate::terrain::TerrainPlugin;

/// Snow, wetness and hydrology for a `VoxelVolume`: put it on the volume's entity.
///
/// The volume keeps its own `field`; this crate wraps it in a [`SnowyField`] and swaps that in
/// as the snow changes, so every edit to the base field made through
/// [`VoxelVolume::replace_field`](bevy_isomesh::VoxelVolume::replace_field) is seen (it must be
/// ordered `.before(`[`VoxelGroundSystems`]`)` together with the `NeedsRemesh` marks of the
/// chunks it changed).
///
/// A volume with a `VoxelGround` owns the crate's [`Terrain`](crate::Terrain): spawn no other
/// (one terrain exists at a time) and no second `VoxelGround`. Its grid covers the volume's
/// chunk entities as they are at the first scan; chunks that appear later outside that box get
/// no snow. The chunk entities need identity transforms and no parent offset (field space is
/// world space), and the voxel cell size must be a power of two, as isomesh requires. The field is
/// in world units ([`ElementsUnits`](crate::ElementsUnits)): the terrain the crate spawns under
/// it has its extent and heights in metres (the field's divided by the units), and the snow is
/// laid on the slab in world units.
///
/// [`ErosionSettings::enabled`](crate::ErosionSettings) is ignored for voxel ground (one
/// warning is logged when it is on): the volume's field is the truth and erosion would rewrite
/// only the heightfield under it. Water still flows, and snow, wetness and rain run as on any
/// terrain.
///
/// A ground material on the volume's meshes must gate its snow and wetness by depth below the
/// column top, or a tunnel's walls are shaded white wherever the map above them has snow: see
/// [`GroundExtension::with_column_gate_m`](crate::GroundExtension::with_column_gate_m).
#[derive(Component, Clone, Copy, Debug, PartialEq, Eq, Reflect)]
pub struct VoxelGround {
    /// The terrain grid in cells (clamped to the quality level and rounded down to a power of
    /// two, 32 to [`ElementsQuality::terrain_max_resolution`](crate::ElementsQuality)). Choose
    /// the volume's XZ extent divided by its voxel cell size, rounded to a power of two, so one
    /// terrain cell is one voxel wide (64 m of 0.5 m voxels is 128).
    pub resolution: UVec2,
}

impl VoxelGround {
    /// A voxel ground with a terrain grid of `resolution` cells.
    pub fn new(resolution: UVec2) -> Self {
        Self { resolution }
    }
}

/// A snow slab on a `VoxelVolume` that is not the ground: put it on any number of volumes (walls,
/// rocks, ruins over a game's own ground). It spawns no terrain and runs no hydrology.
///
/// Each column's depth is the ground-cover map's at the column's world position where a
/// [`Terrain`](crate::Terrain) with [`GroundCoverPlugin`] covers it (so a wall under a
/// [`PrecipitationBlocker`](crate::PrecipitationBlocker) roof stays bare, and the walls melt
/// with the ground), and elsewhere the point snowpack of
/// [`SurfaceWeathering`](crate::SurfaceWeathering) (exposed flat ground), times the shed-slope
/// exposure of the column's top ([`SnowSettings::shed_slope_deg`](crate::SnowSettings)). The
/// slab, the per-column hold, the edit rescans and the field swaps are [`VoxelGround`]'s; the
/// columns are one per voxel over the volume's chunk entities at the first scan (wider, by
/// doubling, past [`ElementsQuality::terrain_max_resolution`](crate::ElementsQuality) a side).
///
/// The chunk entities are children of the volume (as bevy_isomesh documents), so the volume's
/// transform places them; it may translate, turn about the vertical and scale uniformly. A
/// tilted or unevenly scaled volume gets no snow (one warning). A volume with a `VoxelGround`
/// too is a voxel ground. See `docs/research/voxel-volumes.md`.
#[derive(Component, Clone, Copy, Debug, Default, PartialEq, Eq, Reflect)]
pub struct VoxelSnow;

/// The systems that scan, edit and swap the snow of [`VoxelGround`] and [`VoxelSnow`] volumes.
/// Ordered after [`ElementsSystems::Readback`] (they read the snow map's readback) and before
/// bevy_isomesh's `IsomeshSystems`.
///
/// A system that edits a volume's field or marks its chunks `NeedsRemesh` orders itself
/// `.before(VoxelGroundSystems)`: the ground sees the edit in the frame it is made and rescans
/// the column tops under the marked chunks before they are meshed.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VoxelGroundSystems;

/// Snow on voxel volumes: see [`VoxelGround`] and [`VoxelSnow`]. Added by
/// [`ElementsPlugins`](crate::ElementsPlugins) with the feature `isomesh`; it does not add
/// bevy_isomesh's `IsomeshPlugin` (the consumer does). `VoxelGround` needs [`TerrainPlugin`],
/// [`HydrologyPlugin`] and [`GroundCoverPlugin`] to have been added before it, and idles with a
/// warning otherwise; `VoxelSnow` runs without them.
pub struct VoxelPlugin;

impl Plugin for VoxelPlugin {
    fn build(&self, app: &mut App) {
        configure_sets(app);
        if !(app.is_plugin_added::<TerrainPlugin>()
            && app.is_plugin_added::<HydrologyPlugin>()
            && app.is_plugin_added::<GroundCoverPlugin>())
        {
            warn!(
                "VoxelPlugin needs TerrainPlugin, HydrologyPlugin and GroundCoverPlugin added before it for VoxelGround: voxel ground idles (VoxelSnow runs)"
            );
            add_snow_systems(app);
            return;
        }
        add_systems(app);
    }
}

/// The set's place in the frame.
fn configure_sets(app: &mut App) {
    app.configure_sets(
        Update,
        VoxelGroundSystems
            .after(ElementsSystems::Readback)
            .before(IsomeshSystems),
    );
}

/// The systems, apart from the plugins they need (the unit tests add them to a bare app).
fn add_systems(app: &mut App) {
    app.add_systems(
        Update,
        (
            voxel_ground_maintain,
            voxel_ground_build,
            voxel_snow_build,
            voxel_ground_edits,
            voxel_ground_snow,
            voxel_snow_refresh,
            voxel_ground_sync,
        )
            .chain()
            .in_set(VoxelGroundSystems),
    )
    .add_systems(
        Update,
        voxel_ground_erosion_guard.before(ElementsSystems::Simulation),
    );
}

/// The `VoxelSnow` systems alone, for an app without the terrain, hydrology and cover plugins.
fn add_snow_systems(app: &mut App) {
    app.add_systems(
        Update,
        (
            voxel_ground_maintain,
            voxel_snow_build,
            voxel_ground_edits,
            voxel_snow_refresh,
        )
            .chain()
            .in_set(VoxelGroundSystems),
    );
}
