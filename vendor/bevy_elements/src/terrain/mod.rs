//! Heightfield terrain: the `Terrain` component, its generator or heightmap import, the layer
//! buffers the hydrology and erosion kernels write, the packed render buffer, and the readback
//! behind [`Environment::terrain_height`](crate::Environment::terrain_height). See
//! `docs/research/terrain.md`.
//!
//! One terrain exists at a time (a second `Terrain` entity logs a warning and is ignored). It is
//! centred on its entity's `Transform`: the grid's minimum corner is `translation.xz − extent / 2`
//! and a stored height `h` is the height `translation.y + h` above the entity's origin, all in
//! the metre frame (world ÷ [`ElementsUnits::units_per_metre`](crate::ElementsUnits): the
//! entity's translation is read through it, the patches are written ×units). Rotation and scale
//! are not supported.

mod gpu;
#[cfg(test)]
mod gpu_tests;
pub(crate) mod lod;
mod material;
pub(crate) mod model;
mod textures;

use std::collections::HashMap;
use std::ops::Range;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use crate::gpu::grid::zeroed_buffer;
use crate::gpu::readback::{Readbacks, decode};
use crate::plugin::{
    ElementsFocus, ElementsQuality, ElementsSystems, ElementsUnits, ElementsView, ensure_gpu_core,
};
use crate::query::EnvironmentSnapshot;
use bevy::asset::{RenderAssetUsages, embedded_asset};
use bevy::shader::load_shader_library;
use bevy::camera::primitives::Aabb;
use bevy::camera::visibility::NoAutoAabb;
use bevy::math::UVec4;
use bevy::mesh::{Indices, MeshTag, PrimitiveTopology};
use bevy::pbr::MaterialPlugin;
use bevy::prelude::*;
use bevy::render::render_resource::{ShaderType, TextureFormat};
use bevy::render::storage::ShaderBuffer;
use bevy::transform::TransformSystems;
pub(crate) use gpu::TerrainDispatch;
use lod::{BoundsTree, LodConfig, Patch, select_patches};
pub use material::{
    GroundExtension, GroundMaterial, GroundParams, TerrainExtension, TerrainExtensionKey,
    TerrainLook, TerrainMaterial, TerrainMaterialParams,
};
pub(crate) use material::{
    NO_TERRAIN, insert_placeholder, lod_centre_uniform, lod_uniform, shadow_inputs,
};
use model::{
    F_HALF, GAIN, LACUNARITY, LEAF_CELLS, OCTAVE_ROTATION, R_LOW, R_SPAN, octave_count,
    resolve_resolution, seed_mix,
};
pub use model::{TerrainGrid, TerrainSnapshot};
pub use textures::{GroundLayerTextures, TerrainTextures};

/// Largest readback edge: a larger simulation grid is box-averaged down by 2 first.
const READBACK_MAX_CELLS: u32 = 1024;

/// How a terrain's surface is made.
#[derive(Clone, Debug, PartialEq)]
pub enum TerrainSource {
    /// Generated on the GPU from noise, once, when the terrain spawns.
    Procedural(ProceduralTerrain),
    /// Resampled from the red channel of an image. Heights are
    /// `lerp(height_range_m.start, height_range_m.end, value)` for a value in 0–1.
    ///
    /// The image must be read linearly: a `*Srgb` format (what the image loader yields for an
    /// 8-bit file by default) logs an error and leaves the terrain unspawned. Load the file with
    /// `ImageLoaderSettings { is_srgb: false, expand_grayscale: false, .. }`; 16-bit files are
    /// recommended (an 8-bit map has 256 height steps). Accepted formats: `R8Unorm`,
    /// `Rgba8Unorm`, `R16Unorm`, `Rgba16Unorm`, `R16Uint`, `R32Float`, `Rgba16Float`,
    /// `Rgba32Float`.
    Heightmap {
        /// The height map.
        image: Handle<Image>,
        /// Heights of the lowest and highest values, m.
        height_range_m: Range<f32>,
    },
    /// Level ground at this height, m.
    Flat(f32),
}

/// Parameters of the procedural generator (Musgrave fBm blended with a ridged multifractal).
#[derive(Clone, Copy, Debug, PartialEq, Reflect)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ProceduralTerrain {
    /// Seed of the noise offsets.
    pub seed: u64,
    /// Height of the lowest ground, m. 0 in every preset.
    pub base_height_m: f32,
    /// Height range of one noise component, m. Blending the two components compresses the range
    /// of the sum (`terrain.md`: hills measure 96.7 m of 120).
    pub relief_m: f32,
    /// Wavelength of the base octave, m. The ratio `relief_m / feature_size_m` sets the slopes.
    pub feature_size_m: f32,
    /// Octaves, at most `⌊log₂(feature / 2 cell)⌋ + 1` (the finest wavelength stays above two
    /// cells).
    pub octaves: u32,
    /// Blend of the ridged multifractal into the fBm, 0–1.
    pub ridged: f32,
    /// Sediment laid over the generated surface, m (the surface stays the generated field; the
    /// bedrock is lowered by this depth).
    pub initial_sand_m: f32,
}

impl ProceduralTerrain {
    const fn preset(
        seed: u64,
        feature: f32,
        relief: f32,
        ridged: f32,
        octaves: u32,
        sand: f32,
    ) -> Self {
        Self {
            seed,
            base_height_m: 0.0,
            relief_m: relief,
            feature_size_m: feature,
            octaves,
            ridged,
            initial_sand_m: sand,
        }
    }

    /// Rolling hills: relief 120 m, features 600 m, a little ridging.
    pub const fn hills(seed: u64) -> Self {
        Self::preset(seed, 600.0, 120.0, 0.2, 8, 0.0)
    }

    /// Mountains: relief 900 m, features 1500 m, mostly ridged. Steep: 70 % of the ground starts
    /// above 40° (erosion relaxes it to the talus angle).
    pub const fn mountains(seed: u64) -> Self {
        Self::preset(seed, 1500.0, 900.0, 0.7, 8, 0.0)
    }

    /// Dunes: relief 40 m, features 300 m, smooth, under 4 m of sand.
    pub const fn dunes(seed: u64) -> Self {
        Self::preset(seed, 300.0, 40.0, 0.0, 7, 4.0)
    }

    /// Plains: relief 15 m, features 800 m.
    pub const fn plains(seed: u64) -> Self {
        Self::preset(seed, 800.0, 15.0, 0.0, 8, 0.0)
    }
}

/// Who draws the ground.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TerrainRender {
    /// The crate draws the ground with its own level-of-detail mesh and layer-blending material.
    #[default]
    Owned,
    /// The consumer's meshes are the ground: the crate simulates and shades through
    /// `GroundExtension` only.
    External,
}

/// The terrain. Spawn one entity with it. It is centred on the entity's `Transform` (the grid's
/// minimum corner is `translation.xz − extent / 2` and a stored height `h` is the height
/// `translation.y + h`, both read through [`ElementsUnits`](crate::ElementsUnits): `Terrain`'s
/// own lengths are metres); rotation and scale are ignored. One terrain exists at a time: a second
/// `Terrain` entity logs a warning and is ignored.
#[derive(Component, Clone, Debug, PartialEq)]
#[require(Transform, Visibility)]
pub struct Terrain {
    /// Where the surface comes from.
    pub source: TerrainSource,
    /// World size, m. Default 2048 × 2048.
    pub extent_m: Vec2,
    /// Cells per axis: clamped to
    /// [`ElementsQuality::terrain_max_resolution`] and rounded down to a power of two. Cells
    /// are square: when `extent_m` and `resolution` disagree about the cell size, the coarser
    /// axis's cell is kept and both counts are re-derived from it (with a warning), so neither
    /// exceeds the cap. Default 1024².
    pub resolution: UVec2,
    /// Who draws it.
    pub render: TerrainRender,
}

impl Terrain {
    fn with_source(source: TerrainSource) -> Self {
        Self {
            source,
            extent_m: Vec2::splat(2048.0),
            resolution: UVec2::splat(1024),
            render: TerrainRender::Owned,
        }
    }

    /// A procedurally generated terrain.
    pub fn procedural(procedural: ProceduralTerrain) -> Self {
        Self::with_source(TerrainSource::Procedural(procedural))
    }

    /// A terrain resampled from a height map; see [`TerrainSource::Heightmap`].
    pub fn from_heightmap(image: Handle<Image>, height_range_m: Range<f32>) -> Self {
        Self::with_source(TerrainSource::Heightmap {
            image,
            height_range_m,
        })
    }

    /// Level ground at `height_m`.
    pub fn flat(height_m: f32) -> Self {
        Self::with_source(TerrainSource::Flat(height_m))
    }

    /// Set the world size, m.
    pub fn with_extent(mut self, extent_m: Vec2) -> Self {
        self.extent_m = extent_m;
        self
    }

    /// Set the cells per axis (before clamping to the quality level).
    pub fn with_resolution(mut self, resolution: UVec2) -> Self {
        self.resolution = resolution;
        self
    }

    /// The consumer draws the ground ([`TerrainRender::External`]).
    pub fn external(mut self) -> Self {
        self.render = TerrainRender::External;
        self
    }
}

/// The terrain's GPU layers and where they sit. Exists while a [`Terrain`] does.
///
/// Every layer is a row-major per-cell storage buffer on [`grid`](Self::grid) (row index along
/// world +Z). The surface height is `bedrock + sediment`; the packed `cells` and `flow_cells`
/// are what materials and other kernels read. Hydrology and erosion write the simulation layers;
/// everything else treats them as read-only.
#[derive(Resource, Clone, Debug)]
pub struct TerrainState {
    /// The terrain entity.
    pub entity: Entity,
    /// Placement of the simulation grid.
    pub grid: TerrainGrid,
    /// Bedrock height, `array<f32>`, m above `grid.y0`.
    pub bedrock: Handle<ShaderBuffer>,
    /// Loose sediment depth, `array<f32>`, m.
    pub sediment: Handle<ShaderBuffer>,
    /// Water depth above the surface, `array<f32>`, m.
    pub water: Handle<ShaderBuffer>,
    /// Outflow fluxes `(left, right, top, bottom)`, `array<vec4<f32>>`, m³/s.
    pub flux: Handle<ShaderBuffer>,
    /// Depth-averaged water velocity, `array<vec2<f32>>`, m/s.
    pub velocity: Handle<ShaderBuffer>,
    /// Suspended sediment, `array<f32>`, height-equivalent m.
    pub suspended: Handle<ShaderBuffer>,
    /// Scratch of the flux kernel, `array<vec2<f32>>`.
    pub transport: Handle<ShaderBuffer>,
    /// Water source map, `array<vec2<f32>>`.
    pub source_map: Handle<ShaderBuffer>,
    /// Thermal erosion fluxes, a ping-pong pair of `array<vec4<f32>>`.
    pub thermal_flux: [Handle<ShaderBuffer>; 2],
    /// The packed render buffer, `array<vec4<f32>>`: surface height, sediment depth, water depth,
    /// `bitcast<f32>(pack2x16float(∂h/∂x, ∂h/∂z))`.
    pub cells: Handle<ShaderBuffer>,
    /// The packed water-render buffer, `array<vec4<f32>>`: velocity xy, turbidity, spare.
    pub flow_cells: Handle<ShaderBuffer>,
    /// Surface height and sediment thickness per readback cell, `array<vec2<f32>>`
    /// (`readback_size` cells; the grid box-averaged by 2 above 1024²).
    pub height_readback: Handle<ShaderBuffer>,
    /// Bounds of every 32 × 32-cell leaf, `array<vec4<f32>>`: surface min, surface max, water
    /// surface max, water depth max.
    pub leaf_bounds: Handle<ShaderBuffer>,
    /// Cells of `height_readback` per axis.
    pub readback_size: UVec2,
    /// The generator's `n`-th completed pack, 0 before the terrain exists on the GPU. Rises
    /// whenever the packed buffer was rewritten (hydrology, erosion and edits bump it).
    pub version: Arc<AtomicU32>,
}

impl TerrainState {
    /// Whether the generator and the first pack have run on the GPU.
    pub fn is_live(&self) -> bool {
        self.version.load(Ordering::Relaxed) > 0
    }
}

/// Uniform of the terrain kernels and, embedded, of every shader that reads `terrain_cells`.
/// Mirrors `TerrainParams` of `common.wesl`.
#[derive(Clone, Copy, Debug, Default, PartialEq, ShaderType)]
pub(crate) struct TerrainParamsGpu {
    origin: Vec2,
    cell: f32,
    pad0: f32,
    size: UVec2,
    pad1: UVec2,
    /// x: y0. y: readback downsample. z: live flag.
    misc: Vec4,
}

impl TerrainParamsGpu {
    pub(crate) fn new(grid: &TerrainGrid, downsample: u32, live: bool) -> Self {
        Self {
            origin: grid.origin,
            cell: grid.cell_m,
            pad0: 0.0,
            size: grid.size,
            pad1: UVec2::ZERO,
            misc: Vec4::new(grid.y0, downsample as f32, f32::from(u8::from(live)), 0.0),
        }
    }
}

/// Uniform of `generate.wesl`.
#[derive(Clone, Copy, Debug, Default, PartialEq, ShaderType)]
pub(crate) struct GenParams {
    size: UVec4,
    shape: Vec4,
    base: Vec4,
    consts: Vec4,
    octave: Vec4,
}

/// Uniform of `heightmap.wesl`.
#[derive(Clone, Copy, Debug, Default, PartialEq, ShaderType)]
pub(crate) struct HeightmapParams {
    size: UVec4,
    range: Vec4,
}

/// The source of a terrain's first heights, for the kernels.
#[derive(Clone, Debug)]
pub(crate) enum GenerateSource {
    Procedural(GenParams),
    Heightmap {
        image: Handle<Image>,
        uint: bool,
        params: HeightmapParams,
    },
}

/// The GPU resources of the terrain and this frame's parameters; extracted to the render world
/// every frame.
#[derive(Resource, Clone)]
pub(crate) struct TerrainGpu {
    pub state: TerrainState,
    pub params: TerrainParamsGpu,
    pub source: GenerateSource,
    /// Bumped by anything that changed bedrock, sediment or water this frame; the render world
    /// runs the pack, bounds and readback kernels while it is ahead of `packed`. Extracted by
    /// value, so a later frame's bump never reaches an earlier render frame. Starts at 1: the
    /// generation itself is a change.
    pub changes: u64,
    /// The `changes` value the render world last packed.
    pub packed: Arc<std::sync::atomic::AtomicU64>,
    /// Set by the render world once the generator has run.
    pub generated: Arc<std::sync::atomic::AtomicBool>,
}

/// Entities per patch pool: the measured worst case of the selection (187 on flat Ultra, 133 at
/// High) plus margin (`terrain.md` Deviation 9).
pub(crate) const POOL_CAPACITY: usize = 256;

/// One entity of the patch pool and the patch it currently draws.
struct PoolSlot {
    entity: Entity,
    patch: Option<Patch>,
}

/// Main-world bookkeeping of the active terrain.
#[derive(Resource, Default)]
pub(crate) struct TerrainRuntime {
    /// What the current resources were built from.
    active: Option<(Entity, Terrain, ElementsQuality)>,
    /// A terrain that failed validation: not retried until it changes.
    failed: Option<Terrain>,
    warned_extra: bool,
    warned_transform: bool,
    warned_capacity: bool,
    /// The quadtree geometry of the active terrain.
    pub(crate) lod: Option<LodConfig>,
    /// Min and max heights of every node, from the last bounds readback (before the first, the
    /// generator's analytic range).
    bounds: Option<BoundsTree>,
    bounds_version: u32,
    /// The owned render's patch pool, empty for an external terrain.
    pool: Vec<PoolSlot>,
    patch_mesh: Option<Handle<Mesh>>,
    /// The patches the last selection chose (the owned pool's contents, or what it would hold
    /// for an external terrain): the water pool draws the wet ones.
    pub(crate) selected: Vec<Patch>,
    /// Surface min and water-surface max of every node, for the water patches' boxes.
    pub(crate) water_box: Option<BoundsTree>,
    /// Water depth max of every node (both entries): `get(..).1` is the deepest cell's depth.
    pub(crate) water_depth: Option<BoundsTree>,
    /// What the patches are selected and morphed about under an orthographic followed camera,
    /// metre frame (`docs/research/orthographic-lod.md` Model 1); `None`: the eye.
    pub(crate) lod_centre: Option<Vec3>,
}

impl TerrainRuntime {
    /// Whether the crate draws the active terrain with its own patches (not a consumer's mesh).
    pub(crate) fn draws_ground(&self) -> bool {
        self.active
            .as_ref()
            .is_some_and(|(_, terrain, _)| terrain.render == TerrainRender::Owned)
    }

    /// The point the patches are selected and morphed about for a followed camera at `eye`
    /// (metre frame).
    pub(crate) fn morph_centre(&self, eye: Vec3) -> Vec3 {
        self.lod_centre.unwrap_or(eye)
    }
}

/// Heightmap formats the generator reads linearly, and whether the kernel is the integer one.
fn heightmap_format(format: TextureFormat) -> Option<bool> {
    match format {
        TextureFormat::R8Unorm
        | TextureFormat::Rgba8Unorm
        | TextureFormat::R16Unorm
        | TextureFormat::Rgba16Unorm
        | TextureFormat::R32Float
        | TextureFormat::Rgba16Float
        | TextureFormat::Rgba32Float => Some(false),
        TextureFormat::R16Uint => Some(true),
        _ => None,
    }
}

/// Heightfield terrain with GPU-generated layers.
///
/// Spawn a [`Terrain`]; the layers are created and generated on the next frame and
/// [`TerrainState`] appears. Needs an [`ElementsCamera`](crate::ElementsCamera) only for the
/// level-of-detail selection of the owned render.
pub struct TerrainPlugin;

impl Plugin for TerrainPlugin {
    fn build(&self, app: &mut App) {
        ensure_gpu_core(app);
        embedded_asset!(app, "generate.wesl");
        embedded_asset!(app, "heightmap.wesl");
        embedded_asset!(app, "pack.wesl");
        embedded_asset!(app, "flow.wesl");
        embedded_asset!(app, "render.wesl");
        embedded_asset!(app, "ground.wesl");
        // The ground's detail textures' hex-tiling, imported by `render.wesl`.
        load_shader_library!(app, "textures.wesl");
        app.add_plugins((
            MaterialPlugin::<TerrainMaterial>::default(),
            MaterialPlugin::<GroundMaterial>::default(),
        ))
        .init_resource::<TerrainRuntime>()
        .init_resource::<textures::GroundTexturePack>()
        .init_resource::<TerrainLook>()
        .add_systems(
            Update,
            (ensure_terrain, update_lod_centre).in_set(ElementsSystems::Simulation),
        )
        .add_systems(
            Update,
            (
                read_terrain.in_set(ElementsSystems::Readback),
                textures::pack_terrain_textures,
                material::update_materials,
            )
                .chain()
                .after(ElementsSystems::Simulation),
        )
        .add_systems(
            PostUpdate,
            update_patches.before(TransformSystems::Propagate),
        );
        gpu::plugin(app);
    }
}

#[allow(clippy::too_many_arguments)]
fn ensure_terrain(
    mut commands: Commands,
    quality: Res<ElementsQuality>,
    terrains: Query<(Entity, &Terrain, &Transform, Option<&GlobalTransform>)>,
    units: Res<ElementsUnits>,
    asset_server: Res<AssetServer>,
    images: Res<Assets<Image>>,
    state: Option<ResMut<TerrainState>>,
    gpu: Option<ResMut<TerrainGpu>>,
    mut runtime: ResMut<TerrainRuntime>,
    mut snapshot: ResMut<EnvironmentSnapshot>,
    mut buffers: ResMut<Assets<ShaderBuffer>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<TerrainMaterial>>,
) {
    // The terrain in use: the one already built, else the lowest entity.
    let active_entity = runtime
        .active
        .as_ref()
        .map(|a| a.0)
        .filter(|e| terrains.contains(*e))
        .or_else(|| terrains.iter().map(|t| t.0).min());
    let Some((entity, terrain, local, global)) = active_entity.and_then(|e| terrains.get(e).ok())
    else {
        if state.is_some() {
            commands.remove_resource::<TerrainState>();
            commands.remove_resource::<TerrainGpu>();
            snapshot.terrain = None;
            snapshot.terrain_leaves = None;
        }
        clear_pool(&mut commands, &mut runtime);
        runtime.active = None;
        return;
    };
    if terrains.iter().count() > 1 && !runtime.warned_extra {
        warn!(
            "more than one Terrain entity exists; only {entity} is used and the others are ignored"
        );
        runtime.warned_extra = true;
    }
    let transform = global.map_or(*local, |g| g.compute_transform());
    // The grid lives in the metre frame; the entity's translation is in world units.
    let translation = units.point_to_metres(transform.translation);
    if !runtime.warned_transform
        && (transform.rotation.angle_between(Quat::IDENTITY) > 1e-4
            || (transform.scale - Vec3::ONE).abs().max_element() > 1e-4)
    {
        warn!(
            "a Terrain's rotation and scale are ignored: its grid is axis-aligned and its heights are in metres"
        );
        runtime.warned_transform = true;
    }

    let wanted = (entity, terrain.clone(), *quality);
    let max = quality.terrain_max_resolution();
    let (size, cell, adjusted) = resolve_resolution(terrain.extent_m, terrain.resolution, max);
    // The grid follows the transform every frame; the layers are rebuilt only when what they
    // were built from changes.
    let grid = TerrainGrid {
        origin: translation.xz() - size.as_vec2() * cell * 0.5,
        cell_m: cell,
        size,
        y0: translation.y,
    };
    if runtime.active.as_ref() == Some(&wanted) {
        if let (Some(mut state), Some(mut gpu)) = (state, gpu) {
            state.grid = grid;
            gpu.state.grid = grid;
            gpu.params = TerrainParamsGpu::new(&grid, readback_downsample(size), true);
        }
        return;
    }
    if runtime.failed.as_ref() == Some(terrain) {
        return;
    }

    let source = match &terrain.source {
        TerrainSource::Procedural(p) => {
            let octaves = octave_count(p.octaves, p.feature_size_m, cell);
            GenerateSource::Procedural(GenParams {
                size: UVec4::new(size.x, size.y, octaves, seed_mix(p.seed)),
                shape: Vec4::new(
                    cell,
                    p.feature_size_m.max(cell),
                    p.relief_m,
                    p.ridged.clamp(0.0, 1.0),
                ),
                base: Vec4::new(p.base_height_m, p.initial_sand_m.max(0.0), 0.0, 0.0),
                consts: Vec4::new(F_HALF, R_LOW, R_SPAN, GAIN),
                octave: Vec4::new(LACUNARITY, OCTAVE_ROTATION, 0.0, 0.0),
            })
        }
        TerrainSource::Flat(height) => GenerateSource::Procedural(GenParams {
            size: UVec4::new(size.x, size.y, 1, 0),
            shape: Vec4::new(cell, 1000.0, 0.0, 0.0),
            base: Vec4::new(*height, 0.0, 0.0, 0.0),
            consts: Vec4::new(F_HALF, R_LOW, R_SPAN, GAIN),
            octave: Vec4::new(LACUNARITY, OCTAVE_ROTATION, 0.0, 0.0),
        }),
        TerrainSource::Heightmap {
            image,
            height_range_m,
        } => {
            use bevy::asset::LoadState;
            match asset_server.load_state(image) {
                LoadState::Failed(error) => {
                    error!(
                        "terrain heightmap {:?} failed to load, the terrain stays unspawned: {error}",
                        image.path()
                    );
                    runtime.failed = Some(terrain.clone());
                    return;
                }
                LoadState::Loaded => {}
                // Still loading (or a handle that was added to the assets directly).
                _ if images.get(image).is_none() => return,
                _ => {}
            }
            let Some(loaded) = images.get(image) else {
                return;
            };
            let format = loaded.texture_descriptor.format;
            let Some(uint) = heightmap_format(format) else {
                error!(
                    "terrain heightmap {:?} has format {format:?}, which is not read linearly; the terrain stays unspawned. Load it with ImageLoaderSettings {{ is_srgb: false, expand_grayscale: false, .. }} (and prefer 16 bit)",
                    image.path()
                );
                runtime.failed = Some(terrain.clone());
                return;
            };
            if matches!(format, TextureFormat::R8Unorm | TextureFormat::Rgba8Unorm)
                && (height_range_m.end - height_range_m.start).abs() > 100.0
            {
                info!(
                    "terrain heightmap {:?} is 8 bit: its {:.0} m range has 256 height steps; a 16-bit map is recommended",
                    image.path(),
                    (height_range_m.end - height_range_m.start).abs()
                );
            }
            GenerateSource::Heightmap {
                image: image.clone(),
                uint,
                params: HeightmapParams {
                    size: UVec4::new(size.x, size.y, 0, 0),
                    range: Vec4::new(height_range_m.start, height_range_m.end, 0.0, 0.0),
                },
            }
        }
    };
    if adjusted {
        warn!(
            "terrain extent {:?} and resolution {:?} give non-square cells; the grid is re-derived as {} for {:.3} m cells",
            terrain.extent_m, terrain.resolution, size, cell
        );
    }

    let cells = u64::from(size.x) * u64::from(size.y);
    let ds = readback_downsample(size);
    let readback_size = size / ds;
    let leaves = u64::from(size.x / LEAF_CELLS) * u64::from(size.y / LEAF_CELLS);
    let mut layer = |label: &'static str, bytes_per_cell: u64| {
        zeroed_buffer(&mut buffers, label, cells * bytes_per_cell)
    };
    let new_state = TerrainState {
        entity,
        grid,
        bedrock: layer("elements_terrain_bedrock", 4),
        sediment: layer("elements_terrain_sediment", 4),
        water: layer("elements_terrain_water", 4),
        flux: layer("elements_terrain_flux", 16),
        velocity: layer("elements_terrain_velocity", 8),
        suspended: layer("elements_terrain_suspended", 4),
        transport: layer("elements_terrain_transport", 8),
        source_map: layer("elements_terrain_source_map", 8),
        thermal_flux: [
            layer("elements_terrain_thermal_flux_a", 16),
            layer("elements_terrain_thermal_flux_b", 16),
        ],
        cells: layer("elements_terrain_cells", 16),
        flow_cells: layer("elements_terrain_flow_cells", 16),
        height_readback: zeroed_buffer(
            &mut buffers,
            "elements_terrain_height_readback",
            u64::from(readback_size.x) * u64::from(readback_size.y) * 8,
        ),
        leaf_bounds: zeroed_buffer(&mut buffers, "elements_terrain_leaf_bounds", leaves * 16),
        readback_size,
        version: Arc::new(AtomicU32::new(0)),
    };
    commands.insert_resource(TerrainGpu {
        state: new_state.clone(),
        params: TerrainParamsGpu::new(&grid, ds, true),
        source,
        changes: 1,
        packed: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        generated: Arc::new(std::sync::atomic::AtomicBool::new(false)),
    });
    let cells_handle = new_state.cells.clone();
    commands.insert_resource(new_state);
    snapshot.terrain = None;
    snapshot.terrain_leaves = None;
    clear_pool(&mut commands, &mut runtime);
    let lod = LodConfig::new(size.as_vec2() * cell, cell);
    let (low, high) = analytic_range(&terrain.source);
    runtime.bounds = Some(BoundsTree::uniform(&lod, low, high));
    runtime.water_box = Some(BoundsTree::uniform(&lod, low, high));
    runtime.water_depth = Some(BoundsTree::uniform(&lod, 0.0, 0.0));
    runtime.selected.clear();
    runtime.bounds_version = 0;
    runtime.lod = Some(lod);
    if terrain.render == TerrainRender::Owned {
        let mesh = runtime
            .patch_mesh
            .get_or_insert_with(|| meshes.add(patch_mesh()))
            .clone();
        let material = materials.add(TerrainMaterial {
            base: StandardMaterial {
                base_color: Color::WHITE,
                perceptual_roughness: 0.9,
                metallic: 0.0,
                ..default()
            },
            extension: TerrainExtension {
                params: TerrainMaterialParams::default(),
                cells: cells_handle,
                cover: NO_TERRAIN,
                cloud_shadow: None,
                trail: None,
                ground: None,
                path: None,
            },
        });
        // Hidden until the first selection puts a patch in a slot.
        let analytic = Aabb::from_min_max(
            Vec3::new(0.0, units.world(low - 2.0), 0.0),
            Vec3::new(1.0, units.world(high + 2.0), 1.0),
        );
        let mut slots = Vec::with_capacity(POOL_CAPACITY);
        commands.entity(entity).with_children(|parent| {
            for _ in 0..POOL_CAPACITY {
                let entity = parent
                    .spawn((
                        Name::new("Terrain patch"),
                        Mesh3d(mesh.clone()),
                        MeshMaterial3d(material.clone()),
                        Transform::default(),
                        MeshTag::new(0),
                        analytic,
                        NoAutoAabb,
                        Visibility::Hidden,
                    ))
                    .id();
                slots.push(PoolSlot {
                    entity,
                    patch: None,
                });
            }
        });
        runtime.pool = slots;
    }
    runtime.active = Some(wanted);
    runtime.failed = None;
}

/// Despawn the patch pool (the terrain's children), if any.
fn clear_pool(commands: &mut Commands, runtime: &mut TerrainRuntime) {
    for slot in runtime.pool.drain(..) {
        commands.entity(slot.entity).despawn();
    }
}

/// The range of surface heights the source can produce before the first bounds readback, m.
fn analytic_range(source: &TerrainSource) -> (f32, f32) {
    match source {
        TerrainSource::Procedural(p) => {
            let (a, b) = (p.base_height_m, p.base_height_m + p.relief_m);
            (a.min(b), a.max(b))
        }
        TerrainSource::Heightmap { height_range_m, .. } => (
            height_range_m.start.min(height_range_m.end),
            height_range_m.start.max(height_range_m.end),
        ),
        TerrainSource::Flat(h) => (*h, *h),
    }
}

/// The patch every pool entity shares: a unit square of 32 × 32 quads (33 × 33 vertices,
/// `u16` indices) in the XZ plane. The vertex shader places the vertices; only the grid index
/// (position × 32) matters.
pub(crate) fn patch_mesh() -> Mesh {
    const N: u16 = 32;
    let mut positions = Vec::with_capacity(usize::from((N + 1) * (N + 1)));
    for z in 0..=N {
        for x in 0..=N {
            positions.push([
                f32::from(x) / f32::from(N),
                0.0,
                f32::from(z) / f32::from(N),
            ]);
        }
    }
    let mut indices = Vec::with_capacity(usize::from(N * N * 6));
    for z in 0..N {
        for x in 0..N {
            let i = z * (N + 1) + x;
            indices.extend_from_slice(&[i, i + N + 1, i + 1, i + 1, i + N + 1, i + N + 2]);
        }
    }
    Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::RENDER_WORLD,
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
    .with_inserted_indices(Indices::U16(indices))
}

/// Slots for the `wanted` patches of a pool whose slots currently hold `current`: a patch keeps
/// the slot it already has, the others take the free ones (the farthest are dropped when there
/// are none). Returns the assignment and which slots are in use.
pub(crate) fn assign_slots(
    wanted: Vec<Patch>,
    current: &[Option<Patch>],
) -> (Vec<(Patch, usize)>, Vec<bool>) {
    let slot_of: HashMap<Patch, usize> = current
        .iter()
        .enumerate()
        .filter_map(|(i, patch)| patch.map(|p| (p, i)))
        .collect();
    let mut used = vec![false; current.len()];
    let mut assigned: Vec<(Patch, usize)> = Vec::with_capacity(wanted.len());
    let mut unmatched = Vec::new();
    for patch in wanted {
        match slot_of.get(&patch) {
            Some(&i) => {
                used[i] = true;
                assigned.push((patch, i));
            }
            None => unmatched.push(patch),
        }
    }
    let mut free = (0..used.len())
        .filter(|i| !used[*i])
        .collect::<Vec<_>>()
        .into_iter();
    for patch in unmatched {
        if let Some(i) = free.next() {
            used[i] = true;
            assigned.push((patch, i));
        }
    }
    (assigned, used)
}

/// The transform of a patch relative to the terrain entity: its corner on the vertex lattice
/// (the node's tile half a cell in from the grid's corner, so the vertices sit on the height
/// samples), scaled to its edge.
pub(crate) fn patch_transform(lod: &LodConfig, extent: Vec2, patch: Patch) -> (Vec3, Vec3) {
    let size = lod.node_m(patch.level);
    let offset = lod.sample_offset();
    (
        Vec3::new(
            patch.ix as f32 * size + offset - extent.x * 0.5,
            0.0,
            patch.iz as f32 * size + offset - extent.y * 0.5,
        ),
        Vec3::new(size, 1.0, size),
    )
}

/// Find what the patches are selected and morphed about this frame
/// (`docs/research/orthographic-lod.md` Model 1). Under a perspective followed camera, the eye
/// (`None`): Bevy's vertex stage morphs about the view's own translation. Under an orthographic
/// one, whose eye stands far back from what it shows, its [`ElementsFocus`], or without one
/// where its forward ray first meets the ground; a view that meets no ground keeps the eye.
pub(crate) fn update_lod_centre(
    mut runtime: ResMut<TerrainRuntime>,
    view: Res<ElementsView>,
    snapshot: Res<EnvironmentSnapshot>,
    cameras: Query<(&Projection, &GlobalTransform, Has<ElementsFocus>)>,
) {
    let centre = view
        .entity
        .and_then(|entity| cameras.get(entity).ok())
        .and_then(|(projection, transform, focused)| {
            if !matches!(projection, Projection::Orthographic(_)) {
                return None;
            }
            // A focus that resolved to the eye (an entity without a transform) is no focus.
            if focused && view.position != view.eye {
                return Some(view.position);
            }
            let snap = snapshot.terrain.as_ref()?;
            let (lo, hi) = runtime.bounds.as_ref()?.extremes();
            snap.ray_hit(
                view.eye,
                transform.forward().as_vec3(),
                snap.grid.y0 + lo,
                snap.grid.y0 + hi,
            )
        });
    if runtime.lod_centre != centre {
        runtime.lod_centre = centre;
    }
}

/// Select the patches for the followed camera and write them into the pool, touching only the
/// slots whose patch, transform or bounds changed. Idles without a camera and keeps the last
/// pool. Runs before transform propagation so the new transforms are drawn this frame.
pub(crate) fn update_patches(
    mut runtime: ResMut<TerrainRuntime>,
    state: Option<Res<TerrainState>>,
    view: Res<ElementsView>,
    units: Res<ElementsUnits>,
    snapshot: Res<EnvironmentSnapshot>,
    mut patches: Query<(&mut Transform, &mut Visibility, &mut MeshTag, &mut Aabb)>,
) {
    let (Some(state), Some(lod)) = (state, runtime.lod) else {
        return;
    };
    if view.entity.is_none() {
        return;
    }
    if let Some(leaves) = snapshot
        .terrain_leaves
        .as_ref()
        .filter(|l| l.version != runtime.bounds_version)
    {
        let tree =
            BoundsTree::from_leaves(&lod, leaves.dims, leaves.bounds.iter().map(|b| (b.x, b.y)));
        // The ranges follow the steepest node of every level.
        runtime.lod = Some(lod.derive(Some(&tree)));
        runtime.bounds = Some(tree);
        runtime.water_box = Some(BoundsTree::from_leaves(
            &lod,
            leaves.dims,
            leaves.bounds.iter().map(|b| (b.x, b.z)),
        ));
        runtime.water_depth = Some(BoundsTree::from_leaves(
            &lod,
            leaves.dims,
            leaves.bounds.iter().map(|b| (b.w, b.w)),
        ));
        runtime.bounds_version = leaves.version;
    }
    let Some(lod) = runtime.lod else {
        return;
    };
    let Some(tree) = runtime.bounds.take() else {
        return;
    };
    let grid = state.grid;
    // The same centre the vertex stage morphs about: the eye, Bevy's `lod_view_world_position`,
    // or the uniform `lod_centre` under an orthographic camera.
    let centre = runtime.morph_centre(view.eye);
    let camera = Vec3::new(
        centre.x - grid.origin.x,
        centre.y - grid.y0,
        centre.z - grid.origin.y,
    );
    let mut wanted = select_patches(&lod, &tree, camera);
    if wanted.len() > POOL_CAPACITY {
        if !runtime.warned_capacity {
            warn!(
                "terrain selection wants {} patches, more than the pool's {POOL_CAPACITY}; the farthest are dropped",
                wanted.len()
            );
            runtime.warned_capacity = true;
        }
        wanted.truncate(POOL_CAPACITY);
    }
    runtime.selected.clone_from(&wanted);
    if runtime.pool.is_empty() {
        runtime.bounds = Some(tree);
        return;
    }
    let current: Vec<Option<Patch>> = runtime.pool.iter().map(|s| s.patch).collect();
    let (assigned, used) = assign_slots(wanted, &current);
    let extent = grid.extent();
    for (i, slot) in runtime.pool.iter_mut().enumerate() {
        if used[i] {
            continue;
        }
        if slot.patch.take().is_some()
            && let Ok((_, mut visibility, _, _)) = patches.get_mut(slot.entity)
            && *visibility != Visibility::Hidden
        {
            *visibility = Visibility::Hidden;
        }
    }
    for (patch, i) in assigned {
        let entity = runtime.pool[i].entity;
        runtime.pool[i].patch = Some(patch);
        let Ok((mut transform, mut visibility, mut tag, mut aabb)) = patches.get_mut(entity) else {
            continue;
        };
        // The patch is placed in the world: its metre-frame corner and edge × units.
        let (translation, scale) = patch_transform(&lod, extent, patch);
        let (translation, scale) = (
            units.point_to_world(translation),
            Vec3::new(units.world(scale.x), scale.y, units.world(scale.z)),
        );
        if transform.translation != translation || transform.scale != scale {
            transform.translation = translation;
            transform.scale = scale;
        }
        if tag.value != patch.level {
            *tag = MeshTag::new(patch.level);
        }
        let (low, high) = tree.get(patch.level, patch.ix, patch.iz);
        let bounds = Aabb::from_min_max(
            Vec3::new(0.0, units.world(low - model::AABB_PAD_M), 0.0),
            Vec3::new(1.0, units.world(high + model::AABB_PAD_M), 1.0),
        );
        if *aabb != bounds {
            *aabb = bounds;
        }
        if *visibility != Visibility::Inherited {
            *visibility = Visibility::Inherited;
        }
    }
    runtime.bounds = Some(tree);
}

/// The factor the readback box-averages the grid down by.
fn readback_downsample(size: UVec2) -> u32 {
    if size.max_element() > READBACK_MAX_CELLS {
        2
    } else {
        1
    }
}

/// Bounds of every leaf patch at the last readback.
#[derive(Clone, Debug)]
pub(crate) struct TerrainLeaves {
    /// The surface version the bounds were read at.
    pub version: u32,
    /// Leaves per axis.
    pub dims: UVec2,
    /// Surface min, surface max, water-surface max, water depth max, per leaf, row-major.
    pub bounds: Vec<Vec4>,
}

/// Copy the height field and the leaf bounds back whenever the GPU's version moved.
fn read_terrain(
    state: Option<Res<TerrainState>>,
    snapshot: Res<EnvironmentSnapshot>,
    mut readbacks: Readbacks,
) {
    let Some(state) = state else {
        return;
    };
    let version = state.version.load(Ordering::Relaxed);
    if version == 0 {
        return;
    }
    if snapshot
        .terrain
        .as_ref()
        .is_none_or(|t| t.version != version)
    {
        let ds = state.grid.size.x / state.readback_size.x;
        let grid = TerrainGrid {
            cell_m: state.grid.cell_m * ds as f32,
            size: state.readback_size,
            ..state.grid
        };
        let expected = (grid.size.x * grid.size.y) as usize;
        readbacks.request(
            "terrain_height",
            state.height_readback.clone(),
            move |bytes, snapshot| {
                let Some(cells) = decode::<[f32; 2]>(bytes) else {
                    return;
                };
                if cells.len() < expected {
                    return;
                }
                let cells = &cells[..expected];
                snapshot.terrain = Some(TerrainSnapshot {
                    grid,
                    surface: cells.iter().map(|c| c[0]).collect(),
                    sediment: cells.iter().map(|c| c[1]).collect(),
                    version,
                });
            },
        );
    }
    if snapshot
        .terrain_leaves
        .as_ref()
        .is_none_or(|t| t.version != version)
    {
        let dims = state.grid.size / LEAF_CELLS;
        let expected = (dims.x * dims.y) as usize;
        readbacks.request(
            "terrain_bounds",
            state.leaf_bounds.clone(),
            move |bytes, snapshot| {
                let Some(bounds) = decode::<[f32; 4]>(bytes) else {
                    return;
                };
                if bounds.len() < expected {
                    return;
                }
                snapshot.terrain_leaves = Some(TerrainLeaves {
                    version,
                    dims,
                    bounds: bounds[..expected]
                        .iter()
                        .copied()
                        .map(Vec4::from_array)
                        .collect(),
                });
            },
        );
    }
}
