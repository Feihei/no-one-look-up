//! Trails: footprints, ruts and tracks pressed into snow, sand, soil and mud
//! (`docs/research/trails.md`, `docs/research/trails-game-ground.md`).
//!
//! An entity with an [`Imprint`] presses into the ground under it while its translation (the
//! sole) is within 0.1 m of the undisturbed surface: as deep as the ground gives under its load
//! (Bekker–Wong sinkage, the CRREL compaction limit in snow), with a rim of the material it
//! cannot compress. The field is a centimetre window that follows the [`ElementsFocus`]
//! (else the camera), simulated on the GPU (stamp, merge, slump and wind) and drawn per pixel by
//! the terrain and ground materials and by a [`WeatheredMaterial`] that opts in. New snow fills
//! it, wind smooths it, melt takes it, wet sand and mud slump. The ground is the crate's
//! [`Terrain`](crate::Terrain) with its cover maps, or a game's own, described by a
//! [`TrailGround`].
//!
//! [`ElementsFocus`]: crate::ElementsFocus
//! [`WeatheredMaterial`]: crate::WeatheredMaterial

mod gpu;
pub(crate) mod model;

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use bevy::asset::{RenderAssetUsages, embedded_asset};
use bevy::image::{ImageAddressMode, ImageFilterMode, ImageSampler, ImageSamplerDescriptor};
use bevy::prelude::*;
use bevy::render::render_resource::{
    Extent3d, ShaderType, TextureDimension, TextureFormat, TextureUsages,
};
use bevy::render::storage::ShaderBuffer;

use crate::celestial::SunState;
use crate::celestial::{CelestialFrame, WorldClock};
use crate::climate::{SurfaceWeathering, Weather};
use crate::clouds::CloudShadows;
use crate::cover::{CoverState, SnowSettings};
use crate::dust::DustSettings;
use crate::gpu::grid::zeroed_buffer;
use crate::gpu::readback::{ReadbackTick, Readbacks, decode};
use crate::plugin::{
    ElementsQuality, ElementsSystems, ElementsUnits, ElementsView, ensure_gpu_core,
};
use crate::terrain::{TerrainGpu, TerrainParamsGpu, TerrainState};
use model::{ImprintGpu, MAX_IMPRINTS, MAX_SWEEP_M, window_origin};

/// Presses into the ground under it: a footprint, a hoof, a wheel, a body.
///
/// The contact patch is an ellipse of `half_extents` (m) along the entity's local X and Z, centred
/// on its translation, which is the sole: it presses while the sole is no more than 0.1 m above
/// the undisturbed ground (the terrain or the [`TrailGround`], plus the snow) under it, so lift it
/// to stop. It sinks as deep as the ground gives under `load_n` newtons spread over the ellipse:
/// about 3 cm into dry sand, a few millimetres into dry soil, up to 30 cm into mud and fresh snow
/// for a 77 kg walker on a 10 × 26 cm sole. Between frames it carves the whole path it swept, so
/// a fast wheel leaves an unbroken rut. The 64 nearest to the focus press each frame.
#[derive(Component, Clone, Copy, Debug, PartialEq, Reflect)]
#[require(Transform)]
pub struct Imprint {
    /// Half extents of the contact ellipse along the entity's local X and Z, m.
    pub half_extents: Vec2,
    /// The load it presses with, N (a standing 77.5 kg person on one foot: 760 N).
    pub load_n: f32,
}

impl Imprint {
    /// An ellipse of `half_extents` (m) loaded with `load_n` newtons.
    pub fn new(half_extents: Vec2, load_n: f32) -> Self {
        Self {
            half_extents,
            load_n,
        }
    }
}

/// A game's own ground for the trails, when the crate's [`Terrain`](crate::Terrain) is not the
/// ground (`docs/research/trails-game-ground.md`).
///
/// `height` maps a world XZ to the ground's world height (world units, the shape of
/// `bevy_flora`'s `FloraGround`, so one closure can feed both); it is called once per
/// [`Imprint`] a frame for the contact test, so it should be cheap: an analytic height or a
/// heightmap lookup, not a ray cast. The ground is soil (turning to mud with
/// [`SurfaceWeathering::wetness`]) or, [`with_sand`](Self::with_sand), sand; snow lies on it as
/// [`SurfaceWeathering`]'s snowpack says. While it exists the trails run on it, with or without
/// a crate `Terrain`, and nothing is spawned. Draw them with
/// [`WeatheredExtension::with_trails`](crate::WeatheredExtension::with_trails) on the ground's
/// material. Cloning is cheap (the closure is shared).
#[derive(Resource, Clone)]
pub struct TrailGround {
    height: Arc<dyn Fn(Vec2) -> f32 + Send + Sync>,
    sand: bool,
}

impl TrailGround {
    /// The ground whose world height at world XZ `xz` is `height(xz)`; soil.
    pub fn new(height: impl Fn(Vec2) -> f32 + Send + Sync + 'static) -> Self {
        Self {
            height: Arc::new(height),
            sand: false,
        }
    }

    /// Flat ground at world height `y`; soil.
    pub fn flat(y: f32) -> Self {
        Self::new(move |_| y)
    }

    /// The same ground, of sand (dry sand slumps to 34°, wet sand holds a wall).
    pub fn with_sand(mut self) -> Self {
        self.sand = true;
        self
    }

    /// Whether the ground is sand.
    pub fn is_sand(&self) -> bool {
        self.sand
    }

    /// The ground's world height at world XZ `xz`.
    pub fn height(&self, xz: Vec2) -> f32 {
        (self.height)(xz)
    }
}

impl std::fmt::Debug for TrailGround {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TrailGround")
            .field("sand", &self.sand)
            .finish_non_exhaustive()
    }
}

/// Whether trails are pressed and drawn, and how fine. Insert it before the plugin to override
/// the default.
#[derive(Resource, Clone, Debug, PartialEq, Reflect)]
pub struct TrailSettings {
    /// Press and draw trails. Default `true`.
    pub enabled: bool,
    /// The window's texel edge, m, instead of the quality's (`ElementsQuality::trail_window`:
    /// 5 cm at High). The window keeps the quality's texel count, so a coarser texel reaches
    /// farther at the same cost: 8 cm covers 82 m at High, enough for an isometric camera.
    /// Default `None`.
    pub texel_m: Option<f32>,
}

impl Default for TrailSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            texel_m: None,
        }
    }
}

impl TrailSettings {
    /// The window (texels a side, texel edge m) at `quality`.
    fn window(&self, quality: ElementsQuality) -> (u32, f32) {
        let (size, texel) = quality.trail_window();
        let texel = self
            .texel_m
            .filter(|t| t.is_finite() && *t > 0.0)
            .unwrap_or(texel);
        (size, texel)
    }
}

/// The published trail field, for a custom ground shader: an `rgba16float` texture of
/// `(h, occupancy, ∂h/∂x, ∂h/∂z)` per texel, `h` the surface's offset from the undisturbed ground
/// (m, negative in a trench), `occupancy` the largest `|h|` over the texel's 8 × 8 block and the
/// eight blocks around it (0 where there is nothing to draw), the gradient the field's Sobel
/// gradient; toroidal by world texel. Sample it at `world_xz / (texel_m · texels)` with a
/// repeating sampler, within `texels · texel_m / 2` of [`centre`](Self::centre)
/// (`bevy_elements::trails::surface` does it). Positions are in the crate's metre frame (world ÷
/// [`ElementsUnits`](crate::ElementsUnits)).
#[derive(Resource, Clone, Debug)]
pub struct TrailState {
    /// The field texture.
    pub field: Handle<Image>,
    /// Texel edge, m.
    pub texel_m: f32,
    /// Texels a side.
    pub texels: u32,
    /// The window's centre this frame, metre-frame XZ: the corner of the texel the
    /// [`ElementsFocus`](crate::ElementsFocus) (else the camera) stands in.
    pub centre: Vec2,
    /// The share of the light on a horizontal surface that comes from the sun or moon's beam
    /// (the self-shadow darkens only that share).
    pub(crate) direct_share: f32,
}

/// Uniform of the trail kernels. Mirrors `TrailParams` of `trails.wesl`.
#[derive(Clone, Copy, Debug, Default, PartialEq, ShaderType)]
pub(crate) struct TrailParams {
    terrain: TerrainParamsGpu,
    origin: IVec2,
    prev_origin: IVec2,
    /// x: texels a side. y: imprints uploaded. z: 1 to clear the window. w: stamp tile edge.
    counts: UVec4,
    /// x: texel edge, m. y: game time of this step, s. z, w: wind diffusivity in snow and sand,
    /// m²/s.
    sim: Vec4,
    /// A game's own ground, uniform over the window: snow depth (m), snow density (kg/m³),
    /// wetness, sediment (m: 1 for sand, 0 for soil).
    ground: Vec4,
    /// x: 1 on a game's own ground. y: texels a side of a readback cell.
    mode: UVec4,
}

/// This frame's work; extracted to the render world every frame.
#[derive(Resource, Clone)]
pub(crate) struct TrailGpu {
    pub state_a: Handle<ShaderBuffer>,
    pub state_b: Handle<ShaderBuffer>,
    pub stamp: Handle<ShaderBuffer>,
    /// The largest |h| per 8 × 8 block of slots (the materials' occupancy).
    pub blocks: Handle<ShaderBuffer>,
    /// The terrain's cells and the cover maps' (one-cell placeholders on a game's own ground).
    pub cells: Handle<ShaderBuffer>,
    pub cover: Handle<ShaderBuffer>,
    /// The deepest `h` per readback cell, after a four-word header (the window origin in
    /// texels, cells a side, texels a cell).
    pub readback: Handle<ShaderBuffer>,
    pub field: Handle<Image>,
    pub params: TrailParams,
    /// Never empty: `params.counts.y` says how many are real.
    pub imprints: Vec<ImprintGpu>,
    /// Run the kernels this frame.
    pub run: bool,
    /// Fill the readback buffer this frame (a readback tick).
    pub read: bool,
    /// Frames the kernels ran.
    pub executed: Arc<AtomicU64>,
    /// The render world has every pipeline and buffer it needs.
    pub ready: Arc<AtomicBool>,
}

/// What the field is built on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Source {
    /// The crate's terrain, by its water buffer (a new terrain clears the field).
    Terrain(AssetId<ShaderBuffer>),
    /// A game's own ground ([`TrailGround`]).
    Game,
}

/// Main-world bookkeeping.
#[derive(Resource, Default)]
struct TrailRuntime {
    /// The window the GPU resources were built for: (texels, texel edge).
    built: Option<(u32, f32)>,
    /// The ground they were built on.
    source: Option<Source>,
    /// Where each imprint was last frame, world XZ.
    last: HashMap<Entity, Vec2>,
}

/// Trails pressed into the ground by [`Imprint`]s, drawn by the terrain and ground materials and
/// by every [`WeatheredMaterial`](crate::WeatheredMaterial) that opts in.
///
/// Runs on a game's own ground while a [`TrailGround`] exists, else on a
/// [`Terrain`](crate::Terrain) with [`GroundCoverPlugin`](crate::GroundCoverPlugin) (the field
/// reads their maps); idles without either. Inserts, if absent, [`TrailSettings`]. Add it with
/// [`ElementsPlugins`](crate::ElementsPlugins).
pub struct TrailsPlugin;

impl Plugin for TrailsPlugin {
    fn build(&self, app: &mut App) {
        ensure_gpu_core(app);
        embedded_asset!(app, "trails.wesl");
        app.init_resource::<TrailSettings>()
            .init_resource::<SnowSettings>()
            .init_resource::<TrailRuntime>()
            .add_systems(
                Update,
                (
                    (ensure_trails, run_trails)
                        .chain()
                        .in_set(ElementsSystems::Simulation),
                    request_readback.in_set(ElementsSystems::Readback),
                ),
            );
        gpu::plugin(app);
    }
}

/// A `size²` `rgba16float` texture written by a compute kernel and sampled by the materials with
/// a linear, repeating sampler (the field is toroidal).
fn field_image(images: &mut Assets<Image>, size: u32) -> Handle<Image> {
    let mut image = Image::new_fill(
        Extent3d {
            width: size,
            height: size,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        &[0u8; 8],
        TextureFormat::Rgba16Float,
        RenderAssetUsages::RENDER_WORLD,
    );
    image.texture_descriptor.usage = TextureUsages::STORAGE_BINDING
        | TextureUsages::TEXTURE_BINDING
        | TextureUsages::COPY_DST
        | TextureUsages::COPY_SRC;
    image.sampler = ImageSampler::Descriptor(ImageSamplerDescriptor {
        address_mode_u: ImageAddressMode::Repeat,
        address_mode_v: ImageAddressMode::Repeat,
        mag_filter: ImageFilterMode::Linear,
        min_filter: ImageFilterMode::Linear,
        ..default()
    });
    images.add(image)
}

/// Texels a side of a readback cell for a window of `size` texels: at most 512² cells (1 MB).
pub(crate) fn readback_cell(size: u32) -> u32 {
    (size / 512).max(2)
}

/// Create the field when a game's ground or a terrain with ground cover exists and trails are
/// on; rebuild it for another ground, a new terrain or a new window size; drop it otherwise.
#[allow(clippy::too_many_arguments)]
fn ensure_trails(
    mut commands: Commands,
    settings: Res<TrailSettings>,
    quality: Res<ElementsQuality>,
    game: Option<Res<TrailGround>>,
    terrain: Option<Res<TerrainState>>,
    cover: Option<Res<CoverState>>,
    gpu: Option<Res<TrailGpu>>,
    mut runtime: ResMut<TrailRuntime>,
    mut buffers: ResMut<Assets<ShaderBuffer>>,
    mut images: ResMut<Assets<Image>>,
) {
    // A game's own ground wins: inserting it says the ground is the game's.
    let maps = match (&game, &terrain, &cover) {
        (Some(_), _, _) => Some((Source::Game, None)),
        (None, Some(t), Some(c)) => Some((
            Source::Terrain(t.water.id()),
            Some((t.cells.clone(), c.cover_cells.clone())),
        )),
        _ => None,
    };
    let (Some((source, maps)), true) = (maps, settings.enabled) else {
        if gpu.is_some() {
            commands.remove_resource::<TrailGpu>();
            commands.remove_resource::<TrailState>();
            *runtime = TrailRuntime::default();
        }
        return;
    };
    let window = settings.window(*quality);
    if gpu.is_some() && runtime.built == Some(window) && runtime.source == Some(source) {
        return;
    }
    let (size, texel_m) = window;
    let texels = u64::from(size) * u64::from(size);
    let cells = u64::from(size / readback_cell(size));
    let field = field_image(&mut images, size);
    let (cells_buffer, cover_buffer) = maps.unwrap_or_else(|| {
        (
            zeroed_buffer(&mut buffers, "elements_trails_no_terrain", 16),
            zeroed_buffer(&mut buffers, "elements_trails_no_cover", 16),
        )
    });
    *runtime = TrailRuntime {
        built: Some(window),
        source: Some(source),
        ..TrailRuntime::default()
    };
    commands.insert_resource(TrailState {
        field: field.clone(),
        texel_m,
        texels: size,
        centre: Vec2::ZERO,
        direct_share: 0.0,
    });
    commands.insert_resource(TrailGpu {
        state_a: zeroed_buffer(&mut buffers, "elements_trails_state_a", texels * 8),
        state_b: zeroed_buffer(&mut buffers, "elements_trails_state_b", texels * 8),
        stamp: zeroed_buffer(&mut buffers, "elements_trails_stamp", texels * 8),
        blocks: zeroed_buffer(&mut buffers, "elements_trails_blocks", texels / 64 * 4),
        cells: cells_buffer,
        cover: cover_buffer,
        readback: zeroed_buffer(
            &mut buffers,
            "elements_trails_readback",
            (4 + cells * cells) * 4,
        ),
        field,
        params: TrailParams::default(),
        imprints: vec![ImprintGpu::default()],
        run: false,
        read: false,
        executed: Arc::new(AtomicU64::new(0)),
        ready: Arc::new(AtomicBool::new(false)),
    });
}

/// The yaw of an entity's local X axis in world XZ.
fn yaw_of(transform: &GlobalTransform) -> f32 {
    let x = transform.right();
    x.z.atan2(x.x)
}

/// The share of the light on a horizontal surface carried by the brightest body's beam: the
/// clear sky's direct share (the clouds' `sky_share`: 85 % at high sun, none in twilight) times
/// the beam the lights carry under the clouds (their mean over the shadow map); the moon's
/// light is all beam ([INFERENCE]: the note's self-shadow weighting).
fn direct_share(sun: &SunState, frame: &CelestialFrame, shadows: Option<&CloudShadows>) -> f32 {
    let beam = shadows
        .map_or(1.0, |s| s.mean_transmittance)
        .clamp(0.0, 1.0);
    if sun.altitude_deg > 0.0 {
        let t = ((sun.altitude_deg + 2.0) / 12.0).clamp(0.0, 1.0);
        let diffuse = 1.0 + (0.15 - 1.0) * t * t * (3.0 - 2.0 * t);
        (1.0 - diffuse) * beam
    } else if sun.moon_altitude_deg > 0.0 && frame.moon_lux > 0.0 {
        beam
    } else {
        0.0
    }
}

/// The uniform ground of a game's own ground: snow depth (m), snow density (kg/m³), wetness and
/// sediment (1 m for sand, none for soil), from the point snowpack.
fn game_ground(ground: &TrailGround, weathering: Option<&SurfaceWeathering>) -> Vec4 {
    let (snow, density, wetness) = weathering.map_or((0.0, 100.0, 0.0), |w| {
        (w.snow_depth_m.max(0.0), w.snow_density(), w.wetness)
    });
    Vec4::new(
        snow,
        density,
        wetness.clamp(0.0, 1.0),
        if ground.sand { 1.0 } else { 0.0 },
    )
}

/// What the trails can run on: a game's own ground with the point snowpack, or the terrain.
type GroundSources<'w> = (
    Option<Res<'w, TrailGround>>,
    Option<Res<'w, SurfaceWeathering>>,
    Option<Res<'w, TerrainState>>,
    Option<Res<'w, TerrainGpu>>,
);

/// Gather this frame's imprints, place the window and fill the uniform.
#[allow(clippy::too_many_arguments)]
fn run_trails(
    view: Res<ElementsView>,
    weather: Res<Weather>,
    clock: Res<WorldClock>,
    time: Res<Time<Virtual>>,
    (sun, frame, shadows): (
        Res<SunState>,
        Res<CelestialFrame>,
        Option<Res<CloudShadows>>,
    ),
    snow: Res<SnowSettings>,
    tick: Res<ReadbackTick>,
    dust: Option<Res<DustSettings>>,
    (game, weathering, terrain, terrain_gpu): GroundSources<'_>,
    gpu: Option<ResMut<TrailGpu>>,
    state: Option<ResMut<TrailState>>,
    mut runtime: ResMut<TrailRuntime>,
    imprints: Query<(Entity, &Imprint, Ref<GlobalTransform>)>,
    units: Res<ElementsUnits>,
) {
    let (Some(mut gpu), Some(mut state)) = (gpu, state) else {
        return;
    };
    let Some((size, texel_m)) = runtime.built else {
        return;
    };
    // The ground: a game's own (uniform, its contact tested here), or the terrain's maps.
    let game = game.as_deref();
    let (terrain_params, ground, live) = match (game, terrain.as_deref(), terrain_gpu) {
        (Some(g), _, _) => (
            TerrainParamsGpu::default(),
            game_ground(g, weathering.as_deref()),
            true,
        ),
        (None, Some(t), Some(t_gpu)) => (
            TerrainParamsGpu::new(&t.grid, 1, true),
            Vec4::ZERO,
            t.is_live() && t_gpu.generated.load(Ordering::Relaxed),
        ),
        _ => return,
    };
    state.direct_share = direct_share(&sun, &frame, shadows.as_deref());
    let camera = Vec2::new(view.position.x, view.position.z);
    let origin = window_origin(camera, texel_m, size);
    state.centre = (camera / texel_m).floor() * texel_m;

    // The imprints: the swept segment from where each was last frame. An imprint's first frame
    // carves nothing (its `GlobalTransform` is not propagated yet), and a jump of more than
    // MAX_SWEEP_M in a frame (a teleport, a respawn) starts a new path.
    let mut seen = HashMap::with_capacity(runtime.last.len());
    let mut keyed: Vec<(f32, Entity, ImprintGpu)> = Vec::new();
    for (entity, imprint, transform) in &imprints {
        if transform.is_added() {
            continue;
        }
        // The sole in the metre frame: the sweep limit and the sinkage test are metres.
        let transform_m = units.to_metres(&transform);
        let p = transform_m.translation();
        let xz = Vec2::new(p.x, p.z);
        let from = runtime
            .last
            .get(&entity)
            .copied()
            .filter(|from| from.distance(xz) <= MAX_SWEEP_M)
            .unwrap_or(xz);
        seen.insert(entity, xz);
        // On a game's ground the kernel's surface is the plane y = 0 and the sole is uploaded
        // at its height above the ground and the snow under it.
        let sole_y = match game {
            Some(g) => {
                let world = transform.translation();
                let ground_y = g.height(Vec2::new(world.x, world.z)) / units.per_metre();
                p.y - ground_y - ground.x
            }
            None => p.y,
        };
        if let Some(upload) = ImprintGpu::new(
            imprint.half_extents,
            yaw_of(&transform_m),
            imprint.load_n,
            from,
            xz,
            sole_y,
            texel_m,
        ) {
            keyed.push((xz.distance_squared(camera), entity, upload));
        }
    }
    runtime.last = seen;
    if keyed.len() > MAX_IMPRINTS {
        keyed.select_nth_unstable_by(MAX_IMPRINTS, |a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        keyed.truncate(MAX_IMPRINTS);
    }
    let tile = keyed
        .iter()
        .map(|(_, _, i)| i.tile.w as u32)
        .max()
        .unwrap_or(0)
        .div_ceil(16)
        * 16;

    // Wind smoothing: the saltation flux as a diffusivity, m²/s.
    let u = weather.wind_speed_ms;
    let snow_d = if snow.drift {
        snow.drift_coefficient * (u - snow.drift_threshold_ms).max(0.0).powi(3)
            / snow.new_density.max(1.0)
    } else {
        0.0
    };
    let sand_d = dust.as_deref().filter(|d| d.enabled).map_or(0.0, |d| {
        d.flux_coefficient * u * u * (u - d.threshold_ms).max(0.0)
    });
    let dt = if clock.paused {
        0.0
    } else {
        time.delta_secs() * clock.time_scale.max(0.0)
    };

    gpu.run = gpu.ready.load(Ordering::Relaxed) && live;
    gpu.read = gpu.run && tick.is_due();
    // The render world fills `prev_origin` and the clear flag: only it knows which frames ran.
    gpu.params = TrailParams {
        terrain: terrain_params,
        origin,
        prev_origin: origin,
        counts: UVec4::new(size, keyed.len() as u32, 0, tile),
        sim: Vec4::new(texel_m, dt, snow_d, sand_d),
        ground,
        mode: UVec4::new(u32::from(game.is_some()), readback_cell(size), 0, 0),
    };
    gpu.imprints = if keyed.is_empty() {
        vec![ImprintGpu::default()]
    } else {
        keyed.into_iter().map(|(_, _, i)| i).collect()
    };
}

/// Read the field's deepest points back for [`Environment::trail_depth`](crate::Environment).
fn request_readback(
    gpu: Option<Res<TrailGpu>>,
    state: Option<Res<TrailState>>,
    mut readbacks: Readbacks,
) {
    let (Some(gpu), Some(state)) = (gpu, state) else {
        return;
    };
    if !gpu.read {
        return;
    }
    let (texel_m, texels) = (state.texel_m, state.texels);
    readbacks.request("trails", gpu.readback.clone(), move |bytes, snapshot| {
        if let Some(values) = decode::<f32>(bytes)
            && let Some(trails) = TrailSnapshot::decode(&values, texel_m, texels)
        {
            snapshot.trails = Some(trails);
        }
    });
}

/// CPU copy of the trail field's deepest points: one per `cell` × `cell` block of texels.
#[derive(Clone, Debug)]
pub(crate) struct TrailSnapshot {
    /// World texel of the window's minimum corner when it was read.
    origin: IVec2,
    /// Texel edge, m.
    texel_m: f32,
    /// Texels a side of the window.
    texels: u32,
    /// Texels a side of a cell.
    cell: u32,
    /// The deepest `h` per cell (m, negative in a trench), row-major by slot cell.
    deepest: Arc<[f32]>,
}

impl TrailSnapshot {
    /// The snapshot in a readback buffer of a window of `texels` texels of `texel_m`: a header
    /// of the window origin (texels), cells a side and texels a cell, then the cells. `None`
    /// for a buffer that does not match the window (a frame that ran no kernel).
    pub(crate) fn decode(values: &[f32], texel_m: f32, texels: u32) -> Option<Self> {
        let (header, rest) = values.split_at_checked(4)?;
        let cell = readback_cell(texels);
        let cells = (texels / cell) as usize;
        if header[2] != cells as f32 || header[3] != cell as f32 || rest.len() < cells * cells {
            return None;
        }
        Some(Self {
            origin: IVec2::new(header[0] as i32, header[1] as i32),
            texel_m,
            texels,
            cell,
            deepest: rest[..cells * cells].into(),
        })
    }

    /// How far the ground at metre-frame XZ is pressed below the undisturbed surface, m: 0 on a
    /// rim or untouched ground, `None` outside the window.
    pub(crate) fn depth_at(&self, xz: Vec2) -> Option<f32> {
        let w = (xz / self.texel_m).floor().as_ivec2();
        let r = w - self.origin;
        let n = self.texels as i32;
        if r.min_element() < 0 || r.max_element() >= n {
            return None;
        }
        // The slot of world texel `w` (toroidal), and its cell: a whole number of cells fits the
        // window, so a slot cell is a world-aligned block.
        let slot = w.rem_euclid(IVec2::splat(n));
        let c = slot / self.cell as i32;
        let cells = (self.texels / self.cell) as i32;
        let h = self.deepest.get((c.y * cells + c.x) as usize)?;
        Some((-h).max(0.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A readback of a 16-texel window (cells of 2) whose origin is (−5, 7), with one pressed
    /// texel at world texel (−3, 9).
    fn sample() -> Vec<f32> {
        let texels = 16u32;
        let cell = readback_cell(texels);
        let cells = (texels / cell) as i32;
        let origin = IVec2::new(-5, 7);
        let mut values = vec![origin.x as f32, origin.y as f32, cells as f32, cell as f32];
        let mut deepest = vec![0.0f32; (cells * cells) as usize];
        let slot = IVec2::new(-3, 9).rem_euclid(IVec2::splat(texels as i32));
        let c = slot / cell as i32;
        deepest[(c.y * cells + c.x) as usize] = -0.12;
        values.extend(deepest);
        values
    }

    #[test]
    fn the_readback_places_a_print_where_the_window_held_it() {
        let texel = 0.05;
        let snapshot = TrailSnapshot::decode(&sample(), texel, 16).expect("a matching buffer");
        // The pressed texel's centre, and its cell-mate.
        let at = |w: IVec2| (w.as_vec2() + 0.5) * texel;
        assert_eq!(snapshot.depth_at(at(IVec2::new(-3, 9))), Some(0.12));
        assert_eq!(snapshot.depth_at(at(IVec2::new(-4, 8))), Some(0.12));
        // Untouched ground inside the window, and outside it.
        assert_eq!(snapshot.depth_at(at(IVec2::new(5, 15))), Some(0.0));
        assert_eq!(snapshot.depth_at(at(IVec2::new(-6, 9))), None);
        assert_eq!(snapshot.depth_at(at(IVec2::new(11, 9))), None);
    }

    #[test]
    fn a_readback_of_another_window_is_refused() {
        assert!(TrailSnapshot::decode(&sample(), 0.05, 32).is_none());
        assert!(TrailSnapshot::decode(&sample()[..10], 0.05, 16).is_none());
    }
}
