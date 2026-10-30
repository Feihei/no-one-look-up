//! Waves: ripples and wakes from bodies in the water (`docs/research/waves.md`).
//!
//! An entity with a [`WaterContact`] (an ellipsoid) presses on the water surface where it cuts
//! it, with the pressure head of its occupied column: at rest it holds the water down under it,
//! moving it leaves a wake bounded by Kelvin's angle, bobbing it sends out rings. The surface is
//! a linear dispersive wave field (Airy's capillary-gravity dispersion through a fitted 13 × 13
//! kernel, Tessendorf's iWave scheme) in three camera-following cascades a factor of four apart,
//! reflected by the shores, carried by the current, and drawn by `WaterPlugin`'s surface and
//! absorb passes.

mod gpu;
pub(crate) mod model;

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use bevy::asset::{RenderAssetUsages, embedded_asset};
use bevy::image::{ImageAddressMode, ImageFilterMode, ImageSampler, ImageSamplerDescriptor};
use bevy::prelude::*;
use bevy::render::render_resource::{
    Extent3d, ShaderType, TextureDimension, TextureFormat, TextureUsages, TextureViewDescriptor,
    TextureViewDimension,
};
use bevy::render::storage::ShaderBuffer;

use crate::gpu::grid::zeroed_buffer;
use crate::plugin::{
    ElementsQuality, ElementsSystems, ElementsUnits, ElementsView, ensure_gpu_core,
};
use crate::query::Environment;
use crate::terrain::{TerrainGpu, TerrainParamsGpu, TerrainState};
use crate::trails::model::window_origin;
use model::{
    CASCADES, ContactGpu, MAX_CONTACTS, MAX_SUBSTEPS, SUBSTEP_S, WaveKernel, band_centre,
    cascade_texel, entry_head,
};

/// A body in the water: an ellipsoid of `half_extents` (m) along the entity's local axes, centred
/// on its translation, any rotation.
///
/// Where it cuts the water surface it presses the surface down by the pressure head of its
/// occupied column (its draft, for a shallow hull; less for a deep, narrow one, whose lower part
/// the waves do not reach): at rest the water stays flat around it, moving it leaves a wake,
/// rising or sinking it sends out rings. Approximate a long body (a hull, a log) with one
/// stretched ellipsoid and a person wading with one per leg. The 32 nearest to the camera
/// press each frame.
#[derive(Component, Clone, Copy, Debug, PartialEq, Reflect)]
#[require(Transform)]
pub struct WaterContact {
    /// Half extents of the ellipsoid along the entity's local X, Y and Z, m.
    pub half_extents: Vec3,
}

impl WaterContact {
    /// An ellipsoid of `half_extents` (m).
    pub fn new(half_extents: Vec3) -> Self {
        Self { half_extents }
    }

    /// A sphere of `radius` (m).
    pub fn sphere(radius: f32) -> Self {
        Self::new(Vec3::splat(radius))
    }
}

/// The wave field's settings. Insert it before the plugin to override the default.
#[derive(Resource, Clone, Debug, PartialEq, Reflect)]
pub struct WaveSettings {
    /// Simulate and draw the waves. Default `true`.
    pub enabled: bool,
    /// The waves' energy loss, s⁻¹: an amplitude falls by `e` in `2 / damping` seconds. Default
    /// 0.05 (Crest's), which a calm pond's surface films and the wind account for; viscosity alone
    /// would be below 10⁻³.
    pub damping_per_s: f32,
}

impl Default for WaveSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            damping_per_s: 0.05,
        }
    }
}

/// The published wave field, for a custom water shader: a texture array of one `rgba16float`
/// layer per cascade holding `(h, ∂h/∂x, ∂h/∂z, 0)` per texel, `h` the surface's rise above the
/// undisturbed water (m), toroidal by world texel. Cascade `c` has texels of `texel_m[c]`; sample
/// it at `world_xz / (texel_m[c] · texels)` with a repeating sampler, within
/// `texels · texel_m[c] / 2` of the followed camera's texel. The surface is the sum of the layers.
#[derive(Resource, Clone, Debug)]
pub struct WaveState {
    /// The field texture array.
    pub field: Handle<Image>,
    /// Texel edge of each cascade, m, finest first.
    pub texel_m: [f32; CASCADES],
    /// Texels a side, every cascade.
    pub texels: u32,
}

/// Uniform of the wave kernels. Mirrors `WaveParams` of `waves.wesl`.
#[derive(Clone, Copy, Debug, Default, PartialEq, ShaderType)]
pub(crate) struct WaveParams {
    terrain: TerrainParamsGpu,
    /// Per cascade: xy the window's minimum world texel this frame, zw last frame's.
    origins: [IVec4; CASCADES],
    /// x: texels a side. y: contacts uploaded. z: 1 to clear the whole field. w: the head tile's
    /// edge in finest texels.
    counts: UVec4,
    /// x: the finest texel, m. y: substep, s. z: damping, s⁻¹.
    sim: Vec4,
    /// xyz: each cascade's band-centre wavenumber, rad/m.
    band: Vec4,
    /// Each cascade's 28 kernel weights.
    kernel: [Vec4; 7 * CASCADES],
}

/// This frame's work; extracted to the render world every frame.
#[derive(Resource, Clone)]
pub(crate) struct WaveGpu {
    pub state_a: Handle<ShaderBuffer>,
    pub state_b: Handle<ShaderBuffer>,
    /// The bodies' pressure head per plane (`waves.wesl`), as f32 bits for `atomicMax`.
    pub head: Handle<ShaderBuffer>,
    /// This frame's smoothed forcing per cascade texel.
    pub force: Handle<ShaderBuffer>,
    pub field: Handle<Image>,
    pub params: WaveParams,
    /// Never empty: `params.counts.y` says how many are real.
    pub contacts: Vec<ContactGpu>,
    /// Substeps to run this frame.
    pub substeps: u32,
    /// Frames the kernels ran.
    pub executed: Arc<AtomicU64>,
    /// The render world has every pipeline and buffer it needs.
    pub ready: Arc<AtomicBool>,
}

/// Main-world bookkeeping.
#[derive(Resource, Default)]
struct WaveRuntime {
    /// The window the GPU resources were built for: (texels, finest texel edge).
    built: Option<(u32, f32)>,
    /// The terrain's water buffer they were built against (a new terrain clears them).
    terrain: Option<AssetId<ShaderBuffer>>,
    /// The kernels of the built cascades.
    kernel: [Vec4; 7 * CASCADES],
    /// Virtual time not yet simulated, s.
    debt: f32,
    /// Each body's centre height last frame, m.
    last: HashMap<Entity, f32>,
}

/// Ripples and wakes on the hydrology's water from [`WaterContact`]s, drawn by `WaterPlugin`.
///
/// Needs a [`Terrain`](crate::Terrain) with [`HydrologyPlugin`](crate::HydrologyPlugin) (the waves
/// live on its water); idles without them. Inserts, if absent, [`WaveSettings`]. Add it with
/// [`ElementsPlugins`](crate::ElementsPlugins).
pub struct WavesPlugin;

impl Plugin for WavesPlugin {
    fn build(&self, app: &mut App) {
        ensure_gpu_core(app);
        embedded_asset!(app, "waves.wesl");
        app.init_resource::<WaveSettings>()
            .init_resource::<WaveRuntime>()
            .add_systems(
                Update,
                (ensure_waves, run_waves)
                    .chain()
                    .in_set(ElementsSystems::Simulation),
            );
        gpu::plugin(app);
    }
}

/// A `size²` texture array of one `rgba16float` layer per cascade, written by a compute kernel
/// and sampled by the water with a linear, repeating sampler (the field is toroidal).
fn field_image(images: &mut Assets<Image>, size: u32) -> Handle<Image> {
    let mut image = Image::new_fill(
        Extent3d {
            width: size,
            height: size,
            depth_or_array_layers: CASCADES as u32,
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
    image.texture_view_descriptor = Some(TextureViewDescriptor {
        dimension: Some(TextureViewDimension::D2Array),
        ..default()
    });
    image.sampler = ImageSampler::Descriptor(ImageSamplerDescriptor {
        address_mode_u: ImageAddressMode::Repeat,
        address_mode_v: ImageAddressMode::Repeat,
        mag_filter: ImageFilterMode::Linear,
        min_filter: ImageFilterMode::Linear,
        ..default()
    });
    images.add(image)
}

/// Create the field when a terrain exists and waves are on; rebuild it for a new terrain or a
/// new window; drop it otherwise.
#[allow(clippy::too_many_arguments)]
fn ensure_waves(
    mut commands: Commands,
    settings: Res<WaveSettings>,
    quality: Res<ElementsQuality>,
    terrain: Option<Res<TerrainState>>,
    gpu: Option<Res<WaveGpu>>,
    mut runtime: ResMut<WaveRuntime>,
    mut buffers: ResMut<Assets<ShaderBuffer>>,
    mut images: ResMut<Assets<Image>>,
) {
    let (Some(terrain), true) = (terrain, settings.enabled) else {
        if gpu.is_some() {
            commands.remove_resource::<WaveGpu>();
            commands.remove_resource::<WaveState>();
            *runtime = WaveRuntime::default();
        }
        return;
    };
    let window = quality.wave_window();
    if gpu.is_some() && runtime.built == Some(window) && runtime.terrain == Some(terrain.water.id())
    {
        return;
    }
    let (size, base) = window;
    let texels = u64::from(size) * u64::from(size);
    let field = field_image(&mut images, size);
    let mut kernel = [Vec4::ZERO; 7 * CASCADES];
    for c in 0..CASCADES {
        kernel[7 * c..7 * c + 7].copy_from_slice(&WaveKernel::fit(cascade_texel(base, c)).packed());
    }
    *runtime = WaveRuntime {
        built: Some(window),
        terrain: Some(terrain.water.id()),
        kernel,
        debt: 0.0,
        last: HashMap::new(),
    };
    commands.insert_resource(WaveState {
        field: field.clone(),
        texel_m: std::array::from_fn(|c| cascade_texel(base, c)),
        texels: size,
    });
    let cascades = CASCADES as u64;
    commands.insert_resource(WaveGpu {
        // Three planes per cascade (`h_n`, `h_{n−1}`, foam), and four of forcing (`waves.wesl`).
        state_a: zeroed_buffer(
            &mut buffers,
            "elements_waves_state_a",
            cascades * texels * 12,
        ),
        state_b: zeroed_buffer(
            &mut buffers,
            "elements_waves_state_b",
            cascades * texels * 12,
        ),
        head: zeroed_buffer(
            &mut buffers,
            "elements_waves_head",
            (2 * cascades - 1) * texels * 4,
        ),
        force: zeroed_buffer(&mut buffers, "elements_waves_force", cascades * texels * 16),
        field,
        params: WaveParams::default(),
        contacts: vec![ContactGpu::default()],
        substeps: 0,
        executed: Arc::new(AtomicU64::new(0)),
        ready: Arc::new(AtomicBool::new(false)),
    });
}

/// Gather this frame's bodies, place the windows, count the substeps and fill the uniform.
#[allow(clippy::too_many_arguments)]
fn run_waves(
    view: Res<ElementsView>,
    time: Res<Time<Virtual>>,
    settings: Res<WaveSettings>,
    terrain: Option<Res<TerrainState>>,
    terrain_gpu: Option<Res<TerrainGpu>>,
    gpu: Option<ResMut<WaveGpu>>,
    mut runtime: ResMut<WaveRuntime>,
    contacts: Query<(Entity, &WaterContact, &GlobalTransform)>,
    env: Environment,
    units: Res<ElementsUnits>,
) {
    let (Some(terrain), Some(terrain_gpu), Some(mut gpu)) = (terrain, terrain_gpu, gpu) else {
        return;
    };
    let Some((size, base)) = runtime.built else {
        return;
    };
    let camera = Vec2::new(view.position.x, view.position.z);
    let mut origins = [IVec4::ZERO; CASCADES];
    for (c, origin) in origins.iter_mut().enumerate() {
        let o = window_origin(camera, cascade_texel(base, c), size);
        *origin = IVec4::new(o.x, o.y, o.x, o.y);
    }

    // The previous frame's centre of each body, for its entry head.
    let dt = time.delta_secs();
    let mut seen = HashMap::with_capacity(runtime.last.len());
    let mut keyed: Vec<(f32, Entity, ContactGpu)> = contacts
        .iter()
        .filter_map(|(entity, contact, transform)| {
            // The body in the metre frame: its half extents × scale and its height are metres.
            let transform = units.to_metres(transform);
            let (scale, rotation, translation) = transform.to_scale_rotation_translation();
            let half = contact.half_extents * scale;
            let mut upload = ContactGpu::new(translation, half, rotation)?;
            let xz = Vec2::new(translation.x, translation.z);
            seen.insert(entity, translation.y);
            if let (Some(prev), Some(ground), Some(depth)) = (
                runtime.last.get(&entity),
                env.terrain_height_metres(xz),
                env.water_depth_metres(xz),
            ) {
                upload.m1.z = entry_head(
                    *prev,
                    translation.y,
                    ContactGpu::half_height(half, rotation),
                    ground + depth,
                    dt,
                );
            }
            Some((xz.distance_squared(camera), entity, upload))
        })
        .collect();
    runtime.last = seen;
    if keyed.len() > MAX_CONTACTS {
        keyed.select_nth_unstable_by(MAX_CONTACTS, |a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        keyed.truncate(MAX_CONTACTS);
    }
    // The head tile: every contact's bounding square in finest texels, with a texel of margin
    // each side for the supersampled edge.
    let tile = keyed
        .iter()
        .map(|(_, _, c)| (2.0 * c.centre.w / base).ceil() as u32 + 3)
        .max()
        .unwrap_or(0)
        .min(1024);

    // Fixed substeps of virtual time.
    runtime.debt = (runtime.debt + time.delta_secs()).min(MAX_SUBSTEPS as f32 * SUBSTEP_S);
    let substeps = (runtime.debt / SUBSTEP_S).floor() as u32;
    runtime.debt -= substeps as f32 * SUBSTEP_S;

    let live = gpu.ready.load(Ordering::Relaxed)
        && terrain.is_live()
        && terrain_gpu.generated.load(Ordering::Relaxed);
    gpu.substeps = if live { substeps } else { 0 };
    // The render world fills last frame's origins and the clear flag: only it knows which frames
    // ran.
    gpu.params = WaveParams {
        terrain: TerrainParamsGpu::new(&terrain.grid, 1, true),
        origins,
        counts: UVec4::new(size, keyed.len() as u32, 0, tile),
        sim: Vec4::new(base, SUBSTEP_S, settings.damping_per_s.max(0.0), 0.0),
        band: Vec4::new(
            band_centre(cascade_texel(base, 0)),
            band_centre(cascade_texel(base, 1)),
            band_centre(cascade_texel(base, 2)),
            0.0,
        ),
        kernel: runtime.kernel,
    };
    gpu.contacts = if keyed.is_empty() {
        vec![ContactGpu::default()]
    } else {
        keyed.into_iter().map(|(_, _, c)| c).collect()
    };
}
