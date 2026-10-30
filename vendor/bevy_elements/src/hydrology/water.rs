//! Water rendering: the free surface of the hydrology simulation drawn over the terrain with
//! absorption, flow-mapped normals and foam. See `docs/research/water-rendering.md`.
//!
//! A second pool of CDLOD patches shares the terrain's selection and draws the patches that hold
//! water (`water_depth_max ≥ 3 mm`), displaced to bedrock + sediment + water. Every patch is two
//! entities with one [`WaterExtension`] and two materials: an *absorb* pass (`AlphaMode::Multiply`)
//! that filters what is under the water per colour channel (Beer–Lambert on the thickness the depth
//! prepass gives), and a *surface* pass (`AlphaMode::Add`) that adds the reflection of the sky and
//! sun, the body colour of the silt and the foam. The pool exists for
//! [`TerrainRender::External`](crate::TerrainRender::External) too.
//!
//! The water needs a camera with a depth prepass, which [`ElementsCamera`](crate::ElementsCamera)
//! has; on a view without one the shader does not compile and the pipeline cache logs it.

#[cfg(test)]
mod model;

use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use bevy::asset::{AssetPath, embedded_asset};
use bevy::camera::primitives::Aabb;
use bevy::camera::visibility::NoAutoAabb;
use bevy::image::{ImageAddressMode, ImageFilterMode, ImageSampler, ImageSamplerDescriptor};
use bevy::mesh::MeshTag;
use bevy::pbr::{ExtendedMaterial, MaterialExtension, MaterialPlugin};
use bevy::prelude::*;
use bevy::render::render_resource::{
    AsBindGroup, Extent3d, ShaderType, TextureDimension, TextureFormat, TextureUsages,
};
use bevy::render::storage::ShaderBuffer;
use bevy::shader::ShaderRef;
use bevy::transform::TransformSystems;

use super::water_gpu;
use crate::clouds::CloudShadows;
use crate::plugin::{ElementsUnits, ElementsView, ensure_gpu_core, needs_depth_prepass};
use crate::terrain::lod::MAX_LEVELS;
use crate::terrain::{
    NO_TERRAIN, TerrainParamsGpu, TerrainRuntime, TerrainState, assign_slots, lod_centre_uniform,
    lod_uniform, patch_mesh, patch_transform, shadow_inputs,
};
use crate::waves::WaveState;
use crate::wind::WindState;

/// Edge of the generated normal map, texels.
pub(crate) const NORMAL_TEXELS: u32 = 512;
/// Mip levels of the normal map (512 → 1 texel).
pub(crate) const NORMAL_MIPS: u32 = 10;
/// Octaves of the normal map's height field: three, from 0.25 m to 6.25 cm
/// (`docs/research/water-surface.md`, Model 2).
pub(crate) const NORMAL_OCTAVES: u32 = 3;
/// The base octave's wavelength in texels of the full-resolution map (16 = 0.25 m at the 8 m
/// tile), so the finest band is at the kernel's four-texel limit.
pub(crate) const NORMAL_BASE_TEXELS: u32 = 16;
/// Octaves of the foam fbm in the map's alpha.
pub(crate) const FOAM_OCTAVES: u32 = 4;
/// The rms slope of the normal map's mip 0 (Cox & Munk 1954 at a 4 m/s wind).
pub(crate) const NORMAL_RMS_SLOPE: f32 = 0.15;
/// The factor that turns the central difference of the height field (two texels) into the slope
/// that has [`NORMAL_RMS_SLOPE`] on mip 0, measured by `normal_gain_gives_the_cox_munk_slope`.
/// It is smaller than the 4.7 of the metre-scale spectrum because a flat slope spectrum over
/// three octaves has a smaller total gradient (`docs/research/water-surface.md`, Model 1).
pub(crate) const NORMAL_SLOPE_GAIN: f32 = 1.8575;
/// The wind speed the normal map's slope belongs to, m/s.
const MAP_WIND_MS: f32 = 4.0;
/// The largest number of patches the pool draws (two entities each).
pub(crate) const WATER_POOL_CAPACITY: usize = 192;
/// A patch with less water than this (m) is not drawn.
pub(crate) const MIN_PATCH_DEPTH_M: f32 = 0.003;
/// Padding of a water patch's box (m): a flood front rises faster than the terrain's pad allows.
const WATER_AABB_PAD_M: f32 = 4.0;
/// The reflectance that gives water's `F₀ = 0.02` through Bevy's `0.16 r²`.
const WATER_REFLECTANCE: f32 = 0.353_553_4;
/// Bevy clamps the perceptual roughness to this in its lighting.
const WATER_ROUGHNESS: f32 = 0.089;
const EMBEDDED_SHADER: &str = "embedded://bevy_elements/hydrology/water.wesl";

/// How the water looks. Insert it before the plugin to override the defaults.
#[derive(Resource, Clone, Debug, PartialEq, Reflect)]
pub struct WaterSettings {
    /// Absorption of the water per linear colour channel, m⁻¹: the light's transmittance through
    /// `t` metres is `exp(−σ t)`. Default `(0.45, 0.06, 0.03)`: slightly tinted natural water
    /// (pure water is about `(0.30, 0.05, 0.004)`).
    pub absorption: Vec3,
    /// Index of refraction of the water. Default 1.33.
    pub refractive_index: f32,
    /// The colour of suspended silt, sRGB-encoded like the terrain's palette. Default
    /// `(0.45, 0.38, 0.25)`.
    pub silt_tint: Vec3,
    /// The body colour of clear deep water, linear reflectance. Default `(0, 0.005, 0.025)`.
    pub deep_color: Vec3,
    /// Scattering of the water per unit volume fraction of suspended sediment, m⁻¹. Default 6.
    pub turbidity_scattering: f32,
    /// Scattering of clear water, m⁻¹: how soon the body colour shows with depth. Default 0.05.
    pub clear_scattering: f32,
    /// Flow speed above which the water foams, m/s. Default 1.5.
    pub foam_speed_ms: f32,
    /// Thickness below which the water foams, m: the shoreline and the contact with objects.
    /// Default 0.05.
    pub foam_thickness_m: f32,
    /// Albedo of foam. Default 0.9.
    pub foam_albedo: f32,
    /// Perceptual roughness of foam. Default 0.8.
    pub foam_roughness: f32,
    /// World size of one tile of the normal map, m. Default 8.
    pub tile_m: f32,
    /// Seconds of one flow cycle of the normal map's two phases. Default 1.5.
    pub flow_cycle_s: f32,
    /// The largest distortion of the normal map by the flow, in tiles. Default `1/3`.
    pub flow_distortion: f32,
    /// The 10 m wind speed the ripples' slope follows, m/s; `None` follows
    /// [`WindState::speed_10m`] (4 m/s without a wind). Default `None`.
    pub wind_speed_ms: Option<f32>,
    /// Multiplies the ripples' slope: 1 is the wind's, 0 is a flat surface. Default 1.
    pub ripple_scale: f32,
}

impl Default for WaterSettings {
    fn default() -> Self {
        Self {
            absorption: Vec3::new(0.45, 0.06, 0.03),
            refractive_index: 1.33,
            silt_tint: Vec3::new(0.45, 0.38, 0.25),
            deep_color: Vec3::new(0.0, 0.005, 0.025),
            turbidity_scattering: 6.0,
            clear_scattering: 0.05,
            foam_speed_ms: 1.5,
            foam_thickness_m: 0.05,
            foam_albedo: 0.9,
            foam_roughness: 0.8,
            tile_m: 8.0,
            flow_cycle_s: 1.5,
            flow_distortion: 1.0 / 3.0,
            wind_speed_ms: None,
            ripple_scale: 1.0,
        }
    }
}

/// The Cox & Munk rms slope of a clean sea surface at a 10 m wind of `u10` m/s.
pub(crate) fn ripple_slope(u10: f32) -> f32 {
    (0.003 + 0.00512 * u10.max(0.0)).sqrt()
}

/// Uniform of the water materials. Mirrors `WaterParams` of `water.wesl`.
#[derive(Clone, Copy, Debug, PartialEq, ShaderType, Reflect)]
#[reflect(opaque)]
pub struct WaterParams {
    pub(crate) terrain: TerrainParamsGpu,
    /// Per level: x morph start, y range (m).
    pub(crate) lod: [Vec4; MAX_LEVELS],
    /// x: coarsest level.
    pub(crate) top: Vec4,
    /// What the vertex stage morphs about (`TerrainMaterialParams::lod_centre`).
    pub(crate) lod_centre: Vec4,
    /// Cloud shadow map origin xy (metre frame), extent, the lights' mean transmittance.
    pub(crate) shadow: Vec4,
    /// Unit direction toward the light; w = 1 with a map.
    pub(crate) light: Vec4,
    /// Absorption σ (m⁻¹) and the index of refraction.
    pub(crate) absorption: Vec4,
    /// Linear silt tint and the turbidity scattering κ.
    pub(crate) silt: Vec4,
    /// Linear deep body colour and the clear-water scattering.
    pub(crate) deep: Vec4,
    /// Flow cycle, tile, distortion limit, ripple slope relative to the map's.
    pub(crate) flow: Vec4,
    /// Foam speed threshold, thickness threshold, albedo, roughness.
    pub(crate) foam: Vec4,
    /// x: the finest wave cascade's texel (m). y: texels a side. z: 1 when a field is bound.
    pub(crate) waves: Vec4,
    /// World units to the metre ([`ElementsUnits`](crate::ElementsUnits)): the vertex and
    /// fragment stages divide the positions Bevy gives them by it and multiply the ones they
    /// hand back.
    pub(crate) units: f32,
}

impl Default for WaterParams {
    fn default() -> Self {
        Self {
            terrain: TerrainParamsGpu::default(),
            lod: [Vec4::ZERO; MAX_LEVELS],
            top: Vec4::ZERO,
            lod_centre: Vec4::ZERO,
            shadow: Vec4::ZERO,
            light: Vec4::ZERO,
            absorption: Vec4::ZERO,
            silt: Vec4::ZERO,
            deep: Vec4::ZERO,
            flow: Vec4::ZERO,
            foam: Vec4::ZERO,
            waves: Vec4::ZERO,
            units: 1.0,
        }
    }
}

/// The extension that makes a `StandardMaterial` one pass of the water: the vertex shader
/// displaces the terrain's CDLOD patches to the water surface and the fragment shader absorbs,
/// reflects and foams. The crate fills it; the two passes are two material assets that differ in
/// the base's `alpha_mode`.
#[derive(Asset, AsBindGroup, Reflect, Debug, Clone)]
pub struct WaterExtension {
    #[uniform(100)]
    pub(crate) params: WaterParams,
    #[storage(101, read_only)]
    pub(crate) cells: Handle<ShaderBuffer>,
    #[storage(102, read_only)]
    pub(crate) flow: Handle<ShaderBuffer>,
    #[texture(103)]
    #[sampler(104)]
    pub(crate) normals: Handle<Image>,
    #[texture(105)]
    #[sampler(106)]
    pub(crate) cloud_shadow: Option<Handle<Image>>,
    /// The wave field (`WaveState`), one layer per cascade.
    #[texture(107, dimension = "2d_array")]
    #[sampler(108)]
    pub(crate) waves: Option<Handle<Image>>,
}

impl MaterialExtension for WaterExtension {
    fn vertex_shader() -> ShaderRef {
        ShaderRef::Path(AssetPath::parse(EMBEDDED_SHADER))
    }

    fn fragment_shader() -> ShaderRef {
        ShaderRef::Path(AssetPath::parse(EMBEDDED_SHADER))
    }

    /// Blended passes are not queued in the prepass, and a film of water casts no useful shadow.
    fn enable_prepass() -> bool {
        false
    }

    fn enable_shadows() -> bool {
        false
    }
}

/// One pass of the water: a `StandardMaterial` extended with [`WaterExtension`].
pub type WaterMaterial = ExtendedMaterial<StandardMaterial, WaterExtension>;

/// Marks the entities of the water pool. A patch is drawn twice, by an absorb entity and a
/// surface entity with the same transform and bounds.
#[derive(Component, Clone, Copy, Debug, PartialEq, Eq)]
pub struct WaterPatch {
    /// `true` for the entity of the surface pass, `false` for the absorb pass.
    pub surface: bool,
}

/// The water's GPU assets.
#[derive(Resource, Clone, Debug)]
pub struct WaterState {
    /// The tileable normal map (512² `rgba8unorm`, 10 mips; alpha is the foam patchiness).
    pub normals: Handle<Image>,
    /// Set once the `water_normals` kernel has written every mip.
    pub normals_ready: Arc<AtomicBool>,
}

struct WaterSlot {
    absorb: Entity,
    surface: Entity,
    patch: Option<crate::terrain::lod::Patch>,
}

/// Main-world bookkeeping of the water pool.
#[derive(Resource, Default)]
struct WaterRuntime {
    /// The terrain entity the pool is parented to.
    terrain: Option<Entity>,
    absorb: Handle<WaterMaterial>,
    surface: Handle<WaterMaterial>,
    slots: Vec<WaterSlot>,
    mesh: Option<Handle<Mesh>>,
    warned_capacity: bool,
}

/// Water rendering. Needs the [`TerrainPlugin`](crate::TerrainPlugin) (it reads its selection and
/// layers) and draws whatever the hydrology simulation puts on the terrain.
pub struct WaterPlugin;

impl Plugin for WaterPlugin {
    fn build(&self, app: &mut App) {
        ensure_gpu_core(app);
        needs_depth_prepass(app);
        embedded_asset!(app, "water.wesl");
        embedded_asset!(app, "water_normals.wesl");
        app.add_plugins(MaterialPlugin::<WaterMaterial>::default())
            .init_resource::<WaterSettings>()
            .init_resource::<WaterRuntime>()
            .add_systems(Startup, create_normal_map)
            .add_systems(
                Update,
                (ensure_water_pool, update_water_materials)
                    .chain()
                    .after(crate::plugin::ElementsSystems::Readback),
            )
            .add_systems(
                PostUpdate,
                update_water_patches
                    .after(crate::terrain::update_patches)
                    .before(TransformSystems::Propagate),
            );
        water_gpu::plugin(app);
    }
}

/// Create the normal map's image: written by the `water_normals` kernel, sampled with repeat
/// addressing, linear filtering across the mips and 16× anisotropy.
fn create_normal_map(mut commands: Commands, mut images: ResMut<Assets<Image>>) {
    let mut image = Image::new_uninit(
        Extent3d {
            width: NORMAL_TEXELS,
            height: NORMAL_TEXELS,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        TextureFormat::Rgba8Unorm,
        bevy::asset::RenderAssetUsages::RENDER_WORLD,
    );
    image.texture_descriptor.mip_level_count = NORMAL_MIPS;
    image.texture_descriptor.usage =
        TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_SRC;
    image.sampler = ImageSampler::Descriptor(ImageSamplerDescriptor {
        address_mode_u: ImageAddressMode::Repeat,
        address_mode_v: ImageAddressMode::Repeat,
        mag_filter: ImageFilterMode::Linear,
        min_filter: ImageFilterMode::Linear,
        mipmap_filter: ImageFilterMode::Linear,
        anisotropy_clamp: 16,
        ..default()
    });
    commands.insert_resource(WaterState {
        normals: images.add(image),
        normals_ready: Arc::new(AtomicBool::new(false)),
    });
}

/// The two materials of the pool: the absorb pass multiplies the background, the surface pass
/// adds to it and sorts after its patch's absorb pass (`depth_bias`).
fn pass_materials(
    materials: &mut Assets<WaterMaterial>,
    normals: &Handle<Image>,
) -> (Handle<WaterMaterial>, Handle<WaterMaterial>) {
    let extension = WaterExtension {
        params: WaterParams::default(),
        cells: NO_TERRAIN,
        flow: NO_TERRAIN,
        normals: normals.clone(),
        cloud_shadow: None,
        waves: None,
    };
    let base = StandardMaterial {
        base_color: Color::BLACK,
        perceptual_roughness: WATER_ROUGHNESS,
        reflectance: WATER_REFLECTANCE,
        metallic: 0.0,
        ..default()
    };
    let absorb = materials.add(WaterMaterial {
        base: StandardMaterial {
            alpha_mode: AlphaMode::Multiply,
            ..base.clone()
        },
        extension: extension.clone(),
    });
    let surface = materials.add(WaterMaterial {
        base: StandardMaterial {
            alpha_mode: AlphaMode::Add,
            depth_bias: 1.0,
            ..base
        },
        extension,
    });
    (absorb, surface)
}

/// Spawn the pool under the terrain entity when a terrain appears.
fn ensure_water_pool(
    mut commands: Commands,
    state: Option<Res<TerrainState>>,
    water: Option<Res<WaterState>>,
    mut runtime: ResMut<WaterRuntime>,
    mut materials: ResMut<Assets<WaterMaterial>>,
    mut meshes: ResMut<Assets<Mesh>>,
) {
    let (Some(state), Some(water)) = (state, water) else {
        if runtime.terrain.take().is_some() {
            runtime.slots.clear();
        }
        return;
    };
    if runtime.terrain == Some(state.entity) {
        return;
    }
    let (absorb, surface) = pass_materials(&mut materials, &water.normals);
    runtime.absorb = absorb.clone();
    runtime.surface = surface.clone();
    let mesh = runtime
        .mesh
        .get_or_insert_with(|| meshes.add(patch_mesh()))
        .clone();
    // Hidden until the first selection puts a patch in a slot.
    let hidden = Aabb::from_min_max(Vec3::ZERO, Vec3::ONE);
    let mut slots = Vec::with_capacity(WATER_POOL_CAPACITY);
    commands.entity(state.entity).with_children(|parent| {
        for _ in 0..WATER_POOL_CAPACITY {
            let mut spawn = |surface_pass: bool| {
                parent
                    .spawn((
                        Name::new(if surface_pass {
                            "Water surface"
                        } else {
                            "Water absorb"
                        }),
                        WaterPatch {
                            surface: surface_pass,
                        },
                        Mesh3d(mesh.clone()),
                        MeshMaterial3d(if surface_pass {
                            surface.clone()
                        } else {
                            absorb.clone()
                        }),
                        Transform::default(),
                        MeshTag::new(0),
                        hidden,
                        NoAutoAabb,
                        Visibility::Hidden,
                    ))
                    .id()
            };
            let absorb = spawn(false);
            let surface = spawn(true);
            slots.push(WaterSlot {
                absorb,
                surface,
                patch: None,
            });
        }
    });
    runtime.slots = slots;
    runtime.terrain = Some(state.entity);
}

/// The selected patches that hold water: the leaf bounds' deepest cell of the patch is at least
/// [`MIN_PATCH_DEPTH_M`].
pub(crate) fn wet_patches(
    selected: &[crate::terrain::lod::Patch],
    depth: &crate::terrain::lod::BoundsTree,
) -> Vec<crate::terrain::lod::Patch> {
    selected
        .iter()
        .copied()
        .filter(|p| depth.get(p.level, p.ix, p.iz).1 >= MIN_PATCH_DEPTH_M)
        .collect()
}

/// Put the terrain's selected wet patches into the pool, touching only the slots whose patch,
/// transform or bounds changed.
fn update_water_patches(
    terrain: Res<TerrainRuntime>,
    state: Option<Res<TerrainState>>,
    view: Res<ElementsView>,
    mut runtime: ResMut<WaterRuntime>,
    mut patches: Query<(&mut Transform, &mut Visibility, &mut MeshTag, &mut Aabb)>,
    units: Res<ElementsUnits>,
) {
    let (Some(state), Some(lod)) = (state, terrain.lod) else {
        return;
    };
    let (Some(boxes), Some(depth)) = (terrain.water_box.as_ref(), terrain.water_depth.as_ref())
    else {
        return;
    };
    if runtime.slots.is_empty() || view.entity.is_none() {
        return;
    }
    let mut wanted = wet_patches(&terrain.selected, depth);
    if wanted.len() > WATER_POOL_CAPACITY {
        if !runtime.warned_capacity {
            warn!(
                "the water pool wants {} patches, more than its {WATER_POOL_CAPACITY}; the farthest are dropped",
                wanted.len()
            );
            runtime.warned_capacity = true;
        }
        wanted.truncate(WATER_POOL_CAPACITY);
    }
    let current: Vec<_> = runtime.slots.iter().map(|s| s.patch).collect();
    let (assigned, used) = assign_slots(wanted, &current);
    let extent = state.grid.extent();
    for (i, slot) in runtime.slots.iter_mut().enumerate() {
        if used[i] {
            continue;
        }
        if slot.patch.take().is_some() {
            for entity in [slot.absorb, slot.surface] {
                if let Ok((_, mut visibility, _, _)) = patches.get_mut(entity)
                    && *visibility != Visibility::Hidden
                {
                    *visibility = Visibility::Hidden;
                }
            }
        }
    }
    for (patch, i) in assigned {
        let slot = &mut runtime.slots[i];
        slot.patch = Some(patch);
        // The patch is placed in the world: its metre-frame corner and edge × units.
        let (translation, scale) = patch_transform(&lod, extent, patch);
        let (translation, scale) = (
            units.point_to_world(translation),
            Vec3::new(units.world(scale.x), scale.y, units.world(scale.z)),
        );
        let (low, high) = boxes.get(patch.level, patch.ix, patch.iz);
        let bounds = Aabb::from_min_max(
            Vec3::new(0.0, units.world(low - WATER_AABB_PAD_M), 0.0),
            Vec3::new(1.0, units.world(high + WATER_AABB_PAD_M), 1.0),
        );
        for entity in [slot.absorb, slot.surface] {
            let Ok((mut transform, mut visibility, mut tag, mut aabb)) = patches.get_mut(entity)
            else {
                continue;
            };
            if transform.translation != translation || transform.scale != scale {
                transform.translation = translation;
                transform.scale = scale;
            }
            if tag.value != patch.level {
                *tag = MeshTag::new(patch.level);
            }
            if *aabb != bounds {
                *aabb = bounds;
            }
            if *visibility != Visibility::Inherited {
                *visibility = Visibility::Inherited;
            }
        }
    }
}

/// The uniform of the water materials for this frame.
pub(crate) fn water_params(
    settings: &WaterSettings,
    terrain: TerrainParamsGpu,
    lod: [Vec4; MAX_LEVELS],
    top: Vec4,
    shadows: (Vec4, Vec4),
    wind_ms: f32,
    units: f32,
) -> WaterParams {
    let u10 = settings.wind_speed_ms.unwrap_or(wind_ms);
    let silt = Color::srgb(
        settings.silt_tint.x,
        settings.silt_tint.y,
        settings.silt_tint.z,
    )
    .to_linear();
    WaterParams {
        terrain,
        lod,
        top,
        lod_centre: Vec4::ZERO,
        shadow: shadows.0,
        light: shadows.1,
        absorption: settings.absorption.extend(settings.refractive_index),
        silt: Vec3::new(silt.red, silt.green, silt.blue).extend(settings.turbidity_scattering),
        deep: settings.deep_color.extend(settings.clear_scattering),
        flow: Vec4::new(
            settings.flow_cycle_s,
            settings.tile_m,
            settings.flow_distortion,
            settings.ripple_scale * ripple_slope(u10) / NORMAL_RMS_SLOPE,
        ),
        foam: Vec4::new(
            settings.foam_speed_ms,
            settings.foam_thickness_m,
            settings.foam_albedo,
            settings.foam_roughness,
        ),
        waves: Vec4::ZERO,
        units,
    }
}

/// Keep both passes' uniform, buffers and shadow map current. Writes an asset only when
/// something changed, so a still scene re-prepares nothing.
#[allow(clippy::too_many_arguments)]
fn update_water_materials(
    state: Option<Res<TerrainState>>,
    terrain: Res<TerrainRuntime>,
    settings: Res<WaterSettings>,
    shadows: Option<Res<CloudShadows>>,
    wind: Option<Res<WindState>>,
    waves: Option<Res<WaveState>>,
    runtime: Res<WaterRuntime>,
    units: Res<ElementsUnits>,
    mut materials: ResMut<Assets<WaterMaterial>>,
) {
    let (Some(state), Some(lod)) = (state, terrain.lod.as_ref()) else {
        return;
    };
    let (shadow, light, map) = shadow_inputs(shadows.as_deref());
    let (lod, top) = lod_uniform(lod);
    // Quantised to 0.5 m/s: the wind drifts every frame, and a changed uniform re-prepares the
    // bind group.
    let wind_ms = wind.map_or(MAP_WIND_MS, |w| (w.speed_10m * 2.0).round() * 0.5);
    let mut params = water_params(
        &settings,
        TerrainParamsGpu::new(&state.grid, 1, state.is_live()),
        lod,
        top,
        (shadow, light),
        wind_ms,
        units.per_metre(),
    );
    params.lod_centre = lod_centre_uniform(terrain.lod_centre);
    let field = waves.as_deref().map(|w| w.field.clone());
    if let Some(w) = waves.as_deref() {
        params.waves = Vec4::new(w.texel_m[0], w.texels as f32, 1.0, 0.0);
    }
    for handle in [&runtime.absorb, &runtime.surface] {
        let Some(material) = materials.get(handle) else {
            continue;
        };
        let e = &material.extension;
        if (e.params != params
            || e.cells != state.cells
            || e.flow != state.flow_cells
            || e.cloud_shadow != map
            || e.waves != field)
            && let Some(mut material) = materials.get_mut(handle)
        {
            material.extension.params = params;
            material.extension.cells = state.cells.clone();
            material.extension.flow = state.flow_cells.clone();
            material.extension.cloud_shadow = map.clone();
            material.extension.waves = field.clone();
        }
    }
}
