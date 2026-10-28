//! The terrain's materials: `TerrainMaterial` (the owned render's displaced, layer-blended
//! ground) and `GroundExtension` (the same cloud shadows and water darkening on a consumer's own
//! meshes). Both read the packed `terrain_cells` buffer; the crate keeps their uniforms and buffer
//! handle current.

use bevy::asset::{AssetPath, RenderAssetUsages, uuid_handle};
use bevy::pbr::{ExtendedMaterial, MaterialExtension};
use bevy::prelude::*;
use bevy::render::render_resource::{AsBindGroup, ShaderType};
use bevy::render::storage::ShaderBuffer;
use bevy::shader::ShaderRef;

use super::lod::{LodConfig, MAX_LEVELS};
use super::{TerrainParamsGpu, TerrainState};
use crate::clouds::CloudShadows;
use crate::cover::{CoverState, WetnessSettings};
use crate::weathered::ElementsGlobals;

/// A one-cell `terrain_cells` stand-in for shaders that bind the terrain before one exists. Its
/// uniform says "no terrain" (`TerrainParams::misc.z = 0`).
pub(crate) const NO_TERRAIN: Handle<ShaderBuffer> =
    uuid_handle!("2f3c9a40-6a44-4b3a-9a67-0d1b5e7c4a11");

/// Insert the placeholder buffer asset. A zero cell: height 0, no sediment, no water, flat.
pub(crate) fn insert_placeholder(buffers: &mut Assets<ShaderBuffer>) {
    let mut buffer = ShaderBuffer::new(vec![[0f32; 4]], RenderAssetUsages::RENDER_WORLD);
    buffer.label = "elements_terrain_none".into();
    if let Err(error) = buffers.insert(NO_TERRAIN.id(), buffer) {
        error!("could not create the terrain placeholder buffer: {error}");
    }
}

/// Uniform of the owned terrain material. Mirrors `TerrainMaterialParams` of `render.wesl`.
#[derive(Clone, Copy, Debug, Default, PartialEq, ShaderType, Reflect)]
#[reflect(opaque)]
pub struct TerrainMaterialParams {
    pub(crate) terrain: TerrainParamsGpu,
    /// Per level: x morph start, y range (m).
    pub(crate) lod: [Vec4; MAX_LEVELS],
    /// x: coarsest level.
    pub(crate) top: Vec4,
    /// Cloud shadow map origin xy, extent, camera transmittance.
    pub(crate) shadow: Vec4,
    /// Unit direction toward the light; w = 1 with a map.
    pub(crate) light: Vec4,
    /// The water's absorption σ (m⁻¹) in xyz and its refractive index in w (0: no sun-leg
    /// attenuation under water).
    pub(crate) absorption: Vec4,
    /// Ground cover: x 1 when the cover maps are bound, y rain intensity `min(1, R_liquid / 20)`,
    /// z puddle start depth (m), w full-puddle depth (m).
    pub(crate) cover: Vec4,
    /// Phase of each of the four ripple layers, 0–1 (zero while it does not rain).
    pub(crate) ripple: Vec4,
}

/// The extension that turns a `StandardMaterial` into the terrain's displaced, layer-blended
/// ground. The crate fills it.
#[derive(Asset, AsBindGroup, Reflect, Debug, Clone)]
pub struct TerrainExtension {
    #[uniform(100)]
    pub(crate) params: TerrainMaterialParams,
    #[storage(101, read_only)]
    pub(crate) cells: Handle<ShaderBuffer>,
    /// The cover pack (snow depth, wetness, SWE): read by the vertex stage for the snow
    /// displacement too.
    #[storage(102, read_only)]
    pub(crate) cover: Handle<ShaderBuffer>,
    #[texture(103)]
    #[sampler(104)]
    pub(crate) cloud_shadow: Option<Handle<Image>>,
}

const RENDER_SHADER: &str = "embedded://bevy_elements/terrain/render.wesl";
const GROUND_SHADER: &str = "embedded://bevy_elements/terrain/ground.wesl";

impl MaterialExtension for TerrainExtension {
    fn vertex_shader() -> ShaderRef {
        ShaderRef::Path(AssetPath::parse(RENDER_SHADER))
    }

    fn fragment_shader() -> ShaderRef {
        ShaderRef::Path(AssetPath::parse(RENDER_SHADER))
    }

    fn prepass_vertex_shader() -> ShaderRef {
        ShaderRef::Path(AssetPath::parse(RENDER_SHADER))
    }

    fn deferred_vertex_shader() -> ShaderRef {
        ShaderRef::Path(AssetPath::parse(RENDER_SHADER))
    }

    fn deferred_fragment_shader() -> ShaderRef {
        ShaderRef::Path(AssetPath::parse(RENDER_SHADER))
    }
}

/// The terrain's ground material: a `StandardMaterial` extended with [`TerrainExtension`].
pub type TerrainMaterial = ExtendedMaterial<StandardMaterial, TerrainExtension>;

/// Uniform of [`GroundExtension`]. Mirrors `GroundParams` of `ground.wesl`.
#[derive(Clone, Copy, Debug, PartialEq, ShaderType, Reflect)]
#[reflect(opaque)]
pub struct GroundParams {
    pub(crate) terrain: TerrainParamsGpu,
    /// Cloud shadow map origin xy, extent, camera transmittance.
    pub(crate) shadow: Vec4,
    /// Unit direction toward the light; w = 1 with a map.
    pub(crate) light: Vec4,
    /// The water's absorption σ (m⁻¹) in xyz and its refractive index in w.
    pub(crate) absorption: Vec4,
    /// As `TerrainMaterialParams::cover`.
    pub(crate) cover: Vec4,
    /// As `TerrainMaterialParams::ripple`.
    pub(crate) ripple: Vec4,
    /// x: `column_gate_m`, the depth below the column top where snow, wetness and puddles stop
    /// (m); `NO_COLUMN_GATE` for none.
    pub(crate) gate: Vec4,
}

/// `GroundParams::gate.x` for a ground that is not gated by depth below the column top.
pub(crate) const NO_COLUMN_GATE: f32 = -1.0e30;

impl Default for GroundParams {
    fn default() -> Self {
        Self {
            terrain: TerrainParamsGpu::default(),
            shadow: Vec4::ZERO,
            light: Vec4::ZERO,
            absorption: Vec4::ZERO,
            cover: Vec4::ZERO,
            ripple: Vec4::ZERO,
            gate: Vec4::new(NO_COLUMN_GATE, 0.0, 0.0, 0.0),
        }
    }
}

/// Shades a consumer's own mesh as the terrain's ground
/// ([`TerrainRender::External`](crate::TerrainRender::External)): snow covers it, it darkens and
/// glosses when wet, water standing on it shows as a puddle with ripples while it rains, and
/// cloud shadows fall on it, all by world XZ. It does not displace or blend layers; the mesh
/// keeps its own geometry and vertex shader (so snow whitens it but does not raise it).
///
/// Use it as `ExtendedMaterial<StandardMaterial, GroundExtension>` with
/// `GroundExtension::default()`; the crate keeps its buffer, grid and shadow map current every
/// frame, and until a terrain exists it shades as plain `StandardMaterial` (no water, no
/// terrain) with the cloud shadows.
#[derive(Asset, AsBindGroup, Reflect, Debug, Clone)]
pub struct GroundExtension {
    #[uniform(100)]
    pub(crate) params: GroundParams,
    #[storage(101, read_only)]
    pub(crate) terrain: Handle<ShaderBuffer>,
    #[storage(102, read_only)]
    pub(crate) cover: Handle<ShaderBuffer>,
    #[texture(103)]
    #[sampler(104)]
    pub(crate) cloud_shadow: Option<Handle<Image>>,
}

impl Default for GroundExtension {
    fn default() -> Self {
        Self {
            params: GroundParams::default(),
            terrain: NO_TERRAIN,
            cover: NO_TERRAIN,
            cloud_shadow: None,
        }
    }
}

impl GroundExtension {
    /// Snow, wetness and puddles shade only a fragment within `metres` of the column top or above
    /// it: the terrain's surface height at the fragment's XZ, minus `metres`. A fragment deeper
    /// than that (the walls and floor of a tunnel under a snowy hill) is left as the material
    /// made it. The maps are heightfields, so without the gate a voxel volume's tunnels are
    /// shaded white wherever the map above them has snow.
    ///
    /// Set it on the ground material of a [`VoxelGround`](crate::VoxelGround) volume to the
    /// volume's voxel cell size; the slab of snow sits above `top − cell`. The default, no gate,
    /// is right for a ground that is itself a heightfield.
    #[must_use]
    pub fn with_column_gate_m(mut self, metres: f32) -> Self {
        self.params.gate.x = metres;
        self
    }
}

impl MaterialExtension for GroundExtension {
    fn fragment_shader() -> ShaderRef {
        ShaderRef::Path(AssetPath::parse(GROUND_SHADER))
    }
}

/// The ground material for a consumer's mesh: see [`GroundExtension`].
pub type GroundMaterial = ExtendedMaterial<StandardMaterial, GroundExtension>;

/// The cloud shadow part of both uniforms: map origin and extent, the camera's transmittance, the
/// light direction, and the map handle.
pub(crate) fn shadow_inputs(shadows: Option<&CloudShadows>) -> (Vec4, Vec4, Option<Handle<Image>>) {
    match shadows {
        Some(s) => (
            Vec4::new(s.origin.x, s.origin.y, s.extent, s.camera_transmittance),
            s.light_direction.extend(1.0),
            Some(s.map.clone()),
        ),
        None => (
            Vec4::new(0.0, 0.0, 1.0, 1.0),
            Vec4::new(0.0, 1.0, 0.0, 0.0),
            None,
        ),
    }
}

/// The per-level morph ranges and the coarsest level, as the vertex shaders read them.
pub(crate) fn lod_uniform(config: &LodConfig) -> ([Vec4; MAX_LEVELS], Vec4) {
    (
        std::array::from_fn(|k| {
            if (k as u32) < config.levels {
                Vec4::new(
                    config.morph_start_m(k as u32),
                    config.range_m(k as u32),
                    0.0,
                    0.0,
                )
            } else {
                Vec4::ZERO
            }
        }),
        Vec4::new((config.levels - 1) as f32, 0.0, 0.0, 0.0),
    )
}

/// The cover part of both uniforms: the live flag, the rain intensity and the puddle depths, and
/// the ripple phases (frozen at zero while it does not rain, so a dry scene re-prepares nothing).
fn cover_inputs(
    live: bool,
    wetness: Option<&WetnessSettings>,
    globals: Option<&ElementsGlobals>,
) -> (Vec4, Vec4) {
    let defaults = WetnessSettings::default();
    let wetness = wetness.unwrap_or(&defaults);
    let rain = globals.map_or(0.0, |g| g.rain_intensity);
    let ripple = match globals {
        Some(g) if rain > 0.0 => Vec4::from_array(g.ripple_phase),
        _ => Vec4::ZERO,
    };
    (
        Vec4::new(
            f32::from(u8::from(live)),
            rain,
            wetness.puddle_min_m,
            wetness.puddle_full_m,
        ),
        ripple,
    )
}

/// Keep every terrain and ground material's uniform, buffer and shadow map current. Writes an
/// asset only when something changed, so a still scene re-prepares nothing.
#[allow(clippy::too_many_arguments)]
pub(super) fn update_materials(
    state: Option<Res<TerrainState>>,
    runtime: Res<super::TerrainRuntime>,
    shadows: Option<Res<CloudShadows>>,
    cover: Option<Res<CoverState>>,
    wetness: Option<Res<WetnessSettings>>,
    globals: Option<Res<ElementsGlobals>>,
    mut terrain_materials: ResMut<Assets<TerrainMaterial>>,
    mut ground_materials: ResMut<Assets<GroundMaterial>>,
    water: Option<Res<crate::hydrology::WaterSettings>>,
) {
    let (shadow, light, map) = shadow_inputs(shadows.as_deref());
    let absorption = water.map_or(Vec4::ZERO, |w| w.absorption.extend(w.refractive_index));
    let live = state.as_ref().is_some_and(|s| s.is_live());
    let terrain_params = state
        .as_ref()
        .map(|s| TerrainParamsGpu::new(&s.grid, 1, live))
        .unwrap_or_default();
    let cells = state.as_ref().map_or(NO_TERRAIN, |s| s.cells.clone());
    let cover_cells = cover.as_ref().map_or(NO_TERRAIN, |c| c.cover_cells.clone());
    let (cover_params, ripple) = cover_inputs(
        cover.is_some() && live,
        wetness.as_deref(),
        globals.as_deref(),
    );

    if let (Some(state), Some(config)) = (state.as_ref(), runtime.lod.as_ref()) {
        let (lod, top) = lod_uniform(config);
        let params = TerrainMaterialParams {
            terrain: terrain_params,
            lod,
            top,
            shadow,
            light,
            absorption,
            cover: cover_params,
            ripple,
        };
        let ids: Vec<_> = terrain_materials.iter().map(|(id, _)| id).collect();
        for id in ids {
            let Some(material) = terrain_materials.get(id) else {
                continue;
            };
            let e = &material.extension;
            if (e.params != params
                || e.cells != state.cells
                || e.cover != cover_cells
                || e.cloud_shadow != map)
                && let Some(mut material) = terrain_materials.get_mut(id)
            {
                material.extension.params = params;
                material.extension.cells = state.cells.clone();
                material.extension.cover = cover_cells.clone();
                material.extension.cloud_shadow = map.clone();
            }
        }
    }

    let params = GroundParams {
        terrain: terrain_params,
        shadow,
        light,
        absorption,
        cover: cover_params,
        ripple,
        gate: GroundParams::default().gate,
    };
    let ids: Vec<_> = ground_materials.iter().map(|(id, _)| id).collect();
    for id in ids {
        let Some(material) = ground_materials.get(id) else {
            continue;
        };
        let e = &material.extension;
        // The gate is the material's own: everything else is shared.
        let params = GroundParams {
            gate: e.params.gate,
            ..params
        };
        if (e.params != params
            || e.terrain != cells
            || e.cover != cover_cells
            || e.cloud_shadow != map)
            && let Some(mut material) = ground_materials.get_mut(id)
        {
            material.extension.params = params;
            material.extension.terrain = cells.clone();
            material.extension.cover = cover_cells.clone();
            material.extension.cloud_shadow = map.clone();
        }
    }
}

#[cfg(test)]
mod tests {
    /// Terrain validation 9: each extension declares at most two storage buffers (terrain cells
    /// now, the cover pack with M4), so with Bevy's five in the fragment stage the WebGPU default
    /// of eight holds. Counted from the declarations: the layout itself needs a device.
    #[test]
    fn each_extension_binds_at_most_two_storage_buffers() {
        let source = include_str!("material.rs");
        for extension in ["pub struct TerrainExtension", "pub struct GroundExtension"] {
            let body = &source[source.find(extension).expect("the struct")..];
            let body = &body[..body.find("\n}").expect("the end of the struct")];
            let storage = body.matches("#[storage(").count();
            assert!(
                (1..=2).contains(&storage),
                "{extension}: {storage} storage buffers"
            );
        }
    }
}
