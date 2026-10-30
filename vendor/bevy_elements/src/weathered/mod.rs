//! Weathered materials for any mesh: [`WeatheredMaterial`] darkens and glosses when it is wet,
//! streaks on vertical faces while it rains, turns white on up-facing normals under snow and takes
//! the cloud shadows, all from one shared storage buffer ([`ElementsGlobals`]) the crate keeps
//! current. See `docs/research/wetness.md` ("WeatheredMaterial and ElementsGlobals") and
//! `docs/research/snow-cover.md` ("Coverage and shading").
//!
//! The values are the scene's: [`SurfaceWeathering`] (the CPU point model until the cover maps
//! replace it), the [`Weather`] and the [`CloudShadows`]. A material's own
//! [`WeatheredExtension::exposure`] scales every weather effect: 1 is open sky, 0 is sheltered.

mod model;

#[cfg(test)]
mod mirror;

use bevy::asset::{AssetPath, RenderAssetUsages, embedded_asset, uuid_handle};
use bevy::pbr::{ExtendedMaterial, MaterialExtension, MaterialPlugin};
use bevy::prelude::*;
use bevy::render::render_resource::AsBindGroup;
use bevy::render::storage::ShaderBuffer;
use bevy::shader::ShaderRef;

use crate::climate::{SurfaceWeathering, Weather};
use crate::clouds::CloudShadows;
use crate::plugin::{ElementsSystems, ElementsUnits, ensure_gpu_core};
use crate::trails::TrailState;
use crate::trails::model::MARCH_BOUND_M;

/// Linear albedo of snow, bit-equal to `bevy_flora`'s `SNOW_ALBEDO` so trees and buildings match.
pub const SNOW_ALBEDO: LinearRgba = LinearRgba::new(0.80, 0.83, 0.88, 1.0);

/// Perceptual roughness of snow (`bevy_flora`'s 0.9; the plan's 0.6 is Deviation 4 of
/// `docs/research/snow-cover.md`).
pub const SNOW_ROUGHNESS: f32 = 0.9;

/// The crate's globals as a GPU storage buffer, for a material or pass of your own: the 128-byte
/// [`ElementsGlobals`] (wetness, snow cover and snow depth of exposed ground, the rain intensity,
/// the mean cloud transmittance, the cloud shadow map's placement, the ripple and streak phases,
/// the wind, the light direction, the world units per metre and the trail window), laid out as
/// the WGSL `bevy_elements::surface_cover::ElementsGlobals` (eight `vec4<f32>`; the field
/// comments there give each component).
///
/// [`WeatheredPlugin`] (in `ElementsPlugins`) creates the asset and rewrites it whenever a value
/// changes, after [`ElementsSystems::Simulation`]; the handle is a fixed id, so a material can
/// name it before the plugin has run. Bind it read-only as a storage buffer at any free slot of
/// your own `AsBindGroup` (`#[storage(N, read_only)] globals: Handle<ShaderBuffer>` set to this)
/// and import the struct; the cloud shadow map is [`CloudShadows::map`]
/// (`docs/research/integration.md`).
pub const ELEMENTS_GLOBALS_BUFFER: Handle<ShaderBuffer> =
    uuid_handle!("6b0e2d51-7c1f-4f0d-8a53-2e9a41c7d0b8");

/// `vec4<f32>`s of the GPU buffer (`ElementsGlobals` of `surface_cover.wesl`).
const GLOBALS_VEC4S: usize = 8;

/// What the crate tells every weathered material, rewritten each frame after
/// [`ElementsSystems::Simulation`] by [`WeatheredPlugin`] and uploaded to one shared GPU
/// storage buffer (`surface_cover::ElementsGlobals`, eight `vec4<f32>`, 128 B).
///
/// Wetness and snow cover are those of exposed flat ground, [`SurfaceWeathering`]'s. The
/// shader scales them by each material's [`WeatheredExtension::exposure`] and, for snow, by the
/// face's normal.
#[derive(Resource, Clone, Copy, Debug, PartialEq, Reflect)]
pub struct ElementsGlobals {
    /// Wetness of exposed ground, 0 dry – 1 soaked.
    pub wetness: f32,
    /// Snow cover of exposed ground, 0–1.
    pub snow_cover: f32,
    /// Rain intensity `min(1, R_liquid / 20)`, with `R_liquid` the liquid precipitation in mm/h.
    pub rain_intensity: f32,
    /// Mean direct cloud transmittance over the cloud shadow map, 0–1 (1 without clouds): the
    /// share of the clear beam the lights carry (`CloudShadows::mean_transmittance`).
    pub mean_transmittance: f32,
    /// Metre-frame XZ of the cloud shadow map's minimum corner, m (world ÷ `units_per_metre`).
    pub shadow_origin: Vec2,
    /// Side of the cloud shadow map, m.
    pub shadow_extent: f32,
    /// Whether a cloud shadow map exists.
    pub shadow_enabled: bool,
    /// Phase of each of the four ripple layers, 0–1, computed in f64 from the virtual clock.
    pub ripple_phase: [f32; 4],
    /// The wind the weather blows toward, XZ, m/s.
    pub wind: Vec2,
    /// Phase of the rain streaks, s: the virtual clock wrapped at 8192 s, the lattice period of
    /// the streak noise over its drift speed, so the wrap is seamless.
    pub streak_phase: f32,
    /// Unit direction toward the light the cloud shadow map was made for.
    pub light_direction: Vec3,
    /// World units per metre ([`crate::ElementsUnits::units_per_metre`]), as the shader divides
    /// a world position by it to reach the metre frame the fields above are in.
    pub units_per_metre: f32,
    /// Snow depth of exposed ground, m ([`SurfaceWeathering::snow_depth_m`]): a trail pressed
    /// deeper shows the ground under the snow.
    pub snow_depth_m: f32,
    /// The trail window's texel edge, m ([`TrailState`]); 0 without trails.
    pub trail_texel_m: f32,
    /// Texels a side of the trail window; 0 without trails.
    pub trail_texels: u32,
    /// The trail window's centre, metre-frame XZ ([`TrailState::centre`]).
    pub trail_centre: Vec2,
    /// The share of the light on a horizontal surface carried by the sun or moon's beam, which a
    /// trail's self-shadow removes.
    pub trail_direct_share: f32,
}

impl Default for ElementsGlobals {
    fn default() -> Self {
        Self {
            wetness: 0.0,
            snow_cover: 0.0,
            rain_intensity: 0.0,
            mean_transmittance: 1.0,
            shadow_origin: Vec2::ZERO,
            shadow_extent: 1.0,
            shadow_enabled: false,
            ripple_phase: [0.0; 4],
            wind: Vec2::ZERO,
            streak_phase: 0.0,
            light_direction: Vec3::Y,
            units_per_metre: 1.0,
            snow_depth_m: 0.0,
            trail_texel_m: 0.0,
            trail_texels: 0,
            trail_centre: Vec2::ZERO,
            trail_direct_share: 0.0,
        }
    }
}

impl ElementsGlobals {
    /// The globals for this frame. `seconds` is the virtual clock in f64 (the ripple phases are
    /// not representable as a wrapped f32 clock, `docs/research/wetness.md`); `units` is
    /// [`crate::ElementsUnits::per_metre`].
    pub(crate) fn gather(
        weathering: Option<&SurfaceWeathering>,
        weather: &Weather,
        shadows: Option<&CloudShadows>,
        trails: Option<&TrailState>,
        seconds: f64,
        units: f32,
    ) -> Self {
        let (shadow_origin, shadow_extent, shadow_enabled, mean_transmittance, light) =
            match shadows {
                Some(s) => (
                    s.origin,
                    s.extent,
                    true,
                    s.mean_transmittance,
                    s.light_direction,
                ),
                None => (Vec2::ZERO, 1.0, false, 1.0, Vec3::Y),
            };
        Self {
            wetness: weathering.map_or(0.0, |w| w.wetness),
            snow_cover: weathering.map_or(0.0, |w| w.snow_cover),
            rain_intensity: model::rain_intensity(weather),
            mean_transmittance,
            shadow_origin,
            shadow_extent,
            shadow_enabled,
            ripple_phase: model::ripple_phases(seconds),
            wind: weather.wind_direction * weather.wind_speed_ms,
            streak_phase: model::streak_phase(seconds),
            light_direction: light,
            units_per_metre: units,
            snow_depth_m: weathering.map_or(0.0, |w| w.snow_depth_m),
            trail_texel_m: trails.map_or(0.0, |t| t.texel_m),
            trail_texels: trails.map_or(0, |t| t.texels),
            trail_centre: trails.map_or(Vec2::ZERO, |t| t.centre),
            trail_direct_share: trails.map_or(0.0, |t| t.direct_share),
        }
    }

    /// The buffer's contents, `surface_cover::ElementsGlobals` field for field.
    pub(crate) fn pack(&self) -> [[f32; 4]; GLOBALS_VEC4S] {
        [
            [
                self.wetness,
                self.snow_cover,
                self.rain_intensity,
                self.mean_transmittance,
            ],
            [
                self.shadow_origin.x,
                self.shadow_origin.y,
                self.shadow_extent,
                f32::from(u8::from(self.shadow_enabled)),
            ],
            self.ripple_phase,
            [self.wind.x, self.wind.y, self.streak_phase, 0.0],
            [
                self.light_direction.x,
                self.light_direction.y,
                self.light_direction.z,
                0.0,
            ],
            [self.units_per_metre, 0.0, 0.0, 0.0],
            [
                self.trail_texel_m,
                self.trail_texels as f32,
                f32::from(u8::from(self.trail_texels > 0)),
                MARCH_BOUND_M,
            ],
            [
                self.trail_direct_share,
                self.trail_centre.x,
                self.trail_centre.y,
                self.snow_depth_m,
            ],
        ]
    }
}

fn globals_buffer(globals: &ElementsGlobals) -> ShaderBuffer {
    // Main-world data is kept (`RenderAssetUsages::default()`), so replacing it rewrites the GPU
    // copy in place: no bind group is re-prepared.
    let mut buffer = ShaderBuffer::new(globals.pack().to_vec(), RenderAssetUsages::default());
    buffer.label = "elements_globals".into();
    buffer
}

/// The extension that weathers a `StandardMaterial`: see [`WeatheredMaterial`].
///
/// Binds the crate's shared [`ElementsGlobals`] buffer (one storage buffer, binding 101,
/// fragment stage), the cloud shadow map and the trail field; the crate keeps them current.
/// Build one with [`WeatheredExtension::default`] (exposed) or
/// [`WeatheredExtension::with_exposure`], and [`with_trails`](Self::with_trails) for a ground.
#[derive(Asset, AsBindGroup, Reflect, Debug, Clone)]
pub struct WeatheredExtension {
    /// How much of the weather reaches this material: 1 under open sky, 0 sheltered (no wetness,
    /// no snow, no streaks). Every weather effect is multiplied by it. It does not follow a
    /// [`PrecipitationBlocker`](crate::PrecipitationBlocker) by itself: set it from your own
    /// shelter test.
    #[uniform(100)]
    pub exposure: f32,
    /// 1 when the material draws trails ([`with_trails`](Self::with_trails)).
    #[uniform(100)]
    trails: f32,
    /// The stone grain's feature size, m: 0 for none ([`with_grain`](Self::with_grain)).
    #[uniform(100)]
    grain_m: f32,
    /// A colour grade of the base colour (tint, saturation): the identity, except on the crate's
    /// scattered rocks, which carry [`TerrainLook`](crate::TerrainLook).
    #[uniform(100)]
    pub(crate) look: Vec4,
    #[storage(101, read_only, visibility(fragment))]
    pub(crate) globals: Handle<ShaderBuffer>,
    #[texture(102)]
    #[sampler(103)]
    pub(crate) cloud_shadow: Option<Handle<Image>>,
    /// The trail field (`TrailState::field`), bound while the material draws trails.
    #[texture(104)]
    #[sampler(105)]
    pub(crate) trail: Option<Handle<Image>>,
}

impl Default for WeatheredExtension {
    fn default() -> Self {
        Self::with_exposure(1.0)
    }
}

impl WeatheredExtension {
    /// An extension with the given [`exposure`](Self::exposure).
    pub fn with_exposure(exposure: f32) -> Self {
        Self {
            exposure,
            trails: 0.0,
            grain_m: 0.0,
            look: Vec4::ONE,
            globals: ELEMENTS_GLOBALS_BUFFER,
            cloud_shadow: None,
            trail: None,
        }
    }

    /// The same extension, drawing the trails [`Imprint`](crate::Imprint)s press into it: put it
    /// on the material of a ground the trails run on (a [`TrailGround`](crate::TrailGround), or
    /// a mesh over the crate's terrain). The trails are drawn per pixel by parallax occlusion
    /// mapping over the face's own plane, on faces up to 60° from horizontal, in forward
    /// rendering; the field is a 2D map by world XZ, so keep it off roofs and floors above a
    /// path (`docs/research/trails-game-ground.md`).
    pub fn with_trails(mut self) -> Self {
        self.trails = 1.0;
        self
    }

    /// Whether the material draws trails.
    pub fn draws_trails(&self) -> bool {
        self.trails > 0.5
    }

    /// The same extension with a **stone grain**: the crate's band-limited 3D noise at the world
    /// point, a mottle at `feature_m` and a speckle at an eighth of it (two octaves each),
    /// varying the albedo and bending the shading normal, so a stone surface has relief at arm's
    /// length without a texture or UVs. It is in world space, so it suits meshes that do not
    /// move (the scattered rocks use it); each octave fades out where it is finer than the
    /// pixel's footprint. 0 turns it off.
    pub fn with_grain(mut self, feature_m: f32) -> Self {
        self.grain_m = feature_m.max(0.0);
        self
    }

    /// The stone grain's feature size, m (0 for none).
    pub fn grain_m(&self) -> f32 {
        self.grain_m
    }
}

const SHADER: &str = "embedded://bevy_elements/weathered/weathered.wesl";

impl MaterialExtension for WeatheredExtension {
    fn fragment_shader() -> ShaderRef {
        ShaderRef::Path(AssetPath::parse(SHADER))
    }

    // The base shader would drop the weather in deferred (`extended_material.rs:294-299`); the
    // prepass fragment stays the base's, since nothing here changes alpha or geometry.
    fn deferred_fragment_shader() -> ShaderRef {
        ShaderRef::Path(AssetPath::parse(SHADER))
    }
}

/// A `StandardMaterial` that is rained and snowed on: wet surfaces darken and gloss, rain runs
/// down vertical faces, up-facing faces whiten under snow with the curve `bevy_flora` uses, and
/// cloud shadows fall on it. See [`WeatheredExtension`].
///
/// ```no_run
/// # use bevy::prelude::*;
/// # use bevy_elements::*;
/// fn spawn(mut commands: Commands, mut meshes: ResMut<Assets<Mesh>>, mut materials: ResMut<Assets<WeatheredMaterial>>) {
///     commands.spawn((
///         Mesh3d(meshes.add(Cuboid::new(8.0, 6.0, 8.0))),
///         MeshMaterial3d(materials.add(WeatheredMaterial {
///             base: StandardMaterial { base_color: Color::srgb(0.6, 0.35, 0.3), perceptual_roughness: 0.9, ..default() },
///             extension: WeatheredExtension::default(),
///         })),
///     ));
/// }
/// ```
pub type WeatheredMaterial = ExtendedMaterial<StandardMaterial, WeatheredExtension>;

/// Weathered materials: [`WeatheredMaterial`], and the [`ElementsGlobals`] they share.
///
/// Reads [`SurfaceWeathering`], [`Weather`] and [`CloudShadows`] each frame and writes
/// [`ElementsGlobals`] and its GPU buffer; the cloud shadow map is attached to every
/// `WeatheredMaterial` asset.
pub struct WeatheredPlugin;

impl Plugin for WeatheredPlugin {
    fn build(&self, app: &mut App) {
        ensure_gpu_core(app);
        embedded_asset!(app, "weathered.wesl");
        app.add_plugins(MaterialPlugin::<WeatheredMaterial>::default())
            .init_resource::<ElementsGlobals>()
            .add_systems(
                Update,
                (update_globals, update_materials)
                    .chain()
                    .after(ElementsSystems::Simulation),
            );
    }
}

/// Gather this frame's [`ElementsGlobals`] and rewrite the shared buffer when they changed.
#[allow(clippy::too_many_arguments)]
fn update_globals(
    weathering: Option<Res<SurfaceWeathering>>,
    weather: Res<Weather>,
    shadows: Option<Res<CloudShadows>>,
    trails: Option<Res<TrailState>>,
    time: Res<Time<Virtual>>,
    units: Res<ElementsUnits>,
    mut globals: ResMut<ElementsGlobals>,
    buffers: Option<ResMut<Assets<ShaderBuffer>>>,
) {
    let next = ElementsGlobals::gather(
        weathering.as_deref(),
        &weather,
        shadows.as_deref(),
        trails.as_deref(),
        time.elapsed_secs_f64(),
        units.per_metre(),
    );
    let Some(mut buffers) = buffers else {
        // Headless: no renderer, nothing to upload.
        globals.set_if_neq(next);
        return;
    };
    if (globals.set_if_neq(next) || !buffers.contains(ELEMENTS_GLOBALS_BUFFER.id()))
        && let Err(error) = buffers.insert(ELEMENTS_GLOBALS_BUFFER.id(), globals_buffer(&next))
    {
        error!("could not write the weathered materials' globals buffer: {error}");
    }
}

/// Attach the cloud shadow map to every weathered material, and the trail field to those that
/// draw trails. Writes an asset only when one changed, so a still scene re-prepares nothing.
fn update_materials(
    shadows: Option<Res<CloudShadows>>,
    trails: Option<Res<TrailState>>,
    mut materials: ResMut<Assets<WeatheredMaterial>>,
) {
    let map = shadows.as_ref().map(|s| s.map.clone());
    let field = trails.as_ref().map(|t| t.field.clone());
    let wanted = |e: &WeatheredExtension| field.clone().filter(|_| e.draws_trails());
    let stale: Vec<_> = materials
        .iter()
        .filter(|(_, m)| {
            m.extension.cloud_shadow != map || m.extension.trail != wanted(&m.extension)
        })
        .map(|(id, _)| id)
        .collect();
    for id in stale {
        if let Some(mut material) = materials.get_mut(id) {
            material.extension.cloud_shadow = map.clone();
            material.extension.trail = wanted(&material.extension);
        }
    }
}

#[cfg(test)]
mod tests;
