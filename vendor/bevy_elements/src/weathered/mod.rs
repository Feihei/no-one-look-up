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
use crate::plugin::{ElementsSystems, ensure_gpu_core};

/// Linear albedo of snow, bit-equal to `bevy_flora`'s `SNOW_ALBEDO` so trees and buildings match.
pub const SNOW_ALBEDO: LinearRgba = LinearRgba::new(0.80, 0.83, 0.88, 1.0);

/// Perceptual roughness of snow (`bevy_flora`'s 0.9; the plan's 0.6 is Deviation 4 of
/// `docs/research/snow-cover.md`).
pub const SNOW_ROUGHNESS: f32 = 0.9;

/// The shared buffer every weathered material binds. Its asset is inserted and rewritten by
/// [`WeatheredPlugin`]; the handle is a fixed id so a material can name it before the plugin has
/// run.
pub(crate) const GLOBALS_BUFFER: Handle<ShaderBuffer> =
    uuid_handle!("6b0e2d51-7c1f-4f0d-8a53-2e9a41c7d0b8");

/// `vec4<f32>`s of the GPU buffer (`ElementsGlobals` of `surface_cover.wesl`).
const GLOBALS_VEC4S: usize = 5;

/// What the crate tells every weathered material, rewritten each frame after
/// [`ElementsSystems::Simulation`] by [`WeatheredPlugin`] and uploaded to one shared GPU
/// storage buffer (`surface_cover::ElementsGlobals`, five `vec4<f32>`, 80 B).
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
    /// Mean direct cloud transmittance at the camera, 0–1 (1 without clouds).
    pub camera_transmittance: f32,
    /// World XZ of the cloud shadow map's minimum corner, m.
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
}

impl Default for ElementsGlobals {
    fn default() -> Self {
        Self {
            wetness: 0.0,
            snow_cover: 0.0,
            rain_intensity: 0.0,
            camera_transmittance: 1.0,
            shadow_origin: Vec2::ZERO,
            shadow_extent: 1.0,
            shadow_enabled: false,
            ripple_phase: [0.0; 4],
            wind: Vec2::ZERO,
            streak_phase: 0.0,
            light_direction: Vec3::Y,
        }
    }
}

impl ElementsGlobals {
    /// The globals for this frame. `seconds` is the virtual clock in f64 (the ripple phases are
    /// not representable as a wrapped f32 clock, `docs/research/wetness.md`).
    pub(crate) fn gather(
        weathering: Option<&SurfaceWeathering>,
        weather: &Weather,
        shadows: Option<&CloudShadows>,
        seconds: f64,
    ) -> Self {
        let (shadow_origin, shadow_extent, shadow_enabled, camera_transmittance, light) =
            match shadows {
                Some(s) => (
                    s.origin,
                    s.extent,
                    true,
                    s.camera_transmittance,
                    s.light_direction,
                ),
                None => (Vec2::ZERO, 1.0, false, 1.0, Vec3::Y),
            };
        Self {
            wetness: weathering.map_or(0.0, |w| w.wetness),
            snow_cover: weathering.map_or(0.0, |w| w.snow_cover),
            rain_intensity: model::rain_intensity(weather),
            camera_transmittance,
            shadow_origin,
            shadow_extent,
            shadow_enabled,
            ripple_phase: model::ripple_phases(seconds),
            wind: weather.wind_direction * weather.wind_speed_ms,
            streak_phase: model::streak_phase(seconds),
            light_direction: light,
        }
    }

    /// The buffer's contents, `surface_cover::ElementsGlobals` field for field.
    pub(crate) fn pack(&self) -> [[f32; 4]; GLOBALS_VEC4S] {
        [
            [
                self.wetness,
                self.snow_cover,
                self.rain_intensity,
                self.camera_transmittance,
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
/// fragment stage) and the cloud shadow map; the crate keeps both current. Build one with
/// [`WeatheredExtension::default`] (exposed) or [`WeatheredExtension::with_exposure`].
#[derive(Asset, AsBindGroup, Reflect, Debug, Clone)]
pub struct WeatheredExtension {
    /// How much of the weather reaches this material: 1 under open sky, 0 sheltered (no wetness,
    /// no snow, no streaks). Every weather effect is multiplied by it. It does not follow a
    /// [`PrecipitationBlocker`](crate::PrecipitationBlocker) by itself: set it from your own
    /// shelter test.
    #[uniform(100)]
    pub exposure: f32,
    #[storage(101, read_only, visibility(fragment))]
    pub(crate) globals: Handle<ShaderBuffer>,
    #[texture(102)]
    #[sampler(103)]
    pub(crate) cloud_shadow: Option<Handle<Image>>,
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
            globals: GLOBALS_BUFFER,
            cloud_shadow: None,
        }
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
fn update_globals(
    weathering: Option<Res<SurfaceWeathering>>,
    weather: Res<Weather>,
    shadows: Option<Res<CloudShadows>>,
    time: Res<Time<Virtual>>,
    mut globals: ResMut<ElementsGlobals>,
    buffers: Option<ResMut<Assets<ShaderBuffer>>>,
) {
    let next = ElementsGlobals::gather(
        weathering.as_deref(),
        &weather,
        shadows.as_deref(),
        time.elapsed_secs_f64(),
    );
    let Some(mut buffers) = buffers else {
        // Headless: no renderer, nothing to upload.
        globals.set_if_neq(next);
        return;
    };
    if (globals.set_if_neq(next) || !buffers.contains(GLOBALS_BUFFER.id()))
        && let Err(error) = buffers.insert(GLOBALS_BUFFER.id(), globals_buffer(&next))
    {
        error!("could not write the weathered materials' globals buffer: {error}");
    }
}

/// Attach the cloud shadow map to every weathered material. Writes an asset only when the map
/// changed, so a still scene re-prepares nothing.
fn update_materials(
    shadows: Option<Res<CloudShadows>>,
    mut materials: ResMut<Assets<WeatheredMaterial>>,
) {
    let map = shadows.as_ref().map(|s| s.map.clone());
    let stale: Vec<_> = materials
        .iter()
        .filter(|(_, m)| m.extension.cloud_shadow != map)
        .map(|(id, _)| id)
        .collect();
    for id in stale {
        if let Some(mut material) = materials.get_mut(id) {
            material.extension.cloud_shadow = map.clone();
        }
    }
}

#[cfg(test)]
mod tests;
