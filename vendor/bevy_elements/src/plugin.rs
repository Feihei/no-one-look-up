//! The plugin group, the system sets every domain orders against, the camera marker and the
//! quality knob.

use bevy::app::PluginGroupBuilder;
use bevy::camera::Hdr;
use bevy::core_pipeline::prepass::DepthPrepass;
use bevy::ecs::system::SystemParam;
use bevy::light::atmosphere::ScatteringMedium;
use bevy::light::{Atmosphere, AtmosphereEnvironmentMapLight, VolumetricFog};
use bevy::pbr::{AtmosphereSettings, DistanceFog};
use bevy::prelude::*;
use bevy::render::view::Msaa;
use bevy::shader::load_shader_library;

use crate::celestial::{CelestialPlugin, SunState, clock_plugin};
use crate::climate::{ClimatePlugin, Weather};
use crate::clouds::CloudPlugin;
use crate::cover::GroundCoverPlugin;
use crate::dust::DustPlugin;
use crate::fog::FogPlugin;
use crate::gpu::readback::ReadbackPlugin;
use crate::hydrology::{HydrologyPlugin, WaterPlugin};
use crate::lightning::LightningPlugin;
use crate::paths::PathsPlugin;
use crate::precipitation::PrecipitationPlugin;
use crate::query::EnvironmentSnapshot;
use crate::rocks::RocksPlugin;
use crate::terrain::TerrainPlugin;
use crate::trails::TrailsPlugin;
use crate::waves::WavesPlugin;
use crate::weathered::WeatheredPlugin;
use crate::wind::{WindPlugin, WindState};

/// Every domain plugin that exists in this version, in dependency order.
///
/// Each member is also public on its own, so a game can drop one:
///
/// ```no_run
/// # use bevy::prelude::*;
/// # use bevy_elements::*;
/// App::new().add_plugins((DefaultPlugins, ElementsPlugins.build().disable::<FogPlugin>()));
/// ```
pub struct ElementsPlugins;

impl PluginGroup for ElementsPlugins {
    fn build(self) -> PluginGroupBuilder {
        let group = PluginGroupBuilder::start::<Self>()
            .add(ClimatePlugin)
            .add(CelestialPlugin)
            .add(WindPlugin)
            .add(TerrainPlugin)
            .add(HydrologyPlugin)
            .add(WaterPlugin)
            .add(GroundCoverPlugin)
            .add(DustPlugin)
            .add(TrailsPlugin)
            .add(PathsPlugin)
            .add(WavesPlugin)
            .add(FogPlugin)
            .add(CloudPlugin)
            .add(LightningPlugin)
            .add(PrecipitationPlugin)
            .add(WeatheredPlugin)
            // Rocks on the terrain, drawn with the weathered material above.
            .add(RocksPlugin);
        // Snow on `bevy_isomesh` voxel volumes; it needs the terrain, hydrology and cover
        // plugins above, so it comes last.
        #[cfg(feature = "isomesh")]
        let group = group.add(crate::voxel::VoxelPlugin);
        group
    }
}

/// The phases of one environment frame, chained in this order in `Update`.
///
/// Systems that write `WorldClock`, `WeatherDirector` or settings run
/// `.before(ElementsSystems::Clock)`; systems that read `Environment` run
/// `.after(ElementsSystems::Readback)`.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ElementsSystems {
    /// The world clock advances and the followed camera is chosen.
    Clock,
    /// The climate schedules weather and the weather blends toward its target.
    Weather,
    /// Sun, moon and stars are placed for the current clock.
    Celestial,
    /// GPU simulation inputs (uniforms, gusts, sources) are written for this frame.
    Simulation,
    /// Readback requests are issued and completed readbacks have filled `EnvironmentSnapshot`.
    Readback,
}

/// Marks the camera the environment follows and renders for.
///
/// Adding it dresses the camera for the crate's physical look, each part a flag that defaults
/// on (`docs/research/integration.md`): `atmosphere` (`AtmosphereSettings`,
/// `AtmosphereEnvironmentMapLight`, one Earth `Atmosphere` in the world, and the night sky's glow
/// as the clear colour of a camera that kept Bevy's default), `distance_fog` (a `DistanceFog` the
/// crate drives from the visibility), `volumetric_fog` (`VolumetricFog`: the ground fog and the
/// light shafts), `ambient` (an `AmbientLight` the crate drives with the cloud-scattered sky light)
/// and `msaa_off` (`Msaa::Off`). A component the camera already has is kept; turning a flag off at
/// runtime removes only what the crate inserted, and stops the crate driving the game's own.
/// `DepthPrepass` is inserted while a plugin that reads the depth prepass (clouds, precipitation,
/// water) is added. Bevy gives every `Camera3d` an `Exposure`; `auto_exposure` decides whether the
/// crate drives it.
///
/// Without `atmosphere` the sky is the game's and the cloud layer is not drawn on that camera
/// (it needs the atmosphere's lookup tables), but the cloud shadows still fall; without
/// `msaa_off` the cloud and precipitation passes skip the camera, which they read single-sampled.
///
/// With more than one active `ElementsCamera` the one with the lowest `Camera::order` is followed
/// and a warning is logged once; inactive cameras (`Camera::is_active == false`) are ignored. With
/// none, the camera-relative simulations (wind, fog volume, clouds) idle while the clock, weather
/// and gameplay queries keep running.
#[derive(Component, Clone, Copy, Debug, PartialEq, Eq, Reflect)]
#[require(Hdr)]
pub struct ElementsCamera {
    /// Drive the camera's `Exposure` from the sun, moon and sky illuminance. Default `true`.
    pub auto_exposure: bool,
    /// Bevy's physical atmosphere: the sky, its light and the aerial perspective. Default `true`.
    pub atmosphere: bool,
    /// A `DistanceFog` driven by the weather's visibility. Default `true`.
    pub distance_fog: bool,
    /// `VolumetricFog`, which draws the ground fog volume and the light shafts. Default `true`.
    pub volumetric_fog: bool,
    /// An `AmbientLight` driven with the cloud-scattered sky light. Default `true`.
    pub ambient: bool,
    /// `Msaa::Off`. Default `true`.
    pub msaa_off: bool,
}

impl Default for ElementsCamera {
    fn default() -> Self {
        Self {
            auto_exposure: true,
            atmosphere: true,
            distance_fog: true,
            volumetric_fog: true,
            ambient: true,
            msaa_off: true,
        }
    }
}

impl ElementsCamera {
    /// The weather on the game's own look: every part of the dressing off, so the crate inserts
    /// and drives nothing on the camera but the `DepthPrepass` its passes read (and keeps the
    /// game's MSAA, which the cloud and precipitation passes then skip).
    pub fn bare() -> Self {
        Self {
            auto_exposure: false,
            atmosphere: false,
            distance_fog: false,
            volumetric_fog: false,
            ambient: false,
            msaa_off: false,
        }
    }
}

/// What the camera-relative domains centre on, on the followed [`ElementsCamera`]: the rain and
/// snow box, the wind window and its gusts, the ground fog volume, the strike disc and the
/// thunder distance, the cloud weather and shadow maps, the trails and waves windows, and the
/// mean the surface weathering takes around the view. Without it they centre on the camera.
///
/// An isometric or top-down game whose orthographic camera stands far back from what it shows
/// puts the focus on the player, or on the point the camera looks at
/// (`docs/research/view-anchor.md`). An `Entity` focus without a `GlobalTransform` falls back
/// to the camera.
#[derive(Component, Clone, Copy, Debug, PartialEq, Reflect)]
pub enum ElementsFocus {
    /// That entity's `GlobalTransform` translation, read every frame.
    Entity(Entity),
    /// A world point, in world units.
    Point(Vec3),
}

/// Resolution and step counts for every domain. Read once per frame; changing it at runtime
/// re-creates the affected GPU resources.
#[derive(Resource, Default, Clone, Copy, Debug, PartialEq, Eq, Hash, Reflect)]
pub enum ElementsQuality {
    /// Integrated GPUs and the web.
    Low,
    /// Mid-range discrete GPUs.
    Medium,
    /// The default.
    #[default]
    High,
    /// Large budgets; the original research resolutions.
    Ultra,
}

impl ElementsQuality {
    /// Wind grid cells (x, y, z); each cell is 4 m on a side.
    pub fn wind_grid(self) -> UVec3 {
        match self {
            Self::Low => UVec3::new(32, 8, 32),
            Self::Medium => UVec3::new(48, 12, 48),
            Self::High => UVec3::new(64, 16, 64),
            Self::Ultra => UVec3::new(96, 24, 96),
        }
    }

    /// Jacobi pressure iterations per wind step (even: the ping-pong ends in the same buffer).
    /// The pressure is warm-started from the last frame, so few are needed: on the High grid 10
    /// iterations leave the same divergence as 30 (8.4e-4 vs 8.2e-4 s⁻¹, `tests/gpu_wind.rs`).
    pub fn wind_pressure_iterations(self) -> u32 {
        match self {
            Self::Low | Self::Medium => 10,
            Self::High => 12,
            Self::Ultra => 40,
        }
    }

    /// The most cells per axis the terrain's simulation grid has (`Terrain::resolution` is
    /// clamped to this and rounded down to a power of two). The hydrology and erosion kernels
    /// run on this grid, so this is the one number to lower if they do not fit their budget.
    ///
    /// `High` takes `Medium`'s value: hydrology and erosion at 1024² and 8 substeps cost
    /// 1.86 ms per frame against the plan's 0.6 ms (`docs/research/hydrology.md`, Deviation 20),
    /// and 512² with 4 substeps costs 0.26 ms. `Ultra` keeps the research resolution.
    pub fn terrain_max_resolution(self) -> u32 {
        match self {
            Self::Low => 256,
            Self::Medium | Self::High => 512,
            Self::Ultra => 2048,
        }
    }

    /// The most hydrology and erosion substeps per frame (each is `HydrologySettings::step_s`
    /// of simulated time); an erosion bake runs this many every frame. `High` takes `Medium`'s
    /// value, as [`terrain_max_resolution`](Self::terrain_max_resolution).
    pub fn hydrology_substeps(self) -> u32 {
        match self {
            Self::Low => 2,
            Self::Medium | Self::High => 4,
            Self::Ultra => 16,
        }
    }

    /// Ray-march steps for Bevy's volumetric fog.
    pub fn fog_steps(self) -> u32 {
        match self {
            Self::Low => 32,
            Self::Medium => 48,
            Self::High => 64,
            Self::Ultra => 96,
        }
    }

    /// Particles in the precipitation buffer; rain and snow take a share of them by rate.
    pub fn precipitation_particles(self) -> u32 {
        match self {
            Self::Low => 20_000,
            Self::Medium => 60_000,
            Self::High => 150_000,
            Self::Ultra => 300_000,
        }
    }

    /// The trail field's window: texels a side and the texel edge (m). Rise of the Tomb Raider's
    /// 1024² × 4 cm is the reference (`docs/research/trails.md`).
    pub fn trail_window(self) -> (u32, f32) {
        match self {
            Self::Low => (512, 0.08),
            Self::Medium => (1024, 0.06),
            Self::High => (1024, 0.05),
            Self::Ultra => (2048, 0.03),
        }
    }

    /// The wave field's window: texels a side and the finest cascade's texel edge (m); the
    /// coarser cascades are 4× and 16× (`docs/research/waves.md`).
    pub fn wave_window(self) -> (u32, f32) {
        match self {
            Self::Low => (256, 0.08),
            Self::Medium => (512, 0.06),
            Self::High => (512, 0.05),
            Self::Ultra => (1024, 0.04),
        }
    }

    /// How far scattered rocks are drawn, m per m of a rock's diameter: a rock vanishes at the
    /// same size on screen whatever its size (`docs/research/rock-scatter.md` Deviation 3). Under
    /// an orthographic camera the same factor is read as that screen size, which is a threshold
    /// on a rock's diameter instead of a distance from the eye
    /// (`docs/research/orthographic-rocks.md` Model 1).
    pub fn rock_range(self) -> f32 {
        match self {
            Self::Low => 60.0,
            Self::Medium => 80.0,
            Self::High => 100.0,
            Self::Ultra => 140.0,
        }
    }
}

/// How many world units make a metre: 1.0 (the default) for a world in metres, 3.2808 for one in
/// feet, 100.0 in centimetres (`docs/research/units.md`).
///
/// The crate simulates in metres. Positions it reads or writes in the world (every `Transform`,
/// [`ElementsFocus::Point`], the [`Environment`](crate::Environment) query positions,
/// [`Environment::terrain_height`](crate::Environment::terrain_height), the strike, thunder,
/// flash and gust positions) are world units; everything else stays SI: lengths and depths in
/// metres ([`Thunder::distance_m`](crate::Thunder), [`Terrain`](crate::Terrain) extents and
/// heights, the half extents of blockers, contacts and imprints, which their transforms scale),
/// speeds in m/s. The grids the crate publishes (the wind window, the terrain grid, the cloud
/// shadow map's placement) are in the metre frame, world ÷ `units_per_metre`. The crate scales
/// what it puts in Bevy's world to match: the atmosphere it spawns, its fog, shadow cascades and
/// lightning light.
///
/// Changing it at runtime moves the camera-relative domains on the next frame; a terrain keeps
/// the metre frame it was built in until it is rebuilt. A value that is not positive and finite
/// reads as 1.0 (one warning).
#[derive(Resource, Clone, Copy, Debug, PartialEq, Reflect)]
pub struct ElementsUnits {
    /// World units to the metre.
    pub units_per_metre: f32,
}

impl Default for ElementsUnits {
    fn default() -> Self {
        Self {
            units_per_metre: 1.0,
        }
    }
}

impl ElementsUnits {
    /// World units to the metre, 1.0 for a value that is not positive and finite.
    pub fn per_metre(&self) -> f32 {
        let u = self.units_per_metre;
        if u.is_finite() && u > 0.0 { u } else { 1.0 }
    }

    /// `metres` in world units.
    pub fn world(&self, metres: f32) -> f32 {
        metres * self.per_metre()
    }

    /// `world` units in metres.
    pub fn metres(&self, world: f32) -> f32 {
        world / self.per_metre()
    }

    /// A world point in the metre frame.
    pub fn point_to_metres(&self, world: Vec3) -> Vec3 {
        world / self.per_metre()
    }

    /// A metre-frame point in the world.
    pub fn point_to_world(&self, metres: Vec3) -> Vec3 {
        metres * self.per_metre()
    }

    /// A world transform in the metre frame: the same rotation, its translation and scale
    /// divided by the units (a box of local half extents `h` under it measures `h · scale` m).
    pub fn to_metres(&self, transform: &GlobalTransform) -> GlobalTransform {
        GlobalTransform::from_scale(Vec3::splat(1.0 / self.per_metre())) * *transform
    }
}

/// Says once that the units are not usable, and keeps the crate's atmosphere in metres.
fn apply_units(
    units: Res<ElementsUnits>,
    mut atmospheres: Query<(&mut Transform, &Atmosphere), With<ElementsAtmosphere>>,
    mut warned: Local<bool>,
) {
    if !units.is_changed() {
        return;
    }
    let u = units.units_per_metre;
    if !(u.is_finite() && u > 0.0) && !*warned {
        warn!("ElementsUnits::units_per_metre is {u}: reading it as 1.0");
        *warned = true;
    }
    for (mut transform, atmosphere) in &mut atmospheres {
        *transform = planet_transform(atmosphere, units.per_metre());
    }
}

/// The transform of the crate's `Atmosphere`: Bevy reads its `GlobalTransform` as the planet's
/// centre and scale, so the centre lies `inner_radius` metres below the world origin (Bevy's own
/// default, which a `Transform` would otherwise overwrite with the origin) and the planet is
/// scaled to the world units.
fn planet_transform(atmosphere: &Atmosphere, units: f32) -> Transform {
    Transform::from_translation(Vec3::NEG_Y * (atmosphere.inner_radius * units))
        .with_scale(Vec3::splat(units))
}

/// The `Atmosphere` the crate spawned: its planet centred below the origin and scaled by
/// [`ElementsUnits`], so Bevy's sky is in metres.
#[derive(Component, Clone, Copy, Debug)]
pub(crate) struct ElementsAtmosphere;

/// The camera the camera-relative simulations follow this frame, in the metre frame (world ÷
/// [`ElementsUnits::units_per_metre`]).
#[derive(Resource, Default, Clone, Copy, Debug)]
pub(crate) struct ElementsView {
    /// The followed `ElementsCamera`, if any.
    pub entity: Option<Entity>,
    /// What the camera-relative domains centre on, m: its [`ElementsFocus`], else the camera.
    pub position: Vec3,
    /// The camera's position, m.
    pub eye: Vec3,
    /// The camera's velocity, m/s: the finite difference of `eye` over the last frame, zero on
    /// the frame the camera is first followed.
    pub velocity: Vec3,
}

/// Shared CPU setup every domain needs: the system sets, the clock, the followed view, the
/// quality knob and the resources [`Environment`](crate::Environment) reads. Runs without a
/// renderer (`MinimalPlugins`). Added once by whichever domain comes first.
pub(crate) struct ElementsCorePlugin;

impl Plugin for ElementsCorePlugin {
    fn build(&self, app: &mut App) {
        clock_plugin(app);
        app.configure_sets(
            Update,
            (
                ElementsSystems::Clock,
                ElementsSystems::Weather,
                ElementsSystems::Celestial,
                ElementsSystems::Simulation,
                ElementsSystems::Readback,
            )
                .chain(),
        )
        .init_resource::<ElementsQuality>()
        .init_resource::<ElementsUnits>()
        .init_resource::<ElementsView>()
        .init_resource::<EnvironmentSnapshot>()
        .init_resource::<Weather>()
        .init_resource::<SunState>()
        .init_resource::<WindState>()
        .add_systems(
            Update,
            (apply_units, follow_camera)
                .chain()
                .in_set(ElementsSystems::Clock),
        );
    }
}

/// Shared rendering setup: the public WESL libraries, the readback scheduler and the camera
/// dressing. Added once by whichever GPU domain comes first.
pub(crate) struct ElementsGpuCorePlugin;

impl Plugin for ElementsGpuCorePlugin {
    fn build(&self, app: &mut App) {
        ensure_core(app);
        load_shader_library!(app, "noise.wesl");
        load_shader_library!(app, "grid.wesl");
        load_shader_library!(app, "wind.wesl");
        load_shader_library!(app, "surface_cover.wesl");
        // The one-cell `terrain_cells` that wind, precipitation and ground materials bind until
        // a terrain exists.
        if let Some(mut buffers) = app
            .world_mut()
            .get_resource_mut::<Assets<bevy::render::storage::ShaderBuffer>>()
        {
            crate::terrain::insert_placeholder(&mut buffers);
        }
        load_shader_library!(app, "climate/snow.wesl");
        // The terrain grid's placement, imported by wind, precipitation, cover and water
        // shaders whether or not `TerrainPlugin` is added.
        load_shader_library!(app, "terrain/common.wesl");
        // The trail field's sampling and parallax, imported by the terrain and ground materials
        // whether or not `TrailsPlugin` is added.
        load_shader_library!(app, "trails/surface.wesl");
        // The path mask's distance field, imported by the terrain and ground materials.
        load_shader_library!(app, "paths.wesl");
        app.init_resource::<CameraNeeds>()
            .add_plugins(ReadbackPlugin)
            .add_observer(dress_camera)
            .add_systems(Update, redress_cameras.in_set(ElementsSystems::Clock));
    }
}

/// Add [`ElementsCorePlugin`] unless another domain already did.
pub(crate) fn ensure_core(app: &mut App) {
    if !app.is_plugin_added::<ElementsCorePlugin>() {
        app.add_plugins(ElementsCorePlugin);
    }
}

/// Add [`ElementsGpuCorePlugin`] (and the CPU core) unless another domain already did.
pub(crate) fn ensure_gpu_core(app: &mut App) {
    if !app.is_plugin_added::<ElementsGpuCorePlugin>() {
        app.add_plugins(ElementsGpuCorePlugin);
    }
}

/// What the domains added to the app need on every [`ElementsCamera`], and whether the crate has
/// spawned the world's `Atmosphere`.
#[derive(Resource, Default, Clone, Copy, Debug)]
pub(crate) struct CameraNeeds {
    /// A plugin that reads the depth prepass is added (clouds, precipitation, water).
    pub depth_prepass: bool,
    atmosphere_spawned: bool,
}

/// Record that a plugin reads the depth prepass, so every [`ElementsCamera`] gets a
/// `DepthPrepass`.
pub(crate) fn needs_depth_prepass(app: &mut App) {
    app.world_mut()
        .get_resource_or_init::<CameraNeeds>()
        .depth_prepass = true;
}

/// What the crate put on an [`ElementsCamera`], so turning a flag off removes only that.
#[derive(Component, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct ElementsDressing {
    atmosphere: bool,
    sky_background: bool,
    distance_fog: bool,
    volumetric_fog: bool,
    ambient: bool,
    msaa: bool,
    depth_prepass: bool,
}

type DressParts = (
    &'static ElementsCamera,
    Option<&'static mut ElementsDressing>,
    &'static mut Camera,
    Has<AtmosphereSettings>,
    Has<DistanceFog>,
    Has<VolumetricFog>,
    Has<AmbientLight>,
    Has<DepthPrepass>,
);

/// Dresses an [`ElementsCamera`] for its flags.
#[derive(SystemParam)]
struct Dresser<'w, 's> {
    commands: Commands<'w, 's>,
    quality: Res<'w, ElementsQuality>,
    units: Res<'w, ElementsUnits>,
    needs: ResMut<'w, CameraNeeds>,
    atmospheres: Query<'w, 's, (), With<Atmosphere>>,
    media: ResMut<'w, Assets<ScatteringMedium>>,
    cameras: Query<'w, 's, DressParts>,
}

/// Insert `make()` when wanted and absent (and record it as the crate's); remove it when not
/// wanted and the crate inserted it. A component the game put there is never removed.
fn toggle<B: Bundle>(
    entity: &mut EntityCommands,
    want: bool,
    present: bool,
    ours: &mut bool,
    make: impl FnOnce() -> B,
) {
    if want && !present {
        entity.insert(make());
        *ours = true;
    } else if !want && *ours {
        entity.remove::<B>();
        *ours = false;
    }
}

impl Dresser<'_, '_> {
    fn dress(&mut self, entity: Entity) {
        let fog_steps = self.quality.fog_steps();
        let depth_prepass = self.needs.depth_prepass;
        let Ok((want, done, mut camera, atmosphere, fog, volumetric, ambient, prepass)) =
            self.cameras.get_mut(entity)
        else {
            return;
        };
        let want = *want;
        let mut ours = done.as_deref().copied().unwrap_or_default();
        let mut e = self.commands.entity(entity);
        toggle(
            &mut e,
            want.atmosphere,
            atmosphere,
            &mut ours.atmosphere,
            || {
                (
                    AtmosphereSettings::default(),
                    AtmosphereEnvironmentMapLight::default(),
                )
            },
        );
        toggle(
            &mut e,
            depth_prepass,
            prepass,
            &mut ours.depth_prepass,
            || DepthPrepass,
        );
        toggle(
            &mut e,
            want.volumetric_fog,
            volumetric,
            &mut ours.volumetric_fog,
            || VolumetricFog {
                ambient_intensity: 0.0,
                step_count: fog_steps,
                ..default()
            },
        );
        toggle(
            &mut e,
            want.distance_fog,
            fog,
            &mut ours.distance_fog,
            DistanceFog::default,
        );
        // Sky light comes from the atmosphere's environment map; Bevy's default ambient is a
        // constant 80 cd/m² that the night exposure would lift to white. Starts dark; the
        // celestial systems drive it with the cloud-scattered diffuse light.
        toggle(&mut e, want.ambient, ambient, &mut ours.ambient, || {
            AmbientLight {
                brightness: 0.0,
                ..default()
            }
        });
        // The cloud and precipitation passes read the depth prepass as single-sampled, and
        // `Camera` requires a default `Msaa::Sample4`, so there is no "absent" to fill.
        if want.msaa_off {
            e.insert(Msaa::Off);
            ours.msaa = true;
        } else if ours.msaa {
            e.insert(Msaa::default());
            ours.msaa = false;
        }
        // Behind the atmosphere is the night sky's own faint glow, not Bevy's default
        // `ClearColor` (sRGB 43, 44, 47), which is written before the sky, never exposed, and
        // showed through the atmosphere as a constant brown night sky. A camera that kept Bevy's
        // default gets `SkyBackground`, and the celestial systems set its clear colour each frame.
        if want.atmosphere && matches!(camera.clear_color, ClearColorConfig::Default) {
            camera.clear_color = ClearColorConfig::Custom(Color::BLACK);
            e.insert(SkyBackground);
            ours.sky_background = true;
        } else if !want.atmosphere && ours.sky_background {
            camera.clear_color = ClearColorConfig::Default;
            e.remove::<SkyBackground>();
            ours.sky_background = false;
        }
        match done {
            Some(mut done) => {
                if *done != ours {
                    *done = ours;
                }
            }
            None => {
                e.insert(ours);
            }
        }
        if want.atmosphere && self.atmospheres.is_empty() && !self.needs.atmosphere_spawned {
            let medium = self.media.add(ScatteringMedium::default());
            let atmosphere = Atmosphere::earth(medium);
            let transform = planet_transform(&atmosphere, self.units.per_metre());
            self.commands.spawn((
                Name::new("Elements atmosphere"),
                atmosphere,
                ElementsAtmosphere,
                transform,
            ));
            self.needs.atmosphere_spawned = true;
        }
    }
}

/// On an [`ElementsCamera`] whose clear colour the crate owns (it kept Bevy's default): the
/// celestial systems clear it to the night sky's background radiance at the camera's exposure.
#[derive(Component, Clone, Copy, Debug, Default)]
pub(crate) struct SkyBackground;

fn dress_camera(add: On<Add<ElementsCamera>>, mut dresser: Dresser) {
    dresser.dress(add.entity);
}

/// Dress again a camera whose flags changed: a game can toggle the physical sky at runtime.
fn redress_cameras(changed: Query<Entity, Changed<ElementsCamera>>, mut dresser: Dresser) {
    for entity in &changed {
        dresser.dress(entity);
    }
}

fn follow_camera(
    time: Res<Time<Virtual>>,
    cameras: Query<
        (Entity, &Camera, &GlobalTransform, Option<&ElementsFocus>),
        With<ElementsCamera>,
    >,
    transforms: Query<&GlobalTransform>,
    units: Res<ElementsUnits>,
    mut view: ResMut<ElementsView>,
    mut warned: Local<bool>,
) {
    let mut count = 0usize;
    let mut best: Option<(Entity, isize, Vec3, Option<ElementsFocus>)> = None;
    for (entity, camera, transform, focus) in &cameras {
        if !camera.is_active {
            continue;
        }
        count += 1;
        if best.is_none_or(|(_, order, _, _)| camera.order < order) {
            best = Some((
                entity,
                camera.order,
                units.point_to_metres(transform.translation()),
                focus.copied(),
            ));
        }
    }
    if count > 1 && !*warned {
        warn!(
            "{count} active entities carry ElementsCamera; the environment follows the one with the lowest Camera::order"
        );
        *warned = true;
    }
    match best {
        Some((entity, _, eye, focus)) => {
            let dt = time.delta_secs();
            view.velocity = if view.entity == Some(entity) && dt > 0.0 {
                (eye - view.eye) / dt
            } else {
                Vec3::ZERO
            };
            view.entity = Some(entity);
            view.eye = eye;
            view.position = match focus {
                Some(ElementsFocus::Point(p)) => units.point_to_metres(p),
                Some(ElementsFocus::Entity(e)) => transforms
                    .get(e)
                    .map_or(eye, |t| units.point_to_metres(t.translation())),
                None => eye,
            };
        }
        None => {
            view.entity = None;
            view.velocity = Vec3::ZERO;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_inactive_camera_is_not_followed() {
        let mut app = App::new();
        app.add_plugins((MinimalPlugins, ElementsCorePlugin));
        // The inactive camera has the lower order, so it would win if it were counted.
        app.world_mut().spawn((
            Camera {
                order: -1,
                is_active: false,
                ..default()
            },
            ElementsCamera::default(),
            GlobalTransform::from_xyz(100.0, 0.0, 0.0),
        ));
        app.world_mut().spawn((
            Camera::default(),
            ElementsCamera::default(),
            GlobalTransform::from_xyz(5.0, 2.0, 0.0),
        ));
        app.update();
        assert_eq!(
            app.world().resource::<ElementsView>().position,
            Vec3::new(5.0, 2.0, 0.0)
        );
    }

    /// The followed view centres on the camera's `ElementsFocus`: a point, an entity's
    /// transform, or the camera itself when that entity has no transform; the eye stays the
    /// camera's.
    #[test]
    fn the_view_centres_on_the_focus() {
        let mut app = App::new();
        app.add_plugins((MinimalPlugins, ElementsCorePlugin));
        let eye = Vec3::new(0.0, 860.0, 1229.0);
        let player = app
            .world_mut()
            .spawn(GlobalTransform::from_xyz(3.0, 0.5, -4.0))
            .id();
        let bare = app.world_mut().spawn_empty().id();
        let camera = app
            .world_mut()
            .spawn((
                Camera::default(),
                ElementsCamera::default(),
                GlobalTransform::from_translation(eye),
                ElementsFocus::Point(Vec3::new(1.0, 0.0, 2.0)),
            ))
            .id();
        for (focus, expected) in [
            (
                ElementsFocus::Point(Vec3::new(1.0, 0.0, 2.0)),
                Vec3::new(1.0, 0.0, 2.0),
            ),
            (ElementsFocus::Entity(player), Vec3::new(3.0, 0.5, -4.0)),
            (ElementsFocus::Entity(bare), eye),
        ] {
            app.world_mut().entity_mut(camera).insert(focus);
            app.update();
            let view = app.world().resource::<ElementsView>();
            assert_eq!((view.position, view.eye), (expected, eye), "{focus:?}");
        }
        app.world_mut().entity_mut(camera).remove::<ElementsFocus>();
        app.update();
        assert_eq!(app.world().resource::<ElementsView>().position, eye);
    }

    /// Bevy reads the atmosphere's `GlobalTransform` as the planet's centre: the crate's
    /// atmosphere keeps it `inner_radius` metres below the origin, in world units, after the
    /// transforms propagate and when the units change. With the centre at the origin the camera
    /// stands inside the planet and the sky is garbage (iteration 24's regression).
    #[test]
    fn the_crate_atmosphere_centres_the_planet_below_the_origin() {
        let mut app = App::new();
        app.add_plugins((
            MinimalPlugins,
            bevy::transform::TransformPlugin,
            ElementsCorePlugin,
        ))
        .init_resource::<CameraNeeds>()
        .insert_resource(Assets::<ScatteringMedium>::default())
        .add_observer(dress_camera);
        app.world_mut()
            .spawn((Camera3d::default(), ElementsCamera::default()));
        let centre = |app: &mut App| {
            let mut atmospheres = app
                .world_mut()
                .query_filtered::<(&GlobalTransform, &Atmosphere), With<ElementsAtmosphere>>();
            let (global, atmosphere) = atmospheres.single(app.world()).map_or_else(
                |e| panic!("one crate atmosphere: {e}"),
                |(g, a)| (*g, a.inner_radius),
            );
            let (scale, _, translation) = global.to_scale_rotation_translation();
            (translation, scale, atmosphere)
        };
        for u in [1.0_f32, 3.2808] {
            app.insert_resource(ElementsUnits { units_per_metre: u });
            for _ in 0..3 {
                app.update();
            }
            let (translation, scale, radius) = centre(&mut app);
            assert_eq!(translation, Vec3::NEG_Y * (radius * u), "units {u}");
            assert_eq!(scale, Vec3::splat(u), "units {u}");
        }
    }
}
