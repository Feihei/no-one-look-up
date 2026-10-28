//! The plugin group, the system sets every domain orders against, the camera marker and the
//! quality knob.

use bevy::app::PluginGroupBuilder;
use bevy::camera::Exposure;
use bevy::camera::Hdr;
use bevy::core_pipeline::prepass::DepthPrepass;
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
use crate::precipitation::PrecipitationPlugin;
use crate::query::EnvironmentSnapshot;
use crate::terrain::TerrainPlugin;
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
            .add(FogPlugin)
            .add(CloudPlugin)
            .add(LightningPlugin)
            .add(PrecipitationPlugin)
            .add(WeatheredPlugin);
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
/// Adding it inserts, where absent: `AtmosphereSettings`, `AtmosphereEnvironmentMapLight`,
/// `DepthPrepass`, `VolumetricFog`, `DistanceFog` and `Exposure`, and spawns one Earth
/// `Atmosphere` if the world has none. It always sets `Msaa::Off` (a `Camera3d` requires `Msaa`,
/// so there is no "absent" to fill): the cloud pass reads the non-multisampled depth prepass.
/// With more than one `ElementsCamera` the one with the lowest `Camera::order` is followed and a
/// warning is logged once. With none, the camera-relative simulations (wind, fog volume, clouds)
/// idle while the clock, weather and gameplay queries keep running.
#[derive(Component, Clone, Copy, Debug, Reflect)]
#[require(Hdr)]
pub struct ElementsCamera {
    /// Drive `Exposure::ev100` from the sun, moon and sky illuminance. Default `true`.
    pub auto_exposure: bool,
}

impl Default for ElementsCamera {
    fn default() -> Self {
        Self {
            auto_exposure: true,
        }
    }
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
}

/// The camera the camera-relative simulations follow this frame.
#[derive(Resource, Default, Clone, Copy, Debug)]
pub(crate) struct ElementsView {
    /// The followed `ElementsCamera`, if any.
    pub entity: Option<Entity>,
    /// Its world position, m.
    pub position: Vec3,
    /// Its velocity, m/s: the finite difference of `position` over the last frame, zero on the
    /// frame the camera is first followed.
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
        .init_resource::<ElementsView>()
        .init_resource::<EnvironmentSnapshot>()
        .init_resource::<Weather>()
        .init_resource::<SunState>()
        .init_resource::<WindState>()
        .add_systems(Update, follow_camera.in_set(ElementsSystems::Clock));
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
        app.add_plugins(ReadbackPlugin).add_observer(dress_camera);
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

fn dress_camera(
    add: On<Add<ElementsCamera>>,
    mut commands: Commands,
    quality: Res<ElementsQuality>,
    atmospheres: Query<(), With<Atmosphere>>,
    mut media: ResMut<Assets<ScatteringMedium>>,
) {
    commands.entity(add.entity).insert_if_new((
        AtmosphereSettings::default(),
        AtmosphereEnvironmentMapLight::default(),
        DepthPrepass,
        // (`Msaa::Off` is inserted below: `Camera` requires `Msaa`, so it is never absent here.)
        VolumetricFog {
            ambient_intensity: 0.0,
            step_count: quality.fog_steps(),
            ..default()
        },
        DistanceFog::default(),
        Exposure::SUNLIGHT,
        // Sky light comes from the atmosphere's environment map; Bevy's default ambient is a
        // constant 80 cd/m² that the night exposure would lift to white. Starts dark; the
        // celestial systems drive it with the cloud-scattered diffuse light.
        AmbientLight {
            brightness: 0.0,
            ..default()
        },
    ));
    // The cloud and precipitation passes read the depth prepass as single-sampled, and `Camera`
    // requires a default `Msaa::Sample4`, so `insert_if_new` would never replace it. A game that
    // wants MSAA inserts `Msaa` after spawning the camera.
    commands.entity(add.entity).insert(Msaa::Off);
    if atmospheres.is_empty() {
        let medium = media.add(ScatteringMedium::default());
        commands.spawn((Name::new("Elements atmosphere"), Atmosphere::earth(medium)));
    }
}

fn follow_camera(
    time: Res<Time<Virtual>>,
    cameras: Query<(Entity, &Camera, &GlobalTransform), With<ElementsCamera>>,
    mut view: ResMut<ElementsView>,
    mut warned: Local<bool>,
) {
    let mut count = 0usize;
    let mut best: Option<(Entity, isize, Vec3)> = None;
    for (entity, camera, transform) in &cameras {
        count += 1;
        if best.is_none_or(|(_, order, _)| camera.order < order) {
            best = Some((entity, camera.order, transform.translation()));
        }
    }
    if count > 1 && !*warned {
        warn!(
            "{count} entities carry ElementsCamera; the environment follows the one with the lowest Camera::order"
        );
        *warned = true;
    }
    match best {
        Some((entity, _, position)) => {
            let dt = time.delta_secs();
            view.velocity = if view.entity == Some(entity) && dt > 0.0 {
                (position - view.position) / dt
            } else {
                Vec3::ZERO
            };
            view.entity = Some(entity);
            view.position = position;
        }
        None => {
            view.entity = None;
            view.velocity = Vec3::ZERO;
        }
    }
}
