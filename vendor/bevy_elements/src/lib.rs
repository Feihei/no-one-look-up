#![doc = include_str!("../README.md")]
#![forbid(unsafe_code)]

mod celestial;
mod climate;
mod clouds;
mod cover;
mod dust;
mod flash;
#[cfg(feature = "flora")]
mod flora;
mod fog;
mod gpu;
mod hydrology;
mod lightning;
mod paths;
mod plugin;
mod precipitation;
mod query;
mod rocks;
mod terrain;
mod trails;
#[cfg(feature = "isomesh")]
mod voxel;
mod waves;
mod weathered;
mod wind;

// Plugin group, system sets, camera, its focus, quality, world units.
pub use plugin::{
    ElementsCamera, ElementsFocus, ElementsPlugins, ElementsQuality, ElementsSystems,
    ElementsUnits,
};

// Readback cadence.
pub use gpu::readback::ElementsReadback;

// Gameplay queries.
pub use query::{Environment, EnvironmentSnapshot};

// Celestial: clock, location, sun, moon, exposure; `astro` holds the ephemeris functions.
pub use celestial::{
    CelestialPlugin, CelestialSettings, CivilDate, ElementsExposure, GeoLocation, Moon, Sun,
    SunState, WorldClock, astro, light,
};

// Climate, seasons, weather.
pub use climate::{
    Climate, ClimateHistory, ClimatePlugin, ClimateSummary, DayConditions, FREEZING_C,
    LAPSE_RATE_C_PER_M, Season, SeasonChanged, SeasonState, SnowSeason, SurfaceWeathering,
    WET_DAY_MM, Weather, WeatherChanged, WeatherDirector, WeatherPreset,
};

// Wind.
pub use wind::{
    MAX_GUSTS, WIND_CELL_M, WIND_ROUGHNESS_M, WindFieldParams, WindGust, WindPlugin, WindState,
    log_wind_speed,
};

// Precipitation.
pub use precipitation::{
    MAX_PRECIPITATION_BLOCKERS, PRECIPITATION_BOX_M, PrecipitationBlocker, PrecipitationLook,
    PrecipitationPlugin, PrecipitationState,
};

// Fog.
pub use fog::{
    CLEAR_AIR_EXTINCTION, CLEAR_VISIBILITY_M, FogPlugin, GROUND_FOG_EXTENT, GROUND_FOG_TEXELS,
    GroundFog, KOSCHMIEDER, distance_fog_falloff, extinction_for_visibility, fog_hue,
    orthographic_fog_falloff, orthographic_near_depth, sky_air_extinction,
    volume_light_compensation,
};

// Clouds.
pub use clouds::{CloudLayer, CloudPlugin, CloudQuality, CloudShadows, CloudWeatherMap};

// Lightning and thunder; `LightningFlash` is shared with the cloud and precipitation passes.
pub use flash::LightningFlash;
pub use lightning::{
    FLASH_LUMENS, LightningPlugin, LightningStrike, SPEED_OF_SOUND_MS, STRIKE_RADIUS_M,
    StrikeLightning, Thunder,
};

// Terrain.
pub use terrain::{
    GroundExtension, GroundLayerTextures, GroundMaterial, GroundParams, ProceduralTerrain, Terrain,
    TerrainExtension, TerrainExtensionKey, TerrainGrid, TerrainLook, TerrainMaterial,
    TerrainMaterialParams, TerrainPlugin, TerrainRender, TerrainSnapshot, TerrainSource,
    TerrainState, TerrainTextures,
};

// Aeolian transport.
pub use dust::{DustPlugin, DustSettings, DustStats};

// Hydrology and erosion.
pub use hydrology::{
    ErodeTerrain, ErosionSettings, HydrologyBoundary, HydrologyPlugin, HydrologySettings,
    HydrologyStats, MAX_WATER_SOURCES, TerrainBaked, WaterExtension, WaterMaterial, WaterParams,
    WaterPatch, WaterPlugin, WaterSettings, WaterSink, WaterSource, WaterState, evaporation_mm_h,
};

// Ground cover: snow and wetness maps.
pub use cover::{CoverState, CoverStats, GroundCoverPlugin, SnowSettings, WetnessSettings};

// Trails: footprints, ruts and tracks in snow, sand and mud.
pub use trails::{Imprint, TrailGround, TrailSettings, TrailState, TrailsPlugin};

// Paths: routes laid across the terrain, worn into its ground.
pub use paths::{Path, PathHit, PathKind, PathNetwork, PathState, PathSurface, PathsPlugin};

// Waves: ripples and wakes from bodies in the water.
pub use waves::{WaterContact, WaveSettings, WaveState, WavesPlugin};

// Rocks scattered on the ground: boulders and stones as meshes.
pub use rocks::{RockSettings, RockStats, RocksPlugin, ScatteredRock};

// `bevy_flora` integration.
#[cfg(feature = "flora")]
pub use flora::{ElementsFloraPlugin, FLORA_FULL_WIND_MS};

// Weathered materials.
pub use weathered::{
    ELEMENTS_GLOBALS_BUFFER, ElementsGlobals, SNOW_ALBEDO, SNOW_ROUGHNESS, WeatheredExtension,
    WeatheredMaterial, WeatheredPlugin,
};

// Volumetric snow on `bevy_isomesh` voxel volumes.
#[cfg(feature = "isomesh")]
pub use voxel::{SnowyField, VoxelGround, VoxelGroundSystems, VoxelPlugin, VoxelSnow};
