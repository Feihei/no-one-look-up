//! Sky system plugin (R1).
//!
//! Wraps bevy_elements behind a locked golden-hour sky: 17:30 local time,
//! paused clock, forced clear weather, Beijing-like observer location
//! (ADR-0001/0003). bevy_elements subsystems the game does not use (wind,
//! precipitation, lightning, dust, terrain, hydrology, water, ground cover,
//! weathering) are disabled at plugin-group level (R1.4).
//!
//! Ordering contract (root AGENTS.md: no implicit schedule dependence): every
//! system this plugin adds is explicitly ordered against
//! [`bevy_elements::ElementsSystems`].

use std::time::Duration;

use bevy::camera::Exposure;
use bevy::core_pipeline::tonemapping::Tonemapping;
use bevy::post_process::bloom::Bloom;
use bevy::prelude::*;

use bevy_elements::{
    CivilDate, Climate, CloudLayer, DustPlugin, ElementsCamera, ElementsPlugins, ElementsSystems,
    GeoLocation, GroundCoverPlugin, HydrologyPlugin, LightningPlugin, PrecipitationPlugin,
    TerrainPlugin, WaterPlugin, WeatherDirector, WeatherPreset, WeatheredPlugin, WindPlugin,
    WorldClock,
};

/// Golden-hour sky configuration (ADR-0001: fixed dusk, no weather changes).
///
/// Insert this resource before adding [`SkyPlugin`] to override the defaults.
#[derive(Resource, Clone, Debug)]
pub struct HeadsUpSky {
    /// Calendar date driving both the sun altitude at the golden hour and the
    /// moon phase. The two are coupled through the date (a fixed location +
    /// fixed hour leaves the date as the single tuning knob).
    pub date: CivilDate,
    /// Locked local hour. The game freezes time here (golden hour, 17:30).
    pub golden_hour: f64,
    /// Observer location; also selects the time zone for the local hour.
    ///
    /// Longitude is tuned so the fixed local hour lands the sun mid-way
    /// through the dusk tint (≈5° altitude): the game's world is fictional,
    /// so the coordinates only need to produce the adr-0001 gradient.
    pub location: GeoLocation,
    /// Overrides `Weather.cloud_cover` every frame (0..1). `None` keeps the
    /// forced `WeatherPreset::Clear` value (0.05, sparse clouds).
    pub cloud_cover: Option<f32>,
    /// Overrides the derived `CloudLayer` altitudes (m). Applied with one
    /// frame of lag (see `apply_cloud_layer_override`).
    pub cloud_layer: Option<(f32, f32)>,
    /// Base photographic exposure (ev100). ADR-0005 maps the golden-hour look
    /// to a fixed `ev100 ≈ 13.0` with ACES tonemapping.
    pub exposure_ev100: f32,
    /// Peak amplitude of the world-position exposure perturbation (R1.5), in
    /// ev100 stops. Deterministic function of the camera position.
    pub exposure_noise_amplitude: f32,
}

impl Default for HeadsUpSky {
    fn default() -> Self {
        Self {
            // Equinox: sunset over Beijing is ~18:15, so 17:30 lands the sun
            // ~10° above the horizon — the `DUSK_ONSET_DEG = 10°` band where
            // bevy_elements blends toward its golden dusk tint.
            date: CivilDate::new(2026, 3, 21),
            golden_hour: 17.5,
            location: GeoLocation {
                latitude_deg: 39.9042,
                // Tuned so 17:30 lands the sun at ~2.6° altitude: dusk tint
                // factor ≈ 0.75 (deeper orange horizon) and a relatively
                // darker blue-purple zenith (Feihei: "slightly more
                // orange-purple").
                longitude_deg: 126.4,
                utc_offset_hours: 8.0,
            },
            cloud_cover: None,
            cloud_layer: None,
            exposure_ev100: 13.0,
            exposure_noise_amplitude: 0.15,
        }
    }
}

pub struct SkyPlugin;

impl Plugin for SkyPlugin {
    fn build(&self, app: &mut App) {
        let sky = app
            .world()
            .get_resource::<HeadsUpSky>()
            .cloned()
            .unwrap_or_default();
        app.insert_resource(sky.clone());

        // Frozen golden hour: time_scale 0.0 and paused are equivalent
        // (bevy_elements-notes.md §3); paused states the intent explicitly.
        let clock = WorldClock::new(sky.date)
            .at_hour(sky.golden_hour)
            .with_time_scale(0.0);
        let mut director = WeatherDirector::default();
        director.force(WeatherPreset::Clear, Duration::ZERO);

        app.insert_resource(clock)
            .insert_resource(Climate::temperate())
            .insert_resource(sky.location.clone())
            .insert_resource(director)
            .add_plugins(
                ElementsPlugins
                    .build()
                    // R1.4: the game has no weather changes — drop everything
                    // outside climate/celestial/fog/cloud (ADR-0001).
                    .disable::<WindPlugin>()
                    .disable::<PrecipitationPlugin>()
                    .disable::<LightningPlugin>()
                    .disable::<DustPlugin>()
                    .disable::<TerrainPlugin>()
                    .disable::<HydrologyPlugin>()
                    .disable::<WaterPlugin>()
                    .disable::<GroundCoverPlugin>()
                    .disable::<WeatheredPlugin>(),
            )
            .add_systems(
                Update,
                (
                    // Runs between the weather director (set `Weather`) and
                    // the celestial/cloud consumers (set `Celestial`), so one
                    // coherent overridden `Weather` reaches lighting and the
                    // cloud layer derivation.
                    apply_cloud_cover_override
                        .after(ElementsSystems::Weather)
                        .before(ElementsSystems::Celestial),
                    // `update_layer` (in `Simulation`) re-derives the layer
                    // from `Weather` every frame; rewriting the resource after
                    // the set means the GPU pass consumes the override on the
                    // next frame — indistinguishable with a paused clock.
                    apply_cloud_layer_override.after(ElementsSystems::Simulation),
                    perturb_exposure,
                ),
            );
    }
}

fn apply_cloud_cover_override(sky: Res<HeadsUpSky>, mut weather: ResMut<bevy_elements::Weather>) {
    let Some(cover) = sky.cloud_cover else {
        return;
    };
    weather.cloud_cover = cover.clamp(0.0, 1.0);
}

fn apply_cloud_layer_override(sky: Res<HeadsUpSky>, mut layer: ResMut<CloudLayer>) {
    let Some((base_m, top_m)) = sky.cloud_layer else {
        return;
    };
    layer.base_m = base_m;
    layer.top_m = top_m.max(base_m + 1.0);
}

/// R1.5: subtle exposure drift driven by a deterministic function of the
/// camera's world position (sum-of-sines noise; no RNG on the frame path).
///
/// Note: the atmosphere's `ScatteringMedium` is a GPU LUT asset — per-position
/// density changes would rebuild it every frame, so only the exposure half of
/// the ADR's "density multiplier / exposure jitter" is implemented (recorded
/// in bevy-elements-notes.md).
fn perturb_exposure(
    sky: Res<HeadsUpSky>,
    cameras: Query<(&GlobalTransform, &mut Exposure), With<ElementsCamera>>,
) {
    let amplitude = sky.exposure_noise_amplitude;
    if amplitude == 0.0 {
        return;
    }
    for (transform, mut exposure) in cameras {
        let p = transform.translation();
        let noise = (p.x * 0.013).sin() * 0.5
            + (p.z * 0.021 + 1.7).sin() * 0.3
            + ((p.x + p.z) * 0.005 + 4.1).sin() * 0.2;
        exposure.ev100 = sky.exposure_ev100 + amplitude * noise;
    }
}

/// Camera component set for the golden-hour look (ADR-0005 §4): fixed
/// exposure, ACES tonemapping, natural bloom. Attach alongside
/// `ElementsCamera`.
#[derive(Bundle)]
pub struct GoldenHourCamera {
    pub elements_camera: ElementsCamera,
    pub exposure: Exposure,
    pub tonemapping: Tonemapping,
    pub bloom: Bloom,
}

impl Default for GoldenHourCamera {
    fn default() -> Self {
        Self {
            elements_camera: ElementsCamera {
                auto_exposure: false,
            },
            exposure: Exposure {
                ev100: HeadsUpSky::default().exposure_ev100,
            },
            tonemapping: Tonemapping::AcesFitted,
            // Above NATURAL's 0.15: spreads the low-sun glow across the
            // horizon band (golden-hour look, R1.3 tuning).
            bloom: Bloom {
                intensity: 0.3,
                ..Bloom::NATURAL
            },
        }
    }
}
