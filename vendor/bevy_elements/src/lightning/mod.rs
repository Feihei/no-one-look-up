//! Lightning and thunder: a seeded strike process, Reed & Wyvill bolts rendered as additive
//! ribbons, a flash of light, and thunder that arrives at the speed of sound.
//!
//! Strikes are a Poisson process at `Weather::storminess / 60` per second of `Time<Virtual>`,
//! uniform over an 8 km disc around the followed camera, drawn from `hash64(Climate::seed, i)`;
//! a game can add its own with [`StrikeLightning`]. Each strike writes a [`LightningStrike`], a
//! [`Thunder`] `distance / 343` s of real time later, and (with a renderer) a bolt, a flash
//! `PointLight`, and the [`LightningFlash`] resource the cloud and precipitation passes read.
//!
//! The strike process, the flash envelope and the messages need no renderer, so they run on
//! `MinimalPlugins`; the bolt mesh and the light are added when the app has mesh assets.

mod bolt;
mod strikes;
mod visual;

use bevy::prelude::*;

pub use strikes::{LightningStrike, SPEED_OF_SOUND_MS, STRIKE_RADIUS_M, StrikeLightning, Thunder};

use crate::climate::Climate;
use crate::flash::LightningFlash;
use crate::plugin::{ElementsSystems, ensure_core};
use strikes::{ActiveBolt, Strikes};

/// Luminous power of a first return stroke averaged over one 1/60 s frame, lm
/// (`docs/research/lightning.md`, Deviation 8). Candela is this over 4π.
pub const FLASH_LUMENS: f32 = 1.0e8;
/// Height of the flash light above the strike point, m.
pub(crate) const FLASH_HEIGHT_M: f32 = 300.0;
/// Range of the flash light, m.
pub(crate) const FLASH_RANGE_M: f32 = 5000.0;

/// Lightning, thunder and the flash resource.
///
/// Inserts (if absent) [`Climate`] and [`LightningFlash`], adds the [`StrikeLightning`],
/// [`LightningStrike`] and [`Thunder`] messages, and runs in [`ElementsSystems::Simulation`].
pub struct LightningPlugin;

impl Plugin for LightningPlugin {
    fn build(&self, app: &mut App) {
        ensure_core(app);
        app.init_resource::<Climate>()
            .init_resource::<LightningFlash>()
            .init_resource::<Strikes>()
            .add_message::<StrikeLightning>()
            .add_message::<LightningStrike>()
            .add_message::<Thunder>()
            .add_systems(
                Update,
                (strikes::run_strikes, advance_bolts, write_flash)
                    .chain()
                    .in_set(ElementsSystems::Simulation),
            );
        // Meshes exist only with a renderer; without one the bolt is a timeline and nothing else.
        if app.world().contains_resource::<Assets<Mesh>>() {
            visual::plugin(app);
        }
    }
}

/// Move every flash's clock on by this frame and average its envelope over it.
fn advance_bolts(
    time: Res<Time<Virtual>>,
    mut bolts: Query<(Entity, &mut ActiveBolt)>,
    mut commands: Commands,
) {
    let dt = time.delta_secs_f64();
    for (entity, mut bolt) in &mut bolts {
        let (from, to) = (bolt.age, bolt.age + dt);
        bolt.mean = bolt.strokes.mean(from, to) as f32;
        bolt.mean_first = bolt.strokes.mean_first(from, to) as f32;
        bolt.age = to;
        if bolt.strokes.finished(to) {
            commands.entity(entity).despawn();
        }
    }
}

/// Publish the brightest flash this frame as [`LightningFlash`]: its channel top at the cloud
/// base, `Φ·e / 4π` candela; intensity 0 when nothing is flashing.
pub(crate) fn write_flash(bolts: Query<&ActiveBolt>, mut flash: ResMut<LightningFlash>) {
    let brightest = bolts
        .iter()
        .filter(|b| b.mean > 0.0)
        .max_by(|a, b| a.mean.total_cmp(&b.mean));
    let next = match brightest {
        Some(bolt) => LightningFlash {
            position: bolt.bolt.top(),
            intensity_cd: FLASH_LUMENS * bolt.mean / (4.0 * std::f32::consts::PI),
        },
        None => LightningFlash {
            position: flash.position,
            intensity_cd: 0.0,
        },
    };
    if *flash != next {
        *flash = next;
    }
}
