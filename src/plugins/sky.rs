//! Sky system plugin (R1).
//!
//! Wraps bevy_elements (Atmosphere/Celestial/Cloud) behind a `HeadsUpSky`
//! Resource with the golden hour locked at 17:30 and clear weather forced
//! (ADR-0001/0003). Filled in during milestone R1.

use bevy::prelude::*;

pub struct SkyPlugin;

impl Plugin for SkyPlugin {
    fn build(&self, _app: &mut App) {
        // R1: integrate bevy_elements + HeadsUpSky resource.
    }
}
