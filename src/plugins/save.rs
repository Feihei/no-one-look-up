//! Save system plugin (R4).
//!
//! Save = seed + map size + season enum + exploration bitmap + player pose;
//! restore must complete in under a second (ADR-0001). Filled in during
//! milestone R4.

use bevy::prelude::*;

pub struct SavePlugin;

impl Plugin for SavePlugin {
    fn build(&self, _app: &mut App) {
        // R4: exploration bitmap, serde JSON persistence, save slots.
    }
}
