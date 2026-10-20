//! Season system plugin (R3).
//!
//! Four-season parameter table (leaf_density/leaf_color/sway_strength) shared
//! through a single uniform/storage buffer; season is chosen at run start and
//! never changes within a run (ADR-0004). Filled in during milestone R3.

use bevy::prelude::*;

pub struct SeasonPlugin;

impl Plugin for SeasonPlugin {
    fn build(&self, _app: &mut App) {
        // R3: season resource + tree MaterialExtension WESL wiring.
    }
}
