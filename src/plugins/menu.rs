//! Menu UI plugin (R4).
//!
//! Menu-driven run start (no auto world generation), pause menu, save panel,
//! exploration-complete dialog and HUD, built with Bevy UI (em/rem units).
//! Filled in during milestone R4.

use bevy::prelude::*;

pub struct MenuPlugin;

impl Plugin for MenuPlugin {
    fn build(&self, _app: &mut App) {
        // R4: main menu, pause menu, HUD; is_game_active input gating.
    }
}
