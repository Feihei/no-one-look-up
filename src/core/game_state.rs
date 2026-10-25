//! Game phase state (R4.5/4.6): menus gate the world.
//!
//! The phase is the single source of truth for `is_game_active`: player
//! movement, chunk streaming and exploration tracking all stop outside
//! [`GamePhase::Playing`], so the menus can rebuild the world (via the
//! R4.3 restore flow) without the walker stepping on it.

use bevy::prelude::*;

/// Which screen the player is on. `Playing` and `Paused` both keep the
/// streamed world alive; only `Playing` runs movement and exploration.
#[derive(Resource, Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GamePhase {
    /// Main menu: seed input + start/load/quit. No world interaction.
    #[default]
    MainMenu,
    /// The game runs: movement, streaming, exploration, autosave.
    Playing,
    /// Esc pause menu: resume / save / back to menu. The world is frozen
    /// but kept alive (a resume must not re-stream).
    Paused,
}

impl GamePhase {
    /// True while the game actually runs (movement/streaming/exploration).
    pub fn is_game_active(&self) -> bool {
        matches!(self, GamePhase::Playing)
    }
}

/// The main menu's seed input buffer: digits typed by the player, parsed on
/// start (empty = the default seed).
#[derive(Resource, Debug, Clone, Default)]
pub struct SeedInput(pub String);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_playing_is_game_active() {
        assert!(!GamePhase::MainMenu.is_game_active());
        assert!(!GamePhase::Paused.is_game_active());
        assert!(GamePhase::Playing.is_game_active());
    }

    #[test]
    fn default_phase_is_the_main_menu() {
        assert_eq!(GamePhase::default(), GamePhase::MainMenu);
    }
}
