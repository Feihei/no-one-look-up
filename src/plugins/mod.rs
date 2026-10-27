//! Feature plugins for Heads-Up.
//!
//! Plugin split mirrors the original Godot Manager layout (see ADR-0005
//! mapping table): SkyPlugin, WorldPlugin, SeasonPlugin, SavePlugin,
//! MenuPlugin. Each milestone fills in the corresponding plugin.

pub mod menu;
pub mod save;
pub mod save_data;
pub mod season;
pub mod sky;
// R3.9: the F7 gallery is retired. It was a one-shot snapshot pinned where the
// player stood, so walking away left it behind and it needed two F7 presses to
// rebuild — which read as "trees don't stream with chunks". World trees now
// come from chunk content (`world::chunks` + `world::tree_mesh`). The module
// is kept out of the build; `examples/tree_gallery.rs` is the shape reference.
#[cfg(any())]
pub mod tree_preview;
pub mod world;

pub use sky::GoldenHourCamera;

use bevy::app::PluginGroupBuilder;
use bevy::prelude::*;

use menu::MenuPlugin;
use save::SavePlugin;
use season::SeasonPlugin;
use sky::SkyPlugin;
use world::WorldPlugin;

use crate::world::debug_viz::DebugVizPlugin;

/// Group plugin wiring every feature plugin in dependency order.
pub struct HeadsUpPlugins;

impl PluginGroup for HeadsUpPlugins {
    fn build(self) -> PluginGroupBuilder {
        PluginGroupBuilder::start::<Self>()
            .add(SkyPlugin)
            .add(WorldPlugin)
            .add(DebugVizPlugin)
            // NOTE (R3.9): the F7 gallery is NOT registered. It was a
            // one-shot snapshot pinned where the player stood, so walking away
            // left it behind and it needed two F7 presses to rebuild — which
            // read as "trees don't stream with chunks". World trees now come
            // from chunk content (`chunks.rs` + `world::tree_mesh`); the
            // gallery module stays as a shape reference for
            // `examples/tree_gallery.rs`.
            .add(SeasonPlugin)
            .add(SavePlugin)
            .add(MenuPlugin)
    }
}
