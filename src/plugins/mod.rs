//! Feature plugins for Heads-Up.
//!
//! Plugin split mirrors the original Godot Manager layout (see ADR-0005
//! mapping table): SkyPlugin, WorldPlugin, SeasonPlugin, SavePlugin,
//! MenuPlugin. Each milestone fills in the corresponding plugin.

pub mod menu;
pub mod save;
pub mod season;
pub mod sky;
pub mod world;

use bevy::app::PluginGroupBuilder;
use bevy::prelude::*;

use menu::MenuPlugin;
use save::SavePlugin;
use season::SeasonPlugin;
use sky::SkyPlugin;
use world::WorldPlugin;

/// Group plugin wiring every feature plugin in dependency order.
pub struct HeadsUpPlugins;

impl PluginGroup for HeadsUpPlugins {
    fn build(self) -> PluginGroupBuilder {
        PluginGroupBuilder::start::<Self>()
            .add(SkyPlugin)
            .add(WorldPlugin)
            .add(SeasonPlugin)
            .add(SavePlugin)
            .add(MenuPlugin)
    }
}
