//! Heads-Up library crate: sky/world/season/save/menu plugin wiring.
//!
//! The binary (`main.rs`) only assembles the app; all systems and plugins
//! live here so integration tests can drive them headless.

pub mod core;
pub mod plugins;
pub mod world;
