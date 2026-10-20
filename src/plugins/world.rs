//! World plugin wiring (R2).
//!
//! The world plugin itself lives in `crate::world::chunks` (layout
//! generation + chunk streaming, ADR-0002/0004); this module re-exports it
//! for the plugin group so the file layout mirrors the original Godot
//! Manager split (see ADR-0005 mapping table).

pub use crate::world::chunks::WorldPlugin;
