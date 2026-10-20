//! Project-wide error handling convention.
//!
//! All fallible code paths use [`HeadsUpResult`]. Systems on the Update/Render
//! frame path must not panic (see root AGENTS.md); errors are surfaced as
//! explicit `Result` values and logged or handled by the caller.
#![allow(dead_code)] // consumed per milestone: R2 generation, R4 saves

use std::fmt;

/// Convenience alias used across the project.
pub type HeadsUpResult<T> = Result<T, HeadsUpError>;

/// Top-level error type for Heads-Up.
#[derive(Debug)]
pub enum HeadsUpError {
    /// An asset failed to load or validate (glTF models, textures, ...).
    Asset(String),
    /// Serialized save data is malformed or from an incompatible version.
    SaveFormat(String),
    /// A generation invariant was violated (layout graph, chunk streaming...).
    Generation(String),
    /// Anything else that does not fit a dedicated variant.
    Other(String),
}

impl fmt::Display for HeadsUpError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HeadsUpError::Asset(msg) => write!(f, "asset error: {msg}"),
            HeadsUpError::SaveFormat(msg) => write!(f, "save format error: {msg}"),
            HeadsUpError::Generation(msg) => write!(f, "generation error: {msg}"),
            HeadsUpError::Other(msg) => write!(f, "{msg}"),
        }
    }
}

impl std::error::Error for HeadsUpError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_includes_variant_context() {
        let err = HeadsUpError::SaveFormat("missing field `seed`".to_string());
        assert_eq!(err.to_string(), "save format error: missing field `seed`");
    }
}
