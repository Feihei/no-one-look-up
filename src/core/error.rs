//! Project-wide error handling convention.
//!
//! All fallible code paths use [`GameResult`]. Systems on the Update/Render
//! frame path must not panic (see root AGENTS.md); errors are surfaced as
//! explicit `Result` values and logged or handled by the caller.
#![allow(dead_code)] // consumed per milestone: R2 generation, R4 saves

use std::fmt;

/// Convenience alias used across the project.
pub type GameResult<T> = Result<T, GameError>;

/// Top-level error type for No One Looks Up Even Alone.
#[derive(Debug)]
pub enum GameError {
    /// An asset failed to load or validate (glTF models, textures, ...).
    Asset(String),
    /// Serialized save data is malformed or from an incompatible version.
    SaveFormat(String),
    /// A generation invariant was violated (layout graph, chunk streaming...).
    Generation(String),
    /// Anything else that does not fit a dedicated variant.
    Other(String),
}

impl fmt::Display for GameError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GameError::Asset(msg) => write!(f, "asset error: {msg}"),
            GameError::SaveFormat(msg) => write!(f, "save format error: {msg}"),
            GameError::Generation(msg) => write!(f, "generation error: {msg}"),
            GameError::Other(msg) => write!(f, "{msg}"),
        }
    }
}

impl std::error::Error for GameError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_includes_variant_context() {
        let err = GameError::SaveFormat("missing field `seed`".to_string());
        assert_eq!(err.to_string(), "save format error: missing field `seed`");
    }
}
