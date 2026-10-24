//! Exploration tracking (R4.1, ADR-0001: the original's exploration bitmap).
//!
//! A chunk counts as explored once the player has had it inside the active
//! window — standing in it is not required, matching the original's
//! "walked past" semantics. The set is a plain `HashSet` of chunk keys so the
//! R4.2 save can serialize it directly and regeneration stays bit-exact:
//! the same walk always marks the same keys.

use bevy::prelude::*;
use std::collections::HashSet;

use crate::world::GRID_CELLS;
use crate::world::collision::chunk_key;

/// Chunk keys the player has seen, at 50 m chunk granularity.
#[derive(Resource, Debug, Default, Clone)]
pub struct ExploredBitmap {
    explored: HashSet<(i32, i32)>,
}

impl ExploredBitmap {
    /// Marks one chunk explored. Returns true when it was new.
    pub fn mark(&mut self, key: (i32, i32)) -> bool {
        self.explored.insert(key)
    }

    /// True if the chunk has been explored.
    pub fn is_explored(&self, key: (i32, i32)) -> bool {
        self.explored.contains(&key)
    }

    /// Number of explored chunks.
    pub fn len(&self) -> usize {
        self.explored.len()
    }

    /// True when nothing is explored yet.
    pub fn is_empty(&self) -> bool {
        self.explored.is_empty()
    }

    /// Fraction of the world explored (0.0-1.0).
    pub fn fraction(&self) -> f32 {
        let total = (GRID_CELLS * GRID_CELLS) as f32;
        self.explored.len() as f32 / total
    }

    /// True once every chunk has been visited (R4.6 completion gate).
    pub fn is_complete(&self) -> bool {
        self.explored.len() >= (GRID_CELLS * GRID_CELLS) as usize
    }

    /// Explored keys, sorted, for deterministic serialization (R4.2).
    pub fn sorted_keys(&self) -> Vec<(i32, i32)> {
        let mut keys: Vec<_> = self.explored.iter().copied().collect();
        keys.sort_unstable();
        keys
    }

    /// Restores a previously saved bitmap (R4.3); keys are trusted input.
    pub fn restore(keys: impl IntoIterator<Item = (i32, i32)>) -> Self {
        ExploredBitmap {
            explored: keys.into_iter().collect(),
        }
    }
}

/// Message sent when the player's chunk changed (R4.1: the save system
/// listens to mark exploration; R4.6 may also use it for autosave timing).
#[derive(Message, Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlayerChunkChanged {
    /// The chunk the player is now in.
    pub new_chunk: (i32, i32),
    /// The chunk the player left (None on the first frame).
    pub old_chunk: Option<(i32, i32)>,
}

/// Marks the player's current chunk explored and sends the change message.
///
/// Called from the streaming system (it already computes the camera
/// position); tracking inside one system keeps the semantics single-sourced.
pub(crate) fn track_player_chunk(
    cameras: Query<&Transform, With<Camera3d>>,
    mut bitmap: ResMut<ExploredBitmap>,
    mut last: Local<Option<(i32, i32)>>,
    mut changed: MessageWriter<PlayerChunkChanged>,
) {
    let Ok(camera) = cameras.single() else {
        return;
    };
    let key = chunk_key(Vec2::new(camera.translation.x, camera.translation.z));
    bitmap.mark(key);
    if *last != Some(key) {
        changed.write(PlayerChunkChanged {
            new_chunk: key,
            old_chunk: *last,
        });
        *last = Some(key);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mark_tracks_new_keys_once() {
        let mut bitmap = ExploredBitmap::default();
        assert!(bitmap.mark((3, 4)), "first visit is new");
        assert!(!bitmap.mark((3, 4)), "revisit is not");
        assert!(bitmap.is_explored((3, 4)));
        assert_eq!(bitmap.len(), 1);
        assert!(bitmap.fraction() > 0.0);
        assert!(!bitmap.is_complete());
    }

    #[test]
    fn completion_requires_every_chunk() {
        let mut bitmap = ExploredBitmap::default();
        let total = (GRID_CELLS * GRID_CELLS) as i32;
        for i in 0..total {
            bitmap.mark((i % GRID_CELLS as i32, i / GRID_CELLS as i32));
        }
        assert!(bitmap.is_complete());
        assert!((bitmap.fraction() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn sorted_keys_round_trip_is_bit_exact() {
        let mut bitmap = ExploredBitmap::default();
        for k in [(7_i32, 2_i32), (0, 0), (39, 39), (7, 2)] {
            bitmap.mark(k);
        }
        let restored = ExploredBitmap::restore(bitmap.sorted_keys());
        assert_eq!(bitmap.len(), restored.len());
        for key in bitmap.sorted_keys() {
            assert!(restored.is_explored(key));
        }
        assert_eq!(bitmap.sorted_keys(), vec![(0, 0), (7, 2), (39, 39)]);
    }
}
