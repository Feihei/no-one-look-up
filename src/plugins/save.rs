//! Save system plugin (R4).
//!
//! Save = seed + season enum + exploration bitmap + player pose; restore
//! regenerates the layout from the seed (ADR-0001) and must complete in
//! under a second. R4.2 owns the format and file IO ([`save_data`]); this
//! plugin owns the restore flow: a [`LoadRequest`] message tears down the
//! streamed world, rebuilds it from the saved seed and restores the
//! exploration state and player pose.

use bevy::prelude::*;

use crate::plugins::save_data;

use crate::player::{EYE_HEIGHT, PlayerBody};
use crate::world::chunks::{ChunkManager, ChunkRoot, WorldSystems};
use crate::world::collision::CollisionIndex;
use crate::world::explored::ExploredBitmap;
use crate::world::path_network::{HeadsUpSeed, generate_world};

pub use self::save_data::{SaveData, Season, read_save, save_path, write_save};

/// Requests a restore from a captured save. Sent by the R4.5 menu (and by
/// tests); consumed once per frame.
#[derive(Message, Debug, Clone)]
pub struct LoadRequest(pub SaveData);

pub struct SavePlugin;

impl Plugin for SavePlugin {
    fn build(&self, app: &mut App) {
        app.add_message::<LoadRequest>().add_systems(
            Update,
            handle_load_request
                // Restore before streaming: the world this frame is rebuilt
                // around the restored pose, not the one the walker left.
                .before(WorldSystems::Streaming),
        );
    }
}

/// Tears down the streamed world and rebuilds it from the save.
fn handle_load_request(
    mut requests: MessageReader<LoadRequest>,
    mut commands: Commands,
    chunk_roots: Query<Entity, With<ChunkRoot>>,
    mut manager: ResMut<ChunkManager>,
    mut collision: ResMut<CollisionIndex>,
    mut bitmap: ResMut<ExploredBitmap>,
    mut players: Query<&mut Transform, With<Camera3d>>,
) {
    let Some(request) = requests.read().next() else {
        return;
    };
    let data = &request.0;
    if data.version != save_data::SAVE_VERSION {
        error!(
            "save version {} not supported (this build expects {})",
            data.version,
            save_data::SAVE_VERSION
        );
        return;
    }

    // Regenerate the layout from the seed (bit-exact; tested in path_network).
    let started = std::time::Instant::now();
    match generate_world(data.seed) {
        Ok((grid, graph)) => {
            info!(
                "save restored: seed={:#x} blocks={} explored={} in {:.1?}",
                data.seed,
                graph.blocks.len(),
                data.explored.len(),
                started.elapsed()
            );
            commands.insert_resource(grid);
            commands.insert_resource(graph);
        }
        Err(err) => {
            error!("save restore failed: world generation error: {err}");
            return;
        }
    }

    // Streamed state: drop every chunk group and start the window over
    // around the restored pose (streaming refills it within a few frames).
    for entity in &chunk_roots {
        commands.entity(entity).try_despawn();
    }
    *manager = ChunkManager::default();
    *collision = CollisionIndex::default();
    *bitmap = data.explored_bitmap();

    // The seed resource must agree with the restored world for anything that
    // reads it later (e.g. a menu-initiated restart).
    commands.insert_resource(HeadsUpSeed(data.seed));

    // Player pose: eye height, saved XZ, saved facing (pitch resets to the
    // golden-hour default; the format keeps room for it if that ever moves).
    if let Ok(mut transform) = players.single_mut() {
        transform.translation = Vec3::new(data.player_pos[0], EYE_HEIGHT, data.player_pos[1]);
        transform.rotation = Quat::from_euler(
            bevy::math::EulerRot::YXZ,
            data.player_yaw,
            PlayerBody::default().pitch,
            0.0,
        );
    } else {
        warn!("save restore: no camera found, player pose not applied");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::chunks::WorldPlugin;
    use bevy::MinimalPlugins;

    fn app_with_world() -> App {
        let mut app = App::new();
        app.add_plugins((MinimalPlugins, WorldPlugin, SavePlugin))
            .init_resource::<ButtonInput<KeyCode>>();
        app.world_mut().spawn((
            Camera3d::default(),
            PlayerBody::default(),
            Transform::from_xyz(1000.0, EYE_HEIGHT, 1000.0),
        ));
        app
    }

    #[test]
    fn load_request_restores_bitmap_and_pose() {
        let mut app = app_with_world();
        app.update(); // Startup + first streaming frame.

        // Capture the save at the spawn pose, then wander away and explore.
        let (bitmap_snapshot, saved) = {
            let bitmap = app.world().resource::<ExploredBitmap>();
            let data = SaveData::capture(
                crate::world::chunks::DEFAULT_SEED,
                Season::Summer,
                bitmap,
                [1000.0, 1000.0],
                0.0,
            );
            (bitmap.clone(), data)
        };
        assert!(bitmap_snapshot.is_explored((20, 20)));

        // Wander: teleport far away and let streaming mark exploration.
        let mut cameras = app
            .world_mut()
            .query_filtered::<&mut Transform, With<Camera3d>>();
        for mut transform in cameras.iter_mut(app.world_mut()) {
            transform.translation = Vec3::new(1500.0, EYE_HEIGHT, 1500.0);
        }
        let _ = &mut cameras;
        app.update();
        app.update();
        let after_walk = app.world().resource::<ExploredBitmap>().clone();
        assert!(
            after_walk.len() > bitmap_snapshot.len(),
            "walking adds explored chunks"
        );

        // Restore: the message resets everything to the saved state.
        app.world_mut().write_message(LoadRequest(saved.clone()));
        app.update();
        app.update();

        let bitmap = app.world().resource::<ExploredBitmap>();
        assert_eq!(bitmap.len(), saved.explored.len(), "bitmap restored");
        assert!(bitmap.is_explored((20, 20)));
        assert!(!bitmap.is_explored((30, 30)), "post-walk chunks unlearned");

        let mut cameras = app
            .world_mut()
            .query_filtered::<&Transform, With<Camera3d>>();
        let transform = cameras.single(app.world()).expect("camera exists");
        assert!((transform.translation.x - 1000.0).abs() < 1e-4);
        assert!((transform.translation.z - 1000.0).abs() < 1e-4);

        // Streaming rebuilds the window around the restored pose.
        let manager = app.world().resource::<ChunkManager>();
        assert!(!manager.active.is_empty(), "chunks respawn after restore");
    }

    #[test]
    fn future_versions_are_rejected_safely() {
        let mut app = app_with_world();
        app.update();

        let mut future = SaveData::capture(
            crate::world::chunks::DEFAULT_SEED,
            Season::Summer,
            &ExploredBitmap::default(),
            [0.0, 0.0],
            0.0,
        );
        future.version = save_data::SAVE_VERSION + 1;
        app.world_mut().write_message(LoadRequest(future));
        app.update();

        // The world is untouched: chunks are still active.
        let manager = app.world().resource::<ChunkManager>();
        assert!(!manager.active.is_empty(), "reject must not tear down");
    }

    #[test]
    fn regeneration_from_seed_stays_under_a_second() {
        // ADR-0001: restore must complete in under a second. The dominant
        // cost is layout generation; assert it directly (release-target).
        let started = std::time::Instant::now();
        let result = generate_world(0x4845_4144);
        let elapsed = started.elapsed();
        assert!(result.is_ok());
        assert!(elapsed.as_secs_f64() < 1.0, "generation took {elapsed:?}");
    }
}
