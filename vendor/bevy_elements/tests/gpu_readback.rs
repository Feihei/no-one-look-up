//! Gameplay queries in an app with no window: nothing presents, so nothing paces the frames, and
//! the readbacks behind `Environment` must still arrive (`docs/research/readback-pacing.md`).
//!
//! Needs a GPU: `cargo test --test gpu_readback -- --ignored`.

mod support;

use std::time::Duration;

use bevy::app::AppExit;
use bevy::ecs::system::RunSystemOnce;
use bevy::prelude::*;
use bevy::time::TimeUpdateStrategy;
use bevy_elements::*;

const FRAME_S: f64 = 1.0 / 60.0;

fn setup(mut commands: Commands, mut images: ResMut<Assets<Image>>) {
    commands.spawn((
        Camera3d::default(),
        support::image_target(&mut images),
        ElementsCamera::default(),
        Transform::from_xyz(0.0, 30.0, 0.0),
    ));
    commands.spawn(
        Terrain::flat(0.0)
            .with_extent(Vec2::splat(512.0))
            .with_resolution(UVec2::splat(256)),
    );
}

/// Every domain's field (wind, clouds, terrain, water, ground cover) is read back at most a
/// few frames after the interval, frame after frame, for twenty simulated seconds. Without
/// pacing the queue grew every frame and so did the readback latency: the oldest snapshot here
/// was 683 ms, and in the ground-cover suite the snapshot froze for good three seconds in.
#[test]
#[ignore = "needs a GPU"]
fn the_queries_stay_fresh_without_a_window() {
    let interval = Duration::from_millis(100);
    let mut app = App::new();
    app.add_plugins((support::headless_plugins(), ElementsPlugins))
        .insert_resource(TimeUpdateStrategy::ManualDuration(Duration::from_secs_f64(
            FRAME_S,
        )))
        .insert_resource(ElementsQuality::Ultra)
        .insert_resource(ElementsReadback { interval })
        .add_systems(Startup, setup);
    support::finish(&mut app);

    // Bring-up: pipelines compile and every field is read once.
    for _ in 0..240 {
        app.update();
    }
    // The interval, plus the frames a request takes to be copied, mapped and delivered.
    let limit = interval + Duration::from_secs_f64(6.0 * FRAME_S);
    let mut worst = Duration::ZERO;
    for frame in 0..1200 {
        app.update();
        let (age, depth, wetness) = app
            .world_mut()
            .run_system_once(|env: Environment| {
                (
                    env.snapshot_age(),
                    env.water_depth(Vec2::ZERO),
                    env.wetness(Vec2::ZERO),
                )
            })
            .expect("the query runs");
        assert!(
            depth.is_some() && wetness.is_some(),
            "frame {frame}: the terrain's fields were never read back"
        );
        worst = worst.max(age);
        assert!(
            age <= limit,
            "frame {frame}: the snapshot is {age:?} old, limit {limit:?}"
        );
    }
    println!("RESULT oldest snapshot over 1200 frames: {worst:?} (limit {limit:?})");
    app.world_mut().write_message(AppExit::Success);
    app.update();
}
