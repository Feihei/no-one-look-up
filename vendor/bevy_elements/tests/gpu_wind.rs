//! The wind field on a real GPU: the kernels compile, the solver relaxes to the log-law wind and
//! the projection leaves it nearly divergence-free, read back through `Environment::wind_at`.
//!
//! Needs a GPU: `cargo test --test gpu_wind -- --ignored`.

mod support;

use std::time::Duration;

use bevy::app::AppExit;
use bevy::ecs::system::RunSystemOnce;
use bevy::prelude::*;
use bevy::time::TimeUpdateStrategy;
use bevy_elements::*;

fn headless_app() -> App {
    headless_app_with(None)
}

fn headless_app_with(terrain: Option<Terrain>) -> App {
    let mut app = App::new();
    app.add_plugins((support::headless_plugins(), ElementsPlugins))
        .insert_resource(TimeUpdateStrategy::ManualDuration(Duration::from_secs_f64(
            1.0 / 60.0,
        )))
        // Overcast blows 5 m/s × mean_wind/4: mean 8 m/s gives exactly 10 m/s at 10 m.
        .insert_resource(Climate {
            mean_wind_ms: 8.0,
            ..Climate::temperate()
        })
        .add_systems(Startup, setup);
    if let Some(terrain) = terrain {
        app.add_systems(Startup, move |mut commands: Commands| {
            commands.spawn(terrain.clone());
        });
    }
    support::finish(&mut app);
    app
}

fn setup(
    mut commands: Commands,
    mut director: ResMut<WeatherDirector>,
    mut images: ResMut<Assets<Image>>,
) {
    commands.spawn((
        Camera3d::default(),
        support::image_target(&mut images),
        ElementsCamera::default(),
        Transform::from_xyz(0.0, 10.0, 0.0),
    ));
    director.force(WeatherPreset::Overcast, Duration::ZERO);
}

#[test]
#[ignore = "needs a GPU"]
fn the_wind_field_relaxes_to_the_log_law_and_stays_divergence_free() {
    let mut app = headless_app();
    for _ in 0..120 {
        app.update();
    }

    let (age, u10, mean_speed, mean_div) = app
        .world_mut()
        .run_system_once(|env: Environment| {
            let h = WIND_CELL_M;
            let centre = |c: i32| (c as f32 + 0.5) * h;
            let mut speed = 0.0;
            let mut div = 0.0;
            let mut n = 0.0;
            // Interior cells of the High grid (64 × 16 × 64 around x = z = 0), 10 m above ground.
            for cx in -24..24 {
                for cz in -24..24 {
                    let p = Vec3::new(centre(cx), 10.0, centre(cz));
                    speed += env.wind_at(p).length();
                    let d = (env.wind_at(p + Vec3::X * h).x - env.wind_at(p - Vec3::X * h).x
                        + env.wind_at(p + Vec3::Y * h).y
                        - env.wind_at(p - Vec3::Y * h).y
                        + env.wind_at(p + Vec3::Z * h).z
                        - env.wind_at(p - Vec3::Z * h).z)
                        / (2.0 * h);
                    div += d.abs();
                    n += 1.0;
                }
            }
            (
                env.snapshot_age(),
                env.weather().wind_speed_ms,
                speed / n,
                div / n,
            )
        })
        .expect("the query system runs");
    app.world_mut().write_message(AppExit::Success);

    println!(
        "u10 {u10} m/s, mean |v| at 10 m {mean_speed} m/s, mean |div| {mean_div} 1/s, snapshot age {age:?}"
    );
    assert!(
        age < Duration::from_secs(1),
        "no wind readback arrived (age {age:?})"
    );
    assert!(
        (u10 - 10.0).abs() < 1e-4,
        "forced Overcast should blow 10 m/s, got {u10}"
    );
    assert!(
        (8.0..=12.0).contains(&mean_speed),
        "mean |v| at 10 m was {mean_speed} m/s"
    );
    assert!(
        mean_div < 1e-2,
        "mean |div| after projection was {mean_div} 1/s"
    );
}

/// Over rolling hills the projection still leaves the wind nearly divergence-free: the solid mask
/// follows the terrain, the window follows the ground, and the open window edges let the air leave
/// (`terrain.md` wiring; wind.md Deviation 1).
#[test]
#[ignore = "needs a GPU"]
fn the_wind_over_hills_stays_divergence_free() {
    let mut app = headless_app_with(Some(
        Terrain::procedural(ProceduralTerrain::hills(42)).with_resolution(UVec2::splat(256)),
    ));
    for _ in 0..360 {
        app.update();
    }
    let (mean_div, cells, mean_speed) = app
        .world_mut()
        .run_system_once(|env: Environment, wind: Res<WindState>| {
            let h = WIND_CELL_M;
            let origin = wind.field_origin;
            let extent = wind.field_extent;
            let centre = Vec2::new(origin.x + 0.5 * extent.x, origin.z + 0.5 * extent.z);
            let (mut div, mut speed, mut n) = (0.0, 0.0, 0usize);
            for ix in -20..20 {
                for iz in -20..20 {
                    let xz = centre + Vec2::new(ix as f32, iz as f32) * h;
                    let Some(ground) = env.terrain_height(xz) else {
                        continue;
                    };
                    let p = Vec3::new(xz.x, ground + 12.0, xz.y);
                    // The whole stencil clear of the ground and inside the window.
                    let clear = [Vec2::X, -Vec2::X, Vec2::Y, -Vec2::Y].iter().all(|o| {
                        env.terrain_height(xz + *o * h)
                            .is_some_and(|g| g + 4.0 < p.y - h)
                    });
                    if !clear || p.y + 2.0 * h > origin.y + extent.y || p.y - h < origin.y {
                        continue;
                    }
                    div += ((env.wind_at(p + Vec3::X * h).x - env.wind_at(p - Vec3::X * h).x
                        + env.wind_at(p + Vec3::Y * h).y
                        - env.wind_at(p - Vec3::Y * h).y
                        + env.wind_at(p + Vec3::Z * h).z
                        - env.wind_at(p - Vec3::Z * h).z)
                        / (2.0 * h))
                        .abs();
                    speed += env.wind_at(p).length();
                    n += 1;
                }
            }
            (div / n.max(1) as f32, n, speed / n.max(1) as f32)
        })
        .expect("the query system runs");
    app.world_mut().write_message(AppExit::Success);
    println!("over hills: mean |div| {mean_div} 1/s, mean |v| {mean_speed} m/s over {cells} cells");
    assert!(cells > 200, "too few air cells ({cells})");
    assert!(mean_div < 1e-2, "mean |div| over hills was {mean_div} 1/s");
    assert!(
        mean_speed > 3.0,
        "the wind died over the hills ({mean_speed} m/s)"
    );
}
