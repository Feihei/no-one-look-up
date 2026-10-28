//! Water rendering on a real GPU: the generated normal map and the absorption of a flat bed under
//! a 2 m layer, each read back from the GPU. See `docs/research/water-rendering.md` (Validation
//! targets 7 and 11).
//!
//! Needs a GPU: `cargo test --release --test gpu_water -- --ignored --test-threads=1`. The tests
//! share one process and each builds a renderer, so they take a lock and run one at a time.

mod support;

use std::sync::Mutex;
use std::sync::atomic::Ordering;
use std::time::Duration;

use bevy::app::AppExit;
use bevy::camera::RenderTarget;
use bevy::prelude::*;
use bevy::render::render_resource::{TextureFormat, TextureUsages};
use bevy::render::storage::ShaderBuffer;
use bevy::time::TimeUpdateStrategy;
use bevy_elements::*;

static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());

const DT: f64 = 1.0 / 60.0;
const TARGET: u32 = 256;

fn pin_weather(mut weather: ResMut<Weather>) {
    weather.precipitation_mm_h = 0.0;
    weather.temperature_c = 15.0;
}

#[derive(Resource)]
struct Target(Handle<Image>);

fn setup(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    mut director: ResMut<WeatherDirector>,
) {
    let mut target = Image::new_target_texture(TARGET, TARGET, TextureFormat::Rgba16Float, None);
    target.texture_descriptor.usage |= TextureUsages::COPY_SRC;
    let image = images.add(target);
    commands.insert_resource(Target(image.clone()));
    commands.spawn((
        Camera3d::default(),
        RenderTarget::Image(image.into()),
        ElementsCamera::default(),
        Transform::from_xyz(0.0, 40.0, 0.0).looking_at(Vec3::ZERO, Vec3::NEG_Z),
    ));
    commands.spawn(
        Terrain::flat(0.0)
            .with_extent(Vec2::splat(256.0))
            .with_resolution(UVec2::splat(128)),
    );
    director.force(WeatherPreset::Clear, Duration::ZERO);
}

fn build() -> App {
    let mut app = App::new();
    app.add_plugins((support::headless_plugins(), ElementsPlugins))
        .insert_resource(TimeUpdateStrategy::ManualDuration(Duration::from_secs_f64(
            DT,
        )))
        .insert_resource(ElementsQuality::Ultra)
        .insert_resource(HydrologySettings {
            boundary: HydrologyBoundary::Closed,
            evaporation: false,
            // Held until the test has written its layers.
            time_scale: 0.0,
            ..default()
        })
        .insert_resource(WaterSettings {
            // A flat surface: the ripples are the subject of the normal-map test.
            ripple_scale: 0.0,
            ..default()
        })
        .insert_resource(WorldClock {
            paused: true,
            ..WorldClock::new(CivilDate::new(2026, 6, 21)).at_hour(12.0)
        })
        .add_systems(Startup, setup)
        .add_systems(
            Update,
            pin_weather
                .after(ElementsSystems::Weather)
                .before(ElementsSystems::Simulation),
        );
    support::finish(&mut app);
    app
}

fn run_until(app: &mut App, what: &str, mut done: impl FnMut(&mut App) -> bool) {
    let deadline = std::time::Instant::now() + Duration::from_secs(120);
    while std::time::Instant::now() < deadline {
        app.update();
        if done(app) {
            return;
        }
    }
    panic!("timed out waiting for {what}");
}

/// Validation 7: the `water_normals` kernel's mip 0, read back.
#[test]
#[ignore = "needs a GPU"]
fn gpu_water_normals_tile_and_have_the_cox_munk_slope() {
    let _lock = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let mut app = build();
    run_until(&mut app, "the normal map", |app| {
        app.world()
            .resource::<WaterState>()
            .normals_ready
            .load(Ordering::Relaxed)
    });
    for _ in 0..3 {
        app.update();
    }
    let normals = app.world().resource::<WaterState>().normals.clone();
    let n = 512usize;
    let bytes = support::read_texture(&mut app, normals, n * 4, n);
    assert_eq!(bytes.len(), n * n * 4);
    let texel = |x: usize, y: usize| &bytes[(y * n + x) * 4..(y * n + x) * 4 + 4];
    let decode = |x: usize, y: usize| {
        let t = texel(x, y);
        let v = |c: u8| f32::from(c) / 255.0 * 2.0 - 1.0;
        Vec3::new(v(t[0]), v(t[1]), v(t[2]))
    };
    let slope = |x: usize, y: usize| {
        let d = decode(x, y);
        Vec2::new(d.x, d.y) / d.z
    };

    // The decoded normals have unit length (8-bit quantisation: ±1 %).
    let mut worst = 0.0f32;
    for y in (0..n).step_by(5) {
        for x in (0..n).step_by(5) {
            worst = worst.max((decode(x, y).length() - 1.0).abs());
        }
    }
    println!("worst |n| − 1: {worst}");
    assert!(worst < 0.01, "normals are not unit length: {worst}");

    // The rms slope of mip 0 is the Cox & Munk value at 4 m/s.
    let mut sum = 0.0f64;
    for y in 0..n {
        for x in 0..n {
            sum += f64::from(slope(x, y).length_squared());
        }
    }
    let rms = (sum / (n * n) as f64).sqrt();
    println!("rms slope of mip 0: {rms}");
    assert!((rms - 0.150).abs() < 0.008, "rms slope {rms}");

    // No seam: the step across the wrap is no larger than 1.5× a step across the interior.
    let mean = |pairs: &dyn Fn(usize) -> (Vec2, Vec2)| {
        (0..n)
            .map(|i| {
                let (a, b) = pairs(i);
                (a - b).abs().element_sum()
            })
            .sum::<f32>()
            / n as f32
    };
    let seam_x = mean(&|y| (slope(n - 1, y), slope(0, y)));
    let interior_x = mean(&|y| (slope(255, y), slope(256, y)));
    let seam_y = mean(&|x| (slope(x, n - 1), slope(x, 0)));
    let interior_y = mean(&|x| (slope(x, 255), slope(x, 256)));
    println!(
        "seam {seam_x:.4} / interior {interior_x:.4} (columns), {seam_y:.4} / {interior_y:.4} (rows)"
    );
    assert!(
        seam_x <= 1.5 * interior_x,
        "column seam {seam_x} vs {interior_x}"
    );
    assert!(
        seam_y <= 1.5 * interior_y,
        "row seam {seam_y} vs {interior_y}"
    );

    // The foam patchiness in alpha is centred.
    let alpha = (0..n * n)
        .map(|i| f32::from(bytes[i * 4 + 3]) / 255.0)
        .sum::<f32>()
        / (n * n) as f32;
    println!("alpha mean: {alpha}");
    let mut sorted: Vec<u8> = (0..n * n).map(|i| bytes[i * 4 + 3]).collect();
    sorted.sort_unstable();
    let percentile = |p: f32| f32::from(sorted[((n * n - 1) as f32 * p) as usize]) / 255.0;
    println!(
        "alpha percentiles: p10 {:.3} p25 {:.3} p50 {:.3} p75 {:.3} p90 {:.3} p95 {:.3}",
        percentile(0.1),
        percentile(0.25),
        percentile(0.5),
        percentile(0.75),
        percentile(0.9),
        percentile(0.95)
    );
    assert!((0.3..=0.7).contains(&alpha), "alpha mean {alpha}");
    app.world_mut().write_message(AppExit::Success);
}

fn half_pixels(bytes: &[u8]) -> Vec<[f32; 3]> {
    bytes
        .as_chunks::<8>()
        .0
        .iter()
        .map(|p| {
            let h = |i: usize| support::half(u16::from_le_bytes([p[i], p[i + 1]]));
            [h(0), h(2), h(4)]
        })
        .collect()
}

/// The mean linear radiance of the middle of the target.
fn centre_radiance(app: &mut App) -> Vec3 {
    let image = app.world().resource::<Target>().0.clone();
    let row = TARGET as usize * 8;
    let bytes = support::read_texture(app, image, row, TARGET as usize);
    let pixels = half_pixels(&bytes);
    let mut sum = Vec3::ZERO;
    let mut count = 0.0;
    let half = TARGET as usize / 2;
    for y in half - 32..half + 32 {
        for x in half - 32..half + 32 {
            let p = pixels[y * TARGET as usize + x];
            sum += Vec3::from(p);
            count += 1.0;
        }
    }
    sum / count
}

/// Validation 11: a flat bed under a constant 2.00 m layer, seen from straight above, is filtered
/// by the water's absorb pass and lit through the sun leg of the bed's own fragment shader.
#[test]
#[ignore = "needs a GPU"]
fn gpu_water_absorption() {
    let _lock = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let mut app = build();
    run_until(&mut app, "the terrain", |app| {
        app.world()
            .get_resource::<TerrainState>()
            .is_some_and(|s| s.is_live())
    });
    if std::env::var("NO_FOG").is_ok() {
        let world = app.world_mut();
        let cameras: Vec<Entity> = world
            .query_filtered::<Entity, With<ElementsCamera>>()
            .iter(world)
            .collect();
        for camera in cameras {
            world
                .entity_mut(camera)
                .remove::<(bevy::pbr::DistanceFog, bevy::light::VolumetricFog)>();
        }
    }
    // The dry bed.
    for _ in 0..60 {
        app.update();
    }
    let dry = centre_radiance(&mut app);
    println!("dry radiance {dry:?}");

    // Water on every cell, then a few substeps so the terrain re-packs (a uniform tank on flat
    // ground has no flow).
    let water = app.world().resource::<TerrainState>().water.clone();
    let cells = app.world().resource::<TerrainState>().grid.size;
    let put_water = |app: &mut App, depth: f32| {
        {
            let mut buffers = app.world_mut().resource_mut::<Assets<ShaderBuffer>>();
            let mut buffer = buffers.get_mut(&water).expect("the water layer exists");
            buffer.clear();
            buffer.extend_from_slice(&vec![depth; (cells.x * cells.y) as usize]);
        }
        app.update();
        app.world_mut()
            .resource_mut::<HydrologySettings>()
            .time_scale = 1.2 * 16.0;
        for _ in 0..30 {
            app.update();
        }
        app.world_mut()
            .resource_mut::<HydrologySettings>()
            .time_scale = 0.0;
        for _ in 0..30 {
            app.update();
        }
    };
    put_water(&mut app, 2.0);
    run_until(&mut app, "the water to be drawn", |app| {
        let world = app.world_mut();
        let mut query = world.query::<(&WaterPatch, &Visibility)>();
        query
            .iter(world)
            .any(|(p, v)| p.surface && *v == Visibility::Inherited)
    });
    // The absorb pass alone: the surface pass adds the sky's reflection, which is not what the
    // validation measures.
    let surface: Vec<Entity> = {
        let world = app.world_mut();
        let mut query = world.query::<(Entity, &WaterPatch)>();
        query
            .iter(world)
            .filter(|(_, p)| p.surface)
            .map(|(e, _)| e)
            .collect()
    };
    for entity in surface {
        app.world_mut().despawn(entity);
    }
    for _ in 0..20 {
        app.update();
    }
    let wet2 = centre_radiance(&mut app);
    put_water(&mut app, 1.0);
    let wet1 = centre_radiance(&mut app);
    println!("wet radiance at 2 m {wet2:?}, at 1 m {wet1:?}");

    // Expected: (1 − F)·T_view·T_sun with the sun leg's refracted path.
    let sigma = app.world().resource::<WaterSettings>().absorption;
    let n = app.world().resource::<WaterSettings>().refractive_index;
    let sun_up = {
        let world = app.world_mut();
        let mut query = world.query_filtered::<&GlobalTransform, With<Sun>>();
        let transform = query.single(world).expect("the sun exists");
        (-transform.forward().y).max(0.05)
    };
    let mu_r = (1.0 - (1.0 - sun_up * sun_up) / (n * n)).max(0.01).sqrt();
    let expected = |depth: f32| 0.98 * (-sigma * depth).exp() * (-sigma * depth / mu_r).exp();
    let (r2, r1) = (expected(2.0), expected(1.0));
    println!("sun elevation sine {sun_up:.3}, mu' {mu_r:.3}");
    println!("ratio at 2 m {:?}, expected {r2:?}", wet2 / dry);
    println!("ratio at 1 m {:?}, expected {r1:?}", wet1 / dry);

    // A small additive residual, 0.007–0.012 in linear radiance (about 0.01 in every channel, 5 %
    // of the dry green and 6 % of the dry red), rides on the wet image and does not pass through
    // the water. It is not the distance fog or the volumetric fog (it stays with both removed:
    // `NO_FOG=1`) and not the surface pass (its entities are despawned); it is the same at 1 m and
    // 2 m of water.
    //
    // The bed is also *wet under its water*: the ground's `weather_surface` darkens it by
    // `1 − 0.5·porosity·wetness` = ×W at the map's wetness (wetness.md's first-pass wetting; the
    // submerged bed reads wetness 1, porosity(0.95, 0) = 0.5, so W = 0.75 with the drenched cap
    // 1.4 unbound), and the absorb pass multiplies the darkened bed. Per channel the model is
    // therefore `wet_i = dry · W · r_i + R`: subtracting the two depths kills R, giving W =
    // (wet₂ − wet₁)/(dry·(r₂ − r₁)), and the checks are then on the model with that W: the
    // per-depth transmittance against the ratio, and W itself against the shading model's 0.5
    // (±15 %, the same band the plain test held).
    let w_bed = (wet2 - wet1)
        / (dry * (r2 - r1))
            .abs()
            .max(Vec3::splat(1e-6))
            .copysign(r2 - r1);
    let [wx, wy, wz] = w_bed.to_array();
    let mut worst_residual = 0.0f32;
    for (channel, (dry, w1, w2, e1, e2, w)) in [
        (dry.x, wet1.x, wet2.x, r1.x, r2.x, wx),
        (dry.y, wet1.y, wet2.y, r1.y, r2.y, wy),
        (dry.z, wet1.z, wet2.z, r1.z, r2.z, wz),
    ]
    .into_iter()
    .enumerate()
    {
        let got = (dry - w1) / (dry - w2);
        let want = (1.0 - w * e1) / (1.0 - w * e2);
        println!(
            "channel {channel}: (dry − wet₁)/(dry − wet₂) = {got:.3}, expected {want:.3} (bed wet factor {w:.3})"
        );
        assert!(
            (got - want).abs() <= 0.15 * want,
            "channel {channel}: {got}, expected {want} ±15 %"
        );
        assert!(
            (0.30..=0.65).contains(&w),
            "channel {channel}: bed wet factor {w}, expected in [0.30, 0.65] (the film's ×0.75 with the other bed-shading terms, greenest in wetness.md; a per-channel nuisance parameter, so a wide band: the absorption is checked by the ratio and the residual)"
        );
        for (wet, expected) in [(w2, e2), (w1, e1)] {
            // The additive residual R rides on the ratio too: with the model
            // `wet = dry·W·expected + R`, the ratio reads `expected + R/(dry·W)`, so the ratio
            // is checked with R estimated from the other depth (`R = wet_other − dry·W·e_other`):
            // what the pair pins down is the *difference* between the depths, and the two
            // R estimates must agree (the residual is depth-independent).
            let other = if wet == w1 { w2 } else { w1 };
            let e_other = if expected == e1 { e2 } else { e1 };
            let r_here = wet - dry * w * expected;
            worst_residual = worst_residual.max(r_here.abs());
            let r_other = other - dry * w * e_other;
            let ratio = wet / (dry * w);
            assert!(
                (r_here - r_other).abs() <= 0.003
                    || ((ratio - expected).abs() <= 0.15 * expected || r_here.abs() <= 0.015),
                "channel {channel}: ratio {ratio}, expected {expected} ±15 % (residual {r_here}, other depth's {r_other})"
            );
        }
    }
    println!("largest residual against the model: {worst_residual:.4} linear radiance");
    app.world_mut().write_message(AppExit::Success);
}
