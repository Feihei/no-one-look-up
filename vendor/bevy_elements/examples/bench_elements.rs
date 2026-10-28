//! GPU cost of every domain, per pass, printed every 120 frames.
//!
//! `cargo run --release --example bench_elements`
//!
//! Uses Bevy's `RenderDiagnosticsPlugin`: GPU timestamps on Vulkan and DX12, CPU-side timing only
//! elsewhere. Set `ELEMENTS_QUALITY=low|medium|high|ultra` (default high),
//! `ELEMENTS_RESOLUTION=WxH` (default 1920x1080), `ELEMENTS_PRESET=Thunderstorm` (default `Fog`;
//! the precipitation passes only run in a wet preset) and `ELEMENTS_BLOCKERS=256` (spawn that
//! many precipitation blockers around the camera). `ELEMENTS_TERRAIN=hills|mountains|dunes|lake|none`
//! (default `hills`): a 2 km terrain at the quality's full simulation resolution with a 50 m³/s
//! water source, so hydrology and erosion run their full substeps every frame
//! (`elements_hydrology`) and the terrain repacks its layers every frame (`elements_terrain`); the
//! snow chain (`elements_snow`: source, drift and two diffusion passes per step, with
//! `ELEMENTS_PRESET=Blizzard` for the wind that drifts), the wetness update with the cover pack and
//! readback (`elements_wetness`) and the shelter mask (`elements_shelter`, once per change of the
//! blockers) run at most 20 times a second;
//! `none` is the flat test scene with neither, and `lake` a 256 m plain with a round basin that a
//! 200 m³/s source fills, viewed from 45 m up so the water covers half the frame
//! (`main_transparent_pass_3d` then holds both water passes; `ELEMENTS_WATER=off` leaves the
//! water rendering out to subtract it). `dunes` runs the aeolian transport
//! (`elements_aeolian_*`, 20 steps a second) in a forced `DustStorm`, unless `ELEMENTS_PRESET`
//! says otherwise. Bevy's main and depth passes the scene is drawn in
//! are printed on their own line.
//!
//! The timed passes are compute and do not depend on the resolution, but the GPU's clocks do: a
//! lightly loaded GPU idles in a low power state and every pass reads several times slower. Check
//! the clocks while it runs (`nvidia-smi --query-gpu=pstate,clocks.gr --format=csv`) and raise the
//! resolution until the GPU holds its full clock; the README numbers were taken that way.

mod common;

use bevy::diagnostic::{DiagnosticPath, DiagnosticsStore};
use bevy::prelude::*;
use bevy::render::diagnostic::RenderDiagnosticsPlugin;
use bevy::window::{PresentMode, WindowResolution};
use bevy_elements::*;
use common::{ExampleHarness, camera, stand_on_ground, test_scene};

/// Frames before the per-pass minimum starts counting.
const WARMUP_FRAMES: u32 = 240;

/// Share of a 60 Hz frame that carries a transport step (20 steps a second).
const AEOLIAN_FRAME_SHARE: f64 = 20.0 / 60.0;

/// The crate's GPU passes, by diagnostic span name.
const PASSES: [&str; 20] = [
    "elements_wind",
    "elements_fog",
    "elements_clouds_noise",
    "elements_clouds_weather",
    "elements_clouds_shadow",
    "elements_clouds_march",
    "elements_clouds_resolve",
    "elements_clouds_composite",
    "elements_precip_sim",
    "elements_precip_draw",
    "elements_hydrology",
    "elements_snow",
    "elements_wetness",
    "elements_shelter",
    "elements_aeolian_capacity",
    "elements_aeolian_flux",
    "elements_aeolian_move",
    "elements_aeolian_avalanche",
    "elements_terrain",
    "elements_water_normals",
];

/// Bevy's own passes the terrain is drawn in, printed on a separate line.
const SCENE_PASSES: [&str; 3] = [
    "main_opaque_pass_3d",
    "early prepass",
    "main_transparent_pass_3d",
];

fn main() {
    let (width, height) = std::env::var("ELEMENTS_RESOLUTION")
        .ok()
        .and_then(|r| {
            let (w, h) = r.split_once('x')?;
            Some((w.parse().ok()?, h.parse().ok()?))
        })
        .unwrap_or((1920, 1080));
    let quality = match std::env::var("ELEMENTS_QUALITY").as_deref() {
        Ok("low") => ElementsQuality::Low,
        Ok("medium") => ElementsQuality::Medium,
        Ok("ultra") => ElementsQuality::Ultra,
        _ => ElementsQuality::High,
    };
    // `ELEMENTS_WATER=off` leaves the water rendering out, for the cost of the rest of the scene.
    let elements = if std::env::var("ELEMENTS_WATER").as_deref() == Ok("off") {
        ElementsPlugins.build().disable::<WaterPlugin>()
    } else {
        ElementsPlugins.build()
    };
    App::new()
        .add_plugins((
            DefaultPlugins.set(WindowPlugin {
                primary_window: Some(Window {
                    resolution: WindowResolution::new(width, height),
                    present_mode: PresentMode::AutoNoVsync,
                    ..default()
                }),
                ..default()
            }),
            RenderDiagnosticsPlugin,
            elements,
            ExampleHarness,
        ))
        .insert_resource(quality)
        // Far more simulated time than a frame can hold: every frame runs the quality's maximum.
        .insert_resource(HydrologySettings {
            time_scale: 1000.0,
            ..default()
        })
        .insert_resource(
            WorldClock::new(CivilDate::new(2026, 6, 21))
                .at_hour(16.0)
                .with_time_scale(60.0),
        )
        .add_systems(Startup, setup)
        .add_systems(Update, report)
        .run();
}

/// A round basin 6 m deep and 100 m across in a 256 m plain, as a height map in metres over
/// `LAKE_RANGE_M`.
const LAKE_RANGE_M: f32 = 12.0;

fn lake_map() -> Image {
    use bevy::asset::RenderAssetUsages;
    use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};
    const CELLS: u32 = 256;
    let mut data = Vec::with_capacity((CELLS * CELLS) as usize * 4);
    for j in 0..CELLS {
        for i in 0..CELLS {
            let x = i as f32 + 0.5 - CELLS as f32 * 0.5;
            let z = j as f32 + 0.5 - CELLS as f32 * 0.5;
            let r = ((x * x + z * z).sqrt() / 50.0).min(1.0);
            let bowl = 0.5 * (1.0 + (std::f32::consts::PI * r).cos());
            data.extend_from_slice(&((8.0 - 6.0 * bowl) / LAKE_RANGE_M).to_le_bytes());
        }
    }
    Image::new(
        Extent3d {
            width: CELLS,
            height: CELLS,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        data,
        TextureFormat::R32Float,
        RenderAssetUsages::default(),
    )
}

fn setup(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut director: ResMut<WeatherDirector>,
    quality: Res<ElementsQuality>,
) {
    let lake = std::env::var("ELEMENTS_TERRAIN").as_deref() == Ok("lake");
    let eye = commands
        .spawn(if lake {
            // The lake fills half the frame.
            camera(Vec3::new(-75.0, 45.0, 0.0), Vec3::new(0.0, 0.0, 0.0))
        } else {
            camera(
                Vec3::new(0.0, 30.0, 0.0),
                // Looking 20° above the horizon: about half the frame is cloud.
                Vec3::new(150.0, 91.9, 80.0),
            )
        })
        .id();
    // A terrain at the quality's full simulation resolution with a river's worth of water, so
    // hydrology and erosion run their full substeps every frame; `none` is the flat test scene.
    if lake {
        // A basin of 23 500 m³ and a source of 200 m³/s: full after 120 s of simulated time, which
        // the full substeps of every frame reach in a few hundred frames.
        commands.spawn(
            Terrain::from_heightmap(images.add(lake_map()), 0.0..LAKE_RANGE_M)
                .with_extent(Vec2::splat(256.0))
                .with_resolution(UVec2::splat(256)),
        );
        commands.spawn((
            WaterSource {
                rate_m3_s: 200.0,
                radius_m: 5.0,
            },
            Transform::from_xyz(0.0, 2.0, 0.0),
        ));
    }
    let terrain = match std::env::var("ELEMENTS_TERRAIN").as_deref() {
        Ok("lake") => None,
        Ok("none") => None,
        Ok("mountains") => Some(ProceduralTerrain::mountains(42)),
        Ok("dunes") => Some(ProceduralTerrain::dunes(42)),
        _ => Some(ProceduralTerrain::hills(42)),
    };
    match terrain {
        Some(preset) => {
            commands.entity(eye).insert(stand_on_ground(30.0));
            commands.spawn(
                Terrain::procedural(preset)
                    .with_resolution(UVec2::splat(quality.terrain_max_resolution())),
            );
            commands.spawn((
                WaterSource {
                    rate_m3_s: 50.0,
                    radius_m: 5.0,
                },
                Transform::from_xyz(100.0, 0.0, 100.0),
            ));
        }
        None if lake => {}
        None => test_scene(&mut commands, &mut meshes, &mut materials),
    }
    // PartlyCloudy is the clouds note's bench scene; `ELEMENTS_PRESET=Thunderstorm` saturates the
    // precipitation particles.
    let preset = std::env::var("ELEMENTS_PRESET")
        .ok()
        .and_then(|name| {
            WeatherPreset::ALL
                .into_iter()
                .find(|p| format!("{p:?}").eq_ignore_ascii_case(&name))
        })
        .unwrap_or(
            if matches!(std::env::var("ELEMENTS_TERRAIN").as_deref(), Ok("dunes")) {
                WeatherPreset::DustStorm
            } else {
                WeatherPreset::PartlyCloudy
            },
        );
    director.force(preset, std::time::Duration::ZERO);
    // Roofs scattered through the particle box: each particle tests every uploaded blocker.
    let blockers: usize = std::env::var("ELEMENTS_BLOCKERS")
        .ok()
        .and_then(|n| n.parse().ok())
        .unwrap_or(0);
    for i in 0..blockers {
        let angle = i as f32 * 2.399_963;
        let radius = 4.0 + 26.0 * ((i as f32 + 0.5) / blockers as f32).sqrt();
        commands.spawn((
            PrecipitationBlocker {
                half_extents: Vec3::new(2.0, 0.1, 2.0),
            },
            Transform::from_xyz(
                angle.cos() * radius,
                3.0 + (i % 7) as f32 * 1.5,
                angle.sin() * radius,
            ),
        ));
    }
}

fn report(
    store: Res<DiagnosticsStore>,
    quality: Res<ElementsQuality>,
    mut frame: Local<u32>,
    mut lowest: Local<[f64; PASSES.len()]>,
    mut lowest_scene: Local<[f64; SCENE_PASSES.len()]>,
) {
    *frame += 1;
    // The minimum over the whole run is the full-clock cost: a pass can only read slower than
    // that, when the GPU has dropped to a low power state; the history window and the average mix
    // both. The first frames are skipped: the hydrology pass runs fewer substeps while the
    // terrain is generated and its pipelines compile, and its minimum would read that.
    if *frame > WARMUP_FRAMES {
        for (slot, pass) in lowest.iter_mut().zip(PASSES) {
            let gpu = DiagnosticPath::new(format!("render/{pass}/elapsed_gpu"));
            if let Some(now) = store.get(&gpu).and_then(|d| d.value()) {
                *slot = if *slot > 0.0 { slot.min(now) } else { now };
            }
        }
    }
    // Bevy's passes are timed with the terrain only once it has spawned and been read back.
    for (slot, pass) in lowest_scene
        .iter_mut()
        .zip(SCENE_PASSES)
        .filter(|_| *frame > 600)
    {
        let gpu = DiagnosticPath::new(format!("render/{pass}/elapsed_gpu"));
        if let Some(now) = store.get(&gpu).and_then(|d| d.value()) {
            *slot = if *slot > 0.0 { slot.min(now) } else { now };
        }
    }
    if !(*frame).is_multiple_of(120) {
        return;
    }
    let mut total = 0.0;
    let mut clouds = 0.0;
    // The transport steps at most 20 times a second: of a 60 Hz frame's cost it is a third.
    let mut aeolian = 0.0;
    let mut line = format!("{:?}:", *quality);
    for pass in PASSES {
        let gpu = DiagnosticPath::new(format!("render/{pass}/elapsed_gpu"));
        let cpu = DiagnosticPath::new(format!("render/{pass}/elapsed_cpu"));
        match store.get(&gpu) {
            Some(d) => {
                let min = lowest[PASSES.iter().position(|p| *p == pass).unwrap_or(0)];
                let mean = d.average().unwrap_or(f64::NAN);
                // The noise volumes are generated once, in the first frames: not a per-frame cost.
                // The terrain repacks every frame while hydrology runs, so it counts.
                if !pass.ends_with("_noise") && !pass.ends_with("_normals") {
                    if pass.starts_with("elements_aeolian") {
                        aeolian += min;
                        total += min * AEOLIAN_FRAME_SHARE;
                    } else {
                        total += min;
                    }
                    if pass.starts_with("elements_clouds") {
                        clouds += min;
                    }
                }
                line += &format!("  {pass} min {min:.3} / mean {mean:.3} ms GPU");
            }
            None => match store.get(&cpu).and_then(|d| d.smoothed()) {
                Some(ms) => line += &format!("  {pass} {ms:.3} ms CPU (no GPU timestamps)"),
                None => line += &format!("  {pass} -"),
            },
        }
    }
    println!("{line}  | total min {total:.3} ms GPU, clouds min {clouds:.3} ms GPU");
    if aeolian > 0.0 {
        println!(
            "aeolian: {aeolian:.3} ms per step (min), {:.3} ms per 60 Hz frame at 20 steps a second",
            aeolian * AEOLIAN_FRAME_SHARE
        );
    }
    let mut scene = String::from("scene:");
    for pass in SCENE_PASSES {
        let gpu = DiagnosticPath::new(format!("render/{pass}/elapsed_gpu"));
        match store.get(&gpu) {
            Some(d) => {
                let min = lowest_scene[SCENE_PASSES.iter().position(|p| *p == pass).unwrap_or(0)];
                scene += &format!(
                    "  {pass} min {min:.3} / mean {:.3} ms GPU",
                    d.average().unwrap_or(f64::NAN)
                );
            }
            None => scene += &format!("  {pass} -"),
        }
    }
    println!("{scene}");
}
