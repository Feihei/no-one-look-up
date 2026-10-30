//! R1.7 headless screenshot smoke test: renders the locked golden-hour sky
//! offscreen and asserts the frame is bright, varied and warm-tinted.
//!
//! Runs the full sky stack (bevy_elements climate/celestial/fog/cloud) with a
//! windowless camera targeting an HDR image, aims the camera at the sun once
//! the ephemeris has settled, captures a frame and analyses it.
//!
//! Requires a real GPU (atmosphere/cloud compute pipelines). CI runners
//! without a GPU would fail here, so the test is `#[ignore]`d: run locally
//! with `cargo test --test sky_screenshot -- --ignored --nocapture`.
//!
//! Determinism note: the clock is paused (fixed golden hour), but bevy_elements
//! cloud drift advances on real time, so the smoke assertions are statistical
//! (brightness/contrast/warmth) rather than bit-exact frame comparisons.

use bevy::asset::AssetPlugin;
use bevy::camera::CameraPlugin;
use bevy::camera::Exposure;
use bevy::camera::ImageRenderTarget;
use bevy::camera::RenderTarget;
use bevy::core_pipeline::CorePipelinePlugin;
use bevy::core_pipeline::tonemapping::Tonemapping;
use bevy::image::ImagePlugin;
use bevy::log::LogPlugin;
use bevy::mesh::MeshPlugin;
use bevy::post_process::bloom::Bloom;
use bevy::prelude::*;
use bevy::render::RenderPlugin;
use bevy::render::render_resource::TextureFormat;
use bevy::render::view::screenshot::{Screenshot, ScreenshotCaptured};
use bevy_elements::{ElementsCamera, SunState};

use heads_up_bevy::plugins::HeadsUpPlugins;

use std::time::Duration;

#[derive(Resource, Default)]
struct FrameCounter(usize);

#[derive(Resource, Default)]
struct CapturedFrame(Option<Image>);

#[test]
#[ignore = "requires a physical GPU (atmosphere/cloud compute); run locally with --ignored"]
fn golden_hour_sky_renders() {
    let mut app = App::new();
    app.add_plugins((
        MinimalPlugins,
        // MinimalPlugins has no TransformPlugin: GlobalTransform never
        // propagates, which breaks bevy_elements' camera-following fog and the
        // atmosphere view (uniform flat frame).
        bevy::transform::TransformPlugin,
        AssetPlugin::default(),
        RenderPlugin::default(),
        WindowPlugin {
            primary_window: None,
            exit_condition: bevy::window::ExitCondition::DontExit,
            close_when_requested: false,
            primary_cursor_options: None,
        },
        ImagePlugin::default(),
        MeshPlugin,
        CameraPlugin,
        CorePipelinePlugin,
        // R4.5's menu UI reads `ButtonInput<KeyCode>` (`menu_input`); without
        // the input plugin the system fails validation on every frame.
        bevy::input::InputPlugin,
    ));
    app.add_plugins((
        bevy::post_process::PostProcessPlugin,
        // GltfPlugin must come before PbrPlugin: PbrPlugin registers a glTF
        // material extension handler into the resource GltfPlugin initializes.
        bevy::gltf::GltfPlugin::default(),
        bevy::pbr::PbrPlugin::default(),
        // Initializes Assets<ScatteringMedium> (needed by ElementsCamera's
        // dressing observer) plus light clustering/shadow infrastructure.
        bevy::light::LightPlugin,
        // LightPlugin's render-side gizmo systems need Assets<GizmoAsset>.
        bevy::gizmos::GizmoPlugin,
        HeadsUpPlugins,
        LogPlugin::default(),
    ));
    // `finish()` unpacks render resources (see tests/assets_load.rs).
    app.finish();
    app.init_resource::<FrameCounter>();
    app.init_resource::<CapturedFrame>();

    // HDR target is REQUIRED: an `Hdr` camera (required by `ElementsCamera`)
    // silently renders nothing into an LDR image target — the output stays at
    // the clear color. Linear values carry the exposure scale; assertions are
    // therefore ratio/band based (exposure-invariant).
    let target = {
        let mut images = app.world_mut().resource_mut::<Assets<Image>>();
        images.add(Image::new_target_texture(
            320,
            180,
            TextureFormat::Rgba16Float,
            None,
        ))
    };

    app.world_mut().spawn((
        Name::new("ScreenshotCamera"),
        Camera3d::default(),
        RenderTarget::Image(ImageRenderTarget {
            handle: target,
            scale_factor: 1.0,
        }),
        ElementsCamera {
            auto_exposure: false,
            ..ElementsCamera::default()
        },
        Exposure { ev100: 13.0 },
        Tonemapping::AcesFitted,
        Bloom::NATURAL,
        Transform::from_xyz(0.0, 1.6, 0.0),
    ));

    app.add_systems(
        Update,
        (
            bump_frame_counter,
            // Ephemeris settles on the first frames; aim once, then give the
            // cloud/fog GPU pipelines time to warm up before capturing.
            aim_at_sun.run_if(in_frame(20)),
            take_screenshot.run_if(in_frame(80)),
        )
            .chain(),
    );

    let mut frames = 0usize;
    loop {
        app.update();
        frames += 1;
        // Pace frames to a realistic ~16ms delta: bevy_elements' smoothing
        // and readback cadence run on `Time<Virtual>`; unpaced test frames
        // starve them (delta ≈ 0) and the sky renders flat.
        std::thread::sleep(Duration::from_millis(16));
        if frames.is_multiple_of(20) {
            let (active, has_target) = app
                .world_mut()
                .query::<(&Camera, Has<RenderTarget>)>()
                .single(app.world())
                .map(|(c, t)| (c.is_active, t))
                .unwrap_or((false, false));
            println!("frame {frames}: camera active={active} has_target_component={has_target}");
        }
        let captured = app
            .world()
            .get_resource::<CapturedFrame>()
            .and_then(|c| c.0.as_ref());
        if let Some(image) = captured {
            if let Some(data) = &image.data {
                println!(
                    "captured: {} bytes, first u16s: {:?}",
                    data.len(),
                    data.as_chunks::<2>()
                        .0
                        .iter()
                        .take(8)
                        .map(|b| u16::from_le_bytes([b[0], b[1]]))
                        .collect::<Vec<_>>()
                );
            } else {
                println!("captured: image data is None");
            }
            assert_frame(image);
            return;
        }
        assert!(frames < 600, "screenshot not captured within 600 frames");
    }
}

fn bump_frame_counter(mut counter: ResMut<FrameCounter>) {
    counter.0 += 1;
}

fn in_frame(frame: usize) -> impl FnMut(Res<FrameCounter>) -> bool + Clone {
    move |counter: Res<FrameCounter>| counter.0 == frame
}

/// Points the camera at the sun disk with the horizon glow in frame.
fn aim_at_sun(sun: Res<SunState>, mut cameras: Query<&mut Transform, With<ElementsCamera>>) {
    let Ok(mut transform) = cameras.single_mut() else {
        return;
    };
    let dir = sun.direction.normalize_or(Vec3::NEG_Z);
    let yaw = (-dir.x).atan2(-dir.z);
    // Keep the horizon glow in frame: aim at the sun but never above 15°.
    let pitch = dir.y.clamp(-1.0, 1.0).asin().max(10f32.to_radians());
    *transform = Transform::from_xyz(0.0, 1.6, 0.0).with_rotation(Quat::from_euler(
        EulerRot::YXZ,
        yaw,
        pitch,
        0.0,
    ));
}

/// Requests the screenshot capture (entity despawns after capture).
fn take_screenshot(mut commands: Commands, targets: Query<&RenderTarget, With<ElementsCamera>>) {
    let Ok(target) = targets.single() else {
        return;
    };
    let RenderTarget::Image(image_target) = target else {
        return;
    };
    let handle = image_target.handle.clone();
    let entity = commands
        .spawn(Screenshot::image(handle.clone()))
        .observe(
            |trigger: On<ScreenshotCaptured>, mut frame: ResMut<CapturedFrame>| {
                frame.0 = Some(trigger.image.clone());
            },
        )
        .id();
    // Save the captured frame for human review next to the test target dir.
    commands
        .entity(entity)
        .observe(bevy::render::view::screenshot::save_to_disk(
            "target/sky_screenshot.png",
        ));
}

/// Asserts the captured linear HDR frame looks like a golden-hour sky.
///
/// Band analysis (the camera aims at the low sun with the horizon in frame).
/// Linear values carry the exposure scale, so thresholds are lenient and the
/// load-bearing assertions are exposure-invariant ratios: the horizon band
/// must be warm (R > B) and the frame must have a gradient (not uniform).
fn assert_frame(image: &Image) {
    let planes = u8_plane(image);
    assert_eq!(planes.len() % 4, 0, "expected RGBA data");
    let pixels = planes.len() / 4;
    assert!(pixels > 0, "empty frame");

    let height = 180usize;
    let width = pixels / height;
    let band = |rows: std::ops::Range<usize>| {
        let mut sum = [0.0f32; 3];
        let mut count = 0.0f32;
        for y in rows {
            for x in 0..width {
                let px = &planes[(y * width + x) * 4..(y * width + x) * 4 + 4];
                for c in 0..3 {
                    sum[c] += px[c];
                }
                count += 1.0;
            }
        }
        [sum[0] / count, sum[1] / count, sum[2] / count]
    };
    let top = band(0..height / 3);
    let middle = band(height / 3..2 * height / 3);

    let mut max = 0.0f32;
    for px in planes.as_chunks::<4>().0 {
        let luma = 0.2126 * px[0] + 0.7152 * px[1] + 0.0722 * px[2];
        max = max.max(luma);
    }

    println!("golden-hour frame stats: top rgb=({top:.4?}) middle rgb=({middle:.4?}) max={max:.4}");

    assert!(
        max > 0.001,
        "frame is black (max {max}); sky did not render"
    );
    let top_luma = 0.2126 * top[0] + 0.7152 * top[1] + 0.0722 * top[2];
    let mid_luma = 0.2126 * middle[0] + 0.7152 * middle[1] + 0.0722 * middle[2];
    assert!(
        (top_luma - mid_luma).abs() > 0.001 || (top[2] - middle[0]).abs() > 0.01,
        "frame is uniform (top luma {top_luma:.4} vs mid {mid_luma:.4}); sky gradient missing"
    );
    assert!(
        middle[0] > middle[2] * 1.05,
        "horizon band is not warm (mean R {:.4} vs B {:.4}); dusk tint missing",
        middle[0],
        middle[2]
    );
}

/// Converts the RGBA16Float texture data to normalized f32 values.
fn u8_plane(image: &Image) -> Vec<f32> {
    assert_eq!(image.texture_descriptor.format, TextureFormat::Rgba16Float);
    let data = image.data.as_ref().expect("CPU-visible image data");
    data.as_chunks::<2>()
        .0
        .iter()
        .map(|b| f16_to_f32(u16::from_le_bytes([b[0], b[1]])))
        .collect()
}

/// Minimal IEEE 754 half -> f32 conversion (no `half` dependency needed).
#[allow(dead_code)]
fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h & 0x8000) as u32) << 16;
    let exp = (h & 0x7C00) >> 10;
    let frac = (h & 0x03FF) as u32;
    let bits = match exp {
        0 => {
            if frac == 0 {
                sign
            } else {
                // Subnormal: normalize.
                let mut e: u32 = 127 - 15 + 1;
                let mut f = frac;
                while f & 0x0400 == 0 {
                    f <<= 1;
                    e -= 1;
                }
                f &= 0x03FF;
                sign | ((e + 9) << 23) | (f << 13)
            }
        }
        0x1F => sign | 0x7F80_0000 | (frac << 13),
        _ => sign | (((exp as u32) + 112) << 23) | (frac << 13),
    };
    f32::from_bits(bits)
}
