//! Dev tool: render an orthographic top-down view of the generated world
//! layout (debug gizmos on) and save it as a PNG.
//!
//! Usage (requires a GPU; same plugin set as the sky screenshot test):
//!
//! ```text
//! cargo test --test top_view -- --ignored --nocapture
//! # optional: SEED=0x1234 cargo test --test top_view -- --ignored --nocapture
//! #           WORLD_TOP_VIEW_OUT=target/my.png ...
//! ```
//!
//! Output: `target/world_top_view.png` (default). The camera is orthographic,
//! looking straight down from `CAM_HEIGHT`, framed to cover the whole
//! `WORLD_SIZE` world; debug gizmos are forced on for the capture.

use std::time::Duration;

use bevy::asset::AssetPlugin;
use bevy::camera::{CameraPlugin, Exposure, ImageRenderTarget, RenderTarget};
use bevy::core_pipeline::CorePipelinePlugin;
use bevy::core_pipeline::tonemapping::Tonemapping;
use bevy::image::ImagePlugin;
use bevy::log::LogPlugin;
use bevy::mesh::MeshPlugin;
use bevy::prelude::*;
use bevy::render::RenderPlugin;
use bevy::render::render_resource::TextureFormat;
use bevy::render::view::screenshot::{Screenshot, ScreenshotCaptured};
use bevy_elements::ElementsCamera;

use heads_up_bevy::plugins::HeadsUpPlugins;
use heads_up_bevy::world::WORLD_SIZE;
use heads_up_bevy::world::debug_viz::DebugVizEnabled;
use heads_up_bevy::world::path_network::HeadsUpSeed;

const IMAGE_SIZE: u32 = 1600;
/// Orthographic camera height above the ground plane (only the projection
/// extent matters for an orthographic view).
const CAM_HEIGHT: f32 = 500.0;

#[derive(Resource, Default)]
struct FrameCounter(usize);

#[derive(Resource, Default)]
struct CapturedFrame(Option<Image>);

fn main() {
    let seed = std::env::var("SEED")
        .ok()
        .and_then(|raw| {
            let raw = raw.trim();
            match raw.strip_prefix("0x") {
                Some(hex) => u64::from_str_radix(hex, 16).ok(),
                None => raw.parse().ok(),
            }
        })
        .unwrap_or(heads_up_bevy::world::chunks::DEFAULT_SEED);
    let out = std::env::var("WORLD_TOP_VIEW_OUT")
        .unwrap_or_else(|_| "target/world_top_view.png".to_string());

    let mut app = App::new();
    app.add_plugins((
        MinimalPlugins,
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
    ));
    app.add_plugins((
        bevy::post_process::PostProcessPlugin,
        bevy::world_serialization::WorldSerializationPlugin,
        bevy::gltf::GltfPlugin::default(),
        bevy::pbr::PbrPlugin::default(),
        bevy::light::LightPlugin,
        bevy::gizmos::GizmoPlugin,
        HeadsUpPlugins,
        LogPlugin::default(),
    ));
    app.finish();
    // MinimalPlugins carries no keyboard input resource, but the world
    // plugin's walk/streaming systems require it (dev-only stand-in).
    app.init_resource::<ButtonInput<KeyCode>>();
    // The glTF loader registers AnimationClip handles for animated models,
    // but this minimal stack carries no AnimationPlugin: init the asset type
    // so loading the trees does not panic in the loader.
    app.init_asset::<bevy::animation::AnimationClip>();
    app.insert_resource(HeadsUpSeed(seed));
    app.insert_resource(DebugVizEnabled(true));
    app.init_resource::<FrameCounter>();
    app.init_resource::<CapturedFrame>();

    let target = {
        let mut images = app.world_mut().resource_mut::<Assets<Image>>();
        images.add(Image::new_target_texture(
            IMAGE_SIZE,
            IMAGE_SIZE,
            TextureFormat::Rgba16Float,
            None,
        ))
    };

    // Orthographic, straight down, north (-Z) up: world (0,0) is top-left.
    let scale = WORLD_SIZE;
    app.world_mut().spawn((
        Name::new("TopViewCamera"),
        Camera3d::default(),
        // `Infinity` depth range is fine for a single ortho frame.
        Projection::Orthographic(OrthographicProjection {
            scaling_mode: bevy::camera::ScalingMode::FixedVertical {
                viewport_height: scale,
            },
            ..OrthographicProjection::default_3d()
        }),
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
        Transform::from_xyz(WORLD_SIZE * 0.5, CAM_HEIGHT, WORLD_SIZE * 0.5).looking_at(
            Vec3::new(WORLD_SIZE * 0.5, 0.0, WORLD_SIZE * 0.5),
            Vec3::NEG_Z,
        ),
    ));

    app.add_systems(
        Update,
        (
            bump_frame_counter,
            disable_camera_controls,
            take_screenshot_at(40).run_if(in_frame(40)),
        )
            .chain(),
    );

    let mut frames = 0usize;
    loop {
        app.update();
        frames += 1;
        std::thread::sleep(Duration::from_millis(16));
        if frames.is_multiple_of(20) {
            println!("frame {frames}");
        }
        if app.world().resource::<CapturedFrame>().0.is_some() {
            // Give the async disk write a moment to flush, then exit.
            for _ in 0..30 {
                app.update();
                std::thread::sleep(Duration::from_millis(16));
            }
            println!("top view saved to {out} (seed {seed:#x})");
            return;
        }
        assert!(
            frames < 400,
            "top view frame not captured within 400 frames"
        );
    }
}

fn bump_frame_counter(mut counter: ResMut<FrameCounter>) {
    counter.0 += 1;
}

fn in_frame(frame: usize) -> impl FnMut(Res<FrameCounter>) -> bool + Clone {
    move |counter: Res<FrameCounter>| counter.0 == frame
}

/// The debug walk system would otherwise move the camera (no input, so it
/// stays put, but keep the frame stable explicitly by clearing input).
fn disable_camera_controls(mut input: ResMut<ButtonInput<KeyCode>>) {
    input.clear();
}

fn take_screenshot_at(_frame: usize) -> impl FnMut(Commands, Query<&RenderTarget, With<Camera3d>>) {
    move |mut commands: Commands, targets: Query<&RenderTarget, With<Camera3d>>| {
        let Ok(target) = targets.single() else {
            return;
        };
        let RenderTarget::Image(image_target) = target else {
            return;
        };
        let handle = image_target.handle.clone();
        let out = std::env::var("WORLD_TOP_VIEW_OUT")
            .unwrap_or_else(|_| "target/world_top_view.png".to_string());
        let entity = commands
            .spawn(Screenshot::image(handle.clone()))
            .observe(
                |trigger: On<ScreenshotCaptured>, mut frame: ResMut<CapturedFrame>| {
                    frame.0 = Some(trigger.image.clone());
                },
            )
            .id();
        commands
            .entity(entity)
            .observe(bevy::render::view::screenshot::save_to_disk(out));
    }
}
