//! R0 asset-chain verification: tree glTF models exported from Blender load
//! through Bevy's AssetServer in a windowless headless app.
//!
//! Assets: `assets/models/*.glb` exported from the original Godot project's
//! `.blend` files (aspen_1, tree_2, tree_3).
//!
//! Note: `RenderPlugin` is required even for pure asset loads — `bevy_image`
//! only pre-registers the ImageLoader and the real loader is registered by the
//! renderer (compressed-texture formats depend on the device). Without it,
//! asset loads hang forever in the `Pending` loader state.

use bevy::asset::{AssetPlugin, AssetServer, Assets, LoadState};
use bevy::gltf::{Gltf, GltfPlugin};
use bevy::image::ImagePlugin;
use bevy::log::LogPlugin;
use bevy::mesh::MeshPlugin;
use bevy::prelude::*;
use bevy::render::RenderPlugin;
use bevy::scene::ScenePlugin;
use std::time::{Duration, Instant};

const TREE_GLBS: [&str; 3] = [
    "models/aspen_1.glb",
    "models/tree_2.glb",
    "models/tree_3.glb",
];

#[test]
fn tree_gltf_assets_load_headless() {
    let mut app = App::new();
    app.add_plugins((
        MinimalPlugins,
        AssetPlugin::default(),
        RenderPlugin::default(),
        // `camera_system` reads `WindowResized`; WindowPlugin initializes the
        // message (primary_window: None avoids creating a real window).
        WindowPlugin {
            primary_window: None,
            exit_condition: bevy::window::ExitCondition::DontExit,
            close_when_requested: false,
            primary_cursor_options: None,
        },
        ImagePlugin::default(),
        MeshPlugin,
        bevy::camera::CameraPlugin,
        bevy::core_pipeline::CorePipelinePlugin,
        bevy::world_serialization::WorldSerializationPlugin,
        ScenePlugin,
        GltfPlugin::default(),
        LogPlugin::default(),
    ));
    // `app.finish()` unpacks the render resources (including DeviceErrorHandler)
    // and registers the real ImageLoader; normally `app.run()` does this, but a
    // test loop calls `update()` directly.
    app.finish();
    // Asset types produced by the GltfLoader that would otherwise come from
    // AnimationPlugin/PbrPlugin (not needed for pure asset loads).
    app.init_asset::<bevy::animation::AnimationClip>();
    app.init_asset::<bevy::pbr::StandardMaterial>();

    let server = app.world().resource::<AssetServer>().clone();
    let handles: Vec<Handle<Gltf>> = TREE_GLBS
        .iter()
        .map(|path| {
            let fs_path = format!("assets/{path}");
            assert!(
                std::path::Path::new(&fs_path).exists(),
                "missing exported asset: {fs_path}"
            );
            // AssetServer paths are relative to the `assets` root.
            server.load(*path)
        })
        .collect();

    let deadline = Instant::now() + Duration::from_secs(60);
    loop {
        app.update();

        let mut all_loaded = true;
        for (path, handle) in TREE_GLBS.iter().zip(&handles) {
            match server.load_state(handle) {
                LoadState::Loaded => {}
                LoadState::Failed(err) => panic!("failed to load {path}: {err}"),
                _ => all_loaded = false,
            }
        }
        if all_loaded {
            // Also confirm the parsed Gltf assets are registered.
            let assets = app.world().resource::<Assets<Gltf>>();
            for handle in &handles {
                assert!(
                    assets.contains(handle),
                    "Gltf asset registered as loaded but not in Assets<Gltf>"
                );
            }
            break;
        }
        assert!(
            Instant::now() < deadline,
            "tree glTF assets did not load within 60s"
        );
        std::thread::sleep(Duration::from_millis(16));
    }
}
