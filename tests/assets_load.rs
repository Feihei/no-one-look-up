//! R0 asset-chain verification: tree glTF models exported from Blender load
//! through Bevy's AssetServer in a windowless headless app.
//!
//! Assets: `assets/models/*.glb` exported from the original Godot project's
//! `.blend` files (aspen_1, tree_2, tree_3). Those exports are local-only
//! (`.gitignore` keeps `/assets/*` untracked), so the test skips when none of
//! them is present — a fresh clone or CI run has nothing to verify. A partial
//! export still fails, because that is a real packaging mistake.
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

/// Which of [`TREE_GLBS`] are missing below `assets_root` (the directory the
/// `AssetServer` reads from).
///
/// The exports are ~32 MB of Blender output and stay untracked (`.gitignore`
/// keeps `/assets/*` local), so a fresh clone, CI included, has no
/// `assets/models/` at all. Callers read "all missing" as "nothing to verify
/// here" and "some missing" as a broken export.
fn missing_tree_glbs(assets_root: &std::path::Path) -> Vec<&'static str> {
    TREE_GLBS
        .iter()
        .copied()
        .filter(|path| !assets_root.join(path).exists())
        .collect()
}

#[test]
fn tree_gltf_assets_load_headless() {
    let missing = missing_tree_glbs(std::path::Path::new("assets"));
    if missing.len() == TREE_GLBS.len() {
        // Local-only assets: CI never receives them, and a green build must not
        // depend on files the repository does not ship.
        println!("skipped: no tree glTF under assets/models (local-only exports, see README)");
        return;
    }
    assert!(
        missing.is_empty(),
        "partially exported tree assets, missing: {missing:?}"
    );

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

#[cfg(test)]
mod tests {
    use super::{TREE_GLBS, missing_tree_glbs};
    use std::path::PathBuf;

    /// A unique scratch directory under the OS temp root; removed on drop.
    struct ScratchDir(PathBuf);

    impl ScratchDir {
        fn new(tag: &str) -> Self {
            let path = std::env::temp_dir()
                .join(format!("no_one_assets_load_{tag}_{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&path);
            std::fs::create_dir_all(&path).expect("scratch dir");
            Self(path)
        }

        fn path(&self) -> &std::path::Path {
            &self.0
        }
    }

    impl Drop for ScratchDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn no_assets_means_every_glb_is_missing() {
        let dir = ScratchDir::new("empty");
        assert_eq!(missing_tree_glbs(dir.path()).len(), TREE_GLBS.len());
    }

    #[test]
    fn a_partial_export_reports_exactly_the_absent_files() {
        let dir = ScratchDir::new("partial");
        let models = dir.path().join("models");
        std::fs::create_dir_all(&models).expect("models dir");
        std::fs::write(models.join("aspen_1.glb"), b"").expect("placeholder");

        let missing = missing_tree_glbs(dir.path());
        assert_eq!(missing, vec!["models/tree_2.glb", "models/tree_3.glb"]);
    }

    #[test]
    fn a_complete_export_reports_nothing_missing() {
        let dir = ScratchDir::new("full");
        let models = dir.path().join("models");
        std::fs::create_dir_all(&models).expect("models dir");
        for path in TREE_GLBS {
            std::fs::write(dir.path().join(path), b"").expect("placeholder");
        }

        assert!(missing_tree_glbs(dir.path()).is_empty());
    }
}
