//! Shared setup for the GPU tests: a windowless app that still renders and computes.

#![allow(dead_code)]

use bevy::app::{PluginGroupBuilder, PluginsState};
use bevy::camera::RenderTarget;
use bevy::prelude::*;
use bevy::render::RenderPlugin;
use bevy::render::render_resource::TextureFormat;
use bevy::tasks::tick_global_task_pools_on_main_thread;
use bevy::window::ExitCondition;
use bevy::winit::WinitPlugin;

/// `DefaultPlugins` without a window or event loop, compiling pipelines on first use so frames
/// measure the simulation rather than the shader compiler.
pub fn headless_plugins() -> PluginGroupBuilder {
    DefaultPlugins
        .set(WindowPlugin {
            primary_window: None,
            exit_condition: ExitCondition::DontExit,
            ..default()
        })
        .set(RenderPlugin {
            synchronous_pipeline_compilation: true,
            ..default()
        })
        .disable::<WinitPlugin>()
}

/// Wait for the renderer (created asynchronously) and finish plugin setup, as `App::run` does.
pub fn finish(app: &mut App) {
    while app.plugins_state() == PluginsState::Adding {
        tick_global_task_pools_on_main_thread();
    }
    app.finish();
    app.cleanup();
}

/// A camera needs a target to have a view: Bevy 0.20's atmosphere environment probe unwraps the
/// view uniforms, so a target-less camera panics the render world. 256 × 256 is enough.
pub fn image_target(images: &mut Assets<Image>) -> RenderTarget {
    RenderTarget::Image(
        images
            .add(Image::new_target_texture(
                256,
                256,
                TextureFormat::Rgba8UnormSrgb,
                None,
            ))
            .into(),
    )
}

/// Read a texture back, tightly packed: the copy pads every row to 256 bytes, which is removed
/// here. `rows` is height × depth. Spawns a persistent `Readback`, updates the app until the
/// bytes arrive and despawns it.
pub fn read_texture(app: &mut App, image: Handle<Image>, row_bytes: usize, rows: usize) -> Vec<u8> {
    read_textures(app, vec![(image, row_bytes, rows)])
        .pop()
        .expect("one texture requested")
}

/// Read several textures back as they were in the same frame (for example both halves of a
/// ping-pong pair): every `Readback` is spawned before the next update, so their copies are
/// recorded in one frame, and only each one's first result is kept. Each entry is
/// `(image, row_bytes, rows)` as in [`read_texture`].
pub fn read_textures(app: &mut App, requests: Vec<(Handle<Image>, usize, usize)>) -> Vec<Vec<u8>> {
    use std::sync::{Arc, Mutex};

    use bevy::render::gpu_readback::{Readback, ReadbackComplete};

    let slots: Vec<Arc<Mutex<Option<Vec<u8>>>>> = requests.iter().map(|_| Arc::default()).collect();
    let entities: Vec<Entity> = requests
        .iter()
        .zip(&slots)
        .map(|((image, _, _), slot)| {
            let sink = slot.clone();
            app.world_mut()
                .spawn(Readback::texture(image.clone()))
                .observe(move |done: On<ReadbackComplete>| {
                    let mut sink = sink.lock().expect("lock");
                    if sink.is_none() {
                        *sink = Some(done.data.clone());
                    }
                })
                .id()
        })
        .collect();
    // A wall-clock bound, not a frame count: updates do not wait for the GPU, so on a GPU busy
    // with other work a readback can take more than a hundred frames, and over 20 s, to arrive.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(60);
    while std::time::Instant::now() < deadline {
        app.update();
        if slots.iter().all(|s| s.lock().expect("lock").is_some()) {
            break;
        }
    }
    for entity in entities {
        app.world_mut().despawn(entity);
    }
    requests
        .iter()
        .zip(slots)
        .map(|((_, row_bytes, rows), slot)| {
            let padded = slot
                .lock()
                .expect("lock")
                .take()
                .expect("the readback completed");
            let stride = row_bytes.next_multiple_of(256);
            assert_eq!(padded.len(), stride * rows);
            padded
                .chunks_exact(stride)
                .flat_map(|row| row[..*row_bytes].iter().copied())
                .collect()
        })
        .collect()
}

/// IEEE half to f32.
pub fn half(bits: u16) -> f32 {
    let sign = if bits & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exponent = i32::from((bits >> 10) & 0x1F);
    let mantissa = f32::from(bits & 0x3FF);
    match exponent {
        0 => sign * mantissa * 2f32.powi(-24),
        31 => sign * f32::INFINITY,
        e => sign * (1.0 + mantissa / 1024.0) * 2f32.powi(e - 15),
    }
}
