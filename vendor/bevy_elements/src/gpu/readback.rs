//! Throttled async readback of GPU fields into [`EnvironmentSnapshot`].
//!
//! Every [`ElementsReadback::interval`] each domain may request one readback per field. A request
//! spawns an entity with Bevy's one-shot `ReadbackOnce` bundle; its `ReadbackComplete` observer
//! decodes the bytes into the snapshot and despawns the entity. A field whose previous request is
//! still in flight skips the tick, and a request that never completes (the buffer was not on the
//! GPU yet) is dropped after a timeout so the field is retried.

use std::time::Duration;

use bevy::ecs::system::SystemParam;
use bevy::prelude::*;
use bevy::render::gpu_readback::{ReadbackComplete, ReadbackOnce};
use bevy::render::render_resource::PollType;
use bevy::render::renderer::RenderDevice;
use bevy::render::storage::ShaderBuffer;
use bevy::render::view::ExtractedWindow;
use bevy::render::{Render, RenderApp, RenderSystems};

use crate::plugin::ElementsSystems;
use crate::query::EnvironmentSnapshot;

/// How often GPU state is copied back for gameplay queries.
#[derive(Resource, Clone, Debug, Reflect)]
pub struct ElementsReadback {
    /// Time between readbacks, real time. Default 250 ms (4 Hz).
    pub interval: Duration,
}

impl Default for ElementsReadback {
    fn default() -> Self {
        Self {
            interval: Duration::from_millis(250),
        }
    }
}

/// Whether this frame is a readback tick.
#[derive(Resource, Default)]
pub(crate) struct ReadbackTick {
    since: Duration,
    due: bool,
}

impl ReadbackTick {
    /// Whether this frame is a readback tick.
    pub(crate) fn is_due(&self) -> bool {
        self.due
    }
}

/// An outstanding readback of one named field.
#[derive(Component)]
pub(crate) struct ReadbackField {
    key: &'static str,
    issued: Duration,
}

pub(crate) struct ReadbackPlugin;

impl Plugin for ReadbackPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ElementsReadback>()
            .init_resource::<ReadbackTick>()
            .add_systems(
                Update,
                (
                    tick.in_set(ElementsSystems::Clock),
                    drop_stale.in_set(ElementsSystems::Readback),
                ),
            );
        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app.add_systems(
                Render,
                pace_without_window.in_set(RenderSystems::PostCleanup),
            );
        }
    }
}

/// Without a window nothing paces the frames: presenting to a swap chain is what makes the CPU
/// wait for the GPU, and an app that renders only to images submits frames faster than the GPU
/// retires them. The queue then grows without bound, every readback's latency grows with it
/// (measured: 2, 4, 8 … frames, past the drop timeout within 3 s), and the queries freeze. So
/// a frame with no window waits for its own work, which also delivers the readback mapped this
/// frame. See `docs/research/readback-pacing.md`.
fn pace_without_window(device: Res<RenderDevice>, windows: Query<(), With<ExtractedWindow>>) {
    if !windows.is_empty() {
        return;
    }
    if let Err(error) = device.poll(PollType::wait_indefinitely()) {
        error!("waiting for the GPU's frame failed: {error}");
    }
}

fn tick(time: Res<Time<Real>>, settings: Res<ElementsReadback>, mut tick: ResMut<ReadbackTick>) {
    tick.since += time.delta();
    tick.due = tick.since >= settings.interval;
    if tick.due {
        tick.since = Duration::ZERO;
    }
}

fn drop_stale(
    mut commands: Commands,
    time: Res<Time<Real>>,
    settings: Res<ElementsReadback>,
    fields: Query<(Entity, &ReadbackField)>,
) {
    let limit = settings.interval * 4 + Duration::from_secs(1);
    for (entity, field) in &fields {
        if time.elapsed().saturating_sub(field.issued) > limit {
            commands.entity(entity).despawn();
        }
    }
}

/// Issue readbacks from a domain system in [`ElementsSystems::Readback`].
#[derive(SystemParam)]
pub(crate) struct Readbacks<'w, 's> {
    commands: Commands<'w, 's>,
    outstanding: Query<'w, 's, &'static ReadbackField>,
    tick: Res<'w, ReadbackTick>,
    time: Res<'w, Time<Real>>,
}

impl Readbacks<'_, '_> {
    /// Read `buffer` back and hand the bytes to `decode`, unless it is not a tick or `key` is
    /// still in flight. `decode` captures whatever describes the data at request time (grid
    /// origin, size), so a field that moved meanwhile is still placed where it was read.
    pub fn request(
        &mut self,
        key: &'static str,
        buffer: Handle<ShaderBuffer>,
        decode: impl Fn(&[u8], &mut EnvironmentSnapshot) + Send + Sync + 'static,
    ) {
        if !self.tick.due || self.outstanding.iter().any(|f| f.key == key) {
            return;
        }
        self.commands
            .spawn((
                ReadbackField {
                    key,
                    issued: self.time.elapsed(),
                },
                ReadbackOnce::buffer(buffer),
            ))
            .observe(
                move |done: On<ReadbackComplete>,
                      mut snapshot: ResMut<EnvironmentSnapshot>,
                      time: Res<Time<Real>>,
                      mut commands: Commands| {
                    decode(&done.data, &mut snapshot);
                    snapshot.updated_at = Some(time.elapsed());
                    if let Ok(mut entity) = commands.get_entity(done.entity) {
                        entity.despawn();
                    }
                },
            );
    }
}

/// Decode a tightly packed `array<T>` readback. The bytes come from a `Vec<u8>`, which carries no
/// alignment guarantee, so they are copied rather than cast in place. `None` for a length that is
/// not a whole number of `T`.
pub(crate) fn decode<T: bytemuck::Pod>(bytes: &[u8]) -> Option<Vec<T>> {
    bytes
        .len()
        .is_multiple_of(size_of::<T>())
        .then(|| bytemuck::pod_collect_to_vec(bytes))
}
