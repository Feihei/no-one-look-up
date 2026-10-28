//! A compute kernel: one queued pipeline plus the layout its bind group is built against.
//!
//! Bevy offers the raw pipeline cache and nothing above it; every domain here dispatches a handful
//! of kernels per frame and needs the same three answers: is it compiled, did it fail, and what
//! layout does its bind group use.

use std::borrow::Cow;
use std::sync::atomic::{AtomicBool, Ordering};

use bevy::prelude::*;
use bevy::render::render_resource::{
    BindGroupLayoutDescriptor, CachedComputePipelineId, CachedPipelineState, ComputePipeline,
    ComputePipelineDescriptor, PipelineCache,
};
use bevy::shader::{ShaderCacheError, ShaderDefVal};

/// Workgroup edge for 3D kernels (`@workgroup_size(4, 4, 4)`).
pub(crate) const WG3: u32 = 4;

/// One compute entry point, queued in the pipeline cache.
pub(crate) struct Kernel {
    id: CachedComputePipelineId,
    /// The layout of bind group 0.
    pub layout: BindGroupLayoutDescriptor,
    entry: &'static str,
    reported: AtomicBool,
}

/// Queue `entry` of `shader` with bind group 0 laid out as `layout`.
pub(crate) fn queue_kernel(
    cache: &PipelineCache,
    shader: Handle<Shader>,
    entry: &'static str,
    layout: BindGroupLayoutDescriptor,
    defs: Vec<ShaderDefVal>,
) -> Kernel {
    let id = cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some(Cow::Borrowed(entry)),
        layout: vec![layout.clone()],
        shader,
        shader_defs: defs,
        entry_point: Some(Cow::Borrowed(entry)),
        zero_initialize_workgroup_memory: true,
        ..default()
    });
    Kernel {
        id,
        layout,
        entry,
        reported: AtomicBool::new(false),
    }
}

/// The compiled pipeline, or `None` while it compiles or after it failed.
///
/// A failure other than a not-yet-loaded shader is reported once with `error!` naming the
/// kernel; the pipeline cache has already logged the compiler's message. The domain then stays
/// idle: there is no fallback path.
pub(crate) fn ready<'a>(cache: &'a PipelineCache, kernel: &Kernel) -> Option<&'a ComputePipeline> {
    match cache.get_compute_pipeline_state(kernel.id) {
        CachedPipelineState::Ok(_) => cache.get_compute_pipeline(kernel.id),
        CachedPipelineState::Err(
            ShaderCacheError::ShaderNotLoaded(_) | ShaderCacheError::ShaderImportNotYetAvailable,
        ) => None,
        CachedPipelineState::Err(error) => {
            if !kernel.reported.swap(true, Ordering::Relaxed) {
                error!(
                    "bevy_elements kernel `{}` failed to compile and its domain will stay idle: {error}",
                    kernel.entry
                );
            }
            None
        }
        CachedPipelineState::Queued | CachedPipelineState::Creating(_) => None,
    }
}

/// Workgroups needed to cover `cells` with groups of `edge`.
pub(crate) fn groups(cells: u32, edge: u32) -> u32 {
    cells.div_ceil(edge)
}
