//! The public WESL imports compile in a consumer's shader: `bevy_elements::noise`,
//! `bevy_elements::grid` and `bevy_elements::wind`, exactly as their doc comments show.
//!
//! Needs a GPU: `cargo test --test gpu_imports -- --ignored`.

mod support;

use std::borrow::Cow;
use std::sync::{Arc, Mutex};

use bevy::app::AppExit;
use bevy::prelude::*;
use bevy::render::render_resource::binding_types::{
    sampler, storage_buffer_sized, texture_3d, uniform_buffer, uniform_buffer_sized,
};
use bevy::render::render_resource::{
    BindGroupLayoutDescriptor, BindGroupLayoutEntries, CachedComputePipelineId,
    CachedPipelineState, ComputePipelineDescriptor, PipelineCache, SamplerBindingType,
    ShaderStages, TextureSampleType,
};
use bevy::render::{Render, RenderApp, RenderStartup, RenderSystems};
use bevy_elements::*;

const CONSUMER: &str = r#"
import bevy_elements::noise::{perlin3, perlin3_d, fbm3, worley3, curl3};
import bevy_elements::grid::{GridParams, grid_taps};
import bevy_elements::wind::{WindFieldParams, elements_wind};

@group(0) @binding(0) var wind_field: texture_3d<f32>;
@group(0) @binding(1) var wind_sampler: sampler;
@group(0) @binding(2) var<uniform> wind_params: WindFieldParams;
@group(0) @binding(3) var<uniform> grid_params: GridParams;
@group(0) @binding(4) var<storage, read_write> result: array<vec4<f32>>;

@compute @workgroup_size(1)
fn main() {
    let v = elements_wind(wind_field, wind_sampler, wind_params, vec3<f32>(1.0, 2.0, 3.0));
    let t = grid_taps(grid_params, vec2<f32>(1.0, 2.0));
    let n = perlin3(v) + fbm3(v, 2u) + worley3(v, 0) + curl3(v).x + perlin3_d(v).y;
    result[0] = vec4<f32>(v, f32(t.i00) + t.w.x + n);
}
"#;

#[derive(Resource)]
struct ConsumerShader(Handle<Shader>);

#[derive(Resource)]
struct Check(CachedComputePipelineId);

/// The pipeline state as seen from the render world, which runs on its own thread.
#[derive(Resource, Clone, Default)]
struct Outcome(Arc<Mutex<Option<String>>>);

fn queue(mut commands: Commands, cache: Res<PipelineCache>, shader: Res<ConsumerShader>) {
    let layout = BindGroupLayoutDescriptor::new(
        "import_check",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                texture_3d(TextureSampleType::Float { filterable: true }),
                sampler(SamplerBindingType::Filtering),
                uniform_buffer::<WindFieldParams>(false),
                uniform_buffer_sized(false, None),
                storage_buffer_sized(false, None),
            ),
        ),
    );
    commands.insert_resource(Check(cache.queue_compute_pipeline(
        ComputePipelineDescriptor {
            label: Some(Cow::Borrowed("import_check")),
            layout: vec![layout],
            shader: shader.0.clone(),
            entry_point: Some(Cow::Borrowed("main")),
            ..default()
        },
    )));
}

fn observe(cache: Res<PipelineCache>, check: Option<Res<Check>>, outcome: Res<Outcome>) {
    let Some(check) = check else {
        return;
    };
    let state = match cache.get_compute_pipeline_state(check.0) {
        CachedPipelineState::Ok(_) => Some("ok".to_string()),
        CachedPipelineState::Err(e) => Some(format!("{e}")),
        _ => None,
    };
    if let (Some(state), Ok(mut slot)) = (state, outcome.0.lock()) {
        *slot = Some(state);
    }
}

#[test]
#[ignore = "needs a GPU"]
fn a_consumer_shader_can_import_every_public_module() {
    let mut app = App::new();
    app.add_plugins((support::headless_plugins(), ElementsPlugins));
    let shader = app
        .world_mut()
        .resource_mut::<Assets<Shader>>()
        .add(Shader::from_wesl(
            CONSUMER,
            "embedded://import_check/consumer.wesl",
        ));
    let outcome = Outcome::default();
    app.sub_app_mut(RenderApp)
        .insert_resource(ConsumerShader(shader))
        .insert_resource(outcome.clone())
        .add_systems(RenderStartup, queue)
        .add_systems(Render, observe.in_set(RenderSystems::Cleanup));
    support::finish(&mut app);

    let mut state = None;
    for _ in 0..120 {
        app.update();
        state = outcome.0.lock().ok().and_then(|s| s.clone());
        if state.is_some() {
            break;
        }
    }
    app.world_mut().write_message(AppExit::Success);
    assert_eq!(
        state.as_deref(),
        Some("ok"),
        "the consumer shader did not compile: {state:?}"
    );
}
