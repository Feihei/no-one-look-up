// The water readback kernels, dispatched on readback ticks only:
//
//   downsample_depth     box mean of `water` over ds × ds cells (ds = 1 up to 1024², 2 above)
//   downsample_velocity  2 × 2 mean of `velocity` weighted by depth: Σ d v / Σ d, 0 where
//                        Σ d < 1e-3 m
//
// All bindings are read_write so the two entry points share one layout.

import bevy_elements::grid::GridParams;

struct HydroParams {
    grid: GridParams,
    sim: vec4<f32>,
    weather: vec4<f32>,
    misc: vec4<f32>,
    flags: vec4<u32>,
    erosion: vec4<f32>,
    thermal: vec4<f32>,
    // y: depth readback downsample factor.
    extra: vec4<f32>,
}

@group(0) @binding(0) var<uniform> params: HydroParams;
@group(0) @binding(1) var<storage, read_write> water: array<f32>;
@group(0) @binding(2) var<storage, read_write> velocity: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read_write> depth_rb: array<f32>;
@group(0) @binding(4) var<storage, read_write> velocity_rb: array<vec2<f32>>;

const MIN_DEPTH_SUM: f32 = 1e-3;

@compute @workgroup_size(8, 8, 1)
fn downsample_depth(@builtin(global_invocation_id) id: vec3<u32>) {
    let ds = u32(params.extra.y);
    let out_size = params.grid.size / ds;
    if id.x >= out_size.x || id.y >= out_size.y {
        return;
    }
    var sum = 0.0;
    for (var j = 0u; j < ds; j += 1u) {
        for (var i = 0u; i < ds; i += 1u) {
            sum += water[(id.y * ds + j) * params.grid.size.x + id.x * ds + i];
        }
    }
    depth_rb[id.y * out_size.x + id.x] = sum / f32(ds * ds);
}

@compute @workgroup_size(8, 8, 1)
fn downsample_velocity(@builtin(global_invocation_id) id: vec3<u32>) {
    let out_size = params.grid.size / 2u;
    if id.x >= out_size.x || id.y >= out_size.y {
        return;
    }
    var weight = 0.0;
    var momentum = vec2<f32>(0.0);
    for (var j = 0u; j < 2u; j += 1u) {
        for (var i = 0u; i < 2u; i += 1u) {
            let k = (id.y * 2u + j) * params.grid.size.x + id.x * 2u + i;
            weight += water[k];
            momentum += water[k] * velocity[k];
        }
    }
    var mean = vec2<f32>(0.0);
    if weight >= MIN_DEPTH_SUM {
        mean = momentum / weight;
    }
    velocity_rb[id.y * out_size.x + id.x] = mean;
}
