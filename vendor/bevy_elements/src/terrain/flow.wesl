// flow_pack: the water renderer's packed buffer (docs/research/terrain.md, "Layer buffers and the
// render pack"): x, y depth-averaged velocity (m/s), z turbidity (suspended / max(depth, 1e-3)).

import bevy_elements::grid::GridParams;

struct FlowParams {
    grid: GridParams,
    misc: vec4<f32>,
}

@group(0) @binding(0) var<uniform> params: FlowParams;
@group(0) @binding(1) var<storage, read> velocity: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> suspended: array<f32>;
@group(0) @binding(3) var<storage, read> water: array<f32>;
@group(0) @binding(4) var<storage, read_write> flow_cells: array<vec4<f32>>;

@compute @workgroup_size(8, 8, 1)
fn flow_pack(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = params.grid.size;
    if id.x >= size.x || id.y >= size.y {
        return;
    }
    let k = id.y * size.x + id.x;
    flow_cells[k] = vec4<f32>(velocity[k], suspended[k] / max(water[k], 1e-3), 0.0);
}
