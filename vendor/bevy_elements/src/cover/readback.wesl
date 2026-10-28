// cover_readback: box mean of (snow depth, wetness) over ds × ds cells (ds = 1 up to 1024², 2
// above) into the compact buffer the CPU reads on a readback tick.

import bevy_elements::cover::common::CoverParams;

@group(0) @binding(0) var<uniform> params: CoverParams;
@group(0) @binding(1) var<storage, read_write> cover_cells: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> readback: array<vec2<f32>>;

@compute @workgroup_size(8, 8, 1)
fn cover_readback(@builtin(global_invocation_id) id: vec3<u32>) {
    let ds = params.flags.z;
    let out_size = params.grid.size / ds;
    if id.x >= out_size.x || id.y >= out_size.y {
        return;
    }
    var sum = vec2<f32>(0.0);
    for (var j = 0u; j < ds; j += 1u) {
        for (var i = 0u; i < ds; i += 1u) {
            let c = cover_cells[(id.y * ds + j) * params.grid.size.x + id.x * ds + i];
            sum += c.xy;
        }
    }
    readback[id.y * out_size.x + id.x] = sum / f32(ds * ds);
}
