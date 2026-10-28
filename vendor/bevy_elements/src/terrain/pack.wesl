// The terrain's small per-frame kernels: the packed render buffer, the height readback and the
// leaf bounds. See docs/research/terrain.md ("Layer buffers and the render pack", GPU mapping).
//
//   terrain_pack      bedrock + sediment + water → terrain_cells (surface, sediment, water, slope)
//   terrain_readback  terrain_cells.x → one f32 per cell (2× box-averaged above 1024²)
//   terrain_minmax    terrain_cells → per 32×32-cell leaf (surface min, surface max, water-surface max, water depth max)
//   flow_pack         velocity + suspended + water → flow_cells (u, v, turbidity)

import bevy_elements::grid::{GridParams, grid_index};

struct TerrainParams {
    grid: GridParams,
    // x: y0 (m). y: readback downsample factor.
    misc: vec4<f32>,
}

@group(0) @binding(0) var<uniform> params: TerrainParams;
@group(0) @binding(1) var<storage, read> bedrock: array<f32>;
@group(0) @binding(2) var<storage, read> sediment: array<f32>;
@group(0) @binding(3) var<storage, read> water: array<f32>;
@group(0) @binding(4) var<storage, read_write> cells: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> height_rb: array<f32>;
@group(0) @binding(6) var<storage, read_write> leaf_bounds: array<vec4<f32>>;

fn surface_at(c: vec2<i32>) -> f32 {
    let k = grid_index(params.grid, c);
    return bedrock[k] + sediment[k];
}

@compute @workgroup_size(8, 8, 1)
fn terrain_pack(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = params.grid.size;
    if id.x >= size.x || id.y >= size.y {
        return;
    }
    let c = vec2<i32>(id.xy);
    let n = vec2<i32>(size);
    let lo = max(c - 1, vec2<i32>(0));
    let hi = min(c + 1, n - 1);
    let cell = params.grid.cell;
    // Central differences over ±1 cell; one-sided at the edge.
    let gx = (surface_at(vec2<i32>(hi.x, c.y)) - surface_at(vec2<i32>(lo.x, c.y))) / (f32(hi.x - lo.x) * cell);
    let gz = (surface_at(vec2<i32>(c.x, hi.y)) - surface_at(vec2<i32>(c.x, lo.y))) / (f32(hi.y - lo.y) * cell);
    let k = id.y * size.x + id.x;
    cells[k] = vec4<f32>(bedrock[k] + sediment[k], sediment[k], water[k], bitcast<f32>(pack2x16float(vec2<f32>(gx, gz))));
}

@compute @workgroup_size(8, 8, 1)
fn terrain_readback(@builtin(global_invocation_id) id: vec3<u32>) {
    let ds = u32(params.misc.y);
    let out_size = params.grid.size / ds;
    if id.x >= out_size.x || id.y >= out_size.y {
        return;
    }
    var sum = 0.0;
    for (var j = 0u; j < ds; j += 1u) {
        for (var i = 0u; i < ds; i += 1u) {
            let c = vec2<u32>(id.x * ds + i, id.y * ds + j);
            sum += cells[c.y * params.grid.size.x + c.x].x;
        }
    }
    height_rb[id.y * out_size.x + id.x] = sum / f32(ds * ds);
}

var<workgroup> reduce_a: array<vec4<f32>, 256>;

// Combine two partial bounds: (min surface, max surface, max water surface, max water depth).
fn combine(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(min(a.x, b.x), max(a.y, b.y), max(a.z, b.z), max(a.w, b.w));
}

fn cell_bounds(c: vec2<i32>) -> vec4<f32> {
    let size = vec2<i32>(params.grid.size);
    let k = clamp(c, vec2<i32>(0), size - 1);
    let v = cells[u32(k.y) * params.grid.size.x + u32(k.x)];
    return vec4<f32>(v.x, v.x, v.x + v.z, v.z);
}

// One workgroup per 32×32-cell leaf; each of the 256 threads reduces a 2×2 block plus a one-cell
// halo (4×4 cells): a vertex on the leaf's edge is a bilinear blend with the neighbouring leaf's
// first cell, so the leaf's bounds must cover it.
@compute @workgroup_size(16, 16, 1)
fn terrain_minmax(
    @builtin(workgroup_id) leaf: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(local_invocation_index) index: u32,
) {
    let origin = vec2<i32>(leaf.xy * 32u + lid.xy * 2u);
    var acc = cell_bounds(origin);
    for (var j = -1; j <= 2; j += 1) {
        for (var i = -1; i <= 2; i += 1) {
            acc = combine(acc, cell_bounds(origin + vec2<i32>(i, j)));
        }
    }
    reduce_a[index] = acc;
    workgroupBarrier();
    for (var stride = 128u; stride > 0u; stride = stride >> 1u) {
        if index < stride {
            reduce_a[index] = combine(reduce_a[index], reduce_a[index + stride]);
        }
        workgroupBarrier();
    }
    if index == 0u {
        let leaves_x = params.grid.size.x / 32u;
        leaf_bounds[leaf.y * leaves_x + leaf.x] = reduce_a[0];
    }
}
