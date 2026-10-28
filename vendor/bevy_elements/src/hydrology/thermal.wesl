// Thermal erosion: every 4th substep, two gather dispatches (docs/research/erosion.md).
//
//   thermal_flux    per cell, the material that leaves toward each of the eight neighbours
//                   (axis neighbours in tflux_a, diagonals in tflux_b) and what it takes from the
//                   sediment and bedrock layers (in transport, a scratch of the substep kernels)
//   thermal_apply   per cell, subtract its own move and add what its neighbours send it
//
// tflux_a = (L, R, T, B), tflux_b = (TL, TR, BL, BR); T is row − 1. All bindings are read_write in
// both entry points so they share one layout; thermal_flux reads other cells only from the layers
// it does not write, thermal_apply only writes its own cell.

import bevy_elements::grid::GridParams;

struct HydroParams {
    grid: GridParams,
    sim: vec4<f32>,
    weather: vec4<f32>,
    misc: vec4<f32>,
    flags: vec4<u32>,
    erosion: vec4<f32>,
    // x: tan(talus). y: tan(talus + 10°). z: thermal fraction per pass.
    thermal: vec4<f32>,
    extra: vec4<f32>,
}

@group(0) @binding(0) var<uniform> params: HydroParams;
@group(0) @binding(1) var<storage, read_write> bedrock: array<f32>;
@group(0) @binding(2) var<storage, read_write> sediment: array<f32>;
@group(0) @binding(3) var<storage, read_write> tflux_a: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> tflux_b: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> moved: array<vec2<f32>>;

const SQRT2: f32 = 1.4142135;

fn in_grid(c: vec2<i32>) -> bool {
    let n = vec2<i32>(params.grid.size);
    return c.x >= 0 && c.y >= 0 && c.x < n.x && c.y < n.y;
}

fn cell_index(c: vec2<i32>) -> u32 {
    return u32(c.y) * params.grid.size.x + u32(c.x);
}

// The excess of the drop to the neighbour at `dir` over the talus and over the bedrock threshold:
// (e, e_bedrock), zero off the grid.
fn excess(c: vec2<i32>, dir: vec2<i32>, surface: f32) -> vec2<f32> {
    let nb = c + dir;
    if !in_grid(nb) {
        return vec2<f32>(0.0);
    }
    let k = cell_index(nb);
    let drop = surface - (bedrock[k] + sediment[k]);
    var dist = params.grid.cell;
    if dir.x != 0 && dir.y != 0 {
        dist = dist * SQRT2;
    }
    return max(vec2<f32>(drop - dist * params.thermal.x, drop - dist * params.thermal.y), vec2<f32>(0.0));
}

@compute @workgroup_size(8, 8, 1)
fn thermal_flux(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = params.grid.size;
    if id.x >= size.x || id.y >= size.y {
        return;
    }
    let c = vec2<i32>(id.xy);
    let k = id.y * size.x + id.x;
    let surface = bedrock[k] + sediment[k];
    let e0 = excess(c, vec2<i32>(-1, 0), surface);
    let e1 = excess(c, vec2<i32>(1, 0), surface);
    let e2 = excess(c, vec2<i32>(0, -1), surface);
    let e3 = excess(c, vec2<i32>(0, 1), surface);
    let e4 = excess(c, vec2<i32>(-1, -1), surface);
    let e5 = excess(c, vec2<i32>(1, -1), surface);
    let e6 = excess(c, vec2<i32>(-1, 1), surface);
    let e7 = excess(c, vec2<i32>(1, 1), surface);
    let ea = vec4<f32>(e0.x, e1.x, e2.x, e3.x);
    let eb = vec4<f32>(e4.x, e5.x, e6.x, e7.x);
    let ba = vec4<f32>(e0.y, e1.y, e2.y, e3.y);
    let bb = vec4<f32>(e4.y, e5.y, e6.y, e7.y);
    let max_e = max(max(max(ea.x, ea.y), max(ea.z, ea.w)), max(max(eb.x, eb.y), max(eb.z, eb.w)));
    let max_b = max(max(max(ba.x, ba.y), max(ba.z, ba.w)), max(max(bb.x, bb.y), max(bb.z, bb.w)));
    let rate = params.thermal.z;
    let from_sediment = min(0.5 * rate * max_e, sediment[k]);
    let from_bedrock = min(0.5 * rate * max_e - from_sediment, 0.5 * rate * max_b);
    let sum_e = ea.x + ea.y + ea.z + ea.w + eb.x + eb.y + eb.z + eb.w;
    let sum_b = ba.x + ba.y + ba.z + ba.w + bb.x + bb.y + bb.z + bb.w;
    var qa = vec4<f32>(0.0);
    var qb = vec4<f32>(0.0);
    if sum_e > 0.0 {
        qa = ea * (from_sediment / sum_e);
        qb = eb * (from_sediment / sum_e);
    }
    if sum_b > 0.0 {
        qa = qa + ba * (from_bedrock / sum_b);
        qb = qb + bb * (from_bedrock / sum_b);
    }
    tflux_a[k] = qa;
    tflux_b[k] = qb;
    moved[k] = vec2<f32>(from_sediment, from_bedrock);
}

@compute @workgroup_size(8, 8, 1)
fn thermal_apply(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = params.grid.size;
    if id.x >= size.x || id.y >= size.y {
        return;
    }
    let c = vec2<i32>(id.xy);
    let k = id.y * size.x + id.x;
    var received = 0.0;
    // The neighbour at c + o sends toward c through the pipe pointing at −o.
    let nl = c + vec2<i32>(-1, 0);
    let nr = c + vec2<i32>(1, 0);
    let nt = c + vec2<i32>(0, -1);
    let nb = c + vec2<i32>(0, 1);
    if in_grid(nl) { received += tflux_a[cell_index(nl)].y; }
    if in_grid(nr) { received += tflux_a[cell_index(nr)].x; }
    if in_grid(nt) { received += tflux_a[cell_index(nt)].w; }
    if in_grid(nb) { received += tflux_a[cell_index(nb)].z; }
    let ntl = c + vec2<i32>(-1, -1);
    let ntr = c + vec2<i32>(1, -1);
    let nbl = c + vec2<i32>(-1, 1);
    let nbr = c + vec2<i32>(1, 1);
    if in_grid(ntl) { received += tflux_b[cell_index(ntl)].w; }
    if in_grid(ntr) { received += tflux_b[cell_index(ntr)].z; }
    if in_grid(nbl) { received += tflux_b[cell_index(nbl)].y; }
    if in_grid(nbr) { received += tflux_b[cell_index(nbr)].x; }
    let own = moved[k];
    sediment[k] = sediment[k] - own.x + received;
    bedrock[k] = bedrock[k] - own.y;
}
