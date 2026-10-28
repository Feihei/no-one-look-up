// source_map_build: rasterise the water sources and sinks and the precipitation blockers into the
// per-cell source map. Runs when sources, sinks, blockers or the grid change, not every substep.
//
//   source_map[cell].x   sum of the depth rates of the sources covering the cell, m/s (sinks < 0)
//   source_map[cell].y   rain mask: 0 when a blocker box lies above the cell's surface, else 1
//
// A source covers the cells whose centres lie within its radius and at least the cell that holds
// its centre (the CPU counts the cells with the same expression, `model::covers`). A cell is
// shaded when the vertical line up from the surface point at its centre meets a blocker box (the
// same inverse-matrix box test as the precipitation kernel).

import bevy_elements::grid::GridParams;

struct HydroParams {
    grid: GridParams,
    sim: vec4<f32>,
    weather: vec4<f32>,
    misc: vec4<f32>,
    // z: sources uploaded. w: blockers uploaded.
    flags: vec4<u32>,
    erosion: vec4<f32>,
    thermal: vec4<f32>,
    // x: terrain y0 (m).
    extra: vec4<f32>,
}

struct Blocker {
    // Rows of the inverse of the box's affine matrix: local = rot · (world − centre).
    rot: array<vec4<f32>, 3>,
    centre: vec3<f32>,
    extent: vec3<f32>,
}

@group(0) @binding(0) var<uniform> params: HydroParams;
@group(0) @binding(1) var<storage, read> bedrock: array<f32>;
@group(0) @binding(2) var<storage, read> sediment: array<f32>;
// xy: world XZ of the centre, z: radius (m), w: depth rate per covered cell (m/s).
@group(0) @binding(3) var<storage, read> sources: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> blockers: array<Blocker>;
@group(0) @binding(5) var<storage, read_write> source_map: array<vec2<f32>>;

const MISS: f32 = 2.0;
// How far above the surface a blocker still shelters it, m.
const SHELTER_HEIGHT_M: f32 = 10000.0;

// Whether the segment p0 → p1 meets the blocker box.
fn segment_hits(b: Blocker, p0: vec3<f32>, p1: vec3<f32>) -> bool {
    let d0 = p0 - b.centre;
    let d1 = p1 - b.centre;
    let q0 = vec3<f32>(dot(b.rot[0].xyz, d0), dot(b.rot[1].xyz, d0), dot(b.rot[2].xyz, d0));
    let q1 = vec3<f32>(dot(b.rot[0].xyz, d1), dot(b.rot[1].xyz, d1), dot(b.rot[2].xyz, d1));
    let d = q1 - q0;
    var t_in = 0.0;
    var t_out = 1.0;
    for (var a = 0; a < 3; a += 1) {
        if abs(d[a]) < 1e-7 {
            if abs(q0[a]) > b.extent[a] {
                return false;
            }
        } else {
            let inv = 1.0 / d[a];
            let t_a = (-b.extent[a] - q0[a]) * inv;
            let t_b = (b.extent[a] - q0[a]) * inv;
            t_in = max(t_in, min(t_a, t_b));
            t_out = min(t_out, max(t_a, t_b));
        }
    }
    return t_in <= t_out;
}

@compute @workgroup_size(8, 8, 1)
fn source_map_build(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = params.grid.size;
    if id.x >= size.x || id.y >= size.y {
        return;
    }
    let c = vec2<i32>(id.xy);
    let k = id.y * size.x + id.x;
    let centre = params.grid.origin + (vec2<f32>(c) + 0.5) * params.grid.cell;
    var rate = 0.0;
    for (var i = 0u; i < params.flags.z; i += 1u) {
        let s = sources[i];
        let d = centre - s.xy;
        let home = vec2<i32>(floor((s.xy - params.grid.origin) / params.grid.cell));
        if dot(d, d) <= s.z * s.z || (home.x == c.x && home.y == c.y) {
            rate += s.w;
        }
    }
    var mask = 1.0;
    if params.flags.w > 0u {
        let p0 = vec3<f32>(centre.x, params.extra.x + bedrock[k] + sediment[k], centre.y);
        let p1 = p0 + vec3<f32>(0.0, SHELTER_HEIGHT_M, 0.0);
        for (var i = 0u; i < params.flags.w; i += 1u) {
            if segment_hits(blockers[i], p0, p1) {
                mask = 0.0;
                break;
            }
        }
    }
    source_map[k] = vec2<f32>(rate, mask);
}
