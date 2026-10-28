// shelter_mask: which ground is sheltered from rain and snow by a PrecipitationBlocker. A ground
// point is sheltered when the ray from it toward where the precipitation came from, (−wind,
// fall speed), meets a blocker box (docs/research/wetness.md, "Shelter"; the slab test of the
// particle kernel). Runs when the blocker set or the fall directions change, not every step.
//
//   shelter[cell] = (rain exposure, snow exposure), 0 sheltered, 1 exposed

import bevy_elements::cover::common::CoverParams;

struct Blocker {
    // Rows of the inverse of the box's affine matrix: local = rot · (world − centre).
    rot: array<vec4<f32>, 3>,
    centre: vec3<f32>,
    extent: vec3<f32>,
}

@group(0) @binding(0) var<uniform> params: CoverParams;
@group(0) @binding(1) var<storage, read> blockers: array<Blocker>;
@group(0) @binding(2) var<storage, read_write> cells: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> shelter: array<vec2<f32>>;

// How high above the ground a blocker still shelters it, m.
const SHELTER_HEIGHT_M: f32 = 500.0;

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

// The segment from `p0` up the ray (dx, v_f, dz) to SHELTER_HEIGHT_M.
fn sheltered(p0: vec3<f32>, ray: vec4<f32>) -> bool {
    let p1 = p0 + vec3<f32>(ray.x, ray.w, ray.y) * (SHELTER_HEIGHT_M / ray.w);
    for (var i = 0u; i < params.flags.x; i += 1u) {
        if segment_hits(blockers[i], p0, p1) {
            return true;
        }
    }
    return false;
}

@compute @workgroup_size(8, 8, 1)
fn shelter_mask(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = params.grid.size;
    if id.x >= size.x || id.y >= size.y {
        return;
    }
    let k = id.y * size.x + id.x;
    let centre = params.grid.origin + (vec2<f32>(id.xy) + 0.5) * params.grid.cell;
    let p0 = vec3<f32>(centre.x, params.terrain.y + cells[k].x, centre.y);
    shelter[k] = vec2<f32>(
        select(1.0, 0.0, sheltered(p0, params.rain)),
        select(1.0, 0.0, sheltered(p0, params.snow_ray)),
    );
}
