// Procedural terrain generator: one thread per cell. See docs/research/terrain.md ("Procedural
// generator"). The CPU transliteration is `model/mirror.rs`; keep them in step.
//
// With p = local_xz / F and octave k sampling perlin3(R_k p 2^k + v_k, 0.5 + k) (R_k the golden
// angle rotation, v_k a seeded whole-cell offset), two sums are formed:
//   f = Σ g^k n_k / Σ g^k                          (fBm)
//   r = Σ g^k s_k / Σ g^k, s_k = w_k (1 − |n_k|)²  (ridged multifractal, w_0 = 1, w_k = clamp(2 s_{k−1}, 0, 1))
// and the surface is  base + R · ((1 − q) u_f + q u_r)  with u_f, u_r the unit-range maps of f, r.

import bevy_elements::noise::perlin3;

struct GenParams {
    // x, y: cells per axis. z: octave count. w: the seed's contribution to the octave offsets.
    size: vec4<u32>,
    // x: cell edge (m). y: feature size F (m). z: relief R (m). w: ridged mix q.
    shape: vec4<f32>,
    // x: base height (m). y: initial sand depth (m).
    base: vec4<f32>,
    // x: fBm half range. y: ridged low end. z: ridged span. w: gain.
    consts: vec4<f32>,
    // x: lacunarity. y: octave rotation (rad).
    octave: vec4<f32>,
}

@group(0) @binding(0) var<uniform> params: GenParams;
@group(0) @binding(1) var<storage, read_write> bedrock: array<f32>;
@group(0) @binding(2) var<storage, read_write> sediment: array<f32>;

@compute @workgroup_size(8, 8, 1)
fn terrain_generate(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= params.size.x || id.y >= params.size.y {
        return;
    }
    let local = (vec2<f32>(id.xy) + 0.5) * params.shape.x;
    let p = local / params.shape.y;
    var f_sum = 0.0;
    var r_sum = 0.0;
    var norm = 0.0;
    var gain = 1.0;
    var weight = 1.0;
    for (var k = 0u; k < params.size.z; k += 1u) {
        let a = params.octave.y * f32(k);
        let s = sin(a);
        let c = cos(a);
        let q = vec2<f32>(c * p.x - s * p.y, s * p.x + c * p.y) * pow(params.octave.x, f32(k));
        let v = f32((params.size.w + k * 1031u) % 4096u);
        let n = perlin3(vec3<f32>(q + vec2<f32>(v), 0.5 + f32(k)));
        let ridge = max(1.0 - abs(n), 0.0);
        let s_k = weight * ridge * ridge;
        f_sum += gain * n;
        r_sum += gain * s_k;
        norm += gain;
        weight = clamp(2.0 * s_k, 0.0, 1.0);
        gain *= params.consts.w;
    }
    let f = f_sum / norm;
    let r = r_sum / norm;
    let u_f = clamp(0.5 + f / (2.0 * params.consts.x), 0.0, 1.0);
    let u_r = clamp((r - params.consts.y) / params.consts.z, 0.0, 1.0);
    let q_mix = params.shape.w;
    let unit = (1.0 - q_mix) * u_f + q_mix * u_r;
    let sand = params.base.y;
    let i = id.y * params.size.x + id.x;
    // The surface is the generated field; the sand rests on bedrock lowered by its depth.
    bedrock[i] = params.base.x + params.shape.z * unit - sand;
    sediment[i] = sand;
}
