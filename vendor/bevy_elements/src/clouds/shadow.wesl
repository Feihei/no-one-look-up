// The cloud shadow map: direct transmittance toward the light through the clouds for the plane
// y = 0, 1024² texels over an 8 km square centred on the camera. See docs/research/clouds.md
// ("Cloud shadow map and the sun's illuminance").
//
//   cloud_shadow          R transmittance, G optical depth (≤ 60), B precipitation darkening;
//                         every fourth texel's transmittance (texel 4k + 2 on each axis) also
//                         goes to `coarse`, the 256² grid the gameplay query and the lights'
//                         mean transmittance read back

import bevy_pbr::render::utils::interleaved_gradient_noise;
import bevy_elements::clouds::common::{CloudParams, cloud_medium};

@group(0) @binding(0) var<uniform> params: CloudParams;
@group(0) @binding(1) var weather_tex: texture_2d<f32>;
@group(0) @binding(2) var base_tex: texture_3d<f32>;
@group(0) @binding(3) var smp: sampler;
@group(0) @binding(4) var shadow_out: texture_storage_2d<rgba16float, write>;
@group(0) @binding(5) var<storage, read_write> coarse: array<f32>;

const TEXELS: u32 = 1024u;
/// Texels per coarse cell edge.
const COARSE_STRIDE: u32 = 4u;
const STEPS: u32 = 8u;
/// Lowest sine of the light's elevation the map is generated for (the layer would otherwise
/// lie beyond the weather map): arcsin 0.1 = 5.74°.
const MIN_LIGHT_SINE: f32 = 0.1;

/// (T, τ, g) toward the light at a shadow-map texel position (texel units, centre at +0.5).
fn shadow_at(texel: vec2<f32>) -> vec3<f32> {
    let q = params.shadow.xy + texel * (params.shadow.z / f32(TEXELS));
    var l = params.light.xyz;
    if l.y < MIN_LIGHT_SINE {
        let flat_len = length(l.xz);
        let flat_dir = select(vec2<f32>(1.0, 0.0), l.xz / max(flat_len, 1e-6), flat_len > 1e-4);
        let c = sqrt(1.0 - MIN_LIGHT_SINE * MIN_LIGHT_SINE);
        l = vec3<f32>(flat_dir.x * c, MIN_LIGHT_SINE, flat_dir.y * c);
    }
    // The pattern is fixed in space (frame 0): grain that does not shimmer.
    let jitter = interleaved_gradient_noise(floor(texel), 0u);
    let h_b = params.layer.x;
    let thickness = params.layer.y - h_b;
    let dh = thickness / f32(STEPS);
    let ds = dh / l.y;
    var tau = 0.0;
    var g_sum = 0.0;
    var s_sum = 0.0;
    for (var i = 0u; i < STEPS; i += 1u) {
        let h = h_b + (f32(i) + jitter) * dh;
        let m = cloud_medium(h, q + l.xz * (h / l.y), 0.0, params, base_tex, base_tex, weather_tex, smp);
        tau += (1.0 + m.g) * m.sigma * ds;
        g_sum += m.g * m.sigma;
        s_sum += m.sigma;
    }
    return vec3<f32>(exp(-tau), min(tau, 60.0), g_sum / max(s_sum, 1e-6));
}

@compute @workgroup_size(8, 8, 1)
fn cloud_shadow(@builtin(global_invocation_id) id: vec3<u32>) {
    if any(id.xy >= vec2<u32>(TEXELS)) {
        return;
    }
    let s = shadow_at(vec2<f32>(id.xy) + 0.5);
    textureStore(shadow_out, vec2<i32>(id.xy), vec4<f32>(s.x, s.y, s.z, 0.0));
    if all(id.xy % COARSE_STRIDE == vec2<u32>(COARSE_STRIDE / 2u)) {
        let cell = id.xy / COARSE_STRIDE;
        coarse[cell.y * (TEXELS / COARSE_STRIDE) + cell.x] = s.x;
    }
}
