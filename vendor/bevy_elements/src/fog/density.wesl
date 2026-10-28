// Ground-fog density for Bevy's FogVolume: extinction coefficient in m⁻¹ in `.r` of a
// 128×32×128 rgba16float texture that spans the volume's unit cube. See docs/research/fog.md
// and docs/research/precipitation.md.
//
//   σ(p) = σ_haze · exp(−h/H) · (0.6 + 0.4 · f(p − w t))
//        + σ_p · clamp(g(x_ref) / g_nom, 0, 1) · (0.4 + 1.2 · f_s(x_ref − w t))
//
// σ_haze + σ_p = σ_V = 3.912 / visibility: the weather's visibility stays the one authority and
// precipitation only redistributes part of it into shafts. x_ref follows the slanted fall of
// the rain, and g is the precipitation channel of the weather map.

import bevy_elements::noise::fbm3;

struct FogParams {
    // xyz: world position of the volume's minimum corner, m. w: 1 / max |fbm| (noise → [0, 1]).
    origin: vec4<f32>,
    // xyz: volume size, m. w: σ_haze, m⁻¹.
    extent: vec4<f32>,
    // x: scale height H (m), y: ground height (m), zw: wind offset w·t (world XZ, m).
    shape: vec4<f32>,
    // xyz: texels per axis. w: unused.
    size: vec4<u32>,
    // x: σ_p, m⁻¹. y: shaft slant per m/s of wind, s/m. z: weather map extent, m. w: g_nom.
    precip: vec4<f32>,
    // xy: wind at 10 m, m/s. zw: weather map origin, world XZ, m.
    slant: vec4<f32>,
}

@group(0) @binding(0) var<uniform> params: FogParams;
@group(0) @binding(1) var density: texture_storage_3d<rgba16float, write>;
@group(0) @binding(2) var weather_map: texture_2d<f32>;
@group(0) @binding(3) var weather_sampler: sampler;

// fBm wavelength, m.
const WAVELENGTH_M: f32 = 80.0;
// Shaft fBm base wavelength, m: octaves at 120 and 60 m, at least 32 texels across.
const SHAFT_WAVELENGTH_M: f32 = 120.0;

@compute @workgroup_size(4, 4, 4)
fn fog_density(@builtin(global_invocation_id) id: vec3<u32>) {
    if any(id >= params.size.xyz) {
        return;
    }
    let uvw = (vec3<f32>(id) + 0.5) / vec3<f32>(params.size.xyz);
    let p = params.origin.xyz + uvw * params.extent.xyz;
    let h = max(p.y - params.shape.y, 0.0);
    let q = (p - vec3<f32>(params.shape.z, 0.0, params.shape.w)) / WAVELENGTH_M;
    let f = clamp(0.5 + 0.5 * fbm3(q, 2u) * params.origin.w, 0.0, 1.0);
    let haze = params.extent.w * exp(-h / params.shape.x) * (0.6 + 0.4 * f);

    // Precipitation shafts. A drop at height y left the cloud upwind, so the map and the noise
    // are sampled where the drop was at the volume's mid height.
    let y_ref = params.origin.y + 0.5 * params.extent.y;
    let x_ref = p.xz + params.slant.xy * params.precip.y * (p.y - y_ref);
    let g = textureSampleLevel(
        weather_map,
        weather_sampler,
        (x_ref - params.slant.zw) / params.precip.z,
        0.0,
    ).g;
    let coverage = clamp(g / max(params.precip.w, 1e-4), 0.0, 1.0);
    let qs = vec3<f32>(
        (x_ref.x - params.shape.z) / SHAFT_WAVELENGTH_M,
        0.37,
        (x_ref.y - params.shape.w) / SHAFT_WAVELENGTH_M,
    );
    let f_s = clamp(0.5 + 0.5 * fbm3(qs, 2u) * params.origin.w, 0.0, 1.0);
    let shafts = params.precip.x * coverage * (0.4 + 1.2 * f_s);

    textureStore(density, vec3<i32>(id), vec4<f32>(haze + shafts, 0.0, 0.0, 0.0));
}
