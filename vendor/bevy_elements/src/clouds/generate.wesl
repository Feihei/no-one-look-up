// Cloud noise textures, generated once by compute (the recipe of Hillaire's TileableVolumeNoise,
// which realises Schneider 2015). Every channel tiles with the texture period.
//
//   base   128³: R Perlin-Worley, G/B/A Worley fBm at 2/4/8-cell octaves (cells ×4)
//   detail  32³: R/G/B Worley fBm at 1/2/4/8-cell octaves (cells ×2)
//
// See docs/research/clouds.md ("Noise textures"). One kernel per texture; the base texture is
// produced in four z-slabs over four frames (`params.tail.x` is the first slice of the slab).

import bevy_elements::noise::{worley3, fbm3_periodic};
import bevy_elements::clouds::common::CloudParams;

@group(0) @binding(0) var<uniform> params: CloudParams;
@group(0) @binding(1) var base_out: texture_storage_3d<rgba8unorm, write>;
@group(0) @binding(2) var detail_out: texture_storage_3d<rgba8unorm, write>;

/// Largest |fbm| of the 3-octave periodic Perlin sum, measured by
/// `the_noise_normalisation_constants_match_the_noise` over 10⁵ points.
const PERLIN_FBM_MAX: f32 = 0.55;

/// Inverted Worley noise with `k` cells across the tile; `x` is in [0, 1).
fn inverted_worley(x: vec3<f32>, k: i32) -> f32 {
    return 1.0 - min(worley3(x * f32(k), k), 1.0);
}

@compute @workgroup_size(4, 4, 4)
fn cloud_noise_base(@builtin(global_invocation_id) id: vec3<u32>) {
    let z = id.z + params.tail.x;
    if id.x >= 128u || id.y >= 128u || z >= 128u {
        return;
    }
    let texel = vec3<u32>(id.x, id.y, z);
    let uvw = (vec3<f32>(texel) + 0.5) / 128.0;
    let w8 = inverted_worley(uvw, 8);
    let w16 = inverted_worley(uvw, 16);
    let w32 = inverted_worley(uvw, 32);
    let w56 = inverted_worley(uvw, 56);
    let w64 = inverted_worley(uvw, 64);
    let perlin = clamp(0.5 + 0.5 * fbm3_periodic(uvw * 8.0, 8, 3) / PERLIN_FBM_MAX, 0.0, 1.0);
    let dilated = 0.625 * w8 + 0.25 * w32 + 0.125 * w56;
    let r = dilated + perlin * (1.0 - dilated);
    let g = 0.625 * w8 + 0.25 * w16 + 0.125 * w32;
    let b = 0.625 * w16 + 0.25 * w32 + 0.125 * w64;
    let a = 0.75 * w32 + 0.25 * w64;
    textureStore(base_out, vec3<i32>(texel), vec4<f32>(r, g, b, a));
}

@compute @workgroup_size(4, 4, 4)
fn cloud_noise_detail(@builtin(global_invocation_id) id: vec3<u32>) {
    if any(id >= vec3<u32>(32u)) {
        return;
    }
    let uvw = (vec3<f32>(id) + 0.5) / 32.0;
    let w2 = inverted_worley(uvw, 2);
    let w4 = inverted_worley(uvw, 4);
    let w8 = inverted_worley(uvw, 8);
    let w16 = inverted_worley(uvw, 16);
    let r = 0.625 * w2 + 0.25 * w4 + 0.125 * w8;
    let g = 0.625 * w4 + 0.25 * w8 + 0.125 * w16;
    let b = 0.75 * w8 + 0.25 * w16;
    textureStore(detail_out, vec3<i32>(id), vec4<f32>(r, g, b, 1.0));
}
