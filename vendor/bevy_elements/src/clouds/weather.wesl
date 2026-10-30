// The cloud weather map: a camera-centred, 32 km periodic coverage field, thresholded by a
// per-frame histogram so the area fraction of cloud equals `Weather::cloud_cover`.
// See docs/research/clouds.md ("Weather map").
//
//   cloud_histogram_clear   zero the 256 bins
//   cloud_field             histogram of the field, 8 bit
//   cloud_threshold         scan the bins from the top to the quantile 1 − cover
//   cloud_weather           R coverage, G precipitation, B type

import bevy_elements::noise::fbm3_periodic;
import bevy_elements::clouds::common::CloudParams;

@group(0) @binding(0) var<uniform> params: CloudParams;
@group(0) @binding(1) var<storage, read_write> histogram: array<atomic<u32>, 256>;
@group(0) @binding(2) var<storage, read_write> threshold: array<f32, 4>;
@group(0) @binding(3) var weather_out: texture_storage_2d<rgba8unorm, write>;

var<workgroup> local_bins: array<atomic<u32>, 256>;

/// Texels per side of the weather map.
const TEXELS: u32 = 512u;
/// Soft edge of the coverage threshold, field units.
const SOFT_EDGE: f32 = 0.18;
/// Width over which the dome factor (the map's A channel) rises from 0 at the cloud edge to 1,
/// field units.
// [PATCH heads-up] 0.37 -> 0.30. The upstream value spreads the dome across
// more of the blob, so the full-height core is wide and the dome reads as a
// flat disc with a bump; 0.30 keeps full height only in a blob's core.
const DOME_WIDTH: f32 = 0.30;
/// Largest |fbm| of the 3-octave periodic Perlin field on its z slice, measured by
/// `the_noise_normalisation_constants_match_the_noise`; maps the noise onto [0, 1].
const FIELD_MAX: f32 = 0.385;

/// The coverage noise at a texel, [0, 1]: 3 octaves of 2D Perlin with wavelengths 8, 4 and 2 km
/// on a lattice that wraps every 32 km (4 base cells), evaluated at the texel's world position
/// minus the cloud drift. Periodic in the map extent, so its histogram does not depend on where
/// the window sits and the repeat addressing of the map has no seam.
fn coverage_field(texel: vec2<u32>) -> f32 {
    let xz = params.weather.xy + (vec2<f32>(texel) + 0.5) * (params.weather.z / f32(TEXELS));
    let q = (xz - params.drift.xy) / 8000.0;
    let n = fbm3_periodic(vec3<f32>(q.x, q.y, 0.31), 4, 3);
    return clamp(0.5 + 0.5 * n / FIELD_MAX, 0.0, 1.0);
}

@compute @workgroup_size(256)
fn cloud_histogram_clear(@builtin(local_invocation_index) i: u32) {
    atomicStore(&histogram[i], 0u);
}

@compute @workgroup_size(8, 8, 1)
fn cloud_field(
    @builtin(global_invocation_id) id: vec3<u32>,
    @builtin(local_invocation_index) lane: u32,
) {
    if all(id.xy < vec2<u32>(TEXELS)) {
        let bin = min(u32(coverage_field(id.xy) * 256.0), 255u);
        atomicAdd(&local_bins[bin], 1u);
    }
    workgroupBarrier();
    for (var i = lane; i < 256u; i += 64u) {
        let count = atomicLoad(&local_bins[i]);
        if count > 0u {
            atomicAdd(&histogram[i], count);
        }
    }
}

@compute @workgroup_size(1)
fn cloud_threshold() {
    let target_count = params.layer.w * f32(TEXELS * TEXELS);
    var cumulative = 0.0;
    var theta = 256.0;
    for (var k = 255; k >= 0; k -= 1) {
        let count = f32(atomicLoad(&histogram[k]));
        if cumulative + count >= target_count {
            // Bin k spans field values [k, k + 1) / 256; the mass fills it from the top.
            theta = f32(k) + 1.0 - (target_count - cumulative) / max(count, 1.0);
            break;
        }
        cumulative += count;
    }
    threshold[0] = theta;
}

@compute @workgroup_size(8, 8, 1)
fn cloud_weather(@builtin(global_invocation_id) id: vec3<u32>) {
    if any(id.xy >= vec2<u32>(TEXELS)) {
        return;
    }
    let cover = params.layer.w;
    var coverage = clamp(
        0.5 + (coverage_field(id.xy) * 256.0 - threshold[0]) / (SOFT_EDGE * 256.0),
        0.0,
        1.0,
    );
    if cover >= 0.995 {
        coverage = 1.0;
    } else if cover <= 0.005 {
        coverage = 0.0;
    }
    let precipitation = params.weather.w * coverage;
    let dome = clamp((coverage_field(id.xy) * 256.0 - threshold[0]) / (DOME_WIDTH * 256.0), 0.0, 1.0);
    textureStore(
        weather_out,
        vec2<i32>(id.xy),
        vec4<f32>(coverage, precipitation, params.layer.z, dome),
    );
}
