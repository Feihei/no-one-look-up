// Heightmap import: resample the red channel of a texture to the terrain's resolution with
// hand-written taps (no sampler: R32Float is not filterable and a kernel needs none). See
// docs/research/terrain.md ("Heightmap import"). Built twice: with UINT_SOURCE for R16Uint.
//
// Per axis, when the source is smaller than the destination the taps are bilinear (two texels);
// when it is larger, `min(ceil(src / dst), 4)` taps at the centres of equal parts of the cell's
// footprint are averaged.

struct HeightmapParams {
    // x, y: destination cells per axis.
    size: vec4<u32>,
    // x: height at value 0 (m). y: height at value 1 (m).
    range: vec4<f32>,
}

@group(0) @binding(0) var<uniform> params: HeightmapParams;
@group(0) @binding(1) var<storage, read_write> bedrock: array<f32>;
@group(0) @binding(2) var<storage, read_write> sediment: array<f32>;
@if(UINT_SOURCE)
@group(0) @binding(3) var source: texture_2d<u32>;
@if(!UINT_SOURCE)
@group(0) @binding(3) var source: texture_2d<f32>;

fn value_at(texel: vec2<i32>) -> f32 {
    let dims = vec2<i32>(textureDimensions(source));
    let t = clamp(texel, vec2<i32>(0), dims - 1);
    @if(UINT_SOURCE)
    return f32(textureLoad(source, t, 0).r) / 65535.0;
    @if(!UINT_SOURCE)
    return textureLoad(source, t, 0).r;
}

// Number of taps along one axis for a source/destination ratio.
fn tap_count(ratio: f32) -> i32 {
    if ratio <= 1.0 {
        return 2;
    }
    return i32(min(ceil(ratio), 4.0));
}

// Tap `k` of destination cell `d` along one axis: (source texel, weight).
fn axis_tap(d: u32, ratio: f32, k: i32) -> vec2<f32> {
    if ratio <= 1.0 {
        let g = (f32(d) + 0.5) * ratio - 0.5;
        let base = floor(g);
        let f = g - base;
        return vec2<f32>(base + f32(k), select(1.0 - f, f, k == 1));
    }
    let n = f32(tap_count(ratio));
    let centre = f32(d) * ratio + (f32(k) + 0.5) / n * ratio;
    return vec2<f32>(floor(centre), 1.0 / n);
}

@compute @workgroup_size(8, 8, 1)
fn terrain_heightmap(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= params.size.x || id.y >= params.size.y {
        return;
    }
    let src = vec2<f32>(textureDimensions(source));
    let dst = vec2<f32>(params.size.xy);
    let ratio = src / dst;
    let nx = tap_count(ratio.x);
    let ny = tap_count(ratio.y);
    var sum = 0.0;
    for (var j = 0; j < ny; j += 1) {
        let ty = axis_tap(id.y, ratio.y, j);
        for (var i = 0; i < nx; i += 1) {
            let tx = axis_tap(id.x, ratio.x, i);
            sum += tx.y * ty.y * value_at(vec2<i32>(i32(tx.x), i32(ty.x)));
        }
    }
    let k = id.y * params.size.x + id.x;
    bedrock[k] = mix(params.range.x, params.range.y, sum);
    sediment[k] = 0.0;
}
