// The water's tileable normal map, generated once (docs/research/water-rendering.md, "Generated
// normal map"): 512² `rgba8unorm`, rgb = a tangent-space normal `(−K ∇h, 1)` encoded `n·0.5 + 0.5`,
// a = a tileable fbm for the foam patchiness. Every mip level is the same kernel dispatched with
// the octaves whose wavelength is below the level's two texels left out (a band-limited analytic
// mip: no downsample pass and no energy bleeding across the tile seam).

import bevy_elements::noise::{perlin3_periodic, fbm3_periodic};

struct NormalParams {
    // x: mip level. y: edge of that level in texels. z: octaves of the height kept.
    // w: octaves of the foam fbm kept.
    level: vec4<u32>,
    // x: the slope gain K that makes the rms slope of mip 0 the Cox & Munk value.
    gain: vec4<f32>,
}

@group(0) @binding(0) var<uniform> params: NormalParams;
@group(0) @binding(1) var out_tex: texture_storage_2d<rgba8unorm, write>;

// Texels of the full-resolution map: the height is a function of the texel on this lattice.
const BASE_TEXELS: i32 = 512;
// Height amplitude ratio per octave: 0.5, so that every octave's *slope* (amplitude ×
// frequency) is the same — a flat slope spectrum, as a surface's appearance depends on the slope
// and not on the height (Lagae et al. 2010 §4). The map's total rms slope is unchanged (Cox &
// Munk 1954); only its distribution over the spectrum changes
// (`docs/research/water-surface.md`, Model 1).
const OCTAVE_GAIN: f32 = 0.5;
// Lattice cells of the base octave across the tile: 16, so the map's wavelengths run 0.25 m to
// 6.25 cm (three octaves, the finest at the kernel's four-texel limit). A sheltered body has no
// fetch and no metre-scale wind chop; the metre band is the wave field's (`waves`, fitted to
// Airy's dispersion), and duplicating it here is what smeared the reflection into blobs
// (`water-surface.md`, Model 2).
const BASE_CELLS: f32 = 16.0;
const FOAM_CELLS: f32 = 4.0;
const FOAM_STRETCH: f32 = 2.5;

// Height at integer texel `t` of the full-resolution map, wrapped. Octave `o` has `16·2^o` lattice
// cells across the tile (wavelength `32/2^o` texels) and the same period, so the sum tiles.
fn height(t: vec2<i32>) -> f32 {
    let w = vec2<i32>(t.x & (BASE_TEXELS - 1), t.y & (BASE_TEXELS - 1));
    let u = (vec2<f32>(w) + 0.5) / f32(BASE_TEXELS);
    var h = 0.0;
    var amplitude = 1.0;
    for (var o = 0u; o < params.level.z; o += 1u) {
        let cells = BASE_CELLS * f32(1u << o);
        h += amplitude * perlin3_periodic(vec3<f32>(u * cells, 0.5 + f32(o)), i32(cells));
        amplitude *= OCTAVE_GAIN;
    }
    return h;
}

@compute @workgroup_size(8, 8, 1)
fn water_normals(@builtin(global_invocation_id) id: vec3<u32>) {
    let edge = params.level.y;
    if id.x >= edge || id.y >= edge {
        return;
    }
    // The texel's centre on the full-resolution lattice: level m texel i covers 2^m texels.
    let stride = 512u / edge;
    let centre = vec2<i32>(id.xy * stride) + i32(stride / 2u);
    let hx = height(centre + vec2<i32>(1, 0)) - height(centre - vec2<i32>(1, 0));
    let hy = height(centre + vec2<i32>(0, 1)) - height(centre - vec2<i32>(0, 1));
    let n = normalize(vec3<f32>(-params.gain.x * hx, -params.gain.x * hy, 1.0));
    let u = (vec2<f32>(id.xy) + 0.5) / f32(edge);
    // The periodic fbm is narrow (p10–p90 of 0.41–0.58 once centred), so it is stretched to use
    // the range the patchiness thresholds of `water.wesl` select from.
    let foam = 0.5 + 0.5 * clamp(FOAM_STRETCH * fbm3_periodic(vec3<f32>(u * FOAM_CELLS, 7.3), i32(FOAM_CELLS), params.level.w), -1.0, 1.0);
    textureStore(out_tex, vec2<i32>(id.xy), vec4<f32>(n * 0.5 + 0.5, clamp(foam, 0.0, 1.0)));
}
