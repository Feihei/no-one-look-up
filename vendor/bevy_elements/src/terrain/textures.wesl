// The ground's detail textures (docs/research/ground-textures.md): one ground layer's maps in a
// texture array, sampled without visible repetition by hex-tiling (Mikkelsen 2022, "Practical
// Real-Time Hex-Tiling", JCGT 11(3)) and decoded into an albedo multiplier, a height, a slope, a
// roughness multiplier and an occlusion multiplier.
//
// Layer `2k` of the array is the albedo detail (RGB, `value * 64 / 255`) and the height (A);
// layer `2k + 1` is the tangent normal's u and v (RG, `(c * 127 + 128) / 255`), the roughness
// detail (B, `value * 128 / 255`) and the occlusion detail (A, `value * 128 / 255`, mean 1). u
// runs along world X and v along world Z.

import bevy_elements::noise::hash33;

// The hex grid's scale in texture tiles: Mikkelsen's `2√3`, so a hex spans about a third of the
// texture and three taps from three places in it meet at every point.
const HEX_SCALE: f32 = 3.4641016;
// How much a tap's height wins the blend between the hexes, 0–1 (Mikkelsen's fall-off contrast
// β, on the height rather than the luminance): the taller grain is in front at a hex's edge.
const HEX_CONTRAST: f32 = 0.6;
// The decoded albedo multiplier of a stored 64, and the roughness and occlusion multipliers of a
// stored 128: exactly one.
const ALBEDO_DECODE: f32 = 255.0 / 64.0;
const MULTIPLIER_DECODE: f32 = 255.0 / 128.0;
// The steepest slope a detail normal may give, so a grazing texel cannot fold the normal over.
const MAX_SLOPE: f32 = 2.0;
// The smallest share (`w⁷`, normalised over the three hexes) a hex tap needs to be fetched at
// all. The tap's height can raise its weight against another tap's by at most four times (the
// factor runs from 0.4 to 1.6), so a tap below this ends with at most 0.5 % of the blend. Over
// the hex plane that leaves 1.8 taps of 3 on average, and one in 37 % of the pixels.
const TAP_MIN: f32 = 0.00125;
// A texel of the second array layer that decodes to no change: a flat normal, roughness and
// occlusion ×1. A tap that is not fetched holds it, at zero weight.
const NEUTRAL_SURFACE: vec4<f32> = vec4<f32>(128.0 / 255.0, 128.0 / 255.0, 128.0 / 255.0, 128.0 / 255.0);
// The relief (the normal, roughness and occlusion of the second array layer) is fetched only
// where the pixel resolves it: its mips have averaged it flat by the time the footprint on the
// ground grows from the first of these to the second, m, and it fades out in between. Its
// occlusion is a multiplier with mean 1, so the fade moves no mean.
const RELIEF_RESOLVED_M: f32 = 0.02;
const RELIEF_LOST_M: f32 = 0.06;
// The albedo detail, likewise: by a footprint of half a metre its mips are its mean (1), so it
// fades out over the first to the second of these, m, and the layer is not sampled beyond.
const ALBEDO_RESOLVED_M: f32 = 0.25;
const ALBEDO_LOST_M: f32 = 0.5;

// The three hex-grid vertices around a point and their barycentric weights.
struct HexGrid {
    v1: vec2<i32>,
    v2: vec2<i32>,
    v3: vec2<i32>,
    w: vec3<f32>,
}

// One tap: its coordinate in the texture and its orientation.
struct HexTap {
    st: vec2<f32>,
    rot: mat2x2<f32>,
}

// One ground layer's maps at a point, decoded.
struct LayerMaps {
    // The albedo multiplier, mean 1.
    albedo: vec3<f32>,
    // The relief's height, 0–1, mean 0.5.
    height: f32,
    // The relief's slope in the world, (∂h/∂x, ∂h/∂z).
    slope: vec2<f32>,
    // The roughness multiplier, mean 1.
    rough: f32,
    // The occlusion multiplier of the ambient light, mean 1.
    occlusion: f32,
}

// The triangle of the hex grid's vertices around `st` (texture tiles), Mikkelsen's
// `TriangleGrid`: the point is skewed into a simplex grid, and its barycentric weights are the
// weights of the three hexes that meet there.
fn hex_grid(st: vec2<f32>) -> HexGrid {
    let s = st * HEX_SCALE;
    let skewed = vec2<f32>(s.x - 0.57735027 * s.y, 1.15470054 * s.y);
    let base = vec2<i32>(floor(skewed));
    let f = skewed - floor(skewed);
    let z = 1.0 - f.x - f.y;
    let up = select(0.0, 1.0, z <= 0.0);
    let flip = 2.0 * up - 1.0;
    let u = i32(up);
    var g: HexGrid;
    g.w = vec3<f32>(-z * flip, up - f.y * flip, up - f.x * flip);
    g.v1 = base + vec2<i32>(u, u);
    g.v2 = base + vec2<i32>(u, 1 - u);
    g.v3 = base + vec2<i32>(1 - u, u);
    return g;
}

// A hex vertex's centre in texture tiles (Mikkelsen's `MakeCenST`).
fn hex_centre(v: vec2<i32>) -> vec2<f32> {
    let p = vec2<f32>(v);
    return vec2<f32>(p.x + 0.5 * p.y, 0.8660254 * p.y) / HEX_SCALE;
}

// The tap of the hex at vertex `v` for ground layer `layer`: the texture in one of its eight
// orientations about the hex's centre (a swap of the axes and a sign on each, from three bits of
// the hash) and moved by a random offset, so no two hexes show the same piece of it the same way
// round. Eight orientations hide an isotropic texture's repetition as a free angle does, and
// cost a swizzle where the angle cost a sine and a cosine per tap (Deviation 19).
fn hex_tap(v: vec2<i32>, layer: u32, st: vec2<f32>) -> HexTap {
    let h = hash33(vec3<i32>(v, i32(layer)));
    let k = u32(h.z * 8.0);
    let swap = (k & 1u) != 0u;
    let signs = vec2<f32>(select(1.0, -1.0, (k & 2u) != 0u), select(1.0, -1.0, (k & 4u) != 0u));
    var tap: HexTap;
    tap.rot = mat2x2<f32>(
        select(vec2<f32>(1.0, 0.0), vec2<f32>(0.0, 1.0), swap) * signs,
        select(vec2<f32>(0.0, 1.0), vec2<f32>(1.0, 0.0), swap) * signs,
    );
    let centre = hex_centre(v);
    tap.st = tap.rot * (st - centre) + centre + h.xy;
    return tap;
}

// A tap's normal map texel as a slope in the world: the tangent normal's (u, v) over its
// height, turned back by the tap's orientation (the slope is a gradient, so it turns by the
// transpose).
fn tap_slope(texel: vec4<f32>, rot: mat2x2<f32>) -> vec2<f32> {
    let n = (texel.rg * 255.0 - vec2<f32>(128.0)) / 127.0;
    let nz = sqrt(max(1.0 - dot(n, n), 0.0));
    let slope = -n / max(nz, 1.0 / MAX_SLOPE);
    return transpose(rot) * slope;
}

// The maps that decode to no change.
fn neutral_maps() -> LayerMaps {
    return LayerMaps(vec3<f32>(1.0), 0.5, vec2<f32>(0.0), 1.0, 1.0);
}

// The running sums of a layer's hex taps, each weighted (`add_tap`).
struct TapSums {
    colour: vec3<f32>,
    elevation: f32,
    gradient: vec2<f32>,
    surface: vec2<f32>,
    weight: f32,
}

// Add the tap of the hex at vertex `v` (barycentric `w⁷` share `bary7`) to `sums`: its albedo
// and height, and where `relief` is above zero its normal, roughness and occlusion, all weighted
// by `bary7` times the tap's height (Mikkelsen's β). One tap at a time, so a layer holds its
// running sums and not all three taps' texels: the terrain's fragment shader is long, and the
// registers it needs at its peak decide how many pixels run at once over all of it.
fn add_tap(
    maps: texture_2d_array<f32>,
    maps_sampler: sampler,
    layer: u32,
    v: vec2<i32>,
    bary7: f32,
    st: vec2<f32>,
    dx: vec2<f32>,
    dy: vec2<f32>,
    relief: f32,
    sums: TapSums,
) -> TapSums {
    let tap = hex_tap(v, layer, st);
    let albedo_layer = i32(2u * layer);
    let a = textureSampleGrad(maps, maps_sampler, tap.st, albedo_layer, tap.rot * dx, tap.rot * dy);
    let k = bary7 * mix(1.0, 2.0 * a.a, HEX_CONTRAST);
    var out = sums;
    out.colour = out.colour + k * a.rgb;
    out.elevation = out.elevation + k * a.a;
    out.weight = out.weight + k;
    if relief > 0.0 {
        let b = textureSampleGrad(maps, maps_sampler, tap.st, albedo_layer + 1, tap.rot * dx, tap.rot * dy);
        out.gradient = out.gradient + k * tap_slope(b, tap.rot);
        out.surface = out.surface + k * b.ba;
    } else {
        out.surface = out.surface + k * NEUTRAL_SURFACE.ba;
    }
    return out;
}

// Ground layer `layer`'s detail at `st` (texture tiles), with the coordinate's screen
// derivatives `dx` and `dy` (texture tiles a pixel) and the pixel's footprint on the ground (m):
// the hex taps of each of the layer's two array layers, weighted by Mikkelsen's `w^γ` (γ = 7)
// times the tap's height, so the hexes' seams are narrow and the taller grain is in front. Only
// what can show is fetched: the taps that can carry weight (`TAP_MIN`), the relief where it is
// resolved, the albedo detail where its mips are not yet their mean.
fn layer_maps(
    maps: texture_2d_array<f32>,
    maps_sampler: sampler,
    layer: u32,
    st: vec2<f32>,
    dx: vec2<f32>,
    dy: vec2<f32>,
    footprint_m: f32,
) -> LayerMaps {
    let detail = 1.0 - smoothstep(ALBEDO_RESOLVED_M, ALBEDO_LOST_M, footprint_m);
    if detail <= 0.0 {
        return neutral_maps();
    }
    let relief = 1.0 - smoothstep(RELIEF_RESOLVED_M, RELIEF_LOST_M, footprint_m);
    let g = hex_grid(st);
    // w⁷ by multiplication: a `pow` with a computed exponent fails silently in WESL.
    let w2 = g.w * g.w;
    let w4 = w2 * w2;
    let bary7 = w4 * w2 * g.w;
    let cut = TAP_MIN * (bary7.x + bary7.y + bary7.z);
    var sums = TapSums(vec3<f32>(0.0), 0.0, vec2<f32>(0.0), vec2<f32>(0.0), 0.0);
    if bary7.x > cut {
        sums = add_tap(maps, maps_sampler, layer, g.v1, bary7.x, st, dx, dy, relief, sums);
    }
    if bary7.y > cut {
        sums = add_tap(maps, maps_sampler, layer, g.v2, bary7.y, st, dx, dy, relief, sums);
    }
    if bary7.z > cut {
        sums = add_tap(maps, maps_sampler, layer, g.v3, bary7.z, st, dx, dy, relief, sums);
    }
    let norm = 1.0 / max(sums.weight, 1e-12);
    let shown = detail * relief;
    let surface = sums.surface * (norm * MULTIPLIER_DECODE);
    var d: LayerMaps;
    d.albedo = mix(vec3<f32>(1.0), sums.colour * (norm * ALBEDO_DECODE), detail);
    d.height = mix(0.5, sums.elevation * norm, detail);
    d.slope = sums.gradient * (norm * shown);
    d.rough = mix(1.0, surface.x, shown);
    d.occlusion = mix(1.0, surface.y, shown);
    return d;
}

// How sharply the ground layers' heights decide their blend: a weight is scaled by
// `exp(HEIGHT_BLEND · (h − ½))`, so where two layers meet, the taller texel's layer shows
// (grass blades standing out of the soil) instead of a cross-fade.
const HEIGHT_BLEND: f32 = 8.0;
// A ground layer's textures are sampled only above this weight.
const DETAIL_GATE: f32 = 0.01;
// The height blend is a nonlinear function of *filtered* heights: where the pixel no longer
// resolves the heights' features (blades, crumbs, pebbles: millimetres to a few centimetres),
// it flips each pixel between two layers by a hair's difference in their mipped heights, a red
// and green dither that sparkles in motion. Its sharpness fades to a linear blend as the pixel's
// footprint on the ground grows from the first to the second of these, m.
const HEIGHT_BLEND_RESOLVED_M: f32 = 0.01;
const HEIGHT_BLEND_LOST_M: f32 = 0.03;

// The height blend's sharpness for a pixel whose footprint on the ground is `footprint_m`.
fn height_sharpness(footprint_m: f32) -> f32 {
    return HEIGHT_BLEND * (1.0 - smoothstep(HEIGHT_BLEND_RESOLVED_M, HEIGHT_BLEND_LOST_M, footprint_m));
}

// A ground layer's lift in the height blend: its weight `weight` less the gate, scaled by
// `exp(sharpness · (h − ½))` for its height `h` (`height_sharpness`). The layers' shares are
// their lifts scaled to the ground's own share (`ground_textured`), so the sum is unchanged, a
// layer's share is continuous where its sampling starts, and equal heights change nothing but
// the gate's own share.
fn height_lift(weight: f32, h: f32, sharpness: f32) -> f32 {
    return max(weight - DETAIL_GATE, 0.0) * exp(sharpness * (h - 0.5));
}
