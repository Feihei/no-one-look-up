// bevy_elements::paths — the path mask in a ground material (`docs/research/paths.md`).
//
// The mask is `rgba8unorm`, a distance field over the terrain's extent (the format the terrain's
// ground-texture array uses, which this material is proven to sample):
//   r  the **signed distance** to the nearest route's edge, over ±PATH_DISTANCE_RANGE_M
//   g  the kind plus the rut depth in its fraction (m / 0.5)
//   b  the lateral offset from the route's centre line as a fraction of the half width, -1..1
//   a  a per-texel **grain** 0-1: the near field's stone at the mask's own resolution (the edge
//      fade is derived from the half width instead)
// A distance field, not a coverage mask, so a 70 cm footpath's edge interpolates between texels
// instead of stepping at the grid (Green 2007, *Improved alpha-tested magnification for vector
// textures*; RTR4 §15.4.3).
//
// `lane` is the material's `path` uniform: xy the mask's minimum corner (metre frame), z its texel
// edge (m), w its texels a side. A w below 0.5 is no mask.

import bevy_elements::noise::perlin2_d;

struct PathSample {
    // 1 on the route's surface, 0 in the ground around it: a smoothstep over the edge.
    cover: f32,
    // 0 earth, 1 gravel, 2 stone, 3 snow track.
    kind: f32,
    // The route's rut depth at this point, m (0 without ruts or off a wheel's band).
    rut: f32,
    // 1 where a cart track's crown keeps its turf between the wheels.
    crown: f32,
    // How far the route's surface is pressed below the untouched ground here, m (0 outside).
    dish: f32,
    // The per-texel grain the mask carries, 0-1.
    grain: f32,
    // The route's cross-section slope, m/m (positive rising outward), and the world XZ direction
    // it rises in (the distance field's own gradient, so it follows a curving route). It is the
    // dish *and* the wheel ruts: a rut is a groove, so its slope dips on the inner side and climbs
    // on the outer, which is what a low sun rakes.
    wall: f32,
    away: vec2<f32>,
}

fn path_none() -> PathSample {
    var none: PathSample;
    none.cover = 0.0;
    none.kind = 0.0;
    none.rut = 0.0;
    none.crown = 0.0;
    none.grain = 0.5;
    none.dish = 0.0;
    none.wall = 0.0;
    none.away = vec2<f32>(0.0);
    return none;
}

// The mask's channel ranges, as the CPU encodes them (`paths.rs`).
const PATH_DISTANCE_RANGE_M: f32 = 8.0;
const PATH_FADE_RANGE_M: f32 = 2.0;
// The rut depth the green lane's fraction spans, m (1 mm of it, 8 bits).
const PATH_RUT_RANGE_M: f32 = 0.25;

// How deep a route is pressed below the untouched ground at its centre, m, as a share of the
// path's half width: a trodden lane is a dish a few centimetres deep, a road is graded and flatter.
// Without it a route is a painted band; with it the ground at the verge stands a little proud and
// the low sun rakes the shallow wall.
const PATH_DISH_LANE_M: f32 = 0.030;
const PATH_DISH_ROAD_M: f32 = 0.018;
const PATH_DISH_STONE_M: f32 = 0.010;

// Where a vehicle's wheels run, as a fraction of the route's half width (a cart's track is 1.5 m
// apart on a 2 m half width's 0.6), and the band's width in the same units.
const PATH_RUT_OFFSET: f32 = 0.6;
const PATH_RUT_SIGMA: f32 = 0.10;

// The path at world XZ `xz`. `lane.w` below 0.5 is no mask.
fn path_at(
    mask: texture_2d<f32>,
    smp: sampler,
    xz: vec2<f32>,
    lane: vec4<f32>,
    route_half_width_m: f32,
) -> PathSample {
    if lane.w < 0.5 {
        return path_none();
    }
    // A texel (i, j) is centred at origin + (i + 0.5, j + 0.5) x texel, which is what `uv` of
    // (i + 0.5) / N means to a bilinear sampler.
    let uv = (xz - lane.xy) / (lane.z * lane.w);
    if any(uv < vec2<f32>(0.0)) || any(uv > vec2<f32>(1.0)) {
        return path_none();
    }
    let texel = textureSampleLevel(mask, smp, uv, 0.0);
    // The 8-bit channels back to metres and shares (see the header): green is
    // (kind + depth / PATH_RUT_RANGE_M) / 4, so one byte carries both.
    let distance = (texel.x - 0.5) * PATH_DISTANCE_RANGE_M;
    // The edge fade, from the half width (a third of the route's width): derived, so the mask's
    // alpha channel is free for the grain.
    let half_width = max(route_half_width_m, 0.3);
    let fade = max(half_width * 0.33, 0.12);
    let packed = texel.y * 4.0;
    var out: PathSample;
    // The distance is negative inside the route, so the coverage grows inward from the edge.
    out.cover = smoothstep(0.0, -fade, distance);
    out.grain = texel.w;
    out.kind = floor(packed);
    let depth = (packed - out.kind) * PATH_RUT_RANGE_M;
    // The lateral offset, back from 0..1 to a fraction of the half width, -1..1.
    let lateral = (texel.z - 0.5) * 2.0;
    let l = abs(lateral);
    // The wheel lines: a Gaussian band each side, `PATH_RUT_OFFSET` of the half width out.
    let wheel = (l - PATH_RUT_OFFSET) / PATH_RUT_SIGMA;
    let in_band = abs(wheel) <= 2.5;
    out.rut = select(depth, 0.0, !in_band) * exp(-0.5 * wheel * wheel);
    // A cart track is two ruts with a grassy crown between them (its axle clears the middle): the
    // crown keeps the turf, its wheel bands and verges do not.
    let between = smoothstep(0.28, 0.40, l) * (1.0 - smoothstep(0.80, 0.95, l));
    out.crown = select(0.0, between, out.kind > 0.5 && out.kind < 1.5);
    // The cross-section: a dish, deepest on the centre line and rising to the ground at the verge.
    let dish_depth = select(
        select(PATH_DISH_LANE_M, PATH_DISH_STONE_M, out.kind > 1.5),
        PATH_DISH_ROAD_M,
        out.kind > 0.5 && out.kind < 1.5,
    );
    out.dish = dish_depth * (1.0 - l * l) * out.cover;
    // The slope across the route, m per metre: the dish's own (`d/dl` of `depth (1 - l^2)`, scaled
    // by the half width) and each rut's groove wall (the Gaussian's derivative, scaled so a 6 cm
    // rut on a 1 m half width tilts ~37 degrees, which a low sun rakes).
    let dish_slope = 2.0 * dish_depth * l * out.cover / half_width;
    let rut_slope = select(0.0, (wheel / PATH_RUT_SIGMA) * out.rut / half_width, in_band);
    out.wall = dish_slope - sign(lateral) * rut_slope;
    // The direction the wall rises in is where the signed distance grows: its gradient, which
    // points outward for a route and follows a curve.
    let du = 12.0 / (lane.z * lane.w);
    let dx = textureSampleLevel(mask, smp, uv + vec2<f32>(du, 0.0), 0.0).x
        - textureSampleLevel(mask, smp, uv - vec2<f32>(du, 0.0), 0.0).x;
    let dz = textureSampleLevel(mask, smp, uv + vec2<f32>(0.0, du), 0.0).x
        - textureSampleLevel(mask, smp, uv - vec2<f32>(0.0, du), 0.0).x;
    out.away = normalize(vec2<f32>(dx, dz) + vec2<f32>(1e-6, 0.0));
    return out;
}

// The route's own surface colour, linear (the crate's albedos are linear, and its `SOIL` is a wet
// forest floor at 0.07: a worn lane is *not* that — it is dry, compacted, dusty bare earth).
fn path_colour(kind: f32) -> vec3<f32> {
    if kind > 2.5 {
        // A snow track changes no colour: it compacts the snow, and the cover draws that.
        return vec3<f32>(1.0);
    }
    if kind > 1.5 {
        // Laid stone: pale and slightly cool.
        return vec3<f32>(0.150, 0.150, 0.145);
    }
    if kind > 0.5 {
        // Gravel: a *grey* stone road (`0.13` linear is a mid grey under this sun, not a beige).
        return vec3<f32>(0.115, 0.112, 0.102);
    }
    // Earth: a trodden lane is dark, damp-packed bare soil — well below the gravel's grey and
    // browner, so a footpath and a road never read as the same band.
    return vec3<f32>(0.058, 0.042, 0.028);
}

// The share of the ground's own surface the route's takes (a snow track takes none: the cover
// draws it).
fn path_bare(kind: f32) -> f32 {
    return select(1.0, 0.0, kind > 2.5);
}

struct PathGrain {
    // An albedo multiplier around 1.
    albedo: f32,
    // The grain's tilt: ∂h/∂x and ∂h/∂z in x and z, m/m (as the terrain's own `grad`).
    grad: vec3<f32>,
}

// The grain's base feature, m: gravel's chips are one to three centimetres (a coarse field at
// 10 cm reads as broken concrete), earth's crust is broken at a hand's width, laid stone's courses
// are a slab's width.
const PATH_GRAVEL_M: f32 = 0.022;
const PATH_EARTH_M: f32 = 0.11;
const PATH_STONE_M: f32 = 0.30;

// The route's own grain: the ground's detail belongs to turf and soil, and a lane needs a surface
// of its own. Band-limited by the pixel's footprint, as every detail in this crate is.
fn path_grain(p: vec2<f32>, kind: f32, footprint_m: f32, mask_grain: f32) -> PathGrain {
    var out: PathGrain;
    out.albedo = 1.0;
    out.grad = vec3<f32>(0.0);
    if kind > 2.5 {
        return out;
    }
    // The base feature, its weight and the albedo's amplitude, per kind: earth's crust is broken
    // at a hand's width, gravel's chips are centimetres, laid stone's courses are broad.
    var feature = PATH_EARTH_M;
    var weight = 0.45;
    var amplitude = 0.24;
    if kind > 1.5 {
        feature = PATH_STONE_M;
        weight = 0.34;
        amplitude = 0.12;
    } else if kind > 0.5 {
        feature = PATH_GRAVEL_M;
        weight = 0.55;
        amplitude = 0.24;
    }
    var sum = 0.0;
    var grad = vec2<f32>(0.0);
    // The coarsest octave is deliberately wide (four times the base) so the surface still varies
    // at 20-40 m, where the chips and the crust are sub-pixel and the band limit drops them: a
    // road that flattens to a flat colour by 15 m reads as paint.
    // Two wide octaves carry the surface past the chips' band limit: a road that flattens to one
    // colour by 12 m reads as paint (the capture review, `/tmp/i54/z3_road.png`).
    for (var k = 0u; k < 2u; k += 1u) {
        let wide = feature * select(4.0, 14.0, k == 1u);
        let keep_wide = 1.0 - smoothstep(0.5 * wide, 1.5 * wide, footprint_m);
        if keep_wide > 0.0 {
            let n = perlin2_d(p / wide + vec2<f32>(19.7, 5.3) * f32(k + 1u));
            let amp = select(0.30, 0.22, k == 1u);
            sum += amp * keep_wide * n.x;
            grad += amp * keep_wide * vec2<f32>(n.y, n.z) / wide;
        }
    }
    var q = p / feature;
    for (var o = 0u; o < 3u; o += 1u) {
        // An octave the pixel cannot resolve is dropped, so a distant road does not shimmer.
        let keep = 1.0 - smoothstep(0.5 * feature, 1.5 * feature, footprint_m);
        if keep > 0.0 {
            let n = perlin2_d(q + vec2<f32>(7.31, 3.17) * f32(o));
            sum += weight * keep * n.x;
            grad += weight * keep * vec2<f32>(n.y, n.z) / feature;
        }
        weight *= 0.55;
        feature *= 0.5;
        q *= 2.0;
    }
    // The mask's own grain is the surface's speckle at 0.175 m — the chips of a gravel road, the
    // trodden clods of a lane — and it survives to 40 m where the noise octaves above are gone.
    let speckle = (mask_grain - 0.5) * 2.0;
    sum += 0.55 * speckle;
    // A gentle sharpening keeps the chips *specks* rather than a smooth swell, without the dark
    // rims of a hard cube.
    let sharp = sum * (1.0 + 1.1 * sum * sum) * amplitude;
    out.albedo = 1.0 + sharp;
    out.grad = vec3<f32>(grad.x, 0.0, grad.y) * amplitude;
    return out;
}
