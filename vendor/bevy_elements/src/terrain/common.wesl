// bevy_elements::terrain::common — the terrain grid's placement for shaders that read the packed
// `terrain_cells` buffer (wind solid mask, precipitation ground, ground extension, terrain and
// water materials).
//
//     import bevy_elements::grid::grid_taps;
//     import bevy_elements::terrain::common::{TerrainParams, terrain_live};
//
// `terrain_cells` is `array<vec4<f32>>`, one entry per cell, row-major: x surface height
// (bedrock + sediment, m above `misc.x`), y sediment depth (m), z water depth above the surface
// (m), w `bitcast<f32>(pack2x16float(∂h/∂x, ∂h/∂z))`. Sample it with `grid_taps(params.grid, xz)`
// and four loads; the world height is `misc.x + surface`.

import bevy_elements::grid::GridParams;
import bevy_elements::noise::{cell2, cell3, hash31, perlin2_d, perlin3, perlin3_d, seed_offset};

struct TerrainParams {
    grid: GridParams,
    // x: world height of a stored height of 0 (m). y: readback downsample factor.
    // z: 1 when a terrain is bound, 0 when the binding is a one-cell placeholder.
    misc: vec4<f32>,
}

fn terrain_live(params: TerrainParams) -> bool {
    return params.misc.z > 0.5;
}

// ---------------------------------------------------------------------------------------------
// The CDLOD patches (`docs/research/orthographic-lod.md`)
// ---------------------------------------------------------------------------------------------

/// The last height sample on each axis, metre-frame XZ. A patch's vertices sit on the samples
/// (its translation is half a cell in from the node's tile), so a vertex past the last one is
/// clamped onto it and the drawn terrain spans the cell centres.
fn last_sample(grid: GridParams) -> vec2<f32> {
    return grid.origin + (vec2<f32>(grid.size) - 0.5) * grid.cell;
}

/// What a patch vertex morphs about, metre frame: the uniform's `centre` when its w is set (an
/// orthographic followed camera's focus, the same in every view), else the view's own LOD
/// position `lod_view` (world units, `units` to the metre).
fn morph_centre(lod_view: vec3<f32>, centre: vec4<f32>, units: f32) -> vec3<f32> {
    return select(lod_view / units, centre.xyz, centre.w > 0.5);
}

// ---------------------------------------------------------------------------------------------
// Ground detail (`docs/research/ground-detail.md`)
// ---------------------------------------------------------------------------------------------

/// Feature size of the coarse detail's base octave, m.
const DETAIL_COARSE_M: f32 = 1.25;
/// Feature size of the fine detail's base octave, m: the near-field grain.
const DETAIL_FINE_M: f32 = 0.09;
/// Amplitude of the coarse detail as a fraction of the albedo.
const DETAIL_COARSE_AMP: f32 = 0.20;
/// Amplitude of the fine detail as a fraction of the albedo.
const DETAIL_FINE_AMP: f32 = 0.85;
/// Height amplitude of each scale's base octave, m: the bump gradient is the gradient of a
/// height field of this amplitude, so the coarse scale reproduces the old
/// `0.04 × 0.8 = 0.032` per unit of the noise's derivative (`ground-detail.md` Parameters).
const DETAIL_COARSE_HEIGHT_M: f32 = 0.045;
const DETAIL_FINE_HEIGHT_M: f32 = 0.018;
/// Octaves of each scale.
const DETAIL_COARSE_OCTAVES: u32 = 4u;
const DETAIL_FINE_OCTAVES: u32 = 3u;

struct GroundDetail {
    /// Multiplier on the albedo, around 1.
    albedo: f32,
    /// The detail's gradient, already scaled by the bump of each octave that survived.
    grad: vec3<f32>,
    /// 1 where any octave survived the footprint test, 0 in the distance.
    present: f32,
    /// The coarse scale's own value, in [-1, 1]: the field at 1.25 m that the turf's grain is
    /// made of. The palette uses it as the *clump* of a tone, so a patch's edge is broken and
    /// the near field has colour variation in it without another noise evaluation
    /// (`ground-colour.md` Model 3).
    coarse: f32,
    /// The fine scale's own value, in [-1, 1]: the field at nine centimetres.
    fine: f32,
}

// ---------------------------------------------------------------------------------------------
// The ground's colour (`docs/research/ground-colour.md`)
// ---------------------------------------------------------------------------------------------

/// The palette of ground tones, as multipliers on the turf's albedo: fresh grass, the yellow of
/// a dry patch, the straw of a bleached one, and the earth of a worn place.
///
/// A meadow is not one green. Papavasiliou 2015 §3.3 is the model: each terrain type has a
/// reference colour, and a point's score for that type is the inverse distance of the
/// reference colour to the point's base map colour; the scores are the weights of a palette
/// blend. The tones here are the *variation* of one material rather than four materials, so the
/// palette is mixed in the material's own hue and its mean is the material's own colour.
const TURF_FRESH: vec3<f32> = vec3<f32>(1.00, 1.00, 1.00);
const TURF_DRY: vec3<f32> = vec3<f32>(1.35, 1.20, 0.69);
const TURF_STRAW: vec3<f32> = vec3<f32>(1.55, 1.37, 0.82);
const TURF_WORN: vec3<f32> = vec3<f32>(1.13, 0.95, 0.73);

/// The scale of the palette's patch, m: a patch is a clump of grass a couple of metres across.
const TURF_PATCH_M: f32 = 5.0;
/// The scale of the palette's *distribution* field, m: the field decides how much of each tone
/// there is around, over tens of metres, so the variation reaches the far field.
const TURF_FIELD_M: f32 = 38.0;
/// How far the palette moves the ground's colour towards a tone, 0 to 1.
const TURF_STRENGTH: f32 = 0.85;

/// The ground's colour: a multiplier on the turf's albedo.
///
/// The palette is chosen per *patch*, not per pixel: at 2 m a pixel is a centimetre and a straw
/// blade is a decimetre, so a per-pixel choice of a tone is a dithered speckle. The patch is
/// selected by a field, and the boundary between two tones is soft, so the tones' grain crosses
/// it (Lefèbvre & Neyret 2003's dithering of a low-resolution areas map, done with a smooth
/// boundary instead of a dither so the transition is not a speckle).
///
/// The result is normalised so that its mean is 1: a game's lighting is balanced around its
/// ground's colour, so the change must be a *variation* and not a shift (the measured hue's
/// spread over a near-ground patch was 0.010 on the plains before this note).
///
/// `base` is the ground's own colour, and its *chroma* decides how far the tone goes: a place
/// where the ground is already the soil's brown does not also get the straw's yellow, and a
/// shadow does not turn the grass into straw (Papavasiliou's weight vector, which drops the
/// lightness for exactly that reason).
fn turf_palette(p: vec3<f32>, base: vec3<f32>, field: f32, coarse: f32, fine: f32) -> vec3<f32> {
    // The distribution field — how much of each tone there is around — is the ground's own
    // large-scale field, which the caller already has: the palette is given it, so the
    // palette's only new evaluation is the *selection* field below. (The two extra 3D noise
    // evaluations the palette first carried cost the mountains' opaque pass 0.132 ms -> 0.148
    // ms, and the ground's field and the palette's were the same field at 45 m and 38 m.)
    //
    // The clump: which tone this piece of ground wears. A field at five metres picks it, and
    // the ground's own *detail* — a field at a metre, which the turf already evaluates for its
    // grain — is the clump's finer octave, so the near field has variation in it without
    // another noise evaluation.
    let patch_field = perlin3(p / TURF_PATCH_M + seed_offset(97u)) / 0.2268;
    // The clump is the patch's field at five metres *plus* the turf's own fine field. The fine
    // field is at nine centimetres, which is far too fine to break a five-metre edge on its
    // own, but it is what makes the edge ragged rather than a smooth curve: a dry patch's
    // edge is broken, and the mixture of the two gives that.
    let clump = clamp(patch_field + coarse * 0.55 + fine * 0.45, -1.0, 1.0);
    // The score of each tone: how much of it belongs here. The field is the *distribution* (the
    // amount of the tone around) and the clump the *selection* (this piece of ground), and the
    // score is their product, so a tone is common where the field is high and present where the
    // clump is.
    //
    // The scores are raised to the power `p`, which is Papavasiliou's exponent: "using lower
    // values for p, broader transitional regions between terrain types can be achieved while
    // larger values create sharper, more defined transitions". At p = 1 the dry grass faded in
    // over the whole field and the meadow read as a haze; at p = 2 the patches have an edge.
    //
    // The scores are *shares* of the turf's own colour: the three dry tones take what they need
    // and the fresh grass has what is left, so the four sum to 1 and the palette is a convex
    // combination of them. A convex combination cannot drift the ground's mean: the mixture is
    // between the tones and the tones bracket the fresh one. (Normalising the mixture instead
    // was tried three ways and each failed: by the unweighted mean of the tones the ground's
    // mean brightness drifted 3.2 DN, by the mixture's own colour the hue was cancelled
    // entirely, and by a per-channel mean the hue was cancelled again.)
    // The scores are squared — Papavasiliou's exponent p = 2. A square is a multiply, not a
    // `pow`: three `pow` calls per pixel cost the mountains' opaque pass 0.137 ms -> 0.252 ms,
    // and a `pow` is a transcendental.
    //
    // The exponent is a *literal* everywhere it is written as a `pow` elsewhere: `pow(x, p)`
    // with `p` in a `let` makes naga emit a broken shader *without an error*, and the ground
    // renders as a dark navy (the same symptom as an unimported name; see AGENTS.md).
    let d = max(field, 0.0) * max(0.5 + clump, 0.0) * 1.6;
    let st = max(field - 0.45, 0.0) * 3.0 * max(0.5 + clump, 0.0) * 1.6;
    let w = max(-field - 0.55, 0.0) * 3.0 * max(0.5 - clump, 0.0) * 1.6;
    var s_dry = min(d * d, 1.0);
    var s_straw = min(st * st, 1.0);
    var s_worn = min(w * w, 1.0);
    let others = s_dry + s_straw + s_worn;
    if others > 1.0 {
        let k = 1.0 / others;
        s_dry = s_dry * k;
        s_straw = s_straw * k;
        s_worn = s_worn * k;
    }
    // The fresh share is what is left, and the subtraction can leave it a hair below zero
    // (−2.2e−8 at field 0.5, clump 0.75), which is a negative weight.
    let s_fresh = max(1.0 - s_dry - s_straw - s_worn, 0.0);
    // The palette, mixed by the shares.
    let mixed = s_fresh * TURF_FRESH + s_dry * TURF_DRY + s_straw * TURF_STRAW + s_worn * TURF_WORN;
    var tint = mix(vec3<f32>(1.0), mixed, TURF_STRENGTH);
    // The base colour's own chroma decides how far the tone goes.
    // The base colour's own chroma decides how far the tone goes. The chroma is measured
    // *relative* to the colour's own brightness: a dark albedo has a small absolute chroma by
    // construction (the crate's grass is (0.0397, 0.0835, 0.0134), whose chroma is 0.065), and
    // an absolute measure would leave the palette at 39 % of its strength on the ground and
    // nowhere near it on a bright one.
    let lum = dot(base, vec3<f32>(0.2126, 0.7152, 0.0722));
    let chroma = length(base - vec3<f32>(lum)) / max(lum, 1e-4);
    let carry = clamp(chroma * 1.6, 0.0, 1.0);
    return mix(vec3<f32>(1.0), tint, carry);
}

/// Weight of an octave of feature size `feature_m` in a pixel whose footprint is `footprint_m`
/// wide: 1 while the octave is resolved (at least two footprints per feature), falling to 0 over
/// the band where it is not. Lagae et al. 2010 §5.2.1, frequency clamping: a band the pixel
/// cannot represent is cancelled rather than aliased (`ground-detail.md` Model 2).
fn detail_keep(feature_m: f32, footprint_m: f32) -> f32 {
    return 1.0 - smoothstep(0.5, 1.0, footprint_m / feature_m);
}

/// One detail scale's contribution: the weighted octave sum, the weights' total, its bump
/// gradient, and 1 where any octave survived the footprint test.
struct ScaleDetail {
    sum: f32,
    norm: f32,
    grad: vec3<f32>,
    present: f32,
}

/// One detail scale: `octaves` Perlin octaves from `base_m` down by halving (persistence 0.5, as
/// `fbm3` in `noise.wesl`), each band-limited by its own feature size.
///
/// The octaves are the crate's **2D** gradient noise in the world XZ plane, and the gradient is
/// carried back to the world as a surface gradient. This is the ground's *turf*, and the turf
/// is on the near-horizontal surfaces, where the plane's projection is the surface.
///
/// It must not reach a slope. A 2D noise in the world XZ plane is the *projection* of the
/// surface onto that plane, so on a slope the plane's lattice is stretched along the slope and
/// the lattice shows through as a weave at a fixed world direction: 43.5 degrees at 2 m and
/// 44.0 degrees at 6 m on a 45 degree face (`rock-material.md` Model 1). The upness gate in
/// `render.wesl` keeps the turf off the faces, and the rock's own material — the 3D noise at the
/// 3D point, which has no projection at all — takes over there.
fn detail_scale(
    p: vec3<f32>,
    base_m: f32,
    octaves: u32,
    footprint_m: f32,
    seed: u32,
) -> ScaleDetail {
    var out = ScaleDetail(0.0, 0.0, vec3<f32>(0.0), 0.0);
    var feature = base_m;
    var weight = 0.5;
    var q = p.xz / base_m;
    for (var o = 0u; o < octaves; o += 1u) {
        let keep = detail_keep(feature, footprint_m);
        if keep > 0.0 {
            let off = seed_offset(seed + o);
            let n = perlin2_d(q + off.xz);
            out.sum += weight * keep * n.x;
            // The gradient is in the plane; the bump tilts the surface by it.
            out.grad += weight * keep * vec3<f32>(n.y, 0.0, n.z) / feature;
            out.present = max(out.present, keep);
        }
        out.norm += weight;
        weight *= 0.5;
        feature *= 0.5;
        q *= 2.0;
    }
    return out;
}

/// One detail scale in the crate's **3D** noise, evaluated at the 3D surface point: the same
/// octave sum as `detail_scale`, but with no projection, so it is the same field seen from every
/// direction and it stays put on the surface.
///
/// This is what the rock's material is made of. A per-pixel *surface frame* was tried and
/// rejected: the normal comes from the terrain's 2 m cell gradient, so the frame rotates from
/// pixel to pixel, the noise's lattice swims in it, and the result is a herringbone moiré
/// (measured on the capture: a fine diagonal hatch over the whole face). Lagae et al. 2010 §5's
/// warning is about *slicing* a 3D noise onto a 2D surface (taking `z = const`), not about
/// evaluating the 3D function at the surface's own point.
fn detail_scale_3d(
    p: vec3<f32>,
    base_m: f32,
    octaves: u32,
    footprint_m: f32,
    seed: u32,
) -> ScaleDetail {
    var out = ScaleDetail(0.0, 0.0, vec3<f32>(0.0), 0.0);
    var feature = base_m;
    var weight = 0.5;
    var q = p / base_m;
    for (var o = 0u; o < octaves; o += 1u) {
        let keep = detail_keep(feature, footprint_m);
        if keep > 0.0 {
            let off = seed_offset(seed + o);
            let n = perlin3_d(q + off);
            out.sum += weight * keep * n.x;
            out.grad += weight * keep * n.yzw / feature;
            out.present = max(out.present, keep);
        }
        out.norm += weight;
        weight *= 0.5;
        feature *= 0.5;
        q *= 2.0;
    }
    return out;
}

// ---------------------------------------------------------------------------------------------
// Rock's own material (`docs/research/rock-material.md`)
// ---------------------------------------------------------------------------------------------

/// Spacing of the beds, m of world height. A bed is a *thick* band of rock, and a cliff shows a
/// few of them, not a hatch: at 3.5 m a 20 m face shows six beds, at 12 m it shows under two.
/// At 4.5 m the face a player stands at — four metres of it — shows one or two bands, which is
/// what reads as bedding; at 12 m it showed none and the rock was a wash.
const ROCK_BED_M: f32 = 4.5;
/// The bedding plane's tilt: the direction the beds run and how steeply they rise. A bed is a
/// surface inside the rock, and it is *tilted* — the world's height is not the rock's bedding,
/// and a massif banded by its own contours is a topographic map, not a mountain.
const ROCK_BED_TILT: vec3<f32> = vec3<f32>(0.32, 0.90, 0.28);
/// The warp's amplitude, m: how far the beds dip and undulate. It must be a *fraction* of a
/// bed's thickness, or the beds stop being beds: at 3.5 m the warp displaced the bed
/// coordinate by up to 0.78 of a bed, and a 4.5 m grid distorted by a 45 m-scale field is a
/// set of blotches and not a set of bands — which is what the massif from 120 m showed
/// (`docs/research/bedding.md`).
const ROCK_BED_WARP_M: f32 = 0.9;
/// The bed lattice's warp, in bed units: how far a boundary wanders from the uniform lattice.
/// It is what makes the *gaps* — the beds' thicknesses — vary, and it is bounded, so the
/// lattice does not drift and the index can be inverted in a few steps.
const ROCK_BED_WARP_AMP: f32 = 1.0;
/// How far a bed's own tone differs from the rock's, as a fraction of the albedo. A succession's
/// beds are *different rock* — a sandstone, a shale, a limestone — and the tone is a draw on
/// the bed's index, so the beds differ from one another and not only in their thickness.
const ROCK_BED_TONE: f32 = 0.34;
/// The **region**'s scale, m: the field that biases the bed's colour, so a whole massif leans
/// towards one rock and the massifs separate into warm and cool. A range of mountains is made
/// of *formations*, and a formation is a massif — so the field is at the massif's scale,
/// hundreds of metres, and the bed's own draw only tints it.
const ROCK_BED_REGION_M: f32 = 260.0;
/// How far the region's field biases the bed's selection. It is **weak**: the field is a bias
/// and not a replacement, because a bias that saturates the selection pins one tone over a
/// whole massif and the rock's mean colour then depends on which tone a massif happened to get.
const ROCK_BED_REGION_BIAS: f32 = 0.30;

/// A bed's own value: the tone it wears, and how much of it there is.
///
/// A succession's beds are *different rock* — a sandstone, a shale, a limestone — and the
/// cheap way to say that is a **per-bed tone** drawn from the bed's index. So the tone is a
/// draw like the thickness is, and the beds of a cliff differ from one another the way a
/// succession's do, instead of every bed wearing the same colour.
struct BedValue {
    /// The bed's own **colour**, a multiplier on the rock's albedo, one channel each. It is a
    /// *low-frequency* feature — a bed's own colour, over metres of rock — so it survives every
    /// distance the eye can see the cliff from. This is what a distant massif shows: the
    /// succession's colours and not its stripes.
    ///
    /// It is a **vector** and not a scalar, because a scalar can only make a rock lighter or
    /// darker and a distant range is many *colours* (`docs/research/bed-colour.md`).
    tone: vec3<f32>,
    /// 1 inside the bed's own layer, 0 outside it: the bed's edge.
    mask: f32,
    /// 1 where the bed's edge survives the pixel's footprint, 0 where it does not.
    present: f32,
}

/// The bed's structure at the coordinate `bed` (in units of the nominal spacing): which bed
/// this is, how thick it is, and where its edges are.
///
/// The beds **stack**: each bed's *top* is its own bottom plus its own thickness, and the
/// thickness is a **draw** on the bed's index — a real succession's thicknesses are "the sum of a
/// series of log-normal frequency distributions" (Talling 2001), so a few beds are much
/// thicker than the rest. The beds therefore *overlap* where a bed is thicker than the
/// nominal and leave a *gap* where it is thinner, and the layer that shows is the one whose
/// span covers the coordinate.
///
/// A true cumulative sum would need an unbounded prefix sum, and the shader cannot afford
/// one. The draw is the same rhythm done in a bounded way: each bed's own span is its draw,
/// and the *sequence* of spans is what the eye reads as a succession.
fn bed_value(p: vec3<f32>, bed: f32, footprint_m: f32, colour_keep: f32) -> BedValue {
    // The index whose boundary is at or below the coordinate: the fixed point of
    // `i = (bed - warp(i)) / mean`. The warp is a bounded displacement, so the iteration
    // converges in a couple of steps.
    var index = bed;
    for (var k = 0u; k < 4u; k += 1u) {
        index = bed - bed_warp(index);
    }
    // The candidate bed, and the one below it: the coordinate is in one of them, and the one
    // whose bottom is at or below the coordinate and whose top is above it is the bed. The
    // iteration must be *stable*: taking the floor at the end of it would flip between two
    // beds at a boundary.
    var bed_index = floor(index);
    var bottom = bed_boundary(bed_index);
    if bottom > bed {
        // The estimate overshot into the bed above: step down.
        bed_index = bed_index - 1.0;
        bottom = bed_boundary(bed_index);
    }
    let top = bed_boundary(bed_index + 1.0);
    // Inside the bed? The bed's own layer, between its bottom and its top.
    let inside = step(bottom, bed) * step(bed, top);
    // The bed's *edge*: the distance to the nearer of its two boundaries, and the edge is
    // sharp — a bed is a band of rock with an edge, not a gradient.
    //
    // The edge's width is **softened over the pixel's own footprint**: a bed's edge is a line
    // a fraction of a bed wide, and a hard step on a line that narrow stair-steps at every
    // distance (measured: the seams on a face at 3 m were aliased diagonals). The width grows
    // with the footprint, so the seam blurs into the rock rather than aliasing along it.
    let edge_width = max(0.06, footprint_m / (ROCK_BED_M * 0.5));
    let to_edge = min(bed - bottom, top - bed);
    let edge = 1.0 - smoothstep(0.0, edge_width, to_edge);
    // The band-limit: the bed's edge is a line a fraction of a bed wide, so it aliases long
    // before the bed does. Where it cannot be resolved it is *cancelled* — and what survives
    // is the bed's own *tone*, which is a low-frequency feature. That is the whole trick: at a
    // distance the cliff shows its succession and not its stripes.
    let keep = detail_keep(ROCK_BED_M * 0.5, footprint_m);
    // The bed's tone: a draw on the index, so the beds differ from one another. The crate's
    // own integer hash, for the same reason the thickness's is: a `sin` of a large argument
    // loses its precision in f32.
    let h = hash31(vec3<i32>(i32(floor(index)), 7, 0));
    // The bed's own **colour**, and not only its brightness (`docs/research/bed-colour.md`).
    //
    // A bed's colour is its **iron**: goethite is yellow and hematite is red, and the ratio
    // of the surface's reflectance at 700 nm to its reflectance at 400 nm is the standard
    // quantitative description of a sediment's colour (Debret et al. 2011: "the numerical
    // description of the general slope of VIS", running "from 1.5 to 6" across the sediments
    // they measured and near 1 for the greyest). So the bed's draw is its iron, and the
    // palette's members are the iron-bearing rocks.
    //
    // The score is a **narrow split** on the draw, exactly as the fragments' palette is
    // (iteration 38): a product of smooth fields is smooth, and a smooth field blends, and
    // the blend is what left the cliff at one hue. The split's width is Papavasiliou's `p`,
    // written as a `smoothstep`'s width and not as a `pow` — the crate's own rule is that
    // three `pow` calls per pixel cost the mountains' opaque pass 0.137 ms → 0.252 ms.
    //
    // The **region** is a *slow* field that biases the selection, so a whole massif leans
    // towards one rock and the massifs separate into warm and cool. A range of mountains is
    // made of *formations*, and a formation is a massif — so the field is at the massif's
    // scale, hundreds of metres.
    //
    // The field's bias is **weak** and its value is the noise's *own* range: the field is a
    // bias and not a replacement. A bias that saturates the selection pins one tone over a
    // whole massif, and the rock's mean colour then depends on which tone a massif happened
    // to get — the frame's mean moved 66 DN, and the crate's own rule is that the mean stays
    // where it was.
    //
    // The field is only evaluated where the bed's **colour** is wanted: the fragments carry
    // the near field and the beds carry the far one, so where a fragment resolves the bed's
    // colour is multiplied by zero and the region — a whole `perlin3_d` — is computed and
    // thrown away. The gate is the colour's own keep, which is the fragments' band-limit.
    //
    // The gate skips the **evaluation** and not the field's *value*: the value is the field's
    // own, unmodified. Scaling it by the keep instead would weaken the bias in the band
    // where the keep is between zero and one, and the bed's colour would change there — a
    // performance change must not change the image.
    var region = 0.0;
    if colour_keep > 0.0 {
        region = perlin3(p / ROCK_BED_REGION_M + seed_offset(113u));
    }
    let sel = smoothstep(
        0.5 - ROCK_PALETTE_EDGE,
        0.5 + ROCK_PALETTE_EDGE,
        clamp(h + region * ROCK_BED_REGION_BIAS, 0.0, 1.0),
    );
    // The bed's second draw: the distribution, which member of the half. A *separate* hash of
    // the same index, so the two are independent.
    let h2 = hash31(vec3<i32>(i32(floor(index)), 11, 0));
    let reg = smoothstep(
        0.5 - ROCK_PALETTE_EDGE,
        0.5 + ROCK_PALETTE_EDGE,
        clamp(h2 + region * ROCK_BED_REGION_BIAS, 0.0, 1.0),
    );
    let s_basalt = (1.0 - sel) * (1.0 - reg);
    let s_sandstone = (1.0 - sel) * reg;
    let s_limestone = sel * (1.0 - reg);
    let s_varnish = sel * reg;
    let shares = s_basalt + s_sandstone + s_limestone + s_varnish;
    let inv = 1.0 / max(shares, 1e-5);
    // The mixture: a convex combination of the palette, so the rock's mean colour cannot
    // drift (the crate's own rule from the ground's palette).
    let mixed = (s_basalt * ROCK_TONE_BASALT
        + s_sandstone * ROCK_TONE_SANDSTONE
        + s_limestone * ROCK_TONE_LIMESTONE
        + s_varnish * ROCK_TONE_VARNISH) * inv;
    // The bed's *value*: the bed's own brightness spread, as it was before the colour.
    let value = 1.0 + ROCK_BED_TONE * (h * 2.0 - 1.0);
    // The bed's **colour** is band-limited against the *fragments*: the fragments carry the
    // near field and the beds carry the far one. Where a fragment is visible it wears its own
    // rock and the bed's colour is not mixed in at all, so the near face is untouched; where
    // the fragments are gone the bed's own rock shows through.
    //
    // The limit is the *fragments'* own: their band-limit is the joint's width, and the
    // blocks' `present` is that limit, so the two are band-limited against each other exactly
    // once and there is no gap and no overlap.
    let tone = mix(vec3<f32>(value), mixed * value, ROCK_PALETTE_STRENGTH * colour_keep);
    return BedValue(tone, inside * edge, keep);
}

/// The warp of the bed lattice, in bed units: where the boundary between the bed whose index
/// is `i` and the next one falls.
///
/// The displacement is **bounded** — the beds wander, and the *gaps* between them carry the
/// draw, so a bed's thickness varies. It is not a cumulative sum of the beds' draws: that
/// would need an unbounded prefix sum, and the shader cannot afford one. A bounded
/// displacement whose *differences* are the draws is the same rhythm, and it is finite.
fn bed_warp(index: f32) -> f32 {
    // The warp is **smooth** — a few low frequencies — because it is the warp's
    // *derivative* that sets the spacing between two boundaries, and a smooth field's
    // derivative is bounded and can be large, while a per-bed hash's difference is
    // uncorrelated and tiny. So the smooth part is what makes the beds' *thicknesses* vary,
    // and the per-bed hash is only a little local jitter on top.
    let h = hash31(vec3<i32>(i32(floor(index)), 3, 0));
    return ROCK_BED_WARP_AMP * (
        0.60 * sin(index * 0.21)
        + 0.28 * sin(index * 0.53 + 1.3)
        + 0.12 * sin(index * 1.17 + 2.6)
    ) + 0.10 * (h - 0.5);
}

/// The boundary between the bed whose index is `i` and the next one, in bed units.
fn bed_boundary(index: f32) -> f32 {
    // The lattice's spacing is the *nominal* spacing, and the warp is in bed units of it, so
    // the boundaries are at `index + warp(index)` in units of the nominal spacing.
    return index + bed_warp(index);
}

/// The second rock tone, relative to the first: a *darker*, browner bed. Bedded rock alternates
/// light and dark bands, and the dark one is darker rather than lighter: the crate's rock
/// albedo is already 0.10 linear, and a brighter second tone only washed the face further.
///
/// The tone is a *value* change and not only a hue one: at (1.06, 0.93, 0.76) the luminance
/// barely moved (1.9 % at the old mix), and the massif from 90 m was one colour — a max−min of
/// 1.00 DN, where the crate's own rock note asks for 9. A bed is a *darker* band, and the
/// value is what carries at a distance.
const ROCK_SECOND_TONE: vec3<f32> = vec3<f32>(0.80, 0.73, 0.66);
/// The rock's **palette**: the colours a cliff is made of (`docs/research/rock-palette.md`).
///
/// A cliff is not one grey, and the crate's rock system was monochrome by construction: every
/// tone in it was a *scalar*, and a scalar can only make a rock lighter or darker and never
/// *another colour*. Measured on the build before this note, the face at dusk was a **single
/// hue** — the circular resultant of the hue angles 0.998 and the spread 2.1° — and the
/// rock-only patch at 3 m had a chroma's median of 0.0157, which is nearly grey.
///
/// The palette's members are *different minerals*, and each is a whole colour and not a tint
/// of the rock's grey:
///
/// - `ROCK_TONE_BASALT` — a cool grey, a fresh volcanic rock, at 215° and a saturation of
///   0.10;
/// - `ROCK_TONE_SANDSTONE` — a warm ochre, an iron-stained sandstone, at 35° and 0.30 (Tansem &
///   Storemyr 2020: the red bedrock surfaces "are composed of inorganic iron films");
/// - `ROCK_TONE_LIMESTONE` — a pale warm buff, the lightest rock in the succession, at 45° and
///   0.10;
/// - `ROCK_TONE_VARNISH` — the dark brown of a ferromanganese varnish, at 25° and 0.20
///   (Krinsley et al. 1998: "a thin coating of ferromanganese oxides, clay minerals, and trace
///   elements").
///
/// The saturations are **low** — 0.10 to 0.30 — because rock is a *desaturated* colour. The
/// first version of the palette was written as raw RGB triples and its tones came out at
/// saturations of 0.84 and 0.75: a strong blue and a strong orange, which are not rock
/// colours but *primaries*, and the cliff read as a rust-painted fence. The palette is
/// therefore written in *hue and saturation* and converted, so the hues are the ones a
/// geologist would name and the saturations are the ones a rock has.
///
/// The tones are **multipliers on the rock's own albedo**, so the crate's albedo
/// `(0.072, 0.069, 0.067)` — "a stone face under a blue sky is grey with the blue in it" —
/// still sets the rock's overall value and the palette only moves the *hue*.
///
/// The palette's **equal-weight mean is exactly (1, 1, 1)**, so the rock's mean colour and
/// mean brightness cannot move: the tones are *normalised* so that they do. The first
/// version of the palette was not, and the face's mean luminance fell 2.7 DN — a game that
/// changes colour changes its lighting, and the crate's own rule from the ground's palette
/// is that "the *mean* colour and the *mean* brightness stay where they were". The tones'
/// own luminances then span 0.59 to 1.26, so a fragment is a *lighter or darker* piece of
/// rock as well, which is the point.
const ROCK_TONE_BASALT: vec3<f32> = vec3<f32>(0.848, 1.015, 1.309);
const ROCK_TONE_SANDSTONE: vec3<f32> = vec3<f32>(1.223, 1.083, 0.832);
const ROCK_TONE_LIMESTONE: vec3<f32> = vec3<f32>(1.214, 1.270, 1.256);
const ROCK_TONE_VARNISH: vec3<f32> = vec3<f32>(0.715, 0.632, 0.603);
/// The width of a split in the palette, as a fraction of the draw's range. A narrow edge
/// makes a fragment emphatically ONE rock; a wide one blends two and the cliff is at one hue
/// again. It is Papavasiliou's exponent `p`, written as a `smoothstep`'s width.
const ROCK_PALETTE_EDGE: f32 = 0.12;
/// How far the palette is mixed in, 0 to 1. The tones bracket the base, so the mixture is a
/// convex combination of them and the rock's mean colour cannot drift — the crate's own rule
/// from the ground's palette (`ground-colour.md`): "the ground is a place and not a paint
/// colour, while the *mean* colour and the *mean* brightness stay where they were".
const ROCK_PALETTE_STRENGTH: f32 = 0.90;
/// The **rind's** own colour: the iron-and-manganese film on the rock's surface. It is a
/// separate *layer* and not a tint: Tansem & Storemyr's "rocks with red iron films, rocks
/// that are now generally gray" are the same rock, and the film is what the surface shows.
const ROCK_RIND_TONE: vec3<f32> = vec3<f32>(1.225, 0.956, 0.770);
/// The rind's own tone has a luminance of exactly 1, so the film moves the surface's *hue*
/// and not its *value*: a rind is a colour on the rock and not a darkening of it.
///
/// The rind's **coverage**: how much of the surface the film takes, and how it is mixed in.
/// The varnish's "percentage of clast surfaces covered" is the age indicator (Reneau et al.
/// 1993), so the coverage is a fraction and not a tint of everything.
const ROCK_RIND_MIX: f32 = 0.55;
/// The rind's patch size, m: the scale of the film's own mottling. The film is a *fine*
/// feature — a coating 100 to 500 um thick — and the patches of it are decimetres across, so
/// the film is band-limited away long before the rocks' own colours are.
const ROCK_RIND_M: f32 = 0.55;
/// How far the beds' tone is mixed in, 0 to 1. The tone's luminance is 0.74, so at 0.55 the
/// beds span an 14 % albedo drop — enough to read as bands on a lit face and to carry the
/// massif's colour at a distance.
const ROCK_BED_MIX: f32 = 0.55;
/// The rock's own grain scales, m: the coarse one is the field the fractures are ridges of.
const ROCK_GRAIN_COARSE_M: f32 = 2.6;
const ROCK_GRAIN_FINE_M: f32 = 0.12;
/// The fracture's width, as a fraction of the ridge's range.
const ROCK_FRACTURE_WIDTH: f32 = 0.30;
/// How dark the fractures are, as a multiplier on the rock's albedo. A crack is a shadow: the
/// face beside it sees less sky.
const ROCK_FRACTURE_DARKEN: f32 = 0.26;
/// The rock's own grain amplitude, as a fraction of the rock's albedo.
const ROCK_GRAIN_AMP: f32 = 0.60;

// The cliff's blocks (`docs/research/rock-blocks.md`)
// ---------------------------------------------------------------------------------------------

/// The block network's cell, m: the scale of the *pieces* the rock breaks into. A block a
/// player can read is a metre or two across, and the rock face in the capture is 2 m across
/// the frame at 3 m — so this is the largest feature the crate's rock can carry.
const ROCK_BLOCK_M: f32 = 0.95;
/// How far the blocks stretch along the bedding, as a factor: a rock breaks along *joint
/// sets*, which are families of parallel planes, so its blocks are tabular and aligned rather
/// than a random heap (La Pointe & Hudson 1985 on joint sets).
const ROCK_BLOCK_TABULAR: f32 = 2.6;
/// The **second joint set's** normal: a cross joint, roughly perpendicular to the bedding and
/// tilted, so the two sets are not parallel and the blocks between them are *slabs* and not
/// boxes. A rock breaks along *joint sets* — families of parallel planes (La Pointe & Hudson
/// 1985) — and a cliff has more than one.
const ROCK_JOINT2_NORMAL: vec3<f32> = vec3<f32>(0.71, 0.46, -0.53);
/// The second joint set's **spacing range**, m: the blocks between two sets of planes are
/// slabs of this thickness, and a formation's spacing is a property of the formation.
const ROCK_JOINT2_MIN_M: f32 = 0.55;
const ROCK_JOINT2_MAX_M: f32 = 2.60;
/// The **formation's** cell, m: the scale at which the second set's spacing is constant. The
/// spacing must be constant *within* a formation — a spacing that varies smoothly makes the
/// planes *fold* instead of fracture — and it changes from one formation to the next.
const ROCK_FORMATION_M: f32 = 46.0;
/// The second joint set's joint's width, as a fraction of its own spacing: a joint is a
/// feature of the rock's own scale, and a crack is a line a fraction of a cell wide.
const ROCK_JOINT2_WIDTH: f32 = 0.075;
/// How far the second set's planes **wander** from their regular positions, as a fraction of
/// their own spacing. A joint set's planes are non-persistent: they end, and their spacing
/// varies along them (Strauhal & Zangerl 2021). A perfectly straight, evenly spaced plane
/// set reads as a wire fence across the face.
const ROCK_JOINT2_WANDER: f32 = 0.48;
/// How long the second set's segments are, in multiples of their own spacing: the plane
/// wanders by a fraction of a spacing over a few metres and then the joint is simply absent
/// (Strauhal & Zangerl 2021 on the non-persistent fractures).
const ROCK_JOINT2_SEGMENT_M: f32 = 1.9;
/// How sharply a segment *ends*: the threshold on the finer field. A plane that bends but
/// never ends still reads as a wire fence across the face.
const ROCK_JOINT2_BREAK: f32 = 0.30;
/// The **power** of the draw that sets the second set's spacing. A block's size is the
/// *product* of three spacings (Verdian et al. 2021: the latent size distribution is "set by
/// the spacing and orientation of fractures, foliations, and bedding planes"), and a product
/// of three heavy-tailed draws has a heavier tail than one - and the heavier tail is what
/// "slabs and chips in the same wall" is.
const ROCK_BLOCK_SIZE_POWER: f32 = 2.4;
/// The warp that breaks the cells' shape, as a fraction of the cell. A cell network's cells
/// are convex polygons of nearly the same area, and a face of them reads as a mosaic — or,
/// with the blocks' step, as *shingles*. A rock breaks into irregular fragments, and the warp
/// is what makes the fragments irregular.
///
/// The amplitude is in **metres**, and the warp's vector is the noise's own *gradient*, whose
/// mean magnitude is 0.94 — so the amplitude is also the displacement's scale, in metres. The
/// displacement in *cells* is the amplitude over the cell: at 0.50 m over a 1.5 m cell the
/// mean displacement is 0.31 of a cell and the max 0.79, which breaks the cells' regularity
/// without tearing them. The note's earlier 0.09 was calibrated for the noise's *value* and
/// not for its *gradient*, so it displaced a boundary by 0.06 of a cell — a fifth of what it
/// needed to be, and the face came out a quilt.
const ROCK_BLOCK_WARP_M: f32 = 0.30;
/// How many cells the warp's noise varies over. The warp must displace the lattice
/// *coherently*: a warp that varies over the length of a cell moves different points by
/// different amounts and scrambles the lattice instead of bending it.
const ROCK_BLOCK_WARP_CELLS: f32 = 4.0;
/// The block's own tone, as a multiplier on the rock's albedo: the cell's random factor, so
/// every block is a slightly different piece of the rock. The spread is narrow — a block is a
/// piece of *one* bed, so it carries the bed's tone and not a new one.
const ROCK_BLOCK_TONE: f32 = 0.22;
/// The joint between two blocks, as a multiplier on the rock's albedo. A broken edge is a
/// shadow: the joint darkens the rock beside it, and it is the one feature of a cliff a
/// player reads as "solid" at a distance.
const ROCK_JOINT_DARKEN: f32 = 0.76;
/// The joint's width, as a fraction of the cell's half-boundary distance. A joint is a line a
/// fraction of a cell wide — the crate's own rule for the fractures, and the reason they are
/// band-limited by their own width and not by the cell.
const ROCK_JOINT_WIDTH: f32 = 0.19;
/// The block's relief, as the *slope* of its face: the distance field's gradient domes the
/// block, so the sun rakes its edge and the light finds the joint.
const ROCK_BLOCK_RELIEF: f32 = 0.95;
/// The fine block network's cell, as a fraction of the coarse one, and **zero**: the fine
/// generation is not used.
///
/// A cliff's blocks are not all the same size — the fracture spacing follows a Weibull whose
/// geometric standard deviation is about 0.31 (Verdian et al. 2021, whose fifteen sites run
/// 0.214 to 0.551), and the spacings themselves run from 2 mm to 5000 mm against a site
/// median of 10 to 390 mm — and a second generation of blocks, a third of the size, in the
/// gaps between the big ones, is the scree's own construction and the obvious way to get the
/// size spread.
///
/// It does not work here, and the measurement says why: a fine generation of 0.45 m cells on
/// a face at 3 m puts a seam every few pixels, and the face came out a **dark speckle**
/// (the material's own high-pass fell from 10.0 DN to 6.5 DN, below the baseline's 6.7). A
/// chip is not a crack, so the fine generation was made to carry its tone and no seam — and
/// then it was a mottle of patches of a slightly different rock, which is not what a cliff is
/// either.
///
/// The size spread comes from the **cells themselves** instead: the network's feature points
/// are drawn towards their cell's centre, so the cells are unequal (`CELL_INSET` in
/// `noise.wesl`), and a face of unequal cells is a scatter of fragments and not a mosaic.
/// The edge orientations' resultant — how strongly the blocks' edges agree on a direction,
/// which is the quilt's signature — falls from 0.43 to **0.20**.
const ROCK_BLOCK_FINE: f32 = 0.0;
/// The Weibull's shape for the joint's width, the distribution the fracture spacing follows
/// (Verdian et al. 2021). A shape of 1 is the exponential and 2 is the Rayleigh; 1.6 is
/// between, with the longer tail a cliff's spacings have.
const ROCK_BLOCK_WEIBULL: f32 = 1.6;
/// The Weibull's scale, as a multiple of the joint's width.
const ROCK_BLOCK_WEIBULL_SPREAD: f32 = 0.85;
/// The Weibull's draw's ceiling: the 95th percentile, and not the unbounded tail. The
/// fracture spacing's own distribution is *undersampled* in its lower tail, so the tail is
/// bounded in the data and it is bounded here.
const ROCK_BLOCK_WEIBULL_MAX: f32 = 2.0;

// The scree (`docs/research/scree.md`)
//
// A scree is a *scatter* of stones of many sizes with turf between them, not a band of
// mixed colour. The stones come from the cliff above them, and their sizes follow the cliff's
// fracture spacing: a real talus runs from 2 mm to 575 mm, with the median moving by more than
// 10-fold between sites (Verdian et al. 2021, Earth Surface Dynamics 9:1073).

/// The cell network's cell, m: the *spacing* of the stones in it. A scree's stones are fist-
/// to boulder-sized — Verdian et al. 2021 measure the median b-axis at 10 to 575 mm — and at
/// 0.45 m the cell holds a 0.2 m stone beside a 0.5 m one, which is what a scree looks like
/// from four metres away.
const SCREE_CELL_M: f32 = 0.45;
/// The spread of the stones' sizes, as the range of the per-cell factor: the smallest stone is
/// 0.35 of its cell and the largest 2.2 of it, a 6:1 range. Verdian et al.: 2 mm to 575 mm.
const SCREE_SIZE_MIN: f32 = 0.35;
const SCREE_SIZE_MAX: f32 = 2.20;
/// How much of the cell the largest stone fills: the cell is a *spacing*, not a stone's size,
/// and a stone that filled its cell would touch its neighbours and the scatter would be a
/// pavement. At 0.72 the turf shows between the stones.
const SCREE_FILL: f32 = 0.72;
/// How dark a stone is against the ground it lies on, and how much the turf's colour shows
/// between them: the scree is *stones with turf between them*, not a sheet of stone.
const SCREE_STONE_TINT: f32 = 0.34;
/// The stone's relief, as the *slope* of its dome at the rim. The distance field's gradient
/// tilts the normal outward, so the sun rakes the stone's edge and it reads as a thing rather
/// than as a stain. It is a slope and not a height, so it does not depend on the stone's size:
/// a big stone is a big dome and a small one a small dome, and both catch the light the same
/// way.
const SCREE_RELIEF: f32 = 0.30;

/// Rock's own material: the colour the rock's texture gives it, the grain's bump gradient, and
/// whether the grain survives the pixel's footprint. A cliff face is bedded, jointed and
/// two-toned; without this it is one flat colour, which is what the crate had
/// (`rock-material.md` Model 2).
struct RockMaterial {
    /// A multiplier on the rock's albedo: the beds' tone, the fractures' darkening, and the
    /// grain's modulation.
    tint: vec3<f32>,
    /// The grain's bump gradient, in the world.
    grad: vec3<f32>,
    /// 1 where the rock's own detail survives the pixel's footprint.
    present: f32,
    /// How much of the sky the point sees, 0 to 1: the joints' and fractures' own darkening
    /// ("a crack is a shadow: the face beside it sees less sky"), which hides the sky's
    /// reflection as it hides its light, so a wet joint stays dark (`wet-ground.md`).
    sky: f32,
}

/// The rock's material at the world point `p`, for a pixel footprint of `footprint_m` metres.
///
/// The **beds** are bands of the world height, warped by a low-frequency field so they dip and
/// undulate like real bedding instead of ruling a stripe pattern; a second, darker tone is mixed
/// in by them, so the cliff has two rock colours. The **fractures** are *lines* — the joints
/// along which rock breaks — and they are the one feature of a cliff a player reads as "solid" at
/// a distance. The **grain** is the rock's own band-limited octaves, so the rock's surface does
/// not carry the grass's tufts.
///
/// The grain's octaves and the fractures' octaves are **one** set of noise evaluations: a
/// fracture is a ridge of the same field the grain is made of, so the ridged transform is
/// applied to the noise the grain already has. That is half the noise the two would cost
/// separately, and the opaque pass is the crate's hot path (measured: the mountains' opaque
/// pass went from 0.145 ms to 0.309 ms with the two sums separate).
fn rock_material(p: vec3<f32>, warp: f32, footprint_m: f32) -> RockMaterial {
    // The beds: the rock's *own* bedding plane, not the world's height. A bed is a surface
    // inside the rock, and it is usually tilted — and if the beds followed the world height
    // they would be the terrain's own contours, which is a topographic map: the massif came
    // out banded with rings round every hill (measured: the mountain from 700 m was a set of
    // concentric contour bands, and no rock is bedded in the shape of the hill above it).
    // The bed's coordinate is a *tilted* height, so the beds cut across the terrain at an
    // angle, and the warp tilts them further.
    let tilted = dot(p, ROCK_BED_TILT);
    let bed = (tilted + warp * ROCK_BED_WARP_M) / ROCK_BED_M;
    // The bed's own value: the *stacked* bed — which bed this is, how thick it is, where its
    // edges are, and what tone it wears (`docs/research/bedding.md`). The beds stack, so their
    // thicknesses vary and they are a succession and not a stripe pattern; and the bed's
    // *edge* is band-limited away with the distance while the bed's *tone* is not, so a
    // distant cliff shows its layers and not its stripes.
    // The bed's own colour is band-limited against the *fragments*, and the fragments' own
    // band-limit is a **known** number — the joint's width — so the limit is computed here
    // and the blocks' lookup below is not needed for it. The two are band-limited against
    // each other exactly once: where a fragment resolves it wears its own rock, and where it
    // is gone the bed's own rock shows through.
    let bed_colour_keep = 1.0
        - detail_keep(ROCK_BLOCK_M * ROCK_JOINT_WIDTH * 1.85, footprint_m);
    let bed_value = bed_value(p, bed, footprint_m, bed_colour_keep);
    let bed_mask = bed_value.mask;
    // The rock's own field: two octaves, one coarse and one fine, band-limited by their own
    // feature sizes. The fractures are the ridges of the coarse one.
    let coarse = detail_scale_3d(p, ROCK_GRAIN_COARSE_M, 1u, footprint_m, 51u);
    let fine = detail_scale_3d(p, ROCK_GRAIN_FINE_M, 1u, footprint_m, 61u);
    let coarse_norm = max(coarse.norm, 1e-6);
    let fine_norm = max(fine.norm, 1e-6);
    let grain_value = 0.5 * (coarse.sum / coarse_norm + fine.sum / fine_norm);
    let present = max(coarse.present, fine.present);
    // The relief's gradient is the *coarse* octave's alone. The fine octave's gradient is
    // anisotropic on a slope: a 3D noise's gradient has a vertical component, and the surface
    // compresses it, so the fine relief was a directional weave (measured: the near band of a
    // rock face had a structure-tensor coherence of 0.71 at 29-39 degrees, against the flat
    // ground's 0.36). The fine octave stays in the albedo, where a direction is invisible.
    let world_grad = coarse.grad / coarse_norm;
    // The fractures: the ridge of the coarse octave, which is near zero on a line where the
    // noise crosses zero. A Worley cell network cannot make them — it bounds *cells*, so a face
    // reads as a giraffe's hide (measured: the cell pattern dominated the face at 2 m, and a
    // strong domain warp turned it into coral).
    // The fractures are band-limited by *their own* width, not by the cell they live in: a
    // crack is a line a fraction of a cell wide, so it aliases long before the cell does.
    // Band-limited by the cell, the cracks survived as a fine corduroy over the whole massif
    // at 700 m (measured: the mountain read as fabric, not stone).
    let fracture_keep = detail_keep(ROCK_GRAIN_COARSE_M * ROCK_FRACTURE_WIDTH, footprint_m);
    let ridged = 1.0 - abs(coarse.sum / coarse_norm) * abs(coarse.sum / coarse_norm);
    let crack = (1.0 - smoothstep(ROCK_FRACTURE_WIDTH, ROCK_FRACTURE_WIDTH * 3.0, ridged)) * fracture_keep;
    // The colour: the beds' own tone, the fractures' darkening, and the grain's modulation.
    //
    // A bed **is** a colour: the succession's tones are the massif's colour at a distance, and
    // they are a *low-frequency* feature, so they survive every distance. The bed's own tone
    // is a draw on the bed's index, so the beds of a cliff differ from one another the way a
    // succession's do.
    //
    // The *edge* is the other half: the seam between two beds, where the second tone — a
    // browner and darker rock — shows. The edge is a line a fraction of a bed wide, so it is
    // band-limited away with the distance, and what is left is the succession's tones and not
    // its stripes. That is the whole trick.
    // The rock's *blocks*: the pieces it breaks into (`docs/research/rock-blocks.md`). Their
    // own colour, the joint's darkening and their own relief — the three things that turn a
    // smooth face into a face made of pieces.
    //
    // The lookup is here and not lower down because the bed's own colour is band-limited
    // against it: the fragments carry the near field and the beds carry the far one.
    let blocks = rock_blocks(p, footprint_m);
    let blocks_present = blocks.present;
    // The bed's own **colour** is mixed in only where the *fragments* do not resolve
    // (`docs/research/bed-colour.md`). The fragments carry the near field and the beds carry
    // the far one, and the two are band-limited against each other: where a fragment is
    // visible it wears its own rock, and where it is gone the bed's own rock shows through.
    // Without the split the two palettes would compound and the near face would be tinted.
    let bed_tone = bed_value.tone
        * mix(vec3<f32>(1.0), ROCK_SECOND_TONE, bed_mask * bed_value.present * ROCK_BED_MIX);
    var tint = bed_tone;
    var sky = 1.0;
    if blocks_present > 0.0 {
        // The block's own **colour** multiplies the rock's colour, and the joint darkens it.
        let joint = mix(1.0, ROCK_JOINT_DARKEN, blocks.joint);
        tint = tint * blocks.tone * joint;
        sky = joint;
    }
    let dark = mix(1.0, ROCK_FRACTURE_DARKEN, crack);
    let grain_tint = 1.0 + ROCK_GRAIN_AMP * grain_value;
    // The rock's relief. A face with no relief is a painted colour: the light does no work on
    // it, and it reads as fog or as concrete. The grain's own gradient gives the surface its
    // small relief, and the fractures give it the *step* a crack has: the ridge's gradient
    // turns the crack from a drawn line into a groove the sun rakes.
    var relief = world_grad * (ROCK_GRAIN_AMP * 0.20);
    // The blocks' own relief: the joint's gradient, so the sun rakes each fragment's edge.
    if blocks_present > 0.0 {
        relief = relief + blocks.grad;
    }
    // The rock's **rind**: the film of iron and manganese on the surface. It is a separate
    // *layer* and not a tint — Tansem & Storemyr's "rocks with red iron films, rocks that are
    // now generally gray" are the same rock, and the film is what the surface shows
    // (`docs/research/rock-palette.md`).
    //
    // The film's **coverage** is a fraction of the surface and it is patchy: the varnish's
    // "percentage of clast surfaces covered" is the age indicator (Reneau et al. 1993), so the
    // film covers *some* of the rock and not all of it. The patchiness is a field at the
    // film's own scale, and the film is band-limited away with the distance — it is a fine
    // feature, a coating 100 to 500 um thick, and the rocks' own colours are what survive.
    let rind_keep = detail_keep(ROCK_RIND_M, footprint_m);
    if rind_keep > 0.0 {
        // `patch` is a WGSL reserved word, so the film's own field is the *film*.
        let film = perlin3(p / ROCK_RIND_M + seed_offset(83u)) / 0.2268;
        // The coverage: the film is on the faces that have weathered longest, and it is a
        // *fraction* — so the coverage is a threshold on the film's own field, softened.
        let coverage = smoothstep(0.05, 0.45, film) * rind_keep;
        tint = mix(tint, tint * ROCK_RIND_TONE, coverage * ROCK_RIND_MIX);
    }
    return RockMaterial(
        tint * dark * grain_tint,
        relief,
        present,
        sky * dark,
    );
}

/// The rock's *blocks*: the pieces the rock breaks into, at the world point `p` (metres), for a
/// pixel footprint of `footprint_m` metres per pixel.
///
/// A rock mass is a set of blocks held in one piece (Zhang et al. 2020: "discontinuous in
/// geometry while continuous in mechanics"), and a cliff face shows the blocks. They are the
/// cells of a **3D** cellular network, in the rock's own frame: a 2D network in the ground
/// plane would stretch along a slope and weave, exactly as the crate's turf does.
///
/// The blocks are **tabular**: the network's coordinates are squashed along the bedding
/// plane's normal, so the cells stretch into slabs along the beds. A rock breaks along *joint
/// sets* — families of parallel planes — so its pieces are aligned, not a random heap (La
/// Pointe & Hudson 1985).
///
/// A block is *shaded*, not outlined: its own tone, and its own relief from the distance
/// field's gradient, so the sun rakes its edge and the light finds the joint. The crate's
/// rock note is right that a cell network's boundaries read as a giraffe's hide — that is a
/// statement about drawing the boundaries as *lines*; a block is a volume.
struct RockBlocks {
    /// The block's own **colour**: a multiplier on the rock's albedo, one channel each.
    tone: vec3<f32>,
    /// The joint's darkening, 0 to 1: how close the point is to a block's edge.
    joint: f32,
    /// The block's relief gradient, in the world.
    grad: vec3<f32>,
    /// 1 where the point is inside a block and 0 on a joint.
    inside: f32,
    /// 1 where the blocks survive the pixel's footprint.
    present: f32,
}

/// One generation of blocks: the cells of the 3D network at `cell_m`, and the block that the
/// nearest cell owns.
struct RockBlockScale {
    /// The block's own **colour**: a multiplier on the rock's albedo, one channel each, so a
    /// fragment can be another *hue* and not only another brightness.
    tone: vec3<f32>,
    /// The joint's darkening, 0 to 1: how close the point is to a block's edge.
    joint: f32,
    /// The block's relief gradient, in the world.
    grad: vec3<f32>,
    /// 1 where the point is *inside* a block, 0 on a joint.
    inside: f32,
}

/// One generation of the blocks: the 3D network at `cell_m`, the block the nearest cell owns,
/// its tone, its joint and its facet.
///
/// The network's coordinates are **stretched** along the bedding plane's normal so the cells
/// become *shorter* across the beds and so a face shows slabs and not gravel. The bedding
/// plane is the same tilted plane the beds run on: a rock breaks along joint sets, families
/// of parallel planes, so its pieces are aligned.
///
/// The component along the normal is *multiplied*, so the lattice is finer across the beds
/// and the cells are long in the bedding's plane. (Squashing it instead — the other way
/// round — makes the cells long *across* the beds, and the face reads as gravel.)
fn rock_block_scale(p: vec3<f32>, cell_m: f32, footprint_m: f32, jointed: bool) -> RockBlockScale {
    let normal = normalize(ROCK_BED_TILT);
    let along = p - normal * dot(p, normal) * (1.0 - 1.0 / ROCK_BLOCK_TABULAR);
    // The coordinates are **warped** before the lookup, so the cells' boundaries wander: a
    // cell network's cells are convex polygons of nearly the same area, and a face of them
    // reads as a mosaic — or, with the blocks' step, as *shingles*. A rock breaks into
    // irregular fragments, and a warp is what makes the fragments irregular.
    //
    // The warp is the crate's own 3D noise at the 3D point, so it does not weave on a slope
    // (the rock's own rule: the material is "the crate's 3D noise at the 3D point, which has
    // no projection at all"). Its scale is a fraction of the cell, so a block's *shape* is
    // broken and not its *position*.
    //
    // The warp vector is the noise's own **gradient** — one evaluation for a whole 3D vector
    // field, where three separate noise *values* would be three lattice walks. The gradient is
    // the better field anyway: it is smooth and it *points* somewhere, so the cells' boundaries
    // are pushed along a consistent direction rather than at random.
    //
    // `perlin3_d` returns (value, ∂x, ∂y, ∂z), so the gradient is **free** with the value.
    //
    // The noise's *scale* is many cells, so the warp displaces the lattice **coherently**: a
    // warp that varies over the length of a cell moves different points by different amounts
    // and *scrambles* the lattice instead of displacing it — the cell's index stopped being
    // monotonic along a straight cut, and the joint was crossed zero times in 30 m. A warp
    // that varies over ten cells bends the lattice, which is what a rock's fracture set does.
    //
    // The noise is sampled at the *unwarped* point, so the warp does not chase itself.
    let warp_sample = perlin3_d(p / (cell_m * ROCK_BLOCK_WARP_CELLS));
    let warp = warp_sample.yzw;
    let q = (along + warp * ROCK_BLOCK_WARP_M) / cell_m;
    let cell = cell3(q);
    // The **second joint set**: a family of parallel planes across the bedding, and the blocks
    // between the two sets are *slabs* (`docs/research/block-sizes.md`).
    //
    // A family of planes is a *period*, and a period is a coordinate: the second set's planes
    // are at the multiples of its spacing, so the joint is where the coordinate crosses one.
    // That is **no lookup at all** — a dot product, a division and a `fract` — and it is the
    // whole reason a second set is affordable on the crate's hottest path.
    //
    // The **spacing** is a draw, and it is the spacing's own distribution: the fracture
    // spacing's is a Weibull (Verdian et al. 2021: the Weibull "yielded the best fit to the
    // data in most cases", with a geometric standard deviation of 0.214 to 0.551 across the
    // fifteen sites). The draw is the *formation's* own hash, so the spacing is **constant
    // within a formation** and changes from one formation to the next — a spacing that varied
    // *smoothly* would make the planes fold instead of fracture, and the face came out a moiré
    // of concentric rings.
    let formation = floor(p / ROCK_FORMATION_M);
    let formation_hash = hash31(vec3<i32>(
        i32(formation.x),
        i32(formation.y),
        i32(formation.z),
    ));
    // The spacing's own distribution is a **Weibull** (Verdian et al. 2021: the Weibull form
    // "yielded the best fit to the data in most cases", with a geometric standard deviation of
    // 0.214 to 0.551 across the fifteen sites), so the draw is the Weibull's inverse
    // transform and not a power of a uniform: a power of a uniform concentrates the values
    // near zero, and the spacing's spread came out at 2.2:1 where the Weibull's is wider.
    let spacing_draw = min(
        pow(-log(max(1.0 - formation_hash, 1e-6)), 1.0 / ROCK_BLOCK_WEIBULL),
        ROCK_BLOCK_WEIBULL_MAX,
    );
    let spacing = mix(
        ROCK_JOINT2_MIN_M,
        ROCK_JOINT2_MAX_M,
        spacing_draw / ROCK_BLOCK_WEIBULL_MAX,
    );
    // The second set's coordinate: the distance from the plane, in metres.
    let cross = dot(p, normalize(ROCK_JOINT2_NORMAL));
    // The distance to the nearest plane of the second set.
    let phase = cross / spacing;
    let to_plane = abs(phase - round(phase)) * spacing;
    // The block's own **colour**: a draw from the rock's palette, so the cliff is a
    // succession of *different rock* and not one grey in many shades
    // (`docs/research/rock-palette.md`).
    //
    // The palette's score is the *distribution* (which rocks are around) times the
    // *selection* (which rock this fragment is). The **selection** is the cell's own factor,
    // so a fragment is a **piece of one rock** and not a blend of several — which is what a
    // fracture gives you: a broken piece of a bed, and a bed is one rock.
    //
    // The **distribution** is the cell's *second* draw, so it is at the *fragment's* scale and
    // not at a field's: a cliff's colour must vary from fragment to fragment, or the cliff is
    // a set of coloured regions and every fragment in a region is one rock. A field at six
    // cells was tried first and it read as blotches.
    let draw = cell.factor;
    let region = cell.factor2;
    // The four scores: the palette's members. The selection splits the palette in two — the
    // *cool* half and the *warm* half — and the distribution splits each half again, so the
    // four members are the four combinations.
    //
    // The scores are **squared** — Papavasiliou's exponent `p`, and the crate's own reason
    // for squaring the turf's: a square is a multiply and not a `pow`, and three `pow` calls
    // per pixel cost the mountains' opaque pass 0.137 ms → 0.252 ms. The square also makes
    // the *selection* the dominant term, so a fragment is emphatically ONE rock and not a
    // blend of two — and the blend is what left the cliff at one hue.
    // The two splits as weights, so a fragment on the *edge* of a split is a blend of two
    // rocks and a fragment in the middle of one is emphatically that rock. The width of the
    // edge is the *score's* own sharpness, Papavasiliou's `p`.
    let sel_w = smoothstep(0.5 - ROCK_PALETTE_EDGE, 0.5 + ROCK_PALETTE_EDGE, draw);
    let reg_w = smoothstep(0.5 - ROCK_PALETTE_EDGE, 0.5 + ROCK_PALETTE_EDGE, region);
    let s_basalt = (1.0 - sel_w) * (1.0 - reg_w);
    let s_sandstone = (1.0 - sel_w) * reg_w;
    let s_limestone = sel_w * (1.0 - reg_w);
    let s_varnish = sel_w * reg_w;
    let shares = s_basalt + s_sandstone + s_limestone + s_varnish;
    let inv = 1.0 / max(shares, 1e-5);
    // The mixture: a convex combination of the palette, so it lies between the tones and the
    // rock's mean colour cannot drift (the crate's own rule from the ground's palette).
    let mixed = (s_basalt * ROCK_TONE_BASALT
        + s_sandstone * ROCK_TONE_SANDSTONE
        + s_limestone * ROCK_TONE_LIMESTONE
        + s_varnish * ROCK_TONE_VARNISH) * inv;
    // The palette's *value* spread: the fragments are lighter and darker pieces of the same
    // rock as well, the way a bed's own tone is.
    let value = 1.0 + ROCK_BLOCK_TONE * (draw * 2.0 - 1.0);
    // The block's own colour, and its brightness.
    let tone = mix(vec3<f32>(value), mixed * value, ROCK_PALETTE_STRENGTH);
    // The joint: the cell's own **boundary** — the surface where two cells meet, so a point
    // close to it is at a joint and a point deep inside it is in the middle of a block.
    //
    // The boundary is the *nearest point's own cell's* far side, measured from the point: the
    // feature point sits somewhere inside its cell, and the cell's wall is the distance from
    // the point to the cell's edge *away* from it. That is the standard construction for a
    // cell's boundary, and it needs only the **nearest** point — the second-nearest point's
    // distance would give the same answer at twice the cost, and the rock is the crate's
    // hottest path.
    let to_centre = cell.point - fract(q);
    let len = max(length(to_centre), 1e-6);
    let outward = to_centre / len;
    // The distance from the point to its cell's wall, along the direction away from the
    // feature point: the point is at `fract(q)` inside the cell, and the cell's wall is at
    // the far side of it.
    let inside = fract(q);
    let to_wall = select(1.0 - inside, inside, outward < vec3<f32>(0.0));
    let half = min(min(to_wall.x, to_wall.y), to_wall.z);
    // The joint's own width is a fraction of the cell, the way a crack is (the crate's own
    // rule for the fractures: "a crack is a line a fraction of a cell wide").
    //
    // The joint's *width* is the cell's own draw, and it is what sets the block's **size**: a
    // cell whose joint is wide is a *small* piece of rock and one whose joint is narrow is a
    // big slab. The cells of a network are all the same size, so without this every block
    // would be the same and the face would read as a cobblestone. The cell's factor is used
    // for the block's tone *and* for its joint's width, so a cell is consistently one piece —
    // a small block is a differently coloured one as well.
    //
    // The draw is **a Weibull**, the shape the fracture spacing's distribution follows
    // (Verdian et al. 2021: "the Weibull form of the exponential distribution … yielded the
    // best fit to the data in most cases"), so the widths have the distribution a real cliff's
    // spacings have and not the uniform one a linear draw has. The Weibull's inverse
    // transform is `λ · (−ln(1 − u))^(1/k)`, and the *shape* `k` below is 1.6: a cliff's
    // spacings have a longer tail than a uniform draw and a shorter one than a pure
    // exponential.
    // The draw is **clamped** at the 95th percentile, and the note says why: "the lower
    // tails of the distributions follow a steeper trend than the rest of the data ...
    // potentially reflecting an undersampling of the smallest fractures". A Weibull's tail is
    // unbounded, and an unbounded joint's width eats the block: at the 99th percentile the
    // width was 47 % of the cell's half-boundary, so a block in one cell was a sliver.
    let weibull = min(
        pow(-log(max(1.0 - cell.factor, 1e-6)), 1.0 / ROCK_BLOCK_WEIBULL),
        ROCK_BLOCK_WEIBULL_MAX,
    );
    let width = ROCK_JOINT_WIDTH * (0.25 + ROCK_BLOCK_WEIBULL_SPREAD * weibull);
    // The joint's edge is *softened* over the pixel's own footprint: a joint is a line a
    // fraction of a cell wide, and a hard edge on a line that narrow stair-steps at every
    // distance. The width grows with the footprint, so the joint blurs into the rock rather
    // than aliasing along it.
    let soft = max(width, footprint_m / (cell_m * 0.5));
    // The **second set's** joint: the plane of the second family, and its own width. A joint
    // is a feature of the rock's own scale, so the second set's width is a fraction of *its*
    // spacing and not of the cell's.
    //
    // The joint is **not** a straight line: a joint set's planes are *non-persistent* — they
    // end, and their spacing varies along them, which is the whole difference between a
    // fracture network and a lattice (Strauhal & Zangerl 2021 on the difficulty of
    // determining the block size distribution "when non-persistent fractures in a rock mass
    // should be considered"). A perfectly straight, evenly spaced plane set reads as a **wire
    // fence** across the face, and it is the second set's *regularity* that gave the face away
    // and not its spacing.
    //
    // So the plane's *position* is displaced by a slow field — the same field the warp uses,
    // which is a free component of the warp's own `perlin3_d` — and the displacement is a
    // fraction of the spacing, so the plane wanders within its own cell of the family.
    let wander = warp_sample.x * spacing * ROCK_JOINT2_WANDER;
    // The planes are **non-persistent**: they end, and a second, finer wander breaks them
    // into segments. One smooth wander made the planes *bend* but not *end*, and a bend
    // across a whole face still reads as a wire fence — the lines ran unbroken from edge to
    // edge. The finer field is at the *segment's* scale, a few spacings long, so the plane
    // wanders by a fraction of a spacing over a few metres and then the joint is simply
    // **absent**.
    //
    // The absence is a *threshold* on the finer field, and it is the finest thing the joint
    // draws, so it is band-limited with the joint's own width: a segment that is shorter
    // than the pixel's footprint is not drawn at all, and the distance stays calm.
    //
    // The finer field is only evaluated **where the point is near a plane**, and the plane's
    // distance is already known: a joint is a line a fraction of the spacing wide, so the
    // overwhelming majority of the face is *not* on a plane and the field is computed and
    // thrown away. The gate is the plane's own width, so the field is fetched exactly where
    // it can change the answer.
    let cross_w = cross + wander;
    let to_plane_w = abs(cross_w / spacing - round(cross_w / spacing)) * spacing;
    let soft2 = max(spacing * ROCK_JOINT2_WIDTH, footprint_m * 0.5);
    var joint2 = 0.0;
    {
        let segment = perlin3(p / (spacing * ROCK_JOINT2_SEGMENT_M) + seed_offset(137u));
        let present2 = smoothstep(
            -ROCK_JOINT2_BREAK,
            ROCK_JOINT2_BREAK,
            segment,
        );
        joint2 = (1.0 - smoothstep(0.0, soft2, to_plane_w)) * present2;
    }
    // The two sets' joints together: the block is bounded by both, so the block's *size* is
    // the smaller of the two — a slab where the second set is close, a block where it is far.
    // The two sets' joints together: the block is bounded by both, so the block is a slab
    // where the second set is close and a block where it is far.
    let joint = max(1.0 - smoothstep(0.0, soft, half), joint2);
    // The block's *facet*: a rock block's face is **flat**, not domed, and the step is at the
    // joint. So the relief is the joint's own gradient — the boundary's direction — and the
    // block's face is level. A dome over each block reads as a pebble pressed into the face
    // (which is what the first attempt rendered), and a rock face is not that.
    //
    // The step's gradient: it rises towards the joint, so the face tilts back at the edge. Its
    // width is the joint's own softened width, so the step and the darkening are the same
    // feature.
    //
    // The second set's step is along *its* normal, so the two sets' steps cross and the
    // block's edge is a *corner* — which is what a block between two joint sets is.
    let step = (1.0 - smoothstep(0.0, soft * 2.0, half)) * ROCK_BLOCK_RELIEF;
    let step2 = (1.0 - smoothstep(0.0, soft2 * 2.0, to_plane)) * ROCK_BLOCK_RELIEF;
    let grad = (-outward * step + normalize(ROCK_JOINT2_NORMAL) * step2) / cell_m;
    // A generation that is **not jointed** carries its tone and no crack and no relief: the
    // fine generation is the *chips* lying on a face, and a crack round every chip — and a
    // relief step at every chip — turned the face into a dark speckle. A chip is a patch of a
    // slightly different rock, and its edge is a shadow the *coarse* generation's own relief
    // already draws.
    if !jointed {
        return RockBlockScale(tone, 0.0, vec3<f32>(0.0), 1.0);
    }
    return RockBlockScale(tone, joint, grad, 1.0 - joint);
}

fn rock_blocks(p: vec3<f32>, footprint_m: f32) -> RockBlocks {
    // Band-limited by the **joint's own width**, and not by the cell: the joint is the finest
    // thing the blocks draw, and it is a line a fraction of a cell across, so it aliases long
    // before the cell does. The crate's own rule for the fractures, and the reason they are
    // band-limited by their own width: "a crack is a line a fraction of a cell wide, so it
    // aliases long before the cell does". Band-limited by the cell, the blocks survive at 90 m
    // and the massifs read as a scaly, reptilian texture.
    //
    // The *widest* joint's width is the band-limit, and not the cell's own: a band-limit that
    // varies per cell would flicker as the cell's draw changes from pixel to pixel.
    let keep = detail_keep(ROCK_BLOCK_M * ROCK_JOINT_WIDTH * 1.85, footprint_m);
    // The band-limit is *known before the network is looked up*, and the lookup is the
    // expensive part — 27 cells, each a hash and a distance. So where the blocks cannot
    // resolve, they are not looked up at all: the far terrain pays nothing for them, and the
    // mountains' opaque pass does not carry the cost over the whole massif.
    if keep <= 0.0 {
        return RockBlocks(vec3<f32>(1.0), 0.0, vec3<f32>(0.0), 0.0, 0.0);
    }
    // The coarse generation: the big slabs.
    let coarse = rock_block_scale(p, ROCK_BLOCK_M, footprint_m, true);
    // The fine generation: the small fragments in the gaps between the big ones. A cliff's
    // blocks are not all the same size — the fracture spacing follows a Weibull whose
    // geometric standard deviation is about 0.31 (Verdian et al. 2021, whose fifteen sites run
    // 0.214 to 0.551), and the spacings run from 2 mm to 5000 mm against a site median of 10 to
    // 390 mm — so a face of equal cells is a *quilt*, and the quilt is what the capture
    // showed.
    //
    // The fine network is only looked up where the coarse one *resolves*: it is a third of
    // the cell, so it is gone long before the coarse one is, and the far terrain pays
    // nothing for it.
    var tone = coarse.tone;
    var joint = coarse.joint;
    var grad = coarse.grad;
    var inside = coarse.inside;
    let fine_keep = detail_keep(ROCK_BLOCK_M * ROCK_BLOCK_FINE * ROCK_JOINT_WIDTH * 1.85, footprint_m);
    if fine_keep > 0.0 {
        let fine = rock_block_scale(p, ROCK_BLOCK_M * ROCK_BLOCK_FINE, footprint_m, false);
        // The fine fragment only shows where the coarse block does *not* cover, so a small
        // fragment never grows out of the side of a big one. This is the scree's own
        // construction (`docs/research/scree.md`), and the reason one network at one size
        // reads as a grid of pebbles and two nested read as a scatter.
        let fine_inside = fine.inside * (1.0 - coarse.inside);
        if fine_inside > 0.0 {
            // A fine fragment carries its **tone** and not a joint: a chip lying on a face is
            // a patch of a slightly different rock, and a *crack* round every chip turned the
            // face into a dark speckle. The darkening is a crack, and a chip is not a crack —
            // the joint belongs to the big blocks, which are the ones that parted.
            tone = select(fine.tone, coarse.tone, coarse.inside > fine_inside);
            grad = select(fine.grad, coarse.grad, coarse.inside > fine_inside);
            inside = max(coarse.inside, fine_inside);
        }
    }
    return RockBlocks(
        mix(vec3<f32>(1.0), tone, keep),
        joint * keep,
        grad * keep,
        inside * keep,
        keep,
    );
}

/// The scree: a scatter of stones with turf between them, at `p` (metres) in the world XZ
/// plane, for a view whose footprint is `footprint_m` metres per pixel.
///
/// The stones are the *cells* of a 2D cellular network: the nearest feature point is a stone's
/// centre, and the distance to it is the stone's radius. The cell's own random factor makes
/// the stones many sizes (Verdian et al. 2021: a talus runs from 2 mm to 575 mm), and the
/// distance field's gradient domes each stone, so the sun rakes its edge and it reads as a
/// thing rather than as a stain.
///
/// The network is in the **world XZ plane**, the turf's own frame, because the scree lies on
/// the near-horizontal surfaces where the plane's projection is the surface; on a steep face it
/// would stretch along the slope, which is the weave the crate's rock note warns about.
///
/// `tint` is the multiplier on the ground's albedo (around 1), `grad` the surface gradient the
/// bump adds, and `coverage` how much of the ground the stones take (the rest is the turf).
struct Scree {
    tint: f32,
    grad: vec2<f32>,
    coverage: f32,
    present: f32,
    /// The stone's occlusion of the matrix beside it, 0 to 1: the contact shadow that makes
    /// the stones sit *in* the dirt.
    contact: f32,
}

/// One scale's stones: the cell network's cells at `cell_m`, the stone each cell owns, and
/// the size factor that makes the stones many sizes.
struct ScreeScale {
    /// How much of the point the stone covers, 0 to 1.
    inside: f32,
    /// The stone's dome gradient, in the surface's frame (x, z).
    grad: vec2<f32>,
    /// The stone's own tone, a multiplier around 1.
    tint: f32,
    /// The stone's occlusion of the matrix beside it: the contact shadow that makes the
    /// stones sit *in* the dirt.
    contact: f32,
}

/// The stones of one network at `cell_m`: the cell's own stone, its size, its dome and the
/// contact shadow it casts on the matrix beside it.
fn scree_scale(xz: vec2<f32>, cell_m: f32, size_min: f32, size_max: f32) -> ScreeScale {
    let q = xz / cell_m;
    let cell = cell2(q);
    // The stone's own size: the cell's factor, remapped to the spread. A power law's shape —
    // most cells small, a few large — is what a size distribution looks like (Verdian et al.
    // 2021: the median moves by more than 10-fold between sites, and the tail is long).
    let size = mix(size_min, size_max, pow(cell.factor, 3.0));
    // The stone's radius in lattice units.
    let radius = 0.5 * size * SCREE_FILL;
    let d = cell.distance;
    let inside = 1.0 - smoothstep(radius * 0.80, radius, d);
    // The contact shadow: the matrix just outside the stone, in the stone's own occlusion of
    // the ground it lies on. The ground's shader has no access to the sun's direction, so this
    // is not a *cast* shadow — it is what makes the stones sit in the dirt rather than on it.
    let contact = smoothstep(radius * 0.80, radius, d)
        * (1.0 - smoothstep(radius, radius * 1.9, d));
    if inside <= 0.0 {
        return ScreeScale(0.0, vec2<f32>(0.0), 1.0, contact);
    }
    // The stone's dome: the distance field's gradient, which points away from the stone's
    // centre, tilts the normal outward.
    let to_centre = cell.point - fract(q);
    let len = max(length(to_centre), 1e-6);
    let outward = to_centre / len;
    // The dome's slope at the point: the slope of a dome of a fixed shape, so it does *not*
    // depend on the stone's size. A big stone is a big dome and a small one a small dome, and
    // both catch the light the same way. (A slope that scaled with the cell instead made the
    // smallest stones' rims tilt by 85°, and they rendered as black crescents.)
    let rim = clamp(d / max(radius, 1e-6), 0.0, 1.0);
    let slope = SCREE_RELIEF * rim / sqrt(max(1.0 - rim * rim, 0.05));
    // The gradient is a *tilt*, the same for every stone: the surface's normal is turned by
    // the dome's slope, in the surface's own frame. It is the dimensionless slope, so it does
    // not carry the cell's scale — a stone in the fine network is a small stone with the same
    // tilt at its rim as a big one.
    let grad = -outward * slope;
    // The stone's own tone: the rock, a little darker than the bare rock it came from (a stone
    // in the shadow of its neighbours). The grain is the same for both scales and is
    // evaluated once, by the caller.
    let tint = 1.0 - SCREE_STONE_TINT * 0.35;
    return ScreeScale(inside, grad, tint, contact);
}

fn scree_scatter(xz: vec2<f32>, footprint_m: f32) -> Scree {
    // Band-limited by the *coarse* cell, like the fractures: a stone is a fraction of a cell
    // across and a pixel at 700 m is a metre, and a scatter that is not band-limited is a
    // corduroy.
    let keep = detail_keep(SCREE_CELL_M, footprint_m);
    // The band-limit is known before the networks are looked up, and the lookups are the
    // expensive part, so where the stones cannot resolve they are not looked up at all.
    if keep <= 0.0 {
        return Scree(1.0, vec2<f32>(0.0), 0.0, 0.0, 0.0);
    }
    // Two networks, nested: a coarse one whose cells hold the big stones and a fine one whose
    // cells hold the small ones. That is how a scree is built — a size distribution with a
    // long tail, and the tail is the small stones filling the gaps between the big ones
    // (Verdian et al. 2021: the finest talus clasts are 2 mm, against a median of 10 to
    // 575 mm). One network at one size reads as *pebbles on grass*; two nested read as a
    // scree.
    let coarse = scree_scale(xz, SCREE_CELL_M, SCREE_SIZE_MIN, SCREE_SIZE_MAX);
    let fine = scree_scale(xz, SCREE_CELL_M * 0.38, SCREE_SIZE_MIN * 0.8, SCREE_SIZE_MAX * 0.8);
    // The stone's own grain, so it is not a flat patch: one evaluation for both scales.
    let grain = perlin3(vec3<f32>(xz.x, 0.0, xz.y) / 0.11);
    // The contact shadows: the matrix beside each stone, in both networks.
    let contact = max(coarse.contact, fine.contact);
    // The fine stones only show where the coarse one does not cover, so a fine stone never
    // grows out of the side of a big one.
    let fine_inside = fine.inside * (1.0 - coarse.inside);
    let inside = max(coarse.inside, fine_inside);
    if inside <= 0.0 {
        return Scree(1.0, vec2<f32>(0.0), 0.0, keep, contact * keep);
    }
    // The gradient is the stone's own, and the fine one's only where the fine stone is.
    let grad = select(fine.grad, coarse.grad, coarse.inside > fine_inside);
    let tint = select(fine.tint, coarse.tint, coarse.inside > fine_inside) * (1.0 + 0.35 * grain);
    return Scree(
        mix(1.0, tint, keep),
        grad * inside * keep,
        inside * keep,
        keep,
        contact * keep,
    );
}

/// The ground's detail at `p` (metres) for a view whose footprint is `footprint_m` metres per
/// pixel: two octave sums, each band-limited by their own scale, whose albedo modulation and bump
/// gradient replace the one fixed 0.8 m⁻¹ band faded by a distance constant. `albedo` is a
/// multiplier around 1; `grad` is the surface gradient the bump adds.
///
/// The octaves are the crate's 3D noise at the 3D point, so the detail is the same field on
/// every surface and does not weave the cliff faces (`rock-material.md` Model 1).
fn ground_detail(p: vec3<f32>, footprint_m: f32) -> GroundDetail {
    let coarse = detail_scale(p, DETAIL_COARSE_M, DETAIL_COARSE_OCTAVES, footprint_m, 1u);
    let fine = detail_scale(p, DETAIL_FINE_M, DETAIL_FINE_OCTAVES, footprint_m, 21u);
    let coarse_value = coarse.sum / max(coarse.norm, 1e-6);
    let fine_value = fine.sum / max(fine.norm, 1e-6);
    let world_grad = DETAIL_COARSE_HEIGHT_M * coarse.grad + DETAIL_FINE_HEIGHT_M * fine.grad;
    return GroundDetail(
        1.0 + DETAIL_COARSE_AMP * coarse_value + DETAIL_FINE_AMP * fine_value,
        world_grad,
        max(coarse.present, fine.present),
        coarse_value,
        fine_value,
    );
}
