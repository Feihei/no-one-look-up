//! CPU transliterations of the terrain kernels that the tests compare against: the procedural
//! generator of `generate.wesl` and the layer weights of `terrain.wesl`.

use bevy::math::{IVec2, IVec3, Mat2, UVec2, Vec2, Vec3};

use super::{F_HALF, GAIN, LACUNARITY, OCTAVE_ROTATION, R_LOW, R_SPAN, octave_count, seed_mix};
use crate::gpu::noise_mirror::{hash33, perlin3_d};
use crate::terrain::ProceduralTerrain;

/// Both noise sums of a sample point: `(f, r)` of the research note.
pub(crate) fn noise_sums(local_xz: Vec2, feature_m: f32, octaves: u32, seed: u64) -> (f32, f32) {
    let p = local_xz / feature_m;
    let mix = seed_mix(seed);
    let (mut f_sum, mut r_sum, mut norm, mut gain) = (0.0, 0.0, 0.0, 1.0);
    let mut weight = 1.0_f32;
    for k in 0..octaves {
        let (s, c) = (OCTAVE_ROTATION * k as f32).sin_cos();
        let q = Vec2::new(c * p.x - s * p.y, s * p.x + c * p.y) * LACUNARITY.powi(k as i32);
        let v = ((mix + k * 1031) % 4096) as f32;
        let n = perlin3_d(Vec3::new(q.x + v, q.y + v, 0.5 + k as f32)).x;
        let ridge = (1.0 - n.abs()).max(0.0);
        let s_k = weight * ridge * ridge;
        f_sum += gain * n;
        r_sum += gain * s_k;
        norm += gain;
        weight = (2.0 * s_k).clamp(0.0, 1.0);
        gain *= GAIN;
    }
    (f_sum / norm, r_sum / norm)
}

/// The unit terrain value `(1 − q) u_f + q u_r` of the two sums.
pub(crate) fn unit_height(f: f32, r: f32, ridged: f32) -> f32 {
    let u_f = (0.5 + f / (2.0 * F_HALF)).clamp(0.0, 1.0);
    let u_r = ((r - R_LOW) / R_SPAN).clamp(0.0, 1.0);
    (1.0 - ridged) * u_f + ridged * u_r
}

/// The surface height at a point `local_xz` metres from the terrain's minimum corner, relative
/// to the terrain transform: the base height plus the relief times the unit value.
pub(crate) fn surface_height(p: &ProceduralTerrain, octaves: u32, local_xz: Vec2) -> f32 {
    let (f, r) = noise_sums(local_xz, p.feature_size_m, octaves, p.seed);
    p.base_height_m + p.relief_m * unit_height(f, r, p.ridged.clamp(0.0, 1.0))
}

/// The generated surface at every cell of a `size` grid of `cell_m` cells, row-major.
pub(crate) fn generate_surface(p: &ProceduralTerrain, size: UVec2, cell_m: f32) -> Vec<f32> {
    let octaves = octave_count(p.octaves, p.feature_size_m, cell_m);
    let mut out = Vec::with_capacity((size.x * size.y) as usize);
    for j in 0..size.y {
        for i in 0..size.x {
            let local = (Vec2::new(i as f32, j as f32) + 0.5) * cell_m;
            out.push(surface_height(p, octaves, local));
        }
    }
    out
}

/// The slope angle of a surface at a cell by central differences over ±1 cell (one-sided at the
/// edge), degrees.
pub(crate) fn slope_deg(surface: &[f32], size: UVec2, cell_m: f32, i: u32, j: u32) -> f32 {
    let at = |x: u32, z: u32| surface[(z * size.x + x) as usize];
    let (xl, xh) = (i.saturating_sub(1), (i + 1).min(size.x - 1));
    let (zl, zh) = (j.saturating_sub(1), (j + 1).min(size.y - 1));
    let gx = (at(xh, j) - at(xl, j)) / ((xh - xl) as f32 * cell_m);
    let gz = (at(i, zh) - at(i, zl)) / ((zh - zl) as f32 * cell_m);
    (gx * gx + gz * gz).sqrt().atan().to_degrees()
}

/// Percentile `q` (0–1) of `values` (sorted in place).
pub(crate) fn percentile(values: &mut [f32], q: f32) -> f32 {
    values.sort_by(f32::total_cmp);
    values[((values.len() - 1) as f32 * q).round() as usize]
}

/// `smoothstep(a, b, x)` of WGSL.
pub(crate) fn smoothstep(a: f32, b: f32, x: f32) -> f32 {
    let t = ((x - a) / (b - a)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Andersson 2007 §5.2.3, Listing 2: the Overlay blend of `detail` onto `mask` at full opacity.
pub(crate) fn overlay(mask: f32, detail: f32) -> f32 {
    if mask < 0.5 {
        2.0 * mask * detail
    } else {
        1.0 - 2.0 * (1.0 - mask) * (1.0 - detail)
    }
}

/// Feature size of the coarse ground detail's base octave, m (`terrain/common.wesl`).
pub(crate) const DETAIL_COARSE_M: f32 = 1.25;
/// Feature size of the fine ground detail's base octave, m.
pub(crate) const DETAIL_FINE_M: f32 = 0.125;

/// `detail_keep` of `terrain/common.wesl`: the weight of an octave of feature size `feature_m`
/// in a pixel whose footprint is `footprint_m` wide. 1 while the octave is resolved (at least
/// two footprints per feature), falling to 0 over the band where it is not (Lagae et al. 2010
/// §5.2.1, frequency clamping; `docs/research/ground-detail.md` Model 2).
pub(crate) fn detail_keep(feature_m: f32, footprint_m: f32) -> f32 {
    let t = ((footprint_m / feature_m - 0.5) / 0.5).clamp(0.0, 1.0);
    1.0 - t * t * (3.0 - 2.0 * t)
}

/// The four layer weights `(rock, sand, soil, grass)` of `render.wesl` for a slope angle
/// (degrees), a sediment depth (m) and a detail value in [0, 1] (0.5: none).
pub(crate) fn layer_weights(slope_deg: f32, sediment_m: f32, detail: f32) -> [f32; 4] {
    let rock = overlay(smoothstep(30.0, 38.0, slope_deg), detail);
    let sand = overlay(smoothstep(0.3, 0.7, sediment_m), detail)
        * (1.0 - smoothstep(26.0, 30.0, slope_deg))
        * (1.0 - rock);
    let band = overlay(smoothstep(16.0, 26.0, slope_deg), detail) * (1.0 - smoothstep(22.0, 29.0, slope_deg));
    let rest = (1.0 - rock) * (1.0 - sand);
    [rock, sand, rest * band, rest * (1.0 - band)]
}

/// The scale of the ground's large-scale variation, m (`render.wesl`).
pub(crate) const LAYER_VARIATION_M: f32 = 45.0;
/// How far the variation moves the layer weights (`render.wesl`).
pub(crate) const LAYER_VARIATION_DEPTH: f32 = 0.25;
/// The rms of the crate's 3D noise, measured in the crate's own mirror: the field is
/// normalised by it so that `LAYER_VARIATION_DEPTH` is the fraction of the grass a patch turns
/// to soil.
pub(crate) const NOISE_RMS: f32 = 0.2268;

/// The ground's large-scale variation of the layer weights (`render.wesl`): a low-frequency
/// field that takes soil from the grass, so a plain — where the soil's weight is zero — gets
/// patches instead of one colour. The field is the `ground_fields` function's large one.
pub(crate) fn layer_variation(field: f32, w: [f32; 4]) -> [f32; 4] {
    let field = field.clamp(-1.0, 1.0);
    let shift = field * LAYER_VARIATION_DEPTH * w[3];
    let mut v = w;
    // The shift is taken from the grass and given to the soil, and both ends are clamped: a
    // field more negative than the soil's weight would take the soil below zero (measured:
    // −0.054 on a plain), and a negative weight is a negative albedo.
    let taken = shift.clamp(-v[2], v[3]);
    v[2] += taken;
    v[3] -= taken;
    let total = v.iter().sum::<f32>();
    v.map(|x| x / total.max(1e-6))
}

/// The turf's palette (`terrain/common.wesl`): the four tones, as multipliers on the turf's
/// own colour. The fresh tone is the identity.
pub(crate) const TURF_FRESH: [f32; 3] = [1.00, 1.00, 1.00];
pub(crate) const TURF_DRY: [f32; 3] = [1.35, 1.20, 0.69];
pub(crate) const TURF_STRAW: [f32; 3] = [1.55, 1.37, 0.82];
pub(crate) const TURF_WORN: [f32; 3] = [1.13, 0.95, 0.73];
/// How far the palette moves the ground's colour towards a tone, 0 to 1.
pub(crate) const TURF_STRENGTH: f32 = 0.85;

/// The turf's palette at the given fields (`terrain/common.wesl`): the *shares* of the four
/// tones, which sum to one by construction, and the mixture they make.
pub(crate) fn turf_palette(field: f32, clump: f32) -> ([f32; 3], [f32; 4]) {
    let p = 2.0f32;
    let mut dry = (field.max(0.0) * (0.5 + clump).max(0.0) * 1.6).powf(p).min(1.0);
    let mut straw = ((field - 0.45).max(0.0) * 3.0 * (0.5 + clump).max(0.0) * 1.6)
        .powf(p)
        .min(1.0);
    let mut worn = ((-field - 0.55).max(0.0) * 3.0 * (0.5 - clump).max(0.0) * 1.6)
        .powf(p)
        .min(1.0);
    let others = dry + straw + worn;
    if others > 1.0 {
        let k = 1.0 / others;
        dry *= k;
        straw *= k;
        worn *= k;
    }
    // The fresh share is what is left, and the subtraction can leave it a hair below zero
    // (−2.2e−8 at field 0.5, clump 0.75), which is a negative weight.
    let fresh = (1.0 - dry - straw - worn).max(0.0);
    let shares = [fresh, dry, straw, worn];
    let mut mixed = [0.0f32; 3];
    for c in 0..3 {
        mixed[c] = fresh * TURF_FRESH[c]
            + dry * TURF_DRY[c]
            + straw * TURF_STRAW[c]
            + worn * TURF_WORN[c];
        mixed[c] = 1.0 + (mixed[c] - 1.0) * TURF_STRENGTH;
    }
    (mixed, shares)
}

// The cliff's blocks (`docs/research/rock-blocks.md`)
// ---------------------------------------------------------------------------------------------

/// The block network's cell, m (`common.wesl`): the scale of the pieces the rock breaks into.
pub(crate) const ROCK_BLOCK_M: f32 = 0.95;
/// How far the blocks stretch along the bedding, as a factor (`common.wesl`).
pub(crate) const ROCK_BLOCK_TABULAR: f32 = 2.6;
/// The joint's width, as a fraction of the cell's half-boundary distance (`common.wesl`).
pub(crate) const ROCK_JOINT_WIDTH: f32 = 0.19;
/// The **second joint set's** normal: a cross joint, roughly perpendicular to the bedding and
/// tilted (`common.wesl`).
pub(crate) const ROCK_JOINT2_NORMAL: Vec3 = Vec3::new(0.71, 0.46, -0.53);
/// The second joint set's **spacing range**, m (`common.wesl`).
pub(crate) const ROCK_JOINT2_MIN_M: f32 = 0.55;
pub(crate) const ROCK_JOINT2_MAX_M: f32 = 2.60;
/// The **formation's** cell, m: the scale at which the second set's spacing is constant
/// (`common.wesl`).
pub(crate) const ROCK_FORMATION_M: f32 = 46.0;
/// How far the second set's planes **wander**, as a fraction of their own spacing
/// (`common.wesl`).
pub(crate) const ROCK_JOINT2_WANDER: f32 = 0.48;
/// How long the second set's segments are, in multiples of their own spacing (`common.wesl`).
pub(crate) const ROCK_JOINT2_SEGMENT_M: f32 = 1.9;
/// How sharply a segment ends (`common.wesl`).
pub(crate) const ROCK_JOINT2_BREAK: f32 = 0.30;
/// The second set's joint's width, as a fraction of its own spacing (`common.wesl`).
pub(crate) const ROCK_JOINT2_WIDTH: f32 = 0.075;
/// The **region's** scale, m: the field that biases the bed's selection (`common.wesl`).
pub(crate) const ROCK_BED_REGION_M: f32 = 260.0;
/// The warp that breaks the cells' shape, in metres (`common.wesl`): the warp's vector is the
/// crate's own 3D noise's *gradient*, whose mean magnitude is 0.94, so the amplitude is also
/// the displacement's scale. At 0.50 m over a 1.5 m cell the mean displacement is 0.31 of a
/// cell and the max 0.79, which breaks the cells' regularity without tearing them.
pub(crate) const ROCK_BLOCK_WARP_M: f32 = 0.30;
/// How many cells the warp's noise varies over (`common.wesl`).
pub(crate) const ROCK_BLOCK_WARP_CELLS: f32 = 4.0;
/// The Weibull's shape for the joint's width, the distribution the fracture spacing follows
/// (Verdian et al. 2021). A shape of 1 is the exponential and 2 is the Rayleigh.
pub(crate) const ROCK_BLOCK_WEIBULL: f32 = 1.6;
/// The Weibull's scale, as a multiple of the joint's width.
pub(crate) const ROCK_BLOCK_WEIBULL_SPREAD: f32 = 0.85;
/// The Weibull's draw's ceiling: the 95th percentile, and not the unbounded tail. The
/// fracture spacing's own distribution is *undersampled* in its lower tail, so the tail is
/// bounded in the data and it is bounded here.
pub(crate) const ROCK_BLOCK_WEIBULL_MAX: f32 = 2.0;

/// The fractional part, as WGSL's `fract` (`common.wesl`).
///
/// **Bevy's `Vec3::fract` is `x - x.trunc()` and so is NEGATIVE for a negative coordinate,
/// while WGSL's is `x - floor(x)` and is always in [0, 1).** The two disagree wherever the
/// coordinate is negative, and the cell's own boundary is computed from the fraction, so the
/// mirror and the shader disagreed about where every cell's wall was. The joint's weight came
/// out at zero on the negative half of the world.
fn wesl_fract(v: Vec3) -> Vec3 {
    v - v.floor()
}

/// The block network's coordinates for a world point, squashed along the bedding plane's normal
/// so the cells stretch into slabs (`common.wesl`).
///
/// The coordinates are **warped** before the lookup, by the crate's own 3D noise's *gradient*,
/// so the cells' boundaries wander: a cell network's cells are convex polygons of nearly the
/// same area, and a face of them reads as a mosaic — or, with the blocks' step, as *shingles*
/// (`docs/research/rock-fragments.md`).
pub(crate) fn rock_block_coords(p: Vec3) -> Vec3 {
    let normal = Vec3::new(0.32, 0.90, 0.28).normalize();
    let along = p - normal * p.dot(normal) * (1.0 - 1.0 / ROCK_BLOCK_TABULAR);
    let warp = crate::gpu::noise_mirror::perlin3_d(p / (ROCK_BLOCK_M * ROCK_BLOCK_WARP_CELLS));
    let gradient = Vec3::new(warp.y, warp.z, warp.w);
    (along + gradient * ROCK_BLOCK_WARP_M) / ROCK_BLOCK_M
}

/// The rock's **palette**: the colours a cliff is made of (`docs/research/rock-palette.md`).
///
/// The tones are multipliers on the rock's own albedo, and their **equal-weight mean is
/// exactly (1, 1, 1)**, so the rock's mean colour and mean brightness cannot move.
pub(crate) const ROCK_TONES: [Vec3; 4] = [
    Vec3::new(0.848, 1.015, 1.309),
    Vec3::new(1.223, 1.083, 0.832),
    Vec3::new(1.214, 1.270, 1.256),
    Vec3::new(0.715, 0.632, 0.603),
];
/// The width of a split in the palette, as a fraction of the draw's range (`common.wesl`).
pub(crate) const ROCK_PALETTE_EDGE: f32 = 0.12;
/// How far the region's field biases the bed's selection (`common.wesl`).
pub(crate) const BED_REGION_BIAS: f32 = 0.30;
/// How far the palette is mixed in, 0 to 1 (`common.wesl`).
pub(crate) const ROCK_PALETTE_STRENGTH: f32 = 0.90;

/// The rock's own **colour** at a world point: the palette's mixture for the cell the point
/// falls in, and its own brightness.
///
/// The **selection** is the cell's own factor and the **distribution** the cell's second
/// draw, so a fragment is a *piece of one rock* and the colour varies from fragment to
/// fragment — a cliff is not a set of coloured regions.
pub(crate) fn rock_palette(p: Vec3) -> Vec3 {
    let q = rock_block_coords(p);
    let cell = crate::gpu::noise_mirror::cell3(q);
    let sel_w = smoothstep(
        0.5 - ROCK_PALETTE_EDGE,
        0.5 + ROCK_PALETTE_EDGE,
        cell.factor,
    );
    let reg_w = smoothstep(
        0.5 - ROCK_PALETTE_EDGE,
        0.5 + ROCK_PALETTE_EDGE,
        cell.factor2,
    );
    let shares = [
        (1.0 - sel_w) * (1.0 - reg_w),
        (1.0 - sel_w) * reg_w,
        sel_w * (1.0 - reg_w),
        sel_w * reg_w,
    ];
    let total: f32 = shares.iter().sum();
    let mut mixed = Vec3::ZERO;
    for (share, tone) in shares.iter().zip(ROCK_TONES.iter()) {
        mixed += *tone * (*share / total);
    }
    // The palette is mixed in at its own strength (`common.wesl`), so the mirror's colour is
    // the shader's.
    Vec3::ONE.lerp(mixed, ROCK_PALETTE_STRENGTH)
}

/// The rock's **palette** at the *bed's* scale (`docs/research/bed-colour.md`).
///
/// The same four rocks as the fragments' palette, and the same normalisation: the
/// equal-weight mean is exactly (1, 1, 1), so the rock's mean colour cannot drift.
pub(crate) const BED_TONES: [Vec3; 4] = ROCK_TONES;
/// The width of a split in the palette, as a fraction of the draw's range (`common.wesl`).
pub(crate) const BED_PALETTE_EDGE: f32 = ROCK_PALETTE_EDGE;
/// How far the palette is mixed in, 0 to 1 (`common.wesl`).
pub(crate) const BED_PALETTE_STRENGTH: f32 = ROCK_PALETTE_STRENGTH;

/// A bed's own **colour** at a bed index: the palette's mixture for that index, and its own
/// brightness.
///
/// The **selection** is the bed's index's own hash and the **distribution** a second,
/// independent hash of the same index. The **region** is a slow field that biases both, so a
/// whole massif leans towards one rock and the massifs separate into warm and cool.
pub(crate) fn bed_palette(index: f32, region: f32) -> Vec3 {
    let i = index.floor() as i32;
    let h = crate::gpu::noise_mirror::hash31(IVec3::new(i, 7, 0));
    let h2 = crate::gpu::noise_mirror::hash31(IVec3::new(i, 11, 0));
    let sel = smoothstep(
        0.5 - BED_PALETTE_EDGE,
        0.5 + BED_PALETTE_EDGE,
        (h + region * BED_REGION_BIAS).clamp(0.0, 1.0),
    );
    let reg = smoothstep(
        0.5 - BED_PALETTE_EDGE,
        0.5 + BED_PALETTE_EDGE,
        (h2 + region * BED_REGION_BIAS).clamp(0.0, 1.0),
    );
    let shares = [
        (1.0 - sel) * (1.0 - reg),
        (1.0 - sel) * reg,
        sel * (1.0 - reg),
        sel * reg,
    ];
    let total: f32 = shares.iter().sum();
    let mut mixed = Vec3::ZERO;
    for (share, tone) in shares.iter().zip(BED_TONES.iter()) {
        mixed += *tone * (*share / total);
    }
    Vec3::ONE.lerp(mixed, BED_PALETTE_STRENGTH)
}

/// The **second joint set's** spacing at a world point, m: a *formation's* own draw.
///
/// A formation's cross-joint spacing is a property of the formation, so the spacing is
/// **constant within** it and changes from one formation to the next (`docs/research/
/// block-sizes.md`).
pub(crate) fn rock_joint2_spacing(p: Vec3) -> f32 {
    let formation = (p / ROCK_FORMATION_M).floor();
    let h = crate::gpu::noise_mirror::hash31(formation.as_ivec3());
    let weibull = (-(1.0 - h).max(1e-6).ln())
        .powf(1.0 / ROCK_BLOCK_WEIBULL)
        .min(ROCK_BLOCK_WEIBULL_MAX);
    ROCK_JOINT2_MIN_M + (ROCK_JOINT2_MAX_M - ROCK_JOINT2_MIN_M) * weibull / ROCK_BLOCK_WEIBULL_MAX
}

/// The **region's** value at a world point: the slow field that biases the bed's selection, so
/// a whole massif leans towards one rock.
///
/// The field is only *evaluated* where the bed's colour is wanted — the fragments carry the
/// near field and the beds carry the far one — and the gate skips the **evaluation** and not
/// the field's *value*. A gate that scaled the value would weaken the bias where the keep is
/// between zero and one, and the face would change: a performance change must not change the
/// image (`docs/research/rock-cost.md`).
pub(crate) fn rock_bed_region(p: Vec3, colour_keep: f32) -> f32 {
    if colour_keep > 0.0 {
        crate::gpu::noise_mirror::perlin3_d(p / ROCK_BED_REGION_M).x
    } else {
        0.0
    }
}

/// The joint's weight at a world point, 0 to 1: how close the point is to a block's boundary.
///
/// The boundary is the distance to the second nearest feature point, and the joint's own width
/// is the cell's draw, so the joints are of many widths.
pub(crate) fn rock_joint(p: Vec3) -> f32 {
    let q = rock_block_coords(p);
    let cell = crate::gpu::noise_mirror::cell3(q);
    // The distance from the point to its cell's wall, along the direction away from the
    // feature point (`common.wesl`).
    let to_centre = cell.point - wesl_fract(q);
    let len = to_centre.length().max(1e-6);
    let outward = to_centre / len;
    let inside = wesl_fract(q);
    // The wall in each axis: the far side of the cell from the feature point.
    let to_wall = Vec3::new(
        if outward.x < 0.0 { inside.x } else { 1.0 - inside.x },
        if outward.y < 0.0 { inside.y } else { 1.0 - inside.y },
        if outward.z < 0.0 { inside.z } else { 1.0 - inside.z },
    );
    let half = to_wall.x.min(to_wall.y).min(to_wall.z);
    // The joint's width is a **Weibull** draw, the shape the fracture spacing's distribution
    // follows (Verdian et al. 2021).
    let weibull = (-(1.0 - cell.factor).max(1e-6).ln())
        .powf(1.0 / ROCK_BLOCK_WEIBULL)
        .min(ROCK_BLOCK_WEIBULL_MAX);
    let width = ROCK_JOINT_WIDTH * (0.25 + ROCK_BLOCK_WEIBULL_SPREAD * weibull);
    let joint = 1.0 - smoothstep(0.0, width, half);
    // The **second joint set**: a family of parallel planes across the bedding, whose spacing
    // is a *formation's* own draw (`docs/research/block-sizes.md`).
    let spacing = rock_joint2_spacing(p);
    let cross = p.dot(ROCK_JOINT2_NORMAL.normalize());
    let wander = crate::gpu::noise_mirror::perlin3_d(p / (ROCK_BLOCK_M * ROCK_BLOCK_WARP_CELLS)).x
        * spacing
        * ROCK_JOINT2_WANDER;
    let cross_w = cross + wander;
    let phase = cross_w / spacing;
    let to_plane = (phase - phase.round()).abs() * spacing;
    // The planes are **non-persistent**: they end, and a finer field breaks them into
    // segments (Strauhal & Zangerl 2021).
    let segment = crate::gpu::noise_mirror::perlin3_d(p / (spacing * ROCK_JOINT2_SEGMENT_M)).x;
    let present2 = smoothstep(-ROCK_JOINT2_BREAK, ROCK_JOINT2_BREAK, segment);
    let joint2 = (1.0 - smoothstep(0.0, spacing * ROCK_JOINT2_WIDTH, to_plane)) * present2;
    joint.max(joint2)
}

// The beds (`docs/research/bedding.md`)
// ---------------------------------------------------------------------------------------------

/// The nominal spacing between beds, m (`common.wesl`).
pub(crate) const ROCK_BED_M: f32 = 4.5;
/// The bed lattice's warp, in bed units (`common.wesl`).
pub(crate) const BED_WARP_AMP: f32 = 1.0;

/// The bed's structure at the coordinate `bed`: which bed it is, where its bottom is and
/// where its top is (`common.wesl`).
///
/// The bed whose bottom is at or below the coordinate: the lattice is uniform, so the
/// candidate is the coordinate's own floor, and the walk goes down while the bed above still
/// covers the coordinate.
/// The bed lattice's warp, in bed units: how far a boundary wanders from the uniform
/// lattice (`common.wesl`).
pub(crate) fn bed_warp(index: f32) -> f32 {
    let cell = IVec3::new(index as i32, 3, 0);
    let h = crate::gpu::noise_mirror::hash31(cell);
    BED_WARP_AMP
        * (0.60 * (index * 0.21).sin()
            + 0.28 * (index * 0.53 + 1.3).sin()
            + 0.12 * (index * 1.17 + 2.6).sin())
        + 0.10 * (h - 0.5)
}

/// The boundary between the bed whose index is `i` and the next one, in bed units.
pub(crate) fn bed_boundary(index: f32) -> f32 {
    index + bed_warp(index)
}

pub(crate) fn bed_structure(bed: f32) -> (f32, f32, f32) {
    // The bed whose span covers the coordinate. The lattice's boundaries are at
    // `index * mean + warp(index)`, and the bed is the one whose boundary is at or below the
    // coordinate.
    //
    // The estimate is the coordinate's own position in the lattice, and the warp is a bounded
    // displacement, so a couple of steps of the fixed point
    // `i = (bed - warp(i)) / mean` bring it to the right bed. The iteration must be *stable*:
    // taking the floor at the end of it would flip between two beds at a boundary, so the
    // candidate is taken as the *largest* bed whose bottom is at or below the coordinate.
    let mut index = bed;
    for _ in 0..4 {
        index = bed - bed_warp(index);
    }
    // The candidate bed, and the one below it: the coordinate is in one of them, and the one
    // whose bottom is at or below the coordinate and whose top is above it is the bed.
    let mut bed_index = index.floor();
    let mut bottom = bed_boundary(bed_index);
    if bottom > bed {
        // The estimate overshot into the bed above: step down.
        bed_index -= 1.0;
        bottom = bed_boundary(bed_index);
    }
    let top = bed_boundary(bed_index + 1.0);
    (bed_index, bottom, top)
}

// The scree (`docs/research/scree.md`)
// ---------------------------------------------------------------------------------------------

/// The cell network's cell, m (`common.wesl`): the *spacing* of the stones in it.
pub(crate) const SCREE_CELL_M: f32 = 0.45;
/// The spread of the stones' sizes, as the range of the per-cell factor (`common.wesl`).
pub(crate) const SCREE_SIZE_MIN: f32 = 0.35;
pub(crate) const SCREE_SIZE_MAX: f32 = 2.20;
/// How much of the cell the largest stone fills (`common.wesl`).
pub(crate) const SCREE_FILL: f32 = 0.72;
/// The fine network's cell, as a fraction of the coarse one (`common.wesl`).
pub(crate) const SCREE_FINE_SCALE: f32 = 0.38;

/// The scree's band: the slope angles between which the stones lie (`render.wesl`). The scree
/// is at the rock's *foot* — from the soil's own band up to the rock's ramp (which opens at
/// 30°) — and the band is *wide*, because a scree is an apron and a narrow band would trace
/// the hillside's break as a contour line.
pub(crate) const SCREE_BAND_DEG: [f32; 4] = [16.0, 26.0, 31.0, 40.0];

/// The scree's weight at a slope angle: 1 at the rock's foot, 0 on the meadow and 0 on the
/// cliff (`render.wesl`).
pub(crate) fn scree_weight(slope_deg: f32) -> f32 {
    smoothstep(SCREE_BAND_DEG[0], SCREE_BAND_DEG[1], slope_deg)
        * (1.0 - smoothstep(SCREE_BAND_DEG[2], SCREE_BAND_DEG[3], slope_deg))
}

/// One scale's stones: how much of the point the stone covers (`common.wesl`).
pub(crate) fn scree_scale_coverage(xz: Vec2, cell_m: f32, size_min: f32, size_max: f32) -> f32 {
    let cell = crate::gpu::noise_mirror::cell2(xz / cell_m);
    let size = size_min + (size_max - size_min) * cell.factor.powi(3);
    let radius = 0.5 * size * SCREE_FILL;
    1.0 - smoothstep(radius * 0.80, radius, cell.distance)
}

/// The scree's coverage at a point: the coarse network's stone, and the fine network's stone
/// where the coarse one does not cover (`common.wesl`).
pub(crate) fn scree_coverage(xz: Vec2) -> f32 {
    let coarse = scree_scale_coverage(xz, SCREE_CELL_M, SCREE_SIZE_MIN, SCREE_SIZE_MAX);
    let fine = scree_scale_coverage(
        xz,
        SCREE_CELL_M * SCREE_FINE_SCALE,
        SCREE_SIZE_MIN * 0.8,
        SCREE_SIZE_MAX * 0.8,
    );
    coarse.max(fine * (1.0 - coarse))
}

/// `textures.wesl`'s hex grid scale (Mikkelsen's `2√3`).
pub(crate) const HEX_SCALE: f32 = 3.464_101_6;
/// `textures.wesl`'s sampling gate, height-blend sharpness and its band-limit (m).
pub(crate) const DETAIL_GATE: f32 = 0.01;
pub(crate) const HEIGHT_BLEND: f32 = 8.0;
pub(crate) const HEIGHT_BLEND_RESOLVED_M: f32 = 0.01;
pub(crate) const HEIGHT_BLEND_LOST_M: f32 = 0.03;

/// `hex_grid`: the three hex-grid vertices around `st` (texture tiles) and their weights.
pub(crate) fn hex_grid(st: Vec2) -> ([IVec2; 3], Vec3) {
    let s = st * HEX_SCALE;
    let skewed = Vec2::new(s.x - 0.577_350_27 * s.y, 1.154_700_5 * s.y);
    let base = skewed.floor();
    let f = skewed - base;
    let base = base.as_ivec2();
    let z = 1.0 - f.x - f.y;
    let up = if z <= 0.0 { 1.0 } else { 0.0 };
    let flip = 2.0 * up - 1.0;
    let u = up as i32;
    let w = Vec3::new(-z * flip, up - f.y * flip, up - f.x * flip);
    let vertices = [
        base + IVec2::new(u, u),
        base + IVec2::new(u, 1 - u),
        base + IVec2::new(1 - u, u),
    ];
    (vertices, w)
}

/// `hex_centre`: a hex vertex's centre in texture tiles.
pub(crate) fn hex_centre(v: IVec2) -> Vec2 {
    let p = v.as_vec2();
    Vec2::new(p.x + 0.5 * p.y, 0.866_025_4 * p.y) / HEX_SCALE
}

/// `layer_maps`' hex blend: Mikkelsen's contrast on the taps' heights and the fetch cut-off.
pub(crate) const HEX_CONTRAST: f32 = 0.6;
pub(crate) const TAP_MIN: f32 = 0.00125;

/// The hexes' `w⁷` shares, normalised.
fn hex_shares(bary: Vec3) -> Vec3 {
    let w2 = bary * bary;
    let b7 = w2 * w2 * w2 * bary;
    b7 / (b7.x + b7.y + b7.z)
}

/// `layer_maps`' tap weights for the barycentric weights `bary` and the taps' heights, with
/// every tap fetched (`all`) or only the live ones.
pub(crate) fn hex_tap_weights(bary: Vec3, heights: Vec3, all: bool) -> Vec3 {
    let shares = hex_shares(bary);
    let live = if all { Vec3::ONE } else { Vec3::select(shares.cmpgt(Vec3::splat(TAP_MIN)), Vec3::ONE, Vec3::ZERO) };
    let weighted = shares * (Vec3::ONE + (2.0 * heights - Vec3::ONE) * HEX_CONTRAST) * live;
    weighted / (weighted.x + weighted.y + weighted.z).max(1e-12)
}

/// `hex_tap`: the hex's coordinate in the texture for ground layer `layer`, and its orientation
/// (one of the eight: an optional swap of the axes and a sign on each).
pub(crate) fn hex_tap(v: IVec2, layer: u32, st: Vec2) -> (Vec2, Mat2) {
    let h = hash33(IVec3::new(v.x, v.y, layer as i32));
    let k = (h.z * 8.0) as u32;
    let swap = k & 1 != 0;
    let signs = Vec2::new(
        if k & 2 != 0 { -1.0 } else { 1.0 },
        if k & 4 != 0 { -1.0 } else { 1.0 },
    );
    let (c0, c1) = if swap { (Vec2::Y, Vec2::X) } else { (Vec2::X, Vec2::Y) };
    let rot = Mat2::from_cols(c0 * signs, c1 * signs);
    let centre = hex_centre(v);
    (rot * (st - centre) + centre + Vec2::new(h.x, h.y), rot)
}

/// `tap_slope`'s turn: a slope read in a tap's texture, back in the world.
pub(crate) fn tap_slope(texture_slope: Vec2, rot: Mat2) -> Vec2 {
    rot.transpose() * texture_slope
}

/// `height_sharpness`: the height blend's sharpness at a pixel footprint, m.
pub(crate) fn height_sharpness(footprint_m: f32) -> f32 {
    HEIGHT_BLEND * (1.0 - smoothstep(HEIGHT_BLEND_RESOLVED_M, HEIGHT_BLEND_LOST_M, footprint_m))
}

/// The ground layers' shares after the height blend, as `ground_textured` assembles them from
/// each layer's `height_lift`: the lifts scaled to the ground's own share (`w`: sand, soil, grass).
pub(crate) fn height_blend(w: Vec3, h: Vec3, sharpness: f32) -> Vec3 {
    let total = w.x + w.y + w.z;
    let lift = |h: f32| (sharpness * (h - 0.5)).exp();
    let e = (w - Vec3::splat(DETAIL_GATE)).max(Vec3::ZERO) * Vec3::new(lift(h.x), lift(h.y), lift(h.z));
    let sum = e.x + e.y + e.z;
    if sum <= 0.0 {
        return w;
    }
    e * (total / sum)
}

/// `ground_textured`'s grain share: the share of the procedural grain's albedo the ground keeps,
/// for the ground layers' shares `b`, their tile sizes (zero: untextured) and the set's `grain`.
pub(crate) fn grain_share(b: Vec3, tile: Vec3, grain: f32) -> f32 {
    let kept = Vec3::select(tile.cmpgt(Vec3::ZERO), Vec3::splat(grain), Vec3::ONE);
    let total = b.x + b.y + b.z;
    if total > 1e-6 { b.dot(kept) / total } else { 1.0 }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The beds are **bands and not blotches**: a vertical cut across a cliff crosses a bed
    /// every few metres, and the crossings are *bounded* — a smear has no bound.
    ///
    /// The measurement that started this: the old warp displaced the bed coordinate by up to
    /// 0.78 of a bed, and a 4.5 m grid distorted by a 45 m-scale field is a set of blotches.
    #[test]
    fn the_beds_are_bands_and_not_blotches() {
        // A vertical cut through the rock, 40 m of it, in the bed's own coordinate.
        let mut crossings = 0;
        let mut spacings = Vec::new();
        let mut last = None;
        let mut last_y = 0.0_f32;
        for i in 0..1000 {
            let y = i as f32 * 0.04;
            // The coordinate in bed units: the height along the bed plane, in the nominal
            // spacing, which is the lattice's spacing.
            let bed = (y * 0.90) / ROCK_BED_M;
            // The bed's *index*: the cut crosses a bed's edge where the index changes.
            let (index, _, _) = bed_structure(bed);
            if let Some(was) = last
                && was != index
            {
                crossings += 1;
                // The spacing is the distance between two crossings. The first is measured
                // from the cut's start, which is in the middle of a bed, so it is dropped.
                if crossings > 1 {
                    spacings.push(y - last_y);
                }
                last_y = y;
            }
            last = Some(index);
        }
        assert!(
            crossings >= 4,
            "the beds are bands: {crossings} crossings in 40 m"
        );
        // The spacings are *bounded*: a bed's thickness is the gap between two boundaries, and
        // the warp is bounded, so the distance between two edges is bounded too. A smear — the
        // old warp — has no bound: it moves the coordinate by most of a bed, and the edges
        // land anywhere.
        let longest = spacings.iter().copied().fold(0.0_f32, f32::max);
        let shortest = spacings.iter().copied().fold(f32::INFINITY, f32::min);
        assert!(
            longest < 3.0 * ROCK_BED_M,
            "the beds' spacing is bounded: the longest is {longest} m"
        );
        assert!(
            shortest > 0.2 * ROCK_BED_M,
            "the beds' spacing is bounded: the shortest is {shortest} m"
        );
    }

    /// The beds **stack**: their thicknesses are a draw, not a constant, and the draw is
    /// heavy-tailed, so a few beds are much thicker than the rest.
    ///
    /// Talling 2001: a bed's thickness follows a log-normal distribution, and a succession's
    /// thicknesses are "the sum of a series of log-normal frequency distributions".
    #[test]
    fn the_beds_thicknesses_vary_and_stack() {
        // The thicknesses are the *gaps* between the lattice's boundaries, and they vary
        // because the warp is a smooth field with a bounded derivative.
        let mut thicknesses = Vec::new();
        for i in 0..2000 {
            thicknesses.push(bed_boundary(i as f32 + 1.0) - bed_boundary(i as f32));
        }
        thicknesses.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let p10 = thicknesses[thicknesses.len() / 10];
        let p90 = thicknesses[thicknesses.len() * 9 / 10];
        // The thicknesses span at least 1.25:1 between the tenth and the ninetieth
        // percentile, and none is degenerate.
        assert!(
            p90 > p10 * 1.25,
            "the beds' thicknesses span only {:.2}:1", p90 / p10
        );
        assert!(p10 > 0.35, "the beds' thicknesses are bounded: {p10}");
        // The beds *stack*: the boundaries are increasing, so every bed is above the one below
        // it and the succession is a sequence and not a set of overlapping bands.
        let mut increasing = 0;
        for i in 0..2000 {
            if bed_boundary(i as f32 + 1.0) > bed_boundary(i as f32) {
                increasing += 1;
            }
        }
        assert_eq!(increasing, 2000, "the beds stack");
    }

    /// The rock's blocks are **discrete**: a face's joint weight crosses the half-way mark
    /// many times along a cut, where a smooth wash crosses it once.
    ///
    /// This is the note's second validation target, and it is the difference between a *face
    /// made of pieces* and a *texture*: the blocks have to be countable.
    #[test]
    fn the_blocks_are_discrete() {
        // A cut across a face, 20 cm apart, 6 m long.
        let mut joint = Vec::new();
        for i in 0..150 {
            let p = Vec3::new(i as f32 * 0.2, 12.0, 30.0);
            joint.push(rock_joint(p));
        }
        let mut crossings = 0;
        for w in joint.windows(2) {
            if (w[0] > 0.5) != (w[1] > 0.5) {
                crossings += 1;
            }
        }
        assert!(
            crossings >= 3,
            "the blocks are discrete, not a wash: {crossings} crossings in 30 m"
        );
        // And the joints are *lines*, not a field: the joint is near zero over most of the cut
        // (it is a line a fraction of a cell wide) and one over the rest.
        let low = joint.iter().filter(|j| **j < 0.15).count();
        assert!(
            low * 2 > joint.len(),
            "the joint is a line, not a field: {low} of {} below 0.15",
            joint.len()
        );
    }

    /// The blocks are of **many sizes**, and not all the same.
    ///
    /// The measure: the *distance between the joints* along a cut, which is the block's own
    /// size. A face of equal cells is a mosaic — the quilt the capture showed — and the
    /// geology says a cliff's blocks are not equal: the fracture spacing follows a Weibull
    /// whose geometric standard deviation is about 0.31 (Verdian et al. 2021).
    #[test]
    fn the_blocks_are_of_many_sizes() {
        // A cut across a face, and the block's boundary along it: the joint's weight crosses
        // the half-way mark where a block ends and the next begins.
        let mut boundaries = Vec::new();
        let mut last = rock_joint(Vec3::new(0.0, 20.0, 0.0)) > 0.5;
        for i in 1..4000 {
            let x = i as f32 * 0.02;
            let now = rock_joint(Vec3::new(x, 20.0, 0.0)) > 0.5;
            if now != last {
                boundaries.push(x);
                last = now;
            }
        }
        assert!(
            boundaries.len() >= 8,
            "the cut crosses enough blocks to measure: {}",
            boundaries.len()
        );
        let sizes: Vec<f32> = boundaries
            .windows(2)
            .map(|w| w[1] - w[0])
            .filter(|s| *s > 0.05)
            .collect();
        assert!(sizes.len() >= 6, "the sizes to measure: {}", sizes.len());
        // The spread: the largest block against the smallest, measured on the *median* and
        // the tails so a single outlier does not carry it.
        let mut sorted = sizes.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let p10 = sorted[sorted.len() / 10];
        let p90 = sorted[sorted.len() * 9 / 10];
        let ratio = p90 / p10.max(1e-6);
        assert!(
            ratio >= 2.0,
            "the blocks are of many sizes: the p90/p10 ratio is {ratio:.2} over {} blocks \
             ({p10:.2} m to {p90:.2} m)",
            sizes.len()
        );
    }

    /// The **second joint set's spacing** varies, and that is what makes the pieces many-sized.
    ///
    /// A block is the intersection of *two* families of planes, so where the second set's
    /// planes are close the pieces are chips and where they are far the pieces are slabs
    /// (`docs/research/block-sizes.md`). The spacing is a *formation's* own draw, so the spread
    /// is *between* formations, and the formations are 46 m apart.
    ///
    /// The measure: the spacing over a run of formations. The spacing's own distribution is a
    /// Weibull (Verdian et al. 2021: the Weibull form "yielded the best fit to the data in
    /// most cases"), so the spread is wide.
    #[test]
    fn the_second_joint_sets_spacing_varies() {
        let mut spacings = Vec::new();
        // The formations are 46 m apart, so the cut crosses many of them.
        for i in 0..400 {
            let p = Vec3::new(i as f32 * 3.1, 18.0 + (i as f32) * 0.91, 0.0);
            spacings.push(rock_joint2_spacing(p));
        }
        let mut sorted = spacings.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let p10 = sorted[sorted.len() / 10];
        let p90 = sorted[sorted.len() * 9 / 10];
        let ratio = p90 / p10.max(1e-6);
        assert!(
            ratio >= 2.5,
            "the second set's spacing varies: the p90/p10 ratio is {ratio:.2} over {} \
             formations ({p10:.2} m to {p90:.2} m)",
            spacings.len()
        );
    }

    /// A **gate** skips the field's *evaluation* and not its *value*, so a performance change
    /// does not change the image.
    ///
    /// The rule: where the gate is *open* the field is the field's own, unmodified — and where
    /// it is *closed* the field is not fetched at all. The first attempt scaled the field by
    /// the keep, and that changed 13 % of the near face, because the keep is between zero and
    /// one where the two band-limits cross and scaling weakens the bias there.
    #[test]
    fn a_gate_skips_the_evaluation_and_not_the_value() {
        // Where the gate is open, the gated field and the ungated one agree exactly.
        let mut same = 0;
        let mut samples = 0;
        for i in 0..200 {
            let p = Vec3::new(i as f32 * 0.37, 18.0 + (i as f32) * 0.11, 0.0);
            let gated = rock_bed_region(p, 1.0);
            let ungated = crate::gpu::noise_mirror::perlin3_d(p / ROCK_BED_REGION_M).x;
            if (gated - ungated).abs() < 1e-6 {
                same += 1;
            }
            samples += 1;
        }
        assert_eq!(same, samples, "the gate does not change the field's value");
        // And where the gate is closed, the field is not fetched: the mirror returns the
        // neutral value, and the shader does not evaluate the noise at all.
        let closed = rock_bed_region(Vec3::new(1.0, 18.0, 0.0), 0.0);
        assert_eq!(closed, 0.0, "the closed gate is the neutral value");
    }

    /// The rock's **palette** does not move the rock's mean colour.
    ///
    /// The rule is the crate's own, from the ground's palette: "the ground is a place and not
    /// a paint colour, while the *mean* colour and the *mean* brightness stay where they were
    /// (a ground that changes colour changes the game's lighting)". A convex combination of
    /// tones that bracket the base cannot drift the mean, and the tones are normalised so
    /// that their equal-weight mean is exactly (1, 1, 1).
    #[test]
    fn the_palette_does_not_move_the_mean() {
        let mut mean = Vec3::ZERO;
        for tone in ROCK_TONES.iter() {
            mean += *tone;
        }
        mean /= 4.0;
        assert!(
            (mean - Vec3::ONE).length() < 1e-3,
            "the palette's equal-weight mean is {mean:?}, not (1, 1, 1)"
        );
    }

    /// The fragments are of **different rock**, and not one rock in many shades.
    ///
    /// The measure: the *hue* of the fragments over a patch of the face. A cliff's fragments
    /// are different minerals, so the palette's hues must differ; and the difference must be
    /// at the *fragment's* scale, so a patch of the face sees several of them.
    #[test]
    fn the_fragments_are_of_different_rock() {
        // The palette's own hues: the four tones must not be the same colour.
        let hues: Vec<f32> = ROCK_TONES
            .iter()
            .map(|t| {
                let max = t.x.max(t.y).max(t.z);
                let min = t.x.min(t.y).min(t.z);
                let d = max - min;
                if d < 1e-6 { return 0.0; }
                let h = if max == t.x {
                    ((t.y - t.z) / d).rem_euclid(6.0)
                } else if max == t.y {
                    (t.z - t.x) / d + 2.0
                } else {
                    (t.x - t.y) / d + 4.0
                };
                h * 60.0
            })
            .collect();
        let mut sorted = hues.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let spread = sorted[3] - sorted[0];
        assert!(
            spread > 60.0,
            "the palette's hues are different colours: the spread is {spread:.1} degrees"
        );
        // And the fragments *use* them: **neighbouring** fragments must differ, or the cliff
        // is one rock in many shades. The measure is the difference between the colours of
        // two points a cell apart — the cell is the fragment, so two points a cell apart are
        // two fragments.
        let mut differ = 0;
        let mut samples = 0;
        for i in 0..200 {
            let p = Vec3::new(i as f32 * 0.37, 20.0 + (i as f32) * 0.11, 0.0);
            let a = rock_palette(p);
            let b = rock_palette(p + Vec3::new(ROCK_BLOCK_M * 0.9, 0.0, 0.0));
            if (a - b).length() > 0.10 {
                differ += 1;
            }
            samples += 1;
        }
        assert!(
            differ * 100 / samples > 60,
            "the fragments are of different rock: {differ} of {samples} neighbours differ"
        );
    }

    /// The **beds** carry a palette too, and the beds of a succession are of *different rock*.
    ///
    /// The measure: the hue of the beds' colours over a run of bed indices. A succession's beds
    /// are a sandstone, a shale, a limestone — different minerals, different colours — and the
    /// crate's bed's tone was a *scalar*, so a distant massif was one grey.
    #[test]
    fn the_beds_are_of_different_rock() {
        // The beds' own hues over a run of indices, and the *region*'s effect on them.
        let mut hues = Vec::new();
        for i in 0..200 {
            let c = bed_palette(i as f32, 0.0);
            let max = c.x.max(c.y).max(c.z);
            let min = c.x.min(c.y).min(c.z);
            let d = max - min;
            if d < 1e-6 {
                continue;
            }
            let h = if max == c.x {
                ((c.y - c.z) / d).rem_euclid(6.0)
            } else if max == c.y {
                (c.z - c.x) / d + 2.0
            } else {
                (c.x - c.y) / d + 4.0
            };
            hues.push(h * 60.0);
        }
        assert!(hues.len() > 150, "the hues to measure: {}", hues.len());
        let mut sorted = hues.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let spread = sorted[sorted.len() - 1] - sorted[0];
        assert!(
            spread > 60.0,
            "the beds are of different rock: the hue's spread is {spread:.1} degrees"
        );
    }

    /// The **region** shifts the beds' colour, so a whole massif is one rock and not a set of
    /// stripes.
    ///
    /// The measure: the *same* bed's colour under two different regions. A range of mountains
    /// is made of *formations*, and a formation is a massif — so the field is at the massif's
    /// scale and the bed's own draw only tints it.
    #[test]
    fn the_region_shifts_the_beds_colour() {
        let mut differ = 0;
        let mut samples = 0;
        for i in 0..200 {
            let a = bed_palette(i as f32, -0.5);
            let b = bed_palette(i as f32, 0.5);
            if (a - b).length() > 0.02 {
                differ += 1;
            }
            samples += 1;
        }
        assert!(
            differ * 100 / samples > 50,
            "the region shifts the beds' colour: {differ} of {samples} beds differ"
        );
    }

    /// The blocks are **tabular**: they stretch along the bedding, so a face shows slabs and
    /// not gravel.
    ///
    /// The measure is the *cell's* own extent in the rock's frame: the network's coordinates
    /// are stretched along the bedding plane's normal, so the cells are longer in the
    /// bedding's plane than across it. The measure is the cell and not the joint's crossings,
    /// because the crossings are the *warp's* bending as much as the lattice's, and the warp
    /// is a long-range field that does not know about the bedding.
    #[test]
    fn the_blocks_are_tabular() {
        let normal = Vec3::new(0.32, 0.90, 0.28).normalize();
        // A direction in the bedding plane, and the one across it.
        let in_plane = (Vec3::new(1.0, 0.0, 0.0) - normal * normal.x).normalize();
        let across = normal.cross(in_plane).normalize();
        // The cell's *own* extent along each direction: the distance between the cell
        // boundaries, measured in the network's coordinates.
        let extent = |dir: Vec3| {
            // The direction in the network's own frame, and the cell's size along it.
            let q_dir = rock_block_coords(dir * 100.0) - rock_block_coords(Vec3::ZERO);
            q_dir.length()
        };
        let along = extent(in_plane);
        let across_cut = extent(across);
        assert!(
            along > across_cut,
            "the blocks stretch along the bedding: the cell is {along:.3} along it and \
             {across_cut:.3} across it"
        );
        // And the squash is in the bedding's own frame, not the world's: the normal is
        // mostly vertical, so the blocks are flattened in the vertical and stretched in the
        // horizontal.
        assert!(
            normal.y > 0.8,
            "the bedding plane is a near-horizontal one: {normal:?}"
        );
    }

    /// The blocks are in the rock's **3D** frame, so they do not weave on a slope.
    ///
    /// The measure: the joint's pattern at two heights on the same vertical line must be
    /// *different* — a 2D network in the ground plane would give the same pattern at both,
    /// which is the weave the crate's rock note warns about.
    #[test]
    fn the_blocks_are_in_the_rocks_own_frame() {
        // The *cell* each point falls in, at two heights on the same vertical line. A 2D
        // network in the ground plane would give the same cell at both heights, which is the
        // weave the crate's rock note warns about; a 3D network gives a different one.
        //
        // The measure is the cell and not the joint: the joint is a *line* a fraction of a
        // cell wide, so two heights are usually both *off* a joint and trivially agree, and
        // the test would pass for a 2D network.
        let mut same = 0;
        for i in 0..200 {
            let x = i as f32 * 0.37;
            let a = rock_block_coords(Vec3::new(x, 10.0, 40.0)).floor();
            let b = rock_block_coords(Vec3::new(x, 14.0, 40.0)).floor();
            if a == b {
                same += 1;
            }
        }
        // A 3D network: the two heights are in different cells, for most of the points. A 2D
        // network in the ground plane would agree on every one.
        assert!(
            same < 100,
            "the blocks are in the rock's 3D frame: {same} of 200 are in the same cell \
             across 4 m of height"
        );
    }

    /// The scree is a *scatter*, not a band: its coverage must be **discrete** — the ground
    /// under the scree is stones and turf between them, not a uniform wash.
    ///
    /// The check is the research note's first validation target: the coverage along a cut
    /// across the scree must have *peaks* — a histogram with a second mode — and not a
    /// monotone gradient. A band's weight is a smooth ramp, so its coverage is smooth too.
    #[test]
    fn the_scree_is_discrete_and_not_a_wash() {
        // A cut across the scree, 40 cm apart, 4 m long.
        let mut coverage = Vec::new();
        for i in 0..200 {
            coverage.push(scree_coverage(Vec2::new(i as f32 * 0.04, 3.0)));
        }
        // The stones: how many times the coverage crosses the middle of its range on its way
        // up. A smooth wash crosses once or twice; a scatter of stones crosses many.
        let mut crossings = 0;
        for w in coverage.windows(2) {
            let a = w[0] > 0.5;
            let b = w[1] > 0.5;
            if a != b {
                crossings += 1;
            }
        }
        assert!(
            crossings >= 6,
            "the scree is a scatter, not a wash: {crossings} crossings in 8 m"
        );
        // And there is turf between the stones: the coverage is below one everywhere, and
        // below the middle over a good part of the cut.
        assert!(
            coverage.iter().all(|c| *c <= 1.0),
            "the coverage never exceeds one"
        );
        let below_half = coverage.iter().filter(|c| **c < 0.5).count();
        assert!(
            below_half * 2 > coverage.len(),
            "the turf shows between the stones: {below_half} of {} below half",
            coverage.len()
        );
    }

    /// The stones have *many* sizes: the note's second validation target, from Verdian et al.
    /// 2021 — a talus runs from 2 mm to 575 mm, and the median moves by more than 10-fold
    /// between sites.
    ///
    /// The measure is the *area* each stone takes: a scatter of one size repeated reads as a
    /// cobble pavement, and a scatter of many sizes reads as scree. The stones' *positions*
    /// are checked too, because a network whose points all landed in one corner of their cells
    /// would be a grid of stones, not a scatter.
    #[test]
    fn the_stones_have_many_sizes() {
        // The coarse network's stones, over a 20 m square, and the area each one covers.
        let mut areas = Vec::new();
        let mut offsets = Vec::new();
        for j in 0..200 {
            for i in 0..200 {
                let p = Vec2::new(i as f32 * 0.1, j as f32 * 0.1);
                let cell = crate::gpu::noise_mirror::cell2(p / SCREE_CELL_M);
                let size = SCREE_SIZE_MIN + (SCREE_SIZE_MAX - SCREE_SIZE_MIN) * cell.factor.powi(3);
                let radius = 0.5 * size * SCREE_FILL;
                areas.push(radius * radius);
                // The point's offset inside its own cell, in [0, 1).
                offsets.push(cell.point.fract());
            }
        }
        areas.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let p10 = areas[areas.len() / 10];
        let p90 = areas[areas.len() * 9 / 10];
        // The areas span at least 4:1 between the tenth and the ninetieth percentile.
        assert!(
            p90 > p10 * 4.0,
            "the stones' areas span only {:.1}:1", p90 / p10
        );
        // The stones are spread inside their cells: the mean offset is near the middle, and
        // the offsets are not all the same point.
        let mean = offsets.iter().fold(Vec2::ZERO, |a, b| a + *b) / offsets.len() as f32;
        assert!(
            mean.x > 0.35 && mean.x < 0.65 && mean.y > 0.35 && mean.y < 0.65,
            "the stones sit in the middle of their cells on average: {mean:?}"
        );
        let spread = offsets
            .iter()
            .fold(Vec2::ZERO, |a, b| a + (*b - mean).abs())
            / offsets.len() as f32;
        assert!(
            spread.x > 0.2 && spread.y > 0.2,
            "the stones are spread inside their cells: {spread:?}"
        );
    }

    /// The scree is at the rock's **foot**: on the meadow and on the cliff there are no
    /// stones, and the band's weight peaks where the soil's band dies and the rock's ramp
    /// begins (the crate's own layer weights: the soil is gone by 30° and the rock is full at
    /// 38°).
    #[test]
    fn the_scree_lies_at_the_rocks_foot() {
        assert_eq!(scree_weight(0.0), 0.0, "the meadow has no stones");
        assert_eq!(scree_weight(15.0), 0.0, "the meadow has no stones");
        assert_eq!(scree_weight(45.0), 0.0, "the cliff has no stones");
        // The band's weight is strongest just below the rock's ramp.
        let peak = (14..42)
            .map(|d| (d as f32, scree_weight(d as f32)))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(d, _)| d)
            .unwrap_or(0.0);
        assert!(
            (22.0..=33.0).contains(&peak),
            "the scree peaks at {peak} degrees, at the rock's foot"
        );
        // And it is on the ground the soil has left and the rock has not taken: at 28° the
        // crate's own layer weights are mostly grass and soil.
        let w = layer_weights(28.0, 0.0, 0.5);
        assert!(w[0] < 0.1, "the rock has not begun at 28 degrees: {:?}", w);
        // The band is an *apron*, not a contour line: it is at least six degrees wide at
        // half its strength, or it traces the hillside's break as a line.
        let wide = (14..42)
            .filter(|d| scree_weight(*d as f32) > 0.5)
            .count();
        assert!(wide >= 6, "the scree's band is {wide} degrees wide, too narrow");
    }

    /// The turf's palette: the four tones' shares must sum to one by construction (so the
    /// mixture is a convex combination and the ground's mean cannot drift), every share must
    /// be in range, and the mixture must stay inside the tones' convex hull — the fresh
    /// grass is the turf's own colour, so a place with no dry tone must be exactly the
    /// identity.
    ///
    /// Three other normalisations were tried and each failed: by the unweighted mean of the
    /// tones the ground's mean brightness drifted 3.2 DN, by the mixture's own colour the hue
    /// was cancelled entirely (the G/R ratio was 1.0 at every field), and by a per-channel
    /// mean the hue was cancelled again. The shares are the fourth way and the only one that
    /// keeps both the hue and the mean.
    #[test]
    fn the_turf_palette_shares_sum_to_one_and_keep_the_mean_and_the_hue() {
        let mut hues = Vec::new();
        for field in (-30..=30).map(|k| k as f32 / 20.0) {
            for clump in (-20..=20).map(|k| k as f32 / 20.0) {
                let (mixed, shares) = turf_palette(field, clump);
                assert!(
                    (shares.iter().sum::<f32>() - 1.0).abs() < 1e-5,
                    "the shares do not sum to one at {field}/{clump}: {shares:?}"
                );
                assert!(
                    shares.iter().all(|s| (0.0..=1.0).contains(s)),
                    "a share is out of range at {field}/{clump}: {shares:?}"
                );
                assert!(
                    mixed.iter().all(|c| (0.5..=2.0).contains(c)),
                    "the mixture left the tones' range at {field}/{clump}: {mixed:?}"
                );
                // The hue must survive: the G/R ratio has to *move* with the field, or the
                // palette is a brightness change and not a colour change.
                hues.push(mixed[1] / mixed[0]);
            }
        }
        let min = hues.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = hues.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        // The ratio's *maximum* is 1.0 — the fresh tone is the identity, so a place with no
        // dry tone is exactly the turf's own hue — and its minimum is the straw's.
        assert!(
            min < 0.95 && max <= 1.0 + 1e-5,
            "the palette's hue does not move: the G/R ratio spans {min} to {max}"
        );
        // A place with no dry tone is exactly the turf's own colour.
        let (mixed, _) = turf_palette(0.0, 0.0);
        for c in mixed {
            assert!((c - 1.0).abs() < 1e-5, "the fresh tone is not the identity: {mixed:?}");
        }
    }

    /// An octave is kept whole while the pixel resolves it, cancelled once it cannot, and the
    /// transition is monotone and smooth; the two scales overlap so no band of the ground's
    /// detail is left uncovered (`ground-detail.md` Validation 1-3).
    #[test]
    fn the_detail_band_limit_keeps_resolved_octaves_and_cancels_the_rest() {
        assert_eq!(detail_keep(DETAIL_COARSE_M, 0.0), 1.0);
        assert_eq!(detail_keep(DETAIL_COARSE_M, DETAIL_COARSE_M / 2.0), 1.0);
        assert_eq!(detail_keep(DETAIL_COARSE_M, DETAIL_COARSE_M), 0.0);
        assert_eq!(detail_keep(DETAIL_COARSE_M, 10.0), 0.0);
        // Monotone in the footprint, and continuous (no popping as the camera pulls back).
        let mut last = 1.0;
        for k in 0..=400 {
            let f = f64::from(k) / 400.0 * f64::from(DETAIL_COARSE_M);
            let w = detail_keep(DETAIL_COARSE_M, f as f32);
            assert!(w <= last + 1e-6, "not monotone at {f}");
            assert!((last - w) < 0.02, "a step of {} at {f}", last - w);
            last = w;
        }
        // The overlap: an octave held whole (half its feature) while the next coarser is gone.
        // The fine scale is 10× the coarse one's frequency, so there is always a band of
        // footprints between them where one has faded and the other has not yet.
        let half_fine = DETAIL_FINE_M * 0.5;
        let half_coarse = DETAIL_COARSE_M * 0.5;
        assert!(detail_keep(DETAIL_FINE_M, half_fine) > 0.99);
        assert_eq!(detail_keep(DETAIL_COARSE_M, half_coarse), 1.0);
        assert!(half_fine < half_coarse);
    }

    /// FNV-1a over the bytes of `values`.
    fn fnv1a(bytes: &[u8]) -> u64 {
        bytes.iter().fold(0xcbf2_9ce4_8422_2325, |h, b| {
            (h ^ u64::from(*b)).wrapping_mul(0x0000_0100_0000_01b3)
        })
    }

    /// The statistics of one preset at 1024² cells of 2 m: (spread of `h − h_base` between the
    /// 0.5 and 99.5 percentiles, slope p50, p95, share above 40° in percent).
    fn stats(p: &ProceduralTerrain) -> (f32, f32, f32, f32) {
        let size = UVec2::splat(1024);
        let surface = generate_surface(p, size, 2.0);
        let mut heights: Vec<f32> = surface.iter().map(|h| h - p.base_height_m).collect();
        let spread = percentile(&mut heights, 0.995) - percentile(&mut heights, 0.005);
        let mut slopes: Vec<f32> = (0..size.y)
            .flat_map(|j| (0..size.x).map(move |i| (i, j)))
            .map(|(i, j)| slope_deg(&surface, size, 2.0, i, j))
            .collect();
        let steep = slopes.iter().filter(|s| **s > 40.0).count() as f32 / slopes.len() as f32;
        (
            spread,
            percentile(&mut slopes, 0.5),
            percentile(&mut slopes, 0.95),
            steep * 100.0,
        )
    }

    /// Validation 1: the two noise sums on this crate's noise land on the constants the unit
    /// maps assume. The percentiles are the stationary ones (pooled over seeds): one seed's
    /// 2048 m tile of a 600 m feature has only 3.4 base-octave cells, so its own percentiles
    /// scatter by ±0.08 (seed 42: f −0.347 / 0.256).
    #[test]
    fn the_noise_sums_fill_their_unit_ranges() {
        let (mut f, mut r) = (Vec::new(), Vec::new());
        for seed in 0..32u64 {
            for j in 0..96 {
                for i in 0..96 {
                    let p = Vec2::new(i as f32, j as f32) * 0.31 + Vec2::splat(0.1);
                    let (a, b) = noise_sums(p, 1.0, 8, seed * 7 + 1);
                    f.push(a);
                    r.push(b);
                }
            }
        }
        let f_lo = percentile(&mut f, 0.005);
        let f_hi = percentile(&mut f, 0.995);
        let r_lo = percentile(&mut r, 0.005);
        let r_hi = percentile(&mut r, 0.995);
        println!("f {f_lo:.3} {f_hi:.3}  r {r_lo:.3} {r_hi:.3}");
        assert!(
            (f_lo + F_HALF).abs() < 0.03 && (f_hi - F_HALF).abs() < 0.03,
            "f {f_lo} {f_hi}"
        );
        assert!(
            (r_lo - R_LOW).abs() < 0.03 && (r_hi - (R_LOW + R_SPAN)).abs() < 0.03,
            "r {r_lo} {r_hi}"
        );
    }

    /// Validation 1: per preset, the percentile spread of the heights, the slope percentiles and
    /// the share of rock-steep ground, re-blessed once on this crate's noise (Deviation 11; the
    /// note's numpy port, a 2D gradient noise, is about 30 % steeper).
    #[test]
    fn the_presets_have_the_measured_statistics() {
        // (preset, spread m, slope p50, slope p95, share above 40° in %), seed 42, 2 m cells.
        let table = [
            (ProceduralTerrain::hills(42), 91.8, 12.4, 25.5, 0.0),
            (ProceduralTerrain::mountains(42), 549.5, 41.1, 61.9, 52.6),
            (ProceduralTerrain::dunes(42), 39.9, 9.1, 18.6, 0.0),
            (ProceduralTerrain::plains(42), 13.7, 1.4, 2.9, 0.0),
        ];
        let mut misses = Vec::new();
        for (preset, spread, p50, p95, steep) in table {
            let (s, m, h, e) = stats(&preset);
            println!(
                "R {} F {}: spread {s:.1} (want {spread}) slope p50 {m:.1} ({p50}) p95 {h:.1} ({p95}) >40° {e:.2} % ({steep})",
                preset.relief_m, preset.feature_size_m
            );
            if (s / spread - 1.0).abs() >= 0.10
                || (m / p50 - 1.0).abs() >= 0.15
                || (h / p95 - 1.0).abs() >= 0.15
                || (e - steep).abs() >= 3.0
            {
                misses.push(format!("R {} F {}", preset.relief_m, preset.feature_size_m));
            }
        }
        assert!(misses.is_empty(), "{misses:?}");
    }

    /// Validation 2: the generator is frozen on the CPU mirror.
    #[test]
    fn the_terrain_is_frozen() {
        let surface = generate_surface(&ProceduralTerrain::hills(42), UVec2::splat(256), 8.0);
        let mm: Vec<i32> = surface
            .iter()
            .map(|h| (h * 1000.0).round() as i32)
            .collect();
        let hash = fnv1a(bytemuck::cast_slice(&mm));
        println!("the_terrain_is_frozen: {hash:#018x}");
        assert_eq!(
            hash, GOLDEN_HILLS_256,
            "the generator changed: re-bless once, deliberately"
        );
    }

    const GOLDEN_HILLS_256: u64 = 0x6f04_892c_64bc_b5eb;

    /// Validation 7: the layer weights sum to one for any detail, pick the plan's layers without
    /// detail, and the detail never moves a layer that is fully on or off.
    #[test]
    fn the_layer_weights_sum_to_one_and_pick_the_right_layer() {
        for a in (0..90).step_by(1) {
            for s in [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0, 4.0] {
                for d in [0.0, 0.25, 0.5, 0.75, 1.0] {
                    let w = layer_weights(a as f32, s, d);
                    assert!(
                        (w.iter().sum::<f32>() - 1.0).abs() < 1e-6,
                        "{a}° {s} m detail {d}: {w:?}"
                    );
                    assert!(w.iter().all(|x| (0.0..=1.0).contains(x)));
                }
            }
        }
        assert!(layer_weights(45.0, 0.0, 0.5)[0] >= 0.99);
        assert!(layer_weights(20.0, 1.0, 0.5)[1] >= 0.99);
        assert_eq!(layer_weights(10.0, 0.0, 0.5)[3], 1.0);
        // The soil band is 20° to 30°, tapering from 26° as the rock's ramp opens, and it must
        // sit *below* the rock's ramp (30° to 38°): where the two overlap, a mix of a warm
        // brown and a cool grey is a pink patch, and the ground showed them on the hills
        // (measured: 0.32 % of the frame).
        let soil = layer_weights(21.0, 0.0, 0.5)[2];
        assert!((0.3..0.7).contains(&soil), "soil at 21°: {soil}");
        assert!(layer_weights(31.0, 0.0, 0.5)[2] < 0.4, "the soil must be gone by 31°");
        // The taper: the soil thins as the rock rises, so the grass meets the stone without a
        // warm brown rim (measured: 1415 orange pixels at RGB 148, 121, 100 in one frame, and
        // none after).
        assert!(layer_weights(28.0, 0.0, 0.5)[2] < 0.35, "the soil must taper into the rock");
        // Fully on or off stays put under any detail: rock at 50°, grass at 10°.
        for d in [0.0, 0.3, 0.7, 1.0] {
            assert_eq!(layer_weights(50.0, 0.0, d)[0], 1.0);
            assert_eq!(layer_weights(10.0, 0.0, d)[3], 1.0);
        }
        // Inside a transition the detail spans the whole range. The rock's ramp is 30° to 38°,
        // so 34° is its middle: there the detail takes the rock from 0 to 1.
        assert_eq!(layer_weights(34.0, 0.0, 0.0)[0], 0.0);
        assert_eq!(layer_weights(34.0, 0.0, 1.0)[0], 1.0);
        // The rock's ramp starts at 30°, so a slope is already half rock at 34°: the ground's
        // turf is the crate's 2D noise in the world XZ plane, and it must be gone before the
        // surface is steep enough to weave it (`rock-material.md` Model 1).
        assert!(layer_weights(34.0, 0.0, 0.5)[0] > 0.3, "the rock must be there by 34°");
        // The ramp is narrow on purpose: a wide one has the rock's *partial* weight mixing
        // with the soil's, and a warm brown mixed with a cool grey is a pink patch (measured:
        // 9.67 % of the mountain's frame was pink before, 0.01 % after).
        assert!(layer_weights(31.0, 0.0, 0.5)[0] < 0.2, "the rock must not creep below 30°");
    }

    /// The ground's large-scale variation must reach a **plain**, where the soil's weight is
    /// zero and the grass's is one. Moving the two weights against each other (each by the
    /// other's weight) does nothing there, and the plain stayed one colour; the shift has to
    /// be taken from the grass and given to the soil. The weights must still sum to one and
    /// stay in range, or the ground's mean colour drifts (`rock-material.md` Model 4).
    #[test]
    fn the_layer_variation_reaches_a_plain_and_keeps_the_weights_summing_to_one() {
        let plain = layer_weights(4.0, 0.0, 0.5);
        assert_eq!(plain[2], 0.0, "a plain has no soil to start with");
        assert_eq!(plain[3], 1.0);
        // Walk a line across the field: the soil must appear, the grass must give it up, and
        // the pair must still sum to one.
        let mut max_soil = 0.0f32;
        let mut min_soil = 1.0f32;
        for k in 0..200 {
            let p = Vec3::new(k as f32 * 3.0, 0.0, 0.0);
            let field = crate::gpu::noise_mirror::perlin3_d(
                p / LAYER_VARIATION_M + crate::gpu::noise_mirror::seed_offset(71),
            )
            .x
                / NOISE_RMS;
            let w = layer_variation(field, plain);
            assert!(
                (w.iter().sum::<f32>() - 1.0).abs() < 1e-5,
                "the weights do not sum to one at {p:?}: {w:?}"
            );
            assert!(w.iter().all(|x| (0.0..=1.0).contains(x)), "{w:?}");
            max_soil = max_soil.max(w[2]);
            min_soil = min_soil.min(w[2]);
        }
        assert!(
            max_soil > 0.05,
            "the variation never reaches a plain (the most soil was {max_soil})"
        );
        assert!(
            min_soil < 0.02,
            "the variation never gives the soil back (the least soil was {min_soil})"
        );
        // The rock and the sand are where the terrain's shape put them, and the variation must
        // not move them.
        let steep = layer_weights(50.0, 0.0, 0.5);
        let field = crate::gpu::noise_mirror::perlin3_d(
            Vec3::new(30.0, 0.0, 0.0) / LAYER_VARIATION_M + crate::gpu::noise_mirror::seed_offset(71),
        )
        .x
            / NOISE_RMS;
        let w = layer_variation(field, steep);
        assert_eq!(w[0], 1.0, "the variation moved the rock");
    }

    /// Validation 5: on the plane `h = 0.5 x` the central-difference slope is atan(0.5) at every
    /// cell, edges included, and a flat field has slope 0.
    #[test]
    fn the_slope_of_a_plane_is_exact_up_to_the_edge() {
        let size = UVec2::new(16, 8);
        let plane: Vec<f32> = (0..size.y)
            .flat_map(|_| (0..size.x).map(|i| 0.5 * (i as f32 + 0.5) * 2.0))
            .collect();
        for j in 0..size.y {
            for i in 0..size.x {
                let a = slope_deg(&plane, size, 2.0, i, j);
                assert!((a - 26.565).abs() < 0.01, "cell {i},{j}: {a}");
            }
        }
        assert_eq!(slope_deg(&vec![3.0; 128], size, 2.0, 0, 0), 0.0);
    }

    /// The hex grid's weights are the point's barycentric coordinates in the triangle of the
    /// three hexes' centres: none is negative, they sum to one, and they rebuild the point. A
    /// vertex paired with another's weight rebuilds a different point.
    #[test]
    fn the_hex_weights_are_the_points_barycentric_coordinates() {
        for i in 0..2000 {
            let st = Vec2::new(i as f32 * 0.013_7 - 13.0, (i * 7919 % 2000) as f32 * 0.011_3 - 11.0);
            let (v, w) = hex_grid(st);
            assert!(w.min_element() >= -1e-6, "{st}: a negative weight {w}");
            assert!((w.x + w.y + w.z - 1.0).abs() < 1e-5, "{st}: weights {w}");
            let rebuilt = w.x * hex_centre(v[0]) + w.y * hex_centre(v[1]) + w.z * hex_centre(v[2]);
            assert!((rebuilt - st).length() < 1e-4, "{st} rebuilt as {rebuilt}");
        }
    }

    /// A hex tap that is not fetched could have carried at most 0.5 % of the blend, whatever
    /// the taps' heights: the fetch cut-off is safe, and the blend with it differs from the full
    /// one by no more than that.
    #[test]
    fn a_tap_that_is_not_fetched_carries_almost_nothing() {
        let mut state = 12_345u32;
        let mut next = || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 8) as f32 / 16_777_216.0
        };
        let mut skipped = 0;
        for i in 0..4000 {
            let st = Vec2::new(i as f32 * 0.009_1 - 18.0, (i * 7919 % 4000) as f32 * 0.008_3 - 16.0);
            let (_, bary) = hex_grid(st);
            for _ in 0..4 {
                let heights = Vec3::new(next(), next(), next());
                let full = hex_tap_weights(bary, heights, true);
                let cut = hex_tap_weights(bary, heights, false);
                let shares = hex_shares(bary);
                for k in 0..3 {
                    if shares[k] <= TAP_MIN {
                        skipped += 1;
                        assert!(full[k] <= 0.005, "{st} {heights}: a skipped tap weighed {}", full[k]);
                    }
                }
                assert!((full - cut).abs().max_element() <= 0.005, "{st}: {full} against {cut}");
            }
        }
        assert!(skipped > 4000, "the cut-off skips taps: {skipped}");
    }

    /// A tap's slope turns back into the world by the transpose of its rotation: a texture whose
    /// height rises along a texture direction rises, in the world, along the direction the tap's
    /// coordinate maps onto it. With the rotation itself instead, the relief would be lit from
    /// the wrong side in every hex but the unrotated ones.
    #[test]
    fn a_taps_slope_turns_back_into_the_world() {
        let rise = Vec2::new(0.3, -0.7);
        let e = 1e-3;
        for vx in -4..4 {
            for vy in -4..4 {
                let v = IVec2::new(vx, vy);
                let st = hex_centre(v) + Vec2::new(0.01, 0.02);
                let height = |p: Vec2| rise.dot(hex_tap(v, 2, p).0);
                let finite = Vec2::new(
                    height(st + Vec2::X * e) - height(st - Vec2::X * e),
                    height(st + Vec2::Y * e) - height(st - Vec2::Y * e),
                ) / (2.0 * e);
                let (_, rot) = hex_tap(v, 2, st);
                let slope = tap_slope(rise, rot);
                assert!((slope - finite).length() < 2e-3, "{v}: {slope} vs {finite}");
            }
        }
    }

    /// The height blend keeps the ground's share whatever the heights, lets the taller layer
    /// show, and starts where the sampling does: a layer at the gate has no share and one just
    /// above it a small one, so its tallest texels do not jump in on a contour line.
    #[test]
    fn the_height_blend_keeps_the_ground_and_starts_at_the_gate() {
        let w = Vec3::new(0.0, 0.3, 0.5);
        for h in [Vec3::new(0.5, 0.2, 0.9), Vec3::new(0.5, 0.9, 0.1), Vec3::splat(0.5)] {
            let b = height_blend(w, h, HEIGHT_BLEND);
            assert!((b.x + b.y + b.z - 0.8).abs() < 1e-6, "{h}: {b}");
        }
        let b = height_blend(Vec3::new(0.0, 0.5, 0.5), Vec3::new(0.5, 0.9, 0.2), HEIGHT_BLEND);
        assert!(b.y > 0.95, "the taller soil shows: {b}");
        let tall = Vec3::new(1.0, 0.5, 0.5);
        let at_gate = height_blend(Vec3::new(DETAIL_GATE, 0.0, 1.0 - DETAIL_GATE), tall, HEIGHT_BLEND);
        assert_eq!(at_gate.x, 0.0, "{at_gate}");
        let above = height_blend(
            Vec3::new(DETAIL_GATE + 1e-4, 0.0, 1.0 - DETAIL_GATE - 1e-4),
            tall,
            HEIGHT_BLEND,
        );
        assert!(above.x > 0.0 && above.x < 0.01, "{above}");
    }

    /// The height blend is band-limited: underfoot it is whole, and where a pixel no longer
    /// resolves the heights' features it no longer depends on them, so two pixels whose
    /// filtered heights differ by a hair cannot flip between layers (the mid field's sparkle).
    #[test]
    fn the_height_blend_fades_where_the_heights_are_not_resolved() {
        assert_eq!(height_sharpness(0.002), HEIGHT_BLEND);
        assert_eq!(height_sharpness(HEIGHT_BLEND_LOST_M), 0.0);
        let w = Vec3::new(0.0, 0.5, 0.5);
        let far = height_sharpness(0.05);
        let a = height_blend(w, Vec3::new(0.5, 0.55, 0.45), far);
        let b = height_blend(w, Vec3::new(0.5, 0.45, 0.55), far);
        assert_eq!(a, b, "far: the heights decide nothing");
        let near = height_sharpness(0.002);
        let a = height_blend(w, Vec3::new(0.5, 0.55, 0.45), near);
        let b = height_blend(w, Vec3::new(0.5, 0.45, 0.55), near);
        assert!(a.y > b.y + 0.3, "near: the taller layer shows ({a} against {b})");
    }

    /// A layer the set leaves untextured (tile zero) keeps the whole procedural grain, so the
    /// dunes keep their mottle under a set that textures only the soil and the grass; a textured
    /// layer keeps the set's share, and a mixed ground keeps the shares' weighted mean.
    #[test]
    fn an_untextured_layer_keeps_its_whole_grain() {
        let set = Vec3::new(0.0, 2.0, 2.0);
        assert_eq!(grain_share(Vec3::new(0.9, 0.0, 0.0), set, 0.3), 1.0);
        assert!((grain_share(Vec3::new(0.0, 0.2, 0.7), set, 0.3) - 0.3).abs() < 1e-6);
        assert!((grain_share(Vec3::new(0.5, 0.0, 0.5), set, 0.3) - 0.65).abs() < 1e-6);
        assert!((grain_share(Vec3::new(0.5, 0.0, 0.5), Vec3::splat(2.0), 0.3) - 0.3).abs() < 1e-6);
        assert_eq!(grain_share(Vec3::ZERO, set, 0.3), 1.0, "no ground: no change");
    }
}
