// bevy_elements::noise — hashing and procedural noise shared by every domain.
//
// A supported import for consumers:
//     import bevy_elements::noise::{perlin3, perlin3_d, fbm3, worley3, curl3};
// The lattice is periodic with NOISE_PERIOD cells on every axis, so callers can wrap their
// coordinates (and drifting offsets) modulo NOISE_PERIOD without a seam and keep f32 precision.
// Deterministic on one GPU; not bit-identical across vendors. `src/gpu/noise_mirror.rs` is a
// CPU transliteration used by the tests.

/// Lattice period of every noise in this module, cells.
const NOISE_PERIOD: i32 = 4096;

/// How far a cell network's feature point is drawn *towards* its cell's centre, 0 to 1.
///
/// A cell network's cells are the same size when the feature points sit on a lattice, and a
/// face of equal cells is a mosaic. The point's offset inside its cell is the only place the
/// cells can differ in size: a point near its cell's corner leaves a big cell behind it, and
/// a point at the centre a small one. An inset of 0 leaves the point anywhere in the cell and
/// the cells' sizes vary the most; an inset of 1 puts every point at the centre and every
/// cell is the same size.
///
/// The rock's blocks are the cells of a 3D network (`docs/research/rock-fragments.md`), and a
/// cliff's blocks are not all the same size: the fracture spacing follows a Weibull whose
/// geometric standard deviation is about 0.31 (Verdian et al. 2021).
const CELL_INSET: f32 = 0.0;

/// PCG3D hash (Jarzynski & Olano 2020, "Hash Functions for GPU Rendering").
fn pcg3d(v_in: vec3<u32>) -> vec3<u32> {
    var v = v_in * 1664525u + 1013904223u;
    v.x += v.y * v.z;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    v ^= v >> vec3<u32>(16u);
    v.x += v.y * v.z;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    return v;
}

/// Hash of an integer lattice point (wrapped to the period) to three floats in [0, 1).
fn hash33(p: vec3<i32>) -> vec3<f32> {
    let w = p & vec3<i32>(NOISE_PERIOD - 1);
    return vec3<f32>(pcg3d(bitcast<vec3<u32>>(w)) >> vec3<u32>(8u)) * (1.0 / 16777216.0);
}

/// Hash of an integer lattice point to one float in [0, 1).
fn hash31(p: vec3<i32>) -> f32 {
    return hash33(p).x;
}

fn lattice_gradient(cell: vec3<i32>) -> vec3<f32> {
    return normalize(hash33(cell) * 2.0 - 1.0 + vec3<f32>(1e-6));
}

/// Gradient (Perlin) noise with its analytic derivative: `.x` the value, roughly in [-1, 1],
/// `.yzw` = ∂/∂(x, y, z) per lattice unit. Quintic interpolation (Quílez's closed form), so the
/// derivative is exact and needs no finite differences.
fn perlin3_d(p: vec3<f32>) -> vec4<f32> {
    let i = vec3<i32>(floor(p));
    let f = p - floor(p);
    let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    let du = 30.0 * f * f * (f * (f - 2.0) + 1.0);

    let ga = lattice_gradient(i);
    let gb = lattice_gradient(i + vec3<i32>(1, 0, 0));
    let gc = lattice_gradient(i + vec3<i32>(0, 1, 0));
    let gd = lattice_gradient(i + vec3<i32>(1, 1, 0));
    let ge = lattice_gradient(i + vec3<i32>(0, 0, 1));
    let gf = lattice_gradient(i + vec3<i32>(1, 0, 1));
    let gg = lattice_gradient(i + vec3<i32>(0, 1, 1));
    let gh = lattice_gradient(i + vec3<i32>(1, 1, 1));

    let va = dot(ga, f);
    let vb = dot(gb, f - vec3<f32>(1.0, 0.0, 0.0));
    let vc = dot(gc, f - vec3<f32>(0.0, 1.0, 0.0));
    let vd = dot(gd, f - vec3<f32>(1.0, 1.0, 0.0));
    let ve = dot(ge, f - vec3<f32>(0.0, 0.0, 1.0));
    let vf = dot(gf, f - vec3<f32>(1.0, 0.0, 1.0));
    let vg = dot(gg, f - vec3<f32>(0.0, 1.0, 1.0));
    let vh = dot(gh, f - vec3<f32>(1.0, 1.0, 1.0));

    let k1 = vb - va;
    let k2 = vc - va;
    let k3 = ve - va;
    let k4 = va - vb - vc + vd;
    let k5 = va - vc - ve + vg;
    let k6 = va - vb - ve + vf;
    let k7 = -va + vb + vc - vd + ve - vf - vg + vh;

    let value = va + u.x * k1 + u.y * k2 + u.z * k3 + u.x * u.y * k4 + u.y * u.z * k5
        + u.z * u.x * k6 + u.x * u.y * u.z * k7;
    let grad = ga + u.x * (gb - ga) + u.y * (gc - ga) + u.z * (ge - ga)
        + u.x * u.y * (ga - gb - gc + gd) + u.y * u.z * (ga - gc - ge + gg)
        + u.z * u.x * (ga - gb - ge + gf) + u.x * u.y * u.z * (-ga + gb + gc - gd + ge - gf - gg + gh)
        + du * (vec3<f32>(k1, k2, k3)
            + u.yzx * vec3<f32>(k4, k5, k6)
            + u.zxy * vec3<f32>(k6, k4, k5)
            + u.yzx * u.zxy * k7);
    return vec4<f32>(value, grad) * 1.1547;
}

/// Gradient (Perlin) noise, roughly in [-1, 1]: the **value only**.
///
/// The same function as `perlin3_d` with the **derivative omitted**: the derivative is a second
/// large polynomial — eight vector differences, a `du` term and three swizzles — and it is
/// computed for every evaluation whether or not the caller wants it.
///
/// The crate makes **five** evaluations per rock pixel and **four** of them use the value
/// alone: the bed's region, the rock's rind, the second joint set's segment field, the turf's
/// patch field and the scree's grain. So the derivative is computed and thrown away most of
/// the time, on the crate's hottest path (`docs/research/rock-cost.md`).
///
/// The value is **bit-identical** to `perlin3_d(p).x`: the same eight lattice gradients and the
/// same interpolation, with only the derivative's terms left out. So a call site can switch
/// without changing a single pixel.
fn perlin3(p: vec3<f32>) -> f32 {
    let i = vec3<i32>(floor(p));
    let f = p - floor(p);
    let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);

    let ga = lattice_gradient(i);
    let gb = lattice_gradient(i + vec3<i32>(1, 0, 0));
    let gc = lattice_gradient(i + vec3<i32>(0, 1, 0));
    let gd = lattice_gradient(i + vec3<i32>(1, 1, 0));
    let ge = lattice_gradient(i + vec3<i32>(0, 0, 1));
    let gf = lattice_gradient(i + vec3<i32>(1, 0, 1));
    let gg = lattice_gradient(i + vec3<i32>(0, 1, 1));
    let gh = lattice_gradient(i + vec3<i32>(1, 1, 1));

    let va = dot(ga, f);
    let vb = dot(gb, f - vec3<f32>(1.0, 0.0, 0.0));
    let vc = dot(gc, f - vec3<f32>(0.0, 1.0, 0.0));
    let vd = dot(gd, f - vec3<f32>(1.0, 1.0, 0.0));
    let ve = dot(ge, f - vec3<f32>(0.0, 0.0, 1.0));
    let vf = dot(gf, f - vec3<f32>(1.0, 0.0, 1.0));
    let vg = dot(gg, f - vec3<f32>(0.0, 1.0, 1.0));
    let vh = dot(gh, f - vec3<f32>(1.0, 1.0, 1.0));

    let k1 = vb - va;
    let k2 = vc - va;
    let k3 = ve - va;
    let k4 = va - vb - vc + vd;
    let k5 = va - vc - ve + vg;
    let k6 = va - vb - ve + vf;
    let k7 = -va + vb + vc - vd + ve - vf - vg + vh;

    let value = va + u.x * k1 + u.y * k2 + u.z * k3 + u.x * u.y * k4 + u.y * u.z * k5
        + u.z * u.x * k6 + u.x * u.y * u.z * k7;
    return value * 1.1547;
}

/// Gradient (Perlin) noise on a lattice that wraps every `period` cells on every axis, so the
/// result tiles with period `period` (in lattice units). `period` must be a power of two no
/// larger than NOISE_PERIOD. Roughly in [-1, 1], quintic interpolation, same gradients as
/// `perlin3` (a cell's gradient is `lattice_gradient` of the wrapped cell).
fn perlin3_periodic(p: vec3<f32>, period: i32) -> f32 {
    let i = vec3<i32>(floor(p));
    let f = p - floor(p);
    let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    let m = vec3<i32>(period - 1);
    let va = dot(lattice_gradient(i & m), f);
    let vb = dot(lattice_gradient((i + vec3<i32>(1, 0, 0)) & m), f - vec3<f32>(1.0, 0.0, 0.0));
    let vc = dot(lattice_gradient((i + vec3<i32>(0, 1, 0)) & m), f - vec3<f32>(0.0, 1.0, 0.0));
    let vd = dot(lattice_gradient((i + vec3<i32>(1, 1, 0)) & m), f - vec3<f32>(1.0, 1.0, 0.0));
    let ve = dot(lattice_gradient((i + vec3<i32>(0, 0, 1)) & m), f - vec3<f32>(0.0, 0.0, 1.0));
    let vf = dot(lattice_gradient((i + vec3<i32>(1, 0, 1)) & m), f - vec3<f32>(1.0, 0.0, 1.0));
    let vg = dot(lattice_gradient((i + vec3<i32>(0, 1, 1)) & m), f - vec3<f32>(0.0, 1.0, 1.0));
    let vh = dot(lattice_gradient((i + vec3<i32>(1, 1, 1)) & m), f - vec3<f32>(1.0, 1.0, 1.0));
    let x0 = mix(mix(va, vb, u.x), mix(vc, vd, u.x), u.y);
    let x1 = mix(mix(ve, vf, u.x), mix(vg, vh, u.x), u.y);
    return mix(x0, x1, u.z) * 1.1547;
}

/// 2D gradient (Perlin) noise in the XZ plane, with its analytic derivative: `.x` the value in
/// roughly [-1, 1], `.yz` = ∂/∂(x, z) per lattice unit. The gradients are unit vectors on the
/// circle, so the noise is isotropic in the plane.
///
/// The ground's *turf* uses this, and the ground's turf is on the near-horizontal surfaces,
/// where the plane's projection is the surface. It is not used on a slope: a 2D noise in the
/// world XZ plane is the *projection* of the surface onto that plane, so on a slope the plane's
/// lattice is stretched along the slope and the lattice shows through as a weave at a fixed
/// world direction — 43.5 degrees at 2 m and 44.0 degrees at 6 m on a 45 degree face
/// (`docs/research/rock-material.md`, Model 1). The rock's material uses the 3D noise at the 3D
/// point instead, which has no projection at all.
fn perlin2_gradient(c: vec2<i32>) -> vec2<f32> {
    let h = hash33(vec3<i32>(c.x, c.y, 0)).xy * 2.0 - vec2<f32>(1.0);
    return normalize(h + vec2<f32>(1e-6));
}

fn perlin2_d(p: vec2<f32>) -> vec3<f32> {
    let i = vec2<i32>(floor(p));
    let f = p - floor(p);
    let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    let du = 30.0 * f * f * (f * (f - 2.0) + 1.0);
    let ga = perlin2_gradient(i);
    let gb = perlin2_gradient(i + vec2<i32>(1, 0));
    let gc = perlin2_gradient(i + vec2<i32>(0, 1));
    let gd = perlin2_gradient(i + vec2<i32>(1, 1));
    let va = dot(ga, f);
    let vb = dot(gb, f - vec2<f32>(1.0, 0.0));
    let vc = dot(gc, f - vec2<f32>(0.0, 1.0));
    let vd = dot(gd, f - vec2<f32>(1.0, 1.0));
    let k1 = vb - va;
    let k2 = vc - va;
    let k3 = va - vb - vc + vd;
    let value = va + u.x * k1 + u.y * k2 + u.x * u.y * k3;
    let grad = ga + u.x * (gb - ga) + u.y * (gc - ga) + u.x * u.y * (ga - gb - gc + gd)
        + du * (vec2<f32>(k1, k2) + vec2<f32>(u.y, u.x) * k3);
    return vec3<f32>(value, grad) * 1.4142;
}

/// Fractal sum of `octaves` periodic Perlin octaves (lacunarity 2, gain 0.5) that tiles with
/// `period` lattice units at the base octave, divided by the sum of the amplitudes. Octave `o`
/// uses lattice period `period · 2^o` and its own whole-cell hash offset.
fn fbm3_periodic(p: vec3<f32>, period: i32, octaves: u32) -> f32 {
    var sum = 0.0;
    var amplitude = 0.5;
    var norm = 0.0;
    var q = p;
    var n = period;
    for (var o = 0u; o < octaves; o += 1u) {
        sum += amplitude * perlin3_periodic(q + seed_offset(o + 1u), n);
        norm += amplitude;
        amplitude *= 0.5;
        q = q * 2.0;
        n = n * 2;
    }
    return sum / max(norm, 1e-6);
}

/// Integer lattice offset that decorrelates octave or component `k` (a different hash seed
/// without moving the lattice).
fn seed_offset(k: u32) -> vec3<f32> {
    return vec3<f32>(f32(k * 1031u % 4096u), f32(k * 2053u % 4096u), f32(k * 3079u % 4096u));
}

/// Fractal sum of `octaves` Perlin octaves (lacunarity 2, gain 0.5, a separate seed per octave),
/// divided by the sum of the amplitudes.
fn fbm3(p: vec3<f32>, octaves: u32) -> f32 {
    var sum = 0.0;
    var amplitude = 0.5;
    var norm = 0.0;
    var q = p;
    for (var o = 0u; o < octaves; o += 1u) {
        sum += amplitude * perlin3(q + seed_offset(o + 1u));
        norm += amplitude;
        amplitude *= 0.5;
        q = q * 2.0;
    }
    return sum / max(norm, 1e-6);
}

/// Worley (cellular) noise: distance to the nearest feature point, roughly in [0, 1].
/// `period` > 0 wraps the lattice so the result tiles with that period.
fn worley3(p: vec3<f32>, period: i32) -> f32 {
    let i = vec3<i32>(floor(p));
    let f = p - floor(p);
    var best = 1e9;
    for (var z = -1; z <= 1; z += 1) {
        for (var y = -1; y <= 1; y += 1) {
            for (var x = -1; x <= 1; x += 1) {
                let o = vec3<i32>(x, y, z);
                var c = i + o;
                if period > 0 {
                    c = ((c % period) + period) % period;
                }
                let d = vec3<f32>(o) + hash33(c) - f;
                best = min(best, dot(d, d));
            }
        }
    }
    return sqrt(best);
}

/// A cell of the 2D cellular network: where its feature point is, how far the point is, and the
/// cell's own random factor in [0, 1).
struct Cell2 {
    /// The distance to the feature point, in lattice units.
    distance: f32,
    /// The feature point's position, in lattice units.
    point: vec2<f32>,
    /// The cell's own random factor, in [0, 1): a per-cell draw, used for the cell's size.
    factor: f32,
}

/// The 2D cellular (Worley) network: the *nearest* feature point, its distance and its cell's
/// random factor.
///
/// The nearest point's *distance* is a field whose gradient points away from the point, so the
/// field is a dome over each point: the crate's scree uses that to shade each stone
/// (`docs/research/scree.md`). The cell's random factor is a per-cell draw, which is how one
/// network carries stones of *many* sizes: the cell's size is its own size times the factor.
///
/// The lattice is periodic with [`NOISE_PERIOD`] cells on both axes, so the caller can wrap the
/// coordinates modulo the period without a seam.
fn cell2(p: vec2<f32>) -> Cell2 {
    let i = vec2<i32>(floor(p));
    let f = p - floor(p);
    var best = 1e9;
    var best_point = vec2<f32>(0.0);
    var best_factor = 0.0;
    for (var y = -1; y <= 1; y += 1) {
        for (var x = -1; x <= 1; x += 1) {
            let o = vec2<i32>(x, y);
            let c = (i + o) & vec2<i32>(NOISE_PERIOD - 1);
            let h = hash33(vec3<i32>(c.x, c.y, 0));
            // The point's position inside its cell, and its factor, from one hash: the point's
            // third component is the factor, so the point does not sit on the cell's centre.
            let point = vec2<f32>(o) + h.xy;
            let d = dot(point - f, point - f);
            if d < best {
                best = d;
                best_point = point;
                best_factor = h.z;
            }
        }
    }
    return Cell2(sqrt(best), best_point, best_factor);
}

/// A cell of the 3D cellular network: where its feature point is, how far the point is, and
/// the cell's own random factor in [0, 1).
struct Cell3 {
    /// The distance to the feature point, in lattice units.
    distance: f32,
    /// The feature point's position, in lattice units.
    point: vec3<f32>,
    /// The cell's own random factor, in [0, 1): a per-cell draw, used for the cell's tone.
    factor: f32,
    /// A **second** per-cell draw, in [0, 1): an independent hash of the cell, for the rock's
    /// palette. The palette wants two draws — the *rock* a fragment is and the *region* it
    /// belongs to — and one hash gives one draw, so the second is a second hash of the same
    /// cell. It is a *separate* hash and not a component of the first, so the two draws are
    /// independent.
    factor2: f32,
}

/// The 3D cellular (Worley) network: the *nearest* feature point, its distance and the cell's
/// random factor.
///
/// The rock's own 3D frame, so a block stays put on a slope: a 2D network in the ground plane
/// would stretch along the slope and weave, exactly as the crate's turf does
/// (`docs/research/rock-material.md` Model 1).
///
/// The lookup is **27 cells**, each a hash, so it is expensive — the crate's rock uses it only
/// where the rock shows, and the rock's material is the crate's hot path. The cells are pruned
/// by the distance to their nearest corner, which is an exact bound, so the result is the same
/// as the full search.
fn cell3(p: vec3<f32>) -> Cell3 {
    let i = vec3<i32>(floor(p));
    let f = p - floor(p);
    var best = 1e9;
    var best_point = vec3<f32>(0.0);
    var best_factor = 0.0;
    var best_factor2 = 0.0;
    // The cells are visited in order of increasing distance — the point's own cell first, then
    // the ring around it, then the ring around that — so the prune's bound tightens quickly.
    // In the natural order the bound is loose and about fourteen of the twenty-seven cells
    // are hashed; in this order it is about seven. The bound is exact, so the result is the
    // same either way, and the crate's rock is its hottest path.
    for (var ring = 0; ring < 3; ring += 1) {
      for (var z = -1; z <= 1; z += 1) {
        for (var y = -1; y <= 1; y += 1) {
            for (var x = -1; x <= 1; x += 1) {
                // The pass is the ring the cell is in, from the point's own cell outwards.
                if max(max(abs(x), abs(y)), abs(z)) != ring {
                    continue;
                }
                let o = vec3<i32>(x, y, z);
                // The cell is pruned by the distance to its *nearest corner*: a feature point
                // lies inside its cell, so if the corner is already further than the best
                // distance so far, the cell cannot hold a closer one and its hash — the
                // expensive part — is not computed at all. The bound is exact, so the result is
                // the same as the full search.
                // The distance from the point to the cell's *far* corner, per axis: the point
                // is at `f` inside its own cell, and the cell spans `[o, o + 1]`. A feature
                // point lies inside its cell, so a cell whose far corner is nearer than the
                // best distance so far cannot hold a closer one, and its hash — the expensive
                // part — is not computed at all. The bound is exact, so the result is the same
                // as the full search.
                //
                // The two distances are the point's offsets from the cell's two ends, and the
                // larger of them is the far one.
                let from_start = vec3<f32>(o) - f;
                let from_end = f - (vec3<f32>(o) + 1.0);
                let lo = max(from_start, from_end);
                if dot(lo, lo) >= best {
                    continue;
                }
                let c = (i + o) & vec3<i32>(NOISE_PERIOD - 1);
                let h = hash33(c);
                // The point's position inside its cell, and its factor, from one hash: the
                // point's third component is the factor, so the point does not sit on the
                // cell's centre.
                //
                // The point's offset is what makes the cells **unequal**, and it is the only
                // place a cell network's cells differ in size: the *cells* of a Voronoi
                // network are the same size when the feature points sit on a lattice, and a
                // face of equal cells is a mosaic. A point that sits near its cell's corner
                // leaves a *big* cell behind it and a point at the centre a small one, so the
                // offset's spread is the blocks' size spread.
                //
                // The offset is the hash's own value, drawn towards the cell's centre by
                // `CELL_INSET`: an inset of 0 leaves the point anywhere in the cell and the
                // cells' sizes vary the most, and an inset of 1 puts every point at the
                // centre and every cell is the same size.
                let inset = mix(h, vec3<f32>(0.5), CELL_INSET);
                let point = vec3<f32>(o) + inset;
                let d = dot(point - f, point - f);
                if d < best {
                    best = d;
                    best_point = point;
                    best_factor = h.z;
                    best_factor2 = hash31(c + vec3<i32>(7919, 104729, 1299709));
                }
            }
        }
      }
    }
    return Cell3(sqrt(best), best_point, best_factor, best_factor2);
}

/// Divergence-free noise (Bridson, Hourihan & Nordenstrm 2007): the curl of three Perlin
/// potentials with separate seeds, from analytic derivatives, per lattice unit. Unnormalised.
fn curl3(p: vec3<f32>) -> vec3<f32> {
    let a = perlin3_d(p + seed_offset(11u));
    let b = perlin3_d(p + seed_offset(12u));
    let c = perlin3_d(p + seed_offset(13u));
    // ψ = (a, b, c): curl = (∂c/∂y − ∂b/∂z, ∂a/∂z − ∂c/∂x, ∂b/∂x − ∂a/∂y)
    return vec3<f32>(c.z - b.w, a.w - c.y, b.y - a.z);
}
