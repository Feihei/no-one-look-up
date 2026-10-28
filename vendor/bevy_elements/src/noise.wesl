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

/// Gradient (Perlin) noise, roughly in [-1, 1].
fn perlin3(p: vec3<f32>) -> f32 {
    return perlin3_d(p).x;
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

/// Divergence-free noise (Bridson, Hourihan & Nordenstam 2007): the curl of three Perlin
/// potentials with separate seeds, from analytic derivatives, per lattice unit. Unnormalised.
fn curl3(p: vec3<f32>) -> vec3<f32> {
    let a = perlin3_d(p + seed_offset(11u));
    let b = perlin3_d(p + seed_offset(12u));
    let c = perlin3_d(p + seed_offset(13u));
    // ψ = (a, b, c): curl = (∂c/∂y − ∂b/∂z, ∂a/∂z − ∂c/∂x, ∂b/∂x − ∂a/∂y)
    return vec3<f32>(c.z - b.w, a.w - c.y, b.y - a.z);
}
