// Per-particle hash and raindrop microphysics shared by the simulation and the draw shaders, so
// a particle's size and speed are the same function in both. See docs/research/precipitation.md.
// `src/precipitation/model.rs` mirrors every function here in the tests.

/// PCG4D hash (Jarzynski & Olano 2020, "Hash Functions for GPU Rendering").
fn pcg4d(v_in: vec4<u32>) -> vec4<u32> {
    var v = v_in * 1664525u + 1013904223u;
    v.x += v.y * v.w;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    v.w += v.y * v.z;
    v ^= v >> vec4<u32>(16u);
    v.x += v.y * v.w;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    v.w += v.y * v.z;
    return v;
}

/// Four independent uniforms in (0, 1) for particle `index` in its `epoch`-th life; `salt`
/// selects an independent stream.
fn particle_rand(index: u32, epoch: u32, salt: u32) -> vec4<f32> {
    let h = pcg4d(vec4<u32>(index, epoch, salt, 0x9E3779B9u));
    return (vec4<f32>(h >> vec4<u32>(8u)) + 0.5) * (1.0 / 16777216.0);
}

/// Diameter in mm of a raindrop drawn from the water-weighted Marshall–Palmer distribution
/// D³·exp(−ΛD), a gamma distribution of shape 4 and rate `lambda` (mm⁻¹): the sum of four
/// exponentials.
fn drop_diameter_mm(u: vec4<f32>, lambda: f32) -> f32 {
    return -log(u.x * u.y * u.z * u.w) / lambda;
}

/// Terminal velocity in m/s of a raindrop of diameter `d_mm`: the Atlas & Ulbrich power law
/// 3.778·D^0.67 (Uijlenhoet 2001), capped at the 9 m/s plateau.
fn drop_fall_speed(d_mm: f32) -> f32 {
    return min(3.778 * pow(max(d_mm, 0.0), 0.67), 9.0);
}

/// Height above the ground in m of a respawning dust sprite for `u` in (0, 1): the exponential
/// profile of scale height `scale` truncated to [lo, hi] (the part of the box above the ground),
/// by inverse CDF: lo − H ln(1 − u (1 − e^(−(hi − lo)/H))). With lo = 0 and hi = 60 it is the
/// note's −H ln(1 − u (1 − e^(−60/H))); `model::dust_height_m` mirrors it.
fn dust_height(u: f32, lo: f32, hi: f32, scale: f32) -> f32 {
    return lo - scale * log(1.0 - u * (1.0 - exp(-(hi - lo) / scale)));
}
