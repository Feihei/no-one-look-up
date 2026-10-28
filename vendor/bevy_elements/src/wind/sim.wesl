// Wind field: camera-centred, toroidally addressed stable-fluids solver (Stam 1999) on a
// collocated grid. See docs/research/wind.md.
//
// Frame order (bind group = velocity in→out, pressure in→out):
//   forcing    A→B          solid mask; relax toward log-law + curl noise + gust bands
//   advect     B→A          semi-Lagrangian
//   divergence A→·, p B→A   divergence; zero pressure in freshly exposed cells
//   jacobi ×n     p A→B, B→A, …  (n even: the result is back in A)
//   project    A→B, p A
//   publish    B→A          copy back to A and write the sampled 3D texture

import bevy_elements::grid::grid_taps;
import bevy_elements::noise::{perlin3_d, seed_offset};
import bevy_elements::terrain::common::{TerrainParams, terrain_live};

struct Gust {
    // xy: band centre (world XZ), zw: unit direction it travels (world XZ).
    a: vec4<f32>,
    // x: width (m), y: peak strength (m/s).
    b: vec4<f32>,
}

struct Params {
    // xyz: cells per axis.
    size: vec4<u32>,
    // xyz: world cell index of the window's minimum corner; w: 1 on a reset frame.
    origin: vec4<i32>,
    // xyz: last frame's origin.
    prev_origin: vec4<i32>,
    // xy: prevailing direction (world XZ, blows toward), z: u10 (m/s), w: gustiness 0–1.
    wind: vec4<f32>,
    // x: z0 (m), y: relaxation factor 1 − e^(−rΔt), z: curl amplitude per unit raw curl (m/s),
    // w: coarse curl octave L (m).
    forcing: vec4<f32>,
    // x: cell edge (m), y: Δt (s), z: unused, w: active gust count.
    misc: vec4<f32>,
    // xyz: noise lattice offset of the coarse octave (drifts downwind, wrapped at the period).
    noise_offset: vec4<f32>,
    // The terrain grid; without a live terrain the ground is the plane y = 0.
    terrain: TerrainParams,
    gusts: array<Gust, 16>,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> vel_in: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> vel_out: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> solid: array<u32>;
@group(0) @binding(4) var<storage, read_write> divergence_field: array<f32>;
@group(0) @binding(5) var<storage, read> p_in: array<f32>;
@group(0) @binding(6) var<storage, read_write> p_out: array<f32>;
@group(0) @binding(7) var field: texture_storage_3d<rgba16float, write>;
@group(0) @binding(8) var<storage, read> terrain_cells: array<vec4<f32>>;

// Kolmogorov octave weights (Kim et al. 2008): (1, 2^(−5/6)) normalised to unit total rms.
const OCTAVE_WEIGHTS: vec2<f32> = vec2<f32>(0.8721, 0.4895);
// The curl noise fades in over this height above ground, m.
const CURL_RAMP_M: f32 = 4.0;

fn dims() -> vec3<i32> {
    return vec3<i32>(params.size.xyz);
}

fn cell() -> f32 {
    return params.misc.x;
}

// Storage index of a world cell: toroidal, so a cell keeps its slot while the window scrolls.
fn slot(w: vec3<i32>) -> u32 {
    let n = dims();
    let s = ((w % n) + n) % n;
    return u32(s.x + n.x * (s.y + n.y * s.z));
}

// Storage index of window coordinate `g`, clamped into the window (the window edge is a wall
// for the stencils: nothing wraps across the seam).
fn at(g: vec3<i32>) -> u32 {
    return slot(params.origin.xyz + clamp(g, vec3<i32>(0), dims() - 1));
}

fn inside(g: vec3<i32>) -> bool {
    return all(g >= vec3<i32>(0)) && all(g < dims());
}

fn centre(g: vec3<i32>) -> vec3<f32> {
    return (vec3<f32>(params.origin.xyz + g) + 0.5) * cell();
}

fn is_solid(g: vec3<i32>) -> bool {
    return solid[at(g)] != 0u;
}

// Bilinear terrain surface (world height, m) at world XZ; the plane y = 0 without a terrain.
fn ground_at(xz: vec2<f32>) -> f32 {
    if !terrain_live(params.terrain) {
        return 0.0;
    }
    let t = grid_taps(params.terrain.grid, xz);
    let a = mix(terrain_cells[t.i00].x, terrain_cells[t.i10].x, t.w.x);
    let b = mix(terrain_cells[t.i01].x, terrain_cells[t.i11].x, t.w.x);
    return params.terrain.misc.x + mix(a, b, t.w.y);
}

// The ground a wind cell centred over `xz` must clear to be air: the highest surface at the
// cell's four quarter points (x ± 1, z ± 1 m), so a slope makes cells solid rather than leaky.
fn solid_ground_at(xz: vec2<f32>) -> f32 {
    return max(
        max(ground_at(xz + vec2<f32>(1.0, 1.0)), ground_at(xz + vec2<f32>(-1.0, 1.0))),
        max(ground_at(xz + vec2<f32>(1.0, -1.0)), ground_at(xz + vec2<f32>(-1.0, -1.0))),
    );
}

// Neutral log-law speed at height z above ground (WMO open-terrain z0).
fn log_law(z: f32) -> f32 {
    let z0 = params.forcing.x;
    return params.wind.z * log(max(z, z0) / z0) / log(10.0 / z0);
}

// curl(A ψ) of one octave, per lattice unit: ψ three Perlin potentials with separate seeds,
// analytic derivatives, and A(z) a ramp from the ground so the noise stays divergence-free
// while fading out at the surface (Bridson et al. 2007 §2.3). `scale` is the octave's lattice
// size in metres.
fn ramped_curl(p: vec3<f32>, z: f32, scale: f32) -> vec3<f32> {
    let a = perlin3_d(p + seed_offset(11u));
    let b = perlin3_d(p + seed_offset(12u));
    let c = perlin3_d(p + seed_offset(13u));
    let curl = vec3<f32>(c.z - b.w, a.w - c.y, b.y - a.z);
    let t = clamp(z / CURL_RAMP_M, 0.0, 1.0);
    let ramp = t * t * (3.0 - 2.0 * t);
    // ∇A × ψ with ∇A = (0, A', 0), A' per lattice unit.
    let slope = 6.0 * t * (1.0 - t) / CURL_RAMP_M * scale;
    return ramp * curl + slope * vec3<f32>(c.x, 0.0, -a.x);
}

// The velocity the forcing relaxes toward; w = turbulence intensity of the noise part.
fn target_wind(pos: vec3<f32>, z: f32) -> vec4<f32> {
    let dir = vec3<f32>(params.wind.x, 0.0, params.wind.y);
    let mean = dir * log_law(z);
    let coarse = params.forcing.w;
    let lattice = pos / coarse + params.noise_offset.xyz;
    let noise = (OCTAVE_WEIGHTS.x * ramped_curl(lattice, z, coarse)
        + OCTAVE_WEIGHTS.y * ramped_curl(2.0 * lattice, z, 0.5 * coarse)) * params.forcing.z;
    var gust = vec3<f32>(0.0);
    let shape = log_law(z) / max(params.wind.z, 1e-3);
    let count = u32(params.misc.w);
    for (var k = 0u; k < count; k += 1u) {
        let g = params.gusts[k];
        let s = dot(pos.xz - g.a.xy, g.a.zw);
        let half_width = 0.5 * g.b.x;
        if abs(s) < half_width {
            let profile = 0.5 * (1.0 + cos(3.14159265 * s / half_width));
            gust += vec3<f32>(g.a.z, 0.0, g.a.w) * (g.b.y * profile * shape);
        }
    }
    let v = mean + noise + gust;
    let intensity = length(noise + gust) / max(length(v), 0.1);
    return vec4<f32>(v, intensity);
}

@compute @workgroup_size(4, 4, 4)
fn forcing(@builtin(global_invocation_id) id: vec3<u32>) {
    if any(id >= params.size.xyz) {
        return;
    }
    let g = vec3<i32>(id);
    let w = params.origin.xyz + g;
    let i = slot(w);
    let pos = centre(g);
    let is_ground = pos.y < solid_ground_at(pos.xz);
    solid[i] = select(0u, 1u, is_ground);
    let z = pos.y - ground_at(pos.xz);
    if is_ground {
        vel_out[i] = vec4<f32>(0.0);
        return;
    }
    let goal = target_wind(pos, z);
    let lo = params.prev_origin.xyz;
    let fresh = params.origin.w != 0 || any(w < lo) || any(w >= lo + dims());
    var v = vel_in[i];
    if fresh {
        v = goal;
    } else {
        v = v + (goal - v) * params.forcing.y;
    }
    vel_out[i] = v;
}

fn sample_in(gp: vec3<f32>) -> vec4<f32> {
    let x = gp - 0.5;
    let b = vec3<i32>(floor(x));
    let f = x - floor(x);
    let c000 = vel_in[at(b)];
    let c100 = vel_in[at(b + vec3<i32>(1, 0, 0))];
    let c010 = vel_in[at(b + vec3<i32>(0, 1, 0))];
    let c110 = vel_in[at(b + vec3<i32>(1, 1, 0))];
    let c001 = vel_in[at(b + vec3<i32>(0, 0, 1))];
    let c101 = vel_in[at(b + vec3<i32>(1, 0, 1))];
    let c011 = vel_in[at(b + vec3<i32>(0, 1, 1))];
    let c111 = vel_in[at(b + vec3<i32>(1, 1, 1))];
    let x00 = mix(c000, c100, f.x);
    let x10 = mix(c010, c110, f.x);
    let x01 = mix(c001, c101, f.x);
    let x11 = mix(c011, c111, f.x);
    return mix(mix(x00, x10, f.y), mix(x01, x11, f.y), f.z);
}

@compute @workgroup_size(4, 4, 4)
fn advect(@builtin(global_invocation_id) id: vec3<u32>) {
    if any(id >= params.size.xyz) {
        return;
    }
    let g = vec3<i32>(id);
    let i = at(g);
    if is_solid(g) {
        vel_out[i] = vec4<f32>(0.0);
        return;
    }
    let v = vel_in[i];
    let back = vec3<f32>(g) + 0.5 - v.xyz * (params.misc.y / cell());
    vel_out[i] = sample_in(back);
}

// Velocity of a neighbour for the divergence stencil (Harris 2004 boundary rules): across a
// solid face the component along the stencil axis is mirrored and the others slip freely; the
// window top is a lid (vertical component mirrored); the other window edges are open (the
// centre's value, zero gradient).
fn vel_nbr(g: vec3<i32>, o: vec3<i32>) -> vec3<f32> {
    let n = g + o;
    let c = vel_in[at(g)].xyz;
    if !inside(n) {
        if o.y > 0 {
            return vec3<f32>(c.x, -c.y, c.z);
        }
        return c;
    }
    if is_solid(n) {
        return c * (vec3<f32>(1.0) - 2.0 * abs(vec3<f32>(o)));
    }
    return vel_in[at(n)].xyz;
}

@compute @workgroup_size(4, 4, 4)
fn divergence(@builtin(global_invocation_id) id: vec3<u32>) {
    if any(id >= params.size.xyz) {
        return;
    }
    let g = vec3<i32>(id);
    let w = params.origin.xyz + g;
    let i = slot(w);
    let lo = params.prev_origin.xyz;
    if params.origin.w != 0 || any(w < lo) || any(w >= lo + dims()) {
        p_out[i] = 0.0;
    }
    if is_solid(g) {
        divergence_field[i] = 0.0;
        return;
    }
    let dx = vel_nbr(g, vec3<i32>(1, 0, 0)).x - vel_nbr(g, vec3<i32>(-1, 0, 0)).x;
    let dy = vel_nbr(g, vec3<i32>(0, 1, 0)).y - vel_nbr(g, vec3<i32>(0, -1, 0)).y;
    let dz = vel_nbr(g, vec3<i32>(0, 0, 1)).z - vel_nbr(g, vec3<i32>(0, 0, -1)).z;
    divergence_field[i] = (dx + dy + dz) / (2.0 * cell());
}

// Pressure of a neighbour: the centre's value across a wall or the lid (Neumann), zero across
// the open window edges (Dirichlet: the air leaves freely, so a terrain that blocks more of the
// window on one side than the other leaves a solvable Poisson problem; with Neumann pressure on
// every edge the net flux through the open edges is unbalanced and the divergence cannot be
// projected out, wind.md Deviation 1).
fn p_nbr(g: vec3<i32>, o: vec3<i32>, centre_p: f32) -> f32 {
    let n = g + o;
    if !inside(n) {
        return select(0.0, centre_p, n.y >= dims().y);
    }
    if is_solid(n) {
        return centre_p;
    }
    return p_in[at(n)];
}

@compute @workgroup_size(4, 4, 4)
fn jacobi(@builtin(global_invocation_id) id: vec3<u32>) {
    if any(id >= params.size.xyz) {
        return;
    }
    let g = vec3<i32>(id);
    let i = at(g);
    if is_solid(g) {
        p_out[i] = 0.0;
        return;
    }
    let pc = p_in[i];
    let sum = p_nbr(g, vec3<i32>(1, 0, 0), pc) + p_nbr(g, vec3<i32>(-1, 0, 0), pc)
        + p_nbr(g, vec3<i32>(0, 1, 0), pc) + p_nbr(g, vec3<i32>(0, -1, 0), pc)
        + p_nbr(g, vec3<i32>(0, 0, 1), pc) + p_nbr(g, vec3<i32>(0, 0, -1), pc);
    let h = cell();
    p_out[i] = (sum - h * h * divergence_field[i]) / 6.0;
}

@compute @workgroup_size(4, 4, 4)
fn project(@builtin(global_invocation_id) id: vec3<u32>) {
    if any(id >= params.size.xyz) {
        return;
    }
    let g = vec3<i32>(id);
    let i = at(g);
    if is_solid(g) {
        vel_out[i] = vec4<f32>(0.0);
        return;
    }
    let pc = p_in[i];
    let grad = vec3<f32>(
        p_nbr(g, vec3<i32>(1, 0, 0), pc) - p_nbr(g, vec3<i32>(-1, 0, 0), pc),
        p_nbr(g, vec3<i32>(0, 1, 0), pc) - p_nbr(g, vec3<i32>(0, -1, 0), pc),
        p_nbr(g, vec3<i32>(0, 0, 1), pc) - p_nbr(g, vec3<i32>(0, 0, -1), pc),
    ) / (2.0 * cell());
    let v = vel_in[i];
    vel_out[i] = vec4<f32>(v.xyz - grad, v.w);
}

@compute @workgroup_size(4, 4, 4)
fn publish(@builtin(global_invocation_id) id: vec3<u32>) {
    if any(id >= params.size.xyz) {
        return;
    }
    let w = params.origin.xyz + vec3<i32>(id);
    let n = dims();
    let s = ((w % n) + n) % n;
    let i = slot(w);
    let v = vel_in[i];
    vel_out[i] = v;
    textureStore(field, s, v);
}
