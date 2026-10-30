// The wave field (docs/research/waves.md): three camera-following cascades, a factor of four
// apart, of a linear capillary-gravity wave field `h` on the hydrology's water, each stepped by
// Tessendorf's iWave scheme with a 13 × 13 kernel fitted to Airy's dispersion over its band, and
// forced by the bodies' pressure head split into the bands like a Laplacian pyramid.
//
//   wave_head     per body and head plane: the pressure head of its column (plus the impact
//                 head of its entry), box-averaged over the plane's texel (4 × 4 samples),
//                 `atomicMax` into the plane
//   wave_force    per cascade texel, once a frame: the band's forcing (3 × 3 binomial), the
//                 water's depth and the current
//   wave_step     per cascade texel, per substep: skip dry ground, advect by the current,
//                 convolve `h + forcing` with the cascade's kernel from a workgroup tile, update,
//                 break, foam, shore mask, window sponge
//   wave_publish  per cascade texel: `(h, ∂h/∂x, ∂h/∂z, foam)` into the texture array; clear the
//                 heads
//
// Windows are toroidal by world texel: world texel `w` lives in slot `w mod N`. Head planes:
// 0..2 the head of cascade c at its own texels (P_c), 3..4 the head of cascade c at the next
// cascade's texels (Q_c); cascade c is forced by `P_c − bilinear(Q_c)`, the last by `P_c`.

import bevy_elements::grid::grid_taps;
import bevy_elements::terrain::common::TerrainParams;

const CASCADES: u32 = 3u;
const PLANES: u32 = 5u;
const RATIO: f32 = 4.0;
const RADIUS: i32 = 6;
const TILE: i32 = 32;
const SPAN: i32 = 44;
// The state's planes per cascade (`h_n`, `h_{n−1}`, foam) and the forcing's (forcing, water
// depth, current x and z), each N² floats: a tile load reads only the planes it needs.
const STATE_PLANES: u32 = 3u;
const H: u32 = 0u;
const PREV: u32 = 1u;
const FOAM: u32 = 2u;
const FORCE_PLANES: u32 = 4u;
const FORCING: u32 = 0u;
const DEPTH: u32 = 1u;
const FLOW_X: u32 = 2u;
const FLOW_Z: u32 = 3u;
const SUPER: i32 = 4;
// The shore: no water below 3 mm, full waves above 3 cm (the water's own edge).
const SHORE_MIN: f32 = 0.003;
const SHORE_FULL: f32 = 0.03;
// The window's outer 6 % damps toward the edge (Crest's feather).
const SPONGE: f32 = 0.06;
const SPONGE_FLOOR: f32 = 0.95;
// Breaking: the steepest slope a wave holds (`ak` = 0.44, amplitude 7 % of the wavelength;
// Jeschke & Wojtan 2017 after Dean & Dalrymple 1991); foam from 75 % of it.
const BREAKING_SLOPE: f32 = 0.44;
const FOAM_ONSET: f32 = 0.75;
// The e-folding life of the foam, s.
const FOAM_LIFE_S: f32 = 3.0;

struct WaveParams {
    terrain: TerrainParams,
    // Per cascade: xy the window's minimum world texel this frame, zw last frame's.
    origins: array<vec4<i32>, 3>,
    // x: texels a side. y: contacts. z: 1 to clear the field. w: head tile edge, finest texels.
    counts: vec4<u32>,
    // x: the finest texel (m). y: substep (s). z: damping (1/s).
    sim: vec4<f32>,
    // xyz: each cascade's band-centre wavenumber (rad/m).
    band: vec4<f32>,
    // Each cascade's 28 weights, seven vec4s per cascade.
    kernel: array<vec4<f32>, 21>,
}

struct ContactGpu {
    // xyz: centre (world, m). w: XZ bounding radius (m).
    centre: vec4<f32>,
    // The ellipsoid's matrix: xx, xy, xz, yy.
    m0: vec4<f32>,
    // yz, zz.
    m1: vec4<f32>,
}

@group(0) @binding(0) var<uniform> params: WaveParams;
@group(0) @binding(1) var<storage, read> state_in: array<f32>;
@group(0) @binding(2) var<storage, read_write> state_out: array<f32>;
@group(0) @binding(3) var<storage, read_write> head: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read> contacts: array<ContactGpu>;
@group(0) @binding(5) var<storage, read> terrain_cells: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read> flow_cells: array<vec4<f32>>;
@group(0) @binding(7) var field: texture_storage_2d_array<rgba16float, write>;
@group(0) @binding(8) var<storage, read_write> force: array<f32>;

// The kernel's distinct offsets (m, n), 0 ≤ n ≤ m ≤ 6, are `model.rs` `taps()` in order: weight
// `i` of a cascade belongs to the i-th, and `wave_step` spells them out.

// The tile's value at `l + (dx, dz)`.
fn at(l: vec2<i32>, dx: i32, dz: i32) -> f32 {
    return tile[(l.y + dz) * SPAN + l.x + dx];
}

// The four images of offset (m, 0).
fn axis(l: vec2<i32>, m: i32) -> f32 {
    return at(l, m, 0) + at(l, -m, 0) + at(l, 0, m) + at(l, 0, -m);
}

// The four images of offset (m, m).
fn diag(l: vec2<i32>, m: i32) -> f32 {
    return at(l, m, m) + at(l, -m, m) + at(l, m, -m) + at(l, -m, -m);
}

// The eight images of offset (m, q), 0 < q < m.
fn skew(l: vec2<i32>, m: i32, q: i32) -> f32 {
    return at(l, m, q) + at(l, -m, q) + at(l, m, -q) + at(l, -m, -q)
        + at(l, q, m) + at(l, -q, m) + at(l, q, -m) + at(l, -q, -m);
}

// Weight `i` (a constant) of cascade `c`'s kernel.
fn kw(c: u32, i: u32) -> f32 {
    return params.kernel[c * 7u + i / 4u][i % 4u];
}

var<workgroup> tile: array<f32, 1936>;

fn size() -> i32 {
    return i32(params.counts.x);
}

fn texel_m(c: u32) -> f32 {
    return params.sim.x * pow(RATIO, f32(c));
}

fn origin(c: u32) -> vec2<i32> {
    return params.origins[c].xy;
}

fn prev_origin(c: u32) -> vec2<i32> {
    return params.origins[c].zw;
}

fn in_window(w: vec2<i32>, o: vec2<i32>) -> bool {
    return all(w >= o) && all(w < o + vec2<i32>(size()));
}

// The slot of world texel `w`: `w mod N`, a mask since every window is a power of two.
fn slot(w: vec2<i32>) -> vec2<i32> {
    return w & vec2<i32>(size() - 1);
}

// Index of world texel `w` of cascade (or head plane) `layer`.
fn index(layer: u32, w: vec2<i32>) -> u32 {
    let n = size();
    let s = slot(w);
    return layer * u32(n * n) + u32(s.y * n + s.x);
}

fn centre(c: u32, w: vec2<i32>) -> vec2<f32> {
    return (vec2<f32>(w) + 0.5) * texel_m(c);
}

// The water at world XZ: (undisturbed surface height, depth), m. No water off the terrain.
fn water_at(xz: vec2<f32>) -> vec2<f32> {
    let t = grid_taps(params.terrain.grid, xz);
    if !t.inside {
        return vec2<f32>(0.0, 0.0);
    }
    let a = mix(terrain_cells[t.i00].xz, terrain_cells[t.i10].xz, t.w.x);
    let b = mix(terrain_cells[t.i01].xz, terrain_cells[t.i11].xz, t.w.x);
    let c = mix(a, b, t.w.y);
    return vec2<f32>(params.terrain.misc.x + c.x + c.y, c.y);
}

fn velocity_at(xz: vec2<f32>) -> vec2<f32> {
    let t = grid_taps(params.terrain.grid, xz);
    let a = mix(flow_cells[t.i00].xy, flow_cells[t.i10].xy, t.w.x);
    let b = mix(flow_cells[t.i01].xy, flow_cells[t.i11].xy, t.w.x);
    return mix(a, b, t.w.y);
}

// ---------------------------------------------------------------------------------------------
// The bodies' pressure head.

// The vertical chord through a contact's ellipsoid at world XZ: (y_bottom, y_top, 1 when hit).
fn chord(k: ContactGpu, xz: vec2<f32>) -> vec3<f32> {
    let dx = xz.x - k.centre.x;
    let dz = xz.y - k.centre.z;
    let qa = k.m0.w;
    let qb = k.m0.y * dx + k.m1.x * dz;
    let qc = k.m0.x * dx * dx + 2.0 * k.m0.z * dx * dz + k.m1.y * dz * dz - 1.0;
    let disc = qb * qb - qa * qc;
    if disc <= 0.0 || qa <= 0.0 {
        return vec3<f32>(0.0);
    }
    let root = sqrt(disc);
    return vec3<f32>(k.centre.y + (-qb - root) / qa, k.centre.y + (-qb + root) / qa, 1.0);
}

// The head of a chord under a surface at `y_w` for waves of wavenumber `kc` (`model.rs` `head`).
fn head_of(y: vec3<f32>, y_w: f32, kc: f32) -> f32 {
    if y.z < 0.5 {
        return 0.0;
    }
    let upper = max(y_w - y.y, 0.0);
    let lower = max(y_w - y.x, 0.0);
    return (exp(-kc * upper) - exp(-kc * lower)) / kc;
}

@compute @workgroup_size(8, 8, 1)
fn wave_head(@builtin(global_invocation_id) id: vec3<u32>) {
    let contact = id.z / PLANES;
    let plane = id.z % PLANES;
    if contact >= params.counts.y {
        return;
    }
    // The plane's grid (its cascade) and wavenumber (the head's cascade).
    let grid = select(plane - 2u, plane, plane < CASCADES);
    let kc = params.band[select(plane - CASCADES, plane, plane < CASCADES)];
    let k = contacts[contact];
    let t = texel_m(grid);
    let reach = k.centre.w + t;
    let edge = u32(ceil(2.0 * reach / t)) + 1u;
    if any(id.xy >= vec2<u32>(edge)) {
        return;
    }
    let lo = vec2<i32>(floor((k.centre.xz - reach) / t));
    let w = lo + vec2<i32>(id.xy);
    if !in_window(w, origin(grid)) {
        return;
    }
    let water = water_at(centre(grid, w));
    if water.y < SHORE_MIN {
        return;
    }
    var sum = 0.0;
    for (var j = 0; j < SUPER; j += 1) {
        for (var i = 0; i < SUPER; i += 1) {
            let xz = (vec2<f32>(w) + (vec2<f32>(f32(i), f32(j)) + 0.5) / f32(SUPER)) * t;
            let y = chord(k, xz);
            // The static head of its column, and where it cuts the surface the impact head of
            // its entry (`model.rs` `entry_head`).
            sum += head_of(y, water.x, kc) + k.m1.z * y.z;
        }
    }
    let eta = sum / f32(SUPER * SUPER);
    if eta > 0.0 {
        atomicMax(&head[index(plane, w)], bitcast<u32>(eta));
    }
}

// ---------------------------------------------------------------------------------------------
// The step.

// The band's share of the head at world texel `w` of cascade `c`, unsmoothed.
fn band_head(c: u32, w: vec2<i32>) -> f32 {
    if !in_window(w, origin(c)) {
        return 0.0;
    }
    var f = bitcast<f32>(atomicLoad(&head[index(c, w)]));
    if c + 1u < CASCADES {
        // Minus the same head averaged over the next cascade's texels, interpolated here.
        let u = (vec2<f32>(w) + 0.5) / RATIO - 0.5;
        let base = vec2<i32>(floor(u));
        let fr = u - floor(u);
        let o = origin(c + 1u);
        var q = array<f32, 4>(0.0, 0.0, 0.0, 0.0);
        let steps = array<vec2<i32>, 4>(vec2<i32>(0, 0), vec2<i32>(1, 0), vec2<i32>(0, 1), vec2<i32>(1, 1));
        for (var i = 0; i < 4; i += 1) {
            let wq = base + steps[i];
            if in_window(wq, o) {
                q[i] = bitcast<f32>(atomicLoad(&head[index(CASCADES + c, wq)]));
            }
        }
        f -= mix(mix(q[0], q[1], fr.x), mix(q[2], q[3], fr.x), fr.y);
    }
    return f;
}

// Whether any body's head can reach world texel `w` of cascade `c` this frame: within its bound,
// the smoothing's texel and the next cascade's interpolation (two of its texels).
fn near_a_body(c: u32, w: vec2<i32>) -> bool {
    let xz = centre(c, w);
    let margin = 2.0 * texel_m(c) + 2.0 * texel_m(min(c + 1u, CASCADES - 1u));
    for (var i = 0u; i < params.counts.y; i += 1u) {
        let k = contacts[i];
        if distance(xz, k.centre.xz) <= k.centre.w + margin {
            return true;
        }
    }
    return false;
}

// Per cascade texel, once a frame: the forcing, its band's head smoothed by a 3 × 3 binomial
// (zero response at the grid's Nyquist, so a body smaller than a texel moving across them does
// not excite the two-texel mode the kernel was not fitted for), and the water's depth.
@compute @workgroup_size(8, 8, 1)
fn wave_force(@builtin(global_invocation_id) id: vec3<u32>) {
    let c = id.z;
    let n = size();
    if any(id.xy >= vec2<u32>(u32(n))) || c >= CASCADES {
        return;
    }
    let w = origin(c) + vec2<i32>(id.xy);
    var f = 0.0;
    if near_a_body(c, w) {
        let b = array<f32, 3>(0.25, 0.5, 0.25);
        for (var j = 0; j < 3; j += 1) {
            for (var i = 0; i < 3; i += 1) {
                f += b[i] * b[j] * band_head(c, w + vec2<i32>(i - 1, j - 1));
            }
        }
    }
    let xz = centre(c, w);
    let velocity = velocity_at(xz);
    force[index(FORCE_PLANES * c + FORCING, w)] = f;
    force[index(FORCE_PLANES * c + DEPTH, w)] = water_at(xz).y;
    force[index(FORCE_PLANES * c + FLOW_X, w)] = velocity.x;
    force[index(FORCE_PLANES * c + FLOW_Z, w)] = velocity.y;
}

// Plane `p` (`H`, `PREV`, `FOAM`) of world texel `w` of cascade `c`, or zero where it has no
// history (outside last frame's window, or the whole field on a clear).
fn stored(c: u32, p: u32, w: vec2<i32>) -> f32 {
    if params.counts.z != 0u || !in_window(w, prev_origin(c)) || !in_window(w, origin(c)) {
        return 0.0;
    }
    return state_in[index(STATE_PLANES * c + p, w)];
}

// Plane `p` at world texel `w` carried by the current `velocity` (m/s) over one substep: fetched
// upstream, bilinearly (Crest's semi-Lagrangian advection). Below a hundredth of a texel a
// substep the fetch is the texel itself: one load instead of four, and no bilinear blur of the
// shortest waves by a still lake's millimetre-a-second drift. The foam drifts with the waves.
fn advected(c: u32, p: u32, w: vec2<i32>, velocity: vec2<f32>) -> f32 {
    let shift = velocity * params.sim.y / texel_m(c);
    if dot(shift, shift) < 1.0e-4 {
        return stored(c, p, w);
    }
    let u = vec2<f32>(w) - shift;
    let base = vec2<i32>(floor(u));
    let fr = u - floor(u);
    let s00 = stored(c, p, base);
    let s10 = stored(c, p, base + vec2<i32>(1, 0));
    let s01 = stored(c, p, base + vec2<i32>(0, 1));
    let s11 = stored(c, p, base + vec2<i32>(1, 1));
    return mix(mix(s00, s10, fr.x), mix(s01, s11, fr.x), fr.y);
}

fn velocity_of(c: u32, w: vec2<i32>) -> vec2<f32> {
    return vec2<f32>(
        force[index(FORCE_PLANES * c + FLOW_X, w)],
        force[index(FORCE_PLANES * c + FLOW_Z, w)],
    );
}

// One output texel of a step: `l` its place in the tile, `at` its slot offset from the origin.
fn step_texel(c: u32, l: vec2<i32>, at: vec2<i32>) {
    let n = size();
    let w = origin(c) + at;
    let depth = force[index(FORCE_PLANES * c + DEPTH, w)];
    // Dry ground holds no waves: the shore mask would zero it anyway.
    if depth < SHORE_MIN {
        state_out[index(STATE_PLANES * c + H, w)] = 0.0;
        state_out[index(STATE_PLANES * c + PREV, w)] = 0.0;
        state_out[index(STATE_PLANES * c + FOAM, w)] = 0.0;
        return;
    }
    let g0 = tile[l.y * SPAN + l.x];
    // Σ w (g(x + offset) − g(x)) over every image of every offset but the centre, unrolled: the
    // offsets are constants, so no per-invocation table and no branches.
    var conv = 0.0;
    conv += kw(c, 1u) * (axis(l, 1) - 4.0 * g0);
    conv += kw(c, 2u) * (diag(l, 1) - 4.0 * g0);
    conv += kw(c, 3u) * (axis(l, 2) - 4.0 * g0);
    conv += kw(c, 4u) * (skew(l, 2, 1) - 8.0 * g0);
    conv += kw(c, 5u) * (diag(l, 2) - 4.0 * g0);
    conv += kw(c, 6u) * (axis(l, 3) - 4.0 * g0);
    conv += kw(c, 7u) * (skew(l, 3, 1) - 8.0 * g0);
    conv += kw(c, 8u) * (skew(l, 3, 2) - 8.0 * g0);
    conv += kw(c, 9u) * (diag(l, 3) - 4.0 * g0);
    conv += kw(c, 10u) * (axis(l, 4) - 4.0 * g0);
    conv += kw(c, 11u) * (skew(l, 4, 1) - 8.0 * g0);
    conv += kw(c, 12u) * (skew(l, 4, 2) - 8.0 * g0);
    conv += kw(c, 13u) * (skew(l, 4, 3) - 8.0 * g0);
    conv += kw(c, 14u) * (diag(l, 4) - 4.0 * g0);
    conv += kw(c, 15u) * (axis(l, 5) - 4.0 * g0);
    conv += kw(c, 16u) * (skew(l, 5, 1) - 8.0 * g0);
    conv += kw(c, 17u) * (skew(l, 5, 2) - 8.0 * g0);
    conv += kw(c, 18u) * (skew(l, 5, 3) - 8.0 * g0);
    conv += kw(c, 19u) * (skew(l, 5, 4) - 8.0 * g0);
    conv += kw(c, 20u) * (diag(l, 5) - 4.0 * g0);
    conv += kw(c, 21u) * (axis(l, 6) - 4.0 * g0);
    conv += kw(c, 22u) * (skew(l, 6, 1) - 8.0 * g0);
    conv += kw(c, 23u) * (skew(l, 6, 2) - 8.0 * g0);
    conv += kw(c, 24u) * (skew(l, 6, 3) - 8.0 * g0);
    conv += kw(c, 25u) * (skew(l, 6, 4) - 8.0 * g0);
    conv += kw(c, 26u) * (skew(l, 6, 5) - 8.0 * g0);
    conv += kw(c, 27u) * (diag(l, 6) - 4.0 * g0);
    let shore = smoothstep(SHORE_MIN, SHORE_FULL, depth);
    let depth_factor = tanh(params.band[c] * depth);
    let velocity = velocity_of(c, w);
    var now = advected(c, H, w, velocity);
    let before = advected(c, PREV, w, velocity);
    let dt = params.sim.y;
    let damp = params.sim.z * dt;
    var next = ((2.0 - damp) * now - before - dt * dt * depth_factor * conv) / (1.0 + damp);
    // Breaking (Jeschke & Wojtan 2017 after Dean & Dalrymple: a wave breaks once its amplitude
    // exceeds 7 % of its wavelength, a slope of `ak` = 0.44): where the surface is steeper, its
    // amplitude is cut back to the limit and the excess turns to foam. Not under a body (nor
    // the texel around it the smoothed forcing reaches), whose own dip is not a wave.
    var foam = advected(c, FOAM, w, velocity) * exp(-dt / FOAM_LIFE_S);
    if abs(force[index(FORCE_PLANES * c + FORCING, w)]) < 1.0e-4 {
        let slope = length(vec2<f32>(
            tile[l.y * SPAN + l.x + 1] - tile[l.y * SPAN + l.x - 1],
            tile[(l.y + 1) * SPAN + l.x] - tile[(l.y - 1) * SPAN + l.x],
        )) / (2.0 * texel_m(c));
        foam = max(foam, smoothstep(FOAM_ONSET * BREAKING_SLOPE, BREAKING_SLOPE, slope));
        if slope > BREAKING_SLOPE {
            let cut = BREAKING_SLOPE / slope;
            next *= cut;
            now *= cut;
        }
    }
    // The window's sponge: distance to the nearest edge, in texels.
    let from_edge = f32(min(min(at.x, at.y), min(n - 1 - at.x, n - 1 - at.y)));
    let keep = shore * mix(SPONGE_FLOOR, 1.0, saturate(from_edge / (SPONGE * f32(n))));
    state_out[index(STATE_PLANES * c + H, w)] = next * keep;
    state_out[index(STATE_PLANES * c + PREV, w)] = now * keep;
    state_out[index(STATE_PLANES * c + FOAM, w)] = foam * shore;
}

// Each workgroup steps a 32 × 32 block of a cascade, every invocation four texels of it, from
// one tile of `h + forcing` with its 6-texel halo (1936 loads for 1024 outputs).
@compute @workgroup_size(16, 16, 1)
fn wave_step(
    @builtin(workgroup_id) group_id: vec3<u32>,
    @builtin(local_invocation_index) lane: u32,
    @builtin(local_invocation_id) local: vec3<u32>,
) {
    let c = group_id.z;
    let n = size();
    let o = origin(c);
    let corner = o + vec2<i32>(group_id.xy) * TILE - RADIUS;
    // Load `h + forcing` over the tile and its halo: zero outside the window and on dry ground,
    // which costs only its depth (the windows often reach far onto the land).
    for (var i = i32(lane); i < SPAN * SPAN; i += 256) {
        let w = corner + vec2<i32>(i % SPAN, i / SPAN);
        var g = 0.0;
        if in_window(w, o) && force[index(FORCE_PLANES * c + DEPTH, w)] >= SHORE_MIN {
            g = advected(c, H, w, velocity_of(c, w)) + force[index(FORCE_PLANES * c + FORCING, w)];
        }
        tile[i] = g;
    }
    workgroupBarrier();
    // Each invocation takes one texel in each 16 × 16 quarter of the block, so neighbouring
    // invocations read neighbouring tile entries (no bank conflicts).
    for (var k = 0; k < 4; k += 1) {
        let inner = vec2<i32>(local.xy) + 16 * vec2<i32>(k & 1, k >> 1u);
        let at = vec2<i32>(group_id.xy) * TILE + inner;
        if all(at < vec2<i32>(n)) {
            step_texel(c, inner + RADIUS, at);
        }
    }
}

// ---------------------------------------------------------------------------------------------
// The published field.

@compute @workgroup_size(8, 8, 1)
fn wave_publish(@builtin(global_invocation_id) id: vec3<u32>) {
    let c = id.z;
    let n = size();
    if any(id.xy >= vec2<u32>(u32(n))) || c >= CASCADES {
        return;
    }
    let o = origin(c);
    let w = o + vec2<i32>(id.xy);
    // The heads are rebuilt every frame: clear this texel's planes on this cascade's grid.
    atomicStore(&head[index(c, w)], 0u);
    if c > 0u {
        atomicStore(&head[index(CASCADES + c - 1u, w)], 0u);
    }
    // Dry ground holds no waves (the step zeroed it): skip its neighbours.
    if force[index(FORCE_PLANES * c + DEPTH, w)] < SHORE_MIN {
        textureStore(field, slot(w), i32(c), vec4<f32>(0.0));
        return;
    }
    let h = state_out[index(STATE_PLANES * c + H, w)];
    let foam = state_out[index(STATE_PLANES * c + FOAM, w)];
    // The four neighbours: the slope, and the foam through a five-point binomial (a breaking
    // texel is a texel-sized square, which the water's lace threshold would sharpen into a block).
    var eh = array<f32, 4>(h, h, h, h);
    var ef = array<f32, 4>(foam, foam, foam, foam);
    let steps = array<vec2<i32>, 4>(vec2<i32>(1, 0), vec2<i32>(-1, 0), vec2<i32>(0, 1), vec2<i32>(0, -1));
    for (var i = 0; i < 4; i += 1) {
        let wn = w + steps[i];
        if in_window(wn, o) {
            eh[i] = state_out[index(STATE_PLANES * c + H, wn)];
            ef[i] = state_out[index(STATE_PLANES * c + FOAM, wn)];
        }
    }
    let grad = vec2<f32>(eh[0] - eh[1], eh[2] - eh[3]) / (2.0 * texel_m(c));
    let smooth_foam = 0.5 * foam + 0.125 * (ef[0] + ef[1] + ef[2] + ef[3]);
    textureStore(field, slot(w), i32(c), vec4<f32>(h, grad.x, grad.y, smooth_foam));
}
