// Aeolian transport of the loose sand layer (docs/research/dust.md "Model adopted").
//
//   aeolian_capacity  per cell, the flux the surface wind carries at saturation: the terrain-
//                     modified wind (Venturi speed-up over the 50 m ring mean, shadow behind
//                     upwind rises) through k u² (u − u_t)+
//   aeolian_flux      per line of cells along x and along z, one sequential scan from the upwind
//                     edge: the flux relaxes toward the capacity over the saturation length
//                     (dq/ds = (q_s − q) / L_sat) and a cell sends at most what arrives plus
//                     the sand it holds (Deviation 9: a scan, not a persistent Jacobi field)
//   aeolian_move      per cell, the bed change by the divergence of the face fluxes
//
// The wind's flux is split by axis: the face flux through a cell's downwind x face is
// q_x = q |û_x| and through its z face q_z = q |û_z|, and the sand the cell holds is shared between
// the two scans by |û_x| : |û_z|, so no cell can lose more than it holds. All bindings are
// read_write in every entry point so the three share one layout; each dispatch writes a buffer
// that its neighbours do not read (capacity writes `cap`, the scan writes `q_x`/`q_z`, move
// writes only its own `sediment`).

import bevy_elements::grid::GridParams;
import bevy_elements::dust::surface_wind::{
    RING_M, SHADOW_MARCH_M, venturi_speedup, shadow_factor, flux_capacity,
};

struct AeolianParams {
    grid: GridParams,
    // xy: unit wind direction (x, z). z: the 10 m wind, m/s. w: 1 when the boundary is closed.
    wind: vec4<f32>,
    // x: flux coefficient k, m²/s per (m/s)³. y: transport threshold u_t, m/s.
    law: vec4<f32>,
    // x: saturation length L_sat, m. y: the bed's time step, s.
    step: vec4<f32>,
}

@group(0) @binding(0) var<uniform> params: AeolianParams;
@group(0) @binding(1) var<storage, read_write> bedrock: array<f32>;
@group(0) @binding(2) var<storage, read_write> sediment: array<f32>;
@group(0) @binding(3) var<storage, read_write> cap: array<f32>;
@group(0) @binding(4) var<storage, read_write> q_x: array<f32>;
@group(0) @binding(5) var<storage, read_write> q_z: array<f32>;

fn in_grid(c: vec2<i32>) -> bool {
    let n = vec2<i32>(params.grid.size);
    return c.x >= 0 && c.y >= 0 && c.x < n.x && c.y < n.y;
}

fn cell_index(c: vec2<i32>) -> u32 {
    return u32(c.y) * params.grid.size.x + u32(c.x);
}

/// The surface height at a cell, the edge cell's for a cell off the grid.
fn surface_at(c: vec2<i32>) -> f32 {
    let m = vec2<i32>(params.grid.size) - vec2<i32>(1);
    let k = cell_index(clamp(c, vec2<i32>(0), m));
    return bedrock[k] + sediment[k];
}

/// The flux sign of the wind per axis: the cell the sand comes from is `c − s`.
fn wind_sign() -> vec2<i32> {
    let u = params.wind.xy;
    return vec2<i32>(select(-1, 1, u.x >= 0.0), select(-1, 1, u.y >= 0.0));
}

@compute @workgroup_size(8, 8, 1)
fn aeolian_capacity(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = params.grid.size;
    if id.x >= size.x || id.y >= size.y {
        return;
    }
    let c = vec2<i32>(id.xy);
    let cell = params.grid.cell;
    let z = surface_at(c);
    let dir = params.wind.xy;

    // The steepest rise of the surface upwind, over the march radius, one tap per cell.
    let taps = max(i32(floor(SHADOW_MARCH_M / cell + 1e-3)), 1);
    var slope = -1e9;
    for (var i = 1; i <= taps; i += 1) {
        let offset = vec2<i32>(round(dir * f32(i)));
        slope = max(slope, (surface_at(c - offset) - z) / (f32(i) * cell));
    }
    let shadow = shadow_factor(slope);

    // The height above the mean of eight taps on the ring.
    var ring_dirs = array<vec2<f32>, 8>(
        vec2<f32>(1.0, 0.0),
        vec2<f32>(0.70710678, 0.70710678),
        vec2<f32>(0.0, 1.0),
        vec2<f32>(-0.70710678, 0.70710678),
        vec2<f32>(-1.0, 0.0),
        vec2<f32>(-0.70710678, -0.70710678),
        vec2<f32>(0.0, -1.0),
        vec2<f32>(0.70710678, -0.70710678),
    );
    let ring = RING_M / cell;
    var mean = 0.0;
    for (var d = 0u; d < 8u; d += 1u) {
        mean += surface_at(c + vec2<i32>(round(ring_dirs[d] * ring)));
    }
    mean = mean * 0.125;

    let u = params.wind.z * venturi_speedup(z - mean) * shadow;
    cap[cell_index(c)] = flux_capacity(u, params.law.x, params.law.y);
}

/// One scan of a row of cells (along x) from the upwind end: writes the flux through each
/// cell's downwind face into `q_x`. The flux that arrives is what the previous cell sent; nothing
/// enters from the edge. On a closed boundary the last cell sends nothing, so what reaches it
/// stays.
fn scan_x(row: u32) {
    let size = params.grid.size;
    let s = wind_sign();
    let a = abs(params.wind.xy);
    let share = a.x / (a.x + a.y);
    let cell = params.grid.cell;
    let dt = params.step.y;
    let rho = 1.0 - exp(-cell / (params.step.x * max(a.x, 1e-3)));
    var arriving = 0.0;
    for (var m = 0u; m < size.x; m += 1u) {
        let i = select(size.x - 1u - m, m, s.x > 0);
        let k = row * size.x + i;
        var out = 0.0;
        if a.x > 1e-6 {
            let relaxed = arriving + (cap[k] * a.x - arriving) * rho;
            let supply = arriving + sediment[k] * cell / dt * share;
            out = max(min(relaxed, supply), 0.0);
            if params.wind.w > 0.5 && m == size.x - 1u {
                out = 0.0;
            }
        }
        q_x[k] = out;
        arriving = out;
    }
}

/// The same along a column (along z), into `q_z`.
fn scan_z(column: u32) {
    let size = params.grid.size;
    let s = wind_sign();
    let a = abs(params.wind.xy);
    let share = a.y / (a.x + a.y);
    let cell = params.grid.cell;
    let dt = params.step.y;
    let rho = 1.0 - exp(-cell / (params.step.x * max(a.y, 1e-3)));
    var arriving = 0.0;
    for (var m = 0u; m < size.y; m += 1u) {
        let j = select(size.y - 1u - m, m, s.y > 0);
        let k = j * size.x + column;
        var out = 0.0;
        if a.y > 1e-6 {
            let relaxed = arriving + (cap[k] * a.y - arriving) * rho;
            let supply = arriving + sediment[k] * cell / dt * share;
            out = max(min(relaxed, supply), 0.0);
            if params.wind.w > 0.5 && m == size.y - 1u {
                out = 0.0;
            }
        }
        q_z[k] = out;
        arriving = out;
    }
}

// Threads 0 .. size.y scan the rows (along x), the next size.x scan the columns (along z).
@compute @workgroup_size(64, 1, 1)
fn aeolian_flux(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = params.grid.size;
    if id.x < size.y {
        scan_x(id.x);
    } else if id.x < size.y + size.x {
        scan_z(id.x - size.y);
    }
}

@compute @workgroup_size(8, 8, 1)
fn aeolian_move(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = params.grid.size;
    if id.x >= size.x || id.y >= size.y {
        return;
    }
    let c = vec2<i32>(id.xy);
    let k = cell_index(c);
    let s = wind_sign();
    let from_x = c - vec2<i32>(s.x, 0);
    let from_z = c - vec2<i32>(0, s.y);
    var arriving = 0.0;
    if in_grid(from_x) {
        arriving += q_x[cell_index(from_x)];
    }
    if in_grid(from_z) {
        arriving += q_z[cell_index(from_z)];
    }
    let leaving = q_x[k] + q_z[k];
    sediment[k] = max(sediment[k] + params.step.y / params.grid.cell * (arriving - leaving), 0.0);
}
