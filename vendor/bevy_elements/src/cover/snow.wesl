// The snow and wetness kernels, one dispatch each per snow step (docs/research/snow-cover.md,
// "GPU mapping"; docs/research/wetness.md):
//
//   snow_source      accumulate, densify, melt and release to hydrology's water layer
//   snow_drift_flux  threshold-cubic drift fluxes by donor cell, limited by the cell's supply
//   snow_drift_apply the divergence: each cell gains what its upwind neighbours sent
//   snow_diffuse_a   Festenberg & Gumhold smoothing, snow → drift
//   snow_diffuse_b   the second pass, drift → snow
//   wetness_update   rain wets, evaporation dries (one update for the whole frame)
//   cover_pack       (depth, wetness, SWE) for the renderer and the readback
//
// `snow` is (depth m, SWE m, pending melt m, —) per cell. Every binding is read_write so the
// entry points share one layout; each kernel reads other cells only from buffers it does not
// write in that dispatch (drift_apply reads `drift` and writes `snow`; diffuse alternates).

import bevy_elements::climate::snow::snow_fraction_at;
import bevy_elements::cover::common::{CoverParams, local_temperature, slope_tangent};

@group(0) @binding(0) var<uniform> params: CoverParams;
@group(0) @binding(1) var<storage, read_write> snow: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> drift: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> wetness: array<f32>;
// (rain exposure, snow exposure) per cell: 0 under a blocker.
@group(0) @binding(4) var<storage, read_write> shelter: array<vec2<f32>>;
@group(0) @binding(5) var<storage, read_write> cells: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read_write> water: array<f32>;
@group(0) @binding(7) var<storage, read_write> cover_cells: array<vec4<f32>>;

// A pending melt below this stays in the snow until it grows past it or the cell empties: an f32
// water depth cannot hold a smaller increment (hydrology.md, Deviation 21).
const RELEASE_QUANTUM_M: f32 = 5e-6;
// Snow depth at which the ground is fully covered (m).
const FULL_COVER_M: f32 = 0.05;
// Diffusion weight D Δt / Δx² of one pass at the reference step, and that step (s).
const DIFFUSE_LAMBDA: f32 = 0.125;
const DIFFUSE_REFERENCE_STEP_S: f32 = 600.0;
// Terrain-scale surface wind (docs/research/dust.md, "Surface wind"; Paris et al. 2019 §4.2-4.3).
const VENTURI_K: f32 = 0.005;
const VENTURI_RING_M: f32 = 50.0;
const SHADOW_RANGE_M: f32 = 10.0;
const SHADOW_MIN_RAD: f32 = 0.17453293; // 10°
const SHADOW_FULL_RAD: f32 = 0.26179939; // 15°
const MAX_SHADOW_TAPS: i32 = 8;

fn in_grid(c: vec2<i32>) -> bool {
    let n = vec2<i32>(params.grid.size);
    return c.x >= 0 && c.y >= 0 && c.x < n.x && c.y < n.y;
}

fn cell_index(c: vec2<i32>) -> u32 {
    return u32(c.y) * params.grid.size.x + u32(c.x);
}

fn clamped_index(c: vec2<i32>) -> u32 {
    let m = vec2<i32>(params.grid.size) - 1;
    return cell_index(clamp(c, vec2<i32>(0), m));
}

// Surface height (m above y0) of the cell nearest world offset `p` (in cells, from cell c).
fn surface_near(c: vec2<i32>, offset_cells: vec2<f32>) -> f32 {
    return cells[clamped_index(c + vec2<i32>(round(offset_cells)))].x;
}

@compute @workgroup_size(8, 8, 1)
fn snow_source(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = params.grid.size;
    if id.x >= size.x || id.y >= size.y {
        return;
    }
    let k = id.y * size.x + id.x;
    let cell = cells[k];
    let t_loc = local_temperature(params, cell.x);
    let alpha = atan(slope_tangent(cell.w));
    let exposure = (1.0 - smoothstep(params.shed.x, params.shed.y, alpha)) * shelter[k].y;
    let dt_h = params.sim.x / 3600.0;

    var s = snow[k];
    // Accumulate at the cell's own rain/snow split.
    let swe_in = 1e-3 * params.sim.y * snow_fraction_at(t_loc) * exposure * dt_h;
    s.x += swe_in * 1000.0 / params.snow.x;
    s.y += swe_in;
    if s.y > 0.0 {
        // Bulk density relaxes toward the settled value, never loosening.
        var rho = 1000.0 * s.y / max(s.x, 1e-9);
        rho += max(params.snow.y - rho, 0.0) * (1.0 - exp(-dt_h / params.snow.z));
        // Degree-day melt (Hock 2003), the density unchanged.
        let melt = min(s.y, params.snow.w * 1e-3 * max(t_loc, 0.0) * dt_h / 24.0);
        s.y -= melt;
        s.z += melt;
        s.x = 1000.0 * s.y / rho;
    }
    if s.y <= 0.0 {
        s.x = 0.0;
        s.y = 0.0;
    }
    // Hand the melt to hydrology in quanta, flushing the remainder when the cell empties.
    if s.z >= RELEASE_QUANTUM_M || (s.y <= 0.0 && s.z > 0.0) {
        water[k] += s.z;
        s.z = 0.0;
    }
    snow[k] = s;
}

// The surface wind (m/s) at a cell: the 10 m wind times the Venturi speed-up over the local
// relief and the 10°–15° lee shadow of the upwind surface.
fn surface_wind(c: vec2<i32>, z: f32) -> f32 {
    let u10 = params.terrain.z;
    let cell = params.grid.cell;
    let dir = params.wind.xy;
    // Venturi: height above the mean of 8 taps on a ring of 50 m.
    let r = VENTURI_RING_M / cell;
    var ring = 0.0;
    for (var i = 0; i < 8; i += 1) {
        let a = 0.78539816 * f32(i);
        ring += surface_near(c, r * vec2<f32>(cos(a), sin(a)));
    }
    let speedup = clamp(1.0 + VENTURI_K * (z - ring * 0.125), 0.5, 2.0);
    // Shadow: the steepest rise toward the wind within 10 m.
    let taps = clamp(i32(floor(SHADOW_RANGE_M / cell)), 1, MAX_SHADOW_TAPS);
    var beta = -1.5;
    for (var i = 1; i <= taps; i += 1) {
        let d = f32(i);
        let zq = surface_near(c, -dir * d);
        beta = max(beta, atan((zq - z) / (d * cell)));
    }
    let sigma = clamp((SHADOW_FULL_RAD - beta) / (SHADOW_FULL_RAD - SHADOW_MIN_RAD), 0.0, 1.0);
    return u10 * speedup * sigma;
}

@compute @workgroup_size(8, 8, 1)
fn snow_drift_flux(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = params.grid.size;
    if id.x >= size.x || id.y >= size.y {
        return;
    }
    let c = vec2<i32>(id.xy);
    let k = id.y * size.x + id.x;
    let s = snow[k];
    var out = vec4<f32>(0.0);
    if s.y > 0.0 {
        let u = surface_wind(c, cells[k].x);
        let excess = max(u - params.shed.z, 0.0);
        if excess > 0.0 {
            let q = params.shed.w * excess * excess * excess;
            let dir = params.wind.xy;
            let sgn = select(vec2<i32>(-1), vec2<i32>(1), dir >= vec2<f32>(0.0));
            let dt_over_cell = params.sim.x / params.grid.cell;
            var o = abs(dir) * (q * dt_over_cell);
            // No export through the edge of the grid.
            if !in_grid(c + vec2<i32>(sgn.x, 0)) {
                o.x = 0.0;
            }
            if !in_grid(c + vec2<i32>(0, sgn.y)) {
                o.y = 0.0;
            }
            let total = o.x + o.y;
            if total > s.y {
                o = o * (s.y / total);
            }
            let depth_per_swe = s.x / s.y;
            out = vec4<f32>(o, o * depth_per_swe);
        }
    }
    drift[k] = out;
}

@compute @workgroup_size(8, 8, 1)
fn snow_drift_apply(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = params.grid.size;
    if id.x >= size.x || id.y >= size.y {
        return;
    }
    let c = vec2<i32>(id.xy);
    let k = id.y * size.x + id.x;
    let sgn = select(vec2<i32>(-1), vec2<i32>(1), params.wind.xy >= vec2<f32>(0.0));
    let own = drift[k];
    var swe = snow[k].y - own.x - own.y;
    var depth = snow[k].x - own.z - own.w;
    // Closed edges: nothing enters from outside the grid, nothing leaves through the downwind
    // edge (`snow_drift_flux`), so drift conserves snow exactly.
    let from_x = c - vec2<i32>(sgn.x, 0);
    if in_grid(from_x) {
        let d = drift[cell_index(from_x)];
        swe += d.x;
        depth += d.z;
    }
    let from_z = c - vec2<i32>(0, sgn.y);
    if in_grid(from_z) {
        let d = drift[cell_index(from_z)];
        swe += d.y;
        depth += d.w;
    }
    var s = snow[k];
    if swe <= 0.0 {
        swe = 0.0;
        depth = 0.0;
    }
    s.x = max(depth, 0.0);
    s.y = swe;
    snow[k] = s;
}

// Diffusion weight of one pass at this step's length: `DIFFUSE_LAMBDA` at the longest step (600 s),
// proportionally less for shorter ones, so the smoothing is a rate in game time and does not
// depend on how many steps a second the frame rate allows.
fn diffuse_lambda() -> f32 {
    return DIFFUSE_LAMBDA * min(params.sim.x / DIFFUSE_REFERENCE_STEP_S, 1.0);
}

fn relax(own: vec4<f32>, n: vec4<f32>, e: vec4<f32>, s: vec4<f32>, w: vec4<f32>) -> vec4<f32> {
    let lambda = diffuse_lambda();
    var out = own;
    out.x = own.x + lambda * (n.x + e.x + s.x + w.x - 4.0 * own.x);
    out.y = own.y + lambda * (n.y + e.y + s.y + w.y - 4.0 * own.y);
    if out.y <= 0.0 {
        out.x = 0.0;
        out.y = 0.0;
    }
    return out;
}

// Whether snow may be exchanged with the cell: a cell under a blocker is a wall, so the shelter
// stays bare and the snow beside it keeps its edge.
fn open_cell(k: u32) -> bool {
    return shelter[k].y > 0.0;
}

@compute @workgroup_size(8, 8, 1)
fn snow_diffuse_a(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = params.grid.size;
    if id.x >= size.x || id.y >= size.y {
        return;
    }
    let c = vec2<i32>(id.xy);
    let k = cell_index(c);
    let own = snow[k];
    if !open_cell(k) {
        drift[k] = own;
        return;
    }
    // No-flux edges: a missing neighbour is the cell itself.
    var n = own;
    var e = own;
    var s = own;
    var w = own;
    let cn = c + vec2<i32>(0, -1);
    let ce = c + vec2<i32>(1, 0);
    let cs = c + vec2<i32>(0, 1);
    let cw = c + vec2<i32>(-1, 0);
    if in_grid(cn) && open_cell(cell_index(cn)) { n = snow[cell_index(cn)]; }
    if in_grid(ce) && open_cell(cell_index(ce)) { e = snow[cell_index(ce)]; }
    if in_grid(cs) && open_cell(cell_index(cs)) { s = snow[cell_index(cs)]; }
    if in_grid(cw) && open_cell(cell_index(cw)) { w = snow[cell_index(cw)]; }
    drift[k] = relax(own, n, e, s, w);
}

@compute @workgroup_size(8, 8, 1)
fn snow_diffuse_b(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = params.grid.size;
    if id.x >= size.x || id.y >= size.y {
        return;
    }
    let c = vec2<i32>(id.xy);
    let k = cell_index(c);
    let own = drift[k];
    if !open_cell(k) {
        snow[k] = own;
        return;
    }
    var n = own;
    var e = own;
    var s = own;
    var w = own;
    let cn = c + vec2<i32>(0, -1);
    let ce = c + vec2<i32>(1, 0);
    let cs = c + vec2<i32>(0, 1);
    let cw = c + vec2<i32>(-1, 0);
    if in_grid(cn) && open_cell(cell_index(cn)) { n = drift[cell_index(cn)]; }
    if in_grid(ce) && open_cell(cell_index(ce)) { e = drift[cell_index(ce)]; }
    if in_grid(cs) && open_cell(cell_index(cs)) { s = drift[cell_index(cs)]; }
    if in_grid(cw) && open_cell(cell_index(cw)) { w = drift[cell_index(cw)]; }
    snow[k] = relax(own, n, e, s, w);
}

@compute @workgroup_size(8, 8, 1)
fn wetness_update(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = params.grid.size;
    if id.x >= size.x || id.y >= size.y {
        return;
    }
    let k = id.y * size.x + id.x;
    let t_loc = local_temperature(params, cells[k].x);
    let snow_part = snow_fraction_at(t_loc);
    let covered = smoothstep(0.0, FULL_COVER_M, snow[k].x);
    let rain_mm_h = params.sim.y * (1.0 - snow_part) * shelter[k].x * (1.0 - covered);
    // E / E0 with E the hydrology evaporation (hydrology::evaporation_mm_h) and E0 = 0.1 mm/h.
    let evaporation = (1.0 + params.air.x / 5.0) * (1.0 + max(t_loc, 0.0) / 20.0) * max(1.0 - params.wet.z, 0.0);
    let drying = params.wet.y * evaporation * (1.0 + params.wet.w);
    let dt_h = params.terrain.w / 3600.0;
    wetness[k] = clamp(wetness[k] + (params.wet.x * rain_mm_h - drying) * dt_h, 0.0, 1.0);
}

@compute @workgroup_size(8, 8, 1)
fn cover_pack(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = params.grid.size;
    if id.x >= size.x || id.y >= size.y {
        return;
    }
    let k = id.y * size.x + id.x;
    cover_cells[k] = vec4<f32>(snow[k].x, wetness[k], snow[k].y, 0.0);
}
