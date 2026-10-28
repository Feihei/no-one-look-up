// The hydrology and erosion substep: two gather dispatches per substep.
//
//   flux          sources → pipe fluxes (Mei 2007 virtual pipes with the pipe area taken from the
//                 local depth, Manning friction, K scaling), sin α and the sediment ratio s/d₁
//   depth_erode   depth, velocity, sediment transport through the same fluxes, dissolve and
//                 deposit, evaporation
//
// See docs/research/hydrology.md and docs/research/erosion.md. Cell (x, y) is column x along world
// +X and row y along world +Z; the pipes are L (x−1), R (x+1), T (row−1), B (row+1).
//
// All eight bindings are read_write in both entry points so they share one layout; every kernel
// reads other cells only from buffers it does not write in that dispatch:
//   flux          writes flux (own cell) and transport (own cell)
//   depth_erode   writes water, velocity, suspended, bedrock, sediment (own cell)

import bevy_elements::grid::GridParams;
import bevy_elements::climate::snow::snow_fraction_at;

struct HydroParams {
    grid: GridParams,
    // x: Δt (s). y: Manning n. z: pipe depth cap (m). w: wet threshold d_wet (m).
    sim: vec4<f32>,
    // x: rain quantum (m). y: evaporation quantum (m). z: terrain y0 − climate reference elevation
    // (m). w: air temperature at the reference elevation (°C).
    weather: vec4<f32>,
    // x: temperature lapse (°C/m, negative). y: speed clamp (m/s). z: 1 = bake (all rain is liquid).
    misc: vec4<f32>,
    // x: 1 = closed boundary. y: 1 = erosion on. z: sources uploaded. w: blockers uploaded.
    flags: vec4<u32>,
    // x: K_c (s). y: min tilt. z: dissolve fraction per step. w: deposit fraction per step.
    erosion: vec4<f32>,
    // x: tan(talus). y: tan(talus + 10°). z: thermal fraction per pass.
    thermal: vec4<f32>,
    // x: terrain y0 (m). y: depth readback downsample factor.
    extra: vec4<f32>,
}

@group(0) @binding(0) var<uniform> params: HydroParams;
@group(0) @binding(1) var<storage, read_write> bedrock: array<f32>;
@group(0) @binding(2) var<storage, read_write> sediment: array<f32>;
@group(0) @binding(3) var<storage, read_write> water: array<f32>;
@group(0) @binding(4) var<storage, read_write> source_map: array<vec2<f32>>;
@group(0) @binding(5) var<storage, read_write> flux: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read_write> velocity: array<vec2<f32>>;
@group(0) @binding(7) var<storage, read_write> suspended: array<f32>;
@group(0) @binding(8) var<storage, read_write> transport: array<vec2<f32>>;

const GRAVITY: f32 = 9.81;

fn in_grid(c: vec2<i32>) -> bool {
    let n = vec2<i32>(params.grid.size);
    return c.x >= 0 && c.y >= 0 && c.x < n.x && c.y < n.y;
}

fn cell_index(c: vec2<i32>) -> u32 {
    return u32(c.y) * params.grid.size.x + u32(c.x);
}

fn surface_at(k: u32) -> f32 {
    return bedrock[k] + sediment[k];
}

// The water depth after this substep's sources: sources and sinks (m/s of depth), the frame's
// rain quantum split by the cell's own snow fraction and masked under blockers, and the
// evaporation quantum, clamped at dry. Both kernels evaluate this identically, so it is never
// stored.
fn depth_after_sources(k: u32, surface: f32) -> f32 {
    let sm = source_map[k];
    var rain = 0.0;
    if params.weather.x > 0.0 {
        var liquid = 1.0;
        if params.misc.z < 0.5 {
            let t = params.weather.w + params.misc.x * (surface + params.weather.z);
            liquid = 1.0 - snow_fraction_at(t);
        }
        rain = params.weather.x * sm.y * liquid;
    }
    return max(0.0, water[k] + params.sim.x * sm.x + rain - params.weather.y);
}

// The outflow flux of the pipe from the cell (surface `s_i`, depth `d1_i`, speed `speed_i`, old
// flux `f_old`) toward the neighbour at offset `dir`, before the K scaling.
fn pipe_flux(c: vec2<i32>, dir: vec2<i32>, s_i: f32, d1_i: f32, speed_i: f32, f_old: f32) -> f32 {
    let nb = c + dir;
    var s_j = s_i;
    var d1_j = 0.0;
    var speed_j = speed_i;
    if in_grid(nb) {
        let k = cell_index(nb);
        s_j = surface_at(k);
        d1_j = depth_after_sources(k, s_j);
        speed_j = length(velocity[k]);
    } else if params.flags.x == 1u {
        return 0.0;
    }
    // Open edge: a ghost neighbour with the same bed and no water, so the water falls out as a
    // free overfall at the edge cell's own bed level.
    let eta_i = s_i + d1_i;
    let eta_j = s_j + d1_j;
    let h_f = min(max(max(eta_i, eta_j) - max(s_i, s_j), 0.0), params.sim.z);
    if h_f <= 0.0 {
        return 0.0;
    }
    let dt = params.sim.x;
    let n = params.sim.y;
    let v_bar = 0.5 * (speed_i + speed_j);
    let friction = 1.0 + dt * GRAVITY * n * n * v_bar / pow(max(h_f, params.sim.w), 4.0 / 3.0);
    return max(0.0, (f_old + dt * GRAVITY * h_f * (eta_i - eta_j)) / friction);
}

@compute @workgroup_size(8, 8, 1)
fn flux_step(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = params.grid.size;
    if id.x >= size.x || id.y >= size.y {
        return;
    }
    let c = vec2<i32>(id.xy);
    let k = id.y * size.x + id.x;
    let l = params.grid.cell;
    let dt = params.sim.x;
    let s_i = surface_at(k);
    let d1 = depth_after_sources(k, s_i);
    let speed_i = length(velocity[k]);
    let old = flux[k];

    var f = vec4<f32>(
        pipe_flux(c, vec2<i32>(-1, 0), s_i, d1, speed_i, old.x),
        pipe_flux(c, vec2<i32>(1, 0), s_i, d1, speed_i, old.y),
        pipe_flux(c, vec2<i32>(0, -1), s_i, d1, speed_i, old.z),
        pipe_flux(c, vec2<i32>(0, 1), s_i, d1, speed_i, old.w),
    );
    // Mei's scaling: no cell sends out more than it holds.
    let total = f.x + f.y + f.z + f.w;
    if total > 0.0 {
        f = f * min(1.0, d1 * l * l / (dt * total));
    }
    flux[k] = f;

    // sin α from the surface before this step's erosion (central differences, one-sided at the
    // edge, as terrain_pack), and the sediment ratio the transport carries.
    let n = vec2<i32>(size);
    let lo = max(c - 1, vec2<i32>(0));
    let hi = min(c + 1, n - 1);
    let gx = (surface_at(cell_index(vec2<i32>(hi.x, c.y))) - surface_at(cell_index(vec2<i32>(lo.x, c.y)))) / (f32(hi.x - lo.x) * l);
    let gz = (surface_at(cell_index(vec2<i32>(c.x, hi.y))) - surface_at(cell_index(vec2<i32>(c.x, lo.y)))) / (f32(hi.y - lo.y) * l);
    let grad = sqrt(gx * gx + gz * gz);
    let sin_a = grad / sqrt(1.0 + grad * grad);
    var ratio = 0.0;
    if d1 >= params.sim.w {
        ratio = suspended[k] / d1;
    }
    transport[k] = vec2<f32>(sin_a, ratio);
}

// Dissolve toward capacity or deposit, with every transfer the amount actually applied after f32
// rounding. Returns (bedrock, sediment, suspended).
fn exchange(b: f32, sed: f32, s: f32, capacity: f32) -> vec3<f32> {
    var bed = b;
    var layer = sed;
    var load = s;
    if capacity > load {
        let a = params.erosion.z * (capacity - load);
        let from_sediment = min(a, layer);
        let layer_after = layer - from_sediment;
        var applied = layer - layer_after;
        layer = layer_after;
        if from_sediment < a {
            let bed_after = bed - (a - from_sediment);
            applied += bed - bed_after;
            bed = bed_after;
        }
        load += applied;
    } else {
        let a = params.erosion.w * (load - capacity);
        let layer_after = layer + a;
        let applied = layer_after - layer;
        layer = layer_after;
        load -= applied;
    }
    return vec3<f32>(bed, layer, load);
}

@compute @workgroup_size(8, 8, 1)
fn depth_erode(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = params.grid.size;
    if id.x >= size.x || id.y >= size.y {
        return;
    }
    let c = vec2<i32>(id.xy);
    let k = id.y * size.x + id.x;
    let l = params.grid.cell;
    let dt = params.sim.x;
    let b = bedrock[k];
    let sed = sediment[k];
    let d1 = depth_after_sources(k, b + sed);
    let out = flux[k];
    let out_sum = out.x + out.y + out.z + out.w;

    // What the neighbours send in, and the sediment it carries (their ratio c_j times the flux).
    var fin = vec4<f32>(0.0); // from L, R, T, B
    var sediment_in = 0.0;
    let nl = c + vec2<i32>(-1, 0);
    let nr = c + vec2<i32>(1, 0);
    let nt = c + vec2<i32>(0, -1);
    let nb = c + vec2<i32>(0, 1);
    if in_grid(nl) {
        let j = cell_index(nl);
        fin.x = flux[j].y;
        sediment_in += fin.x * transport[j].y;
    }
    if in_grid(nr) {
        let j = cell_index(nr);
        fin.y = flux[j].x;
        sediment_in += fin.y * transport[j].y;
    }
    if in_grid(nt) {
        let j = cell_index(nt);
        fin.z = flux[j].w;
        sediment_in += fin.z * transport[j].y;
    }
    if in_grid(nb) {
        let j = cell_index(nb);
        fin.w = flux[j].z;
        sediment_in += fin.w * transport[j].y;
    }
    let in_sum = fin.x + fin.y + fin.z + fin.w;
    let d2 = max(0.0, d1 + dt / (l * l) * (in_sum - out_sum));

    // Velocity from the flux through the cell (Mei Eq. 8–9), clamped (Chentanez §2.1.5).
    var vel = vec2<f32>(0.0);
    let d_bar = 0.5 * (d1 + d2);
    if d_bar >= params.sim.w {
        vel = 0.5 * vec2<f32>(fin.x - out.x + out.y - fin.y, fin.z - out.z + out.w - fin.w) / (l * d_bar);
        let speed = length(vel);
        if speed > params.misc.y {
            vel = vel * (params.misc.y / speed);
        }
    }

    var bed = b;
    var layer = sed;
    var load = suspended[k];
    if params.flags.y == 1u {
        // Transport in flux form: the sediment moves with exactly the water that moves.
        load = max(0.0, load + dt / (l * l) * (sediment_in - transport[k].y * out_sum));
        if d2 <= params.sim.w {
            // A dry cell has no transport: everything settles.
            let layer_after = layer + load;
            load -= layer_after - layer;
            layer = layer_after;
        } else {
            let capacity = params.erosion.x * max(transport[k].x, params.erosion.y) * length(vel);
            let r = exchange(bed, layer, load, capacity);
            bed = r.x;
            layer = r.y;
            load = r.z;
        }
    }
    water[k] = d2;
    velocity[k] = vel;
    suspended[k] = load;
    bedrock[k] = bed;
    sediment[k] = layer;
}
