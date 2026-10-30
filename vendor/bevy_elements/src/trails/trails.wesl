// Trails: the camera-following centimetre field of imprints in snow, sand and soil
// (docs/research/trails.md). Four kernels a frame on one layout:
//
// 1. `trail_stamp`: one 16×16 workgroup grid per imprint over its stamp tile; proposes the swept
//    ellipse's carve and the rim around it (atomicMax of their f32 bits).
// 2. `trail_merge`: every texel; resets a texel that entered the window, applies and clears the
//    stamp (both plastic targets), refills trenches by the snow the cover map gained, clamps to
//    the snow; `a` → `b`.
// 3. `trail_blocks`: the largest |h| per 8 × 8 block, the materials' occupancy.
// 4. `trail_relax`: every texel; slump above the material's angle and wind diffusion, `b` → `a`,
//    and publishes (h, occupancy, ∂h/∂x, ∂h/∂z) to the field the materials sample.
// 5. `trail_readback` (on a readback tick, its own layout): the deepest `h` per readback cell, for
//    `Environment::trail_depth`.
//
// The ground is the terrain's and the cover maps', or on a game's own ground
// (docs/research/trails-game-ground.md) one uniform ground over the window, whose contact the
// main world tests: the imprint's sole height is then its height above that ground.
//
// The window is toroidal by world texel: world texel `w` lives at `w mod N`, so it slides without
// copying.

import bevy_elements::grid::grid_taps;
import bevy_elements::terrain::common::TerrainParams;

struct TrailParams {
    terrain: TerrainParams,
    // World texel of the window's minimum corner, this frame and last frame.
    origin: vec2<i32>,
    prev_origin: vec2<i32>,
    // x: texels a side. y: imprints uploaded. z: 1 to clear the whole window. w: tile edge, texels.
    counts: vec4<u32>,
    // x: texel edge, m. y: game time of this step, s. z: wind diffusivity in snow, m²/s.
    // w: in sand, m²/s.
    sim: vec4<f32>,
    // A game's own ground: snow depth (m), snow density (kg/m³), wetness, sediment (m).
    ground: vec4<f32>,
    // x: 1 on a game's own ground. y: texels a side of a readback cell.
    mode: vec4<u32>,
}

struct ImprintGpu {
    // World XZ where it was last carved (xy) and where it is now (zw).
    segment: vec4<f32>,
    // Half extents (m), and (cos, sin) of its local x axis in world XZ.
    shape: vec4<f32>,
    // x: sole height, m. y: pressure, kPa. z: plate width, m. w: area over perimeter, m.
    load: vec4<f32>,
    // x: ring width, m. yz: world texel of the tile's minimum corner. w: unused.
    tile: vec4<f32>,
}

@group(0) @binding(0) var<uniform> params: TrailParams;
@group(0) @binding(1) var<storage, read_write> state_a: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> state_b: array<vec2<f32>>;
// Two words per texel: the deepest carve and the highest rim proposed this frame (f32 bits; both
// non-negative, so their bits order as their values).
@group(0) @binding(3) var<storage, read_write> stamp: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read> imprints: array<ImprintGpu>;
@group(0) @binding(5) var<storage, read> terrain_cells: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read> cover_cells: array<vec4<f32>>;
@group(0) @binding(7) var field: texture_storage_2d<rgba16float, write>;
// The largest |h| per BLOCK × BLOCK block of slots.
@group(0) @binding(8) var<storage, read_write> blocks: array<f32>;
// `trail_readback`'s output: the window origin (texels), cells a side and texels a cell, then
// the deepest `h` per cell, row-major by slot cell.
@group(0) @binding(9) var<storage, read_write> readback: array<f32>;

// Texels a side of an occupancy block.
const BLOCK: i32 = 8;

// Bekker / Wong (NASA TM-20250006958, Table 1): n, k_c (kN/m^(n+1)), k_φ (kN/m^(n+2)).
const SNOW: vec3<f32> = vec3<f32>(1.6, 4.37, 196.72);
const DRY_SAND: vec3<f32> = vec3<f32>(1.1, 0.99, 1528.43);
const DRY_SOIL: vec3<f32> = vec3<f32>(0.7, 5.27, 1515.04);
const MUD: vec3<f32> = vec3<f32>(0.5, 0.77, 51.91);
const COMPACTED_SNOW_DENSITY: f32 = 500.0;
const LOOSE_LAYER_M: f32 = 0.30;
const SNOW_GROUND_M: f32 = 0.01;
const RIM_SNOW: f32 = 0.1;
const RIM_SAND: f32 = 1.0;
const RIM_SOIL: f32 = 0.7;
const SLUMP_DRY_SAND_DEG: f32 = 34.0;
const SLUMP_SATURATED_SAND_DEG: f32 = 10.0;
const SLUMP_MUD_DEG: f32 = 63.0;
const SLUMP_NONE_DEG: f32 = 90.0;
const CONTACT_M: f32 = 0.1;
// Share of a slope's excess that moves per step (Sumner's roughness σ).
const SLUMP_SHARE: f32 = 0.2;
// The highest rim, m: the parallax march starts 0.1 m above the ground (`TRAIL_TOP_M`).
const RIM_MAX_M: f32 = 0.1;

struct Ground {
    snow_m: f32,
    snow_density: f32,
    sediment_m: f32,
    slope_deg: f32,
    wetness: f32,
}

fn texel_m() -> f32 {
    return params.sim.x;
}

fn size() -> i32 {
    return i32(params.counts.x);
}

// Storage index of world texel `w` (toroidal).
fn slot(w: vec2<i32>) -> vec2<i32> {
    let n = size();
    return ((w % n) + n) % n;
}

fn index(w: vec2<i32>) -> u32 {
    let s = slot(w);
    return u32(s.y * size() + s.x);
}

// The world texel stored at slot `s` this frame.
fn world_of(s: vec2<i32>) -> vec2<i32> {
    let n = size();
    let o = params.origin;
    return o + (((s - o) % n) + n) % n;
}

fn in_window(w: vec2<i32>, origin: vec2<i32>) -> bool {
    let r = w - origin;
    return all(r >= vec2<i32>(0)) && all(r < vec2<i32>(size()));
}

fn centre(w: vec2<i32>) -> vec2<f32> {
    return (vec2<f32>(w) + 0.5) * texel_m();
}

fn game_ground() -> bool {
    return params.mode.x != 0u;
}

fn bekker(k: vec3<f32>, p_kpa: f32, b: f32) -> f32 {
    return pow(max(p_kpa, 0.0) / (k.y / max(b, 1e-3) + k.z), 1.0 / k.x);
}

// The ground at world XZ from the terrain and cover maps (bilinear), or a game's uniform one.
fn ground_at(xz: vec2<f32>) -> Ground {
    if game_ground() {
        var flat: Ground;
        flat.snow_m = max(params.ground.x, 0.0);
        flat.snow_density = clamp(params.ground.y, 50.0, 917.0);
        flat.wetness = clamp(params.ground.z, 0.0, 1.0);
        flat.sediment_m = params.ground.w;
        flat.slope_deg = 0.0;
        return flat;
    }
    let t = grid_taps(params.terrain.grid, xz);
    let c00 = terrain_cells[t.i00];
    let c10 = terrain_cells[t.i10];
    let c01 = terrain_cells[t.i01];
    let c11 = terrain_cells[t.i11];
    let k00 = cover_cells[t.i00];
    let k10 = cover_cells[t.i10];
    let k01 = cover_cells[t.i01];
    let k11 = cover_cells[t.i11];
    let cover = mix(mix(k00, k10, t.w.x), mix(k01, k11, t.w.x), t.w.y);
    let sediment = mix(mix(c00.y, c10.y, t.w.x), mix(c01.y, c11.y, t.w.x), t.w.y);
    let g = mix(
        mix(unpack2x16float(bitcast<u32>(c00.w)), unpack2x16float(bitcast<u32>(c10.w)), t.w.x),
        mix(unpack2x16float(bitcast<u32>(c01.w)), unpack2x16float(bitcast<u32>(c11.w)), t.w.x),
        t.w.y,
    );
    var out: Ground;
    out.snow_m = max(cover.x, 0.0);
    out.snow_density = clamp(1000.0 * cover.z / max(cover.x, 1e-4), 50.0, 917.0);
    out.sediment_m = sediment;
    out.slope_deg = degrees(atan(length(g)));
    out.wetness = clamp(cover.y, 0.0, 1.0);
    return out;
}

// The undisturbed surface's world height at XZ: terrain plus snow. On a game's own ground the
// sole comes as its height above that surface, so the surface is 0.
fn surface_y(xz: vec2<f32>) -> f32 {
    if game_ground() {
        return 0.0;
    }
    let t = grid_taps(params.terrain.grid, xz);
    let h = mix(mix(terrain_cells[t.i00].x, terrain_cells[t.i10].x, t.w.x), mix(terrain_cells[t.i01].x, terrain_cells[t.i11].x, t.w.x), t.w.y);
    let s = mix(mix(cover_cells[t.i00].x, cover_cells[t.i10].x, t.w.x), mix(cover_cells[t.i01].x, cover_cells[t.i11].x, t.w.x), t.w.y);
    return params.terrain.misc.x + h + max(s, 0.0);
}

fn is_snow(g: Ground) -> bool {
    return g.snow_m > SNOW_GROUND_M;
}

fn rock(g: Ground) -> f32 {
    return smoothstep(35.0, 45.0, g.slope_deg);
}

fn sand(g: Ground) -> f32 {
    return smoothstep(0.3, 0.7, g.sediment_m) * (1.0 - smoothstep(26.0, 30.0, g.slope_deg)) * (1.0 - rock(g));
}

// Target depth (m) under `p` kPa on a plate `b` m wide (`model.rs` `Ground::sinkage`).
fn sinkage(g: Ground, p: f32, b: f32) -> f32 {
    if is_snow(g) {
        let compaction = g.snow_m * (1.0 - g.snow_density / COMPACTED_SNOW_DENSITY);
        return min(min(bekker(SNOW, p, b), max(compaction, 0.0)), LOOSE_LAYER_M);
    }
    let dry = max(min(bekker(DRY_SOIL, p, b), LOOSE_LAYER_M), 1e-6);
    let wet = max(min(bekker(MUD, p, b), LOOSE_LAYER_M), 1e-6);
    let soil = exp(mix(log(dry), log(wet), g.wetness));
    let s = sand(g);
    let z = s * min(bekker(DRY_SAND, p, b), LOOSE_LAYER_M) + (1.0 - s) * soil;
    return z * (1.0 - rock(g));
}

fn rim_share(g: Ground) -> f32 {
    if is_snow(g) {
        return RIM_SNOW;
    }
    let s = sand(g);
    return s * RIM_SAND + (1.0 - s) * RIM_SOIL;
}

fn slump_tan(g: Ground) -> f32 {
    if is_snow(g) {
        return 1.0e6;
    }
    let w = g.wetness;
    let damp = smoothstep(0.05, 0.2, w);
    let saturated = smoothstep(0.8, 1.0, w);
    let sand_deg = SLUMP_DRY_SAND_DEG + (SLUMP_NONE_DEG - SLUMP_DRY_SAND_DEG) * damp
        + (SLUMP_SATURATED_SAND_DEG - SLUMP_NONE_DEG) * saturated;
    let soil_deg = SLUMP_NONE_DEG + (SLUMP_MUD_DEG - SLUMP_NONE_DEG) * smoothstep(0.4, 0.8, w);
    let s = sand(g);
    let deg = s * sand_deg + (1.0 - s) * soil_deg;
    return select(tan(radians(deg)), 1.0e6, deg >= 89.0);
}

@compute @workgroup_size(16, 16, 1)
fn trail_stamp(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.z >= params.counts.y || any(id.xy >= vec2<u32>(params.counts.w)) {
        return;
    }
    let im = imprints[id.z];
    let w = vec2<i32>(im.tile.yz) + vec2<i32>(id.xy);
    if !in_window(w, params.origin) {
        return;
    }
    // In contact only while the sole is at most CONTACT_M above the undisturbed surface.
    if im.load.x > surface_y(im.segment.zw) + CONTACT_M {
        return;
    }
    let xz = centre(w);
    let a = im.shape.x;
    let b = im.shape.y;
    let c = im.shape.z;
    let s = im.shape.w;
    // Into the ellipse's frame, scaled to a unit circle: the swept ellipse is then the set
    // within 1 of the scaled segment.
    let to_local = mat2x2<f32>(vec2<f32>(c / a, -s / b), vec2<f32>(s / a, c / b));
    let p = to_local * xz;
    let s0 = to_local * im.segment.xy;
    let s1 = to_local * im.segment.zw;
    let d = s1 - s0;
    let k = select(clamp(dot(p - s0, d) / dot(d, d), 0.0, 1.0), 0.0, dot(d, d) < 1e-12);
    let off = p - (s0 + k * d);
    let r = length(off);
    let g = ground_at(xz);
    let i = index(w);
    let want = sinkage(g, im.load.y, im.load.z);
    if r <= 1.0 {
        // A flat floor inside a wall two and a half texels wide (narrower, the bilinear field
        // steps along a curved edge), at most half the ellipse's smaller half extent so a narrow
        // sole keeps its floor.
        let edge = min(2.5 * texel_m(), 0.5 * min(a, b)) / min(a, b);
        let z = want * (1.0 - smoothstep(1.0 - edge, 1.0, r));
        if z > 0.0 {
            atomicMax(&stamp[2u * i], bitcast<u32>(z));
        }
        return;
    }
    // The rim: outside the edge by `out` metres, within the ring.
    let dir = off / r;
    // The direction back in world units: the unit-circle direction scaled by (a, b), rotated.
    let local_m = dir * vec2<f32>(a, b);
    let radius = length(local_m);
    let out = (r - 1.0) * radius;
    // The displaced material spreads about as far as the carve is deep (and at least the
    // imprint's own ring), so a deep rut's rim is wide and low, not a fin.
    let ring = max(im.tile.x, want);
    if out > ring {
        return;
    }
    // The rim this imprint's carve raises here: the displaced share of a carve of `want` per
    // unit length of edge (area over perimeter), spread over the ring as a half sine (zero at the
    // wall and at the ring's edge, unit area), no higher than the parallax march looks above the
    // ground. A crest at the wall itself is a crease the field aliases along a diagonal rut.
    // Around a curved edge the ring's area grows as 1 + κ·out (κ the edge's curvature: the
    // ellipse's at a cap, zero along the straight sides of a sweep), so the height falls by it
    // and the rim keeps the displaced volume however wide the ring.
    let q = dir * vec2<f32>(b, a);
    let kappa = select(a * b / pow(dot(q, q), 1.5), 0.0, k > 0.0 && k < 1.0);
    let profile = 1.5707964 / ring * sin(3.1415927 * out / ring) / (1.0 + kappa * out);
    let rim = min(rim_share(g) * want * im.load.w * profile, RIM_MAX_M);
    if rim > 0.0 {
        atomicMax(&stamp[2u * i + 1u], bitcast<u32>(rim));
    }
}

@compute @workgroup_size(8, 8, 1)
fn trail_merge(@builtin(global_invocation_id) id: vec3<u32>) {
    if any(id.xy >= vec2<u32>(params.counts.x)) {
        return;
    }
    let s_slot = vec2<i32>(id.xy);
    let w = world_of(s_slot);
    let i = u32(s_slot.y * size() + s_slot.x);
    var s = state_a[i];
    if params.counts.z != 0u || !in_window(w, params.prev_origin) {
        s = vec2<f32>(0.0, 0.0);
    }
    let carve = bitcast<f32>(atomicExchange(&stamp[2u * i], 0u));
    let rim = bitcast<f32>(atomicExchange(&stamp[2u * i + 1u], 0u));
    // Undisturbed ground that nothing presses stays as it is, and its snow reference is not
    // needed until something does (most of the window, most frames: no ground lookup).
    let undisturbed = s.x == 0.0;
    if undisturbed && carve == 0.0 && rim == 0.0 {
        state_b[i] = s;
        return;
    }
    let g = ground_at(centre(w));
    if undisturbed {
        s.y = g.snow_m;
    }
    // Both are plastic targets: the carve lowers the surface to it, the rim raises it to it;
    // pressing again where it already is changes nothing. A rim never fills a trench: a moving
    // imprint's ring lies over what it carved a moment ago.
    if carve > 0.0 {
        s.x = min(s.x, -carve);
    } else if rim > 0.0 && s.x >= 0.0 {
        s.x = max(s.x, rim);
    }
    // New snow fills a trench by what the cover map gained; the flat and the rims rise with the
    // snow surface itself.
    let grow = g.snow_m - s.y;
    if grow > 0.0 && s.x < 0.0 {
        s.x = min(s.x + grow, 0.0);
    }
    s.y = g.snow_m;
    if is_snow(g) {
        s.x = clamp(s.x, -g.snow_m, 0.5 * g.snow_m);
    }
    state_b[i] = s;
}

@compute @workgroup_size(8, 8, 1)
fn trail_relax(@builtin(global_invocation_id) id: vec3<u32>) {
    if any(id.xy >= vec2<u32>(params.counts.x)) {
        return;
    }
    let s_slot = vec2<i32>(id.xy);
    let w = world_of(s_slot);
    let i = u32(s_slot.y * size() + s_slot.x);
    let s = state_b[i];
    // Occupancy for the materials: the largest |h| over this texel's block and its eight
    // neighbours (toroidal), so a fragment can skip the parallax where there is nothing.
    let nb = size() / BLOCK;
    let b = s_slot / BLOCK;
    var occupied = 0.0;
    for (var dy = -1; dy <= 1; dy += 1) {
        for (var dx = -1; dx <= 1; dx += 1) {
            let q = (((b + vec2<i32>(dx, dy)) % nb) + nb) % nb;
            occupied = max(occupied, blocks[u32(q.y * nb + q.x)]);
        }
    }
    // Nothing within a block of here: this texel and its neighbours are flat, so is the result.
    if occupied == 0.0 {
        state_a[i] = s;
        textureStore(field, s_slot, vec4<f32>(0.0));
        return;
    }
    let g = ground_at(centre(w));
    let h = s.x;
    let c = texel_m();
    let wall = c * slump_tan(g);
    let snow = select(0.0, 1.0, is_snow(g));
    let diffusivity = mix(sand(g) * params.sim.w, params.sim.z, snow);
    let kappa = min(diffusivity * params.sim.y / (c * c), 0.2);
    var change = 0.0;
    // Slump toward all eight neighbours (the diagonals a √2 longer run, half the share): with the
    // four axial ones only, a wall at an angle to the grid slumps in steps, a notch wherever its
    // edge crosses a texel row.
    var hn = array<f32, 8>(h, h, h, h, h, h, h, h);
    let steps = array<vec2<i32>, 8>(
        vec2<i32>(1, 0), vec2<i32>(-1, 0), vec2<i32>(0, 1), vec2<i32>(0, -1),
        vec2<i32>(1, 1), vec2<i32>(-1, 1), vec2<i32>(1, -1), vec2<i32>(-1, -1),
    );
    for (var k = 0; k < 8; k += 1) {
        let wn = w + steps[k];
        if !in_window(wn, params.origin) {
            continue;
        }
        let hj = state_b[index(wn)].x;
        hn[k] = hj;
        let dh = hj - h;
        if k < 4 {
            change += 0.5 * SLUMP_SHARE * sign(dh) * max(abs(dh) - wall, 0.0);
            change += kappa * dh;
        } else {
            change += 0.25 * SLUMP_SHARE * sign(dh) * max(abs(dh) - 1.4142135 * wall, 0.0);
        }
    }
    let next = h + change;
    state_a[i] = vec2<f32>(next, s.y);
    // The Sobel gradient (the four diagonals too): smoother along a wall that runs at an angle to
    // the texel grid than central differences, whose normals step at every texel.
    let grad = vec2<f32>(
        (hn[4] + 2.0 * hn[0] + hn[6]) - (hn[5] + 2.0 * hn[1] + hn[7]),
        (hn[4] + 2.0 * hn[2] + hn[5]) - (hn[6] + 2.0 * hn[3] + hn[7]),
    ) / (8.0 * c);
    textureStore(field, s_slot, vec4<f32>(next, occupied, grad.x, grad.y));
}

// The largest |h| over each BLOCK × BLOCK block of slots, from the merged state.
@compute @workgroup_size(8, 8, 1)
fn trail_blocks(@builtin(global_invocation_id) id: vec3<u32>) {
    let nb = size() / BLOCK;
    let b = vec2<i32>(id.xy);
    if any(b >= vec2<i32>(nb)) {
        return;
    }
    var m = 0.0;
    for (var y = 0; y < BLOCK; y += 1) {
        for (var x = 0; x < BLOCK; x += 1) {
            let s = b * BLOCK + vec2<i32>(x, y);
            m = max(m, abs(state_b[u32(s.y * size() + s.x)].x));
        }
    }
    blocks[u32(b.y * nb + b.x)] = m;
}

// The deepest h per readback cell of `mode.y` × `mode.y` slots (a world-aligned block: a whole
// number of cells fits the window), after relax.
@compute @workgroup_size(8, 8, 1)
fn trail_readback(@builtin(global_invocation_id) id: vec3<u32>) {
    let k = i32(max(params.mode.y, 1u));
    let cells = size() / k;
    let c = vec2<i32>(id.xy);
    if any(c >= vec2<i32>(cells)) {
        return;
    }
    if all(c == vec2<i32>(0)) {
        // Exact as f32 below 2^24 texels (840 km at 5 cm).
        readback[0] = f32(params.origin.x);
        readback[1] = f32(params.origin.y);
        readback[2] = f32(cells);
        readback[3] = f32(k);
    }
    var deepest = 0.0;
    for (var y = 0; y < k; y += 1) {
        for (var x = 0; x < k; x += 1) {
            let s = c * k + vec2<i32>(x, y);
            deepest = min(deepest, state_a[u32(s.y * size() + s.x)].x);
        }
    }
    readback[4u + u32(c.y * cells + c.x)] = deepest;
}
