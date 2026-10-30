// Precipitation particles: one thread per particle in a cube around the camera. See
// docs/research/precipitation.md ("Motion", "Blockers", "Splashes") and docs/research/dust.md
// ("Particles").
//
//   activate  a parked particle that becomes active is seeded uniformly in the box (dust:
//             exponentially in height above the ground, the haze's profile)
//   velocity  rain: wind + (0, −v(D), 0) with D from the water-weighted Marshall–Palmer
//             distribution; snow: wind + (0, −1, 0) + flutter; dust: wind + (0, −0.05, 0)
//   collide   the step segment against every blocker box and the ground; a hit respawns the
//             particle at the top of the box, and a rain hit near the camera appends a splash.
//             Dust sees no blockers and makes no splash: it respawns at a new place in the box
//   wrap      toroidally around the camera

import bevy_elements::grid::grid_taps;
import bevy_elements::terrain::common::{TerrainParams, terrain_live};
import bevy_elements::wind::{WindFieldParams, elements_wind_metres};
import bevy_elements::precipitation::common::{particle_rand, drop_diameter_mm, drop_fall_speed, dust_height};

struct Particle {
    pos: vec3<f32>,
    // Bits 0–1: kind (0 rain, 1 snow, 2 dust, 3 parked). Bits 2–31: respawn counter.
    kind: u32,
    vel: vec3<f32>,
    // Age since the last respawn, s.
    life: f32,
}

struct Blocker {
    // Rows of the inverse of the box's affine matrix: local = rot · (world − centre).
    rot: array<vec4<f32>, 3>,
    centre: vec3<f32>,
    extent: vec3<f32>,
}

struct SplashRing {
    counter: atomic<u32>,
    pad0: u32,
    pad1: u32,
    pad2: u32,
    ring: array<vec4<f32>, 4096>,
}

struct Params {
    // xyz: followed camera, m. w: Δt, s.
    camera: vec4<f32>,
    // x: shader clock, s. y: box half edge, m. z: splash radius, m. w: ground height, m.
    motion: vec4<f32>,
    // x: rain count. y: snow count. z: capacity. w: blocker count.
    counts: vec4<u32>,
    // x: Marshall–Palmer Λ, mm⁻¹. y: the standing water an impact on a live terrain needs to
    // ring, m (`docs/research/wet-ground.md`, Model adopted 5).
    rain: vec4<f32>,
    // x: flutter speed, m/s. y: snow fall speed, m/s. zw: flutter frequency range, Hz.
    snow: vec4<f32>,
    // x: dust fall speed, m/s. y: dust height scale, m.
    dust: vec4<f32>,
    // x: dust count.
    dust_counts: vec4<u32>,
    wind: WindFieldParams,
    // The terrain grid; without a live terrain the ground is the plane y = motion.w.
    terrain: TerrainParams,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(2) var<storage, read_write> splashes: SplashRing;
@group(0) @binding(3) var<storage, read> blockers: array<Blocker>;
@group(0) @binding(4) var wind_field: texture_3d<f32>;
@group(0) @binding(5) var wind_sampler: sampler;
@group(0) @binding(6) var<storage, read> terrain_cells: array<vec4<f32>>;

const KIND_RAIN: u32 = 0u;
const KIND_SNOW: u32 = 1u;
const KIND_DUST: u32 = 2u;
const KIND_PARKED: u32 = 3u;
const TAU: f32 = 6.2831853;
const TILE: u32 = 64u;
const MISS: f32 = 2.0;
const SPLASH_LIFT_M: f32 = 0.02;

var<workgroup> tile: array<Blocker, 64>;

/// Entry parameter in [0, 1] of the segment p0 → p1 into the blocker box (0 when p0 is already
/// inside), or MISS.
fn segment_enter(b: Blocker, p0: vec3<f32>, p1: vec3<f32>) -> f32 {
    let d0 = p0 - b.centre;
    let d1 = p1 - b.centre;
    let q0 = vec3<f32>(dot(b.rot[0].xyz, d0), dot(b.rot[1].xyz, d0), dot(b.rot[2].xyz, d0));
    let q1 = vec3<f32>(dot(b.rot[0].xyz, d1), dot(b.rot[1].xyz, d1), dot(b.rot[2].xyz, d1));
    let d = q1 - q0;
    var t_in = 0.0;
    var t_out = 1.0;
    for (var a = 0; a < 3; a += 1) {
        if abs(d[a]) < 1e-7 {
            if abs(q0[a]) > b.extent[a] {
                return MISS;
            }
        } else {
            let inv = 1.0 / d[a];
            let t_a = (-b.extent[a] - q0[a]) * inv;
            let t_b = (b.extent[a] - q0[a]) * inv;
            t_in = max(t_in, min(t_a, t_b));
            t_out = min(t_out, max(t_a, t_b));
        }
    }
    return select(MISS, t_in, t_in <= t_out);
}

/// World height of the surface rain lands on at world XZ: the terrain's surface plus the water
/// standing on it (a flood splashes), or the plane y = motion.w without a terrain.
fn ground_at(xz: vec2<f32>) -> f32 {
    if !terrain_live(params.terrain) {
        return params.motion.w;
    }
    let t = grid_taps(params.terrain.grid, xz);
    let a = terrain_cells[t.i00];
    let b = terrain_cells[t.i10];
    let c = terrain_cells[t.i01];
    let d = terrain_cells[t.i11];
    let h = mix(mix(a.x + a.z, b.x + b.z, t.w.x), mix(c.x + c.z, d.x + d.z, t.w.x), t.w.y);
    return params.terrain.misc.x + h;
}

/// Whether a rain impact on the ground at world XZ rings: a ring is a wave on a free water
/// surface, so on a live terrain only where the water standing on it is at least `rain.y` deep;
/// a drop on grass is caught by a leaf and one on soil soaks in (`docs/research/wet-ground.md`,
/// Model adopted 5). The plane of a scene without a terrain is a surface the kernel does not
/// know, and rings as before.
fn ground_rings(xz: vec2<f32>) -> bool {
    if !terrain_live(params.terrain) {
        return true;
    }
    let t = grid_taps(params.terrain.grid, xz);
    let a = terrain_cells[t.i00].z;
    let b = terrain_cells[t.i10].z;
    let c = terrain_cells[t.i01].z;
    let d = terrain_cells[t.i11].z;
    return mix(mix(a, b, t.w.x), mix(c, d, t.w.x), t.w.y) >= params.rain.y;
}

/// A position uniform in the part of the box above the ground.
fn seed_position(index: u32, epoch: u32) -> vec3<f32> {
    let cam = params.camera.xyz;
    let half_edge = params.motion.y;
    let r = particle_rand(index, epoch, 3u);
    let top = cam.y + half_edge;
    let xz = vec2<f32>(
        cam.x + (r.x * 2.0 - 1.0) * half_edge,
        cam.z + (r.z * 2.0 - 1.0) * half_edge,
    );
    let bottom = min(max(ground_at(xz), cam.y - half_edge), top);
    return vec3<f32>(xz.x, mix(bottom, top, r.y), xz.y);
}

/// A dust sprite's position: uniform over the box in the horizontal, exponential in height above
/// the ground with the haze's scale height, truncated to the part of the box above the ground.
fn dust_position(index: u32, epoch: u32) -> vec3<f32> {
    let cam = params.camera.xyz;
    let half_edge = params.motion.y;
    let r = particle_rand(index, epoch, 4u);
    let top = cam.y + half_edge;
    let xz = vec2<f32>(
        cam.x + (r.x * 2.0 - 1.0) * half_edge,
        cam.z + (r.z * 2.0 - 1.0) * half_edge,
    );
    let ground = ground_at(xz);
    let lo = max(ground, cam.y - half_edge) - ground;
    let hi = max(top - ground, lo + 0.01);
    let h = dust_height(r.y, lo, hi, params.dust.y);
    return vec3<f32>(xz.x, min(ground + h, top - 0.001), xz.y);
}

fn wrap_axis(x: f32, centre: f32, half_edge: f32) -> f32 {
    let size = 2.0 * half_edge;
    let shifted = x - centre + half_edge;
    return centre + (shifted - size * floor(shifted / size)) - half_edge;
}

@compute @workgroup_size(64, 1, 1)
fn elements_precip_sim(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_index) local_index: u32,
) {
    let index = gid.x;
    let capacity = params.counts.z;
    let n_blockers = params.counts.w;
    let dt = params.camera.w;
    let cam = params.camera.xyz;
    let half_edge = params.motion.y;

    var kind = KIND_PARKED;
    var epoch = 0u;
    var pos = vec3<f32>(0.0);
    var life = 0.0;
    var vel = vec3<f32>(0.0);
    var live = false;
    if index < capacity {
        let p = particles[index];
        epoch = p.kind >> 2u;
        pos = p.pos;
        life = p.life;
        if index < params.counts.x {
            kind = KIND_RAIN;
        } else if index < params.counts.x + params.counts.y {
            kind = KIND_SNOW;
        } else if index < params.counts.x + params.counts.y + params.dust_counts.x {
            kind = KIND_DUST;
        }
        if kind == KIND_PARKED {
            particles[index] = Particle(pos, (epoch << 2u) | KIND_PARKED, vec3<f32>(0.0), 0.0);
        } else {
            live = true;
            if (p.kind & 3u) == KIND_PARKED {
                epoch += 1u;
                if kind == KIND_DUST {
                    pos = dust_position(index, epoch);
                } else {
                    pos = seed_position(index, epoch);
                }
                life = 0.0;
            }
            let wind = elements_wind_metres(wind_field, wind_sampler, params.wind, pos);
            if kind == KIND_RAIN {
                let d = drop_diameter_mm(particle_rand(index, epoch, 0u), params.rain.x);
                vel = wind + vec3<f32>(0.0, -drop_fall_speed(d), 0.0);
            } else if kind == KIND_DUST {
                vel = wind + vec3<f32>(0.0, -params.dust.x, 0.0);
            } else {
                let u = particle_rand(index, epoch, 1u);
                let frequency = mix(params.snow.z, params.snow.w, u.y);
                let heading = TAU * u.x;
                let phase = TAU * u.z;
                let sway = params.snow.x * sin(TAU * frequency * life + phase);
                vel = wind + vec3<f32>(0.0, -params.snow.y, 0.0)
                    + vec3<f32>(cos(heading), 0.0, sin(heading)) * sway;
            }
        }
    }

    let p0 = pos;
    let p1 = pos + vel * dt;

    // Blockers: tiled through workgroup memory, every invocation takes part in the barriers.
    var t_hit = MISS;
    let tiles = (n_blockers + TILE - 1u) / TILE;
    for (var t = 0u; t < tiles; t += 1u) {
        let bi = t * TILE + local_index;
        if bi < n_blockers {
            tile[local_index] = blockers[bi];
        }
        workgroupBarrier();
        if live && kind != KIND_DUST {
            let n = min(TILE, n_blockers - t * TILE);
            for (var k = 0u; k < n; k += 1u) {
                t_hit = min(t_hit, segment_enter(tile[k], p0, p1));
            }
        }
        workgroupBarrier();
    }
    if !live {
        return;
    }

    // Ground: the terrain surface (plus standing water) under each end of the step. A particle
    // already below it respawns without a splash. The blockers' entry is kept apart: a splash
    // on a blocker's top face always rings, one on the ground only where it is water.
    let t_block = t_hit;
    var t_ground = MISS;
    let g0 = ground_at(p0.xz);
    let g1 = ground_at(p1.xz);
    if p0.y <= g0 {
        t_ground = 0.0;
    } else if p1.y <= g1 {
        let above = p0.y - g0;
        t_ground = above / (above - (p1.y - g1));
    }
    t_hit = min(t_block, t_ground);
    // A sprite carried out of the top of the box respawns in it (wrapping would drop it under the
    // ground for a frame).
    if kind == KIND_DUST && p1.y > cam.y + half_edge {
        t_hit = 0.0;
    }

    var out_pos = p1;
    var out_life = life + dt;
    if t_hit <= 1.0 {
        if kind == KIND_DUST {
            epoch += 1u;
            out_pos = dust_position(index, epoch);
        } else {
            // A segment that enters from outside (t > 0) is an impact; one that starts inside a
            // blocker or below the ground is a respawn that landed badly and makes no splash.
            if kind == KIND_RAIN && t_hit > 0.0 && (t_block < t_ground || ground_rings(mix(p0, p1, t_hit).xz)) {
                let hit = mix(p0, p1, t_hit);
                let dxz = hit.xz - cam.xz;
                if dot(dxz, dxz) <= params.motion.z * params.motion.z {
                    let slot = atomicAdd(&splashes.counter, 1u);
                    splashes.ring[slot & 4095u] = vec4<f32>(hit.x, hit.y + SPLASH_LIFT_M, hit.z, params.motion.x);
                }
            }
            epoch += 1u;
            let r = particle_rand(index, epoch, 2u);
            // Where it would be after the rest of the step, had it re-entered at the top.
            out_pos = vec3<f32>(
                cam.x + (r.x * 2.0 - 1.0) * half_edge,
                min(cam.y + half_edge + (1.0 - t_hit) * (p1.y - p0.y), cam.y + half_edge - 0.001),
                cam.z + (r.z * 2.0 - 1.0) * half_edge,
            );
        }
        out_life = 0.0;
    }
    out_pos = vec3<f32>(
        wrap_axis(out_pos.x, cam.x, half_edge),
        wrap_axis(out_pos.y, cam.y, half_edge),
        wrap_axis(out_pos.z, cam.z, half_edge),
    );
    particles[index] = Particle(out_pos, (epoch << 2u) | kind, vel, out_life);
}
