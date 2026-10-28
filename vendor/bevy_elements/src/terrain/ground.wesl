// GroundExtension: a consumer's mesh shaded as the terrain's ground. Water standing on the ground
// darkens it (the wet-shading ramp of docs/research/wetness.md, driven by the water depth until
// M4's wetness map) and cloud shadows fall on it, both by world XZ. Vertex stage and geometry stay
// the consumer's. See docs/research/terrain.md ("GroundExtension").

import bevy_elements::grid::{GridParams, grid_taps};
import bevy_elements::terrain::common::{TerrainParams, terrain_live};
import bevy_elements::surface_cover::{weather_surface, ripple_weights, ripple_offset};
import bevy_pbr::render::{
    mesh_view_bindings::lights,
    pbr_fragment::pbr_input_from_standard_material,
    pbr_functions::{apply_pbr_lighting, main_pass_post_lighting_processing},
    forward_io::{VertexOutput, FragmentOutput},
};

struct GroundParams {
    terrain: TerrainParams,
    // xy: cloud shadow map origin (world XZ), z: its extent, w: camera transmittance.
    shadow: vec4<f32>,
    // xyz: unit direction toward the light the map was made for. w: 1 when a map is bound.
    light: vec4<f32>,
    // xyz: the water's absorption (1/m). w: its refractive index (0 without water rendering).
    absorption: vec4<f32>,
    // x: 1 when the cover maps are bound. y: rain intensity min(1, R_liquid / 20). z: puddle start
    // depth (m). w: full-puddle depth (m).
    cover: vec4<f32>,
    // Phase of each of the four ripple layers, 0–1.
    ripple: vec4<f32>,
    // x: `column_gate_m`, how far below the column top snow, wetness and puddles still shade (m);
    // below -1e29 for no gate.
    gate: vec4<f32>,
}

@group(constants::MATERIAL_BIND_GROUP) @binding(100) var<uniform> ground: GroundParams;
@group(constants::MATERIAL_BIND_GROUP) @binding(101) var<storage, read> terrain_cells: array<vec4<f32>>;
// (snow depth m, wetness 0–1, snow water equivalent m, —) per cell; a one-cell placeholder while
// `ground.cover.x` is 0.
@group(constants::MATERIAL_BIND_GROUP) @binding(102) var<storage, read> cover_cells: array<vec4<f32>>;
@group(constants::MATERIAL_BIND_GROUP) @binding(103) var cloud_map: texture_2d<f32>;
@group(constants::MATERIAL_BIND_GROUP) @binding(104) var cloud_sampler: sampler;

// Water depth above the ground (m) at world XZ; 0 without a terrain.
fn water_depth(xz: vec2<f32>) -> f32 {
    if !terrain_live(ground.terrain) {
        return 0.0;
    }
    let t = grid_taps(ground.terrain.grid, xz);
    if !t.inside {
        return 0.0;
    }
    let a = mix(terrain_cells[t.i00].z, terrain_cells[t.i10].z, t.w.x);
    let b = mix(terrain_cells[t.i01].z, terrain_cells[t.i11].z, t.w.x);
    return mix(a, b, t.w.y);
}

// The sun's way down to the ground through `depth` metres of water, per channel: `exp(-σ d / μ')`
// (docs/research/water-rendering.md, "The bed under the water").
fn bed_transmittance(depth: f32) -> vec3<f32> {
    let n = ground.absorption.w;
    if n <= 1.0 || depth <= 0.0 || lights.n_directional_lights == 0u {
        return vec3<f32>(1.0);
    }
    // The brightest directional light is the sun (the moon is a second light).
    var brightest = 0.0;
    var up = 0.0;
    for (var i = 0u; i < lights.n_directional_lights; i += 1u) {
        let c = lights.directional_lights[i].color;
        let luminance = c.r + c.g + c.b;
        if luminance > brightest {
            brightest = luminance;
            up = lights.directional_lights[i].direction_to_light.y;
        }
    }
    if brightest <= 0.0 {
        return vec3<f32>(1.0);
    }
    let mu = clamp(up, 0.05, 1.0);
    let mu_r = sqrt(max(1.0 - (1.0 - mu * mu) / (n * n), 0.01));
    return exp(-ground.absorption.xyz * depth / mu_r);
}

// (snow depth m, wetness) at world XZ; zero without the maps.
fn cover_at(xz: vec2<f32>) -> vec2<f32> {
    if ground.cover.x < 0.5 || !terrain_live(ground.terrain) {
        return vec2<f32>(0.0);
    }
    let t = grid_taps(ground.terrain.grid, xz);
    if !t.inside {
        return vec2<f32>(0.0);
    }
    let a = mix(cover_cells[t.i00].xy, cover_cells[t.i10].xy, t.w.x);
    let b = mix(cover_cells[t.i01].xy, cover_cells[t.i11].xy, t.w.x);
    return mix(a, b, t.w.y);
}

// 1 where a fragment at world position `p` is within `ground.gate.x` metres below the column top
// (the terrain's surface height at its XZ) or above it, 0 deeper. `gate.x` below -1e29 is no gate.
fn column_open(p: vec3<f32>) -> f32 {
    if ground.gate.x < -1.0e29 || !terrain_live(ground.terrain) {
        return 1.0;
    }
    let t = grid_taps(ground.terrain.grid, p.xz);
    if !t.inside {
        return 1.0;
    }
    let a = mix(terrain_cells[t.i00].x, terrain_cells[t.i10].x, t.w.x);
    let b = mix(terrain_cells[t.i01].x, terrain_cells[t.i11].x, t.w.x);
    let top = ground.terrain.misc.x + mix(a, b, t.w.y);
    return select(0.0, 1.0, p.y >= top - ground.gate.x);
}

fn cloud_shadow_ratio(p: vec3<f32>) -> f32 {
    let l = ground.light.xyz;
    var t = 1.0;
    if ground.light.w > 0.5 && l.y > 0.1 {
        let uv = (p.xz - l.xz * (p.y / l.y) - ground.shadow.xy) / ground.shadow.z;
        if all(uv >= vec2<f32>(0.0)) && all(uv <= vec2<f32>(1.0)) {
            t = textureSampleLevel(cloud_map, cloud_sampler, uv, 0.0).r;
        }
    }
    let here = 0.25 + 0.75 * t;
    let camera = max(0.25 + 0.75 * ground.shadow.w, 0.01);
    return here / camera;
}

@fragment
fn fragment(in: VertexOutput, @builtin(front_facing) is_front: bool) -> FragmentOutput {
    var pbr_input = pbr_input_from_standard_material(in, is_front);
    let xz = in.world_position.xz;
    // A fragment deeper than the gate below the column top (a tunnel under the hill) is not
    // under the maps' snow, wetness or water.
    let gated = column_open(in.world_position.xyz);
    let cover = cover_at(xz) * gated;
    let depth = water_depth(xz) * gated;
    let puddle = smoothstep(ground.cover.z, ground.cover.w, depth);
    var ripple = vec2<f32>(0.0);
    if puddle > 0.0 && ground.cover.y > 0.0 {
        ripple = ripple_offset(xz, ground.ripple, ripple_weights(ground.cover.y));
    }
    let m = pbr_input.material;
    let surface = weather_surface(
        m.base_color.rgb, m.perceptual_roughness, m.metallic, m.reflectance, pbr_input.N,
        cover.y, puddle, smoothstep(0.0, 0.05, cover.x), ripple,
    );
    pbr_input.material.base_color = vec4<f32>(surface.base_color, m.base_color.a);
    pbr_input.material.perceptual_roughness = surface.roughness;
    pbr_input.material.metallic = surface.metallic;
    pbr_input.material.reflectance = surface.reflectance;
    pbr_input.N = surface.normal;
    pbr_input.world_normal = surface.normal;
    var out: FragmentOutput;
    out.color = apply_pbr_lighting(pbr_input);
    out.color = vec4<f32>(out.color.rgb * cloud_shadow_ratio(in.world_position.xyz) * bed_transmittance(depth), out.color.a);
    out.color = main_pass_post_lighting_processing(pbr_input, out.color);
    return out;
}
