// Water rendering: the CDLOD patch pool displaced to bedrock + sediment + water, and two blended
// passes of one fragment shader (docs/research/water-rendering.md, "Model adopted"):
//
//   absorb   (`AlphaMode::Multiply`)  dst ← dst · lerp(1, X, a),  X = (1 − F) · T(t_eff) · (1 − foam)
//   surface  (`AlphaMode::Add`)       dst ← dst + a · (F · L_refl + (1 − F) · L_body + foam · L_foam)
//
// Bevy sets `BLEND_MULTIPLY` for the first. The vertex entry point displaces exactly like the
// terrain's (`terrain/render.wesl`): same morph, same packed `terrain_cells`, plus the water depth
// and a 1 mm lift. The thickness comes from the depth prepass, which holds the opaque scene only.

import bevy_elements::grid::{GridParams, grid_taps};
import bevy_elements::noise::perlin3;
import bevy_elements::terrain::common::TerrainParams;
import bevy_pbr::render::{
    mesh_functions,
    mesh_view_bindings::{view, globals},
    pbr_types,
    view_transformations::{position_world_to_clip, position_world_to_view, depth_ndc_to_view_z},
    pbr_fragment::pbr_input_from_standard_material,
    pbr_functions::{apply_pbr_lighting, main_pass_post_lighting_processing},
    forward_io::{VertexOutput, FragmentOutput},
};
import bevy_pbr::prepass::utils::prepass_depth;

struct WaterParams {
    terrain: TerrainParams,
    // Per level: x morph start, y range (m).
    lod: array<vec4<f32>, 16>,
    // x: coarsest level.
    top: vec4<f32>,
    // xy: cloud shadow map origin (world XZ), z: its extent, w: camera transmittance.
    shadow: vec4<f32>,
    // xyz: unit direction toward the light the map was made for. w: 1 when a map is bound.
    light: vec4<f32>,
    // xyz: absorption σ (1/m). w: index of refraction n.
    absorption: vec4<f32>,
    // xyz: silt tint (linear). w: κ, scattering per unit volume fraction of sediment (1/m).
    silt: vec4<f32>,
    // xyz: deep-water body reflectance (linear). w: σ_clear, the scattering of clear water (1/m).
    deep: vec4<f32>,
    // x: flow cycle T (s). y: normal-map tile L (m). z: distortion limit D (tiles).
    // w: ripple slope relative to the normal map's rms slope.
    flow: vec4<f32>,
    // x: foam speed threshold (m/s). y: foam thickness threshold (m). z: foam albedo.
    // w: foam perceptual roughness.
    foam: vec4<f32>,
}

@group(constants::MATERIAL_BIND_GROUP) @binding(100) var<uniform> water: WaterParams;
@group(constants::MATERIAL_BIND_GROUP) @binding(101) var<storage, read> terrain_cells: array<vec4<f32>>;
@group(constants::MATERIAL_BIND_GROUP) @binding(102) var<storage, read> flow_cells: array<vec4<f32>>;
@group(constants::MATERIAL_BIND_GROUP) @binding(103) var normal_map: texture_2d<f32>;
@group(constants::MATERIAL_BIND_GROUP) @binding(104) var normal_sampler: sampler;
@group(constants::MATERIAL_BIND_GROUP) @binding(105) var cloud_map: texture_2d<f32>;
@group(constants::MATERIAL_BIND_GROUP) @binding(106) var cloud_sampler: sampler;

// 1 mm: the water vertex sits this far above the surface so it never z-fights the ground where
// the fragment discard (3 mm) has not yet removed it.
const LIFT: f32 = 0.001;
const DISCARD_DEPTH: f32 = 0.003;
const FADE_DEPTH: f32 = 0.03;
const FRESNEL_F0: f32 = 0.02;
// Half widths of the foam ramps around the thresholds: ±10 % of the speed, ±0.03 m of thickness.
const FOAM_SPEED_RAMP: f32 = 0.1;
const FOAM_THICKNESS_RAMP: f32 = 0.03;
// Offset of the second flow layer, in tiles (Vlachos 2010: layers are offset against repetition).
const LAYER_OFFSET: vec2<f32> = vec2<f32>(0.37, 0.59);
const PHASE_NOISE_M: f32 = 37.0;
const STILL_RIPPLE: f32 = 0.25;
const MAX_THICKNESS: f32 = 1000.0;

struct WaterVertex {
    @builtin(instance_index) instance_index: u32,
    @location(0) position: vec3<f32>,
}

fn terrain_height(xz: vec2<f32>) -> f32 {
    let t = grid_taps(water.terrain.grid, xz);
    let a = mix(terrain_cells[t.i00].x, terrain_cells[t.i10].x, t.w.x);
    let b = mix(terrain_cells[t.i01].x, terrain_cells[t.i11].x, t.w.x);
    return mix(a, b, t.w.y);
}

fn water_depth_at(xz: vec2<f32>) -> f32 {
    let t = grid_taps(water.terrain.grid, xz);
    let a = mix(terrain_cells[t.i00].z, terrain_cells[t.i10].z, t.w.x);
    let b = mix(terrain_cells[t.i01].z, terrain_cells[t.i11].z, t.w.x);
    return mix(a, b, t.w.y);
}

fn morph_k(distance: f32, lod: u32) -> f32 {
    if f32(lod) >= water.top.x {
        return 0.0;
    }
    let range = water.lod[lod];
    return clamp((distance - range.x) / (range.y - range.x), 0.0, 1.0);
}

// The terrain's CDLOD vertex with the morph measured against the terrain surface (so the water
// morphs with the ground under it), lifted by the water depth.
fn water_vertex_position(local: vec3<f32>, instance_index: u32) -> vec4<f32> {
    let m = mesh_functions::get_world_from_local(instance_index);
    let lod = mesh_functions::get_tag(instance_index);
    let gi = round(local.xz * 32.0);
    let step = water.terrain.grid.cell * f32(1u << lod);
    var wxz = vec2<f32>(m[3].x, m[3].z) + gi * step;
    let d = distance(view.lod_view_world_position, vec3<f32>(wxz.x, m[3].y + terrain_height(wxz), wxz.y));
    let odd = gi - 2.0 * floor(gi * 0.5);
    wxz = wxz - odd * step * morph_k(d, lod);
    return vec4<f32>(wxz.x, m[3].y + terrain_height(wxz) + water_depth_at(wxz) + LIFT, wxz.y, 1.0);
}

@vertex
fn vertex(v: WaterVertex) -> VertexOutput {
    var out: VertexOutput;
    let world = water_vertex_position(v.position, v.instance_index);
    out.world_position = world;
    out.position = position_world_to_clip(world.xyz);
    out.world_normal = vec3<f32>(0.0, 1.0, 0.0);
    @if(VERTEX_OUTPUT_INSTANCE_INDEX)
    out.instance_index = v.instance_index;
    return out;
}

// ---- The pieces the CPU mirrors in `water.rs` -------------------------------------------------

// Beer–Lambert transmittance of `t` metres of water, per channel.
fn beer_lambert(sigma: vec3<f32>, t: f32) -> vec3<f32> {
    return exp(-sigma * t);
}

// Schlick's Fresnel reflectance of water at `cos_v`.
fn fresnel(cos_v: f32) -> f32 {
    let k = 1.0 - clamp(cos_v, 0.0, 1.0);
    let k2 = k * k;
    return FRESNEL_F0 + (1.0 - FRESNEL_F0) * k2 * k2 * k;
}

// Thickness along the refracted ray under a flat surface from the thickness along the view ray:
// `t_ray · cos θ_v / cos θ_r` (Snell's law).
fn refracted_thickness(t_ray: f32, cos_v: f32, n: f32) -> f32 {
    let cos_r = sqrt(max(1.0 - (1.0 - cos_v * cos_v) / (n * n), 1e-4));
    return t_ray * cos_v / cos_r;
}

// Vlachos's two-phase weights for the phase `p` of a layer: a triangle that is 0 at the restart.
fn phase_weight(p: f32) -> f32 {
    return 1.0 - abs(2.0 * p - 1.0);
}

fn foam_from_speed(speed: f32) -> f32 {
    let thr = water.foam.x;
    return smoothstep(thr * (1.0 - FOAM_SPEED_RAMP), thr * (1.0 + FOAM_SPEED_RAMP), speed);
}

fn foam_from_thickness(t_ray: f32) -> f32 {
    let thr = water.foam.y;
    return 1.0 - smoothstep(thr - FOAM_THICKNESS_RAMP, thr + FOAM_THICKNESS_RAMP, t_ray);
}

fn edge_alpha(depth: f32) -> f32 {
    return smoothstep(DISCARD_DEPTH, FADE_DEPTH, depth);
}

// Cloud shadow scale ρ = (0.25 + 0.75 T_px) / (0.25 + 0.75 T_cam) at a world position, as the
// terrain's fragment computes it.
fn cloud_shadow_ratio(p: vec3<f32>) -> f32 {
    let l = water.light.xyz;
    var t = 1.0;
    if water.light.w > 0.5 && l.y > 0.1 {
        let uv = (p.xz - l.xz * (p.y / l.y) - water.shadow.xy) / water.shadow.z;
        if all(uv >= vec2<f32>(0.0)) && all(uv <= vec2<f32>(1.0)) {
            t = textureSampleLevel(cloud_map, cloud_sampler, uv, 0.0).r;
        }
    }
    let here = 0.25 + 0.75 * t;
    let camera = max(0.25 + 0.75 * water.shadow.w, 0.01);
    return clamp(here / camera, 0.0, 4.0);
}

struct Flow {
    // The ripple normal (world, y up) and the foam patchiness source (the map's alpha).
    normal: vec3<f32>,
    patch_noise: f32,
}

// Two-phase flow-mapped normals (Vlachos 2010): two layers of the tiling normal map, each advected
// along the velocity by up to D tiles over a cycle, half a cycle apart and weighted by a triangle
// so that a layer is invisible when it restarts.
fn flow_normal(p: vec2<f32>, velocity: vec2<f32>, gx: vec2<f32>, gy: vec2<f32>) -> Flow {
    let cycle = water.flow.x;
    let tile = water.flow.y;
    let limit = water.flow.z;
    let uv = p / tile;
    var offset = velocity * cycle / tile;
    let len = length(offset);
    if len > limit {
        offset = offset * (limit / len);
    }
    let psi = 0.5 + 0.5 * perlin3(vec3<f32>(p / PHASE_NOISE_M, 0.5));
    let phase = globals.time / cycle + psi;
    let p0 = fract(phase);
    let p1 = fract(phase + 0.5);
    let w0 = phase_weight(p0);
    let w1 = phase_weight(p1);
    let s0 = textureSampleGrad(normal_map, normal_sampler, uv - offset * p0, gx, gy);
    let s1 = textureSampleGrad(normal_map, normal_sampler, uv - offset * p1 + LAYER_OFFSET, gx, gy);
    let n0 = s0.xyz * 2.0 - 1.0;
    let n1 = s1.xyz * 2.0 - 1.0;
    let slope = w0 * (n0.xy / max(n0.z, 0.1)) + w1 * (n1.xy / max(n1.z, 0.1));
    let strength = water.flow.w * mix(STILL_RIPPLE, 1.0, saturate(length(velocity) / water.foam.x));
    var out: Flow;
    out.normal = normalize(vec3<f32>(slope.x * strength, 1.0, slope.y * strength));
    out.patch_noise = w0 * s0.a + w1 * s1.a;
    return out;
}

struct Column {
    depth: f32,
    velocity: vec2<f32>,
    turbidity: f32,
}

// The water column at a world XZ: four bilinear taps of `terrain_cells` (depth) and `flow_cells`
// (velocity, turbidity). The velocity is faded in with the depth: it is noisy where the film is
// a few millimetres deep.
fn column_at(xz: vec2<f32>) -> Column {
    let t = grid_taps(water.terrain.grid, xz);
    var c: Column;
    c.depth = mix(
        mix(terrain_cells[t.i00].z, terrain_cells[t.i10].z, t.w.x),
        mix(terrain_cells[t.i01].z, terrain_cells[t.i11].z, t.w.x),
        t.w.y,
    );
    let f00 = flow_cells[t.i00];
    let f10 = flow_cells[t.i10];
    let f01 = flow_cells[t.i01];
    let f11 = flow_cells[t.i11];
    let f = mix(mix(f00, f10, t.w.x), mix(f01, f11, t.w.x), t.w.y);
    c.velocity = f.xy * smoothstep(DISCARD_DEPTH, FADE_DEPTH, c.depth);
    c.turbidity = max(f.z, 0.0);
    return c;
}

fn silt_body(turbidity: f32) -> vec3<f32> {
    let kappa = water.silt.w;
    let sigma_c = kappa * turbidity;
    let tau = sigma_c / (sigma_c + water.deep.w);
    return mix(water.deep.xyz, water.silt.xyz, tau);
}

@fragment
fn fragment(in: VertexOutput, @builtin(front_facing) is_front: bool) -> FragmentOutput {
    let p = in.world_position.xyz;
    // Derivatives of the tile coordinate, taken before the discard (uniform control flow).
    let gx = dpdx(p.xz / water.flow.y);
    let gy = dpdy(p.xz / water.flow.y);
    let column = column_at(p.xz);
    if column.depth < DISCARD_DEPTH {
        discard;
    }
    let a_edge = edge_alpha(column.depth);

    let flow = flow_normal(p.xz, column.velocity, gx, gy);
    let n = flow.normal;
    let v_dir = normalize(view.world_position - p);
    let cos_v = max(dot(v_dir, n), 0.02);
    let f = fresnel(cos_v);

    // Thickness from the depth prepass: the opaque scene under the water along the view ray.
    let view_pos = position_world_to_view(p);
    let ray = normalize(view_pos);
    let z_scene = depth_ndc_to_view_z(max(prepass_depth(in.position, 0u), 1e-9));
    let t_ray = clamp((view_pos.z - z_scene) / max(-ray.z, 1e-3), 0.0, MAX_THICKNESS);
    let t_eff = refracted_thickness(t_ray, cos_v, water.absorption.w);

    // Foam: fast flow, and the contact with anything the water meets (shoreline, rocks, posts).
    let blotch = smoothstep(0.45, 0.75, flow.patch_noise);
    let f_speed = foam_from_speed(length(column.velocity)) * blotch;
    let f_contact = foam_from_thickness(t_ray) * mix(1.0, blotch, smoothstep(0.02, 0.03, t_ray));
    let foam = max(f_speed, f_contact);

@if(BLEND_MULTIPLY) {
    let x = (1.0 - f) * beer_lambert(water.absorption.xyz, t_eff) * (1.0 - foam);
    var out: FragmentOutput;
    // Premultiplied by the edge alpha, written directly: a transmittance is not fogged or tonemapped.
    out.color = vec4<f32>(x * a_edge, a_edge);
    return out;
} @else {
    var pbr_input = pbr_input_from_standard_material(in, is_front);
    let sigma_s = water.deep.w + water.silt.w * column.turbidity;
    let alpha_body = 1.0 - exp(-sigma_s * t_eff);
    let body = silt_body(column.turbidity) * alpha_body;
    let foam_albedo = vec3<f32>(water.foam.z);
    pbr_input.material.base_color = vec4<f32>(mix(body, foam_albedo, foam), 1.0);
    pbr_input.material.perceptual_roughness = mix(pbr_input.material.perceptual_roughness, water.foam.w, foam);
    pbr_input.material.metallic = 0.0;
    pbr_input.N = n;
    pbr_input.world_normal = n;
    // The ground fog already scattered over the bed this pixel shows; the film adds no fog of its
    // own (a fog blend would pull the added radiance toward the fog colour twice).
    pbr_input.material.flags = pbr_input.material.flags & ~pbr_types::STANDARD_MATERIAL_FLAGS_FOG_ENABLED_BIT;
    var out: FragmentOutput;
    let lit = apply_pbr_lighting(pbr_input);
    out.color = vec4<f32>(lit.rgb * cloud_shadow_ratio(p), a_edge);
    out.color = main_pass_post_lighting_processing(pbr_input, out.color);
    return out;
}
}
