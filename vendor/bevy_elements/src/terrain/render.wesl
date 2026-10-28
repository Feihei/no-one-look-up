// The owned terrain render: CDLOD vertex displacement from the packed `terrain_cells` buffer and
// the procedural layer blend. See docs/research/terrain.md ("Vertex stage", "Fragment stage").
//
// One file for the forward, prepass and deferred pipelines (like bevy_pbr's pbr.wesl): the vertex
// entry point is the vertex shader of all three, so the main pass and the depth prepass displace
// bit-identically. The fragment entry point is the forward shader and the deferred one; the
// depth/normal prepass fragment stays Bevy's.

import bevy_elements::grid::{GridParams, grid_taps};
import bevy_elements::noise::{perlin3_d, fbm3};
import bevy_elements::terrain::common::TerrainParams;
import bevy_elements::surface_cover::{weather_surface, ripple_weights, ripple_offset};
import bevy_pbr::render::{
    mesh_functions,
    mesh_view_bindings::{view, lights},
    view_transformations::position_world_to_clip,
    pbr_fragment::pbr_input_from_standard_material,
};

@if(PREPASS_PIPELINE)
import bevy_pbr::prepass::io::VertexOutput;
@else
import bevy_pbr::render::forward_io::{VertexOutput, FragmentOutput};

@if(PREPASS_PIPELINE && DEFERRED_PREPASS)
import bevy_pbr::prepass::io::FragmentOutput;
@if(PREPASS_PIPELINE && DEFERRED_PREPASS)
import bevy_pbr::deferred::functions::deferred_output;

@if(!PREPASS_PIPELINE)
import bevy_pbr::render::pbr_functions::{apply_pbr_lighting, main_pass_post_lighting_processing};

struct TerrainMaterialParams {
    terrain: TerrainParams,
    // Per level: x morph start, y range (m).
    lod: array<vec4<f32>, 16>,
    // x: coarsest level.
    top: vec4<f32>,
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
}

@group(constants::MATERIAL_BIND_GROUP) @binding(100) var<uniform> material: TerrainMaterialParams;
@group(constants::MATERIAL_BIND_GROUP) @binding(101) var<storage, read> terrain_cells: array<vec4<f32>>;
// (snow depth m, wetness 0–1, snow water equivalent m, —) per cell; a one-cell placeholder while
// `material.cover.x` is 0.
@group(constants::MATERIAL_BIND_GROUP) @binding(102) var<storage, read> cover_cells: array<vec4<f32>>;
@group(constants::MATERIAL_BIND_GROUP) @binding(103) var cloud_map: texture_2d<f32>;
@group(constants::MATERIAL_BIND_GROUP) @binding(104) var cloud_sampler: sampler;

struct TerrainVertex {
    @builtin(instance_index) instance_index: u32,
    @location(0) position: vec3<f32>,
}

// Height above the terrain's origin (m) at world XZ, from the four taps around it.
fn terrain_height(xz: vec2<f32>) -> f32 {
    let t = grid_taps(material.terrain.grid, xz);
    let a = mix(terrain_cells[t.i00].x, terrain_cells[t.i10].x, t.w.x);
    let b = mix(terrain_cells[t.i01].x, terrain_cells[t.i11].x, t.w.x);
    return mix(a, b, t.w.y);
}

// The surface gradient (∂h/∂x, ∂h/∂z) at world XZ: a bilinear blend of the stored central
// differences, smooth across cell edges.
fn terrain_gradient(xz: vec2<f32>) -> vec2<f32> {
    let t = grid_taps(material.terrain.grid, xz);
    let g00 = unpack2x16float(bitcast<u32>(terrain_cells[t.i00].w));
    let g10 = unpack2x16float(bitcast<u32>(terrain_cells[t.i10].w));
    let g01 = unpack2x16float(bitcast<u32>(terrain_cells[t.i01].w));
    let g11 = unpack2x16float(bitcast<u32>(terrain_cells[t.i11].w));
    return mix(mix(g00, g10, t.w.x), mix(g01, g11, t.w.x), t.w.y);
}

// Snow depth (m) at world XZ from the cover pack; 0 without the maps.
fn snow_depth(xz: vec2<f32>) -> f32 {
    if material.cover.x < 0.5 {
        return 0.0;
    }
    let t = grid_taps(material.terrain.grid, xz);
    let a = mix(cover_cells[t.i00].x, cover_cells[t.i10].x, t.w.x);
    let b = mix(cover_cells[t.i01].x, cover_cells[t.i11].x, t.w.x);
    return mix(a, b, t.w.y);
}

fn morph_k(distance: f32, lod: u32) -> f32 {
    if f32(lod) >= material.top.x {
        return 0.0;
    }
    let range = material.lod[lod];
    return clamp((distance - range.x) / (range.y - range.x), 0.0, 1.0);
}

// World position of a patch vertex with the CDLOD morph applied (Strugar's morphVertex): the
// instance matrix's translation is the patch origin (x, terrain y, z); `MeshTag` is the level.
fn terrain_vertex_position(local: vec3<f32>, instance_index: u32) -> vec4<f32> {
    let m = mesh_functions::get_world_from_local(instance_index);
    let lod = mesh_functions::get_tag(instance_index);
    let gi = round(local.xz * 32.0);
    let step = material.terrain.grid.cell * f32(1u << lod);
    var wxz = vec2<f32>(m[3].x, m[3].z) + gi * step;
    let d = distance(view.lod_view_world_position, vec3<f32>(wxz.x, m[3].y + terrain_height(wxz), wxz.y));
    let odd = gi - 2.0 * floor(gi * 0.5);
    wxz = wxz - odd * step * morph_k(d, lod);
    // The snow lies on the surface: the same depth in the prepass, so the depths agree.
    return vec4<f32>(wxz.x, m[3].y + terrain_height(wxz) + snow_depth(wxz), wxz.y, 1.0);
}

@vertex
fn vertex(v: TerrainVertex) -> VertexOutput {
    var out: VertexOutput;
    let world = terrain_vertex_position(v.position, v.instance_index);
    out.world_position = world;
    out.position = position_world_to_clip(world.xyz);
    @if(!PREPASS_PIPELINE || NORMAL_PREPASS_OR_DEFERRED_PREPASS) {
        let g = terrain_gradient(world.xz);
        out.world_normal = normalize(vec3<f32>(-g.x, 1.0, -g.y));
    }
    @if(PREPASS_PIPELINE && MOTION_VECTOR_PREPASS)
    out.previous_world_position = world;
    @if(PREPASS_PIPELINE && UNCLIPPED_DEPTH_ORTHO_EMULATION) {
        out.unclipped_depth = out.position.z;
        out.position.z = min(out.position.z, 1.0);
    }
    @if(VERTEX_OUTPUT_INSTANCE_INDEX)
    out.instance_index = v.instance_index;
    return out;
}

// Linear albedo of the four layers: the plan's sRGB colours (Deviation 4).
const ROCK: vec3<f32> = vec3<f32>(0.1005, 0.0890, 0.0783);
const SOIL: vec3<f32> = vec3<f32>(0.0732, 0.0397, 0.0196);
const SAND: vec3<f32> = vec3<f32>(0.5382, 0.3931, 0.1960);
const GRASS: vec3<f32> = vec3<f32>(0.0397, 0.0835, 0.0134);

// Andersson 2007 §5.2.3, Listing 2: the Overlay blend of a detail value onto a computed mask, at
// full opacity. It keeps 0 and 1, and 0.5 is the identity, so the detail breaks up a transition
// without moving where a layer is fully on or off.
fn overlay(mask: f32, detail: f32) -> f32 {
    return select(1.0 - 2.0 * (1.0 - mask) * (1.0 - detail), 2.0 * mask * detail, mask < 0.5);
}

// The four layer weights (rock, sand, soil, grass) for a slope angle (degrees), a sediment depth
// (m) and a detail value in [0, 1] (0.5: none). They sum to 1 by construction; `model/mirror.rs`
// is the CPU copy (terrain.md Deviation 19).
fn layer_weights(slope_deg: f32, sediment: f32, detail: f32) -> vec4<f32> {
    let rock = overlay(smoothstep(35.0, 45.0, slope_deg), detail);
    let sand = overlay(smoothstep(0.3, 0.7, sediment), detail) * (1.0 - smoothstep(26.0, 30.0, slope_deg)) * (1.0 - rock);
    let band = overlay(smoothstep(26.0, 36.0, slope_deg), detail);
    let rest = (1.0 - rock) * (1.0 - sand);
    return vec4<f32>(rock, sand, rest * band, rest * (1.0 - band));
}

/// The raw cloud deck transmittance over the ground at `p` (1.0 outside the map or without one):
/// the deck the sky's light comes through, so the shade ramp's signal (`snow-cover.md`
/// Deviation 15).
fn cloud_deck(p: vec3<f32>) -> f32 {
    let l = material.light.xyz;
    var t = 1.0;
    if material.light.w > 0.5 && l.y > 0.1 {
        let uv = (p.xz - l.xz * (p.y / l.y) - material.shadow.xy) / material.shadow.z;
        if all(uv >= vec2<f32>(0.0)) && all(uv <= vec2<f32>(1.0)) {
            t = textureSampleLevel(cloud_map, cloud_sampler, uv, 0.0).r;
        }
    }
    return t;
}

// Cloud shadow scale ρ = (0.25 + 0.75 T_px) / (0.25 + 0.75 T_cam) at a world position, the same
// formula the weathered materials use (`surface_cover.wesl::cloud_shadow_ratio`; clouds.md,
// "Cloud shadow map").
fn cloud_shadow_ratio(p: vec3<f32>) -> f32 {
    let here = 0.25 + 0.75 * cloud_deck(p);
    let camera = max(0.25 + 0.75 * material.shadow.w, 0.01);
    return clamp(here / camera, 0.0, 4.0);
}

// The light's way down to the bed through `depth` metres of water, per channel (water-rendering.md,
// "The bed under the water"): `exp(-σ d / μ')` with `μ'` the sine of the sun's elevation after
// refraction at the surface. The view leg back up is the water's own absorb pass.
fn bed_transmittance(depth: f32) -> vec3<f32> {
    let n = material.absorption.w;
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
    return exp(-material.absorption.xyz * depth / mu_r);
}

@if(!PREPASS_PIPELINE || DEFERRED_PREPASS)
@fragment
fn fragment(in: VertexOutput, @builtin(front_facing) is_front: bool) -> FragmentOutput {
    var pbr_input = pbr_input_from_standard_material(in, is_front);
    let p = in.world_position.xyz;

    // Cell fetch: sediment and water bilinear; the normal from the stored gradient.
    let t = grid_taps(material.terrain.grid, p.xz);
    let c00 = terrain_cells[t.i00];
    let c10 = terrain_cells[t.i10];
    let c01 = terrain_cells[t.i01];
    let c11 = terrain_cells[t.i11];
    let sediment = mix(mix(c00.y, c10.y, t.w.x), mix(c01.y, c11.y, t.w.x), t.w.y);
    let water_depth = mix(mix(c00.z, c10.z, t.w.x), mix(c01.z, c11.z, t.w.x), t.w.y);
    let g = mix(
        mix(unpack2x16float(bitcast<u32>(c00.w)), unpack2x16float(bitcast<u32>(c10.w)), t.w.x),
        mix(unpack2x16float(bitcast<u32>(c01.w)), unpack2x16float(bitcast<u32>(c11.w)), t.w.x),
        t.w.y,
    );
    var n = normalize(vec3<f32>(-g.x, 1.0, -g.y));

    // Slope angle and the mask detail: four fBm octaves from 48 m down to 6 m, contrast-scaled
    // into [0, 1] and overlaid on the masks (Andersson 5.2.3), so the transitions are irregular
    // rather than iso-slope lines.
    let alpha = degrees(acos(clamp(n.y, -1.0, 1.0)));
    let detail = clamp(0.5 + 0.75 * fbm3(p / 48.0, 4u), 0.0, 1.0);
    let w = layer_weights(alpha, sediment, detail);
    var albedo = w.x * ROCK + w.y * SAND + w.z * SOIL + w.w * GRASS;
    let rough = mix(0.9, 0.95, w.z);

    // Detail: 3D noise at the world position (a procedural function has no projection axis).
    let dist = distance(view.world_position, p);
    albedo = albedo * (1.0 + 0.25 * fbm3(p * 0.8, 3u) * (1.0 - smoothstep(40.0, 120.0, dist)));
    let grad = perlin3_d(p * 0.8).yzw * 0.8;
    let bump = 0.04 * (1.0 - smoothstep(0.0, 60.0, dist));
    n = normalize(n - bump * (grad - dot(grad, n) * n));

    // Ground cover: wet darkening and gloss, puddles with ripples while it rains, snow
    // (docs/research/wetness.md, snow-cover.md). The water depth is the hydrology layer's.
    var wet = 0.0;
    var snow_cover = 0.0;
    var puddle = 0.0;
    var ripple = vec2<f32>(0.0);
    if material.cover.x > 0.5 {
        let k00 = cover_cells[t.i00];
        let k10 = cover_cells[t.i10];
        let k01 = cover_cells[t.i01];
        let k11 = cover_cells[t.i11];
        let cover = mix(mix(k00.xy, k10.xy, t.w.x), mix(k01.xy, k11.xy, t.w.x), t.w.y);
        let water = mix(mix(c00.z, c10.z, t.w.x), mix(c01.z, c11.z, t.w.x), t.w.y);
        snow_cover = smoothstep(0.0, 0.05, cover.x);
        wet = cover.y;
        puddle = smoothstep(material.cover.z, material.cover.w, water);
        if puddle > 0.0 && material.cover.y > 0.0 {
            ripple = ripple_offset(p.xz, material.ripple, ripple_weights(material.cover.y));
        }
    }
    let surface = weather_surface(
        albedo, rough, 0.0, pbr_input.material.reflectance, n, wet, puddle, snow_cover, ripple,
    );
    albedo = surface.base_color;
    n = surface.normal;

    pbr_input.material.base_color = vec4<f32>(albedo, 1.0);
    pbr_input.material.perceptual_roughness = surface.roughness;
    pbr_input.material.metallic = surface.metallic;
    pbr_input.material.reflectance = surface.reflectance;
    pbr_input.world_normal = n;
    pbr_input.N = n;

    let ratio = vec3<f32>(cloud_shadow_ratio(p)) * bed_transmittance(water_depth);
@if(PREPASS_PIPELINE) {
    // Deferred: the lighting runs later, so the shadow scales the albedo instead.
    pbr_input.material.base_color = vec4<f32>(albedo * ratio, 1.0);
    return deferred_output(in, pbr_input);
}
@if(!PREPASS_PIPELINE) {
    var out: FragmentOutput;
    out.color = apply_pbr_lighting(pbr_input);
    // The Moon & Spencer 1942 overcast dome as a shade factor (top 1, side 0.68, bottom 0.5;
    // `docs/research/snow-cover.md` Deviation 15): the sky is brightest at the zenith, so a
    // surface sees less light from below its horizon. It ramps with the deck over this ground
    // (`cloud_deck`, 0 under cloud, 1 in clear air — exactly no change there). Under a deck the
    // beam inside `out` is ~0 (the cloud dims the DirectionalLights), so scaling the lit colour
    // scales the sky's fill alone; the beam share this also dims at partial cover is in
    // Deviation 15.
    out.color = vec4<f32>(
        out.color.rgb * mix(
            1.0,
            (1.0 - 0.35 * pow(max(1.0 - pbr_input.N.y, 0.0), 0.3)),
            1.0 - cloud_deck(p),
        ),
        out.color.a,
    );
    out.color = vec4<f32>(out.color.rgb * ratio, out.color.a);
    out.color = main_pass_post_lighting_processing(pbr_input, out.color);
    return out;
}
}
