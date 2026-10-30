// WeatheredExtension: any mesh, rained and snowed on. Wet surfaces darken and gloss, rain runs down
// vertical faces, up-facing faces whiten under the flora snow curve, and cloud shadows fall on
// everything, all from the shared ElementsGlobals buffer and scaled by the material's exposure
// (0 = sheltered). See docs/research/wetness.md ("WeatheredMaterial and ElementsGlobals").
//
// One file for the forward and deferred pipelines (like bevy_pbr's pbr.wesl): the depth/normal
// prepass fragment stays Bevy's, because nothing here changes alpha or geometry. Skeleton:
// pbr.wesl:59-67, build the input, edit it, light it.
//
// A material `with_trails()` also draws the trail field (docs/research/trails-game-ground.md) in
// forward rendering: the parallax march of `bevy_elements::trails::surface` over the face's own
// plane, as `GroundExtension` does.

import bevy_elements::surface_cover::{
    ElementsGlobals, cloud_shadow_ratio, cloud_shadow_uv, graded, mesh_snow_cover, overcast_relief,
    rain_streak, ripple_offset, ripple_weights, weather_surface,
};
import bevy_elements::terrain::common::detail_scale_3d;
import bevy_pbr::render::{
    pbr_fragment::pbr_input_from_standard_material,
    pbr_functions::alpha_discard,
    pbr_types::STANDARD_MATERIAL_FLAGS_UNLIT_BIT,
};
import bevy_pbr::decal::clustered::apply_decals;

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
@if(!PREPASS_PIPELINE)
import bevy_pbr::render::mesh_view_bindings::{view, lights};
@if(!PREPASS_PIPELINE)
import bevy_elements::trails::surface::{trail_fade, trail_march, trail_none};

@if(VISIBILITY_RANGE_DITHER)
import bevy_pbr::render::pbr_functions::visibility_range_dither;

struct WeatheredParams {
    // 1 exposed, 0 sheltered.
    exposure: f32,
    // 1 when the material draws trails.
    trails: f32,
    // The stone grain's feature size, m; 0 for none.
    grain_m: f32,
    // A colour grade of the base colour: xyz tint, w saturation (the identity by default; the
    // crate's scattered rocks carry `TerrainLook`, docs/research/terrain-look.md).
    look: vec4<f32>,
}

@group(constants::MATERIAL_BIND_GROUP) @binding(100) var<uniform> params: WeatheredParams;
@group(constants::MATERIAL_BIND_GROUP) @binding(101) var<storage, read> elements: ElementsGlobals;
@group(constants::MATERIAL_BIND_GROUP) @binding(102) var cloud_map: texture_2d<f32>;
@group(constants::MATERIAL_BIND_GROUP) @binding(103) var cloud_sampler: sampler;
@group(constants::MATERIAL_BIND_GROUP) @binding(104) var trail_field: texture_2d<f32>;
@group(constants::MATERIAL_BIND_GROUP) @binding(105) var trail_sampler: sampler;

// Compacted snow's albedo relative to fresh snow on a trench floor (trails.md Deviation 8).
const COMPACTED_SNOW_TINT: vec3<f32> = vec3<f32>(0.86, 0.91, 0.97);

// The stone grain (`docs/research/rock-scatter.md` Model 7): a **mottle** at the grain's own size
// and a **speckle** at an eighth of it, each two octaves of the terrain rock's band-limited 3D
// noise (`detail_scale_3d`), so a scattered rock and the cliff it fell from share a surface. A
// stone face reads at arm's length by its centimetre speckle (mineral grains, pits); one fBm from
// the grain's size down puts almost nothing there (the octaves halve), and the rocks read as clay.
// The albedo's swing per unit of each normalised field, and the relief's height per metre of
// each feature size.
const STONE_MOTTLE_ALBEDO: f32 = 1.2;
const STONE_SPECKLE_ALBEDO: f32 = 2.0;
const STONE_SPECKLE_SCALE: f32 = 0.125;
const STONE_MOTTLE_RELIEF: f32 = 0.15;
const STONE_SPECKLE_RELIEF: f32 = 0.35;

// The direction toward the brightest directional light (the sun, else the moon); straight up
// without one.
@if(!PREPASS_PIPELINE)
fn brightest_light() -> vec3<f32> {
    var brightest = 0.0;
    var l = vec3<f32>(0.0, 1.0, 0.0);
    for (var i = 0u; i < lights.n_directional_lights; i += 1u) {
        let c = lights.directional_lights[i].color;
        let luminance = c.r + c.g + c.b;
        if luminance > brightest {
            brightest = luminance;
            l = lights.directional_lights[i].direction_to_light;
        }
    }
    return l;
}

@if(!PREPASS_PIPELINE || DEFERRED_PREPASS)
@fragment
fn fragment(in: VertexOutput, @builtin(front_facing) is_front: bool) -> FragmentOutput {
@if(VISIBILITY_RANGE_DITHER)
    visibility_range_dither(in.position, in.visibility_range_dither);

    var pbr_input = pbr_input_from_standard_material(in, is_front);
    pbr_input.material.base_color = alpha_discard(
        pbr_input.material.flags,
        pbr_input.material.alpha_cutoff,
        pbr_input.material.base_color,
    );
    apply_decals(&pbr_input);

    // Weather: the geometric normal drives snow and streaks, never the normal map.
    let g = elements;
    let exposure = params.exposure;
    let n = normalize(pbr_input.world_normal);
    let p = in.world_position.xyz / g.units.x;

    // The stone grain: mottles the albedo and bends the shading normal, never the geometric one
    // (snow and streaks follow the face). Each octave fades where it is finer than the pixel.
    let footprint = max(length(dpdx(p)), length(dpdy(p)));
    if params.grain_m > 0.0 {
        let mottle = detail_scale_3d(p, params.grain_m, 2u, footprint, 71u);
        let speckle_m = params.grain_m * STONE_SPECKLE_SCALE;
        let speckle = detail_scale_3d(p, speckle_m, 2u, footprint, 79u);
        if mottle.present > 0.0 {
            let m_norm = max(mottle.norm, 1e-6);
            let s_norm = max(speckle.norm, 1e-6);
            let tone = 1.0 + STONE_MOTTLE_ALBEDO * mottle.sum / m_norm
                + STONE_SPECKLE_ALBEDO * speckle.sum / s_norm;
            pbr_input.material.base_color = vec4<f32>(
                pbr_input.material.base_color.rgb * max(tone, 0.2),
                pbr_input.material.base_color.a,
            );
            let grad = mottle.grad / m_norm * (STONE_MOTTLE_RELIEF * params.grain_m)
                + speckle.grad / s_norm * (STONE_SPECKLE_RELIEF * speckle_m);
            let tangential = grad - dot(grad, pbr_input.N) * pbr_input.N;
            pbr_input.N = normalize(pbr_input.N - tangential);
        }
    }

    // Trails: the hit of the view ray on the field over this face's plane. Snow and streaks keep
    // the face's own normal (a trench's wall is snow); what the trench changes is its depth.
    var pressed = 0.0;
    var grad = vec2<f32>(0.0);
    var lit = 1.0;
    var xz = p.xz;
@if(!PREPASS_PIPELINE) {
    if params.trails > 0.5 {
        let slope = -n.xz / max(n.y, 0.2);
        // Walls and steep faces are not walked on.
        let fade = trail_fade(p.xz, g.trail_light.yz, g.trail) * smoothstep(0.5, 0.6, n.y);
        if fade > 0.0 {
            let eye = view.world_position / g.units.x;
            let trail = trail_march(trail_field, trail_sampler, g.trail, p, eye, slope, brightest_light(), fade);
            let pv = vec3<f32>(trail.xz.x, p.y + trail.h + dot(slope, trail.xz - p.xz), trail.xz.y);
            pressed = max(-trail.h, 0.0);
            grad = trail.grad;
            lit = trail.lit;
            xz = pv.xz;
            // The field's slope tilts the shading normal (normal map included) and the geometric
            // one; flat ground leaves both as they were.
            let tilt = vec3<f32>(trail.grad.x, 0.0, trail.grad.y) * n.y;
            pbr_input.N = normalize(pbr_input.N - tilt);
            pbr_input.world_normal = normalize(n - tilt);
            pbr_input.world_position = vec4<f32>(pv * g.units.x, 1.0);
            pbr_input.diffuse_occlusion = pbr_input.diffuse_occlusion * trail.open;
        }
    }
}

    // The grade (identity unless the crate set it: its scattered rocks follow `TerrainLook`),
    // after the vertex colours and the stone grain, before the weather, so snow and water keep
    // their own colours.
    pbr_input.material.base_color = vec4<f32>(
        graded(pbr_input.material.base_color.rgb, params.look),
        pbr_input.material.base_color.a,
    );
    var wet = g.weather.x * exposure;
    let rain = g.weather.z * exposure;
    if rain > 0.0 {
        // The streak noise is in metres; the fragment's world position is in world units.
        wet = wet + rain_streak(p, n, rain, g.wind.z);
    }
    // A trench pressed to the bottom of the snow shows the surface under it (only where pressed:
    // the snow cover is the authority on unpressed ground).
    let snow_depth = g.trail_light.w;
    let bared = select(0.0, smoothstep(snow_depth - 0.02, snow_depth, pressed), pressed > 0.0);
    let snow = mesh_snow_cover(g.weather.y, n.y) * exposure * (1.0 - bared);
    // Ruts in saturated ground hold water on their floors (trails.md Deviation 6).
    let puddle = smoothstep(0.6, 1.0, wet) * smoothstep(0.01, 0.04, pressed)
        * (1.0 - smoothstep(0.3, 0.8, length(grad))) * (1.0 - snow);
    if wet > 0.0 || snow > 0.0 || pressed > 0.0 {
        var ripple = vec2<f32>(0.0);
        if puddle > 0.0 && rain > 0.0 {
            ripple = ripple_offset(xz, g.ripple, ripple_weights(rain));
        }
        let s = weather_surface(
            pbr_input.material.base_color.rgb,
            pbr_input.material.perceptual_roughness,
            pbr_input.material.metallic,
            pbr_input.material.reflectance,
            pbr_input.N,
            wet,
            puddle,
            snow,
            ripple,
        );
        // Compacted snow on a trench floor.
        let packed = snow * smoothstep(0.005, 0.04, pressed);
        pbr_input.material.base_color = vec4<f32>(
            s.base_color * mix(vec3<f32>(1.0), COMPACTED_SNOW_TINT, packed),
            pbr_input.material.base_color.a,
        );
        pbr_input.material.perceptual_roughness = mix(s.roughness, s.roughness * 0.75, packed);
        pbr_input.material.metallic = s.metallic;
        if puddle > 0.0 {
            pbr_input.material.reflectance = s.reflectance;
            pbr_input.N = s.normal;
            pbr_input.world_normal = s.normal;
        }
    }

    // Cloud shadow: the lit colour (the albedo in deferred, whose lighting runs later) scales by
    // the shadow's transmittance against the camera's. Outside the map, or without one, the
    // transmittance is 1 (and the camera's is 1 without clouds, so the scale is exactly 1).
    var t = 1.0;
    let lookup = cloud_shadow_uv(g, in.world_position.xyz / g.units.x);
    if lookup.z > 0.5 {
        t = textureSampleLevel(cloud_map, cloud_sampler, lookup.xy, 0.0).r;
    }
    let ratio = cloud_shadow_ratio(t, g.weather.w);

@if(PREPASS_PIPELINE) {
    pbr_input.material.base_color = vec4<f32>(
        pbr_input.material.base_color.rgb * ratio,
        pbr_input.material.base_color.a,
    );
    return deferred_output(in, pbr_input);
}
@if(!PREPASS_PIPELINE) {
    var out: FragmentOutput;
    if (pbr_input.material.flags & STANDARD_MATERIAL_FLAGS_UNLIT_BIT) == 0u {
        // The overcast dome shades the surface through the environment map (the clouds' sky
        // probe, `docs/research/sky-light.md`); the relief cue adds what the dome lacks, ramped
        // by the deck: `t` is the raw cloud map transmittance sampled above (1 outside the map
        // or with none), so clear air is exactly no change.
        out.color = apply_pbr_lighting(pbr_input);
        // A trail's self-shadow takes the beam's share of the light (trails.md, Rendering).
        let trail_shade = 1.0 - g.trail_light.x * (1.0 - lit);
        out.color = vec4<f32>(out.color.rgb * overcast_relief(pbr_input.N.y, 1.0 - t) * trail_shade, out.color.a);
    } else {
        out.color = pbr_input.material.base_color;
    }
    out.color = vec4<f32>(out.color.rgb * ratio, out.color.a);
    out.color = main_pass_post_lighting_processing(pbr_input, out.color);
    return out;
}
}
