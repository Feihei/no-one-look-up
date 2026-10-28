// WeatheredExtension: any mesh, rained and snowed on. Wet surfaces darken and gloss, rain runs down
// vertical faces, up-facing faces whiten under the flora snow curve, and cloud shadows fall on
// everything, all from the shared ElementsGlobals buffer and scaled by the material's exposure
// (0 = sheltered). See docs/research/wetness.md ("WeatheredMaterial and ElementsGlobals").
//
// One file for the forward and deferred pipelines (like bevy_pbr's pbr.wesl): the depth/normal
// prepass fragment stays Bevy's, because nothing here changes alpha or geometry. Skeleton:
// pbr.wesl:59-67, build the input, edit it, light it.

import bevy_elements::surface_cover::{
    ElementsGlobals, cloud_shadow_ratio, cloud_shadow_uv, mesh_snow_cover, rain_streak,
    weather_surface,
};
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

@if(VISIBILITY_RANGE_DITHER)
import bevy_pbr::render::pbr_functions::visibility_range_dither;

// 1 exposed, 0 sheltered.
@group(constants::MATERIAL_BIND_GROUP) @binding(100) var<uniform> exposure: f32;
@group(constants::MATERIAL_BIND_GROUP) @binding(101) var<storage, read> elements: ElementsGlobals;
@group(constants::MATERIAL_BIND_GROUP) @binding(102) var cloud_map: texture_2d<f32>;
@group(constants::MATERIAL_BIND_GROUP) @binding(103) var cloud_sampler: sampler;

/// The delta between the sky's shaped light over the flat ambient inside `apply_pbr_lighting`.
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
    let n = normalize(pbr_input.world_normal);
    var wet = g.weather.x * exposure;
    let rain = g.weather.z * exposure;
    if rain > 0.0 {
        wet = wet + rain_streak(in.world_position.xyz, n, rain, g.wind.z);
    }
    let snow = mesh_snow_cover(g.weather.y, n.y) * exposure;
    if wet > 0.0 || snow > 0.0 {
        let s = weather_surface(
            pbr_input.material.base_color.rgb,
            pbr_input.material.perceptual_roughness,
            pbr_input.material.metallic,
            pbr_input.material.reflectance,
            pbr_input.N,
            wet,
            0.0,
            snow,
            vec2<f32>(0.0),
        );
        pbr_input.material.base_color = vec4<f32>(s.base_color, pbr_input.material.base_color.a);
        pbr_input.material.perceptual_roughness = s.roughness;
        pbr_input.material.metallic = s.metallic;
    }

    // Cloud shadow: the lit colour (the albedo in deferred, whose lighting runs later) scales by
    // the shadow's transmittance against the camera's. Outside the map, or without one, the
    // transmittance is 1 (and the camera's is 1 without clouds, so the scale is exactly 1).
    var t = 1.0;
    let lookup = cloud_shadow_uv(g, in.world_position.xyz);
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
        out.color = apply_pbr_lighting(pbr_input);
        // The Moon & Spencer 1942 overcast dome as a shade factor (top 1, side 0.68, bottom
        // 0.5; `docs/research/snow-cover.md` Deviation 15), ramped by the deck: `t` is the raw
        // cloud map transmittance sampled above (1.0 outside the map or with none), so clear
        // air is exactly 1 (no change). Under a deck the beam is ~0, so scaling the lit colour
        // scales the sky's fill alone; the beam share at partial cover is in Deviation 15.
out.color = vec4<f32>(
        out.color.rgb * mix(
            1.0,
            (1.0 - 0.35 * pow(max(1.0 - pbr_input.N.y, 0.0), 0.3)),
            1.0 - t,
        ),
        out.color.a,
    );
    } else {
        out.color = pbr_input.material.base_color;
    }
    out.color = vec4<f32>(out.color.rgb * ratio, out.color.a);
    out.color = main_pass_post_lighting_processing(pbr_input, out.color);
    return out;
}
}
