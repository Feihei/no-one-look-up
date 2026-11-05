// Season-aware tree vertex stage (R3.5), shared by the forward and prepass
// pipelines.
//
// This file replaces ONLY the vertex stage of `StandardMaterial`. The fragment
// stage stays Bevy's built-in `render/pbr.wesl`, which keeps lighting, distance
// fog, tonemapping and the deferred path identical to every other material in
// the scene. That is also why the season colour travels as a vertex colour:
// `pbr_fragment.wesl` already does
//
//     pbr_input.material.base_color = in.color;        // VERTEX_COLORS
//     pbr_input.material.base_color *= material.base_color;
//
// so a tint in `in.color` reaches the lighting maths with no custom fragment
// code at all.
//
// # Why one file serves both pipelines
//
// The forward and prepass stages MUST produce bit-identical positions: the
// forward pipeline depth-tests with `CompareFunction::GreaterEqual` against
// prepass depth, so a vertex the main pass places behind where the prepass put
// it is discarded -- leaves swaying away from the camera would vanish in
// patches that depend on wind phase. (The deferred pipeline, one
// `OpaqueRendererMethod` away, uses `Equal`, where any disagreement disappears
// outright.)
//
// Keeping the maths in ONE source file makes that agreement structural. The
// `PREPASS_PIPELINE` shader def selects the io module, the vertex struct and the
// clip-space transform; everything else below is one implementation.
//
// # Why the clock lives in the material uniform
//
// `globals.time` is the natural clock, but it sits at `@group(0) @binding(11)`
// in the forward pass and `@group(0) @binding(1)` in the prepass, and one file
// cannot declare both. The season uniform is a MATERIAL uniform, bound at the
// same group and binding in both pipelines, so the frame time rides along with
// it: one value per frame, read identically by both stages.

// Imported from the `bevy_pbr` package, not `package::`: `package::` resolves
// to the CRATE THAT OWNS THE SHADER, and this shader is a game asset whose
// module path is `shaders::tree_sway`, which has no `render` submodule. Bevy's
// own shaders can write `package::` because they live inside bevy_pbr.
import bevy_pbr::render::{mesh_functions, mesh_view_bindings::view};

@if(PREPASS_PIPELINE)
import bevy_pbr::prepass::io::{Vertex, VertexOutput, decompress_vertex};
@else
import bevy_pbr::render::forward_io::{Vertex, VertexOutput};
import bevy_pbr::render::view_transformations::position_world_to_clip;

// Season parameters, mirrored from `plugins::season::SeasonUniform`.
// Group 3 is Bevy's material bind group (`MATERIAL_BIND_GROUP_INDEX`); binding
// 100 leaves 0-99 to the base `StandardMaterial`.
struct SeasonParams {
    // rgb: foliage tint in linear space with unit luminance, so multiplying it
    // into the silhouette base colour changes hue only. a: leaf density,
    // 0 = bare branches.
    tint_density: vec4<f32>,
    // x: sway strength (m) at weight 1.0. y: sway speed (rad/s).
    // z: phase length (1/m). w: frame time (s), shared by both stages.
    sway: vec4<f32>,
};

@group(3) @binding(100) var<uniform> season: SeasonParams;

// Peak sway displacement (m) for one vertex.
//
// `weight` is the vertex-colour R channel (ADR-0004's art interface): foliage
// carries ~1.0, wood ~0.12, and both grow with height so the crown travels
// further than the trunk.
fn sway_offset(world_position: vec3<f32>, weight: f32, phase: f32) -> vec3<f32> {
    let strength = season.sway.x * weight;
    if strength <= 0.0 {
        return vec3<f32>(0.0);
    }
    let speed = season.sway.y;
    let phase_len = season.sway.z;

    // Three axes with incommensurate frequencies, so the motion never settles
    // into a visible loop and neighbouring branches drift out of step. The
    // multipliers are the original Godot shader's (1.123 / 1.12412 / 1.3123):
    // they were picked by eye to avoid an obvious period.
    let t = season.sway.w * speed + phase * 6.2831853;
    let x = sin(world_position.x * phase_len * 1.123 + t);
    // The vertical term is damped: a crown that bobs as much as it sways reads
    // as a bouncing object rather than as wind.
    let y = sin(world_position.y * phase_len + t * 1.12412) * 0.45;
    let z = sin(world_position.z * phase_len * 0.9123 + t * 1.3123);
    return vec3<f32>(x, y, z) * strength;
}

// The one displacement function both stages call: local position in, world
// position out.
//
// Sway is added in WORLD space and driven by the world position, so neighbouring
// cards move coherently; driving it from the local position would make every
// card oscillate about its own origin instead of with the tree.
//
// The geometry itself is NOT displaced any more. Leaf density used to collapse
// each cluster toward its centre, which was how an opaque crown lost its leaves
// in winter. The cards now carry a leaf texture, so density is the material's
// base-colour alpha times the texture alpha, and the alpha test does the work --
// with correct depth, because Bevy's prepass honours texture alpha.
fn season_world_position(
    world_from_local: mat4x4<f32>,
    local_position: vec3<f32>,
    vertex_color: vec4<f32>,
) -> vec3<f32> {
    let world = (world_from_local * vec4<f32>(local_position, 1.0)).xyz;
    return world + sway_offset(world, vertex_color.r, vertex_color.b);
}

@vertex
fn vertex(vertex_no_morph: Vertex) -> VertexOutput {
    var out: VertexOutput;
    let instance_index = vertex_no_morph.instance_index;
    let world_from_local = mesh_functions::get_world_from_local(instance_index);

    // The prepass io module stores vertices in a form that has to be
    // decompressed first (compression is optional, so this is often a no-op);
    // the forward module hands over plain fields. The two structs are
    // different types, hence the branch rather than one shared call.
@if(PREPASS_PIPELINE)
    let v = decompress_vertex(vertex_no_morph, instance_index);
@else
    let v = vertex_no_morph;

    // The bake writes the sway weight, foliage mask and per-card phase into
    // COLOR. UVs now belong to the leaf texture and mean nothing to the sway,
    // so this stage reads none of them.
    //
    // The read stays behind its shader def: the field genuinely does not exist
    // when the layout lacks the attribute.
    var color = vec4<f32>(1.0, 0.0, 0.0, 1.0);
@if(VERTEX_COLORS)
    color = v.color;

    let world_position = season_world_position(world_from_local, v.position, color);

    out.world_position = vec4<f32>(world_position, 1.0);

    // Tree LODs crossfade over the `VisibilityRange` margins, which the
    // fragment stage implements by dithering. The level must be written here
    // explicitly -- this file replaces Bevy's vertex stage, so nothing fills
    // the field in for us. Distance is measured to the baked origin
    // (`world_from_local[3]`), not the swayed vertex, so the fade does not
    // shimmer with the wind phase.
@if(VISIBILITY_RANGE_DITHER)
    out.visibility_range_dither = mesh_functions::get_visibility_range_dither_level(
        instance_index,
        world_from_local[3],
    );
@if(PREPASS_PIPELINE)
    out.position = view.clip_from_world * vec4<f32>(world_position, 1.0);
@else
    out.position = position_world_to_clip(world_position);

    // Directional-light shadow views disable depth clipping on hardware that
    // supports it; on hardware that does not, the depth is clamped in the
    // fragment stage instead. Both branches only matter for the prepass, and
    // dropping them would produce shadows that pop where a crown leaves the
    // light frustum.
@if(PREPASS_PIPELINE && UNCLIPPED_DEPTH_ORTHO_EMULATION)
    out.unclipped_depth = out.position.z;
    out.position.z = min(out.position.z, 1.0);

@if(VERTEX_UVS_A)
    out.uv = v.uv;
@if(VERTEX_UVS_B)
    out.uv_b = v.uv_b;

@if(PREPASS_PIPELINE && NORMAL_PREPASS_OR_DEFERRED_PREPASS && VERTEX_NORMALS)
    out.world_normal = mesh_functions::mesh_normal_local_to_world(v.normal, instance_index);
@if(!PREPASS_PIPELINE && VERTEX_NORMALS)
    out.world_normal = mesh_functions::mesh_normal_local_to_world(v.normal, instance_index);

@if(PREPASS_PIPELINE && NORMAL_PREPASS_OR_DEFERRED_PREPASS && VERTEX_TANGENTS)
    out.world_tangent = mesh_functions::mesh_tangent_local_to_world(
        world_from_local,
        v.tangent,
        instance_index,
    );
@if(!PREPASS_PIPELINE && VERTEX_TANGENTS)
    out.world_tangent = mesh_functions::mesh_tangent_local_to_world(
        world_from_local,
        v.tangent,
        instance_index,
    );

    // Motion vectors (TAA): the previous position is the baked one, because the
    // sway is procedural and has no history to look up. Only compiled when the
    // camera asks for motion vectors.
@if(PREPASS_PIPELINE && MOTION_VECTOR_PREPASS)
    out.previous_world_position = vec4<f32>(world_position, 1.0);

    // The season tint. `in.color` carries the sway weight and cluster phase in
    // R and B, which must NOT leak into the shading, so the forward stage
    // writes a colour built from the mask alone: wood keeps the bare silhouette
    // (a 1.0 multiplier), foliage gets the season's hue. The prepass has no
    // colour target; it only forwards the raw value for the fragment input.
@if(VERTEX_COLORS && !PREPASS_PIPELINE)
    out.color = vec4<f32>(
        mix(vec3<f32>(1.0), season.tint_density.rgb, color.g),
        1.0,
    );
@if(VERTEX_COLORS && PREPASS_PIPELINE)
    out.color = color;

    // Tree LODs crossfade with `VisibilityRange` margins (see the dither level
    // written above); this only forwards the instance index the shadow and
    // motion-vector stages need.
@if(VERTEX_OUTPUT_INSTANCE_INDEX)
    out.instance_index = instance_index;

    return out;
}
