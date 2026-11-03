// Facade vertex stage (R3.11): per-face UVs that keep every window the same
// physical size on every building, from one shared unit cube and one shared
// material per (style, variant, wear).
//
// # Why the UVs are computed here instead of baked into the mesh
//
// Buildings are a *shared* unit cube scaled per instance. Their local positions
// are always ±0.5, so a baked UV would stretch with the building: a 4 m
// shophouse and a 48 m shed would show the same four windows, the shed's each
// 12 m wide. The pattern must be metric, so it has to be derived at draw time
// from something that knows the size.
//
// # Why not world-space UVs
//
// `fract(world.xz / tile)` is the obvious way to get metric windows, and it is
// what this step first used. It has one defect that cannot be fixed in the
// shader: the pattern is anchored to the world, not to the building, so every
// facade ends mid-window at its corner. At a 2 m bay on a 4.6 m shophouse that
// is a sliced window on both ends of every building on the street.
//
// Instead the extents come from the instance transform itself:
//
//     scale.x = length(world_from_local[0].xyz)
//
// which is exact for the TRS matrices the chunk spawns (no shear, no negative
// scale). Placement snaps length and height to whole bays/floors, so
//
//     bays   = round(scale.x / bay)
//     floors = round(scale.y / floor)
//
// recovers the integer the placement used, and `u = (local.x + 0.5) * bays`
// puts a bay boundary exactly on each corner with the windows centred inside.
// No per-instance component, no extra mesh, nothing to keep in sync — the
// building's own Transform is the source of truth, which is also why this
// cannot drift when placement changes its snapping.
//
// # Two constants come from the material
//
// `bay` and `floor` live in `FacadeParams` (one uniform per style), because
// they are properties of the *style*, not of the instance. The shader also
// reads the tile's bay/floor counts so the 4-bay texture is not a second
// hard-coded 4 in WGSL.
//
// # The roof face
//
// A box's top face is the roof, and it must not show windows: at 60 degrees of
// pitch the player sees the roofline constantly, and a window pattern lying
// flat on the roof reads immediately as a box. Vertical-ish normals are
// therefore parked on a constant point inside a wall band (between two window
// rows), which renders the roof as flat wall colour. Roofs proper are R3.11
// step 11.3; a flat placeholder is the honest interim.
//
// # Only the vertex stage is replaced
//
// The fragment stage stays Bevy's built-in `pbr.wesl`, exactly as `tree_sway`
// does: lighting, the environment map (which is what makes the glass reflect
// the dusk sky), distance fog, tonemapping and the prepass all keep working
// with no custom code. The prepass vertex stage is left alone as well — this
// shader moves nothing, and the material is opaque, so the prepass never
// samples a texture and never needs these UVs.

import bevy_pbr::render::{
    mesh_functions,
    forward_io::{Vertex, VertexOutput},
    view_transformations::position_world_to_clip,
};

// Mirrors `plugins::facade::FacadeParams`.
//
// Group 3 is Bevy's material bind group; binding 100 leaves 0-99 to the base
// `StandardMaterial` (the same slot `tree_sway` uses).
struct FacadeParams {
    // x: bay width (m). y: floor height (m).
    // z: bays per texture tile. w: floors per texture tile.
    dims: vec4<f32>,
};

@group(3) @binding(100) var<uniform> facade: FacadeParams;

// UV for a point on the unit cube, in tile units (0..1 = one whole tile).
//
// `local` is the vertex position in the mesh's own space, i.e. always ±0.5 on
// the box; `extent` is the instance's metric size on each axis.
fn facade_uv(local: vec3<f32>, normal: vec3<f32>, extent: vec3<f32>) -> vec2<f32> {
    let bay = max(facade.dims.x, 0.01);
    let storey = max(facade.dims.y, 0.01);
    let bays_per_tile = max(facade.dims.z, 1.0);
    let floors_per_tile = max(facade.dims.w, 1.0);

    // Whole floors, at least one: a degenerate scale must not collapse the
    // facade to a single texel.
    let floors = max(1.0, round(extent.y / storey));

    // v grows downward from the roofline, so a height that is not a whole
    // number of floors loses part of the GROUND floor rather than the top one.
    // The top row of windows therefore never gets sliced, and that is the row
    // the player actually sees against the sky.
    let v = (0.5 - local.y) * floors / floors_per_tile;

    if abs(normal.y) > 0.7 {
        // Roof / underside: a fixed point inside a wall band. `v = 0.94` of a
        // floor lands below the window row (windows occupy 0.25..0.70), so this
        // is plain wall in every style.
        return vec2<f32>(0.31, 0.94);
    }

    // Which local axis runs along the street is NOT a guess: `placement` sets
    // `yaw = atan2(-dir.z, dir.x)`, which maps local +X onto the street
    // direction and local +Z onto its normal. So `size.x` is the frontage — the
    // axis placement snapped to whole bays — and `size.z` is the depth.
    if abs(normal.x) > 0.7 {
        // Gable faces (normal across the street): `u` spans the DEPTH.
        //
        // Depth is deliberately not snapped to bays — the block-fit tests rely
        // on it being free to shrink — so these faces end on a partial bay,
        // which is what a terrace's party wall looks like from the side.
        let bays = max(1.0, round(extent.z / bay));
        return vec2<f32>((local.z + 0.5) * bays / bays_per_tile, v);
    }

    // Street-facing faces: `u` spans the FRONTAGE (`size.x`).
    let bays = max(1.0, round(extent.x / bay));
    return vec2<f32>((local.x + 0.5) * bays / bays_per_tile, v);
}

@vertex
fn vertex(vertex_no_morph: Vertex) -> VertexOutput {
    var out: VertexOutput;
    let instance_index = vertex_no_morph.instance_index;
    let world_from_local = mesh_functions::get_world_from_local(instance_index);

    let world_position = (world_from_local * vec4<f32>(vertex_no_morph.position, 1.0)).xyz;
    out.world_position = vec4<f32>(world_position, 1.0);
    out.position = position_world_to_clip(world_position);

    // Column lengths of the TRS matrix are the instance's metric scale. Any
    // uniform scale baked into the mesh would be folded in too, which is what
    // we want: the facade follows what is actually drawn.
    let extent = vec3<f32>(
        length(world_from_local[0].xyz),
        length(world_from_local[1].xyz),
        length(world_from_local[2].xyz),
    );

@if(VERTEX_NORMALS) {
    out.world_normal = mesh_functions::mesh_normal_local_to_world(
        vertex_no_morph.normal,
        instance_index,
    );
}

    // The UV is consumed by the base-colour, metallic/roughness and emissive maps
    // alike: `pbr_fragment.wesl` samples all three with `in.uv` unless a channel
    // override is set, and `StandardMaterial::uv_transform` starts as the
    // identity. The field only exists when the mesh carries UVs, which the shared
    // cube always does.
@if(VERTEX_UVS_A) {
    var facade_coords = vec2<f32>(0.5, 0.94);
@if(VERTEX_NORMALS) {
    facade_coords = facade_uv(vertex_no_morph.position, vertex_no_morph.normal, extent);
}
    out.uv = facade_coords;
}

@if(VERTEX_OUTPUT_INSTANCE_INDEX) {
    out.instance_index = instance_index;
}

    return out;
}
