// Cirrus dome shader (R5.6).
//
// The mesh is a large inside-out sphere centred on the camera, so `in.world_position`
// minus the camera position is the view direction for that pixel. Everything below is
// a function of that direction in WORLD space: the pattern stays put as the player
// walks and turns, which is what makes it read as a sheet several kilometres up.
//
// Shape: cirrus is stretched fibre. An isotropic fbm reads as cotton wool; stretching
// the noise domain along one axis (and shearing it by the same axis, so streaks curve)
// gives the long filaments. A second, coarser fbm gates the result so the sky has clear
// gaps rather than a uniform haze.

import bevy_pbr::render::forward_io::{Vertex, VertexOutput};
import bevy_pbr::render::mesh_functions;
// `mesh_view_bindings` supplies the `view` uniform at @group(0) @binding(0),
// which every mesh pipeline already binds. Importing `View` alone defines the
// type but binds nothing.
import bevy_pbr::render::mesh_view_bindings::view;

// Group 3 is bevy's material bind group (MATERIAL_BIND_GROUP_INDEX = 3).
@group(3) @binding(0) var<uniform> cirrus: CirrusUniform;

struct CirrusUniform {
    // xyz: unit direction toward the sun. w: amount (0..1).
    sun: vec4<f32>,
    // rgb: sky-scattered tint (linear). a: brightness of that light.
    tint: vec4<f32>,
    // x: stretch. y: scale. z: directionality. w: drift time (s).
    params: vec4<f32>,
};

// Cheap 3D value noise. A full gradient noise would be prettier, but cirrus is
// extremely low-contrast against the dusk sky and the extra cost is wasted.
fn hash3(p: vec3<f32>) -> f32 {
    var q = fract(p * 0.1031);
    q += dot(q, q.yzx + 33.33);
    return fract((q.x + q.y) * q.z);
}

fn value_noise(p: vec3<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    // Smoothstep weights: a plain mix leaves visible diagonal creases.
    let w = f * f * (3.0 - 2.0 * f);
    let c000 = hash3(i);
    let c100 = hash3(i + vec3<f32>(1.0, 0.0, 0.0));
    let c010 = hash3(i + vec3<f32>(0.0, 1.0, 0.0));
    let c110 = hash3(i + vec3<f32>(1.0, 1.0, 0.0));
    let c001 = hash3(i + vec3<f32>(0.0, 0.0, 1.0));
    let c101 = hash3(i + vec3<f32>(1.0, 0.0, 1.0));
    let c011 = hash3(i + vec3<f32>(0.0, 1.0, 1.0));
    let c111 = hash3(i + vec3<f32>(1.0, 1.0, 1.0));
    let x00 = mix(c000, c100, w.x);
    let x10 = mix(c010, c110, w.x);
    let x01 = mix(c001, c101, w.x);
    let x11 = mix(c011, c111, w.x);
    return mix(mix(x00, x10, w.y), mix(x01, x11, w.y), w.z);
}

fn fbm(p: vec3<f32>, octaves: i32) -> f32 {
    var sum = 0.0;
    var amp = 0.5;
    var q = p;
    for (var i = 0; i < octaves; i += 1) {
        sum += amp * value_noise(q);
        q *= 2.03;
        amp *= 0.5;
    }
    return sum;
}

// Standard mesh vertex stage: transform the dome's vertices and pass the world
// position through. The dome is a plain sphere; all the work is in the
// fragment stage.
@vertex
fn vertex(vertex: Vertex) -> VertexOutput {
    var out: VertexOutput;
    out.position = view.clip_from_world * mesh_functions::mesh_position_local_to_world(
        mesh_functions::get_world_from_local(vertex.instance_index),
        vec4<f32>(vertex.position, 1.0),
    );
    out.world_position = mesh_functions::mesh_position_local_to_world(
        mesh_functions::get_world_from_local(vertex.instance_index),
        vec4<f32>(vertex.position, 1.0),
    );
    out.world_normal = mesh_functions::mesh_normal_local_to_world(
        vertex.normal,
        vertex.instance_index,
    );
    return out;
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    let amount = cirrus.sun.w;
    if (amount <= 0.001) {
        discard;
    }

    // View direction, rebuilt from the PIXEL's clip position rather than
    // interpolated from the dome's vertices.
    //
    // Interpolating `world_position` across a 900 m dome with subdiv-3
    // triangles (edge ~135 m) makes the direction deviate by up to
    // 2.8e-3 rad from the true one, and the error changes abruptly across
    // every triangle edge -- which reads as seams radiating over the sky.
    // Unprojecting the pixel is exact for every fragment, seam-free, and it
    // also removes the dependence on how many triangles the dome has.
    // NDC from the fragment's pixel coordinate (viewport origin at 0,0 here).
    let ndc = vec2<f32>(in.position.xy / view.viewport.zw) * 2.0 - 1.0;

    // Bevy uses REVERSE-Z with an INFINITE far plane: depth 1.0 is the near
    // plane and 0.0 is the point at infinity. Unprojecting depth 0.0 divides
    // by `w == 0` and yields NaN/inf, which normalises to NaN -- and a NaN
    // direction fails the `dir.y` test below, so every fragment is discarded
    // and the cirrus silently disappears.
    //
    // Take two FINITE depths instead and use the difference: exact in
    // direction, immune to the degenerate far plane, and independent of where
    // the near plane happens to sit.
    let a_h = view.world_from_clip * vec4<f32>(ndc, 0.95, 1.0);
    let b_h = view.world_from_clip * vec4<f32>(ndc, 0.55, 1.0);
    let dir = normalize(b_h.xyz / b_h.w - a_h.xyz / a_h.w);
    // Cirrus lives above the horizon; everything below is the ground's business.
    if (dir.y < 0.02) {
        discard;
    }

    let params = cirrus.params;
    let drift = params.w;
    let scale = params.y;

    // Wind the sample point around the vertical axis so streaks follow the sky
    // rather than the screen, then stretch along one axis and shear it.
    let wind = vec3<f32>(0.82, 0.0, 0.57);
    var p = dir * shape_scale(scale);

    // Stretch: compress the coordinate ALONG the streak direction, so noise that
    // is isotropic becomes long in that direction. The shear term bends the
    // streaks instead of leaving them ruler-straight.
    let along = dot(p, wind);
    p -= wind * along * (1.0 - 1.0 / params.x);
    p += wind * sin(along * 1.7 + drift * 0.05) * params.z * 0.35;

    // Two layers at different frequencies and drift rates; their difference is
    // what gives fibres of uneven length.
    let fine = fbm(p + wind * drift * 0.02, 5);
    let coarse = fbm(p * 0.35 - wind * drift * 0.008 + 11.0, 3);

    // Gate: only where the coarse field is high enough. This produces the gaps
    // and hooked edges that make cirrus look drawn rather than painted.
    let gate = smoothstep(0.42, 0.78, coarse);
    var density = clamp((fine - 0.35) * 2.2, 0.0, 1.0) * gate;

    // Thin where the view grazes the horizon (more air, more occlusion) and
    // thicken toward the zenith.
    density *= smoothstep(0.02, 0.35, dir.y);
    density *= amount;

    if (density <= 0.002) {
        discard;
    }

    // Lighting: backlit cirrus glows near the sun and greys out away from it.
    let sun_dir = normalize(cirrus.sun.xyz);
    let to_sun = max(dot(dir, sun_dir), 0.0);
    let forward_scatter = pow(to_sun, 6.0) * 0.9 + pow(to_sun, 1.5) * 0.35;

    // The tint is the sky's own scattered light, so cirrus sits inside the dusk
    // palette instead of fighting it. A touch brighter than the sky keeps it
    // legible against the gradient.
    let sky = cirrus.tint.rgb * max(cirrus.tint.a, 0.001);
    let lit = mix(sky * 0.85, sky * 2.6 + vec3<f32>(0.25, 0.16, 0.10), forward_scatter);

    // Alpha grows with density but saturates well below 1: cirrus is never
    // opaque, and a solid patch would read as a low cloud.
    let alpha = clamp(density * 0.75, 0.0, 0.62);
    return vec4<f32>(lit, alpha);
}

// Maps the panel's `scale` onto the noise domain. Larger scale = finer detail.
fn shape_scale(scale: f32) -> f32 {
    return 1.0 + scale * 3.0;
}
