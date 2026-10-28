// The star field: one quad per star, placed at infinity, expanded to a fixed pixel size, radiance
// from catalogue illuminance. See `stars.rs` for the model and `docs/research/celestial.md` §Stars.
//
//   L = E / Ω · (3/π) · (1 − ρ²)² · colour · T_atm · exposure   [cd/m²]
//   Ω = (1.5 p)²,  p = 2 tan(fov_y / 2) / viewport_height  [rad per pixel]
//   T_atm = exp(−τ · X),  X = Kasten–Young airmass at the star's altitude

import bevy_pbr::render::mesh_view_bindings::view;

struct StarsParams {
    // Columns of the J2000 equatorial → world matrix, xyz.
    c0: vec4<f32>,
    c1: vec4<f32>,
    c2: vec4<f32>,
    // xyz: zenith optical depth per channel. w: unused (1).
    extinction_cloud: vec4<f32>,
    // xyz: unit world direction of the moon. w: cosine of its angular radius.
    moon: vec4<f32>,
    // x: sprite half-width, px. y: profile energy normaliser, 3/π.
    sprite: vec4<f32>,
}

@group(constants::MATERIAL_BIND_GROUP) @binding(0) var<uniform> params: StarsParams;

struct VertexInput {
    // J2000 unit direction (Mesh::ATTRIBUTE_POSITION).
    @location(0) j2000: vec3<f32>,
    // Quad corner, (±1, ±1).
    @location(1) corner: vec2<f32>,
    // rgb: linear colour at luminance 1. w: illuminance outside the atmosphere, lx.
    @location(2) colour_flux: vec4<f32>,
}

struct VertexOutput {
    @builtin(position) clip: vec4<f32>,
    @location(0) corner: vec2<f32>,
    @location(1) @interpolate(flat) radiance: vec3<f32>,
}

// Kasten & Young 1989: X = 1 / (sin h + 0.50572 (h° + 6.07995)^−1.6364).
fn airmass(sin_altitude: f32) -> f32 {
    let s = clamp(sin_altitude, 0.0, 1.0);
    let altitude_deg = degrees(asin(s));
    return 1.0 / (s + 0.50572 * pow(altitude_deg + 6.07995, -1.6364));
}

// Outside the clip volume: the quad is not rasterised.
const CULLED: vec4<f32> = vec4<f32>(0.0, 0.0, 2.0, 1.0);

@vertex
fn vertex(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.corner = in.corner;
    out.radiance = vec3<f32>(0.0);
    out.clip = CULLED;

    let world = params.c0.xyz * in.j2000.x + params.c1.xyz * in.j2000.y + params.c2.xyz * in.j2000.z;
    let sin_altitude = world.y;
    // Below the horizon the ground hides it; inside the moon's disc the moon hides it.
    if sin_altitude <= 0.0 || dot(world, params.moon.xyz) > params.moon.w {
        return out;
    }

    // A point at infinity: rotation only, then clip z = 0 (reverse-Z far plane, depth 0).
    let view_dir = (view.view_from_world * vec4<f32>(world, 0.0)).xyz;
    var clip = view.clip_from_view * vec4<f32>(view_dir, 1.0);
    if clip.w <= 1.0e-4 {
        return out;
    }
    let px = vec2<f32>(2.0 / view.viewport.z, 2.0 / view.viewport.w);
    clip = vec4<f32>(clip.xy + in.corner * params.sprite.x * px * clip.w, 0.0, clip.w);
    out.clip = clip;

    // Pixel angle at the view centre, rad.
    let pixel_angle = 2.0 / (view.clip_from_view[1][1] * view.viewport.w);
    let omega = pow(params.sprite.x * pixel_angle, 2.0);
    let transmittance = exp(-params.extinction_cloud.xyz * airmass(sin_altitude));
    out.radiance = in.colour_flux.rgb * (in.colour_flux.w * params.sprite.y / omega)
        * transmittance * params.extinction_cloud.w * view.exposure;
    return out;
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    let r2 = dot(in.corner, in.corner);
    if r2 >= 1.0 {
        discard;
    }
    let profile = (1.0 - r2) * (1.0 - r2);
    // Alpha 0: AlphaMode::Add is premultiplied blending, so this is purely added.
    return vec4<f32>(in.radiance * profile, 0.0);
}
