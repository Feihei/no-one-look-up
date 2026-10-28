// The moon disc: a quad placed at infinity whose fragment shader draws a Lommel–Seeliger-lit
// sphere from an equirectangular albedo map. See `moon.rs` for the model and
// `docs/research/celestial.md` §Moon disc.
//
//   L = A · E_sun/π · (2 μ0 / (μ0 + μ) + 0.0002 (1 − k)) · T_atm · exposure
//   A = Ā · texel / 128                                                         [cd/m²]
//
// `exposure` is the darker of the scene's and that of an eye adapted to the lit lunar surface
// (local adaptation, Larson et al. 1997): a night exposure would push the whole disc past white,
// and the dimmer earthshine side would then read as the brighter, textured one.

import bevy_pbr::render::mesh_view_bindings::view;

struct MoonParams {
    // xyz: unit world direction to the moon. w: angular radius, rad.
    moon: vec4<f32>,
    // xyz: moon-fixed +Z (north), unit, perpendicular to `moon`. w: illuminated fraction k.
    north: vec4<f32>,
    // xyz: unit world direction to the sun. w: unused (1).
    sun_cloud: vec4<f32>,
    // xyz: zenith optical depth per channel. w: exposure adapted to the lit lunar surface.
    extinction: vec4<f32>,
    // x: albedo per unit of the unorm texel (Ā · 255 / 128). y: sun illuminance, lx.
    // z: earthshine factor. w: quad margin, px.
    photometry: vec4<f32>,
}

@group(constants::MATERIAL_BIND_GROUP) @binding(0) var<uniform> params: MoonParams;
@group(constants::MATERIAL_BIND_GROUP) @binding(1) var albedo_map: texture_2d<f32>;
@group(constants::MATERIAL_BIND_GROUP) @binding(2) var albedo_sampler: sampler;

struct VertexInput {
    // Quad corner (±1, ±1, 0) (Mesh::ATTRIBUTE_POSITION).
    @location(0) corner: vec3<f32>,
}

struct VertexOutput {
    @builtin(position) clip: vec4<f32>,
    // Tangent-plane offset from the moon's centre along screen-right and screen-up, at unit
    // distance (= radians, small-angle).
    @location(0) tangent: vec2<f32>,
}

const PI: f32 = 3.14159265358979;
// Texels per radian of longitude in the 512-wide map.
const TEXELS_PER_RADIAN: f32 = 512.0 / (2.0 * PI);

// Pixel angle at the view centre, rad: 2 tan(fov_y / 2) / viewport height.
fn pixel_angle() -> f32 {
    return 2.0 / (view.clip_from_view[1][1] * view.viewport.w);
}

// Kasten & Young 1989: X = 1 / (sin h + 0.50572 (h° + 6.07995)^−1.6364).
fn airmass(sin_altitude: f32) -> f32 {
    let s = clamp(sin_altitude, 0.0, 1.0);
    let altitude_deg = degrees(asin(s));
    return 1.0 / (s + 0.50572 * pow(altitude_deg + 6.07995, -1.6364));
}

@vertex
fn vertex(in: VertexInput) -> VertexOutput {
    let m = params.moon.xyz;
    let up = params.north.xyz;
    // Screen-right for a camera looking along `m` with `up` up; also lunar east.
    let right = cross(m, up);
    let half = tan(params.moon.w) + params.photometry.w * pixel_angle();
    let tangent = in.corner.xy * half;
    let world = m + right * tangent.x + up * tangent.y;

    // A point at infinity: rotation only, then clip z = 0 (reverse-Z far plane, depth 0).
    let view_dir = (view.view_from_world * vec4<f32>(world, 0.0)).xyz;
    let clip = view.clip_from_view * vec4<f32>(view_dir, 1.0);

    var out: VertexOutput;
    // Behind the camera: outside the clip volume.
    if clip.w <= 1.0e-4 {
        out.clip = vec4<f32>(0.0, 0.0, 2.0, 1.0);
    } else {
        out.clip = vec4<f32>(clip.xy, 0.0, clip.w);
    }
    out.tangent = tangent;
    return out;
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    let m = params.moon.xyz;
    let up = params.north.xyz;
    let right = cross(m, up);
    let tan_rho = tan(params.moon.w);
    let px = pixel_angle();

    // Orthographic unit sphere in the frame (right, up, toward the observer).
    let r = in.tangent / tan_rho;
    let r2 = dot(r, r);
    let mu = sqrt(max(1.0 - r2, 0.0));
    let n = vec3<f32>(r, mu);

    // Selenographic latitude and longitude of the surface point; +X toward the observer,
    // +Y east = screen-right, +Z north = screen-up.
    let lat = asin(clamp(n.y, -1.0, 1.0));
    let lon = atan2(n.x, n.z);
    let uv = vec2<f32>(lon / (2.0 * PI) + 0.5, 0.5 - lat / PI);
    // Texels covered by one pixel, stretched toward the limb: pick the mip by hand so the limb's
    // exploding derivatives do not select the 1×1 level.
    let arc_per_pixel = px / tan_rho / max(mu, 0.25);
    let lod = log2(max(arc_per_pixel * TEXELS_PER_RADIAN, 1.0));
    let albedo = textureSampleLevel(albedo_map, albedo_sampler, uv, lod).r * params.photometry.x;

    // Lommel–Seeliger, the sun in the same frame.
    let sun = params.sun_cloud.xyz;
    let s = vec3<f32>(dot(sun, right), dot(sun, up), dot(sun, -m));
    let mu0 = max(dot(n, s), 0.0);
    var lommel = 0.0;
    if mu0 + mu > 1.0e-6 {
        lommel = 2.0 * mu0 / (mu0 + mu);
    }
    let earthshine = params.photometry.z * (1.0 - params.north.w);
    let radiance = albedo * params.photometry.y / PI * (lommel + earthshine);

    // Anti-aliased limb: coverage from the distance to the edge in pixels.
    let edge_px = (1.0 - sqrt(r2)) * tan_rho / px;
    let coverage = clamp(edge_px + 0.5, 0.0, 1.0);

    let transmittance = exp(-params.extinction.xyz * airmass(m.y));
    let exposure = min(view.exposure, params.extinction.w);
    let colour = radiance * transmittance * params.sun_cloud.w * exposure * coverage;
    // Alpha 0: AlphaMode::Add is premultiplied blending, so this is purely added.
    return vec4<f32>(colour, 0.0);
}
