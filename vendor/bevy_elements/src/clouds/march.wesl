// The cloud ray march, one texel per k×k block of pixels (k = 4 or 2), one pixel of each block
// per frame. Output: premultiplied, exposed radiance (rgb) and transmittance (a). See
// docs/research/clouds.md ("Ray march", "Lighting at a fine sample", "Aerial perspective").
//
// A second entry point, `cloud_sky_probe`, marches the same clouds into a cube around the camera
// and composes it with Bevy's atmosphere cube: the environment map the scene's reflections and
// diffuse sky light come from (docs/research/sky-light.md). It binds no view, depth or aerial
// LUT; the functions it shares with the screen march take no screen position and return
// luminance (unexposed).

import bevy_render::view::View;
import bevy_pbr::render::utils::sample_cube_dir;
import bevy_elements::noise::hash31;
import bevy_elements::clouds::common::{
    CloudParams, Medium, MAX_MARCH_M, PI, altitude, cloud_medium, coarse_shape, detail_weight,
    ground_before_layer, is_orthographic, light_at, phase, sample_transmittance, scene_distance,
    shell_range, sky_through_clouds, slot_offset, view_ray,
};

@group(0) @binding(0) var<uniform> view: View;
@group(0) @binding(1) var<uniform> params: CloudParams;
@group(0) @binding(2) var base_tex: texture_3d<f32>;
@group(0) @binding(3) var detail_tex: texture_3d<f32>;
@group(0) @binding(4) var weather_tex: texture_2d<f32>;
@group(0) @binding(5) var transmittance_lut: texture_2d<f32>;
@group(0) @binding(6) var aerial_lut: texture_3d<f32>;
@group(0) @binding(7) var depth_tex: texture_depth_2d;
@group(0) @binding(8) var smp_repeat: sampler;
@group(0) @binding(9) var smp_clamp: sampler;
@group(0) @binding(10) var march_out: texture_storage_2d<rgba16float, write>;
@group(0) @binding(11) var sky_probe: texture_cube<f32>;
// The sky probe's own: its cloud state of the last frame and of this one (premultiplied
// luminance and transmittance per cube texel), and the composed cube.
@group(0) @binding(12) var probe_prev: texture_2d_array<f32>;
@group(0) @binding(13) var probe_next: texture_storage_2d_array<rgba16float, write>;
@group(0) @binding(14) var probe_out: texture_storage_2d_array<rgba16float, write>;

/// Directions over the upper hemisphere the sky's light is averaged from.
const SKY_SAMPLES: u32 = 16u;

/// The night sky's own colour: the *scotopic* one, a cool blue-green. Jensen et al. 2001 §7
/// on the blue shift — night scenes "exhibit a bluish tint" — and the rod's curve, whose peak
/// is at 507 nm.
const NIGHT_SKY_COLOUR: vec3<f32> = vec3<f32>(0.72, 0.95, 1.28);

/// The clear sky's irradiance on the layer, `π L̄`, with `L̄` the mean radiance of Bevy's sky over
/// the upper hemisphere (clouds.md Deviation 30): a spherical Fibonacci set uniform in solid
/// angle over the camera's atmosphere environment map (z flipped, cubemaps are left-handed; a
/// white cube without one), undimmed by the cloud shadow at the camera (`air.y`), plus the night
/// sky's neutral background behind the atmosphere (`atmo.w`).
fn sky_irradiance() -> vec3<f32> {
    var sum = vec3<f32>(0.0);
    for (var i = 0u; i < SKY_SAMPLES; i += 1u) {
        let y = (f32(i) + 0.5) / f32(SKY_SAMPLES);
        let r = sqrt(max(1.0 - y * y, 0.0));
        let phi = 2.3999632 * f32(i);
        let d = vec3<f32>(r * cos(phi), y, -r * sin(phi));
        sum += textureSampleLevel(sky_probe, smp_clamp, d, 0.0).rgb;
    }
    return PI * (sum / f32(SKY_SAMPLES) * params.air.y + vec3<f32>(params.atmo.w));
}

/// The ambient source of luminance `s` in the colour of the sky's irradiance `sky_e`: the
/// skylight in the cloud. The sunlight the cloud has scattered out of the beam is
/// `cloud_scatter`'s diffusion term (cloud-lighting.md Deviation 3).
fn ambient_light(sky_e: vec3<f32>, s: f32) -> vec3<f32> {
    let lum = dot(sky_e, vec3<f32>(0.2126, 0.7152, 0.0722));
    if !(lum > 1e-12) {
        return vec3<f32>(s);
    }
    return sky_e * (s / lum);
}

/// Consecutive empty fine samples after which the march returns to coarse steps (Nubis).
const EMPTY_LIMIT: u32 = 10u;
/// Transmittance below which the march stops (Högfeldt, Schneider's alpha → 1).
const MIN_TRANSMITTANCE: f32 = 0.01;
/// Tangent of the light-sample cone half angle.
const CONE_TAN: f32 = 0.1;
/// Bottom weight of the ambient gradient (ground bounce).
const AMBIENT_BOTTOM: f32 = 0.2;
/// Distance at which cloud extinction starts to fade toward the march limit, m.
const FADE_START_M: f32 = 16000.0;
/// Path length through the layer beyond `FADE_START_M` over which the far field builds up to its
/// mean cover (1 − 1/e of it at this length), m: a few cloud widths, so that under full cover
/// the sky does not show through.
const FAR_LENGTH_M: f32 = 4000.0;
/// Optical depth toward the light from the surface of the far cloud, along the vertical: the
/// order of the model stratus's column depth (6.04, Validation 7), set to 5 by comparison with the
/// marched cloud (Deviation 27 of the research note).
const FAR_TAU: f32 = 5.0;
/// Cover over which the far field turns from broken cloud to a uniform deck, whose light path
/// slants with the light.
const DECK_COVER_FROM: f32 = 0.6;
const DECK_COVER_TO: f32 = 0.95;
/// Lowest sine of the light's elevation the deck's slanted light path is built for.
const MIN_LIGHT_SINE: f32 = 0.1;
/// Ray elevation (sine) below the horizon over which the far veil fades out.
const HORIZON_BAND: f32 = 0.03;

fn medium_at(h: f32, xz: vec2<f32>, detail: f32) -> Medium {
    return cloud_medium(h, xz, detail, params, base_tex, detail_tex, weather_tex, smp_repeat);
}

/// Optical depth toward the light from the point `off` (offset from the camera), m⁻¹·m: six
/// samples in a cone, base shape only. Segment lengths 30, 60, 120, 240, 480 m and a far sample
/// at 1800 m covering 1740 m; times (1 + g) per sample. Each sample sits at a fixed golden-angle
/// azimuth around the light direction (Schneider's fixed cone kernel): the lighting is a smooth
/// function of position, so pixels do not carry independent noise.
fn light_optical_depth(off: vec3<f32>, l: vec3<f32>, cam: vec3<f32>) -> f32 {
    var dist = array<f32, 6>(15.0, 60.0, 150.0, 330.0, 690.0, 1800.0);
    var len = array<f32, 6>(30.0, 60.0, 120.0, 240.0, 480.0, 1740.0);
    let helper = select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), abs(l.y) > 0.99);
    let e1 = normalize(cross(l, helper));
    let e2 = cross(l, e1);
    var tau = 0.0;
    for (var j = 0u; j < 6u; j += 1u) {
        let phi = 2.39996323 * f32(j + 1u);
        let perp = cos(phi) * e1 + sin(phi) * e2;
        let o = off + l * dist[j] + perp * (CONE_TAN * dist[j]);
        let h = altitude(o.xz, cam.y + o.y, params.camera.w);
        let m = medium_at(h, cam.xz + o.xz, 0.0);
        tau += (1.0 + m.g) * m.sigma * len[j];
    }
    return tau;
}

/// Diffusion's extrapolation length in optical depths, `2 / (3 (1 − g))` with the forward lobe's
/// `g = 0.8` (Jensen et al. 2001, the boundary condition `φ − 2AD ∂φ/∂n = 0`, A = 1);
/// cloud-lighting.md fits 3.39 to the Monte Carlo slabs.
const DIFFUSE_EXTRAPOLATION: f32 = 3.3333333;
/// Albedo of the diffuse source, fitted to the Monte Carlo slabs (cloud-lighting.md).
const DIFFUSE_ALBEDO: f32 = 0.68;
/// How far below a sample (away from the light) the cloud behind it is sampled, m; the sample
/// stands for twice this.
const BELOW_M: f32 = 300.0;

/// Radiance of cloud lit by `sun` through a light path of optical depth `tau_l`, with `tau_b` of
/// cloud behind the sample along the light, seen at angle `cos_t` to the light that makes
/// cosine `mu_l` with the local vertical: single scattering with the dual-lobe phase (the
/// powder darkening near the lit surface) plus the multiply scattered light as photon diffusion
/// in a conservative slab, whose fluence falls linearly between extrapolated boundaries
/// `(τ_b + z) / (τ_l + τ_b + 2z)` (cloud-lighting.md, Model), and the ambient source `ambient`.
fn cloud_scatter(
    sun: vec3<f32>,
    tau_l: f32,
    tau_b: f32,
    mu_l: f32,
    cos_t: f32,
    powder_weight: f32,
    ambient: vec3<f32>,
) -> vec3<f32> {
    let powder = mix(1.0, 1.0 - exp(-2.0 * tau_l), powder_weight);
    let z = DIFFUSE_EXTRAPOLATION;
    let diffuse =
        DIFFUSE_ALBEDO / PI * sqrt(max(mu_l, 0.0)) * (tau_b + z) / (tau_l + tau_b + 2.0 * z);
    return sun * (powder * exp(-tau_l) * phase(cos_t) + diffuse) + ambient;
}

/// Optical depth of the cloud behind the point `off` along the light (away from it), m⁻¹·m: one
/// base-shape sample `BELOW_M` beyond it standing for `2 · BELOW_M`, times `1 + g`.
fn below_optical_depth(off: vec3<f32>, l: vec3<f32>, cam: vec3<f32>) -> f32 {
    let o = off - l * BELOW_M;
    let h = altitude(o.xz, cam.y + o.y, params.camera.w);
    let m = medium_at(h, cam.xz + o.xz, 0.0);
    return (1.0 + m.g) * m.sigma * 2.0 * BELOW_M;
}

/// Aerial perspective from the camera to `depth` along `dir`: the transmittance of the air
/// (rgb) and its in-scatter (rgb of the second return), luminance.
struct Aerial {
    transmittance: vec3<f32>,
    in_scatter: vec3<f32>,
}

/// Transmittance of the air from the camera at radius `o_r` to `depth` along `dir`. The LUT holds
/// the transmittance to the top of the atmosphere, which a descending ray never reaches (it meets
/// the ground): the air between two points is then taken along the reversed ray, which leaves
/// the atmosphere through the camera (Bruneton's `GetTransmittance`). A camera above the clouds
/// looking down sees them through that much air, not through the ground's (cloud-lighting.md
/// Deviation 1).
fn air_transmittance(dir: vec3<f32>, depth: f32, o_r: f32) -> vec3<f32> {
    let r_d = sqrt(max(o_r * o_r + 2.0 * o_r * depth * dir.y + depth * depth, 1.0));
    let mu_d = (o_r * dir.y + depth) / r_d;
    if dir.y < 0.0 {
        let t_from_far = sample_transmittance(transmittance_lut, smp_clamp, r_d, -mu_d, params);
        let t_from_cam = sample_transmittance(transmittance_lut, smp_clamp, o_r, -dir.y, params);
        return min(t_from_far / max(t_from_cam, vec3<f32>(1e-6)), vec3<f32>(1.0));
    }
    let t_cam = sample_transmittance(transmittance_lut, smp_clamp, o_r, dir.y, params);
    let t_far = sample_transmittance(transmittance_lut, smp_clamp, r_d, mu_d, params);
    return min(t_cam / max(t_far, vec3<f32>(1e-6)), vec3<f32>(1.0));
}

/// Aerial perspective through the screen position `uv`. In-scatter is Bevy's aerial-view LUT,
/// lit by the sun light that carries the beam only, plus the air lit by the cloud-scattered
/// diffuse light, `(1 − T) · L` for the uniform sky radiance `L` (Deviation 25 of the research
/// note).
fn aerial_perspective(uv: vec2<f32>, dir: vec3<f32>, depth: f32, o_r: f32) -> Aerial {
    let slices = params.atmo.z;
    let range = params.atmo.y;
    let slice = clamp(depth / range - 0.5 / slices, 0.0, 1.0);
    let lut = textureSampleLevel(aerial_lut, smp_clamp, vec3<f32>(uv, slice), 0.0);
    let t_ap = air_transmittance(dir, depth, o_r);
    let beam = exp(lut.rgb) * clamp(depth / (range / slices), 0.0, 1.0);
    return Aerial(t_ap, beam + (vec3<f32>(1.0) - t_ap) * params.sky.yzw);
}

/// Aerial perspective along a sky-probe direction, where there is no screen to index Bevy's
/// aerial-view LUT by (sky-light.md Model 2): the in-scatter is the clear sky's radiance `sky`
/// along `dir` (the in-scatter of the whole path to the top of the atmosphere, lit as the LUT
/// is) times the share of the path's extinction that lies in front of the cloud, plus the air
/// lit by the cloud-scattered diffuse light as on the screen.
fn aerial_along(dir: vec3<f32>, depth: f32, o_r: f32, sky: vec3<f32>) -> Aerial {
    let t_ap = air_transmittance(dir, depth, o_r);
    let t_top = sample_transmittance(transmittance_lut, smp_clamp, o_r, dir.y, params);
    let share = clamp(
        (vec3<f32>(1.0) - t_ap) / max(vec3<f32>(1.0) - t_top, vec3<f32>(1e-4)),
        vec3<f32>(0.0),
        vec3<f32>(1.0),
    );
    return Aerial(t_ap, sky * share + (vec3<f32>(1.0) - t_ap) * params.sky.yzw);
}

/// A ray that meets the planet before the layer (past the edge of a finite ground): no cloud, and
/// the air to the ground is the camera's distance fog, as on every mesh, `scene · T + fog · (1 −
/// T)` with `T = exp(−σ d)` (bevy_pbr's atmospheric fog with equal extinction and in-scattering).
/// Bevy's sky pixel there is the planet lit by the sun light, which under cloud carries only the
/// beam; the fog's colour carries the diffuse light. The transmittance LUT cannot give this path:
/// it holds rays that leave the atmosphere (Deviation 28 of the research note). The distance is
/// floored at the march range: a below-horizon pixel with no mesh shows ground the fog has fully
/// carried — nothing nearer than the far field is visible beyond a finite terrain, and the bare
/// planet within the march range would otherwise show through as a dark band at the horizon
/// (measured 68 DN on `ground_cover` at 6 m eye). Luminance: the fog colour is display-referred,
/// so it is divided by the camera's exposure.
fn ground_haze(ground_m: f32, sky: vec3<f32>) -> vec4<f32> {
    // The weather's air over the far field, as the composite fogs a sky pixel (`air.x` is the
    // extinction above clear air, so a clear sky leaves this exactly 0): what the slab shows is
    // the atmosphere in that column seen through the weather, converging on the fog colour as
    // the weather thickens — the same law the ground below obeys.
    let air = 1.0 - exp(-params.air.x * MAX_MARCH_M);
    let colour = mix(sky, params.haze.xyz / max(params.probe.x, 1e-30), air);
    let t = exp(-params.haze.w * max(ground_m, MAX_MARCH_M));
    return vec4<f32>(colour * (1.0 - t), t);
}

/// The sky's own radiance `HORIZON_BAND` above the horizon in the ray's azimuth, luminance: the
/// air a below-horizon ray converges on (`ground_haze`). Read from the sky cube, which holds the
/// atmosphere (and the cloud, once composed), so a slab that fills the frame past a finite
/// terrain is the sky in *that column* rather than one mean over the whole horizon: a dusk
/// horizon is not the same colour every way round, and a mean of it is a band of its own
/// (`docs/research/horizon-colour.md`). The direction is raised above the horizon so the sample
/// is sky and not the cube's own ground.
fn horizon_sky(dir: vec3<f32>) -> vec3<f32> {
    let flat = vec3<f32>(dir.x, 0.0, dir.z);
    let along = select(vec3<f32>(0.0, 0.0, -1.0), normalize(flat), length(flat) > 1e-4);
    let world = along * cos(HORIZON_SAMPLE_BAND) + vec3<f32>(0.0, sin(HORIZON_SAMPLE_BAND), 0.0);
    // Cubemaps are left-handed: the cube direction is the world's with z negated.
    let cube = vec3<f32>(world.x, world.y, -world.z);
    // The cube is lit by the sun light, which carries only the cloud's beam `T̄`, so it holds the
    // clear sky dimmed to that; `sky_gain` (`params.sky.x`) restores the dome to the total light,
    // exactly as the composite restores a sky pixel (`clouds.md` Deviation 25). The night sky's
    // own light lies behind the atmosphere, so it is added rather than gained (`params.atmo.w`).
    return textureSampleLevel(sky_probe, smp_clamp, cube, 0.0).rgb * params.sky.x
        + vec3<f32>(params.atmo.w) * NIGHT_SKY_COLOUR;
}

/// Elevation of the band a below-horizon ray reads the sky from, rad: 0.5°. The cube's texel at
/// the horizon of a 128² face spans about 0.7°, so a fetch closer than that bilinearly averages
/// the cube's own ground in and the slab goes dark (measured at 0.1°: 12.9 against a sky of 14.2).
/// The cost of sampling 0.5° up the atmosphere's gradient is 1–2 % (below).
const HORIZON_SAMPLE_BAND: f32 = 0.0087;

/// Where along a ray the far veil's cloud lies: about one build-up length past where it starts.
fn veil_depth(shell: vec2<f32>) -> f32 {
    return max(shell.x, FADE_START_M) + FAR_LENGTH_M;
}

/// The cloud beyond the march and its fade: a uniform veil of the layer's mean cover over the
/// path through the layer past `FADE_START_M`, lit as the marched base is lit where the march
/// fades and seen through the air `air` to the cloud's own distance (`veil_depth`).
/// Premultiplied luminance (rgb) and transmittance (a). Under full cover the horizon keeps the
/// cloud; at cover 0.05 it stays clear sky. Smooth in the ray, so it cannot alias (Deviations 26
/// and 27 of the research note). A ray that meets the planet first sees no cloud, only the air
/// to the ground (`ground_haze`).
fn far_veil(dir: vec3<f32>, shell: vec2<f32>, o_r: f32, air: Aerial, sky_e: vec3<f32>) -> vec4<f32> {
    let ground = ground_before_layer(o_r, dir.y, shell.x, params.camera.w);
    if ground > 0.0 {
        return ground_haze(ground, horizon_sky(dir));
    }
    let t_start = max(shell.x, FADE_START_M);
    let length = shell.y - t_start;
    let above = smoothstep(-HORIZON_BAND, 0.0, dir.y);
    let a = params.layer.w * (1.0 - exp(-max(length, 0.0) / FAR_LENGTH_M)) * above;
    if a <= 0.0 {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }
    let l = params.light.xyz;
    // The surface the march sees at grazing angles is the base of the cloud: the optical depth
    // toward the viewer reaches one within metres of it, so the sample is at the base, where the
    // march's ambient gradient is `AMBIENT_BOTTOM`. A uniform deck lets the light in along the
    // slant of its column; a broken layer lights its sides through about the column depth.
    let h_b = params.layer.x;
    let off = dir * t_start;
    let mu_s = (dot(off.xz, l.xz) + (o_r + off.y) * l.y) / (params.camera.w + h_b);
    let sun = light_at(transmittance_lut, smp_clamp, h_b, mu_s, params);
    let deck = smoothstep(DECK_COVER_FROM, DECK_COVER_TO, params.layer.w);
    let tau_l = FAR_TAU * mix(1.0, 1.0 / max(l.y, MIN_LIGHT_SINE), deck) * (1.0 + params.weather.w);
    let cos_t = dot(dir, l);
    // Behind a broken layer's lit side lies about as much cloud as in front of it; below a
    // uniform deck's base, none.
    let radiance = cloud_scatter(
        sun,
        tau_l,
        FAR_TAU * (1.0 - deck),
        mu_s,
        cos_t,
        clamp(0.5 - 0.5 * cos_t, 0.0, 1.0),
        ambient_light(sky_e, params.light_e.w * AMBIENT_BOTTOM),
    );
    return vec4<f32>(a * (radiance * air.transmittance + air.in_scatter), 1.0 - a);
}

/// What a march along a ray gathered: premultiplied scattered luminance and transmittance, and
/// the transmittance-weighted sum of its sample depths.
struct Integrated {
    radiance: vec3<f32>,
    transmittance: f32,
    depth_sum: f32,
    depth_weight: f32,
}

/// March the cloud layer from `cam` along `dir` between `t0` and `t1` (metres) in about `steps`
/// primary steps, the samples offset by `jitter` (0–1) of a step, the ambient in the colour of
/// the sky irradiance `sky_e`.
fn integrate(
    cam: vec3<f32>,
    dir: vec3<f32>,
    t0: f32,
    t1: f32,
    jitter: f32,
    sky_e: vec3<f32>,
    steps: u32,
) -> Integrated {
    let big_r = params.camera.w;
    let h_b = params.layer.x;
    let h_t = params.layer.y;
    let l = params.light.xyz;
    let n = f32(steps);
    let dt = max(clamp((t1 - t0) / n, 25.0, 400.0), (t1 - t0) / (1.5 * n));
    let cos_t = dot(dir, l);
    let detail = detail_weight(dt);
    let powder_weight = clamp(0.5 - 0.5 * cos_t, 0.0, 1.0);

    var radiance = vec3<f32>(0.0);
    var transmittance = 1.0;
    var depth_sum = 0.0;
    var depth_weight = 0.0;
    var t = t0 + dt * jitter;
    var coarse = true;
    var empty = 0u;
    var resume = t;
    let max_iterations = steps * 3u;
    for (var it = 0u; it < max_iterations; it += 1u) {
        if t >= t1 || transmittance < MIN_TRANSMITTANCE {
            break;
        }
        let off = dir * t;
        let h = altitude(off.xz, cam.y + off.y, big_r);
        let xz = cam.xz + off.xz;
        if coarse {
            let shape = coarse_shape(h, xz, params, base_tex, weather_tex, smp_repeat);
            if shape > 0.0 {
                coarse = false;
                empty = 0u;
                t = max(t - 2.0 * dt, resume);
            } else {
                t += 2.0 * dt;
            }
            continue;
        }
        var m = medium_at(h, xz, detail);
        // Distant cloud thins out with distance instead of ending in a hard wall at the march
        // limit (Deviation 11 of the research note).
        m.sigma *= 1.0 - smoothstep(FADE_START_M, MAX_MARCH_M, t);
        if m.sigma <= 0.0 {
            empty += 1u;
            if empty >= EMPTY_LIMIT {
                coarse = true;
                resume = t + dt;
            }
            t += dt;
            continue;
        }
        empty = 0u;
        // Each sample stands for the segment of length dt it starts (the jitter shifts the
        // sample inside its segment, not the segment).
        let seg = clamp(t1 - (t - dt * jitter), 0.0, dt);
        let step_t = exp(-m.sigma * seg);

        let mu_s = (dot(off.xz, l.xz) + (big_r + cam.y + off.y) * l.y) / (big_r + h);
        let sun = light_at(transmittance_lut, smp_clamp, h, mu_s, params);
        let tau_l = light_optical_depth(off, l, cam);
        let tau_b = below_optical_depth(off, l, cam);
        let f = (h - h_b) / (h_t - h_b);
        var scatter = cloud_scatter(
            sun,
            tau_l,
            tau_b,
            mu_s,
            cos_t,
            powder_weight,
            ambient_light(sky_e, params.light_e.w * (AMBIENT_BOTTOM + (1.0 - AMBIENT_BOTTOM) * f)),
        );
        if params.flash.w > 0.0 {
            let d = length(cam + off - params.flash.xyz);
            let source = params.flash.w / (4.0 * PI * max(d * d, 2500.0))
                * exp(-0.5 * m.sigma * min(d, 150.0));
            scatter += vec3<f32>(source);
        }
        radiance += transmittance * scatter * (1.0 - step_t);
        depth_sum += transmittance * t;
        depth_weight += transmittance;
        transmittance *= step_t;
        t += dt;
    }
    // A march that stopped on its transmittance floor met an opaque cloud: the rest of the
    // column is assumed to scatter like what was met, and nothing behind shows through — the
    // sun disc is 10⁵ times the sky, so its last 1 % would still burn white in the cloud
    // (Schneider & Vos 2015: alpha 1 at the exit; broken-cloud-light.md, Model 4).
    if transmittance < MIN_TRANSMITTANCE {
        radiance /= 1.0 - transmittance;
        transmittance = 0.0;
    }
    return Integrated(radiance, transmittance, depth_sum, depth_weight);
}

/// The marched cloud seen through the air `air` to its mean depth, fading into the haze toward
/// the march range, over the far field `veil` behind it: premultiplied luminance and
/// transmittance. `r` met cloud (`depth_weight > 0`).
fn finish_ray(r: Integrated, air: Aerial, veil: vec4<f32>) -> vec4<f32> {
    let depth = r.depth_sum / r.depth_weight;
    let fade = smoothstep(20000.0, 32000.0, depth);
    let rgb = (1.0 - fade) * (r.radiance * air.transmittance + air.in_scatter * (1.0 - r.transmittance));
    let alpha = 1.0 - (1.0 - fade) * (1.0 - r.transmittance);
    // The far field behind it: the cloud in front covers it by `1 − alpha`.
    return vec4<f32>(rgb + alpha * veil.rgb, alpha * veil.a);
}

/// Luminance to the view's display-referred value.
fn exposed(c: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(c.rgb * view.exposure, c.a);
}

@compute @workgroup_size(8, 8, 1)
fn cloud_march(@builtin(global_invocation_id) id: vec3<u32>) {
    let low = params.sizes.zw;
    if any(id.xy >= low) {
        return;
    }
    let k = params.counts.z;
    let size = params.sizes.xy;
    let frame = params.counts.x;
    let px = min(id.xy * k + slot_offset(frame, k), size - vec2<u32>(1u));
    let uv = (vec2<f32>(px) + 0.5) / vec2<f32>(size);
    let orthographic = is_orthographic(view.clip_from_view);
    let ray = view_ray(uv, view.world_from_clip, view.world_position, orthographic, params.camera.xyz, params.units.x);
    // The planet's centre lies straight below the ray's origin: the eye of a perspective view,
    // the pixel's point on the focus plane of an orthographic one.
    let cam = ray.origin;
    let dir = ray.dir;
    let big_r = params.camera.w;
    let o_r = big_r + cam.y;
    let h_b = params.layer.x;
    let h_t = params.layer.y;

    let shell = shell_range(o_r, dir.y, h_b, h_t, big_r);
    let t0 = shell.x;
    let t1 = min(shell.y, MAX_MARCH_M);
    if t1 <= t0 {
        let air = aerial_perspective(uv, dir, veil_depth(shell), o_r);
        textureStore(march_out, vec2<i32>(id.xy), exposed(far_veil(dir, shell, o_r, air, sky_irradiance())));
        return;
    }

    // A block whose pixels all show surfaces nearer than the cloud entry shows no cloud.
    var hidden = true;
    for (var j = 0u; j < k && hidden; j += 1u) {
        for (var i = 0u; i < k && hidden; i += 1u) {
            let q = min(id.xy * k + vec2<u32>(i, j), size - vec2<u32>(1u));
            let quv = (vec2<f32>(q) + 0.5) / vec2<f32>(size);
            let d = textureLoad(depth_tex, vec2<i32>(q), 0);
            let q_ray = view_ray(quv, view.world_from_clip, view.world_position, orthographic, params.camera.xyz, params.units.x);
            let q_shell = shell_range(big_r + q_ray.origin.y, q_ray.dir.y, h_b, h_t, big_r);
            if scene_distance(quv, d, view.world_from_clip, q_ray, params.units.x) >= q_shell.x {
                hidden = false;
            }
        }
    }
    if hidden && params.misc.y < 0.5 {
        textureStore(march_out, vec2<i32>(id.xy), vec4<f32>(0.0, 0.0, 0.0, 1.0));
        return;
    }

    let l = params.light.xyz;
    if params.misc.y > 1.5 {
        // Debug: the light at the entry point, for the sun-colour test.
        let o = dir * t0;
        let h = altitude(o.xz, cam.y + o.y, big_r);
        let mu = (dot(o.xz, l.xz) + (big_r + cam.y + o.y) * l.y) / (big_r + h);
        let e = light_at(transmittance_lut, smp_clamp, h, mu, params) / max(params.light_e.xyz, vec3<f32>(1e-6));
        textureStore(march_out, vec2<i32>(id.xy), vec4<f32>(e, 1.0));
        return;
    }

    let jitter = hash31(vec3<i32>(i32(px.x), i32(px.y), i32(frame)));
    let sky_e = sky_irradiance();
    let r = integrate(cam, dir, t0, t1, jitter, sky_e, params.counts.y);
    let veil = far_veil(dir, shell, o_r, aerial_perspective(uv, dir, veil_depth(shell), o_r), sky_e);
    if r.depth_weight <= 0.0 {
        textureStore(march_out, vec2<i32>(id.xy), exposed(veil));
        return;
    }
    // Aerial perspective at the mean cloud depth, then the fade into haze.
    let air = aerial_perspective(uv, dir, r.depth_sum / r.depth_weight, o_r);
    textureStore(march_out, vec2<i32>(id.xy), exposed(finish_ray(r, air, veil)));
}

/// Primary steps of a sky-probe ray against a screen ray: the probe is prefiltered into a cube
/// for reflections and sky light, so its rays take half the steps (Unreal's reflection sample
/// count scale on volumetric clouds; Hillaire 2016 marches the reflection view at a quarter of
/// the resolution), never fewer than this floor.
const PROBE_STEP_DIVISOR: u32 = 2u;
const PROBE_MIN_STEPS: u32 = 16u;

/// The clouds along a sky-probe direction that reaches the layer: premultiplied luminance and
/// transmittance, as `cloud_march` gives a sky pixel before exposure.
fn probe_ray(
    cam: vec3<f32>,
    dir: vec3<f32>,
    shell: vec2<f32>,
    o_r: f32,
    sky: vec3<f32>,
    jitter: f32,
) -> vec4<f32> {
    let sky_e = sky_irradiance();
    let veil = far_veil(dir, shell, o_r, aerial_along(dir, veil_depth(shell), o_r, sky), sky_e);
    let t0 = shell.x;
    let t1 = min(shell.y, MAX_MARCH_M);
    if t1 <= t0 {
        return veil;
    }
    let steps = max(params.counts.y / PROBE_STEP_DIVISOR, PROBE_MIN_STEPS);
    let r = integrate(cam, dir, t0, t1, jitter, sky_e, steps);
    if r.depth_weight <= 0.0 {
        return veil;
    }
    return finish_ray(r, aerial_along(dir, r.depth_sum / r.depth_weight, o_r, sky), veil);
}

/// The sky probe: one thread per texel of the cube (`z` the face). A texel in this frame's slot
/// of its k×k block marches its direction and blends the result into its history; every texel
/// is composed with Bevy's atmosphere cube as the screen composite composes a sky pixel
/// (`sky_through_clouds`), or, toward the ground, is that cube's ground restored to the dome's
/// total light by the sky gain (sky-light.md, Model 1–4 and 6). Luminance throughout.
@compute @workgroup_size(8, 8, 1)
fn cloud_sky_probe(@builtin(global_invocation_id) id: vec3<u32>) {
    let n = u32(params.probe.z);
    if id.x >= n || id.y >= n || id.z >= 6u {
        return;
    }
    let texel = vec2<i32>(id.xy);
    let face = i32(id.z);
    let uv = (vec2<f32>(id.xy) + 0.5) / f32(n);
    // Cubemaps are left-handed: the world direction of a texel is its cube direction with z
    // negated (Bevy's `atmosphere/environment.wesl`), and the cube is sampled at the cube one.
    let c = sample_cube_dir(uv, id.z);
    let dir = vec3<f32>(c.x, c.y, -c.z);
    let sky = textureSampleLevel(sky_probe, smp_clamp, c, 0.0).rgb;
    let cam = params.camera.xyz;
    let big_r = params.camera.w;
    let o_r = big_r + cam.y;
    let shell = shell_range(o_r, dir.y, params.layer.x, params.layer.y, big_r);
    if ground_before_layer(o_r, dir.y, shell.x, big_r) > 0.0 {
        textureStore(probe_next, texel, face, vec4<f32>(0.0, 0.0, 0.0, 1.0));
        textureStore(probe_out, texel, face, vec4<f32>(sky * params.sky.x, 1.0));
        return;
    }
    var cloud = textureLoad(probe_prev, texel, face, 0);
    let k = u32(params.probe.y);
    let frame = params.counts.x;
    if all(id.xy % k == slot_offset(frame, k)) {
        let jitter = hash31(vec3<i32>(i32(id.x + id.z * n), i32(id.y), i32(frame)));
        cloud = mix(cloud, probe_ray(cam, dir, shell, o_r, sky, jitter), params.probe.w);
    }
    textureStore(probe_next, texel, face, cloud);
    let haze = params.haze.xyz / max(params.probe.x, 1e-30);
    let s = sky_through_clouds(cloud, shell, params, haze);
    // The night sky's own light, behind the atmosphere, so the cloud does not dim it. The
    // cube is display-referred (`params.probe.x` is the camera's exposure), so the radiance
    // is converted with it, as the fog's haze is. Without it a moonless night is a black
    // screen: the sky is lit and the ground on it is not (`docs/research/night-sky-light.md`).
    let night = vec3<f32>(params.atmo.w) * NIGHT_SKY_COLOUR * params.probe.x;
    textureStore(probe_out, texel, face, vec4<f32>(s.rgb + sky * s.a + night, 1.0));
}
