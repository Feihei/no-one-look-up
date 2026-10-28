// The cloud ray march, one texel per k×k block of pixels (k = 4 or 2), one pixel of each block
// per frame. Output: premultiplied, exposed radiance (rgb) and transmittance (a). See
// docs/research/clouds.md ("Ray march", "Lighting at a fine sample", "Aerial perspective").

import bevy_render::view::View;
import bevy_elements::noise::hash31;
import bevy_elements::clouds::common::{
    CloudParams, Medium, MAX_MARCH_M, PI, altitude, cloud_medium, coarse_shape,
    ground_before_layer, light_at, phase_octave, ray_dir, sample_transmittance, scene_distance,
    shell_range, slot_offset,
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

fn medium_at(h: f32, xz: vec2<f32>, detail: bool) -> Medium {
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
        let m = medium_at(h, cam.xz + o.xz, false);
        tau += (1.0 + m.g) * m.sigma * len[j];
    }
    return tau;
}

/// Radiance of cloud lit by `sun` through a light path of optical depth `tau_l`, seen at angle
/// `cos_t` to the light: three octaves of multiple scattering (weight a^n, light-path extinction
/// b^n, phase g c^n) and the ambient source `ambient` (the gradient already applied).
fn cloud_scatter(sun: vec3<f32>, tau_l: f32, cos_t: f32, powder_weight: f32, ambient: f32) -> vec3<f32> {
    let powder = mix(1.0, 1.0 - exp(-2.0 * tau_l), powder_weight);
    return sun * (
        powder * exp(-tau_l) * phase_octave(cos_t, 1.0)
        + 0.5 * exp(-0.5 * tau_l) * phase_octave(cos_t, 0.5)
        + 0.25 * exp(-0.25 * tau_l) * phase_octave(cos_t, 0.25)
    ) + vec3<f32>(ambient);
}

/// Aerial perspective from the camera to `depth` along `dir` (through the screen position `uv`):
/// the transmittance of the air (rgb) and its in-scatter (rgb of the second return). In-scatter
/// is Bevy's aerial-view LUT, lit by the sun light that carries the beam only, plus the air lit
/// by the cloud-scattered diffuse light, `(1 − T) · L` for the uniform sky radiance `L`
/// (Deviation 25 of the research note).
struct Aerial {
    transmittance: vec3<f32>,
    in_scatter: vec3<f32>,
}

fn aerial_perspective(uv: vec2<f32>, dir: vec3<f32>, depth: f32, o_r: f32) -> Aerial {
    let slices = params.atmo.z;
    let range = params.atmo.y;
    let slice = clamp(depth / range - 0.5 / slices, 0.0, 1.0);
    let lut = textureSampleLevel(aerial_lut, smp_clamp, vec3<f32>(uv, slice), 0.0);
    let r_d = sqrt(max(o_r * o_r + 2.0 * o_r * depth * dir.y + depth * depth, 1.0));
    let mu_d = (o_r * dir.y + depth) / r_d;
    let t_cam = sample_transmittance(transmittance_lut, smp_clamp, o_r, dir.y, params);
    let t_far = sample_transmittance(transmittance_lut, smp_clamp, r_d, mu_d, params);
    let t_ap = min(t_cam / max(t_far, vec3<f32>(1e-6)), vec3<f32>(1.0));
    let beam = exp(lut.rgb) * clamp(depth / (range / slices), 0.0, 1.0);
    return Aerial(t_ap, beam + (vec3<f32>(1.0) - t_ap) * params.sky.yzw);
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
/// (measured 68 DN on `ground_cover` at 6 m eye).
fn ground_haze(ground_m: f32) -> vec4<f32> {
    let t = exp(-params.haze.w * max(ground_m, MAX_MARCH_M));
    return vec4<f32>(params.haze.xyz * (1.0 - t), t);
}

/// The cloud beyond the march and its fade: a uniform veil of the layer's mean cover over the
/// path through the layer past `FADE_START_M`, lit as the marched base is lit where the march
/// fades and seen through the air to the cloud's own distance. Premultiplied exposed radiance
/// (rgb) and transmittance (a). Under full cover the horizon keeps the cloud; at cover 0.05 it
/// stays clear sky. Smooth in the ray, so it cannot alias (Deviations 26 and 27 of the research
/// note). A ray that meets the planet first sees no cloud, only the air to the ground
/// (`ground_haze`).
fn far_veil(uv: vec2<f32>, dir: vec3<f32>, shell: vec2<f32>, o_r: f32) -> vec4<f32> {
    let ground = ground_before_layer(o_r, dir.y, shell.x, params.camera.w);
    if ground > 0.0 {
        return ground_haze(ground);
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
    let radiance = cloud_scatter(
        sun,
        tau_l,
        cos_t,
        clamp(0.5 - 0.5 * cos_t, 0.0, 1.0),
        params.light_e.w * AMBIENT_BOTTOM,
    );
    // The visible cloud of the veil lies about one build-up length past where it starts.
    let air = aerial_perspective(uv, dir, t_start + FAR_LENGTH_M, o_r);
    return vec4<f32>(a * (radiance * air.transmittance + air.in_scatter) * view.exposure, 1.0 - a);
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
    let cam = view.world_position;
    let dir = ray_dir(uv, view.world_from_clip, cam);
    let big_r = params.camera.w;
    let o_r = big_r + cam.y;
    let h_b = params.layer.x;
    let h_t = params.layer.y;

    let shell = shell_range(o_r, dir.y, h_b, h_t, big_r);
    let t0 = shell.x;
    let t1 = min(shell.y, MAX_MARCH_M);
    if t1 <= t0 {
        textureStore(march_out, vec2<i32>(id.xy), far_veil(uv, dir, shell, o_r));
        return;
    }

    // A block whose pixels all show surfaces nearer than the cloud entry shows no cloud.
    var hidden = true;
    for (var j = 0u; j < k && hidden; j += 1u) {
        for (var i = 0u; i < k && hidden; i += 1u) {
            let q = min(id.xy * k + vec2<u32>(i, j), size - vec2<u32>(1u));
            let quv = (vec2<f32>(q) + 0.5) / vec2<f32>(size);
            let d = textureLoad(depth_tex, vec2<i32>(q), 0);
            let qdir = ray_dir(quv, view.world_from_clip, cam);
            let q_shell = shell_range(o_r, qdir.y, h_b, h_t, big_r);
            if scene_distance(quv, d, view.world_from_clip, cam) >= q_shell.x {
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

    let steps = f32(params.counts.y);
    let dt = max(clamp((t1 - t0) / steps, 25.0, 400.0), (t1 - t0) / (1.5 * steps));
    let jitter = hash31(vec3<i32>(i32(px.x), i32(px.y), i32(frame)));
    let cos_t = dot(dir, l);
    let powder_weight = clamp(0.5 - 0.5 * cos_t, 0.0, 1.0);
    let ambient = params.light_e.w;

    var radiance = vec3<f32>(0.0);
    var transmittance = 1.0;
    var depth_sum = 0.0;
    var depth_weight = 0.0;
    var t = t0 + dt * jitter;
    var coarse = true;
    var empty = 0u;
    var resume = t;
    let max_iterations = params.counts.y * 3u;
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
        var m = medium_at(h, xz, true);
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
        let f = (h - h_b) / (h_t - h_b);
        var scatter = cloud_scatter(
            sun,
            tau_l,
            cos_t,
            powder_weight,
            ambient * (AMBIENT_BOTTOM + (1.0 - AMBIENT_BOTTOM) * f),
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

    if depth_weight <= 0.0 {
        textureStore(march_out, vec2<i32>(id.xy), far_veil(uv, dir, shell, o_r));
        return;
    }

    // Aerial perspective at the mean cloud depth, then the fade into haze.
    let depth = depth_sum / depth_weight;
    let fade = smoothstep(20000.0, 32000.0, depth);
    let air = aerial_perspective(uv, dir, depth, o_r);

    let rgb = (1.0 - fade) * (radiance * air.transmittance + air.in_scatter * (1.0 - transmittance));
    let alpha = 1.0 - (1.0 - fade) * (1.0 - transmittance);
    // The far field behind it: the cloud in front covers it by `1 − alpha`.
    let veil = far_veil(uv, dir, shell, o_r);
    textureStore(march_out, vec2<i32>(id.xy), vec4<f32>(rgb * view.exposure + alpha * veil.rgb, alpha * veil.a));
}
