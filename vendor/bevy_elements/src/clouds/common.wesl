// bevy_elements::clouds::common — the cloud model shared by every cloud kernel: the uniform,
// shell geometry, density, phase function and the atmosphere lookups. No bindings: each kernel
// declares its own and passes the textures in. See docs/research/clouds.md.
//
// Geometry: one spherical shell about a planet centre straight below the camera. A point is
// described by its horizontal offset `rel_xz` from the camera and its world height `y`; its
// altitude above the surface is computed without forming radii of 6.36e6 m in f32.

/// Mirrors the Rust `CloudParams`. Every field is a vec4 so the layouts agree.
struct CloudParams {
    /// xyz: camera world position, m. w: planet radius R, m.
    camera: vec4<f32>,
    /// x: atmosphere outer radius, m. y: aerial-view LUT range, m. z: aerial-view LUT slices.
    atmo: vec4<f32>,
    /// x: gain on the sky pixels of the composite. yzw: radiance of the air lit by the
    /// cloud-scattered diffuse light, cd/m².
    sky: vec4<f32>,
    /// x: layer base altitude, m. y: layer top altitude, m. z: cloud type 0–1. w: cloud cover.
    layer: vec4<f32>,
    /// xyz: unit vector toward the light (sun, or moon at night).
    light: vec4<f32>,
    /// xyz: illuminance of the light outside the atmosphere per channel, lx. w: ambient source
    /// term E_d / 2π, cd/m².
    light_e: vec4<f32>,
    /// xy: cloud drift for the weather field (wrapped to its 32 km period), m. zw: for the base
    /// noise (wrapped to 8192 m).
    drift: vec4<f32>,
    /// xy: cloud drift for the detail noise (wrapped to 1024 m). zw: unit wind direction (XZ).
    drift_detail: vec4<f32>,
    /// xy: world XZ of the weather map's minimum corner. z: its extent, m. w: precipitation
    /// darkening min(R/25, 1).
    weather: vec4<f32>,
    /// xy: world XZ of the shadow map's minimum corner. z: its extent, m.
    shadow: vec4<f32>,
    /// xyz: flash position, world m. w: intensity, cd.
    flash: vec4<f32>,
    /// x: frame counter. y: primary steps Q. z: block edge k. w: light samples.
    counts: vec4<u32>,
    /// xy: full resolution. zw: march resolution.
    sizes: vec4<u32>,
    /// x: exposure(previous frame) / exposure. y: debug mode (0 off, 1 uniform slab of
    /// extinction z, 2 sun transmittance at the layer entry). z: debug extinction, m⁻¹.
    /// w: 1 on the first frame of a history (no valid history).
    misc: vec4<f32>,
    /// x: first z slice of this frame's noise generation.
    tail: vec4<u32>,
    /// xyz: the camera's distance fog colour (linear, after exposure). w: its extinction, m⁻¹.
    haze: vec4<f32>,
    /// x: extinction of the weather's air above clear air, m⁻¹; the composite fogs sky pixels
    /// with it, so the sky converges on `haze.xyz` as the ground does. 0 for clear air.
    air: vec4<f32>,
}

const PI: f32 = 3.14159265358979;
/// Farthest distance the march goes, m: the weather map extent and the aerial LUT range.
const MAX_MARCH_M: f32 = 32000.0;
/// World size of the base and detail noise tiles, m.
const BASE_TILE_M: f32 = 8192.0;
const DETAIL_TILE_M: f32 = 1024.0;
/// Horizontal lean of cloud tops per unit of height fraction, m (Nubis).
const WIND_SKEW_M: f32 = 500.0;
/// Cloud-side detail erosion strength (Schneider 2015 / GPU Pro 7).
const DETAIL_EROSION: f32 = 0.2;
/// Large number standing for "no surface": the sky.
const NO_SURFACE: f32 = 1.0e30;

/// Peak extinction by cloud type, m⁻¹: stratus 0.05, cumulus 0.085, cumulonimbus 0.12.
fn sigma_peak(kind: f32) -> f32 {
    return select(mix(0.085, 0.12, (kind - 0.5) * 2.0), mix(0.05, 0.085, kind * 2.0), kind <= 0.5);
}

/// Height-gradient quadruple (a0, a1, a2, a3) by type.
fn gradient_quad(kind: f32) -> vec4<f32> {
    let stratus = vec4<f32>(0.0, 0.1, 0.2, 0.3);
    let cumulus = vec4<f32>(0.0, 0.1625, 0.88, 0.98);
    let nimbus = vec4<f32>(0.0, 0.1, 0.9, 1.0);
    let t = clamp(kind * 2.0, 0.0, 1.0);
    let u = clamp(kind * 2.0 - 1.0, 0.0, 1.0);
    return mix(mix(stratus, cumulus, t), nimbus, u);
}

/// Smallest fraction of the layer height a cloud at the edge of its coverage reaches.
const MIN_HEIGHT_SCALE: f32 = 0.3;

/// The height gradient with the cloud height following the dome factor (the weather map's A
/// channel, 0 at the edge of a blob, 1 deep inside): a blob has full height in its core and is
/// shorter toward its edge, so cumulus domes round off instead of standing
/// as prisms with flat tops and vertical walls (Deviation 12 of the research note).
// [PATCH heads-up] Dome shaping. The upstream curve is
//     scale = mix(1, mix(0.3, 1, sqrt(dome)), clamp(kind * 4, 0, 1))
// which (a) saturates completely for any `kind >= 0.25`, so every cumulus-like
// cloud gets an identical dome, and (b) uses `sqrt`, which rises so fast that
// the profile is a clean half-ellipse -- read as "a flat disc with half an egg
// on top". Two changes:
//   * the kind ramp is a smoothstep over a wider band, so intermediate types
//     get intermediate domes instead of snapping to full;
//   * the dome profile is a gentler exponent, and is modulated per-height by
//     the cloud's own detail noise (`dome_jitter`, 0..1), so blobs are not
//     identical copies of one shape.
fn cloud_gradient(f: f32, kind: f32, dome: f32, dome_jitter: f32) -> f32 {
    let lean = smoothstep(0.05, 0.55, kind);
    // 0.62 instead of sqrt: a flatter shoulder, closer to a cumulus pile than
    // to a dome. `dome_jitter` breaks the radial symmetry by up to +-35 %.
    let profile = pow(dome, 0.62) * mix(0.65, 1.35, dome_jitter);
    let scale = mix(1.0, mix(MIN_HEIGHT_SCALE, 1.0, clamp(profile, 0.0, 1.0)), lean);
    return height_gradient(min(f / max(scale, 0.05), 1.0), gradient_quad(kind));
}

fn height_gradient(f: f32, q: vec4<f32>) -> f32 {
    let up = clamp((f - q.x) / max(q.y - q.x, 1e-4), 0.0, 1.0);
    let down = 1.0 - clamp((f - q.z) / max(q.w - q.z, 1e-4), 0.0, 1.0);
    return up * down;
}

/// Altitude above the surface of the point `rel_xz` away from the camera horizontally at world
/// height `y`, m: (r² − R²) / (r + R) with r² − R² built from small terms.
fn altitude(rel_xz: vec2<f32>, y: f32, big_r: f32) -> f32 {
    let h2 = dot(rel_xz, rel_xz);
    let a = h2 + y * (2.0 * big_r + y);
    let r = sqrt(h2 + (big_r + y) * (big_r + y));
    return a / (r + big_r);
}

/// The two roots (near, far) of |o + t d| = r for a ray from radius `o_r` whose direction has
/// cosine `mu` with the local vertical. (1, −1) when the ray misses the sphere. Numerically
/// stable form (Numerical Recipes q) so a 1.5 km shell is found to centimetres in f32.
fn sphere_roots(o_r: f32, mu: f32, r: f32) -> vec2<f32> {
    let b = o_r * mu;
    let c0 = (o_r - r) * (o_r + r);
    let disc = b * b - c0;
    if disc < 0.0 {
        return vec2<f32>(1.0, -1.0);
    }
    let s = sqrt(disc);
    let q = -(b + select(-s, s, b >= 0.0));
    var t2 = 0.0;
    if abs(q) > 1e-6 {
        t2 = c0 / q;
    }
    return vec2<f32>(min(q, t2), max(q, t2));
}

/// Where a ray from the camera is inside the cloud shell: (t_enter, t_exit); empty (exit ≤
/// enter) when it never is. Not clamped to the march range.
fn shell_range(o_r: f32, mu: f32, h_b: f32, h_t: f32, big_r: f32) -> vec2<f32> {
    let top = sphere_roots(o_r, mu, big_r + h_t);
    if top.y <= 0.0 {
        return vec2<f32>(0.0, 0.0);
    }
    let t_in = max(top.x, 0.0);
    let t_out = top.y;
    let base = sphere_roots(o_r, mu, big_r + h_b);
    if base.x > base.y {
        return vec2<f32>(t_in, t_out);
    }
    // The ray passes through the hollow below the base between base.x and base.y.
    let first_end = min(t_out, base.x);
    if first_end > t_in {
        return vec2<f32>(t_in, first_end);
    }
    return vec2<f32>(max(t_in, base.y), t_out);
}

/// Distance to the planet's surface along a ray that meets it before it enters the cloud layer
/// at `shell_in` (a ray past the edge of a finite ground), or −1 when it does not. Such a ray sees
/// no cloud: `shell_range` does not know the planet and finds the layer on its far side.
fn ground_before_layer(o_r: f32, mu: f32, shell_in: f32, big_r: f32) -> f32 {
    let ground = sphere_roots(o_r, mu, big_r);
    if ground.x > 0.0 && ground.x <= ground.y && ground.x < shell_in {
        return ground.x;
    }
    return -1.0;
}

/// Unit world direction of the ray through screen position `uv` (0–1, top-left origin).
fn ray_dir(uv: vec2<f32>, world_from_clip: mat4x4<f32>, cam: vec3<f32>) -> vec3<f32> {
    let ndc = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    let p = world_from_clip * vec4<f32>(ndc, 1.0, 1.0);
    return normalize(p.xyz / p.w - cam);
}

/// Distance from the camera to the surface seen at `uv` with reverse-Z `depth`; a huge value
/// for the sky (depth 0).
fn scene_distance(uv: vec2<f32>, depth: f32, world_from_clip: mat4x4<f32>, cam: vec3<f32>) -> f32 {
    if depth <= 0.0 {
        return NO_SURFACE;
    }
    let ndc = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    let p = world_from_clip * vec4<f32>(ndc, depth, 1.0);
    return length(p.xyz / p.w - cam);
}

/// Offset of the pixel a march texel owns this frame inside its k×k block: the Bayer order
/// (2×2 for k = 2, 4×4 for k = 4), so every pixel is marched once per k² frames.
fn slot_offset(frame: u32, k: u32) -> vec2<u32> {
    var bayer2 = array<vec2<u32>, 4>(
        vec2<u32>(0u, 0u), vec2<u32>(1u, 1u), vec2<u32>(1u, 0u), vec2<u32>(0u, 1u),
    );
    var bayer4 = array<vec2<u32>, 16>(
        vec2<u32>(0u, 0u), vec2<u32>(2u, 2u), vec2<u32>(2u, 0u), vec2<u32>(0u, 2u),
        vec2<u32>(1u, 1u), vec2<u32>(3u, 3u), vec2<u32>(3u, 1u), vec2<u32>(1u, 3u),
        vec2<u32>(1u, 0u), vec2<u32>(3u, 2u), vec2<u32>(3u, 0u), vec2<u32>(1u, 2u),
        vec2<u32>(0u, 1u), vec2<u32>(2u, 3u), vec2<u32>(2u, 1u), vec2<u32>(0u, 3u),
    );
    if k == 4u {
        return bayer4[frame % 16u];
    }
    if k == 2u {
        return bayer2[frame % 4u];
    }
    return vec2<u32>(0u, 0u);
}

/// Extinction and precipitation darkening at a point.
struct Medium {
    /// σ_t, m⁻¹.
    sigma: f32,
    /// Precipitation darkening g ≥ 0 (extra light-path extinction factor 1 + g).
    g: f32,
}

/// The base-shape term `S′·r′`, 0 outside cloud: what the coarse march samples.
fn coarse_shape(
    h: f32,
    xz: vec2<f32>,
    p: CloudParams,
    base_tex: texture_3d<f32>,
    weather_tex: texture_2d<f32>,
    smp: sampler,
) -> f32 {
    let h_b = p.layer.x;
    let f = (h - h_b) / (p.layer.y - h_b);
    if f < 0.0 || f > 1.0 {
        return 0.0;
    }
    if p.misc.y > 0.5 && p.misc.y < 1.5 {
        return 1.0;
    }
    let w = textureSampleLevel(weather_tex, smp, (xz - p.weather.xy) / p.weather.z, 0.0);
    if w.r <= 0.0 {
        return 0.0;
    }
    let kind = p.layer.z;
    let lean = xz - f * p.drift_detail.zw * WIND_SKEW_M;
    let uvw = vec3<f32>(
        (lean.x - p.drift.z) / BASE_TILE_M,
        h / BASE_TILE_M,
        (lean.y - p.drift.w) / BASE_TILE_M,
    );
    let b4 = textureSampleLevel(base_tex, smp, uvw, 0.0);
    let lf = 0.625 * b4.g + 0.25 * b4.b + 0.125 * b4.a;
    let shape = clamp((b4.r + 1.0 - lf) / (2.0 - lf), 0.0, 1.0);
    let beta = smoothstep(0.5, 1.0, kind);
    let e = mix(1.0, mix(1.0, 0.5, beta), clamp((f - 0.7) / 0.1, 0.0, 1.0));
    let r = pow(w.r, e);
    // [PATCH heads-up] Break the radial symmetry: the base noise's green
    // channel varies per blob, so identical domes stop being identical.
    let dome_jitter = clamp(b4.g, 0.0, 1.0);
    return shape * cloud_gradient(f, kind, w.a, dome_jitter) * r;
}

/// The medium at a point of altitude `h` above world XZ `xz`: the weather map gives coverage,
/// precipitation and (uniform) type, the base noise the shape, the detail noise (when
/// `detail`) erodes the edges.
fn cloud_medium(
    h: f32,
    xz: vec2<f32>,
    detail: bool,
    p: CloudParams,
    base_tex: texture_3d<f32>,
    detail_tex: texture_3d<f32>,
    weather_tex: texture_2d<f32>,
    smp: sampler,
) -> Medium {
    var m = Medium(0.0, 0.0);
    let h_b = p.layer.x;
    let f = (h - h_b) / (p.layer.y - h_b);
    if f < 0.0 || f > 1.0 {
        return m;
    }
    if p.misc.y > 0.5 && p.misc.y < 1.5 {
        m.sigma = p.misc.z;
        return m;
    }
    let w = textureSampleLevel(weather_tex, smp, (xz - p.weather.xy) / p.weather.z, 0.0);
    if w.r <= 0.0 {
        return m;
    }
    let kind = p.layer.z;
    let lean = xz - f * p.drift_detail.zw * WIND_SKEW_M;
    let uvw = vec3<f32>(
        (lean.x - p.drift.z) / BASE_TILE_M,
        h / BASE_TILE_M,
        (lean.y - p.drift.w) / BASE_TILE_M,
    );
    let b4 = textureSampleLevel(base_tex, smp, uvw, 0.0);
    let lf = 0.625 * b4.g + 0.25 * b4.b + 0.125 * b4.a;
    let dome_jitter = clamp(b4.g, 0.0, 1.0);
    let shape = clamp((b4.r + 1.0 - lf) / (2.0 - lf), 0.0, 1.0)
        * cloud_gradient(f, kind, w.a, dome_jitter);
    let beta = smoothstep(0.5, 1.0, kind);
    let e = mix(1.0, mix(1.0, 0.5, beta), clamp((f - 0.7) / 0.1, 0.0, 1.0));
    let r = pow(w.r, e);
    var rho = clamp((shape - (1.0 - r)) / max(r, 1e-4), 0.0, 1.0) * r;
    if detail && rho > 0.0 {
        let duvw = vec3<f32>(
            (lean.x - p.drift_detail.x) / DETAIL_TILE_M,
            h / DETAIL_TILE_M,
            (lean.y - p.drift_detail.y) / DETAIL_TILE_M,
        );
        let d4 = textureSampleLevel(detail_tex, smp, duvw, 0.0);
        let hf = 0.625 * d4.r + 0.25 * d4.g + 0.125 * d4.b;
        let wispy = mix(hf, 1.0 - hf, clamp(10.0 * f, 0.0, 1.0));
        let lo = DETAIL_EROSION * wispy;
        rho = clamp((rho - lo) / (1.0 - lo), 0.0, 1.0);
    }
    m.sigma = sigma_peak(kind) * rho;
    m.g = w.g;
    return m;
}

/// Henyey–Greenstein phase function, sr⁻¹.
fn hg(cos_t: f32, g: f32) -> f32 {
    let g2 = g * g;
    return (1.0 - g2) / (4.0 * PI * pow(1.0 + g2 - 2.0 * g * cos_t, 1.5));
}

/// Dual-lobe phase of octave `n` of the multiple-scattering sum: both lobes' g scaled by
/// c^n (`c_pow`), forward lobe weighted 0.7.
fn phase_octave(cos_t: f32, c_pow: f32) -> f32 {
    return 0.7 * hg(cos_t, 0.8 * c_pow) + 0.3 * hg(cos_t, -0.5 * c_pow);
}

/// Visibility of the light from a point of altitude `h` whose local vertical makes cosine `mu`
/// with the light: 0 below the planet's horizon, blended over ±0.0044 rad (half a sun disc).
fn planet_visibility(h: f32, mu: f32, big_r: f32) -> f32 {
    let mu_h = -sqrt(max(h * (2.0 * big_r + h), 0.0)) / (big_r + h);
    return smoothstep(mu_h - 0.0044, mu_h + 0.0044, mu);
}

/// UV in Bevy's transmittance LUT for radius `r` and zenith cosine `mu` (Bruneton's mapping,
/// derived from its definition; the LUT is a function of the medium only).
fn transmittance_uv(r: f32, mu: f32, ri: f32, ro: f32) -> vec2<f32> {
    let big_h = sqrt(ro * ro - ri * ri);
    let rho = sqrt(max(r * r - ri * ri, 0.0));
    let d = max(-r * mu + sqrt(max(r * r * (mu * mu - 1.0) + ro * ro, 0.0)), 0.0);
    let d_min = ro - r;
    let d_max = rho + big_h;
    return vec2<f32>((d - d_min) / (d_max - d_min), rho / big_h);
}

/// Transmittance from radius `r` along zenith cosine `mu` to the top of the atmosphere.
fn sample_transmittance(
    lut: texture_2d<f32>,
    smp: sampler,
    r: f32,
    mu: f32,
    p: CloudParams,
) -> vec3<f32> {
    return textureSampleLevel(lut, smp, transmittance_uv(r, mu, p.camera.w, p.atmo.x), 0.0).rgb;
}

/// Illuminance of the light at altitude `h` where the light makes cosine `mu` with the local
/// vertical: outside-atmosphere illuminance × atmospheric transmittance × planet visibility.
fn light_at(
    lut: texture_2d<f32>,
    smp: sampler,
    h: f32,
    mu: f32,
    p: CloudParams,
) -> vec3<f32> {
    let r = p.camera.w + h;
    let v = planet_visibility(h, mu, p.camera.w);
    if v <= 0.0 {
        return vec3<f32>(0.0);
    }
    return p.light_e.xyz * sample_transmittance(lut, smp, r, clamp(mu, -1.0, 1.0), p) * v;
}
