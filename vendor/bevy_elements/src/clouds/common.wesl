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
    /// x: world units per metre (`ElementsUnits`). Everything above is in metres; Bevy's view
    /// matrices and depth are in world units, so the shaders divide what they take from them by
    /// this and multiply what they hand back.
    units: vec4<f32>,
    /// The sky probe (sky-light.md): x the followed camera's exposure (the display-referred
    /// `haze.xyz` over it is the fog's luminance), y the slot block edge k, z texels per face edge,
    /// w the weight a marched texel takes in its history (1 when the history is invalid).
    probe: vec4<f32>,
}

const PI: f32 = 3.14159265358979;
/// Farthest distance the march goes, m: the weather map extent and the aerial LUT range.
const MAX_MARCH_M: f32 = 32000.0;
/// World size of the base and detail noise tiles, m (docs/research/cumulus-shape.md: lobes of
/// about a kilometre, turrets and billows of 50–200 m).
const BASE_TILE_M: f32 = 4096.0;
const DETAIL_TILE_M: f32 = 640.0;
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
//     the cloud's own base noise (`dome_jitter`, 0..1), so blobs are not
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

/// Weight of the cumuliform density (cumulus-shape.md, Model): 0 for stratus up to type 0.25,
/// 1 from cumulus (0.5) up.
fn cumuliform(kind: f32) -> f32 {
    return clamp(4.0 * kind - 1.0, 0.0, 1.0);
}

/// Flat base of a cumuliform cloud: the density rises over this fraction of the layer.
const CUMULUS_BASE_RISE: f32 = 0.06;
/// The top rounds off over the upper part of the blob's height, from this fraction of it.
const CUMULUS_TOP_ROUND: f32 = 0.25;
/// Height of a cumuliform cloud at the edge of its blob, as a fraction of the layer.
const CUMULUS_EDGE_HEIGHT: f32 = 0.45;

/// The rounded top of a cumuliform cloud at height fraction `f` with the dome factor `dome`:
/// full up to the upper part of the blob's height `s`, rounding off toward it. C¹, so neither
/// the corners of a linear gradient nor the dome texels' kinks crease the cloud.
fn cumulus_envelope(f: f32, dome: f32) -> f32 {
    // [PATCH heads-up] 0.62 instead of sqrt: a flatter shoulder, matching the
    // dome profile change in cloud_gradient, so the cumuliform top does not
    // round off as a half-ellipse either.
    let s = mix(CUMULUS_EDGE_HEIGHT, 1.0, pow(dome, 0.62));
    return 1.0 - smoothstep(s * CUMULUS_TOP_ROUND, s, f);
}

/// The flat base of a cumuliform cloud: a factor on the eroded density, not on the profile the
/// noise erodes, so the condensation level is a plane and no billow hangs below it.
fn cumulus_base(f: f32) -> f32 {
    return smoothstep(0.0, CUMULUS_BASE_RISE, f);
}

/// Weight of the billowy composite: wispy at the base, billowy above it (Nubis Evolved's
/// `noise_height_blend` around the type).
fn billow_weight(f: f32) -> f32 {
    return smoothstep(0.05, 0.25, f);
}

/// Nubis' value erosion `saturate(noise − (1 − d))`, normalised to reach 1 where the noise is 1
/// and raised to the edge exponent ½: 0 wherever the noise is below `1 − d`, and a cloud's core
/// is as dense as its type allows however thin the profile that carved it.
fn value_erosion(noise: f32, d: f32) -> f32 {
    if d <= 0.0 {
        return 0.0;
    }
    return sqrt(clamp((noise - (1.0 - d)) / d, 0.0, 1.0));
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

/// A view ray: its origin and unit direction.
struct ViewRay {
    origin: vec3<f32>,
    dir: vec3<f32>,
}

/// Whether `clip_from_view` is an orthographic projection: `w` is 1 everywhere (Bevy's own
/// test, `cluster_raster.wesl`).
fn is_orthographic(clip_from_view: mat4x4<f32>) -> bool {
    return clip_from_view[3].w == 1.0;
}

/// The ray through screen position `uv` (0–1, top-left origin), in metres: the view matrices
/// are in world units, `units` of them to the metre. A perspective view's starts at
/// the eye `cam` (world). An orthographic view's runs along the view direction from the pixel's point on
/// the plane through `focus` (metres) across it (Marschner & Shirley §4.3.1, view-anchor.md Model 3 and
/// Deviation 5): an isometric camera's stand-off is not air, and a camera far back and high up
/// would otherwise start its rays inside or above a low deck that is not between the viewer and
/// the scene. Without a focus `focus` is the camera and the plane is its own. The direction
/// joins the pixel's points at reverse-Z depths 1 and 0.5, finite for both projections.
fn view_ray(
    uv: vec2<f32>,
    world_from_clip: mat4x4<f32>,
    cam: vec3<f32>,
    orthographic: bool,
    focus: vec3<f32>,
    units: f32,
) -> ViewRay {
    let ndc = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    let n = world_from_clip * vec4<f32>(ndc, 1.0, 1.0);
    let on_near = n.xyz / n.w / units;
    if orthographic {
        let m = world_from_clip * vec4<f32>(ndc, 0.5, 1.0);
        let dir = normalize(m.xyz / m.w / units - on_near);
        return ViewRay(on_near + dir * dot(focus - on_near, dir), dir);
    }
    let eye = cam / units;
    return ViewRay(eye, normalize(on_near - eye));
}

/// Distance along the ray (metres) from its `origin` to the surface seen at `uv` with reverse-Z
/// `depth`, negative for a surface in front of the origin; a huge value for the sky (depth 0).
fn scene_distance(
    uv: vec2<f32>,
    depth: f32,
    world_from_clip: mat4x4<f32>,
    ray: ViewRay,
    units: f32,
) -> f32 {
    if depth <= 0.0 {
        return NO_SURFACE;
    }
    let ndc = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    let p = world_from_clip * vec4<f32>(ndc, depth, 1.0);
    return dot(p.xyz / p.w / units - ray.origin, ray.dir);
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

/// The base-shape term, 0 outside cloud: what the coarse march samples. For a cumuliform
/// cloud it is the value erosion of the largest noise composite the detail could give, so the
/// coarse march never steps over a cloud the fine one would draw.
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
    let b4 = textureSampleLevel(base_tex, smp, base_uvw(lean, h, p), 0.0);
    let lf = 0.625 * b4.g + 0.25 * b4.b + 0.125 * b4.a;
    let shape = clamp((b4.r + 1.0 - lf) / (2.0 - lf), 0.0, 1.0);
    let r = pow(w.r, coverage_exponent(f, kind));
    let k = cumuliform(kind);
    var stratiform = 0.0;
    if k < 1.0 {
        stratiform = shape * cloud_gradient(f, kind, w.a, b4.g) * r;
    }
    var cumulus = 0.0;
    if k > 0.0 {
        // Only the sign matters to the coarse march: no division or power.
        cumulus = max(noise_bound(f, shape, lf) - (1.0 - cumulus_profile(f, w.a, r, b4.g, p.layer.w)), 0.0);
    }
    return mix(stratiform, cumulus, k);
}

/// Exponent of the coverage erosion `r′ = r^e`: the anvil inflation of the top.
fn coverage_exponent(f: f32, kind: f32) -> f32 {
    let beta = smoothstep(0.5, 1.0, kind);
    return mix(1.0, mix(1.0, 0.5, beta), clamp((f - 0.7) / 0.1, 0.0, 1.0));
}

/// Base-noise coordinate of a (leaned) point.
fn base_uvw(lean: vec2<f32>, h: f32, p: CloudParams) -> vec3<f32> {
    return vec3<f32>((lean.x - p.drift.z) / BASE_TILE_M, h / BASE_TILE_M, (lean.y - p.drift.w) / BASE_TILE_M);
}

/// Share of the cumuliform profile the cloud's core keeps, so the noise's lows still open gaps
/// inside a coverage blob and it breaks into separate clouds.
const CORE_PROFILE: f32 = 0.75;

/// The dimensional profile of a cumuliform cloud (Nubis Evolved): the rounded envelope times
/// the eroded coverage times the convective cell (`cell`, the base texture's Worley cells: a
/// coverage blob is a field of separate thermals, not one mass). The gaps close as the weather's
/// `cover` reaches 1 (`cover⁴`): a sky the weather calls overcast has no holes, its billows
/// pack into a deck (Nubis³: "Increasing Density = Layered Billows").
fn cumulus_profile(f: f32, dome: f32, r: f32, cell: f32, cover: f32) -> f32 {
    let c2 = clamp(cover, 0.0, 1.0) * clamp(cover, 0.0, 1.0);
    let closed = c2 * c2;
    let gaps = mix(CORE_PROFILE * smoothstep(0.25, 0.65, cell), 1.0, closed);
    return cumulus_envelope(f, dome) * r * gaps;
}

/// The noise composite of a cumuliform cloud from the base Perlin-Worley `shape`, the base
/// Worley octaves `lf` and the detail Worley octaves `hf`: wispy toward the base, billowy above
/// (cumulus-shape.md, Model 3).
fn noise_composite(f: f32, shape: f32, lf: f32, hf: f32) -> f32 {
    let wispy = 0.6 * shape + 0.4 * (1.0 - hf);
    let billowy = 0.5 * shape + 0.3 * lf + 0.2 * hf;
    return mix(wispy, billowy, billow_weight(f));
}

/// The largest `noise_composite` any detail value gives: the coarse march's bound.
fn noise_bound(f: f32, shape: f32, lf: f32) -> f32 {
    return mix(0.6 * shape + 0.4, 0.5 * shape + 0.3 * lf + 0.2, billow_weight(f));
}

/// The medium at a point of altitude `h` above world XZ `xz`: the weather map gives coverage,
/// precipitation and (uniform) type, the base noise the shape, the detail noise erodes the
/// edges. `detail` (0–1) is how much of the detail noise the sample can resolve: 0 skips the
/// tap (the shadow map, the light march), and the march lowers it as its step outgrows the
/// billows (`detail_weight`), so a long step sees their mean instead of aliasing them.
fn cloud_medium(
    h: f32,
    xz: vec2<f32>,
    detail: f32,
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
    let b4 = textureSampleLevel(base_tex, smp, base_uvw(lean, h, p), 0.0);
    let lf = 0.625 * b4.g + 0.25 * b4.b + 0.125 * b4.a;
    let base_shape = clamp((b4.r + 1.0 - lf) / (2.0 - lf), 0.0, 1.0);
    let r = pow(w.r, coverage_exponent(f, kind));
    // Without the detail tap the detail octaves take their mean.
    var raw_hf = 0.5;
    if detail > 0.0 {
        let duvw = vec3<f32>(
            (lean.x - p.drift_detail.x) / DETAIL_TILE_M,
            h / DETAIL_TILE_M,
            (lean.y - p.drift_detail.y) / DETAIL_TILE_M,
        );
        let d4 = textureSampleLevel(detail_tex, smp, duvw, 0.0);
        raw_hf = 0.625 * d4.r + 0.25 * d4.g + 0.125 * d4.b;
    }
    let hf = mix(0.5, raw_hf, detail);
    let k = cumuliform(kind);
    var stratiform = 0.0;
    if k < 1.0 {
        let shape = base_shape * cloud_gradient(f, kind, w.a, b4.g);
        stratiform = clamp((shape - (1.0 - r)) / max(r, 1e-4), 0.0, 1.0) * r;
        if detail > 0.0 && stratiform > 0.0 {
            let wispy = mix(raw_hf, 1.0 - raw_hf, clamp(10.0 * f, 0.0, 1.0));
            let lo = DETAIL_EROSION * wispy;
            stratiform = clamp((stratiform - lo) / (1.0 - lo), 0.0, 1.0);
        }
    }
    var cumulus = 0.0;
    if k > 0.0 {
        let d = cumulus_profile(f, w.a, r, b4.g, p.layer.w);
        cumulus = value_erosion(noise_composite(f, base_shape, lf, hf), d) * cumulus_base(f);
    }
    m.sigma = sigma_peak(kind) * mix(stratiform, cumulus, k);
    m.g = w.g;
    return m;
}

/// How much of the detail noise a fine step of `dt` metres resolves: all of it up to an eighth
/// of the detail tile (its finest cells are a sixteenth), none past a quarter.
fn detail_weight(dt: f32) -> f32 {
    return 1.0 - smoothstep(DETAIL_TILE_M / 8.0, DETAIL_TILE_M / 4.0, dt);
}

/// Henyey–Greenstein phase function, sr⁻¹.
fn hg(cos_t: f32, g: f32) -> f32 {
    let g2 = g * g;
    return (1.0 - g2) / (4.0 * PI * pow(1.0 + g2 - 2.0 * g * cos_t, 1.5));
}

/// The single-scattering dual lobe: forward lobe (g = 0.8) weighted 0.7, backward lobe
/// (g = −0.5) weighted 0.3.
fn phase(cos_t: f32) -> f32 {
    return 0.7 * hg(cos_t, 0.8) + 0.3 * hg(cos_t, -0.5);
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

/// Cloud transmittance below which the sky behind is hidden, and above which it is really in
/// view: the sky gain ramps in between (`clouds.md` Deviation 28).
const OPAQUE_BELOW: f32 = 0.02;
const SKY_ABOVE: f32 = 0.2;

/// The cloud layer and the weather's air over the sky, along a ray that reaches the layer (no
/// ground before it): premultiplied `(L, T)` of `cloud`, to be composed as `L + L_sky · T`.
///
/// Bevy's atmosphere is lit by the sun light, which carries only the beam `T̄` under cloud, but
/// the sky above the cloud is not dimmed by the cloud below it: the sky gain (`params.sky.x`)
/// ramps in above a transmittance of 0.02, since the march stops at 0.01 and an opaque cloud
/// must not amplify the sky behind it. Then the weather's air over the path to the layer entry
/// (`air.x = σ_fog − 3.912/40 km`): a sky pixel already has Bevy's atmosphere, which draws the
/// clear air, so this is what the weather adds on top, in-scatter and extinction both the fog's,
/// so the sky converges on the fog colour `haze` as distant ground does and clear air (σ_air = 0)
/// is an exact no-op (`clouds.md` Deviation 29). A ray that never enters the layer
/// (`shell.y <= 0`) has the weather's whole extent of air; a ray starting inside it has none.
/// `haze` is in the units of `cloud`: display-referred for the screen, luminance for the probe.
fn sky_through_clouds(
    cloud: vec4<f32>,
    shell: vec2<f32>,
    p: CloudParams,
    haze: vec3<f32>,
) -> vec4<f32> {
    let a = cloud.a * mix(1.0, p.sky.x, smoothstep(OPAQUE_BELOW, SKY_ABOVE, cloud.a));
    let air_path = select(shell.x, MAX_MARCH_M, shell.y <= 0.0);
    let t = exp(-p.air.x * air_path);
    return vec4<f32>(cloud.rgb * t + haze * (1.0 - t), a * t);
}
