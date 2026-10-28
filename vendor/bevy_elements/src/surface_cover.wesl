// bevy_elements::surface_cover — how wet and snowy a surface looks, shared by the extensions that
// weather a material (WeatheredExtension; the terrain extensions import it too). The model is
// docs/research/wetness.md ("Shading, shared by all three extensions") and
// docs/research/snow-cover.md ("Coverage and shading"); `src/weathered/mirror.rs` is the CPU
// transliteration the tests check.
//
//     import bevy_elements::surface_cover::{ElementsGlobals, weather_surface, mesh_snow_cover};
//     @group(constants::MATERIAL_BIND_GROUP) @binding(101) var<storage, read> elements: ElementsGlobals;
//
// Bind the crate's shared `ElementsGlobals` buffer (`WeatheredExtension` does, at binding 101).
// Every function is pure: textures and the extension's own bindings stay in the caller.

import bevy_elements::noise::{hash33, perlin3};

/// The shared globals buffer, five `vec4<f32>` (80 B). Mirrors the Rust `ElementsGlobals`.
struct ElementsGlobals {
    /// x wetness 0–1, y snow cover 0–1 (both of exposed flat ground), z rain intensity
    /// `min(1, R_liquid / 20)`, w the camera's mean cloud transmittance.
    weather: vec4<f32>,
    /// xy cloud shadow map origin (world XZ, m), z extent (m), w 1 when a map is bound.
    shadow: vec4<f32>,
    /// Phase of each of the four ripple layers, 0–1.
    ripple: vec4<f32>,
    /// xy wind the weather blows toward (m/s), z streak phase (s, wraps with the noise lattice).
    wind: vec4<f32>,
    /// xyz unit direction toward the light the shadow map was made for.
    light: vec4<f32>,
}

/// Linear albedo of snow; bit-equal to `bevy_flora`'s `SNOW_ALBEDO` (`common.wesl:388`).
const SNOW_ALBEDO: vec3<f32> = vec3<f32>(0.80, 0.83, 0.88);
/// Roughness of snow; `bevy_flora` `common.wesl:424` (the plan's 0.6 is Deviation 4 of snow-cover.md).
const SNOW_ROUGHNESS: f32 = 0.9;
/// Fraction of albedo a fully wet, fully porous surface loses (plan 4.1).
const WET_ALBEDO_LOSS: f32 = 0.5;
/// Roughness a wet surface approaches (plan 4.1).
const WET_ROUGHNESS: f32 = 0.1;
/// Roughness of standing water (Lagarde 2013, gloss 1).
const PUDDLE_ROUGHNESS: f32 = 0.02;
/// Bevy `reflectance` for water: `F0 = 0.16 · r²` (pbr_functions.wesl:279) equals 0.02, so
/// `r = sqrt(0.02 / 0.16)`.
const WATER_REFLECTANCE: f32 = 0.35355339;
/// Ripple normal strength (Lagarde 2013, `* 0.35`).
const RIPPLE_STRENGTH: f32 = 0.35;
/// Ripple lattice cell and ring radius, m (wetness.md Deviation 4).
const RIPPLE_CELL_M: f32 = 0.4;
const RIPPLE_RADIUS_M: f32 = 0.2;
/// Largest offset of a ring's centre from its cell centre, in cells.
const RIPPLE_JITTER: f32 = 0.25;
const RIPPLE_PI: f32 = 3.14159265;
/// Rain streak amplitude, the rain intensity at which a wall streams fully, and the noise scales
/// (wetness.md, Model adopted 6, and Deviation 9).
const STREAK_AMPLITUDE: f32 = 0.4;
const STREAK_SATURATION: f32 = 0.2;
/// The noise range a rivulet occupies: Perlin seldom gets above 0.6 (`n` 0.8), so the
/// note's smoothstep(0.55, 1) never reached 1 and drew 3% contrast (measured, Deviation 9).
const STREAK_EDGE_LOW: f32 = 0.55;
const STREAK_EDGE_HIGH: f32 = 0.8;
/// Wetness a rivulet adds beyond a full film: 1.4 darkens a porous surface to ×0.3, the
/// 0.2–0.3 of Lagarde's empirical wet albedo (wetness.md, Model adopted 2).
const DRENCHED_MAX: f32 = 1.4;
const STREAK_FREQUENCY_X: f32 = 8.0;
const STREAK_FREQUENCY_Y: f32 = 0.4;
const STREAK_SPEED: f32 = 0.5;

/// How much of its water a surface soaks up, 0–1: Lagarde's gloss-based porosity estimate with
/// Bevy's roughness standing in for `1 − gloss`. Rough walls and soil (`r ≥ 0.9`) are 1, plastic
/// and glazed tile (`r ≤ 0.5`) 0, metal 0.
fn porosity(roughness: f32, metallic: f32) -> f32 {
    return saturate((roughness - 0.5) / 0.4) * (1.0 - metallic);
}

/// Snow cover of a mesh face from the scene's snow cover and the face's geometric world normal
/// `y`: the flora curve `snow · smoothstep(0.85 − 0.6 snow, +0.2, n.y)`.
fn mesh_snow_cover(snow: f32, normal_y: f32) -> f32 {
    let lo = 0.85 - 0.6 * snow;
    return snow * smoothstep(lo, lo + 0.2, normal_y);
}

/// Wetness a rain streak adds on top of a vertical face's film: `0.4 · min(1, f / 0.2) · s ·
/// (1 − n_y²)` with `s` a noise stretched down the wall that flows with `phase`. `world` is the
/// fragment's position, `normal` its geometric world normal, `intensity` the rain intensity `f`.
/// The noise lattice repeats every 4096 units, so a `phase` wrapped at 8192 s is seamless.
fn rain_streak(world: vec3<f32>, normal: vec3<f32>, intensity: f32, phase: f32) -> f32 {
    let vertical = 1.0 - normal.y * normal.y;
    let horizontal = length(normal.xz);
    var along = vec2<f32>(1.0, 0.0);
    if horizontal > 1e-4 {
        along = vec2<f32>(-normal.z, normal.x) / horizontal;
    }
    let x = dot(world.xz, along);
    let n = 0.5 + 0.5 * perlin3(vec3<f32>(
        STREAK_FREQUENCY_X * x,
        STREAK_FREQUENCY_Y * world.y + STREAK_SPEED * phase,
        0.5,
    ));
    return STREAK_AMPLITUDE * saturate(intensity / STREAK_SATURATION) * smoothstep(STREAK_EDGE_LOW, STREAK_EDGE_HIGH, n) * vertical;
}

/// Weights of the four ripple layers for rain intensity `f`: `saturate((f − 0.25 k) · 4)`.
fn ripple_weights(intensity: f32) -> vec4<f32> {
    return saturate((vec4<f32>(intensity) - vec4<f32>(0.0, 0.25, 0.5, 0.75)) * 4.0);
}

/// One ripple layer: the horizontal normal offset at world `xz` of the ring in its lattice cell,
/// after Lagarde 2013 (`ComputeRipple`) with a procedural ring instead of his tiled texture.
/// `phase` is the layer's 0–1 phase, `weight` its weight; a layer of weight 0 is absent.
fn ripple_layer(xz: vec2<f32>, layer: u32, phase: f32, weight: f32) -> vec2<f32> {
    if weight <= 0.0 {
        return vec2<f32>(0.0);
    }
    let q = xz / RIPPLE_CELL_M + f32(layer) * vec2<f32>(0.37, 0.23);
    let cell = floor(q);
    let h = hash33(vec3<i32>(vec2<i32>(cell), i32(layer)));
    let centre = cell + vec2<f32>(0.5) + (h.xy - 0.5) * RIPPLE_JITTER;
    let d = (q - centre) * RIPPLE_CELL_M;
    let dist = length(d);
    let x = 1.0 - dist / RIPPLE_RADIUS_M;
    if x <= 0.0 {
        return vec2<f32>(0.0);
    }
    let drop_frac = fract(phase + h.z);
    let time_frac = drop_frac - 1.0 + x;
    let drop_factor = saturate(0.2 + weight * 0.8 - drop_frac);
    let ring = drop_factor * x * sin(clamp(time_frac * 9.0, 0.0, 3.0) * RIPPLE_PI);
    return d / max(dist, 1e-5) * (ring * RIPPLE_STRENGTH);
}

/// The horizontal normal offset of all four ripple layers at world `xz`.
fn ripple_offset(xz: vec2<f32>, phases: vec4<f32>, weights: vec4<f32>) -> vec2<f32> {
    return ripple_layer(xz, 0u, phases.x, weights.x)
        + ripple_layer(xz, 1u, phases.y, weights.y)
        + ripple_layer(xz, 2u, phases.z, weights.z)
        + ripple_layer(xz, 3u, phases.w, weights.w);
}

/// What `weather_surface` returns: the material inputs after wetness, puddles and snow.
struct WeatheredSurface {
    base_color: vec3<f32>,
    roughness: f32,
    metallic: f32,
    reflectance: vec3<f32>,
    normal: vec3<f32>,
}

/// Wet, puddled and snowed-on material inputs, in the order of the note: darken and gloss by the
/// wetness `max(wetness, puddle)`; standing water is a mirror with water's `F0`, a level normal
/// plus `ripple` (the horizontal offset of `ripple_offset`); snow covers it all.
///
/// With wetness, puddle and snow all 0 the inputs come back bit-for-bit.
fn weather_surface(
    base_color: vec3<f32>,
    roughness: f32,
    metallic: f32,
    reflectance: vec3<f32>,
    normal: vec3<f32>,
    wetness: f32,
    puddle: f32,
    snow: f32,
    ripple: vec2<f32>,
) -> WeatheredSurface {
    var s = WeatheredSurface(base_color, roughness, metallic, reflectance, normal);
    let wet = min(max(wetness, puddle), DRENCHED_MAX);
    if wet > 0.0 {
        s.base_color = base_color * (1.0 - WET_ALBEDO_LOSS * wet * porosity(roughness, metallic));
        s.roughness = mix(roughness, min(roughness, WET_ROUGHNESS), saturate(wet));
    }
    if puddle > 0.0 {
        s.roughness = mix(s.roughness, PUDDLE_ROUGHNESS, puddle);
        s.reflectance = mix(reflectance, vec3<f32>(WATER_REFLECTANCE), puddle);
        s.normal = normalize(mix(normal, vec3<f32>(ripple.x, 1.0, ripple.y), puddle));
    }
    if snow > 0.0 {
        s.base_color = mix(s.base_color, SNOW_ALBEDO, snow);
        s.roughness = mix(s.roughness, SNOW_ROUGHNESS, snow);
        s.metallic = metallic * (1.0 - snow);
    }
    return s;
}

/// Where the cloud shadow map is sampled for world position `p`: `xy` the texture coordinate,
/// `z` 1 when the point lies inside a bound map (else its transmittance is 1).
fn cloud_shadow_uv(g: ElementsGlobals, p: vec3<f32>) -> vec3<f32> {
    let l = g.light.xyz;
    if g.shadow.w > 0.5 && l.y > 0.1 {
        let uv = (p.xz - l.xz * (p.y / l.y) - g.shadow.xy) / g.shadow.z;
        if all(uv >= vec2<f32>(0.0)) && all(uv <= vec2<f32>(1.0)) {
            return vec3<f32>(uv, 1.0);
        }
    }
    return vec3<f32>(0.0);
}

/// Scale of the lit colour from the map's transmittance `t` at the point against the camera's
/// mean transmittance: `(0.25 + 0.75 t) / (0.25 + 0.75 t_cam)` (clouds.md, "Cloud shadow map").
fn cloud_shadow_ratio(t: f32, camera_transmittance: f32) -> f32 {
    let here = 0.25 + 0.75 * t;
    let camera = max(0.25 + 0.75 * camera_transmittance, 0.01);
    return clamp(here / camera, 0.0, 4.0);
}
