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

/// The shared globals buffer, eight `vec4<f32>` (128 B). Mirrors the Rust `ElementsGlobals`.
struct ElementsGlobals {
    /// x wetness 0–1, y snow cover 0–1 (both of exposed flat ground), z rain intensity
    /// `min(1, R_liquid / 20)`, w the mean cloud transmittance the lights carry.
    weather: vec4<f32>,
    /// xy cloud shadow map origin (metre-frame XZ, m), z extent (m), w 1 when a map is bound.
    shadow: vec4<f32>,
    /// Phase of each of the four ripple layers, 0–1.
    ripple: vec4<f32>,
    /// xy wind the weather blows toward (m/s), z streak phase (s, wraps with the noise lattice).
    wind: vec4<f32>,
    /// xyz unit direction toward the light the shadow map was made for.
    light: vec4<f32>,
    /// x world units per metre (`ElementsUnits`); the rest 0. A shader that takes a world position
    /// from Bevy divides it by this before the metre-based functions below (`rain_streak`,
    /// `ripple_offset`, `cloud_shadow_uv`).
    units: vec4<f32>,
    /// The trail field's window, as `bevy_elements::trails::surface` takes it: x texel edge (m),
    /// y texels a side, z 1 when trails run, w the parallax march's bound on |h| (m).
    trail: vec4<f32>,
    /// x the beam's share of the light on a horizontal surface (a trail's self-shadow removes
    /// it), yz the trail window's centre (metre-frame XZ), w the snow depth of exposed ground (m).
    trail_light: vec4<f32>,
}

/// Linear albedo of snow; bit-equal to `bevy_flora`'s `SNOW_ALBEDO` (`common.wesl:388`).
const SNOW_ALBEDO: vec3<f32> = vec3<f32>(0.80, 0.83, 0.88);
/// Roughness of snow; `bevy_flora` `common.wesl:424` (the plan's 0.6 is Deviation 4 of snow-cover.md).
const SNOW_ROUGHNESS: f32 = 0.9;
/// The water film's diffuse reflectances, external and internal, for water's n = 1.33: a wet
/// porous surface's albedo is `(1 − r_e)(1 − r_i) ρ / (1 − r_i ρ)` per channel, the film physics
/// of `wetness.md` Model adopted 2 (Jensen et al. 1999 §2.1). A dark channel loses more than a
/// bright one, so wet colour is more saturated (`wet-ground.md`, Model adopted 1).
const WET_FILM_R_E: f32 = 0.0659;
const WET_FILM_R_I: f32 = 0.4720;
/// The albedo a porous surface drenched to `DRENCHED_MAX` keeps: Lagarde's empirical 0.2–0.3.
const DRENCHED_ALBEDO: f32 = 0.3;
/// Roughness of a water film: a smooth or metal surface holds one as soon as it is wet, a porous
/// one only when it is drenched past saturation (wet-ground.md, Model adopted 2).
const WET_ROUGHNESS: f32 = 0.1;
/// How far a soaked, fully porous surface's gloss moves towards water's: Lagarde 3b's
/// `lerp(1, factor, 0.5 · WetLevel)` with his `factor` 0.2, i.e. 0.5 × (1 − 0.2), so the roughness
/// becomes `r · (1 − 0.4 · w · φ)` (wet-ground.md, Model adopted 1).
const WET_GLOSS_BOOST: f32 = 0.4;
/// How much a wet leaf's saturation rises at full wetness; `bevy_flora`'s `WET_LEAF_SATURATE`
/// (`common.wesl:538`), so a meadow matches the wet trees on it.
const WET_LEAF_SATURATE: f32 = 0.25;
/// The excess green `2g − r − b` (chromatic coordinates) over which a surface turns from bare
/// ground to canopy: Woebbecke et al. 1995's index, the ramp from the terrain's own layer colours
/// (grass 0.834, sand 0.046, rock 0.000, soil −0.101; `wet-ground.md`, Model adopted 3).
const CANOPY_EXG_LOW: f32 = 0.10;
const CANOPY_EXG_HIGH: f32 = 0.80;
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
/// fragment's position in metres, `normal` its geometric world normal, `intensity` the rain intensity `f`.
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

/// Wet, puddled and snowed-on material inputs, in the order of the note: darken by the wetness
/// `max(wetness, puddle)` and the porosity; gloss a porous surface by Lagarde's limited boost and
/// lay a water film on a smooth one, or on a porous one drenched past saturation
/// (`docs/research/wet-ground.md`); standing water is a mirror with water's `F0`, a level normal
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
        let phi = porosity(roughness, metallic);
        let soaked = min(wet, 1.0);
        let drenched = saturate((wet - 1.0) / (DRENCHED_MAX - 1.0));
        let filmed = (1.0 - WET_FILM_R_E) * (1.0 - WET_FILM_R_I) * base_color
            / (vec3<f32>(1.0) - WET_FILM_R_I * base_color);
        s.base_color = mix(mix(base_color, filmed, soaked * phi), base_color * DRENCHED_ALBEDO, drenched * phi);
        let porous = roughness * (1.0 - WET_GLOSS_BOOST * soaked * phi);
        let film = (1.0 - phi) * soaked + phi * drenched;
        s.roughness = mix(porous, min(porous, WET_ROUGHNESS), film);
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

/// A wet canopy's colour, `wetness` 0–1 times the canopy's share of the surface: `bevy_flora`'s wet
/// leaf (`common.wesl` `wet_on`), the saturation raised by a quarter at full wetness with the
/// luminance kept. A blade's wax beads the rain, so a canopy takes no film and no gloss; the
/// caller passes `weather_surface` only the wetness of the ground between the blades
/// (`docs/research/wet-ground.md`, Model adopted 3). Exactly the input at `wetness` 0.
fn wet_canopy(colour: vec3<f32>, wetness: f32) -> vec3<f32> {
    if wetness <= 0.0 {
        return colour;
    }
    return saturated(colour, 1.0 + WET_LEAF_SATURATE * wetness);
}

/// A colour's saturation scaled by `s` around its own linear Rec. 709 luminance: 0 is its grey,
/// 1 the colour itself, above 1 more saturated; the luminance kept, no channel below zero.
fn saturated(colour: vec3<f32>, s: f32) -> vec3<f32> {
    let grey = vec3<f32>(dot(colour, vec3<f32>(0.2126, 0.7152, 0.0722)));
    return max(grey + (colour - grey) * s, vec3<f32>(0.0));
}

/// A game's grade of an albedo, `look` = (tint, saturation) (`TerrainLook`,
/// `docs/research/terrain-look.md`): the saturation first, then the tint, so a cooling tint
/// survives a desaturation. The default (1, 1, 1, 1) returns the albedo itself to the bit, and a
/// tint alone is exactly a multiply.
fn graded(albedo: vec3<f32>, look: vec4<f32>) -> vec3<f32> {
    return saturated(albedo, max(look.w, 0.0)) * look.xyz;
}

/// The share of a surface that is green canopy, 0–1, read from its albedo: Woebbecke et al.'s
/// excess green `2g − r − b` on chromatic coordinates (each channel over their sum), which tells
/// living plants from soil, residue and stone (`docs/research/wet-ground.md`, Model adopted 3).
/// Black is not canopy.
fn canopy_share(colour: vec3<f32>) -> f32 {
    let sum = colour.r + colour.g + colour.b;
    if sum <= 1e-6 {
        return 0.0;
    }
    let c = colour / sum;
    return smoothstep(CANOPY_EXG_LOW, CANOPY_EXG_HIGH, 2.0 * c.g - c.r - c.b);
}

/// Where the cloud shadow map is sampled for position `p` in metres (world ÷ `g.units.x`): `xy`
/// the texture coordinate, `z` 1 when the point lies inside a bound map (else its transmittance is 1).
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

/// Scale of the lit colour from the map's transmittance `t` at the point against the mean
/// transmittance the lights carry: `(0.25 + 0.75 t) / (0.25 + 0.75 t_mean)`
/// (broken-cloud-light.md, Model 2).
fn cloud_shadow_ratio(t: f32, mean_transmittance: f32) -> f32 {
    let here = 0.25 + 0.75 * t;
    let lights = max(0.25 + 0.75 * mean_transmittance, 0.01);
    return clamp(here / lights, 0.0, 4.0);
}

/// The relief cue under a deck (`docs/research/snow-cover.md` Deviation 15, as revised by
/// `sky-light.md` Deviation 9): the lit colour's scale for a surface whose normal has vertical
/// component `n_y`, `deck` 0 in clear air (exactly no change) to 1 under a deck. The environment
/// map carries the overcast dome, which shades a surface by its cosine, `1 − 0.30 (1 − n_y)`
/// (measured on the sky probe's irradiance map: a side 0.70 and an underside 0.40 of a top); the
/// iteration-7 curve `1 − 0.35 (1 − n_y)^0.3` was fitted so gentle relief reads (−6.7 % at 5°),
/// standing for the sky occlusion real relief has. Their quotient, never above 1, is what the
/// dome lacks: about −6.5 % on any tilted surface, and no second darkening of a wall.
fn overcast_relief(n_y: f32, deck: f32) -> f32 {
    let tilt = max(1.0 - n_y, 0.0);
    let shape = 1.0 - 0.35 * pow(tilt, 0.3);
    let dome = 1.0 - 0.30 * tilt;
    return mix(1.0, min(shape / dome, 1.0), deck);
}
