//! CPU transliteration of `src/surface_cover.wesl` (f32, same operation order), used by the tests
//! to check the shading numbers of `docs/research/wetness.md`, `docs/research/wet-ground.md` and
//! `docs/research/snow-cover.md`.

use bevy::math::{IVec3, Vec2, Vec3, Vec3Swizzles, Vec4};

use crate::gpu::noise_mirror::{hash33, perlin3_d};

pub(crate) const SNOW_ALBEDO: Vec3 = Vec3::new(0.80, 0.83, 0.88);
pub(crate) const SNOW_ROUGHNESS: f32 = 0.9;
pub(crate) const WET_FILM_R_E: f32 = 0.0659;
pub(crate) const WET_FILM_R_I: f32 = 0.4720;
pub(crate) const DRENCHED_ALBEDO: f32 = 0.3;
pub(crate) const WET_ROUGHNESS: f32 = 0.1;
pub(crate) const WET_GLOSS_BOOST: f32 = 0.4;
pub(crate) const WET_LEAF_SATURATE: f32 = 0.25;
pub(crate) const CANOPY_EXG_LOW: f32 = 0.10;
pub(crate) const CANOPY_EXG_HIGH: f32 = 0.80;
pub(crate) const PUDDLE_ROUGHNESS: f32 = 0.02;
pub(crate) const WATER_REFLECTANCE: f32 = 0.353_553_39;
pub(crate) const RIPPLE_STRENGTH: f32 = 0.35;
pub(crate) const RIPPLE_CELL_M: f32 = 0.4;
pub(crate) const RIPPLE_RADIUS_M: f32 = 0.2;
const RIPPLE_JITTER: f32 = 0.25;
const RIPPLE_PI: f32 = std::f32::consts::PI;
const STREAK_AMPLITUDE: f32 = 0.4;
const STREAK_SATURATION: f32 = 0.2;
const STREAK_EDGE_LOW: f32 = 0.55;
const STREAK_EDGE_HIGH: f32 = 0.8;
pub(crate) const DRENCHED_MAX: f32 = 1.4;
const STREAK_FREQUENCY_X: f32 = 8.0;
const STREAK_FREQUENCY_Y: f32 = 0.4;
const STREAK_SPEED: f32 = 0.5;

fn saturate(x: f32) -> f32 {
    x.clamp(0.0, 1.0)
}

fn mix(a: f32, b: f32, t: f32) -> f32 {
    a * (1.0 - t) + b * t
}

fn mix3(a: Vec3, b: Vec3, t: f32) -> Vec3 {
    a * (1.0 - t) + b * t
}

pub(crate) fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = saturate((x - e0) / (e1 - e0));
    t * t * (3.0 - 2.0 * t)
}

pub(crate) fn porosity(roughness: f32, metallic: f32) -> f32 {
    saturate((roughness - 0.5) / 0.4) * (1.0 - metallic)
}

pub(crate) fn mesh_snow_cover(snow: f32, normal_y: f32) -> f32 {
    let lo = 0.85 - 0.6 * snow;
    snow * smoothstep(lo, lo + 0.2, normal_y)
}

pub(crate) fn rain_streak(world: Vec3, normal: Vec3, intensity: f32, phase: f32) -> f32 {
    let vertical = 1.0 - normal.y * normal.y;
    let horizontal = normal.xz().length();
    let mut along = Vec2::X;
    if horizontal > 1e-4 {
        along = Vec2::new(-normal.z, normal.x) / horizontal;
    }
    let x = world.xz().dot(along);
    let n = 0.5
        + 0.5
            * perlin3_d(Vec3::new(
                STREAK_FREQUENCY_X * x,
                STREAK_FREQUENCY_Y * world.y + STREAK_SPEED * phase,
                0.5,
            ))
            .x;
    STREAK_AMPLITUDE
        * saturate(intensity / STREAK_SATURATION)
        * smoothstep(STREAK_EDGE_LOW, STREAK_EDGE_HIGH, n)
        * vertical
}

pub(crate) fn ripple_weights(intensity: f32) -> [f32; 4] {
    std::array::from_fn(|k| saturate((intensity - 0.25 * k as f32) * 4.0))
}

pub(crate) fn ripple_layer(xz: Vec2, layer: u32, phase: f32, weight: f32) -> Vec2 {
    if weight <= 0.0 {
        return Vec2::ZERO;
    }
    let q = xz / RIPPLE_CELL_M + layer as f32 * Vec2::new(0.37, 0.23);
    let cell = q.floor();
    let h = hash33(IVec3::new(cell.x as i32, cell.y as i32, layer as i32));
    let centre = cell + Vec2::splat(0.5) + (Vec2::new(h.x, h.y) - Vec2::splat(0.5)) * RIPPLE_JITTER;
    let d = (q - centre) * RIPPLE_CELL_M;
    let dist = d.length();
    let x = 1.0 - dist / RIPPLE_RADIUS_M;
    if x <= 0.0 {
        return Vec2::ZERO;
    }
    let drop_frac = (phase + h.z).fract();
    let time_frac = drop_frac - 1.0 + x;
    let drop_factor = saturate(0.2 + weight * 0.8 - drop_frac);
    let ring = drop_factor * x * ((time_frac * 9.0).clamp(0.0, 3.0) * RIPPLE_PI).sin();
    d / dist.max(1e-5) * (ring * RIPPLE_STRENGTH)
}

#[allow(dead_code)]
pub(crate) fn ripple_offset(xz: Vec2, phases: [f32; 4], weights: [f32; 4]) -> Vec2 {
    (0..4u32)
        .map(|k| ripple_layer(xz, k, phases[k as usize], weights[k as usize]))
        .sum()
}

/// What `weather_surface` returns.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Surface {
    pub base_color: Vec3,
    pub roughness: f32,
    pub metallic: f32,
    pub reflectance: Vec3,
    pub normal: Vec3,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn weather_surface(
    base_color: Vec3,
    roughness: f32,
    metallic: f32,
    reflectance: Vec3,
    normal: Vec3,
    wetness: f32,
    puddle: f32,
    snow: f32,
    ripple: Vec2,
) -> Surface {
    let mut s = Surface {
        base_color,
        roughness,
        metallic,
        reflectance,
        normal,
    };
    let wet = wetness.max(puddle).min(DRENCHED_MAX);
    if wet > 0.0 {
        let phi = porosity(roughness, metallic);
        let soaked = wet.min(1.0);
        let drenched = saturate((wet - 1.0) / (DRENCHED_MAX - 1.0));
        let filmed = (1.0 - WET_FILM_R_E) * (1.0 - WET_FILM_R_I) * base_color
            / (Vec3::ONE - WET_FILM_R_I * base_color);
        s.base_color = mix3(
            mix3(base_color, filmed, soaked * phi),
            base_color * DRENCHED_ALBEDO,
            drenched * phi,
        );
        let porous = roughness * (1.0 - WET_GLOSS_BOOST * soaked * phi);
        let film = (1.0 - phi) * soaked + phi * drenched;
        s.roughness = mix(porous, porous.min(WET_ROUGHNESS), film);
    }
    if puddle > 0.0 {
        s.roughness = mix(s.roughness, PUDDLE_ROUGHNESS, puddle);
        s.reflectance = mix3(reflectance, Vec3::splat(WATER_REFLECTANCE), puddle);
        s.normal = mix3(normal, Vec3::new(ripple.x, 1.0, ripple.y), puddle).normalize();
    }
    if snow > 0.0 {
        s.base_color = mix3(s.base_color, SNOW_ALBEDO, snow);
        s.roughness = mix(s.roughness, SNOW_ROUGHNESS, snow);
        s.metallic = metallic * (1.0 - snow);
    }
    s
}

pub(crate) fn wet_canopy(colour: Vec3, wetness: f32) -> Vec3 {
    if wetness <= 0.0 {
        return colour;
    }
    saturated(colour, 1.0 + WET_LEAF_SATURATE * wetness)
}

pub(crate) fn saturated(colour: Vec3, s: f32) -> Vec3 {
    let grey = Vec3::splat(colour.dot(Vec3::new(0.2126, 0.7152, 0.0722)));
    (grey + (colour - grey) * s).max(Vec3::ZERO)
}

pub(crate) fn graded(albedo: Vec3, look: Vec4) -> Vec3 {
    saturated(albedo, look.w.max(0.0)) * look.truncate()
}

pub(crate) fn canopy_share(colour: Vec3) -> f32 {
    let sum = colour.x + colour.y + colour.z;
    if sum <= 1e-6 {
        return 0.0;
    }
    let c = colour / sum;
    smoothstep(CANOPY_EXG_LOW, CANOPY_EXG_HIGH, 2.0 * c.y - c.x - c.z)
}

pub(crate) fn cloud_shadow_ratio(t: f32, mean_transmittance: f32) -> f32 {
    let here = 0.25 + 0.75 * t;
    let lights = (0.25 + 0.75 * mean_transmittance).max(0.01);
    (here / lights).clamp(0.0, 4.0)
}
