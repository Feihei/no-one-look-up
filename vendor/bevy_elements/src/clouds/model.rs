//! A CPU mirror, in `f64`, of the cloud model of `common.wesl` and `march.wesl`, and the CPU
//! oracle tests of `docs/research/clouds.md` ("Validation targets" 1, 2, 4, 5, 6, 7, 8, 14, 16).
//!
//! The density is the shader's `cloud_medium` with the noise textures replaced by the texel
//! generators of `generate.wesl` evaluated at continuous coordinates (`noise_mirror`) and the
//! weather map replaced by `coverage_field` with the exact quantile threshold. The integrator is
//! the shader's march on a flat shell. The GPU tests of `gpu.rs` compare the kernels with these
//! functions.

use bevy::math::{DVec3, Mat4, Vec2, Vec3, Vec4};

use crate::gpu::noise_mirror::{cloud_base_at, cloud_coverage_field, cloud_detail_at};

const PI: f64 = std::f64::consts::PI;

/// Radius of the planet and of the top of the atmosphere of `Atmosphere::earth`, m.
pub(crate) const PLANET_RADIUS_M: f64 = 6_360_000.0;
const ATMOSPHERE_TOP_RADIUS_M: f64 = 6_460_000.0;
/// Hillaire 2020, Table 1: Rayleigh scattering (per channel) with an 8 km scale height, Mie
/// extinction (scattering plus absorption) with 1.2 km, ozone absorption in a tent of width
/// 30 km centred at 25 km.
const RAYLEIGH_PER_M: [f64; 3] = [5.802e-6, 13.558e-6, 33.1e-6];
const RAYLEIGH_SCALE_M: f64 = 8_000.0;
const MIE_EXTINCTION_PER_M: f64 = 4.44e-6;
const MIE_SCALE_M: f64 = 1_200.0;
const OZONE_PER_M: [f64; 3] = [0.650e-6, 1.881e-6, 0.085e-6];
const OZONE_CENTRE_M: f64 = 25_000.0;
const OZONE_HALF_WIDTH_M: f64 = 15_000.0;

/// Outside-the-atmosphere illuminance of the sun, lx.
pub(crate) const SUN_LUX: f64 = 130_000.0;
/// World size of the weather map, m, and its texels per side.
const WEATHER_EXTENT_M: f64 = 32_000.0;
const WEATHER_TEXELS: usize = 512;
/// Soft edge and dome width of the coverage threshold, field units (`weather.wesl`).
const SOFT_EDGE: f64 = 0.18;
const DOME_WIDTH: f64 = 0.37;
const BASE_TILE_M: f64 = 4096.0;
const DETAIL_TILE_M: f64 = 640.0;
const WIND_SKEW_M: f64 = 500.0;
const DETAIL_EROSION: f64 = 0.2;
const MIN_HEIGHT_SCALE: f64 = 0.3;
/// The cumuliform density of `common.wesl` (`docs/research/cumulus-shape.md`).
const CUMULUS_BASE_RISE: f64 = 0.06;
const CUMULUS_TOP_ROUND: f64 = 0.25;
const CUMULUS_EDGE_HEIGHT: f64 = 0.45;
const CORE_PROFILE: f64 = 0.75;
const MAX_MARCH_M: f64 = 32_000.0;
const FADE_START_M: f64 = 16_000.0;
const EMPTY_LIMIT: u32 = 10;
const MIN_TRANSMITTANCE: f64 = 0.01;
const CONE_TAN: f64 = 0.1;
const AMBIENT_BOTTOM: f64 = 0.2;
/// Distance along the light direction of each light-path sample, and the length it stands for.
const LIGHT_DISTANCES_M: [f64; 6] = [15.0, 60.0, 150.0, 330.0, 690.0, 1800.0];
const LIGHT_LENGTHS_M: [f64; 6] = [30.0, 60.0, 120.0, 240.0, 480.0, 1740.0];

/// `smoothstep(a, b, x)`.
fn smoothstep(a: f64, b: f64, x: f64) -> f64 {
    let t = ((x - a) / (b - a)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// WGSL `mix`.
fn mix(a: f64, b: f64, t: f64) -> f64 {
    a * (1.0 - t) + b * t
}

// ---------------------------------------------------------------------------------------------
// Phase function and multiple scattering
// ---------------------------------------------------------------------------------------------

/// Henyey–Greenstein phase function, sr⁻¹.
fn hg(cos_t: f64, g: f64) -> f64 {
    let g2 = g * g;
    (1.0 - g2) / (4.0 * PI * (1.0 + g2 - 2.0 * g * cos_t).powf(1.5))
}

/// `phase` of `common.wesl`: the single-scattering dual lobe, forward lobe (g = 0.8) weighted
/// 0.7, backward lobe (g = −0.5) weighted 0.3.
pub(crate) fn phase(cos_t: f64) -> f64 {
    0.7 * hg(cos_t, 0.8) + 0.3 * hg(cos_t, -0.5)
}

/// Diffusion's extrapolation length in optical depths, `2 / (3 (1 − g))` with `g = 0.8`
/// (`march.wesl`; `cloud-lighting.md`).
pub(crate) const DIFFUSE_EXTRAPOLATION: f64 = 10.0 / 3.0;
/// Albedo of the diffuse source, fitted to the Monte Carlo slabs.
pub(crate) const DIFFUSE_ALBEDO: f64 = 0.68;
/// How far behind a sample (away from the light) the cloud behind it is sampled, m; the sample
/// stands for twice this.
const BELOW_M: f64 = 300.0;

/// `cloud_scatter` of `march.wesl` per unit of the light's illuminance, without the ambient:
/// single scattering `powder · e^{−τ_l} p(θ)` plus the diffuse fluence of a conservative slab
/// between extrapolated boundaries, `α √μ_l / π · (τ_b + z) / (τ_l + τ_b + 2z)`.
pub(crate) fn scatter_source(tau_l: f64, tau_b: f64, mu_l: f64, cos_t: f64, powder: f64) -> f64 {
    let z = DIFFUSE_EXTRAPOLATION;
    let diffuse =
        DIFFUSE_ALBEDO / PI * mu_l.max(0.0).sqrt() * (tau_b + z) / (tau_l + tau_b + 2.0 * z);
    powder * (-tau_l).exp() * phase(cos_t) + diffuse
}

// ---------------------------------------------------------------------------------------------
// The atmosphere
// ---------------------------------------------------------------------------------------------

/// Transmittance to space of the Earth medium (Hillaire 2020, Table 1) along a straight ray from
/// a point at `altitude_m` whose direction makes cosine `cos_zenith` with the local vertical:
/// the optical depth integrated numerically on the true spherical geometry, to the top of the
/// atmosphere whether or not the ray meets the planet first.
pub(crate) fn optical_transmittance(altitude_m: f64, cos_zenith: f64) -> [f64; 3] {
    const SAMPLES: usize = 4000;
    let r0 = PLANET_RADIUS_M + altitude_m;
    let mu = cos_zenith;
    let disc = r0 * r0 * (mu * mu - 1.0) + ATMOSPHERE_TOP_RADIUS_M * ATMOSPHERE_TOP_RADIUS_M;
    let length = (-r0 * mu + disc.max(0.0).sqrt()).max(0.0);
    let ds = length / SAMPLES as f64;
    let mut tau = [0.0_f64; 3];
    for k in 0..SAMPLES {
        let s = (k as f64 + 0.5) * ds;
        let r = (r0 * r0 + 2.0 * r0 * mu * s + s * s).sqrt();
        let h = (r - PLANET_RADIUS_M).max(0.0);
        let rayleigh = (-h / RAYLEIGH_SCALE_M).exp();
        let mie = (-h / MIE_SCALE_M).exp();
        let ozone = (1.0 - (h - OZONE_CENTRE_M).abs() / OZONE_HALF_WIDTH_M).max(0.0);
        for (t, (ray, oz)) in tau
            .iter_mut()
            .zip(RAYLEIGH_PER_M.into_iter().zip(OZONE_PER_M))
        {
            *t += ds * (ray * rayleigh + MIE_EXTINCTION_PER_M * mie + oz * ozone);
        }
    }
    tau.map(|t| (-t).exp())
}

/// The same transmittance for the medium Bevy's `ScatteringMedium::earth` actually builds, which
/// the sky and therefore the cloud light LUT use: its falloffs are functions of the height as a
/// fraction `x = h / T` of the atmosphere's thickness `T` = 100 km (`Atmosphere::earth`), not of
/// the absolute height Hillaire's table assumes. Exponential: `(e^{−x/s} − e^{−1/s}) / (1 −
/// e^{−1/s})` with `s` = 8/60 (Rayleigh) and 1.2/60 (Mie), i.e. scale heights of 13.3 km and
/// 2 km on a 100 km atmosphere; ozone is a tent of centre 0.25 and half-width 0.15 (25 ± 15 km,
/// as in Hillaire). Extinction coefficients as in Hillaire's table.
pub(crate) fn bevy_optical_transmittance(altitude_m: f64, cos_zenith: f64) -> [f64; 3] {
    const SAMPLES: usize = 4000;
    let thickness = ATMOSPHERE_TOP_RADIUS_M - PLANET_RADIUS_M;
    let exponential = |x: f64, s: f64| {
        let end = (-1.0 / s).exp();
        (((-x / s).exp()) - end) / (1.0 - end)
    };
    let r0 = PLANET_RADIUS_M + altitude_m;
    let mu = cos_zenith;
    let disc = r0 * r0 * (mu * mu - 1.0) + ATMOSPHERE_TOP_RADIUS_M * ATMOSPHERE_TOP_RADIUS_M;
    let length = (-r0 * mu + disc.max(0.0).sqrt()).max(0.0);
    let ds = length / SAMPLES as f64;
    let mut tau = [0.0_f64; 3];
    for k in 0..SAMPLES {
        let s = (k as f64 + 0.5) * ds;
        let r = (r0 * r0 + 2.0 * r0 * mu * s + s * s).sqrt();
        let x = ((r - PLANET_RADIUS_M) / thickness).clamp(0.0, 1.0);
        let rayleigh = exponential(x, 8.0 / 60.0);
        let mie = exponential(x, 1.2 / 60.0);
        let ozone = (1.0 - (x - 0.25).abs() / 0.15).max(0.0);
        for (t, (ray, oz)) in tau
            .iter_mut()
            .zip(RAYLEIGH_PER_M.into_iter().zip(OZONE_PER_M))
        {
            *t += ds * (ray * rayleigh + MIE_EXTINCTION_PER_M * mie + oz * ozone);
        }
    }
    tau.map(|t| (-t).exp())
}

/// Mirror of `planet_visibility` of `common.wesl`: 0 below the planet's horizon as seen from
/// altitude `altitude_m`, blended over ±0.0044 rad (half a sun disc).
pub(crate) fn planet_visibility(altitude_m: f64, cos_zenith: f64) -> f64 {
    let h = altitude_m;
    let r = PLANET_RADIUS_M;
    let mu_h = -(h * (2.0 * r + h)).max(0.0).sqrt() / (r + h);
    smoothstep(mu_h - 0.0044, mu_h + 0.0044, cos_zenith)
}

// ---------------------------------------------------------------------------------------------
// Density
// ---------------------------------------------------------------------------------------------

/// Peak extinction by cloud type, m⁻¹: stratus 0.05, cumulus 0.085, cumulonimbus 0.12.
pub(crate) fn sigma_peak(kind: f64) -> f64 {
    if kind <= 0.5 {
        mix(0.05, 0.085, kind * 2.0)
    } else {
        mix(0.085, 0.12, (kind - 0.5) * 2.0)
    }
}

/// Height-gradient quadruple by type: stratus, cumulus and cumulonimbus lerped.
pub(crate) fn gradient_quad(kind: f64) -> [f64; 4] {
    const STRATUS: [f64; 4] = [0.0, 0.1, 0.2, 0.3];
    const CUMULUS: [f64; 4] = [0.0, 0.1625, 0.88, 0.98];
    const NIMBUS: [f64; 4] = [0.0, 0.1, 0.9, 1.0];
    let t = (kind * 2.0).clamp(0.0, 1.0);
    let u = (kind * 2.0 - 1.0).clamp(0.0, 1.0);
    std::array::from_fn(|i| mix(mix(STRATUS[i], CUMULUS[i], t), NIMBUS[i], u))
}

/// Two combined remaps: rises from `q[0]` to `q[1]`, falls from `q[2]` to `q[3]`.
pub(crate) fn height_gradient(f: f64, q: [f64; 4]) -> f64 {
    let up = ((f - q[0]) / (q[1] - q[0]).max(1e-4)).clamp(0.0, 1.0);
    let down = 1.0 - ((f - q[2]) / (q[3] - q[2]).max(1e-4)).clamp(0.0, 1.0);
    up * down
}

/// `cloud_gradient`: the height gradient with the cloud height following the dome factor (the
/// weather map's A channel): a blob is full height in its core and shorter toward its edge.
pub(crate) fn cloud_gradient(f: f64, kind: f64, dome: f64) -> f64 {
    let lean = (kind * 4.0).clamp(0.0, 1.0);
    let scale = mix(1.0, mix(MIN_HEIGHT_SCALE, 1.0, dome.sqrt()), lean);
    height_gradient((f / scale).min(1.0), gradient_quad(kind))
}

/// The exponent of the coverage erosion `r′ = r^e`: 1 below 70 % of the height, falling to
/// `lerp(1, 0.5, β)` at 80 % with `β = S(0.5, 1, kind)`, the anvil inflation of the top.
pub(crate) fn coverage_exponent(f: f64, kind: f64) -> f64 {
    let beta = smoothstep(0.5, 1.0, kind);
    mix(1.0, mix(1.0, 0.5, beta), ((f - 0.7) / 0.1).clamp(0.0, 1.0))
}

/// The eroded coverage `r′` at height fraction `f`.
pub(crate) fn eroded_coverage(r: f64, f: f64, kind: f64) -> f64 {
    r.powf(coverage_exponent(f, kind))
}

/// `cumuliform`: weight of the cumuliform density, 0 for stratus up to type 0.25, 1 from
/// cumulus (0.5) up.
pub(crate) fn cumuliform(kind: f64) -> f64 {
    (4.0 * kind - 1.0).clamp(0.0, 1.0)
}

/// `cumulus_envelope`: the rounded top, full up to `CUMULUS_TOP_ROUND` of the blob's height
/// `s = mix(CUMULUS_EDGE_HEIGHT, 1, √dome)` and 0 from `s` up, C¹.
pub(crate) fn cumulus_envelope(f: f64, dome: f64) -> f64 {
    let s = mix(CUMULUS_EDGE_HEIGHT, 1.0, dome.sqrt());
    1.0 - smoothstep(s * CUMULUS_TOP_ROUND, s, f)
}

/// `cumulus_base`: the flat base, a factor on the eroded density.
pub(crate) fn cumulus_base(f: f64) -> f64 {
    smoothstep(0.0, CUMULUS_BASE_RISE, f)
}

/// `cumulus_profile`: the dimensional profile, the top times the eroded coverage times the
/// convective cell (the base texture's G channel), whose gaps close as `cover⁴` → 1.
pub(crate) fn cumulus_profile(f: f64, dome: f64, r: f64, cell: f64, cover: f64) -> f64 {
    let closed = cover.clamp(0.0, 1.0).powi(4);
    let gaps = mix(CORE_PROFILE * smoothstep(0.25, 0.65, cell), 1.0, closed);
    cumulus_envelope(f, dome) * r * gaps
}

/// `billow_weight`: wispy at the base, billowy above it.
fn billow_weight(f: f64) -> f64 {
    smoothstep(0.05, 0.25, f)
}

/// `noise_composite` of the base Perlin-Worley `shape`, the base Worley octaves `lf` and the
/// detail octaves `hf`.
pub(crate) fn noise_composite(f: f64, shape: f64, lf: f64, hf: f64) -> f64 {
    let wispy = 0.6 * shape + 0.4 * (1.0 - hf);
    let billowy = 0.5 * shape + 0.3 * lf + 0.2 * hf;
    mix(wispy, billowy, billow_weight(f))
}

/// `noise_bound`: the largest `noise_composite` any detail value gives.
pub(crate) fn noise_bound(f: f64, shape: f64, lf: f64) -> f64 {
    mix(
        0.6 * shape + 0.4,
        0.5 * shape + 0.3 * lf + 0.2,
        billow_weight(f),
    )
}

/// `value_erosion`: `√saturate((noise − (1 − d)) / d)`, 0 for `d ≤ 0`.
pub(crate) fn value_erosion(noise: f64, d: f64) -> f64 {
    if d <= 0.0 {
        return 0.0;
    }
    ((noise - (1.0 - d)) / d).clamp(0.0, 1.0).sqrt()
}

/// `detail_weight`: how much of the detail noise a fine step of `dt` metres resolves.
pub(crate) fn detail_weight(dt: f64) -> f64 {
    1.0 - smoothstep(DETAIL_TILE_M / 8.0, DETAIL_TILE_M / 4.0, dt)
}

/// Field value of the weather map at a world position, `[0, 1]`: the position is wrapped to the
/// field's 32 km period in `f64` first, so the `f32` field keeps its precision at any drift.
fn field_at(xz: [f64; 2], drift: [f64; 2]) -> f64 {
    let wrap = |x: f64, d: f64| (x - d).rem_euclid(WEATHER_EXTENT_M) as f32;
    f64::from(cloud_coverage_field(
        [wrap(xz[0], drift[0]), wrap(xz[1], drift[1])],
        [0.0, 0.0],
    ))
}

/// The coverage field of every texel of the 512² weather map (texel 62.5 m, origin 0), times
/// 256 (the GPU's histogram units), sorted ascending.
pub(crate) fn sorted_field(drift: [f64; 2]) -> Vec<f32> {
    let texel = WEATHER_EXTENT_M / WEATHER_TEXELS as f64;
    let mut values = Vec::with_capacity(WEATHER_TEXELS * WEATHER_TEXELS);
    for j in 0..WEATHER_TEXELS {
        for i in 0..WEATHER_TEXELS {
            let xz = [(i as f64 + 0.5) * texel, (j as f64 + 0.5) * texel];
            values.push((field_at(xz, drift) * 256.0) as f32);
        }
    }
    values.sort_unstable_by(f32::total_cmp);
    values
}

/// The exact quantile threshold of an ascending-sorted field: the value above which
/// `round(cover · N)` of the `N` texels lie (midway between the two neighbouring field values).
/// 256 when no texel is covered, the field minimum when every texel is.
pub(crate) fn threshold_of(sorted: &[f32], cover: f64) -> f64 {
    let n = sorted.len();
    let k = ((cover * n as f64).round() as usize).min(n);
    match k {
        0 => 256.0,
        k if k == n => f64::from(sorted[0]),
        k => f64::from(sorted[n - k - 1] + sorted[n - k]) / 2.0,
    }
}

/// Threshold `θ` of the coverage field, in the GPU's units (field × 256, `threshold[0]` of
/// `weather.wesl`), such that exactly `round(cover · 512²)` texels of the 512² weather map
/// (texel 62.5 m, origin 0) have `n ≥ θ`.
pub(crate) fn coverage_threshold(cover: f64, drift: [f64; 2]) -> f64 {
    threshold_of(&sorted_field(drift), cover)
}

/// What `cloud_medium` reads from the weather map and the base noise at a point.
struct ShapeSample {
    /// Height fraction in the layer.
    f: f64,
    /// Horizontal position of the noise lookup (wind skew applied), m.
    lean: [f64; 2],
    /// Coverage `r`, `(0, 1]`.
    coverage: f64,
    /// Dome factor (the weather map's A channel).
    dome: f64,
    /// Base Perlin-Worley shape `S`.
    base_shape: f64,
    /// Base Worley octaves `lf`.
    lf: f64,
    /// The base texture's G channel: the convective cell.
    cell: f64,
}

impl ShapeSample {
    /// The stratiform base shape times the height gradient, `S · G`.
    fn stratiform_shape(&self, kind: f64) -> f64 {
        self.base_shape * cloud_gradient(self.f, kind, self.dome)
    }

    /// The cumuliform dimensional profile at eroded coverage `r` under the weather's `cover`.
    fn profile(&self, r: f64, cover: f64) -> f64 {
        cumulus_profile(self.f, self.dome, r, self.cell, cover)
    }
}

/// The cloud layer and weather of a mirror evaluation.
#[derive(Clone, Debug)]
pub(crate) struct MirrorParams {
    /// Layer base altitude, m.
    pub(crate) base_m: f64,
    /// Layer top altitude, m.
    pub(crate) top_m: f64,
    /// Cloud type 0–1.
    pub(crate) kind: f64,
    /// Cloud cover 0–1.
    pub(crate) cover: f64,
    /// Cloud drift (weather field, base and detail noise), m.
    pub(crate) drift: [f64; 2],
    /// Unit XZ wind direction: the cloud tops lean along it.
    pub(crate) wind: [f64; 2],
    /// Precipitation darkening `min(R / 25, 1)`.
    pub(crate) precipitation: f64,
    /// Coverage threshold, field × 256.
    theta: f64,
}

impl MirrorParams {
    /// A layer with the exact quantile threshold of `cover` for `drift`. The wind is +X and there
    /// is no precipitation; set the fields for anything else.
    pub(crate) fn new(base_m: f64, top_m: f64, kind: f64, cover: f64, drift: [f64; 2]) -> Self {
        Self::from_sorted(base_m, top_m, kind, cover, drift, &sorted_field(drift))
    }

    /// Like [`new`](Self::new) with the sorted field of `drift` already computed.
    pub(crate) fn from_sorted(
        base_m: f64,
        top_m: f64,
        kind: f64,
        cover: f64,
        drift: [f64; 2],
        sorted: &[f32],
    ) -> Self {
        Self {
            base_m,
            top_m,
            kind,
            cover,
            drift,
            wind: [1.0, 0.0],
            precipitation: 0.0,
            theta: threshold_of(sorted, cover),
        }
    }

    /// Weather map at a world position: coverage `r` and the dome factor (A channel).
    pub(crate) fn weather_at(&self, xz: [f64; 2]) -> (f64, f64) {
        let n = field_at(xz, self.drift) * 256.0;
        let mut coverage = (0.5 + (n - self.theta) / (SOFT_EDGE * 256.0)).clamp(0.0, 1.0);
        if self.cover >= 0.995 {
            coverage = 1.0;
        } else if self.cover <= 0.005 {
            coverage = 0.0;
        }
        let dome = ((n - self.theta) / (DOME_WIDTH * 256.0)).clamp(0.0, 1.0);
        (coverage, dome)
    }

    /// Base shape, height gradient and coverage at altitude `h` over `xz`; `None` outside the
    /// layer or where the weather map has no cloud.
    fn shape_at(&self, h: f64, xz: [f64; 2]) -> Option<ShapeSample> {
        let f = (h - self.base_m) / (self.top_m - self.base_m);
        if !(0.0..=1.0).contains(&f) {
            return None;
        }
        let (coverage, dome) = self.weather_at(xz);
        if coverage <= 0.0 {
            return None;
        }
        let lean = [
            xz[0] - f * self.wind[0] * WIND_SKEW_M,
            xz[1] - f * self.wind[1] * WIND_SKEW_M,
        ];
        let b = cloud_base_at(Self::noise_coordinate(
            (lean[0] - self.drift[0]) / BASE_TILE_M,
            h / BASE_TILE_M,
            (lean[1] - self.drift[1]) / BASE_TILE_M,
        ))
        .map(f64::from);
        let lf = 0.625 * b[1] + 0.25 * b[2] + 0.125 * b[3];
        Some(ShapeSample {
            f,
            lean,
            coverage,
            dome,
            base_shape: ((b[0] + 1.0 - lf) / (2.0 - lf)).clamp(0.0, 1.0),
            lf,
            cell: b[1],
        })
    }

    /// A tile coordinate wrapped to [0, 1) in `f64` before it loses precision in `f32`.
    fn noise_coordinate(u: f64, v: f64, w: f64) -> Vec3 {
        let wrap = |x: f64| (x - x.floor()) as f32;
        Vec3::new(wrap(u), wrap(v), wrap(w))
    }

    /// Mirror of `cloud_medium`: extinction σ_t, m⁻¹, and the precipitation darkening g at
    /// altitude `h` over `xz`; `detail` (0–1) is how much of the detail noise is resolved.
    pub(crate) fn medium(&self, h: f64, xz: [f64; 2], detail: f64) -> (f64, f64) {
        let Some(s) = self.shape_at(h, xz) else {
            return (0.0, 0.0);
        };
        let r = eroded_coverage(s.coverage, s.f, self.kind);
        // Without the detail tap the detail octaves take their mean.
        let mut raw_hf = 0.5;
        if detail > 0.0 {
            let d = cloud_detail_at(Self::noise_coordinate(
                (s.lean[0] - self.drift[0]) / DETAIL_TILE_M,
                h / DETAIL_TILE_M,
                (s.lean[1] - self.drift[1]) / DETAIL_TILE_M,
            ))
            .map(f64::from);
            raw_hf = 0.625 * d[0] + 0.25 * d[1] + 0.125 * d[2];
        }
        let hf = mix(0.5, raw_hf, detail);
        let k = cumuliform(self.kind);
        let mut stratiform = 0.0;
        if k < 1.0 {
            stratiform =
                ((s.stratiform_shape(self.kind) - (1.0 - r)) / r.max(1e-4)).clamp(0.0, 1.0) * r;
            if detail > 0.0 && stratiform > 0.0 {
                let wispy = mix(raw_hf, 1.0 - raw_hf, (10.0 * s.f).clamp(0.0, 1.0));
                let lo = DETAIL_EROSION * wispy;
                stratiform = ((stratiform - lo) / (1.0 - lo)).clamp(0.0, 1.0);
            }
        }
        let mut cumulus = 0.0;
        if k > 0.0 {
            cumulus = value_erosion(
                noise_composite(s.f, s.base_shape, s.lf, hf),
                s.profile(r, self.cover),
            ) * cumulus_base(s.f);
        }
        (
            sigma_peak(self.kind) * mix(stratiform, cumulus, k),
            self.precipitation * s.coverage,
        )
    }
}

/// Extinction and precipitation darkening of the density mirror with the detail noise on, at
/// altitude `altitude_m` over world `xz`.
pub(crate) fn density(altitude_m: f64, xz: [f64; 2], p: &MirrorParams) -> (f64, f64) {
    p.medium(altitude_m, xz, 1.0)
}

/// Vertical optical depth `τ_col = ∫ σ_t dh` through the layer over `xz`, midpoint rule with
/// `steps` samples, detail noise on.
pub(crate) fn column_optical_depth(p: &MirrorParams, xz: [f64; 2], steps: u32) -> f64 {
    let dh = (p.top_m - p.base_m) / f64::from(steps);
    (0..steps)
        .map(|k| p.medium(p.base_m + (f64::from(k) + 0.5) * dh, xz, 1.0).0 * dh)
        .sum()
}

// ---------------------------------------------------------------------------------------------
// The march
// ---------------------------------------------------------------------------------------------

/// What the march samples: the coarse base shape and the extinction.
pub(crate) trait Medium {
    /// Layer base altitude, m.
    fn base_m(&self) -> f64;
    /// Layer top altitude, m.
    fn top_m(&self) -> f64;
    /// `coarse_shape`: the base-shape term `S′·r′`, 0 outside cloud.
    fn coarse(&self, altitude: f64, xz: [f64; 2]) -> f64;
    /// `cloud_medium`: (σ_t, g).
    fn sample(&self, altitude: f64, xz: [f64; 2], detail: f64) -> (f64, f64);
}

impl Medium for MirrorParams {
    fn base_m(&self) -> f64 {
        self.base_m
    }

    fn top_m(&self) -> f64 {
        self.top_m
    }

    fn coarse(&self, altitude: f64, xz: [f64; 2]) -> f64 {
        self.shape_at(altitude, xz).map_or(0.0, |s| {
            let r = eroded_coverage(s.coverage, s.f, self.kind);
            let k = cumuliform(self.kind);
            let mut stratiform = 0.0;
            if k < 1.0 {
                stratiform = s.stratiform_shape(self.kind) * r;
            }
            let mut cumulus = 0.0;
            if k > 0.0 {
                cumulus = (noise_bound(s.f, s.base_shape, s.lf) - (1.0 - s.profile(r, self.cover)))
                    .max(0.0);
            }
            mix(stratiform, cumulus, k)
        })
    }

    fn sample(&self, altitude: f64, xz: [f64; 2], detail: f64) -> (f64, f64) {
        self.medium(altitude, xz, detail)
    }
}

/// The uniform slab of the shader's debug mode: constant extinction between two altitudes.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Slab {
    /// Extinction, m⁻¹.
    pub(crate) sigma: f64,
    /// Slab base altitude, m.
    pub(crate) base_m: f64,
    /// Slab top altitude, m.
    pub(crate) top_m: f64,
}

impl Slab {
    fn contains(&self, altitude: f64) -> bool {
        (self.base_m..=self.top_m).contains(&altitude)
    }
}

impl Medium for Slab {
    fn base_m(&self) -> f64 {
        self.base_m
    }

    fn top_m(&self) -> f64 {
        self.top_m
    }

    fn coarse(&self, altitude: f64, _xz: [f64; 2]) -> f64 {
        if self.contains(altitude) { 1.0 } else { 0.0 }
    }

    fn sample(&self, altitude: f64, _xz: [f64; 2], _detail: f64) -> (f64, f64) {
        if self.contains(altitude) {
            (self.sigma, 0.0)
        } else {
            (0.0, 0.0)
        }
    }
}

/// A ray through a flat cloud shell and the light it is lit by. The atmosphere's transmittance
/// is 1 and the planet never shadows the light: `sun` is the illuminance at every sample.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Scene {
    /// Camera: world x, altitude, z; m.
    pub(crate) camera: DVec3,
    /// Unit view direction.
    pub(crate) dir: DVec3,
    /// Unit vector toward the light.
    pub(crate) light: DVec3,
    /// Cosine between the view and light directions, the phase function's argument.
    pub(crate) cos_theta: f64,
    /// Illuminance of the light, lx.
    pub(crate) sun: f64,
    /// Ambient source term `E_d / 2π`, cd/m².
    pub(crate) ambient: f64,
    /// Whether the powder term darkens single scattering.
    pub(crate) powder: bool,
    /// Primary steps Q.
    pub(crate) steps: u32,
    /// Jitter of the first sample inside its step, 0–1.
    pub(crate) jitter: f64,
}

impl Scene {
    /// Full sun, no ambient, powder on, 64 steps; the phase cosine is that of `dir` and `light`.
    pub(crate) fn new(camera: DVec3, dir: DVec3, light: DVec3) -> Self {
        let (dir, light) = (dir.normalize(), light.normalize());
        Self {
            camera,
            dir,
            light,
            cos_theta: dir.dot(light),
            sun: SUN_LUX,
            ambient: 0.0,
            powder: true,
            steps: 64,
            jitter: 0.5,
        }
    }
}

/// Result of a march: radiance, cd/m² (a single channel, before exposure and aerial
/// perspective), and the view transmittance.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Marched {
    /// Radiance toward the camera.
    pub(crate) radiance: f64,
    /// Transmittance of the cloud along the ray when the march ended.
    pub(crate) transmittance: f64,
}

/// Mirror of `view_ray` of `common.wesl`: the origin and unit direction, in metres, of the ray
/// through screen position `uv` (0–1, top-left origin) of a view whose matrices are in world
/// units, `units` of them to the metre. A perspective view's starts at the eye `cam` (world);
/// an orthographic view's runs along the view direction from the pixel's point on the plane
/// through `focus` (metres) across it.
pub(crate) fn view_ray(
    uv: Vec2,
    world_from_clip: Mat4,
    cam: Vec3,
    orthographic: bool,
    focus: Vec3,
    units: f32,
) -> (Vec3, Vec3) {
    let ndc = Vec2::new(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    let n = world_from_clip * Vec4::new(ndc.x, ndc.y, 1.0, 1.0);
    let on_near = n.truncate() / n.w / units;
    if orthographic {
        let m = world_from_clip * Vec4::new(ndc.x, ndc.y, 0.5, 1.0);
        let dir = (m.truncate() / m.w / units - on_near).normalize();
        return (on_near + dir * (focus - on_near).dot(dir), dir);
    }
    let eye = cam / units;
    (eye, (on_near - eye).normalize())
}

/// Where a ray from altitude `y` with vertical component `dy` is between two altitudes (the
/// flat shell): `(t_enter, t_exit)`, `t_enter ≥ 0`.
fn flat_range(y: f64, dy: f64, base_m: f64, top_m: f64) -> (f64, f64) {
    if dy.abs() < 1e-12 {
        return if (base_m..=top_m).contains(&y) {
            (0.0, f64::INFINITY)
        } else {
            (0.0, 0.0)
        };
    }
    let (a, b) = ((base_m - y) / dy, (top_m - y) / dy);
    (a.min(b).max(0.0), a.max(b))
}

/// Optical depth toward the light from the point `off` (offset from the camera): the shader's six
/// base-shape samples at golden-angle cone directions, times `1 + g` per sample.
fn light_optical_depth(off: DVec3, light: DVec3, camera: DVec3, medium: &impl Medium) -> f64 {
    let helper = if light.y.abs() > 0.99 {
        DVec3::X
    } else {
        DVec3::Y
    };
    let e1 = light.cross(helper).normalize();
    let e2 = light.cross(e1);
    let mut tau = 0.0;
    for (j, (dist, len)) in LIGHT_DISTANCES_M
        .into_iter()
        .zip(LIGHT_LENGTHS_M)
        .enumerate()
    {
        let phi = 2.399_963_23 * (j + 1) as f64;
        let perp = phi.cos() * e1 + phi.sin() * e2;
        let o = off + light * dist + perp * (CONE_TAN * dist);
        let (sigma, g) = medium.sample(camera.y + o.y, [camera.x + o.x, camera.z + o.z], 0.0);
        tau += (1.0 + g) * sigma * len;
    }
    tau
}

/// `below_optical_depth` of `march.wesl`: one base-shape sample [`BELOW_M`] behind the point
/// along the light, standing for twice that, times `1 + g`.
fn below_optical_depth(off: DVec3, light: DVec3, camera: DVec3, medium: &impl Medium) -> f64 {
    let o = off - light * BELOW_M;
    let (sigma, g) = medium.sample(camera.y + o.y, [camera.x + o.x, camera.z + o.z], 0.0);
    (1.0 + g) * sigma * 2.0 * BELOW_M
}

/// Mirror of the fine march of `march.wesl` (coarse and fine phases, early exit, per-sample
/// segment, energy-conserving `S (1 − T_s)` integration, ambient) on a flat shell.
pub(crate) fn march(scene: &Scene, medium: &impl Medium) -> Marched {
    let (lo, hi) = flat_range(scene.camera.y, scene.dir.y, medium.base_m(), medium.top_m());
    let (t0, t1) = (lo, hi.min(MAX_MARCH_M));
    if t1 <= t0 {
        return Marched {
            radiance: 0.0,
            transmittance: 1.0,
        };
    }
    let steps = f64::from(scene.steps);
    let dt = ((t1 - t0) / steps)
        .clamp(25.0, 400.0)
        .max((t1 - t0) / (1.5 * steps));
    let powder_weight = (0.5 - 0.5 * scene.cos_theta).clamp(0.0, 1.0);
    let mut radiance = 0.0;
    let mut transmittance = 1.0;
    let mut t = t0 + dt * scene.jitter;
    let mut coarse = true;
    let mut empty = 0;
    let mut resume = t0;
    for _ in 0..scene.steps * 3 {
        if t >= t1 || transmittance < MIN_TRANSMITTANCE {
            break;
        }
        let off = scene.dir * t;
        let h = scene.camera.y + off.y;
        let xz = [scene.camera.x + off.x, scene.camera.z + off.z];
        if coarse {
            if medium.coarse(h, xz) > 0.0 {
                coarse = false;
                empty = 0;
                t = (t - 2.0 * dt).max(resume);
            } else {
                t += 2.0 * dt;
            }
            continue;
        }
        let (mut sigma, _) = medium.sample(h, xz, detail_weight(dt));
        sigma *= 1.0 - smoothstep(FADE_START_M, MAX_MARCH_M, t);
        if sigma <= 0.0 {
            empty += 1;
            if empty >= EMPTY_LIMIT {
                coarse = true;
                resume = t + dt;
            }
            t += dt;
            continue;
        }
        empty = 0;
        let seg = (t1 - (t - dt * scene.jitter)).clamp(0.0, dt);
        let step_t = (-sigma * seg).exp();
        let tau_l = light_optical_depth(off, scene.light, scene.camera, medium);
        let tau_b = below_optical_depth(off, scene.light, scene.camera, medium);
        let powder = if scene.powder {
            mix(1.0, 1.0 - (-2.0 * tau_l).exp(), powder_weight)
        } else {
            1.0
        };
        let mut source =
            scene.sun * scatter_source(tau_l, tau_b, scene.light.y, scene.cos_theta, powder);
        let f = (h - medium.base_m()) / (medium.top_m() - medium.base_m());
        source += scene.ambient * (AMBIENT_BOTTOM + (1.0 - AMBIENT_BOTTOM) * f);
        radiance += transmittance * source * (1.0 - step_t);
        transmittance *= step_t;
        t += dt;
    }
    // An exhausted march is opaque (march.wesl; broken-cloud-light.md, Model 4).
    if transmittance < MIN_TRANSMITTANCE {
        radiance /= 1.0 - transmittance;
        transmittance = 0.0;
    }
    Marched {
        radiance,
        transmittance,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::climate::Weather;
    use crate::clouds::{CloudLayer, CloudQuality};
    use crate::plugin::ElementsQuality;

    // -----------------------------------------------------------------------------------------
    // Validation 1: the cloud type function
    // -----------------------------------------------------------------------------------------

    /// A preset's target weather at a temperate climate (`climate/weather.rs`, wind factor 1) and
    /// the layer the note's plan table gives for it.
    struct Preset {
        name: &'static str,
        cover: f32,
        mm_h: f32,
        rh: f32,
        wind: f32,
        storm: f32,
        kind: f32,
        base_m: f32,
        top_m: f32,
        darkening: f32,
    }

    #[rustfmt::skip]
    const PRESETS: [Preset; 10] = [
        Preset { name: "Clear",        cover: 0.05, mm_h: 0.0,  rh: 0.50, wind: 3.0,  storm: 0.0, kind: 0.5, base_m: 1500.0, top_m: 3300.0, darkening: 0.0 },
        Preset { name: "PartlyCloudy", cover: 0.40, mm_h: 0.0,  rh: 0.60, wind: 4.0,  storm: 0.0, kind: 0.5, base_m: 1500.0, top_m: 3300.0, darkening: 0.0 },
        Preset { name: "Overcast",     cover: 0.95, mm_h: 0.0,  rh: 0.75, wind: 5.0,  storm: 0.0, kind: 0.0, base_m: 1500.0, top_m: 2300.0, darkening: 0.0 },
        Preset { name: "Fog",          cover: 0.60, mm_h: 0.0,  rh: 0.99, wind: 1.0,  storm: 0.0, kind: 0.0, base_m: 600.0,  top_m: 1400.0, darkening: 0.0 },
        Preset { name: "Drizzle",      cover: 0.95, mm_h: 0.5,  rh: 0.92, wind: 4.0,  storm: 0.0, kind: 0.0, base_m: 600.0,  top_m: 1400.0, darkening: 0.02 },
        Preset { name: "Rain",         cover: 1.00, mm_h: 4.0,  rh: 0.95, wind: 7.0,  storm: 0.0, kind: 0.0, base_m: 1500.0, top_m: 2300.0, darkening: 0.16 },
        Preset { name: "Thunderstorm", cover: 1.00, mm_h: 25.0, rh: 0.97, wind: 14.0, storm: 6.0, kind: 1.0, base_m: 1500.0, top_m: 9500.0, darkening: 1.0 },
        Preset { name: "Snow",         cover: 1.00, mm_h: 1.5,  rh: 0.90, wind: 4.0,  storm: 0.0, kind: 0.0, base_m: 1500.0, top_m: 2300.0, darkening: 0.06 },
        Preset { name: "Blizzard",     cover: 1.00, mm_h: 4.0,  rh: 0.90, wind: 18.0, storm: 0.0, kind: 1.0, base_m: 1500.0, top_m: 9500.0, darkening: 0.16 },
        Preset { name: "DustStorm",    cover: 0.30, mm_h: 0.0,  rh: 0.15, wind: 16.0, storm: 0.0, kind: 0.5, base_m: 1500.0, top_m: 3300.0, darkening: 0.0 },
    ];

    fn weather_of(cover: f32, mm_h: f32, rh: f32, wind: f32, storm: f32) -> Weather {
        Weather {
            cloud_cover: cover,
            precipitation_mm_h: mm_h,
            relative_humidity: rh,
            wind_speed_ms: wind,
            storminess: storm,
            ..Weather::default()
        }
    }

    impl Preset {
        fn weather(&self) -> Weather {
            weather_of(self.cover, self.mm_h, self.rh, self.wind, self.storm)
        }
    }

    #[test]
    fn the_layer_of_every_preset_is_the_one_of_the_plan_table() {
        for p in &PRESETS {
            let w = p.weather();
            let layer = CloudLayer::from_weather(&w);
            let kind = CloudLayer::kind_from_weather(&w);
            let darkening = CloudLayer::precipitation_from_weather(&w);
            assert!((kind - p.kind).abs() <= 0.001, "{} type {kind}", p.name);
            assert!(
                (layer.base_m - p.base_m).abs() <= 1.0,
                "{} base {}",
                p.name,
                layer.base_m
            );
            assert!(
                (layer.top_m - p.top_m).abs() <= 1.0,
                "{} top {}",
                p.name,
                layer.top_m
            );
            assert!(
                (darkening - p.darkening).abs() <= 0.001,
                "{} darkening {darkening}",
                p.name
            );
        }
    }

    /// Blends compared per pair of presets. The note asks for 20 evenly spaced blends and
    /// |Δb| ≤ 0.05 between neighbours, which no continuous function of this shape meets: Clear
    /// to Blizzard steps by 0.53 between the blends 0.90 and 0.95 (the wind crosses its
    /// 12–16 m/s ramp while the cover crosses 0.9–1.0), and Clear to Fog by 0.20. The type is
    /// continuous (its steepest slope is about 10.6 per unit of blend) and meets the 0.05 bound
    /// from 400 blends on (worst 0.036, Clear to Blizzard).
    const BLENDS: usize = 400;

    #[test]
    fn the_cloud_type_is_continuous_across_every_weather_blend() {
        for from in &PRESETS {
            for to in &PRESETS {
                let blend = |s: f32| {
                    let l = |a: f32, b: f32| a + (b - a) * s;
                    let w = weather_of(
                        l(from.cover, to.cover),
                        l(from.mm_h, to.mm_h),
                        l(from.rh, to.rh),
                        l(from.wind, to.wind),
                        l(from.storm, to.storm),
                    );
                    CloudLayer::kind_from_weather(&w)
                };
                let mut previous = blend(0.0);
                for k in 1..=BLENDS {
                    let b = blend(k as f32 / BLENDS as f32);
                    assert!(
                        (b - previous).abs() <= 0.05,
                        "{} to {} jumps from {previous} to {b} at blend {k}/{BLENDS}",
                        from.name,
                        to.name
                    );
                    previous = b;
                }
            }
        }
    }

    // -----------------------------------------------------------------------------------------
    // view-anchor.md Validation 3: the rays of both projections
    // -----------------------------------------------------------------------------------------

    /// `world_from_clip` of a camera at `eye` looking at `at`, through `projection` on a 16:9
    /// target.
    fn world_from_clip(projection: bevy::camera::Projection, eye: Vec3, at: Vec3) -> Mat4 {
        let mut projection = projection;
        projection.update(1920.0, 1080.0);
        let world_from_view = bevy::prelude::Transform::from_translation(eye)
            .looking_at(at, Vec3::Y)
            .to_matrix();
        world_from_view * projection.get_clip_from_view().inverse()
    }

    /// Under an orthographic projection the rays of two pixels start at different points of the
    /// plane through the focus across the view and share the view direction, and without a focus
    /// the plane is the camera's own; under a perspective one they start at the eye and differ in
    /// direction (issue #7).
    #[test]
    fn orthographic_rays_are_parallel_and_perspective_rays_share_the_eye() {
        use bevy::camera::{
            OrthographicProjection, PerspectiveProjection, Projection, ScalingMode,
        };
        let (eye, focus) = (Vec3::new(0.0, 860.4, 1228.7), Vec3::ZERO);
        let forward = (focus - eye).normalize();
        let ortho = world_from_clip(
            Projection::from(OrthographicProjection {
                scaling_mode: ScalingMode::FixedVertical {
                    viewport_height: 80.0,
                },
                far: 4000.0,
                ..OrthographicProjection::default_3d()
            }),
            eye,
            focus,
        );
        let (a, b) = (Vec2::new(0.1, 0.2), Vec2::new(0.9, 0.7));
        let (oa, da) = view_ray(a, ortho, eye, true, focus, 1.0);
        let (ob, db) = view_ray(b, ortho, eye, true, focus, 1.0);
        assert!(oa.distance(ob) > 10.0, "{oa} {ob}");
        assert!(
            da.distance(db) < 1e-5 && da.distance(forward) < 1e-5,
            "{da} {db} {forward}"
        );
        // The origins lie on the plane through the focus across the view direction.
        assert!((oa - focus).dot(forward).abs() < 1e-2 && (ob - focus).dot(forward).abs() < 1e-2);
        // Without a focus (it is the camera) they lie on the camera's own plane.
        let (own, _) = view_ray(a, ortho, eye, true, eye, 1.0);
        assert!((own - eye).dot(forward).abs() < 1e-2 && own.distance(oa) > 1000.0);

        let persp = world_from_clip(
            Projection::from(PerspectiveProjection::default()),
            eye,
            focus,
        );
        let (pa, ea) = view_ray(a, persp, eye, false, focus, 1.0);
        let (pb, eb) = view_ray(b, persp, eye, false, focus, 1.0);
        assert_eq!((pa, pb), (eye, eye));
        assert!(ea.distance(eb) > 0.1 && (ea.length() - 1.0).abs() < 1e-5);
        // The centre of the screen looks along the view direction in both (a perspective ray is
        // the difference of the eye and a point 0.1 m from it, in f32 1500 m out: 1e-3), and an
        // orthographic one starts at the focus.
        let centre = Vec2::splat(0.5);
        assert!(
            view_ray(centre, persp, eye, false, focus, 1.0)
                .1
                .distance(forward)
                < 1e-3
        );
        assert!(
            view_ray(centre, ortho, eye, true, focus, 1.0)
                .0
                .distance(focus)
                < 1e-2
        );
    }

    /// The same scene built in feet (the camera, and so every matrix, ×3.2808) gives the same
    /// rays in metres: the same origin and direction under both projections.
    #[test]
    fn a_view_in_feet_gives_the_metre_rays_of_the_same_scene() {
        use bevy::camera::{
            OrthographicProjection, PerspectiveProjection, Projection, ScalingMode,
        };
        let u = 3.2808_f32;
        let (eye, focus) = (Vec3::new(0.0, 860.4, 1228.7), Vec3::ZERO);
        let uv = Vec2::new(0.3, 0.6);
        for (projection, orthographic) in [
            (
                Projection::from(OrthographicProjection {
                    scaling_mode: ScalingMode::FixedVertical {
                        viewport_height: 80.0,
                    },
                    far: 4000.0,
                    ..OrthographicProjection::default_3d()
                }),
                true,
            ),
            (Projection::from(PerspectiveProjection::default()), false),
        ] {
            let metres = view_ray(
                uv,
                world_from_clip(projection.clone(), eye, focus),
                eye,
                orthographic,
                focus,
                1.0,
            );
            // The world in feet: the camera at `eye · u`, and a projection whose extents are
            // in feet too.
            let mut feet_projection = projection;
            if let Projection::Orthographic(o) = &mut feet_projection {
                o.scaling_mode = ScalingMode::FixedVertical {
                    viewport_height: 80.0 * u,
                };
                o.far *= u;
            }
            let feet = view_ray(
                uv,
                world_from_clip(feet_projection, eye * u, focus * u),
                eye * u,
                orthographic,
                focus,
                u,
            );
            assert!(
                // (A perspective direction is the difference of the eye and a point 0.1 from it,
                // in f32 thousands out: ~1e-2.)
                metres.0.distance(feet.0) < 0.5 && metres.1.distance(feet.1) < 2e-2,
                "{metres:?} {feet:?}"
            );
        }
    }

    // -----------------------------------------------------------------------------------------
    // Validation 4: the phase function
    // -----------------------------------------------------------------------------------------

    #[test]
    fn the_phase_function_integrates_to_one() {
        // ∫ p dΩ = 2π ∫ p dμ, midpoint rule fine enough for the forward lobe (width ~0.03).
        let n = 400_000;
        let integral: f64 = (0..n)
            .map(|k| phase(-1.0 + (f64::from(k) + 0.5) * 2.0 / f64::from(n)))
            .sum::<f64>()
            * (2.0 / f64::from(n))
            * 2.0
            * PI;
        assert!((integral - 1.0).abs() < 1e-3, "{integral}");
    }

    #[test]
    fn the_phase_function_has_the_values_of_the_note() {
        for (cos, expected) in [(1.0, 2.5120), (0.0, 0.0224), (-1.0, 0.1467)] {
            let p = phase(cos);
            assert!(
                (p - expected).abs() <= 5.0e-5,
                "p({cos}) = {p}, note {expected}"
            );
        }
    }

    // -----------------------------------------------------------------------------------------
    // cloud-lighting.md Validation 1 and 3: the slab against Monte Carlo
    // -----------------------------------------------------------------------------------------

    /// `(τ, μ₀, view (0 above, 1 below), view zenith cosine, azimuth from the sun's side in
    /// degrees, L/E)` of a homogeneous conservative slab, by Monte Carlo transport with the
    /// march's phase (`docs/research/cloud-lighting-mc.py`).
    const MONTE_CARLO_SLAB: &[(f64, f64, u32, f64, f64, f64)] = &include!("monte_carlo_slab.in");

    /// `L / E` the march sees on a 900 m slab of optical depth `tau` lit from zenith cosine
    /// `mu0`, viewed from above or below along the row's direction, with `steps` primary steps.
    fn slab_view(tau: f64, mu0: f64, below: bool, mu_v: f64, azimuth: f64, steps: u32) -> f64 {
        let thickness = 900.0;
        let slab = Slab {
            sigma: tau / thickness,
            base_m: 1500.0,
            top_m: 1500.0 + thickness,
        };
        let light = DVec3::new((1.0 - mu0 * mu0).sqrt(), mu0, 0.0);
        let sv = (1.0 - mu_v * mu_v).sqrt();
        let a = azimuth.to_radians();
        // The viewer sits along `toward` and looks back at the slab.
        let (from, toward) = if below {
            (slab.base_m, DVec3::new(-sv, -mu_v, 0.0))
        } else {
            (slab.top_m, DVec3::new(sv * a.cos(), mu_v, sv * a.sin()))
        };
        let mut scene = Scene::new(DVec3::new(0.0, from, 0.0) + toward * 100.0, -toward, light);
        scene.sun = 1.0;
        scene.steps = steps;
        march(&scene, &slab).radiance
    }

    /// The march returns the light a cloud slab reflects and transmits: the mean squared log
    /// error over the 96 Monte Carlo cases (floor 0.01) is at most 0.08, every thick top within
    /// half to twice the reference and every base within 0.4 to 2.5 times (`cloud-lighting.md`
    /// Deviation 2). Measured: 0.067, thick tops 0.57–1.12×, bases 0.45–1.19×. Three Wrenninge
    /// octaves gave 0.82, thick tops at a sixth and bases at nothing.
    #[test]
    fn the_slab_matches_monte_carlo() {
        let mut error = 0.0;
        for &(tau, mu0, view, mu_v, azimuth, reference) in MONTE_CARLO_SLAB {
            let below = view == 1;
            let marched = slab_view(tau, mu0, below, mu_v, azimuth, 64);
            let weight = if below { 0.5 } else { 1.0 };
            error += weight * ((marched + 0.01).ln() - (reference + 0.01).ln()).powi(2);
            let ratio = marched / reference;
            if below {
                assert!(
                    (0.4..=2.5).contains(&ratio),
                    "base τ {tau} μ₀ {mu0} μ_v {mu_v}: {marched} against {reference}"
                );
            } else if tau >= 10.0 {
                assert!(
                    (0.5..=2.0).contains(&ratio),
                    "top τ {tau} μ₀ {mu0} μ_v {mu_v} az {azimuth}: {marched} against {reference}"
                );
            }
        }
        let error = error / MONTE_CARLO_SLAB.len() as f64;
        println!("mean squared log error {error:.4}");
        assert!(error <= 0.08, "{error}");
    }

    /// The march at Q = 64 is within 5 % of the same march converged at Q = 1024.
    #[test]
    fn the_integrator_converges_on_the_slab() {
        for (tau, mu0, below) in [(1.0, 0.9, false), (10.0, 0.5, false), (10.0, 0.9, true)] {
            let coarse = slab_view(tau, mu0, below, 0.7, 90.0, 64);
            let fine = slab_view(tau, mu0, below, 0.7, 90.0, 1024);
            assert!(
                (coarse / fine - 1.0).abs() <= 0.05,
                "τ {tau}, μ₀ {mu0}, below {below}: {coarse} at Q = 64, {fine} at Q = 1024"
            );
        }
    }

    // -----------------------------------------------------------------------------------------
    // Validation 6: energy
    // -----------------------------------------------------------------------------------------

    /// The most light one unit of illuminance can return per unit of source: the phase
    /// function's forward peak plus the diffuse term at its largest, `α / π`.
    fn source_bound() -> f64 {
        let peak = (0..=2000)
            .map(|k| phase(-1.0 + f64::from(k) / 1000.0))
            .fold(0.0_f64, f64::max);
        peak + DIFFUSE_ALBEDO / PI
    }

    #[test]
    fn the_source_bound_is_the_phase_peak_and_the_diffuse_albedo() {
        let bound = source_bound();
        assert!((bound - 2.7285).abs() < 1e-3, "source bound {bound}");
        for tau_l in [0.0, 1.0, 10.0] {
            for tau_b in [0.0, 5.0, 100.0] {
                for cos in [-1.0, 0.0, 1.0] {
                    let s = scatter_source(tau_l, tau_b, 1.0, cos, 1.0);
                    assert!((0.0..=bound + 1e-12).contains(&s), "{s}");
                }
            }
        }
    }

    /// Directions of the stress scenes: toward and away from the light, and rings at several
    /// elevations.
    fn stress_directions(light: DVec3) -> Vec<DVec3> {
        let mut dirs = vec![light, -light];
        for elevation in [-60.0_f64, -25.0, -8.0, 8.0, 25.0, 60.0] {
            for k in 0..12 {
                let azimuth = f64::from(k) * 30.0_f64.to_radians();
                let e = elevation.to_radians();
                dirs.push(DVec3::new(
                    e.cos() * azimuth.cos(),
                    e.sin(),
                    e.cos() * azimuth.sin(),
                ));
            }
        }
        dirs
    }

    #[test]
    fn cloud_radiance_never_exceeds_the_light_that_arrives() {
        let light = DVec3::new(0.4, 0.8, 0.3).normalize();
        let mut cumulonimbus = MirrorParams::new(1500.0, 9500.0, 1.0, 1.0, [0.0, 0.0]);
        cumulonimbus.precipitation = 0.5;
        let bound = source_bound() * SUN_LUX;
        let ambient = SUN_LUX * 0.15 / (2.0 * PI);
        let mut brightest: f64 = 0.0;
        let mut hits = 0;
        // Camera inside the layer, above it and below it.
        for altitude in [5_000.0, 10_500.0, 600.0] {
            for dir in stress_directions(light) {
                for ambient in [0.0, ambient] {
                    let mut scene = Scene::new(DVec3::new(1234.0, altitude, 987.0), dir, light);
                    scene.ambient = ambient;
                    let m = march(&scene, &cumulonimbus);
                    assert!(m.radiance.is_finite() && m.radiance >= 0.0);
                    assert!((0.0..=1.0).contains(&m.transmittance));
                    assert!(
                        m.radiance <= bound + ambient,
                        "camera {altitude} m, view {dir:?}: {} cd/m² against {}",
                        m.radiance,
                        bound + ambient
                    );
                    if m.radiance > 0.0 {
                        hits += 1;
                    }
                    brightest = brightest.max(m.radiance - ambient);
                }
            }
        }
        assert!(hits > 100, "only {hits} stress rays met cloud");
        assert!(
            brightest > 1_000.0,
            "the brightest stress ray reaches only {brightest} cd/m²"
        );
        // A cumulonimbus self-shadows far below the bound; a thin slab seen toward the light
        // comes within a third of it.
        let slab = Slab {
            sigma: 1.0 / 900.0,
            base_m: 1500.0,
            top_m: 2400.0,
        };
        let toward_light = Scene::new(DVec3::new(0.0, 1400.0, 0.0), DVec3::Y, DVec3::Y);
        let thin = march(&toward_light, &slab).radiance;
        assert!(
            thin <= bound && thin > 0.3 * bound,
            "thin slab {thin} cd/m² against {bound}"
        );
        assert!(bound < 3.56e5);
    }

    /// A stratus slab of optical depth 40 (σ = 0.05, the stratus peak, over 800 m), sun off,
    /// ambient source term 1 (`E_d = 2π`): looking down at it the radiance is the ambient source
    /// at the top, `w_a(1) = 1`; looking up at it from below it is the one at the base,
    /// `w_a(0) = 0.2`. Within 3 % of the source term; the note asks for τ ≥ 20, where the
    /// gradient of the ambient weight across the first mean free paths leaves 4 %.
    #[test]
    fn a_thick_slab_in_a_white_furnace_returns_the_ambient_source() {
        let slab = Slab {
            sigma: 0.05,
            base_m: 1500.0,
            top_m: 2300.0,
        };
        let furnace = |camera_y: f64, dir: DVec3| {
            let mut scene = Scene::new(DVec3::new(0.0, camera_y, 0.0), dir, DVec3::Y);
            scene.sun = 0.0;
            scene.ambient = 1.0;
            march(&scene, &slab)
        };
        let from_above = furnace(3000.0, -DVec3::Y);
        let from_below = furnace(1000.0, DVec3::Y);
        assert!(
            (from_above.radiance - 1.0).abs() <= 0.03,
            "from above {}",
            from_above.radiance
        );
        assert!(
            (from_below.radiance - 0.2).abs() <= 0.03,
            "from below {}",
            from_below.radiance
        );
        assert!(from_above.transmittance < 0.01 && from_below.transmittance < 0.01);
    }

    // -----------------------------------------------------------------------------------------
    // Validation 7: columns
    // -----------------------------------------------------------------------------------------

    fn column(sigma: f64, depth: f64, steps: u32) -> Marched {
        let slab = Slab {
            sigma,
            base_m: 1500.0,
            top_m: 1500.0 + depth,
        };
        let mut scene = Scene::new(DVec3::new(0.0, 1500.0, 0.0), DVec3::Y, DVec3::Y);
        scene.steps = steps;
        march(&scene, &slab)
    }

    #[test]
    fn a_homogeneous_column_transmits_exp_minus_sigma_d() {
        for steps in [32, 48, 64, 96] {
            for (sigma, depth) in [(0.002, 1000.0), (0.0004, 1000.0)] {
                let t = column(sigma, depth, steps).transmittance;
                let exact = (-sigma * depth).exp();
                assert!(
                    (t / exact - 1.0).abs() <= 0.01,
                    "σ {sigma}, d {depth}, Q {steps}: {t} against {exact}"
                );
            }
        }
    }

    /// Validation 3 of `broken-cloud-light.md`: a column the march gives up on hides what is
    /// behind it (the sun disc is 10⁵ times the sky), and its radiance is what it integrated,
    /// renormalised by at most `1/0.99`.
    #[test]
    fn an_opaque_column_hides_what_is_behind_it() {
        for steps in [32, 64, 96] {
            let m = column(0.05, 1000.0, steps);
            assert!(m.radiance.is_finite());
            assert_eq!(m.transmittance, 0.0);
            assert!(m.radiance > 0.0);
        }
        // A thin column the march finishes keeps its transmittance.
        let thin = column(0.0004, 1000.0, 64);
        assert!(
            (thin.transmittance - (-0.4f64).exp()).abs() < 0.02,
            "T {}",
            thin.transmittance
        );
    }

    /// `cumulus-shape.md` Validation 4: carving the cumulus out of the coverage keeps the share
    /// of the ground in their shade (vertical columns with `T < 0.5`, the shadow map's own
    /// sampling without the detail noise) at the weather map's fraction, so `cloud_cover` still
    /// means what it says. Measured over 2048 columns of the 32 km field: 0.048 at cover 0.05,
    /// 0.377 at 0.4, 0.818 at 0.8, 0.966 at 0.95.
    #[test]
    fn the_cumulus_shadow_covers_what_the_weather_says() {
        for cover in [0.05, 0.4, 0.8] {
            let p = MirrorParams::new(1500.0, 3300.0, 0.5, cover, [0.0, 0.0]);
            let n = 2048;
            let dh = (p.top_m - p.base_m) / 32.0;
            let (mut map, mut dark) = (0, 0);
            for i in 0..n {
                let x = (0.5 + f64::from(i) * 0.754_877_666_246_692_7).fract() * WEATHER_EXTENT_M;
                let z = (0.5 + f64::from(i) * 0.569_840_290_998_053_2).fract() * WEATHER_EXTENT_M;
                map += usize::from(p.weather_at([x, z]).0 >= 0.5);
                let tau: f64 = (0..32)
                    .map(|k| {
                        p.medium(p.base_m + (f64::from(k) + 0.5) * dh, [x, z], 0.0)
                            .0
                            * dh
                    })
                    .sum();
                dark += usize::from(tau > std::f64::consts::LN_2);
            }
            let (map, dark) = (map as f64 / f64::from(n), dark as f64 / f64::from(n));
            println!("cover {cover}: map {map:.3}, shade {dark:.3}");
            assert!(
                (dark - map).abs() <= 0.03,
                "cover {cover}: map {map}, shade {dark}"
            );
        }
    }

    /// Vertical optical depth of the default stratus layer (1500–2300 m, kind 0, cover 1) averaged
    /// over 4096 columns of the weather map, 64 samples per column with the detail noise.
    /// Measured: τ_col = 6.04 (T_dir = 0.0024; 4.3–7.2 over the columns), above the prior
    /// estimate 2.4–3.2: at cover 1 the shape term alone sets ρ, whose mean is 0.15 (the
    /// gradient's mean 0.20 times a shape term of about 0.75), not the assumed 0.06–0.08.
    #[test]
    fn the_stratus_column_has_the_optical_depth_the_note_expects() {
        let p = MirrorParams::new(1500.0, 2300.0, 0.0, 1.0, [0.0, 0.0]);
        let n = 4096;
        let mean = (0..n)
            .map(|i| {
                // R2 low-discrepancy points over the 32 km map.
                let x = (0.5 + f64::from(i) * 0.754_877_666_246_692_7).fract() * WEATHER_EXTENT_M;
                let z = (0.5 + f64::from(i) * 0.569_840_290_998_053_2).fract() * WEATHER_EXTENT_M;
                column_optical_depth(&p, [x, z], 64)
            })
            .sum::<f64>()
            / f64::from(n);
        println!("stratus tau_col {mean}, T_dir {}", (-mean).exp());
        assert!((2.0..=8.0).contains(&mean), "stratus τ_col {mean}");
    }

    // -----------------------------------------------------------------------------------------
    // Validation 2 (CPU part): the coverage field
    // -----------------------------------------------------------------------------------------

    #[test]
    fn the_exact_quantile_gives_every_cover_as_the_area_fraction() {
        // The area fraction is measured on a 128² grid unrelated to the 512² texels the
        // threshold was taken from.
        for drift in [
            [0.0, 0.0],
            [7_000.0, 0.0],
            [0.0, 31_000.0],
            [31_000.0, 7_000.0],
        ] {
            let sorted = sorted_field(drift);
            for cover in [0.05, 0.4, 0.6, 0.95] {
                let p = MirrorParams::from_sorted(1500.0, 3300.0, 0.5, cover, drift, &sorted);
                assert!(p.theta < 256.0, "threshold {}", p.theta);
                let n = 128;
                let covered = (0..n * n)
                    .filter(|k| {
                        let xz = [
                            (f64::from(k % n) + 0.37) * 250.0,
                            (f64::from(k / n) + 0.61) * 250.0,
                        ];
                        let (coverage, _) = p.weather_at(xz);
                        coverage >= 0.5
                    })
                    .count();
                let fraction = covered as f64 / f64::from(n * n);
                assert!(
                    (fraction - cover).abs() <= 0.005,
                    "drift {drift:?}, cover {cover}: area fraction {fraction}"
                );
            }
        }
    }

    #[test]
    fn the_coverage_threshold_rises_as_the_cover_falls() {
        let drift = [0.0, 0.0];
        let sorted = sorted_field(drift);
        let thresholds = [0.05, 0.4, 0.6, 0.95].map(|c| threshold_of(&sorted, c));
        assert!(thresholds.windows(2).all(|w| w[0] > w[1]), "{thresholds:?}");
        assert!((coverage_threshold(0.5, drift) - threshold_of(&sorted, 0.5)).abs() < 1e-9);
        assert_eq!(threshold_of(&sorted, 0.0), 256.0);
        assert_eq!(threshold_of(&sorted, 1.0), f64::from(sorted[0]));
    }

    #[test]
    fn the_coverage_field_repeats_every_thirty_two_kilometres() {
        for k in 0..200 {
            let x = (f64::from(k) * 977.3) % 32_000.0;
            let z = (f64::from(k) * 1_311.7) % 32_000.0;
            let drift = [(f64::from(k) * 411.0) % 32_000.0, 0.0];
            let at = |x: f64, z: f64| {
                cloud_coverage_field([x as f32, z as f32], [drift[0] as f32, drift[1] as f32])
            };
            let here = at(x, z);
            assert!(
                (here - at(x + 32_000.0, z)).abs() < 1e-4,
                "along x at ({x}, {z})"
            );
            assert!(
                (here - at(x, z + 32_000.0)).abs() < 1e-4,
                "along z at ({x}, {z})"
            );
            assert!((here - at(x - 32_000.0, z - 32_000.0)).abs() < 1e-4);
        }
    }

    // -----------------------------------------------------------------------------------------
    // Validation 8 (CPU part): the atmosphere
    // -----------------------------------------------------------------------------------------

    fn assert_close(label: &str, got: [f64; 3], want: [f64; 3], tolerance: f64) {
        for (g, w) in got.into_iter().zip(want) {
            assert!(
                (g / w - 1.0).abs() <= tolerance,
                "{label}: {got:?} against {want:?} (tolerance {tolerance})"
            );
        }
    }

    #[test]
    fn the_zenith_transmittance_of_the_layer_is_the_value_of_the_note() {
        assert_close(
            "1.5 km",
            optical_transmittance(1500.0, 1.0),
            [0.9515, 0.8872, 0.8007],
            2e-3,
        );
        assert_close(
            "2.0 km",
            optical_transmittance(2000.0, 1.0),
            [0.9542, 0.8925, 0.8118],
            2e-3,
        );
        assert_close(
            "3.3 km",
            optical_transmittance(3300.0, 1.0),
            [0.9600, 0.9045, 0.8379],
            2e-3,
        );
    }

    #[test]
    fn the_oblique_transmittance_follows_the_flat_slab_oracle_above_twelve_degrees() {
        // T(μ) ≈ T_zenith^(1/μ) within 5 % for μ ≥ 0.2 (the note), at 1.5 km.
        let zenith = optical_transmittance(1500.0, 1.0);
        for mu in [0.5, 0.3, 0.2] {
            let spherical = optical_transmittance(1500.0, mu);
            let flat = zenith.map(|t| t.powf(1.0 / mu));
            assert_close(&format!("μ = {mu}"), spherical, flat, 0.05);
        }
        // Lower, the curvature shortens the path: the flat oracle is an upper bound on the
        // optical depth, so the sphere transmits more.
        let spherical = optical_transmittance(1500.0, 0.1736);
        let flat = zenith.map(|t| t.powf(1.0 / 0.1736));
        for (s, f) in spherical.into_iter().zip(flat) {
            assert!(s >= f && s < 1.0, "spherical {s} against flat {f}");
        }
    }

    #[test]
    fn the_layer_keeps_the_sun_until_the_horizon_dip() {
        let mu = |elevation_deg: f64| elevation_deg.to_radians().sin();
        // Dip arccos(R / (R + h)): 1.244° at 1.5 km, 3.13° at 9.5 km.
        assert!(planet_visibility(1500.0, mu(-1.0)) > 0.0);
        assert_eq!(planet_visibility(1500.0, mu(-1.5)), 0.0);
        assert!(planet_visibility(1500.0, mu(0.5)) > 0.99);
        assert!(planet_visibility(9500.0, mu(-3.0)) > 0.0);
        // The note puts the cut at −3.3° for 9.5 km; the ±0.0044 rad blend reaches 0 at −3.4°
        // (−3.3° still passes 7 % of the light).
        assert_eq!(planet_visibility(9500.0, mu(-3.5)), 0.0);
        // The blend is centred on the dip.
        let dip = (PLANET_RADIUS_M / (PLANET_RADIUS_M + 1500.0))
            .acos()
            .to_degrees();
        assert!((dip - 1.244).abs() < 1e-3);
        assert!((planet_visibility(1500.0, mu(-dip)) - 0.5).abs() < 1e-3);
    }

    // -----------------------------------------------------------------------------------------
    // Validation 14: quality
    // -----------------------------------------------------------------------------------------

    #[test]
    fn the_cloud_quality_table_is_the_one_of_the_note() {
        let table = [
            (ElementsQuality::Low, 4, 32, 16),
            (ElementsQuality::Medium, 4, 48, 16),
            (ElementsQuality::High, 2, 64, 4),
            (ElementsQuality::Ultra, 2, 96, 4),
        ];
        for (quality, block, steps, window) in table {
            let q = CloudQuality::of(quality);
            assert_eq!(
                (q.block, q.steps, q.light_samples, q.window),
                (block, steps, 6, window),
                "{quality:?}"
            );
        }
    }

    // -----------------------------------------------------------------------------------------
    // Shape: gradient, dome and anvil
    // -----------------------------------------------------------------------------------------

    #[test]
    fn the_height_gradient_has_the_mean_of_the_note() {
        let mean = |kind: f64| {
            (0..10_000)
                .map(|k| cloud_gradient((f64::from(k) + 0.5) / 10_000.0, kind, 1.0))
                .sum::<f64>()
                / 10_000.0
        };
        assert!((mean(0.0) - 0.200).abs() < 5e-4, "stratus {}", mean(0.0));
        assert!((mean(0.5) - 0.849).abs() < 5e-4, "cumulus {}", mean(0.5));
        // The cumulonimbus is dense from 10 % to 90 % of its height.
        assert!((cloud_gradient(0.5, 1.0, 1.0) - 1.0).abs() < 1e-12);
        assert_eq!(cloud_gradient(1.0, 1.0, 1.0), 0.0);
    }

    #[test]
    fn deep_inside_a_blob_the_gradient_is_the_unscaled_one() {
        for kind in [0.0, 0.25, 0.5, 0.75, 1.0] {
            for k in 0..=100 {
                let f = f64::from(k) / 100.0;
                assert_eq!(
                    cloud_gradient(f, kind, 1.0),
                    height_gradient(f, gradient_quad(kind)),
                    "kind {kind}, f {f}"
                );
            }
        }
    }

    #[test]
    fn at_the_edge_of_a_blob_a_cumulus_is_short_and_a_stratus_unchanged() {
        // Dome 0: the cumulus reaches only 0.3 of its height, so nothing is above it.
        for k in 31..=100 {
            let f = f64::from(k) / 100.0;
            assert_eq!(cloud_gradient(f, 0.5, 0.0), 0.0, "cumulus at f = {f}");
        }
        assert!(cloud_gradient(0.2, 0.5, 0.0) > 0.0);
        for k in 0..=100 {
            let f = f64::from(k) / 100.0;
            assert_eq!(
                cloud_gradient(f, 0.0, 0.0),
                cloud_gradient(f, 0.0, 1.0),
                "stratus at f = {f}"
            );
        }
        // The height grows with the dome factor.
        let tall = |dome: f64| {
            (1..=100)
                .filter(|k| cloud_gradient(f64::from(*k) / 100.0, 0.5, dome) > 0.0)
                .count()
        };
        assert!(tall(0.0) < tall(0.25) && tall(0.25) < tall(1.0));
    }

    #[test]
    fn the_anvil_top_is_wider_than_the_body() {
        // r′(f = 0.8) ≥ r′(f = 0.6) for r < 1 and a cumulonimbus; the body is untouched.
        for k in 1..100 {
            let r = f64::from(k) / 100.0;
            let top = eroded_coverage(r, 0.8, 1.0);
            let body = eroded_coverage(r, 0.6, 1.0);
            assert!(top > body, "r {r}: top {top}, body {body}");
            assert!((body - r).abs() < 1e-12);
            // A cumulus (β = 0) has no anvil.
            assert!((eroded_coverage(r, 0.8, 0.5) - r).abs() < 1e-12);
        }
        assert!((coverage_exponent(0.8, 1.0) - 0.5).abs() < 1e-12);
    }

    // -----------------------------------------------------------------------------------------
    // The density mirror
    // -----------------------------------------------------------------------------------------

    #[test]
    fn the_density_is_zero_outside_the_layer_and_bounded_inside() {
        let p = MirrorParams::new(1500.0, 3300.0, 0.5, 0.6, [0.0, 0.0]);
        assert_eq!(density(1499.0, [5000.0, 5000.0], &p), (0.0, 0.0));
        assert_eq!(density(3301.0, [5000.0, 5000.0], &p), (0.0, 0.0));
        let mut dense = 0;
        for i in 0..400 {
            let xz = [f64::from(i) * 80.0, f64::from(i) * 31.0];
            let (sigma, g) = density(2300.0, xz, &p);
            assert!(
                (0.0..=sigma_peak(0.5) + 1e-12).contains(&sigma),
                "σ {sigma}"
            );
            assert_eq!(g, 0.0);
            // Where the march's coarse test sees nothing, the medium has no extinction.
            let coarse = p.coarse(2300.0, xz);
            assert!(coarse > 0.0 || p.medium(2300.0, xz, 1.0).0 == 0.0);
            if sigma > 0.0 {
                dense += 1;
            }
        }
        assert!(dense > 0, "no cloud found at mid height");
    }

    /// The coarse march samples the base noise only; it must never step over a cloud the fine
    /// march would draw, for any type, height or detail weight.
    #[test]
    fn the_coarse_test_never_misses_a_cloud() {
        for kind in [0.0, 0.35, 0.5, 1.0] {
            let p = MirrorParams::new(1500.0, 3300.0, kind, 0.6, [0.0, 0.0]);
            for i in 0..600 {
                let xz = [f64::from(i) * 97.0, f64::from(i) * 53.0];
                let h = 1510.0 + f64::from(i % 29) * 62.0;
                for detail in [0.0, 0.5, 1.0] {
                    let fine = p.medium(h, xz, detail).0;
                    assert!(
                        fine == 0.0 || p.coarse(h, xz) > 0.0,
                        "type {kind}: σ {fine} at {xz:?}, {h} with a zero coarse test"
                    );
                }
            }
        }
    }

    #[test]
    fn detail_noise_only_removes_stratus_density() {
        let p = MirrorParams::new(1500.0, 2300.0, 0.0, 0.95, [0.0, 0.0]);
        for i in 0..300 {
            let xz = [f64::from(i) * 97.0, f64::from(i) * 53.0];
            let h = 1510.0 + f64::from(i % 17) * 45.0;
            let with = p.medium(h, xz, 1.0).0;
            let without = p.medium(h, xz, 0.0).0;
            assert!(
                with <= without + 1e-12,
                "detail added density at {xz:?}, {h}"
            );
        }
    }

    /// `cumulus-shape.md` Validation 5: the envelope is 0 above the blob's height and 1 below
    /// its rounding, C¹ (no jump in slope: the terraces were slope jumps), the base is a plane,
    /// and the erosion is a density in [0, 1] that grows with the noise and is 0 below `1 − d`.
    #[test]
    fn the_cumulus_envelope_is_smooth_and_the_erosion_bounded() {
        for dome in [0.0, 0.3, 1.0] {
            let s = mix(CUMULUS_EDGE_HEIGHT, 1.0, f64::sqrt(dome));
            assert_eq!(cumulus_envelope(s, dome), 0.0);
            assert_eq!(cumulus_envelope(0.99 * s * CUMULUS_TOP_ROUND, dome), 1.0);
            // A kink (the old linear gradient's corners) jumps the slope by a whole 1/width
            // however fine the step; a C¹ curve by its curvature times the step.
            let h = 1e-5;
            let slope = |f: f64| {
                (cumulus_envelope(f + h, dome) - cumulus_envelope(f - h, dome)) / (2.0 * h)
            };
            let mut f = 0.0005;
            while f < 0.999 {
                let jump = (slope(f + 0.0005) - slope(f)).abs();
                assert!(jump < 0.1, "slope jumps by {jump} at f {f}, dome {dome}");
                f += 0.0005;
            }
        }
        assert_eq!(cumulus_base(0.0), 0.0);
        assert_eq!(cumulus_base(CUMULUS_BASE_RISE), 1.0);
        for d in [0.1, 0.5, 0.75] {
            assert_eq!(value_erosion(1.0 - d - 1e-6, d), 0.0);
            assert!((value_erosion(1.0, d) - 1.0).abs() < 1e-12);
            let mut last = 0.0;
            for i in 0..=100 {
                let v = value_erosion(f64::from(i) / 100.0, d);
                assert!((last..=1.0).contains(&v), "not monotone at {i}");
                last = v;
            }
        }
        assert_eq!(value_erosion(1.0, 0.0), 0.0);
    }

    #[test]
    fn rain_darkens_only_where_there_is_cloud() {
        let mut p = MirrorParams::new(1500.0, 3300.0, 0.0, 0.5, [0.0, 0.0]);
        p.precipitation = 0.16;
        for i in 0..300 {
            let xz = [f64::from(i) * 101.0, f64::from(i) * 67.0];
            let (coverage, _) = p.weather_at(xz);
            let (_, g) = p.medium(1800.0, xz, 0.0);
            if coverage <= 0.0 {
                assert_eq!(g, 0.0);
            } else {
                assert!((g - 0.16 * coverage).abs() < 1e-12 && g <= 0.16);
            }
        }
    }
}
