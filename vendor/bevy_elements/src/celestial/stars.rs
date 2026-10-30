//! The star field: the 8404 Yale Bright Star Catalogue stars with V ≤ 6.5, drawn as one additive
//! mesh of camera-facing quads. See `docs/research/celestial.md` §Stars and `PROVENANCE.md`.
//!
//! # How a star is drawn
//!
//! One `Mesh` is built at startup from `stars.bin`: four vertices per star carrying the J2000 unit
//! direction, the quad corner `(±1, ±1)`, the linear-sRGB colour from B−V (luminance 1) and the
//! outside-the-atmosphere illuminance E★. Each frame the CPU writes the 3×3 J2000 → world matrix
//! ([`CelestialFrame::j2000_to_world`]) and the moon's position into the material
//! uniform; `stars.wesl` does the rest:
//!
//! - **Placement.** The star's world direction is rotated into view space and projected as a point
//!   at infinity: the vertex shader ignores the mesh transform, uses only the rotation part of the
//!   view matrix, and writes clip `z = 0`. Under Bevy's infinite reverse-Z projection that is depth
//!   0, i.e. the far plane, so a star is hidden by anything drawn in the opaque pass, can never
//!   show through terrain, and does not depend on the camera's `far` (which Bevy uses only to cull,
//!   and the stars are never culled). Nothing needs to follow the camera.
//! - **Size.** The quad is expanded in clip space to a half-width of
//!   [`STAR_SPRITE_RADIUS_PX`] pixels. The pixel angle `p = 2·tan(fov_y/2)/H` is read from the
//!   view uniform (`clip_from_view[1][1]` and the viewport height), so it is right for any
//!   viewport, resolution or screenshot target without a uniform to keep in sync.
//! - **Radiance.** The sprite's solid angle is Ω = (1.5·p)². The soft disc profile
//!   `g(ρ) = (1 − ρ²)²` over `ρ = |corner| ≤ 1` is normalised to unit energy over Ω
//!   (∫ g d²c = π/3, so the peak is `3/π` times the mean), giving
//!   `L = E★/Ω · (3/π) · g(ρ) · colour · T_atm · exposure` cd/m².
//! - **Extinction.** `T_atm = exp(−τ·X)` with X the Kasten–Young airmass at the star's altitude and
//!   τ = [`ZENITH_OPTICAL_DEPTH`] per channel.
//! - **Culling.** Stars below the horizon and stars inside the moon's disc are moved outside the
//!   clip volume in the vertex shader.
//! - **Clouds.** The cloud composite attenuates the stars by the march transmittance, so no cloud
//!   factor is applied here. With `CloudPlugin` disabled the stars are not dimmed by cloud cover.
//!
//! # Blending and depth
//!
//! `AlphaMode::Add` in Bevy 0.20 is premultiplied blending (`src + dst·(1 − src.a)`); the shader
//! writes alpha 0, so the star is purely added. The pass is `Transparent3d`, after the atmosphere's
//! sky pass, depth-tested without depth writes.

use bevy::asset::{AssetPath, RenderAssetUsages, embedded_asset};
use bevy::camera::visibility::NoFrustumCulling;
use bevy::light::NotShadowCaster;
use bevy::mesh::{
    Indices, Mesh, MeshVertexAttribute, MeshVertexBufferLayoutRef, PrimitiveTopology, VertexFormat,
};
use bevy::pbr::{Material, MaterialPipeline, MaterialPipelineKey, MaterialPlugin};
use bevy::prelude::*;
use bevy::render::render_resource::{
    AsBindGroup, RenderPipelineDescriptor, ShaderType, SpecializedMeshPipelineError,
};
use bevy::shader::ShaderRef;

use super::{CelestialFrame, CelestialSettings, SunState};
use crate::gpu::readback::decode;
use crate::plugin::{ElementsSystems, ElementsView};

/// The catalogue: little-endian `[f32; 4]` records `(ra_rad, dec_rad, vmag, b_minus_v)`, J2000.
const CATALOG: &[u8] = include_bytes!("stars.bin");

/// Half-width of a star's sprite, pixels.
pub(super) const STAR_SPRITE_RADIUS_PX: f32 = 1.5;
/// Factor that makes the profile `(1 − ρ²)²` integrate to 1 over the unit disc: 1 / (π/3).
pub(super) const STAR_PROFILE_NORMALISER: f32 = 3.0 / std::f32::consts::PI;
/// Zero point of the star illuminance, magnitudes: `E = 10^(−0.4 (V + 13.99))` lx
/// (Krisciunas & Schaefer 1991, Eq. 8: `16.57 − 2.5·log10(10.76391 lx/fc)`).
const ILLUMINANCE_ZERO_POINT: f64 = 13.99;
/// Colour index of the stars the catalogue gives none for, magnitudes. Kept in sync with
/// `tools/src/bin/gen_stars.rs`, which writes it into `stars.bin`.
#[cfg(test)]
const DEFAULT_B_MINUS_V: f32 = 0.48;

/// Zenith optical depth (extinction: scattering plus absorption) of Bevy's Earth
/// `ScatteringMedium`, per linear-sRGB channel, for the whole 100 km atmosphere of
/// `Atmosphere::earth`.
///
/// Derivation. `ScatteringMedium::earth` (`bevy_light-0.20.0/src/atmosphere.rs:197`) has three
/// terms, each multiplying its coefficient by a falloff `f(p)` with `p = 1` at the ground and
/// `p = 0` at the top of the atmosphere (`outer_radius − inner_radius = 100 000 m` for
/// `Atmosphere::earth`). With `a = 1 − p` the normalised altitude, the vertical integral is
/// `τ = (β_absorb + β_scatter) · H · ∫₀¹ f da`:
///
/// - Rayleigh, `β = (5.802, 13.558, 33.100)·10⁻⁶ m⁻¹`, exponential falloff with scale `s = 8/60`:
///   `f = (e^(−a/s) − e^(−1/s)) / (1 − e^(−1/s))`, so `∫ f da = s − e^(−1/s)/(1 − e^(−1/s)) =
///   0.132780`. Contribution `(0.07704, 0.18002, 0.43950)`.
/// - Mie, absorption `3.996·10⁻⁶` + scattering `0.444·10⁻⁶` = `4.440·10⁻⁶ m⁻¹` in every channel,
///   `s = 1.2/60`, `∫ f da = 0.02`. Contribution `0.00888` per channel.
/// - Ozone, absorption `(0.650, 1.881, 0.085)·10⁻⁶ m⁻¹`, tent falloff `center = 0.75`,
///   `width = 0.3`: `∫ f dp = width/2 = 0.15`. Contribution `(0.00975, 0.028215, 0.001275)`.
///
/// Sum: `(0.095669, 0.217118, 0.449657)`. The test `zenith_optical_depth_matches_the_bevy_medium`
/// integrates Bevy's own `ScatteringMedium::earth` terms numerically and compares.
///
/// Note: Bevy's doc says "60 km atmosphere", but `Atmosphere::earth` spans 100 km, so the
/// effective Rayleigh scale height is `8/60 · 100 km = 13.3 km` rather than 8 km. The stars use
/// the same medium as the rendered sky, so they dim exactly as much as the sky says the air is
/// thick.
pub(super) const ZENITH_OPTICAL_DEPTH: Vec3 = Vec3::new(0.095_669, 0.217_118, 0.449_657);

/// CPU mirror of the shader's airmass (Kasten & Young 1989):
/// `X = 1 / (sin h + 0.50572 (h° + 6.07995)^−1.6364)`; 1 at the zenith, ≈ 38 at the horizon.
#[cfg(test)]
fn airmass(sin_altitude: f32) -> f32 {
    let s = sin_altitude.clamp(0.0, 1.0);
    let altitude_deg = s.asin().to_degrees();
    1.0 / (s + 0.505_72 * (altitude_deg + 6.079_95).powf(-1.6364))
}

/// CPU mirror of the shader's transmittance: the fraction of light that survives the atmosphere at
/// `sin_altitude`, per channel.
#[cfg(test)]
fn atmospheric_transmittance(sin_altitude: f32) -> Vec3 {
    let x = airmass(sin_altitude);
    Vec3::new(
        (-ZENITH_OPTICAL_DEPTH.x * x).exp(),
        (-ZENITH_OPTICAL_DEPTH.y * x).exp(),
        (-ZENITH_OPTICAL_DEPTH.z * x).exp(),
    )
}

/// Illuminance of a star of visual magnitude `vmag` outside the atmosphere, lx.
pub(super) fn star_illuminance_lux(vmag: f32) -> f32 {
    10f64.powf(-0.4 * (f64::from(vmag) + ILLUMINANCE_ZERO_POINT)) as f32
}

/// Effective temperature of a star of colour index `b_minus_v`, K (Ballesteros 2012, Eq. 14),
/// clamped to the validity range of the Planckian-locus fit below.
///
/// The index is first clamped to `[−0.4, 4.0]`: Eq. 14 has poles at B−V = −0.674 and −1.848, and the
/// catalogue's stars lie in `[−0.28, 3.86]`.
pub(super) fn effective_temperature(b_minus_v: f32) -> f32 {
    let bv = f64::from(b_minus_v.clamp(-0.4, 4.0));
    let t = 4600.0 * (1.0 / (0.92 * bv + 1.7) + 1.0 / (0.92 * bv + 0.62));
    t.clamp(1667.0, 25000.0) as f32
}

/// CIE 1931 chromaticity `(x, y)` of a black body at `kelvin` (Kang et al. 2002 cubic splines,
/// valid 1667–25000 K).
pub(super) fn planckian_chromaticity(kelvin: f32) -> (f64, f64) {
    let t = f64::from(kelvin).clamp(1667.0, 25000.0);
    let (t1, t2, t3) = (1e3 / t, 1e6 / (t * t), 1e9 / (t * t * t));
    let x = if t < 4000.0 {
        -0.266_123_9 * t3 - 0.234_358_9 * t2 + 0.877_695_6 * t1 + 0.179_910
    } else {
        -3.025_846_9 * t3 + 2.107_037_9 * t2 + 0.222_634_7 * t1 + 0.240_390
    };
    let y = if t < 2222.0 {
        -1.106_381_4 * x * x * x - 1.348_110_20 * x * x + 2.185_558_32 * x - 0.202_196_83
    } else if t < 4000.0 {
        -0.954_947_6 * x * x * x - 1.374_185_93 * x * x + 2.091_370_15 * x - 0.167_488_67
    } else {
        3.081_758_0 * x * x * x - 5.873_386_70 * x * x + 3.751_129_97 * x - 0.370_014_83
    };
    (x, y)
}

/// Linear sRGB of a chromaticity at luminance Y = 1 (D65 matrix), before any clamping.
pub(super) fn chromaticity_to_linear_srgb(x: f64, y: f64) -> [f64; 3] {
    let (cx, cy, cz) = (x / y, 1.0, (1.0 - x - y) / y);
    [
        3.240_454_2 * cx - 1.537_138_5 * cy - 0.498_531_4 * cz,
        -0.969_266_0 * cx + 1.876_010_8 * cy + 0.041_556_0 * cz,
        0.055_643_4 * cx - 0.204_025_9 * cy + 1.057_225_2 * cz,
    ]
}

/// Linear-sRGB colour of a star of colour index `b_minus_v`, normalised to luminance 1 so the
/// brightness comes from the illuminance alone.
pub(super) fn star_color(b_minus_v: f32) -> [f32; 3] {
    let (x, y) = planckian_chromaticity(effective_temperature(b_minus_v));
    let rgb = chromaticity_to_linear_srgb(x, y).map(|c| c.max(0.0));
    let luminance = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2];
    rgb.map(|c| (c / luminance) as f32)
}

/// J2000 equatorial unit vector `(cos δ cos α, cos δ sin α, sin δ)`.
fn j2000_direction(ra_rad: f32, dec_rad: f32) -> [f32; 3] {
    let (sin_dec, cos_dec) = dec_rad.sin_cos();
    let (sin_ra, cos_ra) = ra_rad.sin_cos();
    [cos_dec * cos_ra, cos_dec * sin_ra, sin_dec]
}

/// Per-vertex J2000 direction is `Mesh::ATTRIBUTE_POSITION` (shader location 0).
/// Quad corner `(±1, ±1)`, shader location 1.
pub(super) const ATTRIBUTE_STAR_CORNER: MeshVertexAttribute =
    MeshVertexAttribute::new("StarCorner", 0x57A5_0001, VertexFormat::Float32x2);
/// Linear-sRGB colour (luminance 1) in `.xyz` and illuminance E★ in lx in `.w`, shader location 2.
pub(super) const ATTRIBUTE_STAR_COLOR_FLUX: MeshVertexAttribute =
    MeshVertexAttribute::new("StarColorFlux", 0x57A5_0002, VertexFormat::Float32x4);

/// Corners of a star's quad, counter-clockwise.
const CORNERS: [[f32; 2]; 4] = [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]];

/// One quad per star.
fn star_mesh(catalog: &[[f32; 4]]) -> Mesh {
    let n = catalog.len();
    let mut directions = Vec::with_capacity(4 * n);
    let mut corners = Vec::with_capacity(4 * n);
    let mut color_flux = Vec::with_capacity(4 * n);
    let mut indices = Vec::with_capacity(6 * n);
    for (i, &[ra, dec, vmag, b_minus_v]) in catalog.iter().enumerate() {
        let direction = j2000_direction(ra, dec);
        let [r, g, b] = star_color(b_minus_v);
        let flux = star_illuminance_lux(vmag);
        for corner in CORNERS {
            directions.push(direction);
            corners.push(corner);
            color_flux.push([r, g, b, flux]);
        }
        let base = (4 * i) as u32;
        indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }
    // The data is only ever read by the GPU.
    Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::RENDER_WORLD,
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, directions)
    .with_inserted_attribute(ATTRIBUTE_STAR_CORNER, corners)
    .with_inserted_attribute(ATTRIBUTE_STAR_COLOR_FLUX, color_flux)
    .with_inserted_indices(Indices::U32(indices))
}

/// The star field's uniform, in `stars.wesl` order. Vectors are padded to `vec4`.
#[derive(Clone, Copy, Debug, PartialEq, ShaderType)]
pub(super) struct StarsParams {
    /// Columns of the J2000 → world matrix, `.w` unused.
    pub j2000_to_world: [Vec4; 3],
    /// `.xyz`: zenith optical depth per channel. `.w`: unused (1).
    pub extinction_cloud: Vec4,
    /// `.xyz`: unit world direction of the moon. `.w`: cosine of its angular radius.
    pub moon: Vec4,
    /// `.x`: sprite half-width, px. `.y`: profile normaliser. `.zw` unused.
    pub sprite: Vec4,
}

impl Default for StarsParams {
    fn default() -> Self {
        Self {
            j2000_to_world: [Vec4::X, Vec4::Y, Vec4::Z],
            extinction_cloud: ZENITH_OPTICAL_DEPTH.extend(1.0),
            moon: Vec4::new(0.0, -1.0, 0.0, 2.0),
            sprite: Vec4::new(STAR_SPRITE_RADIUS_PX, STAR_PROFILE_NORMALISER, 0.0, 0.0),
        }
    }
}

/// The additive, unlit star material.
#[derive(Asset, TypePath, AsBindGroup, Clone, Debug, Default)]
pub(super) struct StarsMaterial {
    /// Binding 0 of the material bind group (group 3).
    #[uniform(0)]
    pub params: StarsParams,
}

const SHADER: &str = "embedded://bevy_elements/celestial/stars.wesl";

impl Material for StarsMaterial {
    fn vertex_shader() -> ShaderRef {
        ShaderRef::Path(AssetPath::parse(SHADER))
    }

    fn fragment_shader() -> ShaderRef {
        ShaderRef::Path(AssetPath::parse(SHADER))
    }

    fn alpha_mode(&self) -> AlphaMode {
        AlphaMode::Add
    }

    fn enable_prepass() -> bool {
        false
    }

    fn enable_shadows() -> bool {
        false
    }

    fn specialize(
        _pipeline: &MaterialPipeline,
        descriptor: &mut RenderPipelineDescriptor,
        layout: &MeshVertexBufferLayoutRef,
        _key: MaterialPipelineKey<Self>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        descriptor.vertex.buffers = vec![layout.0.get_layout(&[
            Mesh::ATTRIBUTE_POSITION.at_shader_location(0),
            ATTRIBUTE_STAR_CORNER.at_shader_location(1),
            ATTRIBUTE_STAR_COLOR_FLUX.at_shader_location(2),
        ])?];
        // The quads face the camera by construction; a stray winding flip must not delete stars.
        descriptor.primitive.cull_mode = None;
        Ok(())
    }
}

/// Marks the star-field entity.
#[derive(Component, Clone, Copy, Debug, Default)]
pub(super) struct StarField;

pub(super) fn plugin(app: &mut App) {
    embedded_asset!(app, "stars.wesl");
    app.add_plugins(MaterialPlugin::<StarsMaterial>::default())
        .add_systems(Startup, spawn_stars)
        .add_systems(Update, update_stars.after(ElementsSystems::Celestial));
}

fn spawn_stars(
    mut commands: Commands,
    settings: Res<CelestialSettings>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StarsMaterial>>,
) {
    if !settings.sky {
        return;
    }
    let Some(catalog) = decode::<[f32; 4]>(CATALOG) else {
        error!("stars.bin is not a whole number of 16-byte records; the star field is not drawn");
        return;
    };
    commands.spawn((
        Name::new("Stars"),
        StarField,
        Mesh3d(meshes.add(star_mesh(&catalog))),
        MeshMaterial3d(materials.add(StarsMaterial::default())),
        Transform::default(),
        Visibility::Hidden,
        // The shader places every vertex itself, so the mesh bounds mean nothing.
        NoFrustumCulling,
        NotShadowCaster,
    ));
}

/// This frame's uniform from the sky state.
pub(super) fn stars_params(frame: &CelestialFrame, sun: &SunState) -> StarsParams {
    let m = frame.j2000_to_world;
    let moon_radius = 0.5 * frame.moon_angular_diameter_deg.to_radians();
    StarsParams {
        j2000_to_world: [
            m.x_axis.extend(0.0),
            m.y_axis.extend(0.0),
            m.z_axis.extend(0.0),
        ],
        extinction_cloud: ZENITH_OPTICAL_DEPTH.extend(1.0),
        moon: sun.moon_direction.extend(moon_radius.cos()),
        ..default()
    }
}

fn update_stars(
    frame: Res<CelestialFrame>,
    sun: Res<SunState>,
    view: Res<ElementsView>,
    mut materials: ResMut<Assets<StarsMaterial>>,
    mut fields: Query<(&MeshMaterial3d<StarsMaterial>, &mut Visibility), With<StarField>>,
) {
    let params = stars_params(&frame, &sun);
    let visibility = if view.entity.is_some() {
        Visibility::Visible
    } else {
        Visibility::Hidden
    };
    for (handle, mut shown) in &mut fields {
        if *shown != visibility {
            *shown = visibility;
        }
        if materials
            .get(&handle.0)
            .is_some_and(|material| material.params != params)
            && let Some(mut material) = materials.get_mut(&handle.0)
        {
            material.params = params;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::light::Atmosphere;
    use bevy::light::atmosphere::{Falloff, ScatteringMedium};

    fn records() -> Vec<[f32; 4]> {
        decode::<[f32; 4]>(CATALOG).expect("whole records")
    }

    /// Validation 20: 8404 records, 134 464 bytes, 244 of them without a measured B−V.
    #[test]
    fn the_catalogue_has_8404_stars() {
        assert_eq!(CATALOG.len(), 134_464);
        let stars = records();
        assert_eq!(stars.len(), 8404);
        assert!(stars.iter().all(|s| s[2] <= 6.5));
        assert_eq!(
            stars.iter().filter(|s| s[2] == -1.46).count(),
            1,
            "exactly one star is as bright as Sirius"
        );
        let brightest = stars.iter().map(|s| s[2]).fold(f32::INFINITY, f32::min);
        assert_eq!(brightest, -1.46);
        // 244 stars have no B−V; some measured ones are 0.48 too.
        let defaulted = stars.iter().filter(|s| s[3] == DEFAULT_B_MINUS_V).count();
        assert!(
            defaulted >= 244,
            "{defaulted} stars carry the default colour index"
        );
        assert!(stars.iter().all(|s| (-0.4..=4.0).contains(&s[3])));
    }

    /// Validation 21: Sirius, HR 2491, is `06h45m08.9s −16°42′58″`, V −1.46, B−V 0.00.
    #[test]
    fn sirius_matches_the_cds_line() {
        let sirius = records()
            .into_iter()
            .find(|s| s[2] == -1.46)
            .expect("Sirius");
        let ra = (6.0 + 45.0 / 60.0 + 8.9 / 3600.0) * 15.0_f64.to_radians();
        let dec = -(16.0 + 42.0 / 60.0 + 58.0 / 3600.0_f64).to_radians();
        assert!((f64::from(sirius[0]) - ra).abs() < 1e-6, "{}", sirius[0]);
        assert!((f64::from(sirius[1]) - dec).abs() < 1e-6, "{}", sirius[1]);
        assert!((sirius[2] - -1.46).abs() < 0.005);
        assert!(sirius[3].abs() < 0.005);
        assert!((sirius[0] - 1.767_793).abs() < 1e-6);
        assert!((sirius[1] - -0.291_751).abs() < 1e-6);
    }

    /// Validation 22.
    #[test]
    fn star_illuminance_follows_the_magnitude_scale() {
        for (v, expected) in [
            (0.0, 2.5351e-6),
            (-1.46, 9.7275e-6),
            (1.0, 1.0093e-6),
            (6.5, 6.368e-9),
        ] {
            let e = star_illuminance_lux(v);
            assert!(
                ((e - expected) / expected).abs() < 1e-4,
                "V {v}: {e} vs {expected}"
            );
        }
        let sun = star_illuminance_lux(-26.74);
        assert!(((sun - 1.26e5) / 1.26e5).abs() < 0.005, "{sun}");
    }

    /// Validation 23.
    #[test]
    fn ballesteros_temperatures() {
        for (bv, kelvin) in [
            (0.65, 5778.4),
            (0.0, 10125.2),
            (1.0, 4742.7),
            (1.5, 3793.5),
            (-0.33, 17832.7),
        ] {
            let t = effective_temperature(bv);
            assert!((t - kelvin).abs() < 0.1, "B-V {bv}: {t} vs {kelvin}");
        }
        // Outside the fit's domain the index is clamped, never fed through the formula's poles.
        assert_eq!(effective_temperature(-1.0), effective_temperature(-0.4));
        assert!((effective_temperature(-0.4) - 21707.4).abs() < 0.1);
        assert_eq!(effective_temperature(10.0), effective_temperature(4.0));
        assert!((effective_temperature(4.0) - 1925.0).abs() < 5.0);
        for bv in [-10.0, -0.28, 3.86, 10.0] {
            assert!((1667.0..=25000.0).contains(&effective_temperature(bv)));
        }
    }

    /// Validation 24.
    #[test]
    fn kang_locus_and_normalised_colours() {
        for (kelvin, x, y) in [
            (5778.0, 0.32634, 0.33573),
            (10125.0, 0.28002, 0.28750),
            (6504.0, 0.31343, 0.32360),
        ] {
            let (cx, cy) = planckian_chromaticity(kelvin);
            assert!(
                (cx - x).abs() < 1e-4 && (cy - y).abs() < 1e-4,
                "{kelvin}: {cx} {cy}"
            );
        }
        for (kelvin, rgb) in [
            (5778.0, [1.1109, 0.9757, 0.9142]),
            (10125.0, [0.8690, 0.9945, 1.4406]),
            (3794.0, [1.4711, 0.9123, 0.4810]),
        ] {
            let (x, y) = planckian_chromaticity(kelvin);
            let got = chromaticity_to_linear_srgb(x, y);
            for (g, want) in got.iter().zip(rgb) {
                assert!((g - want).abs() < 1e-3, "{kelvin} K: {got:?} vs {rgb:?}");
            }
        }
    }

    #[test]
    fn every_star_colour_has_unit_luminance() {
        for [_, _, _, bv] in records() {
            let [r, g, b] = star_color(bv);
            assert!(r >= 0.0 && g >= 0.0 && b >= 0.0);
            let luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b;
            assert!((luminance - 1.0).abs() < 1e-5, "B-V {bv}: {luminance}");
        }
        // Blue stars are blue, red stars red.
        let (blue, red) = (star_color(-0.3), star_color(1.6));
        assert!(blue[2] > blue[0] && red[0] > red[2]);
    }

    /// The sprite profile `(1 − ρ²)²` times [`STAR_PROFILE_NORMALISER`] integrates to 1 over the
    /// unit disc, so a sprite carries exactly E★ of illuminance.
    #[test]
    fn the_sprite_profile_has_unit_energy() {
        let n = 2000;
        let mut sum = 0.0f64;
        for j in 0..n {
            for i in 0..n {
                let x = (f64::from(i) + 0.5) / f64::from(n) * 2.0 - 1.0;
                let y = (f64::from(j) + 0.5) / f64::from(n) * 2.0 - 1.0;
                let r2 = x * x + y * y;
                if r2 < 1.0 {
                    sum += (1.0 - r2) * (1.0 - r2);
                }
            }
        }
        let cell = (2.0 / f64::from(n)).powi(2);
        let energy = sum * cell * f64::from(STAR_PROFILE_NORMALISER);
        assert!((energy - 1.0).abs() < 1e-3, "{energy}");
    }

    #[test]
    fn airmass_follows_kasten_young() {
        assert!((airmass(1.0) - 1.0).abs() < 1e-3);
        // Kasten & Young 1989 table values: X(30°) = 1.995, X(10°) = 5.60, X(0°) = 37.9.
        assert!((airmass(30f32.to_radians().sin()) - 1.995).abs() < 0.01);
        assert!((airmass(10f32.to_radians().sin()) - 5.60).abs() < 0.03);
        assert!((airmass(0.0) - 37.9).abs() < 0.3);
        let t = atmospheric_transmittance(1.0);
        assert!(t.x > t.y && t.y > t.z, "blue is extinguished most");
        assert!(atmospheric_transmittance(0.0).z < 1e-6);
    }

    /// Integrate Bevy's own Earth medium and compare with the constant.
    #[test]
    fn zenith_optical_depth_matches_the_bevy_medium() {
        let medium = ScatteringMedium::earth(256, 256);
        let height = {
            // `Atmosphere::earth` spans 6 360 km to 6 460 km.
            let atmosphere = Atmosphere::earth(Handle::default());
            f64::from(atmosphere.outer_radius - atmosphere.inner_radius)
        };
        let steps = 200_000;
        let mut tau = [0.0f64; 3];
        for term in &medium.terms {
            let falloff: &Falloff = &term.falloff;
            let mut integral = 0.0f64;
            for i in 0..steps {
                let p = (f64::from(i) + 0.5) / f64::from(steps);
                integral += f64::from(falloff.sample(p as f32));
            }
            integral /= f64::from(steps);
            let extinction = term.absorption + term.scattering;
            tau[0] += f64::from(extinction.x) * height * integral;
            tau[1] += f64::from(extinction.y) * height * integral;
            tau[2] += f64::from(extinction.z) * height * integral;
        }
        for (got, want) in tau.iter().zip(ZENITH_OPTICAL_DEPTH.to_array()) {
            assert!(
                (got - f64::from(want)).abs() < 5e-4,
                "{tau:?} vs {ZENITH_OPTICAL_DEPTH}"
            );
        }
    }

    #[test]
    fn the_mesh_has_one_quad_per_star() {
        let stars = records();
        let mesh = star_mesh(&stars);
        assert_eq!(mesh.count_vertices(), 4 * stars.len());
        assert_eq!(mesh.indices().map(Indices::len), Some(6 * stars.len()));
        let directions = mesh
            .attribute(Mesh::ATTRIBUTE_POSITION)
            .and_then(|a| a.as_float3())
            .expect("positions");
        assert!(directions.iter().all(|d| {
            let len = Vec3::from(*d).length();
            (len - 1.0).abs() < 1e-5
        }));
    }

    #[test]
    fn the_uniform_carries_the_frame() {
        let frame = CelestialFrame {
            j2000_to_world: Mat3::from_rotation_y(0.5),
            moon_angular_diameter_deg: 0.52,
            ..default()
        };
        let sun = SunState {
            moon_direction: Vec3::X,
            ..default()
        };
        let p = stars_params(&frame, &sun);
        assert_eq!(p.extinction_cloud.w, 1.0);
        assert_eq!(p.j2000_to_world[0].truncate(), frame.j2000_to_world.x_axis);
        assert!((p.moon.w - (0.26f32.to_radians()).cos()).abs() < 1e-7);
    }
}
