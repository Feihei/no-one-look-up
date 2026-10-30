//! The moon disc: one quad whose fragment shader draws a lit sphere from a 512×256 albedo map.
//! See `docs/research/celestial.md` §Moon disc and `PROVENANCE.md`.
//!
//! # How the disc is drawn
//!
//! Like the stars (see `stars.rs`) the disc is placed at infinity by its vertex shader: only the
//! rotation part of the view matrix is used, clip `z = 0`, no culling, the mesh transform ignored.
//! The quad spans the moon's angular diameter plus a 2-pixel margin for the anti-aliased limb. In
//! the fragment shader, with `(x, y)` the tangent-plane offset from the moon's centre along the
//! screen-right and screen-up axes and `r = (x, y)/tan ρ` (`ρ` the angular radius):
//!
//! - The sphere is orthographic at this distance: the normal in the frame (right, up, toward the
//!   observer) is `n = (r, √(1 − |r|²))`, `μ = n_z`.
//! - The moon-fixed frame is +X toward the observer, +Z = the north ecliptic pole projected
//!   perpendicular to the view direction (so the sub-observer point is latitude 0, longitude 0,
//!   libration and the 1.54° axis tilt are neglected) and +Y = view direction × +Z, which is
//!   screen-right and lunar east. Selenographic latitude is `asin(n_up)`, longitude
//!   `atan2(n_right, n_toward)`.
//! - The map (`moon_albedo.r8`, row 0 = north, column 0 = longitude −180°) is sampled as an
//!   `R8Unorm` filtered texture with a hand-computed mip level, so a disc a few pixels wide does
//!   not shimmer. `A = Ā · texel / 128` with `Ā` = [`MOON_MEAN_ALBEDO`] and the disc mean of the
//!   texel value normalised to 128 by `tools/src/bin/gen_moon.rs`.
//! - Lommel–Seeliger shading `L = A · E_sun/π · 2μ₀/(μ₀ + μ)` with `μ₀ = n · ŝ` (0 on the night
//!   side), plus earthshine `0.0002 · A · E_sun/π · (1 − k)` everywhere (negligible on the lit
//!   side), where `k` is the illuminated fraction.
//! - Times the atmospheric transmittance of a star at the moon's altitude, the camera exposure and
//!   the limb coverage.
//!
//! The cloud composite attenuates the disc by the march transmittance, so no cloud factor is
//! applied here; with `CloudPlugin` disabled the disc is not dimmed by cloud cover.
//!
//! The disc is hidden below [`MOON_HIDE_BELOW_DEG`] altitude and when no camera is followed.

use bevy::asset::{AssetPath, RenderAssetUsages, embedded_asset};
use bevy::camera::visibility::NoFrustumCulling;
use bevy::image::{ImageAddressMode, ImageFilterMode, ImageSampler, ImageSamplerDescriptor};
use bevy::light::NotShadowCaster;
use bevy::light::light_consts::lux;
use bevy::mesh::{Indices, Mesh, MeshVertexBufferLayoutRef, PrimitiveTopology};
use bevy::pbr::{Material, MaterialPipeline, MaterialPipelineKey, MaterialPlugin};
use bevy::prelude::*;
use bevy::render::render_resource::{
    AsBindGroup, Extent3d, RenderPipelineDescriptor, ShaderType, SpecializedMeshPipelineError,
    TextureDimension, TextureFormat,
};
use bevy::shader::ShaderRef;

use super::stars::ZENITH_OPTICAL_DEPTH;
use super::{CelestialFrame, CelestialSettings, SunState};
use crate::plugin::{ElementsSystems, ElementsView};

/// The albedo map, raw `u8`, row-major, 512×256, row 0 = north, column 0 = longitude −180°.
const ALBEDO: &[u8] = include_bytes!("moon_albedo.r8");
/// Map width, texels.
const ALBEDO_WIDTH: u32 = 512;
/// Map height, texels.
const ALBEDO_HEIGHT: u32 = 256;
const _: () = assert!(ALBEDO.len() == (ALBEDO_WIDTH * ALBEDO_HEIGHT) as usize);

/// Disc-mean luminance of the full moon, cd/m²: Krisciunas & Schaefer's full-moon illuminance
/// (0.31331 lx) over the disc's solid angle (`docs/research/celestial.md` Validation 19).
const FULL_MOON_DISC_LUMINANCE: f64 = 4843.0;
/// Disc-mean albedo `Ā` that makes the full moon's disc-mean luminance
/// [`FULL_MOON_DISC_LUMINANCE`]: `4843 · π / 130000 = 0.117036`. The research note rounds this to
/// 0.117, which would give 4841.5 cd/m², outside its own ±1 cd/m² target.
pub(super) const MOON_MEAN_ALBEDO: f32 =
    (FULL_MOON_DISC_LUMINANCE * std::f64::consts::PI / lux::RAW_SUNLIGHT as f64) as f32;
/// The texel value whose albedo is [`MOON_MEAN_ALBEDO`] (the disc mean of the map).
const MEAN_TEXEL: f32 = 128.0;
/// Earthshine relative to the sunlit radiance, times `1 − k`.
const EARTHSHINE: f32 = 0.0002;
/// Quad margin around the limb for anti-aliasing, pixels.
const QUAD_MARGIN_PX: f32 = 2.0;
/// Altitude below which the disc is hidden, degrees.
const MOON_HIDE_BELOW_DEG: f32 = -1.0;

/// The moon's uniform, in `moon.wesl` order. Vectors are padded to `vec4`.
#[derive(Clone, Copy, Debug, PartialEq, ShaderType)]
pub(super) struct MoonParams {
    /// `.xyz`: unit world direction to the moon. `.w`: angular radius, rad.
    pub moon: Vec4,
    /// `.xyz`: the moon-fixed +Z axis (north), unit and perpendicular to `moon`. `.w`: illuminated
    /// fraction `k`.
    pub north: Vec4,
    /// `.xyz`: unit world direction to the sun. `.w`: unused (1).
    pub sun_cloud: Vec4,
    /// `.xyz`: zenith optical depth per channel. `.w`: exposure for an eye adapted to the lit
    /// lunar surface (the disc is shown at the darker of this and the scene's exposure).
    pub extinction: Vec4,
    /// `.x`: albedo per unit of the `R8Unorm` texel (`Ā · 255 / 128`). `.y`: sun illuminance, lx.
    /// `.z`: earthshine factor. `.w`: quad margin, px.
    pub photometry: Vec4,
}

impl Default for MoonParams {
    fn default() -> Self {
        Self {
            moon: Vec4::new(0.0, -1.0, 0.0, 0.0045),
            north: Vec4::new(0.0, 0.0, -1.0, 0.0),
            sun_cloud: Vec4::new(0.0, 1.0, 0.0, 1.0),
            extinction: ZENITH_OPTICAL_DEPTH.extend(super::exposure_for_adaptation(
                FULL_MOON_DISC_LUMINANCE as f32,
            )),
            photometry: Vec4::new(
                MOON_MEAN_ALBEDO * 255.0 / MEAN_TEXEL,
                lux::RAW_SUNLIGHT,
                EARTHSHINE,
                QUAD_MARGIN_PX,
            ),
        }
    }
}

/// The moon disc's material: additive and unlit, with the albedo map.
#[derive(Asset, TypePath, AsBindGroup, Clone, Debug)]
pub(super) struct MoonMaterial {
    /// Binding 0 of the material bind group (group 3).
    #[uniform(0)]
    pub params: MoonParams,
    /// Bindings 1 (texture) and 2 (sampler).
    #[texture(1)]
    #[sampler(2)]
    pub albedo: Handle<Image>,
}

const SHADER: &str = "embedded://bevy_elements/celestial/moon.wesl";

impl Material for MoonMaterial {
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
        // The corner `(±1, ±1, 0)` is the quad's only vertex attribute.
        descriptor.vertex.buffers = vec![
            layout
                .0
                .get_layout(&[Mesh::ATTRIBUTE_POSITION.at_shader_location(0)])?,
        ];
        descriptor.primitive.cull_mode = None;
        Ok(())
    }
}

/// Marks the moon-disc entity.
#[derive(Component, Clone, Copy, Debug, Default)]
pub(super) struct MoonDisc;

pub(super) fn plugin(app: &mut App) {
    embedded_asset!(app, "moon.wesl");
    app.add_plugins(MaterialPlugin::<MoonMaterial>::default())
        .add_systems(Startup, spawn_moon)
        .add_systems(Update, update_moon.after(ElementsSystems::Celestial));
}

/// The box-filtered mip chain of a `width × height` byte map, level 0 first, as one buffer
/// (wgpu's layer-major order for a single layer). Each level halves each side, to a minimum of 1,
/// averaging the source texels it covers; levels continue to 1×1.
fn mip_chain(base: &[u8], width: u32, height: u32) -> (Vec<u8>, u32) {
    let mut data = base.to_vec();
    let (mut w, mut h) = (width as usize, height as usize);
    let mut level_start = 0;
    let mut levels = 1;
    while w > 1 || h > 1 {
        let (nw, nh) = ((w / 2).max(1), (h / 2).max(1));
        let level_len = w * h;
        for y in 0..nh {
            for x in 0..nw {
                let mut sum = 0u32;
                let mut count = 0u32;
                for sy in (y * 2)..(y * 2 + 2).min(h) {
                    for sx in (x * 2)..(x * 2 + 2).min(w) {
                        sum += u32::from(data[level_start + sy * w + sx]);
                        count += 1;
                    }
                }
                data.push(((sum + count / 2) / count) as u8);
            }
        }
        level_start += level_len;
        (w, h) = (nw, nh);
        levels += 1;
    }
    (data, levels)
}

/// The albedo map as a filtered, mipmapped `R8Unorm` image: longitude wraps, latitude clamps.
fn albedo_image() -> Image {
    let (data, levels) = mip_chain(ALBEDO, ALBEDO_WIDTH, ALBEDO_HEIGHT);
    let mut image = Image::new_uninit(
        Extent3d {
            width: ALBEDO_WIDTH,
            height: ALBEDO_HEIGHT,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        TextureFormat::R8Unorm,
        RenderAssetUsages::RENDER_WORLD,
    );
    image.data = Some(data);
    image.texture_descriptor.mip_level_count = levels;
    image.sampler = ImageSampler::Descriptor(ImageSamplerDescriptor {
        address_mode_u: ImageAddressMode::Repeat,
        address_mode_v: ImageAddressMode::ClampToEdge,
        mag_filter: ImageFilterMode::Linear,
        min_filter: ImageFilterMode::Linear,
        mipmap_filter: ImageFilterMode::Linear,
        ..default()
    });
    image
}

/// A unit quad `(±1, ±1, 0)`; the vertex shader scales and places it.
fn quad_mesh() -> Mesh {
    Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::RENDER_WORLD,
    )
    .with_inserted_attribute(
        Mesh::ATTRIBUTE_POSITION,
        vec![
            [-1.0f32, -1.0, 0.0],
            [1.0, -1.0, 0.0],
            [1.0, 1.0, 0.0],
            [-1.0, 1.0, 0.0],
        ],
    )
    .with_inserted_indices(Indices::U32(vec![0, 1, 2, 0, 2, 3]))
}

fn spawn_moon(
    mut commands: Commands,
    settings: Res<CelestialSettings>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut images: ResMut<Assets<Image>>,
    mut materials: ResMut<Assets<MoonMaterial>>,
) {
    if !settings.sky {
        return;
    }
    let material = MoonMaterial {
        params: MoonParams::default(),
        albedo: images.add(albedo_image()),
    };
    commands.spawn((
        Name::new("Moon disc"),
        MoonDisc,
        Mesh3d(meshes.add(quad_mesh())),
        MeshMaterial3d(materials.add(material)),
        Transform::default(),
        Visibility::Hidden,
        // The shader places every vertex itself, so the mesh bounds mean nothing.
        NoFrustumCulling,
        NotShadowCaster,
    ));
}

/// This frame's uniform from the sky state.
pub(super) fn moon_params(frame: &CelestialFrame, sun: &SunState) -> MoonParams {
    let m = sun.moon_direction.normalize_or(Vec3::Y);
    // Moon-fixed +Z: the ecliptic pole, projected perpendicular to the line of sight.
    let up = (frame.ecliptic_north - m * frame.ecliptic_north.dot(m))
        .try_normalize()
        .unwrap_or_else(|| m.any_orthonormal_vector());
    MoonParams {
        moon: m.extend(0.5 * frame.moon_angular_diameter_deg.to_radians()),
        north: up.extend(sun.moon_illuminated_fraction.clamp(0.0, 1.0)),
        sun_cloud: sun.direction.normalize_or(Vec3::Y).extend(1.0),
        ..default()
    }
}

fn update_moon(
    frame: Res<CelestialFrame>,
    sun: Res<SunState>,
    view: Res<ElementsView>,
    mut materials: ResMut<Assets<MoonMaterial>>,
    mut discs: Query<(&MeshMaterial3d<MoonMaterial>, &mut Visibility), With<MoonDisc>>,
) {
    let params = moon_params(&frame, &sun);
    let visibility = if view.entity.is_some() && sun.moon_altitude_deg > MOON_HIDE_BELOW_DEG {
        Visibility::Visible
    } else {
        Visibility::Hidden
    };
    for (handle, mut shown) in &mut discs {
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
    use std::f64::consts::PI;

    // ---- CPU mirror of `moon.wesl` -------------------------------------------------------

    /// Texture coordinates of selenographic `lat`/`lon` (radians), as in the shader.
    fn albedo_uv(lat: f64, lon: f64) -> (f64, f64) {
        (lon / (2.0 * PI) + 0.5, 0.5 - lat / PI)
    }

    /// Bilinear sample of the map at texture coordinates, longitude wrapping, latitude clamping
    /// (the sampler's `Repeat`/`ClampToEdge`), as a unorm value 0..1.
    fn sample_unorm(u: f64, v: f64) -> f64 {
        let x = u * f64::from(ALBEDO_WIDTH) - 0.5;
        let y = v * f64::from(ALBEDO_HEIGHT) - 0.5;
        let (x0, y0) = (x.floor(), y.floor());
        let (fx, fy) = (x - x0, y - y0);
        let at = |xi: f64, yi: f64| {
            let col = (xi as i64).rem_euclid(i64::from(ALBEDO_WIDTH)) as usize;
            let row = (yi as i64).clamp(0, i64::from(ALBEDO_HEIGHT) - 1) as usize;
            f64::from(ALBEDO[row * ALBEDO_WIDTH as usize + col]) / 255.0
        };
        let top = at(x0, y0) * (1.0 - fx) + at(x0 + 1.0, y0) * fx;
        let bottom = at(x0, y0 + 1.0) * (1.0 - fx) + at(x0 + 1.0, y0 + 1.0) * fx;
        top * (1.0 - fy) + bottom * fy
    }

    /// Albedo at the disc point with orthographic coordinates `(rx, ry)` (right, up).
    fn albedo_at(rx: f64, ry: f64) -> f64 {
        let mu = (1.0 - rx * rx - ry * ry).max(0.0).sqrt();
        let (u, v) = albedo_uv(ry.asin(), rx.atan2(mu));
        sample_unorm(u, v) * f64::from(MOON_MEAN_ALBEDO) * 255.0 / f64::from(MEAN_TEXEL)
    }

    /// Radiance of the disc point `(rx, ry)` with albedo `a`, sun at `s` in the (right, up,
    /// toward-observer) frame, illuminated fraction `k`; cd/m², before cloud, atmosphere, exposure.
    fn shade(rx: f64, ry: f64, a: f64, s: [f64; 3], k: f64) -> f64 {
        let mu = (1.0 - rx * rx - ry * ry).max(0.0).sqrt();
        let mu0 = (rx * s[0] + ry * s[1] + mu * s[2]).max(0.0);
        let lommel = if mu0 + mu > 1e-6 {
            2.0 * mu0 / (mu0 + mu)
        } else {
            0.0
        };
        a * f64::from(lux::RAW_SUNLIGHT) / PI * (lommel + f64::from(EARTHSHINE) * (1.0 - k))
    }

    /// The disc-mean quadrature of `tools/src/bin/gen_moon.rs` (`DISC_SAMPLES = 1024`).
    fn disc_mean(f: impl Fn(f64, f64) -> f64) -> f64 {
        let n = 1024;
        let (mut sum, mut count) = (0.0, 0u64);
        for j in 0..n {
            let y = (f64::from(j) + 0.5) / f64::from(n) * 2.0 - 1.0;
            for i in 0..n {
                let x = (f64::from(i) + 0.5) / f64::from(n) * 2.0 - 1.0;
                if x * x + y * y < 1.0 {
                    sum += f(x, y);
                    count += 1;
                }
            }
        }
        sum / count as f64
    }

    // ---- tests ---------------------------------------------------------------------------

    /// The generator normalises the disc-mean texel to 128.
    #[test]
    fn the_map_is_normalised_on_the_disc() {
        let mean_texel = disc_mean(|x, y| {
            let mu = (1.0 - x * x - y * y).sqrt();
            let (u, v) = albedo_uv(y.asin(), x.atan2(mu));
            sample_unorm(u, v) * 255.0
        });
        assert!((mean_texel - 128.0).abs() < 0.01, "{mean_texel}");
    }

    /// Validation 19: full-moon disc-mean luminance is E_full/Ω = 4843 cd/m² ± 1.
    #[test]
    fn the_full_moon_disc_averages_4843_cd_per_m2() {
        // Full moon: the sun is behind the observer, so ŝ = the toward-observer axis.
        let sun = [0.0, 0.0, 1.0];
        let mean = disc_mean(|x, y| shade(x, y, albedo_at(x, y), sun, 1.0));
        assert!((mean - 4843.0).abs() < 1.0, "{mean}");
        // Lommel–Seeliger keeps the full moon flat: with a uniform albedo every point is equal.
        let flat = [(0.0, 0.0), (0.6, 0.3), (0.0, -0.95), (-0.9, 0.2)]
            .map(|(x, y)| shade(x, y, f64::from(MOON_MEAN_ALBEDO), sun, 1.0));
        for l in flat {
            assert!((l - flat[0]).abs() < 1e-6 * flat[0], "{flat:?}");
        }
        assert!((MOON_MEAN_ALBEDO - 0.117_036).abs() < 1e-6);
    }

    /// Validation 19: at quarter phase the brightest point is on the limb toward the sun's
    /// projected direction (flat albedo, so only the geometry decides).
    #[test]
    fn the_lit_limb_faces_the_sun() {
        for sun_azimuth_deg in [0.0f64, 37.0, 90.0, 123.4, 180.0, 250.0, 315.0] {
            let a = sun_azimuth_deg.to_radians();
            // k = 0.5: the sun is 90° from the line of sight, in the tangent plane.
            let sun = [a.cos(), a.sin(), 0.0];
            let (mut best, mut best_l) = (0.0f64, 0.0f64);
            for i in 0..3600 {
                let phi = f64::from(i) * 0.1_f64.to_radians();
                for r in [0.9, 0.99, 0.999] {
                    let l = shade(
                        r * phi.cos(),
                        r * phi.sin(),
                        f64::from(MOON_MEAN_ALBEDO),
                        sun,
                        0.5,
                    );
                    if l > best_l {
                        (best, best_l) = (phi, l);
                    }
                }
            }
            let mut diff = (best.to_degrees() - sun_azimuth_deg).rem_euclid(360.0);
            if diff > 180.0 {
                diff -= 360.0;
            }
            assert!(
                diff.abs() < 1.0,
                "sun {sun_azimuth_deg}°: brightest at {}°",
                best.to_degrees()
            );
        }
    }

    /// The map's orientation: dark maria where the real Moon has them. Mare Crisium
    /// (17°N, 59°E) and Oceanus Procellarum (18°N, 57°W) against the southern highlands near
    /// Tycho (43°S, 11°W) and the far-side highlands.
    #[test]
    fn the_map_is_oriented_like_the_moon() {
        let patch = |lat_deg: f64, lon_deg: f64| {
            let mut sum = 0.0;
            let mut n = 0.0;
            for dy in -3..=3 {
                for dx in -3..=3 {
                    let (u, v) = albedo_uv(
                        (lat_deg + f64::from(dy)).to_radians(),
                        (lon_deg + f64::from(dx)).to_radians(),
                    );
                    sum += sample_unorm(u, v);
                    n += 1.0;
                }
            }
            sum / n
        };
        let crisium = patch(17.0, 59.0);
        let procellarum = patch(18.0, -57.0);
        let highlands = patch(-43.0, -11.0);
        let far_side = patch(0.0, 150.0);
        assert!(crisium < 0.7 * highlands, "{crisium} vs {highlands}");
        assert!(
            procellarum < 0.7 * highlands,
            "{procellarum} vs {highlands}"
        );
        assert!(procellarum < 0.7 * far_side, "{procellarum} vs {far_side}");
        // A mirrored map would put Procellarum on the east side.
        assert!(patch(18.0, 57.0) > 1.3 * procellarum);
    }

    #[test]
    fn lunar_east_is_screen_right() {
        let frame = CelestialFrame {
            ecliptic_north: Vec3::new(0.2, 0.9, -0.3).normalize(),
            moon_angular_diameter_deg: 0.52,
            ..default()
        };
        let sun = SunState {
            moon_direction: Vec3::new(0.3, 0.5, -0.8).normalize(),
            moon_illuminated_fraction: 0.4,
            ..default()
        };
        let p = moon_params(&frame, &sun);
        let m = p.moon.truncate();
        let up = p.north.truncate();
        assert!((up.length() - 1.0).abs() < 1e-5);
        assert!(
            up.dot(m).abs() < 1e-5,
            "+Z is perpendicular to the line of sight"
        );
        assert!(
            up.dot(frame.ecliptic_north) > 0.0,
            "and points to ecliptic north"
        );
        // Screen right for a camera looking along m with `up` up is m × up; the shader uses it
        // as lunar east (+Y), and (−m) × (m × up) = up closes the frame.
        let right = m.cross(up);
        assert!(((-m).cross(right) - up).length() < 1e-5);
        assert_eq!(p.north.w, 0.4);
        assert_eq!(p.sun_cloud.w, 1.0);
        assert!((p.moon.w - 0.26f32.to_radians()).abs() < 1e-7);
    }

    #[test]
    fn the_mip_chain_is_a_box_filter_down_to_one_texel() {
        let (data, levels) = mip_chain(ALBEDO, ALBEDO_WIDTH, ALBEDO_HEIGHT);
        assert_eq!(levels, 10);
        let expected: usize = (0..levels)
            .map(|l| ((ALBEDO_WIDTH >> l).max(1) * (ALBEDO_HEIGHT >> l).max(1)) as usize)
            .sum();
        assert_eq!(data.len(), expected);
        assert_eq!(&data[..ALBEDO.len()], ALBEDO);
        // 4×2 of known values: level 1 is the 2×2 means, level 2 the 1×1 mean.
        let (small, levels) = mip_chain(&[0, 4, 8, 12, 4, 8, 12, 16], 4, 2);
        assert_eq!(levels, 3);
        assert_eq!(&small[8..], &[4, 12, 8]);
    }

    #[test]
    fn the_albedo_image_is_filtered_and_mipmapped() {
        let image = albedo_image();
        assert_eq!(image.texture_descriptor.format, TextureFormat::R8Unorm);
        assert_eq!(image.texture_descriptor.mip_level_count, 10);
        assert_eq!(image.width(), 512);
        assert_eq!(image.height(), 256);
        assert!(matches!(image.sampler, ImageSampler::Descriptor(_)));
    }
}
