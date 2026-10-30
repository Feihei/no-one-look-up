//! The bolt on screen: crossed additive ribbons, a flash light, and the display-referred
//! brightness that keeps the HDR target finite. Added when the app has mesh assets.

use bevy::asset::RenderAssetUsages;
use bevy::camera::Exposure;
use bevy::camera::visibility::NoFrustumCulling;
use bevy::ecs::system::SystemParam;
use bevy::mesh::{Indices, PrimitiveTopology};
use bevy::prelude::*;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};

use super::bolt::Channel;
use super::strikes::ActiveBolt;
use super::{FLASH_HEIGHT_M, FLASH_LUMENS, FLASH_RANGE_M};
use crate::climate::Weather;
use crate::fog::extinction_for_visibility;
use crate::plugin::{ElementsSystems, ElementsUnits, ElementsView};

/// Scene luminance of the channel core at a first stroke's peak, cd/m² (plan 2.3).
const BOLT_LUMINANCE: f32 = 50_000.0;
/// Blue-white, linear RGB.
const TINT: Vec3 = Vec3::new(0.85, 0.92, 1.0);
/// Half the 6.5 cm mean channel diameter, m.
const PHYSICAL_HALF_WIDTH_M: f32 = 0.0325;
/// The core is at least this many pixels wide.
const CORE_PIXELS: f32 = 2.0;
/// Glow scale over core half-width (Reed & Wyvill: 3.84 / 0.4).
const GLOW_OVER_CORE: f32 = 9.6;
/// The quad's half-width in glow scales, so the Gaussian glow is below 1e-4 at the edge.
const QUAD_GLOW_SCALES: f32 = 3.0;
/// Core profile exponent `n` in `exp(-(r/w)^n)` (Reed & Wyvill Fig. 4).
const CORE_EXPONENT: f32 = 10.0;
/// Glow amplitude against the core's 1.
const GLOW_AMPLITUDE: f32 = 0.0004;
/// Texels across the ramp.
const RAMP_TEXELS: u32 = 256;
/// Pixel angle of a 1080p, 60° view, rad: the width used when the followed camera's is unknown.
const DEFAULT_PIXEL: PixelSize = PixelSize::Angle(1.069e-3);

/// A bolt's render objects.
#[derive(Component)]
struct BoltVisual {
    main: Handle<StandardMaterial>,
    branches: Option<Handle<StandardMaterial>>,
    light: Entity,
}

/// The shared cross-section ramp.
#[derive(Resource)]
struct Ramp(Handle<Image>);

pub(super) fn plugin(app: &mut App) {
    app.add_systems(Startup, warm_up).add_systems(
        Update,
        (attach, animate)
            .chain()
            .after(super::advance_bolts)
            .before(super::write_flash)
            .in_set(ElementsSystems::Simulation),
    );
}

/// The angle one pixel subtends vertically, rad.
pub(crate) fn pixel_angle(fov_y: f32, height_px: f32) -> f32 {
    2.0 * (fov_y * 0.5).tan() / height_px
}

/// The world size of one pixel of the followed camera at a distance from it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum PixelSize {
    /// A perspective camera's: the angle one pixel subtends vertically, rad.
    Angle(f32),
    /// An orthographic camera's: one pixel's height in world units, at every distance.
    Fixed(f32),
}

impl PixelSize {
    /// One pixel's height at `distance_m` from the camera, m.
    pub(crate) fn at(self, distance_m: f32) -> f32 {
        match self {
            Self::Angle(angle) => angle * distance_m,
            Self::Fixed(size) => size,
        }
    }
}

/// Half-width of the channel core at `distance_m`: the physical channel, or two pixels if that
/// is wider on screen.
pub(crate) fn core_half_width(distance_m: f32, pixel: PixelSize) -> f32 {
    PHYSICAL_HALF_WIDTH_M.max(0.5 * CORE_PIXELS * pixel.at(distance_m))
}

/// Half-width of the ribbon quad around a core of half-width `core`.
pub(crate) fn quad_half_width(core: f32) -> f32 {
    core * GLOW_OVER_CORE * QUAD_GLOW_SCALES
}

/// The display value of scene luminance `luminance` (cd/m²) under camera exposure `exposure`
/// (`2^−EV₁₀₀ / 1.2`): Bevy writes an unlit colour as it is, so the exposure is applied here,
/// and the result is clamped so it stays finite in `Rgba16Float`.
pub(crate) use crate::celestial::light::display_value;

/// Material colour for a layer at envelope `mean` under `exposure`.
pub(crate) fn layer_color(mean: f32, exposure: f32) -> LinearRgba {
    let v = display_value(BOLT_LUMINANCE * mean.max(0.0), exposure);
    LinearRgba::new(TINT.x * v, TINT.y * v, TINT.z * v, 1.0)
}

/// `x ≥ 0` as IEEE half-precision bits (round to nearest; below the normal range it is a
/// subnormal, above 65 504 it saturates to the largest finite half).
fn f16_bits(x: f32) -> u16 {
    const SUBNORMAL_STEP: f32 = 1.0 / 16_777_216.0;
    if x < 6.103_515_6e-5 {
        return (x / SUBNORMAL_STEP).round() as u16;
    }
    if x >= 65_504.0 {
        return 0x7BFF;
    }
    let bits = x.to_bits();
    let exponent = (bits >> 23) as i32 - 127 + 15;
    let mantissa = ((bits & 0x007F_FFFF) as f32 / 8192.0).round() as u32;
    // A rounded-up mantissa carries into the exponent, which the addition handles.
    (((exponent as u32) << 10) + mantissa).min(0x7BFF) as u16
}

/// The cross-section values, `RAMP_TEXELS` per row over the quad's width: row 0 is core plus
/// glow, row 1 the core alone.
fn ramp_values() -> [Vec<f32>; 2] {
    let n = RAMP_TEXELS as usize;
    [GLOW_AMPLITUDE, 0.0].map(|glow| {
        (0..n)
            .map(|i| {
                // r in units of the glow scale W, across the quad.
                let x = ((i as f32 + 0.5) / n as f32 * 2.0 - 1.0) * QUAD_GLOW_SCALES;
                let core = (-(GLOW_OVER_CORE * x.abs()).powf(CORE_EXPONENT)).exp();
                core + glow * (-x * x).exp()
            })
            .collect()
    })
}

/// The ramp as a two-row `Rgba16Float` texture. Half floats keep the glow's tail smooth: at the
/// largest display value (4096) an 8-bit step would show as an edge. A vertex's `v` blends the
/// rows by the channel's glow factor.
fn ramp_image() -> Image {
    let one = f16_bits(1.0).to_le_bytes();
    let mut data = Vec::with_capacity(RAMP_TEXELS as usize * 2 * 8);
    for row in ramp_values() {
        for value in row {
            let bytes = f16_bits(value.min(1.0)).to_le_bytes();
            data.extend_from_slice(&[bytes, bytes, bytes, one].concat());
        }
    }
    let mut image = Image::new(
        Extent3d {
            width: RAMP_TEXELS,
            height: 2,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        data,
        TextureFormat::Rgba16Float,
        RenderAssetUsages::RENDER_WORLD,
    );
    image.sampler = bevy::image::ImageSampler::linear();
    image
}

/// A crossed pair of axial ribbons along each channel, positions relative to `origin`. Ribbon
/// widths follow the channel's distance from `view` so the core stays two pixels wide, and the
/// vertex colour is the fog's transmittance `exp(−extinction · distance)` along that distance.
pub(crate) fn ribbon_mesh(
    channels: &[&Channel],
    origin: Vec3,
    view: Vec3,
    pixel: PixelSize,
    extinction: f32,
) -> Mesh {
    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut uvs: Vec<[f32; 2]> = Vec::new();
    let mut colors: Vec<[f32; 4]> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();
    for channel in channels {
        let points = &channel.points;
        let base = positions.len() as u32;
        let v = 0.25 + 0.5 * (1.0 - channel.glow.clamp(0.0, 1.0));
        for (k, &p) in points.iter().enumerate() {
            let before = if k > 0 {
                (p - points[k - 1]).normalize_or_zero()
            } else {
                Vec3::ZERO
            };
            let after = if k + 1 < points.len() {
                (points[k + 1] - p).normalize_or_zero()
            } else {
                Vec3::ZERO
            };
            let tangent = (before + after).try_normalize().unwrap_or(Vec3::Y);
            let s1 = tangent
                .cross(view - p)
                .try_normalize()
                .unwrap_or_else(|| tangent.any_orthonormal_vector());
            let s2 = tangent.cross(s1);
            let distance = p.distance(view);
            let core = core_half_width(distance, pixel) * channel.width;
            let q = quad_half_width(core);
            let t = (-extinction * distance).exp();
            let local = p - origin;
            for (side, dir) in [(-1.0, s1), (1.0, s1), (-1.0, s2), (1.0, s2)] {
                positions.push((local + dir * q * side).to_array());
                uvs.push([if side < 0.0 { 0.0 } else { 1.0 }, v]);
                colors.push([t, t, t, 1.0]);
            }
        }
        for k in 0..points.len() as u32 - 1 {
            let a = base + 4 * k;
            let b = a + 4;
            for ribbon in [0, 2] {
                let (l0, r0, l1, r1) = (a + ribbon, a + ribbon + 1, b + ribbon, b + ribbon + 1);
                indices.extend_from_slice(&[l0, r0, l1, r0, r1, l1]);
            }
        }
    }
    Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
    .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, uvs)
    .with_inserted_attribute(Mesh::ATTRIBUTE_COLOR, colors)
    .with_inserted_indices(Indices::U32(indices))
}

fn material(ramp: &Handle<Image>) -> StandardMaterial {
    StandardMaterial {
        base_color: Color::BLACK,
        base_color_texture: Some(ramp.clone()),
        unlit: true,
        // Bevy's fog runs before the additive premultiply, so on a black texel it would still add
        // the fog colour over the whole quad. The ribbon's vertex colour carries the Koschmieder
        // transmittance instead.
        fog_enabled: false,
        alpha_mode: AlphaMode::Add,
        double_sided: true,
        cull_mode: None,
        ..default()
    }
}

/// Pipelines compile asynchronously on first use, which would cost the first strike its first
/// frames: draw one degenerate ribbon with the bolt's material from the start, so the pipeline
/// is ready when lightning is.
fn warm_up(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut images: ResMut<Assets<Image>>,
) {
    let ramp = images.add(ramp_image());
    let mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, vec![[0.0f32; 3]; 3])
    .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, vec![[0.0f32; 2]; 3])
    .with_inserted_attribute(Mesh::ATTRIBUTE_COLOR, vec![[0.0f32; 4]; 3])
    .with_inserted_indices(Indices::U32(vec![0, 1, 2]));
    commands.spawn((
        Name::new("Lightning pipeline warm-up"),
        Mesh3d(meshes.add(mesh)),
        MeshMaterial3d(materials.add(material(&ramp))),
        NoFrustumCulling,
    ));
    commands.insert_resource(Ramp(ramp));
}

/// The followed camera, for the width of one pixel.
#[derive(SystemParam)]
struct ViewPixels<'w, 's> {
    view: Res<'w, ElementsView>,
    units: Res<'w, ElementsUnits>,
    cameras: Query<'w, 's, (&'static Camera, &'static Projection)>,
}

impl ViewPixels<'_, '_> {
    /// The size in metres of one pixel of the followed camera: the angle a perspective camera's
    /// pixel subtends, or an orthographic camera's pixel height; a 1080p, 60° view's if there is
    /// no followed camera with a target yet.
    fn size(&self) -> PixelSize {
        self.view
            .entity
            .and_then(|e| self.cameras.get(e).ok())
            .and_then(|(camera, projection)| {
                let height_px = camera.physical_target_size()?.y as f32;
                match projection {
                    Projection::Perspective(p) => {
                        Some(PixelSize::Angle(pixel_angle(p.fov, height_px)))
                    }
                    Projection::Orthographic(o) => Some(PixelSize::Fixed(
                        self.units.metres(o.area.height()) / height_px,
                    )),
                    Projection::Custom(_) => None,
                }
            })
            .filter(|p| match *p {
                PixelSize::Angle(a) | PixelSize::Fixed(a) => a.is_finite() && a > 0.0,
            })
            .unwrap_or(DEFAULT_PIXEL)
    }
}

/// Build the ribbons and the light for every bolt that just struck.
#[allow(clippy::too_many_arguments)]
fn attach(
    mut commands: Commands,
    new: Query<(Entity, &ActiveBolt), Added<ActiveBolt>>,
    pixels: ViewPixels,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    ramp: Res<Ramp>,
    weather: Res<Weather>,
    units: Res<ElementsUnits>,
) {
    if new.is_empty() {
        return;
    }
    let pixel = pixels.size();
    let ramp = ramp.0.clone();
    let extinction = extinction_for_visibility(weather.visibility_m);
    for (entity, active) in &new {
        let ground = active.ground();
        let main_mesh = ribbon_mesh(
            &[&active.bolt.channels[0]],
            ground,
            active.view,
            pixel,
            extinction,
        );
        let main = materials.add(material(&ramp));
        let branch_channels: Vec<&Channel> = active.bolt.channels[1..].iter().collect();
        let branches = (!branch_channels.is_empty()).then(|| {
            (
                meshes.add(ribbon_mesh(
                    &branch_channels,
                    ground,
                    active.view,
                    pixel,
                    extinction,
                )),
                materials.add(material(&ramp)),
            )
        });
        let mut light = Entity::PLACEHOLDER;
        let branch_material = branches.as_ref().map(|(_, m)| m.clone());
        // The bolt is built in metres about its attachment point: its entity stands at that
        // point in the world and is scaled by the units, so the ribbons and the light's height
        // follow. A light's range and intensity are not transformed: they are written in world
        // units (range × u, intensity × u², `E = I / d²`).
        let u = units.per_metre();
        commands.entity(entity).insert((
            Name::new("Lightning bolt"),
            Transform::from_translation(units.point_to_world(ground)).with_scale(Vec3::splat(u)),
            Visibility::default(),
        ));
        commands.entity(entity).with_children(|parent| {
            parent.spawn((Mesh3d(meshes.add(main_mesh)), MeshMaterial3d(main.clone())));
            if let Some((mesh, material)) = branches {
                parent.spawn((Mesh3d(mesh), MeshMaterial3d(material)));
            }
            light = parent
                .spawn((
                    PointLight {
                        color: Color::linear_rgb(TINT.x, TINT.y, TINT.z),
                        intensity: 0.0,
                        range: FLASH_RANGE_M * u,
                        shadow_maps_enabled: false,
                        ..default()
                    },
                    Transform::from_xyz(0.0, FLASH_HEIGHT_M, 0.0),
                ))
                .id();
        });
        commands.entity(entity).insert(BoltVisual {
            main,
            branches: branch_material,
            light,
        });
    }
}

/// Set this frame's brightness: the ribbons' display colour and the light's power.
fn animate(
    bolts: Query<(&ActiveBolt, &BoltVisual)>,
    view: Res<ElementsView>,
    units: Res<ElementsUnits>,
    exposures: Query<&Exposure>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut lights: Query<&mut PointLight>,
) {
    let exposure = view
        .entity
        .and_then(|e| exposures.get(e).ok())
        .map_or_else(|| Exposure::default().exposure(), Exposure::exposure);
    for (active, visual) in &bolts {
        if let Some(mut material) = materials.get_mut(&visual.main) {
            material.base_color = layer_color(active.mean, exposure).into();
        }
        if let Some(mut material) = visual
            .branches
            .as_ref()
            .and_then(|handle| materials.get_mut(handle))
        {
            material.base_color = layer_color(active.mean_first, exposure).into();
        }
        if let Ok(mut light) = lights.get_mut(visual.light) {
            let u = units.per_metre();
            light.intensity = FLASH_LUMENS * active.mean * u * u;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::celestial::light::DISPLAY_MAX;
    use crate::lightning::bolt::generate;

    fn exposure(ev100: f32) -> f32 {
        Exposure { ev100 }.exposure()
    }

    #[test]
    fn display_values_follow_the_notes_table() {
        let v = |ev| display_value(BOLT_LUMINANCE, exposure(ev));
        assert!((v(13.336) - 4.03).abs() < 0.02, "noon {}", v(13.336));
        assert!((v(6.478) - 467.0).abs() < 2.0, "sunrise {}", v(6.478));
        for night in [0.857, -2.235, -4.747] {
            assert_eq!(v(night), DISPLAY_MAX, "EV {night}");
        }
        // The clamp turns on at EV 3.347.
        assert_eq!(v(3.30), DISPLAY_MAX);
        assert!(v(3.40) < DISPLAY_MAX);
        let threshold = (BOLT_LUMINANCE / 1.2 / DISPLAY_MAX).log2();
        assert!((threshold - 3.347).abs() < 0.01, "{threshold}");
    }

    #[test]
    fn no_channel_is_ever_above_the_clamp_or_non_finite() {
        let mut ev = -6.0;
        while ev <= 16.0 {
            for mean in [0.0, 0.01, 0.2, 1.0, 1.2] {
                let c = layer_color(mean, exposure(ev));
                for channel in [c.red, c.green, c.blue, c.alpha] {
                    assert!(
                        channel.is_finite() && channel <= DISPLAY_MAX,
                        "EV {ev}: {c:?}"
                    );
                }
            }
            ev += 0.25;
        }
    }

    #[test]
    fn the_ribbon_keeps_two_pixels_of_core() {
        let delta = pixel_angle(60f32.to_radians(), 1080.0);
        assert!((delta - 1.069e-3).abs() < 1e-6, "{delta}");
        let delta = PixelSize::Angle(delta);
        let core = |d| 2.0 * core_half_width(d, delta);
        assert!((core(1000.0) - 2.14).abs() < 0.01, "{}", core(1000.0));
        assert!(
            (quad_half_width(core_half_width(1000.0, delta)) / (GLOW_OVER_CORE * QUAD_GLOW_SCALES)
                - 1.07)
                .abs()
                < 0.01
        );
        // The ribbon proper (one glow scale each side) is 9.6x the core: 20.5 m at 1 km.
        assert!((GLOW_OVER_CORE * core(1000.0) - 20.5).abs() < 0.1);
        // 20 m away the physical 6.5 cm channel is wider than two pixels (4.3 cm).
        assert!((core(20.0) - 0.065).abs() < 1e-6, "{}", core(20.0));
        assert!((core(5000.0) - 10.7).abs() < 0.05, "{}", core(5000.0));
    }

    /// Under an orthographic camera a pixel is the same size at every distance, so the core is
    /// two pixels wide wherever the bolt strikes (issue #7).
    #[test]
    fn an_orthographic_ribbon_keeps_two_pixels_at_every_distance() {
        // A 1080-pixel view 200 m high: 0.185 m a pixel, 0.370 m of core.
        let pixel = PixelSize::Fixed(200.0 / 1080.0);
        for d in [10.0, 1500.0, 8000.0] {
            let core = 2.0 * core_half_width(d, pixel);
            assert!(
                core.is_finite() && (core - 0.3704).abs() < 1e-3,
                "{d} m: {core}"
            );
        }
    }

    #[test]
    fn a_ribbon_pair_is_four_triangles_per_segment() {
        let bolt = generate(11, Vec3::ZERO, 1500.0);
        let mesh = ribbon_mesh(
            &[&bolt.channels[0]],
            Vec3::ZERO,
            Vec3::new(1000.0, 5.0, 0.0),
            PixelSize::Angle(1e-3),
            0.0,
        );
        let triangles = mesh.indices().map_or(0, Indices::len) / 3;
        assert_eq!(triangles, 4 * bolt.channels[0].segments());
        let positions = mesh
            .attribute(Mesh::ATTRIBUTE_POSITION)
            .map_or(0, |a| a.len());
        assert_eq!(positions, 4 * bolt.channels[0].points.len());
    }

    #[test]
    fn vertex_colour_is_the_fogs_transmittance() {
        // 2 % at the visibility (Koschmieder), 1 with no fog.
        let bolt = generate(11, Vec3::ZERO, 1500.0);
        let view = Vec3::new(1500.0, 0.0, 0.0);
        let colours = |extinction| {
            let mesh = ribbon_mesh(
                &[&bolt.channels[0]],
                Vec3::ZERO,
                view,
                PixelSize::Angle(1e-3),
                extinction,
            );
            match mesh.attribute(Mesh::ATTRIBUTE_COLOR) {
                Some(bevy::mesh::VertexAttributeValues::Float32x4(c)) => c.clone(),
                other => panic!("no vertex colours: {other:?}"),
            }
        };
        assert!(colours(0.0).iter().all(|c| *c == [1.0, 1.0, 1.0, 1.0]));
        let foggy = colours(extinction_for_visibility(1500.0));
        let ground = foggy[0][0];
        assert!((ground - 0.02).abs() < 1e-3, "{ground}");
        // Farther nodes are dimmer; the top of the channel is a little farther than the foot.
        assert!(foggy.iter().all(|c| c[0] > 0.0 && c[0] <= 1.0));
    }

    /// Half-precision bits back to `f32`, for the ramp test.
    fn f16_value(bits: u16) -> f32 {
        let (exponent, mantissa) = (i32::from(bits >> 10) & 31, f32::from(bits & 1023));
        if exponent == 0 {
            mantissa / 16_777_216.0
        } else {
            (1.0 + mantissa / 1024.0) * 2f32.powi(exponent - 15)
        }
    }

    #[test]
    fn half_floats_round_trip_over_the_ramps_range() {
        for value in [
            1.0, 0.5, 1e-3, 1.2e-7, 4.0e-4, 0.123_456, 6.1e-5, 3e-5, 1e-7,
        ] {
            let back = f16_value(f16_bits(value));
            assert!(
                (back / value - 1.0).abs() < 1e-3 || (back - value).abs() < 6e-8,
                "{value} -> {back}"
            );
        }
        assert_eq!(f16_bits(1.0), 0x3C00);
        assert_eq!(f16_bits(0.0), 0);
        assert_eq!(f16_bits(1e9), 0x7BFF);
    }

    #[test]
    fn the_ramp_has_a_hot_core_a_faint_glow_and_a_vanishing_edge() {
        let image = ramp_image();
        let data = image.data.as_ref().map_or(&[][..], |d| &d[..]);
        let texel = |row: usize, i: usize| {
            let at = (row * RAMP_TEXELS as usize + i) * 8;
            f16_value(u16::from_le_bytes([data[at], data[at + 1]]))
        };
        let n = RAMP_TEXELS as usize;
        assert!(texel(0, n / 2) > 0.99 && texel(0, n / 2 - 1) > 0.99);
        // The glow tail is below any visible level at the quad's edge, even at the 4096 clamp.
        assert!(4096.0 * texel(0, 0) < 1e-2 && 4096.0 * texel(0, n - 1) < 1e-2);
        // Away from the core the glow row is the glow alone and the core row is zero.
        let x = (n / 2 + 40) as f32;
        assert!(texel(0, x as usize) > 1e-4 && texel(1, x as usize) < 1e-9);
        // The glow falls monotonically outward.
        assert!((n / 2..n - 1).all(|i| texel(0, i + 1) <= texel(0, i)));
    }

    #[test]
    fn flash_photometry_matches_bevys_point_light() {
        // Candela is lumens over 4 pi (bevy_pbr light.rs:574); the surface faces the light, the
        // range window is (1 - (d^2/R^2)^2)^2 (pbr_lighting.wesl:128-136).
        let candela = FLASH_LUMENS / (4.0 * std::f32::consts::PI);
        assert!((candela / 7.958e6 - 1.0).abs() < 1e-3);
        let lux = |horizontal: f32| {
            let d2 = horizontal * horizontal + FLASH_HEIGHT_M * FLASH_HEIGHT_M;
            let f = d2 / (FLASH_RANGE_M * FLASH_RANGE_M);
            let window = (1.0 - f * f).clamp(0.0, 1.0).powi(2);
            candela * window / d2
        };
        for (x, want) in [
            (300.0, 44.2),
            (500.0, 23.4),
            (1000.0, 7.27),
            (2000.0, 1.84),
            (3000.0, 0.659),
        ] {
            assert!((lux(x) / want - 1.0).abs() < 0.01, "{x} m: {}", lux(x));
        }
        assert!(lux(5000.0) < 1e-6);
        assert!((lux(0.0) - 88.4).abs() < 0.5, "{}", lux(0.0));
    }
}
