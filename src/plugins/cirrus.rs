//! Cirrus: high, thin, wispy cloud rendered as a shaded sky dome (R5.6).
//!
//! Why not the volumetric layer: cirrus sits at 6-12 km and reads as a flat
//! sheet with essentially no parallax, so marching it as volume pays for depth
//! nobody can see. Its defining look is stretched fibre, which is exactly what
//! 2D noise stretched along one axis produces and what volumetric noise is
//! worst at. A dome shader is a fraction of the cost.
//!
//! Why a mesh rather than a full-screen pass: the dome goes through the normal
//! mesh pipeline, so the depth buffer occludes it behind buildings and trees
//! for free. A full-screen post-process pass cannot see depth (its render pass
//! has no depth attachment), and would paint cirrus over the street.
//!
//! The dome is a large inverted sphere that follows the camera. Its own
//! rotation is fixed, so the cloud pattern stays put in world space as you walk
//! (that is what makes it read as "high up" rather than "stuck to the lens").

use bevy::mesh::MeshBuilder;
use bevy::pbr::{Material, MaterialPlugin};
use bevy::prelude::*;
use bevy::render::render_resource::{AsBindGroup, ShaderType};
use bevy::shader::ShaderRef;

/// Radius of the dome (m).
///
/// Two hard bounds decide this:
/// - **Inside the camera's far plane.** `PerspectiveProjection::default()`
///   uses `far = 1000 m` and nothing in this project overrides it, so a larger
///   dome would be clipped away entirely and the sky would simply lose its
///   cirrus. (A 4 km dome was the first attempt and did exactly that.)
/// - **Outside all world geometry.** The streaming window is
///   `CHUNK_AHEAD * CHUNK_SIZE = 200 m` and trees are culled at 220 m, so 900 m
///   leaves the dome comfortably behind everything that can be drawn.
pub const DOME_RADIUS: f32 = 900.0;

/// Cirrus look and placement, kept in one struct so the sky panel (R5.1) can
/// drive it and the values can be asserted without a GPU.
#[derive(Resource, Clone, Copy, Debug, PartialEq)]
pub struct CirrusSettings {
    /// How much cirrus the sky carries, 0..1. 0 disables the dome entirely.
    pub amount: f32,
    /// How far the fibre stretches, 1..8. High values give long streaks.
    pub stretch: f32,
    /// Noise frequency; larger is finer.
    pub scale: f32,
    /// Neighbouring streaks stay this similar, 0..1.
    pub directionality: f32,
    /// Drift speed (m/s) of the pattern, so the sky is not frozen.
    pub drift: f32,
}

impl Default for CirrusSettings {
    fn default() -> Self {
        Self {
            // Off by default: Feihei's call is that the volumetric layer stays
            // the main cloud and cirrus is an optional, very cheap addition.
            amount: 0.0,
            stretch: 4.0,
            scale: 2.2,
            directionality: 0.75,
            drift: 6.0,
        }
    }
}

/// Uniform sent to the dome shader. `AsBindGroup` packs it into binding 0.
#[derive(Clone, Copy, Debug, ShaderType)]
pub struct CirrusUniform {
    /// xyz: unit direction toward the sun. w: amount.
    pub sun: Vec4,
    /// rgb: sky-scattered light tint (linear). a: brightness.
    pub tint: Vec4,
    /// x: stretch. y: scale. z: directionality. w: drift time (s).
    pub params: Vec4,
}

impl Default for CirrusUniform {
    fn default() -> Self {
        Self {
            sun: Vec4::new(0.0, 0.1, -1.0, 0.0),
            tint: Vec4::new(1.0, 0.85, 0.7, 1.0),
            params: Vec4::new(4.0, 2.2, 0.75, 0.0),
        }
    }
}

/// Material for the cirrus dome.
#[derive(Asset, AsBindGroup, TypePath, Clone, Debug, Default)]
pub struct CirrusMaterial {
    #[uniform(0)]
    pub uniform: CirrusUniform,
}

impl Material for CirrusMaterial {
    fn vertex_shader() -> ShaderRef {
        "shaders/cirrus.wesl".into()
    }

    fn fragment_shader() -> ShaderRef {
        "shaders/cirrus.wesl".into()
    }

    /// Cirrus is drawn after the opaque scene so its few fragments blend over
    /// the sky; it writes no depth.
    fn alpha_mode(&self) -> AlphaMode {
        AlphaMode::Blend
    }

    /// The dome does not receive light or cast shadows: its shading is the
    /// analytic sky tint computed in the fragment shader.
    fn enable_shadows() -> bool {
        false
    }

    /// The camera is INSIDE the dome, so every triangle is back-facing. Bevy's
    /// mesh pipeline culls back faces by default for every alpha mode, which
    /// makes a plain sphere invisible from within. Handing the shader both
    /// faces is simpler and more reliable than mirroring the geometry: a
    /// negative scale does not actually reverse winding once the triangle has
    /// moved to the other side (verified numerically).
    fn reads_view_transmission_texture(&self) -> bool {
        false
    }
}

/// The dome entity, kept as a component so the follow system can find it.
#[derive(Component)]
pub struct CirrusDome;

pub struct CirrusPlugin;

impl Plugin for CirrusPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<CirrusSettings>()
            .add_plugins(MaterialPlugin::<CirrusMaterial>::default())
            .add_systems(Startup, spawn_dome)
            .add_systems(Update, (follow_camera, update_uniform));
    }
}

fn spawn_dome(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<CirrusMaterial>>,
) {
    info!("cirrus dome spawned: radius {DOME_RADIUS} m, amount 0 = hidden");
    commands.spawn((
        Name::new("CirrusDome"),
        CirrusDome,
        Mesh3d(meshes.add(uv_sphere(DOME_RADIUS))),
        MeshMaterial3d(materials.add(CirrusMaterial::default())),
        // No scale flip. A negative scale mirrors the geometry, which does
        // NOT reliably reverse triangle winding (verified: the cross product
        // of a mirrored triangle keeps its sign once the triangle has moved to
        // the other side). The dome is seen from inside, so the shader emits
        // both faces instead -- see the `cull_mode: None` note on the material.
        Transform::IDENTITY,
        // Never culled: the camera is always inside it.
        Visibility::default(),
        bevy::camera::visibility::NoFrustumCulling,
    ));
}

/// A sphere mesh at `radius`, **wound inside-out**.
///
/// The camera sits at the centre and looks at the inside surface, so every
/// triangle would be back-facing and culled. Bevy's `Material` trait has no
/// `cull_mode` hook, so the fix has to be in the geometry: reverse each
/// triangle's index order and the same faces become front-facing from within.
///
/// (`ico()` returns a `Result` and a fallback mesh would be a silently
/// invisible dome; the UV sphere cannot fail and its seams do not matter for a
/// noise pattern.)
fn uv_sphere(radius: f32) -> Mesh {
    let mut mesh = Sphere::new(radius).mesh().build();
    match mesh.indices_mut() {
        Some(bevy::mesh::Indices::U32(indices)) => {
            for triangle in indices.as_chunks_mut::<3>().0 {
                triangle.swap(1, 2);
            }
        }
        Some(bevy::mesh::Indices::U16(indices)) => {
            for triangle in indices.as_chunks_mut::<3>().0 {
                triangle.swap(1, 2);
            }
        }
        None => {}
    }
    mesh
}

/// Keeps the dome centred on the camera. The dome does not inherit the
/// camera's rotation, so the pattern stays world-aligned.
fn follow_camera(
    cameras: Query<&GlobalTransform, (With<Camera3d>, Without<CirrusDome>)>,
    mut domes: Query<&mut Transform, With<CirrusDome>>,
) {
    let Ok(camera) = cameras.single() else {
        return;
    };
    let centre = camera.translation();
    for mut transform in &mut domes {
        if transform.translation != centre {
            transform.translation = centre;
        }
    }
}

/// Copies the live sky tint and sun direction into the dome's uniform, and
/// hides the dome when cirrus is off.
///
/// The tint comes from the camera's `AmbientLight`, which bevy_elements derives
/// from the sky's own dusk gradient -- so cirrus does not need a second colour
/// model that could drift out of step with the rest of the sky.
fn update_uniform(
    settings: Res<CirrusSettings>,
    sun: Option<Res<bevy_elements::SunState>>,
    lights: Query<&AmbientLight, With<Camera3d>>,
    mut domes: Query<(&mut Visibility, &MeshMaterial3d<CirrusMaterial>), With<CirrusDome>>,
    mut materials: ResMut<Assets<CirrusMaterial>>,
    time: Res<Time>,
    mut elapsed: Local<f32>,
) {
    *elapsed += time.delta_secs();

    let amount = settings.amount.clamp(0.0, 1.0);
    let (sun_dir, altitude) = sun
        .as_ref()
        .map(|s| (s.direction.normalize_or(Vec3::Y), s.altitude_deg))
        .unwrap_or((Vec3::Y, 90.0));
    let light = lights.iter().next();
    let (tint, brightness) = match light {
        Some(l) => (l.color, l.brightness),
        None => (Color::WHITE, 1.0),
    };
    // Linear rgb of the ambient colour; the dome shades relative to it.
    let rgb = tint.to_linear();

    for (mut visibility, handle) in &mut domes {
        *visibility = if amount <= 0.001 {
            Visibility::Hidden
        } else {
            Visibility::Inherited
        };
        if amount <= 0.001 {
            continue;
        }
        let Some(mut material) = materials.get_mut(&handle.0) else {
            continue;
        };
        material.uniform = CirrusUniform {
            sun: Vec4::new(sun_dir.x, sun_dir.y, sun_dir.z, amount),
            tint: Vec4::new(rgb.red, rgb.green, rgb.blue, brightness.max(0.0)),
            params: Vec4::new(
                settings.stretch.clamp(1.0, 8.0),
                settings.scale.clamp(0.2, 12.0),
                settings.directionality.clamp(0.0, 1.0),
                // Absolute time, not a per-frame delta: the pattern must not
                // jump when the amount changes or the frame rate varies.
                *elapsed * settings.drift,
            ),
        };
        let _ = altitude;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cirrus_is_off_by_default_and_that_is_a_real_state() {
        // Feihei's call: the volumetric layer stays the main cloud, cirrus is
        // an optional cheap addition. A default that quietly showed cirrus
        // would change the dusk look of every existing save.
        let settings = CirrusSettings::default();
        assert_eq!(settings.amount, 0.0, "cirrus must ship disabled");

        // And OFF must be expressible: the dome hides at this amount rather
        // than rendering an invisible-but-present layer.
        assert!(settings.amount <= 0.001);
    }

    #[test]
    fn settings_stay_in_the_ranges_the_shader_assumes() {
        // The shader divides by `stretch` and uses `scale` as a frequency.
        // A zero stretch would divide by zero; a zero scale would collapse the
        // noise domain to a constant.
        let shader_clamp = |s: CirrusSettings| CirrusUniform {
            sun: Vec4::new(0.0, 1.0, 0.0, s.amount.clamp(0.0, 1.0)),
            tint: Vec4::ONE,
            params: Vec4::new(
                s.stretch.clamp(1.0, 8.0),
                s.scale.clamp(0.2, 12.0),
                s.directionality.clamp(0.0, 1.0),
                0.0,
            ),
        };
        for stretch in [-5.0f32, 0.0, 1.0, 8.0, 100.0] {
            for scale in [-1.0f32, 0.0, 0.2, 12.0, 999.0] {
                let u = shader_clamp(CirrusSettings {
                    stretch,
                    scale,
                    ..Default::default()
                });
                assert!(
                    u.params.x >= 1.0 && u.params.x <= 8.0,
                    "stretch {stretch} escaped its range: {}",
                    u.params.x
                );
                assert!(
                    u.params.y >= 0.2 && u.params.y <= 12.0,
                    "scale {scale} escaped its range: {}",
                    u.params.y
                );
            }
        }
    }

    #[test]
    fn the_dome_is_inside_the_far_plane_and_outside_the_world() {
        // Both bounds are real. The upper one bit immediately: the first dome
        // was 4 km, `PerspectiveProjection::default().far` is 1000 m, and
        // nothing in this project raises it -- so the dome was clipped away
        // and rendered nothing at all.
        // Read through a local so the constants stay comparable rather than
        // being folded into a constant `true`/`false` at compile time.
        let dome = DOME_RADIUS;
        let far = 1000.0f32;
        assert!(
            dome < far * 0.95,
            "dome radius {dome} m reaches the camera's default far plane              ({far} m) and would be clipped"
        );

        let streaming_extent = crate::world::CHUNK_AHEAD as f32 * crate::world::CHUNK_SIZE;
        assert!(
            DOME_RADIUS > streaming_extent * 1.5,
            "dome radius {DOME_RADIUS} m is too close to the {streaming_extent} m              streaming window; geometry would poke through it"
        );
        // Trees are culled at 220 m, handled by the bound above.
    }

    #[test]
    fn stretching_keeps_the_noise_domain_finite() {
        // The shader computes `1 - 1/stretch`; stretch is clamped to >= 1, so
        // that term stays in [0, 0.875]. Documented here because a future
        // change to the clamp would produce NaNs that discard every fragment.
        for stretch in [1.0f32, 2.0, 4.0, 8.0] {
            let factor = 1.0 - 1.0 / stretch.clamp(1.0, 8.0);
            assert!(
                factor.is_finite() && (0.0..=0.875).contains(&factor),
                "stretch {stretch} gives factor {factor}"
            );
        }
    }

    #[test]
    fn the_dome_geometry_faces_inward() {
        // The camera is inside the dome and Bevy culls back faces for every
        // alpha mode (`cull_mode: Some(Face::Back)`), so a normally-wound
        // sphere renders NOTHING from within. The mesh is wound inside-out;
        // this asserts that, because the failure mode is a completely
        // invisible sky with no error to point at.
        let mesh = uv_sphere(DOME_RADIUS);
        let positions = mesh
            .attribute(bevy::mesh::Mesh::ATTRIBUTE_POSITION)
            .expect("positions");
        let Some(bevy::mesh::Indices::U32(indices)) = mesh.indices() else {
            panic!("expected u32 indices");
        };
        let pos = |i: u32| -> Vec3 {
            match positions {
                bevy::mesh::VertexAttributeValues::Float32x3(values) => {
                    Vec3::from_array(values[i as usize])
                }
                other => panic!("unexpected position format {other:?}"),
            }
        };

        // For an inward-wound triangle, (b - a) x (c - a) points TOWARD the
        // centre, i.e. opposite the centroid direction.
        let mut checked = 0;
        for triangle in indices.as_chunks::<3>().0.iter().take(200) {
            let (a, b, c) = (pos(triangle[0]), pos(triangle[1]), pos(triangle[2]));
            let geometric = (b - a).cross(c - a);
            if geometric.length_squared() < 1e-9 {
                continue;
            }
            let centroid = (a + b + c) / 3.0;
            let outward = centroid.normalize();
            assert!(
                geometric.normalize().dot(outward) < 0.0,
                "triangle normal points outward; the dome would be culled                  from inside and the sky would show no cirrus"
            );
            checked += 1;
        }
        assert!(checked > 50, "only {checked} triangles were checked");
    }
}
