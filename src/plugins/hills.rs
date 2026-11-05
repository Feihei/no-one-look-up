//! Distant hill silhouettes on the horizon.
//!
//! The walk is enclosed by a 0.8 km city, so the skyline above the outer
//! rooftops used to be bare sky. Three concentric bands -- nearest darkest,
//! farthest palest -- give the eye a "there is landscape beyond the city" cue
//! without adding world geometry: the bands are 640-840 m away and cost a
//! handful of triangles.
//!
//! Why opaque flat bands rather than a transparent gradient: the layers
//! overlap in screen space, and transparent sorting is per-entity, not per
//! pixel. Two blended rings can draw far-then-near correctly or near-then-far
//! incorrectly depending on the sort, and the wrong order puts a pale far hill
//! on top of a dark near one. Opaque geometry is depth-tested in any order,
//! so layering is guaranteed; the "fading with distance" is carried by each
//! layer's own colour instead of by alpha.
//!
//! The rings follow the camera horizontally (like the cirrus dome) so the
//! player can never reach them: the map is 800 m across while the rings start
//! at 640 m, and a fixed ring would loom at the map's far edge. Vertical
//! position stays at zero so the horizon line does not move with the eye.

use bevy::asset::RenderAssetUsages;
use bevy::mesh::{Indices, Mesh, PrimitiveTopology};
use bevy::prelude::*;
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;

/// How far below the horizon each band starts.
///
/// The band must not end at y = 0: the streamed ground stops at the chunk
/// window (~200 m), so beyond that a band based exactly at eye level would
/// leave a sliver of sky between the ground edge and the hill foot. Extending
/// far below the horizon lets the ground (and the outer building fence) occlude
/// the foot, so the hills always grow out of the skyline.
const BASE_Y: f32 = -200.0;

/// Angular resolution of one band. At 640 m a 256-segment band has ~16 m
/// chords, well under a pixel at the distances these layers sit.
const SEGMENTS: usize = 256;

/// One silhouette layer.
pub struct HillLayer {
    /// Distance from the camera (m). Must stay inside the sky dome.
    pub radius: f32,
    /// Height of the highest ridge (m).
    pub peak: f32,
    /// Unlit silhouette colour. Farther layers are paler so the stack reads
    /// as depth haze rather than three equally dark cutouts.
    pub tint: [f32; 3],
}

/// Three layers, near to far. Peak heights give roughly 3.8°, 4.6° and 5.4°
/// above the horizon; tall enough to clear the outer rooftops, short enough to
/// stay a *distant* range.
pub const LAYERS: [HillLayer; 3] = [
    HillLayer {
        radius: 640.0,
        peak: 42.0,
        tint: [0.20, 0.22, 0.28],
    },
    HillLayer {
        radius: 740.0,
        peak: 60.0,
        tint: [0.30, 0.32, 0.40],
    },
    HillLayer {
        radius: 840.0,
        peak: 80.0,
        tint: [0.40, 0.42, 0.51],
    },
];

/// Marker for a hill band entity.
#[derive(Component)]
pub struct HillBand;

pub struct HillsPlugin;

impl Plugin for HillsPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, spawn_bands)
            .add_systems(Update, follow_camera);
    }
}

fn spawn_bands(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    for (index, layer) in LAYERS.iter().enumerate() {
        let color = Color::srgb(layer.tint[0], layer.tint[1], layer.tint[2]);
        commands.spawn((
            Name::new(format!("HillBand{index}")),
            HillBand,
            Mesh3d(meshes.add(hill_band(index, layer.radius, layer.peak))),
            MeshMaterial3d(materials.add(StandardMaterial {
                base_color: color,
                // No lighting, no shadowing: a silhouette is a painted shape.
                unlit: true,
                // Seen from inside the ring; both faces keep it safe under
                // Bevy's per-material back-face culling.
                cull_mode: None,
                ..default()
            })),
            Transform::IDENTITY,
            // Never cull: the camera is always inside the ring.
            bevy::camera::visibility::NoFrustumCulling,
        ));
    }
}

/// Keeps the bands centred on the camera in XZ, at ground level in Y.
fn follow_camera(
    cameras: Query<&GlobalTransform, (With<Camera3d>, Without<HillBand>)>,
    mut bands: Query<&mut Transform, With<HillBand>>,
) {
    let Ok(camera) = cameras.single() else {
        return;
    };
    let centre = camera.translation();
    for mut transform in &mut bands {
        let target = Vec3::new(centre.x, 0.0, centre.z);
        if transform.translation != target {
            transform.translation = target;
        }
    }
}

/// A vertical band whose top edge is a periodic ridge line.
fn hill_band(layer: usize, radius: f32, peak: f32) -> Mesh {
    let mut positions = Vec::with_capacity((SEGMENTS + 1) * 2);
    let mut indices = Vec::with_capacity(SEGMENTS * 6);
    for i in 0..=SEGMENTS {
        let theta = i as f32 / SEGMENTS as f32 * std::f32::consts::TAU;
        let (sin, cos) = theta.sin_cos();
        let height = peak * ridge(layer, theta);
        positions.push([radius * cos, BASE_Y, radius * sin]);
        positions.push([radius * cos, height, radius * sin]);
    }
    for i in 0..SEGMENTS as u32 {
        let bottom = i * 2;
        let top = bottom + 1;
        let next_bottom = bottom + 2;
        let next_top = bottom + 3;
        indices.extend_from_slice(&[bottom, next_bottom, top, top, next_bottom, next_top]);
    }
    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_indices(Indices::U32(indices));
    mesh
}

/// Ridge profile in `0.35..=1.0`, periodic in theta.
///
/// Only integer harmonics of theta are used, so the profile closes on itself
/// exactly at the seam (`ridge(0) == ridge(TAU)`) -- otherwise the ring's
/// first and last vertices would differ and leave a visible crack at one
/// bearing.
pub fn ridge(layer: usize, theta: f32) -> f32 {
    // Fixed harmonics with seeded phases: deterministic for every run and
    // independent of `thread_rng` (ADR-0002).
    let mut rng = StdRng::seed_from_u64(0x5EED_0000 + layer as u64);
    let mut sum = 0.0;
    let mut amplitude = 0.55;
    let mut frequency = 2.0;
    for _ in 0..4 {
        let phase = rng.random::<f32>() * std::f32::consts::TAU;
        sum += amplitude * (frequency * theta + phase).sin();
        amplitude *= 0.5;
        frequency *= 2.0;
    }
    // `sum` spans roughly -1.08..1.08; normalise and lift it so valleys stay
    // clearly above the horizon.
    let normalised = (sum / 1.08).clamp(-1.0, 1.0);
    0.35 + 0.65 * (normalised * 0.5 + 0.5)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layers_stack_inside_the_sky_dome() {
        // The cirrus dome sits at 900 m actual mesh radius while its material
        // is culled beyond it; a band at or past the dome would be clipped or
        // would intersect the sphere.
        for (i, layer) in LAYERS.iter().enumerate() {
            assert!(
                layer.radius < 900.0 - 40.0,
                "layer {i} at {} m reaches the 900 m sky dome",
                layer.radius
            );
            if let Some(prev) = i.checked_sub(1) {
                assert!(
                    layer.radius > LAYERS[prev].radius + 40.0,
                    "layers {} and {i} are too close to read as separate ranges",
                    prev
                );
            }
        }
    }

    #[test]
    fn ridge_is_periodic_and_keeps_the_band_above_the_horizon() {
        for layer in 0..LAYERS.len() {
            let start = ridge(layer, 0.0);
            let end = ridge(layer, std::f32::consts::TAU);
            assert!(
                (start - end).abs() < 1e-4,
                "layer {layer} ridge does not close at the seam: {start} vs {end}"
            );
            let mut min = f32::INFINITY;
            let mut max = f32::NEG_INFINITY;
            for i in 0..4096 {
                let theta = i as f32 / 4096.0 * std::f32::consts::TAU;
                let h = ridge(layer, theta);
                min = min.min(h);
                max = max.max(h);
            }
            assert!(
                min >= 0.34,
                "layer {layer} valley {min} dips to the horizon"
            );
            assert!(max <= 1.0, "layer {layer} ridge {max} exceeds the peak");
            assert!(
                max - min > 0.2,
                "layer {layer} is too flat to read as hills"
            );
        }
    }

    #[test]
    fn bands_are_finite_and_fully_indexed() {
        for (i, layer) in LAYERS.iter().enumerate() {
            let mesh = hill_band(i, layer.radius, layer.peak);
            let positions = match mesh.attribute(Mesh::ATTRIBUTE_POSITION) {
                Some(bevy::mesh::VertexAttributeValues::Float32x3(p)) => p.clone(),
                other => panic!("unexpected positions: {other:?}"),
            };
            assert_eq!(positions.len(), (SEGMENTS + 1) * 2);
            assert!(positions.iter().flatten().all(|v| v.is_finite()));
            let Some(Indices::U32(indices)) = mesh.indices() else {
                panic!("expected u32 indices");
            };
            assert_eq!(indices.len(), SEGMENTS * 6);
            assert!(indices.iter().all(|&v| v < positions.len() as u32));
        }
    }
}
