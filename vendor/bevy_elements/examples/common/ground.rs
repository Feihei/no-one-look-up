//! The examples' ground plane in the crate's own [`WeatheredMaterial`]: per-pixel cloud shadows,
//! wet darkening and gloss when it rains, rain streaks, and up-facing snow when it snows. A
//! consumer's own ground does the same (or, with a `Terrain` simulating the surface and the
//! cover maps' snow wanted, `ExtendedMaterial<StandardMaterial, GroundExtension>`).
//!
//! [`test_scene`](super::test_scene) spawns the plane with a plain `StandardMaterial` — one
//! material asset type per example — and this plugin swaps it for the weathered one on the first
//! frame.

use bevy::prelude::*;
use bevy_elements::{WeatheredExtension, WeatheredMaterial};

/// Marks the plane [`test_scene`](super::test_scene) spawns; the harness gives it the weathered
/// material.
#[derive(Component)]
pub struct GroundPlane;

/// Marks the showcase's boxes, columns and spheres; they take the weathered material with the
/// plane, so the overcast dome shade (and the rain and snow) reaches their vertical faces too.
#[derive(Component)]
pub struct ShowcaseProp;

/// The harness's re-materialled entities: the plane or a showcase prop, still on their plain
/// material.
type Weatherable<'w, 's> = Query<
    'w,
    's,
    (Entity, &'static MeshMaterial3d<StandardMaterial>),
    Or<(With<GroundPlane>, With<ShowcaseProp>)>,
>;

pub fn plugin(app: &mut App) {
    app.add_systems(Update, adopt_weathered);
}

fn adopt_weathered(
    mut commands: Commands,
    standard: Res<Assets<StandardMaterial>>,
    mut grounds: ResMut<Assets<WeatheredMaterial>>,
    props: Weatherable,
) {
    for (entity, material) in &props {
        let Some(base) = standard.get(&material.0) else {
            continue;
        };
        commands
            .entity(entity)
            .remove::<(GroundPlane, ShowcaseProp, MeshMaterial3d<StandardMaterial>)>()
            .insert(MeshMaterial3d(grounds.add(WeatheredMaterial {
                base: base.clone(),
                extension: WeatheredExtension::default(),
            })));
    }
}
