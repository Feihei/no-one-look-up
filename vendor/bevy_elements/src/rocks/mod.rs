//! Rocks on the ground: boulders and stones scattered over the terrain as meshes, so the ground
//! has silhouettes and cast shadows and not only a painted texture
//! (`docs/research/rock-scatter.md`).
//!
//! Where rocks lie comes from the terrain readback (the slope and the sediment), how big they are
//! from Golombek's exponential rock-abundance law, where exactly from a jittered lattice per size
//! class thinned by the local cover, and how far they are drawn from their size. Each rock is
//! seated on the surface the terrain's level-of-detail patches actually draw, and drawn with the
//! crate's [`WeatheredMaterial`] (snow, wetness, cloud shadows) with a stone grain.

mod mesh;
pub(crate) mod model;

use bevy::camera::visibility::VisibilityRange;
use bevy::platform::collections::HashMap;
use bevy::prelude::*;

use crate::plugin::{ElementsQuality, ElementsSystems, ElementsUnits, ElementsView};
use crate::query::EnvironmentSnapshot;
use crate::terrain::{TerrainLook, TerrainRuntime, TerrainSnapshot, TerrainState};
use crate::weathered::{WeatheredExtension, WeatheredMaterial, WeatheredPlugin};
use mesh::{FAR_SUBDIVISIONS, NEAR_SUBDIVISIONS, SHAPES, ShapeInfo, rock_geometry, rock_mesh};
use model::{
    CLASSES, ClassLattice, CoverParams, Lattices, PatchLevels, Placed, TILT_TO_SLOPE, tile_rocks,
};

/// How many rocks lie on the ground and where. Insert it before the plugin to override the
/// defaults; a change rebuilds the scatter.
///
/// The covers are Golombek's `k`: the fraction of the ground covered by rocks of every size
/// (Golombek et al. 2021, `docs/research/rock-scatter.md`). Rocks smaller than 25 cm are not
/// drawn as meshes (the terrain material's scree and turf carry them), so a cover of 0.30 shows
/// as 17 % of the ground under rocks of 25 cm and up.
#[derive(Resource, Clone, Debug, PartialEq, Reflect)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct RockSettings {
    /// Cover of a talus apron: the band of slope at a cliff's foot where the terrain material
    /// draws its scree (16°–40°, densest at 26°). Default 0.30.
    pub talus_cover: f32,
    /// Cover of the meadow where it is rocky (about a quarter of it, in patches a few hundred
    /// metres across, and more of it the steeper the hillside). Default 0.05 (Golombek's rocky
    /// ground, 4–5 %).
    pub meadow_cover: f32,
    /// Cover of the flat meadow away from the rocky patches. Default 0.015 (Golombek's plains,
    /// 1–2 %).
    pub meadow_floor: f32,
    /// A multiplier on how far rocks are drawn ([`ElementsQuality::rock_range`] metres per metre
    /// of a rock's diameter). Default 1; 0 draws none.
    pub range_scale: f32,
    /// Seed of the scatter.
    pub seed: u64,
}

impl Default for RockSettings {
    fn default() -> Self {
        Self {
            talus_cover: 0.30,
            meadow_cover: 0.05,
            meadow_floor: 0.015,
            range_scale: 1.0,
            seed: 0,
        }
    }
}

impl RockSettings {
    fn cover(&self) -> CoverParams {
        CoverParams {
            talus_cover: self.talus_cover.max(0.0),
            meadow_floor: self.meadow_floor.max(0.0),
            meadow_cover: self.meadow_cover.max(0.0),
        }
    }
}

/// One scattered rock: the entity that carries its transform. Its two children draw it (a near
/// and a far mesh, crossfaded by [`VisibilityRange`]); the near child's mesh is kept in the main
/// world, so a game can build a collider from it when this component is added.
#[derive(Component, Clone, Copy, Debug, PartialEq, Reflect)]
pub struct ScatteredRock {
    /// Its diameter in plan, m (the mean of its long and middle axes).
    pub diameter_m: f32,
    /// Its height, m, of which a quarter to a third is below the ground.
    pub height_m: f32,
}

/// How many rocks are on the ground now, and in how many tiles.
#[derive(Resource, Clone, Copy, Debug, Default, PartialEq, Eq, Reflect)]
pub struct RockStats {
    /// Rocks spawned.
    pub rocks: u32,
    /// Tiles spawned, over every size class.
    pub tiles: u32,
}

/// Rocks scattered on the [`Terrain`](crate::Terrain): see [`RockSettings`].
///
/// Needs the [`TerrainPlugin`](crate::TerrainPlugin) and an
/// [`ElementsCamera`](crate::ElementsCamera); adds the [`WeatheredPlugin`] if it is missing.
pub struct RocksPlugin;

impl Plugin for RocksPlugin {
    fn build(&self, app: &mut App) {
        if !app.is_plugin_added::<WeatheredPlugin>() {
            app.add_plugins(WeatheredPlugin);
        }
        app.init_resource::<RockSettings>()
            .init_resource::<RockStats>()
            .init_resource::<RockField>()
            .add_systems(Startup, create_assets)
            .add_systems(
                Update,
                (update_levels, scatter, seat_rocks)
                    .chain()
                    .after(ElementsSystems::Readback),
            )
            .add_systems(Update, grade_rocks);
    }
}

/// The scattered rocks are the crate's own debris on the terrain, so they take the terrain's grade
/// ([`TerrainLook`], `docs/research/terrain-look.md`): written into their shared material only
/// when it differs, so a still look re-prepares nothing.
fn grade_rocks(
    look: Option<Res<TerrainLook>>,
    assets: Option<Res<RockAssets>>,
    materials: Option<ResMut<Assets<WeatheredMaterial>>>,
) {
    let (Some(look), Some(assets), Some(mut materials)) = (look, assets, materials) else {
        return;
    };
    let packed = look.packed();
    if materials
        .get(&assets.material)
        .is_some_and(|m| m.extension.look != packed)
        && let Some(mut material) = materials.get_mut(&assets.material)
    {
        material.extension.look = packed;
    }
}

/// Tiles built per frame at most, nearest first: a camera that jumps fills its rings over a few
/// frames instead of stalling one.
const TILES_PER_FRAME: usize = 24;
/// The camera moves this far, m, before the rocks are seated again (the morph follows it).
const SEAT_STEP_M: f32 = 0.25;
/// A seat that moved less than this, m, is left alone (no change to the transform).
const SEAT_TOLERANCE_M: f32 = 0.001;
/// The near mesh is drawn to this many metres per metre of diameter (Deviation 4).
const NEAR_PER_M: f32 = 22.0;
/// A rock shrinks to nothing over the last fifth of its range, and its two meshes switch
/// abruptly. Bevy's dithered `VisibilityRange` crossfade left black holes in every fading rock
/// under the crate's depth prepass, mask or not (`AGENTS.md`), and a rock a few pixels wide that
/// shrinks away does not pop.
const FADE: f32 = 0.2;
/// Diameter bins per class for the draw ranges: few distinct [`VisibilityRange`]s, so Bevy's
/// table (64 entries where storage buffers are scarce) holds them all.
const RANGE_BINS: usize = 6;
/// The vertical field of view the per-metre range factors are calibrated for, radians: Bevy's own
/// default perspective view (`PerspectiveProjection::default()`). A rock's draw range `c · D` is
/// the distance at which a rock of diameter `D` is a fixed share of the view's height. Under an
/// orthographic view the same share is reached at the distance a perspective camera of this field
/// of view would need, so the two projections agree about what is worth drawing
/// (`docs/research/orthographic-rocks.md` Model 1).
const REFERENCE_FOV: f32 = std::f32::consts::FRAC_PI_3;
/// Largest ring the scatter's magnification may reach under an orthographic view, m: a 4 km
/// region view shows its rocks, but the whole world's are not built for it.
const ORTHO_RING_MAX_M: f32 = 1200.0;

/// How far rocks are drawn this frame.
///
/// A perspective camera draws a rock out to `range_per_m · D` (metres per metre of diameter),
/// which is a constant screen size: at that distance a rock of diameter `D` subtends the view's
/// height over `2 · range_per_m · tan(θ/2)` of it, whatever `D` is. Under an orthographic camera
/// distance says nothing about a rock's size on screen — "objects maintain the same size
/// regardless of distance to the camera" (Akenine-Möller et al. §4.7.1) — so the same screen size
/// is a threshold on the rock's **diameter** instead, and everything within the view is equally
/// far (`docs/research/orthographic-rocks.md` Model 1).
#[derive(Clone, Copy, Debug, PartialEq)]
enum ViewScale {
    /// A perspective camera: draw a rock out to `range_per_m · D`.
    Distance { range_per_m: f32 },
    /// An orthographic camera: draw a rock whose diameter is at least `min_diameter_m` anywhere
    /// within `ring_m` of the view's centre, and the near mesh from `near_diameter_m` up.
    Diameter {
        min_diameter_m: f32,
        near_diameter_m: f32,
        ring_m: f32,
    },
}

impl ViewScale {
    /// The ring a size class spanning up to `class_max_m` of diameter is built out to, m: zero
    /// where the view does not draw that class at all.
    fn ring_radius(&self, class_max_m: f32) -> f32 {
        match *self {
            Self::Distance { range_per_m, .. } => range_per_m * class_max_m,
            Self::Diameter {
                min_diameter_m,
                ring_m,
                ..
            } => {
                if class_max_m >= min_diameter_m {
                    ring_m
                } else {
                    0.0
                }
            }
        }
    }

    /// Whether the view draws a rock of diameter `d` at all.
    fn draws(&self, d: f32) -> bool {
        match *self {
            Self::Distance { range_per_m, .. } => range_per_m > 0.0,
            Self::Diameter { min_diameter_m, .. } => d >= min_diameter_m,
        }
    }

    /// Whether the rock of diameter `d` takes the near mesh.
    fn near_mesh(&self, d: f32) -> bool {
        match *self {
            // Bevy's `VisibilityRange` switches the meshes at `NEAR_PER_M · d` from the camera.
            Self::Distance { .. } => true,
            Self::Diameter {
                near_diameter_m, ..
            } => d >= near_diameter_m,
        }
    }

    /// The discriminant and values a scatter is built for: a change rebuilds it.
    fn key(&self) -> [f32; 4] {
        match *self {
            Self::Distance { range_per_m, .. } => [0.0, range_per_m, 0.0, 0.0],
            Self::Diameter {
                min_diameter_m,
                ring_m,
                ..
            } => [1.0, min_diameter_m, ring_m, 0.0],
        }
    }
}

/// This frame's [`ViewScale`] for a followed camera of `projection`, whose rocks are measured from
/// the frame's centre (the terrain's morph centre: the eye under a perspective view, the focus
/// under an orthographic one) and whose view spans `footprint_m` of ground across.
fn view_scale(
    projection: Option<&Projection>,
    range_per_m: f32,
    footprint_m: f32,
    units: &ElementsUnits,
) -> ViewScale {
    let height_m = match projection {
        Some(Projection::Orthographic(o)) => units.metres(o.area.height()),
        _ => f32::NAN,
    };
    if !(height_m.is_finite() && height_m > 0.0) {
        return ViewScale::Distance { range_per_m };
    }
    // The distance at which a perspective camera of [`REFERENCE_FOV`] spans `height_m`: the
    // screen size a rock of diameter `d` reaches at range `R` is `d / (2 R tan(θ/2))` of the
    // view's height, so the ranges `range_per_m · d` are the diameter thresholds
    // `height_m / (2 · range_per_m · tan(θ/2))`.
    let threshold = 2.0 * (0.5 * REFERENCE_FOV).tan();
    // The ring is where a rock has *gone*: the fade runs over the last [`FADE`] of the range, so
    // the ring must reach past the view's ground by that much or the fade is drawn inside the
    // frame — stones at the corners shrink to nothing while a player watches.
    let ground_m = 0.5 * footprint_m / (1.0 - FADE);
    ViewScale::Diameter {
        min_diameter_m: height_m / (range_per_m * threshold),
        near_diameter_m: height_m / (NEAR_PER_M * threshold),
        ring_m: ground_m.min(ORTHO_RING_MAX_M),
    }
}

#[derive(Resource)]
struct RockAssets {
    /// Per shape: the near and the far mesh.
    meshes: Vec<[Handle<Mesh>; 2]>,
    shapes: Vec<ShapeInfo>,
    material: Handle<WeatheredMaterial>,
}

/// The terrain rock's albedo (`ROCK` in `render.wesl`); the shapes' vertex colours carry the
/// palette's tone.
const ROCK_ALBEDO: [f32; 3] = [0.076, 0.071, 0.066];
/// The stone grain's feature size, m (Parameters).
const GRAIN_M: f32 = 0.30;

fn create_assets(
    mut commands: Commands,
    meshes: Option<ResMut<Assets<Mesh>>>,
    materials: Option<ResMut<Assets<WeatheredMaterial>>>,
) {
    let (Some(mut meshes), Some(mut materials)) = (meshes, materials) else {
        return;
    };
    let mut handles = Vec::with_capacity(SHAPES);
    let mut shapes = Vec::with_capacity(SHAPES);
    for shape in 0..SHAPES {
        let near = rock_geometry(shape, NEAR_SUBDIVISIONS);
        shapes.push(near.info);
        let far = rock_geometry(shape, FAR_SUBDIVISIONS);
        handles.push([meshes.add(rock_mesh(near)), meshes.add(rock_mesh(far))]);
    }
    let material = materials.add(WeatheredMaterial {
        base: StandardMaterial {
            base_color: Color::linear_rgb(ROCK_ALBEDO[0], ROCK_ALBEDO[1], ROCK_ALBEDO[2]),
            perceptual_roughness: 0.9,
            reflectance: 0.35,
            ..default()
        },
        extension: WeatheredExtension::default().with_grain(GRAIN_M),
    });
    commands.insert_resource(RockAssets {
        meshes: handles,
        shapes,
        material,
    });
}

/// What the scatter was built for: a change rebuilds it.
#[derive(Clone, Debug, PartialEq)]
struct FieldKey {
    grid_origin: Vec2,
    grid_cell: f32,
    grid_size: UVec2,
    grid_y0: f32,
    cover: CoverParams,
    seed: u64,
    /// The view's scale: a camera that zooms or a focus that moves to another orthographic view
    /// changes what is drawn, so the scatter is rebuilt.
    view: [f32; 4],
    units: f32,
}

/// The live scatter.
#[derive(Resource, Default)]
struct RockField {
    key: Option<FieldKey>,
    lattices: Vec<ClassLattice>,
    /// Per (class, tile): the tile's entity and its rock count.
    tiles: HashMap<(usize, IVec2), (Entity, u32)>,
    /// The patch levels the terrain draws, for the seat.
    levels: Option<PatchLevels>,
    /// Set when `levels` changed since the last seat.
    levels_changed: bool,
}

/// Where a rock sits, for re-seating it when the camera or the terrain moves.
#[derive(Component, Clone, Copy, Debug)]
struct RockSeat {
    /// Metre-frame XZ.
    xz: Vec2,
    /// The rock's up (its short axis), unit.
    up: Vec3,
    /// From the surface along `up` to the mesh's centre, m.
    lift_m: f32,
    /// The mesh's scale, m (its long axis's length).
    scale_m: f32,
    /// Where it has shrunk away, m from the view's centre.
    far_m: f32,
    /// The two surfaces the patches draw here, the camera's distance morphing between them;
    /// read at the surface `version`, from the patch levels when `patched`.
    lattices: Lattices,
    version: u32,
    patched: bool,
}

/// The tile entity of one class: the parent of its rocks.
#[derive(Component)]
struct RockTile;

/// Mirror the terrain's selected patches into a level per leaf, for the seat.
fn update_levels(
    runtime: Option<Res<TerrainRuntime>>,
    state: Option<Res<TerrainState>>,
    mut field: ResMut<RockField>,
) {
    let levels = runtime.zip(state).and_then(|(runtime, state)| {
        let lod = runtime.lod?;
        if !runtime.draws_ground() {
            return None;
        }
        let leaves = lod.nodes(0);
        let mut level = vec![u8::MAX; (leaves.x * leaves.y) as usize];
        for patch in &runtime.selected {
            let span = 1u32 << patch.level;
            for z in patch.iz * span..((patch.iz + 1) * span).min(leaves.y) {
                for x in patch.ix * span..((patch.ix + 1) * span).min(leaves.x) {
                    level[(z * leaves.x + x) as usize] = patch.level as u8;
                }
            }
        }
        Some(PatchLevels {
            cell_m: state.grid.cell_m,
            // The patches' vertex lattice: half a cell in from the grid's corner.
            origin: state.grid.origin + Vec2::splat(lod.sample_offset()),
            top: lod.levels.saturating_sub(1),
            morph: (0..lod.levels)
                .map(|k| (lod.morph_start_m(k), lod.range_m(k)))
                .collect(),
            leaf_m: lod.leaf_m,
            leaves,
            level,
        })
    });
    if field.levels != levels {
        field.levels = levels;
        field.levels_changed = true;
    }
}

/// The rock's draw range, m: binned per class so few distinct ranges exist.
fn binned_diameter(class: usize, diameter_m: f32) -> f32 {
    let (a, b) = CLASSES[class];
    let t = ((diameter_m / a).ln() / (b / a).ln()).clamp(0.0, 0.999);
    let bin = (t * RANGE_BINS as f32).floor();
    a * (b / a).powf((bin + 0.5) / RANGE_BINS as f32)
}

/// The ground a followed orthographic camera spans across, m: the diagonal of its footprint, its
/// width by the height it covers over the sine of its look-down angle (`precipitation` sizes its
/// box the same way, `docs/research/view-anchor.md` Deviation 4).
fn footprint_m(projection: &Projection, transform: &GlobalTransform, units: &ElementsUnits) -> f32 {
    let Projection::Orthographic(o) = projection else {
        return 0.0;
    };
    let down = (-transform.forward().y).clamp(0.2, 1.0);
    units
        .metres(o.area.width())
        .hypot(units.metres(o.area.height()) / down)
}

/// Build the tiles that entered each class's ring and drop the ones that left it.
#[allow(clippy::too_many_arguments)]
fn scatter(
    mut commands: Commands,
    settings: Res<RockSettings>,
    quality: Res<ElementsQuality>,
    units: Res<ElementsUnits>,
    view: Res<ElementsView>,
    cameras: Query<(&Projection, &GlobalTransform)>,
    snapshot: Res<EnvironmentSnapshot>,
    terrain: Option<Res<TerrainRuntime>>,
    assets: Option<Res<RockAssets>>,
    mut field: ResMut<RockField>,
    mut stats: ResMut<RockStats>,
) {
    let Some(assets) = assets else {
        return;
    };
    let Some(snap) = snapshot.terrain.as_ref() else {
        clear(&mut commands, &mut field, &mut stats);
        field.key = None;
        return;
    };
    if view.entity.is_none() {
        return;
    }
    let cover = settings.cover();
    let range_per_m = quality.rock_range() * settings.range_scale.max(0.0);
    let u = units.per_metre();
    // What the terrain's patches are selected and morphed about: the eye, or an orthographic
    // camera's focus. The rocks fade, ring and seat about it, so a rock is culled by the same
    // centre as the ground it sits on (`docs/research/orthographic-rocks.md` Model 2).
    let centre = terrain.map_or(view.eye, |t| t.morph_centre(view.eye));
    let camera = view.entity.and_then(|e| cameras.get(e).ok());
    let scale = view_scale(
        camera.map(|(projection, _)| projection),
        range_per_m,
        camera.map_or(0.0, |(projection, transform)| {
            footprint_m(projection, transform, &units)
        }),
        &units,
    );
    let key = FieldKey {
        grid_origin: snap.grid.origin,
        grid_cell: snap.grid.cell_m,
        grid_size: snap.grid.size,
        grid_y0: snap.grid.y0,
        cover,
        seed: settings.seed,
        view: scale.key(),
        units: u,
    };
    if field.key.as_ref() != Some(&key) {
        clear(&mut commands, &mut field, &mut stats);
        field.lattices = (0..CLASSES.len())
            .map(|class| {
                ClassLattice::with_radius(
                    class,
                    cover.max_cover(),
                    scale.ring_radius(CLASSES[class].1),
                )
            })
            .collect();
        field.key = Some(key);
    }
    if cover.max_cover() <= 0.0 || range_per_m <= 0.0 {
        return;
    }
    // A class the view does not draw keeps an empty lattice, so nothing is built for it.
    if field.lattices.iter().all(|lattice| lattice.radius_m <= 0.0) {
        return;
    }
    let centre_xz = centre.xz();
    // Drop the tiles that left their ring by more than a tile (the margin keeps a camera on a
    // tile's edge from rebuilding it every frame).
    let lattices = field.lattices.clone();
    field.tiles.retain(|&(class, tile), &mut (entity, rocks)| {
        let lattice = &lattices[class];
        if lattice.tile_distance(tile, centre_xz) > lattice.radius_m + lattice.tile_m() {
            commands.entity(entity).despawn();
            stats.rocks = stats.rocks.saturating_sub(rocks);
            stats.tiles = stats.tiles.saturating_sub(1);
            false
        } else {
            true
        }
    });
    let mut wanted: Vec<(f32, usize, IVec2)> = lattices
        .iter()
        // A class the view does not draw has no ring, and `tiles_near` would still offer the
        // tile under the centre: skipping it here keeps `TILES_PER_FRAME` for real tiles.
        .filter(|lattice| lattice.radius_m > 0.0)
        .flat_map(|lattice| {
            lattice
                .tiles_near(centre_xz, 0.0)
                .into_iter()
                .map(move |tile| (lattice.tile_distance(tile, centre_xz), lattice.class, tile))
        })
        .filter(|(_, class, tile)| !field.tiles.contains_key(&(*class, *tile)))
        .collect();
    wanted.sort_by(|a, b| a.0.total_cmp(&b.0));
    for &(_, class, tile) in wanted.iter().take(TILES_PER_FRAME) {
        let lattice = &lattices[class];
        if lattice.radius_m <= 0.0 {
            continue;
        }
        let rocks = tile_rocks(snap, &cover, lattice, tile, settings.seed);
        let parent = commands
            .spawn((RockTile, Transform::IDENTITY, Visibility::default()))
            .id();
        // Counted as drawn, not as placed: an orthographic view drops the rocks below its
        // diameter threshold, and `RockStats` reports what is on the ground.
        let mut count = 0;
        for rock in &rocks {
            if !scale.draws(rock.diameter_m) {
                continue;
            }
            count += 1;
            let child = spawn_rock(
                &mut commands,
                &assets,
                class,
                rock,
                scale,
                u,
                centre,
                snap,
                field.levels.as_ref(),
            );
            commands.entity(parent).add_child(child);
        }
        field.tiles.insert((class, tile), (parent, count));
        stats.rocks += count;
        stats.tiles += 1;
    }
}

/// Spawn one rock and its two meshes, seated on the surface the patches draw about `centre`
/// (metre frame), drawn as this frame's [`ViewScale`] says (`docs/research/orthographic-rocks.md`).
#[allow(clippy::too_many_arguments)]
fn spawn_rock(
    commands: &mut Commands,
    assets: &RockAssets,
    class: usize,
    rock: &Placed,
    scale: ViewScale,
    units: f32,
    centre: Vec3,
    snap: &TerrainSnapshot,
    levels: Option<&PatchLevels>,
) -> Entity {
    let info = assets.shapes[rock.shape];
    // The diameter in plan is the mean of the long and middle axes.
    let scale_m = 2.0 * rock.diameter_m / (1.0 + info.b);
    let height_m = info.c * scale_m;
    let up = Quat::from_euler(EulerRot::XYZ, rock.jitter.x, 0.0, rock.jitter.y)
        * Vec3::Y.lerp(rock.normal, TILT_TO_SLOPE).normalize();
    let rotation = Quat::from_rotation_arc(Vec3::Y, up) * Quat::from_rotation_y(rock.yaw);
    let d = binned_diameter(class, rock.diameter_m);
    let seat = RockSeat {
        xz: rock.xz,
        up,
        lift_m: height_m * (0.5 - rock.bury),
        scale_m,
        far_m: match scale {
            ViewScale::Distance { range_per_m, .. } => range_per_m * d,
            // An orthographic view draws the whole ring at one distance: the fade is over the
            // last fifth of the ring instead, so a rock still arrives small at the edge.
            ViewScale::Diameter { ring_m, .. } => ring_m,
        },
        lattices: Lattices::at(snap, levels, rock.xz),
        version: snap.version,
        patched: levels.is_some(),
    };
    let surface = seat.lattices.height(levels, centre, rock.xz);
    let (translation, rock_scale) = seat_transform(&seat, surface, centre, units);
    let [near_mesh, far_mesh] = assets.meshes[rock.shape].clone();
    // Bevy's `VisibilityRange` measures from the camera, which under an orthographic view is one
    // distance for the whole scene: the near and far children would take the same verdict and a
    // rock would never switch mesh (Model 4). There the crate picks the mesh itself.
    let (near_range, far_range, near_shown, far_shown) = match scale {
        ViewScale::Distance { .. } => (
            VisibilityRange::abrupt(0.0, NEAR_PER_M * d * units),
            VisibilityRange::abrupt(NEAR_PER_M * d * units, seat.far_m * units),
            Visibility::Inherited,
            Visibility::Inherited,
        ),
        ViewScale::Diameter { .. } => (
            VisibilityRange::abrupt(0.0, f32::MAX),
            VisibilityRange::abrupt(0.0, f32::MAX),
            if scale.near_mesh(d) {
                Visibility::Inherited
            } else {
                Visibility::Hidden
            },
            if scale.near_mesh(d) {
                Visibility::Hidden
            } else {
                Visibility::Inherited
            },
        ),
    };
    commands
        .spawn((
            ScatteredRock {
                diameter_m: rock.diameter_m,
                height_m,
            },
            seat,
            Transform {
                translation,
                rotation,
                scale: Vec3::splat(rock_scale),
            },
            Visibility::default(),
        ))
        .with_children(|r| {
            r.spawn((
                Mesh3d(near_mesh),
                MeshMaterial3d(assets.material.clone()),
                near_range,
                near_shown,
            ));
            r.spawn((
                Mesh3d(far_mesh),
                MeshMaterial3d(assets.material.clone()),
                far_range,
                far_shown,
            ));
        })
        .id()
}

/// The rock's translation and scale (world units) on a surface at `surface_m` (metre frame): it
/// shrinks to nothing over the last fifth of its range, about its buried base, so a shrinking
/// rock stays in the ground. The range is measured from `view` — the frame's own centre, the
/// same one the terrain's patches morph about — not from where the camera stands.
fn seat_transform(seat: &RockSeat, surface_m: f32, view: Vec3, units: f32) -> (Vec3, f32) {
    let ground = Vec3::new(seat.xz.x, surface_m, seat.xz.y);
    // Distance from what the view shows, not from where the camera stands: an isometric camera
    // stands hundreds of metres back from every rock it draws (`docs/research/orthographic-rocks.md`).
    let t =
        ((view.distance(ground) - seat.far_m * (1.0 - FADE)) / (seat.far_m * FADE)).clamp(0.0, 1.0);
    // Never exactly zero: a zero scale has no inverse for the normals.
    let fade = (1.0 - t * t * (3.0 - 2.0 * t)).max(1e-3);
    (
        (ground + seat.up * seat.lift_m * fade) * units,
        seat.scale_m * fade * units,
    )
}

fn clear(commands: &mut Commands, field: &mut RockField, stats: &mut RockStats) {
    for (_, (entity, _)) in field.tiles.drain() {
        commands.entity(entity).despawn();
    }
    *stats = RockStats::default();
}

/// Seat every rock again when the camera or the patches' morph centre has moved (the morph and
/// the rocks' shrink follow them), when the patch levels changed or when the terrain's surface
/// changed.
fn seat_rocks(
    view: Res<ElementsView>,
    units: Res<ElementsUnits>,
    snapshot: Res<EnvironmentSnapshot>,
    terrain: Option<Res<TerrainRuntime>>,
    mut field: ResMut<RockField>,
    mut rocks: Query<(&mut RockSeat, &mut Transform)>,
    mut last: Local<Option<(Vec3, Vec3, u32)>>,
) {
    let Some(snap) = snapshot.terrain.as_ref() else {
        return;
    };
    let eye = view.eye;
    // What the patches morph about: the eye, or an orthographic camera's centre.
    let centre = terrain.map_or(eye, |t| t.morph_centre(eye));
    let surface_changed = last.is_none_or(|(_, _, version)| version != snap.version);
    let moved = last.is_none_or(|(at, about, _)| {
        at.distance(eye) > SEAT_STEP_M || about.distance(centre) > SEAT_STEP_M
    });
    // The morph centre is what a rock fades about, so a moved centre re-fades every rock even
    // when the eye itself has not moved (an orthographic camera walks its focus, not its eye).
    let levels_changed = std::mem::take(&mut field.levels_changed);
    if !moved && !surface_changed && !levels_changed {
        return;
    }
    *last = Some((eye, centre, snap.version));
    let levels = field.levels.as_ref();
    let u = units.per_metre();
    let tolerance = SEAT_TOLERANCE_M * u;
    for (mut seat, mut transform) in &mut rocks {
        // The lattices depend on the surface and the covering patch's level alone: read them
        // again only when either changed, and morph between them every pass.
        let stale = seat.version != snap.version
            || seat.patched != levels.is_some()
            || (levels_changed
                && levels.is_some_and(|l| l.level_at(seat.xz).unwrap_or(0) != seat.lattices.level));
        if stale {
            seat.lattices = Lattices::at(snap, levels, seat.xz);
            seat.version = snap.version;
            seat.patched = levels.is_some();
        }
        let surface = seat.lattices.height(levels, centre, seat.xz);
        let (translation, scale) = seat_transform(&seat, surface, centre, u);
        if transform.translation.distance(translation) > tolerance
            || (transform.scale.x - scale).abs() > tolerance
        {
            transform.translation = translation;
            transform.scale = Vec3::splat(scale);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const U: ElementsUnits = ElementsUnits {
        units_per_metre: 1.0,
    };

    /// An orthographic view spanning `height_m` of ground top to bottom, its area set as Bevy's
    /// `camera_system` sets it from a 16:9 viewport.
    fn orthographic(height_m: f32) -> Projection {
        let mut projection = OrthographicProjection {
            scaling_mode: bevy::camera::ScalingMode::FixedVertical {
                viewport_height: height_m,
            },
            ..OrthographicProjection::default_3d()
        };
        projection.area = bevy::math::Rect::from_center_size(
            Vec2::ZERO,
            Vec2::new(height_m * 16.0 / 9.0, height_m),
        );
        Projection::from(projection)
    }

    #[test]
    fn a_perspective_view_draws_a_rock_at_the_same_range_as_before() {
        // The change is a no-op for a perspective camera: the per-metre range factor, measured
        // from the eye.
        let scale = view_scale(None, 100.0, 0.0, &U);
        assert_eq!(scale, ViewScale::Distance { range_per_m: 100.0 });
        // A 0.25 m stone is drawn to 25 m, a 3.6 m boulder to 360 m: unchanged.
        assert!((scale.ring_radius(CLASSES[0].1) - 50.0).abs() < 1e-4);
        assert!((scale.ring_radius(CLASSES[3].1) - 360.0).abs() < 1e-3);
        assert!(scale.draws(0.25));
        assert!(scale.near_mesh(0.25));
    }

    #[test]
    fn an_orthographic_view_draws_the_same_screen_size_a_perspective_one_would() {
        // A rock is drawn while it covers at least the share of the view's height a perspective
        // camera of the reference field of view gives it at its own range.
        let height_m = 18.0;
        let range_per_m = 100.0;
        let scale = view_scale(Some(&orthographic(height_m)), range_per_m, 40.0, &U);
        let ViewScale::Diameter {
            min_diameter_m,
            near_diameter_m,
            ring_m,
            ..
        } = scale
        else {
            panic!("an orthographic view must take the diameter rule: {scale:?}");
        };
        // The threshold is the diameter that a perspective camera of the reference field of view
        // draws at its own range, as a share of the view's height.
        let tan_half = (0.5 * REFERENCE_FOV).tan();
        let perspective_share = 1.0 / (2.0 * range_per_m * tan_half);
        assert!(
            (min_diameter_m / height_m - perspective_share).abs() < 1e-6,
            "{min_diameter_m} {height_m} {perspective_share}"
        );
        // 18 m over 2·100·tan(30°) is a 16 cm stone: the smallest class (25 cm and up) is drawn,
        // where before every rock was shrunk to nothing at 457 m from the eye.
        assert!((min_diameter_m - 0.1559).abs() < 1e-3, "{min_diameter_m}");
        assert!(scale.draws(CLASSES[0].0));
        assert!(!scale.draws(0.1));
        // The near mesh, for a rock whose facets a game can see: 18 over 2·22·tan(30°) = 0.71 m.
        assert!((near_diameter_m - height_m / (2.0 * NEAR_PER_M * tan_half)).abs() < 1e-5);
        assert!(!scale.near_mesh(0.4));
        assert!(scale.near_mesh(1.0));
        // The ring reaches past the half footprint by the fade, so the shrink happens outside
        // the frame: 20 m of ground is a 25 m ring. Every class is built out to it.
        assert!((ring_m - 25.0).abs() < 1e-4, "{ring_m}");
        assert!((scale.ring_radius(CLASSES[3].1) - 25.0).abs() < 1e-4);
    }

    #[test]
    fn a_wide_orthographic_view_draws_only_the_stones_that_read() {
        // A rock is worth a mesh only while it is a few pixels tall: a 200 m view keeps every
        // class, a 2 km one only the boulders, and a 4 km one none at all (a 3.6 m boulder is
        // 1 px there). The ring is capped, so no view builds the world.
        let ring = |height_m: f32| {
            let scale = view_scale(Some(&orthographic(height_m)), 100.0, 9e3, &U);
            (
                scale.draws(CLASSES[0].0),
                scale.draws(CLASSES[3].1),
                scale.ring_radius(CLASSES[3].1),
            )
        };
        // The threshold is `height / (2 · 100 · tan 30°)` = `height / 115.47`: 22 cm at the
        // isometric 18 m view, so the smallest class (25 cm and up) is drawn; 26 cm at 30 m, so
        // it is not; 1.7 m at 200 m; 17 m at 2 km and 35 m at 4 km, where no class is drawn and
        // the ring is empty.
        assert_eq!(ring(18.0), (true, true, ORTHO_RING_MAX_M));
        assert_eq!(ring(28.0), (true, true, ORTHO_RING_MAX_M));
        assert_eq!(ring(30.0), (false, true, ORTHO_RING_MAX_M));
        assert_eq!(ring(200.0), (false, true, ORTHO_RING_MAX_M));
        assert_eq!(ring(2000.0), (false, false, 0.0));
        assert_eq!(ring(4000.0), (false, false, 0.0));
    }

    #[test]
    fn a_world_in_feet_sizes_the_view_in_metres() {
        // The view's area is in world units; the conversion is the units resource's.
        let feet = ElementsUnits {
            units_per_metre: 3.280_84,
        };
        let metres = view_scale(Some(&orthographic(18.0)), 100.0, 40.0, &U);
        // `footprint_m` is already in metres (the caller converts with the same units).
        let in_feet = view_scale(
            Some(&orthographic(18.0 * 3.280_84)),
            100.0,
            40.0,
            &feet,
        );
        assert_eq!(metres, in_feet);
    }

    #[test]
    fn a_rock_fades_about_the_focus_not_the_eye() {
        // The isometric camera stands 457 m back: measuring from the eye faded every rock to the
        // floor. From the focus, a rock at the focus is at full scale and one at the ring's edge
        // is at the floor.
        let centre = Vec3::new(45.0, 12.0, 52.0);
        let scale = view_scale(Some(&orthographic(18.0)), 100.0, 40.0, &U);
        let seat = |xz: Vec2, far_m: f32| RockSeat {
            xz,
            up: Vec3::Y,
            lift_m: 0.0,
            scale_m: 1.0,
            far_m,
            lattices: Lattices {
                level: 0,
                fine: 12.0,
                coarse: 12.0,
            },
            version: 0,
            patched: false,
        };
        let ViewScale::Diameter { ring_m, .. } = scale else {
            panic!("an orthographic view must take the diameter rule");
        };
        let (_, at_focus) = seat_transform(&seat(centre.xz(), ring_m), 12.0, centre, 1.0);
        let (_, at_edge) = seat_transform(
            &seat(centre.xz() + Vec2::new(ring_m, 0.0), ring_m),
            12.0,
            centre,
            1.0,
        );
        assert!((at_focus - 1.0).abs() < 1e-4, "{at_focus}");
        assert!(at_edge < 2e-3, "{at_edge}");
        // Monotone between: nearer is never smaller.
        let mut last = 0.0;
        for step in 0..=10 {
            let d = ring_m * 0.5 * step as f32 / 10.0;
            let (_, s) = seat_transform(
                &seat(centre.xz() + Vec2::new(d, 0.0), ring_m),
                12.0,
                centre,
                1.0,
            );
            assert!(s >= last - 1e-6, "{s} {last}");
            last = s;
        }
    }
}
