//! Season system plugin (R3.5, ADR-0004 §6).
//!
//! The parameter table lives in [`crate::world::season`] as pure data; this
//! module owns the GPU side and the run-time state:
//!
//! - [`SeasonUniform`] is the material uniform the tree vertex shader reads
//!   (foliage tint, leaf density, sway strength/speed/phase length, frame
//!   time). It sits on the ONE shared tree material, so switching seasons
//!   rewrites 48 bytes instead of walking every tree entity — which is what
//!   ADR-0004's "no per-material traversal, zero-cost season switch" asks for,
//!   and it comes for free here because every tree in the world already shares
//!   a single `Handle<Material>` (R3.9).
//! - [`TreeSeasonMaterial`] is `StandardMaterial` plus that uniform, wired as
//!   an `ExtendedMaterial` whose vertex stage is `assets/shaders/tree_sway.wesl`
//!   and whose fragment stage stays Bevy's built-in PBR shader.
//! - Dev builds cycle seasons with the 1-4 keys (ADR-0004 §6: "development
//!   shortcut ... replaced by the menu once it exists"). The menu picks the
//!   season at run start and a save carries it, so the keys only ever run
//!   outside a run, or they would break "the season is fixed for the run".
//!
//! Ordering matters in one place: the uniform is pushed to the GPU in
//! `Update` after the player and the sky have run, so a season change and the
//! frame that shows it agree.

use bevy::pbr::{ExtendedMaterial, MaterialExtension};
use bevy::prelude::*;
use bevy::render::render_resource::{AsBindGroup, ShaderType};
use bevy::shader::ShaderRef;

use crate::core::game_state::GamePhase;
use crate::core::input::{
    DEV_SEASON_AUTUMN, DEV_SEASON_SPRING, DEV_SEASON_SUMMER, DEV_SEASON_WINTER,
};
use crate::world::season::{SWAY_PHASE_LEN, SWAY_SPEED, Season, SeasonState};
use crate::world::tree_mesh::SILHOUETTE_BASE_COLOR_SRGB;

/// The shader that both the forward and the prepass vertex stage use.
const TREE_SHADER: &str = "shaders/tree_sway.wesl";

/// Uniform the tree vertex shader reads.
///
/// Layout mirrors `SeasonParams` in `assets/shaders/tree_sway.wesl`. Two
/// `vec4`s rather than named scalars because WGSL uniforms are 16-byte aligned:
/// packing the frame time into `sway.w` is what keeps the struct 32 bytes and
/// the material usable on every adapter.
#[derive(Clone, Copy, Debug, ShaderType, PartialEq)]
pub struct SeasonUniform {
    /// rgb: foliage tint (linear, unit luminance). a: leaf density.
    pub tint_density: Vec4,
    /// x: sway strength. y: sway speed. z: phase length. w: frame time.
    pub sway: Vec4,
}

impl Default for SeasonUniform {
    fn default() -> Self {
        Self::from_season(Season::default(), 0.0)
    }
}

impl SeasonUniform {
    /// Builds the uniform for `season` at `time` seconds.
    pub fn from_season(season: Season, time: f32) -> Self {
        let params = season.params();
        let tint = params.foliage_tint();
        Self {
            tint_density: Vec4::new(tint.x, tint.y, tint.z, params.shader_density()),
            sway: Vec4::new(params.shader_sway(), SWAY_SPEED, SWAY_PHASE_LEN, time),
        }
    }

    /// True when two uniforms differ in anything but the clock, i.e. when the
    /// season itself changed.
    ///
    /// The time in `w` changes every frame, so comparing whole uniforms would
    /// report "changed" 60 times a second and defeat the point of a shared
    /// material.
    pub fn season_differs(&self, other: &Self) -> bool {
        self.tint_density != other.tint_density || self.sway.truncate() != other.sway.truncate()
    }
}

/// `StandardMaterial` extended with the season uniform and the tree vertex
/// shader.
///
/// The base material keeps the near-black silhouette (ADR-0001) and the
/// extension supplies only the hue; the PBR fragment stage multiplies them, so
/// no season can brighten a tree past the silhouette's value.
pub type TreeSeasonMaterial = ExtendedMaterial<StandardMaterial, TreeSeasonExtension>;

/// The extension half of [`TreeSeasonMaterial`].
#[derive(Asset, AsBindGroup, TypePath, Debug, Clone, Default)]
pub struct TreeSeasonExtension {
    /// Binding 100: slots 0-99 belong to the base `StandardMaterial`, and the
    /// prepass binds this same group at the same index.
    #[uniform(100)]
    pub season: SeasonUniform,
}

impl MaterialExtension for TreeSeasonExtension {
    fn vertex_shader() -> ShaderRef {
        TREE_SHADER.into()
    }

    /// Deliberately `ShaderRef::Default`: the fragment stage stays Bevy's PBR
    /// shader. Replacing it would mean re-implementing lighting, distance fog,
    /// tonemapping, decals and the OIT paths, and every one of those would be a
    /// place for the trees to diverge from the rest of the scene.
    fn fragment_shader() -> ShaderRef {
        ShaderRef::Default
    }

    /// The prepass vertex stage must run the same displacement as the forward
    /// stage, or the main pass (which depth-tests `GreaterEqual` against
    /// prepass depth) discards whatever swayed away from the camera.
    fn prepass_vertex_shader() -> ShaderRef {
        TREE_SHADER.into()
    }

    /// No prepass fragment stage: the trees are opaque and never discard, so
    /// the default depth-only prepass is exactly right. Returning a shader here
    /// would also disable Bevy's depth-only batching optimisation
    /// (`prepass_reads_material`).
    fn prepass_fragment_shader() -> ShaderRef {
        ShaderRef::Default
    }

    /// Shadows are cast by the tree's real geometry; the shadow pass runs the
    /// prepass vertex stage, so it inherits the sway automatically.
    fn enable_shadows() -> bool {
        true
    }
}

/// The base `StandardMaterial` for a tree: the silhouette from ADR-0001.
pub fn tree_base_material() -> StandardMaterial {
    StandardMaterial {
        base_color: Color::srgb(
            SILHOUETTE_BASE_COLOR_SRGB[0],
            SILHOUETTE_BASE_COLOR_SRGB[1],
            SILHOUETTE_BASE_COLOR_SRGB[2],
        ),
        perceptual_roughness: 1.0,
        metallic: 0.0,
        // No specular at all: a rim highlight would read as wet plastic and
        // break the cut-out look the whole design rests on.
        reflectance: 0.0,
        ..default()
    }
}

pub struct SeasonPlugin;

impl Plugin for SeasonPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(MaterialPlugin::<TreeSeasonMaterial>::default())
            .init_resource::<SeasonState>()
            .add_systems(
                Update,
                (
                    dev_keys.in_set(SeasonSystems::Input),
                    // Explicit ordering: a season change and the frame that
                    // shows it must agree. Without this the uniform could be
                    // pushed before the key is read and the switch would land
                    // one frame late.
                    push_uniform.after(SeasonSystems::Input),
                ),
            );
    }
}

/// System sets so the dev keys land before the uniform reaches the GPU.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SeasonSystems {
    /// Season changes from the keyboard.
    Input,
}

/// Dev-only season cycling (ADR-0004 §6, R3.7).
///
/// Gated on `!is_game_active()`: ADR-0001 makes the season a property of the
/// run, so changing it mid-walk would break the invariant the save format and
/// the world were built around. Outside a run the keys are free, which is where
/// tuning happens — the same place the sky panel lives (F5).
pub(crate) fn dev_keys(
    keys: Res<ButtonInput<KeyCode>>,
    phase: Res<GamePhase>,
    mut state: ResMut<SeasonState>,
) {
    if phase.is_game_active() {
        return;
    }
    let picked = [
        (DEV_SEASON_SPRING, Season::Spring),
        (DEV_SEASON_SUMMER, Season::Summer),
        (DEV_SEASON_AUTUMN, Season::Autumn),
        (DEV_SEASON_WINTER, Season::Winter),
    ]
    .into_iter()
    .find(|(key, _)| keys.just_pressed(*key))
    .map(|(_, season)| season);
    let Some(season) = picked else {
        return;
    };
    if state.season == season {
        return;
    }
    state.season = season;
    info!("season -> {}", season.label());
}

/// Copies the season state (and the frame clock) into every tree material.
///
/// This is the whole "zero-cost season switch": one `uniform` write on one
/// material asset. Bevy re-uploads that material's bind group and every tree in
/// the streaming window follows, because they all share the handle.
fn push_uniform(
    state: Res<SeasonState>,
    time: Res<Time>,
    mut materials: ResMut<Assets<TreeSeasonMaterial>>,
) {
    // The clock changes every frame, so the uniform is only rewritten when the
    // season changed or the stored value is stale. Writing it unconditionally
    // would dirty the asset (and re-extract it to the render world) 60 times a
    // second for a value that is usually identical.
    let now = time.elapsed_secs();
    for (_, material) in materials.iter_mut() {
        let next = SeasonUniform::from_season(state.season, now);
        if material.extension.season == next {
            continue;
        }
        material.extension.season = next;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::season::SILHOUETTE_LUMINANCE;

    #[test]
    fn uniform_is_32_bytes_and_16_byte_aligned() {
        // WGSL uniform structs are 16-byte aligned and padded to a multiple of
        // 16. A mismatch here fails at pipeline creation (a runtime error, not
        // a compile error), so it is asserted rather than assumed. `Vec4`'s
        // alignment is what provides it — a struct of eight bare `f32`s would
        // be 32 bytes but only 4-byte aligned, and the bind group would reject
        // it.
        assert_eq!(std::mem::size_of::<SeasonUniform>(), 32);
        assert_eq!(std::mem::align_of::<SeasonUniform>(), 16);
        // Field order must match `SeasonParams` in the shader: a swap would
        // compile and then quietly feed density into the tint's alpha.
        assert_eq!(std::mem::offset_of!(SeasonUniform, tint_density), 0);
        assert_eq!(std::mem::offset_of!(SeasonUniform, sway), 16);
    }

    #[test]
    fn uniform_carries_the_season_table() {
        for season in Season::ALL {
            let uniform = SeasonUniform::from_season(season, 12.5);
            let params = season.params();
            let tint = params.foliage_tint();
            assert_eq!(uniform.tint_density.x, tint.x);
            assert_eq!(uniform.tint_density.y, tint.y);
            assert_eq!(uniform.tint_density.z, tint.z);
            assert_eq!(uniform.tint_density.w, params.leaf_density);
            assert_eq!(uniform.sway.x, params.sway_strength);
            assert_eq!(uniform.sway.y, SWAY_SPEED);
            assert_eq!(uniform.sway.z, SWAY_PHASE_LEN);
            assert_eq!(uniform.sway.w, 12.5, "the clock must reach the shader");
        }
    }

    #[test]
    fn season_differs_ignores_the_clock() {
        // The whole point of the predicate: `w` changes every frame, and a
        // "changed" test that looked at it would dirty the material 60 times a
        // second and re-extract it to the render world.
        let a = SeasonUniform::from_season(Season::Autumn, 1.0);
        let b = SeasonUniform::from_season(Season::Autumn, 99.0);
        assert!(!a.season_differs(&b), "a clock tick is not a season change");

        let c = SeasonUniform::from_season(Season::Winter, 1.0);
        assert!(a.season_differs(&c), "autumn vs winter must differ");
    }

    #[test]
    fn winter_collapses_foliage_and_keeps_wood() {
        // Density 0 is what the shader's collapse reads; if it were clamped
        // somewhere else the tree would keep its leaves in winter.
        let uniform = SeasonUniform::from_season(Season::Winter, 0.0);
        assert_eq!(uniform.tint_density.w, 0.0);
        let summer = SeasonUniform::from_season(Season::Summer, 0.0);
        assert_eq!(summer.tint_density.w, 1.0);
    }

    #[test]
    fn every_season_sends_a_real_tint() {
        // A zero tint would multiply the crown to black; the shader relies on
        // the uniform being unit-luminance, and nothing downstream re-checks it.
        for season in Season::ALL {
            let uniform = SeasonUniform::from_season(season, 0.0);
            let tint = uniform.tint_density.truncate();
            let luminance = 0.2126 * tint.x + 0.7152 * tint.y + 0.0722 * tint.z;
            assert!(
                (luminance - 1.0).abs() < 1e-4,
                "{} sends tint luminance {luminance}",
                season.label()
            );
        }
    }

    /// Builds an app with the asset stores the season systems write to.
    ///
    /// `MinimalPlugins` supplies `Time` and the schedule; the two asset
    /// resources are what `build_placement_assets` insists on in a headless
    /// test, and `Assets<TreeSeasonMaterial>` is where `push_uniform` writes.
    fn app_with_material() -> App {
        let mut app = App::new();
        app.add_plugins(MinimalPlugins)
            .init_resource::<SeasonState>()
            .init_resource::<GamePhase>()
            .init_resource::<ButtonInput<KeyCode>>()
            .init_resource::<Assets<TreeSeasonMaterial>>()
            .add_systems(
                Update,
                (
                    dev_keys.in_set(SeasonSystems::Input),
                    // Explicit ordering: a season change and the frame that
                    // shows it must agree. Without this the uniform could be
                    // pushed before the key is read and the switch would land
                    // one frame late.
                    push_uniform.after(SeasonSystems::Input),
                ),
            );
        app
    }

    #[test]
    fn switching_the_season_rewrites_the_shared_material() {
        // The whole "zero-cost switch" claim rests on this path: one resource
        // change -> one uniform write -> every tree in the window follows,
        // because they all share the handle. A test of the pure functions
        // alone would pass while the wiring did nothing.
        let mut app = app_with_material();
        let handle = app
            .world_mut()
            .resource_mut::<Assets<TreeSeasonMaterial>>()
            .add(TreeSeasonMaterial {
                base: tree_base_material(),
                extension: TreeSeasonExtension::default(),
            });

        // Summer (the default) after a frame.
        app.update();
        let summer = {
            let materials = app.world().resource::<Assets<TreeSeasonMaterial>>();
            materials.get(&handle).unwrap().extension.season
        };
        assert_eq!(summer.tint_density.w, 1.0, "summer density");

        // Switch to winter and run one frame.
        app.world_mut().resource_mut::<SeasonState>().season = Season::Winter;
        app.update();
        let winter = {
            let materials = app.world().resource::<Assets<TreeSeasonMaterial>>();
            materials.get(&handle).unwrap().extension.season
        };
        assert_eq!(winter.tint_density.w, 0.0, "winter must clear the leaves");
        assert!(
            winter.tint_density.truncate() != summer.tint_density.truncate(),
            "winter must also send a different tint"
        );
    }

    #[test]
    fn the_uniform_carries_the_frame_clock_so_the_sway_animates() {
        // The sway phase is driven by the clock inside the material uniform,
        // so a stale clock means frozen trees. This also pins the honest cost
        // of the design: the uniform is rewritten once per frame (32 bytes),
        // not once per tree -- the saving is the traversal, not the upload.
        let mut app = app_with_material();
        let handle = app
            .world_mut()
            .resource_mut::<Assets<TreeSeasonMaterial>>()
            .add(TreeSeasonMaterial {
                base: tree_base_material(),
                extension: TreeSeasonExtension::default(),
            });
        app.update();
        let first = app
            .world()
            .resource::<Assets<TreeSeasonMaterial>>()
            .get(&handle)
            .unwrap()
            .extension
            .season
            .sway
            .w;

        app.world_mut()
            .resource_mut::<Time>()
            .advance_by(std::time::Duration::from_millis(500));
        app.update();
        let second = app
            .world()
            .resource::<Assets<TreeSeasonMaterial>>()
            .get(&handle)
            .unwrap()
            .extension
            .season
            .sway
            .w;

        assert!(
            second > first,
            "the shader clock did not advance ({first} -> {second}); the crowns would be frozen"
        );
    }

    /// The tree shader source, for the tripwires below.
    fn tree_shader_source() -> String {
        std::fs::read_to_string("assets/shaders/tree_sway.wesl")
            .expect("the tree shader must be in the repo (assets/shaders is tracked)")
    }

    #[test]
    fn the_shader_runs_the_same_displacement_in_both_stages() {
        // The failure this guards against is silent and ugly: if only the
        // forward stage swayed, leaves moving away from the camera would be
        // behind prepass depth and discarded (`GreaterEqual`) in patches.
        //
        // A single shader FILE is what guarantees agreement (the prepass entry
        // and the forward entry are the same function), so the assertion is
        // that the file wires the prepass stage to itself rather than to a
        // separate copy.
        let src = tree_shader_source();
        assert!(
            src.contains("PREPASS_PIPELINE"),
            "the shader no longer branches on the pipeline"
        );
        // One `@vertex` entry only: two would mean a divergent copy.
        assert_eq!(
            src.matches("@vertex").count(),
            1,
            "the shader must have exactly one vertex entry point"
        );
        // And the prepass must be told to use it.
        let plugin = std::fs::read_to_string("src/plugins/season.rs").unwrap();
        assert!(
            plugin.contains("fn prepass_vertex_shader() -> ShaderRef {"),
            "the material must override the prepass vertex stage"
        );
    }

    #[test]
    fn the_shader_declares_the_uniform_the_rust_side_sends() {
        // `AsBindGroup` and the WGSL declaration are two independent
        // descriptions of the same buffer. If they drift, the failure is a
        // pipeline creation error at run time, not a compile error.
        let src = tree_shader_source();
        assert!(
            src.contains("@group(3) @binding(100)"),
            "the season uniform moved; the Rust side still writes group 3 binding 100"
        );
        assert!(
            src.contains("tint_density: vec4<f32>"),
            "uniform field order must match SeasonUniform"
        );
        assert!(
            src.contains("sway: vec4<f32>"),
            "uniform field order must match SeasonUniform"
        );
        // Field order matters: the struct is read positionally.
        let tint = src.find("tint_density: vec4<f32>").unwrap();
        let sway = src.find("sway: vec4<f32>").unwrap();
        assert!(tint < sway, "tint_density must be declared before sway");
    }

    #[test]
    fn the_shader_collapses_leaves_but_never_wood() {
        // Winter must strip the crown and leave the tree standing. If the
        // collapse stopped checking the foliage mask it would shrink the trunk
        // to a point and the world would lose its trees every winter.
        let src = tree_shader_source();
        assert!(
            src.contains("vertex_color.g <= 0.5"),
            "the foliage mask gate disappeared; wood would collapse with the leaves"
        );
        assert!(
            src.contains("season.tint_density.a - 1.0"),
            "the density collapse must scale the offset by (density - 1)"
        );
    }

    #[test]
    fn the_shader_uses_the_material_clock_not_globals() {
        // `globals.time` sits at a different binding in the prepass (1) than in
        // the forward pass (11), and one file cannot declare both. Using it
        // would break one of the two stages; the material uniform is the only
        // clock both can read.
        let src = tree_shader_source();
        // Check the CODE, not the prose: the header explains why `globals` is
        // unusable and would make a naive text search fail on its own comment.
        let code: String = src
            .lines()
            .filter(|line| !line.trim_start().starts_with("//"))
            .collect::<Vec<_>>()
            .join(
                "
",
            );
        assert!(
            !code.contains("globals"),
            "globals is bound differently per pipeline; the clock must ride in the material uniform"
        );
        assert!(
            code.contains("season.sway.w"),
            "the sway clock must come from the material uniform"
        );
    }

    #[test]
    fn base_material_is_the_silhouette() {
        // ADR-0001: the tree is a shape against the sky. The season tint only
        // modulates this value, so if the base drifted brighter every season
        // would drift with it.
        let material = tree_base_material();
        let Color::Srgba(base) = material.base_color else {
            panic!("base colour must be sRGB to match the shipped material");
        };
        for (actual, expected) in [
            (base.red, SILHOUETTE_BASE_COLOR_SRGB[0]),
            (base.green, SILHOUETTE_BASE_COLOR_SRGB[1]),
            (base.blue, SILHOUETTE_BASE_COLOR_SRGB[2]),
        ] {
            assert!((actual - expected).abs() < 1e-6);
        }
        assert_eq!(material.metallic, 0.0);
        assert_eq!(material.perceptual_roughness, 1.0);
        assert_eq!(material.reflectance, 0.0);

        // And the table's normalisation constant must agree with it, or the
        // tint would be normalised against a tree that does not exist.
        let linear = |c: f32| {
            if c <= 0.04045 {
                c / 12.92
            } else {
                ((c + 0.055) / 1.055).powf(2.4)
            }
        };
        let luminance =
            0.2126 * linear(base.red) + 0.7152 * linear(base.green) + 0.0722 * linear(base.blue);
        assert!(
            (luminance - SILHOUETTE_LUMINANCE).abs() < 1e-4,
            "SILHOUETTE_LUMINANCE {SILHOUETTE_LUMINANCE} vs material {luminance}"
        );
    }
}
