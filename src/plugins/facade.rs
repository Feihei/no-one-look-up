//! Facade material (R3.11 step 11.2): `StandardMaterial` plus the metric UV
//! parameters the facade vertex stage needs.
//!
//! # One material per (style, variant, wear bucket)
//!
//! The whole city's walls are `3 styles x 2 variants x 4 wear buckets` = 24
//! materials. That number is the point: with one material per building the
//! renderer would batch nothing and a streaming window would cost thousands of
//! draw calls. Twenty-four is small enough to batch by the hundred and large
//! enough that two neighbours rarely share a pattern.
//!
//! # What lives in the base `StandardMaterial` and what lives here
//!
//! The base keeps everything the built-in PBR path understands: the three maps
//! (albedo, metallic/roughness, emissive), the emissive luminance, the style
//! tint and the wear grade. The extension adds exactly one `vec4` — bay width,
//! floor height, bays per tile, floors per tile — because those are the numbers
//! the vertex stage divides by and no `StandardMaterial` field carries them.
//!
//! Keeping the split this way means the fragment stage is Bevy's own `pbr.wesl`:
//! lighting, the environment map that makes the glass reflect, distance fog and
//! tonemapping are not reimplemented. That is the same trade the tree material
//! made, and it is why the glass can reflect the dusk sky at all — the sky is an
//! `AtmosphereEnvironmentMapLight` that only the built-in PBR path reads.

use bevy::pbr::{ExtendedMaterial, MaterialExtension};
use bevy::prelude::*;
use bevy::render::render_resource::{AsBindGroup, ShaderType};
use bevy::shader::ShaderRef;

use crate::world::facade::{BAYS_PER_TILE, FLOORS_PER_TILE, FacadeStyle};

/// The vertex stage that turns local positions into metric facade UVs.
const FACADE_SHADER: &str = "shaders/facade_uv.wesl";

/// How bright a lit window is, as a material emissive multiplier.
///
/// `StandardMaterial::emissive` multiplies the emissive texture, so this is the
/// single "how awake is the city" knob. 3.0 was picked against the sky: at the
/// golden-hour exposure (ev100 13, ACES) a lit pane has to clear the wall it sits
/// in but stay well under the sky, or the windows become the brightest thing on
/// screen and the silhouette inverts. Asserted in `a_lit_window_is_visible_but_
/// dimmer_than_the_sky` so a "let's make it prettier" tweak cannot silently
/// break ADR-0001.
pub const WINDOW_EMISSIVE: f32 = 3.0;

/// Uniform the facade vertex stage reads.
///
/// One `vec4` and not four scalars: WGSL uniforms are 16-byte aligned, so four
/// separate `f32` fields would need 12 bytes of padding anyway.
#[derive(Clone, Copy, Debug, ShaderType, PartialEq)]
pub struct FacadeParams {
    /// x: bay width (m). y: floor height (m).
    /// z: bays per texture tile. w: floors per texture tile.
    pub dims: Vec4,
}

impl Default for FacadeParams {
    fn default() -> Self {
        Self::of(FacadeStyle::OldCity)
    }
}

impl FacadeParams {
    /// Parameters for a style. The tile counts come from the baker, so the
    /// shader and the texture cannot disagree about how many bays a tile holds.
    pub fn of(style: FacadeStyle) -> Self {
        let dims = style.dims();
        Self {
            dims: Vec4::new(
                dims.bay,
                dims.floor,
                BAYS_PER_TILE as f32,
                FLOORS_PER_TILE as f32,
            ),
        }
    }
}

/// `StandardMaterial` extended with [`FacadeParams`].
pub type FacadeMaterial = ExtendedMaterial<StandardMaterial, FacadeExtension>;

/// The extension half of [`FacadeMaterial`].
#[derive(Asset, AsBindGroup, TypePath, Debug, Clone, Default)]
pub struct FacadeExtension {
    /// Binding 100: 0-99 belong to the base `StandardMaterial`.
    #[uniform(100)]
    pub params: FacadeParams,
}

impl MaterialExtension for FacadeExtension {
    fn vertex_shader() -> ShaderRef {
        FACADE_SHADER.into()
    }

    /// `ShaderRef::Default` on purpose: the fragment stage stays Bevy's PBR
    /// shader, so the glass reflects the sky's IBL and everything else in the
    /// scene keeps working without a second lighting model.
    fn fragment_shader() -> ShaderRef {
        ShaderRef::Default
    }

    /// The vertex stage moves nothing, so the default prepass positions already
    /// agree with it. Returning a shader here would also defeat Bevy's
    /// depth-only prepass batching for no benefit.
    fn prepass_vertex_shader() -> ShaderRef {
        ShaderRef::Default
    }

    fn prepass_fragment_shader() -> ShaderRef {
        ShaderRef::Default
    }

    /// Walls cast the shadows that make the street read; the sun is 2-3 degrees
    /// up at this hour, so they are long.
    fn enable_shadows() -> bool {
        true
    }
}

/// Builds one facade material.
///
/// `tint` is the style colour after the wear grade has been applied, exactly as
/// the BoxMesh era computed it: the maps carry the *pattern* and stay neutral,
/// the tint carries the style, so wear keeps working unchanged.
///
/// # `perceptual_roughness` and `metallic` are 1.0 here, and that is not a bug
/// The maps store absolute values in G (roughness) and B (metallic), and Bevy
/// *multiplies* the scalars into them. Leaving the defaults (0.5 / 0.0) would
/// halve every roughness and zero every metallic, which is how the glass ends
/// up matte and the corrugated steel ends up plastic.
pub fn facade_material(
    style: FacadeStyle,
    albedo: Handle<Image>,
    metallic_roughness: Handle<Image>,
    emissive: Handle<Image>,
    tint: [f32; 3],
) -> FacadeMaterial {
    FacadeMaterial {
        base: StandardMaterial {
            base_color_texture: Some(albedo),
            // The tint is a multiplier on the map, so the plaster stays pale and
            // the brick keeps its own value; only the hue shifts per style.
            base_color: Color::srgb(tint[0], tint[1], tint[2]),
            metallic_roughness_texture: Some(metallic_roughness),
            metallic: 1.0,
            perceptual_roughness: 1.0,
            emissive_texture: Some(emissive),
            // Linear, because `emissive` is not an sRGB colour field.
            emissive: LinearRgba::rgb(WINDOW_EMISSIVE, WINDOW_EMISSIVE, WINDOW_EMISSIVE),
            // Glass needs a specular response: reflectance is the dielectric F0
            // and 0.5 is Bevy's physically-based default (4%). Setting it to 0
            // the way the trees do would kill the reflection this whole step is
            // for.
            reflectance: 0.5,
            // No normal map (see `world::facade`): the mesh has no tangents and
            // a scaled cube cannot grow them cleanly.
            ..default()
        },
        extension: FacadeExtension {
            params: FacadeParams::of(style),
        },
    }
}

/// Registers the facade material type with the renderer.
///
/// `MaterialPlugin` needs the render sub-app, so it cannot live in
/// [`crate::world::chunks::WorldPlugin`]: the headless tests build the world with
/// `MinimalPlugins` and no renderer at all. They only need `Assets<FacadeMaterial>`
/// to exist, which `WorldPlugin` provides separately. Splitting the two is the
/// same arrangement the season material uses.
pub struct FacadePlugin;

impl Plugin for FacadePlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(MaterialPlugin::<FacadeMaterial>::default());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn params_are_16_bytes_and_match_the_baker() {
        // A mismatch here fails at pipeline creation, which is a runtime error
        // and therefore a test not a comment.
        assert_eq!(std::mem::size_of::<FacadeParams>(), 16);
        assert_eq!(std::mem::align_of::<FacadeParams>(), 16);
        for style in FacadeStyle::ALL {
            let params = FacadeParams::of(style);
            let dims = style.dims();
            assert_eq!(params.dims.x, dims.bay);
            assert_eq!(params.dims.y, dims.floor);
            // The shader divides `u` by these, so they must be the baker's
            // numbers or every window lands in the wrong place.
            assert_eq!(params.dims.z, BAYS_PER_TILE as f32);
            assert_eq!(params.dims.w, FLOORS_PER_TILE as f32);
        }
    }

    #[test]
    fn the_shader_reads_the_same_binding_the_material_writes() {
        // The binding is a number in two languages (Rust attribute and WGSL
        // `@binding`). It is the only thing that can go wrong silently here: a
        // wrong binding still compiles and then reads garbage or nothing.
        let shader = include_str!("../../assets/shaders/facade_uv.wesl");
        assert!(
            shader.contains("@group(3) @binding(100)"),
            "the facade uniform must sit at the material group's binding 100"
        );
        assert!(
            shader.contains("dims: vec4<f32>"),
            "the uniform field must be one vec4 named `dims`"
        );
        assert!(
            shader.contains("facade_uv"),
            "the UV function is what the rest of the file is for"
        );
    }

    #[test]
    fn the_vertex_stage_does_not_reach_for_the_fragment_stage() {
        // Keeping the fragment stage default is the whole reason the glass
        // reflects the sky. If a future edit gives this material a fragment
        // shader, it has to reimplement lighting to keep up -- so the decision
        // is asserted, not left to memory.
        let shader = include_str!("../../assets/shaders/facade_uv.wesl");
        assert!(
            !shader.contains("@fragment"),
            "no fragment entry point here"
        );
        assert!(
            !shader.contains("pbr_functions::apply_pbr_lighting"),
            "lighting stays in Bevy's fragment stage"
        );
    }

    #[test]
    fn a_lit_window_is_visible_but_dimmer_than_the_sky() {
        // The dusk budget: a window has to clear its own wall, and must stay
        // under the sky or the sky stops being the subject (ADR-0001).
        assert!(
            (1.0..=6.0).contains(&WINDOW_EMISSIVE),
            "window emissive {WINDOW_EMISSIVE} is outside the dusk window budget"
        );
    }

    #[test]
    fn glass_keeps_a_specular_response() {
        // `reflectance: 0.0` is the tree recipe (a rim highlight reads as wet
        // plastic on a silhouette). Copying it here would be invisible in a
        // unit test and would silently delete the reflection.
        let m = facade_material_for_test();
        assert!(
            m.base.reflectance > 0.0,
            "a facade with no specular cannot reflect the sky"
        );
        assert_eq!(m.base.metallic, 1.0, "the metal map must pass through");
        assert_eq!(
            m.base.perceptual_roughness, 1.0,
            "the roughness map must pass through"
        );
        assert!(m.base.emissive_texture.is_some());
    }

    /// The material built with placeholder handles; only the scalar fields are
    /// under test here.
    fn facade_material_for_test() -> FacadeMaterial {
        facade_material(
            FacadeStyle::Apartment,
            Handle::default(),
            Handle::default(),
            Handle::default(),
            [0.5, 0.5, 0.6],
        )
    }
}
