//! CPU checks of the weathered shading against the numbers of `docs/research/wetness.md`
//! (Validation 6, 7, 10) and `docs/research/snow-cover.md` (Validation 2), through the
//! transliteration of `surface_cover.wesl` in `mirror.rs`.

use bevy::math::{Vec2, Vec3};

use super::mirror::{self, Surface};
use super::model;
use super::{ElementsGlobals, SNOW_ALBEDO, SNOW_ROUGHNESS};
use crate::climate::Weather;

fn close(a: f32, b: f32, tolerance: f32, what: &str) {
    assert!((a - b).abs() <= tolerance, "{what}: {a} vs {b}");
}

fn surface(base: Vec3, roughness: f32, metallic: f32, wet: f32, puddle: f32, snow: f32) -> Surface {
    mirror::weather_surface(
        base,
        roughness,
        metallic,
        Vec3::splat(0.5),
        Vec3::Y,
        wet,
        puddle,
        snow,
        Vec2::ZERO,
    )
}

/// Snow-cover Validation 2: the flora curve, its albedo and roughness are flora's.
#[test]
fn snow_on_meshes_follows_the_flora_curve() {
    for (ny, expected) in [
        (1.0, 0.5),
        (0.9, 0.5),
        (0.8, 0.5),
        (0.7, 0.422),
        (0.6, 0.078),
        (0.5, 0.0),
    ] {
        close(
            mirror::mesh_snow_cover(0.5, ny),
            expected,
            1e-3,
            &format!("snow 0.5 at n_y {ny}"),
        );
    }
    close(
        mirror::mesh_snow_cover(1.0, 0.45),
        1.0,
        1e-6,
        "snow 1 at n_y 0.45",
    );
    for ny in [-1.0, 0.0, 0.5, 1.0] {
        assert_eq!(mirror::mesh_snow_cover(0.0, ny), 0.0, "snow 0 at {ny}");
    }
    // `bevy_flora` common.wesl:388 and :424, bit for bit.
    assert_eq!(
        (SNOW_ALBEDO.red, SNOW_ALBEDO.green, SNOW_ALBEDO.blue),
        (0.80, 0.83, 0.88)
    );
    assert_eq!(mirror::SNOW_ALBEDO, Vec3::new(0.80, 0.83, 0.88));
    assert_eq!(SNOW_ROUGHNESS, 0.9);
    assert_eq!(mirror::SNOW_ROUGHNESS, SNOW_ROUGHNESS);
}

/// Wetness Validation 6: porosity from roughness, the albedo and roughness numbers, the puddle's
/// `F0`, snow over wetness, and the dry identity.
#[test]
fn the_shading_numbers_are_the_notes() {
    let base = Vec3::new(0.6, 0.4, 0.2);
    // Porosity: 0 at r = 0.5, 0.5 at 0.7, 1 at 0.9; metal is never porous.
    close(mirror::porosity(0.5, 0.0), 0.0, 1e-6, "φ(0.5)");
    close(mirror::porosity(0.7, 0.0), 0.5, 1e-6, "φ(0.7)");
    close(mirror::porosity(0.9, 0.0), 1.0, 1e-6, "φ(0.9)");
    assert_eq!(mirror::porosity(0.9, 1.0), 0.0);

    // w = 1, φ = 1: albedo × 0.5, roughness min(r, 0.1).
    let s = surface(base, 0.9, 0.0, 1.0, 0.0, 0.0);
    for (got, want) in [
        (s.base_color.x, 0.3),
        (s.base_color.y, 0.2),
        (s.base_color.z, 0.1),
    ] {
        close(got, want, 1e-6, "albedo × 0.5");
    }
    close(s.roughness, 0.1, 1e-6, "wet roughness");
    // A surface smoother than 0.1 stays as smooth as it was.
    close(
        surface(base, 0.05, 0.0, 1.0, 0.0, 0.0).roughness,
        0.05,
        1e-6,
        "never rougher",
    );
    // w = 0.5, φ = 0.5 (r = 0.7): × 0.875.
    let s = surface(base, 0.7, 0.0, 0.5, 0.0, 0.0);
    close(s.base_color.x / base.x, 0.875, 1e-6, "× 0.875");
    // φ = 0 (r = 0.5): the albedo is unchanged, only the gloss moves.
    let s = surface(base, 0.5, 0.0, 1.0, 0.0, 0.0);
    assert_eq!(s.base_color, base);
    close(s.roughness, 0.1, 1e-6, "glossy even when not porous");

    // A puddle: roughness 0.02 and Bevy's F0 = 0.16 · reflectance² = 0.02.
    let s = surface(base, 0.9, 0.0, 0.0, 1.0, 0.0);
    close(
        s.roughness,
        mirror::PUDDLE_ROUGHNESS,
        1e-6,
        "puddle roughness",
    );
    close(s.reflectance.x, 0.354, 1e-3, "puddle reflectance");
    close(
        0.16 * s.reflectance.x * s.reflectance.x,
        0.02,
        1e-4,
        "puddle F0",
    );
    // A puddle shows the ground beneath it darkened with w_e = 1.
    close(s.base_color.x, 0.3, 1e-6, "puddle albedo");
    assert_eq!(s.normal, Vec3::Y, "a level, unrippled puddle");

    // Snow wins: the colour is SNOW_ALBEDO and the roughness 0.9 whatever the wetness.
    for wet in [0.0, 0.5, 1.0] {
        for puddle in [0.0, 1.0] {
            let s = surface(base, 0.3, 1.0, wet, puddle, 1.0);
            assert_eq!(
                s.base_color,
                mirror::SNOW_ALBEDO,
                "wet {wet} puddle {puddle}"
            );
            assert_eq!(s.roughness, 0.9);
            assert_eq!(s.metallic, 0.0, "snow covers metal");
        }
    }

    // Dry, no snow: every input comes back bit for bit.
    let reflectance = Vec3::new(0.5, 0.5, 0.5);
    let n = Vec3::new(0.3, 0.9, -0.1).normalize();
    let s = mirror::weather_surface(
        base,
        0.37,
        0.2,
        reflectance,
        n,
        0.0,
        0.0,
        0.0,
        Vec2::new(0.2, 0.1),
    );
    assert_eq!(
        s,
        Surface {
            base_color: base,
            roughness: 0.37,
            metallic: 0.2,
            reflectance,
            normal: n,
        }
    );
    // And the cloud-shadow scale is exactly 1 without clouds.
    assert_eq!(mirror::cloud_shadow_ratio(1.0, 1.0), 1.0);
}

/// Wetness Validation 6, "documentation test, not a gate": the water-film model the plan's 0.5 is
/// compared with. `ρ_w = (1 − r_e)(1 − r_i) ρ / (1 − r_i ρ)`, `r_e` = 0.0659, `r_i = 1 − (1 −
/// r_e) / n²`, n = 1.33.
#[test]
fn the_film_model_table_in_the_note_is_reproduced() {
    let (re, n) = (0.0659_f32, 1.33_f32);
    let ri = 1.0 - (1.0 - re) / (n * n);
    close(ri, 0.4720, 5e-4, "r_i");
    for (rho, ratio) in [
        (0.05, 0.505),
        (0.1, 0.518),
        (0.2, 0.545),
        (0.3, 0.575),
        (0.5, 0.646),
        (0.8, 0.792),
    ] {
        let wet = (1.0 - re) * (1.0 - ri) * rho / (1.0 - ri * rho);
        close(wet / rho, ratio, 1.5e-3, &format!("ρ = {rho}"));
    }
}

/// Wetness Validation 7: layer weights, the ring amplitude and the phases.
#[test]
fn ripples_follow_lagardes_layers() {
    for (f, expected) in [
        (0.2, [0.8, 0.0, 0.0, 0.0]),
        (0.5, [1.0, 1.0, 0.0, 0.0]),
        (1.0, [1.0, 1.0, 1.0, 1.0]),
        (0.0, [0.0; 4]),
    ] {
        let w = mirror::ripple_weights(f);
        for k in 0..4 {
            close(w[k], expected[k], 1e-6, &format!("f {f} layer {k}"));
        }
    }

    // No weight, no ring; with full weight one layer never exceeds the normal strength 0.35.
    assert_eq!(
        mirror::ripple_layer(Vec2::new(0.1, 0.1), 0, 0.3, 0.0),
        Vec2::ZERO
    );
    let mut peak = 0.0_f32;
    for k in 0..4u32 {
        for phase in (0..20).map(|i| i as f32 / 20.0) {
            for ix in 0..120 {
                for iz in 0..120 {
                    let xz = Vec2::new(ix as f32, iz as f32) * 0.0173;
                    peak = peak.max(mirror::ripple_layer(xz, k, phase, 1.0).length());
                }
            }
        }
    }
    assert!(peak <= mirror::RIPPLE_STRENGTH + 1e-6, "peak {peak}");
    assert!(peak > 0.15, "a ring is visible: peak {peak}");
    println!("peak ripple normal offset of one layer: {peak:.4} (strength 0.35)");

    // The four layers add.
    let (phases, weights) = ([0.1, 0.4, 0.7, 0.9], mirror::ripple_weights(1.0));
    let xz = Vec2::new(0.13, 0.29);
    let sum: Vec2 = (0..4)
        .map(|k| mirror::ripple_layer(xz, k as u32, phases[k], weights[k]))
        .sum();
    assert_eq!(mirror::ripple_offset(xz, phases, weights), sum);

    // Phases are f64 values, continuous across an hour wrap and a day wrap, and each layer's
    // period is 1 / (1.6 · TimeMul).
    for t0 in [3599.99, 86_399.99, 1.0e6] {
        let (a, b) = (model::ripple_phases(t0), model::ripple_phases(t0 + 0.02));
        for k in 0..4 {
            let step = (b[k] - a[k]).rem_euclid(1.0);
            let expected = (0.02 * model::RIPPLE_TIME_MUL[k] * model::RIPPLE_RATE) as f32;
            close(step, expected, 2e-3, &format!("t {t0} layer {k}"));
        }
    }
    close(
        model::ripple_phases(0.0)[1],
        (0.2 * 1.6_f32).fract(),
        1e-6,
        "TimeAdd",
    );
}

/// Wetness Model 6 and Deviation 9: streaks live on vertical faces, scale with the rain up to the
/// 4 mm/h that makes a wall stream fully, stay under 0.4 and wrap seamlessly.
#[test]
fn rain_streaks_run_down_walls_only_while_raining() {
    let wall = Vec3::X;
    let mut peak = 0.0_f32;
    let mut lit = 0;
    for ix in 0..200 {
        for iy in 0..200 {
            let p = Vec3::new(3.0, iy as f32 * 0.05, ix as f32 * 0.011);
            let s = mirror::rain_streak(p, wall, 1.0, 12.5);
            peak = peak.max(s);
            lit += usize::from(s > 0.05);
            // Half the saturating intensity has half the streak; any more rain, the same.
            close(
                mirror::rain_streak(p, wall, 0.1, 12.5),
                0.5 * s,
                1e-6,
                "linear in f below 0.2",
            );
            assert_eq!(
                mirror::rain_streak(p, wall, 0.2, 12.5),
                s,
                "saturated at 0.2"
            );
            assert_eq!(mirror::rain_streak(p, wall, 0.0, 12.5), 0.0);
            // A roof or a floor gets none.
            assert_eq!(mirror::rain_streak(p, Vec3::Y, 1.0, 12.5), 0.0);
        }
    }
    println!("streak coverage s>0.05: {} of 40000, peak {peak}", lit);
    assert!(peak <= 0.4 + 1e-6, "peak {peak}");
    assert!(peak > 0.15, "a rivulet is visible: peak {peak}");
    assert!(
        lit > 0 && lit < 200 * 200 / 2,
        "streaks cover part of a wall: {lit}"
    );
    // Wrapping the phase at 8192 s moves the noise by exactly its lattice period (up to f32
    // rounding of the coordinate).
    for i in 0..50 {
        let p = Vec3::new(1.0 + 0.37 * i as f32, 0.9 + 0.11 * i as f32, 2.0);
        let before = mirror::rain_streak(p, wall, 1.0, 8191.9);
        let after = mirror::rain_streak(p, wall, 1.0, 8191.9 - model::STREAK_WRAP_S as f32);
        close(before, after, 6e-3, "streak wrap");
    }
    close(
        model::streak_phase(8192.0 + 5.0),
        5.0,
        1e-4,
        "streak phase wraps",
    );
}

/// The rain intensity is the particle fraction `min(1, R_liquid / 20)`.
#[test]
fn rain_intensity_is_the_liquid_fraction() {
    let weather = |mm_h, t| Weather {
        precipitation_mm_h: mm_h,
        temperature_c: t,
        ..Weather::default()
    };
    close(
        model::rain_intensity(&weather(4.0, 15.0)),
        0.2,
        1e-6,
        "Rain",
    );
    close(
        model::rain_intensity(&weather(25.0, 15.0)),
        1.0,
        1e-6,
        "Storm",
    );
    assert_eq!(model::rain_intensity(&weather(1.5, -5.0)), 0.0, "Snow");
    // Mixed phase: at 0.5 °C half of 10 mm/h falls as rain.
    close(
        model::rain_intensity(&weather(10.0, 0.5)),
        0.25,
        1e-6,
        "sleet",
    );
}

/// The buffer the shader reads is `surface_cover::ElementsGlobals`, five `vec4<f32>`, field for
/// field.
#[test]
fn the_globals_pack_as_the_shader_struct() {
    let globals = ElementsGlobals {
        wetness: 0.1,
        snow_cover: 0.2,
        rain_intensity: 0.3,
        camera_transmittance: 0.4,
        shadow_origin: Vec2::new(5.0, 6.0),
        shadow_extent: 7.0,
        shadow_enabled: true,
        ripple_phase: [8.0, 9.0, 10.0, 11.0],
        wind: Vec2::new(12.0, 13.0),
        streak_phase: 14.0,
        light_direction: Vec3::new(15.0, 16.0, 17.0),
    };
    assert_eq!(
        globals.pack(),
        [
            [0.1, 0.2, 0.3, 0.4],
            [5.0, 6.0, 7.0, 1.0],
            [8.0, 9.0, 10.0, 11.0],
            [12.0, 13.0, 14.0, 0.0],
            [15.0, 16.0, 17.0, 0.0],
        ]
    );
    let source = include_str!("../surface_cover.wesl");
    let body = &source[source.find("struct ElementsGlobals").expect("the struct")..];
    let body = &body[..body.find("\n}").expect("its end")];
    assert_eq!(body.matches("vec4<f32>,").count(), 5, "five vec4 fields");
    // Without clouds the camera's transmittance is 1 and no map is bound.
    let none = ElementsGlobals::gather(None, &Weather::default(), None, 0.0);
    assert_eq!(none.camera_transmittance, 1.0);
    assert!(!none.shadow_enabled);
}

/// Wetness Validation 10 (declaration level; `tests/gpu_weathered.rs` checks the real layout):
/// `WeatheredExtension` binds exactly one storage buffer, at slot 101, visible to the fragment
/// stage only, so the fragment stage's budget of storage buffers keeps its two spare slots
/// (Bevy uses five of the WebGPU default eight).
#[test]
fn the_extension_binds_one_fragment_storage_buffer_at_101() {
    let source = include_str!("mod.rs");
    let body = &source[source
        .find("pub struct WeatheredExtension")
        .expect("the struct")..];
    let body = &body[..body.find("\n}").expect("the end of the struct")];
    assert_eq!(body.matches("#[storage(").count(), 1);
    assert!(
        body.contains("#[storage(101, read_only, visibility(fragment))]"),
        "{body}"
    );
    assert!(body.contains("#[uniform(100)]"));
}

/// Deviation 9: a rivulet on a soaked wall is drenched, 1.4 of film, which darkens a porous
/// surface to ×0.3 (Lagarde's 0.2–0.3) and cannot gloss past the film's roughness.
#[test]
fn a_drenched_surface_is_darker_than_a_wet_one() {
    let base = Vec3::new(0.6, 0.4, 0.2);
    let s = surface(base, 0.9, 0.0, mirror::DRENCHED_MAX, 0.0, 0.0);
    close(s.base_color.x / base.x, 0.3, 1e-6, "drenched albedo ×0.3");
    close(s.roughness, 0.1, 1e-6, "drenched roughness");
    // Past the cap nothing changes.
    assert_eq!(surface(base, 0.9, 0.0, 3.0, 0.0, 0.0), s);
}
