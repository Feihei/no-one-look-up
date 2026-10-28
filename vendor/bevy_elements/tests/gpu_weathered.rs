//! `WeatheredMaterial` on a real GPU: the rendered pixels of a weathered cube beside the same cube
//! in a plain `StandardMaterial` that carries the colour and roughness the note says the weather
//! produces. Both are lit by one directional light and seen by an orthographic camera, so equal
//! inputs give equal pixels.
//!
//! Needs a GPU: `cargo test --test gpu_weathered -- --ignored --test-threads=1`.

mod support;

use std::sync::{Arc, Mutex, MutexGuard};

use bevy::camera::ScalingMode;
use bevy::prelude::*;
use bevy::render::render_resource::{AsBindGroup, BindingType, BufferBindingType, ShaderStages};
use bevy::render::renderer::RenderDevice;
use bevy::render::{RenderApp, RenderStartup};
use bevy_elements::*;

static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());

fn serial() -> MutexGuard<'static, ()> {
    ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner())
}

/// What the test pins the scene's weather to each frame.
#[derive(Resource, Clone, Copy, Default)]
struct Pin {
    wetness: f32,
    snow_cover: f32,
    rain_mm_h: f32,
}

fn pin_weather(
    pin: Res<Pin>,
    mut weathering: ResMut<SurfaceWeathering>,
    mut weather: ResMut<Weather>,
) {
    weathering.wetness = pin.wetness;
    weathering.snow_cover = pin.snow_cover;
    weather.precipitation_mm_h = pin.rain_mm_h;
    weather.temperature_c = 15.0;
}

const SIZE: u32 = 256;
const BASE: Color = Color::srgb(0.6, 0.4, 0.2);

struct Rig {
    app: App,
    left: Handle<WeatheredMaterial>,
    right: Handle<StandardMaterial>,
    camera: Entity,
    image: Handle<Image>,
}

/// Two 6 m cubes, 8 m apart, the left weathered and the right plain, seen from straight above
/// or from the front by an orthographic camera 16 m tall.
fn rig() -> Rig {
    rig_with(false)
}

/// The same rig, rendered by Bevy's deferred path when `deferred` (the extension supplies its own
/// deferred fragment shader).
fn rig_with(deferred: bool) -> Rig {
    let mut app = App::new();
    if deferred {
        app.insert_resource(bevy::pbr::DefaultOpaqueRendererMethod::deferred());
    }
    app.add_plugins((support::headless_plugins(), WeatheredPlugin))
        .init_resource::<SurfaceWeathering>()
        .init_resource::<Pin>()
        .add_systems(Update, pin_weather.in_set(ElementsSystems::Weather));
    support::finish(&mut app);

    let base = StandardMaterial {
        base_color: BASE,
        perceptual_roughness: 0.9,
        ..default()
    };
    let (left, right, camera, image, cube) = {
        let world = app.world_mut();
        let left = world
            .resource_mut::<Assets<WeatheredMaterial>>()
            .add(WeatheredMaterial {
                base: base.clone(),
                extension: WeatheredExtension::default(),
            });
        let right = world.resource_mut::<Assets<StandardMaterial>>().add(base);
        let cube = world
            .resource_mut::<Assets<Mesh>>()
            .add(Cuboid::new(6.0, 6.0, 6.0));
        let (target, image) = {
            let mut images = world.resource_mut::<Assets<Image>>();
            let mut target = Image::new_target_texture(
                SIZE,
                SIZE,
                bevy::render::render_resource::TextureFormat::Rgba8UnormSrgb,
                None,
            );
            // The readback copies out of the texture.
            target.texture_descriptor.usage |=
                bevy::render::render_resource::TextureUsages::COPY_SRC;
            let handle = images.add(target);
            (
                bevy::camera::RenderTarget::Image(handle.clone().into()),
                handle,
            )
        };
        let camera = world
            .spawn((
                Camera3d::default(),
                bevy::core_pipeline::tonemapping::DebandDither::Disabled,
                target,
                Projection::from(OrthographicProjection {
                    scaling_mode: ScalingMode::FixedVertical {
                        viewport_height: 16.0,
                    },
                    ..OrthographicProjection::default_3d()
                }),
                Transform::from_xyz(0.0, 30.0, 0.0).looking_at(Vec3::ZERO, Vec3::NEG_Z),
            ))
            .id();
        if deferred {
            world.entity_mut(camera).insert((
                bevy::core_pipeline::prepass::DepthPrepass,
                bevy::core_pipeline::prepass::DeferredPrepass,
                Msaa::Off,
            ));
        }
        world.spawn((
            DirectionalLight {
                illuminance: 2500.0,
                ..default()
            },
            Transform::from_xyz(-1.0, 3.0, 2.0).looking_at(Vec3::ZERO, Vec3::Y),
        ));
        (left, right, camera, image, cube)
    };
    let world = app.world_mut();
    world.spawn((
        Mesh3d(cube.clone()),
        MeshMaterial3d(left.clone()),
        Transform::from_xyz(-4.0, 3.0, 0.0),
    ));
    world.spawn((
        Mesh3d(cube),
        MeshMaterial3d(right.clone()),
        Transform::from_xyz(4.0, 3.0, 0.0),
    ));
    Rig {
        app,
        left,
        right,
        camera,
        image,
    }
}

impl Rig {
    /// From +Z: the cubes' walls, `n_y = 0` (the rig starts looking straight down at their tops).
    fn look_at_the_walls(&mut self) {
        let transform =
            Transform::from_xyz(0.0, 3.0, 30.0).looking_at(Vec3::new(0.0, 3.0, 0.0), Vec3::Y);
        *self
            .app
            .world_mut()
            .get_mut::<Transform>(self.camera)
            .expect("the camera") = transform;
    }

    fn set_weather(&mut self, wetness: f32, snow_cover: f32, rain_mm_h: f32) {
        *self.app.world_mut().resource_mut::<Pin>() = Pin {
            wetness,
            snow_cover,
            rain_mm_h,
        };
    }

    fn set_exposure(&mut self, exposure: f32) {
        let mut materials = self
            .app
            .world_mut()
            .resource_mut::<Assets<WeatheredMaterial>>();
        materials
            .get_mut(&self.left)
            .expect("the weathered material")
            .extension
            .exposure = exposure;
    }

    /// The plain cube's material: the colour and roughness the weather should have produced.
    fn set_plain(&mut self, color: Color, roughness: f32) {
        let mut materials = self
            .app
            .world_mut()
            .resource_mut::<Assets<StandardMaterial>>();
        let mut m = materials.get_mut(&self.right).expect("the plain material");
        m.base_color = color;
        m.perceptual_roughness = roughness;
    }

    /// Render (a few frames so every change has reached the GPU) and read the image back.
    fn render(&mut self) -> Vec<u8> {
        for _ in 0..6 {
            self.app.update();
        }
        support::read_texture(
            &mut self.app,
            self.image.clone(),
            SIZE as usize * 4,
            SIZE as usize,
        )
    }
}

fn pixel(image: &[u8], x: usize, y: usize) -> [u8; 3] {
    let i = (y * SIZE as usize + x) * 4;
    [image[i], image[i + 1], image[i + 2]]
}

/// The centre of each cube's face: (left, right).
fn centres(image: &[u8]) -> ([u8; 3], [u8; 3]) {
    (pixel(image, 64, 128), pixel(image, 192, 128))
}

fn max_diff(a: [u8; 3], b: [u8; 3]) -> u8 {
    (0..3).map(|c| a[c].abs_diff(b[c])).max().unwrap_or(0)
}

fn done(rig: Rig) {
    let mut app = rig.app;
    app.world_mut().write_message(bevy::app::AppExit::Success);
    app.update();
}

fn snow_color() -> Color {
    Color::from(SNOW_ALBEDO)
}

/// Wetness Validation 9 (plan M4), on the faces that face up: dry is a plain `StandardMaterial`
/// bit for bit; full snow is the snow albedo at roughness 0.9; full wetness is the albedo × 0.5 at
/// roughness 0.1; and `exposure` 0 shelters from both.
#[test]
#[ignore = "needs a GPU"]
fn weathered_tops_match_the_plain_material_with_the_notes_numbers() {
    let _lock = serial();
    let mut rig = rig();
    let base = BASE.to_linear();

    // Dry: identical to the plain material, to the bit.
    let (l, r) = centres(&rig.render());
    assert_eq!(l, r, "dry weathered pixel {l:?} vs plain {r:?}");

    // Snow: the plain cube in snow, and brighter than the dry one.
    rig.set_weather(0.0, 1.0, 0.0);
    rig.set_plain(snow_color(), 0.9);
    let (snowy, plain_snow) = centres(&rig.render());
    assert!(
        max_diff(snowy, plain_snow) <= 1,
        "snow: weathered {snowy:?} vs plain {plain_snow:?}"
    );
    assert!(snowy[0] > l[0] + 20, "snow is brighter: {snowy:?} vs {l:?}");

    // Half snow cover on an up-facing face: the colour is half way, in linear light.
    rig.set_weather(0.0, 0.5, 0.0);
    let half = Color::from(LinearRgba::new(
        0.5 * (base.red + SNOW_ALBEDO.red),
        0.5 * (base.green + SNOW_ALBEDO.green),
        0.5 * (base.blue + SNOW_ALBEDO.blue),
        1.0,
    ));
    rig.set_plain(half, 0.9);
    let (mixed, plain_mixed) = centres(&rig.render());
    assert!(
        max_diff(mixed, plain_mixed) <= 1,
        "half snow: weathered {mixed:?} vs plain {plain_mixed:?}"
    );

    // Sheltered: snow does not reach it.
    rig.set_weather(0.0, 1.0, 0.0);
    rig.set_exposure(0.0);
    rig.set_plain(BASE, 0.9);
    let (sheltered, plain) = centres(&rig.render());
    assert_eq!(sheltered, plain, "exposure 0 under snow");

    // Wet: albedo × 0.5 (porosity 1 at roughness 0.9) and roughness 0.1.
    rig.set_exposure(1.0);
    rig.set_weather(1.0, 0.0, 0.0);
    rig.set_plain(
        Color::from(LinearRgba::new(
            base.red * 0.5,
            base.green * 0.5,
            base.blue * 0.5,
            1.0,
        )),
        0.1,
    );
    let (wet, plain_wet) = centres(&rig.render());
    assert!(
        max_diff(wet, plain_wet) <= 1,
        "wet: weathered {wet:?} vs plain {plain_wet:?}"
    );
    assert!(wet[0] < l[0], "wet is darker: {wet:?} vs {l:?}");

    // Sheltered from the rain too.
    rig.set_exposure(0.0);
    rig.set_plain(BASE, 0.9);
    let (sheltered, plain) = centres(&rig.render());
    assert_eq!(sheltered, plain, "exposure 0 under rain");
    println!("dry {l:?} snow {snowy:?} half {mixed:?} wet {wet:?}");
    done(rig);
}

/// The deferred pipeline runs the extension's own deferred fragment (the base shader would drop
/// the weather): snow and wetness reach the G-buffer, `exposure` 0 shelters, dry is plain.
#[test]
#[ignore = "needs a GPU"]
fn deferred_tops_match_the_plain_material() {
    let _lock = serial();
    let mut rig = rig_with(true);
    let base = BASE.to_linear();

    let (l, r) = centres(&rig.render());
    assert!(max_diff(l, r) <= 1, "deferred dry {l:?} vs plain {r:?}");

    rig.set_weather(0.0, 1.0, 0.0);
    rig.set_plain(snow_color(), 0.9);
    let (snowy, plain_snow) = centres(&rig.render());
    assert!(
        max_diff(snowy, plain_snow) <= 2,
        "deferred snow: {snowy:?} vs {plain_snow:?}"
    );
    assert!(snowy[0] > l[0] + 20, "snow is brighter: {snowy:?} vs {l:?}");

    rig.set_weather(1.0, 0.0, 0.0);
    rig.set_plain(
        Color::from(LinearRgba::new(
            base.red * 0.5,
            base.green * 0.5,
            base.blue * 0.5,
            1.0,
        )),
        0.1,
    );
    let (wet, plain_wet) = centres(&rig.render());
    assert!(
        max_diff(wet, plain_wet) <= 2,
        "deferred wet: {wet:?} vs {plain_wet:?}"
    );
    assert!(wet[0] < l[0], "wet is darker: {wet:?} vs {l:?}");

    rig.set_exposure(0.0);
    rig.set_plain(BASE, 0.9);
    let (sheltered, plain) = centres(&rig.render());
    assert!(max_diff(sheltered, plain) <= 1, "deferred exposure 0");
    println!("deferred dry {l:?} snow {snowy:?} wet {wet:?}");
    done(rig);
}

/// Wetness Validation 9, on the faces that do not: rain draws streaks down a wall (some pixels
/// darker than the plain material, none lighter), and snow leaves the wall bare.
#[test]
#[ignore = "needs a GPU"]
fn walls_streak_in_rain_and_stay_bare_in_snow() {
    let _lock = serial();
    let mut rig = rig();
    rig.look_at_the_walls();

    let region = |image: &[u8]| -> Vec<([u8; 3], [u8; 3])> {
        let mut out = Vec::new();
        for y in 105..185 {
            for x in 0..70 {
                out.push((pixel(image, 30 + x, y), pixel(image, 158 + x, y)));
            }
        }
        out
    };

    // Dry: the same pixels.
    let dry = region(&rig.render());
    assert!(
        dry.iter().all(|(l, r)| l == r),
        "a dry wall is the plain wall"
    );

    // Snow: a vertical face stays bare (the flora curve is 0 at n_y = 0).
    rig.set_weather(0.0, 1.0, 0.0);
    let snow = region(&rig.render());
    assert!(
        snow.iter().all(|(l, r)| l == r),
        "a wall under snow stays bare"
    );

    // Rain at 25 mm/h (intensity 1) with dry ground: rivulets only. A rivulet darkens the wall
    // (the sum of the channels falls by 4 levels or more); the gloss it brings adds at most a
    // level to one channel, so nothing is brighter.
    rig.set_weather(0.0, 0.0, 25.0);
    let rain = region(&rig.render());
    let sum = |p: [u8; 3]| p.iter().map(|&c| i32::from(c)).sum::<i32>();
    let darker = rain.iter().filter(|(l, r)| sum(*l) + 4 <= sum(*r)).count();
    let lighter = rain.iter().filter(|(l, r)| sum(*l) >= sum(*r) + 4).count();
    println!(
        "streaks: {darker} of {} wall pixels darker by 4+ levels in sum, {lighter} lighter",
        rain.len()
    );
    assert!(
        darker > rain.len() / 100,
        "streaks are visible: {darker} pixels"
    );
    assert!(
        darker < rain.len() * 9 / 10,
        "streaks are not a wash: {darker} pixels"
    );
    assert_eq!(lighter, 0, "rain never brightens a wall");

    // Sheltered: no streaks.
    rig.set_exposure(0.0);
    let sheltered = region(&rig.render());
    assert!(
        sheltered.iter().all(|(l, r)| l == r),
        "exposure 0 draws no streaks"
    );
    done(rig);
}

/// Wetness Validation 10 on the real layout: `WeatheredExtension` declares exactly one storage
/// buffer, at 101, visible to the fragment stage only.
#[test]
#[ignore = "needs a GPU"]
fn the_bind_group_layout_has_one_fragment_storage_buffer_at_101() {
    let _lock = serial();
    type Found = Arc<Mutex<Option<Vec<(u32, ShaderStages)>>>>;
    #[derive(Resource, Clone)]
    struct Report(Found);

    fn inspect(device: Res<RenderDevice>, report: Res<Report>) {
        let storage = <WeatheredExtension as AsBindGroup>::bind_group_layout_entries(&device, true)
            .into_iter()
            .filter(|e| {
                matches!(
                    e.ty,
                    BindingType::Buffer {
                        ty: BufferBindingType::Storage { .. },
                        ..
                    }
                )
            })
            .map(|e| (e.binding, e.visibility))
            .collect();
        *report.0.lock().unwrap_or_else(|e| e.into_inner()) = Some(storage);
    }

    let found: Found = Arc::default();
    let mut app = App::new();
    app.add_plugins((support::headless_plugins(), WeatheredPlugin));
    app.sub_app_mut(RenderApp)
        .insert_resource(Report(found.clone()))
        .add_systems(RenderStartup, inspect);
    support::finish(&mut app);
    for _ in 0..4 {
        app.update();
    }
    let storage = found
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .clone()
        .expect("the render world ran its startup");
    assert_eq!(storage, vec![(101, ShaderStages::FRAGMENT)]);
    app.world_mut().write_message(bevy::app::AppExit::Success);
    app.update();
}
