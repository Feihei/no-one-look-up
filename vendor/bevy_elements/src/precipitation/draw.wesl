// Precipitation draw pass: rain streaks, snowflakes, dust sprites and splash rings as procedural
// quads (no vertex buffers), blended premultiplied into the HDR main texture after the main pass
// and soft faded against the depth prepass. See docs/research/precipitation.md ("Rendering
// geometry", "Coverage-preserving alpha", "Lighting", "Fades") and docs/research/dust.md
// ("Particles").
//
// Streaks and flakes are screen-space capsules from the particle's head to its tail
// p − (v − v_cam)·T. A dust sprite is drawn as a flake of the dust's albedo. Splashes are
// horizontal quads on the ground.

import bevy_render::view::View;
import bevy_elements::precipitation::common::{particle_rand, drop_diameter_mm};

struct Particle {
    pos: vec3<f32>,
    kind: u32,
    vel: vec3<f32>,
    life: f32,
}

struct SplashRing {
    counter: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
    ring: array<vec4<f32>, 4096>,
}

struct DrawParams {
    // xyz: unit vector toward the sun. w: its direct illuminance, lx.
    sun: vec4<f32>,
    // xyz: unit vector toward the moon. w: its direct illuminance, lx.
    moon: vec4<f32>,
    // x: sky illuminance, lx. y: extinction σ_V, 1/m. z: soft fade, m. w: splash soft fade, m.
    ambient: vec4<f32>,
    // xyz: lightning flash position, m (metre frame). w: intensity, cd (0 = none).
    flash: vec4<f32>,
    // x: Marshall–Palmer Λ, 1/mm. y: streak alpha. z: exposure time, s. w: edge fade, m.
    rain: vec4<f32>,
    // x: snow albedo. y: ring radius, m. z: ring life, s. w: shader clock, s.
    snow: vec4<f32>,
    // rgb: dust albedo. w: dust sprite diameter, m.
    dust: vec4<f32>,
    // xyz: camera velocity, m/s. w: box half edge, m.
    camera: vec4<f32>,
    // xyz: the box centre (the view's focus), m. w: under an orthographic view, the depth at which
    // its distance fog starts, m.
    centre: vec4<f32>,
    // x: world units per metre. The particles are in metres; the view's matrices and depth are
    // in world units. yzw: unused.
    units: vec4<f32>,
    // The game's look (`PrecipitationLook`, docs/research/precipitation-look.md), draw-only: x the
    // rain streak's fall scale, y its wind scale, z the flakes' log-normal size spread σ, w the
    // smallest flake half width, px.
    look: vec4<f32>,
    // rgb: multiplier on a rain streak's radiance. a: on its opacity.
    tint: vec4<f32>,
}

@group(0) @binding(0) var<uniform> view: View;
@group(0) @binding(1) var<uniform> draw: DrawParams;
@group(0) @binding(2) var<storage, read> particles: array<Particle>;
@group(0) @binding(3) var<storage, read> splashes: SplashRing;
@group(0) @binding(4) var scene_depth: texture_depth_2d;

const KIND_RAIN: f32 = 0.0;
const KIND_SNOW: f32 = 1.0;
const KIND_SPLASH: f32 = 2.0;
const TIME_WRAP: f32 = 1024.0;
const RAIN_PHASE_G: f32 = 0.8;
const PI: f32 = 3.14159265;
const NEAR_FADE_START: f32 = 0.15;
const NEAR_FADE_LENGTH: f32 = 0.35;
const SNOW_ALPHA: f32 = 0.8;
/// Opacity of one dust sprite: a cluster-scale veil of grains, so below a flake's.
const DUST_ALPHA: f32 = 0.5;
// A flake's image is its disc stretched by at most this many diameters along the relative
// velocity (Deviation 17 of the research note): at most 2:1 head to tail.
const SNOW_MAX_STRETCH: f32 = 1.0;
// Half width, px, below which a dust sprite's image does not shrink: a soft blob (a rain
// streak's is half a pixel, a flake's `draw.look.w`).
const DUST_MIN_HALF_WIDTH: f32 = 1.0;
const SPLASH_ALPHA: f32 = 0.6;
// Under an orthographic view (view-anchor.md Deviation 3): a rain streak is drawn over this many
// exposures, so the short fall of one frame reads as a streak and not a dash, and at this
// coverage (opacity `STREAK_ALPHA · 0.12 = 0.18`), so the drops read as rain and not sleet, and
// the whole column the view looks down through does not grey the scene. Chosen in the capture
// review.
const ORTHOGRAPHIC_STREAK_EXPOSURES: f32 = 3.0;
const ORTHOGRAPHIC_RAIN_COVERAGE: f32 = 0.12;
const SPLASH_QUAD: f32 = 1.3;

struct VertexOut {
    @builtin(position) position: vec4<f32>,
    // Streaks and flakes: pixels across and along the capsule. Splashes: ring-radius units.
    @location(0) local: vec2<f32>,
    // x: half width, px (splash: age / life). y: length, px. z: view depth, m. w: kind.
    @location(1) @interpolate(flat) shape: vec4<f32>,
    // rgb: radiance times exposure and transmittance. a: opacity before coverage and fades.
    @location(2) @interpolate(flat) colour: vec4<f32>,
}

fn degenerate() -> VertexOut {
    return VertexOut(vec4<f32>(2.0, 2.0, 2.0, 1.0), vec2<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0));
}

/// Henyey–Greenstein phase function, integrates to 1 over the sphere.
fn henyey_greenstein(cos_theta: f32, g: f32) -> f32 {
    let g2 = g * g;
    return (1.0 - g2) / (4.0 * PI * pow(1.0 + g2 - 2.0 * g * cos_theta, 1.5));
}

/// Brightness of the lightning flash at `p` as an illuminance-like term (0 when there is no
/// flash); added to the sky light and, as its share of the total, makes rain more transparent
/// (Tatarchuk §3.6.2).
fn flash_at(p: vec3<f32>) -> f32 {
    let d = p - draw.flash.xyz;
    return draw.flash.w / max(dot(d, d), 1.0);
}

/// The fade at the sides of the box: 1 inside, falling to 0 over the outer `draw.rain.w` metres.
fn edge_fade(p: vec3<f32>) -> f32 {
    let half_edge = draw.camera.w;
    let d = abs(p - draw.centre.xyz);
    let m = max(max(d.x, d.y), d.z);
    return 1.0 - smoothstep(half_edge - draw.rain.w, half_edge, m);
}

/// An orthographic view: `w` is 1 everywhere (Bevy's own test, `cluster_raster.wesl`).
fn orthographic() -> bool {
    return view.clip_from_view[3].w == 1.0;
}

/// Unit direction from the eye toward the world position `p`: the view direction everywhere in an
/// orthographic view.
fn eye_ray(p: vec3<f32>) -> vec3<f32> {
    if orthographic() {
        return normalize(-view.world_from_view[2].xyz);
    }
    return normalize(p - view.world_position);
}

/// Distance of the metre-frame point `p` in front of the eye along the view axis, m.
fn view_depth(p: vec3<f32>) -> f32 {
    return -(view.view_from_world * vec4<f32>(p * draw.units.x, 1.0)).z / draw.units.x;
}

/// The distance fog's transmittance to `p`, as the meshes carry it: a perspective view's falls
/// with the distance from the eye; an orthographic view's counts from the nearest ground it
/// shows, linearly at the visibility's extinction (`fog::orthographic_fog_falloff`).
fn fog_transmittance(p: vec3<f32>) -> f32 {
    if orthographic() {
        return 1.0 - saturate((view_depth(p) - draw.centre.w) * draw.ambient.y);
    }
    return exp(-draw.ambient.y * distance(p * draw.units.x, view.world_position) / draw.units.x);
}

/// A flake's drawn size factor `exp(σ z − σ²/2)`: a log-normal of mean 1 about its own size, `z`
/// the standard normal Box–Muller makes of the particle's own hash (salt 5, used by no other
/// draw), clamped to ±2 so no flake grows past e^{2σ} of its size into a blur. Exactly 1 at σ = 0.
/// `look::flake_size_factor` mirrors it.
fn flake_size_factor(index: u32, epoch: u32, sigma: f32) -> f32 {
    let u = particle_rand(index, epoch, 5u);
    let z = clamp(sqrt(-2.0 * log(u.x)) * cos(2.0 * PI * u.y), -2.0, 2.0);
    return exp(sigma * z - 0.5 * sigma * sigma);
}

fn corner_of(vi: u32) -> u32 {
    var corners = array<u32, 6>(0u, 1u, 2u, 2u, 1u, 3u);
    return corners[vi];
}

fn to_px(clip: vec4<f32>, size: vec2<f32>) -> vec2<f32> {
    let ndc = clip.xy / clip.w;
    return vec2<f32>((ndc.x * 0.5 + 0.5) * size.x, (0.5 - ndc.y * 0.5) * size.y);
}

@vertex
fn particle_vertex(
    @builtin(vertex_index) vi: u32,
    @builtin(instance_index) ii: u32,
) -> VertexOut {
    let p = particles[ii];
    let kind = p.kind & 3u;
    if kind > 2u {
        return degenerate();
    }
    let epoch = p.kind >> 2u;
    let size = view.viewport.zw;
    let focal_px = 0.5 * size.y * view.clip_from_view[1][1];
    // A perspective view clips at the near plane on `w`; an orthographic `w` is 1 everywhere,
    // and the rasteriser clips on depth.
    let near = select(max(view.clip_from_view[3][2], 1e-3), -1.0, orthographic());

    let v_rel = p.vel - draw.camera.xyz;
    var diameter_m: f32;
    if kind == 0u {
        diameter_m = drop_diameter_mm(particle_rand(ii, epoch, 0u), draw.rain.x) * 1e-3;
    } else if kind == 1u {
        diameter_m = (4.0 + 4.0 * particle_rand(ii, epoch, 1u).w) * 1e-3;
    } else {
        diameter_m = draw.dust.w;
    }
    // The streak: the drop's path in the exposure time, in metres (the particle is simulated in
    // metres). A flake is a soft disc: its streak is capped at `SNOW_MAX_STRETCH` diameters, so
    // wind does not draw it as a dash.
    var streak = v_rel * draw.rain.z;
    if kind == 0u && orthographic() {
        streak *= ORTHOGRAPHIC_STREAK_EXPOSURES;
    }
    if kind == 0u {
        // The game's look: the fall and the wind drift scaled apart (`look::styled_streak`).
        streak = vec3<f32>(streak.x * draw.look.y, streak.y * draw.look.x, streak.z * draw.look.y);
    }
    if kind != 0u {
        streak *= min(1.0, SNOW_MAX_STRETCH * diameter_m / max(length(streak), 1e-6));
    }
    // The drop is simulated in metres; Bevy's matrices take world units.
    let u = draw.units.x;
    var head = view.clip_from_world * vec4<f32>(p.pos * u, 1.0);
    var tail = view.clip_from_world * vec4<f32>((p.pos - streak) * u, 1.0);
    if head.w < near && tail.w < near {
        return degenerate();
    }
    if head.w < near {
        head = mix(head, tail, (near - head.w) / (tail.w - head.w));
    }
    if tail.w < near {
        tail = mix(tail, head, (near - tail.w) / (head.w - tail.w));
    }
    let z = 0.5 * (head.w + tail.w);

    // Half width in px: the drop's own, but at least half a pixel (a flake's `draw.look.w`, a dust
    // sprite's `DUST_MIN_HALF_WIDTH`); opacity follows the true coverage, relative to that minimum,
    // so a sub-pixel drop fades as 1/z (Garg & Nayar 2007, Eqs. 21–22). An orthographic view has
    // no distance at which a drop shrinks: its pixel size is the zoom the game chose, and the
    // coverage fade would hide the precipitation at every zoom a strategy view uses, so a flake
    // keeps its minimum width at full opacity and a drop at `ORTHOGRAPHIC_RAIN_COVERAGE`
    // (view-anchor.md Deviation 3). A flake's drawn size then spreads about its own by the look's
    // `flake_size_factor`.
    let min_half_width = select(select(DUST_MIN_HALF_WIDTH, draw.look.w, kind == 1u), 0.5, kind == 0u);
    // The drop's diameter is in metres, `focal_px / z` is pixels per world unit.
    let true_half_width = 0.5 * diameter_m * u * focal_px / z;
    var half_width = max(true_half_width, min_half_width);
    if kind == 1u {
        half_width *= flake_size_factor(ii, epoch, draw.look.z);
    }
    var coverage = min(1.0, true_half_width / min_half_width);
    if orthographic() {
        coverage = select(1.0, ORTHOGRAPHIC_RAIN_COVERAGE, kind == 0u);
    }

    let head_px = to_px(head, size);
    let tail_px = to_px(tail, size);
    let along_vec = tail_px - head_px;
    let length_px = length(along_vec);
    var along_dir = vec2<f32>(0.0, 1.0);
    if length_px > 1e-4 {
        along_dir = along_vec / length_px;
    }
    let across_dir = vec2<f32>(-along_dir.y, along_dir.x);
    let margin = half_width + 0.5;

    let k = corner_of(vi);
    let at_tail = (k & 1u) == 1u;
    let side = f32(k >> 1u) * 2.0 - 1.0;
    let along_offset = select(-margin, length_px + margin, at_tail);
    let px = head_px + along_dir * along_offset + across_dir * side * margin;

    // Light: the drop refracts the environment; the flake scatters diffusely.
    let to_sun = draw.sun.xyz;
    let to_moon = draw.moon.xyz;
    let ray = eye_ray(p.pos * u);
    let flash = flash_at(p.pos);
    let sky = draw.ambient.x + flash;
    var radiance: vec3<f32>;
    var opacity: f32;
    if kind == 0u {
        radiance = draw.tint.rgb * (sky / PI
            + draw.sun.w * henyey_greenstein(dot(to_sun, ray), RAIN_PHASE_G)
            + draw.moon.w * henyey_greenstein(dot(to_moon, ray), RAIN_PHASE_G));
        // Tatarchuk §3.6.2: rain turns more transparent under a flash.
        let flash_share = flash / max(sky, 1e-6);
        opacity = min(draw.rain.y * coverage * draw.tint.a, 0.9) * (1.0 - 0.5 * flash_share);
    } else {
        // A flake or a dust sprite scatters diffusely, with its own albedo.
        let albedo = select(draw.dust.xyz, vec3<f32>(draw.snow.x), kind == 1u);
        radiance = albedo * (sky + draw.sun.w * abs(dot(to_sun, ray))
            + draw.moon.w * abs(dot(to_moon, ray))) / PI;
        // Dust is drawn at cluster scale (`DUST_SPRITE_DRAW_M`): each sprite is a veil of many
        // grains, so it carries its own opacity — below a flake's — rather than a puff's.
        opacity = select(DUST_ALPHA, SNOW_ALPHA, kind == 1u) * coverage;
    }
    let transmittance = fog_transmittance(p.pos);
    opacity *= edge_fade(p.pos);

    var out: VertexOut;
    out.position = vec4<f32>(
        px.x / size.x * 2.0 - 1.0,
        1.0 - px.y / size.y * 2.0,
        0.5,
        1.0,
    );
    out.local = vec2<f32>(side * margin, along_offset);
    // The soft fade compares view depths: a perspective `w` is the view depth; an orthographic
    // `w` is 1, so the depth is taken from the view matrix.
    let depth_z = select(z / u, 0.5 * (view_depth(p.pos) + view_depth(p.pos - streak)), orthographic());
    out.shape = vec4<f32>(half_width, length_px, depth_z, select(KIND_RAIN, KIND_SNOW, kind != 0u));
    out.colour = vec4<f32>(radiance * view.exposure * transmittance, opacity);
    return out;
}

@vertex
fn splash_vertex(
    @builtin(vertex_index) vi: u32,
    @builtin(instance_index) ii: u32,
) -> VertexOut {
    let entry = splashes.ring[ii];
    if entry.w < 0.0 {
        return degenerate();
    }
    var age = draw.snow.w - entry.w;
    if age < 0.0 {
        age += TIME_WRAP;
    }
    let life = draw.snow.z;
    if age > life {
        return degenerate();
    }
    let k = corner_of(vi);
    let corner = vec2<f32>(f32(k & 1u) * 2.0 - 1.0, f32(k >> 1u) * 2.0 - 1.0) * SPLASH_QUAD;
    let world = entry.xyz + vec3<f32>(corner.x, 0.0, corner.y) * draw.snow.y;
    let u = draw.units.x;
    let centre = view.clip_from_world * vec4<f32>(entry.xyz * u, 1.0);
    let splash_z = select(centre.w / u, view_depth(entry.xyz), orthographic());
    let radiance = (draw.ambient.x + draw.sun.w * max(draw.sun.y, 0.0)
        + draw.moon.w * max(draw.moon.y, 0.0)) / PI;
    let transmittance = fog_transmittance(entry.xyz);
    var out: VertexOut;
    out.position = view.clip_from_world * vec4<f32>(world * u, 1.0);
    out.local = corner;
    out.shape = vec4<f32>(age / life, 0.0, splash_z, KIND_SPLASH);
    out.colour = vec4<f32>(
        vec3<f32>(radiance) * view.exposure * transmittance,
        SPLASH_ALPHA * edge_fade(entry.xyz),
    );
    return out;
}

@fragment
fn particle_fragment(in: VertexOut) -> @location(0) vec4<f32> {
    var coverage: f32;
    var soft = draw.ambient.z;
    if in.shape.w == KIND_SPLASH {
        // An expanding ring that thins and fades as it ages.
        let age = in.shape.x;
        let radius = mix(0.35, 1.0, age);
        let ring = exp(-pow((length(in.local) - radius) / 0.1, 2.0));
        coverage = ring * (1.0 - age) * (1.0 - age);
        soft = draw.ambient.w;
    } else {
        let along = clamp(in.local.y, 0.0, in.shape.y);
        let d = length(vec2<f32>(in.local.x, in.local.y - along));
        if in.shape.w == KIND_RAIN {
            coverage = saturate(in.shape.x + 0.5 - d);
        } else {
            let s = saturate(1.0 - d / (in.shape.x + 0.5));
            coverage = s * s * (3.0 - 2.0 * s);
        }
    }
    // Soft particles: fade as the scene surface comes within `soft` of the particle, and as the
    // particle nears the camera. Reverse-Z infinite far: linear depth is near / depth; an
    // orthographic view's comes back through `view_from_clip`.
    let depth = textureLoad(scene_depth, vec2<i32>(in.position.xy), 0);
    var scene_z = 1e9;
    if orthographic() {
        let v = view.view_from_clip * vec4<f32>(0.0, 0.0, depth, 1.0);
        scene_z = -v.z / v.w / draw.units.x;
    } else if depth > 0.0 {
        scene_z = max(view.clip_from_view[3][2], 1e-3) / depth / draw.units.x;
    }
    let z = in.shape.z;
    let fade = saturate((scene_z - z) / soft) * saturate((z - NEAR_FADE_START) / NEAR_FADE_LENGTH);
    let alpha = in.colour.a * coverage * fade;
    return vec4<f32>(in.colour.rgb * alpha, alpha);
}
