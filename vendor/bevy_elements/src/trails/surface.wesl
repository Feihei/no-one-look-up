// bevy_elements::trails::surface — drawing the trail field on a ground material
// (docs/research/trails.md, "Rendering"): parallax occlusion mapping of the camera-following
// centimetre height field (Tatarchuk 2006: a linear search with a piecewise-linear intersection,
// the sample count by the view angle, soft self-shadows from the blocker-to-receiver ratio), its
// normal, and how open the floor is to the sky.
//
// The field is `rgba16float`, (h, occupancy, ∂h/∂x, ∂h/∂z) with `h` the surface's offset from
// the undisturbed ground (m), toroidal by world texel: sample it at `xz / (texel · N)` with a
// repeating sampler. `window` is (texel edge m, texels a side, 1 when bound, bound on |h| m);
// `centre` is the window's centre, metre-frame XZ (`TrailState::centre`: the texel of the
// `ElementsFocus`, else the camera), which the crate's materials carry in their uniforms.

// 1 well inside the window, 0 outside, a ramp over its outer eighth: the field is cleared where
// it enters and wraps around beyond, and neither may show.
fn trail_fade(xz: vec2<f32>, centre: vec2<f32>, window: vec4<f32>) -> f32 {
    if window.z < 0.5 {
        return 0.0;
    }
    let half = 0.5 * window.x * window.y;
    let d = abs(xz - centre);
    return 1.0 - smoothstep(0.85 * half, 0.97 * half, max(d.x, d.y));
}

fn trail_sample(field: texture_2d<f32>, smp: sampler, xz: vec2<f32>, window: vec4<f32>) -> vec4<f32> {
    return textureSampleLevel(field, smp, xz / (window.x * window.y), 0.0);
}

// The field reconstructed with a cubic B-spline from four bilinear taps (Sigg & Hadwiger 2005,
// *Fast third-order texture filtering*, GPU Gems 2 ch. 20): smooth where the bilinear field has
// a kink at every texel edge.
fn trail_smooth(field: texture_2d<f32>, smp: sampler, xz: vec2<f32>, window: vec4<f32>) -> vec4<f32> {
    // Texel coordinates with texel k's centre at k.
    let t = xz / window.x - 0.5;
    let i = floor(t);
    let f = t - i;
    let f2 = f * f;
    let f3 = f2 * f;
    let w0 = (1.0 - 3.0 * f + 3.0 * f2 - f3) / 6.0;
    let w1 = (4.0 - 6.0 * f2 + 3.0 * f3) / 6.0;
    let w2 = (1.0 + 3.0 * f + 3.0 * f2 - 3.0 * f3) / 6.0;
    let w3 = f3 / 6.0;
    let g0 = w0 + w1;
    let g1 = w2 + w3;
    // The two bilinear taps per axis that sum the four weighted texels.
    let h0 = (i - 1.0 + w1 / g0 + 0.5) * window.x;
    let h1 = (i + 1.0 + w3 / g1 + 0.5) * window.x;
    let a = trail_sample(field, smp, vec2<f32>(h0.x, h0.y), window);
    let b = trail_sample(field, smp, vec2<f32>(h1.x, h0.y), window);
    let c = trail_sample(field, smp, vec2<f32>(h0.x, h1.y), window);
    let d = trail_sample(field, smp, vec2<f32>(h1.x, h1.y), window);
    return g0.y * (g0.x * a + g1.x * b) + g1.y * (g0.x * c + g1.x * d);
}

struct TrailHit {
    // World XZ where the view ray meets the field, and the field's height there over the
    // undisturbed plane, m.
    xz: vec2<f32>,
    h: f32,
    // ∂h/∂x, ∂h/∂z at the hit.
    grad: vec2<f32>,
    // 1 where the floor sees the whole sky, less in a trench.
    open: f32,
    // 1 where the light reaches the hit past the field, 0 in its self-shadow.
    lit: f32,
}

// The field's absence at world XZ: flat, open and lit.
fn trail_none(xz: vec2<f32>) -> TrailHit {
    var empty: TrailHit;
    empty.xz = xz;
    empty.h = 0.0;
    empty.grad = vec2<f32>(0.0);
    empty.open = 1.0;
    empty.lit = 1.0;
    return empty;
}

// How high a rim may stand above the undisturbed ground, m: the march starts there.
const TRAIL_TOP_M: f32 = 0.1;
// The most ground a march crosses either side of the fragment, m.
const TRAIL_REACH_M: f32 = 2.5;

// The view ray from `eye` through the fragment at `p` (on the undisturbed surface whose slope is
// `slope`, ∂y/∂x and ∂y/∂z) against the field, and what the light `l` (unit, toward the light)
// sees there. `fade` scales the field (the window's edge).
fn trail_march(
    field: texture_2d<f32>,
    smp: sampler,
    window: vec4<f32>,
    p: vec3<f32>,
    eye: vec3<f32>,
    slope: vec2<f32>,
    l: vec3<f32>,
    fade: f32,
) -> TrailHit {
    var hit = trail_none(p.xz);
    let v = normalize(p - eye);
    // The ray's rise over the local plane per metre along it.
    let rise = v.y - dot(slope, v.xz);
    if rise > -1.0e-3 || fade <= 0.0 {
        return hit;
    }
    // From TRAIL_TOP_M above the plane (the rims) to the window's bound below it (the deepest
    // trench), at most TRAIL_REACH_M of ground either side of `p`.
    let along = max(length(v.xz), 1.0e-3);
    let t0 = -min(TRAIL_TOP_M / -rise, TRAIL_REACH_M / along);
    let t1 = min(window.w / -rise, TRAIL_REACH_M / along);
    // Nothing near the ray's start, middle or end (the field's dilated block occupancy): flat.
    let occupied = max(
        trail_sample(field, smp, p.xz, window).y,
        max(
            trail_sample(field, smp, p.xz + v.xz * t0, window).y,
            trail_sample(field, smp, p.xz + v.xz * t1, window).y,
        ),
    );
    if occupied < 1.0e-3 {
        return hit;
    }
    // One sample per texel of ground the ray crosses (Tatarchuk's linear search, its count by
    // the angle), then a bisection on the piecewise-linear profile.
    let n = i32(clamp(ceil((t1 - t0) * along / window.x), 8.0, 96.0));
    var lo = t0;
    var hi = t1;
    var prev_gap = TRAIL_TOP_M;
    var crossed = false;
    for (var i = 1; i <= n; i += 1) {
        let t = mix(t0, t1, f32(i) / f32(n));
        let gap = rise * t - trail_sample(field, smp, p.xz + v.xz * t, window).x * fade;
        if gap <= 0.0 {
            hi = t;
            crossed = true;
            break;
        }
        lo = t;
        prev_gap = gap;
    }
    if !crossed {
        // The ray stays above the field over the whole reach (grazing): the field under `p`.
        let s = trail_sample(field, smp, p.xz, window) * fade;
        hit.h = s.x;
        hit.grad = s.zw;
        return hit;
    }
    // The bisection and the hit on the smooth (B-spline) field: the bilinear one steps along any
    // edge at a shallow angle to the texel grid. The bracket came from the bilinear field, so
    // widen it by up to two steps each way until the smooth field changes sign inside it.
    let dt = (t1 - t0) / f32(n);
    var lo_gap = rise * lo - trail_smooth(field, smp, p.xz + v.xz * lo, window).x * fade;
    for (var j = 0; j < 2 && lo_gap <= 0.0; j += 1) {
        lo = max(lo - dt, t0);
        lo_gap = rise * lo - trail_smooth(field, smp, p.xz + v.xz * lo, window).x * fade;
    }
    var hi_gap = rise * hi - trail_smooth(field, smp, p.xz + v.xz * hi, window).x * fade;
    for (var j = 0; j < 2 && hi_gap > 0.0; j += 1) {
        hi = min(hi + dt, t1);
        hi_gap = rise * hi - trail_smooth(field, smp, p.xz + v.xz * hi, window).x * fade;
    }
    prev_gap = lo_gap;
    for (var k = 0; k < 4; k += 1) {
        let mid = 0.5 * (lo + hi);
        let gap = rise * mid - trail_smooth(field, smp, p.xz + v.xz * mid, window).x * fade;
        if gap <= 0.0 {
            hi = mid;
            hi_gap = gap;
        } else {
            lo = mid;
            prev_gap = gap;
        }
    }
    let t_hit = mix(lo, hi, clamp(prev_gap / max(prev_gap - hi_gap, 1.0e-6), 0.0, 1.0));
    hit.xz = p.xz + v.xz * t_hit;
    let s = trail_smooth(field, smp, hit.xz, window) * fade;
    hit.h = s.x;
    // The normal from central differences of the B-spline height a texel apart: any gradient
    // stored per texel steps along a wall at an angle to the grid, which a glossy (wet) surface
    // shows as fluting.
    let e = vec2<f32>(window.x, 0.0);
    hit.grad = fade * vec2<f32>(
        trail_smooth(field, smp, hit.xz + e.xy, window).x - trail_smooth(field, smp, hit.xz - e.xy, window).x,
        trail_smooth(field, smp, hit.xz + e.yx, window).x - trail_smooth(field, smp, hit.xz - e.yx, window).x,
    ) / (2.0 * window.x);
    // Openness: how far the floor lies below a ring around it.
    let r = 0.12;
    var ring = 0.0;
    ring += trail_sample(field, smp, hit.xz + vec2<f32>(r, 0.0), window).x;
    ring += trail_sample(field, smp, hit.xz - vec2<f32>(r, 0.0), window).x;
    ring += trail_sample(field, smp, hit.xz + vec2<f32>(0.0, r), window).x;
    ring += trail_sample(field, smp, hit.xz - vec2<f32>(0.0, r), window).x;
    hit.open = 1.0 - 0.6 * clamp((0.25 * ring * fade - hit.h) / 0.25, 0.0, 1.0);
    // Soft self-shadow toward the light: the largest blocker-to-receiver ratio along it, as far
    // as a ray from the deepest floor needs to climb out.
    if l.y > 0.02 {
        let l_rise = max(l.y - dot(slope, l.xz), 0.02);
        let reach = min((TRAIL_TOP_M - hit.h) / l_rise, 1.5);
        // One sample per texel of ground the light ray crosses (at most 32 over the longest
        // reach), at the same spacing for every pixel: spacing by each pixel's own reach moves
        // every sample when the count changes, and the shadow's edge breaks into ticks there.
        let du = max(window.x / max(length(l.xz), 1.0e-3), 1.5 / 32.0);
        let ns = i32(clamp(ceil(reach / du), 1.0, 32.0));
        var shadow = 0.0;
        for (var k = 1; k <= ns; k += 1) {
            let u = du * f32(k);
            // The blocker on the B-spline field: on the bilinear one the shadow's edge steps at
            // every texel along a rut's wall.
            let blocker = trail_smooth(field, smp, hit.xz + l.xz * u, window).x * fade;
            let ray = hit.h + l_rise * u;
            // A penumbra that widens with the distance to the blocker (Tatarchuk's
            // blocker-to-receiver ratio).
            shadow = max(shadow, smoothstep(0.0, 1.0, (blocker - ray) / (0.01 + 0.25 * u)));
        }
        hit.lit = 1.0 - clamp(shadow, 0.0, 1.0);
    }
    return hit;
}
