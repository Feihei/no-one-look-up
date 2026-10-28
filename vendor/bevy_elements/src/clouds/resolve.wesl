// Temporal resolve: one pixel of every k×k block takes this frame's march, every other pixel
// takes last frame's history reprojected through the previous view, clamped to the colour box of
// this frame's fresh neighbourhood so a changed or uncovered region cannot ghost.
// See docs/research/clouds.md ("Temporal reprojection").

import bevy_render::view::View;
import bevy_elements::clouds::common::{CloudParams, ray_dir, shell_range, slot_offset};

/// Mirrors Bevy's `PreviousViewData`.
struct PreviousView {
    view_from_world: mat4x4<f32>,
    clip_from_world: mat4x4<f32>,
    unjittered_clip_from_world: mat4x4<f32>,
    clip_from_view: mat4x4<f32>,
    world_from_clip: mat4x4<f32>,
    view_from_clip: mat4x4<f32>,
}

@group(0) @binding(0) var<uniform> view: View;
@group(0) @binding(1) var<uniform> previous: PreviousView;
@group(0) @binding(2) var<uniform> params: CloudParams;
@group(0) @binding(3) var march_tex: texture_2d<f32>;
@group(0) @binding(4) var history_in: texture_2d<f32>;
@group(0) @binding(5) var smp: sampler;
@group(0) @binding(6) var history_out: texture_storage_2d<rgba16float, write>;

/// Weight of the fresh sample in the history at the pixel that was marched this frame.
const OWNER_BLEND: f32 = 0.35;

@compute @workgroup_size(8, 8, 1)
fn cloud_resolve(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = params.sizes.xy;
    if any(id.xy >= size) {
        return;
    }
    let low = vec2<i32>(params.sizes.zw);
    let k = params.counts.z;
    let block = min(vec2<i32>(id.xy / k), low - vec2<i32>(1));
    let fresh = textureLoad(march_tex, block, 0);
    let owner = all((id.xy % vec2<u32>(k)) == slot_offset(params.counts.x, k));
    if params.misc.w > 0.5 {
        textureStore(history_out, vec2<i32>(id.xy), fresh);
        return;
    }

    let uv = (vec2<f32>(id.xy) + 0.5) / vec2<f32>(size);
    let cam = view.world_position;
    let dir = ray_dir(uv, view.world_from_clip, cam);
    let big_r = params.camera.w;
    let shell = shell_range(big_r + cam.y, dir.y, params.layer.x, params.layer.y, big_r);
    if shell.y <= shell.x {
        textureStore(history_out, vec2<i32>(id.xy), vec4<f32>(0.0, 0.0, 0.0, 1.0));
        return;
    }

    // Where this pixel's cloud was on the previous screen: the point of the shell it enters.
    let clip = previous.clip_from_world * vec4<f32>(cam + dir * shell.x, 1.0);
    let ndc = clip.xy / clip.w;
    if clip.w <= 0.0 || any(abs(ndc) > vec2<f32>(1.0)) {
        textureStore(history_out, vec2<i32>(id.xy), fresh);
        return;
    }
    var history = textureSampleLevel(history_in, smp, vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5), 0.0);
    history = vec4<f32>(history.rgb * params.misc.x, history.a);

    var lo = fresh;
    var hi = fresh;
    for (var j = -1; j <= 1; j += 1) {
        for (var i = -1; i <= 1; i += 1) {
            let n = textureLoad(march_tex, clamp(block + vec2<i32>(i, j), vec2<i32>(0), low - vec2<i32>(1)), 0);
            lo = min(lo, n);
            hi = max(hi, n);
        }
    }
    var resolved = clamp(history, lo, hi);
    if owner {
        // The fresh pixel joins the history as an exponential moving average, which averages
        // the step-jitter noise of successive cycles (Deviation 13 of the research note).
        resolved = mix(resolved, fresh, OWNER_BLEND);
    }
    textureStore(history_out, vec2<i32>(id.xy), resolved);
}
