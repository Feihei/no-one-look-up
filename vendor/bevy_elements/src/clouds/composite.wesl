// Composite the resolved cloud history over the scene: `scene · T + L` with the blend
// `src.rgb + dst.rgb · src.a`. A pixel whose scene surface is nearer than the cloud entry of its
// ray shows no cloud. See docs/research/clouds.md ("Aerial perspective and composite").

import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput;
import bevy_render::view::View;
import bevy_elements::clouds::common::{
    CloudParams, MAX_MARCH_M, ground_before_layer, ray_dir, scene_distance, shell_range,
};

@group(0) @binding(0) var<uniform> view: View;
@group(0) @binding(1) var<uniform> params: CloudParams;
@group(0) @binding(2) var history: texture_2d<f32>;
@group(0) @binding(3) var depth_tex: texture_depth_2d;

/// Transmittances below this are an opaque cloud; above the second the sky is really in view.
const OPAQUE_BELOW: f32 = 0.02;
const SKY_ABOVE: f32 = 0.2;

@fragment
fn cloud_composite(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    // Debug mode 4: the composite adds nothing (the tests render the same frame without it).
    if params.misc.y > 3.5 {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }
    let pixel = vec2<i32>(in.position.xy);
    let depth = textureLoad(depth_tex, pixel, 0);
    if depth > 0.0 {
        let cam = view.world_position;
        let dir = ray_dir(in.uv, view.world_from_clip, cam);
        let big_r = params.camera.w;
        let shell = shell_range(big_r + cam.y, dir.y, params.layer.x, params.layer.y, big_r);
        if scene_distance(in.uv, depth, view.world_from_clip, cam) < shell.x {
            return vec4<f32>(0.0, 0.0, 0.0, 1.0);
        }
    }
    var cloud = textureLoad(history, pixel, 0);
    if depth <= 0.0 {
        // The sky: Bevy's atmosphere is lit by the sun light, which carries only the beam
        // `T̄` under cloud, but the sky above the cloud is not dimmed by the cloud below it.
        // The march stops at a transmittance of 0.01, so the gain ramps in above that: an
        // opaque cloud must not amplify the sky behind it. A ray that meets the planet before
        // the layer (past the edge of a finite ground) ends beneath the deck, which the beam
        // lights correctly: no gain there (clouds.md Deviation 28).
        let cam = view.world_position;
        let dir = ray_dir(in.uv, view.world_from_clip, cam);
        let big_r = params.camera.w;
        let o_r = big_r + cam.y;
        let shell = shell_range(o_r, dir.y, params.layer.x, params.layer.y, big_r);
        if ground_before_layer(o_r, dir.y, shell.x, big_r) < 0.0 {
            cloud.a *= mix(1.0, params.sky.x, smoothstep(OPAQUE_BELOW, SKY_ABOVE, cloud.a));
            // The weather's air, over the path to the sky this pixel shows: the camera's
            // distance fog (σ_fog) is what every mesh carries, and a sky pixel already has
            // Bevy's atmosphere, which draws the clear air — so `air.x = σ_fog − 3.912/40 km`
            // is what the sky must add on top of it. Without this the sky above the horizon
            // keeps the deck's own radiance while the ground converges on the fog, and a
            // whiteout shows a horizon line (measured 185–195 against fogged snow 225 at
            // 200 m; clouds.md Deviation 29). Both the in-scatter and the extinction are the
            // fog's, so a sky pixel over a long enough path is the fog colour exactly as a
            // distant mesh is, and clear air (σ_air = 0) is the exact no-op this was before.
            // The path is the layer entry. A ray that never enters the layer at all
            // (`shell_range` found no crossing) shows only sky, and the weather's air over it
            // is the weather's whole extent; a ray that starts inside the layer has no air
            // between the camera and the cloud it draws.
            let air_path = select(shell.x, MAX_MARCH_M, shell.y <= 0.0);
            let t = exp(-params.air.x * air_path);
            cloud = vec4<f32>(
                cloud.rgb * t + params.haze.xyz * (1.0 - t),
                cloud.a * t,
            );
        }
    }
    return cloud;
}
