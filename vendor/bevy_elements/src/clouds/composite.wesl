// Composite the resolved cloud history over the scene: `scene · T + L` with the blend
// `src.rgb + dst.rgb · src.a`. A pixel whose scene surface is nearer than the cloud entry of its
// ray shows no cloud. See docs/research/clouds.md ("Aerial perspective and composite").

import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput;
import bevy_render::view::View;
import bevy_elements::clouds::common::{
    CloudParams, ground_before_layer, is_orthographic, scene_distance, shell_range,
    sky_through_clouds, view_ray,
};

@group(0) @binding(0) var<uniform> view: View;
@group(0) @binding(1) var<uniform> params: CloudParams;
@group(0) @binding(2) var history: texture_2d<f32>;
@group(0) @binding(3) var depth_tex: texture_depth_2d;

@fragment
fn cloud_composite(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    // Debug mode 4: the composite adds nothing (the tests render the same frame without it).
    if params.misc.y > 3.5 {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }
    let pixel = vec2<i32>(in.position.xy);
    let depth = textureLoad(depth_tex, pixel, 0);
    let ray = view_ray(
        in.uv,
        view.world_from_clip,
        view.world_position,
        is_orthographic(view.clip_from_view),
        params.camera.xyz,
        params.units.x,
    );
    let cam = ray.origin;
    let dir = ray.dir;
    if depth > 0.0 {
        let big_r = params.camera.w;
        let shell = shell_range(big_r + cam.y, dir.y, params.layer.x, params.layer.y, big_r);
        if scene_distance(in.uv, depth, view.world_from_clip, ray, params.units.x) < shell.x {
            return vec4<f32>(0.0, 0.0, 0.0, 1.0);
        }
    }
    var cloud = textureLoad(history, pixel, 0);
    if depth <= 0.0 {
        // The sky: the cloud over Bevy's atmosphere, with the sky gain and the weather's air
        // (`sky_through_clouds`; the sky probe composes its cube the same way). A ray that meets
        // the planet before the layer (past the edge of a finite ground) ends beneath the deck,
        // which the beam lights correctly: no gain there (clouds.md Deviation 28).
        let big_r = params.camera.w;
        let o_r = big_r + cam.y;
        let shell = shell_range(o_r, dir.y, params.layer.x, params.layer.y, big_r);
        if ground_before_layer(o_r, dir.y, shell.x, big_r) < 0.0 {
            cloud = sky_through_clouds(cloud, shell, params, params.haze.xyz);
        }
    }
    return cloud;
}
