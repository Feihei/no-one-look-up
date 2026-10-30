// bevy_elements::wind — sample the GPU wind field from any shader.
//
// A supported import for consumers:
//
//     import bevy_elements::wind::{WindFieldParams, elements_wind};
//     @group(3) @binding(110) var wind_field: texture_3d<f32>;
//     @group(3) @binding(111) var wind_sampler: sampler;
//     @group(3) @binding(112) var<uniform> wind_params: WindFieldParams;
//     let v = elements_wind(wind_field, wind_sampler, wind_params, world_position);
//
// Bind `WindState::field` as a filterable 3D float texture with its own (repeating, linear)
// sampler and `WindState::field_params()` as the uniform.

/// Where the wind field sits. Mirrors the Rust `WindFieldParams`. The field is in the metre frame
/// (world ÷ `units`).
struct WindFieldParams {
    /// xyz: metre-frame position of the field's minimum corner, m. w: unused.
    origin: vec4<f32>,
    /// xyz: field size, m. w: 1 when the field holds simulated data, else 0.
    extent: vec4<f32>,
    /// xyz: prevailing wind at 10 m above ground, m/s, used outside the field. w: unused.
    prevailing: vec4<f32>,
    /// x: world units per metre. yzw: unused.
    units: vec4<f32>,
}

/// Wind velocity, m/s, at `metres`: a position in the metre frame (world ÷ `params.units.x`).
/// The field is stored toroidally (a cell's texel is its cell index modulo the grid), so the
/// texture coordinate is the position divided by the extent, wrapped; outside the current window
/// the prevailing wind is returned.
fn elements_wind_metres(
    field: texture_3d<f32>,
    s: sampler,
    params: WindFieldParams,
    metres: vec3<f32>,
) -> vec3<f32> {
    let local = metres - params.origin.xyz;
    let inside = all(local >= vec3<f32>(0.0)) && all(local <= params.extent.xyz);
    if !inside || params.extent.w < 0.5 {
        return params.prevailing.xyz;
    }
    let uvw = fract(metres / params.extent.xyz);
    return textureSampleLevel(field, s, uvw, 0.0).xyz;
}

/// Wind velocity at `world`, a world position, m/s.
fn elements_wind(
    field: texture_3d<f32>,
    s: sampler,
    params: WindFieldParams,
    world: vec3<f32>,
) -> vec3<f32> {
    return elements_wind_metres(field, s, params, world / params.units.x);
}
