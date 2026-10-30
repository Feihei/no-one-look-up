// bevy_elements::cover::common — the uniform of the ground-cover kernels (snow, wetness, shelter,
// readback) and the helpers they share. `CoverParams` mirrors the Rust struct of the same name in
// `src/cover/mod.rs` field for field. See docs/research/snow-cover.md and wetness.md.
//
//     import bevy_elements::cover::common::{CoverParams, local_temperature, slope_tangent};

import bevy_elements::grid::GridParams;

struct CoverParams {
    grid: GridParams,
    // x: Δt of one snow step (s). y: precipitation (mm/h, all phases). z: air temperature at the
    // reference elevation (°C). w: temperature lapse (°C/m, negative).
    sim: vec4<f32>,
    // x: terrain y0 − reference elevation (m). y: terrain y0 (m). z: 10 m wind speed (m/s).
    // w: Δt of the wetness update (s): the whole frame.
    terrain: vec4<f32>,
    // x: new-snow density. y: settled density (kg/m³). z: settling time constant (h).
    // w: degree-day factor (mm w.e. per °C per day).
    snow: vec4<f32>,
    // x, y: shed slope range (rad). z: drift threshold u_t (m/s). w: k_d / ρ_w, so the drift flux
    // is q = w · (u − u_t)³ (m²/s of water equivalent).
    shed: vec4<f32>,
    // xy: unit XZ direction the wind blows toward. zw: unused.
    wind: vec4<f32>,
    // x: wetness per mm of rain. y: dry rate (1/h). z: relative humidity. w: sun factor.
    wet: vec4<f32>,
    // x: wind speed for evaporation (m/s). y: Weather::snow_possible (0 outside the snow season).
    // zw: unused.
    air: vec4<f32>,
    // xy: horizontal velocity of falling rain (the wind, m/s) negated, w: its fall speed (m/s):
    // the ray from the ground toward where the rain came from is (x, w, y) in (x, y, z).
    rain: vec4<f32>,
    snow_ray: vec4<f32>,
    // x: blockers uploaded. y: unused. z: readback downsample factor. w: unused.
    flags: vec4<u32>,
    // The snowpack `cover_seed` writes: x: depth (m). y: SWE (m). z: wetness. w: unused.
    seed: vec4<f32>,
}

/// Air temperature (°C) at a ground surface height (m above the terrain's y0).
fn local_temperature(params: CoverParams, surface: f32) -> f32 {
    return params.sim.z + params.sim.w * (surface + params.terrain.x);
}

/// `tan α` of the slope, from the packed gradient of `terrain_cells.w`.
fn slope_tangent(packed: f32) -> f32 {
    return length(unpack2x16float(bitcast<u32>(packed)));
}
