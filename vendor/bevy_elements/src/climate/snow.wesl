// bevy_elements::climate::snow — the one definition of how precipitation splits into rain and
// snow, shared by the kernels that add rain to the water layer (hydrology) and snow to the cover
// layer (M4 snow). Both evaluate it at the temperature of the cell, so the two shares add up to
// the precipitation in every cell and nothing is counted twice.
//
//     import bevy_elements::climate::snow::snow_fraction_at;
//
// Mirrors `Weather::snow_fraction` (`climate/weather.rs`).

/// Fraction of precipitation that falls as snow at air temperature `t_c` (°C): 1 at −1 °C and
/// below, 0 at 2 °C and above, linear between.
fn snow_fraction_at(t_c: f32) -> f32 {
    return clamp((2.0 - t_c) / 3.0, 0.0, 1.0);
}
