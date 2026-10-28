// bevy_elements::grid — addressing and bilinear sampling of the crate's 2D `array<f32>` grids.
//
// A supported import for consumers: `import bevy_elements::grid::{GridParams, grid_taps};`.
// WGSL cannot pass a storage-buffer pointer to a function portably, so the module returns the
// four indices and weights and the caller does the four loads:
//
//     let t = grid_taps(params, world.xz);
//     let h = mix(mix(height[t.i00], height[t.i10], t.w.x), mix(height[t.i01], height[t.i11], t.w.x), t.w.y);

/// Placement of a row-major `size.x × size.y` grid whose cell (0, 0) centre sits at
/// `origin + 0.5 * cell`.
struct GridParams {
    /// World XZ of the grid's minimum corner, m.
    origin: vec2<f32>,
    /// Cell edge, m.
    cell: f32,
    _pad: f32,
    /// Cells per axis.
    size: vec2<u32>,
    _pad2: vec2<u32>,
}

/// Four row-major indices around a sample point and its bilinear weights.
struct GridTaps {
    i00: u32,
    i10: u32,
    i01: u32,
    i11: u32,
    w: vec2<f32>,
    /// Whether the point lies inside the grid; outside, the taps are clamped to the edge.
    inside: bool,
}

/// Row-major index of a clamped cell coordinate.
fn grid_index(params: GridParams, c: vec2<i32>) -> u32 {
    let m = vec2<i32>(params.size) - 1;
    let k = vec2<u32>(clamp(c, vec2<i32>(0), m));
    return k.y * params.size.x + k.x;
}

/// Bilinear taps for world XZ `p`.
fn grid_taps(params: GridParams, p: vec2<f32>) -> GridTaps {
    let g = (p - params.origin) / params.cell - 0.5;
    let base = vec2<i32>(floor(g));
    var t: GridTaps;
    t.i00 = grid_index(params, base);
    t.i10 = grid_index(params, base + vec2<i32>(1, 0));
    t.i01 = grid_index(params, base + vec2<i32>(0, 1));
    t.i11 = grid_index(params, base + vec2<i32>(1, 1));
    t.w = g - floor(g);
    let extent = vec2<f32>(params.size) * params.cell;
    let local = p - params.origin;
    t.inside = all(local >= vec2<f32>(0.0)) && all(local <= extent);
    return t;
}
