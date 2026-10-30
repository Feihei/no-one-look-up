//! The rocks' CPU model (`docs/research/rock-scatter.md`): where rocks lie (the cover field),
//! how big they are (Golombek's exponential law), where exactly (a jittered lattice thinned by
//! the cover), and the height of the surface the terrain's patches draw (the seat). Pure
//! functions; the systems in `mod.rs` call them.

use std::f32::consts::FRAC_PI_4;

use bevy::math::{IVec2, UVec2, Vec2, Vec3};

use crate::terrain::TerrainSnapshot;

/// The smallest rock drawn as a mesh, m: below it the scree's painted stones and the turf carry
/// the ground (Model 2).
pub(crate) const D_MIN_M: f32 = 0.25;
/// The largest, m.
pub(crate) const D_MAX_M: f32 = 3.6;
/// The size classes, m: octaves of diameter, each with its own lattice and ring (Model 2).
pub(crate) const CLASSES: [(f32, f32); 4] =
    [(D_MIN_M, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, D_MAX_M)];

/// The scree's band (`render.wesl`): where the talus lies, by slope in degrees.
const TALUS_RISE_DEG: [f32; 2] = [16.0, 26.0];
const TALUS_FALL_DEG: [f32; 2] = [31.0, 40.0];
/// The meadow's rocks thin out on slopes the talus takes over, and none rest on the rock.
const MEADOW_FALL_DEG: [f32; 2] = [30.0, 40.0];
/// A steeper hillside erodes faster, its rock spends less time weathering in the soil, and more
/// of it reaches the surface as rocks (Verdian et al. 2021): the meadow's cover rises with the
/// slope over this band, by this share of the way from the floor to the rocky patches' cover.
const MEADOW_STEEP_DEG: [f32; 2] = [10.0, 26.0];
const MEADOW_STEEP_SHARE: f32 = 0.6;
/// Sediment that buries rocks, m: the material's sand starts at 0.5 m.
const SAND_BURIES_M: [f32; 2] = [0.15, 0.5];
/// The rocky patches' field: its scale, m, and the threshold that leaves about a quarter of the
/// meadow rocky (Deviation 2).
const PATCH_SCALE_M: f32 = 220.0;
const PATCH_EDGE: [f32; 2] = [0.56, 0.70];
/// A cliff: the slope at which the ground is the material's rock (35°–45°, centred at 40°).
const CLIFF_DEG: [f32; 2] = [36.0, 44.0];
/// How far up the slope a cliff feeds its apron, m: the points the cliff test samples. A talus
/// reaches a few tens of metres below its cliff.
const CLIFF_REACH_M: [f32; 4] = [6.0, 14.0, 24.0, 36.0];
/// The rocks' clusters: the field's scale, m, and its range (mean 1).
const CLUSTER_SCALE_M: f32 = 18.0;
const CLUSTER_RANGE: [f32; 2] = [0.35, 1.65];

/// Golombek's `q(k) = 1.79 + 0.152 / k`: how abruptly the covered area falls with the diameter.
pub(crate) fn q_of(k: f32) -> f32 {
    1.79 + 0.152 / k.max(1e-4)
}

/// Golombek's cumulative fractional area `F_k(D) = k·exp(−q(k)·D)`: the fraction of the ground
/// covered by rocks of diameter `d` or larger, for a total cover `k`. The tests hold the scatter
/// to it.
#[cfg(test)]
fn cumulative_area(k: f32, d: f32) -> f32 {
    if k <= 0.0 {
        return 0.0;
    }
    k * (-q_of(k) * d).exp()
}

/// Intervals of the Simpson and trapezoid integrations over one class.
const STEPS: usize = 32;

/// Rocks per square metre with a diameter in `[a, b)` for a total cover `k`: the area density
/// `−dF/dD = k·q·exp(−q·D)` divided by the area of one rock, `π·D²/4`, integrated (Simpson).
pub(crate) fn count_density(k: f32, a: f32, b: f32) -> f32 {
    if k <= 0.0 || b <= a {
        return 0.0;
    }
    let q = q_of(k);
    let f = |d: f32| k * q * (-q * d).exp() / (FRAC_PI_4 * d * d);
    let h = (b - a) / STEPS as f32;
    let mut sum = f(a) + f(b);
    for i in 1..STEPS {
        let weight = if i % 2 == 1 { 4.0 } else { 2.0 };
        sum += weight * f(a + i as f32 * h);
    }
    sum * h / 3.0
}

/// A diameter in `[a, b)` drawn from the class's own distribution (`∝ exp(−q·D) / D²`) by
/// inverse transform of `u ∈ [0, 1)` on a tabulated cumulative.
pub(crate) fn draw_diameter(k: f32, a: f32, b: f32, u: f32) -> f32 {
    let q = q_of(k);
    let f = |d: f32| (-q * (d - a)).exp() / (d * d);
    let h = (b - a) / STEPS as f32;
    let mut cum = [0.0f32; STEPS + 1];
    for i in 1..=STEPS {
        let d0 = a + (i - 1) as f32 * h;
        cum[i] = cum[i - 1] + 0.5 * (f(d0) + f(d0 + h)) * h;
    }
    let target = u.clamp(0.0, 1.0) * cum[STEPS];
    let i = cum.partition_point(|&c| c < target).clamp(1, STEPS);
    let span = (cum[i] - cum[i - 1]).max(1e-12);
    let t = ((target - cum[i - 1]) / span).clamp(0.0, 1.0);
    a + ((i - 1) as f32 + t) * h
}

fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// The cover parameters, from [`RockSettings`](super::RockSettings).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct CoverParams {
    pub talus_cover: f32,
    pub meadow_floor: f32,
    pub meadow_cover: f32,
}

impl CoverParams {
    /// The largest total cover anywhere: the lattices are sized for it.
    pub(crate) fn max_cover(&self) -> f32 {
        (self.talus_cover * CLUSTER_RANGE[1])
            .max(self.meadow_cover)
            .max(self.meadow_floor)
    }
}

/// The ground at one point, from the terrain readback.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Ground {
    pub slope_deg: f32,
    pub sediment_m: f32,
    /// The surface's unit normal.
    pub normal: Vec3,
}

/// The ground at `p` (metre-frame XZ): the slope from central differences one readback cell
/// either side (the packed gradient's rule), the sediment, the normal. `None` outside the grid.
pub(crate) fn ground_at(snap: &TerrainSnapshot, p: Vec2) -> Option<Ground> {
    let sediment_m = snap.sediment(p)?;
    let dx = snap.grid.cell_m;
    let h = |q: Vec2| snap.height_clamped(q);
    let gx = (h(p + Vec2::X * dx) - h(p - Vec2::X * dx)) / (2.0 * dx);
    let gz = (h(p + Vec2::Y * dx) - h(p - Vec2::Y * dx)) / (2.0 * dx);
    let normal = Vec3::new(-gx, 1.0, -gz).normalize();
    Some(Ground {
        slope_deg: normal.y.clamp(-1.0, 1.0).acos().to_degrees(),
        sediment_m,
        normal,
    })
}

/// How much of the ground the rocks cover at one point (Model 1): the talus's share and the
/// meadow's, each a Golombek `k`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Cover {
    pub talus: f32,
    pub meadow: f32,
}

impl Cover {
    /// The total cover `k`.
    pub(crate) fn k(&self) -> f32 {
        self.talus.max(self.meadow)
    }

    /// Whether the rocks here are the talus's fresh, angular blocks (else the meadow's rounded,
    /// weathered ones).
    pub(crate) fn angular(&self) -> bool {
        self.talus > self.meadow
    }
}

/// The talus band (the scree's, `render.wesl`): 0–1 by slope.
pub(crate) fn talus_band(slope_deg: f32) -> f32 {
    smoothstep(TALUS_RISE_DEG[0], TALUS_RISE_DEG[1], slope_deg)
        * (1.0 - smoothstep(TALUS_FALL_DEG[0], TALUS_FALL_DEG[1], slope_deg))
}

/// How much cliff stands above `p`, 0–1: the steepest of a few points up the slope, as rock
/// (Model 1). A talus is the rock a cliff sheds, so it lies only below one: a steep grass hill
/// with nothing above it is not an apron.
pub(crate) fn cliff_above(snap: &TerrainSnapshot, p: Vec2, ground: &Ground) -> f32 {
    // The normal is (−∂h/∂x, 1, −∂h/∂z): uphill is the gradient, (−n.x, −n.z).
    let uphill = Vec2::new(-ground.normal.x, -ground.normal.z);
    let Some(dir) = uphill.try_normalize() else {
        return 0.0;
    };
    CLIFF_REACH_M
        .iter()
        .filter_map(|&s| ground_at(snap, p + dir * s))
        .map(|g| smoothstep(CLIFF_DEG[0], CLIFF_DEG[1], g.slope_deg))
        .fold(0.0, f32::max)
}

/// The rocks' clusters: a field at 18 m from 0.35 to 1.65 (mean 1), so an apron and a rocky
/// meadow have heaps and bare stretches instead of an even sprinkle (Model 3). Bedrock exposures
/// on a hillslope are patchy (Verdian et al. 2021), and the rocks gather round them.
pub(crate) fn rock_clusters(p: Vec2, seed: u64) -> f32 {
    let n = value_noise(p / CLUSTER_SCALE_M, seed ^ 0xC1);
    CLUSTER_RANGE[0] + (CLUSTER_RANGE[1] - CLUSTER_RANGE[0]) * n
}

/// The cover for the ground, the rocky-patch field's value (0–1), the cliff above (0–1) and the
/// clusters' field.
pub(crate) fn cover(
    params: &CoverParams,
    ground: &Ground,
    patches: f32,
    cliff: f32,
    clusters: f32,
) -> Cover {
    let a = ground.slope_deg;
    let meadow_slope = 1.0 - smoothstep(MEADOW_FALL_DEG[0], MEADOW_FALL_DEG[1], a);
    let rocky = ((patches
        + MEADOW_STEEP_SHARE * smoothstep(MEADOW_STEEP_DEG[0], MEADOW_STEEP_DEG[1], a))
        * clusters)
        .min(1.0);
    let meadow =
        (params.meadow_floor + (params.meadow_cover - params.meadow_floor) * rocky) * meadow_slope;
    let sand = 1.0 - smoothstep(SAND_BURIES_M[0], SAND_BURIES_M[1], ground.sediment_m);
    Cover {
        talus: params.talus_cover * talus_band(a) * cliff * clusters * sand,
        meadow: meadow * sand,
    }
}

/// A 64-bit mix (splitmix64's finaliser).
pub(crate) fn mix64(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// A unit float from 24 bits of `bits`.
pub(crate) fn unit(bits: u64) -> f32 {
    (bits & 0xFF_FFFF) as f32 / 16_777_216.0
}

/// The hash of a lattice cell.
fn cell_hash(seed: u64, salt: u64, c: IVec2) -> u64 {
    mix64(
        seed ^ salt.wrapping_mul(0x9E37_79B9_7F4A_7C15)
            ^ (c.x as i64 as u64).wrapping_mul(0xD6E8_FEB8_6659_FD93)
            ^ (c.y as i64 as u64).wrapping_mul(0xA076_1D64_78BD_642F),
    )
}

/// A smooth value noise in the plane, 0–1: the hashed lattice's values, quintic interpolation.
fn value_noise(p: Vec2, seed: u64) -> f32 {
    let base = p.floor();
    let f = p - base;
    let w = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    let c = base.as_ivec2();
    let v = |dx: i32, dz: i32| unit(cell_hash(seed, 0x51, c + IVec2::new(dx, dz)));
    let a = v(0, 0) + (v(1, 0) - v(0, 0)) * w.x;
    let b = v(0, 1) + (v(1, 1) - v(0, 1)) * w.x;
    a + (b - a) * w.y
}

/// The rocky patches on the meadow, 0–1: two octaves of value noise at 220 m, thresholded so
/// about a quarter of the meadow is rocky (Model 1, Deviation 2).
pub(crate) fn meadow_patches(p: Vec2, seed: u64) -> f32 {
    let q = p / PATCH_SCALE_M;
    let n = (2.0 * value_noise(q, seed) + value_noise(q * 2.03 + 17.3, seed ^ 0x7F)) / 3.0;
    smoothstep(PATCH_EDGE[0], PATCH_EDGE[1], n)
}

/// One size class's lattice: its cell (the footprint `w`, sized so the densest ground expects at
/// most one rock per cell), its tile (cells a side) and its ring's radius, m.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct ClassLattice {
    pub class: usize,
    pub cell_m: f32,
    pub tile_cells: i32,
    pub radius_m: f32,
}

impl ClassLattice {
    /// The lattice of `class` for the densest cover `k_max` and a ring of `radius_m`. An
    /// orthographic view has no distance at which a rock's screen size falls (`docs/research/
    /// orthographic-rocks.md` Model 1), so its ring is the view's own footprint rather than a
    /// multiple of the class's diameter.
    pub(crate) fn with_radius(class: usize, k_max: f32, radius_m: f32) -> Self {
        let (a, b) = CLASSES[class];
        let density = count_density(k_max.max(1e-3), a, b).max(1e-6);
        let cell_m = (1.0 / density).sqrt();
        let tile_cells = ((radius_m / (3.0 * cell_m)).round() as i32).max(1);
        Self {
            class,
            cell_m,
            tile_cells,
            radius_m,
        }
    }

    /// Edge of a tile, m.
    pub(crate) fn tile_m(&self) -> f32 {
        self.cell_m * self.tile_cells as f32
    }

    /// The tiles whose square comes within the ring around `centre` (metre-frame XZ).
    pub(crate) fn tiles_near(&self, centre: Vec2, margin_m: f32) -> Vec<IVec2> {
        let t = self.tile_m();
        let reach = self.radius_m + margin_m;
        let lo = ((centre - reach) / t).floor().as_ivec2();
        let hi = ((centre + reach) / t).floor().as_ivec2();
        let mut out = Vec::new();
        for z in lo.y..=hi.y {
            for x in lo.x..=hi.x {
                let tile = IVec2::new(x, z);
                if self.tile_distance(tile, centre) <= reach {
                    out.push(tile);
                }
            }
        }
        out
    }

    /// Distance from `centre` to the nearest point of a tile's square, m.
    pub(crate) fn tile_distance(&self, tile: IVec2, centre: Vec2) -> f32 {
        let t = self.tile_m();
        let lo = tile.as_vec2() * t;
        let nearest = centre.clamp(lo, lo + t);
        nearest.distance(centre)
    }
}

/// One rock, placed: everything but its height, which the seat gives.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Placed {
    /// Metre-frame XZ.
    pub xz: Vec2,
    pub diameter_m: f32,
    /// Index into the shapes (`mesh::SHAPES`): the angular family is the first half.
    pub shape: usize,
    /// Rotation about the rock's own up, rad.
    pub yaw: f32,
    /// The surface normal the rock rests on.
    pub normal: Vec3,
    /// A small tilt of its own, rad, about X and Z.
    pub jitter: Vec2,
    /// The fraction of its height below the surface.
    pub bury: f32,
}

/// Shapes per family.
pub(crate) const SHAPES_PER_FAMILY: usize = 6;
/// The rock's tilt towards the slope's normal (Deviation 6), and its own jitter, rad.
pub(crate) const TILT_TO_SLOPE: f32 = 0.7;
const TILT_JITTER_RAD: f32 = 0.14;
/// The buried fraction of the height (Deviation 5).
const BURY: [f32; 2] = [0.22, 0.35];

/// The rocks of one tile of one class, in a deterministic order: one candidate per lattice cell
/// at a hashed position, kept where its hashed threshold is under the cell's expected count
/// (HZD's threshold test, Model 3).
pub(crate) fn tile_rocks(
    snap: &TerrainSnapshot,
    params: &CoverParams,
    lattice: &ClassLattice,
    tile: IVec2,
    seed: u64,
) -> Vec<Placed> {
    let (a, b) = CLASSES[lattice.class];
    let n = lattice.tile_cells;
    let w = lattice.cell_m;
    let salt = 0xC1A5 + lattice.class as u64;
    let mut out = Vec::new();
    for j in 0..n {
        for i in 0..n {
            let cell = tile * n + IVec2::new(i, j);
            let h = cell_hash(seed, salt, cell);
            let draw = |k: u64| unit(mix64(h ^ k.wrapping_mul(0x2545_F491_4F6C_DD1D)));
            let xz = (cell.as_vec2() + Vec2::new(draw(1), draw(2))) * w;
            let Some(ground) = ground_at(snap, xz) else {
                continue;
            };
            let cliff = if talus_band(ground.slope_deg) > 0.0 {
                cliff_above(snap, xz, &ground)
            } else {
                0.0
            };
            let clusters = rock_clusters(xz, seed);
            let c = cover(params, &ground, meadow_patches(xz, seed), cliff, clusters);
            let k = c.k();
            if k <= 0.0 {
                continue;
            }
            let expected = count_density(k, a, b) * w * w;
            if draw(3) >= expected {
                continue;
            }
            let family = if c.angular() { 0 } else { SHAPES_PER_FAMILY };
            let shape = family + (draw(5) * SHAPES_PER_FAMILY as f32) as usize % SHAPES_PER_FAMILY;
            out.push(Placed {
                xz,
                diameter_m: draw_diameter(k, a, b, draw(4)),
                shape,
                yaw: draw(6) * std::f32::consts::TAU,
                normal: ground.normal,
                jitter: (Vec2::new(draw(7), draw(8)) * 2.0 - 1.0) * TILT_JITTER_RAD,
                bury: BURY[0] + (BURY[1] - BURY[0]) * draw(9),
            });
        }
    }
    out
}

/// What the seat needs of the terrain's level of detail: the simulation's cell, the levels'
/// morph ranges, and the level of the patch that covers each leaf. `None` for a terrain the crate
/// does not draw (an external one): the seat is then the readback's own surface.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct PatchLevels {
    /// The simulation grid's cell, m: level 0's lattice step.
    pub cell_m: f32,
    /// The patches' vertex lattice origin, metre frame: the grid's minimum corner plus half a
    /// cell, so level 0's nodes are the height samples.
    pub origin: Vec2,
    /// The coarsest level.
    pub top: u32,
    /// Per level: (morph start, range), m.
    pub morph: Vec<(f32, f32)>,
    /// Edge of a leaf patch, m, and leaves per axis.
    pub leaf_m: f32,
    pub leaves: UVec2,
    /// The level drawn over each leaf, row-major; `u8::MAX` where no patch is selected.
    pub level: Vec<u8>,
}

impl PatchLevels {
    /// The level drawn at `p` (metre-frame XZ), if a patch covers it.
    pub(crate) fn level_at(&self, p: Vec2) -> Option<u32> {
        let l = ((p - self.origin) / self.leaf_m).floor().as_ivec2();
        if l.x < 0 || l.y < 0 || l.x >= self.leaves.x as i32 || l.y >= self.leaves.y as i32 {
            return None;
        }
        let level = self.level[(l.y as u32 * self.leaves.x + l.x as u32) as usize];
        (level != u8::MAX).then_some(u32::from(level))
    }

    /// The morph factor of `level` at `distance` m (`morph_k` in `render.wesl`).
    pub(crate) fn morph(&self, level: u32, distance: f32) -> f32 {
        if level >= self.top {
            return 0.0;
        }
        let (start, range) = self.morph[level as usize];
        ((distance - start) / (range - start).max(1e-3)).clamp(0.0, 1.0)
    }
}

/// The height of a lattice of edge `step` (m) at `p`: the patch mesh's own triangles over the
/// readback's surface at the lattice's nodes (each quad split along its (1, 0)–(0, 1) diagonal,
/// as `patch_mesh` indexes it).
fn lattice_height(snap: &TerrainSnapshot, origin: Vec2, step: f32, p: Vec2) -> f32 {
    let g = (p - origin) / step;
    let base = g.floor();
    let f = g - base;
    let node = |dx: f32, dz: f32| snap.height_clamped(origin + (base + Vec2::new(dx, dz)) * step);
    if f.x + f.y <= 1.0 {
        let h00 = node(0.0, 0.0);
        h00 + f.x * (node(1.0, 0.0) - h00) + f.y * (node(0.0, 1.0) - h00)
    } else {
        let h11 = node(1.0, 1.0);
        h11 + (1.0 - f.x) * (node(0.0, 1.0) - h11) + (1.0 - f.y) * (node(1.0, 0.0) - h11)
    }
}

/// The two surfaces the terrain draws at `p` between which the camera's distance morphs: the
/// covering patch's level, its lattice's height and the next level's (metre frame). They depend
/// on the patch level and the surface alone, so a rock keeps them until either changes. Without
/// patch levels, the readback's own surface twice.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Lattices {
    pub level: u32,
    pub fine: f32,
    pub coarse: f32,
}

impl Lattices {
    /// The lattices at `p`.
    pub(crate) fn at(snap: &TerrainSnapshot, levels: Option<&PatchLevels>, p: Vec2) -> Self {
        let Some(levels) = levels else {
            let h = snap.height_clamped(p);
            return Self {
                level: 0,
                fine: h,
                coarse: h,
            };
        };
        let level = levels.level_at(p).unwrap_or(0);
        let step = levels.cell_m * (1u32 << level) as f32;
        Self {
            level,
            fine: lattice_height(snap, levels.origin, step, p),
            coarse: lattice_height(snap, levels.origin, step * 2.0, p),
        }
    }

    /// The height drawn at `p` for patches morphed about `centre` (the eye, or an orthographic
    /// camera's centre): the fine lattice blended to the coarse one by the vertex shader's morph
    /// factor.
    pub(crate) fn height(&self, levels: Option<&PatchLevels>, centre: Vec3, p: Vec2) -> f32 {
        let Some(levels) = levels else {
            return self.fine;
        };
        let distance = centre.distance(Vec3::new(p.x, self.fine, p.y));
        self.fine + (self.coarse - self.fine) * levels.morph(self.level, distance)
    }
}

/// The height (metre frame) of the surface the terrain draws at `p` for a camera at `eye` (the
/// systems keep a rock's [`Lattices`] and morph them; the tests compose the two).
#[cfg(test)]
fn drawn_height(snap: &TerrainSnapshot, levels: Option<&PatchLevels>, eye: Vec3, p: Vec2) -> f32 {
    Lattices::at(snap, levels, p).height(levels, eye, p)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::terrain::TerrainGrid;

    fn params() -> CoverParams {
        CoverParams {
            talus_cover: 0.30,
            meadow_floor: 0.015,
            meadow_cover: 0.05,
        }
    }

    /// A surface `h(x)` along +X, the same along Z, with `sediment` m of sediment: 512 cells of
    /// 2 m centred on the origin.
    fn profile(h: impl Fn(f32) -> f32, sediment: f32) -> TerrainSnapshot {
        let n = 512u32;
        let cell = 2.0;
        let origin = -(n as f32) * cell * 0.5;
        let surface: Vec<f32> = (0..n * n)
            .map(|i| h(origin + ((i % n) as f32 + 0.5) * cell))
            .collect();
        TerrainSnapshot {
            grid: TerrainGrid {
                origin: Vec2::splat(origin),
                cell_m: cell,
                size: UVec2::splat(n),
                y0: 0.0,
            },
            surface: Arc::from(surface),
            sediment: Arc::from(vec![sediment; (n * n) as usize]),
            version: 1,
        }
    }

    /// A plane tilted `slope_deg` along +X.
    fn plane(slope_deg: f32, sediment: f32) -> TerrainSnapshot {
        let g = slope_deg.to_radians().tan();
        profile(|x| x * g, sediment)
    }

    /// A 60° cliff above x = 0 with a 26° slope below it.
    fn cliff_and_apron() -> TerrainSnapshot {
        let (apron, cliff) = (26f32.to_radians().tan(), 60f32.to_radians().tan());
        profile(|x| if x < 0.0 { x * apron } else { x * cliff }, 0.0)
    }

    fn rocks_in(
        snap: &TerrainSnapshot,
        params: &CoverParams,
        class: usize,
        half: i32,
    ) -> Vec<Placed> {
        let lattice =
            ClassLattice::with_radius(class, params.max_cover(), 100.0 * CLASSES[class].1);
        let mut all = Vec::new();
        for z in -half..half {
            for x in -half..half {
                all.extend(tile_rocks(snap, params, &lattice, IVec2::new(x, z), 7));
            }
        }
        all
    }

    #[test]
    fn the_scatter_reproduces_the_cumulative_area() {
        // A uniform cover of k = 0.30 (a flat meadow whose floor is its rocky cover). Measure the
        // fraction of the ground covered by rocks of D ≥ 1 m (classes 2 and 3) over a large patch
        // against k·exp(−q·D).
        let k = 0.30;
        let uniform = CoverParams {
            talus_cover: 0.0,
            meadow_floor: k,
            meadow_cover: k,
        };
        let snap = plane(2.0, 0.0);
        let mut measured = 0.0;
        for class in [2usize, 3] {
            let lattice =
                ClassLattice::with_radius(class, uniform.max_cover(), 100.0 * CLASSES[class].1);
            let half = (300.0 / lattice.tile_m()).ceil() as i32;
            let covered: f32 = rocks_in(&snap, &uniform, class, half)
                .iter()
                .map(|r| FRAC_PI_4 * r.diameter_m * r.diameter_m)
                .sum();
            measured += covered / (2.0 * half as f32 * lattice.tile_m()).powi(2);
        }
        let want = cumulative_area(k, 1.0) - cumulative_area(k, D_MAX_M);
        assert!(
            (measured - want).abs() < 0.25 * want,
            "rocks of 1 m and up cover {measured}, the law says {want}"
        );
    }

    #[test]
    fn no_rock_on_the_cliff_or_the_sand_and_the_talus_is_dense() {
        let class = 1;
        let p = params();
        assert!(
            rocks_in(&plane(45.0, 0.0), &p, class, 4).is_empty(),
            "rocks on a 45° face"
        );
        assert!(
            rocks_in(&plane(26.0, 0.8), &p, class, 4).is_empty(),
            "rocks on 0.8 m of sand"
        );
        assert!(
            rocks_in(&plane(2.0, 0.8), &p, class, 4).is_empty(),
            "rocks on a sand flat"
        );
        // Below a cliff the talus is a boulder field and the flat meadow away from its rocky
        // patches is not: per square metre, rocks of class 1 are at least ten times as many.
        let ground = |slope_deg: f32| Ground {
            slope_deg,
            sediment_m: 0.0,
            normal: Vec3::Y,
        };
        let (a, b) = CLASSES[class];
        let talus = count_density(cover(&p, &ground(26.0), 0.0, 1.0, 1.0).k(), a, b);
        let meadow = count_density(cover(&p, &ground(2.0), 0.0, 0.0, 1.0).k(), a, b);
        assert!(
            talus >= 10.0 * meadow,
            "the talus holds {talus} rocks a square metre, the meadow {meadow}"
        );
        // A steep meadow holds more than a flat one.
        let steep = count_density(cover(&p, &ground(22.0), 0.0, 0.0, 1.0).meadow, a, b);
        assert!(steep > 3.0 * meadow, "steep {steep} vs flat {meadow}");
    }

    #[test]
    fn a_talus_lies_below_a_cliff_and_nowhere_else() {
        // A 26° slope under a 60° cliff: the slope's top 30 m is the cliff's apron, of angular
        // blocks; 70 m and more below the cliff, out of its reach, the same slope is a meadow of
        // rounded rocks, and much sparser.
        let rocks = rocks_in(&cliff_and_apron(), &params(), 1, 4);
        let apron: Vec<_> = rocks
            .iter()
            .filter(|r| (-30.0..-3.0).contains(&r.xz.x))
            .collect();
        let below: Vec<_> = rocks
            .iter()
            .filter(|r| (-130.0..-70.0).contains(&r.xz.x))
            .collect();
        assert!(!apron.is_empty() && apron.iter().all(|r| r.shape < SHAPES_PER_FAMILY));
        assert!(below.iter().all(|r| r.shape >= SHAPES_PER_FAMILY));
        // Per metre of slope: the apron's strip is 27 m wide, the strip below 60 m.
        let (apron_d, below_d) = (apron.len() as f32 / 27.0, below.len() as f32 / 60.0);
        assert!(
            apron_d > 5.0 * below_d,
            "apron {apron_d} vs below {below_d} a metre"
        );
        // A uniform 26° hillside with no cliff above is no apron.
        assert!(
            rocks_in(&plane(26.0, 0.0), &params(), 1, 4)
                .iter()
                .all(|r| r.shape >= SHAPES_PER_FAMILY)
        );
    }

    #[test]
    fn the_scatter_is_not_a_lattice() {
        // Clark–Evans: the mean nearest-neighbour distance over the expected for a Poisson scatter
        // of the same density, 1/(2√ρ). A lattice reads 2.15; a clumped field reads below 1.
        let snap = plane(26.0, 0.0);
        let class = 0;
        let lattice =
            ClassLattice::with_radius(class, params().max_cover(), 100.0 * CLASSES[class].1);
        let rocks = rocks_in(&snap, &params(), class, 3);
        let side = 6.0 * lattice.tile_m();
        let rho = rocks.len() as f32 / (side * side);
        // Skip the border so an edge rock's missing neighbours do not bias it.
        let inner: Vec<_> = rocks
            .iter()
            .filter(|r| r.xz.abs().max_element() < side * 0.35)
            .collect();
        let mean: f32 = inner
            .iter()
            .map(|r| {
                rocks
                    .iter()
                    .filter(|o| o.xz != r.xz)
                    .map(|o| o.xz.distance(r.xz))
                    .fold(f32::MAX, f32::min)
            })
            .sum::<f32>()
            / inner.len() as f32;
        let ratio = mean / (0.5 / rho.sqrt());
        assert!(
            (0.85..1.6).contains(&ratio),
            "Clark–Evans ratio {ratio} over {} rocks",
            inner.len()
        );
    }

    #[test]
    fn a_tile_built_twice_holds_the_same_rocks() {
        let snap = plane(26.0, 0.0);
        let lattice = ClassLattice::with_radius(1, params().max_cover(), 100.0 * CLASSES[1].1);
        let a = tile_rocks(&snap, &params(), &lattice, IVec2::new(3, -2), 7);
        let b = tile_rocks(&snap, &params(), &lattice, IVec2::new(3, -2), 7);
        assert!(!a.is_empty());
        assert_eq!(a, b);
        let other = tile_rocks(&snap, &params(), &lattice, IVec2::new(3, -2), 8);
        assert_ne!(a, other, "the seed moves the rocks");
    }

    #[test]
    fn the_seat_is_the_surface_the_patches_draw() {
        // A bumpy surface: level 0's lattice is the readback's own samples at the nodes; a
        // coarser level differs, and at the end of its morph a level is the next one. The bumps
        // are a product of the two axes, so a quad's two triangles are not its bilinear.
        let n = 256u32;
        let cell = 2.0;
        let surface: Vec<f32> = (0..n * n)
            .map(|i| {
                let (x, z) = ((i % n) as f32, (i / n) as f32);
                3.0 * (x * 0.9).sin() * (z * 0.7).cos() + 2.0 * (z * 0.7).cos()
            })
            .collect();
        let snap = TerrainSnapshot {
            grid: TerrainGrid {
                origin: Vec2::ZERO,
                cell_m: cell,
                size: UVec2::splat(n),
                y0: 0.0,
            },
            surface: Arc::from(surface),
            sediment: Arc::from(vec![0.0; (n * n) as usize]),
            version: 1,
        };
        let leaves = UVec2::splat(n / 32);
        // The patches' lattice sits on the samples, half a cell in from the grid's corner.
        let mut levels = PatchLevels {
            cell_m: cell,
            origin: Vec2::splat(0.5 * cell),
            top: 3,
            morph: vec![(90.0, 128.0), (200.0, 256.0), (400.0, 512.0)],
            leaf_m: 32.0 * cell,
            leaves,
            level: vec![0; (leaves.x * leaves.y) as usize],
        };
        let p = Vec2::new(201.3, 117.9);
        let near = Vec3::new(p.x, 0.0, p.y + 5.0);
        // At a lattice node, level 0 is the stored sample itself: cell (100, 58)'s centre.
        let node = Vec2::new(201.0, 117.0);
        let sample = snap.surface[(58 * n + 100) as usize];
        assert!((drawn_height(&snap, Some(&levels), near, node) - sample).abs() < 1e-4);
        // Inside a quad it is the triangle's plane, not the bilinear sample: they differ.
        let tri = drawn_height(&snap, Some(&levels), near, p);
        assert!((tri - snap.height_clamped(p)).abs() > 1e-4);
        // On a level-1 patch far from the eye (morph 1), the surface is level 2's lattice.
        levels.level.iter_mut().for_each(|l| *l = 1);
        let far = Vec3::new(p.x, 0.0, p.y + 300.0);
        let morphed = drawn_height(&snap, Some(&levels), far, p);
        levels.level.iter_mut().for_each(|l| *l = 2);
        let next = drawn_height(&snap, Some(&levels), Vec3::new(p.x, 0.0, p.y + 1.0), p);
        assert!((morphed - next).abs() < 1e-4, "{morphed} vs {next}");
        // Without levels (a terrain the crate does not draw), the readback's surface.
        assert_eq!(drawn_height(&snap, None, near, p), snap.height_clamped(p));
    }

    #[test]
    fn about_a_quarter_of_the_meadow_is_rocky() {
        let mut rocky = 0;
        let total = 200 * 200;
        for j in 0..200 {
            for i in 0..200 {
                let p = Vec2::new(i as f32, j as f32) * 25.0;
                if meadow_patches(p, 7) > 0.5 {
                    rocky += 1;
                }
            }
        }
        let share = rocky as f32 / total as f32;
        assert!((0.12..0.38).contains(&share), "rocky share {share}");
    }
}
