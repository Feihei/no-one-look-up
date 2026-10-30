//! Paths and roads: routes a game lays across the terrain, worn into its own ground
//! (`docs/research/paths.md`).
//!
//! A [`Path`] is a polyline with a half width, a kind and a rut depth. The crate rasterises the
//! network once, when it changes, into a **coverage distance field** — per texel the signed
//! distance to the nearest path's edge, its kind and rut depth, the lateral offset from its centre
//! line and the edge fade — which the terrain and ground materials read at every fragment: grass
//! gives way to bare, compacted earth over the path's edge, the turf's clods flatten, and a
//! vehicle's ruts are pressed along the route's wheel lines. The distance field is what keeps a
//! 70 cm footpath's edge smooth on a mask of a few texels a metre (Green 2007; RTR4 §15.4.3).
//!
//! The routes are the game's: nothing here decides where a road should go. Wear that traffic adds
//! stays in the trails field, which the weather fills and smooths; a path's mask is the route's own
//! and persists.

use bevy::asset::RenderAssetUsages;
use bevy::image::{ImageAddressMode, ImageFilterMode, ImageSampler, ImageSamplerDescriptor};
use bevy::prelude::*;
use bevy::render::render_resource::{Extent3d, TextureDataOrder, TextureDimension, TextureFormat};
use bevy::tasks::{AsyncComputeTaskPool, Task, block_on, poll_once};

use crate::plugin::{ElementsSystems, ensure_core};
use crate::terrain::TerrainState;

/// What a path is surfaced with.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Reflect)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum PathKind {
    /// A trodden lane: dry bare earth, the turf thinning over its edge.
    #[default]
    Earth,
    /// Gravel: pale stone chips, a fine even grain.
    Gravel,
    /// Laid stone: pale and cool, in courses.
    Stone,
    /// A packed snow track: the snow pressed shallower and denser, its colour unchanged.
    Snow,
}

impl PathKind {
    /// The kind as the mask's lane stores it.
    fn packed(self) -> f32 {
        match self {
            Self::Earth => 0.0,
            Self::Gravel => 1.0,
            Self::Stone => 2.0,
            Self::Snow => 3.0,
        }
    }
}

/// A route across the terrain.
///
/// Its points are world XZ (the Y is ignored: a path follows the terrain's own surface, not the
/// plane it was drawn on). A path is drawn, never a mesh: the terrain's material reads the path
/// mask at each fragment, so the surface follows every hill at the terrain's own resolution, and a
/// terrain with no path pays nothing.
#[derive(Clone, Debug, PartialEq, Reflect)]
pub struct Path {
    /// The route, at least two points, as world XZ in the [`ElementsUnits`](crate::ElementsUnits)
    /// frame (world ÷ units; the Y is ignored).
    pub points: Vec<Vec3>,
    /// Half the path's width, m: 0.35 for a footpath, 1.0 for a cart track, 2.5 for a road.
    pub half_width_m: f32,
    /// What it is surfaced with.
    pub kind: PathKind,
    /// The depth the route's ruts are pressed to, m; 0 for a route without ruts.
    pub rut_depth_m: f32,
}

impl Path {
    /// A path along `points` (world XZ), `half_width_m` either side of the line, without ruts.
    pub fn new(points: impl IntoIterator<Item = Vec3>, half_width_m: f32, kind: PathKind) -> Self {
        Self {
            points: points.into_iter().collect(),
            half_width_m,
            kind,
            rut_depth_m: 0.0,
        }
    }

    /// The same path with ruts `depth_m` deep along its wheel lines.
    pub fn with_ruts(mut self, depth_m: f32) -> Self {
        self.rut_depth_m = depth_m.max(0.0);
        self
    }

    /// Half width, clamped to what a mask can hold, m.
    fn width(&self) -> f32 {
        if self.half_width_m.is_finite() {
            self.half_width_m.clamp(0.01, 64.0)
        } else {
            0.01
        }
    }

    /// Rut depth, clamped, m.
    fn depth(&self) -> f32 {
        if self.rut_depth_m.is_finite() {
            self.rut_depth_m.clamp(0.0, 0.49)
        } else {
            0.0
        }
    }
}

/// The routes over the terrain, published by the game. Insert it or change it at any time: the
/// mask is rasterised off the frame on Bevy's `AsyncComputeTaskPool`, and the previous mask keeps
/// rendering until the new one is ready (`docs/research/paths.md`).
#[derive(Resource, Clone, Debug, Default, PartialEq, Reflect)]
#[reflect(Resource)]
pub struct PathNetwork {
    /// The routes, in any order: where two overlap the nearer to a texel wins.
    pub paths: Vec<Path>,
    /// Texels a side of the mask (a power of two, 16-4096). Default 0: the narrowest route
    /// decides, at least four texels across it.
    pub resolution: u32,
}

impl PathNetwork {
    /// The network of `paths`.
    pub fn new(paths: impl IntoIterator<Item = Path>) -> Self {
        Self {
            paths: paths.into_iter().collect(),
            resolution: 0,
        }
    }

    /// Whether there is anything to draw.
    fn is_empty(&self) -> bool {
        self.paths.iter().all(|p| p.points.len() < 2)
    }
}

/// The path mask the materials read, once it has been rasterised.
#[derive(Resource, Clone, Debug)]
pub struct PathState {
    /// The mask: `rgba8unorm` — r the **signed distance** to the nearest route's edge over
    /// `PATH_DISTANCE_RANGE_M` (0.5 at the edge, lower inside), g the kind plus the rut depth in
    /// its fraction, b the lateral offset from the centre line as a fraction of the half width, a
    /// a per-texel grain. Sampled bilinearly.
    pub mask: Handle<Image>,
    /// The mask's minimum corner, metre frame.
    pub origin: Vec2,
    /// Texel edge, m.
    pub texel_m: f32,
    /// Texels a side.
    pub texels: u32,
    /// The narrowest route's half width, m: the shader's fade and cross-section scale (a route's
    /// own width is not in the mask, because all four of its channels carry the field).
    pub half_width_m: f32,
}

/// The signed distance the mask's red channel spans either side of the edge, m: an 8-bit field
/// resolves about 8 cm of it, which is finer than the edge fade.
pub const PATH_DISTANCE_RANGE_M: f32 = 8.0;



/// The least edge fade a path's coverage uses, m: the turf thins over this much ground at a path's
/// edge. On a coarse mask it is also what antialiases the edge.
pub const PATH_EDGE_FADE_M: f32 = 0.12;

/// A path's edge fade from its half width, m: a third of the width — a lane's verge is narrow, a
/// road's is a wide gradual thing. Never less than [`PATH_EDGE_FADE_M`].
pub fn edge_fade_m(half_width_m: f32) -> f32 {
    (half_width_m * 0.33).max(PATH_EDGE_FADE_M)
}

/// The mask's texels a side for `paths` over an `extent_m` terrain: `resolution` when given, else
/// the finest texel the **narrowest** route needs — four texels across its full width, never finer
/// than half a metre a texel.
///
/// A route narrower than a texel cannot be drawn at all: a 70 cm footpath at a metre a texel is a
/// fraction of one sample, and it renders as a smear or a staircase.
pub(crate) fn mask_resolution(paths: &[Path], extent_m: f32, resolution: u32) -> u32 {
    if resolution >= 16 {
        return resolution.clamp(16, 4096).next_power_of_two().min(4096);
    }
    let narrowest = paths
        .iter()
        .filter(|p| p.points.len() >= 2)
        .map(|p| p.width())
        .fold(f32::INFINITY, f32::min);
    let want = if narrowest.is_finite() {
        (0.5 * narrowest).min(1.0)
    } else {
        1.0
    };
    ((extent_m / want).clamp(16.0, 4096.0).ceil() as u32)
        .next_power_of_two()
        .min(4096)
}

/// The rasterised mask, CPU-side.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct PathMask {
    pub texels: u32,
    pub texel_m: f32,
    pub origin: Vec2,
    /// The narrowest route's half width, m.
    pub half_width_m: f32,
    /// Per texel, row-major: (distance, kind + depth / 0.5, lateral fraction, grain 0-1).
    pub texels_data: Vec<[f32; 4]>,
}

/// The coverage a route's distance draws, which the shader and
/// [`Environment::surface_at`](crate::Environment::surface_at) both use so the picture and the
/// query cannot disagree: 1 well inside the route, 0 outside its edge, a smoothstep over the fade
/// **inward from the edge**. The signed distance is negative inside, so the coverage grows as the
/// distance falls below zero: `smoothstep(0, -fade, distance)`.
pub(crate) fn coverage(distance_m: f32, fade_m: f32) -> f32 {
    smoothstep(0.0, -fade_m.max(1e-3), distance_m)
}

/// How deep in a wheel line a point lies, 0-1, from its lateral offset as a fraction of the route's
/// half width: the shader's Gaussian about each wheel line (`PATH_RUT_OFFSET` and
/// `PATH_RUT_SIGMA` of `paths.wesl`, these same constants).
pub(crate) fn rut_at(lateral_fraction: f32) -> f32 {
    /// Where the wheels run, as a fraction of the half width (a cart's 1.5 m track on a 1 m half
    /// width, the `trails` example's `ELEMENTS_PATHS=track`).
    const RUT_OFFSET: f32 = 0.6;
    /// The band's width, in the same units.
    const RUT_SIGMA: f32 = 0.10;
    let wheel = (lateral_fraction.abs() - RUT_OFFSET) / RUT_SIGMA;
    if wheel.abs() > 2.5 {
        return 0.0;
    }
    (-0.5 * wheel * wheel).exp()
}

#[cfg(test)]
impl PathMask {
    /// A kind lane's integer part (the kind) and its fraction (the rut depth, m).
    pub(crate) fn kind_and_depth(packed: f32) -> (f32, f32) {
        let kind = packed.floor();
        (kind, (packed - kind) * 0.5)
    }
}

/// Rasterise `paths` into a mask over a square `extent` whose corner is `origin` (metre frame), at
/// `texels` a side and `texel_m` a texel.
///
/// Each texel takes the nearest route's values; a route is swept as capsules between its points,
/// so a diagonal route's edge is as smooth as an axial one. Deterministic.
pub(crate) fn rasterise(paths: &[Path], origin: Vec2, texel_m: f32, texels: u32) -> PathMask {
    let mut data = vec![[f32::INFINITY, 0.0, 0.0, 0.5]; (texels * texels) as usize];
    for path in paths {
        if path.points.len() < 2 {
            continue;
        }
        let half = path.width();
        let fade = edge_fade_m(half);
        let kind = path.kind.packed() + path.depth() / 0.5;
        let margin = half + fade + 2.0 * texel_m;
        let (mut lo, mut hi) = (Vec2::splat(f32::INFINITY), Vec2::splat(f32::NEG_INFINITY));
        for p in &path.points {
            lo = lo.min(p.xz());
            hi = hi.max(p.xz());
        }
        let to_texel = |p: Vec2| (p - origin) / texel_m;
        let (lo, hi) = (to_texel(lo - Vec2::splat(margin)), to_texel(hi + Vec2::splat(margin)));
        let (x0, y0) = (lo.x.floor() as i32, lo.y.floor() as i32);
        let (x1, y1) = (
            (hi.x.ceil() as i32).min(texels as i32 - 1),
            (hi.y.ceil() as i32).min(texels as i32 - 1),
        );
        for ty in y0.max(0)..=y1 {
            for tx in x0.max(0)..=x1 {
                let p = origin + (Vec2::new(tx as f32, ty as f32) + 0.5) * texel_m;
                let (d, lateral) = route_distance(&path.points, p, half);
                let i = (ty as u32 * texels + tx as u32) as usize;
                if d < data[i][0] {
                    // The grain: a per-texel value at the mask's own resolution, which for a
                    // footpath is 0.175 m — the scale of a road's chips. Two hashes give a
                    // speckle with a clump, so a road keeps a surface past the band limit that
                    // drops the noise octaves (docs/research/paths.md, Deviation 6a).
                    let grain = grain_at(p, half);
                    data[i] = [d, kind, (lateral / half).clamp(-1.0, 1.0), grain];
                }
            }
        }
    }
    PathMask {
        texels,
        texel_m,
        origin,
        half_width_m: paths
            .iter()
            .filter(|p| p.points.len() >= 2)
            .map(|p| p.width())
            .fold(f32::INFINITY, f32::min)
            .clamp(0.01, 64.0),
        texels_data: data,
    }
}

/// A deterministic per-texel grain at world XZ `p`, 0-1: the mean of a fine hash and a coarser
/// one at a route's own width, so a road has both speckle and clumps.
fn grain_at(p: Vec2, half_width: f32) -> f32 {
    let hash = |v: Vec2| {
        // A 2D integer hash (the crate's noise is a shader import): cheap, deterministic and
        // decorrelated between neighbouring texels.
        let x = v.x as i32 as u32;
        let y = v.y as i32 as u32;
        let h = x.wrapping_mul(0x27d4_eb2d) ^ y.wrapping_mul(0x1656_67b1);
        let h = h.wrapping_mul(0x27d4_eb2d);
        ((h >> 8) & 0xFFFF) as f32 / 65535.0
    };
    let fine = hash((p / (half_width * 0.16).max(0.02)).floor());
    let clump = hash((p / (half_width * 0.55).max(0.06)).floor() + Vec2::new(37.0, 11.0));
    0.6 * fine + 0.4 * clump
}

/// The signed distance from `p` to the nearest route edge (negative inside) and the signed lateral
/// offset from that route's centre line, positive to its left, both m.
fn route_distance(points: &[Vec3], p: Vec2, half_width: f32) -> (f32, f32) {
    let mut best = f32::INFINITY;
    let mut lateral = 0.0;
    for pair in points.windows(2) {
        let (a, b) = (pair[0].xz(), pair[1].xz());
        let ab = b - a;
        let len2 = ab.length_squared();
        let t = if len2 > 1e-12 {
            ((p - a).dot(ab) / len2).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let nearest = a + ab * t;
        let d = p.distance(nearest);
        if d < best {
            best = d;
            let n = Vec2::new(-ab.y, ab.x) / len2.sqrt().max(1e-12);
            lateral = (p - nearest).dot(n);
        }
    }
    (best - half_width, lateral)
}

/// What the ground under a point is, on or near a route ([`Environment::surface_at`],
/// [`Environment::nearest_path`](crate::Environment::nearest_path)).
///
/// The classification a game maps to its own responses — footstep and tyre sounds, movement speed,
/// dust, a print's depth — as shipping engines do with *collision materials* or Unreal's *physical
/// materials* (Gregory 2018, `docs/research/path-query.md`). The crate supplies the surface and the
/// geometry; the assets are the game's.
///
/// [`Environment::surface_at`]: crate::Environment::surface_at
#[derive(Clone, Copy, Debug, PartialEq, Reflect)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PathSurface {
    /// The nearest route's surface.
    pub kind: PathKind,
    /// 1 well inside the route, 0 outside its edge: the shader's own smoothstep over the fade, so
    /// a query at a verge answers what the picture shows there.
    pub coverage: f32,
    /// The route's own half width, m.
    pub half_width_m: f32,
    /// The route's rut depth, m (0 without ruts).
    pub rut_depth_m: f32,
    /// How deep in a wheel line the point lies, 0-1: the shader's Gaussian about the wheel lines at
    /// 0.6 of the half width, so a footstep in a rut can be answered as one.
    pub rut: f32,
}

/// The nearest route to a point ([`Environment::nearest_path`](crate::Environment::nearest_path)).
#[derive(Clone, Copy, Debug, PartialEq, Reflect)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PathHit {
    /// The surface at the point, against that route.
    pub surface: PathSurface,
    /// The signed distance from the point to the route's **edge**, m: negative inside the route,
    /// positive outside it. To the edge and not the centre line, so a game placing a prop 3 m clear
    /// of a road measures what it means.
    pub distance_m: f32,
    /// The point of the route's centre line nearest the query, metre-frame XZ.
    pub point: Vec2,
}

/// The route nearest to `p` (metre-frame XZ), with the surface there, or `None` with no routes.
///
/// The same segment maths the rasteriser uses, so the mask and the answer cannot disagree, and the
/// same coverage and rut curves. Exact and current: the network is the game's own CPU data.
pub(crate) fn nearest(paths: &[Path], p: Vec2) -> Option<PathHit> {
    let mut best: Option<(f32, &Path, Vec2, f32)> = None;
    for path in paths.iter().filter(|p| p.points.len() >= 2) {
        let (edge, point, lateral) = route_query(path, p);
        if best.is_none_or(|(d, _, _, _)| edge < d) {
            best = Some((edge, path, point, lateral));
        }
    }
    let (edge, path, point, lateral) = best?;
    let half = path.width();
    Some(PathHit {
        surface: PathSurface {
            kind: path.kind,
            coverage: coverage(edge, edge_fade_m(half)),
            half_width_m: half,
            rut_depth_m: path.depth(),
            rut: rut_at(lateral),
        },
        distance_m: edge,
        point,
    })
}

/// The signed distance from `p` to `path`'s edge (negative inside), the nearest point of its centre
/// line, and `p`'s lateral offset from that line as a fraction of the half width.
fn route_query(path: &Path, p: Vec2) -> (f32, Vec2, f32) {
    let half = path.width();
    let mut best = f32::INFINITY;
    let mut nearest_point = path.points[0].xz();
    let mut lateral = 0.0;
    for pair in path.points.windows(2) {
        let (a, b) = (pair[0].xz(), pair[1].xz());
        let ab = b - a;
        let len2 = ab.length_squared();
        let t = if len2 > 1e-12 {
            ((p - a).dot(ab) / len2).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let on = a + ab * t;
        let d = p.distance(on);
        if d < best {
            best = d;
            nearest_point = on;
            // The segment's leftward normal, against the offset: positive to the route's left.
            let n = Vec2::new(-ab.y, ab.x) / len2.sqrt().max(1e-12);
            lateral = (p - on).dot(n) / half;
        }
    }
    (best - half, nearest_point, lateral)
}

/// The mask as an `rgba16float` image, sampled linearly (the distance field must interpolate).
pub(crate) fn mask_image(mask: &PathMask) -> Image {
    // `Rgba8Unorm`, the format the terrain's own ground-texture array uses (iteration 50) — the one
    // CPU-filled image this material is proven to sample. An `rgba16float` mask read 0.0 on the GPU
    // in every configuration tried in iteration 53 (formats, usages, a constant field, either size).
    //
    // The channels, 8-bit:
    //   r: the signed edge distance over ±PATH_DISTANCE_RANGE_M, 0.5 at the edge
    //   g: the kind plus the rut depth in the fraction (m / 0.5)
    //   b: the lateral offset as a fraction of the half width, -1..1
    //   a: the per-texel grain, 0-1 (the near field's stone)
    let mut bytes = Vec::with_capacity(mask.texels_data.len() * 4);
    for texel in &mask.texels_data {
        let r = encode(texel[0] / PATH_DISTANCE_RANGE_M, 0.5);
        let g = (texel[1].clamp(0.0, 1.0) * 255.0).round() as u8;
        let b = encode(0.5 * texel[2], 0.5);
        let a = (texel[3].clamp(0.0, 1.0) * 255.0).round() as u8;
        bytes.extend_from_slice(&[r, g, b, a]);
    }
    let mut image = Image::new(
        Extent3d {
            width: mask.texels,
            height: mask.texels,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        bytes,
        TextureFormat::Rgba8Unorm,
        RenderAssetUsages::default(),
    );
    image.data_order = TextureDataOrder::LayerMajor;
    image.sampler = ImageSampler::Descriptor(ImageSamplerDescriptor {
        address_mode_u: ImageAddressMode::ClampToEdge,
        address_mode_v: ImageAddressMode::ClampToEdge,
        mag_filter: ImageFilterMode::Linear,
        min_filter: ImageFilterMode::Linear,
        ..default()
    });
    image
}

/// A signed value scaled by `range` and offset by `bias`, in 0..1, as 8 bits.
fn encode(value: f32, bias: f32) -> u8 {
    ((bias + value).clamp(0.0, 1.0) * 255.0).round() as u8
}

/// The mask's channels back to the values the shader works in, from 8-bit texels.
#[cfg(test)]
pub(crate) fn decode(texel: [u8; 4]) -> [f32; 4] {
    [
        (f32::from(texel[0]) / 255.0 - 0.5) * PATH_DISTANCE_RANGE_M,
        f32::from(texel[1]) / 255.0,
        (f32::from(texel[2]) / 255.0 - 0.5) * 2.0,
        f32::from(texel[3]) / 255.0,
    ]
}

/// GLSL/WGSL `smoothstep`, the curve [`coverage`] and the shader share.
fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    if (edge1 - edge0).abs() < 1e-9 {
        return if x < edge0 { 0.0 } else { 1.0 };
    }
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Lays the game's [`PathNetwork`] into the terrain's ground (`docs/research/paths.md`).
///
/// Idle without a terrain or with an empty network. Insert or change a [`PathNetwork`] to give the
/// terrain its routes. The mask is rasterised off the frame and published as [`PathState`], which
/// the terrain's and a game's ground material read.
pub struct PathsPlugin;

/// What the current mask was built from, so a change rasterises once.
#[derive(Resource, Default)]
struct PathRuntime {
    built: Option<(PathNetwork, Option<(Vec2, f32)>)>,
    task: Option<Task<PathMask>>,
}

impl Plugin for PathsPlugin {
    fn build(&self, app: &mut App) {
        ensure_core(app);
        app.init_resource::<PathNetwork>()
            .init_resource::<PathRuntime>()
            .add_systems(Update, rasterise_paths.in_set(ElementsSystems::Simulation));
    }
}

/// Rasterise the network into the mask when it or the terrain's extent changed, off the frame; the
/// previous mask keeps rendering until the new one lands.
fn rasterise_paths(
    network: Res<PathNetwork>,
    terrain: Option<Res<TerrainState>>,
    mut runtime: ResMut<PathRuntime>,
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
) {
    if let Some(task) = runtime.task.as_mut()
        && let Some(mask) = block_on(poll_once(task))
    {
        commands.insert_resource(PathState {
            mask: images.add(mask_image(&mask)),
            origin: mask.origin,
            texel_m: mask.texel_m,
            texels: mask.texels,
            half_width_m: mask.half_width_m,
        });
        runtime.task = None;
    }
    let extent = terrain.as_ref().map(|t| (t.grid.origin, t.grid.extent().x));
    let key = (network.clone(), extent);
    if runtime.built.as_ref() == Some(&key) {
        return;
    }
    runtime.built = Some(key);
    if network.is_empty() {
        runtime.task = None;
        commands.remove_resource::<PathState>();
        return;
    }
    let Some((origin, extent_m)) = extent else {
        return;
    };
    let texels = mask_resolution(&network.paths, extent_m, network.resolution);
    let texel_m = extent_m / texels as f32;
    let paths = network.paths.clone();
    runtime.task = Some(
        AsyncComputeTaskPool::get().spawn(async move { rasterise(&paths, origin, texel_m, texels) }),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A straight road along +X at z = 0: 5 m wide, gravel.
    fn road() -> Path {
        Path::new(
            vec![Vec3::new(-20.0, 0.0, 0.0), Vec3::new(20.0, 0.0, 0.0)],
            2.5,
            PathKind::Gravel,
        )
    }

    /// The mask sampled as the shader does, at `p`.
    fn sample(mask: &PathMask, p: Vec2) -> [f32; 4] {
        let t = (p - mask.origin) / mask.texel_m - 0.5;
        let base = t.floor();
        let w = t - base;
        let at = |dx: i32, dy: i32| {
            let x = (base.x as i32 + dx).clamp(0, mask.texels as i32 - 1) as u32;
            let y = (base.y as i32 + dy).clamp(0, mask.texels as i32 - 1) as u32;
            mask.texels_data[(y * mask.texels + x) as usize]
        };
        let mut out = [0.0; 4];
        for (c, slot) in out.iter_mut().enumerate() {
            let row0 = at(0, 0)[c] * (1.0 - w.x) + at(1, 0)[c] * w.x;
            let row1 = at(0, 1)[c] * (1.0 - w.x) + at(1, 1)[c] * w.x;
            *slot = row0 * (1.0 - w.y) + row1 * w.y;
        }
        out
    }

    /// The mask's coverage at `p`, through the crate's own curve.
    fn masked_coverage(mask: &PathMask, p: Vec2) -> f32 {
        let s = sample(mask, p);
        // The fade is derived from the route's half width, as the shader derives it: the mask's
        // alpha carries the grain, not the fade.
        super::coverage(s[0], edge_fade_m(mask.half_width_m))
    }

    #[test]
    fn the_field_is_a_signed_distance_along_and_across_the_route() {
        let mask = rasterise(&[road()], Vec2::splat(-32.0), 1.0, 64);
        // Row 32 is centred at z = 0.5, so the signed distance there is 0.5 - 2.5 = -2 m.
        for x in [-15.0, -3.0, 0.0, 7.5, 15.0] {
            let d = mask.texels_data[(32 * 64 + (x + 32.0) as u32) as usize][0];
            assert!((d + 2.0).abs() < 0.01, "at x {x}: {d}");
        }
        // It grows by one texel per texel outward: it is a distance, not a ramp of its own.
        for row in 33..=37u32 {
            let d = mask.texels_data[(row * 64 + 32) as usize][0];
            let want = (row as f32 + 0.5 - 32.0) - 2.5;
            assert!((d - want).abs() < 0.01, "row {row}: {d} against {want}");
        }
        // The kind and depth travel with it.
        let (kind, depth) = PathMask::kind_and_depth(mask.texels_data[(32 * 64 + 32) as usize][1]);
        assert_eq!((kind, depth), (1.0, 0.0));
    }

    #[test]
    fn the_coverage_is_one_inside_and_zero_past_the_fade() {
        let mask = rasterise(&[road()], Vec2::splat(-32.0), 1.0, 64);
        let c = |z: f32| masked_coverage(&mask, Vec2::new(0.0, z));
        // A 5 m road: its edge is at z = 2.5 and the fade runs 0.825 m inward.
        assert_eq!(c(0.0), 1.0);
        assert_eq!(c(1.5), 1.0);
        // The fade is 0.825 m inward from the edge at z = 2.5: at 2.0 (0.5 m in) the smoothstep
        // is ~0.63, at 1.675 (a full fade in) it is 1, and past the edge it is 0.
        assert!((c(2.0) - 0.63).abs() < 0.12, "{}", c(2.0));
        assert!(c(1.6) > 0.99, "{}", c(1.6));
        assert!(c(2.4) < 0.2, "{}", c(2.4));
        assert_eq!(c(2.5), 0.0);
        assert_eq!(c(3.0), 0.0);
        assert_eq!(c(20.0), 0.0);
    }

    #[test]
    fn a_diagonal_route_has_the_same_edge_as_an_axial_one() {
        let diagonal = Path::new(
            vec![Vec3::new(-20.0, 0.0, -20.0), Vec3::new(20.0, 0.0, 20.0)],
            2.5,
            PathKind::Earth,
        );
        let mask = rasterise(&[diagonal], Vec2::splat(-32.0), 1.0, 64);
        let normal = Vec2::new(1.0, -1.0).normalize();
        assert_eq!(masked_coverage(&mask, Vec2::ZERO), 1.0);
        for off in [0.0, 1.5] {
            assert!(masked_coverage(&mask, normal * off) > 0.95, "at {off} m");
        }
        for off in [2.8, 4.0, 9.0] {
            assert_eq!(masked_coverage(&mask, normal * off), 0.0, "at {off} m");
        }
    }

    #[test]
    fn the_lateral_offset_places_the_wheel_lines() {
        // A cart track 2 m wide: its wheel lines are at 0.6 of the half width, 1.2 m either side.
        let track = Path::new(
            vec![Vec3::new(-20.0, 0.0, 0.0), Vec3::new(20.0, 0.0, 0.0)],
            1.0,
            PathKind::Earth,
        )
        .with_ruts(0.06);
        let mask = rasterise(&[track], Vec2::splat(-32.0), 0.25, 256);
        // The route runs along +X, so its leftward normal is -Z: the left wheel line is at
        // z = -0.6 and the right at +0.6, both 0.6 of the half width.
        let left = sample(&mask, Vec2::new(0.0, -0.6));
        let right = sample(&mask, Vec2::new(0.0, 0.6));
        assert!((left[2] + 0.6).abs() < 0.05, "left {left:?}");
        assert!((right[2] - 0.6).abs() < 0.05, "right {right:?}");
        assert!((left[2].abs() - 0.0).abs() > 0.3, "the centre is not a wheel line");
        // The centre's offset is zero, and the rut depth rides the kind's fraction.
        let centre = sample(&mask, Vec2::ZERO);
        assert!(centre[2].abs() < 0.05, "{centre:?}");
        let (kind, depth) = PathMask::kind_and_depth(centre[1]);
        assert_eq!(kind, 0.0);
        assert!((depth - 0.06).abs() < 1e-3, "{depth}");
    }

    #[test]
    fn crossing_routes_take_the_nearer_and_the_mask_is_deterministic() {
        let lane = Path::new(
            vec![Vec3::new(0.0, 0.0, -20.0), Vec3::new(0.0, 0.0, 20.0)],
            0.35,
            PathKind::Earth,
        );
        let a = rasterise(&[road(), lane.clone()], Vec2::splat(-32.0), 0.5, 128);
        let b = rasterise(&[road(), lane], Vec2::splat(-32.0), 0.5, 128);
        assert_eq!(a, b, "the rasteriser is deterministic");
        // Far up the lane it is the lane's surface; far along the road it is the road's.
        assert_eq!(PathMask::kind_and_depth(sample(&a, Vec2::new(0.0, 15.0))[1]).0, 0.0);
        assert_eq!(PathMask::kind_and_depth(sample(&a, Vec2::new(15.0, 0.0))[1]).0, 1.0);
    }

    #[test]
    fn a_route_outside_the_mask_is_clipped() {
        let far = Path::new(
            vec![Vec3::new(200.0, 0.0, 200.0), Vec3::new(260.0, 0.0, 260.0)],
            2.0,
            PathKind::Stone,
        );
        let mask = rasterise(&[far], Vec2::splat(-32.0), 1.0, 64);
        assert!(mask.texels_data.iter().all(|t| t[0] == f32::INFINITY));
        assert_eq!(masked_coverage(&mask, Vec2::ZERO), 0.0);
    }

    #[test]
    fn the_eight_bit_encoding_round_trips_the_field() {
        // The shader's own decode of the encoded channels: a route's -1 m edge, its fade and its
        // lateral fraction survive 8 bits within the range's resolution.
        let mask = rasterise(&[road()], Vec2::splat(-32.0), 0.25, 256);
        let image = mask_image(&mask);
        let data = image.data.as_ref().expect("the mask carries its bytes");
        let n = mask.texels as usize;
        for t in [0.0f32, 1.5, -1.5, 3.0] {
            let p = Vec2::new(0.0, t);
            let tx = ((p.x - mask.origin.x) / mask.texel_m).floor() as usize;
            let ty = ((p.y - mask.origin.y) / mask.texel_m).floor() as usize;
            let i = ty * n + tx;
            let texel = [data[4 * i], data[4 * i + 1], data[4 * i + 2], data[4 * i + 3]];
            let got = decode(texel);
            let want = mask.texels_data[i];
            assert!(
                (got[0] - want[0]).abs() <= PATH_DISTANCE_RANGE_M / 128.0 + 1e-3,
                "distance {got:?} against {want:?}"
            );
            assert!((got[2] - want[2]).abs() <= 0.05, "lateral {got:?} against {want:?}");
            assert!((got[3] - want[3]).abs() <= 1.0 / 255.0, "grain {got:?}");
        }
    }

    /// `docs/research/path-query.md` Validation 1: the query answers the shader's own curves.
    #[test]
    fn the_query_answers_the_shaders_own_curve() {
        let road = road();
        for z in [0.0f32, 1.0, 2.0, 2.5, 2.7, 3.2, 9.0] {
            let hit = nearest(std::slice::from_ref(&road), Vec2::new(0.0, z)).expect("a route");
            let want = coverage(hit.distance_m, edge_fade_m(hit.surface.half_width_m));
            assert_eq!(hit.surface.coverage, want, "at z {z}");
        }
        let track = Path::new(
            vec![Vec3::new(-20.0, 0.0, 0.0), Vec3::new(20.0, 0.0, 0.0)],
            1.0,
            PathKind::Earth,
        )
        .with_ruts(0.06);
        let at = |z: f32| nearest(std::slice::from_ref(&track), Vec2::new(0.0, z)).expect("a route");
        assert!((at(-0.6).surface.rut - 1.0).abs() < 0.05);
        assert_eq!(at(0.0).surface.rut, 0.0, "the crown is no wheel line");
        assert_eq!(at(-0.6).surface.rut_depth_m, 0.06);
    }

    /// Validation 2-4: inside, the edge, outside; the nearest route's kind; the edge distance.
    #[test]
    fn the_query_finds_the_nearest_route_and_its_edge() {
        let road = road();
        let hit = nearest(std::slice::from_ref(&road), Vec2::ZERO).expect("a route");
        assert_eq!(hit.surface.kind, PathKind::Gravel);
        assert_eq!(hit.surface.half_width_m, 2.5);
        assert_eq!(hit.surface.coverage, 1.0);
        assert!((hit.distance_m + 2.5).abs() < 1e-5);
        // Four metres off the centre line: 1.5 m outside the edge, and the nearest point on it.
        let off = nearest(std::slice::from_ref(&road), Vec2::new(3.0, 4.0)).expect("a route");
        assert!((off.distance_m - 1.5).abs() < 1e-5, "{}", off.distance_m);
        assert!((off.point - Vec2::new(3.0, 0.0)).length() < 1e-5);
        assert_eq!(off.surface.coverage, 0.0);
        // Crossing routes: the nearer one's kind wins.
        let lane = Path::new(
            vec![Vec3::new(0.0, 0.0, -20.0), Vec3::new(0.0, 0.0, 20.0)],
            0.35,
            PathKind::Earth,
        );
        let both = [road, lane];
        assert_eq!(nearest(&both, Vec2::new(0.0, 12.0)).expect("a route").surface.kind, PathKind::Earth);
        assert_eq!(nearest(&both, Vec2::new(12.0, 0.0)).expect("a route").surface.kind, PathKind::Gravel);
        assert!(nearest(&[], Vec2::ZERO).is_none());
    }

    /// Validation 6: a village's network answers every query near it.
    ///
    /// The measurement is *reported*, not asserted: a wall-clock bound in a test suite fails on a
    /// loaded machine rather than on a defect (this one did, on the local host, while the same test
    /// read 1.04 ms on bmb). Recorded in `docs/loop-log.md` instead: 1000 queries over 40 routes at
    /// 1.04-1.62 ms, which is a sixteenth of a 60 Hz frame at worst.
    #[test]
    fn a_village_network_answers_every_query() {
        // 40 routes of five segments: a village's road network.
        let paths: Vec<Path> = (0..40)
            .map(|i| {
                let z = i as f32 * 8.0 - 160.0;
                Path::new(
                    vec![
                        Vec3::new(-200.0, 0.0, z),
                        Vec3::new(-80.0, 0.0, z + 6.0),
                        Vec3::new(40.0, 0.0, z - 6.0),
                        Vec3::new(160.0, 0.0, z + 2.0),
                        Vec3::new(220.0, 0.0, z),
                    ],
                    2.0,
                    PathKind::Gravel,
                )
            })
            .collect();
        let start = std::time::Instant::now();
        let mut hits = 0;
        for k in 0..1000 {
            let p = Vec2::new((k % 40) as f32 * 4.0 - 80.0, (k / 40) as f32 * 8.0 - 40.0);
            if nearest(&paths, p).is_some() {
                hits += 1;
            }
        }
        let elapsed = start.elapsed();
        println!("1000 queries over 40 routes: {elapsed:?}");
        // Every point lies near a route, so every query answers: the correctness half of the test.
        assert_eq!(hits, 1000);
        // A generous ceiling, only to catch a catastrophic regression (a linear scan over every
        // segment of every route would blow past it); the loop's real measurement is in the log.
        assert!(
            elapsed.as_secs_f64() < 0.05,
            "{elapsed:?} for 1000 queries is a catastrophic regression"
        );
    }

    #[test]
    fn the_mask_resolves_the_narrowest_route() {
        let lane = Path::new(
            vec![Vec3::new(0.0, 0.0, 0.0), Vec3::new(10.0, 0.0, 0.0)],
            0.35,
            PathKind::Earth,
        );
        let road = Path::new(
            vec![Vec3::new(0.0, 0.0, 0.0), Vec3::new(10.0, 0.0, 0.0)],
            2.5,
            PathKind::Gravel,
        );
        // A 0.7 m footpath wants 0.175 m texels: 2048 / 0.175 = 11 703, capped at 4096.
        assert_eq!(mask_resolution(std::slice::from_ref(&lane), 2048.0, 0), 4096);
        // A road alone asks for no finer than a metre a texel.
        assert_eq!(mask_resolution(std::slice::from_ref(&road), 2048.0, 0), 2048);
        // A game's own resolution wins, clamped to a power of two in range.
        assert_eq!(mask_resolution(std::slice::from_ref(&road), 2048.0, 500), 512);
        assert_eq!(mask_resolution(std::slice::from_ref(&road), 2048.0, 1 << 20), 4096);
    }

    #[test]
    fn the_edge_fade_follows_the_width() {
        assert_eq!(edge_fade_m(0.35), PATH_EDGE_FADE_M);
        assert!((edge_fade_m(2.5) - 0.825).abs() < 1e-6);
        assert!(edge_fade_m(64.0) > edge_fade_m(2.5));
    }
}
