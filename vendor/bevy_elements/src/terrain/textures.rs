//! Ground textures a game sets on the terrain: ordinary PBR maps per ground layer, packed by the
//! crate into the detail array the terrain material samples (`docs/research/terrain-textures.md`;
//! the format and the derivations are `docs/research/ground-textures.md`'s).

use bevy::asset::RenderAssetUsages;
use bevy::image::{ImageAddressMode, ImageFilterMode, ImageSampler, ImageSamplerDescriptor};
use bevy::prelude::*;
use bevy::render::render_resource::{
    Extent3d, TextureDataOrder, TextureDimension, TextureFormat, TextureViewDescriptor,
    TextureViewDimension,
};
use bevy::tasks::{AsyncComputeTaskPool, Task, block_on, poll_once};

use super::Terrain;

/// The ground textures of a [`Terrain`]: for each ground layer (sand, soil, grass) a PBR set — a
/// base colour and, optionally, a normal, roughness, occlusion and height map — which the crate
/// packs into the terrain material's detail textures (`docs/research/terrain-textures.md`). Put
/// it on the terrain's entity; insert, change or remove it at any time. A change is packed off
/// the frame (on Bevy's `AsyncComputeTaskPool`) once every map has loaded, and the previous set
/// keeps rendering until the new one is ready.
///
/// The textures are detail on the crate's ground (ground-textures.md): hex-tiled, so a tile never
/// repeats visibly; blended by their heights where two layers meet; faded out with distance,
/// where their mips average to no change. Each layer's base colour can also replace the layer's
/// own colour at every distance ([`GroundLayerTextures::colour`]). A layer left `None` stays
/// procedural; without the component the ground is the procedural one alone, exactly. The
/// rock is procedural (`docs/research/rock-material.md`).
#[derive(Component, Clone, Debug, Reflect)]
#[reflect(Component)]
pub struct TerrainTextures {
    /// The sand's set, `None` for the procedural sand.
    pub sand: Option<GroundLayerTextures>,
    /// The soil's set, `None` for the procedural soil.
    pub soil: Option<GroundLayerTextures>,
    /// The grass's set, `None` for the procedural grass.
    pub grass: Option<GroundLayerTextures>,
    /// The share of the crate's procedural grain's albedo mottle kept under a textured layer,
    /// 0–1 (default 0.3): the textures carry the near field's colour. The grain's relief is kept
    /// whole: the turf's bumps are what a low sun shows across the mid field, where the
    /// textures' normals have averaged flat.
    pub grain: f32,
    /// Texels per side of the packed textures: a power of two, 16–4096 (default 1024). Every map
    /// is resampled to it.
    pub resolution: u32,
}

impl Default for TerrainTextures {
    fn default() -> Self {
        Self {
            sand: None,
            soil: None,
            grass: None,
            grain: 0.3,
            resolution: 1024,
        }
    }
}

impl TerrainTextures {
    fn layers(&self) -> [Option<&GroundLayerTextures>; 3] {
        [self.sand.as_ref(), self.soil.as_ref(), self.grass.as_ref()]
    }

    /// Whether `id` is one of the sets' maps.
    fn uses(&self, id: AssetId<Image>) -> bool {
        self.layers()
            .into_iter()
            .flatten()
            .any(|layer| layer.maps().into_iter().flatten().any(|h| h.id() == id))
    }
}

/// One ground layer's PBR set (a Unity Terrain Layer, a glTF material's maps). The maps are read
/// as glTF stores them, whatever sRGB flag the images were loaded with: the base colour is
/// sRGB-encoded; the normal is tangent space with +Y up (OpenGL; DirectX maps set
/// [`flip_normal_map_y`](Self::flip_normal_map_y)); the roughness is the G channel and the
/// occlusion the R channel, so one glTF metallic-roughness or occlusion-roughness-metallic image
/// may serve both, and a greyscale map either; the height is the R channel. The images must keep
/// their CPU data (the default when loading).
///
/// The set is detail on the crate's ground: its roughness and occlusion become multipliers with
/// mean 1 (the crate's roughness, which the weather lowers, and the ground's mean light stay),
/// its height is compared with the other layers' where two meet, and its base colour is a
/// detail with mean 1 over the crate's calibrated colour; [`colour`](Self::colour) takes the
/// set's own mean colour instead (a library's base colour is rarely a calibrated albedo: two CC0
/// libraries' sets measured 2-6.5 times the crate's). A map left out is derived from the base
/// colour (height from its luminance, the normal from the height's slope at
/// [`relief_deg`](Self::relief_deg), the occlusion from the cavities, a neutral roughness).
#[derive(Clone, Debug, Reflect)]
pub struct GroundLayerTextures {
    /// The base colour (albedo), sRGB-encoded.
    pub albedo: Handle<Image>,
    /// Tangent-space normal map, +Y up unless [`flip_normal_map_y`](Self::flip_normal_map_y).
    pub normal: Option<Handle<Image>>,
    /// Roughness, read from the G channel.
    pub roughness: Option<Handle<Image>>,
    /// Ambient occlusion, read from the R channel.
    pub occlusion: Option<Handle<Image>>,
    /// Height (displacement), read from the R channel.
    pub height: Option<Handle<Image>>,
    /// One tile's size on the ground, m.
    pub tile_m: f32,
    /// How far the set's mean colour replaces the layer's own colour, 0–1 (default 0): 0 keeps
    /// the crate's calibrated colour (the set brings its detail), 1 is the set's colour, at every
    /// distance — for a stylised set whose colour is the point.
    pub colour: f32,
    /// Scales the normal map's X and Y (glTF's normal scale): 0 flat, 2 twice as pronounced.
    pub normal_scale: f32,
    /// The normal map is DirectX's (−Y up), as `StandardMaterial::flip_normal_map_y`.
    pub flip_normal_map_y: bool,
    /// Without a normal map: the mean tilt of the normals derived from the height, degrees.
    pub relief_deg: f32,
    /// De-lighting: the exponent applied to the base colour's luminance ratio against its local
    /// mean, 1 for a de-lit PBR base colour, below 1 to flatten a photograph's light and shade.
    pub contrast: f32,
    /// The maps already tile; `false` cross-fades them with their half-period offset so they do.
    pub tileable: bool,
}

impl GroundLayerTextures {
    /// A PBR set's base colour, `tile_m` metres a tile: de-lit and tileable, over the crate's
    /// colour ([`with_colour`](Self::with_colour) takes its own). Add its other maps with the
    /// `with_*` methods.
    pub fn new(albedo: Handle<Image>, tile_m: f32) -> Self {
        Self {
            albedo,
            normal: None,
            roughness: None,
            occlusion: None,
            height: None,
            tile_m,
            colour: 0.0,
            normal_scale: 1.0,
            flip_normal_map_y: false,
            relief_deg: 10.0,
            contrast: 1.0,
            tileable: true,
        }
    }

    /// The base colour is a photograph or a generated image (golem's): its light is flattened
    /// (`contrast` 0.65) and it is made tileable.
    pub fn photograph(mut self) -> Self {
        self.contrast = 0.65;
        self.tileable = false;
        self
    }

    /// With the set's own mean colour replacing the layer's by `share`, 0–1.
    pub fn with_colour(mut self, share: f32) -> Self {
        self.colour = share;
        self
    }

    /// With a tangent-space normal map.
    pub fn with_normal(mut self, normal: Handle<Image>) -> Self {
        self.normal = Some(normal);
        self
    }

    /// With a roughness map (G channel).
    pub fn with_roughness(mut self, roughness: Handle<Image>) -> Self {
        self.roughness = Some(roughness);
        self
    }

    /// With an ambient occlusion map (R channel).
    pub fn with_occlusion(mut self, occlusion: Handle<Image>) -> Self {
        self.occlusion = Some(occlusion);
        self
    }

    /// With a height map (R channel).
    pub fn with_height(mut self, height: Handle<Image>) -> Self {
        self.height = Some(height);
        self
    }

    /// With the derived normals' mean tilt, degrees.
    pub fn with_relief_deg(mut self, relief_deg: f32) -> Self {
        self.relief_deg = relief_deg;
        self
    }

    fn maps(&self) -> [Option<&Handle<Image>>; 5] {
        [
            Some(&self.albedo),
            self.normal.as_ref(),
            self.roughness.as_ref(),
            self.occlusion.as_ref(),
            self.height.as_ref(),
        ]
    }
}

/// The packed set the terrain material binds, and the packing in flight.
#[derive(Resource, Default)]
pub(crate) struct GroundTexturePack {
    /// The packed array, once a set has been packed.
    pub(crate) image: Option<Handle<Image>>,
    /// xyz one tile's size (m) for the sand, the soil and the grass, 0 for a procedural layer;
    /// w the procedural grain's albedo share under a textured layer.
    pub(crate) ground: Vec4,
    /// Per layer (sand, soil, grass): the set's mean linear colour (xyz) and its share (w).
    pub(crate) colours: [Vec4; 3],
    task: Option<Task<Packed>>,
    pending: bool,
}

struct Packed {
    image: Image,
    ground: Vec4,
    colours: [Vec4; 3],
}

/// Packs the terrain's [`TerrainTextures`] when they change or one of their maps is reloaded,
/// once every map has loaded, and installs the result when the task finishes.
pub(crate) fn pack_terrain_textures(
    terrains: Query<Ref<TerrainTextures>, With<Terrain>>,
    mut events: MessageReader<AssetEvent<Image>>,
    mut images: ResMut<Assets<Image>>,
    mut pack: ResMut<GroundTexturePack>,
) {
    let Some(textures) = terrains.iter().next() else {
        events.clear();
        if pack.image.is_some() || pack.task.is_some() || pack.pending {
            *pack = GroundTexturePack::default();
        }
        return;
    };
    if textures.is_changed() {
        pack.pending = true;
    }
    for event in events.read() {
        if let AssetEvent::Modified { id } | AssetEvent::LoadedWithDependencies { id } = event
            && textures.uses(*id)
        {
            pack.pending = true;
        }
    }
    if let Some(task) = pack.task.as_mut()
        && let Some(packed) = block_on(poll_once(task))
    {
        pack.task = None;
        pack.image = Some(images.add(packed.image));
        pack.ground = packed.ground;
        pack.colours = packed.colours;
    }
    if pack.pending
        && pack.task.is_none()
        && let Some(inputs) = Inputs::gather(&textures, &images)
    {
        pack.pending = false;
        pack.task = Some(AsyncComputeTaskPool::get().spawn(async move { inputs.pack() }));
    }
}

/// One layer's maps and settings, copied out of the asset store for the packing task.
struct LayerInput {
    albedo: Image,
    normal: Option<Image>,
    roughness: Option<Image>,
    occlusion: Option<Image>,
    height: Option<Image>,
    tile_m: f32,
    colour: f32,
    normal_scale: f32,
    flip_normal_map_y: bool,
    relief_deg: f32,
    contrast: f32,
    tileable: bool,
}

struct Inputs {
    layers: [Option<LayerInput>; 3],
    grain: f32,
    resolution: usize,
}

impl Inputs {
    /// The sets' maps, or `None` while one has not loaded. A map without CPU data cannot be read
    /// and is left out (the layer's albedo: the layer stays procedural), with a warning.
    fn gather(textures: &TerrainTextures, images: &Assets<Image>) -> Option<Self> {
        let loaded = textures
            .layers()
            .into_iter()
            .flatten()
            .flat_map(|layer| layer.maps().into_iter().flatten())
            .all(|h| images.contains(h));
        if !loaded {
            return None;
        }
        let readable = |handle: Option<&Handle<Image>>| -> Option<Image> {
            let image = images.get(handle?)?;
            if image.data.is_none() {
                warn!(
                    "a terrain texture has no CPU data to pack (load it with RenderAssetUsages::MAIN_WORLD); it is left out"
                );
                return None;
            }
            Some(image.clone())
        };
        let layers = textures.layers().map(|layer| {
            let layer = layer?;
            Some(LayerInput {
                albedo: readable(Some(&layer.albedo))?,
                normal: readable(layer.normal.as_ref()),
                roughness: readable(layer.roughness.as_ref()),
                occlusion: readable(layer.occlusion.as_ref()),
                height: readable(layer.height.as_ref()),
                tile_m: layer.tile_m.max(0.0),
                colour: layer.colour.clamp(0.0, 1.0),
                normal_scale: layer.normal_scale,
                flip_normal_map_y: layer.flip_normal_map_y,
                relief_deg: layer.relief_deg.clamp(0.0, 80.0),
                contrast: layer.contrast.max(0.0),
                tileable: layer.tileable,
            })
        });
        Some(Self {
            layers,
            grain: textures.grain.clamp(0.0, 1.0),
            resolution: textures.resolution.clamp(16, 4096).next_power_of_two() as usize,
        })
    }

    fn pack(self) -> Packed {
        let n = self.resolution;
        let mut ground = Vec4::new(0.0, 0.0, 0.0, self.grain);
        let mut colours = [Vec4::ZERO; 3];
        let mut decoded: Vec<[[Grid; 4]; 2]> = Vec::with_capacity(3);
        for (k, layer) in self.layers.iter().enumerate() {
            match layer.as_ref().map(|layer| (layer, derive_layer(layer, n))) {
                Some((layer, Some((pair, mean)))) if layer.tile_m > 0.0 => {
                    ground[k] = layer.tile_m;
                    colours[k] = mean.extend(layer.colour);
                    decoded.push(pair);
                }
                _ => decoded.push(neutral_layer(n)),
            }
        }
        Packed {
            image: encode_array(decoded, n),
            ground,
            colours,
        }
    }
}

/// A square single-channel plane of `n × n` texels, row-major.
#[derive(Clone, Debug, PartialEq)]
struct Grid {
    n: usize,
    v: Vec<f32>,
}

impl Grid {
    fn constant(n: usize, value: f32) -> Self {
        Self {
            n,
            v: vec![value; n * n],
        }
    }

    fn mean(&self) -> f32 {
        (self.v.iter().map(|&x| f64::from(x)).sum::<f64>() / self.v.len() as f64) as f32
    }

    fn map(&self, f: impl Fn(f32) -> f32) -> Self {
        Self {
            n: self.n,
            v: self.v.iter().map(|&x| f(x)).collect(),
        }
    }

    fn zip(&self, other: &Self, f: impl Fn(f32, f32) -> f32) -> Self {
        Self {
            n: self.n,
            v: self
                .v
                .iter()
                .zip(&other.v)
                .map(|(&a, &b)| f(a, b))
                .collect(),
        }
    }

    /// The texel at (column `x`, row `y`), wrapping.
    fn at(&self, x: isize, y: isize) -> f32 {
        let n = self.n as isize;
        self.v[(y.rem_euclid(n) * n + x.rem_euclid(n)) as usize]
    }

    /// 2 × 2 averages: the next mip level.
    fn halve(&self) -> Self {
        let m = (self.n / 2).max(1);
        let mut v = Vec::with_capacity(m * m);
        for y in 0..m {
            for x in 0..m {
                let (x, y) = (2 * x as isize, 2 * y as isize);
                v.push(
                    0.25 * (self.at(x, y)
                        + self.at(x + 1, y)
                        + self.at(x, y + 1)
                        + self.at(x + 1, y + 1)),
                );
            }
        }
        Self { n: m, v }
    }

    /// A Gaussian blur of standard deviation `sigma` texels that wraps (a tileable plane stays
    /// tileable): Kovesi's cascade of four box filters, each separable.
    fn blur(&self, sigma: f32) -> Self {
        let mut out = self.clone();
        for width in box_widths(sigma, 4) {
            if width > 1 {
                out = out.box_rows(width).box_columns(width);
            }
        }
        out
    }

    /// A moving average of odd `width` along each row, wrapping.
    fn box_rows(&self, width: usize) -> Self {
        let n = self.n;
        let r = (width / 2) as isize;
        let wrap = |i: isize| i.rem_euclid(n as isize) as usize;
        let scale = 1.0 / width as f64;
        let mut v = vec![0.0; n * n];
        for (row, out) in self.v.chunks_exact(n).zip(v.chunks_exact_mut(n)) {
            let mut sum: f64 = (-r..=r).map(|i| f64::from(row[wrap(i)])).sum();
            for (x, slot) in out.iter_mut().enumerate() {
                *slot = (sum * scale) as f32;
                let x = x as isize;
                sum += f64::from(row[wrap(x + r + 1)]) - f64::from(row[wrap(x - r)]);
            }
        }
        Self { n, v }
    }

    /// A moving average of odd `width` down each column, wrapping: one running sum per column,
    /// carried down the rows, so the plane is read in order.
    fn box_columns(&self, width: usize) -> Self {
        let n = self.n;
        let r = (width / 2) as isize;
        let row = |y: isize| &self.v[y.rem_euclid(n as isize) as usize * n..][..n];
        let scale = 1.0 / width as f64;
        let mut sums = vec![0.0_f64; n];
        for y in -r..=r {
            for (sum, &value) in sums.iter_mut().zip(row(y)) {
                *sum += f64::from(value);
            }
        }
        let mut v = vec![0.0; n * n];
        for (y, out) in v.chunks_exact_mut(n).enumerate() {
            for (slot, &sum) in out.iter_mut().zip(&sums) {
                *slot = (sum * scale) as f32;
            }
            let y = y as isize;
            for ((sum, &add), &remove) in sums.iter_mut().zip(row(y + r + 1)).zip(row(y - r)) {
                *sum += f64::from(add) - f64::from(remove);
            }
        }
        Self { n, v }
    }

    /// Cross-fade the plane with its half-period roll near the borders, preserving the variance
    /// (Heitz & Neyret 2018), so it tiles; clipped to `[lo, hi]`.
    fn tileable(&self, lo: f32, hi: f32) -> Self {
        let n = self.n;
        let half = (n / 2) as isize;
        let band = 0.18 * n as f32;
        let fade = |i: usize| {
            let edge = (i.min(n - 1 - i) as f32 / band).clamp(0.0, 1.0);
            let w = 1.0 - edge;
            w * w * (3.0 - 2.0 * w)
        };
        let mean = self.mean();
        let mut v = Vec::with_capacity(n * n);
        for y in 0..n {
            for x in 0..n {
                let w = fade(x).max(fade(y));
                let a = self.v[y * n + x] - mean;
                let b = self.at(x as isize + half, y as isize + half) - mean;
                let blended = ((1.0 - w) * a + w * b) / ((1.0 - w).powi(2) + w * w).sqrt();
                v.push((mean + blended).clamp(lo, hi));
            }
        }
        Self { n, v }
    }

    /// Mean-1 normalisation: the plane over its mean, or 1 where the mean is zero.
    fn over_mean(&self) -> Self {
        let mean = self.mean();
        if mean > 1e-6 {
            self.map(|x| x / mean)
        } else {
            Self::constant(self.n, 1.0)
        }
    }

    /// The plane band-passed between σ 1.2 and `n / 24` and normalised to ½ ± 4 std: the
    /// detail's height (ground-textures.md).
    fn band_height(&self) -> Self {
        let fine = self.blur(1.2);
        let coarse = self.blur(self.n as f32 / 24.0);
        let band = fine.zip(&coarse, |a, b| a - b);
        let std = (band.v.iter().map(|&x| f64::from(x).powi(2)).sum::<f64>() / band.v.len() as f64)
            .sqrt() as f32;
        band.map(|x| (0.5 + x / (4.0 * std + 1e-9)).clamp(0.0, 1.0))
    }
}

/// The box widths whose cascade has standard deviation `sigma` (Kovesi 2010): `m` passes of the
/// odd width below the ideal and `passes − m` of the next odd width.
fn box_widths(sigma: f32, passes: usize) -> Vec<usize> {
    let n = passes as f32;
    let s2 = 12.0 * sigma * sigma;
    let ideal = (s2 / n + 1.0).sqrt();
    let mut low = ideal.floor() as i64;
    if low % 2 == 0 {
        low -= 1;
    }
    let low = low.max(1) as f32;
    let m = ((s2 - n * low * low - 4.0 * n * low - 3.0 * n) / (-4.0 * low - 4.0))
        .round()
        .clamp(0.0, n) as usize;
    let low = low as usize;
    (0..passes)
        .map(|i| if i < m { low } else { low + 2 })
        .collect()
}

/// A map's stored RGBA values at `n × n`: halved by 2 × 2 averages while it is at least twice
/// that, then resampled bilinearly with wrap. `srgb` decodes the RGB from the sRGB transfer
/// function first, so the filtering is linear. `None` for an image that is not 2D or whose
/// format `Image::get_color_at` cannot read.
fn read_map(image: &Image, n: usize, srgb: bool) -> Option<[Grid; 4]> {
    let size = image.texture_descriptor.size;
    let (mut w, mut h) = (size.width as usize, size.height as usize);
    if size.depth_or_array_layers != 1 || w == 0 || h == 0 {
        warn!("a terrain texture is not a single 2D image; it is left out");
        return None;
    }
    let mut planes: [Vec<f32>; 4] = std::array::from_fn(|_| Vec::with_capacity(w * h));
    let decode: [f32; 256] = std::array::from_fn(|b| {
        let value = b as f32 / 255.0;
        if srgb { srgb_to_linear(value) } else { value }
    });
    let format = image.texture_descriptor.format;
    let bgra = matches!(
        format,
        TextureFormat::Bgra8Unorm | TextureFormat::Bgra8UnormSrgb
    );
    match image.data.as_deref() {
        // Eight-bit RGBA, the loaders' usual output: the bytes are the stored values.
        Some(data)
            if (bgra
                || matches!(
                    format,
                    TextureFormat::Rgba8Unorm | TextureFormat::Rgba8UnormSrgb
                ))
                && data.len() >= w * h * 4 =>
        {
            let order = if bgra { [2, 1, 0, 3] } else { [0, 1, 2, 3] };
            for texel in data[..w * h * 4].as_chunks::<4>().0 {
                for (c, plane) in planes.iter_mut().enumerate() {
                    let byte = texel[order[c]];
                    plane.push(if c < 3 {
                        decode[byte as usize]
                    } else {
                        f32::from(byte) / 255.0
                    });
                }
            }
        }
        _ => {
            for y in 0..h as u32 {
                for x in 0..w as u32 {
                    let colour = match image.get_color_at(x, y) {
                        Ok(colour) => colour,
                        Err(error) => {
                            warn!("a terrain texture cannot be read ({error:?}); it is left out");
                            return None;
                        }
                    };
                    // The stored value, whatever the format's sRGB flag says about it.
                    let stored = match colour {
                        Color::Srgba(c) => c.to_f32_array(),
                        other => other.to_linear().to_f32_array(),
                    };
                    for (c, plane) in planes.iter_mut().enumerate() {
                        let value = if srgb && c < 3 {
                            srgb_to_linear(stored[c])
                        } else {
                            stored[c]
                        };
                        plane.push(value);
                    }
                }
            }
        }
    }
    while w >= 2 * n || h >= 2 * n {
        let (hw, hh) = (
            if w >= 2 * n { 2 } else { 1 },
            if h >= 2 * n { 2 } else { 1 },
        );
        let (nw, nh) = (w / hw, h / hh);
        for plane in &mut planes {
            let mut out = Vec::with_capacity(nw * nh);
            for y in 0..nh {
                for x in 0..nw {
                    let mut sum = 0.0;
                    for dy in 0..hh {
                        for dx in 0..hw {
                            sum += plane[(y * hh + dy) * w + x * hw + dx];
                        }
                    }
                    out.push(sum / (hw * hh) as f32);
                }
            }
            *plane = out;
        }
        (w, h) = (nw, nh);
    }
    Some(planes.map(|plane| {
        if w == n && h == n {
            return Grid { n, v: plane };
        }
        let at = |x: isize, y: isize| {
            plane[(y.rem_euclid(h as isize) * w as isize + x.rem_euclid(w as isize)) as usize]
        };
        let mut v = Vec::with_capacity(n * n);
        for y in 0..n {
            let fy = (y as f32 + 0.5) * h as f32 / n as f32 - 0.5;
            let (y0, ty) = (fy.floor() as isize, fy - fy.floor());
            for x in 0..n {
                let fx = (x as f32 + 0.5) * w as f32 / n as f32 - 0.5;
                let (x0, tx) = (fx.floor() as isize, fx - fx.floor());
                let top = at(x0, y0) * (1.0 - tx) + at(x0 + 1, y0) * tx;
                let bottom = at(x0, y0 + 1) * (1.0 - tx) + at(x0 + 1, y0 + 1) * tx;
                v.push(top * (1.0 - ty) + bottom * ty);
            }
        }
        Grid { n, v }
    }))
}

fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

const LUMA: [f32; 3] = [0.2126, 0.7152, 0.0722];
/// The albedo detail's encoding: `value · 64 / 255`, range [0, 4) (ground-textures.md,
/// Deviation 6).
const ALBEDO_SCALE: f32 = 64.0;
/// The roughness and occlusion details' encoding: `value · 128 / 255`, range [0, 2).
const DETAIL_SCALE: f32 = 128.0;

/// A layer's two decoded array layers — (albedo detail rgb, height) and (normal u, normal v,
/// roughness detail, occlusion detail) — and the base colour's mean linear colour. `None` when
/// the base colour cannot be read.
fn derive_layer(layer: &LayerInput, n: usize) -> Option<([[Grid; 4]; 2], Vec3)> {
    let tile = |planes: [Grid; 4], lo: f32, hi: f32| {
        if layer.tileable {
            planes
        } else {
            planes.map(|p| p.tileable(lo, hi))
        }
    };
    let [r, g, b, _] = tile(read_map(&layer.albedo, n, true)?, 1e-4, 1.0);
    let mean = Vec3::new(r.mean(), g.mean(), b.mean());
    let given = |image: &Option<Image>| {
        image
            .as_ref()
            .and_then(|image| read_map(image, n, false))
            .map(|planes| tile(planes, 0.0, 1.0))
    };

    // The detail: the albedo over its own low frequencies, mean 1.
    let sigma = n as f32 / 12.0;
    let mut detail = [&r, &g, &b].map(|c| c.zip(&c.blur(sigma), |a, low| a / low.max(1e-6)));
    // De-lighting: the luminance ratio compressed by the exponent, the hue kept.
    if (layer.contrast - 1.0).abs() > 1e-6 {
        let lum = luminance(&detail).map(|l| l.max(1e-4).powf(layer.contrast - 1.0));
        detail = detail.map(|c| c.zip(&lum, |a, f| a * f));
    }
    // The mean restored after the clip, a few times, until the two agree.
    for _ in 0..6 {
        detail = detail.map(|c| {
            let mean = c.mean().max(1e-6);
            c.map(|x| (x / mean).clamp(0.0, 254.0 / ALBEDO_SCALE))
        });
    }

    // The height: a given map's or the detail's luminance, band-passed.
    let height = match given(&layer.height) {
        Some([h, ..]) => h.band_height(),
        None => luminance(&detail).band_height(),
    };

    // The normal, as the packed (u, v): u along the image's columns, v along its rows (the
    // world's +X and +Z; ground-textures.md, Deviation 13).
    let (u, v) = match given(&layer.normal) {
        Some([nx, ny, ..]) => {
            let sign = if layer.flip_normal_map_y { 1.0 } else { -1.0 };
            let s = layer.normal_scale;
            let mut u = Vec::with_capacity(n * n);
            let mut v = Vec::with_capacity(n * n);
            for (&x, &y) in nx.v.iter().zip(&ny.v) {
                let (x, y) = (2.0 * x - 1.0, 2.0 * y - 1.0);
                let z = (1.0 - x * x - y * y).max(0.0).sqrt();
                let normal = Vec3::new(x * s, y * s, z).normalize_or(Vec3::Z);
                // +Y up is −rows: an OpenGL map's y is the packed v's negative.
                u.push(normal.x);
                v.push(sign * normal.y);
            }
            (Grid { n, v: u }, Grid { n, v })
        }
        None => derived_normal(&height, layer.relief_deg),
    };

    let roughness =
        given(&layer.roughness).map_or_else(|| Grid::constant(n, 1.0), |[_, g, ..]| g.over_mean());
    let occlusion = match given(&layer.occlusion) {
        Some([o, ..]) => o.over_mean(),
        None => {
            // The depth of the cavity below its neighbourhood, only in the deepest hollows.
            let around = height.blur(n as f32 / 96.0);
            around
                .zip(&height, |a, h| 1.0 - 0.2 * ((a - h) * 4.0).clamp(0.0, 1.0))
                .over_mean()
        }
    };
    let [dr, dg, db] = detail;
    Some(([[dr, dg, db, height], [u, v, roughness, occlusion]], mean))
}

fn luminance(c: &[Grid; 3]) -> Grid {
    let mut out = c[0].map(|x| x * LUMA[0]);
    for (plane, weight) in c.iter().zip(LUMA).skip(1) {
        out = out.zip(plane, |a, b| a + b * weight);
    }
    out
}

/// The normal of a height plane as the packed (u, v), its slope scaled so the normals' mean tilt
/// is `relief_deg` (ground-textures.md, Deviation 8: a depth in metres is the wrong knob).
fn derived_normal(height: &Grid, relief_deg: f32) -> (Grid, Grid) {
    let n = height.n;
    let mut du = Vec::with_capacity(n * n);
    let mut dv = Vec::with_capacity(n * n);
    for y in 0..n as isize {
        for x in 0..n as isize {
            du.push(0.5 * (height.at(x + 1, y) - height.at(x - 1, y)));
            dv.push(0.5 * (height.at(x, y + 1) - height.at(x, y - 1)));
        }
    }
    // The scale whose mean tilt is the relief's, solved on a subsample.
    let stride = (n / 256).max(1);
    let slopes: Vec<f32> = (0..n)
        .step_by(stride)
        .flat_map(|y| (0..n).step_by(stride).map(move |x| y * n + x))
        .map(|i| du[i].hypot(dv[i]))
        .collect();
    let mean_tilt =
        |scale: f32| slopes.iter().map(|&s| (scale * s).atan()).sum::<f32>() / slopes.len() as f32;
    let target = relief_deg.to_radians();
    let (mut low, mut high) = (0.0_f32, 1.0e3_f32);
    for _ in 0..60 {
        let mid = 0.5 * (low + high);
        if mean_tilt(mid) < target {
            low = mid;
        } else {
            high = mid;
        }
    }
    let scale = 0.5 * (low + high);
    let mut u = Vec::with_capacity(n * n);
    let mut v = Vec::with_capacity(n * n);
    for (&a, &b) in du.iter().zip(&dv) {
        let (a, b) = (a * scale, b * scale);
        let norm = (a * a + b * b + 1.0).sqrt();
        u.push(-a / norm);
        v.push(-b / norm);
    }
    (Grid { n, v: u }, Grid { n, v })
}

/// A layer that changes nothing: detail 1, mid height, a flat normal, roughness and occlusion ×1.
fn neutral_layer(n: usize) -> [[Grid; 4]; 2] {
    let one = Grid::constant(n, 1.0);
    let zero = Grid::constant(n, 0.0);
    [
        [
            one.clone(),
            one.clone(),
            one.clone(),
            Grid::constant(n, 0.5),
        ],
        [zero.clone(), zero, one.clone(), one],
    ]
}

/// Encode a decoded plane value as a `Rgba8Unorm` byte.
fn unit(v: f32) -> u8 {
    (v * 255.0).round().clamp(0.0, 255.0) as u8
}

/// The six-layer `Rgba8Unorm` array with every mip level, each level the 2 × 2 average of the
/// decoded values of the one above (so the details' means stay 1 at every level), level-major.
fn encode_array(mut decoded: Vec<[[Grid; 4]; 2]>, n: usize) -> Image {
    let levels = n.ilog2() as usize + 1;
    let mut bytes = Vec::with_capacity(n * n * 4 * decoded.len() * 2 * 4 / 3 + 64);
    for level in 0..levels {
        for [a, b] in &decoded {
            let m = a[0].n;
            for i in 0..m * m {
                bytes.extend_from_slice(&[
                    unit(a[0].v[i] * ALBEDO_SCALE / 255.0),
                    unit(a[1].v[i] * ALBEDO_SCALE / 255.0),
                    unit(a[2].v[i] * ALBEDO_SCALE / 255.0),
                    unit(a[3].v[i]),
                ]);
            }
            for i in 0..m * m {
                bytes.extend_from_slice(&[
                    unit((b[0].v[i] * 127.0 + 128.0) / 255.0),
                    unit((b[1].v[i] * 127.0 + 128.0) / 255.0),
                    unit(b[2].v[i].clamp(0.0, 254.0 / DETAIL_SCALE) * DETAIL_SCALE / 255.0),
                    unit(b[3].v[i].clamp(0.0, 254.0 / DETAIL_SCALE) * DETAIL_SCALE / 255.0),
                ]);
            }
        }
        if level + 1 < levels {
            decoded = decoded
                .into_iter()
                .map(|pair| pair.map(|planes| planes.map(|p| p.halve())))
                .collect();
        }
    }
    let mut image = Image::new_uninit(
        Extent3d {
            width: n as u32,
            height: n as u32,
            depth_or_array_layers: 6,
        },
        TextureDimension::D2,
        TextureFormat::Rgba8Unorm,
        RenderAssetUsages::RENDER_WORLD,
    );
    image.texture_descriptor.mip_level_count = levels as u32;
    image.data = Some(bytes);
    image.data_order = TextureDataOrder::MipMajor;
    image.texture_view_descriptor = Some(TextureViewDescriptor {
        dimension: Some(TextureViewDimension::D2Array),
        ..default()
    });
    image.sampler = ImageSampler::Descriptor(ImageSamplerDescriptor {
        label: Some("elements_ground_textures".into()),
        address_mode_u: ImageAddressMode::Repeat,
        address_mode_v: ImageAddressMode::Repeat,
        mag_filter: ImageFilterMode::Linear,
        min_filter: ImageFilterMode::Linear,
        mipmap_filter: ImageFilterMode::Linear,
        anisotropy_clamp: 16,
        ..default()
    });
    image
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A noise plane with features of a few texels, deterministic.
    fn noise(n: usize, seed: u32) -> Grid {
        let hash = |x: u32| {
            let mut h = x.wrapping_mul(0x9E37_79B9) ^ seed.wrapping_mul(0x85EB_CA6B);
            h ^= h >> 15;
            h = h.wrapping_mul(0x2C1B_3C6D);
            h ^= h >> 12;
            (h & 0xFFFF) as f32 / 65535.0
        };
        Grid {
            n,
            v: (0..n * n).map(|i| hash(i as u32)).collect(),
        }
        .blur(1.5)
    }

    /// An `Rgba8UnormSrgb` image of `n × n` from a colour per texel.
    fn image(n: usize, texel: impl Fn(usize, usize) -> [u8; 4]) -> Image {
        let data: Vec<u8> = (0..n * n).flat_map(|i| texel(i % n, i / n)).collect();
        Image::new(
            Extent3d {
                width: n as u32,
                height: n as u32,
                depth_or_array_layers: 1,
            },
            TextureDimension::D2,
            data,
            TextureFormat::Rgba8UnormSrgb,
            RenderAssetUsages::default(),
        )
    }

    fn input(albedo: Image) -> LayerInput {
        LayerInput {
            albedo,
            normal: None,
            roughness: None,
            occlusion: None,
            height: None,
            tile_m: 2.0,
            colour: 1.0,
            normal_scale: 1.0,
            flip_normal_map_y: false,
            relief_deg: 10.0,
            contrast: 1.0,
            tileable: true,
        }
    }

    /// The decoded value of byte `b` of every texel of array layer `layer` at mip `level`.
    fn channel(image: &Image, n: usize, level: usize, layer: usize, c: usize) -> Vec<f32> {
        let data = image.data.as_ref().expect("packed data");
        let offset: usize = (0..level).map(|l| (n >> l) * (n >> l) * 4 * 6).sum();
        let m = n >> level;
        let start = offset + layer * m * m * 4;
        (0..m * m)
            .map(|i| f32::from(data[start + 4 * i + c]) / 255.0)
            .collect()
    }

    fn mean(v: &[f32]) -> f32 {
        v.iter().sum::<f32>() / v.len() as f32
    }

    #[test]
    fn the_box_cascade_is_the_gaussian_it_was_asked_for() {
        for sigma in [1.2_f32, 2.7, 5.0, 85.3] {
            let widths = box_widths(sigma, 4);
            let variance: f32 = widths.iter().map(|&w| (w * w - 1) as f32 / 12.0).sum();
            // Kovesi: the integer widths reach σ within his bound (0.17 at n = 5; four passes).
            assert!(
                (variance.sqrt() - sigma).abs() < 0.25,
                "σ {sigma}: {widths:?}"
            );
            assert!(widths.iter().all(|w| w % 2 == 1));
        }
        // An impulse spreads to the Gaussian's variance and keeps its sum.
        let n = 128;
        let mut impulse = Grid::constant(n, 0.0);
        impulse.v[64 * n + 64] = 1.0;
        let blurred = impulse.blur(6.0);
        let total: f32 = blurred.v.iter().sum();
        assert!((total - 1.0).abs() < 1e-4);
        let var: f32 = (0..n)
            .map(|x| blurred.v[64 * n + x] * (x as f32 - 64.0).powi(2))
            .sum::<f32>()
            / (0..n).map(|x| blurred.v[64 * n + x]).sum::<f32>();
        assert!((var.sqrt() - 6.0).abs() < 0.3, "σ {}", var.sqrt());
    }

    #[test]
    fn every_level_of_the_details_averages_to_one() {
        let n = 64;
        let (r, g, b) = (noise(n, 1), noise(n, 2), noise(n, 3));
        let albedo = image(n, |x, y| {
            let i = y * n + x;
            [
                unit(0.2 + 0.5 * r.v[i]),
                unit(0.3 + 0.4 * g.v[i]),
                unit(0.1 + 0.3 * b.v[i]),
                255,
            ]
        });
        let mut layer = input(albedo);
        layer.tileable = false;
        layer.contrast = 0.65;
        let Some((pair, _)) = derive_layer(&layer, n) else {
            panic!("the albedo is readable");
        };
        let packed = encode_array(vec![neutral_layer(n), pair, neutral_layer(n)], n);
        for level in 0..=n.ilog2() as usize {
            for c in 0..3 {
                let detail = mean(&channel(&packed, n, level, 2, c)) * 255.0 / ALBEDO_SCALE;
                assert!(
                    (detail - 1.0).abs() < 0.02,
                    "level {level} channel {c}: {detail}"
                );
            }
            for c in [2, 3] {
                let detail = mean(&channel(&packed, n, level, 3, c)) * 255.0 / DETAIL_SCALE;
                assert!(
                    (detail - 1.0).abs() < 0.02,
                    "level {level} detail {c}: {detail}"
                );
            }
        }
        // The neutral layers decode to no change at every texel.
        assert!(
            channel(&packed, n, 0, 0, 0)
                .iter()
                .all(|&x| x == 64.0 / 255.0)
        );
        assert!(
            channel(&packed, n, 0, 1, 0)
                .iter()
                .all(|&x| x == 128.0 / 255.0)
        );
    }

    #[test]
    fn a_flat_colour_is_no_detail_and_its_own_colour() {
        let n = 32;
        let Some(([detail, _], mean)) =
            derive_layer(&input(image(n, |_, _| [188, 140, 90, 255])), n)
        else {
            panic!("readable");
        };
        for c in &detail[..3] {
            assert!(c.v.iter().all(|&x| (x - 1.0).abs() < 1e-4));
        }
        let expect = Vec3::new(188.0, 140.0, 90.0).map(|c| srgb_to_linear(c / 255.0));
        assert!(
            (mean - expect).abs().max_element() < 1e-5,
            "{mean} vs {expect}"
        );
    }

    #[test]
    fn normal_maps_of_either_convention_agree_with_the_heights_slope() {
        // A ridge along the image's columns: the height rises toward +x for x < n/2.
        let n = 32;
        let height = Grid {
            n,
            v: (0..n * n)
                .map(|i| 0.5 + 0.4 * (std::f32::consts::TAU * (i % n) as f32 / n as f32).sin())
                .collect(),
        };
        let (du, _) = derived_normal(&height, 20.0);
        // A bump whose y slope is non-zero: the height rises toward the image's top rows.
        let tilted = Grid {
            n,
            v: (0..n * n)
                .map(|i| 0.5 + 0.4 * (std::f32::consts::TAU * (i / n) as f32 / n as f32).sin())
                .collect(),
        };
        let (_, dv) = derived_normal(&tilted, 20.0);
        // The same surfaces' normals as OpenGL (+Y up = −rows) and DirectX normal maps.
        let gl = image(n, |x, y| {
            let (u, v) = (du.v[y * n + x], dv.v[y * n + x]);
            [unit(0.5 * u + 0.5), unit(0.5 * -v + 0.5), 255, 255]
        });
        let dx = image(n, |x, y| {
            let (u, v) = (du.v[y * n + x], dv.v[y * n + x]);
            [unit(0.5 * u + 0.5), unit(0.5 * v + 0.5), 255, 255]
        });
        let flat = image(n, |_, _| [128, 128, 128, 255]);
        let mut opengl = input(flat.clone());
        opengl.normal = Some(gl);
        let mut directx = input(flat);
        directx.normal = Some(dx);
        directx.flip_normal_map_y = true;
        let unpack = |layer: &LayerInput| {
            // The maps' bytes are sRGB-flagged; the normal must be read as stored, not decoded.
            let Some(([_, [u, v, ..]], _)) = derive_layer(layer, n) else {
                panic!("readable");
            };
            (u, v)
        };
        let (ug, vg) = unpack(&opengl);
        let (ux, vx) = unpack(&directx);
        for i in 0..n * n {
            assert!(
                (ug.v[i] - du.v[i]).abs() < 0.02,
                "u {i}: {} vs {}",
                ug.v[i],
                du.v[i]
            );
            assert!(
                (vg.v[i] - dv.v[i]).abs() < 0.02,
                "v {i}: {} vs {}",
                vg.v[i],
                dv.v[i]
            );
            // The two maps are quantised separately, so they may differ by one 8-bit step (2/255);
            // a wrong flip would differ by 2|v|.
            assert!((ux.v[i] - ug.v[i]).abs() < 0.01 && (vx.v[i] - vg.v[i]).abs() < 0.01);
        }
    }

    /// terrain-textures.md, Validation target 3: a set put on the terrain is packed; a changed
    /// set is packed while the old one stays bound; a removed set leaves the procedural ground.
    #[test]
    fn a_set_on_the_terrain_is_packed_and_follows_it() {
        let mut app = App::new();
        app.add_plugins((MinimalPlugins, AssetPlugin::default()))
            .init_asset::<Image>()
            .init_resource::<GroundTexturePack>()
            .add_systems(Update, pack_terrain_textures);
        let albedo = app
            .world_mut()
            .resource_mut::<Assets<Image>>()
            .add(image(16, |x, y| {
                [100 + (x * 7 % 50) as u8, 120, 60 + (y * 5 % 40) as u8, 255]
            }));
        let set = |tile_m: f32| TerrainTextures {
            grass: Some(GroundLayerTextures::new(albedo.clone(), tile_m)),
            resolution: 16,
            ..default()
        };
        let terrain = app.world_mut().spawn((Terrain::flat(0.0), set(2.0))).id();
        let settle = |app: &mut App, done: &dyn Fn(&GroundTexturePack) -> bool| {
            for _ in 0..1000 {
                app.update();
                if done(app.world().resource::<GroundTexturePack>()) {
                    return true;
                }
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
            false
        };
        assert!(
            settle(&mut app, &|p| p.image.is_some()),
            "the set was never packed"
        );
        let pack = app.world().resource::<GroundTexturePack>();
        let first = pack.image.clone();
        assert_eq!(
            pack.ground,
            Vec4::new(0.0, 0.0, 2.0, 0.3),
            "only the grass is textured"
        );
        assert_eq!(
            pack.colours[2].w, 0.0,
            "the crate's colour is kept by default"
        );

        app.world_mut().entity_mut(terrain).insert(set(3.0));
        app.update();
        assert_eq!(
            app.world().resource::<GroundTexturePack>().image,
            first,
            "the old set stays bound"
        );
        assert!(
            settle(&mut app, &|p| p.ground.z == 3.0),
            "the change was never packed"
        );
        assert_ne!(app.world().resource::<GroundTexturePack>().image, first);

        app.world_mut()
            .entity_mut(terrain)
            .remove::<TerrainTextures>();
        app.update();
        let pack = app.world().resource::<GroundTexturePack>();
        assert!(
            pack.image.is_none() && pack.ground == Vec4::ZERO,
            "the ground is procedural"
        );
    }

    #[test]
    fn given_roughness_and_occlusion_are_details_with_mean_one() {
        let n = 32;
        let r = noise(n, 7);
        let mut layer = input(image(n, |_, _| [120, 120, 120, 255]));
        // glTF: roughness in G, occlusion in R; a packed ORM map serves both.
        let orm = image(n, |x, y| {
            let v = r.v[y * n + x];
            [unit(0.4 + 0.5 * v), unit(0.3 + 0.6 * (1.0 - v)), 0, 255]
        });
        layer.roughness = Some(orm.clone());
        layer.occlusion = Some(orm);
        let Some(([_, [_, _, rough, occlusion]], _)) = derive_layer(&layer, n) else {
            panic!("readable");
        };
        assert!((rough.mean() - 1.0).abs() < 1e-4 && (occlusion.mean() - 1.0).abs() < 1e-4);
        // Rougher where the map's G is higher, more occluded where its R is lower.
        let i_max = (0..n * n)
            .max_by(|&a, &b| r.v[a].total_cmp(&r.v[b]))
            .unwrap_or(0);
        let i_min = (0..n * n)
            .min_by(|&a, &b| r.v[a].total_cmp(&r.v[b]))
            .unwrap_or(0);
        assert!(rough.v[i_min] > rough.v[i_max]);
        assert!(occlusion.v[i_max] > occlusion.v[i_min]);
    }

    #[test]
    fn a_large_map_is_packed_at_the_resolution() {
        let n = 32;
        // A 2:1 checker of 2 × 2 texel cells, 64 × 128: halves to 32 × 64, then 32 × 32.
        let big = Image::new(
            Extent3d {
                width: 64,
                height: 128,
                depth_or_array_layers: 1,
            },
            TextureDimension::D2,
            (0..64 * 128)
                .flat_map(|i| {
                    let v = if ((i % 64) / 2 + (i / 64) / 2) % 2 == 0 {
                        255
                    } else {
                        0
                    };
                    [v, v, v, 255]
                })
                .collect(),
            TextureFormat::Rgba8Unorm,
            RenderAssetUsages::default(),
        );
        let Some(planes) = read_map(&big, n, false) else {
            panic!("readable");
        };
        assert_eq!(planes[0].v.len(), n * n);
        assert!((planes[0].mean() - 0.5).abs() < 0.01);
    }
}
