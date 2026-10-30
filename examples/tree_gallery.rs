//! Tree skeleton gallery, rendered headless to a PNG (R3.9 step 2).
//!
//! Run with `cargo run --example tree_gallery`. Produces `target/tree_gallery.png`:
//! one **orthographic sheet per species**, four seed variants side by side,
//! stacked top to bottom with a caption strip.
//!
//! Why sheets instead of one perspective view: the first version framed all
//! three species with a single perspective camera. The rows skewed into an
//! unreadable cluster and neighbouring crowns merged, so no individual tree
//! could be judged. An orthographic sheet has no vanishing point: columns line
//! up, spacing is real metres, and variants are comparable.
//!
//! Drawing: branches are capsules with a world-space radius, leaf clusters are
//! shaded spheres — i.e. what the eventual opaque instanced geometry will look
//! like from outside. Both are z-buffered, so overlap resolves correctly.
//!
//! Output: plain PPM (P6) plus a small stored-deflate PNG writer, so the
//! example needs no image crate.

use heads_up_bevy::world::tree_shape::{TREE_PROFILES, TreeSkeleton, grow_tree};

const WIDTH: usize = 2400;
const HEIGHT: usize = 1560;
/// Seed variants per species (one column each).
const VARIANTS: u64 = 4;
/// Minimum centre-to-centre column pitch (m); the real pitch is
/// `crown_width + CROWN_GAP_M`, never less than this.
const COLUMN_SPACING_MIN: f32 = 11.0;
/// Clear air left between two neighbouring crowns (m).
const CROWN_GAP_M: f32 = 2.5;

/// Per-species sheet height in pixels (three sheets stack to `HEIGHT`).
const SHEET_HEIGHT: usize = 500;
const SHEET_GAP: usize = 20;
/// Height of the caption strip at the top of a sheet.
const SHEET_CAPTION_H: usize = 34;
/// Room under the ground line for the variant label.
const SHEET_BASELINE: usize = 40;
const SHEET_MARGIN_X: f32 = 60.0;
/// Upper bound on pixels per metre, so a short species is not blown up until
/// its twigs look like a different tree.
const SHEET_MAX_PX_PER_M: f32 = 30.0;

const GALLERY_SEED: u64 = 0x4845_4144; // "HEAD"

/// Sheet camera: looking down -Z with a slight yaw so a row does not stack
/// into itself.
const SHEET_YAW: f32 = 0.20;

/// One end of a rasterized capsule, already in sheet pixels.
struct CapsuleEnd {
    /// Pixel position (x right, y down).
    pos: (f32, f32),
    /// View depth along the sheet camera's axis.
    depth: f32,
    /// Branch radius in pixels.
    radius: f32,
}

struct Canvas {
    width: usize,
    height: usize,
    color: Vec<[f32; 3]>,
    depth: Vec<f32>,
}

impl Canvas {
    fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            // Sky-ish background so silhouettes have something to read against.
            color: vec![[0.10, 0.12, 0.18]; width * height],
            depth: vec![f32::INFINITY; width * height],
        }
    }

    /// Splats a world-space capsule (branch) into the z-buffer.
    ///
    /// Both ends carry their sheet pixel position, view depth and radius in
    /// one tuple: eight loose scalars tripped clippy's argument limit.
    fn capsule(&mut self, a: CapsuleEnd, b: CapsuleEnd, tint: [f32; 3]) {
        let steps = ((a.pos.0 - b.pos.0).abs().max((a.pos.1 - b.pos.1).abs())) as usize + 1;
        for step in 0..=steps.min(3000) {
            let t = step as f32 / steps.min(3000) as f32;
            let x = a.pos.0 + (b.pos.0 - a.pos.0) * t;
            let y = a.pos.1 + (b.pos.1 - a.pos.1) * t;
            let depth = a.depth + (b.depth - a.depth) * t;
            let r = (a.radius + (b.radius - a.radius) * t).max(0.8);
            let ci = x as i32;
            let cj = y as i32;
            let ri = r.ceil() as i32;
            for dy in -ri..=ri {
                for dx in -ri..=ri {
                    if (dx * dx + dy * dy) as f32 > r * r {
                        continue;
                    }
                    self.put(ci + dx, cj + dy, depth, tint);
                }
            }
        }
    }

    fn put(&mut self, x: i32, y: i32, depth: f32, color: [f32; 3]) {
        if x < 0 || y < 0 || x >= self.width as i32 || y >= self.height as i32 {
            return;
        }
        let index = y as usize * self.width + x as usize;
        if depth < self.depth[index] {
            self.depth[index] = depth;
            self.color[index] = color;
        }
    }

    fn fill_rect(&mut self, x: usize, y: usize, w: usize, h: usize, color: [f32; 3]) {
        for row in y..(y + h).min(self.height) {
            for col in x..(x + w).min(self.width) {
                self.color[row * self.width + col] = color;
            }
        }
    }

    /// 5x7 bitmap text, so the sheet is self-describing without a font asset.
    fn text(&mut self, x: f32, y: f32, text: &str, color: [f32; 3]) {
        let mut cursor = x;
        for ch in text.chars() {
            let glyph = glyph(ch);
            for (row, bits) in glyph.iter().enumerate() {
                for col in 0..5 {
                    if bits & (1 << (4 - col)) != 0 {
                        self.fill_rect(cursor as usize + col, y as usize + row, 2, 2, color);
                    }
                }
            }
            cursor += 12.0;
        }
    }
}

/// 5x7 glyphs for the characters the captions need (uppercase + digits +
/// a few marks). Unknown characters render as a blank cell.
fn glyph(ch: char) -> [u8; 7] {
    match ch.to_ascii_uppercase() {
        'A' => [0x0E, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11],
        'B' => [0x1E, 0x11, 0x11, 0x1E, 0x11, 0x11, 0x1E],
        'C' => [0x0E, 0x11, 0x10, 0x10, 0x10, 0x11, 0x0E],
        'D' => [0x1E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x1E],
        'E' => [0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x1F],
        'F' => [0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x10],
        'G' => [0x0E, 0x11, 0x10, 0x17, 0x11, 0x11, 0x0F],
        'H' => [0x11, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11],
        'I' => [0x0E, 0x04, 0x04, 0x04, 0x04, 0x04, 0x0E],
        'J' => [0x07, 0x02, 0x02, 0x02, 0x02, 0x12, 0x0C],
        'K' => [0x11, 0x12, 0x14, 0x18, 0x14, 0x12, 0x11],
        'L' => [0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x1F],
        'M' => [0x11, 0x1B, 0x15, 0x15, 0x11, 0x11, 0x11],
        'N' => [0x11, 0x19, 0x15, 0x13, 0x11, 0x11, 0x11],
        'O' => [0x0E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E],
        'P' => [0x1E, 0x11, 0x11, 0x1E, 0x10, 0x10, 0x10],
        'Q' => [0x0E, 0x11, 0x11, 0x11, 0x15, 0x12, 0x0D],
        'R' => [0x1E, 0x11, 0x11, 0x1E, 0x14, 0x12, 0x11],
        'S' => [0x0F, 0x10, 0x10, 0x0E, 0x01, 0x01, 0x1E],
        'T' => [0x1F, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04],
        'U' => [0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E],
        'V' => [0x11, 0x11, 0x11, 0x11, 0x11, 0x0A, 0x04],
        'W' => [0x11, 0x11, 0x11, 0x15, 0x15, 0x1B, 0x11],
        'X' => [0x11, 0x11, 0x0A, 0x04, 0x0A, 0x11, 0x11],
        'Y' => [0x11, 0x11, 0x0A, 0x04, 0x04, 0x04, 0x04],
        'Z' => [0x1F, 0x01, 0x02, 0x04, 0x08, 0x10, 0x1F],
        '0' => [0x0E, 0x11, 0x13, 0x15, 0x19, 0x11, 0x0E],
        '1' => [0x04, 0x0C, 0x04, 0x04, 0x04, 0x04, 0x0E],
        '2' => [0x0E, 0x11, 0x01, 0x02, 0x04, 0x08, 0x1F],
        '3' => [0x1F, 0x02, 0x04, 0x02, 0x01, 0x11, 0x0E],
        '4' => [0x02, 0x06, 0x0A, 0x12, 0x1F, 0x02, 0x02],
        '5' => [0x1F, 0x10, 0x1E, 0x01, 0x01, 0x11, 0x0E],
        '6' => [0x06, 0x08, 0x10, 0x1E, 0x11, 0x11, 0x0E],
        '7' => [0x1F, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08],
        '8' => [0x0E, 0x11, 0x11, 0x0E, 0x11, 0x11, 0x0E],
        '9' => [0x0E, 0x11, 0x11, 0x0F, 0x01, 0x02, 0x0C],
        '-' => [0x00, 0x00, 0x00, 0x1F, 0x00, 0x00, 0x00],
        ',' => [0x00, 0x00, 0x00, 0x00, 0x00, 0x0C, 0x08],
        '.' => [0x00, 0x00, 0x00, 0x00, 0x00, 0x0C, 0x0C],
        ':' => [0x00, 0x0C, 0x0C, 0x00, 0x0C, 0x0C, 0x00],
        _ => [0, 0, 0, 0, 0, 0, 0],
    }
}

/// Orthographic sheet projection: world metres -> sheet pixels.
struct Sheet {
    /// Pixels per metre.
    scale: f32,
    /// Sheet pixel column of world x = 0.
    origin_x: f32,
    /// Sheet pixel row of world y = 0 (the ground line).
    ground_y: f32,
    /// Camera yaw (radians) around +Y, applied before projection.
    yaw: f32,
}

impl Sheet {
    /// Projects a world point (tree-local metres) to (pixel, camera depth).
    fn project(&self, p: [f32; 3]) -> ((f32, f32), f32) {
        let (sin, cos) = self.yaw.sin_cos();
        // Yaw about +Y: rotate x/z, keeping y.
        let x = p[0] * cos + p[2] * sin;
        let z = -p[0] * sin + p[2] * cos;
        (
            (
                self.origin_x + x * self.scale,
                self.ground_y - p[1] * self.scale,
            ),
            z,
        )
    }
}

/// Draws one tree into the canvas on its sheet.
fn draw_tree(
    canvas: &mut Canvas,
    tree: &TreeSkeleton,
    sheet: &Sheet,
    species: u8,
    atlas: &heads_up_bevy::world::leaf_shape::AlphaMask,
) {
    for branch in &tree.branches {
        let (a, da) = sheet.project(branch.start);
        let (b, db) = sheet.project(branch.end);
        // Slight exaggeration: a 3 cm twig is sub-pixel at sheet scale.
        let radius_px = (branch.radius * sheet.scale).max(0.9);
        let tint = match branch.level {
            0 => [0.40, 0.29, 0.20],
            1 => [0.50, 0.39, 0.27],
            _ => [0.60, 0.51, 0.38],
        };
        canvas.capsule(
            CapsuleEnd {
                pos: a,
                depth: da,
                radius: radius_px,
            },
            CapsuleEnd {
                pos: b,
                depth: db,
                radius: radius_px,
            },
            tint,
        );
    }
    // Foliage as the game now draws it: crossed leaf cards, not blobs.
    //
    // The sheet used to draw shaded spheres because that WAS the shipped
    // geometry. Cards are what ships now, and a sphere sheet would flatter the
    // design -- the whole point of the switch is the outline, which a blob
    // cannot show. Each card is rasterised as a textured quad sampling the
    // species' alpha mask, so what this sheet shows is what the GPU draws.
    for foliage in &tree.foliage {
        draw_leaf_cards(canvas, foliage, sheet, species, atlas);
    }
}

/// Rasterises a cluster's leaf cards the way the GPU does: textured quads with
/// an alpha test, not outlines.
///
/// The first version of this sheet drew card BORDERS, which made the crown look
/// like a wireframe of big rectangles -- the opposite of what ships. It also
/// hid a real problem: the cards were as large as the cluster, so even with a
/// texture they would read as slabs. Drawing the texture is what makes this
/// sheet worth having.
fn draw_leaf_cards(
    canvas: &mut Canvas,
    foliage: &heads_up_bevy::world::tree_shape::Foliage,
    sheet: &Sheet,
    species: u8,
    atlas: &heads_up_bevy::world::leaf_shape::AlphaMask,
) {
    // Ask the GAME for the geometry instead of re-implementing it here.
    //
    // The first version copied the card placement into the example and the two
    // drifted immediately: the sheet kept showing randomly-oriented cards after
    // the game had been fixed to point petioles at the branch. A preview that
    // can disagree with the shader is worse than no preview.
    let skeleton = heads_up_bevy::world::tree_shape::TreeSkeleton {
        branches: Vec::new(),
        foliage: vec![*foliage],
        species: 0,
        total_length: 0.0,
    };
    let profile = heads_up_bevy::world::tree_shape::TREE_PROFILES[species as usize];
    let parts = heads_up_bevy::world::tree_mesh::bake_skeleton(&skeleton, &profile, species);
    let Some(positions) = parts.leaves.attribute(bevy::mesh::Mesh::ATTRIBUTE_POSITION) else {
        return;
    };
    let bevy::mesh::VertexAttributeValues::Float32x3(positions) = positions else {
        return;
    };
    let Some(uvs) = parts.leaves.attribute(bevy::mesh::Mesh::ATTRIBUTE_UV_0) else {
        return;
    };
    let bevy::mesh::VertexAttributeValues::Float32x2(uvs) = uvs else {
        return;
    };

    for (quad, quad_uv) in positions.chunks(4).zip(uvs.chunks(4)) {
        if quad.len() < 4 {
            continue;
        }
        let projected: Vec<((f32, f32), f32, f32, f32)> = quad
            .iter()
            .zip(quad_uv.iter())
            .map(|(p, uv)| {
                let (px, depth) = sheet.project(*p);
                (px, depth, uv[0], uv[1])
            })
            .collect();

        // Two triangles, barycentric fill, alpha-tested against the mask.
        for tri in [[0usize, 1, 2], [0, 2, 3]] {
            let p0 = projected[tri[0]];
            let p1 = projected[tri[1]];
            let p2 = projected[tri[2]];
            let min_x = p0.0.0.min(p1.0.0).min(p2.0.0).floor() as i32;
            let max_x = p0.0.0.max(p1.0.0).max(p2.0.0).ceil() as i32;
            let min_y = p0.0.1.min(p1.0.1).min(p2.0.1).floor() as i32;
            let max_y = p0.0.1.max(p1.0.1).max(p2.0.1).ceil() as i32;
            let area = edge(p0.0, p1.0, p2.0);
            if area.abs() < 1e-6 {
                continue;
            }
            for py in min_y..=max_y {
                for px in min_x..=max_x {
                    let p = (px as f32 + 0.5, py as f32 + 0.5);
                    let w0 = edge(p1.0, p2.0, p) / area;
                    let w1 = edge(p2.0, p0.0, p) / area;
                    let w2 = edge(p0.0, p1.0, p) / area;
                    if w0 < 0.0 || w1 < 0.0 || w2 < 0.0 {
                        continue;
                    }
                    let u = w0 * p0.2 + w1 * p1.2 + w2 * p2.2;
                    let v = w0 * p0.3 + w1 * p1.3 + w2 * p2.3;
                    // `u` already lies in the species' atlas cell (the game
                    // writes atlas-space UVs), so sample the shared atlas.
                    let tx = (u * atlas.width as f32).clamp(0.0, atlas.width as f32 - 1.0) as u32;
                    let ty = (v * atlas.height as f32).clamp(0.0, atlas.height as f32 - 1.0) as u32;
                    if atlas.get(tx, ty) < 128 {
                        continue;
                    }
                    let depth = w0 * p0.1 + w1 * p1.1 + w2 * p2.1;
                    canvas.put(px, py, depth, [0.10, 0.20, 0.10]);
                }
            }
        }
    }
}

/// Signed area of the triangle (a, b, c): positive for one winding.
fn edge(a: (f32, f32), b: (f32, f32), c: (f32, f32)) -> f32 {
    (b.0 - a.0) * (c.1 - a.1) - (b.1 - a.1) * (c.0 - a.0)
}

fn main() {
    let mut canvas = Canvas::new(WIDTH, HEIGHT);

    // The world's single shared atlas (the same one `chunks.rs` builds); each
    // species' cards sample their own cell via the UVs the game bakes.
    let atlas = heads_up_bevy::world::leaf_shape::bake_species_atlas(
        heads_up_bevy::world::leaf_shape::ATLAS_HEIGHT,
    );

    for (species, profile) in TREE_PROFILES.iter().enumerate() {
        let top = species * (SHEET_HEIGHT + SHEET_GAP);
        canvas.fill_rect(0, top, WIDTH, SHEET_HEIGHT, [0.13, 0.15, 0.21]);

        let title = match species {
            0 => "0 ASPEN/YANG - SLENDER, TALL, UPRIGHT (deltoid, serrate)",
            1 => "1 STREET/HUAI - ROUNDER CROWN, WIDER SPREAD (pinnate compound)",
            _ => "2 BROADLEAF/LIU - LOW FORK, HEAVY LIMBS (lanceolate)",
        };
        canvas.text(SHEET_MARGIN_X, (top + 10) as f32, title, [0.92, 0.90, 0.85]);

        // Build this species' row, then frame it.
        let trees: Vec<TreeSkeleton> = (0..VARIANTS)
            .map(|variant| grow_tree(GALLERY_SEED, species as u8, variant))
            .collect();
        let max_h = trees
            .iter()
            .map(|t| t.crown_height())
            .fold(0.0f32, f32::max);
        let max_w = trees
            .iter()
            .map(|t| t.crown_width())
            .fold(profile.width, f32::max);

        // Column pitch follows the measured crown width with a hard gap, so a
        // 9.5 m broadleaf crown never touches its neighbour. A fixed pitch
        // merged the wide species into a hedge.
        let column_spacing = (max_w + CROWN_GAP_M).max(COLUMN_SPACING_MIN);
        let row_world = (VARIANTS as f32 - 1.0) * column_spacing + max_w;
        let usable_w = WIDTH as f32 - 2.0 * SHEET_MARGIN_X;
        let usable_h = (SHEET_HEIGHT - SHEET_CAPTION_H - SHEET_BASELINE) as f32;
        let scale = (usable_w / row_world)
            .min(usable_h / max_h.max(1.0))
            .min(SHEET_MAX_PX_PER_M);

        let ground_y = (top + SHEET_HEIGHT - SHEET_BASELINE) as f32;
        let first_x = WIDTH as f32 * 0.5 - (VARIANTS as f32 - 1.0) * column_spacing * scale * 0.5;

        // Ground line, so every tree's footing is comparable.
        canvas.fill_rect(
            SHEET_MARGIN_X as usize,
            ground_y as usize,
            (WIDTH as f32 - 2.0 * SHEET_MARGIN_X) as usize,
            1,
            [0.30, 0.32, 0.36],
        );

        for (variant, tree) in trees.iter().enumerate() {
            let sheet = Sheet {
                scale,
                origin_x: first_x + variant as f32 * column_spacing * scale,
                ground_y,
                yaw: SHEET_YAW,
            };
            draw_tree(&mut canvas, tree, &sheet, species as u8, &atlas);
            canvas.text(
                sheet.origin_x - 30.0,
                ground_y + 12.0,
                &format!("V{variant}"),
                [0.72, 0.74, 0.78],
            );
        }
    }

    // Write PPM, then PNG.
    let path = std::path::Path::new("target/tree_gallery.ppm");
    write_ppm(&canvas, path).expect("write ppm");
    println!("wrote {}", path.display());
    let png = std::path::Path::new("target/tree_gallery.png");
    match ppm_to_png(path, png) {
        Ok(()) => println!("wrote {}", png.display()),
        Err(err) => println!("png conversion unavailable ({err}); use {}", path.display()),
    }
}
fn write_ppm(canvas: &Canvas, path: &std::path::Path) -> std::io::Result<()> {
    use std::io::Write;
    let mut file = std::fs::File::create(path)?;
    write!(file, "P6\n{} {}\n255\n", canvas.width, canvas.height)?;
    let mut bytes = Vec::with_capacity(canvas.width * canvas.height * 3);
    for c in &canvas.color {
        for channel in c {
            // Gamma-ish encode so the preview matches what a tonemapper would show.
            bytes.push((channel.max(0.0).powf(1.0 / 2.2) * 255.0).clamp(0.0, 255.0) as u8);
        }
    }
    file.write_all(&bytes)
}

/// Minimal 8-bit RGB PNG encoder (stored-deflate), so the example has no
/// image-crate dependency.
fn ppm_to_png(
    ppm: &std::path::Path,
    png: &std::path::Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let data = std::fs::read(ppm)?;
    let mut header_end = 0;
    let mut newlines = 0;
    for (i, b) in data.iter().enumerate() {
        if *b == b'\n' {
            newlines += 1;
            if newlines == 3 {
                header_end = i + 1;
                break;
            }
        }
    }
    let header = std::str::from_utf8(&data[..header_end])?;
    let mut parts = header.split_whitespace();
    let _magic = parts.next();
    let width: u32 = parts.next().ok_or("width")?.parse()?;
    let height: u32 = parts.next().ok_or("height")?.parse()?;
    let pixels = &data[header_end..];

    // Raw scanlines with filter byte 0.
    let mut raw = Vec::with_capacity(pixels.len() + height as usize);
    for y in 0..height as usize {
        raw.push(0u8);
        let row = y * width as usize * 3;
        raw.extend_from_slice(&pixels[row..row + width as usize * 3]);
    }
    let compressed = zlib_stored(&raw);

    let mut out = Vec::new();
    out.extend_from_slice(&[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]);
    let mut ihdr = Vec::new();
    ihdr.extend_from_slice(&width.to_be_bytes());
    ihdr.extend_from_slice(&height.to_be_bytes());
    ihdr.extend_from_slice(&[8, 2, 0, 0, 0]);
    write_chunk(&mut out, b"IHDR", &ihdr);
    write_chunk(&mut out, b"IDAT", &compressed);
    write_chunk(&mut out, b"IEND", &[]);
    std::fs::write(png, out)?;
    Ok(())
}

fn write_chunk(out: &mut Vec<u8>, kind: &[u8; 4], data: &[u8]) {
    out.extend_from_slice(&(data.len() as u32).to_be_bytes());
    let start = out.len();
    out.extend_from_slice(kind);
    out.extend_from_slice(data);
    let crc = crc32(&out[start..]);
    out.extend_from_slice(&crc.to_be_bytes());
}

fn crc32(data: &[u8]) -> u32 {
    let mut table = [0u32; 256];
    for (i, entry) in table.iter_mut().enumerate() {
        let mut c = i as u32;
        for _ in 0..8 {
            c = if c & 1 != 0 {
                0xEDB8_8320 ^ (c >> 1)
            } else {
                c >> 1
            };
        }
        *entry = c;
    }
    let mut crc = 0xFFFF_FFFFu32;
    for byte in data {
        crc = table[((crc ^ *byte as u32) & 0xFF) as usize] ^ (crc >> 8);
    }
    crc ^ 0xFFFF_FFFF
}

fn adler32(data: &[u8]) -> u32 {
    let (mut a, mut b) = (1u32, 0u32);
    for byte in data {
        a = (a + *byte as u32) % 65521;
        b = (b + a) % 65521;
    }
    (b << 16) | a
}

fn zlib_stored(data: &[u8]) -> Vec<u8> {
    let mut out = vec![0x78, 0x01];
    let mut offset = 0;
    while offset < data.len() {
        let len = (data.len() - offset).min(65535);
        let last = offset + len == data.len();
        out.push(if last { 1 } else { 0 });
        out.extend_from_slice(&(len as u16).to_le_bytes());
        out.extend_from_slice(&(!(len as u16)).to_le_bytes());
        out.extend_from_slice(&data[offset..offset + len]);
        offset += len;
    }
    out.extend_from_slice(&adler32(data).to_be_bytes());
    out
}
