//! Bakes every facade and writes a contact sheet (R3.11 step 11.1).
//!
//! Run with `cargo run --example facade_gallery`. Writes, under the project's
//! gitignored draft area `.tmp/facade_gallery/`:
//!
//! - `<style>_<variant>.png` -- the albedo at full resolution, which is what the
//!   player sees on the wall.
//! - `<style>_light.png`    -- albedo plus the emissive map, i.e. what the dusk
//!   look actually delivers once the material's `emissive` luminance is applied.
//!   The plain albedo hides the lit windows entirely, so this is the sheet to
//!   judge "is the city awake" from.
//! - `facade_sheet.png`     -- all styles and variants as a grid at a common
//!   physical scale, with one line of captions.
//!
//! This is a CPU rasteriser like `tree_gallery`/`leaf_atlas`: no GPU, no window,
//! and therefore none of the headless-camera problems those tools documented.
//! The contact sheet is not a substitute for standing in the street, but it does
//! show the two things unit tests cannot: whether the rhythm reads, and whether
//! the wall is too bright for a silhouette.

use no_one_looks_up::plugins::facade::WINDOW_EMISSIVE;
use no_one_looks_up::world::facade::{
    BASE_SIZE, BAYS_PER_TILE, FacadeStyle, VARIANT_COUNT, bake_facade, sample,
};

/// Metres per pixel in the sheet. 55 px/m puts a 12 m storey band at 1320 px,
/// so the sheet is legible at 100% without being enormous.
const SHEET_PX_PER_M: f32 = 55.0;
/// Sheet layout.
const CAPTION_H: u32 = 34;
const GAP: u32 = 26;
const MARGIN: u32 = 26;
const BG: [u8; 3] = [18, 20, 22];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out_dir = std::path::Path::new(".tmp/facade_gallery");
    std::fs::create_dir_all(out_dir)?;

    for style in FacadeStyle::ALL {
        let dims = style.dims();
        let tile = dims.tile();
        for variant in 0..VARIANT_COUNT {
            let tex = bake_facade(style, variant);
            let path = out_dir.join(format!("{}_{variant}.png", style.label()));
            write_png_rgba(&path, BASE_SIZE, BASE_SIZE, &tex.base)?;
            println!("{} variant {variant}: {}", style.label(), path.display());
        }
        println!(
            "  tile {:.1} x {:.1} m ({} bays of {:.1} m, {} floors of {:.1} m)",
            tile.x,
            tile.y,
            BAYS_PER_TILE,
            dims.bay,
            no_one_looks_up::world::facade::FLOORS_PER_TILE,
            dims.floor
        );
    }

    // Lit sheet: albedo + emissive, on a black background at the same scale.
    let lit = out_dir.join("facade_sheet.png");
    write_sheet(&lit, true)?;
    let plain = out_dir.join("facade_sheet_albedo.png");
    write_sheet(&plain, false)?;
    println!("sheets: {} / {}", lit.display(), plain.display());
    Ok(())
}

/// A grid of every (style, variant), captioned by style.
fn write_sheet(
    path: &std::path::Path,
    with_emissive: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let columns = VARIANT_COUNT as u32;
    let rows = FacadeStyle::ALL.len() as u32;

    let cell_w = (FacadeStyle::OldCity.dims().tile().x * SHEET_PX_PER_M).round() as u32;
    let cell_h = (FacadeStyle::OldCity.dims().tile().y * SHEET_PX_PER_M).round() as u32;
    // Every cell is rendered at the SAME px/m, so a factory tile (24 m) is twice
    // the width of an old-city tile (8 m). Cells are laid out by the widest.
    let widest = FacadeStyle::ALL
        .iter()
        .map(|s| (s.dims().tile().x * SHEET_PX_PER_M).round() as u32)
        .max()
        .unwrap_or(cell_w);
    let tallest = FacadeStyle::ALL
        .iter()
        .map(|s| (s.dims().tile().y * SHEET_PX_PER_M).round() as u32)
        .max()
        .unwrap_or(cell_h);

    let width = MARGIN * 2 + columns * (widest + GAP) - GAP;
    let height = MARGIN * 2 + rows * (CAPTION_H + tallest + GAP) - GAP;
    let mut canvas = vec![0u8; (width * height * 4) as usize];
    for px in canvas.chunks_mut(4) {
        px[..3].copy_from_slice(&BG);
        px[3] = 255;
    }

    for (row, style) in FacadeStyle::ALL.iter().enumerate() {
        for variant in 0..VARIANT_COUNT {
            let dims = style.dims();
            let tile = dims.tile();
            let w = (tile.x * SHEET_PX_PER_M).round() as u32;
            let h = (tile.y * SHEET_PX_PER_M).round() as u32;
            let x0 = MARGIN + variant as u32 * (widest + GAP);
            let y0 = MARGIN + row as u32 * (CAPTION_H + tallest + GAP) + CAPTION_H;

            for y in 0..h {
                for x in 0..w {
                    let u = (x as f32 + 0.5) / w as f32;
                    let v = (y as f32 + 0.5) / h as f32;
                    let s = sample(*style, variant, u, v);
                    let mut rgb = s.albedo;
                    if with_emissive {
                        // The material's emissive luminance is 3.0 (see
                        // `plugins::facade`); clamp here rather than blow out
                        // the sheet so the panes stay readable.
                        for (channel, glow) in rgb.iter_mut().zip(s.emissive) {
                            *channel = (*channel + glow * WINDOW_EMISSIVE).min(1.0);
                        }
                    }
                    // Linear-ish preview: the game's output is ACES-tonemapped,
                    // but a gamma lift is enough to judge rhythm and value.
                    let px = [to_srgb8(rgb[0]), to_srgb8(rgb[1]), to_srgb8(rgb[2]), 255];
                    put(&mut canvas, width, x0 + x, y0 + y, px);
                }
            }
            println!(
                "  sheet cell {} v{variant}: {w}x{h} px at {SHEET_PX_PER_M:.0} px/m",
                style.label()
            );
        }
    }

    write_png_rgba(path, width, height, &canvas)
}

fn to_srgb8(linear: f32) -> u8 {
    let v = linear.clamp(0.0, 1.0);
    // sRGB transfer, so the sheet is not judged in linear space (which would
    // make every wall look far darker than the game does).
    let s = if v <= 0.003_130_8 {
        12.92 * v
    } else {
        1.055 * v.powf(1.0 / 2.4) - 0.055
    };
    (s * 255.0).round() as u8
}

fn put(canvas: &mut [u8], width: u32, x: u32, y: u32, px: [u8; 4]) {
    let i = ((y * width + x) * 4) as usize;
    if i + 4 <= canvas.len() {
        canvas[i..i + 4].copy_from_slice(&px);
    }
}

// --- minimal RGBA PNG encoder (stored deflate) ----------------------------
// Same approach as `tree_gallery` and `leaf_atlas`: no image crate.

fn write_png_rgba(
    path: &std::path::Path,
    width: u32,
    height: u32,
    rgba: &[u8],
) -> Result<(), Box<dyn std::error::Error>> {
    assert_eq!(
        rgba.len(),
        (width * height * 4) as usize,
        "rgba size mismatch"
    );
    let mut raw = Vec::with_capacity(rgba.len() + height as usize);
    for y in 0..height as usize {
        raw.push(0u8);
        let row = y * width as usize * 4;
        raw.extend_from_slice(&rgba[row..row + width as usize * 4]);
    }
    let compressed = zlib_stored(&raw);

    let mut out = Vec::new();
    out.extend_from_slice(&[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]);
    let mut ihdr = Vec::new();
    ihdr.extend_from_slice(&width.to_be_bytes());
    ihdr.extend_from_slice(&height.to_be_bytes());
    ihdr.extend_from_slice(&[8, 6, 0, 0, 0]);
    write_chunk(&mut out, b"IHDR", &ihdr);
    write_chunk(&mut out, b"IDAT", &compressed);
    write_chunk(&mut out, b"IEND", &[]);
    std::fs::write(path, out)?;
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
