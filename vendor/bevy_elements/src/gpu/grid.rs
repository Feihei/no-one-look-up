//! Creating the storage buffers and storage textures the simulations live in.

use bevy::asset::RenderAssetUsages;
use bevy::image::{ImageAddressMode, ImageFilterMode, ImageSampler, ImageSamplerDescriptor};
use bevy::prelude::*;
use bevy::render::render_resource::{
    BufferUsages, Extent3d, TextureDimension, TextureFormat, TextureUsages,
};
use bevy::render::storage::ShaderBuffer;

/// A GPU-only storage buffer holding `data`. The CPU copy is dropped after upload; the buffer can
/// be read back (`COPY_SRC`) and bound by materials as `#[storage(N, read_only)]`.
pub(crate) fn storage_buffer<T: bytemuck::NoUninit>(
    buffers: &mut Assets<ShaderBuffer>,
    label: &'static str,
    data: Vec<T>,
) -> Handle<ShaderBuffer> {
    let mut buffer = ShaderBuffer::new(data, RenderAssetUsages::RENDER_WORLD);
    buffer.label = label.into();
    buffer.buffer_usage = BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST;
    buffers.add(buffer)
}

/// A GPU-only storage buffer of `bytes` zero bytes, created on the GPU with no CPU copy (large
/// simulation layers would otherwise need a zeroed `Vec` as big as they are). Same usages as
/// [`storage_buffer`]; the asset stays in the main world (an uninitialised buffer holds only its
/// size there), so a caller can replace its contents later and the GPU copy is rewritten.
pub(crate) fn zeroed_buffer(
    buffers: &mut Assets<ShaderBuffer>,
    label: &'static str,
    bytes: u64,
) -> Handle<ShaderBuffer> {
    let mut buffer = ShaderBuffer::with_size(bytes, RenderAssetUsages::default());
    buffer.label = label.into();
    buffer.buffer_usage = BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST;
    buffers.add(buffer)
}

/// A 3D `rgba16float` texture written by a compute kernel (write-only storage) and sampled by
/// shaders with a linear, repeating sampler. Starts zeroed.
pub(crate) fn storage_volume(images: &mut Assets<Image>, size: UVec3) -> Handle<Image> {
    let mut image = Image::new_fill(
        Extent3d {
            width: size.x,
            height: size.y,
            depth_or_array_layers: size.z,
        },
        TextureDimension::D3,
        &[0u8; 8],
        TextureFormat::Rgba16Float,
        RenderAssetUsages::RENDER_WORLD,
    );
    image.texture_descriptor.usage =
        TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST;
    image.sampler = ImageSampler::Descriptor(ImageSamplerDescriptor {
        address_mode_u: ImageAddressMode::Repeat,
        address_mode_v: ImageAddressMode::Repeat,
        address_mode_w: ImageAddressMode::Repeat,
        mag_filter: ImageFilterMode::Linear,
        min_filter: ImageFilterMode::Linear,
        mipmap_filter: ImageFilterMode::Linear,
        ..default()
    });
    images.add(image)
}
