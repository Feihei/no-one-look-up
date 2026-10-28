//! Crate-private GPU plumbing shared by every domain: kernels, storage resources and readback.

pub(crate) mod grid;
pub(crate) mod kernel;
#[cfg(test)]
pub(crate) mod noise_mirror;
pub(crate) mod readback;
