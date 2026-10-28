//! The lightning flash, shared by the lightning and cloud domains.

use bevy::prelude::*;

/// The current lightning flash: written each frame by the lightning plugin, read by the cloud
/// march to light the clouds from inside. `intensity_cd` 0 means no flash.
#[derive(Resource, Clone, Copy, Debug, Default, PartialEq, Reflect)]
pub struct LightningFlash {
    /// The top of the channel at the cloud base, world space, m.
    pub position: Vec3,
    /// Luminous intensity, cd.
    pub intensity_cd: f32,
}
