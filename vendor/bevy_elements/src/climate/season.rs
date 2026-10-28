//! Meteorological seasons and the 0–1 phenology curves foliage reads.

use bevy::prelude::*;

/// A meteorological season: whole months, December–February is northern winter.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Reflect)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Season {
    /// March–May (northern hemisphere).
    Spring,
    /// June–August (northern hemisphere).
    #[default]
    Summer,
    /// September–November (northern hemisphere).
    Autumn,
    /// December–February (northern hemisphere).
    Winter,
}

impl Season {
    /// The season of a month (1–12), hemisphere-aware.
    pub fn of_month(month: u8, southern: bool) -> Self {
        let m = if southern {
            (month + 5) % 12 + 1
        } else {
            month
        };
        match m {
            3..=5 => Self::Spring,
            6..=8 => Self::Summer,
            9..=11 => Self::Autumn,
            _ => Self::Winter,
        }
    }
}

/// Where the year is, for foliage and anything else that follows the seasons.
///
/// The curves are smoothsteps over northern-hemisphere dates (shifted by 182 days in the south):
/// `autumn` rises 1 Sep → 15 Oct, holds until 30 Nov, then is 0; `leaf_fall` rises 15 Oct →
/// 30 Nov, holds through winter and falls back to 0 over leaf-out, 1 Apr → 10 May; `blossom`
/// rises 1 Apr → 20 Apr and falls 20 Apr → 10 May.
#[derive(Resource, Clone, Copy, Debug, Default, Reflect)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SeasonState {
    /// The meteorological season.
    pub season: Season,
    /// Fraction of the year elapsed since 1 January (hemisphere-independent), 0–1.
    pub year_fraction: f32,
    /// Autumn colouring, 0–1.
    pub autumn: f32,
    /// Fraction of deciduous leaves fallen, 0–1.
    pub leaf_fall: f32,
    /// Blossom, 0–1.
    pub blossom: f32,
}

// Northern-hemisphere days of year (non-leap) of the curve knots.
const SEP_1: f32 = 244.0;
const OCT_15: f32 = 288.0;
const NOV_30: f32 = 334.0;
const APR_1: f32 = 91.0;
const APR_20: f32 = 110.0;
const MAY_10: f32 = 130.0;

fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

impl SeasonState {
    /// The state at a (fractional, 1-based) day of year and month.
    pub fn at(day_of_year: f32, month: u8, southern: bool, days_in_year: f32) -> Self {
        let d = if southern {
            (day_of_year - 182.0).rem_euclid(365.0)
        } else {
            day_of_year
        };
        let autumn = if d > NOV_30 {
            0.0
        } else {
            smoothstep(SEP_1, OCT_15, d)
        };
        let leaf_fall = if d >= OCT_15 {
            smoothstep(OCT_15, NOV_30, d)
        } else if d >= APR_1 {
            1.0 - smoothstep(APR_1, MAY_10, d)
        } else {
            1.0
        };
        let blossom = smoothstep(APR_1, APR_20, d) * (1.0 - smoothstep(APR_20, MAY_10, d));
        Self {
            season: Season::of_month(month, southern),
            year_fraction: ((day_of_year - 1.0) / days_in_year).clamp(0.0, 1.0),
            autumn,
            leaf_fall,
            blossom,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn curves_hit_their_knots() {
        let mid_oct = SeasonState::at(OCT_15, 10, false, 365.0);
        assert_eq!(mid_oct.season, Season::Autumn);
        assert!((mid_oct.autumn - 1.0).abs() < 1e-6);
        assert!(mid_oct.leaf_fall.abs() < 1e-6);
        let winter = SeasonState::at(20.0, 1, false, 365.0);
        assert_eq!(winter.autumn, 0.0);
        assert_eq!(winter.leaf_fall, 1.0);
        let bloom = SeasonState::at(APR_20, 4, false, 365.0);
        assert!((bloom.blossom - 1.0).abs() < 1e-6);
        let summer = SeasonState::at(200.0, 7, false, 365.0);
        assert_eq!(
            (summer.autumn, summer.leaf_fall, summer.blossom),
            (0.0, 0.0, 0.0)
        );
    }

    #[test]
    fn the_south_is_half_a_year_out_of_phase() {
        let north = SeasonState::at(OCT_15, 10, false, 365.0);
        let south = SeasonState::at(OCT_15 - 182.0, 4, true, 365.0);
        assert_eq!(south.season, Season::Autumn);
        assert!((north.autumn - south.autumn).abs() < 1e-6);
    }
}
