//! The trail model's CPU side: the window's placement and the imprint's upload, and (in tests)
//! the mirror of the kernels' sinkage, rim and slump rules. Every number is
//! `docs/research/trails.md`; `trails.wesl` evaluates the same formulas and the tests here pin
//! them (Validation 1).

use bevy::prelude::*;
use bevy::render::render_resource::ShaderType;

/// The ellipse's area over its perimeter (Ramanujan's perimeter), m: the carved volume per unit
/// length of edge for a uniform carve.
pub(crate) fn area_per_perimeter(a: f32, b: f32) -> f32 {
    let h = ((a - b) / (a + b)).powi(2);
    let perimeter =
        std::f32::consts::PI * (a + b) * (1.0 + 3.0 * h / (10.0 + (4.0 - 3.0 * h).sqrt()));
    std::f32::consts::PI * a * b / perimeter
}

/// Width of the rim's ring, m: two texels or 30 % of the ellipse's smaller half extent.
pub(crate) fn ring_width(a: f32, b: f32, texel_m: f32) -> f32 {
    (2.0 * texel_m).max(0.3 * a.min(b))
}

/// The world texel of the window's minimum corner for a camera at world XZ `camera`: the
/// window is `size` texels of `texel_m`, centred on the camera's texel.
pub(crate) fn window_origin(camera: Vec2, texel_m: f32, size: u32) -> IVec2 {
    (camera / texel_m).floor().as_ivec2() - IVec2::splat((size / 2) as i32)
}

/// Most imprints uploaded a frame (the nearest to the camera).
pub(crate) const MAX_IMPRINTS: usize = 64;
/// Texels a side of the largest stamp tile one imprint may cover.
pub(crate) const STAMP_TILE: u32 = 128;
/// The carve's wall as a share of the ellipse's normalised radius: two and a half texels, at
/// most half the smaller half extent (`trails.wesl` `trail_stamp`).
pub(crate) fn wall_share(half: Vec2, texel_m: f32) -> f32 {
    let narrow = half.min_element();
    (2.5 * texel_m).min(0.5 * narrow) / narrow
}

/// The mean of the carve's profile over the ellipse, `∫₀¹ (1 − smoothstep(1 − e, 1, r)) 2r dr`
/// = `1 − e + 0.3 e²`: the carved volume is `want · area` times this.
pub(crate) fn mean_carve(e: f32) -> f32 {
    1.0 - e + 0.3 * e * e
}

/// The longest path an imprint carves in one frame, m (60 m/s at 60 fps); a longer jump is a
/// teleport and starts a new path.
pub(crate) const MAX_SWEEP_M: f32 = 1.0;
/// The bound the parallax march assumes on a trench's depth, m: the loose layer's depth plus a
/// margin.
pub(crate) const MARCH_BOUND_M: f32 = 0.35;
/// The deepest any imprint presses, m: the loose layer (Alvarado et al. 2022, `L0`).
pub(crate) const MAX_SINKAGE_M: f32 = 0.30;

/// One imprint as the stamp kernel reads it. Mirrors `ImprintGpu` of `trails.wesl`.
#[derive(Clone, Copy, Debug, Default, PartialEq, ShaderType, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub(crate) struct ImprintGpu {
    /// The swept segment: world XZ where it was last carved (xy) and where it is now (zw).
    pub segment: Vec4,
    /// Half extents along its local x and z (m), and its yaw as (cos, sin) of the local x axis
    /// in world XZ.
    pub shape: Vec4,
    /// x: the sole's world height, m. y: pressure, kPa. z: plate width, m. w: the carved volume
    /// per unit length of edge per metre of depth (area over perimeter times the profile's mean), m.
    pub load: Vec4,
    /// x: rim ring width, m. yz: the world texel of the stamp tile's minimum corner (as f32,
    /// exact below 2²⁴). w: the tile's edge, texels.
    pub tile: Vec4,
}

impl ImprintGpu {
    /// The upload for an ellipse of half extents `half` (m) and yaw `yaw` (the world angle of its
    /// local x axis), loaded with `load_n` newtons, swept from `from` to `to` (world XZ) with its
    /// sole at `sole_y`. `None` for a degenerate imprint or one whose sweep does not fit a tile.
    /// `load.w` is the carved volume per unit length of edge per metre of depth: the area over
    /// perimeter times the profile's mean.
    pub(crate) fn new(
        half: Vec2,
        yaw: f32,
        load_n: f32,
        from: Vec2,
        to: Vec2,
        sole_y: f32,
        texel_m: f32,
    ) -> Option<Self> {
        if !(half.x > 0.0 && half.y > 0.0 && load_n > 0.0 && sole_y.is_finite()) {
            return None;
        }
        let ring = ring_width(half.x, half.y, texel_m);
        // The rim's ring widens to the carve's depth, at most the loose layer (`trails.wesl`).
        let reach = half.max_element() + ring.max(MAX_SINKAGE_M) + 2.0 * texel_m;
        let min = from.min(to) - Vec2::splat(reach);
        let max = from.max(to) + Vec2::splat(reach);
        let tile_min = (min / texel_m).floor().as_ivec2();
        let tile_max = (max / texel_m).ceil().as_ivec2();
        let edge = (tile_max - tile_min).max_element();
        if edge > STAMP_TILE as i32 {
            return None;
        }
        let pressure_kpa = load_n / (std::f32::consts::PI * half.x * half.y) / 1000.0;
        Some(Self {
            segment: Vec4::new(from.x, from.y, to.x, to.y),
            shape: Vec4::new(half.x, half.y, yaw.cos(), yaw.sin()),
            load: Vec4::new(
                sole_y,
                pressure_kpa,
                2.0 * half.min_element(),
                area_per_perimeter(half.x, half.y) * mean_carve(wall_share(half, texel_m)),
            ),
            tile: Vec4::new(ring, tile_min.x as f32, tile_min.y as f32, edge as f32),
        })
    }
}

#[cfg(test)]
pub(crate) mod mirror {
    //! CPU mirrors of `trails.wesl`'s ground rules, in the same f32 arithmetic.

    /// Bekker's pressure–sinkage constants `(n, k_c, k_φ)`: `p = (k_c/b + k_φ) z^n` with `p` in
    /// kPa, the plate width `b` and the sinkage `z` in m, `k_c` in kN/m^(n+1), `k_φ` in
    /// kN/m^(n+2).
    #[derive(Clone, Copy, Debug, PartialEq)]
    pub(crate) struct Bekker {
        pub n: f32,
        pub k_c: f32,
        pub k_phi: f32,
    }

    impl Bekker {
        /// Sinkage (m) of a plate of width `b` (m) under the pressure `p` (kPa).
        pub(crate) fn sinkage(self, p_kpa: f32, b: f32) -> f32 {
            let k = self.k_c / b.max(1e-3) + self.k_phi;
            (p_kpa.max(0.0) / k).powf(1.0 / self.n)
        }
    }

    /// Wong's snow (US) (NASA TM-20250006958, Table 1).
    pub(crate) const SNOW: Bekker = Bekker {
        n: 1.6,
        k_c: 4.37,
        k_phi: 196.72,
    };
    /// Wong's dry sand.
    pub(crate) const DRY_SAND: Bekker = Bekker {
        n: 1.1,
        k_c: 0.99,
        k_phi: 1528.43,
    };
    /// Wong's sandy loam at 15 % moisture: dry soil.
    pub(crate) const DRY_SOIL: Bekker = Bekker {
        n: 0.7,
        k_c: 5.27,
        k_phi: 1515.04,
    };
    /// Wong's sandy loam at 32 % moisture: mud.
    pub(crate) const MUD: Bekker = Bekker {
        n: 0.5,
        k_c: 0.77,
        k_phi: 51.91,
    };

    /// Snow compacted under a load below 210 kPa, kg/m³ (CRREL Report 90-9, Eq. A7).
    pub(crate) const COMPACTED_SNOW_DENSITY: f32 = 500.0;
    /// The loose layer a load can press into, m (Alvarado et al. 2022, `L0`).
    pub(crate) const LOOSE_LAYER_M: f32 = 0.30;
    /// Snow thinner than this is not snow ground, m.
    pub(crate) const SNOW_GROUND_M: f32 = 0.01;

    /// Snow depth a load of `p` kPa on a plate `b` m wide presses into snow `depth` m deep of
    /// density `density` kg/m³: Bekker's deep-snow sinkage capped by the CRREL compaction limit.
    pub(crate) fn snow_sinkage(p_kpa: f32, b: f32, depth: f32, density: f32) -> f32 {
        let compaction = depth * (1.0 - density / COMPACTED_SNOW_DENSITY);
        SNOW.sinkage(p_kpa, b)
            .min(compaction.max(0.0))
            .min(LOOSE_LAYER_M)
    }

    /// Soil sinkage from dry soil to mud by the wetness `w` (0–1): log-linear between Wong's
    /// sandy loam at 15 % and 32 % moisture, capped at the loose layer.
    pub(crate) fn soil_sinkage(p_kpa: f32, b: f32, wetness: f32) -> f32 {
        let dry = DRY_SOIL.sinkage(p_kpa, b).clamp(1e-6, LOOSE_LAYER_M);
        let wet = MUD.sinkage(p_kpa, b).clamp(1e-6, LOOSE_LAYER_M);
        (dry.ln() + (wet.ln() - dry.ln()) * wetness.clamp(0.0, 1.0)).exp()
    }

    fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
        let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
        t * t * (3.0 - 2.0 * t)
    }

    /// The ground under a texel, from the maps the crate keeps.
    #[derive(Clone, Copy, Debug, Default, PartialEq)]
    pub(crate) struct Ground {
        /// Snow depth (m) and its density (kg/m³).
        pub snow_m: f32,
        pub snow_density: f32,
        /// The terrain's sediment depth, m.
        pub sediment_m: f32,
        /// Surface slope, degrees.
        pub slope_deg: f32,
        /// The wetness map, 0–1.
        pub wetness: f32,
    }

    impl Ground {
        /// How much of the texel is rock (the terrain material's rule, no noise).
        pub(crate) fn rock(self) -> f32 {
            smoothstep(35.0, 45.0, self.slope_deg)
        }

        /// How much of the texel is sand (the terrain material's rule, no noise).
        pub(crate) fn sand(self) -> f32 {
            smoothstep(0.3, 0.7, self.sediment_m)
                * (1.0 - smoothstep(26.0, 30.0, self.slope_deg))
                * (1.0 - self.rock())
        }

        /// Target depth (m) under `p` kPa on a plate `b` m wide.
        pub(crate) fn sinkage(self, p_kpa: f32, b: f32) -> f32 {
            if self.snow_m > SNOW_GROUND_M {
                return snow_sinkage(p_kpa, b, self.snow_m, self.snow_density);
            }
            let sand = self.sand();
            let z = sand * DRY_SAND.sinkage(p_kpa, b).min(LOOSE_LAYER_M)
                + (1.0 - sand) * soil_sinkage(p_kpa, b, self.wetness);
            z * (1.0 - self.rock())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::mirror::*;
    use super::*;

    /// A 77.5 kg walker on one foot, an ellipse 0.05 × 0.13 m.
    fn walker() -> (f32, f32) {
        let area = std::f32::consts::PI * 0.05 * 0.13;
        (77.5 * 9.81 / area / 1000.0, 0.1)
    }

    fn close(actual: f32, expected: f32, tolerance: f32) -> bool {
        (actual - expected).abs() <= tolerance * expected
    }

    /// Validation 1: the walker's depth in every material, from Wong's constants and the CRREL
    /// cap, against the note's hand-computed values.
    #[test]
    fn a_walker_sinks_as_the_note_computes() {
        let (p, b) = walker();
        assert!(close(p, 37.2, 0.01), "pressure {p}");
        let sand = Ground {
            sediment_m: 2.0,
            ..default()
        };
        assert!(
            close(sand.sinkage(p, b), 0.0339, 0.01),
            "sand {}",
            sand.sinkage(p, b)
        );
        let soil = Ground::default();
        assert!(
            close(soil.sinkage(p, b), 0.0048, 0.02),
            "soil {}",
            soil.sinkage(p, b)
        );
        let mud = Ground {
            wetness: 1.0,
            ..default()
        };
        // Bekker gives 39 cm; the loose layer caps it.
        assert!(close(MUD.sinkage(p, b), 0.390, 0.01));
        assert!(close(mud.sinkage(p, b), LOOSE_LAYER_M, 0.001));
        let fresh = Ground {
            snow_m: 0.3,
            snow_density: 100.0,
            ..default()
        };
        // Bekker 31.2 cm, capped by compaction at 0.3 × (1 − 100/500) = 24 cm.
        assert!(close(SNOW.sinkage(p, b), 0.312, 0.01));
        assert!(close(fresh.sinkage(p, b), 0.240, 0.01));
        let settled = Ground {
            snow_density: 300.0,
            ..fresh
        };
        assert!(close(settled.sinkage(p, b), 0.120, 0.01));
    }

    #[test]
    fn soil_sinks_deeper_the_wetter_it_is() {
        let (p, b) = walker();
        let mut last = 0.0;
        for i in 0..=10 {
            let z = soil_sinkage(p, b, i as f32 / 10.0);
            assert!(z > last, "wetness {}: {z} after {last}", i as f32 / 10.0);
            last = z;
        }
    }

    #[test]
    fn rock_takes_no_imprint() {
        let (p, b) = walker();
        let cliff = Ground {
            slope_deg: 50.0,
            ..default()
        };
        assert_eq!(cliff.sinkage(p, b), 0.0);
    }

    #[test]
    fn the_window_is_centred_on_the_camera_texel() {
        let o = window_origin(Vec2::new(10.02, -3.0), 0.05, 1024);
        assert_eq!(o, IVec2::new(200 - 512, -60 - 512));
        // A camera moving within its texel does not move the window.
        assert_eq!(window_origin(Vec2::new(10.04, -2.99), 0.05, 1024), o);
    }

    #[test]
    fn an_imprint_swept_beyond_a_tile_is_refused() {
        let foot = |to: Vec2| {
            ImprintGpu::new(Vec2::new(0.06, 0.14), 0.0, 760.0, Vec2::ZERO, to, 0.0, 0.05)
        };
        assert!(foot(Vec2::new(0.3, 0.0)).is_some());
        assert!(foot(Vec2::new(8.0, 0.0)).is_none());
    }
}
