//! The wave model's CPU side: the dispersion kernels fitted at start-up, the bodies' upload and
//! their pressure head. Every number is `docs/research/waves.md`; `waves.wesl` evaluates the same
//! formulas and the tests here pin them (Validations 1 and 2).

use bevy::prelude::*;
use bevy::render::render_resource::ShaderType;

/// Gravity, m/s².
pub(crate) const G: f64 = 9.81;
/// Surface tension over density of clean water at 20 °C, m³/s² (Canabal et al. 2016, eq. 2).
pub(crate) const SIGMA_RHO: f64 = 7.28e-5;
/// The kernel's half width, texels (Tessendorf 2004: `P = 6`).
pub(crate) const RADIUS: i32 = 6;
/// Distinct weights of a radially symmetric `(2 RADIUS + 1)²` kernel: `(m, n)` with
/// `0 ≤ n ≤ m ≤ RADIUS`.
pub(crate) const TAPS: usize = 28;
/// The fixed substep, s (Crest's 60 updates a second).
pub(crate) const SUBSTEP_S: f32 = 1.0 / 60.0;
/// Most substeps a frame.
pub(crate) const MAX_SUBSTEPS: u32 = 4;
/// Most bodies uploaded a frame (the nearest to the camera).
pub(crate) const MAX_CONTACTS: usize = 32;
/// Cascades, each `CASCADE_RATIO` times coarser than the last (Ottosson's pyramid, Crest's
/// cascades): a 13 × 13 kernel holds Airy's dispersion to within 2 % only for waves of 3 to 12
/// texels, so the bands are a factor of four apart.
pub(crate) const CASCADES: usize = 3;
/// A cascade's texel over the next finer one's.
pub(crate) const CASCADE_RATIO: u32 = 4;
/// The shortest and longest wavelength a cascade's kernel is fitted for, texels.
pub(crate) const BAND_TEXELS: (f64, f64) = (2.5, 16.0);

/// The texel of cascade `cascade` when the finest is `base_m`, m.
pub(crate) fn cascade_texel(base_m: f32, cascade: usize) -> f32 {
    base_m * (CASCADE_RATIO as f32).powi(cascade as i32)
}

/// Deep-water `ω²` of capillary-gravity waves of wavenumber `k` (rad/m), s⁻² (Canabal eq. 6 with
/// `tanh(k d) = 1`).
pub(crate) fn omega2(k: f64) -> f64 {
    G * k + SIGMA_RHO * k * k * k
}

/// The band-centre wavenumber of a cascade of texel `texel_m`: a wave of eight texels, rad/m.
pub(crate) fn band_centre(texel_m: f32) -> f32 {
    std::f32::consts::TAU / (8.0 * texel_m)
}

/// The largest impact head a body's entry raises, m (a fall of a few metres).
pub(crate) const MAX_ENTRY_HEAD_M: f32 = 2.0;

/// The impact head of a body moving down through the surface this frame: Bernoulli's stagnation
/// head `v²/2g` of its descent while its vertical extent (centre ± `half_height`) swept across
/// the surface at `water_y`, m; 0 for a rising or still body, or one clear of the surface (the
/// head planes keep the largest non-negative head, so a rising body's suction is not modelled).
pub(crate) fn entry_head(prev_y: f32, now_y: f32, half_height: f32, water_y: f32, dt: f32) -> f32 {
    if dt <= 0.0 {
        return 0.0;
    }
    let v = (now_y - prev_y) / dt;
    let lo = prev_y.min(now_y) - half_height;
    let hi = prev_y.max(now_y) + half_height;
    if v >= 0.0 || !(lo..=hi).contains(&water_y) {
        return 0.0;
    }
    (v * v / (2.0 * G as f32)).min(MAX_ENTRY_HEAD_M)
}

/// The `(m, n)` offsets of the kernel's distinct weights, `0 ≤ n ≤ m ≤ RADIUS`, in the order the
/// shader reads them.
pub(crate) fn taps() -> [(i32, i32); TAPS] {
    let mut out = [(0, 0); TAPS];
    let mut i = 0;
    for m in 0..=RADIUS {
        for n in 0..=m {
            out[i] = (m, n);
            i += 1;
        }
    }
    out
}

/// The distinct images of offset `(m, n)` under the kernel's symmetries (sign of each axis and the
/// swap of the axes).
fn images(m: i32, n: i32) -> Vec<(i32, i32)> {
    let mut out: Vec<(i32, i32)> = Vec::with_capacity(8);
    for (a, b) in [(m, n), (n, m)] {
        for (sa, sb) in [(1, 1), (1, -1), (-1, 1), (-1, -1)] {
            let p = (a * sa, b * sb);
            if !out.contains(&p) {
                out.push(p);
            }
        }
    }
    out
}

/// A cascade's kernel: the weights of `(m, n)` in [`taps`] order, applied to every image of the
/// offset. The acceleration is `−Σ K h`; its symbol is `ω²(k)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct WaveKernel {
    pub weights: [f32; TAPS],
}

impl WaveKernel {
    #[cfg(test)]
    /// The kernel's symbol at wave vector `(kx, kz)` (rad/m) for texel `texel_m`, s⁻².
    /// The shader applies it as `Σ w (h(x + offset) − h(x))` over every image of every offset but
    /// the centre, so a uniform rise feels exactly no force whatever the rounding of the weights.
    pub(crate) fn symbol(&self, kx: f64, kz: f64, texel_m: f64) -> f64 {
        taps()
            .iter()
            .zip(self.weights)
            .skip(1)
            .map(|(&(m, n), w)| {
                let imgs = images(m, n);
                f64::from(w)
                    * (imgs
                        .iter()
                        .map(|&(i, j)| ((kx * f64::from(i) + kz * f64::from(j)) * texel_m).cos())
                        .sum::<f64>()
                        - imgs.len() as f64)
            })
            .sum()
    }

    /// The least-squares fit of the symbol to [`omega2`] for a cascade of texel `texel_m`
    /// (`docs/research/waves.md`, "The kernel"): relative error over the band (λ from 2.5 to 16
    /// texels) on the axes, the diagonal and between them. The centre weight is minus the sum of
    /// the others' images (no acceleration from a uniform rise: volume is kept); the shader never
    /// reads it. Outside the band the fit is held near `ω²` with a smaller weight, so the symbol
    /// stays positive and bounded.
    pub(crate) fn fit(texel_m: f32) -> Self {
        let dx = f64::from(texel_m);
        let list = taps();
        // Unknowns: every weight but the centre, which is minus the sum of the others' images.
        let n = TAPS - 1;
        let basis = |kx: f64, kz: f64| -> Vec<f64> {
            list[1..]
                .iter()
                .map(|&(m, nn)| {
                    let imgs = images(m, nn);
                    imgs.iter()
                        .map(|&(i, j)| ((kx * f64::from(i) + kz * f64::from(j)) * dx).cos())
                        .sum::<f64>()
                        - imgs.len() as f64
                })
                .collect()
        };
        let mut ata = vec![0.0f64; n * n];
        let mut atb = vec![0.0f64; n];
        let k_nyquist = std::f64::consts::PI / dx;
        let k_lo = std::f64::consts::TAU / (BAND_TEXELS.1 * dx);
        let k_hi = std::f64::consts::TAU / (BAND_TEXELS.0 * dx);
        let radial = 160;
        let angles = 12;
        for r in 1..=radial {
            // Out to the corner of the Brillouin zone.
            let k = std::f64::consts::SQRT_2 * k_nyquist * f64::from(r) / f64::from(radial);
            let in_band = (k_lo..=k_hi).contains(&k);
            for a in 0..=angles {
                let theta = std::f64::consts::FRAC_PI_4 * f64::from(a) / f64::from(angles);
                let (kx, kz) = (k * theta.cos(), k * theta.sin());
                if kx > k_nyquist || kz > k_nyquist {
                    continue;
                }
                let target = omega2(k);
                let weight = if in_band { 1.0 } else { 0.05 } / target;
                let row = basis(kx, kz);
                for i in 0..n {
                    atb[i] += weight * weight * row[i] * target;
                    for j in 0..n {
                        ata[i * n + j] += weight * weight * row[i] * row[j];
                    }
                }
            }
        }
        // A whisper of ridge keeps the system well conditioned.
        let trace = (0..n).map(|i| ata[i * n + i]).sum::<f64>() / n as f64;
        for i in 0..n {
            ata[i * n + i] += 1e-9 * trace;
        }
        let x = solve(ata, atb, n);
        let mut weights = [0.0f32; TAPS];
        let mut centre = 0.0f64;
        for (i, &(m, nn)) in list[1..].iter().enumerate() {
            weights[i + 1] = x[i] as f32;
            centre -= x[i] * images(m, nn).len() as f64;
        }
        weights[0] = centre as f32;
        Self { weights }
    }

    #[cfg(test)]
    /// The largest value of the symbol over the Brillouin zone, s⁻² (the stiffest mode).
    pub(crate) fn max_symbol(&self, texel_m: f32) -> f64 {
        let dx = f64::from(texel_m);
        let k_nyquist = std::f64::consts::PI / dx;
        let mut most = 0.0f64;
        let steps = 48;
        for i in 0..=steps {
            for j in 0..=i {
                let kx = k_nyquist * f64::from(i) / f64::from(steps);
                let kz = k_nyquist * f64::from(j) / f64::from(steps);
                most = most.max(self.symbol(kx, kz, dx));
            }
        }
        most
    }

    /// The weights as seven `vec4`s, the uniform's layout.
    pub(crate) fn packed(&self) -> [Vec4; 7] {
        let w = self.weights;
        std::array::from_fn(|i| Vec4::new(w[4 * i], w[4 * i + 1], w[4 * i + 2], w[4 * i + 3]))
    }
}

/// Gaussian elimination with partial pivoting of the `n × n` row-major system `a x = b`.
fn solve(mut a: Vec<f64>, mut b: Vec<f64>, n: usize) -> Vec<f64> {
    for col in 0..n {
        let pivot = (col..n)
            .max_by(|&i, &j| a[i * n + col].abs().total_cmp(&a[j * n + col].abs()))
            .unwrap_or(col);
        if pivot != col {
            for k in 0..n {
                a.swap(col * n + k, pivot * n + k);
            }
            b.swap(col, pivot);
        }
        let d = a[col * n + col];
        if d.abs() < f64::MIN_POSITIVE {
            continue;
        }
        for row in col + 1..n {
            let f = a[row * n + col] / d;
            if f == 0.0 {
                continue;
            }
            for k in col..n {
                a[row * n + k] -= f * a[col * n + k];
            }
            b[row] -= f * b[col];
        }
    }
    let mut x = vec![0.0f64; n];
    for row in (0..n).rev() {
        let mut s = b[row];
        for k in row + 1..n {
            s -= a[row * n + k] * x[k];
        }
        let d = a[row * n + row];
        x[row] = if d.abs() < f64::MIN_POSITIVE {
            0.0
        } else {
            s / d
        };
    }
    x
}

/// One body as the head kernel reads it. Mirrors `ContactGpu` of `waves.wesl`.
#[derive(Clone, Copy, Debug, Default, PartialEq, ShaderType, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub(crate) struct ContactGpu {
    /// xyz: the ellipsoid's centre (world, m). w: the radius of its XZ bounding circle, m.
    pub centre: Vec4,
    /// The ellipsoid's matrix `M = R diag(a⁻², b⁻², c⁻²) Rᵀ`: xx, xy, xz, yy.
    pub m0: Vec4,
    /// yz, zz; z: the impact head of its entry this frame, m ([`entry_head`]); w unused.
    pub m1: Vec4,
}

impl ContactGpu {
    /// The upload for an ellipsoid of `half_extents` (m) along the local axes of `rotation`,
    /// centred at `centre`. `None` for a degenerate one.
    pub(crate) fn new(centre: Vec3, half_extents: Vec3, rotation: Quat) -> Option<Self> {
        if !(half_extents.min_element() > 0.0 && centre.is_finite() && rotation.is_finite()) {
            return None;
        }
        let r = Mat3::from_quat(rotation);
        let d = Mat3::from_diagonal(half_extents.recip() * half_extents.recip());
        let m = r * d * r.transpose();
        Some(Self {
            centre: centre.extend(half_extents.max_element()),
            m0: Vec4::new(m.x_axis.x, m.y_axis.x, m.z_axis.x, m.y_axis.y),
            m1: Vec4::new(m.z_axis.y, m.z_axis.z, 0.0, 0.0),
        })
    }

    /// The ellipsoid's vertical half extent (centre to top), m: `√((R diag(a², b², c²) Rᵀ)_yy)`.
    pub(crate) fn half_height(half_extents: Vec3, rotation: Quat) -> f32 {
        let up = rotation.inverse() * Vec3::Y;
        (up * half_extents).length()
    }

    #[cfg(test)]
    /// The vertical chord through the ellipsoid at world XZ `xz`: `(y_bottom, y_top)`, or `None`
    /// where the vertical line misses it.
    pub(crate) fn chord(&self, xz: Vec2) -> Option<(f32, f32)> {
        let dx = xz.x - self.centre.x;
        let dz = xz.y - self.centre.z;
        let (xx, xy, xz_, yy) = (self.m0.x, self.m0.y, self.m0.z, self.m0.w);
        let (yz, zz) = (self.m1.x, self.m1.y);
        let a = yy;
        let b = xy * dx + yz * dz;
        let c = xx * dx * dx + 2.0 * xz_ * dx * dz + zz * dz * dz - 1.0;
        let disc = b * b - a * c;
        if disc <= 0.0 || a <= 0.0 {
            return None;
        }
        let root = disc.sqrt();
        Some((
            self.centre.y + (-b - root) / a,
            self.centre.y + (-b + root) / a,
        ))
    }
}

#[cfg(test)]
/// The pressure head of a body's vertical chord `(y_bottom, y_top)` under a surface at `y_w`,
/// as felt by waves of wavenumber `k` (rad/m): the occupied column weighted by `e^{−k z}` at
/// depth `z`, m (`docs/research/waves.md`, "Pressure head of a body").
pub(crate) fn head(y_bottom: f32, y_top: f32, y_w: f32, k: f32) -> f32 {
    let top = (y_w - y_top).max(0.0);
    let bottom = (y_w - y_bottom).max(0.0);
    ((-k * top).exp() - (-k * bottom).exp()) / k
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Airy's deep-water phase speed of capillary-gravity waves, m/s.
    fn airy_speed(k: f64) -> f64 {
        omega2(k).sqrt() / k
    }

    #[test]
    fn the_kernels_disperse_like_water() {
        // Validation 1: phase speed within 3 % of Airy's for λ from 3 to 12 texels, on the axes,
        // the diagonal and between, for every cascade the qualities use; stable at the substep.
        for base in [0.04f32, 0.05, 0.06, 0.08] {
            for cascade in 0..CASCADES {
                let texel = cascade_texel(base, cascade);
                let kernel = WaveKernel::fit(texel);
                let dx = f64::from(texel);
                let mut worst = 0.0f64;
                for lambda_texels in [3.0, 4.0, 6.0, 8.0, 12.0] {
                    let k = std::f64::consts::TAU / (lambda_texels * dx);
                    for theta in [
                        0.0f64,
                        std::f64::consts::FRAC_PI_8,
                        std::f64::consts::FRAC_PI_4,
                    ] {
                        let s = kernel.symbol(k * theta.cos(), k * theta.sin(), dx);
                        assert!(
                            s > 0.0,
                            "texel {texel}: negative symbol at λ = {lambda_texels}"
                        );
                        let c = s.sqrt() / k;
                        worst = worst.max((c / airy_speed(k) - 1.0).abs());
                    }
                }
                assert!(worst < 0.03, "texel {texel}: phase speed off by {worst:.4}");
                check_stable(&kernel, texel);
            }
        }
    }

    fn check_stable(kernel: &WaveKernel, texel: f32) {
        let dx = f64::from(texel);
        {
            // Leapfrog is stable while ω_max Δt < 2.
            let stiff = kernel.max_symbol(texel).sqrt() * f64::from(SUBSTEP_S);
            assert!(stiff < 2.0, "texel {texel}: ω_max Δt = {stiff:.3}");
            // The symbol is non-negative everywhere (a negative mode would grow).
            let k_nyquist = std::f64::consts::PI / dx;
            for i in 0..=40 {
                for j in 0..=i {
                    let kx = k_nyquist * f64::from(i) / 40.0;
                    let kz = k_nyquist * f64::from(j) / 40.0;
                    assert!(
                        kernel.symbol(kx, kz, dx) > -1e-6,
                        "texel {texel}: negative symbol at ({kx:.2}, {kz:.2})"
                    );
                }
            }
        }
    }

    #[test]
    fn the_head_of_a_floating_sphere() {
        // Validation 2.
        let k = band_centre(0.05);
        let r = 0.5;
        let c = ContactGpu::new(Vec3::new(1.0, 2.0, -3.0), Vec3::splat(r), Quat::IDENTITY).unwrap();
        let (yb, yt) = c.chord(Vec2::new(1.0, -3.0)).unwrap();
        assert!((yb - 1.5).abs() < 1e-5 && (yt - 2.5).abs() < 1e-5);
        // Half submerged: the surface at the centre.
        let h = head(yb, yt, 2.0, k);
        let want = (1.0 - (-k * r).exp()) / k;
        assert!((h - want).abs() < 1e-5, "{h} vs {want}");
        // A shallow hull's head is its draft.
        let draft = head(1.99, 2.5, 2.0, k);
        assert!(
            (draft - 0.01).abs() <= 0.5 * k * 0.01 * 0.01 + 1e-5,
            "{draft}"
        );
        // Above the water: nothing.
        assert_eq!(head(2.1, 3.0, 2.0, k), 0.0);
        // Continuous through submersion.
        let just_under = head(1.0, 1.999, 2.0, k);
        let just_over = head(1.0, 2.001, 2.0, k);
        assert!((just_under - just_over).abs() < 1e-3);
        // Off the body: no chord.
        assert!(c.chord(Vec2::new(1.6, -3.0)).is_none());
    }

    #[test]
    fn a_rotated_ellipsoid_chord_matches_a_march() {
        let rotation = Quat::from_euler(EulerRot::YXZ, 0.7, 0.4, -0.3);
        let half = Vec3::new(0.8, 0.3, 0.5);
        let centre = Vec3::new(0.2, 1.0, -0.4);
        let c = ContactGpu::new(centre, half, rotation).unwrap();
        let inverse = rotation.inverse();
        let inside = |p: Vec3| {
            let l = inverse * (p - centre) / half;
            l.length_squared() <= 1.0
        };
        for (x, z) in [(0.2f32, -0.4f32), (0.5, -0.2), (-0.3, -0.6), (0.6, 0.1)] {
            let march: Vec<f32> = (0..=4000)
                .map(|i| -1.0 + 4.0 * i as f32 / 4000.0)
                .filter(|&y| inside(Vec3::new(x, y, z)))
                .collect();
            match c.chord(Vec2::new(x, z)) {
                Some((yb, yt)) => {
                    let (lo, hi) = (march[0], march[march.len() - 1]);
                    assert!((yb - lo).abs() < 2e-3 && (yt - hi).abs() < 2e-3, "{x},{z}");
                }
                None => assert!(march.is_empty(), "{x},{z}: missed a chord"),
            }
        }
    }

    #[test]
    fn a_falling_body_strikes_the_surface_only_as_it_crosses() {
        let dt = 1.0 / 60.0;
        // A 6 cm pebble at 5.4 m/s (a 1.5 m drop) crossing the surface at 2 m: v²/2g.
        let v = 5.4;
        let struck = entry_head(2.05, 2.05 - v * dt, 0.06, 2.0, dt);
        assert!((struck - v * v / (2.0 * 9.81)).abs() < 1e-3, "{struck}");
        // High above the surface, rising, or still: nothing.
        assert_eq!(entry_head(3.0, 3.0 - v * dt, 0.06, 2.0, dt), 0.0);
        assert_eq!(entry_head(2.0, 2.0 + v * dt, 0.06, 2.0, dt), 0.0);
        assert_eq!(entry_head(2.0, 2.0, 0.06, 2.0, dt), 0.0);
        // A long fall is capped.
        assert_eq!(entry_head(2.5, 1.5, 0.06, 2.0, dt), MAX_ENTRY_HEAD_M);
        // A tilted ellipsoid's half height.
        let tilted = ContactGpu::half_height(Vec3::new(1.0, 0.2, 0.5), Quat::from_rotation_z(0.6));
        let up = Quat::from_rotation_z(-0.6) * Vec3::Y;
        assert!((tilted - (up * Vec3::new(1.0, 0.2, 0.5)).length()).abs() < 1e-6);
    }
}
