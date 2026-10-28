//! Positional astronomy at Meeus accuracy: pure `f64` functions, no ECS.
//!
//! Angles are degrees unless a name says `rad`. `t` is Julian centuries of 36 525 days from
//! J2000.0 in dynamical time (`(JDE − 2451545) / 36525`). Azimuth is measured from north through
//! east. Sources: Meeus, *Astronomical Algorithms* 2nd ed. (ch. 7, 12, 13, 16, 21, 22, 25, 47,
//! 48), Espenak & Meeus ΔT polynomials, Krisciunas & Schaefer 1991; see
//! `docs/research/celestial.md`.

use std::f64::consts::PI;

const D2R: f64 = PI / 180.0;
const R2D: f64 = 180.0 / PI;
/// Astronomical unit, km.
pub const AU_KM: f64 = 149_597_870.7;
/// Earth equatorial radius, km.
pub const EARTH_RADIUS_KM: f64 = 6378.137;
/// Mean lunar radius, km.
pub const MOON_RADIUS_KM: f64 = 1737.4;
/// Full-moon illuminance outside the atmosphere on a surface facing the Moon, lx
/// (Krisciunas & Schaefer 1991 eqs. 8, 9, 20).
pub const FULL_MOON_LUX: f64 = 0.313_31;

fn sin(d: f64) -> f64 {
    (d * D2R).sin()
}
fn cos(d: f64) -> f64 {
    (d * D2R).cos()
}

/// Reduce an angle to [0, 360).
pub fn normalize_deg(d: f64) -> f64 {
    d.rem_euclid(360.0)
}

/// Equatorial coordinates of date.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Equatorial {
    /// Right ascension, degrees in [0, 360).
    pub ra: f64,
    /// Declination, degrees.
    pub dec: f64,
    /// Distance from the Earth's centre, km.
    pub distance_km: f64,
}

/// Julian centuries from J2000.0 for a Julian (Ephemeris) Day.
pub fn centuries(jd: f64) -> f64 {
    (jd - 2_451_545.0) / 36_525.0
}

/// ΔT = TD − UT in seconds for a decimal year (Espenak & Meeus polynomials, 1900–2150; outside
/// that range the long-term parabola, without accuracy claims).
pub fn delta_t_seconds(year: f64) -> f64 {
    let y = year;
    if (1900.0..1920.0).contains(&y) {
        let t = y - 1900.0;
        -2.79 + 1.494_119 * t - 0.059_893_9 * t * t + 0.006_196_6 * t.powi(3)
            - 0.000_197 * t.powi(4)
    } else if (1920.0..1941.0).contains(&y) {
        let t = y - 1920.0;
        21.20 + 0.844_93 * t - 0.076_100 * t * t + 0.002_093_6 * t.powi(3)
    } else if (1941.0..1961.0).contains(&y) {
        let t = y - 1950.0;
        29.07 + 0.407 * t - t * t / 233.0 + t.powi(3) / 2547.0
    } else if (1961.0..1986.0).contains(&y) {
        let t = y - 1975.0;
        45.45 + 1.067 * t - t * t / 260.0 - t.powi(3) / 718.0
    } else if (1986.0..2005.0).contains(&y) {
        let t = y - 2000.0;
        63.86 + 0.3345 * t - 0.060_374 * t * t
            + 0.001_727_5 * t.powi(3)
            + 0.000_651_814 * t.powi(4)
            + 0.000_023_735_99 * t.powi(5)
    } else if (2005.0..2050.0).contains(&y) {
        let t = y - 2000.0;
        62.92 + 0.322_17 * t + 0.005_589 * t * t
    } else if (2050.0..=2150.0).contains(&y) {
        let u = (y - 1820.0) / 100.0;
        -20.0 + 32.0 * u * u - 0.5628 * (2150.0 - y)
    } else {
        let u = (y - 1820.0) / 100.0;
        -20.0 + 32.0 * u * u
    }
}

/// Nutation and obliquity of date (Meeus ch. 22, low accuracy).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Nutation {
    /// Nutation in longitude Δψ, degrees.
    pub dpsi: f64,
    /// Nutation in obliquity Δε, degrees.
    pub deps: f64,
    /// Mean obliquity ε₀, degrees.
    pub eps0: f64,
}

impl Nutation {
    /// True obliquity ε = ε₀ + Δε, degrees.
    pub fn eps(&self) -> f64 {
        self.eps0 + self.deps
    }
}

/// Nutation and obliquity at `t` centuries (Meeus 22.2 low-accuracy terms).
pub fn nutation(t: f64) -> Nutation {
    let omega = 125.044_52 - 1_934.136_261 * t;
    let l = 280.466_5 + 36_000.769_8 * t;
    let lp = 218.316_5 + 481_267.881_3 * t;
    let dpsi = (-17.20 * sin(omega) - 1.32 * sin(2.0 * l) - 0.23 * sin(2.0 * lp)
        + 0.21 * sin(2.0 * omega))
        / 3600.0;
    let deps = (9.20 * cos(omega) + 0.57 * cos(2.0 * l) + 0.10 * cos(2.0 * lp)
        - 0.09 * cos(2.0 * omega))
        / 3600.0;
    let eps0 = 23.0
        + 26.0 / 60.0
        + 21.448 / 3600.0
        + (-46.8150 * t - 0.000_59 * t * t + 0.001_813 * t.powi(3)) / 3600.0;
    Nutation { dpsi, deps, eps0 }
}

/// Mean sidereal time at Greenwich, degrees, for a Julian Day in UT (Meeus 12.4).
pub fn mean_sidereal_deg(jd_ut: f64) -> f64 {
    let d = jd_ut - 2_451_545.0;
    let tu = d / 36_525.0;
    normalize_deg(
        280.460_618_37 + 360.985_647_366_29 * d + 0.000_387_933 * tu * tu
            - tu.powi(3) / 38_710_000.0,
    )
}

/// Local apparent sidereal time, degrees: Greenwich mean sidereal time plus the equation of the
/// equinoxes plus the east longitude.
pub fn local_sidereal_deg(jd_ut: f64, nut: &Nutation, longitude_east: f64) -> f64 {
    normalize_deg(mean_sidereal_deg(jd_ut) + nut.dpsi * cos(nut.eps()) + longitude_east)
}

/// The Sun's apparent position and apparent ecliptic longitude (Meeus ch. 25, low accuracy,
/// about 0.01°).
pub fn sun_apparent(t: f64) -> (Equatorial, f64) {
    let l0 = 280.466_46 + 36_000.769_83 * t + 0.000_303_2 * t * t;
    let m = 357.529_11 + 35_999.050_29 * t - 0.000_153_7 * t * t;
    let e = 0.016_708_634 - 0.000_042_037 * t - 0.000_000_126_7 * t * t;
    let c = (1.914_602 - 0.004_817 * t - 0.000_014 * t * t) * sin(m)
        + (0.019_993 - 0.000_101 * t) * sin(2.0 * m)
        + 0.000_289 * sin(3.0 * m);
    let true_long = l0 + c;
    let nu = m + c;
    let r_au = 1.000_001_018 * (1.0 - e * e) / (1.0 + e * cos(nu));
    let omega = 125.04 - 1934.136 * t;
    let lambda = true_long - 0.005_69 - 0.004_78 * sin(omega);
    let eps = nutation(t).eps0 + 0.002_56 * cos(omega);
    let ra = normalize_deg((cos(eps) * sin(lambda)).atan2(cos(lambda)) * R2D);
    let dec = (sin(eps) * sin(lambda)).asin() * R2D;
    (
        Equatorial {
            ra,
            dec,
            distance_km: r_au * AU_KM,
        },
        normalize_deg(lambda),
    )
}

/// Geocentric ecliptic position of the Moon.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MoonEcliptic {
    /// Geocentric longitude referred to the mean equinox of date (no nutation), degrees.
    pub longitude: f64,
    /// Geocentric latitude, degrees.
    pub latitude: f64,
    /// Distance between the centres of the Earth and Moon, km.
    pub distance_km: f64,
}

// Meeus Table 47.A rows kept (|l| ≥ 500 or |r| ≥ 2000): D, M, M', F, l (1e-6 °), r (1e-3 km).
#[rustfmt::skip]
const MOON_LR: [(i8, i8, i8, i8, i32, i32); 51] = [
    (0,0,1,0,6288774,-20905355),(2,0,-1,0,1274027,-3699111),(2,0,0,0,658314,-2955968),
    (0,0,2,0,213618,-569925),(0,1,0,0,-185116,48888),(0,0,0,2,-114332,-3149),
    (2,0,-2,0,58793,246158),(2,-1,-1,0,57066,-152138),(2,0,1,0,53322,-170733),
    (2,-1,0,0,45758,-204586),(0,1,-1,0,-40923,-129620),(1,0,0,0,-34720,108743),
    (0,1,1,0,-30383,104755),(2,0,0,-2,15327,10321),(0,0,1,2,-12528,0),
    (0,0,1,-2,10980,79661),(4,0,-1,0,10675,-34782),(0,0,3,0,10034,-23210),
    (4,0,-2,0,8548,-21636),(2,1,-1,0,-7888,24208),(2,1,0,0,-6766,30824),
    (1,0,-1,0,-5163,-8379),(1,1,0,0,4987,-16675),(2,-1,1,0,4036,-12831),
    (2,0,2,0,3994,-10445),(4,0,0,0,3861,-11650),(2,0,-3,0,3665,14403),
    (0,1,-2,0,-2689,-7003),(2,0,-1,2,-2602,0),(2,-1,-2,0,2390,10056),
    (1,0,1,0,-2348,6322),(2,-2,0,0,2236,-9884),(0,1,2,0,-2120,5751),
    (0,2,0,0,-2069,0),(2,-2,-1,0,2048,-4950),(2,0,1,-2,-1773,4130),
    (2,0,0,2,-1595,0),(4,-1,-1,0,1215,-3958),(0,0,2,2,-1110,0),
    (3,0,-1,0,-892,3258),(2,1,1,0,-810,2616),(4,-1,-2,0,759,-1897),
    (0,2,-1,0,-713,-2117),(2,2,-1,0,-700,2354),(2,1,-2,0,691,0),
    (2,-1,0,-2,596,0),(4,0,1,0,549,-1423),(0,0,4,0,537,-1117),
    (4,-1,0,0,520,-1571),(0,0,2,-2,-381,-4421),(2,0,-1,-2,0,8752),
];

// Meeus Table 47.B rows kept (|b| ≥ 500): D, M, M', F, b (1e-6 °).
#[rustfmt::skip]
const MOON_B: [(i8, i8, i8, i8, i32); 34] = [
    (0,0,0,1,5128122),(0,0,1,1,280602),(0,0,1,-1,277693),(2,0,0,-1,173237),
    (2,0,-1,1,55413),(2,0,-1,-1,46271),(2,0,0,1,32573),(0,0,2,1,17198),
    (2,0,1,-1,9266),(0,0,2,-1,8822),(2,-1,0,-1,8216),(2,0,-2,-1,4324),
    (2,0,1,1,4200),(2,1,0,-1,-3359),(2,-1,-1,1,2463),(2,-1,0,1,2211),
    (2,-1,-1,-1,2065),(0,1,-1,-1,-1870),(4,0,-1,-1,1828),(0,1,0,1,-1794),
    (0,0,0,3,-1749),(0,1,-1,1,-1565),(1,0,0,1,-1491),(0,1,1,1,-1475),
    (0,1,1,-1,-1410),(0,1,0,-1,-1344),(1,0,0,-1,-1335),(0,0,3,1,1107),
    (4,0,0,-1,1021),(4,0,-1,1,833),(0,0,1,-3,777),(4,0,-2,1,671),
    (2,0,0,-3,607),(2,0,2,-1,596),
];

/// The Moon's geocentric ecliptic position (Meeus ch. 47, truncated series: 51 + 34 terms,
/// within 0.0023° in longitude, 0.0041° in latitude and 2.9 km over 1900–2100).
pub fn moon_position(t: f64) -> MoonEcliptic {
    let lp = 218.316_447_7 + 481_267.881_234_21 * t - 0.001_578_6 * t * t + t.powi(3) / 538_841.0
        - t.powi(4) / 65_194_000.0;
    let d = 297.850_192_1 + 445_267.111_403_4 * t - 0.001_881_9 * t * t + t.powi(3) / 545_868.0
        - t.powi(4) / 113_065_000.0;
    let m = 357.529_109_2 + 35_999.050_290_9 * t - 0.000_153_6 * t * t + t.powi(3) / 24_490_000.0;
    let mp = 134.963_396_4 + 477_198.867_505_5 * t + 0.008_741_4 * t * t + t.powi(3) / 69_699.0
        - t.powi(4) / 14_712_000.0;
    let f = 93.272_095 + 483_202.017_523_3 * t - 0.003_653_9 * t * t - t.powi(3) / 3_526_000.0
        + t.powi(4) / 863_310_000.0;
    let a1 = 119.75 + 131.849 * t;
    let a2 = 53.09 + 479_264.29 * t;
    let a3 = 313.45 + 481_266.484 * t;
    let e = 1.0 - 0.002_516 * t - 0.000_007_4 * t * t;
    let ecc = |mi: i8| match mi.unsigned_abs() {
        0 => 1.0,
        1 => e,
        _ => e * e,
    };

    let mut sl = 0.0;
    let mut sr = 0.0;
    for &(cd, cm, cmp, cf, l, r) in &MOON_LR {
        let arg = d * f64::from(cd) + m * f64::from(cm) + mp * f64::from(cmp) + f * f64::from(cf);
        let k = ecc(cm);
        sl += f64::from(l) * k * sin(arg);
        sr += f64::from(r) * k * cos(arg);
    }
    let mut sb = 0.0;
    for &(cd, cm, cmp, cf, b) in &MOON_B {
        let arg = d * f64::from(cd) + m * f64::from(cm) + mp * f64::from(cmp) + f * f64::from(cf);
        sb += f64::from(b) * ecc(cm) * sin(arg);
    }
    sl += 3958.0 * sin(a1) + 1962.0 * sin(lp - f) + 318.0 * sin(a2);
    sb += -2235.0 * sin(lp)
        + 382.0 * sin(a3)
        + 175.0 * sin(a1 - f)
        + 175.0 * sin(a1 + f)
        + 127.0 * sin(lp - mp)
        - 115.0 * sin(lp + mp);
    MoonEcliptic {
        longitude: normalize_deg(lp + sl * 1e-6),
        latitude: sb * 1e-6,
        distance_km: 385_000.56 + sr * 1e-3,
    }
}

/// Ecliptic (λ, β) to equatorial (α, δ), degrees, for obliquity `eps` (Meeus 13.3, 13.4).
pub fn ecliptic_to_equatorial(lambda: f64, beta: f64, eps: f64) -> (f64, f64) {
    let ra = (sin(lambda) * cos(eps) - (beta * D2R).tan() * sin(eps)).atan2(cos(lambda)) * R2D;
    let dec = (sin(beta) * cos(eps) + cos(beta) * sin(eps) * sin(lambda)).asin() * R2D;
    (normalize_deg(ra), dec)
}

/// The Moon's apparent geocentric equatorial position: [`moon_position`] plus nutation in
/// longitude, converted with the true obliquity.
pub fn moon_apparent(t: f64, nut: &Nutation) -> (Equatorial, MoonEcliptic) {
    let ecl = moon_position(t);
    let (ra, dec) = ecliptic_to_equatorial(ecl.longitude + nut.dpsi, ecl.latitude, nut.eps());
    (
        Equatorial {
            ra,
            dec,
            distance_km: ecl.distance_km,
        },
        ecl,
    )
}

/// Altitude and azimuth (from north through east), degrees, of an object at `ra`/`dec` seen
/// from latitude `lat` at local sidereal time `lst` (Meeus ch. 13).
pub fn horizontal(ra: f64, dec: f64, lat: f64, lst: f64) -> (f64, f64) {
    let h = lst - ra;
    let alt = (sin(lat) * sin(dec) + cos(lat) * cos(dec) * cos(h)).asin() * R2D;
    let az = sin(h).atan2(cos(h) * sin(lat) - (dec * D2R).tan() * cos(lat)) * R2D + 180.0;
    (alt, normalize_deg(az))
}

/// Atmospheric refraction, degrees, to add to a true altitude `alt` (Saemundsson, Meeus 16.4;
/// SPA eq. 42), for pressure `pressure_mbar` and temperature `temp_c`. Zero below −0.8333°.
pub fn refraction_deg(alt: f64, pressure_mbar: f64, temp_c: f64) -> f64 {
    if alt < -0.8333 {
        return 0.0;
    }
    1.02 / (60.0 * ((alt + 10.3 / (alt + 5.11)) * D2R).tan())
        * (pressure_mbar / 1010.0)
        * (283.0 / (273.0 + temp_c))
}

/// Total horizontal illuminance from a body (direct plus sky), as a fraction of its
/// illuminance outside the atmosphere on a surface facing it, for a refracted (apparent)
/// altitude in degrees. Valid from the zenith through twilight (the sky term stays positive below
/// the horizon): 0.925 at the zenith, 0.0057 on the horizon, 2.2e-5 at −6°.
///
/// Subroutine `ATMOS` of Janiczek & DeYoung 1987, *Computer programs for sun and moon
/// illuminance*, US Naval Observatory Circular 171 (X = Earth radius / scale height):
/// `M = X(cos S − sin h) + cos S` with `S = asin(X cos h / (X + 1))`, then
/// `e^(−0.21 M) sin h + 0.0289 e^(−0.042 M)(1 + (h + 90) sin h / 57.29578)`.
pub fn horizontal_illuminance_factor(apparent_altitude: f64) -> f64 {
    let x = 753.661_56;
    let u = sin(apparent_altitude);
    let s = (x * cos(apparent_altitude) / (x + 1.0)).asin();
    let m = x * (s.cos() - u) + s.cos();
    ((-0.21 * m).exp() * u
        + 0.0289 * (-0.042 * m).exp() * (1.0 + (apparent_altitude + 90.0) * u / R2D))
        .max(0.0)
}

/// Unit vector toward altitude/azimuth in the east-north-up frame: (east, north, up).
pub fn enu(alt: f64, az: f64) -> [f64; 3] {
    [cos(alt) * sin(az), cos(alt) * cos(az), sin(alt)]
}

/// Altitude/azimuth of an east-north-up vector, degrees.
pub fn alt_az(v: [f64; 3]) -> (f64, f64) {
    let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt().max(1e-300);
    let alt = (v[2] / n).clamp(-1.0, 1.0).asin() * R2D;
    let az = normalize_deg(v[0].atan2(v[1]) * R2D);
    (alt, az)
}

/// Topocentric altitude/azimuth of a body at geocentric `alt`/`az` and distance `distance_km`:
/// the observer stands one Earth radius above the centre (oblateness neglected).
pub fn topocentric(alt: f64, az: f64, distance_km: f64) -> (f64, f64) {
    let d = enu(alt, az);
    alt_az([
        d[0] * distance_km,
        d[1] * distance_km,
        d[2] * distance_km - EARTH_RADIUS_KM,
    ])
}

/// Phase angle i (0° full, 180° new), degrees, from the geocentric Sun and Moon (Meeus 48.2–48.3).
pub fn phase_angle(sun: &Equatorial, moon: &Equatorial) -> f64 {
    let cos_psi =
        sin(sun.dec) * sin(moon.dec) + cos(sun.dec) * cos(moon.dec) * cos(sun.ra - moon.ra);
    let psi = cos_psi.clamp(-1.0, 1.0).acos();
    (sun.distance_km * psi.sin()).atan2(moon.distance_km - sun.distance_km * psi.cos()) * R2D
}

/// Illuminated fraction k of the Moon's disc for phase angle `i` (Meeus 48.1).
pub fn illuminated_fraction(i: f64) -> f64 {
    (1.0 + cos(i)) / 2.0
}

/// Position angle of the Moon's bright limb, degrees from north through east (Meeus 48.5).
pub fn bright_limb_angle(sun: &Equatorial, moon: &Equatorial) -> f64 {
    let y = cos(sun.dec) * sin(sun.ra - moon.ra);
    let x = sin(sun.dec) * cos(moon.dec) - cos(sun.dec) * sin(moon.dec) * cos(sun.ra - moon.ra);
    normalize_deg(y.atan2(x) * R2D)
}

/// Moonlight relative to full moon for phase angle `i` (Krisciunas & Schaefer 1991 eq. 9):
/// `10^(−0.4·(0.026|i| + 4e−9 i⁴))`, 1 at full moon.
pub fn moon_phase_factor(i: f64) -> f64 {
    let i = i.abs();
    10f64.powf(-0.4 * (0.026 * i + 4e-9 * i.powi(4)))
}

/// Angular diameter of the Moon's disc, degrees, at `distance_km` from the observer.
pub fn moon_angular_diameter(distance_km: f64) -> f64 {
    2.0 * (MOON_RADIUS_KM / distance_km).clamp(-1.0, 1.0).asin() * R2D
}

/// Precession matrix from J2000 equatorial to equatorial of date (Meeus 21.2–21.4), row-major,
/// acting on unit vectors `(cos δ cos α, cos δ sin α, sin δ)`.
pub fn precession_matrix(t: f64) -> [[f64; 3]; 3] {
    let zeta = (2306.2181 * t + 0.301_88 * t * t + 0.017_998 * t.powi(3)) / 3600.0;
    let z = (2306.2181 * t + 1.094_68 * t * t + 0.018_203 * t.powi(3)) / 3600.0;
    let theta = (2004.3109 * t - 0.426_65 * t * t - 0.041_833 * t.powi(3)) / 3600.0;
    let (s_zeta, c_zeta) = (sin(zeta), cos(zeta));
    let (s_z, c_z) = (sin(z), cos(z));
    let (s_t, c_t) = (sin(theta), cos(theta));
    // P = Rz(z) · Ry(−θ) · Rz(ζ), the matrix form of Meeus 21.4.
    [
        [
            c_z * c_t * c_zeta - s_z * s_zeta,
            -c_z * c_t * s_zeta - s_z * c_zeta,
            -c_z * s_t,
        ],
        [
            s_z * c_t * c_zeta + c_z * s_zeta,
            -s_z * c_t * s_zeta + c_z * c_zeta,
            -s_z * s_t,
        ],
        [s_t * c_zeta, -s_t * s_zeta, c_t],
    ]
}

/// Matrix taking an equatorial-of-date unit vector to east-north-up at latitude `lat` and local
/// sidereal time `lst`, row-major.
pub fn equatorial_to_enu(lat: f64, lst: f64) -> [[f64; 3]; 3] {
    let (s, c) = (sin(lst), cos(lst));
    let (sp, cp) = (sin(lat), cos(lat));
    // x' = cos δ cos H = c·x + s·y ; y' = cos δ sin H = s·x − c·y ; z = sin δ
    // E = −y', N = z cos φ − x' sin φ, U = z sin φ + x' cos φ
    [[-s, c, 0.0], [-c * sp, -s * sp, cp], [c * cp, s * cp, sp]]
}

/// Product of two row-major 3×3 matrices.
pub fn mat_mul(a: [[f64; 3]; 3], b: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut out = [[0.0; 3]; 3];
    for (i, row) in out.iter_mut().enumerate() {
        for (j, cell) in row.iter_mut().enumerate() {
            *cell = (0..3).map(|k| a[i][k] * b[k][j]).sum();
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::celestial::{CivilDate, WorldClock};

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol
    }

    fn hms(h: f64, m: f64, s: f64) -> f64 {
        (h + m / 60.0 + s / 3600.0) * 15.0
    }

    #[test]
    fn sidereal_time_matches_meeus_12a_and_12b() {
        assert!(close(mean_sidereal_deg(2_446_895.5), 197.693_195, 1e-5));
        let jd = WorldClock::new(CivilDate::new(1987, 4, 10))
            .at_hour(19.0 + 21.0 / 60.0)
            .julian_day(0.0);
        assert!(close(mean_sidereal_deg(jd), 128.737_873, 1e-5));
        // Apparent: 13h10m46.1351s (Meeus 12.a), with the four-term nutation.
        let t = centuries(2_446_895.5);
        let app = local_sidereal_deg(2_446_895.5, &nutation(t), 0.0);
        assert!(close(app, hms(13.0, 10.0, 46.1351), 5e-4), "{app}");
    }

    #[test]
    fn delta_t_matches_the_espenak_meeus_anchors() {
        assert!(close(delta_t_seconds(2000.0), 63.86, 0.01));
        assert!(close(delta_t_seconds(2050.0), 93.0, 0.01));
    }

    #[test]
    fn the_sun_matches_meeus_25a() {
        let t = centuries(2_448_908.5);
        assert!(close(t, -0.072_183_436, 1e-9));
        let (sun, lambda) = sun_apparent(t);
        assert!(close(lambda, 199.908_89, 5e-4), "{lambda}");
        assert!(close(sun.ra, 198.380_83, 5e-3), "{}", sun.ra);
        assert!(close(sun.dec, -7.785_07, 5e-3), "{}", sun.dec);
        assert!(close(sun.distance_km / AU_KM, 0.997_66, 1e-5));
    }

    #[test]
    fn the_moon_matches_meeus_47a() {
        let t = centuries(2_448_724.5);
        let ecl = moon_position(t);
        assert!(close(ecl.longitude, 133.162_655, 0.01), "{}", ecl.longitude);
        assert!(close(ecl.latitude, -3.229_126, 0.01), "{}", ecl.latitude);
        assert!(
            close(ecl.distance_km, 368_409.7, 10.0),
            "{}",
            ecl.distance_km
        );
        let (eq, _) = moon_apparent(t, &nutation(t));
        assert!(close(eq.ra, 134.688_470, 5e-3), "{}", eq.ra);
        assert!(close(eq.dec, 13.768_368, 5e-3), "{}", eq.dec);
    }

    #[test]
    fn the_phase_matches_meeus_48a() {
        let sun = Equatorial {
            ra: 20.6579,
            dec: 8.6964,
            distance_km: 149_971_520.0,
        };
        let moon = Equatorial {
            ra: 134.6885,
            dec: 13.7684,
            distance_km: 368_410.0,
        };
        let i = phase_angle(&sun, &moon);
        assert!(close(i, 69.0756, 0.01), "{i}");
        assert!(close(illuminated_fraction(i), 0.6786, 5e-4));
        assert!(close(bright_limb_angle(&sun, &moon), 285.0, 0.1));
    }

    #[test]
    fn moonlight_follows_krisciunas_schaefer() {
        assert_eq!(moon_phase_factor(0.0), 1.0);
        assert!(close(moon_phase_factor(30.0), 0.486_08, 1e-5));
        assert!(close(moon_phase_factor(90.0), 0.090_996, 1e-6));
        assert!(close(moon_phase_factor(180.0), 2.8079e-4, 1e-8));
        assert!(close(moon_angular_diameter(368_409.7), 0.5404, 1e-4));
    }

    #[test]
    fn horizontal_illuminance_matches_usno_circular_171() {
        assert!(close(horizontal_illuminance_factor(90.0), 0.925_353, 1e-6));
        assert!(close(
            horizontal_illuminance_factor(0.0) * 133_775.0,
            756.605,
            0.01
        ));
        // Civil twilight (−6°): about 3 lx, the order of the 3.4 lx Bevy calls CIVIL_TWILIGHT.
        assert!(close(
            horizontal_illuminance_factor(-6.0) * 133_775.0,
            2.992_14,
            1e-4
        ));
        assert!(horizontal_illuminance_factor(-18.0) * 133_775.0 < 1e-5);
    }

    #[test]
    fn refraction_matches_saemundsson() {
        assert!(close(refraction_deg(0.0, 1010.0, 10.0), 0.483_03, 1e-5));
        assert!(close(refraction_deg(10.0, 1010.0, 10.0), 0.090_13, 1e-5));
        assert!(close(refraction_deg(45.0, 1010.0, 10.0), 0.016_878, 1e-5));
    }

    /// NREL TP-560-34302 Table A5.1: 2003-10-17 12:30:30 at UTC−7, 39.742476° N, 105.1786° W,
    /// ΔT 67 s, 820 mbar, 11 °C → topocentric zenith 50.11162°, azimuth 194.34024°.
    #[test]
    fn the_sun_matches_the_spa_example() {
        let clock = WorldClock::new(CivilDate::new(2003, 10, 17)).at_hour(12.5 + 30.0 / 3600.0);
        let jd = clock.julian_day(-7.0);
        assert!(close(jd, 2_452_930.312_847, 1e-6));
        let t = centuries(jd + 67.0 / 86_400.0);
        let nut = nutation(t);
        let (sun, _) = sun_apparent(t);
        let lst = local_sidereal_deg(jd, &nut, -105.1786);
        let (alt, az) = horizontal(sun.ra, sun.dec, 39.742_476, lst);
        let parallax = 8.794 / 3600.0 / (sun.distance_km / AU_KM);
        let alt = alt - parallax * cos(alt);
        let alt = alt + refraction_deg(alt, 820.0, 11.0);
        assert!(close(90.0 - alt, 50.111_62, 0.02), "zenith {}", 90.0 - alt);
        assert!(close(az, 194.340_24, 0.02), "azimuth {az}");
    }

    #[test]
    fn the_equatorial_to_enu_matrix_agrees_with_horizontal() {
        let (lat, lst) = (45.0, 123.4);
        let m = equatorial_to_enu(lat, lst);
        for (ra, dec) in [(10.0, 20.0), (200.0, -35.0), (123.4, 80.0)] {
            let v = [cos(dec) * cos(ra), cos(dec) * sin(ra), sin(dec)];
            let e: Vec<f64> = (0..3)
                .map(|i| (0..3).map(|k| m[i][k] * v[k]).sum())
                .collect();
            let (alt, az) = alt_az([e[0], e[1], e[2]]);
            let (alt2, az2) = horizontal(ra, dec, lat, lst);
            assert!(
                close(alt, alt2, 1e-9) && close(az, az2, 1e-9),
                "{alt} {az} vs {alt2} {az2}"
            );
        }
    }

    #[test]
    fn the_precession_matrix_matches_meeus_21_4() {
        let t = 0.288_1; // 2028.8
        let p = precession_matrix(t);
        let (ra, dec): (f64, f64) = (41.054_063, 49.227_750);
        let v = [cos(dec) * cos(ra), cos(dec) * sin(ra), sin(dec)];
        let w: Vec<f64> = (0..3)
            .map(|i| (0..3).map(|k| p[i][k] * v[k]).sum())
            .collect();
        let zeta = (2306.2181 * t + 0.301_88 * t * t + 0.017_998 * t.powi(3)) / 3600.0;
        let z = (2306.2181 * t + 1.094_68 * t * t + 0.018_203 * t.powi(3)) / 3600.0;
        let theta = (2004.3109 * t - 0.426_65 * t * t - 0.041_833 * t.powi(3)) / 3600.0;
        let a = cos(dec) * sin(ra + zeta);
        let b = cos(theta) * cos(dec) * cos(ra + zeta) - sin(theta) * sin(dec);
        let c = sin(theta) * cos(dec) * cos(ra + zeta) + cos(theta) * sin(dec);
        let ra2 = normalize_deg(a.atan2(b) * R2D + z);
        let dec2 = c.asin() * R2D;
        let (ra1, dec1) = (normalize_deg(w[1].atan2(w[0]) * R2D), w[2].asin() * R2D);
        assert!(close(ra1, ra2, 1e-9) && close(dec1, dec2, 1e-9));
    }
}
