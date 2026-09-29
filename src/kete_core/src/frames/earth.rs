//! Earth orientation: the WGS84 coordinate system, precession, nutation,
//! sidereal time, and the TEME frame.
// BSD 3-Clause License
//
// Copyright (c) 2026, Dar Dahlen
// Copyright (c) 2025, California Institute of Technology
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from
//    this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

use nalgebra::{Matrix3, Rotation3, Vector3};
use std::f64::consts::TAU;

use crate::{
    constants::AU_KM,
    desigs::Desig,
    errors::KeteResult,
    state::State,
    time::{TDB, Time, UTC},
};

use super::{Ecliptic, Equatorial, NonInertialFrame};

/// Earth semi major axis in km as defined by WGS84
pub const EARTH_A: f64 = 6378.1370;

/// Earth semi minor axis in km as defined by WGS84
const EARTH_B: f64 = 6356.7523142;

// /// Earth inverse flattening as defined by WGS84
const _EARTH_INV_FLAT: f64 = 298.2572235629972;

/// Earth surface eccentricity squared, calculated from above.
/// e^2 = (2 - flattening) * flattening
const EARTH_E2: f64 = 0.0066943799901413165;

/// Ecliptic obliquity angle in radians at the J2000 epoch. This is using the definition
/// from the 1984 JPL DE Series. These constants allow the conversion between Ecliptic
/// and Equatorial frames. Note that there are more modern definitions for these values,
/// however these are used for compatibility with JPL Horizons and Spice.
///
/// See:
///     - <https://en.wikipedia.org/wiki/Axial_tilt#Short_term>
///     - <https://ssd.jpl.nasa.gov/horizons/manual.html#defs>
pub(super) const OBLIQUITY: f64 = 0.40909280422232897;

/// Prime vertical radius of curvature.
/// This is the radius of curvature of the earth surface at the specific geodetic
/// latitude.
#[must_use]
pub fn prime_vert_radius(geodetic_lat: f64) -> f64 {
    EARTH_A / (1.0 - EARTH_E2 * geodetic_lat.sin().powi(2)).sqrt()
}

/// Compute earths geocentric radius at the specified latitude in km.
///
/// # Arguments
/// * `geodetic_lat` - Geodetic latitude in radians.
#[must_use]
pub fn geocentric_radius(geodetic_lat: f64) -> f64 {
    let (sin, cos) = geodetic_lat.sin_cos();
    let a_cos = EARTH_A * cos;
    let b_sin = EARTH_B * sin;
    (((EARTH_A * a_cos).powi(2) + (EARTH_B * b_sin).powi(2)) / (a_cos.powi(2) + b_sin.powi(2)))
        .sqrt()
}

/// Convert an ECEF position to geodetic latitude, longitude and height.
///
/// `x`, `y` and `z` are the ECEF position in km. The result is the geodetic
/// latitude and longitude in radians, and the height above the WGS84 ellipsoid
/// in km.
#[must_use]
pub fn ecef_to_geodetic_lat_lon(x: f64, y: f64, z: f64) -> (f64, f64, f64) {
    let longitude = f64::atan2(y, x);
    let p = (x * x + y * y).sqrt();

    // Start from the geocentric latitude. The iteration usually converges in one
    // or two steps. A fixed step count avoids a branch.
    let mut geodetic_lat = f64::atan2(z, p);
    let mut height = 0.0;
    // The height p cos(lat) + z sin(lat) - a^2 / N stays well conditioned at the
    // poles. The form p / cos(lat) - N does not.
    for _ in 0..5 {
        let vert_rad = prime_vert_radius(geodetic_lat);
        let (sin, cos) = geodetic_lat.sin_cos();
        height = p * cos + z * sin - EARTH_A * EARTH_A / vert_rad;
        geodetic_lat = f64::atan2(z, p * (1.0 - EARTH_E2 * vert_rad / (vert_rad + height)));
    }

    (geodetic_lat, longitude, height)
}

/// Compute geocentric latitude from geodetic latitude.
///
/// # Arguments
/// * `geodetic_lat` - Geodetic latitude in radians.
/// * `h` - Height above sea level in km.
#[must_use]
pub fn geodetic_lat_to_geocentric(geodetic_lat: f64, height: f64) -> f64 {
    let n = prime_vert_radius(geodetic_lat);
    ((1.0 - EARTH_E2 * n / (n + height)) * geodetic_lat.tan()).atan()
}

/// Compute the ECEF X/Y/Z position in km from geodetic position.
///
/// # Arguments
/// * `geodetic_lat` - Geodetic latitude in radians.
/// * `geodetic_lon` - Geodetic longitude in radians.
/// * `height` - Height above sea level in km.
#[must_use]
pub fn geodetic_lat_lon_to_ecef(geodetic_lat: f64, geodetic_lon: f64, height: f64) -> [f64; 3] {
    let n = prime_vert_radius(geodetic_lat);
    let (sin_gd_lat, cos_gd_lat) = geodetic_lat.sin_cos();
    let (sin_gd_lon, cos_gd_lon) = geodetic_lon.sin_cos();
    let x = (n + height) * cos_gd_lat * cos_gd_lon;
    let y = (n + height) * cos_gd_lat * sin_gd_lon;
    let z = ((1.0 - EARTH_E2) * n + height) * sin_gd_lat;
    [x, y, z]
}

/// Compute MPC parallax constants (rho*cos(phi'), rho*sin(phi')) from geodetic coordinates.
///
/// phi' is the geocentric latitude and rho is the distance from Earth's center, both expressed
/// in units of Earth's equatorial radius. These constants, combined with a longitude, specify
/// the 3D position of an observatory.
///
/// # Arguments
/// * `geodetic_lat` - Geodetic latitude in radians.
/// * `height` - Height above the WGS84 ellipsoid in km.
#[must_use]
pub fn geodetic_to_parallax(geodetic_lat: f64, height: f64) -> (f64, f64) {
    let n = prime_vert_radius(geodetic_lat);
    let (sin_gd_lat, cos_gd_lat) = geodetic_lat.sin_cos();
    let rho_cos = (n + height) * cos_gd_lat / EARTH_A;
    let rho_sin = ((1.0 - EARTH_E2) * n + height) * sin_gd_lat / EARTH_A;
    (rho_cos, rho_sin)
}

/// Compute the mean obliquity of the ecliptic of date in radians.
///
/// `jd` is the time in TDB scaled Julian Days. The expression is the IAU 2006
/// obliquity, equation (39) of the paper cited on
/// [`earth_precession_rotation`]. Use it within several centuries of J2000.
#[must_use]
pub fn earth_obliquity(jd: Time<TDB>) -> f64 {
    // centuries from j2000
    let t = (jd - Time::j2000()).elapsed / 36525.0;
    let arcsec = 84_381.406
        + (-46.836_769
            + (-0.000_183_1 + (0.002_003_40 + (-0.000_000_576 - 0.000_000_043_4 * t) * t) * t) * t)
            * t;
    (arcsec / 3600.0).to_radians()
}

/// Approximate Greenwich mean sidereal time in radians.
///
/// `time` is the time in TDB scaled Julian Days. The result is the angle from
/// the mean equinox of date to the Greenwich meridian.
///
/// The angle is the Earth Rotation Angle plus the accumulated precession in
/// right ascension. The Earth Rotation Angle uses the UT1 of [`approx_ut1`].
/// The precession term uses TDB, and it matches [`earth_precession_rotation`].
/// The two expressions are equations (5.14) and (5.32) of Chapter 5 of:
///
/// > "IERS Conventions (2010)"\\
/// > Petit, G. ; Luzum, B. (eds.)\\
/// > IERS Technical Note No. 36 (2010)
#[must_use]
pub fn greenwich_mean_sidereal_time(time: Time<TDB>) -> f64 {
    let du = approx_ut1(time) - 2451545.0;
    let t = (time - Time::j2000()).elapsed / 36525.0;

    // Remove the whole days before the scale factor to keep the precision.
    let era = TAU * (du.rem_euclid(1.0) + 0.779_057_273_264 + 0.002_737_811_911_354_48 * du);
    let precession = 0.014_506
        + (4_612.156_534
            + (1.391_581_7 + (-0.000_000_44 + (-0.000_029_956 - 0.000_000_036_8 * t) * t) * t) * t)
            * t;
    (era + (precession / 3600.0).to_radians()).rem_euclid(TAU)
}

/// Approximate UT1 as a Julian Date.
///
/// From 1972, when UTC began to track UT1 through leap seconds, UTC is used, as
/// the two differ by less than a second. After the last leap second known to
/// kete, UT1 is taken to follow UTC.
///
/// Before 1972, UT1 is TT minus the Delta T of [`approx_delta_t`]. TDB is used
/// in place of TT, which it matches to within 2 milliseconds.
///
/// # Arguments
/// * `time` - Time in TDB scaled Julian Days.
///
#[must_use]
pub fn approx_ut1(time: Time<TDB>) -> f64 {
    // 1972-01-01, the start of UTC with leap seconds.
    const LEAP_SECOND_START_JD: f64 = 2441317.5;
    let utc = time.utc();
    if utc.jd >= LEAP_SECOND_START_JD {
        utc.jd
    } else {
        time.jd - approx_delta_t(time) / 86400.0
    }
}

/// Approximate Delta T, TT - UT1, in seconds.
///
/// These are the polynomial expressions of Espenak and Meeus (Five Millennium
/// Canon of Solar Eclipses, 2006), which follow the historical record of Earth
/// rotation, and a long-term parabola before the year -500 and after 2150.
/// Delta T is poorly known before the telescopic era, and its uncertainty
/// grows rapidly before 1600. From 2005 onward the expressions are the
/// canon's predictions made in 2006, not observations, and they have since
/// drifted from the measured Earth rotation; [`approx_ut1`] uses UTC instead
/// of this from 1972.
///
/// # Arguments
/// * `time` - Time in TDB scaled Julian Days.
///
#[must_use]
pub fn approx_delta_t(time: Time<TDB>) -> f64 {
    let y = 2000.0 + (time.jd - 2451544.5) / 365.25;
    let parabola = |y: f64| -20.0 + 32.0 * ((y - 1820.0) / 100.0).powi(2);
    match y {
        y if !(-500.0..2150.0).contains(&y) => parabola(y),
        y if y < 500.0 => {
            let u = y / 100.0;
            10583.6
                + u * (-1014.41
                    + u * (33.78311
                        + u * (-5.952053
                            + u * (-0.1798452 + u * (0.022174192 + u * 0.0090316521)))))
        }
        y if y < 1600.0 => {
            let u = (y - 1000.0) / 100.0;
            1574.2
                + u * (-556.01
                    + u * (71.23472
                        + u * (0.319781
                            + u * (-0.8503463 + u * (-0.005050998 + u * 0.0083572073)))))
        }
        y if y < 1700.0 => {
            let t = y - 1600.0;
            120.0 + t * (-0.9808 + t * (-0.01532 + t / 7129.0))
        }
        y if y < 1800.0 => {
            let t = y - 1700.0;
            8.83 + t * (0.1603 + t * (-0.0059285 + t * (0.00013336 - t / 1174000.0)))
        }
        y if y < 1860.0 => {
            let t = y - 1800.0;
            13.72
                + t * (-0.332447
                    + t * (0.0068612
                        + t * (0.0041116
                            + t * (-0.00037436
                                + t * (0.0000121272 + t * (-0.0000001699 + t * 0.000000000875))))))
        }
        y if y < 1900.0 => {
            let t = y - 1860.0;
            7.62 + t
                * (0.5737 + t * (-0.251754 + t * (0.01680668 + t * (-0.0004473624 + t / 233174.0))))
        }
        y if y < 1920.0 => {
            let t = y - 1900.0;
            -2.79 + t * (1.494119 + t * (-0.0598939 + t * (0.0061966 - t * 0.000197)))
        }
        y if y < 1941.0 => {
            let t = y - 1920.0;
            21.20 + t * (0.84493 + t * (-0.076100 + t * 0.0020936))
        }
        y if y < 1961.0 => {
            let t = y - 1950.0;
            29.07 + t * (0.407 + t * (-1.0 / 233.0 + t / 2547.0))
        }
        y if y < 1986.0 => {
            let t = y - 1975.0;
            45.45 + t * (1.067 + t * (-1.0 / 260.0 - t / 718.0))
        }
        y if y < 2005.0 => {
            let t = y - 2000.0;
            63.86
                + t * (0.3345
                    + t * (-0.060374 + t * (0.0017275 + t * (0.000651814 + t * 0.00002373599))))
        }
        y if y < 2050.0 => {
            let t = y - 2000.0;
            62.92 + t * (0.32217 + t * 0.005589)
        }
        y => parabola(y) - 0.5628 * (2150.0 - y),
    }
}

/// Compute the nutation in longitude and obliquity in radians, `(dpsi, deps)`.
///
/// `time` is the time in TDB scaled Julian Days. The model is IAU 2000B, an
/// abridged form of IAU 2000A, from:
///
/// > "An Abridged Model of the Precession-Nutation of the Celestial Pole"\\
/// > McCarthy, D. D. ; Luzum, B. J.\\
/// > Celestial Mechanics and Dynamical Astronomy, v.85, p.37-49 (2003)\\
/// > doi:10.1023/A:1021762727016
///
/// The function sums the series of Table II of the paper over its linear
/// fundamental arguments. It then adds the constant offsets of the paper. The
/// offsets replace the long period lunisolar and planetary terms that the
/// series omits.
///
/// The paper also gives corrections to the IAU 1976 precession and a frame
/// bias. This function does not apply them. [`earth_precession_rotation`] is
/// the IAU 2006 precession, which needs no correction. [`approx_earth_frame`]
/// and [`teme_frame`] apply the frame bias.
#[must_use]
pub fn earth_nutation(time: Time<TDB>) -> (f64, f64) {
    // Julian centuries from J2000.
    let t = (time - Time::j2000()).elapsed / 36525.0;

    // Fundamental arguments in arcseconds, from the paper: mean anomalies of the
    // Moon and the Sun, the Moon's mean longitude minus its node, the elongation of
    // the Moon from the Sun, and the longitude of the Moon's ascending node.
    let arguments = [
        485_868.249_036 + 1_717_915_923.217_8 * t,
        1_287_104.793_05 + 129_596_581.048_1 * t,
        335_779.526_232 + 1_739_527_262.847_8 * t,
        1_072_260.703_69 + 1_602_961_601.209_0 * t,
        450_160.398_036 - 6_962_890.543_1 * t,
    ]
    .map(|arcsec| (arcsec.rem_euclid(1_296_000.0) / 3600.0).to_radians());

    let (mut dpsi, mut deps) = (0.0, 0.0);
    for (multipliers, [a, a_rate, b, b_rate, a_cos, b_sin]) in IAU_2000B_TERMS {
        let arg: f64 = multipliers
            .iter()
            .zip(arguments)
            .map(|(&k, angle)| f64::from(k) * angle)
            .sum();
        let (sin, cos) = arg.sin_cos();
        dpsi += (a + a_rate * t) * sin + a_cos * cos;
        deps += (b + b_rate * t) * cos + b_sin * sin;
    }

    // Coefficients are in units of 0.1 microarcseconds; the offsets are in
    // arcseconds.
    let arcsec = |x: f64| (x / 3600.0).to_radians();
    (
        arcsec(dpsi * 1e-7 - 0.001_583_5),
        arcsec(deps * 1e-7 + 0.001_633_9),
    )
}

/// Compute the frame bias from the mean equator and equinox of J2000 to ICRF.
///
/// The ICRF is the Equatorial frame of the JPL ephemerides. The rotation is the
/// inverse of the bias matrix B of equation (4) of the paper cited on
/// [`earth_precession_rotation`]. The pole offsets are from its equation (15).
/// The equinox offset is from its equation (12).
fn frame_bias() -> Rotation3<f64> {
    let mas = |x: f64| (x / 3.6e6).to_radians();
    let (xi_0, eta_0, d_alpha_0) = (mas(-16.617), mas(-6.819), mas(-14.6));
    // The paper gives B = R1(-eta_0) R2(xi_0) R3(d_alpha_0) as rotations of the
    // frame. This is the inverse of B, as rotations of the vector.
    Rotation3::from_axis_angle(&Vector3::z_axis(), d_alpha_0)
        * Rotation3::from_axis_angle(&Vector3::y_axis(), xi_0)
        * Rotation3::from_axis_angle(&Vector3::x_axis(), -eta_0)
}

/// Compute the True Equator, Mean Equinox (TEME) frame of date.
///
/// `time` is the time in TDB scaled Julian Days. SGP4 states from two-line
/// elements are in this frame.
///
/// - The z axis is the true pole of date, from [`earth_precession_rotation`]
///   and [`earth_nutation`].
/// - The x axis is the mean equinox of date, projected onto the true equator.
///
/// The rotation takes TEME vectors to the Equatorial J2000 frame. It includes
/// the frame bias to the ICRF. The frame rotates at less than 1e-11 rad/s, and
/// the frame has no rotation rate.
pub fn teme_frame(time: Time<TDB>) -> NonInertialFrame {
    let obliquity = earth_obliquity(time);
    let (dpsi, deps) = earth_nutation(time);
    let x_axis = Vector3::x_axis();
    let z_axis = Vector3::z_axis();

    // True equator and equinox of date to mean equator and equinox of date, then
    // mean of date to J2000.
    let nutation = Rotation3::from_axis_angle(&x_axis, obliquity)
        * Rotation3::from_axis_angle(&z_axis, -dpsi)
        * Rotation3::from_axis_angle(&x_axis, -(obliquity + deps));
    let precession = frame_bias() * earth_precession_rotation(time).rotation;

    // The true pole and the mean equinox of date, expressed in J2000.
    let z = (precession * nutation * Vector3::z()).normalize();
    let x = precession * Vector3::x();
    let y = z.cross(&x).normalize();
    let x = y.cross(&z);
    let rotation = Rotation3::from_matrix_unchecked(Matrix3::from_columns(&[x, y, z]));

    NonInertialFrame::from_rotations(time, rotation, None, 1)
}

/// Compute the approximate orientation of the Earth-fixed frame.
///
/// `time` is the time in TDB scaled Julian Days. The rotation takes Earth-fixed
/// vectors, as from [`geodetic_lat_lon_to_ecef`], to the Equatorial J2000
/// frame. It applies these rotations in order:
///
/// 1. Greenwich apparent sidereal time about the true pole of date.
/// 2. The nutation of [`earth_nutation`].
/// 3. The precession of [`earth_precession_rotation`].
/// 4. The frame bias to the ICRF.
///
/// The rotation does not include polar motion. The rotation rate includes only
/// the sidereal rotation. It does not include the rates of precession and
/// nutation.
pub fn approx_earth_frame(time: Time<TDB>) -> NonInertialFrame {
    // Sidereal rotation rate in radians per day. It is the rate of the Earth
    // Rotation Angle plus the linear precession in right ascension.
    const SIDEREAL_RATE: f64 =
        TAU * (1.0 + 0.002_737_811_911_354_48) + 4_612.156_534 / 3600.0 / 36525.0 * (TAU / 360.0);

    let obliquity = earth_obliquity(time);
    let (dpsi, deps) = earth_nutation(time);

    // Apparent sidereal time adds the equation of the equinoxes and its two
    // largest complementary terms. The terms are from Table 5.2e of the IERS
    // Conventions (2010), in arcseconds. The other terms are smaller than the
    // error of the nutation model.
    let t = (time - Time::j2000()).elapsed / 36525.0;
    let node = ((450_160.398_036 - 6_962_890.543_1 * t) / 3600.0).to_radians();
    let complementary = 0.002_640_96 * node.sin() + 0.000_063_52 * (2.0 * node).sin();
    let sidereal = greenwich_mean_sidereal_time(time)
        + dpsi * obliquity.cos()
        + (complementary / 3600.0).to_radians();

    let x_axis = Vector3::x_axis();
    let z_axis = Vector3::z_axis();

    // True equator and equinox of date to mean equator and equinox of date,
    // then to J2000.
    let nutation = Rotation3::from_axis_angle(&x_axis, obliquity)
        * Rotation3::from_axis_angle(&z_axis, -dpsi)
        * Rotation3::from_axis_angle(&x_axis, -(obliquity + deps));
    let to_j2000 = frame_bias() * earth_precession_rotation(time).rotation * nutation;

    let spin = Rotation3::from_axis_angle(&z_axis, sidereal);
    let (sin, cos) = sidereal.sin_cos();
    let spin_rate = Matrix3::new(-sin, -cos, 0.0, cos, -sin, 0.0, 0.0, 0.0, 0.0) * SIDEREAL_RATE;

    NonInertialFrame::from_rotations(
        time,
        to_j2000 * spin,
        Some(to_j2000.matrix() * spin_rate),
        1,
    )
}

/// Compute the precession of the mean equator and equinox from J2000 to a date.
///
/// `time` is the time in TDB scaled Julian Days. The rotation takes vectors in
/// the mean equator and equinox of date to the J2000 Equatorial frame. The
/// equinox precesses by about 50 arcseconds per year. The rotation does not
/// include the frame bias.
///
/// The angles are the P03 precession angles of equation (40). The IAU adopted
/// them as the IAU 2006 precession. The source is:
///
/// > "Expressions for IAU 2000 precession quantities"\\
/// > Capitaine, N. ; Wallace, P. T. ; Chapront, J.\\
/// > Astronomy and Astrophysics, v.412, p.567-586 (2003)\\
/// > doi:10.1051/0004-6361:20031539
///
/// This paper defines the IAU 1976 model, and it discusses the same angles:
///
/// > "Precession matrix based on IAU (1976) system of astronomical constants."\\
/// > Lieske, J. H.\\
/// > Astronomy and Astrophysics, vol. 73, no. 3, Mar. 1979, p. 282-284.
#[inline(always)]
pub fn earth_precession_rotation(time: Time<TDB>) -> NonInertialFrame {
    // centuries since 2000
    let t = (time - Time::j2000()).elapsed / 36525.0;

    // zeta_A, z_A and theta_A of equation (40), in arcseconds.
    let angle_c = -((2.650545
        + (2306.083227
            + (0.2988499 + (0.01801828 + (-0.000005971 - 0.0000003173 * t) * t) * t) * t)
            * t)
        / 3600.0)
        .to_radians();
    let angle_a = -((-2.650545
        + (2306.077181
            + (1.0927348 + (0.01826837 + (-0.000028596 - 0.0000002904 * t) * t) * t) * t)
            * t)
        / 3600.0)
        .to_radians();
    let angle_b = ((2004.191903
        + (-0.4294934 + (-0.04182264 + (-0.000007089 - 0.0000001274 * t) * t) * t) * t)
        * t
        / 3600.0)
        .to_radians();
    let z_axis = Vector3::z_axis();
    // The J2000 to date matrix is R3(-z_A) R2(theta_A) R3(-zeta_A) as rotations
    // of the frame. This is its inverse, as rotations of the vector.
    let rotation = Rotation3::from_axis_angle(&z_axis, angle_c)
        * Rotation3::from_axis_angle(&Vector3::y_axis(), angle_b)
        * Rotation3::from_axis_angle(&z_axis, angle_a);

    NonInertialFrame::from_rotations(time, rotation, None, 1)
}

/// Compute the approximate state of a location on Earth in the Ecliptic frame.
///
/// Use this when SPICE is not available, or when the date is outside the Earth
/// orientation PCK kernels.
///
/// - `time` is the time in TDB scaled Julian Days.
/// - `geodetic_lat` and `geodetic_lon` are the geodetic latitude and longitude
///   in radians.
/// - `height` is the height above the WGS84 ellipsoid in km.
/// - `desig` is the designation of the resulting state.
///
/// The orientation is from [`approx_earth_frame`]. The velocity is the velocity
/// of the Earth's rotation. Before 1972, the Delta T of [`approx_delta_t`] sets
/// the accuracy. Delta T is poorly known before 1600.
///
/// The state is centered on the Earth. A Sun centered state also needs the
/// position of the Earth.
///
/// # Errors
/// Returns the error of [`NonInertialFrame::to_equatorial`]. The frame of
/// [`approx_earth_frame`] references the Equatorial frame directly.
pub fn approx_earth_pos_to_ecliptic(
    time: Time<TDB>,
    geodetic_lat: f64,
    geodetic_lon: f64,
    height: f64,
    desig: Desig,
) -> KeteResult<State<Ecliptic>> {
    let pos = Vector3::from(geodetic_lat_lon_to_ecef(geodetic_lat, geodetic_lon, height)) / AU_KM;
    let (pos, vel) = approx_earth_frame(time).to_equatorial(pos, [0.0; 3])?;
    Ok(State::<Equatorial>::new(desig, time, pos, vel, 399).into_frame())
}

/// Compute the next sunset and sunrise times for a given location.
///
/// This is approximate, but should be good to within a few minutes.
///
/// # Arguments
/// * `geodetic_lat` - Geodetic latitude in radians.
/// * `geodetic_lon` - Geodetic longitude in radians.
/// * `time` - Time in UTC scaled Julian Days.
pub fn next_sunset_sunrise(
    geodetic_lat: f64,
    geodetic_lon: f64,
    time: Time<UTC>,
) -> (Time<UTC>, Time<UTC>) {
    let next_noon = approx_solar_noon(time, geodetic_lon);
    let sun_dec = approx_sun_dec(next_noon);

    let cos_hr_angle = ((-0.833_f64).to_radians().sin() - geodetic_lat.sin() * sun_dec.sin())
        / (geodetic_lat.cos() * sun_dec.cos());

    let hour_angle = cos_hr_angle.acos().to_degrees() / 360.0;

    // if the predicted sunset time is more than 1 day in the future,
    // then we can subtract 1 day from the two times to get the next
    // upcoming sunset and sunrise.
    if (next_noon.jd + hour_angle) > (time.jd + 1.0) {
        (
            (next_noon.jd + hour_angle - 1.0).into(),
            (next_noon.jd - hour_angle).into(),
        )
    } else {
        // otherwise, we are already past sunset, so we will return the next
        // sunrise and sunset times.
        (
            (next_noon.jd + hour_angle).into(),
            (next_noon.jd - hour_angle + 1.0).into(),
        )
    }
}

/// Approximate the Sun's declination angle at solar noon at the specified date.
///
/// Returns the declination in radians.
///
/// # Arguments
/// * `time` - Time in UTC scaled Julian Days.
#[must_use]
pub fn approx_sun_dec(time: Time<UTC>) -> f64 {
    let obliquity = earth_obliquity(time.tdb());

    let time_since_j2000 = (time - Time::j2000()).elapsed;
    let mean_lon_of_sun = (280.459 + 0.98564736 * time_since_j2000).rem_euclid(360.0);
    let mean_anom = (357.529 + 0.98560028 * time_since_j2000)
        .rem_euclid(360.0)
        .to_radians();
    let app_eclip_lon =
        (mean_lon_of_sun + 1.915 * mean_anom.sin() + 0.020 * (2.0 * mean_anom).sin())
            .rem_euclid(360.0)
            .to_radians();

    (obliquity.sin() * app_eclip_lon.sin()).asin()
}

/// Approximate the equation of time in days.
///
/// `time` is the time in UTC scaled Julian Days. The equation of time is the
/// apparent solar time minus the mean solar time. It is positive when the Sun
/// crosses the meridian before mean noon. See
/// <https://en.wikipedia.org/wiki/Equation_of_time>.
///
/// The approximation follows the USNO approximation of the solar position:
/// <https://aa.usno.navy.mil/faq/sun_approx>.
#[must_use]
pub fn equation_of_time(time: Time<UTC>) -> f64 {
    let time_since_j2000 = (time - Time::j2000()).elapsed;
    let mean_lon_of_sun = (280.459 + 0.98564736 * time_since_j2000).rem_euclid(360.0);
    let mean_anom = (357.529 + 0.98560028 * time_since_j2000)
        .rem_euclid(360.0)
        .to_radians();
    let app_eclip_lon = (mean_lon_of_sun
        + 1.9148 * mean_anom.sin()
        + 0.0200 * (2.0 * mean_anom).sin()
        + 0.0003 * (3.0 * mean_anom).sin())
    .rem_euclid(360.0)
    .to_radians();

    0.0069 * (2.0 * app_eclip_lon).sin() - 0.0053 * mean_anom.sin()
}

/// Approximate the next local solar noon time for a given geodetic longitude.
///
/// # Arguments
/// * `time` - Time in UTC scaled Julian Days.
/// * `geodetic_lon` - Geodetic longitude in radians.
pub fn approx_solar_noon(time: Time<UTC>, geodetic_lon: f64) -> Time<UTC> {
    // compute the next clock noon after the given time
    let noon = {
        let (y, m, d, _) = time.year_month_day();

        let frac_of_earth = -geodetic_lon.to_degrees().rem_euclid(360.0) / 360.0;
        Time::<UTC>::from_year_month_day(y.into(), m, d, 0.5 + frac_of_earth).jd
    };
    let mut noon = noon - equation_of_time(Time::<UTC>::new(noon));
    while noon <= time.jd {
        noon += 1.0;
    }

    while noon > time.jd + 1.0 {
        noon -= 1.0;
    }
    Time::<UTC>::new(noon)
}

/// Table II of the paper cited on [`earth_nutation`], the IAU 2000B series.
///
/// Each term holds the integer multipliers of the five fundamental arguments.
/// Then it holds the coefficients `A`, `A'`, `B`, `B'`, `A''` and `B''`. The
/// column headings of the paper give the unit as 0.1 mas. The values are in
/// 0.1 microarcseconds, as the first term shows: it is the 17.2 arcsecond
/// principal nutation.
#[rustfmt::skip]
const IAU_2000B_TERMS: [([i8; 5], [f64; 6]); 78] = [
    ([0, 0, 0, 0, 1], [-172064161.0, -174666.0, 92052331.0, 9086.0, 33386.0, 15377.0]),
    ([0, 0, 2, -2, 2], [-13170906.0, -1675.0, 5730336.0, -3015.0, -13696.0, -4587.0]),
    ([0, 0, 2, 0, 2], [-2276413.0, -234.0, 978459.0, -485.0, 2796.0, 1374.0]),
    ([0, 0, 0, 0, 2], [2074554.0, 207.0, -897492.0, 470.0, -698.0, -291.0]),
    ([0, 1, 0, 0, 0], [1475877.0, -3633.0, 73871.0, -184.0, 11817.0, -1924.0]),
    ([0, 1, 2, -2, 2], [-516821.0, 1226.0, 224386.0, -677.0, -524.0, -174.0]),
    ([1, 0, 0, 0, 0], [711159.0, 73.0, -6750.0, 0.0, -872.0, 358.0]),
    ([0, 0, 2, 0, 1], [-387298.0, -367.0, 200728.0, 18.0, 380.0, 318.0]),
    ([1, 0, 2, 0, 2], [-301461.0, -36.0, 129025.0, -63.0, 816.0, 367.0]),
    ([0, -1, 2, -2, 2], [215829.0, -494.0, -95929.0, 299.0, 111.0, 132.0]),
    ([0, 0, 2, -2, 1], [128227.0, 137.0, -68982.0, -9.0, 181.0, 39.0]),
    ([-1, 0, 2, 0, 2], [123457.0, 11.0, -53311.0, 32.0, 19.0, -4.0]),
    ([-1, 0, 0, 2, 0], [156994.0, 10.0, -1235.0, 0.0, -168.0, 82.0]),
    ([1, 0, 0, 0, 1], [63110.0, 63.0, -33228.0, 0.0, 27.0, -9.0]),
    ([-1, 0, 0, 0, 1], [-57976.0, -63.0, 31429.0, 0.0, -189.0, -75.0]),
    ([-1, 0, 2, 2, 2], [-59641.0, -11.0, 25543.0, -11.0, 149.0, 66.0]),
    ([1, 0, 2, 0, 1], [-51613.0, -42.0, 26366.0, 0.0, 129.0, 78.0]),
    ([-2, 0, 2, 0, 1], [45893.0, 50.0, -24236.0, -10.0, 31.0, 20.0]),
    ([0, 0, 0, 2, 0], [63384.0, 11.0, -1220.0, 0.0, -150.0, 29.0]),
    ([0, 0, 2, 2, 2], [-38571.0, -1.0, 16452.0, -11.0, 158.0, 68.0]),
    ([-2, 0, 0, 2, 0], [-47722.0, 0.0, 477.0, 0.0, -18.0, -25.0]),
    ([2, 0, 2, 0, 2], [-31046.0, -1.0, 13238.0, -11.0, 131.0, 59.0]),
    ([1, 0, 2, -2, 2], [28593.0, 0.0, -12338.0, 10.0, -1.0, -3.0]),
    ([-1, 0, 2, 0, 1], [20441.0, 21.0, -10758.0, 0.0, 10.0, -3.0]),
    ([2, 0, 0, 0, 0], [29243.0, 0.0, -609.0, 0.0, -74.0, 13.0]),
    ([0, 0, 2, 0, 0], [25887.0, 0.0, -550.0, 0.0, -66.0, 11.0]),
    ([0, 1, 0, 0, 1], [-14053.0, -25.0, 8551.0, -2.0, 79.0, -45.0]),
    ([-1, 0, 0, 2, 1], [15164.0, 10.0, -8001.0, 0.0, 11.0, -1.0]),
    ([0, 2, 2, -2, 2], [-15794.0, 72.0, 6850.0, -42.0, -16.0, -5.0]),
    ([0, 0, -2, 2, 0], [21783.0, 0.0, -167.0, 0.0, 13.0, 13.0]),
    ([1, 0, 0, -2, 1], [-12873.0, -10.0, 6953.0, 0.0, -37.0, -14.0]),
    ([0, -1, 0, 0, 1], [-12654.0, 11.0, 6415.0, 0.0, 63.0, 26.0]),
    ([-1, 0, 2, 2, 1], [-10204.0, 0.0, 5222.0, 0.0, 25.0, 15.0]),
    ([0, 2, 0, 0, 0], [16707.0, -85.0, 168.0, -1.0, -10.0, 10.0]),
    ([1, 0, 2, 2, 2], [-7691.0, 0.0, 3268.0, 0.0, 44.0, 19.0]),
    ([-2, 0, 2, 0, 0], [-11024.0, 0.0, 104.0, 0.0, -14.0, 2.0]),
    ([0, 1, 2, 0, 2], [7566.0, -21.0, -3250.0, 0.0, -11.0, -5.0]),
    ([0, 0, 2, 2, 1], [-6637.0, -11.0, 3353.0, 0.0, 25.0, 14.0]),
    ([0, -1, 2, 0, 2], [-7141.0, 21.0, 3070.0, 0.0, 8.0, 4.0]),
    ([0, 0, 0, 2, 1], [-6302.0, -11.0, 3272.0, 0.0, 2.0, 4.0]),
    ([1, 0, 2, -2, 1], [5800.0, 10.0, -3045.0, 0.0, 2.0, -1.0]),
    ([2, 0, 2, -2, 2], [6443.0, 0.0, -2768.0, 0.0, -7.0, -4.0]),
    ([-2, 0, 0, 2, 1], [-5774.0, -11.0, 3041.0, 0.0, -15.0, -5.0]),
    ([2, 0, 2, 0, 1], [-5350.0, 0.0, 2695.0, 0.0, 21.0, 12.0]),
    ([0, -1, 2, -2, 1], [-4752.0, -11.0, 2719.0, 0.0, -3.0, -3.0]),
    ([0, 0, 0, -2, 1], [-4940.0, -11.0, 2720.0, 0.0, -21.0, -9.0]),
    ([-1, -1, 0, 2, 0], [7350.0, 0.0, -51.0, 0.0, -8.0, 4.0]),
    ([2, 0, 0, -2, 1], [4065.0, 0.0, -2206.0, 0.0, 6.0, 1.0]),
    ([1, 0, 0, 2, 0], [6579.0, 0.0, -199.0, 0.0, -24.0, 2.0]),
    ([0, 1, 2, -2, 1], [3579.0, 0.0, -1900.0, 0.0, 5.0, 1.0]),
    ([1, -1, 0, 0, 0], [4725.0, 0.0, -41.0, 0.0, -6.0, 3.0]),
    ([-2, 0, 2, 0, 2], [-3075.0, 0.0, 1313.0, 0.0, -2.0, -1.0]),
    ([3, 0, 2, 0, 2], [-2904.0, 0.0, 1233.0, 0.0, 15.0, 7.0]),
    ([0, -1, 0, 2, 0], [4348.0, 0.0, -81.0, 0.0, -10.0, 2.0]),
    ([1, -1, 2, 0, 2], [-2878.0, 0.0, 1232.0, 0.0, 8.0, 4.0]),
    ([0, 0, 0, 1, 0], [-4230.0, 0.0, -20.0, 0.0, 5.0, -2.0]),
    ([-1, -1, 2, 2, 2], [-2819.0, 0.0, 1207.0, 0.0, 7.0, 3.0]),
    ([-1, 0, 2, 0, 0], [-4056.0, 0.0, 40.0, 0.0, 5.0, -2.0]),
    ([0, -1, 2, 2, 2], [-2647.0, 0.0, 1129.0, 0.0, 11.0, 5.0]),
    ([-2, 0, 0, 0, 1], [-2294.0, 0.0, 1266.0, 0.0, -10.0, -4.0]),
    ([1, 1, 2, 0, 2], [2481.0, 0.0, -1062.0, 0.0, -7.0, -3.0]),
    ([2, 0, 0, 0, 1], [2179.0, 0.0, -1129.0, 0.0, -2.0, -2.0]),
    ([-1, 1, 0, 1, 0], [3276.0, 0.0, -9.0, 0.0, 1.0, 0.0]),
    ([1, 1, 0, 0, 0], [-3389.0, 0.0, 35.0, 0.0, 5.0, -2.0]),
    ([1, 0, 2, 0, 0], [3339.0, 0.0, -107.0, 0.0, -13.0, 1.0]),
    ([-1, 0, 2, -2, 1], [-1987.0, 0.0, 1073.0, 0.0, -6.0, -2.0]),
    ([1, 0, 0, 0, 2], [-1981.0, 0.0, 854.0, 0.0, 0.0, 0.0]),
    ([-1, 0, 0, 1, 0], [4026.0, 0.0, -553.0, 0.0, -353.0, -139.0]),
    ([0, 0, 2, 1, 2], [1660.0, 0.0, -710.0, 0.0, -5.0, -2.0]),
    ([-1, 0, 2, 4, 2], [-1521.0, 0.0, 647.0, 0.0, 9.0, 4.0]),
    ([-1, 1, 0, 1, 1], [1314.0, 0.0, -700.0, 0.0, 0.0, 0.0]),
    ([0, -2, 2, -2, 1], [-1283.0, 0.0, 672.0, 0.0, 0.0, 0.0]),
    ([1, 0, 2, 2, 1], [-1331.0, 0.0, 663.0, 0.0, 8.0, 4.0]),
    ([-2, 0, 2, 2, 2], [1383.0, 0.0, -594.0, 0.0, -2.0, -2.0]),
    ([-1, 0, 0, 0, 2], [1405.0, 0.0, -610.0, 0.0, 4.0, 2.0]),
    ([1, 1, 2, -2, 2], [1290.0, 0.0, -556.0, 0.0, 0.0, 0.0]),
    ([-2, 0, 2, 4, 2], [-1214.0, 0.0, 518.0, 0.0, 5.0, 2.0]),
    ([-1, 0, 4, 0, 2], [1146.0, 0.0, -490.0, 0.0, -3.0, -1.0]),
];

#[cfg(test)]
mod tests {
    use super::*;

    /// Station state against the IAU 2006/2000A terrestrial to celestial
    /// rotation of ERFA `c2t06a`, given the same UT1 and no polar motion. The
    /// expected velocity is a central difference of that rotation.
    #[test]
    fn approx_earth_state_matches_erfa() {
        let time = Time::<TDB>::new(2_460_000.813_7);
        let state = approx_earth_pos_to_ecliptic(time, 0.5, 0.2, 0.0, Desig::Empty).unwrap();
        let state: State<Equatorial> = state.into_frame();
        let pos = Vector3::from(state.pos) * AU_KM;
        let vel = Vector3::from(state.vel) * AU_KM;

        let expected_pos = Vector3::new(855.696557046809, -5536.857745250836, 3037.984432834993);
        let expected_vel = Vector3::new(34884.80804517053, 5348.483691195725, -78.017775219905);
        let pos_err = (pos - expected_pos).norm();
        assert!(pos_err < 1e-4, "position error {pos_err} km");
        // km / day, 0.1 m/s is 8.64 km / day.
        let vel_err = (vel - expected_vel).norm();
        assert!(vel_err < 8.64, "velocity error {vel_err} km/day");
    }

    /// Frame bias against the bias matrix of ERFA `bp06`. The reference vectors
    /// are the mean J2000 equinox and pole in the ICRF.
    #[test]
    fn frame_bias_matches_erfa() {
        let bias = frame_bias();
        let equinox = Vector3::new(
            0.999_999_999_999_994_1,
            -7.078_368_960_971_559e-8,
            8.056_213_977_613_186e-8,
        );
        let pole = Vector3::new(
            -8.056_214_211_620_057e-8,
            -3.305_943_169_468_331e-8,
            0.999_999_999_999_996_2,
        );
        let uas = (1e-6 / 3600.0_f64).to_radians();
        assert!((bias * Vector3::x() - equinox).norm() < uas);
        assert!((bias * Vector3::z() - pole).norm() < uas);
    }

    /// Geodetic coordinates round trip, at the poles and the equator too.
    #[test]
    fn geodetic_round_trip() {
        for lat in [-90.0_f64, -89.999, -45.0, 0.0, 30.0, 89.999, 90.0] {
            for height in [-0.4, 0.0, 3.0, 400.0] {
                let [x, y, z] = geodetic_lat_lon_to_ecef(lat.to_radians(), 0.3, height);
                let (lat_out, lon_out, height_out) = ecef_to_geodetic_lat_lon(x, y, z);
                assert!((lat_out.to_degrees() - lat).abs() < 1e-9, "{lat} {height}");
                assert!(
                    (height_out - height).abs() < 1e-8,
                    "{lat} {height}: {height_out}"
                );
                if lat.abs() < 90.0 {
                    assert!((lon_out - 0.3).abs() < 1e-9, "{lat} {height}");
                }
            }
        }
    }

    /// The equation of time is apparent minus mean solar time. In mid February
    /// the Sun crosses the meridian about 14 minutes after mean noon. In early
    /// November it crosses about 16 minutes before mean noon.
    #[test]
    fn equation_of_time_sign() {
        let february = equation_of_time(Time::from_year_month_day(2025, 2, 11, 0.5)) * 1440.0;
        let november = equation_of_time(Time::from_year_month_day(2025, 11, 3, 0.5)) * 1440.0;
        assert!((february + 14.2).abs() < 0.5, "{february} min");
        assert!((november - 16.4).abs() < 0.5, "{november} min");
    }

    /// Precession against the IAU 2006 precession matrix of ERFA `bp06`, at 1800,
    /// 2025 and 2200. The matrix does not include the frame bias. The reference
    /// vectors are the mean equinox and mean pole of date in J2000.
    #[test]
    fn precession_matches_erfa() {
        let cases = [
            (
                2_378_496.5,
                [
                    0.998_812_530_302_113_1,
                    0.044_675_177_146_083_65,
                    0.019_433_421_172_203_14,
                ],
                [
                    -0.019_433_425_627_861_423,
                    -0.000_434_254_022_659_310_6,
                    0.999_811_058_846_525_3,
                ],
            ),
            (
                2_460_676.5,
                [
                    0.999_981_422_028_832_3,
                    -0.005_590_636_237_461_226,
                    -0.002_429_070_533_004_523,
                ],
                [
                    0.002_429_070_360_083_325,
                    -6.821_017_966_381_851_5e-6,
                    0.999_997_049_780_978,
                ],
            ),
            (
                2_524_593.5,
                [
                    0.998_810_441_453_349_1,
                    -0.044_729_088_555_939_39,
                    -0.019_416_762_879_545_41,
                ],
                [
                    0.019_416_758_379_933_494,
                    -0.000_434_605_984_314_185_7,
                    0.999_811_382_517_549_5,
                ],
            ),
        ];
        for (jd, equinox, pole) in cases {
            let rot = earth_precession_rotation(Time::new(jd)).rotation;
            let equinox_err = (rot * Vector3::x() - Vector3::from(equinox)).norm();
            let pole_err = (rot * Vector3::z() - Vector3::from(pole)).norm();
            let mas = (1e-3 / 3600.0_f64).to_radians();
            assert!(equinox_err < 0.01 * mas, "{jd}: equinox {equinox_err:e}");
            assert!(pole_err < 0.01 * mas, "{jd}: pole {pole_err:e}");
        }
    }

    /// Nutation against the full IAU 2000A model of ERFA `nut00a`, over 1995 to 2050
    /// where the IAU 2000B paper gives the abridged model as within 1 mas in
    /// `dpsi sin(eps)` and in `deps`.
    #[test]
    fn nutation_matches_iau2000a() {
        let mas = (1.0_f64 / 3.6e6).to_radians();
        let sin_eps = earth_obliquity(Time::j2000()).sin();
        for (jd, dpsi_ref, deps_ref) in [
            (
                2_449_718.5,
                5.913_944_422_664_329e-5,
                -3.644_521_126_947_990_5e-5,
            ),
            (
                2_455_000.5,
                6.778_126_399_825_617e-5,
                2.149_933_938_645_781_2e-5,
            ),
            (
                2_460_676.5,
                9.569_537_765_382_347e-7,
                4.122_789_843_289_700_4e-5,
            ),
            (
                2_469_807.5,
                7.355_346_965_280_618e-5,
                -2.583_921_583_470_474_8e-5,
            ),
        ] {
            let (dpsi, deps) = earth_nutation(Time::new(jd));
            assert!(
                (dpsi - dpsi_ref).abs() * sin_eps < mas,
                "{jd}: dpsi {dpsi:e}"
            );
            assert!((deps - deps_ref).abs() < mas, "{jd}: deps {deps:e}");
        }
    }

    /// The TEME rotation is a proper rotation.
    #[test]
    fn teme_is_a_rotation() {
        let rot = *teme_frame(Time::j2000()).rotation.matrix();
        assert!((rot * rot.transpose() - Matrix3::identity()).norm() < 1e-14);
        assert!((rot.determinant() - 1.0).abs() < 1e-14);
    }

    /// The rotation rate is the time derivative of the rotation.
    #[test]
    fn approx_earth_frame_rate_is_derivative() {
        let jd = 2_433_000.3;
        let h = 1e-4;
        let frame = approx_earth_frame(Time::new(jd));
        let rotation_at = |jd| *approx_earth_frame(Time::new(jd)).rotation.matrix();
        let numeric = (rotation_at(jd + h) - rotation_at(jd - h)) / (2.0 * h);
        let err = (frame.rotation_rate.unwrap() - numeric).abs().max();
        assert!(err < 1e-4, "rotation rate error {err:e}");
    }

    /// Delta T against the historical values tabulated by Espenak and Meeus,
    /// which the polynomials were fit to.
    #[test]
    fn delta_t_matches_historical_values() {
        for (year, expected) in [
            (1700.0, 8.8),
            (1750.0, 13.4),
            (1800.0, 13.7),
            (1850.0, 7.1),
            (1900.0, -2.7),
            (1950.0, 29.1),
            (1960.0, 33.2),
            (1970.0, 40.2),
        ] {
            let time = Time::<TDB>::new(2_451_544.5 + (year - 2000.0) * 365.25);
            let delta_t = approx_delta_t(time);
            assert!((delta_t - expected).abs() < 0.3, "{year}: {delta_t} s");
        }
    }

    /// The polynomial pieces meet at their boundaries.
    #[test]
    fn delta_t_is_continuous() {
        for year in [
            -500.0, 500.0, 1600.0, 1700.0, 1800.0, 1860.0, 1900.0, 1920.0, 1941.0, 1961.0, 1986.0,
            2005.0, 2050.0, 2150.0,
        ] {
            let jd = 2_451_544.5 + (year - 2000.0) * 365.25;
            let jump = approx_delta_t(Time::new(jd + 1e-6)) - approx_delta_t(Time::new(jd - 1e-6));
            assert!(jump.abs() < 0.3, "{year}: jump of {jump} s");
        }
    }

    /// Before 1972 UT1 comes from Delta T, afterwards from UTC, and the two
    /// agree where they meet.
    #[test]
    fn ut1_is_continuous_at_1972() {
        let jd = 2_441_317.5 + 42.184 / 86400.0;
        let jump =
            (approx_ut1(Time::new(jd + 1e-4)) - approx_ut1(Time::new(jd - 1e-4)) - 2e-4) * 86400.0;
        assert!(jump.abs() < 0.2, "jump of {jump} s");
    }
}
