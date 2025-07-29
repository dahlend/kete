//! Conversion tools to and from WGS84 coordinate system
// BSD 3-Clause License
//
// Copyright (c) 2025, Dar Dahlen
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

use nalgebra::{Rotation3, Vector3};

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
pub fn prime_vert_radius(geodetic_lat: f64) -> f64 {
    EARTH_A / (1.0 - EARTH_E2 * geodetic_lat.sin().powi(2)).sqrt()
}

/// Compute earths geocentric radius at the specified latitude in km.
pub fn geocentric_radius(geodetic_lat: f64) -> f64 {
    let (sin, cos) = geodetic_lat.sin_cos();
    let a_cos = EARTH_A * cos;
    let b_sin = EARTH_B * sin;
    (((EARTH_A * a_cos).powi(2) + (EARTH_B * b_sin).powi(2)) / (a_cos.powi(2) + b_sin.powi(2)))
        .sqrt()
}

/// Compute geodetic lat/lon/height in radians/km from ECEF position in km.
pub fn ecef_to_geodetic_lat_lon(x: f64, y: f64, z: f64) -> (f64, f64, f64) {
    let longitude = f64::atan2(y, x);
    let p = (x * x + y * y).sqrt();
    let geocen_lat = f64::atan2(p, z);

    // initial guess, and iterate.
    let mut geodetic_lat = geocen_lat;
    let mut h = 0.0;
    // this usually converges in only 1-2 iterations, but to reduce CPU branching
    // don't bother with a convergence check.
    for _ in 0..5 {
        let n = prime_vert_radius(geodetic_lat);
        h = p / geodetic_lat.cos() - n;
        geodetic_lat = f64::atan(z / p / (1.0 - EARTH_E2 * n / (n + h)));
    }

    (geodetic_lat, longitude, h)
}

/// Compute geocentric latitude from geodetic lat/height in radians/km .
pub fn geodetic_lat_to_geocentric(geodetic_lat: f64, h: f64) -> f64 {
    let n = prime_vert_radius(geodetic_lat);
    ((1.0 - EARTH_E2 * n / (n + h)) * geodetic_lat.tan()).atan()
}

/// Compute the ECEF X/Y/Z position in km from geodetic lat/lon/height in radians/km
pub fn geodetic_lat_lon_to_ecef(geodetic_lat: f64, geodetic_lon: f64, h: f64) -> [f64; 3] {
    let n = prime_vert_radius(geodetic_lat);
    let (sin_gd_lat, cos_gd_lat) = geodetic_lat.sin_cos();
    let (sin_gd_lon, cos_gd_lon) = geodetic_lon.sin_cos();
    let x = (n + h) * cos_gd_lat * cos_gd_lon;
    let y = (n + h) * cos_gd_lat * sin_gd_lon;
    let z = ((1.0 - EARTH_E2) * n + h) * sin_gd_lat;
    [x, y, z]
}

/// Compute the angle of obliquity of Earth in radians.
///
/// This is only valid for several centuries near J2000.
///
/// The equation here is from the 2010 Astronomical Almanac.
///
pub fn earth_obliquity(jd: Time<TDB>) -> f64 {
    // centuries from j2000
    let c = (jd - Time::j2000()) / 365.25 / 100.0;
    (23.439279444444444
        + c * (-0.013010213611111
            + c * (-5.08611111111111e-08
                + c * (5.565e-07 - c * (1.6e-10 + -1.1777777777777779e-11 * c)))))
        .to_radians()
}

/// Approximation for the Earth Rotation Angle (ERA) at a given time.
///
/// The ERA is the angle between the Greenwich meridian and the vernal equinox,
/// the Equatorial J2000 X-axis.
///
pub fn earth_rotation_angle(time: Time<UTC>) -> f64 {
    // Note that second number is not j2000, its the j2000 value in UTC time.
    let dt = time.jd - 2451545.0;
    // this is very close to the UT1 conversion, the second value is the
    // number of sidereal days per solar day, (about 365.25 / 364.25)
    ((0.779057273264 + 1.0027379094 * dt) * 360.0).to_radians()
}

/// Rotation which transforms a vector from the J2000 Equatorial frame to the
/// desired epoch.
///
/// Earth's north pole precesses at a rate of about 50 arcseconds per year.
/// This means there was an approximately 20 arcminute rotation of the Equatorial
/// axis from the year 2000 to 2025.
///
/// This implementation is valid for around 200 years on either side of 2000 to
/// within sub micro-arcsecond accuracy.
///
/// This function is an implementation equation (21) from this paper:
///
/// > "Expressions for IAU 2000 precession quantities"  
/// > Capitaine, N. ; Wallace, P. T. ; Chapront, J.  
/// > Astronomy and Astrophysics, v.412, p.567-586 (2003)
///
/// It is recommended to first look at the following paper, as it provides useful
/// discussion to help understand the above model. This defines the model used
/// by JPL Horizons:
///
/// > "Precession matrix based on IAU (1976) system of astronomical constants."  
/// > Lieske, J. H.  
/// > Astronomy and Astrophysics, vol. 73, no. 3, Mar. 1979, p. 282-284.
///
/// The IAU 2000 model paper improves accuracy by approximately ~300 mas/century over
/// the 1976 model.
///
/// # Arguments
///
/// * `tdb_time` - Time in TDB scaled Julian Days.
///
#[inline(always)]
pub fn earth_precession_rotation(time: Time<TDB>) -> NonInertialFrame {
    // centuries since 2000
    let t = (time - Time::j2000()) / 36525.0;

    // angles as defined in the cited paper, equations (21)
    // Note that equation 45 is an even more developed model, which takes into
    // account frame bias in addition to simple precession, however more clarity
    // on the DE source and interpretation is probably required to take advantage
    // of this increased precision.
    let angle_c = -((2.5976176
        + (2306.0809506 + (0.3019015 + (0.0179663 + (-0.0000327 - 0.0000002 * t) * t) * t) * t)
            * t)
        / 3600.0)
        .to_radians();
    let angle_a = -((-2.5976176
        + (2306.0803226 + (1.094779 + (0.0182273 + (0.000047 - 0.0000003 * t) * t) * t) * t) * t)
        / 3600.0)
        .to_radians();
    let angle_b = ((2004.1917476
        + (-0.4269353 + (-0.0418251 + (-0.0000601 - 0.0000001 * t) * t) * t) * t)
        * t
        / 3600.0)
        .to_radians();
    let z_axis = Vector3::z_axis();
    let rotation = Rotation3::from_axis_angle(&z_axis, angle_a)
        * Rotation3::from_axis_angle(&Vector3::y_axis(), angle_b)
        * Rotation3::from_axis_angle(&z_axis, angle_c);

    NonInertialFrame::from_rotations(time, rotation, None, 1, 1000000000)
}

/// Compute the approximate position of a location on Earth in the Ecliptic frame.
///
/// This should be used when either SPICE is unavailable, or when the desired date
/// is before ~1970, when there are no SPICE SPK kernels for the Earth available.
///
/// This will be centered at the Earth, conversion to Sun centered
/// requires a computation of Earths position.
pub fn approx_earth_pos_to_ecliptic(
    time: Time<TDB>,
    geodetic_lat: f64,
    geodetic_lon: f64,
    height: f64,
    desig: Desig,
) -> KeteResult<State<Ecliptic>> {
    let pos: Vector3<f64> = geodetic_lat_lon_to_ecef(geodetic_lat, geodetic_lon, height).into();
    let era = earth_rotation_angle(time.utc());
    let rotation = Rotation3::from_axis_angle(&Vector3::z_axis(), era);
    let pos = rotation * pos / AU_KM;

    let rotation = earth_precession_rotation(time);

    let (pos, vel) = rotation.to_equatorial(pos, [0.0; 3].into())?;

    Ok(State::<Equatorial>::new(desig, time.tdb().jd, pos.into(), vel.into(), 399).into_frame())
}

/// Compute the next sunset and sunrise times for a given location.
///
/// This is approximate, but should be good to within a few minutes.
///
/// # Arguments
///    * `geodetic_lat` - Geodetic latitude in radians.
///    * `geodetic_lon` - Geodetic longitude in radians.
///    * `time` - Time in UTC scaled Julian Days.
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
pub fn approx_sun_dec(time: Time<UTC>) -> f64 {
    let obliquity = earth_obliquity(time.tdb());

    let time_since_j2000 = time - Time::j2000();
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

/// Approximate the Equation of Time at a given time.
///    <https://en.wikipedia.org/wiki/Equation_of_time>
///
/// This is the difference between the apparent solar time and the mean solar time.
///
/// Returned value is in days.
///
/// This is largely based off the approximation used by the USNO:
///     <https://aa.usno.navy.mil/faq/sun_approx>
///
pub fn equation_of_time(time: Time<UTC>) -> f64 {
    let time_since_j2000 = time - Time::j2000();
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

    0.0053 * mean_anom.sin() - 0.0069 * (2.0 * app_eclip_lon).sin()
}

/// Approximate the next local solar noon time for a given geodetic longitude.
///
pub fn approx_solar_noon(time: Time<UTC>, geodetic_lon: f64) -> Time<UTC> {
    // compute the next clock noon after the given time
    let noon = {
        let (y, m, d, _) = time.year_month_day();

        let frac_of_earth = -geodetic_lon.to_degrees().rem_euclid(360.0) / 360.0;
        Time::<UTC>::from_year_month_day(y, m, d, 0.5 + frac_of_earth).jd
    };
    let mut noon = noon + equation_of_time(Time::<UTC>::new(noon));
    while noon <= time.jd {
        noon += 1.0;
    }

    while noon > time.jd + 1.0 {
        noon -= 1.0;
    }
    Time::<UTC>::new(noon)
}
