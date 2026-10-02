//! Python Frame of reference support
use kete_core::frames::{
    approx_earth_pos_to_ecliptic, approx_solar_noon, approx_sun_dec, earth_obliquity,
    earth_precession_rotation, ecef_to_geodetic_lat_lon, equation_of_time,
    geodetic_lat_lon_to_ecef, geodetic_lat_to_geocentric, next_sunset_sunrise,
};
use pyo3::prelude::*;

use crate::{state::PyState, time::PyTime};

/// Defined inertial frames supported by the python side of kete.
///
/// All vectors and states are defined by these coordinate frames.
///
/// Coordinate frames are defined to be equivalent to the J2000 frames used by the
/// JPL Horizons system and SPICE.
///
#[pyclass(frozen, eq, eq_int, name = "Frames", module = "kete", from_py_object)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PyFrames {
    /// Ecliptic Frame
    Ecliptic,
    /// Equatorial Frame
    Equatorial,
    /// Galactic Frame
    Galactic,
    /// FK4 Frame
    FK4,
}

/// Compute a ECEF position from WCS84 Geodetic latitude/longitude/height.
///
/// This returns the X/Y/Z coordinates in km from the geocenter of the Earth.
///
/// Parameters
/// ----------
/// lat :
///     Latitude in degrees.
/// lon :
///     Longitude in degrees.
/// h :
///     Height above the surface of the Earth in km.
#[pyfunction]
#[pyo3(name = "wgs_lat_lon_to_ecef")]
pub fn wgs_lat_lon_to_ecef(lat: f64, lon: f64, h: f64) -> [f64; 3] {
    geodetic_lat_lon_to_ecef(lat.to_radians(), lon.to_radians(), h)
}

/// Compute geocentric latitude from geodetic latitude and height.
///
/// Inputs are in degrees and km.
///
/// Parameters
/// ----------
/// lat :
///     Geodetic Latitude in degrees.
/// h :
///     Height above the surface of the Earth in km from the WGS ellipse.
#[pyfunction]
#[pyo3(name = "geodetic_lat_to_geocentric")]
pub fn geodetic_lat_to_geocentric_py(lat: f64, h: f64) -> f64 {
    geodetic_lat_to_geocentric(lat.to_radians(), h).to_degrees()
}

/// Compute WCS84 Geodetic latitude/longitude/height from a ECEF position.
///
/// This returns the lat, lon, and height from the WGS84 oblate Earth.
///
/// Parameters
/// ----------
/// x :
///     ECEF x position in km.
/// y :
///     ECEF y position in km.
/// z :
///     ECEF z position in km.
#[pyfunction]
#[pyo3(name = "ecef_to_wgs_lat_lon")]
pub fn ecef_to_wgs_lat_lon(x: f64, y: f64, z: f64) -> (f64, f64, f64) {
    let (lat, lon, alt) = ecef_to_geodetic_lat_lon(x, y, z);
    (lat.to_degrees(), lon.to_degrees(), alt)
}

/// Compute the mean obliquity of the ecliptic of date.
///
/// The expression is the IAU 2006 obliquity, equation (39) of the paper cited
/// on :func:`earth_precession_rotation`. Use it within several centuries of
/// J2000.
///
/// Parameters
/// ----------
/// time : float
///   Time in TDB scaled Julian Days.
///
/// Returns
/// -------
/// float
///   Mean obliquity in degrees.
#[pyfunction]
#[pyo3(name = "compute_obliquity")]
pub fn calc_obliquity_py(time: f64) -> f64 {
    earth_obliquity(time.into()).to_degrees()
}

/// Compute the precession rotation from the J2000 epoch to a date.
///
/// The matrix transforms a vector in the J2000 Equatorial frame to the mean
/// equator and equinox of the date. The equinox precesses by about 50
/// arcseconds per year, which is about 20 arcminutes from 2000 to 2025. The
/// matrix does not include the frame bias.
///
/// The angles are the P03 precession angles of equation (40), which the IAU
/// adopted as the IAU 2006 precession. Use them within a few centuries of
/// 2000. The source is:
///
/// .. code-block:: text
///
///     "Expressions for IAU 2000 precession quantities"
///     Capitaine, N. ; Wallace, P. T. ; Chapront, J.
///     Astronomy and Astrophysics, v.412, p.567-586 (2003)
///     doi:10.1051/0004-6361:20031539
///
/// This paper defines the IAU 1976 model, which JPL Horizons uses. It
/// discusses the same angles:
///
/// .. code-block:: text
///
///     "Precession matrix based on IAU (1976) system of astronomical constants."
///     Lieske, J. H.
///     Astronomy and Astrophysics, vol. 73, no. 3, Mar. 1979, p. 282-284.
///
/// Parameters
/// ----------
/// time : Time or float
///   Time, as a :class:`Time` or in TDB scaled Julian Days.
///
/// Returns
/// -------
/// list of list of float
///   The 3x3 rotation matrix.
///
/// Examples
/// --------
/// Convert a vector in the Equatorial J2000 frame to the mean equator and
/// equinox of 2025. The result is not an Equatorial vector as kete defines
/// it, because kete defines that frame at the J2000 epoch.
///
/// .. code-block:: python
///
///     import kete
///     import numpy as np
///
///     jd = kete.Time.from_ymd(2025, 1, 1).jd
///     rotation = np.array(kete.conversion.earth_precession_rotation(jd))
///     new_vec = rotation @ kete.Vector.from_ra_dec(20, 10)
#[pyfunction]
#[pyo3(name = "earth_precession_rotation")]
pub fn calc_earth_precession(time: PyTime) -> Vec<Vec<f64>> {
    earth_precession_rotation(time.into())
        .rotation_to_equatorial()
        .unwrap()
        .matrix()
        .column_iter()
        .map(|x| x.iter().cloned().collect())
        .collect()
}

/// Compute an approximation for the time of the next solar noon at a given geodetic longitude.
///
#[pyfunction]
#[pyo3(name = "next_solar_noon")]
pub fn solar_noon_py(time: PyTime, geodetic_lon: f64) -> f64 {
    approx_solar_noon(time.0.utc(), geodetic_lon.to_radians())
        .tdb()
        .jd()
}

/// Compute the approximate equation of time.
///
/// The equation of time is the apparent solar time minus the mean solar time.
/// It is positive when the Sun crosses the meridian before mean noon.
///
/// Parameters
/// ----------
/// time : Time or float
///   Time, as a :class:`Time` or in TDB scaled Julian Days.
///
/// Returns
/// -------
/// float
///   Equation of time in days.
#[pyfunction]
#[pyo3(name = "equation_of_time")]
pub fn equation_of_time_py(time: PyTime) -> f64 {
    equation_of_time(time.0.utc())
}

/// Compute the approximate solar dec.
#[pyfunction]
#[pyo3(name = "approx_solar_dec")]
pub fn approx_solar_dec_py(time: PyTime) -> f64 {
    approx_sun_dec(time.0.utc())
}

/// Compute an approximation for the time of the next sunset and sunrise at a given
/// geodetic latitude and longitude.
///
#[pyfunction]
#[pyo3(name = "next_sunset_sunrise")]
pub fn next_sunset_sunrise_py(
    time: PyTime,
    geodetic_lat: f64,
    geodetic_lon: f64,
) -> (PyTime, PyTime) {
    let (set, rise) = next_sunset_sunrise(
        geodetic_lat.to_radians(),
        geodetic_lon.to_radians(),
        time.0.utc(),
    );
    (PyTime(set.tdb()), PyTime(rise.tdb()))
}

/// Compute the approximate state of a location on Earth in the Ecliptic frame.
///
/// :func:`kete.spice.earth_pos_to_ecliptic` should be preferred for all modern
/// dates. *This function is only an approximation*.
///
/// This should be used when the desired date is outside the Earth orientation
/// PCK kernels, which begin in 1962.
///
/// The Earth's orientation includes precession, the largest nutation terms, and
/// the Earth's rotation, which also sets the velocity. Polar motion is not
/// included. From 1972 the rotation uses UTC in place of UT1, which differ by
/// less than a second. Before 1972 it uses a model of Delta T, the difference
/// between TT and UT1, which is poorly known before 1600.
///
/// Parameters
/// ----------
/// jd:
///     Julian time (TDB) of the desired state.
/// geodetic_lat:
///     Latitude on Earth's surface in degrees.
/// geodetic_lon:
///     Latitude on Earth's surface in degrees.
/// height:
///     Height of the observer above the surface of the Earth in km.
/// name :
///     Optional name of the position on Earth.
#[pyfunction]
#[pyo3(name = "approx_earth_pos_to_ecliptic", signature = (jd, geodetic_lat, geodetic_lon, height, name=None))]
pub fn approx_earth_pos_to_ecliptic_py(
    jd: PyTime,
    geodetic_lat: f64,
    geodetic_lon: f64,
    height: f64,
    name: Option<String>,
) -> PyResult<PyState> {
    let desig = name
        .map(kete_core::desigs::Desig::Name)
        .unwrap_or(kete_core::desigs::Desig::Empty);
    let time = jd.0;
    let state: PyState = approx_earth_pos_to_ecliptic(
        time,
        geodetic_lat.to_radians(),
        geodetic_lon.to_radians(),
        height,
        desig,
    )?
    .into();
    state.change_center(crate::desigs::NaifIDLike::Int(10))
}
