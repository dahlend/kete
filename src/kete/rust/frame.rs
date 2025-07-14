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
#[pyclass(frozen, eq, eq_int, name = "Frames", module = "kete")]
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

/// Calculate the obliquity angle of the Earth at the specified time.
///
/// This is only valid for several centuries near J2000.
///
/// The equation is from the 2010 Astronomical Almanac.
///
/// Parameters
/// ----------
/// time:
///     Calculate the obliquity angle of the Earth at the specified time.
#[pyfunction]
#[pyo3(name = "compute_obliquity")]
pub fn calc_obliquity_py(time: f64) -> f64 {
    earth_obliquity(time.into()).to_degrees()
}

/// Calculate how far the Earth's north pole has precessed from the J2000 epoch.
///
/// Earth's north pole precesses at a rate of about 50 arcseconds per year.
/// This means there was an approximately 20 arcminute rotation of the Equatorial
/// axis from the year 2000 to 2025.
///
/// This calculates the rotation matrix which transforms a vector from the J2000
/// Equatorial frame to the desired epoch.
///
/// This implementation is valid for around 200 years on either side of 2000 to
/// within sub micro-arcsecond accuracy.
///
/// This function is an implementation equation (21) from this paper:
///
/// .. code-block:: text
///
///     "Expressions for IAU 2000 precession quantities"
///     Capitaine, N. ; Wallace, P. T. ; Chapront, J.
///     Astronomy and Astrophysics, v.412, p.567-586 (2003)
///
/// It is recommended to first look at the following paper, as it provides useful
/// discussion to help understand the above model. This defines the model used
/// by JPL Horizons:
///
/// .. code-block:: text
///
///     "Precession matrix based on IAU (1976) system of astronomical constants."
///     Lieske, J. H.
///     Astronomy and Astrophysics, vol. 73, no. 3, Mar. 1979, p. 282-284.
///
/// The IAU 2000 model paper improves accuracy by approximately ~300 mas/century over
/// the IAU 1976 model.
///
/// Vectors in the Equatorial J2000 frame can be converted to the Equatorial frame
/// at the time of the epoch desired:
///
/// .. code-block:: python
///
///     import kete
///     import numpy as np
///
///     jd = kete.Time.from_ymd(2025, 1, 1).jd
///     rotation = np.array(kete.conversion.earth_precession_rotation(jd))
///
///     new_vec = rotation @ kete.Vector.from_ra_dec(20, 10)
///
///     # keep in mind this is no longer an equatorial vector as defined by kete,
///     # as it would need to be the J2000 epoch under the kete definition.
///
/// Parameters
/// ----------
/// tdb_time:
///     Time in TDB scaled Julian Days.
#[pyfunction]
#[pyo3(name = "earth_precession_rotation")]
pub fn calc_earth_precession(time: f64) -> Vec<Vec<f64>> {
    earth_precession_rotation(time.into())
        .rotations_to_equatorial()
        .unwrap()
        .0
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
        .jd
}

/// Compute the equation of time at a given time.
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

/// Compute the approximate position of a location on Earth in the Ecliptic frame.
///
/// :func:`kete.spice.earth_pos_to_ecliptic` should be preferred for all modern
/// dates. *This function is only an approximation*.
///
/// This should be used when the desired date is before ~1970, when there are no
/// SPICE SPK kernels for the Earth available.
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
