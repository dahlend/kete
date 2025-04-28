//! Python Frame of reference support
use kete_core::frames::*;
use pyo3::prelude::*;

/// Defined inertial frames supported by the python side of kete
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
pub fn wgs_lat_lon_to_ecef(lat: f64, lon: f64, h: f64) -> (f64, f64, f64) {
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
    calc_obliquity(time).to_degrees()
}

/// Calculate rotation matrix which transforms a vector from the J2000 Equatorial
/// frame to the desired epoch.
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
///     # Convert to a Vector if you want, keep in mind this is no longer an
///     # equatorial vector as defined by kete, as it would need to be the J2000
///     # epoch under the kete definition.
///     new_vec = kete.Vector(new_vec, frame=kete.Frames.Undefined)
///
/// Parameters
/// ----------
/// tdb_time:
///     Time in TDB scaled Julian Days.
#[pyfunction]
#[pyo3(name = "earth_precession_rotation")]
pub fn calc_earth_precession(time: f64) -> Vec<Vec<f64>> {
    earth_precession_rotation(time)
        .matrix()
        .column_iter()
        .map(|x| x.iter().cloned().collect())
        .collect()
}
