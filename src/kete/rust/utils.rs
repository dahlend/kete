//! General purpose utility functions.

use kete_core::util::Degrees;
use pyo3::{exceptions::PyValueError, prelude::*};

use crate::maybe_vec::{MaybeVec, maybe_vec_to_pyobj};

/// Convert a Right Ascension in decimal degrees to an "hours minutes seconds" string.
///
/// The angle is wrapped into [0, 24) hours, and seconds are rounded to 3 decimals.
///
/// Parameters
/// ----------
/// ra:
///     Right Ascension in decimal degrees.
///
/// Raises
/// ------
/// ValueError
///     If a value is not finite.
#[pyfunction]
#[pyo3(name = "ra_degrees_to_hms")]
pub fn ra_degrees_to_hms_py(py: Python<'_>, ra: MaybeVec<f64>) -> PyResult<Py<PyAny>> {
    let (ra, was_vec): (Vec<_>, bool) = ra.into();
    let ra = ra
        .into_iter()
        .map(|ra| Degrees::from_degrees(ra).to_hms_str())
        .collect::<Result<Vec<_>, _>>()?;

    maybe_vec_to_pyobj(py, ra, was_vec)
}

/// Convert a declination in degrees to a "degrees arcminutes arcseconds" string.
///
/// Arcseconds are rounded to 2 decimals.
///
/// Parameters
/// ----------
/// dec:
///     Declination in decimal degrees.
///
/// Raises
/// ------
/// ValueError
///     If a value is not finite or is outside [-90, 90].
#[pyfunction]
#[pyo3(name = "dec_degrees_to_dms")]
pub fn dec_degrees_to_dms_py(py: Python<'_>, dec: MaybeVec<f64>) -> PyResult<Py<PyAny>> {
    let (dec, was_vec): (Vec<_>, bool) = dec.into();

    if dec.iter().any(|&d| !(-90.0..=90.0).contains(&d)) {
        return Err(PyErr::new::<PyValueError, _>(
            "Declination must be between -90 and 90 degrees",
        ));
    }

    let dec = dec
        .into_iter()
        .map(|dec| Degrees::from_degrees(dec).to_dms_str())
        .collect::<Result<Vec<_>, _>>()?;

    maybe_vec_to_pyobj(py, dec, was_vec)
}

/// Convert a declination from "degrees arcminutes arcseconds" string to degrees.
///
/// Terms may be separated by spaces, commas, colons, or semicolons. Missing
/// trailing terms are zero. The sign of the degrees applies to the whole angle.
///
/// Parameters
/// ----------
/// dec:
///     Declination in degrees-arcminutes-arcseconds.
///
/// Raises
/// ------
/// ValueError
///     If the string cannot be parsed, arcminutes or arcseconds are outside
///     [0, 60), or the declination is outside [-90, 90].
#[pyfunction]
#[pyo3(name = "dec_dms_to_degrees")]
pub fn dec_dms_to_degrees_py(py: Python<'_>, dec: MaybeVec<String>) -> PyResult<Py<PyAny>> {
    let (dec, was_vec): (Vec<_>, bool) = dec.into();
    let mut results = Vec::with_capacity(dec.len());

    for dms in dec {
        let deg = Degrees::try_from_dms_str(&dms)
            .map_err(|_| {
                PyErr::new::<PyValueError, _>(format!(
                    "Invalid declination format: '{dms}'. Expected 'degrees arcminutes arcseconds'.",
                ))
            })?
            .to_degrees();
        if !(-90.0..=90.0).contains(&deg) {
            return Err(PyErr::new::<PyValueError, _>(format!(
                "Declination '{dms}' must be between -90 and 90 degrees",
            )));
        }
        results.push(deg);
    }

    maybe_vec_to_pyobj(py, results, was_vec)
}

/// Convert a right ascension from "hours minutes seconds" string to degrees.
///
/// Terms may be separated by spaces, commas, colons, or semicolons. Missing
/// trailing terms are zero.
///
/// Parameters
/// ----------
/// ra:
///     Right ascension in hours-minutes-seconds.
///
/// Raises
/// ------
/// ValueError
///     If the string cannot be parsed, minutes or seconds are outside [0, 60),
///     or the right ascension is outside [0, 24) hours.
#[pyfunction]
#[pyo3(name = "ra_hms_to_degrees")]
pub fn ra_hms_to_degrees_py(py: Python<'_>, ra: MaybeVec<String>) -> PyResult<Py<PyAny>> {
    let (ra, was_vec): (Vec<_>, bool) = ra.into();
    let mut results = Vec::with_capacity(ra.len());

    for hms in ra {
        let deg = Degrees::try_from_hms_str(&hms)
            .map_err(|_| {
                PyErr::new::<PyValueError, _>(format!(
                    "Invalid right ascension format: '{hms}'. Expected 'hours minutes seconds'.",
                ))
            })?
            .to_degrees();
        if !(0.0..360.0).contains(&deg) {
            return Err(PyErr::new::<PyValueError, _>(format!(
                "Right ascension '{hms}' must be between 0 and 24 hours",
            )));
        }
        results.push(deg);
    }

    maybe_vec_to_pyobj(py, results, was_vec)
}
