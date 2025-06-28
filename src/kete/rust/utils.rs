//! General purpose utility functions.

use itertools::Itertools;
use kete_core::util::Degrees;
use pyo3::{exceptions::PyValueError, prelude::*};

use crate::maybe_vec::{MaybeVec, maybe_vec_to_pyobj};

/// Convert a Right Ascension in decimal degrees to an "hours minutes seconds" string.
///
/// Parameters
/// ----------
/// ra:
///     Right Ascension in decimal degrees.
#[pyfunction]
#[pyo3(name = "ra_degrees_to_hms")]
pub fn ra_degrees_to_hms_py(py: Python<'_>, ra: MaybeVec<f64>) -> PyResult<PyObject> {
    let (ra, was_vec): (Vec<_>, bool) = ra.into();
    let ra = ra
        .into_iter()
        .map(|ra| {
            let mut deg = Degrees::from_degrees(ra);
            let _ = deg.bound_to_360();
            deg.to_hms_str()
        })
        .collect_vec();

    maybe_vec_to_pyobj(py, ra, was_vec)
}

/// Convert a declination in degrees to a "degrees arcminutes arcseconds" string.
///
/// Parameters
/// ----------
/// dec:
///     Declination in decimal degrees.
#[pyfunction]
#[pyo3(name = "dec_degrees_to_dms")]
pub fn dec_degrees_to_dms_py(py: Python<'_>, dec: MaybeVec<f64>) -> PyResult<PyObject> {
    let (dec, was_vec): (Vec<_>, bool) = dec.into();

    if dec.iter().any(|&d| d.abs() > 90.0) {
        return Err(PyErr::new::<PyValueError, _>(
            "Declination must be between -90 and 90 degrees",
        ));
    }

    let dec = dec
        .into_iter()
        .map(|dec| {
            let mut deg = Degrees::from_degrees(dec);
            let _ = deg.bound_to_pm_180();
            deg.to_dms_str()
        })
        .collect_vec();

    maybe_vec_to_pyobj(py, dec, was_vec)
}

/// Convert a declination from "degrees arcminutes arcseconds" string to degrees.
///
/// This must be formatted with a space between the terms.
///
/// Parameters
/// ----------
/// dec:
///     Declination in degrees-arcminutes-arcseconds.
#[pyfunction]
#[pyo3(name = "dec_dms_to_degrees")]
pub fn dec_dms_to_degrees_py(py: Python<'_>, dec: MaybeVec<String>) -> PyResult<PyObject> {
    let (dec, was_vec): (Vec<_>, bool) = dec.into();
    let mut results = Vec::with_capacity(dec.len());

    for dms in dec {
        Degrees::try_from_dms_str(&dms)
            .map(|deg| {
                let mut deg = deg;
                let _ = deg.bound_to_pm_180();
                results.push(deg.to_degrees());
            })
            .map_err(|_| {
                PyErr::new::<PyValueError, _>(format!(
                    "Invalid declination format: '{dms}'. Expected 'degrees arcminutes arcseconds'.",
                ))
            })?;
    }

    maybe_vec_to_pyobj(py, results, was_vec)
}

///  Convert a right ascension from "hours minutes seconds" string to degrees.
///
/// This must be formatted with a space between the terms.
///
/// Parameters
/// ----------
/// ra:
///     Right ascension in hours-minutes-seconds.
#[pyfunction]
#[pyo3(name = "ra_hms_to_degrees")]
pub fn ra_hms_to_degrees_py(py: Python<'_>, ra: MaybeVec<String>) -> PyResult<PyObject> {
    let (ra, was_vec): (Vec<_>, bool) = ra.into();
    let mut results = Vec::with_capacity(ra.len());

    for hms in ra {
        Degrees::try_from_hms_str(&hms)
            .map(|deg| {
                let mut deg = deg;
                let _ = deg.bound_to_360();
                results.push(deg.to_degrees());
            })
            .map_err(|_| {
                PyErr::new::<PyValueError, _>(format!(
                    "Invalid right ascension format: '{hms}'. Expected 'hours minutes seconds'.",
                ))
            })?;
    }

    maybe_vec_to_pyobj(py, results, was_vec)
}
