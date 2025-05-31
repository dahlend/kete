//! General purpose utility functions.

use itertools::Itertools;
use kete_core::util::Degrees;
use pyo3::{exceptions::PyValueError, prelude::*};

/// Polymorphic support for a single value or a vector of values.
#[derive(Debug, FromPyObject, IntoPyObject)]
pub enum MaybeVec<T> {
    /// A single value of type T.
    Single(T),

    /// A vector of values of type T.
    Multiple(Vec<T>),
}

impl<T> From<MaybeVec<T>> for Vec<T> {
    fn from(maybe_vec: MaybeVec<T>) -> Self {
        match maybe_vec {
            MaybeVec::Single(value) => vec![value],
            MaybeVec::Multiple(vec) => vec,
        }
    }
}

impl<T: Clone> From<Vec<T>> for MaybeVec<T> {
    fn from(maybe_vec: Vec<T>) -> Self {
        if maybe_vec.len() == 1 {
            MaybeVec::Single(maybe_vec[0].clone())
        } else {
            MaybeVec::Multiple(maybe_vec)
        }
    }
}

/// Convert a Right Ascension in decimal degrees to an "hours minutes seconds" string.
///
/// Parameters
/// ----------
/// ra:
///     Right Ascension in decimal degrees.
#[pyfunction]
#[pyo3(name = "ra_degrees_to_hms")]
pub fn ra_degrees_to_hms_py(ra: MaybeVec<f64>) -> MaybeVec<String> {
    let ra: Vec<_> = ra.into();
    ra.into_iter()
        .map(|ra| {
            let mut deg = Degrees::from_degrees(ra);
            deg.bound_to_360();
            deg.to_hms_str()
        })
        .collect_vec()
        .into()
}

/// Convert a declination in degrees to a "degrees arcminutes arcseconds" string.
///
/// Parameters
/// ----------
/// dec:
///     Declination in decimal degrees.
#[pyfunction]
#[pyo3(name = "dec_degrees_to_dms")]
pub fn dec_degrees_to_dms_py(dec: MaybeVec<f64>) -> PyResult<MaybeVec<String>> {
    let dec: Vec<_> = dec.into();

    if dec.iter().any(|&d| d.abs() > 90.0) {
        return Err(PyErr::new::<PyValueError, _>(
            "Declination must be between -90 and 90 degrees",
        ));
    }
    Ok(dec
        .into_iter()
        .map(|dec| {
            let mut deg = Degrees::from_degrees(dec);
            deg.bound_to_pm_180();
            deg.to_dms_str()
        })
        .collect_vec()
        .into())
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
pub fn dec_dms_to_degrees_py(dec: MaybeVec<String>) -> PyResult<MaybeVec<f64>> {
    let dec: Vec<_> = dec.into();
    let mut results = Vec::with_capacity(dec.len());

    for dms in dec {
        Degrees::try_from_dms_str(&dms)
            .map(|deg| {
                let mut deg = deg;
                deg.bound_to_pm_180();
                results.push(deg.to_degrees());
            })
            .map_err(|_| {
                PyErr::new::<PyValueError, _>(format!(
                    "Invalid declination format: '{}'. Expected 'degrees arcminutes arcseconds'.",
                    dms
                ))
            })?;
    }

    Ok(results.into())
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
pub fn ra_hms_to_degrees_py(ra: MaybeVec<String>) -> PyResult<MaybeVec<f64>> {
    let ra: Vec<_> = ra.into();
    let mut results = Vec::with_capacity(ra.len());

    for hms in ra {
        Degrees::try_from_hms_str(&hms)
            .map(|deg| {
                let mut deg = deg;
                deg.bound_to_360();
                results.push(deg.to_degrees());
            })
            .map_err(|_| {
                PyErr::new::<PyValueError, _>(format!(
                    "Invalid right ascension format: '{}'. Expected 'hours minutes seconds'.",
                    hms
                ))
            })?;
    }

    Ok(results.into())
}
