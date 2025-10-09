//! Covariance matrix representation

use std::{collections::HashMap, fmt::Debug};

use crate::state::PyState;
use crate::time::PyTime;
use crate::{elements::PyCometElements, vector::VectorLike};
use kete_core::{errors::Error, io::FileIO};
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

/// Covariance uncertainty representation of an objects state.
#[pyclass(frozen, get_all, module = "kete")]
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Covariance {
    /// Designation of the object
    desig: String,

    /// Epoch of the covariance matrix fit.
    epoch: f64,

    /// Name and best estimate of the parameters in the fit.
    params: Vec<(String, f64)>,

    /// The covariance matrix, where the order of the array corresponds to the parameters.
    cov_matrix: Vec<Vec<f64>>,
}

impl FileIO for Covariance {}

#[pymethods]
impl Covariance {
    /// Create a new covariance object
    #[new]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        desig: String,
        epoch: PyTime,
        params: Vec<(String, f64)>,
        cov_matrix: Vec<Vec<f64>>,
    ) -> Self {
        Self {
            desig,
            epoch: epoch.jd(),
            params,
            cov_matrix,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "Covariance(desig={:?}, epoch={:?}, params={:?}, cov_matrix={:?})",
            self.desig, self.epoch, self.params, self.cov_matrix,
        )
    }

    /// Create a State object from the fit values.
    ///
    /// This looks for either cometary elements or cartesian elements in the parameters.
    /// Cometary Elements must contain the following complete set of keys:
    ///     ["eccentricity", "peri_dist", "peri_time", "lon_of_ascending", "peri_arg", "inclination"]
    /// Cartesian must contain the following complete set of keys:
    ///     ['x', 'y', 'z', 'vx', 'vy', 'vz']
    ///
    /// All units must be in degrees, AU, or AU/Day as appropriate.
    #[getter]
    pub fn state(&self) -> PyResult<PyState> {
        let epoch = self.epoch;
        let desig = self.desig.clone();
        let mut hash = HashMap::new();
        for (key, val) in self.params.iter() {
            let lower_key = key.to_lowercase();
            if hash.insert(lower_key, *val).is_some() {
                return Err(Error::IOError(format!(
                    "Repeat parameter {:?} present in covariance",
                    &key
                )))?;
            }
        }

        if hash.contains_key("eccentricity") {
            let eccentricity = *hash.get("eccentricity").ok_or(Error::ValueError(
                "Covariance missing 'eccentricity'".into(),
            ))?;
            let peri_dist = *hash
                .get("peri_dist")
                .ok_or(Error::ValueError("Covariance missing 'peri_dist'".into()))?;
            let peri_time = *hash
                .get("peri_time")
                .ok_or(Error::ValueError("Covariance missing 'peri_time'".into()))?;
            let lon_of_ascending = *hash.get("lon_of_ascending").ok_or(Error::ValueError(
                "Covariance missing 'lon_of_ascending'".into(),
            ))?;
            let peri_arg = *hash
                .get("peri_arg")
                .ok_or(Error::ValueError("Covariance missing 'peri_arg'".into()))?;
            let inclination = *hash
                .get("inclination")
                .ok_or(Error::ValueError("Covariance missing 'inclination'".into()))?;
            let elem = PyCometElements::new(
                desig,
                epoch.into(),
                eccentricity,
                inclination,
                peri_dist,
                peri_arg,
                peri_time.into(),
                lon_of_ascending,
            );
            elem.state()
        } else if hash.contains_key("x") {
            let x = *hash
                .get("x")
                .ok_or(Error::ValueError("Covariance missing 'x'".into()))?;
            let y = *hash
                .get("y")
                .ok_or(Error::ValueError("Covariance missing 'y'".into()))?;
            let z = *hash
                .get("z")
                .ok_or(Error::ValueError("Covariance missing 'z'".into()))?;
            let vx = *hash
                .get("vx")
                .ok_or(Error::ValueError("Covariance missing 'vx'".into()))?;
            let vy = *hash
                .get("vy")
                .ok_or(Error::ValueError("Covariance missing 'vy'".into()))?;
            let vz = *hash
                .get("vz")
                .ok_or(Error::ValueError("Covariance missing 'vz'".into()))?;
            let pos = VectorLike::Arr([x, y, z]);
            let vel = VectorLike::Arr([vx, vy, vz]);
            Ok(PyState::new(
                Some(desig),
                epoch.into(),
                pos,
                vel,
                None,
                None,
            ))
        } else {
            Err(Error::ValueError("Covariance cannot be converted to a state, \
            the covariance parameters must either contain: \
            ['x', 'y', 'z', 'vx', 'vy', 'vz'] or \
            ['eccentricity', 'peri_dist', 'peri_time', 'lon_of_ascending', 'peri_arg', 'inclination']".into()))?
        }
    }

    /// Save the covariance matrix to a file.
    #[pyo3(name = "save")]
    pub fn py_save(&self, filename: String) -> PyResult<usize> {
        Ok(self.save(filename)?)
    }

    /// Load a covariance matrix from a file.
    #[staticmethod]
    #[pyo3(name = "load")]
    pub fn py_load(filename: String) -> PyResult<Self> {
        Ok(Self::load(filename)?)
    }

    /// Save a list to a binary file.
    ///
    /// Note that this saves a list of Covariances.
    #[staticmethod]
    #[pyo3(name = "save_list")]
    pub fn py_save_list(vec: Vec<Self>, filename: String) -> PyResult<()> {
        Ok(Self::save_vec(&vec, filename)?)
    }

    /// Load a list from a binary file.
    ///
    /// Note that this loads a list of Covariances.
    #[staticmethod]
    #[pyo3(name = "load_list")]
    pub fn py_load_list(filename: String) -> PyResult<Vec<Self>> {
        Ok(Self::load_vec(filename)?)
    }
}
