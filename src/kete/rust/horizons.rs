//! JPL Horizons data representation
use std::fmt::Debug;

use crate::covariance::Covariance;
use crate::elements::PyCometElements;
use crate::state::PyState;
use kete_core::{io::FileIO, prelude};
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

/// Horizons object properties
/// Physical, orbital, and observational properties of a solar system object as recorded in JPL Horizons.
#[pyclass(frozen, get_all, module = "kete")]
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct HorizonsProperties {
    /// The MPC designation of the object.
    desig: String,

    /// An optional group name to associate the object with a group.
    group: Option<String>,

    /// The epoch during which the orbital elements listed are accurate, in JD, TDB.
    epoch: Option<f64>,

    /// The eccentricity of the orbit.
    eccentricity: Option<f64>,

    /// The inclination of the orbit in degrees.
    inclination: Option<f64>,

    /// The longitudinal node of the orbit in degrees.
    lon_of_ascending: Option<f64>,

    /// The argument of perihelion in degrees.
    peri_arg: Option<f64>,

    /// The perihelion distance in AU.
    peri_dist: Option<f64>,

    /// The time of perihelion in JD, TDB scaled time.
    peri_time: Option<f64>,

    /// The H magnitude of the object.
    h_mag: Option<f64>,

    /// The visible albedo of the object, between 0 and 1.
    vis_albedo: Option<f64>,

    /// The diameter of the object in km.
    diameter: Option<f64>,

    /// The minimum orbital intersection distance between the object and Earth in AU.
    moid: Option<f64>,

    /// The g parameter of the object.
    g_phase: Option<f64>,

    /// If the object was previously known, this lists the length of time of the
    /// observations of the object in days.
    arc_len: Option<f64>,

    /// Covariance values in the orbit fit.
    covariance: Option<Covariance>,
}

impl FileIO for HorizonsProperties {}

#[pymethods]
impl HorizonsProperties {
    /// Construct a new HorizonsProperties Object
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (desig, group=None, epoch=None, eccentricity=None, inclination=None,
        lon_of_ascending=None, peri_arg=None, peri_dist=None, peri_time=None, h_mag=None,
        vis_albedo=None, diameter=None, moid=None, g_phase=None, arc_len=None, covariance=None))]
    pub fn new(
        desig: String,
        group: Option<String>,
        epoch: Option<f64>,
        eccentricity: Option<f64>,
        inclination: Option<f64>,
        lon_of_ascending: Option<f64>,
        peri_arg: Option<f64>,
        peri_dist: Option<f64>,
        peri_time: Option<f64>,
        h_mag: Option<f64>,
        vis_albedo: Option<f64>,
        diameter: Option<f64>,
        moid: Option<f64>,
        g_phase: Option<f64>,
        arc_len: Option<f64>,
        covariance: Option<Covariance>,
    ) -> Self {
        Self {
            desig,
            group,
            vis_albedo,
            diameter,
            moid,
            peri_dist,
            eccentricity,
            inclination,
            lon_of_ascending,
            peri_arg,
            peri_time,
            h_mag,
            g_phase,
            epoch,
            arc_len,
            covariance,
        }
    }

    /// Cometary orbital elements.
    #[getter]
    pub fn elements(&self) -> PyResult<PyCometElements> {
        Ok(PyCometElements(prelude::CometElements {
            desig: prelude::Desig::Name(self.desig.clone()),
            epoch: self
                .epoch
                .ok_or(prelude::Error::ValueError("No Epoch defined".into()))?,
            eccentricity: self
                .eccentricity
                .ok_or(prelude::Error::ValueError("No Eccentricity defined".into()))?,
            inclination: self
                .inclination
                .ok_or(prelude::Error::ValueError("No Inclination defined".into()))?
                .to_radians(),
            peri_arg: self
                .peri_arg
                .ok_or(prelude::Error::ValueError("No peri_arg defined".into()))?
                .to_radians(),
            peri_dist: self
                .peri_dist
                .ok_or(prelude::Error::ValueError("No peri_dist defined".into()))?,
            peri_time: self
                .peri_time
                .ok_or(prelude::Error::ValueError("No peri_time defined".into()))?,
            lon_of_ascending: self
                .lon_of_ascending
                .ok_or(prelude::Error::ValueError(
                    "No longitude of ascending node defined".into(),
                ))?
                .to_radians(),
        }))
    }

    /// Convert the orbital elements of the object to a State.
    #[getter]
    pub fn state(&self) -> PyResult<PyState> {
        self.elements()?.state()
    }

    fn __repr__(&self) -> String {
        fn cleanup<T: Debug>(opt: Option<T>) -> String {
            match opt {
                None => "None".into(),
                Some(val) => format!("{val:?}"),
            }
        }

        let cov = match self.covariance {
            Some(_) => "<present>",
            None => "None",
        };

        format!(
            "HorizonsObject(desig={:?}, group={:}, epoch={:}, eccentricity={:}, inclination={:}, \
            lon_of_ascending={:}, peri_arg={:}, peri_dist={:}, peri_time={:}, h_mag={:}, \
            vis_albedo={:}, diameter={:}, moid={:}, g_phase={:}, arc_len={:}, \
            covariance={:})",
            self.desig,
            cleanup(self.group.clone()),
            cleanup(self.epoch),
            cleanup(self.eccentricity),
            cleanup(self.inclination),
            cleanup(self.lon_of_ascending),
            cleanup(self.peri_arg),
            cleanup(self.peri_dist),
            cleanup(self.peri_time),
            cleanup(self.h_mag),
            cleanup(self.vis_albedo),
            cleanup(self.diameter),
            cleanup(self.moid),
            cleanup(self.g_phase),
            cleanup(self.arc_len),
            cov,
        )
    }

    /// Save the horizons query to a file.
    #[pyo3(name = "save")]
    pub fn py_save(&self, filename: String) -> PyResult<usize> {
        Ok(self.save(filename)?)
    }

    /// Load the horizons query from a file.
    #[staticmethod]
    #[pyo3(name = "load")]
    pub fn py_load(filename: String) -> PyResult<Self> {
        Ok(Self::load(filename)?)
    }
}
