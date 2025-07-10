//! Python support for orbital elements
use kete_core::elements;
use kete_core::frames::Ecliptic;
use kete_core::prelude;
use pyo3::{PyResult, pyclass, pymethods};

use crate::state::PyState;

/// Cometary Elements class made accessible to python.
///
/// Angles must be in degrees, distances in AU.
///
/// Parameters
/// ----------
/// desig:
///     The designations of the object.
/// epoch:
///     The epoch time for the orbital elements.
/// eccentricity:
///     The eccentricity of the orbit.
/// inclination:
///     The inclination, must be in degrees.
/// peri_dist:
///     The perihelion distance in AU.
/// peri_arg:
///     The argument of perihelion, must be in degrees.
/// peri_time:
///     The JD time of perihelion.
/// lon_of_ascending:
///     The longitude of ascending node, in degrees.
#[pyclass(module = "kete", frozen, name = "CometElements")]
#[derive(Clone, Debug)]
pub struct PyCometElements(pub elements::CometElements);

#[pymethods]
impl PyCometElements {
    /// Construct a new CometElements object.
    ///
    /// Cometary elements are in the Ecliptic frame.
    ///
    /// Parameters
    /// ----------
    /// desig: str
    ///     Designation of the object.
    /// epoch: float
    ///     Epoch of the orbit fit in JD.
    /// eccentricity: float
    ///     Eccentricity of the orbit.
    /// inclination: float
    ///     Inclination of the orbit in degrees.
    /// peri_dist: float
    ///     Perihelion Distance in au.
    /// peri_arg: float
    ///     Argument of perihelion in degrees.
    /// peri_time: float
    ///     Time of perihelion passage in JD.
    /// lon_of_ascending: float
    ///     Longitude of ascending node in degrees.
    #[new]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        desig: String,
        epoch: f64,
        eccentricity: f64,
        inclination: f64,
        peri_dist: f64,
        peri_arg: f64,
        peri_time: f64,
        lon_of_ascending: f64,
    ) -> Self {
        Self(elements::CometElements {
            desig: prelude::Desig::Name(desig),
            epoch,
            eccentricity,
            inclination: inclination.to_radians(),
            lon_of_ascending: lon_of_ascending.to_radians(),
            peri_time,
            peri_arg: peri_arg.to_radians(),
            peri_dist,
        })
    }

    /// Construct a new CometElements object from a `State`.
    ///
    /// Parameters
    /// ----------
    /// State :
    ///     State Object.
    #[staticmethod]
    pub fn from_state(state: PyState) -> Self {
        Self(elements::CometElements::from_state(&state.raw.into_frame()))
    }

    /// Epoch of the elements in JD.
    #[getter]
    pub fn epoch(&self) -> f64 {
        self.0.epoch
    }

    /// Designation of the object.
    #[getter]
    pub fn desig(&self) -> String {
        match &self.0.desig {
            prelude::Desig::Naif(s) => {
                kete_core::spice::try_name_from_id(*s).unwrap_or(s.to_string())
            }
            _ => self.0.desig.to_string(),
        }
    }

    /// Eccentricity of the orbit.
    #[getter]
    pub fn eccentricity(&self) -> f64 {
        self.0.eccentricity
    }

    /// Inclination of the orbit in degrees.
    #[getter]
    pub fn inclination(&self) -> f64 {
        self.0.inclination.to_degrees()
    }

    /// Longitude of the ascending node of the orbit in degrees.
    #[getter]
    pub fn lon_of_ascending(&self) -> f64 {
        self.0.lon_of_ascending.to_degrees()
    }

    /// Perihelion time of the orbit in JD.
    #[getter]
    pub fn peri_time(&self) -> f64 {
        self.0.peri_time
    }

    /// Argument of Perihelion of the orbit in degrees.
    #[getter]
    pub fn peri_arg(&self) -> f64 {
        self.0.peri_arg.to_degrees()
    }

    /// Distance of Perihelion of the orbit in au.
    #[getter]
    pub fn peri_dist(&self) -> f64 {
        self.0.peri_dist
    }

    /// Semi Major Axis of the orbit in au.
    #[getter]
    pub fn semi_major(&self) -> f64 {
        self.0.semi_major()
    }

    /// Mean Motion of the orbit in degrees.
    #[getter]
    pub fn mean_motion(&self) -> f64 {
        self.0.mean_motion().to_degrees()
    }

    /// Orbital Period in days, nan if non-elliptical.
    #[getter]
    pub fn orbital_period(&self) -> f64 {
        self.0.orbital_period()
    }

    /// Eccentric Anomaly in degrees.
    #[getter]
    pub fn eccentric_anomaly(&self) -> PyResult<f64> {
        Ok(self.0.eccentric_anomaly().map(|x| x.to_degrees())?)
    }

    /// Mean Anomaly in degrees.
    #[getter]
    pub fn mean_anomaly(&self) -> f64 {
        self.0.mean_anomaly().to_degrees()
    }

    /// Aphelion distance in au
    #[getter]
    pub fn aphelion(&self) -> f64 {
        self.0.aphelion()
    }

    /// True Anomaly in degrees.
    #[getter]
    pub fn true_anomaly(&self) -> PyResult<f64> {
        Ok(self.0.true_anomaly().map(|x| x.to_degrees())?)
    }

    /// Convert the orbital elements into a cartesian State.
    #[getter]
    pub fn state(&self) -> PyResult<PyState> {
        Ok(self.0.try_to_state()?.into_frame::<Ecliptic>().into())
    }

    fn __repr__(&self) -> String {
        format!(
            "CometElements(desig={:?}, epoch={}, eccentricity={}, inclination={}, lon_of_ascending={}, peri_time={}, peri_arg={}, peri_dist={})",
            self.desig(),
            self.epoch(),
            self.eccentricity(),
            self.inclination(),
            self.lon_of_ascending(),
            self.peri_time(),
            self.peri_arg(),
            self.peri_dist()
        )
    }
}
