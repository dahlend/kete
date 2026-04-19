//! PyO3 wrappers around [`kete_fitting::HorizonsProperties`] and its fetch API.
use std::fmt::Debug;

use crate::elements::PyCometElements;
use crate::nongrav::PyNonGravModel;
use crate::state::PyState;
use crate::uncertain_state::PyUncertainState;
use pyo3::prelude::*;

/// Horizons object properties
/// Physical, orbital, and observational properties of a solar system object as recorded in JPL Horizons.
#[pyclass(name = "HorizonsProperties", frozen, module = "kete", from_py_object)]
#[derive(Clone, Debug)]
pub struct PyHorizonsProperties(pub kete_fitting::HorizonsProperties);

#[pymethods]
impl PyHorizonsProperties {
    /// Construct a new HorizonsProperties Object
    ///
    /// Parameters
    /// ----------
    /// desig : str
    ///     MPC designation.
    /// covariance_params : list
    ///     Parameter name/value pairs for the covariance (e.g.
    ///     ``[("eccentricity", 0.5), ("peri_dist", 1.2), ...]``).
    /// covariance_matrix : list
    ///     Covariance matrix matching the parameter ordering.
    /// covariance_epoch : float
    ///     Epoch of the covariance (JD, TDB).
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (desig, group=None, epoch=None, eccentricity=None, inclination=None,
        lon_of_ascending=None, peri_arg=None, peri_dist=None, peri_time=None, h_mag=None,
        vis_albedo=None, diameter=None, moid=None, g_phase=None, arc_len=None,
        covariance_params=None, covariance_matrix=None, covariance_epoch=None))]
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
        covariance_params: Option<Vec<(String, f64)>>,
        covariance_matrix: Option<Vec<Vec<f64>>>,
        covariance_epoch: Option<f64>,
    ) -> PyResult<Self> {
        let inner = kete_fitting::HorizonsProperties::new(
            desig,
            group,
            epoch,
            eccentricity,
            inclination,
            lon_of_ascending,
            peri_arg,
            peri_dist,
            peri_time,
            h_mag,
            vis_albedo,
            diameter,
            moid,
            g_phase,
            arc_len,
            covariance_params,
            covariance_matrix,
            covariance_epoch,
        )?;
        Ok(Self(inner))
    }

    /// The MPC designation of the object.
    #[getter]
    fn desig(&self) -> &str {
        &self.0.desig
    }

    /// Optional group name.
    #[getter]
    fn group(&self) -> Option<&str> {
        self.0.group.as_deref()
    }

    /// Epoch of the orbital elements (JD, TDB).
    #[getter]
    fn epoch(&self) -> Option<f64> {
        self.0.epoch
    }

    /// Eccentricity.
    #[getter]
    fn eccentricity(&self) -> Option<f64> {
        self.0.eccentricity
    }

    /// Inclination in degrees.
    #[getter]
    fn inclination(&self) -> Option<f64> {
        self.0.inclination
    }

    /// Longitude of ascending node in degrees.
    #[getter]
    fn lon_of_ascending(&self) -> Option<f64> {
        self.0.lon_of_ascending
    }

    /// Argument of perihelion in degrees.
    #[getter]
    fn peri_arg(&self) -> Option<f64> {
        self.0.peri_arg
    }

    /// Perihelion distance in AU.
    #[getter]
    fn peri_dist(&self) -> Option<f64> {
        self.0.peri_dist
    }

    /// Time of perihelion (JD, TDB).
    #[getter]
    fn peri_time(&self) -> Option<f64> {
        self.0.peri_time
    }

    /// H magnitude.
    #[getter]
    fn h_mag(&self) -> Option<f64> {
        self.0.h_mag
    }

    /// Visible albedo (0-1).
    #[getter]
    fn vis_albedo(&self) -> Option<f64> {
        self.0.vis_albedo
    }

    /// Diameter in km.
    #[getter]
    fn diameter(&self) -> Option<f64> {
        self.0.diameter
    }

    /// MOID to Earth in AU.
    #[getter]
    fn moid(&self) -> Option<f64> {
        self.0.moid
    }

    /// G phase parameter.
    #[getter]
    fn g_phase(&self) -> Option<f64> {
        self.0.g_phase
    }

    /// Arc length in days.
    #[getter]
    fn arc_len(&self) -> Option<f64> {
        self.0.arc_len
    }

    /// The uncertain orbit state, constructed from the Horizons covariance.
    ///
    /// Returns ``None`` if no covariance was provided.
    #[getter]
    fn uncertain_state(&self) -> Option<PyUncertainState> {
        self.0.uncertain_state.clone().map(PyUncertainState)
    }

    /// Non-gravitational force model from Horizons, if available.
    #[getter]
    fn non_grav(&self) -> Option<PyNonGravModel> {
        self.0.non_grav.clone().map(PyNonGravModel)
    }

    /// Alternate designations for this object.
    #[getter]
    fn alternate_desigs(&self) -> Vec<String> {
        self.0.alternate_desigs.clone()
    }

    /// Raw JSON response from SBDB, if fetched via API.
    #[getter]
    fn raw_json(&self) -> Option<String> {
        self.0.raw_json.clone()
    }

    /// Cometary orbital elements.
    #[getter]
    pub fn elements(&self) -> PyResult<PyCometElements> {
        Ok(PyCometElements(self.0.elements()?))
    }

    /// Convert the orbital elements of the object to a State.
    #[getter]
    pub fn state(&self) -> PyResult<PyState> {
        Ok(self.0.state()?.into())
    }

    /// Fetch an object from JPL Horizons.
    ///
    /// Parameters
    /// ----------
    /// name : str
    ///     Name of the object to fetch.
    /// update_name : bool
    ///     If true, replace name with the name that horizons uses for the object.
    /// update_cache : bool
    ///     If true, the current cache contents are ignored, and new values are saved
    ///     after querying horizons.
    /// exact_name : bool
    ///     If true, it is assumed that an exact designation in the format of horizons
    ///     has been provided.
    #[pyo3(signature = (name, update_name=true, update_cache=false, exact_name=false))]
    #[staticmethod]
    pub fn fetch(
        name: &str,
        update_name: bool,
        update_cache: bool,
        exact_name: bool,
    ) -> PyResult<Self> {
        Ok(Self(kete_fitting::HorizonsProperties::fetch(
            name,
            update_name,
            update_cache,
            exact_name,
        )?))
    }

    fn __repr__(&self) -> String {
        fn cleanup<T: Debug>(opt: Option<T>) -> String {
            match opt {
                None => "None".into(),
                Some(val) => format!("{val:?}"),
            }
        }

        let cov = match self.0.uncertain_state {
            Some(_) => "<present>",
            None => "None",
        };

        format!(
            "HorizonsObject(desig={:?}, group={:}, epoch={:}, eccentricity={:}, inclination={:}, \
            lon_of_ascending={:}, peri_arg={:}, peri_dist={:}, peri_time={:}, h_mag={:}, \
            vis_albedo={:}, diameter={:}, moid={:}, g_phase={:}, arc_len={:}, \
            uncertain_state={:})",
            self.0.desig,
            cleanup(self.0.group.clone()),
            cleanup(self.0.epoch),
            cleanup(self.0.eccentricity),
            cleanup(self.0.inclination),
            cleanup(self.0.lon_of_ascending),
            cleanup(self.0.peri_arg),
            cleanup(self.0.peri_dist),
            cleanup(self.0.peri_time),
            cleanup(self.0.h_mag),
            cleanup(self.0.vis_albedo),
            cleanup(self.0.diameter),
            cleanup(self.0.moid),
            cleanup(self.0.g_phase),
            cleanup(self.0.arc_len),
            cov,
        )
    }
}
