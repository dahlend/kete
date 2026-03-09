//! JPL Horizons data representation
use std::fmt::Debug;

use crate::elements::PyCometElements;
use crate::state::PyState;
use crate::uncertain_state::PyUncertainState;
use kete_core::elements::CometElements;
use kete_core::prelude;
use kete_core::propagation::NonGravModel;
use nalgebra::DMatrix;
use pyo3::prelude::*;

/// Horizons object properties
/// Physical, orbital, and observational properties of a solar system object as recorded in JPL Horizons.
#[pyclass(frozen, module = "kete")]
#[derive(Clone, Debug)]
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

    /// Uncertain state built from the Horizons covariance (if provided).
    uncertain_state: Option<PyUncertainState>,
}

#[pymethods]
impl HorizonsProperties {
    /// Construct a new HorizonsProperties Object
    ///
    /// Parameters
    /// ----------
    /// desig : str
    ///     MPC designation.
    /// covariance_params : list[tuple[str, float]], optional
    ///     Parameter name/value pairs for the covariance (e.g.
    ///     ``[("eccentricity", 0.5), ("peri_dist", 1.2), ...]``).
    /// covariance_matrix : list[list[float]], optional
    ///     Covariance matrix matching the parameter ordering.
    /// covariance_epoch : float, optional
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
        let uncertain_state = match (covariance_params, covariance_matrix) {
            (Some(params), Some(cov_matrix)) => {
                let cov_epoch = covariance_epoch.or(epoch).unwrap_or(0.0);
                Some(build_uncertain_state(
                    &desig,
                    cov_epoch,
                    &params,
                    &cov_matrix,
                )?)
            }
            _ => None,
        };
        Ok(Self {
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
            uncertain_state,
        })
    }

    /// The MPC designation of the object.
    #[getter]
    fn desig(&self) -> &str {
        &self.desig
    }

    /// Optional group name.
    #[getter]
    fn group(&self) -> Option<&str> {
        self.group.as_deref()
    }

    /// Epoch of the orbital elements (JD, TDB).
    #[getter]
    fn epoch(&self) -> Option<f64> {
        self.epoch
    }

    /// Eccentricity.
    #[getter]
    fn eccentricity(&self) -> Option<f64> {
        self.eccentricity
    }

    /// Inclination in degrees.
    #[getter]
    fn inclination(&self) -> Option<f64> {
        self.inclination
    }

    /// Longitude of ascending node in degrees.
    #[getter]
    fn lon_of_ascending(&self) -> Option<f64> {
        self.lon_of_ascending
    }

    /// Argument of perihelion in degrees.
    #[getter]
    fn peri_arg(&self) -> Option<f64> {
        self.peri_arg
    }

    /// Perihelion distance in AU.
    #[getter]
    fn peri_dist(&self) -> Option<f64> {
        self.peri_dist
    }

    /// Time of perihelion (JD, TDB).
    #[getter]
    fn peri_time(&self) -> Option<f64> {
        self.peri_time
    }

    /// H magnitude.
    #[getter]
    fn h_mag(&self) -> Option<f64> {
        self.h_mag
    }

    /// Visible albedo (0-1).
    #[getter]
    fn vis_albedo(&self) -> Option<f64> {
        self.vis_albedo
    }

    /// Diameter in km.
    #[getter]
    fn diameter(&self) -> Option<f64> {
        self.diameter
    }

    /// MOID to Earth in AU.
    #[getter]
    fn moid(&self) -> Option<f64> {
        self.moid
    }

    /// G phase parameter.
    #[getter]
    fn g_phase(&self) -> Option<f64> {
        self.g_phase
    }

    /// Arc length in days.
    #[getter]
    fn arc_len(&self) -> Option<f64> {
        self.arc_len
    }

    /// The uncertain orbit state, constructed from the Horizons covariance.
    ///
    /// Returns ``None`` if no covariance was provided.
    #[getter]
    fn uncertain_state(&self) -> Option<PyUncertainState> {
        self.uncertain_state.clone()
    }

    /// Cometary orbital elements.
    #[getter]
    pub fn elements(&self) -> PyResult<PyCometElements> {
        Ok(PyCometElements(CometElements {
            desig: prelude::Desig::Name(self.desig.clone()),
            epoch: self
                .epoch
                .ok_or(prelude::Error::ValueError("No Epoch defined".into()))?
                .into(),
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
                .ok_or(prelude::Error::ValueError("No peri_time defined".into()))?
                .into(),
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

        let cov = match self.uncertain_state {
            Some(_) => "<present>",
            None => "None",
        };

        format!(
            "HorizonsObject(desig={:?}, group={:}, epoch={:}, eccentricity={:}, inclination={:}, \
            lon_of_ascending={:}, peri_arg={:}, peri_dist={:}, peri_time={:}, h_mag={:}, \
            vis_albedo={:}, diameter={:}, moid={:}, g_phase={:}, arc_len={:}, \
            uncertain_state={:})",
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
}

/// Build a [`PyUncertainState`] from raw Horizons covariance data.
///
/// Handles both cometary-element and Cartesian parameterizations,
/// including automatic detection and construction of non-gravitational
/// models when A1/A2/A3 or beta parameters are present.
fn build_uncertain_state(
    desig: &str,
    epoch: f64,
    params: &[(String, f64)],
    cov_matrix: &[Vec<f64>],
) -> PyResult<PyUncertainState> {
    let n_params = params.len();
    if cov_matrix.len() != n_params {
        return Err(prelude::Error::ValueError(format!(
            "Covariance matrix has {} rows but {} parameters",
            cov_matrix.len(),
            n_params
        ))
        .into());
    }
    for (i, row) in cov_matrix.iter().enumerate() {
        if row.len() != n_params {
            return Err(prelude::Error::ValueError(format!(
                "Covariance matrix row {i} has length {}, expected {n_params}",
                row.len()
            ))
            .into());
        }
    }
    let lower_names: Vec<String> = params.iter().map(|(k, _)| k.to_lowercase()).collect();
    let get = |key: &str| -> PyResult<f64> {
        params
            .iter()
            .find(|(k, _)| k.to_lowercase() == key)
            .map(|(_, v)| *v)
            .ok_or_else(|| {
                prelude::Error::ValueError(format!("Horizons covariance missing '{key}'")).into()
            })
    };

    let elem_keys: &[&str] = &[
        "eccentricity",
        "peri_dist",
        "peri_time",
        "lon_of_ascending",
        "peri_arg",
        "inclination",
    ];
    let cart_keys: &[&str] = &["x", "y", "z", "vx", "vy", "vz"];
    let is_cometary = lower_names.iter().any(|k| elem_keys.contains(&k.as_str()));
    let core_keys = if is_cometary { elem_keys } else { cart_keys };

    let core_indices: Vec<usize> = core_keys
        .iter()
        .filter_map(|&key| lower_names.iter().position(|k| k == key))
        .collect();
    let nongrav_indices: Vec<usize> = (0..lower_names.len())
        .filter(|i| !core_indices.contains(i))
        .collect();

    let non_grav = if nongrav_indices.is_empty() {
        None
    } else {
        let ng_hash: std::collections::HashMap<&str, f64> = nongrav_indices
            .iter()
            .map(|&i| (lower_names[i].as_str(), params[i].1))
            .collect();
        build_nongrav_from_hash(&ng_hash)
    };

    let np = non_grav.as_ref().map_or(0, NonGravModel::n_free_params);
    let n = 6 + np;

    let ng_param_names: Vec<&str> = match &non_grav {
        Some(ng) => ng.param_names().to_vec(),
        None => Vec::new(),
    };
    let reorder: Vec<Option<usize>> = (0..n)
        .map(|i| {
            if i < 6 {
                Some(core_indices[i])
            } else {
                let model_name = ng_param_names.get(i - 6)?;
                nongrav_indices
                    .iter()
                    .find(|&&ni| lower_names[ni] == *model_name)
                    .copied()
            }
        })
        .collect();

    if is_cometary {
        let elements = CometElements {
            desig: prelude::Desig::Name(desig.to_string()),
            epoch: epoch.into(),
            eccentricity: get("eccentricity")?,
            inclination: get("inclination")?.to_radians(),
            peri_arg: get("peri_arg")?.to_radians(),
            peri_dist: get("peri_dist")?,
            peri_time: get("peri_time")?.into(),
            lon_of_ascending: get("lon_of_ascending")?.to_radians(),
        };

        let deg2rad = std::f64::consts::PI / 180.0;
        let scale: Vec<f64> = (0..n)
            .map(|i| if (3..6).contains(&i) { deg2rad } else { 1.0 })
            .collect();

        let mat = DMatrix::from_fn(n, n, |r, c| match (reorder[r], reorder[c]) {
            (Some(sr), Some(sc)) => cov_matrix[sr][sc] * scale[r] * scale[c],
            _ => 0.0,
        });

        let us = kete_fitting::UncertainState::from_cometary(&elements, &mat, non_grav)?;
        Ok(PyUncertainState(us))
    } else {
        let x = get("x")?;
        let y = get("y")?;
        let z = get("z")?;
        let vx = get("vx")?;
        let vy = get("vy")?;
        let vz = get("vz")?;

        let desig_val = match desig {
            "" => prelude::Desig::Empty,
            _ => prelude::Desig::Name(desig.to_string()),
        };
        let state: prelude::State<prelude::Equatorial> = prelude::State::new(
            desig_val,
            prelude::Time::new(epoch),
            [x, y, z].into(),
            [vx, vy, vz].into(),
            10,
        );

        let mat = DMatrix::from_fn(n, n, |r, c| match (reorder[r], reorder[c]) {
            (Some(sr), Some(sc)) => cov_matrix[sr][sc],
            _ => 0.0,
        });

        let us = kete_fitting::UncertainState::new(state, mat, non_grav)?;
        Ok(PyUncertainState(us))
    }
}

/// Build a [`NonGravModel`] from leftover (non-orbital) sampled parameters.
///
/// Returns `Some(model)` only when the parameter names match a supported
/// non-gravitational model:
///  - **JplComet**: at least one of `a1`, `a2`, `a3` is present.
///  - **Dust**: `beta` is present.
///
/// Unrecognized parameter sets (e.g. `rho`, `amrat`) yield `None`;
/// the caller should then fall back to a pure orbital covariance.
fn build_nongrav_from_hash(hash: &std::collections::HashMap<&str, f64>) -> Option<NonGravModel> {
    let get = |key: &str, default: f64| -> f64 { hash.get(key).copied().unwrap_or(default) };

    let has_jpl = hash.contains_key("a1") || hash.contains_key("a2") || hash.contains_key("a3");
    let has_dust = hash.contains_key("beta");

    if has_jpl {
        Some(NonGravModel::new_jpl(
            get("a1", 0.0),
            get("a2", 0.0),
            get("a3", 0.0),
            get("alpha", 0.111_262_042_6),
            get("r_0", 2.808),
            get("m", 2.15),
            get("n", 5.093),
            get("k", 4.6142),
            get("dt", 0.0),
        ))
    } else if has_dust {
        Some(NonGravModel::new_dust(get("beta", 0.0)))
    } else {
        let unknown: Vec<&str> = hash.keys().copied().collect();
        eprintln!(
            "Warning: Horizons covariance contains unrecognized non-gravitational \
             parameters {unknown:?}; ignoring and using orbital covariance only."
        );
        None
    }
}
