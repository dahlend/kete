//! Python wrapper for [`kete_core::state::UncertainState`].
//!
//! Bridges between the Rust state shape (`free_params: Vec<f64>`, no
//! force-model state) and the Python user-facing shape (a `non_grav`
//! property that returns a `NonGravModel`). The wrapper stores both:
//! - `state`: the kete_core `UncertainState` carrying covariance and
//!   `free_params`.
//! - `non_grav`: the optional model template (variant + fixed
//!   coefficients) needed to build a `ParameterizedForce` for propagation.

use super::PyState;
use crate::elements::PyCometElements;
use crate::nongrav::PyNonGravModel;
use crate::time::PyTime;
use kete_core::forces::{ForceSet, GravParams, NonGravMask, ParameterizedForce};
use kete_core::frames::{Equatorial, SSB};
use kete_core::prelude::*;
use kete_core::state::StateLike;
use kete_spice::propagation::Recenter;
use kete_spice::propagation::SpkNBody;
use kete_spice::propagation::{propagate_with_diagnosis, sigma_point_divergence};
use kete_spice::spk::LOADED_SPK;
use nalgebra::DMatrix;
use pyo3::prelude::*;

/// Uncertain orbit state: a best-fit Cartesian state together with a
/// covariance matrix that may span the 6 position/velocity components
/// and any fitted non-gravitational parameters.
///
/// The `non_grav` field stores an all-`None` [`ParameterMask`] wrapping
/// the typed ParameterizedForce template; free-parameter values live in
/// `state.free_params`, not in the mask.
#[pyclass(frozen, module = "kete", name = "UncertainState", from_py_object)]
#[derive(Clone)]
pub struct PyUncertainState {
    /// Underlying state with covariance and free-parameter values.
    pub state: UncertainState,
    /// All-`None` parameter mask over the non-grav ParameterizedForce template.
    /// Free-parameter values are stored on `state.free_params`.
    pub non_grav: Option<NonGravMask>,
}

impl std::fmt::Debug for PyUncertainState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PyUncertainState")
            .field("state", &self.state)
            .field("non_grav_present", &self.non_grav.is_some())
            .finish()
    }
}

impl PyUncertainState {
    /// Build the SSB-centered ParameterizedForce used by all propagation paths:
    /// gravity-only `SpkNBody` when `non_grav` is `None`, or
    /// `ForceSet { SpkNBody, Recenter<SSB, _> }` wrapping the typed
    /// non-grav force template otherwise. Captures the `LOADED_SPK`
    /// read guard so the borrow lifetime is well-defined.
    fn build_forces<'a>(
        &self,
        spk: &'a kete_spice::spk::SpkCollection,
        planets: &'a [GravParams],
    ) -> ForceSet<'a, Equatorial, SSB> {
        let mut force_set: ForceSet<'_, Equatorial, SSB> =
            ForceSet::new().with(Box::new(SpkNBody::new(spk, planets)));
        if let Some(ref ng) = self.non_grav {
            force_set = force_set.with(Box::new(Recenter::<SSB, _>::new(spk, ng.clone())));
        }
        force_set
    }
}

#[pymethods]
impl PyUncertainState {
    /// Build an ``UncertainState`` from a state with isotropic diagonal
    /// uncertainties.
    ///
    /// The covariance is initialized to a diagonal matrix with the
    /// given ``pos_sigma`` (AU) and ``vel_sigma`` (AU/day) on the
    /// diagonal.  Useful for seeding MCMC from an IOD candidate.
    ///
    /// The input state is automatically re-centered to the solar
    /// system barycenter if needed.
    ///
    /// Parameters
    /// ----------
    /// state : :class:`~kete.State`
    ///     Object state (any center / frame -- will be converted to
    ///     SSB-centered Equatorial internally).
    /// pos_sigma : float
    ///     1-sigma position uncertainty in AU (default 0.01).
    /// vel_sigma : float
    ///     1-sigma velocity uncertainty in AU/day (default 0.0001).
    /// non_grav : :class:`~kete.propagation.NonGravModel`, optional
    ///     Non-gravitational model template.  If provided, the covariance is
    ///     extended to (6+Np)x(6+Np) with tiny diagonal entries for the
    ///     non-grav parameters.
    #[staticmethod]
    #[pyo3(signature = (state, pos_sigma=0.01, vel_sigma=0.0001, non_grav=None))]
    fn from_state(
        state: PyState,
        pos_sigma: f64,
        vel_sigma: f64,
        non_grav: Option<PyNonGravModel>,
    ) -> PyResult<Self> {
        if pos_sigma <= 0.0 || vel_sigma <= 0.0 {
            return Err(
                Error::ValueError("pos_sigma and vel_sigma must be positive".into()).into(),
            );
        }
        let mut eq_state = state.raw;
        if eq_state.center_id() != 0 {
            let spk = LOADED_SPK.try_read().map_err(Error::from)?;
            spk.try_change_center(&mut eq_state, 0)?;
        }
        let ng_mask = non_grav.as_ref().map(|m| m.to_mask());
        let free_params = non_grav.map(|m| m.initial_values()).unwrap_or_default();
        let np = free_params.len();
        let d = 6 + np;
        let mut cov = DMatrix::<f64>::zeros(d, d);
        for i in 0..3 {
            cov[(i, i)] = pos_sigma * pos_sigma;
        }
        for i in 3..6 {
            cov[(i, i)] = vel_sigma * vel_sigma;
        }
        // Tiny diagonal for free params (so the matrix is positive-definite).
        for i in 6..d {
            cov[(i, i)] = 1e-30;
        }
        let us = UncertainState::new(eq_state, cov, free_params)?;
        Ok(Self {
            state: us,
            non_grav: ng_mask,
        })
    }

    /// Build an ``UncertainState`` from cometary orbital elements and
    /// a covariance expressed in element space.
    ///
    /// The element-space covariance is transformed to a Cartesian
    /// covariance via a numerically evaluated Jacobian.
    ///
    /// Parameters
    /// ----------
    /// elements : CometElements
    ///     Cometary orbital elements (with desig and epoch).
    /// cov_matrix : list[list[float]]
    ///     Covariance matrix in element space, (6+Np)x(6+Np).
    ///     Element order: ``[e, q, tp, node, w, i, <nongrav...>]``.
    /// non_grav : :class:`~kete.propagation.NonGravModel`, optional
    ///     Non-gravitational model template.
    #[staticmethod]
    #[pyo3(signature = (elements, cov_matrix, non_grav=None))]
    fn from_cometary(
        elements: PyCometElements,
        cov_matrix: Vec<Vec<f64>>,
        non_grav: Option<PyNonGravModel>,
    ) -> PyResult<Self> {
        let n = cov_matrix.len();
        for (i, row) in cov_matrix.iter().enumerate() {
            if row.len() != n {
                return Err(Error::ValueError(format!(
                    "Covariance matrix row {i} has length {}, expected {n}",
                    row.len()
                ))
                .into());
            }
        }
        let mat = DMatrix::from_fn(n, n, |r, c| cov_matrix[r][c]);
        let ng_mask = non_grav.as_ref().map(|m| m.to_mask());
        let free_params = non_grav.map(|m| m.initial_values()).unwrap_or_default();
        let us = UncertainState::from_cometary(&elements.0, &mat, free_params)?;
        Ok(Self {
            state: us,
            non_grav: ng_mask,
        })
    }

    /// Best-fit state at the reference epoch (Sun-centered, Ecliptic).
    #[getter]
    fn state(&self) -> PyResult<PyState> {
        let mut st = self.state.state.clone();
        if st.center_id() != 10 {
            let spk = LOADED_SPK.try_read().map_err(Error::from)?;
            spk.try_change_center(&mut st, 10)?;
        }
        Ok(st.into())
    }

    /// Covariance matrix as a list of lists (use ``np.array()`` to convert).
    #[getter]
    fn cov_matrix(&self) -> Vec<Vec<f64>> {
        let n = self.state.cov_matrix.nrows();
        let m = self.state.cov_matrix.ncols();
        (0..n)
            .map(|r| (0..m).map(|c| self.state.cov_matrix[(r, c)]).collect())
            .collect()
    }

    /// Non-gravitational model template, or None.
    ///
    /// The template carries the model variant and fixed coefficients
    /// (g(r) shape, albedo, spin pole, etc.). The fitted parameter
    /// values are stored on the underlying state's `free_params` and
    /// merged here to reconstruct the full model.
    #[getter]
    fn non_grav(&self) -> Option<PyNonGravModel> {
        self.non_grav.as_ref().and_then(|mask| {
            let full = mask.merge(&self.state.free_params).ok()?;
            PyNonGravModel::from_force(&mask.inner, &full)
        })
    }

    /// Object designator (shortcut for ``self.state.desig``).
    #[getter]
    fn desig(&self) -> String {
        self.state.state.desig.to_string()
    }

    /// Reference epoch as a :class:`~kete.Time` (shortcut for ``self.state.epoch``).
    #[getter]
    fn epoch(&self) -> PyTime {
        self.state.state.epoch.jd.into()
    }

    /// Names of all parameters in the covariance matrix, in row/column
    /// order.
    ///
    /// Always starts with ``["x", "y", "z", "vx", "vy", "vz"]``,
    /// followed by any non-gravitational parameter names.
    #[getter]
    fn param_names(&self) -> Vec<String> {
        let mut names: Vec<String> = vec!["x", "y", "z", "vx", "vy", "vz"]
            .into_iter()
            .map(String::from)
            .collect();
        if let Some(ref ng) = self.non_grav {
            names.extend(ng.free_param_names().into_iter().map(String::from));
        }
        names
    }

    /// Draw random samples from the covariance distribution.
    ///
    /// Returns a tuple ``(states, non_gravs)`` where ``states`` is a list
    /// of :class:`~kete.State` objects and ``non_gravs`` is a list of
    /// :class:`~kete.propagation.NonGravModel` or ``None``.
    ///
    /// Parameters
    /// ----------
    /// n_samples : int
    ///     Number of samples to draw.
    /// seed : int
    ///     Random seed for reproducibility (optional).
    #[pyo3(signature = (n_samples, seed=None))]
    pub fn sample(
        &self,
        n_samples: usize,
        seed: Option<u64>,
    ) -> PyResult<(Vec<PyState>, Vec<Option<PyNonGravModel>>)> {
        let samples = self.state.sample(n_samples, seed)?;
        let spk = LOADED_SPK.try_read().map_err(Error::from)?;
        let mut states = Vec::with_capacity(n_samples);
        let mut non_gravs = Vec::with_capacity(n_samples);
        for (mut st, sampled_params) in samples {
            // Re-center to Sun for the Python-facing state.
            if st.center_id() != 10 {
                spk.try_change_center(&mut st, 10)?;
            }
            let py_st: PyState = st.into();
            states.push(py_st);
            // Reconstruct a NonGravModel from the mask + sampled params.
            let ng = self.non_grav.as_ref().and_then(|mask| {
                let raw = if sampled_params.is_empty() {
                    &self.state.free_params
                } else {
                    &sampled_params
                };
                let full = mask.merge(raw).ok()?;
                PyNonGravModel::from_force(&mask.inner, &full)
            });
            non_gravs.push(ng);
        }
        Ok((states, non_gravs))
    }

    /// Propagate this :class:`~kete.UncertainState` linearly to ``jd``.
    ///
    /// The mean state is integrated by the full N-body Radau-15
    /// integrator and the covariance is updated by the augmented
    /// ``(6 + Np) x (6 + Np)`` state transition matrix.  The result is
    /// SSB-centered regardless of the input center.
    ///
    /// Parameters
    /// ----------
    /// jd : :class:`~kete.Time` or float
    ///     Target epoch (TDB).
    /// include_asteroids : bool, optional
    ///     If True, include asteroid masses in the force model.
    #[pyo3(signature = (jd, include_asteroids=false))]
    fn propagate(&self, jd: PyTime, include_asteroids: bool) -> PyResult<Self> {
        let spk = LOADED_SPK.try_read().map_err(Error::from)?;
        let ssb_state = spk.try_to_ssb(self.state.state.clone())?;
        let ssb_us = UncertainState::<Equatorial, SSB>::new(
            ssb_state,
            self.state.cov_matrix.clone(),
            self.state.free_params.clone(),
        )?;
        let result = if include_asteroids {
            let extended = GravParams::selected_masses();
            let forces = self.build_forces(&spk, &extended);
            ssb_us.propagate_with(&forces, jd.into())?
        } else {
            let planets = GravParams::planets();
            let forces = self.build_forces(&spk, &planets);
            ssb_us.propagate_with(&forces, jd.into())?
        };
        // Convert back to DynCenter for storage on the Python wrapper
        // (matches the historical shape).
        let dyn_state: UncertainState =
            UncertainState::new(result.state.into(), result.cov_matrix, result.free_params)?;
        Ok(Self {
            state: dyn_state,
            non_grav: self.non_grav.clone(),
        })
    }

    /// Propagate this :class:`~kete.UncertainState` linearly *and*
    /// compute its sigma-point divergence in a single variational
    /// integration.
    #[pyo3(signature = (jd, n_axes=3, sigma_factor=1.0, include_asteroids=false))]
    fn propagate_with_diagnosis(
        &self,
        jd: PyTime,
        n_axes: usize,
        sigma_factor: f64,
        include_asteroids: bool,
    ) -> PyResult<(Self, f64)> {
        let spk = LOADED_SPK.try_read().map_err(Error::from)?;
        let ssb_state = spk.try_to_ssb(self.state.state.clone())?;
        let ssb_us = UncertainState::<Equatorial, SSB>::new(
            ssb_state,
            self.state.cov_matrix.clone(),
            self.state.free_params.clone(),
        )?;
        let diag = if include_asteroids {
            let extended = GravParams::selected_masses();
            let forces = self.build_forces(&spk, &extended);
            propagate_with_diagnosis(&ssb_us, &forces, jd.into(), n_axes, sigma_factor)?
        } else {
            let planets = GravParams::planets();
            let forces = self.build_forces(&spk, &planets);
            propagate_with_diagnosis(&ssb_us, &forces, jd.into(), n_axes, sigma_factor)?
        };
        let dyn_state: UncertainState = UncertainState::new(
            diag.propagated.state.into(),
            diag.propagated.cov_matrix,
            diag.propagated.free_params,
        )?;
        Ok((
            Self {
                state: dyn_state,
                non_grav: self.non_grav.clone(),
            },
            diag.divergence,
        ))
    }

    /// Sigma-point divergence: a relative measure of how much the
    /// linear (STM-based) propagation deviates from full nonlinear
    /// propagation along the dominant eigenvectors of the covariance.
    #[pyo3(signature = (jd, n_axes=3, sigma_factor=1.0, include_asteroids=false))]
    fn sigma_point_divergence(
        &self,
        jd: PyTime,
        n_axes: usize,
        sigma_factor: f64,
        include_asteroids: bool,
    ) -> PyResult<f64> {
        let spk = LOADED_SPK.try_read().map_err(Error::from)?;
        let ssb_state = spk.try_to_ssb(self.state.state.clone())?;
        let ssb_us = UncertainState::<Equatorial, SSB>::new(
            ssb_state,
            self.state.cov_matrix.clone(),
            self.state.free_params.clone(),
        )?;
        let div = if include_asteroids {
            let extended = GravParams::selected_masses();
            let forces = self.build_forces(&spk, &extended);
            sigma_point_divergence(&ssb_us, &forces, jd.into(), n_axes, sigma_factor)?
        } else {
            let planets = GravParams::planets();
            let forces = self.build_forces(&spk, &planets);
            sigma_point_divergence(&ssb_us, &forces, jd.into(), n_axes, sigma_factor)?
        };
        Ok(div)
    }

    /// String representation.
    fn __repr__(&self) -> String {
        let n = self.state.cov_matrix.nrows();
        format!(
            "UncertainState(desig={}, epoch={:.6}, params={})",
            self.state.state.desig, self.state.state.epoch.jd, n,
        )
    }
}
