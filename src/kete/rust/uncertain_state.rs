//! Python wrapper for [`kete_fitting::UncertainState`].
//!
//! Exposes the Rust-side `UncertainState` to Python as a frozen pyclass
//! with getters for the state, covariance matrix, non-grav model, and
//! convenience methods for sampling and construction.

use crate::elements::PyCometElements;
use crate::nongrav::PyNonGravModel;
use crate::state::PyState;
use crate::time::PyTime;
use kete_core::prelude::*;
use kete_fitting::UncertainState;
use kete_spice::spk::LOADED_SPK;
use nalgebra::DMatrix;
use pyo3::prelude::*;

/// Uncertain orbit state: a best-fit Cartesian state together with a
/// covariance matrix that may span the 6 position/velocity components
/// and any fitted non-gravitational parameters.
///
/// This is the canonical representation of orbit uncertainty in kete,
/// providing correct handling of non-gravitational model templates
/// (preserving fixed physical parameters such as ``alpha``, ``r_0``, etc.).
///
/// Construction
/// ------------
/// - :meth:`from_state` -- from a :class:`~kete.State` with isotropic uncertainties.
/// - :meth:`from_cometary` -- from cometary orbital elements and an
///   element-space covariance (e.g. from JPL Horizons).
/// - Returned as part of :class:`~kete.fitting.OrbitFit` from orbit fitting.
#[pyclass(frozen, module = "kete", name = "UncertainState", from_py_object)]
#[derive(Debug, Clone)]
pub struct PyUncertainState(pub UncertainState);

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
        if eq_state.center_id != 0 {
            let spk = LOADED_SPK.try_read().map_err(Error::from)?;
            spk.try_change_center(&mut eq_state, 0)?;
        }
        let ng = non_grav.map(|m| m.0);
        let np = ng.as_ref().map_or(0, NonGravModel::n_free_params);
        let d = 6 + np;
        let mut cov = DMatrix::<f64>::zeros(d, d);
        for i in 0..3 {
            cov[(i, i)] = pos_sigma * pos_sigma;
        }
        for i in 3..6 {
            cov[(i, i)] = vel_sigma * vel_sigma;
        }
        // Tiny diagonal for non-grav params (so the matrix is positive-definite).
        for i in 6..d {
            cov[(i, i)] = 1e-30;
        }
        let us = UncertainState::new(eq_state, cov, ng)?;
        Ok(Self(us))
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
        let ng = non_grav.map(|m| m.0);
        let us = UncertainState::from_cometary(&elements.0, &mat, ng)?;
        Ok(Self(us))
    }

    /// Best-fit state at the reference epoch (Sun-centered, Ecliptic).
    #[getter]
    fn state(&self) -> PyResult<PyState> {
        let mut st = self.0.state.clone();
        if st.center_id != 10 {
            let spk = LOADED_SPK.try_read().map_err(Error::from)?;
            spk.try_change_center(&mut st, 10)?;
        }
        Ok(st.into())
    }

    /// Covariance matrix as a list of lists (use ``np.array()`` to convert).
    #[getter]
    fn cov_matrix(&self) -> Vec<Vec<f64>> {
        let n = self.0.cov_matrix.nrows();
        let m = self.0.cov_matrix.ncols();
        (0..n)
            .map(|r| (0..m).map(|c| self.0.cov_matrix[(r, c)]).collect())
            .collect()
    }

    /// Non-gravitational model template, or None.
    #[getter]
    fn non_grav(&self) -> Option<PyNonGravModel> {
        self.0.non_grav.clone().map(PyNonGravModel)
    }

    /// Object designator (shortcut for ``self.state.desig``).
    #[getter]
    fn desig(&self) -> String {
        self.0.state.desig.to_string()
    }

    /// Reference epoch as a :class:`~kete.Time` (shortcut for ``self.state.epoch``).
    #[getter]
    fn epoch(&self) -> PyTime {
        self.0.state.epoch.jd.into()
    }

    /// Names of all parameters in the covariance matrix, in row/column
    /// order.
    ///
    /// Always starts with ``["x", "y", "z", "vx", "vy", "vz"]``,
    /// followed by any non-gravitational parameter names.
    #[getter]
    fn param_names(&self) -> Vec<String> {
        self.0.param_names().into_iter().map(String::from).collect()
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
    fn sample(
        &self,
        n_samples: usize,
        seed: Option<u64>,
    ) -> PyResult<(Vec<PyState>, Vec<Option<PyNonGravModel>>)> {
        let samples = self.0.sample(n_samples, seed)?;
        let spk = LOADED_SPK.try_read().map_err(Error::from)?;
        let mut states = Vec::with_capacity(n_samples);
        let mut non_gravs = Vec::with_capacity(n_samples);
        for (mut st, ng) in samples {
            // Re-center to Sun for the Python-facing state.
            if st.center_id != 10 {
                spk.try_change_center(&mut st, 10)?;
            }
            let py_st: PyState = st.into();
            states.push(py_st);
            non_gravs.push(ng.map(PyNonGravModel));
        }
        Ok((states, non_gravs))
    }

    /// String representation.
    fn __repr__(&self) -> String {
        let n = self.0.cov_matrix.nrows();
        format!(
            "UncertainState(desig={}, epoch={:.6}, params={})",
            self.0.state.desig, self.0.state.epoch.jd, n,
        )
    }
}
