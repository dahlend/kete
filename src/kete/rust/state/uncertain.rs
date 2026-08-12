//! Python wrapper for [`kete_core::state::UncertainState`].
//!
//! Bridges between the Rust state shape and the Python user-facing shape,
//! whose `non_grav` property returns a `NonGravModel` rather than the mask the
//! core stores. The wrapper holds nothing of its own: the covariance, the
//! `free_params` and the model they belong to all live on the core state.

use super::PyState;
use crate::elements::{PyCometElements, PyEquinoctialElements};
use crate::nongrav::PyNonGravModel;
use crate::time::PyTime;
use kete_core::forces::ParameterizedForce;
use kete_core::frames::{Ecliptic, Equatorial};
use kete_core::prelude::*;
use kete_spice::propagation::SpkNonGravs;
use kete_spice::propagation::propagate_uncertain;
use kete_spice::spk::LOADED_SPK;
use nalgebra::DMatrix;
use nalgebra::Vector3;
use pyo3::prelude::*;
use std::f64::consts::PI;

/// Row and column of the true longitude in the element covariance.
///
/// The Python interface reports angles in degrees and the core works in radians, so
/// this is the one row of the element covariance that is rescaled at the boundary.
const TRUE_LON_ROW: usize = 5;

/// Rows of the cometary element covariance that hold an angle: the longitude of the
/// ascending node, the argument of perihelion, and the inclination.
const COMETARY_ANGLE_ROWS: [usize; 3] = [3, 4, 5];

/// Uncertain orbit state: a best-fit orbit together with a covariance matrix
/// spanning that orbit's six coordinates and any fitted non-gravitational
/// parameters.
///
/// The best-fit orbit is stored as modified equinoctial orbital elements, and
/// :attr:`cov_matrix` is a covariance over those six elements rather than over
/// cartesian position and velocity. See :attr:`cov_matrix` for the coordinates and
/// the ordering, :attr:`state` for the best-fit orbit as a
/// :class:`~kete.State`, and :attr:`cartesian_cov_matrix` for the covariance in
/// position and velocity.
///
/// `state.non_grav` stores an all-`None` [`ParameterMask`] wrapping the typed
/// ParameterizedForce template; free-parameter values live in
/// `state.free_params`, not in the mask.
#[pyclass(frozen, module = "kete", name = "UncertainState", from_py_object)]
#[derive(Clone)]
pub struct PyUncertainState {
    /// Underlying state with its covariance, free-parameter values and the model
    /// those parameters belong to.
    pub state: UncertainState,
}

impl std::fmt::Debug for PyUncertainState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PyUncertainState")
            .field("state", &self.state)
            .field("non_grav_present", &self.state.non_grav.is_some())
            .finish()
    }
}

/// Resolve the central values of a model's free parameters.
///
/// The mask says WHICH parameters are free; this says what they currently
/// equal. `None` falls back to the model's own starting values, which are
/// zero for every free parameter -- fine for seeding a fit, wrong for a
/// physical cloud, hence the explicit override.
fn resolve_free_params(
    non_grav: &Option<PyNonGravModel>,
    supplied: Option<Vec<f64>>,
) -> PyResult<Vec<f64>> {
    let defaults = non_grav
        .as_ref()
        .map(PyNonGravModel::initial_values)
        .unwrap_or_default();
    let Some(values) = supplied else {
        return Ok(defaults);
    };
    if values.len() != defaults.len() {
        return Err(Error::ValueError(format!(
            "free_params has length {}, but the model has {} free parameter(s); \
             free parameters are those passed as float(\"nan\")",
            values.len(),
            defaults.len()
        ))
        .into());
    }
    if let Some(bad) = values.iter().find(|v| !v.is_finite()) {
        return Err(Error::ValueError(format!("free_params must all be finite, got {bad}")).into());
    }
    Ok(values)
}

impl PyUncertainState {
    /// Build the SSB-centered force model used by all propagation paths.
    ///
    /// Gravity-only `SpkNBody` when `non_grav` is `None`, or
    /// `Sum<SpkNBody, Recenter<SSB, _>>` wrapping the non-grav force
    /// template otherwise. Captures the `LOADED_SPK` read guard so the
    /// borrow lifetime is well-defined.
    fn build_forces<'a>(
        &self,
        spk: &'a kete_spice::spk::SpkCollection,
        include_extended: bool,
    ) -> SpkNonGravs<'a> {
        if let Some(ref ng) = self.state.non_grav {
            SpkNonGravs::with_non_grav_mask(spk, include_extended, ng.clone())
        } else {
            SpkNonGravs::gravity(spk, include_extended)
        }
    }
}

impl PyUncertainState {
    /// Resolves the Sun against the barycenter for the propagation paths below.
    ///
    /// Elements are referred to the Sun while the force models are barycentric, so a state
    /// crosses between them at every epoch the propagation touches.
    fn sun_resolver(
        spk: &kete_spice::spk::SpkCollection,
    ) -> impl Fn(Time<TDB>) -> KeteResult<(Vector3<f64>, Vector3<f64>)> + Sync + '_ {
        move |time| {
            let sun = spk.try_get_state_with_center::<Equatorial>(10, time, 0)?;
            Ok((Vector3::from(sun.pos), Vector3::from(sun.vel)))
        }
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
    /// The input state is automatically re-centered on the Sun if needed, since
    /// orbital elements are defined about a gravitating body. That re-centering is a
    /// translation by a function of time, so it leaves the covariance unchanged.
    ///
    /// Parameters
    /// ----------
    /// state : :class:`~kete.State`
    ///     Object state (any center / frame -- will be converted to Sun-centered
    ///     internally).
    /// pos_sigma : float
    ///     1-sigma position uncertainty in AU (default 0.01).
    /// vel_sigma : float
    ///     1-sigma velocity uncertainty in AU/day (default 0.0001).
    /// non_grav : :class:`~kete.propagation.NonGravModel`, optional
    ///     Non-gravitational model template.  Parameters left free (passed as
    ///     ``float("nan")``) extend the covariance to (6+Np)x(6+Np).
    /// free_params : list[float], optional
    ///     Central values of the free force parameters, in the order given by
    ///     :attr:`param_names`.  Defaults to zero for each, which is rarely
    ///     what a physical cloud means -- a free dust ``beta`` of zero is a
    ///     grain that feels no radiation pressure -- so supply this whenever
    ///     the parameter is free.
    /// param_sigmas : list[float], optional
    ///     1-sigma uncertainty of each free parameter.  Defaults to a
    ///     negligible value, making the parameter effectively fixed at its
    ///     central value while still occupying a covariance row.
    #[staticmethod]
    #[pyo3(signature = (state, pos_sigma=0.01, vel_sigma=0.0001, non_grav=None,
                        free_params=None, param_sigmas=None))]
    fn from_state(
        state: PyState,
        pos_sigma: f64,
        vel_sigma: f64,
        non_grav: Option<PyNonGravModel>,
        free_params: Option<Vec<f64>>,
        param_sigmas: Option<Vec<f64>>,
    ) -> PyResult<Self> {
        if pos_sigma <= 0.0 || vel_sigma <= 0.0 {
            return Err(
                Error::ValueError("pos_sigma and vel_sigma must be positive".into()).into(),
            );
        }
        // Elements are defined about a gravitating body, so this centers on the Sun
        // rather than the barycenter. The propagation paths cross to the force model's
        // center themselves, see `sun_resolver`.
        let mut eq_state = state.raw;
        if eq_state.center_id() != 10 {
            let spk = LOADED_SPK.try_read().map_err(Error::from)?;
            spk.try_change_center(&mut eq_state, 10)?;
        }
        let ng_mask = non_grav.as_ref().map(|m| m.to_mask());
        let free_params = resolve_free_params(&non_grav, free_params)?;
        let np = free_params.len();
        if let Some(ref sig) = param_sigmas
            && sig.len() != np
        {
            return Err(Error::ValueError(format!(
                "param_sigmas has length {}, but the model has {np} free parameter(s)",
                sig.len()
            ))
            .into());
        }
        let d = 6 + np;
        let mut cov = DMatrix::<f64>::zeros(d, d);
        for i in 0..3 {
            cov[(i, i)] = pos_sigma * pos_sigma;
        }
        for i in 3..6 {
            cov[(i, i)] = vel_sigma * vel_sigma;
        }
        // A tiny default keeps the matrix positive-definite when the caller
        // wants the parameter carried but not spread.
        for i in 6..d {
            cov[(i, i)] = match param_sigmas {
                Some(ref sig) => {
                    let s = sig[i - 6];
                    if !s.is_finite() || s < 0.0 {
                        return Err(Error::ValueError(
                            "param_sigmas must be finite and non-negative".into(),
                        )
                        .into());
                    }
                    (s * s).max(1e-30)
                }
                None => 1e-30,
            };
        }
        let mut us = UncertainState::from_state(&eq_state, &cov, free_params)?;
        us.non_grav = ng_mask;
        Ok(Self { state: us })
    }

    /// Build an ``UncertainState`` from a state and a full cartesian covariance
    /// matrix.
    ///
    /// This is the matrix-valued counterpart of :meth:`from_state` and the inverse of
    /// :attr:`cartesian_cov_matrix`: the covariance uses the same convention, rows and
    /// columns 0-5 being ``[x, y, z, vx, vy, vz]`` in AU and AU/day, Sun-centered and
    /// **ecliptic**, the frame :attr:`state` reports in. Rows 6 onward are the fitted
    /// force parameters in the order given by :attr:`param_names`.
    ///
    /// The input state is automatically re-centered on the Sun if needed, since
    /// orbital elements are defined about a gravitating body. That re-centering is a
    /// translation by a function of time, so it leaves the covariance unchanged.
    ///
    /// Use :meth:`conversion_divergence` on the result to bound how faithfully the
    /// stored element-space covariance represents the cartesian one supplied here.
    ///
    /// Parameters
    /// ----------
    /// state : :class:`~kete.State`
    ///     Object state (any center -- will be converted to Sun-centered internally).
    /// cov_matrix : list[list[float]]
    ///     Cartesian covariance matrix, (6+Np)x(6+Np), in the ecliptic frame.
    /// non_grav : :class:`~kete.propagation.NonGravModel`, optional
    ///     Non-gravitational model template. Required when the covariance carries
    ///     rows beyond the sixth; those rows correspond to the parameters left
    ///     free (passed as ``float("nan")``), in :attr:`param_names` order.
    /// free_params : list[float], optional
    ///     Central values of the free force parameters. Defaults to zero for
    ///     each, which for a free dust ``beta`` means a grain feeling no
    ///     radiation pressure -- supply this whenever a parameter is free.
    #[staticmethod]
    #[pyo3(signature = (state, cov_matrix, non_grav=None, free_params=None))]
    fn from_cartesian(
        state: PyState,
        cov_matrix: Vec<Vec<f64>>,
        non_grav: Option<PyNonGravModel>,
        free_params: Option<Vec<f64>>,
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

        // Elements are defined about a gravitating body, so this centers on the Sun
        // rather than the barycenter; the offset is a function of time alone and does
        // not touch the covariance.
        let mut eq_state = state.raw;
        if eq_state.center_id() != 10 {
            let spk = LOADED_SPK.try_read().map_err(Error::from)?;
            spk.try_change_center(&mut eq_state, 10)?;
        }
        // The covariance convention is ecliptic, so the state crosses to that frame
        // before the pair enters the core conversion together.
        let ecl_state: State<Ecliptic> = eq_state.into_frame();

        let ng_mask = non_grav.as_ref().map(|m| m.to_mask());
        let free_params = resolve_free_params(&non_grav, free_params)?;
        let mut us = UncertainState::from_state(&ecl_state, &mat, free_params)?;
        us.non_grav = ng_mask;
        Ok(Self { state: us })
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
    ///     Element order: ``[e, q, tp, node, w, i, <nongrav...>]``, with ``node``,
    ///     ``w`` and ``i`` in degrees to match :class:`~kete.CometElements`. Their
    ///     variances are therefore in degrees squared and their cross terms in
    ///     degrees.
    /// non_grav : :class:`~kete.propagation.NonGravModel`, optional
    ///     Non-gravitational model template.
    /// free_params : list[float], optional
    ///     Central values of the free force parameters, in
    ///     :attr:`param_names` order. Defaults to zero for each.
    #[staticmethod]
    #[pyo3(signature = (elements, cov_matrix, non_grav=None, free_params=None))]
    fn from_cometary(
        elements: PyCometElements,
        cov_matrix: Vec<Vec<f64>>,
        non_grav: Option<PyNonGravModel>,
        free_params: Option<Vec<f64>>,
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
        // The angle rows arrive in degrees, matching `CometElements` itself, and the
        // core works in radians.
        let to_radians = |i: usize| {
            if COMETARY_ANGLE_ROWS.contains(&i) {
                PI / 180.0
            } else {
                1.0
            }
        };
        let mat = DMatrix::from_fn(n, n, |r, c| {
            cov_matrix[r][c] * to_radians(r) * to_radians(c)
        });
        let ng_mask = non_grav.as_ref().map(|m| m.to_mask());
        let free_params = resolve_free_params(&non_grav, free_params)?;
        let mut us = UncertainState::from_cometary(&elements.0, &mat, free_params)?;
        us.non_grav = ng_mask;
        Ok(Self { state: us })
    }

    /// How faithfully the stored covariance and its cartesian image describe the same
    /// distribution, as a sigma-point divergence.
    ///
    /// The conversion between the element and cartesian bases is exact as a linear
    /// map, but a Gaussian in one basis is not a Gaussian in the other: the change
    /// of coordinates is nonlinear off the mean. Probe points placed
    /// ``sigma_factor`` standard deviations out along the cartesian covariance's
    /// principal axes are carried through the exact nonlinear change of coordinates
    /// and compared against the linear image; the result is how many sigma the
    /// exact answer sits from the linear one.
    ///
    /// This asks a different question from :meth:`nonlinearity`, which measures the
    /// propagation rather than the coordinate change, and the two are not on a common
    /// scale: this one probes at ``sigma_factor`` on the cartesian principal axes, and
    /// the propagation metric probes every element direction on its own width.
    ///
    /// Call this immediately after building the state from a fitted or catalog
    /// covariance to bound the loss of the entry conversion itself. A value well
    /// below 1 means the initial Gaussian is faithful, so structure that develops
    /// during propagation is dynamical rather than an entry artifact. A larger value means the uncertainty is already too
    /// wide for a single Gaussian and should be split or sampled instead. Infinity
    /// means it reaches configurations orbital elements cannot represent at all.
    ///
    /// Parameters
    /// ----------
    /// sigma_factor : float
    ///     How many standard deviations out to place the probe points (default 1).
    #[pyo3(signature = (sigma_factor=1.0))]
    fn conversion_divergence(&self, sigma_factor: f64) -> PyResult<f64> {
        Ok(self.state.conversion_divergence(sigma_factor)?)
    }

    /// Best-fit state at the reference epoch (Sun-centered, Ecliptic).
    #[getter]
    fn state(&self) -> PyResult<PyState> {
        // The elements are referred to their central body already, so this is a direct
        // reconstruction rather than a re-centering.
        Ok(self.state.state::<Equatorial>()?.into())
    }

    /// Best-fit orbit as an :class:`~kete.EquinoctialElements`, the basis
    /// :attr:`cov_matrix` is a covariance over.
    ///
    /// This is the mean the covariance is centered on: :attr:`state` gives the same
    /// orbit converted to cartesian, and :attr:`cov_matrix` gives the spread around
    /// this point in these coordinates.
    #[getter]
    fn elements(&self) -> PyEquinoctialElements {
        PyEquinoctialElements(self.state.elements.clone())
    }

    /// Covariance matrix as a list of lists (use ``np.array()`` to convert).
    ///
    /// .. warning::
    ///     Rows and columns 0-5 are **modified equinoctial orbital elements, not
    ///     cartesian position and velocity**. Use :attr:`cartesian_cov_matrix` for the
    ///     cartesian form.
    ///
    /// The six element coordinates, in order and in the ecliptic frame, are the
    /// semi-latus rectum ``p`` in AU, the two eccentricity components
    /// ``f = e cos(w + node)`` and ``g = e sin(w + node)``, the two pole components
    /// ``h = tan(i/2) cos(node)`` and ``k = tan(i/2) sin(node)``, and the true
    /// longitude at the epoch ``L = node + w + nu`` in degrees. All six are
    /// dimensionless except the first, in AU, and the last, in degrees. Rows 6 onward
    /// are the fitted force parameters, unchanged in meaning and in the order given by
    /// :attr:`param_names`.
    ///
    /// The ``L`` row and column are reported in degrees to match
    /// :attr:`~kete.EquinoctialElements.true_lon` and every other angle in the Python
    /// interface, so the variance there is in degrees squared and its cross terms in
    /// degrees. The Rust core holds the same matrix in radians.
    ///
    /// The covariance is stored this way because a cartesian covariance stops
    /// describing the distribution within a fraction of an orbit, as the distribution
    /// shears into a curved shape no linear map can represent. These coordinates stay
    /// accurate considerably longer.
    ///
    /// The true longitude wraps. Differencing two of these covariances, or averaging
    /// their means, has to reduce that coordinate to the shortest signed angle.
    #[getter]
    fn cov_matrix(&self) -> Vec<Vec<f64>> {
        let n = self.state.cov_matrix.nrows();
        let m = self.state.cov_matrix.ncols();
        // The core holds `L` in radians. Rescaling the row and the column carries
        // the variance to degrees squared and every cross term to degrees, which is
        // the same matrix in the units the Python interface states.
        let scale = |i: usize| if i == TRUE_LON_ROW { 180.0 / PI } else { 1.0 };
        (0..n)
            .map(|r| {
                (0..m)
                    .map(|c| self.state.cov_matrix[(r, c)] * scale(r) * scale(c))
                    .collect()
            })
            .collect()
    }

    /// Covariance matrix in cartesian position and velocity, as a list of lists.
    ///
    /// Rows and columns 0-5 are ``[x, y, z, vx, vy, vz]`` in AU and AU/day, Sun-centered
    /// and **ecliptic**, which is the frame :attr:`state` reports its position and
    /// velocity in, so the two can be used together directly. Rows 6 onward are the
    /// fitted force parameters. This is the form to use for reporting, for interchange,
    /// and for comparison against an externally supplied covariance.
    ///
    /// The conversion from the stored element coordinates is exact as a linear map.
    /// What it cannot recover is accuracy already lost: a cartesian covariance is a
    /// much poorer description of the same distribution over a long arc, which is
    /// why it is not the stored form.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If the orbit lies outside the domain where the conversion exists.
    #[getter]
    fn cartesian_cov_matrix(&self) -> PyResult<Vec<Vec<f64>>> {
        let cov = self.state.cartesian_covariance::<Ecliptic>()?;
        let (n, m) = (cov.nrows(), cov.ncols());
        Ok((0..n)
            .map(|r| (0..m).map(|c| cov[(r, c)]).collect())
            .collect())
    }

    /// Non-gravitational model template, or None.
    ///
    /// The template carries the model variant and fixed coefficients
    /// (g(r) shape, albedo, spin pole, etc.). The fitted parameter
    /// values are stored on the underlying state's `free_params` and
    /// merged here to reconstruct the full model.
    #[getter]
    fn non_grav(&self) -> Option<PyNonGravModel> {
        self.state.non_grav.as_ref().and_then(|mask| {
            let full = mask.merge(&self.state.free_params).ok()?;
            PyNonGravModel::from_force(&mask.inner, &full)
        })
    }

    /// Object designator (shortcut for ``self.state.desig``).
    #[getter]
    fn desig(&self) -> String {
        self.state.elements.desig.to_string()
    }

    /// Reference epoch as a :class:`~kete.Time` (shortcut for ``self.state.epoch``).
    #[getter]
    fn epoch(&self) -> PyTime {
        self.state.elements.epoch.jd.into()
    }

    /// Departure from linearity this component is carrying, in sigma of its own
    /// propagated position distribution, or ``None`` if it has never been marched.
    ///
    /// Set by :meth:`kete.DiffuseState.step` and :meth:`kete.DiffuseState.propagate`, and
    /// measured against probes carried since this component last split - so it says how
    /// far the component is from linear now, not what the last leg added.  Read it with
    /// :attr:`residual_meters`.
    #[getter]
    fn eta(&self) -> Option<f64> {
        self.state.eta
    }

    /// The residual behind :attr:`eta`, as a cartesian position offset in meters, or
    /// ``None`` if this component has never been marched.
    ///
    /// The propagator places a position to roughly a meter, so an ``eta`` of ``0.003``
    /// standing on a residual of ``1.2`` m is numerical noise rather than curvature.
    #[getter]
    fn residual_meters(&self) -> Option<f64> {
        self.state.residual_meters
    }

    /// Names of all parameters in the covariance matrix, in row/column
    /// order.
    ///
    /// Always starts with ``["p", "f", "g", "h", "k", "L"]``, the modified equinoctial
    /// elements described on :attr:`cov_matrix`, with ``L`` in degrees, followed by
    /// any non-gravitational parameter names.
    #[getter]
    fn param_names(&self) -> Vec<String> {
        let mut names: Vec<String> = vec!["p", "f", "g", "h", "k", "L"]
            .into_iter()
            .map(String::from)
            .collect();
        if let Some(ref ng) = self.state.non_grav {
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
        let samples = self.state.sample::<Equatorial>(n_samples, seed)?;
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
            let ng = self.state.non_grav.as_ref().and_then(|mask| {
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
    /// ``(6 + Np) x (6 + Np)`` state transition matrix.  The result's
    /// :attr:`state` is Sun-centered and ecliptic, as it is on the input;
    /// the integration crosses to the force model's center internally.
    ///
    /// Parameters
    /// ----------
    /// jd : :class:`~kete.Time` or float
    ///     Target epoch (TDB).
    /// include_asteroids : bool, optional
    ///     If True, include asteroid masses in the force model.
    #[pyo3(signature = (jd, include_asteroids=false))]
    fn propagate(&self, py: Python<'_>, jd: PyTime, include_asteroids: bool) -> PyResult<Self> {
        let target: Time<TDB> = jd.into();
        py.detach(|| {
            let spk = LOADED_SPK.try_read().map_err(Error::from)?;
            let forces = self.build_forces(&spk, include_asteroids);
            let result =
                propagate_uncertain(&self.state, &forces, target, &Self::sun_resolver(&spk))?;
            Ok(Self { state: result })
        })
    }

    /// Save this state to a file.
    ///
    /// The file keeps the covariance, the free-parameter values, the
    /// non-gravitational model those parameters belong to, and any probes the
    /// state was carrying. A state loaded back is the state that was saved, so
    /// a march can continue from it without restarting its ``eta``.
    ///
    /// Use :meth:`save_list` when saving more than one. A directory of
    /// single-state files costs a file and a header for each.
    ///
    /// Parameters
    /// ----------
    /// filename :
    ///     Path to write. The format is the gzipped kete binary format.
    fn save(&self, filename: String) -> PyResult<()> {
        self.state.save(filename)?;
        Ok(())
    }

    /// Load a single state from a file.
    ///
    /// Parameters
    /// ----------
    /// filename :
    ///     Path to read.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If the file holds several states, or a different type. Use
    ///     :meth:`load_list` for a file holding several.
    #[staticmethod]
    fn load(filename: String) -> PyResult<Self> {
        Ok(Self {
            state: UncertainState::load(filename)?,
        })
    }

    /// Save many states to one file.
    ///
    /// Parameters
    /// ----------
    /// states :
    ///     States to save. They do not have to share an epoch or a model.
    /// filename :
    ///     Path to write.
    #[staticmethod]
    fn save_list(states: Vec<Self>, filename: String) -> PyResult<()> {
        let states: Vec<UncertainState> = states.into_iter().map(|s| s.state).collect();
        UncertainState::save_vec(&states, filename)?;
        Ok(())
    }

    /// Load many states from a file.
    ///
    /// A file holding a single state reads back as a list of one.
    ///
    /// Parameters
    /// ----------
    /// filename :
    ///     Path to read.
    #[staticmethod]
    fn load_list(filename: String) -> PyResult<Vec<Self>> {
        Ok(UncertainState::load_vec(filename)?
            .into_iter()
            .map(|state| Self { state })
            .collect())
    }

    /// String representation.
    fn __repr__(&self) -> String {
        let n = self.state.cov_matrix.nrows();
        format!(
            "UncertainState(desig={}, epoch={:.6}, params={})",
            self.state.elements.desig, self.state.elements.epoch.jd, n,
        )
    }
}
