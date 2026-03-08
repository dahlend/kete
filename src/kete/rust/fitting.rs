//! Python bindings for orbit determination and fitting.
//!
//! Wraps `kete_fitting` types and functions for use from Python.

use kete_core::prelude::*;
use kete_core::spice::LOADED_SPK;
use kete_fitting::{Observation, OrbitFit, OrbitSamples, differential_correction, nuts_sample};
use pyo3::{PyResult, pyclass, pyfunction, pymethods};

use crate::nongrav::PyNonGravModel;
use crate::state::PyState;
use crate::time::PyTime;
use crate::uncertain_state::PyUncertainState;

/// Astronomical observation for orbit determination.
///
/// Observations can be optical (RA/Dec), radar range, or radar range-rate.
/// Each carries the observer state, measured value(s), and 1-sigma uncertainties.
///
/// Use the static methods to construct instances:
///
/// - :py:meth:`Observation.optical`
/// - :py:meth:`Observation.radar_range`
/// - :py:meth:`Observation.radar_rate`
#[pyclass(frozen, module = "kete.fitting", name = "Observation")]
#[derive(Debug, Clone)]
pub struct PyObservation(pub Observation);

#[pymethods]
impl PyObservation {
    /// Unused default constructor.
    #[new]
    fn new() -> PyResult<Self> {
        Err(Error::ValueError(
            "Use Observation.optical(), Observation.radar_range(), or \
             Observation.radar_rate() to create observations."
                .into(),
        ))?
    }

    /// Create an optical (RA/Dec) observation.
    ///
    /// The observer state is automatically re-centered to the solar
    /// system barycenter if needed (the fitting engine works
    /// internally in SSB-centered coordinates).
    ///
    /// Parameters
    /// ----------
    /// observer : State
    ///     Observer state (any center / frame - will be converted to
    ///     SSB-centered Equatorial internally). The observation epoch
    ///     is taken from the observer's epoch.
    /// ra : float
    ///     Right ascension in degrees.
    /// dec : float
    ///     Declination in degrees.
    /// sigma_ra : float
    ///     1-sigma RA uncertainty in arcseconds (should include cos(dec)
    ///     factor).
    /// sigma_dec : float
    ///     1-sigma Dec uncertainty in arcseconds.
    #[staticmethod]
    #[pyo3(signature = (observer, ra, dec, sigma_ra, sigma_dec))]
    fn optical(
        observer: PyState,
        ra: f64,
        dec: f64,
        sigma_ra: f64,
        sigma_dec: f64,
    ) -> PyResult<Self> {
        let mut raw = observer.raw;
        if raw.center_id != 0 {
            let spk = LOADED_SPK.try_read().map_err(Error::from)?;
            spk.try_change_center(&mut raw, 0)?;
        }
        let arcsec_to_rad = std::f64::consts::PI / (180.0 * 3600.0);
        Ok(Self(Observation::Optical {
            observer: raw,
            ra: ra.to_radians(),
            dec: dec.to_radians(),
            sigma_ra: sigma_ra * arcsec_to_rad,
            sigma_dec: sigma_dec * arcsec_to_rad,
        }))
    }

    /// Create a radar range observation.
    ///
    /// The observer state is automatically re-centered to SSB.
    ///
    /// Parameters
    /// ----------
    /// observer : State
    ///     Observer state (any center / frame -- will be converted).
    /// range : float
    ///     Measured range in AU.
    /// sigma_range : float
    ///     1-sigma range uncertainty in AU.
    #[staticmethod]
    #[pyo3(signature = (observer, range, sigma_range))]
    fn radar_range(observer: PyState, range: f64, sigma_range: f64) -> PyResult<Self> {
        let mut raw = observer.raw;
        if raw.center_id != 0 {
            let spk = LOADED_SPK.try_read().map_err(Error::from)?;
            spk.try_change_center(&mut raw, 0)?;
        }
        Ok(Self(Observation::RadarRange {
            observer: raw,
            range,
            sigma_range,
        }))
    }

    /// Create a radar range-rate (Doppler) observation.
    ///
    /// The observer state is automatically re-centered to SSB.
    ///
    /// Parameters
    /// ----------
    /// observer : State
    ///     Observer state (any center / frame -- will be converted).
    /// range_rate : float
    ///     Measured range-rate in AU/day (positive = receding).
    /// sigma_range_rate : float
    ///     1-sigma range-rate uncertainty in AU/day.
    #[staticmethod]
    #[pyo3(signature = (observer, range_rate, sigma_range_rate))]
    fn radar_rate(observer: PyState, range_rate: f64, sigma_range_rate: f64) -> PyResult<Self> {
        let mut raw = observer.raw;
        if raw.center_id != 0 {
            let spk = LOADED_SPK.try_read().map_err(Error::from)?;
            spk.try_change_center(&mut raw, 0)?;
        }
        Ok(Self(Observation::RadarRate {
            observer: raw,
            range_rate,
            sigma_range_rate,
        }))
    }

    /// The observation epoch (from the observer state).
    #[getter]
    fn epoch(&self) -> PyTime {
        self.0.epoch().jd.into()
    }

    /// The observer state (Sun-centered, Ecliptic).
    #[getter]
    fn observer(&self) -> PyResult<PyState> {
        let mut st = match &self.0 {
            Observation::Optical { observer, .. }
            | Observation::RadarRange { observer, .. }
            | Observation::RadarRate { observer, .. } => observer.clone(),
        };
        if st.center_id != 10 {
            let spk = LOADED_SPK.try_read().map_err(Error::from)?;
            spk.try_change_center(&mut st, 10)?;
        }
        Ok(st.into())
    }

    /// Right ascension in degrees (optical only, None otherwise).
    #[getter]
    fn ra(&self) -> Option<f64> {
        match &self.0 {
            Observation::Optical { ra, .. } => Some(ra.to_degrees()),
            _ => None,
        }
    }

    /// Declination in degrees (optical only, None otherwise).
    #[getter]
    fn dec(&self) -> Option<f64> {
        match &self.0 {
            Observation::Optical { dec, .. } => Some(dec.to_degrees()),
            _ => None,
        }
    }

    /// 1-sigma RA uncertainty in arcseconds (optical only, None otherwise).
    #[getter]
    fn sigma_ra(&self) -> Option<f64> {
        let rad_to_arcsec = 180.0 * 3600.0 / std::f64::consts::PI;
        match &self.0 {
            Observation::Optical { sigma_ra, .. } => Some(*sigma_ra * rad_to_arcsec),
            _ => None,
        }
    }

    /// 1-sigma Dec uncertainty in arcseconds (optical only, None otherwise).
    #[getter]
    fn sigma_dec(&self) -> Option<f64> {
        let rad_to_arcsec = 180.0 * 3600.0 / std::f64::consts::PI;
        match &self.0 {
            Observation::Optical { sigma_dec, .. } => Some(*sigma_dec * rad_to_arcsec),
            _ => None,
        }
    }

    /// Measured range in AU (radar range only, None otherwise).
    #[getter]
    fn range(&self) -> Option<f64> {
        match &self.0 {
            Observation::RadarRange { range, .. } => Some(*range),
            _ => None,
        }
    }

    /// 1-sigma range uncertainty in AU (radar range only, None otherwise).
    #[getter]
    fn sigma_range(&self) -> Option<f64> {
        match &self.0 {
            Observation::RadarRange { sigma_range, .. } => Some(*sigma_range),
            _ => None,
        }
    }

    /// Measured range-rate in AU/day (radar rate only, None otherwise).
    #[getter]
    fn range_rate(&self) -> Option<f64> {
        match &self.0 {
            Observation::RadarRate { range_rate, .. } => Some(*range_rate),
            _ => None,
        }
    }

    /// 1-sigma range-rate uncertainty in AU/day (radar rate only, None otherwise).
    #[getter]
    fn sigma_range_rate(&self) -> Option<f64> {
        match &self.0 {
            Observation::RadarRate {
                sigma_range_rate, ..
            } => Some(*sigma_range_rate),
            _ => None,
        }
    }

    /// String representation.
    fn __repr__(&self) -> String {
        let epoch = self.0.epoch().jd;
        match &self.0 {
            Observation::Optical { ra, dec, .. } => {
                format!(
                    "Observation.optical(epoch={:.6}, ra={:.8}, dec={:.8})",
                    epoch,
                    ra.to_degrees(),
                    dec.to_degrees()
                )
            }
            Observation::RadarRange { range, .. } => {
                format!(
                    "Observation.radar_range(epoch={:.6}, range={:.10})",
                    epoch, range
                )
            }
            Observation::RadarRate { range_rate, .. } => {
                format!(
                    "Observation.radar_rate(epoch={:.6}, range_rate={:.10})",
                    epoch, range_rate
                )
            }
        }
    }
}

/// Result of orbit determination via batch least squares.
///
/// Attributes
/// ----------
/// uncertain_state : UncertainState
///     The best-fit uncertain orbit (state + covariance + non-grav model).
/// residuals : list[list[float]]
///     Post-fit residuals for included observations (time-sorted).
/// observations : list[Observation]
///     Observations included in the final fit (rejected outliers excluded).
/// rms : float
///     Reduced weighted RMS of post-fit residuals (included observations
///     only), divided by degrees of freedom.
/// converged : bool
///     Whether the solver achieved strict convergence.
#[pyclass(frozen, module = "kete.fitting", name = "OrbitFit")]
#[derive(Debug, Clone)]
pub struct PyOrbitFit(pub OrbitFit);

#[pymethods]
impl PyOrbitFit {
    /// Build an ``OrbitFit`` directly from a state, without running
    /// differential correction.
    ///
    /// The covariance is initialised to a diagonal matrix with the
    /// given ``pos_sigma`` (AU) and ``vel_sigma`` (AU/day) on the
    /// diagonal.  This is useful for seeding MCMC from an IOD
    /// candidate when the differential corrector fails or converges
    /// to an unphysical orbit.
    ///
    /// The input state is automatically re-centered to the solar
    /// system barycenter if needed (the fitting engine works
    /// internally in SSB-centered Equatorial coordinates).
    ///
    /// Parameters
    /// ----------
    /// state : State
    ///     Object state (any center / frame -- will be converted).
    /// pos_sigma : float, optional
    ///     1-sigma position uncertainty in AU (default 0.01).
    /// vel_sigma : float, optional
    ///     1-sigma velocity uncertainty in AU/day (default 0.0001).
    #[staticmethod]
    #[pyo3(signature = (state, pos_sigma=0.01, vel_sigma=0.0001))]
    fn from_state(state: PyState, pos_sigma: f64, vel_sigma: f64) -> PyResult<Self> {
        let mut eq_state = state.raw;
        // Re-center to SSB (the fitting engine expects center_id=0).
        if eq_state.center_id != 0 {
            let spk = LOADED_SPK.try_read().map_err(Error::from)?;
            spk.try_change_center(&mut eq_state, 0)?;
        }
        let mut cov = nalgebra::DMatrix::<f64>::zeros(6, 6);
        for i in 0..3 {
            cov[(i, i)] = pos_sigma * pos_sigma;
        }
        for i in 3..6 {
            cov[(i, i)] = vel_sigma * vel_sigma;
        }
        let uncertain_state = kete_fitting::UncertainState::new(eq_state, cov, None)?;
        Ok(PyOrbitFit(OrbitFit {
            uncertain_state,
            residuals: Vec::new(),
            observations: Vec::new(),
            rms: f64::NAN,
            converged: false,
        }))
    }

    /// The uncertain orbit state (state + covariance + non-grav model).
    #[getter]
    fn uncertain_state(&self) -> PyUncertainState {
        PyUncertainState(self.0.uncertain_state.clone())
    }

    /// Best-fit state at the reference epoch (Sun-centered, Ecliptic).
    ///
    /// Convenience shortcut for ``self.uncertain_state.state``.
    #[getter]
    fn state(&self) -> PyResult<PyState> {
        let mut st = self.0.uncertain_state.state.clone();
        if st.center_id != 10 {
            let spk = LOADED_SPK.try_read().map_err(Error::from)?;
            spk.try_change_center(&mut st, 10)?;
        }
        Ok(st.into())
    }

    /// Post-fit residuals as a list of lists (included observations only,
    /// time-sorted order).
    ///
    /// For optical observations the two elements are (DeltaRA, DeltaDec) in
    /// **arcseconds**.  Radar residuals remain in AU or AU/day.
    #[getter]
    fn residuals(&self) -> Vec<Vec<f64>> {
        let rad_to_arcsec = 180.0 * 3600.0 / std::f64::consts::PI;
        self.0
            .residuals
            .iter()
            .map(|r| {
                // Optical residuals have 2 elements (RA, Dec) in radians;
                // radar residuals have 1 element in AU or AU/day.
                if r.len() == 2 {
                    r.iter().map(|v| v * rad_to_arcsec).collect()
                } else {
                    r.iter().copied().collect()
                }
            })
            .collect()
    }

    /// Observations included in the final fit (time-sorted).
    ///
    /// Rejected outliers are not present in this list.
    #[getter]
    fn observations(&self) -> Vec<PyObservation> {
        self.0
            .observations
            .iter()
            .map(|o| PyObservation(o.clone()))
            .collect()
    }

    /// Reduced weighted RMS of post-fit residuals.
    #[getter]
    fn rms(&self) -> f64 {
        self.0.rms
    }

    /// Fitted non-gravitational model, or None if not fitted.
    ///
    /// Convenience shortcut for ``self.uncertain_state.non_grav``.
    #[getter]
    fn non_grav(&self) -> Option<PyNonGravModel> {
        self.0.uncertain_state.non_grav.clone().map(PyNonGravModel)
    }

    /// Whether the solver achieved strict convergence.
    ///
    /// When ``False`` the fit is the best found within the iteration
    /// limit but the correction norm did not drop below `tol`.
    #[getter]
    fn converged(&self) -> bool {
        self.0.converged
    }

    /// String representation.
    fn __repr__(&self) -> String {
        let n_obs = self.0.observations.len();
        format!(
            "OrbitFit(rms={:.6e}, obs={}, converged={}, epoch={:.6})",
            self.0.rms, n_obs, self.0.converged, self.0.uncertain_state.state.epoch.jd,
        )
    }
}

/// Perform batch least-squares differential correction with optional
/// chi-squared outlier rejection.
///
/// For arcs longer than 180 days, progressively wider time windows are
/// fitted around the reference epoch so that each stage bootstraps from
/// the previous converged solution.  The final pass fits the full arc
/// and re-evaluates all observations for outlier rejection (if enabled).
///
/// Outlier rejection is controlled by ``max_reject_passes``.  When zero
/// (the default), no rejection is performed and all observations are used.
///
/// The per-observation chi-squared is
/// ``sum(residual_k^2 / sigma_k^2)`` over the measurement components
/// (2 for optical: RA + Dec).  For a threshold of 9.0 this corresponds
/// to roughly 3-sigma per component.
///
/// When ``auto_sigma`` is True, the effective threshold is rescaled each
/// rejection pass by a robust estimate of the actual residual scatter
/// (MAD-based).  This is useful when the stated observation uncertainties
/// are unreliable.
///
/// The input state is automatically re-centered to SSB.
///
/// Parameters
/// ----------
/// initial_state : State
///     Initial guess for the object state (any center / frame).
/// observations : list[Observation]
///     Observations to fit.
/// include_asteroids : bool, optional
///     If True, include asteroid masses in the force model (slower but more
///     accurate for near-Earth objects). Default is False.
/// non_grav : NonGravModel, optional
///     Non-gravitational force model, if any.
/// max_iter : int, optional
///     Maximum number of iterations per convergence pass. Default is 50.
/// tol : float, optional
///     Convergence tolerance on the state correction norm. Default is 1e-8.
/// chi2_threshold : float, optional
///     Chi-squared threshold for outlier rejection. Default is 9.0.
///     Only used when ``max_reject_passes > 0``.
/// max_reject_passes : int, optional
///     Maximum number of batch rejection/re-solve cycles. Default is 0
///     (no rejection).
/// auto_sigma : bool, optional
///     If True, rescale the chi-squared threshold each pass using a
///     robust (MAD-based) estimate of the actual residual scatter.
///     Default is False.
///
/// Returns
/// -------
/// OrbitFit
///     The converged orbit fit result.
#[pyfunction]
#[pyo3(
    name = "differential_correction",
    signature = (
        initial_state,
        observations,
        include_asteroids=false,
        non_grav=None,
        max_iter=50,
        tol=1e-8,
        chi2_threshold=9.0,
        max_reject_passes=3,
        auto_sigma=false,
    )
)]
pub fn differential_correction_py(
    initial_state: PyState,
    observations: Vec<PyObservation>,
    include_asteroids: bool,
    non_grav: Option<PyNonGravModel>,
    max_iter: usize,
    tol: f64,
    chi2_threshold: f64,
    max_reject_passes: usize,
    auto_sigma: bool,
) -> PyResult<PyOrbitFit> {
    let mut raw_state = initial_state.raw;

    // Re-center to SSB.
    {
        let spk = &LOADED_SPK.try_read().map_err(Error::from)?;
        spk.try_change_center(&mut raw_state, 0)?;
    }

    let obs: Vec<Observation> = observations.into_iter().map(|o| o.0).collect();
    let ng = non_grav.as_ref().map(|m| &m.0);

    let fit = differential_correction(
        &raw_state,
        &obs,
        include_asteroids,
        ng,
        max_iter,
        tol,
        chi2_threshold,
        max_reject_passes,
        auto_sigma,
    )?;
    Ok(PyOrbitFit(fit))
}

/// Compute an initial orbit from observations.
///
/// Parameters
/// ----------
/// observations : list[Observation]
///     Observations to use for IOD.
///
/// Returns
/// -------
/// list[State]
///     One or more candidate initial states.
#[pyfunction]
#[pyo3(name = "initial_orbit_determination")]
pub fn initial_orbit_determination_py(observations: Vec<PyObservation>) -> PyResult<Vec<PyState>> {
    let obs: Vec<Observation> = observations.into_iter().map(|o| o.0).collect();
    let states = kete_fitting::initial_orbit_determination(&obs)?;
    let spk = LOADED_SPK.try_read().map_err(Error::from)?;
    states
        .into_iter()
        .map(|mut st| {
            if st.center_id != 10 {
                spk.try_change_center(&mut st, 10)?;
            }
            Ok(st.into())
        })
        .collect()
}

/// Short-arc IOD assuming near-circular orbits (Vaisala-like method).
///
/// Works for tracklets spanning minutes to roughly 2 days where the standard
/// :func:`initial_orbit_determination` cannot reliably estimate velocity.
///
/// Parameters
/// ----------
/// observations : list[Observation]
///     At least 2 optical observations from a short tracklet.
///
/// Returns
/// -------
/// list[State]
///     Up to 5 candidate initial states, sorted by residual score.
#[pyfunction]
#[pyo3(name = "short_arc_iod")]
pub fn short_arc_iod_py(observations: Vec<PyObservation>) -> PyResult<Vec<PyState>> {
    let obs: Vec<Observation> = observations.into_iter().map(|o| o.0).collect();
    let states = kete_fitting::short_arc_iod(&obs)?;
    let spk = LOADED_SPK.try_read().map_err(Error::from)?;
    states
        .into_iter()
        .map(|mut st| {
            if st.center_id != 10 {
                spk.try_change_center(&mut st, 10)?;
            }
            Ok(st.into())
        })
        .collect()
}

/// Posterior orbit samples from NUTS MCMC.
///
/// Attributes
/// ----------
/// epoch : float
///     Common reference epoch (JD, TDB) for all draws.
/// draws : list[list[float]]
///     Posterior draws, each row is ``[x, y, z, vx, vy, vz, ...]``
///     at the reference epoch (AU, AU/day, Equatorial SSB).
/// chain_id : list[int]
///     Seed index (0-based) that generated each draw.
/// divergent : list[bool]
///     True if the draw was a divergent transition.
/// logp : list[float]
///     Log-posterior value at each draw (NaN where unavailable).
#[pyclass(frozen, module = "kete.fitting", name = "OrbitSamples")]
#[derive(Debug, Clone)]
pub struct PyOrbitSamples(pub OrbitSamples);

#[pymethods]
impl PyOrbitSamples {
    /// Common reference epoch (JD, TDB).
    #[getter]
    fn epoch(&self) -> f64 {
        self.0.epoch
    }

    /// Designator of the fitted object.
    #[getter]
    fn desig(&self) -> &str {
        &self.0.desig
    }

    /// Posterior draws as a list of :class:`~kete.State` objects.
    ///
    /// Each state is Sun-centered Ecliptic at the reference epoch.
    ///
    /// Non-gravitational parameters (if fitted) are available via
    /// :attr:`raw_draws`.
    #[getter]
    fn draws(&self) -> PyResult<Vec<PyState>> {
        let epoch_jd = self.0.epoch;
        let desig = self.0.desig.clone();
        let spk = LOADED_SPK.try_read().map_err(Error::from)?;
        let sun_state: State<Equatorial> =
            spk.try_get_state_with_center(10, Time::new(epoch_jd), 0)?;
        self.0
            .draws
            .iter()
            .map(|d| {
                // Draws are SSB-centered Equatorial; shift to Sun-centered.
                let pos = [
                    d[0] - sun_state.pos[0],
                    d[1] - sun_state.pos[1],
                    d[2] - sun_state.pos[2],
                ];
                let vel = [
                    d[3] - sun_state.vel[0],
                    d[4] - sun_state.vel[1],
                    d[5] - sun_state.vel[2],
                ];
                // Build the State<Equatorial> directly -- the pos/vel are
                // already in Equatorial components.  Using PyState::new with
                // VectorLike::Arr would incorrectly interpret them as
                // Ecliptic and apply an unwanted rotation.
                let desig_val = Desig::Name(desig.clone());
                let st: State<Equatorial> =
                    State::new(desig_val, Time::new(epoch_jd), pos.into(), vel.into(), 10);
                // From<State<Equatorial>> sets Ecliptic display.
                Ok(st.into())
            })
            .collect()
    }

    /// Raw posterior draws as a list of lists.
    ///
    /// Each inner list is ``[x, y, z, vx, vy, vz, ng_params...]``
    /// in the Equatorial frame at the reference epoch.
    /// Use ``np.array(samples.raw_draws)`` to convert to a 2-D array.
    #[getter]
    fn raw_draws(&self) -> Vec<Vec<f64>> {
        self.0.draws.clone()
    }

    /// Seed index (0-based) that generated each draw.
    #[getter]
    fn chain_id(&self) -> Vec<usize> {
        self.0.chain_id.clone()
    }

    /// Per-draw divergence flag.
    #[getter]
    fn divergent(&self) -> Vec<bool> {
        self.0.divergent.clone()
    }

    /// Per-draw log-posterior value.
    #[getter]
    fn logp(&self) -> Vec<f64> {
        self.0.logp.clone()
    }

    /// Number of posterior draws.
    fn __len__(&self) -> usize {
        self.0.draws.len()
    }

    /// String representation.
    fn __repr__(&self) -> String {
        let n = self.0.draws.len();
        let n_chains = self
            .0
            .chain_id
            .iter()
            .copied()
            .collect::<std::collections::HashSet<_>>()
            .len();
        let n_div = self.0.divergent.iter().filter(|&&d| d).count();
        format!(
            "OrbitSamples(desig={}, draws={n}, chains={n_chains}, divergent={n_div}, epoch={:.6})",
            self.0.desig, self.0.epoch
        )
    }
}

/// Run NUTS MCMC sampling over orbital posteriors.
///
/// This is designed for **short-arc observations** where the Gaussian
/// approximation from differential correction breaks down and the posterior
/// is multi-modal or highly non-Gaussian.  For well-observed objects with
/// long arcs, :func:`differential_correction` alone is usually sufficient
/// and far cheaper -- each NUTS draw requires a full STM propagation, so
/// MCMC is orders of magnitude more expensive.
///
/// Chains are automatically spread across available CPU cores.  When there
/// are fewer seeds than cores, each seed spawns multiple sub-chains (each
/// with its own RNG seed and tuning phase).  The ``chain_id`` in the
/// returned :class:`OrbitSamples` identifies the seed (orbital mode), not
/// the sub-chain.
///
/// ``num_draws`` is the **total** number of posterior draws returned across
/// all seeds.  Each seed receives roughly ``num_draws / len(seeds)`` draws,
/// which are then split across its sub-chains.
///
/// All seeds must share the same reference epoch.
///
/// The non-gravitational model (if any) is taken from each seed's
/// :attr:`OrbitFit.non_grav`, which already contains the fitted parameter
/// values that the covariance was linearized around.
///
/// Parameters
/// ----------
/// seeds : list[OrbitFit]
///     Converged orbit fits, one per orbital mode (from
///     :func:`differential_correction`).
/// observations : list[Observation]
///     Observations to evaluate the likelihood against.
/// include_asteroids : bool, optional
///     If True, include asteroid masses in the force model. Default is False.
/// num_draws : int, optional
///     Total posterior draws across all seeds (after tuning).
///     Default is 1000.
/// num_tune : int, optional
///     Number of tuning (warmup) steps per sub-chain used to adapt the
///     step size and mass matrix.  Because sampling uses whitened
///     coordinates (via the MAP covariance Cholesky), the posterior is
///     approximately standard-normal and adaptation converges quickly.
///     Each sub-chain pays its own warmup cost, so keep this small.
///     Default is 50.
/// student_nu : float, optional
///     Student-t degrees of freedom for the likelihood.  Use ``float('inf')``
///     for Gaussian (default).  Lower values (e.g. 5) down-weight outliers.
///
/// Returns
/// -------
/// OrbitSamples
///     Posterior samples pooled from all chains.
///
/// Raises
/// ------
/// ValueError
///     If ``seeds`` is empty or the seeds have different reference epochs.
#[pyfunction]
#[pyo3(
    name = "nuts_sample",
    signature = (seeds, observations, include_asteroids=false, num_draws=1000, num_tune=50, student_nu=f64::INFINITY)
)]
pub fn nuts_sample_py(
    seeds: Vec<PyOrbitFit>,
    observations: Vec<PyObservation>,
    include_asteroids: bool,
    num_draws: usize,
    num_tune: usize,
    student_nu: f64,
) -> PyResult<PyOrbitSamples> {
    let raw_seeds: Vec<OrbitFit> = seeds.into_iter().map(|s| s.0).collect();
    let obs: Vec<Observation> = observations.into_iter().map(|o| o.0).collect();

    let result = nuts_sample(
        &raw_seeds,
        &obs,
        include_asteroids,
        num_draws,
        num_tune,
        student_nu,
    )?;
    Ok(PyOrbitSamples(result))
}
