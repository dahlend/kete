//! Python bindings for orbit determination and fitting.
//!
//! Wraps `kete_fitting` types and functions for use from Python.

use kete_core::frames::{Equatorial, Vector};
use kete_core::prelude::*;
use kete_core::propagation::NonGravModel;
use kete_core::spice::LOADED_SPK;
use kete_fitting::{Observation, OrbitFit, OrbitSamples, fit_orbit, fit_orbit_mcmc, lambert};
use pyo3::{PyResult, pyclass, pyfunction, pymethods};

use crate::nongrav::PyNonGravModel;
use crate::state::PyState;
use crate::time::PyTime;
use crate::uncertain_state::PyUncertainState;
use crate::vector::PyVector;

/// Radians to arcseconds conversion factor.
const RAD_TO_ARCSEC: f64 = 180.0 * 3600.0 / std::f64::consts::PI;

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
#[pyclass(frozen, module = "kete.fitting", name = "Observation", from_py_object)]
#[derive(Debug, Clone)]
pub struct PyObservation {
    /// The core observation data (astrometry / radar).
    pub obs: Observation,
    /// Photometric filter name (e.g. "V", "W1").  Metadata only.
    pub band: String,
    /// Apparent magnitude.  NaN when unavailable.  Metadata only.
    pub mag: f64,
}

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
    /// observer : :class:`~kete.State`
    ///     Observer state (any center / frame - will be converted to SSB-centered
    ///     Equatorial internally). The observation epoch is taken from the observer's
    ///     epoch.
    /// ra : float
    ///     Right ascension in degrees.
    /// dec : float
    ///     Declination in degrees.
    /// sigma_ra : float
    ///     1-sigma RA uncertainty in arcseconds (should include cos(dec)
    ///     factor).
    /// sigma_dec : float
    ///     1-sigma Dec uncertainty in arcseconds.
    /// band : str
    ///     Photometric filter name (default ``"V"``).
    /// mag : float
    ///     Apparent magnitude (default ``NaN``).
    #[staticmethod]
    #[pyo3(signature = (observer, ra, dec, sigma_ra, sigma_dec, band="V".to_string(), mag=f64::NAN))]
    fn optical(
        observer: PyState,
        ra: f64,
        dec: f64,
        sigma_ra: f64,
        sigma_dec: f64,
        band: String,
        mag: f64,
    ) -> PyResult<Self> {
        if sigma_ra <= 0.0 || sigma_dec <= 0.0 {
            return Err(Error::ValueError("sigma_ra and sigma_dec must be positive".into()).into());
        }
        let mut raw = observer.raw;
        if raw.center_id != 0 {
            let spk = LOADED_SPK.try_read().map_err(Error::from)?;
            spk.try_change_center(&mut raw, 0)?;
        }
        let arcsec_to_rad = 1.0 / RAD_TO_ARCSEC;
        Ok(Self { obs: Observation::Optical {
            observer: raw,
            ra: ra.to_radians(),
            dec: dec.to_radians(),
            sigma_ra: sigma_ra * arcsec_to_rad,
            sigma_dec: sigma_dec * arcsec_to_rad,
        }, band, mag })
    }

    /// Create a radar range observation.
    ///
    /// The observer state is automatically re-centered to SSB.
    ///
    /// Parameters
    /// ----------
    /// observer : :class:`~kete.State`
    ///     Observer state (any center / frame -- will be converted).
    /// range : float
    ///     Measured range in AU.
    /// sigma_range : float
    ///     1-sigma range uncertainty in AU.
    #[staticmethod]
    #[pyo3(signature = (observer, range, sigma_range))]
    fn radar_range(observer: PyState, range: f64, sigma_range: f64) -> PyResult<Self> {
        if sigma_range <= 0.0 {
            return Err(Error::ValueError("sigma_range must be positive".into()).into());
        }
        let mut raw = observer.raw;
        if raw.center_id != 0 {
            let spk = LOADED_SPK.try_read().map_err(Error::from)?;
            spk.try_change_center(&mut raw, 0)?;
        }
        Ok(Self { obs: Observation::RadarRange {
            observer: raw,
            range,
            sigma_range,
        }, band: String::new(), mag: f64::NAN })
    }

    /// Create a radar range-rate (Doppler) observation.
    ///
    /// The observer state is automatically re-centered to SSB.
    ///
    /// Parameters
    /// ----------
    /// observer : :class:`~kete.State`
    ///     Observer state (any center / frame -- will be converted).
    /// range_rate : float
    ///     Measured range-rate in AU/day (positive = receding).
    /// sigma_range_rate : float
    ///     1-sigma range-rate uncertainty in AU/day.
    #[staticmethod]
    #[pyo3(signature = (observer, range_rate, sigma_range_rate))]
    fn radar_rate(observer: PyState, range_rate: f64, sigma_range_rate: f64) -> PyResult<Self> {
        if sigma_range_rate <= 0.0 {
            return Err(Error::ValueError("sigma_range_rate must be positive".into()).into());
        }
        let mut raw = observer.raw;
        if raw.center_id != 0 {
            let spk = LOADED_SPK.try_read().map_err(Error::from)?;
            spk.try_change_center(&mut raw, 0)?;
        }
        Ok(Self { obs: Observation::RadarRate {
            observer: raw,
            range_rate,
            sigma_range_rate,
        }, band: String::new(), mag: f64::NAN })
    }

    /// The observation epoch (from the observer state).
    #[getter]
    fn epoch(&self) -> PyTime {
        self.obs.epoch().jd.into()
    }

    /// The observer state (Sun-centered, Ecliptic).
    #[getter]
    fn observer(&self) -> PyResult<PyState> {
        let mut st = match &self.obs {
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
        match &self.obs {
            Observation::Optical { ra, .. } => Some(ra.to_degrees()),
            _ => None,
        }
    }

    /// Declination in degrees (optical only, None otherwise).
    #[getter]
    fn dec(&self) -> Option<f64> {
        match &self.obs {
            Observation::Optical { dec, .. } => Some(dec.to_degrees()),
            _ => None,
        }
    }

    /// 1-sigma RA uncertainty in arcseconds (optical only, None otherwise).
    #[getter]
    fn sigma_ra(&self) -> Option<f64> {
        match &self.obs {
            Observation::Optical { sigma_ra, .. } => Some(*sigma_ra * RAD_TO_ARCSEC),
            _ => None,
        }
    }

    /// 1-sigma Dec uncertainty in arcseconds (optical only, None otherwise).
    #[getter]
    fn sigma_dec(&self) -> Option<f64> {
        match &self.obs {
            Observation::Optical { sigma_dec, .. } => Some(*sigma_dec * RAD_TO_ARCSEC),
            _ => None,
        }
    }

    /// Measured range in AU (radar range only, None otherwise).
    #[getter]
    fn range(&self) -> Option<f64> {
        match &self.obs {
            Observation::RadarRange { range, .. } => Some(*range),
            _ => None,
        }
    }

    /// 1-sigma range uncertainty in AU (radar range only, None otherwise).
    #[getter]
    fn sigma_range(&self) -> Option<f64> {
        match &self.obs {
            Observation::RadarRange { sigma_range, .. } => Some(*sigma_range),
            _ => None,
        }
    }

    /// Measured range-rate in AU/day (radar rate only, None otherwise).
    #[getter]
    fn range_rate(&self) -> Option<f64> {
        match &self.obs {
            Observation::RadarRate { range_rate, .. } => Some(*range_rate),
            _ => None,
        }
    }

    /// 1-sigma range-rate uncertainty in AU/day (radar rate only, None otherwise).
    #[getter]
    fn sigma_range_rate(&self) -> Option<f64> {
        match &self.obs {
            Observation::RadarRate {
                sigma_range_rate, ..
            } => Some(*sigma_range_rate),
            _ => None,
        }
    }

    /// Photometric filter name (optical only, empty string for radar).
    #[getter]
    fn band(&self) -> &str {
        &self.band
    }

    /// Apparent magnitude (NaN when unavailable).
    #[getter]
    fn mag(&self) -> f64 {
        self.mag
    }

    /// String representation.
    fn __repr__(&self) -> String {
        let epoch = self.obs.epoch().jd;
        match &self.obs {
            Observation::Optical {
                ra,
                dec,
                sigma_ra,
                sigma_dec,
                ..
            } => {
                format!(
                    "Observation.optical(epoch={:.6}, ra={:.8}, dec={:.8}, \
                     sigma_ra={:.4}, sigma_dec={:.4}, band='{}', mag={:.2})",
                    epoch,
                    ra.to_degrees(),
                    dec.to_degrees(),
                    sigma_ra * RAD_TO_ARCSEC,
                    sigma_dec * RAD_TO_ARCSEC,
                    self.band,
                    self.mag,
                )
            }
            Observation::RadarRange {
                range, sigma_range, ..
            } => {
                format!(
                    "Observation.radar_range(epoch={:.6}, range={:.10}, \
                     sigma_range={:.6e})",
                    epoch, range, sigma_range
                )
            }
            Observation::RadarRate {
                range_rate,
                sigma_range_rate,
                ..
            } => {
                format!(
                    "Observation.radar_rate(epoch={:.6}, range_rate={:.10}, \
                     sigma_range_rate={:.6e})",
                    epoch, range_rate, sigma_range_rate
                )
            }
        }
    }
}

/// Result of fitting an orbit to observations.
///
/// Returned by :func:`fit_orbit`.  Contains the best-fit orbital state,
/// its uncertainty (covariance), and diagnostic information about the fit.
#[pyclass(frozen, module = "kete.fitting", name = "OrbitFit", from_py_object)]
#[derive(Debug, Clone)]
pub struct PyOrbitFit(pub OrbitFit);

#[pymethods]
impl PyOrbitFit {
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
        self.0
            .residuals
            .iter()
            .zip(self.0.included.iter())
            .filter(|&(_, &inc)| inc)
            .map(|(r, _)| {
                // Optical residuals have 2 elements (RA, Dec) in radians;
                // radar residuals have 1 element in AU or AU/day.
                if r.len() == 2 {
                    r.iter().map(|v| v * RAD_TO_ARCSEC).collect()
                } else {
                    r.iter().copied().collect()
                }
            })
            .collect()
    }

    /// Observations included in the final fit (time-sorted).
    ///
    /// Rejected outliers are not present in this list.  To see all
    /// observations (including rejected ones), use
    /// ``all_observations``.
    #[getter]
    fn observations(&self) -> Vec<PyObservation> {
        self.0
            .observations
            .iter()
            .zip(self.0.included.iter())
            .filter(|&(_, &inc)| inc)
            .map(|(o, _)| PyObservation { obs: o.clone(), band: String::new(), mag: f64::NAN })
            .collect()
    }

    /// All input observations (time-sorted), including rejected outliers.
    #[getter]
    fn all_observations(&self) -> Vec<PyObservation> {
        self.0
            .observations
            .iter()
            .map(|o| PyObservation { obs: o.clone(), band: String::new(), mag: f64::NAN })
            .collect()
    }

    /// Per-observation inclusion mask (time-sorted).
    ///
    /// ``True`` means the observation was used in the final fit;
    /// ``False`` means it was rejected as an outlier.
    #[getter]
    fn included(&self) -> Vec<bool> {
        self.0.included.clone()
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
        let n_included = self.0.included.iter().filter(|&&v| v).count();
        let n_total = self.0.observations.len();
        format!(
            "OrbitFit(rms={:.6e}, observations={}/{}, converged={}, epoch={:.6})",
            self.0.rms,
            n_included,
            n_total,
            self.0.converged,
            self.0.uncertain_state.state.epoch.jd,
        )
    }
}

/// Fit an orbit to observations using iterative least squares.
///
/// Given an initial guess and a set of observations, this function refines
/// the orbital state until it best matches the data, and estimates the
/// uncertainty of the result via a covariance matrix.  It can also
/// automatically identify and reject outlier observations.
///
/// This is the standard approach for orbit determination when you have
/// a reasonable initial guess (e.g. from
/// :func:`initial_orbit_determination`).  It works well for arcs of any
/// length, but the Gaussian uncertainty estimate is most reliable for
/// **long, well-sampled arcs**.  For short arcs where the uncertainty
/// is non-Gaussian, consider :func:`fit_orbit_mcmc` instead.
///
/// For arcs longer than 180 days, progressively wider time windows are
/// fitted around the reference epoch so that each stage bootstraps from
/// the previous converged solution.  The final pass fits the full arc
/// and re-evaluates all observations for outlier rejection (if enabled).
///
/// The input state is automatically re-centered to the solar system
/// barycenter internally.
///
/// Parameters
/// ----------
/// initial_state : :class:`~kete.State`
///     Initial guess for the object state (any center / frame).
/// observations : list
///     List of :class:`~kete.fitting.Observation` to fit.
/// include_asteroids : bool
///     If True, include asteroid masses in the force model (slower but more
///     accurate for near-Earth objects). Default is False.
/// non_grav : :class:`~kete.propagation.NonGravModel`, optional
///     Non-gravitational force model.
/// max_iter : int
///     Maximum number of iterations per convergence pass. Default is 50.
/// tol : float
///     Convergence tolerance on the state correction norm. Default is 1e-8.
/// chi2_threshold : float
///     Chi-squared threshold for outlier rejection. Default is 9.0
///     (roughly 3-sigma per component).  Only used when
///     ``max_reject_passes > 0``.
/// max_reject_passes : int
///     Maximum number of outlier-rejection cycles. Set to 0 to disable
///     rejection entirely. Default is 3.
/// auto_sigma : bool
///     If True, adaptively rescale the rejection threshold based on the
///     actual residual scatter rather than the stated uncertainties.
///     Useful when observation uncertainties are unreliable.
///     Default is False.
///
/// Returns
/// -------
/// OrbitFit
///     The fitted orbit, including the best-fit state, covariance,
///     residuals, and convergence diagnostics.
#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(
    name = "fit_orbit",
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
pub fn fit_orbit_py(
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

    if raw_state.center_id != 0 {
        let spk = LOADED_SPK.try_read().map_err(Error::from)?;
        spk.try_change_center(&mut raw_state, 0)?;
    }

    let obs: Vec<Observation> = observations.into_iter().map(|o| o.obs).collect();
    let ng = non_grav.as_ref().map(|m| &m.0);

    let fit = fit_orbit(
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
///     At least 2 optical observations.
/// epoch : float
///     Reference epoch (JD, TDB) for returned states (optional).  Defaults
///     to the last observation epoch (for forward prediction).
///
/// Returns
/// -------
/// list[State]
///     One or more candidate initial states at the reference epoch.
#[pyfunction]
#[pyo3(name = "initial_orbit_determination", signature = (observations, epoch=None))]
pub fn initial_orbit_determination_py(
    observations: Vec<PyObservation>,
    epoch: Option<f64>,
) -> PyResult<Vec<PyState>> {
    let obs: Vec<Observation> = observations.into_iter().map(|o| o.obs).collect();
    let epoch_tdb = epoch.map(Time::new);
    let states = kete_fitting::initial_orbit_determination(&obs, epoch_tdb)?;
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

/// Solve Lambert's problem for a single-revolution Keplerian transfer.
///
/// Given two heliocentric position vectors and a transfer time, compute the
/// velocity vectors at departure and arrival that connect them via two-body
/// (Keplerian) motion.
///
/// Parameters
/// ----------
/// r1 : Vector
///     Heliocentric position at departure (AU).
/// r2 : Vector
///     Heliocentric position at arrival (AU).
/// dt : float
///     Transfer time in days. Must be positive.
/// prograde : bool
///     If True (default), selects the short-way transfer (transfer angle
///     less than 180 degrees for prograde orbits). If False, selects
///     the long-way transfer.
///
/// Returns
/// -------
/// tuple[Vector, Vector]
///     ``(v1, v2)`` -- velocity at ``r1`` and ``r2`` respectively (AU/day).
///
/// Raises
/// ------
/// ValueError
///     If ``dt <= 0``, positions are zero-length, or positions are nearly
///     collinear (transfer angle near 0 or 180 degrees).
/// RuntimeError
///     If the iterative solver fails to converge.
#[pyfunction]
#[pyo3(name = "lambert", signature = (r1, r2, dt, prograde=true))]
pub fn lambert_py(
    r1: PyVector,
    r2: PyVector,
    dt: f64,
    prograde: bool,
) -> PyResult<(PyVector, PyVector)> {
    let r1_eq: Vector<Equatorial> = r1.into();
    let r2_eq: Vector<Equatorial> = r2.into();
    let (v1, v2) = lambert(&r1_eq, &r2_eq, dt, prograde)?;
    Ok((v1.into(), v2.into()))
}

/// Collection of plausible orbits from MCMC uncertainty estimation.
///
/// Returned by :func:`fit_orbit_mcmc`.  Each orbit (draw) is statistically
/// consistent with the observations; the spread of the collection represents
/// the uncertainty in the orbit.
#[pyclass(frozen, module = "kete.fitting", name = "OrbitSamples", from_py_object)]
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

    /// Sampled orbits as a list of :class:`~kete.State` objects.
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

    /// Raw orbit samples as a list of lists.
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
    ///
    /// A divergent sample indicates the sampler had difficulty exploring
    /// that region of orbit space.  A small fraction of divergences is
    /// normal; many divergences suggest the model or data are problematic.
    #[getter]
    fn divergent(&self) -> Vec<bool> {
        self.0.divergent.clone()
    }

    /// Number of orbit samples.
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

/// Estimate orbit uncertainty from observations using Markov Chain Monte Carlo.
///
/// Given one or more candidate orbital states (seeds) and a set of
/// observations, this function produces a collection of plausible orbits
/// that are statistically consistent with the data.  The spread
/// of returned orbits represents the **uncertainty** in the orbit
/// determination -- wider spread means less certainty about the true
/// orbit.
///
/// This is most useful for **short-arc observations** (a few nights)
/// where the usual least-squares approach
/// (:func:`fit_orbit`) underestimates the true
/// uncertainty.  For well-observed objects with long arcs,
/// :func:`fit_orbit` alone is usually sufficient and far
/// cheaper.
///
/// Under the hood this uses the No-U-Turn Sampler (NUTS), an adaptive
/// variant of Hamiltonian Monte Carlo that efficiently explores the
/// space of possible orbits.  Each draw requires a full numerical
/// propagation, so this is orders of magnitude more expensive than
/// least squares -- but the result captures non-Gaussian and multi-modal
/// uncertainty that least squares cannot represent.
///
/// Seeds are raw ``State`` objects (typically from
/// :func:`initial_orbit_determination`).  No prior orbit fit is
/// required -- the sampler builds its own internal
/// mass matrix from a linearization at each seed.
///
/// Sampling is parallelized automatically across available CPU cores.
/// When there are fewer seeds than cores, each seed spawns multiple
/// independent sub-chains.  The ``chain_id`` in the returned
/// :class:`~kete.fitting.OrbitSamples` identifies the seed (orbital
/// mode), not the sub-chain.
///
/// ``num_draws`` is the **total** number of orbit samples returned
/// across all seeds.  Each seed receives roughly
/// ``num_draws / len(seeds)`` draws.
///
/// Parameters
/// ----------
/// seeds : list
///     List of :class:`~kete.State` candidate states (e.g. from
///     :func:`initial_orbit_determination`), one per orbital mode.  Seeds at
///     different epochs are automatically propagated to the first seed's epoch.
///     The input states are re-centered to SSB Equatorial internally.
/// observations : list
///     List of :class:`~kete.fitting.Observation` to evaluate against.
/// include_asteroids : bool
///     If True, include asteroid masses in the force model. Default is False.
/// num_draws : int
///     Total orbit samples across all seeds (after tuning). Default is 1000.
/// num_tune : int
///     Number of warmup steps per sub-chain used to adapt internal
///     sampling parameters.  These draws are discarded.  Default is 500.
/// non_grav : :class:`~kete.propagation.NonGravModel`, optional
///     Shared non-gravitational force model applied to all chains.
/// maxdepth : int
///     Maximum tree depth for the sampler.  Higher values allow more
///     thorough exploration at greater computational cost.
///     Default is 10.
/// target_accept : float
///     Target acceptance probability for step-size adaptation during
///     warmup.  Lowering this (e.g. 0.6) makes the sampler take larger
///     leapfrog steps, which helps in poorly constrained situations.
///     Default is 0.8.
///
/// Returns
/// -------
/// OrbitSamples
///     Collection of plausible orbits sampled from the posterior.
///
/// Raises
/// ------
/// ValueError
///     If ``seeds`` is empty or two-body epoch propagation fails.
#[pyfunction]
#[pyo3(
    name = "fit_orbit_mcmc",
    signature = (seeds, observations, include_asteroids=false, num_draws=1000, num_tune=500, non_grav=None, maxdepth=10, target_accept=0.8)
)]
#[allow(clippy::too_many_arguments)]
pub fn fit_orbit_mcmc_py(
    seeds: Vec<PyState>,
    observations: Vec<PyObservation>,
    include_asteroids: bool,
    num_draws: usize,
    num_tune: usize,
    non_grav: Option<PyNonGravModel>,
    maxdepth: u64,
    target_accept: f64,
) -> PyResult<PyOrbitSamples> {
    let spk = LOADED_SPK.try_read().map_err(Error::from)?;
    let raw_seeds: Vec<State<Equatorial>> = seeds
        .into_iter()
        .map(|s| {
            let mut st = s.raw;
            if st.center_id != 0 {
                spk.try_change_center(&mut st, 0)?;
            }
            Ok(st)
        })
        .collect::<KeteResult<Vec<_>>>()?;
    let obs: Vec<Observation> = observations.into_iter().map(|o| o.obs).collect();
    let ng: Option<NonGravModel> = non_grav.map(|m| m.0);

    let result = fit_orbit_mcmc(
        &raw_seeds,
        &obs,
        include_asteroids,
        num_draws,
        num_tune,
        ng.as_ref(),
        maxdepth,
        target_accept,
    )?;
    Ok(PyOrbitSamples(result))
}
