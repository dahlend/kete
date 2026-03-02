//! Python bindings for orbit determination and fitting.
//!
//! Wraps `kete_fitting` types and functions for use from Python.

use kete_core::constants::GravParams;
use kete_core::prelude::*;
use kete_core::spice::LOADED_SPK;
use kete_fitting::{
    Observation, OrbitFit, differential_correction, differential_correction_with_rejection,
    gauss_iod, laplace_iod,
};
use pyo3::{PyResult, pyclass, pyfunction, pymethods};

use crate::nongrav::PyNonGravModel;
use crate::state::PyState;
use crate::time::PyTime;

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
    /// Parameters
    /// ----------
    /// observer : State
    ///     Observer state (SSB-centered, Equatorial frame). The observation
    ///     epoch is taken from the observer's epoch.
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
    fn optical(observer: PyState, ra: f64, dec: f64, sigma_ra: f64, sigma_dec: f64) -> Self {
        let arcsec_to_rad = std::f64::consts::PI / (180.0 * 3600.0);
        Self(Observation::Optical {
            observer: observer.raw,
            ra: ra.to_radians(),
            dec: dec.to_radians(),
            sigma_ra: sigma_ra * arcsec_to_rad,
            sigma_dec: sigma_dec * arcsec_to_rad,
        })
    }

    /// Create a radar range observation.
    ///
    /// Parameters
    /// ----------
    /// observer : State
    ///     Observer state (SSB-centered, Equatorial frame).
    /// range : float
    ///     Measured range in AU.
    /// sigma_range : float
    ///     1-sigma range uncertainty in AU.
    #[staticmethod]
    #[pyo3(signature = (observer, range, sigma_range))]
    fn radar_range(observer: PyState, range: f64, sigma_range: f64) -> Self {
        Self(Observation::RadarRange {
            observer: observer.raw,
            range,
            sigma_range,
        })
    }

    /// Create a radar range-rate (Doppler) observation.
    ///
    /// Parameters
    /// ----------
    /// observer : State
    ///     Observer state (SSB-centered, Equatorial frame).
    /// range_rate : float
    ///     Measured range-rate in AU/day (positive = receding).
    /// sigma_range_rate : float
    ///     1-sigma range-rate uncertainty in AU/day.
    #[staticmethod]
    #[pyo3(signature = (observer, range_rate, sigma_range_rate))]
    fn radar_rate(observer: PyState, range_rate: f64, sigma_range_rate: f64) -> Self {
        Self(Observation::RadarRate {
            observer: observer.raw,
            range_rate,
            sigma_range_rate,
        })
    }

    /// The observation epoch (from the observer state).
    #[getter]
    fn epoch(&self) -> PyTime {
        self.0.epoch().jd.into()
    }

    /// The observer state.
    #[getter]
    fn observer(&self) -> PyState {
        match &self.0 {
            Observation::Optical { observer, .. }
            | Observation::RadarRange { observer, .. }
            | Observation::RadarRate { observer, .. } => observer.clone().into(),
        }
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
/// state : State
///     Best-fit state at the reference epoch.
/// covariance : list[list[float]]
///     Covariance matrix at the reference epoch (6+Np rows and columns).
/// residuals : list[list[float]]
///     Post-fit residuals in time-sorted order. Each inner list has as many
///     elements as the measurement dimension of that observation.
/// included : list[bool]
///     Whether each observation (time-sorted) was included or rejected.
/// rms : float
///     Weighted RMS of post-fit residuals (included observations only).
#[pyclass(frozen, module = "kete.fitting", name = "OrbitFit")]
#[derive(Debug, Clone)]
pub struct PyOrbitFit(pub OrbitFit);

#[pymethods]
impl PyOrbitFit {
    /// Best-fit state at the reference epoch.
    #[getter]
    fn state(&self) -> PyState {
        self.0.state.clone().into()
    }

    /// Covariance matrix as a list of lists (use ``np.array()`` to convert).
    #[getter]
    fn covariance(&self) -> Vec<Vec<f64>> {
        let n = self.0.covariance.nrows();
        let m = self.0.covariance.ncols();
        (0..n)
            .map(|r| (0..m).map(|c| self.0.covariance[(r, c)]).collect())
            .collect()
    }

    /// Post-fit residuals as a list of lists (time-sorted order).
    #[getter]
    fn residuals(&self) -> Vec<Vec<f64>> {
        self.0
            .residuals
            .iter()
            .map(|r| r.iter().copied().collect())
            .collect()
    }

    /// Boolean mask: true if observation was included, false if rejected.
    #[getter]
    fn included(&self) -> Vec<bool> {
        self.0.included.clone()
    }

    /// Weighted RMS of post-fit residuals.
    #[getter]
    fn rms(&self) -> f64 {
        self.0.rms
    }

    /// Fitted non-gravitational model, or None if not fitted.
    #[getter]
    fn non_grav(&self) -> Option<PyNonGravModel> {
        self.0.non_grav.clone().map(PyNonGravModel)
    }

    /// String representation.
    fn __repr__(&self) -> String {
        let n_obs = self.0.included.len();
        let n_inc = self.0.included.iter().filter(|&&b| b).count();
        format!(
            "OrbitFit(rms={:.6e}, obs={}/{}, epoch={:.6})",
            self.0.rms, n_inc, n_obs, self.0.state.epoch.jd,
        )
    }
}

/// Perform batch least-squares differential correction.
///
/// Parameters
/// ----------
/// initial_state : State
///     Initial guess for the object state (SSB-centered, Equatorial).
/// observations : list[Observation]
///     Observations to fit.
/// include_asteroids : bool, optional
///     If True, include asteroid masses in the force model (slower but more
///     accurate for near-Earth objects). Default is False.
/// non_grav : NonGravModel, optional
///     Non-gravitational force model, if any.
/// max_iter : int, optional
///     Maximum number of iterations. Default is 20.
/// tol : float, optional
///     Convergence tolerance on the state correction norm. Default is 1e-8.
///
/// Returns
/// -------
/// OrbitFit
///     The converged orbit fit result.
#[pyfunction]
#[pyo3(
    name = "differential_correction",
    signature = (initial_state, observations, include_asteroids=false, non_grav=None, max_iter=20, tol=1e-8)
)]
pub fn differential_correction_py(
    initial_state: PyState,
    observations: Vec<PyObservation>,
    include_asteroids: bool,
    non_grav: Option<PyNonGravModel>,
    max_iter: usize,
    tol: f64,
) -> PyResult<PyOrbitFit> {
    let mut raw_state = initial_state.raw;

    // Re-center to SSB.
    {
        let spk = &LOADED_SPK.try_read().map_err(Error::from)?;
        spk.try_change_center(&mut raw_state, 0)?;
    }

    let obs: Vec<Observation> = observations.into_iter().map(|o| o.0).collect();
    let ng = non_grav.as_ref().map(|m| &m.0);

    let masses = if include_asteroids {
        GravParams::selected_masses().to_vec()
    } else {
        GravParams::planets()
    };

    let fit = differential_correction(&raw_state, &obs, &masses, ng, max_iter, tol)?;
    Ok(PyOrbitFit(fit))
}

/// Perform differential correction with chi-squared outlier rejection.
///
/// First converges using all observations, then rejects outliers above the
/// chi-squared threshold and re-converges. Repeats up to ``max_reject_passes``
/// times.
///
/// Parameters
/// ----------
/// initial_state : State
///     Initial guess for the object state (SSB-centered, Equatorial).
/// observations : list[Observation]
///     Observations to fit.
/// include_asteroids : bool, optional
///     Include asteroid masses in the force model. Default is False.
/// non_grav : NonGravModel, optional
///     Non-gravitational force model, if any.
/// max_iter : int, optional
///     Maximum iterations per convergence pass. Default is 20.
/// tol : float, optional
///     Convergence tolerance. Default is 1e-8.
/// chi2_threshold : float, optional
///     Chi-squared threshold for outlier rejection. Default is 9.0.
/// max_reject_passes : int, optional
///     Maximum number of rejection/re-solve cycles. Default is 3.
///
/// Returns
/// -------
/// OrbitFit
///     The converged orbit fit result with outlier flags.
#[pyfunction]
#[pyo3(
    name = "differential_correction_with_rejection",
    signature = (
        initial_state,
        observations,
        include_asteroids=false,
        non_grav=None,
        max_iter=20,
        tol=1e-8,
        chi2_threshold=9.0,
        max_reject_passes=3,
    )
)]
pub fn differential_correction_with_rejection_py(
    initial_state: PyState,
    observations: Vec<PyObservation>,
    include_asteroids: bool,
    non_grav: Option<PyNonGravModel>,
    max_iter: usize,
    tol: f64,
    chi2_threshold: f64,
    max_reject_passes: usize,
) -> PyResult<PyOrbitFit> {
    let mut raw_state = initial_state.raw;

    // Re-center to SSB.
    {
        let spk = &LOADED_SPK.try_read().map_err(Error::from)?;
        spk.try_change_center(&mut raw_state, 0)?;
    }

    let obs: Vec<Observation> = observations.into_iter().map(|o| o.0).collect();
    let ng = non_grav.as_ref().map(|m| &m.0);

    let masses = if include_asteroids {
        GravParams::selected_masses().to_vec()
    } else {
        GravParams::planets()
    };

    let fit = differential_correction_with_rejection(
        &raw_state,
        &obs,
        &masses,
        ng,
        max_iter,
        tol,
        chi2_threshold,
        max_reject_passes,
    )?;
    Ok(PyOrbitFit(fit))
}

/// Compute an initial orbit from observations.
///
/// Parameters
/// ----------
/// observations : list[Observation]
///     Observations to use for IOD.
/// method : str
///     IOD method name: ``"gauss"``, ``"laplace"``, or ``"known"``.
/// known_state : State, optional
///     Required when ``method="known"``. The known initial state to use
///     as-is (bypasses IOD computation).
///
/// Returns
/// -------
/// list[State]
///     One or more candidate initial states (multiple roots possible
///     for Gauss and Laplace methods).
#[pyfunction]
#[pyo3(name = "initial_orbit_determination", signature = (observations, method, known_state=None))]
pub fn initial_orbit_determination_py(
    observations: Vec<PyObservation>,
    method: &str,
    known_state: Option<PyState>,
) -> PyResult<Vec<PyState>> {
    let obs: Vec<Observation> = observations.into_iter().map(|o| o.0).collect();

    let states = match method.to_lowercase().as_str() {
        "gauss" => gauss_iod(&obs)?,
        "laplace" => laplace_iod(&obs)?,
        "known" => {
            let state = known_state.ok_or_else(|| {
                Error::ValueError("known_state is required when method='known'".into())
            })?;
            let mut raw = state.raw;
            {
                let spk = &LOADED_SPK.try_read().map_err(Error::from)?;
                spk.try_change_center(&mut raw, 0)?;
            }
            vec![raw]
        }
        _ => {
            return Err(Error::ValueError(format!(
                "Unknown IOD method '{}'. Use 'gauss', 'laplace', or 'known'.",
                method
            ))
            .into());
        }
    };
    Ok(states.into_iter().map(Into::into).collect())
}
