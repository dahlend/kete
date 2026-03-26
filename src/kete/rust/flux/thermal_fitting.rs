//! PyO3 bindings for model fitting (NEATM/FRM/HG).
//!
//! Exposes [`kete_flux::fitting`] types and functions to Python under
//! `kete.flux`.

use crate::frame::PyFrames;
use crate::vector::VectorLike;
use kete_flux::BandInfo;
use kete_flux::fitting::{self, FitResult, FluxObs, FluxPriors, Model, ParamPrior};
use pyo3::prelude::*;

/// Accept either a WISE band name ("W1"-"W4") or a wavelength in nm.
#[derive(Debug, FromPyObject)]
enum BandSpec {
    Name(String),
    Wavelength(f64),
}

impl BandSpec {
    fn into_band_info(self) -> PyResult<BandInfo> {
        match self {
            BandSpec::Name(s) => {
                let wise = BandInfo::new_wise();
                match s.to_uppercase().as_str() {
                    "W1" => Ok(wise[0].clone()),
                    "W2" => Ok(wise[1].clone()),
                    "W3" => Ok(wise[2].clone()),
                    "W4" => Ok(wise[3].clone()),
                    "V" => Ok(BandInfo::new_v()),
                    other => Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "Unknown band name '{other}'. Use 'W1'-'W4', 'V' or a wavelength in nm."
                    ))),
                }
            }
            // Beam area 1.0 and NaN zero-mag flux are safe placeholders:
            // the fitting code works entirely in Jy, never converting to
            // magnitudes, so the zero-mag flux is unused.
            BandSpec::Wavelength(w) => Ok(BandInfo::new(w, 1.0, f64::NAN, None)),
        }
    }
}

/// A single flux observation at a known geometry.
///
/// Parameters
/// ----------
/// flux :
///     Observed flux in Jy (or upper-limit threshold).
/// sigma :
///     1-sigma uncertainty in Jy.
/// band :
///     Band identifier: a WISE name (``"W1"``-``"W4"``) or a wavelength in nm.
/// sun2obj :
///     Sun-to-object vector in AU (Ecliptic frame).
/// sun2obs :
///     Sun-to-observer vector in AU (Ecliptic frame).
/// is_upper_limit :
///     If ``True``, ``flux`` is a non-detection upper limit.
#[pyclass(frozen, module = "kete.flux", name = "FluxObs", from_py_object)]
#[derive(Clone, Debug)]
pub struct PyFluxObs(pub FluxObs);

#[pymethods]
impl PyFluxObs {
    #[new]
    #[pyo3(signature = (flux, sigma, band, sun2obj, sun2obs, is_upper_limit=false))]
    fn new(
        flux: f64,
        sigma: f64,
        band: BandSpec,
        sun2obj: VectorLike,
        sun2obs: VectorLike,
        is_upper_limit: bool,
    ) -> PyResult<Self> {
        Ok(Self(FluxObs {
            flux,
            sigma,
            band: band.into_band_info()?,
            is_upper_limit,
            sun2obj: sun2obj.into_vector(PyFrames::Ecliptic).into(),
            sun2obs: sun2obs.into_vector(PyFrames::Ecliptic).into(),
        }))
    }

    /// Observed flux in Jy.
    #[getter]
    fn flux(&self) -> f64 {
        self.0.flux
    }

    /// 1-sigma uncertainty in Jy.
    #[getter]
    fn sigma(&self) -> f64 {
        self.0.sigma
    }

    /// Band wavelength in nm.
    #[getter]
    fn wavelength(&self) -> f64 {
        self.0.band.wavelength
    }

    /// Whether this is a non-detection upper limit.
    #[getter]
    fn is_upper_limit(&self) -> bool {
        self.0.is_upper_limit
    }

    /// Sun-to-object vector in AU (Ecliptic frame), as ``[x, y, z]``.
    #[getter]
    fn sun2obj(&self) -> [f64; 3] {
        self.0.sun2obj.into()
    }

    /// Sun-to-observer vector in AU (Ecliptic frame), as ``[x, y, z]``.
    #[getter]
    fn sun2obs(&self) -> [f64; 3] {
        self.0.sun2obs.into()
    }

    fn __repr__(&self) -> String {
        format!(
            "FluxObs(flux={:.4e}, sigma={:.4e}, wavelength={:.1}, upper_limit={})",
            self.0.flux, self.0.sigma, self.0.band.wavelength, self.0.is_upper_limit
        )
    }
}

/// Prior configuration for model fitting.
/// Configuration for a single parameter's prior.
///
/// Parameters
/// ----------
/// bounds :
///     ``(low, high)`` logistic-barrier hard bounds.
/// gaussian :
///     Optional ``(mean, sigma)`` Gaussian centering prior.
///     ``None`` means flat (uniform) within the bounds.
#[pyclass(frozen, module = "kete.flux", name = "ParamPrior", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct PyParamPrior(pub ParamPrior);

impl<'a, 'py> FromPyObject<'a, 'py> for PyParamPrior {
    type Error = PyErr;
    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        // Existing ParamPrior instance.
        if let Ok(pp) = ob.cast::<PyParamPrior>() {
            return Ok(pp.get().clone());
        }
        // 4-tuple (low, high, mean, std) -> bounds + gaussian.
        if let Ok((low, high, mean, sigma)) = ob.extract::<(f64, f64, f64, f64)>() {
            return Ok(Self(ParamPrior::with_gaussian(low, high, mean, sigma)));
        }
        // 2-tuple (low, high) -> bounds only.
        if let Ok((low, high)) = ob.extract::<(f64, f64)>() {
            return Ok(Self(ParamPrior::bounds_only(low, high)));
        }
        Err(pyo3::exceptions::PyTypeError::new_err(
            "Expected a ParamPrior, a 2-tuple (low, high), or a 4-tuple (low, high, mean, std)",
        ))
    }
}

#[pymethods]
impl PyParamPrior {
    #[new]
    #[pyo3(signature = (bounds, gaussian=None))]
    fn new(bounds: (f64, f64), gaussian: Option<(f64, f64)>) -> Self {
        Self(match gaussian {
            Some((mean, sigma)) => ParamPrior::with_gaussian(bounds.0, bounds.1, mean, sigma),
            None => ParamPrior::bounds_only(bounds.0, bounds.1),
        })
    }

    #[getter]
    fn bounds(&self) -> (f64, f64) {
        self.0.bounds
    }

    #[getter]
    fn gaussian(&self) -> Option<(f64, f64)> {
        self.0.gaussian
    }

    fn __repr__(&self) -> String {
        let (low, high) = self.0.bounds;
        match self.0.gaussian {
            Some((mean, sigma)) => {
                format!("ParamPrior(bounds=({low}, {high}), gaussian=({mean}, {sigma}))")
            }
            None => format!("ParamPrior(bounds=({low}, {high}))"),
        }
    }
}

/// Priors used in thermal model fitting, below is the list of default priors.
///
/// Priors can be specified either using the ``FluxPriors`` constructor or by directly
/// passing a tuple of the form ``(low, high)`` for bounds-only or
/// ``(low, high, mean, sigma)`` for bounds + Gaussian. Below we see the defaults
/// being specified using the tuple form.
///
/// Not all of these priors are used in every model, this is a comprehensive list for
/// all models.
///
/// If not provided, defaults are used, specifying a prior will overwrite the specified
/// priors while leaving the others at their defaults:
///
/// .. code-block:: python
///
///     kete.flux.FluxPriors(
///         diameter   = (0.001, 1000),
///         beaming    = (0.5, 3.0, 1.0, 0.3),
///         r_ir       = (0.5, 2.0, 1.6, 0.3),
///         h_mag      = (-5.0, 35.0),
///         g_param    = (-0.3, 0.7, 0.2, 0.05),
///         vis_albedo = (0.0, 1.0),
///     )
///
/// Each prior is a :class:`ParamPrior` specifying ``bounds`` (logistic
/// barrier) and an optional ``gaussian`` centering prior ``(mean, sigma)``.
/// To effectively fix a parameter, set tight bounds around the desired
/// value (e.g., ``bounds=(val - 1e-3, val + 1e-3)``).
///
/// Parameters
/// ----------
/// diameter :
///     :class:`ParamPrior` for diameter D in km.
/// beaming :
///     :class:`ParamPrior` for beaming parameter.
/// r_ir :
///     :class:`ParamPrior` for IR-to-visible albedo ratio R_IR.
/// h_mag :
///     :class:`ParamPrior` for H magnitude.
/// g_param :
///     :class:`ParamPrior` for G parameter.
/// vis_albedo :
///     :class:`ParamPrior` for visible geometric albedo.
#[pyclass(frozen, module = "kete.flux", name = "FluxPriors", from_py_object)]
#[derive(Clone, Debug)]
pub struct PyFluxPriors(pub FluxPriors);

#[pymethods]
impl PyFluxPriors {
    #[new]
    #[pyo3(signature = (
        diameter=None,
        beaming=None,
        r_ir=None,
        h_mag=None,
        g_param=None,
        vis_albedo=None,
    ))]
    fn new(
        diameter: Option<PyParamPrior>,
        beaming: Option<PyParamPrior>,
        r_ir: Option<PyParamPrior>,
        h_mag: Option<PyParamPrior>,
        g_param: Option<PyParamPrior>,
        vis_albedo: Option<PyParamPrior>,
    ) -> Self {
        let d = FluxPriors::default();
        Self(FluxPriors {
            diameter: diameter.map_or(d.diameter, |p| p.0),
            beaming: beaming.map_or(d.beaming, |p| p.0),
            r_ir: r_ir.map_or(d.r_ir, |p| p.0),
            h_mag: h_mag.map_or(d.h_mag, |p| p.0),
            g_param: g_param.map_or(d.g_param, |p| p.0),
            vis_albedo: vis_albedo.map_or(d.vis_albedo, |p| p.0),
            f_sigma: d.f_sigma,
        })
    }

    fn __repr__(&self) -> String {
        let p = &self.0;
        format!(
            "FluxPriors(\n  diameter={},\n  beaming={},\n  r_ir={},\n  h_mag={},\n  g_param={},\n  vis_albedo={})",
            PyParamPrior(p.diameter.clone()).__repr__(),
            PyParamPrior(p.beaming.clone()).__repr__(),
            PyParamPrior(p.r_ir.clone()).__repr__(),
            PyParamPrior(p.h_mag.clone()).__repr__(),
            PyParamPrior(p.g_param.clone()).__repr__(),
            PyParamPrior(p.vis_albedo.clone()).__repr__(),
        )
    }

    #[getter]
    fn diameter(&self) -> PyParamPrior {
        PyParamPrior(self.0.diameter.clone())
    }

    #[getter]
    fn beaming(&self) -> PyParamPrior {
        PyParamPrior(self.0.beaming.clone())
    }

    #[getter]
    fn r_ir(&self) -> PyParamPrior {
        PyParamPrior(self.0.r_ir.clone())
    }

    #[getter]
    fn h_mag(&self) -> PyParamPrior {
        PyParamPrior(self.0.h_mag.clone())
    }

    #[getter]
    fn g_param(&self) -> PyParamPrior {
        PyParamPrior(self.0.g_param.clone())
    }

    #[getter]
    fn vis_albedo(&self) -> PyParamPrior {
        PyParamPrior(self.0.vis_albedo.clone())
    }
}

/// Summary statistics for a single fitted parameter (posterior).
#[pyclass(
    frozen,
    module = "kete.flux",
    name = "SampleStats",
    skip_from_py_object
)]
#[derive(Clone, Debug)]
pub struct PySampleStats {
    #[pyo3(get)]
    median: f64,
    #[pyo3(get)]
    std: f64,
    #[pyo3(get)]
    ci_lo: f64,
    #[pyo3(get)]
    ci_hi: f64,
}

#[pymethods]
impl PySampleStats {
    fn __repr__(&self) -> String {
        format!("{self}")
    }
}

impl std::fmt::Display for PySampleStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "SampleStats(median={:.4}, std={:.4}, ci=[{:.4}, {:.4}])",
            self.median, self.std, self.ci_lo, self.ci_hi
        )
    }
}

/// Compute summary statistics from a single column of MCMC draws.
fn stats_from_column(draws: &[Vec<f64>], col: usize) -> PySampleStats {
    let vals: Vec<f64> = draws.iter().map(|r| r[col]).collect();
    let sorted = kete_stats::prelude::SortedData::try_from(vals)
        .expect("draws should be non-empty with finite values");
    PySampleStats {
        median: sorted.median(),
        std: sorted.std(),
        ci_lo: sorted.quantile(0.025),
        ci_hi: sorted.quantile(0.975),
    }
}

/// Full MCMC fitting result.
#[pyclass(frozen, module = "kete.flux", name = "FitResult", from_py_object)]
#[derive(Clone, Debug)]
pub struct PyFitResult(pub FitResult);

#[pymethods]
impl PyFitResult {
    /// Posterior statistics for diameter (km).  ``None`` for HG model.
    #[getter]
    fn diameter(&self) -> Option<PySampleStats> {
        (!self.0.model.is_hg()).then(|| stats_from_column(&self.0.draws, 0))
    }
    /// Posterior statistics for visual geometric albedo.  ``None`` for HG model.
    #[getter]
    fn vis_albedo(&self) -> Option<PySampleStats> {
        (!self.0.model.is_hg()).then(|| stats_from_column(&self.0.draws, 1))
    }
    /// Posterior statistics for beaming (NEATM only).  ``None`` for FRM/HG.
    #[getter]
    fn beaming(&self) -> Option<PySampleStats> {
        self.0
            .model
            .is_neatm()
            .then(|| stats_from_column(&self.0.draws, 2))
    }
    /// Posterior statistics for H magnitude.
    #[getter]
    fn h_mag(&self) -> PySampleStats {
        let col = match self.0.model {
            Model::Neatm => 3,
            Model::Frm => 2,
            Model::Hg => 0,
        };
        stats_from_column(&self.0.draws, col)
    }
    /// Posterior statistics for G parameter.
    #[getter]
    fn g_param(&self) -> PySampleStats {
        let col = match self.0.model {
            Model::Neatm => 4,
            Model::Frm => 3,
            Model::Hg => 1,
        };
        stats_from_column(&self.0.draws, col)
    }
    /// Posterior statistics for uncertainty inflation factor.
    #[getter]
    fn f_sigma(&self) -> PySampleStats {
        let col = match self.0.model {
            Model::Neatm => 6,
            Model::Frm => 5,
            Model::Hg => 2,
        };
        stats_from_column(&self.0.draws, col)
    }
    /// Posterior statistics for IR-to-visible albedo ratio.  ``None`` for HG model.
    #[getter]
    fn ir_albedo_ratio(&self) -> Option<PySampleStats> {
        if self.0.model.is_hg() {
            return None;
        }
        let col = if self.0.model.is_neatm() { 5 } else { 4 };
        Some(stats_from_column(&self.0.draws, col))
    }
    /// Model name (``"Neatm"``, ``"Frm"``, or ``"Hg"``).
    #[getter]
    fn model(&self) -> String {
        format!("{:?}", self.0.model)
    }
    /// Raw MCMC posterior draws.  Each row is one sample; column layout
    /// depends on the model (see :attr:`columns`).
    #[getter]
    fn draws(&self) -> Vec<Vec<f64>> {
        self.0.draws.clone()
    }
    /// Column names for each element of a draw vector.
    #[getter]
    fn columns(&self) -> Vec<&'static str> {
        match self.0.model {
            Model::Neatm => vec![
                "diameter",
                "vis_albedo",
                "beaming",
                "h_mag",
                "g_param",
                "ir_albedo_ratio",
                "f_sigma",
            ],
            Model::Frm => vec![
                "diameter",
                "vis_albedo",
                "h_mag",
                "g_param",
                "ir_albedo_ratio",
                "f_sigma",
            ],
            Model::Hg => vec!["h_mag", "g_param", "f_sigma"],
        }
    }
    /// Per-draw divergence flags from the NUTS sampler.
    #[getter]
    fn divergent(&self) -> Vec<bool> {
        self.0.divergent.clone()
    }
    /// Total number of divergent transitions.
    #[getter]
    fn n_divergent(&self) -> usize {
        self.0.n_divergent
    }
    /// Reduced chi-squared at the MAP point using inflated uncertainties.
    #[getter]
    fn chi2_best(&self) -> f64 {
        self.0.reduced_chi2
    }
    /// Number of non-upper-limit observations.
    #[getter]
    fn nobs(&self) -> usize {
        self.0.nobs
    }
    /// Model fluxes at the MAP point for each observation (Jy).
    #[getter]
    fn best_fit_fluxes(&self) -> Vec<f64> {
        self.0.best_fit_fluxes.clone()
    }
    /// Standardized residuals ``(obs - model) / (f_sigma * sigma)`` at the MAP.
    #[getter]
    fn best_fit_residuals(&self) -> Vec<f64> {
        self.0.best_fit_residuals.clone()
    }
    /// Reflected-light fraction at the MAP point, one per observation.
    #[getter]
    fn best_fit_reflected_frac(&self) -> Vec<f64> {
        self.0.best_fit_reflected_frac.clone()
    }

    fn __repr__(&self) -> String {
        if self.0.model.is_hg() {
            let h = stats_from_column(&self.0.draws, 0);
            let g = stats_from_column(&self.0.draws, 1);
            return format!(
                "FitResult(model=Hg, H={}, G={}, n_draws={}, n_div={})",
                h,
                g,
                self.0.draws.len(),
                self.0.n_divergent,
            );
        }
        let d = stats_from_column(&self.0.draws, 0);
        let pv = stats_from_column(&self.0.draws, 1);
        let (beaming, h_col) = if self.0.model.is_neatm() {
            (Some(stats_from_column(&self.0.draws, 2)), 3)
        } else {
            (None, 2)
        };
        let h = stats_from_column(&self.0.draws, h_col);
        let g = stats_from_column(&self.0.draws, h_col + 1);
        let rir = stats_from_column(&self.0.draws, h_col + 2);

        let beaming_str = beaming
            .map(|b| format!("\n  beaming={b},"))
            .unwrap_or_default();
        format!(
            "FitResult(model={:?},\n  D={d},\n  pV={pv},{beaming_str}\n  H={h},\n  G={g},\n  R_IR={rir},\n  n_draws={}, n_div={})",
            self.0.model,
            self.0.draws.len(),
            self.0.n_divergent,
        )
    }
}

fn extract_obs(obs: &[PyFluxObs]) -> Vec<FluxObs> {
    obs.iter().map(|o| o.0.clone()).collect()
}

fn parse_model(model: &str) -> PyResult<Model> {
    match model.to_lowercase().as_str() {
        "neatm" => Ok(Model::Neatm),
        "frm" => Ok(Model::Frm),
        "hg" => Ok(Model::Hg),
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Unknown model '{other}'. Use 'neatm', 'frm', or 'hg'."
        ))),
    }
}

/// Fit a model to observations using NUTS MCMC.
///
/// H magnitude and G parameter are fitted as free parameters.
/// If ``h_mag`` or ``g_param`` are supplied, they set the center of the
/// corresponding Gaussian prior (keeping its width from ``priors``).
/// ``emissivity`` is a fixed thermal property (not fitted).
///
/// Parameters
/// ----------
/// model :
///     Model name: ``"neatm"``, ``"frm"``, or ``"hg"``.
/// obs :
///     List of :class:`FluxObs` observations.
/// h_mag :
///     Optional H magnitude -- sets the center of the H prior.
/// g_param :
///     Optional G parameter -- sets the center of the G prior.
/// emissivity :
///     Fixed thermal emissivity (default 0.9, not fitted).
/// priors :
///     Prior configuration (:class:`FluxPriors`, default if ``None``).
/// num_chains :
///     Number of MCMC chains (default 10).
/// num_tune :
///     Warmup draws per chain (default 200).
/// num_draws :
///     Posterior draws per chain (default 500).
/// c_hg :
///     HG relationship constant (default 1329.0).
///
/// Returns
/// -------
/// FitResult or None
///     MCMC posterior results, or ``None`` if the fit fails.
#[pyfunction]
#[pyo3(name = "fit_model", signature = (model, obs, h_mag=None, g_param=None,
    emissivity=0.9, priors=None, num_chains=10, num_tune=200, num_draws=500,
    c_hg=None))]
#[allow(clippy::too_many_arguments)]
pub fn fit_model_py(
    model: &str,
    obs: Vec<PyFluxObs>,
    h_mag: Option<f64>,
    g_param: Option<f64>,
    emissivity: f64,
    priors: Option<PyFluxPriors>,
    num_chains: usize,
    num_tune: usize,
    num_draws: usize,
    c_hg: Option<f64>,
) -> PyResult<Option<PyFitResult>> {
    let tm = parse_model(model)?;
    let mut priors = priors.map_or_else(FluxPriors::default, |p| p.0);
    let c_hg_val = c_hg.unwrap_or(kete_core::constants::C_V);

    // Override prior centers from convenience arguments.
    if let Some(h) = h_mag {
        let sigma = priors.h_mag.gaussian.map_or(0.25, |(_, s)| s);
        priors.h_mag.gaussian = Some((h, sigma));
    }
    if let Some(g) = g_param {
        let sigma = priors.g_param.gaussian.map_or(0.05, |(_, s)| s);
        priors.g_param.gaussian = Some((g, sigma));
    }

    let raw_obs = extract_obs(&obs);
    Ok(fitting::fit_mcmc(
        tm, &raw_obs, c_hg_val, emissivity, &priors, num_chains, num_tune, num_draws,
    )
    .map(PyFitResult))
}
