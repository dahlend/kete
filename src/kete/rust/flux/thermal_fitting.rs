//! PyO3 bindings for model fitting (NEATM/FRM/HG).
//!
//! Exposes [`kete_flux::fitting`] types and functions to Python under
//! `kete.flux`.

use crate::frame::PyFrames;
use crate::vector::VectorLike;
use kete_flux::BandInfo;
use kete_flux::fitting::{self, FitResult, FitTask, FluxObs, FluxPriors, Model, ParamPrior};
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
            "FluxObs(flux={:.4e}, sigma={:.4e}, wavelength={:.0}, upper_limit={})",
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
///     ``(lo, hi)`` logistic-barrier hard bounds.
/// gaussian :
///     Optional ``(mean, sigma)`` Gaussian centering prior.
///     ``None`` means flat (uniform) within the bounds.
#[pyclass(frozen, module = "kete.flux", name = "ParamPrior", from_py_object)]
#[derive(Clone, Debug)]
pub struct PyParamPrior(pub ParamPrior);

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
        format!("{:?}", self.0)
    }
}

/// Prior configuration for all fitted parameters.
///
/// Only the scientifically meaningful knobs are exposed.  The nuisance
/// parameter ``f_sigma`` has a sensible fixed prior handled internally.
///
/// If not provided, sensible defaults are used:
///
/// .. code-block:: python
///
///     kete.flux.FluxPriors(
///         ln_diam   = kete.flux.ParamPrior(bounds=(np.ln(0.001), np.ln(1000))),
///         ln_beaming= kete.flux.ParamPrior(bounds=(np.ln(0.5), np.ln(3.0)),
///                                          gaussian=(np.ln(1.0), 0.3)),
///         ln_r_ir   = kete.flux.ParamPrior(bounds=(np.ln(0.5), np.ln(2.0)),
///                                          gaussian=(np.ln(1.6), 0.3)),
///         h_mag     = kete.flux.ParamPrior(bounds=(-5.0, 35.0)),
///         g_param   = kete.flux.ParamPrior(bounds=(-0.3, 0.7),
///                                          gaussian=(0.2, 0.01)),
///         pv        = kete.flux.ParamPrior(bounds=(0.0, 1.0)),
///     )
///
/// Each prior is a :class:`ParamPrior` specifying ``bounds`` (logistic
/// barrier) and an optional ``gaussian`` centering prior ``(mean, sigma)``.
/// To effectively fix a parameter, set tight bounds around the desired
/// value (e.g., ``bounds=(val - 1e-3, val + 1e-3)``).
///
/// Parameters
/// ----------
/// ln_diam :
///     :class:`ParamPrior` for ln(D) in km.
/// ln_beaming :
///     :class:`ParamPrior` for ln(beaming).
/// ln_r_ir :
///     :class:`ParamPrior` for ln(R_IR).
/// h_mag :
///     :class:`ParamPrior` for H magnitude.
/// g_param :
///     :class:`ParamPrior` for G parameter.
/// pv :
///     :class:`ParamPrior` for geometric albedo pV (linear scale).
#[pyclass(frozen, module = "kete.flux", name = "FluxPriors", from_py_object)]
#[derive(Clone, Debug)]
pub struct PyFluxPriors(pub FluxPriors);

#[pymethods]
impl PyFluxPriors {
    #[new]
    #[pyo3(signature = (
        ln_diam=None,
        ln_beaming=None,
        ln_r_ir=None,
        h_mag=None,
        g_param=None,
        pv=None,
    ))]
    fn new(
        ln_diam: Option<PyParamPrior>,
        ln_beaming: Option<PyParamPrior>,
        ln_r_ir: Option<PyParamPrior>,
        h_mag: Option<PyParamPrior>,
        g_param: Option<PyParamPrior>,
        pv: Option<PyParamPrior>,
    ) -> Self {
        let d = FluxPriors::default();
        Self(FluxPriors {
            ln_diam: ln_diam.map_or(d.ln_diam, |p| p.0),
            ln_beaming: ln_beaming.map_or(d.ln_beaming, |p| p.0),
            ln_r_ir: ln_r_ir.map_or(d.ln_r_ir, |p| p.0),
            h_mag: h_mag.map_or(d.h_mag, |p| p.0),
            g_param: g_param.map_or(d.g_param, |p| p.0),
            pv: pv.map_or(d.pv, |p| p.0),
        })
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.0)
    }
}

/// Summary statistics for a single fitted parameter (posterior).
///
/// Attributes
/// ----------
/// median :
///     Posterior median.
/// std :
///     Posterior standard deviation.
/// ci_lo :
///     Lower bound of the 95 % credible interval (2.5th percentile).
/// ci_hi :
///     Upper bound of the 95 % credible interval (97.5th percentile).
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
///
/// Attributes
/// ----------
/// diameter :
///     Posterior statistics for diameter (km).  ``None`` for HG model.
/// vis_albedo :
///     Posterior statistics for visual geometric albedo.  ``None`` for HG model.
/// beaming :
///     Posterior statistics for beaming (NEATM only; ``None`` for FRM/HG).
/// h_mag :
///     Posterior statistics for H magnitude.
/// g_param :
///     Posterior statistics for G parameter.
/// f_sigma :
///     Posterior statistics for uncertainty inflation factor.
/// ir_albedo_ratio :
///     Posterior statistics for IR-to-visible albedo ratio.  ``None`` for HG model.
/// model :
///     Model name string (``"Neatm"``, ``"Frm"``, or ``"Hg"``).
/// draws :
///     Raw MCMC draws as a list of vectors (each row =
///     ``[D, pV, beaming, H, G, R_IR, f_sigma]`` for NEATM,
///     ``[D, pV, H, G, R_IR, f_sigma]`` for FRM, or
///     ``[H, G, f_sigma]`` for HG).
/// divergent :
///     Per-draw divergence flags.
/// n_divergent :
///     Total number of divergent transitions.
/// chi2_best :
///     Reduced chi-squared at the MAP point using inflated uncertainties
///     ``(1/dof) * sum ((obs - model) / (f_sigma * sigma))^2`` where
///     ``dof = nobs - nparams``.  A value near 1.0 indicates a good fit.
/// nobs :
///     Number of non-upper-limit observations.
/// best_fit_fluxes :
///     Model fluxes at the MAP for each observation (Jy).
/// best_fit_residuals :
///     Normalized residuals ``(obs - model) / (f_sigma * sigma)`` at the MAP.
/// best_fit_reflected_frac :
///     Fraction of total flux from reflected light at the MAP, per observation.
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
    #[getter]
    fn model(&self) -> String {
        format!("{:?}", self.0.model)
    }
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
    #[getter]
    fn divergent(&self) -> Vec<bool> {
        self.0.divergent.clone()
    }
    #[getter]
    fn n_divergent(&self) -> usize {
        self.0.n_divergent
    }
    #[getter]
    fn chi2_best(&self) -> f64 {
        self.0.reduced_chi2
    }
    #[getter]
    fn nobs(&self) -> usize {
        self.0.nobs
    }
    #[getter]
    fn best_fit_fluxes(&self) -> Vec<f64> {
        self.0.best_fit_fluxes.clone()
    }
    #[getter]
    fn best_fit_residuals(&self) -> Vec<f64> {
        self.0.best_fit_residuals.clone()
    }
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
        format!(
            "FitResult(model={:?}, D={}, pV={}, n_draws={}, n_div={})",
            self.0.model,
            d,
            pv,
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
///     Number of MCMC chains (default 4).
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
    emissivity=0.9, priors=None, num_chains=4, num_tune=200, num_draws=500,
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
        let sigma = priors.h_mag.gaussian.map_or(0.01, |(_, s)| s);
        priors.h_mag.gaussian = Some((h, sigma));
    }
    if let Some(g) = g_param {
        let sigma = priors.g_param.gaussian.map_or(0.01, |(_, s)| s);
        priors.g_param.gaussian = Some((g, sigma));
    }

    let raw_obs = extract_obs(&obs);
    Ok(fitting::fit_mcmc(
        tm, &raw_obs, c_hg_val, emissivity, &priors, num_chains, num_tune, num_draws,
    )
    .map(PyFitResult))
}

/// Fit many objects in parallel using MCMC (rayon).
///
/// Each element of the input lists corresponds to one object.
/// Returns a list of ``FitResult`` or ``None`` per object.
///
/// ``h_mags`` and ``g_params`` are optional per-object prior-center
/// overrides (same semantics as the scalar arguments in :func:`fit_model`).
/// ``emissivities`` are fixed per-object thermal emissivity values
/// (not fitted).
///
/// Parameters
/// ----------
/// model :
///     Model name: ``"neatm"``, ``"frm"``, or ``"hg"``.
/// obs_list :
///     List of observation lists; one list of :class:`FluxObs` per object.
/// h_mags :
///     Optional H magnitude per object (sets H prior center).
/// g_params :
///     Optional G parameter per object (sets G prior center).
/// emissivities :
///     Optional emissivity per object (default 0.9, not fitted).
/// priors :
///     Common prior configuration (applies to all objects).
/// num_chains :
///     MCMC chains per object (default 4).
/// num_tune :
///     Warmup draws per chain (default 200).
/// num_draws :
///     Posterior draws per chain (default 500).
/// c_hgs :
///     Optional list of ``c_hg`` per object.
///
/// Returns
/// -------
/// list
///     One ``FitResult`` or ``None`` per object.
#[pyfunction]
#[pyo3(name = "fit_model_batch", signature = (model, obs_list, h_mags=None,
    g_params=None, emissivities=None, priors=None, num_chains=4, num_tune=200,
    num_draws=500, c_hgs=None))]
#[allow(clippy::too_many_arguments)]
pub fn fit_model_batch_py(
    model: &str,
    obs_list: Vec<Vec<PyFluxObs>>,
    h_mags: Option<Vec<Option<f64>>>,
    g_params: Option<Vec<Option<f64>>>,
    emissivities: Option<Vec<Option<f64>>>,
    priors: Option<PyFluxPriors>,
    num_chains: usize,
    num_tune: usize,
    num_draws: usize,
    c_hgs: Option<Vec<Option<f64>>>,
) -> PyResult<Vec<Option<PyFitResult>>> {
    let tm = parse_model(model)?;
    let n = obs_list.len();
    let base_priors = priors.map_or_else(FluxPriors::default, |p| p.0);
    let h_mags = h_mags.unwrap_or_else(|| vec![None; n]);
    let g_params = g_params.unwrap_or_else(|| vec![None; n]);
    let emissivities = emissivities.unwrap_or_else(|| vec![None; n]);
    let c_hgs = c_hgs.unwrap_or_else(|| vec![None; n]);

    if h_mags.len() != n || g_params.len() != n || emissivities.len() != n {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "obs_list ({n}), h_mags ({}), g_params ({}), and emissivities ({}) must all have the same length",
            h_mags.len(),
            g_params.len(),
            emissivities.len(),
        )));
    }

    let mut tasks = Vec::with_capacity(n);
    for (i, py_obs) in obs_list.into_iter().enumerate() {
        let mut p = base_priors.clone();
        let c_hg_val = c_hgs[i].unwrap_or(kete_core::constants::C_V);

        if let Some(h) = h_mags[i] {
            let sigma = p.h_mag.gaussian.map_or(1.0, |(_, s)| s);
            p.h_mag.gaussian = Some((h, sigma));
        }
        if let Some(g) = g_params[i] {
            let sigma = p.g_param.gaussian.map_or(0.2, |(_, s)| s);
            p.g_param.gaussian = Some((g, sigma));
        }

        let emissivity_val = emissivities[i].unwrap_or(0.9);

        tasks.push(FitTask {
            model: tm,
            obs: extract_obs(&py_obs),
            c_hg: c_hg_val,
            emissivity: emissivity_val,
            priors: p,
            num_chains,
            num_tune,
            num_draws,
        });
    }

    Ok(fitting::fit_batch(&tasks)
        .into_iter()
        .map(|r| r.map(PyFitResult))
        .collect())
}
