//! PyO3 bindings for model fitting (NEATM/FRM/HG).
//!
//! Exposes [`kete_flux::fitting`] types and functions to Python under
//! `kete.flux`.

use crate::frame::PyFrames;
use crate::vector::VectorLike;
use kete_flux::fitting::{self, FitResult, FitTask, FluxObs, Model, Priors};
use kete_flux::{BandInfo, HGParams};
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
///
/// Only the scientifically meaningful knobs are exposed.  The nuisance
/// parameter ``f_sigma`` has a sensible fixed prior handled internally.
///
/// If not provided, sensible defaults are used:
///
/// - D in [0.001, 1000] km
/// - eta in [0.5, 3.0], Gaussian prior at ln(1.0) with sigma=0.3
/// - R_IR in [0.5, 2.0], Gaussian prior at ln(1.6) with sigma=0.3
/// - H in [-5, 35]
/// - G in [-0.12, 0.7], Gaussian prior at 0.15 with sigma=0.1
///
/// Parameters
/// ----------
/// ln_diam_bounds :
///     (lo, hi) logistic-barrier bounds for ln(D) in km.
/// ln_beaming_prior :
///     Optional (mean, sigma) Gaussian prior on ln(eta).
/// ln_beaming_bounds :
///     (lo, hi) logistic-barrier bounds for ln(eta).
/// disable_beaming_prior :
///     If ``True``, remove the Gaussian centering prior on ln(eta),
///     leaving only the hard bounds.  Default ``False``.
/// ln_r_ir_prior :
///     Optional (mean, sigma) Gaussian prior on ln(R_IR).
/// ln_r_ir_bounds :
///     (lo, hi) logistic-barrier bounds for ln(R_IR).
/// disable_r_ir_prior :
///     If ``True``, remove the Gaussian centering prior on ln(R_IR),
///     leaving only the hard bounds.  Default ``False``.
/// h_mag_prior :
///     Optional (mean, sigma) Gaussian prior on H magnitude (HG model).
/// h_mag_bounds :
///     (lo, hi) logistic-barrier bounds for H magnitude (HG model).
/// g_param_prior :
///     Optional (mean, sigma) Gaussian prior on G parameter (HG model).
/// g_param_bounds :
///     (lo, hi) logistic-barrier bounds for G parameter (HG model).
/// disable_g_param_prior :
///     If ``True``, remove the Gaussian centering prior on G,
///     leaving only the hard bounds.  Default ``False``.
#[pyclass(frozen, module = "kete.flux", name = "Priors", from_py_object)]
#[derive(Clone, Debug)]
pub struct PyPriors(pub Priors);

#[pymethods]
impl PyPriors {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        ln_diam_bounds=None,
        ln_beaming_prior=None,
        ln_beaming_bounds=None,
        disable_beaming_prior=false,
        ln_r_ir_prior=None,
        ln_r_ir_bounds=None,
        disable_r_ir_prior=false,
        h_mag_prior=None,
        h_mag_bounds=None,
        g_param_prior=None,
        g_param_bounds=None,
        disable_g_param_prior=false,
    ))]
    fn new(
        ln_diam_bounds: Option<(f64, f64)>,
        ln_beaming_prior: Option<(f64, f64)>,
        ln_beaming_bounds: Option<(f64, f64)>,
        disable_beaming_prior: bool,
        ln_r_ir_prior: Option<(f64, f64)>,
        ln_r_ir_bounds: Option<(f64, f64)>,
        disable_r_ir_prior: bool,
        h_mag_prior: Option<(f64, f64)>,
        h_mag_bounds: Option<(f64, f64)>,
        g_param_prior: Option<(f64, f64)>,
        g_param_bounds: Option<(f64, f64)>,
        disable_g_param_prior: bool,
    ) -> Self {
        let d = Priors::default();
        let beaming = if disable_beaming_prior {
            None
        } else {
            ln_beaming_prior.or(d.ln_beaming_prior)
        };
        let r_ir = if disable_r_ir_prior {
            None
        } else {
            ln_r_ir_prior.or(d.ln_r_ir_prior)
        };
        let g = if disable_g_param_prior {
            None
        } else {
            g_param_prior.or(d.g_param_prior)
        };
        Self(Priors {
            ln_diam_bounds: ln_diam_bounds.unwrap_or(d.ln_diam_bounds),
            ln_beaming_prior: beaming,
            ln_beaming_bounds: ln_beaming_bounds.unwrap_or(d.ln_beaming_bounds),
            ln_r_ir_prior: r_ir,
            ln_r_ir_bounds: ln_r_ir_bounds.unwrap_or(d.ln_r_ir_bounds),
            h_mag_prior,
            h_mag_bounds: h_mag_bounds.unwrap_or(d.h_mag_bounds),
            g_param_prior: g,
            g_param_bounds: g_param_bounds.unwrap_or(d.g_param_bounds),
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
///     Posterior statistics for beaming eta (NEATM only; ``None`` for FRM/HG).
/// h_mag :
///     Posterior statistics for H magnitude (HG only; ``None`` for thermal models).
/// f_sigma :
///     Posterior statistics for uncertainty inflation factor.
/// ir_albedo_ratio :
///     Posterior statistics for IR-to-visible albedo ratio.  ``None`` for HG model.
/// model :
///     Model name string (``"Neatm"``, ``"Frm"``, or ``"Hg"``).
/// draws :
///     Raw MCMC draws as a list of vectors (each row =
///     ``[D, pV, eta, f_sigma, R_IR]`` for NEATM,
///     ``[D, pV, f_sigma, R_IR]`` for FRM, or
///     ``[H, f_sigma]`` for HG).
/// divergent :
///     Per-draw divergence flags.
/// n_divergent :
///     Total number of divergent transitions.
/// chi2_best :
///     chi^2 at the MAP point using inflated uncertainties
///     ``sum ((obs - model) / (f_sigma * sigma))^2`` (for diagnostics).
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
    /// Posterior statistics for beaming eta (NEATM only).  ``None`` for FRM/HG.
    #[getter]
    fn beaming(&self) -> Option<PySampleStats> {
        self.0
            .model
            .is_neatm()
            .then(|| stats_from_column(&self.0.draws, 2))
    }
    /// Posterior statistics for H magnitude (HG only).  ``None`` for thermal models.
    #[getter]
    fn h_mag(&self) -> Option<PySampleStats> {
        self.0
            .model
            .is_hg()
            .then(|| stats_from_column(&self.0.draws, 0))
    }
    /// Posterior statistics for G parameter (HG only).  ``None`` for thermal models.
    #[getter]
    fn g_param(&self) -> Option<PySampleStats> {
        self.0
            .model
            .is_hg()
            .then(|| stats_from_column(&self.0.draws, 1))
    }
    /// Posterior statistics for uncertainty inflation factor.
    #[getter]
    fn f_sigma(&self) -> PySampleStats {
        let col = match self.0.model {
            Model::Neatm => 3,
            Model::Frm | Model::Hg => 2,
        };
        stats_from_column(&self.0.draws, col)
    }
    /// Posterior statistics for IR-to-visible albedo ratio.  ``None`` for HG model.
    #[getter]
    fn ir_albedo_ratio(&self) -> Option<PySampleStats> {
        if self.0.model.is_hg() {
            return None;
        }
        let col = if self.0.model.is_neatm() { 4 } else { 3 };
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
                "f_sigma",
                "ir_albedo_ratio",
            ],
            Model::Frm => vec!["diameter", "vis_albedo", "f_sigma", "ir_albedo_ratio"],
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
        self.0.chi2_best
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
/// Parameters
/// ----------
/// model :
///     Model name: ``"neatm"``, ``"frm"``, or ``"hg"``.
/// obs :
///     List of :class:`FluxObs` observations.
/// h_mag :
///     Absolute H magnitude (HG system).
/// g_param :
///     Phase slope parameter G (default 0.15).
/// emissivity :
///     Surface emissivity (default 0.9).
/// priors :
///     Prior configuration (:class:`Priors`, default if ``None``).
/// num_chains :
///     Number of MCMC chains (default 4).
/// num_tune :
///     Warmup draws per chain (default 200).
/// num_draws :
///     Posterior draws per chain (default 500).
/// c_hg :
///     HG relationship constant (default 1329.0).
/// diameter :
///     Known diameter in km (alternative to ``h_mag``).
/// vis_albedo :
///     Known visible geometric albedo.
///
/// Returns
/// -------
/// FitResult or None
///     MCMC posterior results, or ``None`` if the fit fails.
#[pyfunction]
#[pyo3(name = "fit_model", signature = (model, obs, h_mag=None, g_param=0.15,
    emissivity=0.9, priors=None, num_chains=4, num_tune=200, num_draws=500,
    c_hg=None, diameter=None, vis_albedo=None))]
#[allow(clippy::too_many_arguments)]
pub fn fit_model_py(
    model: &str,
    obs: Vec<PyFluxObs>,
    h_mag: Option<f64>,
    g_param: f64,
    emissivity: f64,
    priors: Option<PyPriors>,
    num_chains: usize,
    num_tune: usize,
    num_draws: usize,
    c_hg: Option<f64>,
    diameter: Option<f64>,
    vis_albedo: Option<f64>,
) -> PyResult<Option<PyFitResult>> {
    let tm = parse_model(model)?;
    let priors = priors.map_or_else(Priors::default, |p| p.0);
    // For HG fitting, H is a fitted parameter so it need not be provided.
    // Use the midpoint of the prior bounds as a Nelder-Mead seed.
    let hg = if tm.is_hg() && h_mag.is_none() && vis_albedo.is_none() && diameter.is_none() {
        let seed_h = 0.5 * (priors.h_mag_bounds.0 + priors.h_mag_bounds.1);
        HGParams::new("".into(), g_param, seed_h, c_hg)
    } else {
        HGParams::try_new("".into(), g_param, h_mag, c_hg, vis_albedo, diameter)?
    };
    let raw_obs = extract_obs(&obs);
    Ok(fitting::fit_mcmc(
        tm, &raw_obs, &hg, emissivity, &priors, num_chains, num_tune, num_draws,
    )
    .map(PyFitResult))
}

/// Fit many objects in parallel using MCMC (rayon).
///
/// Each element of the input lists corresponds to one object.
/// Returns a list of ``FitResult`` or ``None`` per object.
///
/// Parameters
/// ----------
/// model :
///     Model name: ``"neatm"``, ``"frm"``, or ``"hg"``.
/// obs_list :
///     List of observation lists; one list of :class:`FluxObs` per object.
/// h_mags :
///     Absolute H magnitude per object.
/// g_params :
///     Phase slope G per object.
/// emissivities :
///     Emissivity per object.
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
/// diameters :
///     Optional list of known diameters (km) per object.
/// vis_albedos :
///     Optional list of known visible albedos per object.
///
/// Returns
/// -------
/// list
///     One ``FitResult`` or ``None`` per object.
#[pyfunction]
#[pyo3(name = "fit_model_batch", signature = (model, obs_list, h_mags, g_params,
    emissivities, priors=None, num_chains=4, num_tune=200, num_draws=500,
    c_hgs=None, diameters=None, vis_albedos=None))]
#[allow(clippy::too_many_arguments)]
pub fn fit_model_batch_py(
    model: &str,
    obs_list: Vec<Vec<PyFluxObs>>,
    h_mags: Vec<Option<f64>>,
    g_params: Vec<f64>,
    emissivities: Vec<f64>,
    priors: Option<PyPriors>,
    num_chains: usize,
    num_tune: usize,
    num_draws: usize,
    c_hgs: Option<Vec<Option<f64>>>,
    diameters: Option<Vec<Option<f64>>>,
    vis_albedos: Option<Vec<Option<f64>>>,
) -> PyResult<Vec<Option<PyFitResult>>> {
    let tm = parse_model(model)?;
    let n = obs_list.len();
    if h_mags.len() != n || g_params.len() != n || emissivities.len() != n {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "obs_list ({n}), h_mags ({}), g_params ({}), and emissivities ({}) must all have the same length",
            h_mags.len(),
            g_params.len(),
            emissivities.len(),
        )));
    }
    let priors = priors.map_or_else(Priors::default, |p| p.0);
    let c_hgs = c_hgs.unwrap_or_else(|| vec![None; n]);
    let diameters = diameters.unwrap_or_else(|| vec![None; n]);
    let vis_albedos = vis_albedos.unwrap_or_else(|| vec![None; n]);

    let mut tasks = Vec::with_capacity(n);
    for (i, py_obs) in obs_list.into_iter().enumerate() {
        let hg = if tm.is_hg()
            && h_mags[i].is_none()
            && vis_albedos[i].is_none()
            && diameters[i].is_none()
        {
            let seed_h = 0.5 * (priors.h_mag_bounds.0 + priors.h_mag_bounds.1);
            HGParams::new("".into(), g_params[i], seed_h, c_hgs[i])
        } else {
            HGParams::try_new(
                "".into(),
                g_params[i],
                h_mags[i],
                c_hgs[i],
                vis_albedos[i],
                diameters[i],
            )?
        };
        tasks.push(FitTask {
            model: tm,
            obs: extract_obs(&py_obs),
            hg,
            emissivity: emissivities[i],
            priors: priors.clone(),
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
