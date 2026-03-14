// BSD 3-Clause License
//
// Copyright (c) 2026, Dar Dahlen
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from
//    this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

//! Core types and helper functions shared across the fitting submodules.

use crate::{BandInfo, FrmParams, HGParams, ModelResults, NeatmParams};
use nalgebra::Vector3;

/// Which model to fit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Model {
    /// Near-Earth Asteroid Thermal Model -- beaming eta is a free parameter.
    Neatm,
    /// Fast Rotating Model -- beaming fixed at pi.
    Frm,
    /// HG reflected-light model only -- fits H and G.
    Hg,
}

impl Model {
    /// Number of free parameters.
    ///
    /// NEATM: `[ln_D, ln_eta, ln_f_sigma, ln_R_IR]` -> 4.
    /// FRM:   `[ln_D, ln_f_sigma, ln_R_IR]`          -> 3.
    /// HG:    `[H, G, ln_f_sigma]`                    -> 3.
    #[must_use]
    pub fn dim(self) -> usize {
        match self {
            Self::Neatm => 4,
            Self::Frm | Self::Hg => 3,
        }
    }

    /// Whether this is the NEATM model.
    #[must_use]
    pub fn is_neatm(self) -> bool {
        matches!(self, Self::Neatm)
    }

    /// Whether this is the HG reflected-light-only model.
    #[must_use]
    pub fn is_hg(self) -> bool {
        matches!(self, Self::Hg)
    }

    /// Indices into the log-parameter vector for `f_sigma` and `R_IR`.
    ///
    /// Only valid for thermal models (NEATM / FRM).
    pub(crate) fn nuisance_param_indices(self) -> (usize, usize) {
        match self {
            Self::Neatm => (2, 3),
            Self::Frm => (1, 2),
            Self::Hg => unreachable!("Hg has no R_IR parameter"),
        }
    }

    /// Extract physical parameters from a log-space vector.
    ///
    /// Returns `(diam, eta, f_sigma, r_ir)`.  For FRM, `eta` is pi.
    /// Only valid for thermal models (NEATM / FRM).
    pub(crate) fn unpack_thermal_params(self, x: &[f64]) -> (f64, f64, f64, f64) {
        debug_assert!(!self.is_hg(), "unpack is not valid for Hg");
        let diam = x[0].exp();
        let (fi, ri) = self.nuisance_param_indices();
        let f_sigma = x[fi].exp();
        let r_ir = x[ri].exp();
        let eta = if self.is_neatm() {
            x[1].exp()
        } else {
            std::f64::consts::PI
        };
        (diam, eta, f_sigma, r_ir)
    }

    /// Compute apparent total fluxes for the given geometry.
    pub(crate) fn compute_fluxes(
        self,
        bands: Vec<BandInfo>,
        band_albedos: Vec<f64>,
        eta: f64,
        hg: &HGParams,
        emissivity: f64,
        sun2obj: &Vector3<f64>,
        sun2obs: &Vector3<f64>,
    ) -> Option<ModelResults> {
        match self {
            Self::Neatm => {
                let params = NeatmParams {
                    obs_bands: bands,
                    band_albedos,
                    beaming: eta,
                    hg_params: hg.clone(),
                    emissivity,
                };
                params.apparent_total_flux(sun2obj, sun2obs)
            }
            Self::Frm => {
                let params = FrmParams {
                    obs_bands: bands,
                    band_albedos,
                    hg_params: hg.clone(),
                    emissivity,
                };
                params.apparent_total_flux(sun2obj, sun2obs)
            }
            Self::Hg => unreachable!("Hg uses its own forward-model path"),
        }
    }
}

/// A single flux observation at a known geometry.
#[derive(Debug, Clone)]
pub struct FluxObs {
    /// Observed flux in Jy (or upper-limit threshold if `is_upper_limit`).
    pub flux: f64,
    /// 1-sigma uncertainty in Jy.
    pub sigma: f64,
    /// Band information for this observation.
    pub band: BandInfo,
    /// If true, `flux` is a non-detection upper limit, not a measurement.
    pub is_upper_limit: bool,
    /// Sun-to-object vector in AU (Ecliptic frame).
    pub sun2obj: Vector3<f64>,
    /// Sun-to-observer vector in AU (Ecliptic frame).
    pub sun2obs: Vector3<f64>,
}

/// Prior configuration for model fitting.
///
/// Thermal models (NEATM/FRM) use `ln_diam_bounds`, beaming, and `R_IR` priors.
/// The HG model uses `h_mag_prior` and `h_mag_bounds`.
/// In all cases `f_sigma` has a sensible fixed prior handled internally.
#[derive(Debug, Clone)]
pub struct Priors {
    /// (lo, hi) logistic-barrier bounds for ln D (km).  Used by NEATM/FRM.
    pub ln_diam_bounds: (f64, f64),
    /// Optional Gaussian prior (mean, sigma) on ln(eta) (beaming).
    /// `None` means no Gaussian centering -- only hard bounds apply.
    pub ln_beaming_prior: Option<(f64, f64)>,
    /// (lo, hi) logistic-barrier bounds for ln(eta).  Used by NEATM.
    pub ln_beaming_bounds: (f64, f64),
    /// Optional Gaussian prior (mean, sigma) on `ln(R_IR)`.
    /// `None` means no Gaussian centering -- only hard bounds apply.
    /// Used by NEATM/FRM.
    pub ln_r_ir_prior: Option<(f64, f64)>,
    /// (lo, hi) logistic-barrier bounds for `ln(R_IR)`.  Used by NEATM/FRM.
    pub ln_r_ir_bounds: (f64, f64),
    /// Optional Gaussian prior (mean, sigma) on H magnitude.  Used by HG.
    pub h_mag_prior: Option<(f64, f64)>,
    /// (lo, hi) logistic-barrier bounds for H magnitude.  Used by HG.
    pub h_mag_bounds: (f64, f64),
    /// Optional Gaussian prior (mean, sigma) on G parameter.  Used by HG.
    /// `None` means no Gaussian centering -- only hard bounds apply.
    pub g_param_prior: Option<(f64, f64)>,
    /// (lo, hi) logistic-barrier bounds for G parameter.  Used by HG.
    pub g_param_bounds: (f64, f64),
}

impl Default for Priors {
    /// Sensible defaults:
    /// - D in [0.001, 1000] km
    /// - eta in [0.5, 3.0], Gaussian (ln 1.0, 0.3)
    /// - `R_IR` in [0.5, 2.0], Gaussian (ln 1.6, 0.3)
    /// - H in [-5, 35]
    /// - G in [-0.3, 0.7], Gaussian (0.2, 0.2)
    fn default() -> Self {
        Self {
            ln_diam_bounds: (0.001_f64.ln(), 1000.0_f64.ln()),
            ln_beaming_prior: Some((1.0_f64.ln(), 0.3)),
            ln_beaming_bounds: (0.5_f64.ln(), 3.0_f64.ln()),
            ln_r_ir_prior: Some((1.6_f64.ln(), 0.3)),
            ln_r_ir_bounds: (0.5_f64.ln(), 2.0_f64.ln()),
            h_mag_prior: None,
            h_mag_bounds: (-5.0, 35.0),
            g_param_prior: Some((0.2, 0.2)),
            g_param_bounds: (-0.3, 0.7),
        }
    }
}

/// Logistic barrier prior: smooth wall that is 0 in the interior and -> -inf at
/// the boundaries.
///
/// $$\ln\sigma(k(x - lo)) + \ln\sigma(k(hi - x))$$
///
/// where $\sigma$ is the logistic sigmoid.
///
/// # Arguments
/// * `x`  -- parameter value
/// * `lo` -- lower bound
/// * `hi` -- upper bound
/// * `k`  -- steepness (larger = sharper wall; 30 is typical)
#[must_use]
pub fn logistic_barrier(x: f64, lo: f64, hi: f64, k: f64) -> f64 {
    // ln(sigmoid(z)) = z - ln(1 + exp(z)) but for numerical stability use -ln(1+exp(-z))
    // which is equivalent and avoids overflow for large positive z.
    fn log_sigmoid(z: f64) -> f64 {
        if z > 0.0 {
            -(-z).exp().ln_1p()
        } else {
            z - z.exp().ln_1p()
        }
    }
    log_sigmoid(k * (x - lo)) + log_sigmoid(k * (hi - x))
}

/// Gaussian log-prior: -(x - mu)^2 / (2 sigma^2).
pub(super) fn gaussian_log_prior(x: f64, mean: f64, sigma: f64) -> f64 {
    let z = (x - mean) / sigma;
    -0.5 * z * z
}

/// Given a diameter D (km) and the [`HGParams`] (which hold H and `c_hg`),
/// compute the visual geometric albedo pV.
///
/// pV = (`c_hg` / D)^2 * 10^(-2H/5)
pub(super) fn pv_from_diameter(diam_km: f64, hg: &HGParams) -> f64 {
    let ratio = hg.c_hg() / diam_km;
    ratio * ratio * 10_f64.powf(-0.4 * hg.h_mag)
}

/// Result of evaluating the forward model at a parameter point.
pub(super) struct ForwardModelResult {
    /// Model flux per observation (Jy), in obs-global index order.
    pub model_fluxes: Vec<f64>,
    /// Reflected-light fraction per observation.
    pub reflected_frac: Vec<f64>,
    /// Uncertainty inflation factor.
    pub f_sigma: f64,
}

/// Evaluate the forward model at a parameter point.
///
/// For thermal models (NEATM/FRM), observations sharing the same geometry
/// are batched into a single model call automatically.
/// For HG, each observation's reflected flux is computed independently.
///
/// Returns `None` if the derived albedo is infeasible or the model fails.
pub(super) fn evaluate_forward_model(
    model: Model,
    x: &[f64],
    obs: &[FluxObs],
    hg: &HGParams,
    emissivity: f64,
) -> Option<ForwardModelResult> {
    if model.is_hg() {
        return evaluate_hg_forward_model(x, obs, hg);
    }

    let (diam, eta, f_sigma, r_ir) = model.unpack_thermal_params(x);

    let pv = pv_from_diameter(diam, hg);
    if !pv.is_finite() || pv <= 0.0 || pv > 2.0 {
        return None;
    }

    let hg_model = HGParams::try_new(
        hg.desig.clone(),
        hg.g_param,
        Some(hg.h_mag),
        Some(hg.c_hg()),
        Some(pv),
        Some(diam),
    )
    .unwrap_or_else(|_| hg.clone());

    let n = obs.len();
    let mut model_fluxes = vec![0.0; n];
    let mut reflected_frac = vec![0.0; n];

    // Group observations by geometry so the thermal model is called once per
    // unique (sun2obj, sun2obs) pair.
    let mut remaining: Vec<bool> = vec![true; n];
    for anchor in 0..n {
        if !remaining[anchor] {
            continue;
        }
        remaining[anchor] = false;
        let mut indices = vec![anchor];
        for j in (anchor + 1)..n {
            if remaining[j]
                && obs[j].sun2obj == obs[anchor].sun2obj
                && obs[j].sun2obs == obs[anchor].sun2obs
            {
                remaining[j] = false;
                indices.push(j);
            }
        }
        let bands: Vec<BandInfo> = indices.iter().map(|&i| obs[i].band.clone()).collect();
        let band_albedos: Vec<f64> = bands.iter().map(|_| r_ir * pv).collect();
        let result = model.compute_fluxes(
            bands,
            band_albedos,
            eta,
            &hg_model,
            emissivity,
            &obs[anchor].sun2obj,
            &obs[anchor].sun2obs,
        )?;
        let rf = result.reflected_fraction();
        for (local, &obs_idx) in indices.iter().enumerate() {
            model_fluxes[obs_idx] = result.fluxes[local];
            reflected_frac[obs_idx] = rf[local];
        }
    }

    Some(ForwardModelResult {
        model_fluxes,
        reflected_frac,
        f_sigma,
    })
}

/// HG reflected-light-only forward model.
///
/// Parameters: `x = [H, G, ln_f_sigma]`.
/// Computes reflected flux via `HGParams::apparent_flux` per observation.
fn evaluate_hg_forward_model(
    x: &[f64],
    obs: &[FluxObs],
    hg: &HGParams,
) -> Option<ForwardModelResult> {
    let h = x[0];
    let g = x[1];
    let f_sigma = x[2].exp();

    // Build HGParams with the fitted H and G.  Choose D so that pV = 1;
    // the product D^2 * pV = c_hg^2 * 10^(-2H/5) is determined by H alone.
    let diam = hg.c_hg() * 10_f64.powf(-h / 5.0);
    if !diam.is_finite() || diam <= 0.0 {
        return None;
    }
    let hg_model = HGParams::try_new(
        hg.desig.clone(),
        g,
        Some(h),
        Some(hg.c_hg()),
        Some(1.0),
        Some(diam),
    )
    .ok()?;

    let n = obs.len();
    let mut model_fluxes = Vec::with_capacity(n);
    for ob in obs {
        let flux = hg_model.apparent_flux(&ob.sun2obj, &ob.sun2obs, ob.band.wavelength, 1.0)?
            * ob.band.solar_correction;
        model_fluxes.push(flux);
    }

    Some(ForwardModelResult {
        model_fluxes,
        reflected_frac: vec![1.0; n],
        f_sigma,
    })
}

/// Degrees of freedom for the Student-t likelihood.
pub(super) const STUDENT_NU: f64 = 5.0;

/// Steepness of logistic barriers (sharper = closer to hard wall).
pub(super) const BARRIER_K: f64 = 30.0;

/// Evaluate the Student-t(nu=5) log-likelihood at log-parameter vector `x`.
///
/// Returns a value to be **maximized** (negative of the NLL).
/// Returns `f64::NEG_INFINITY` for infeasible points.
pub(super) fn log_likelihood(
    model: Model,
    x: &[f64],
    obs: &[FluxObs],
    hg: &HGParams,
    emissivity: f64,
) -> f64 {
    let Some(fwd) = evaluate_forward_model(model, x, obs, hg, emissivity) else {
        return f64::NEG_INFINITY;
    };

    let nu = STUDENT_NU;
    let mut ll = 0.0;
    for (i, ob) in obs.iter().enumerate() {
        let mf = fwd.model_fluxes[i];
        let sigma_eff = fwd.f_sigma * ob.sigma;
        let sigma2 = sigma_eff * sigma_eff;
        if ob.is_upper_limit {
            if mf > ob.flux {
                let r = mf - ob.flux;
                ll += -0.5 * (nu + 1.0) * (1.0 + r * r / (nu * sigma2)).ln();
            }
        } else {
            let r = ob.flux - mf;
            ll += -sigma_eff.ln() - 0.5 * (nu + 1.0) * (1.0 + r * r / (nu * sigma2)).ln();
        }
    }

    if ll.is_finite() {
        ll
    } else {
        f64::NEG_INFINITY
    }
}

/// Evaluate the log-prior at log-parameter vector `x`.
///
/// For thermal models: diameter, beaming, and `R_IR` priors plus a hardcoded
/// nuisance prior for `f_sigma`.
/// For HG: H-magnitude prior plus hardcoded `f_sigma` prior.
pub(super) fn log_prior(model: Model, x: &[f64], priors: &Priors) -> f64 {
    let mut lp = 0.0;

    if model.is_hg() {
        // H magnitude bounds + optional Gaussian.
        let h = x[0];
        lp += logistic_barrier(h, priors.h_mag_bounds.0, priors.h_mag_bounds.1, BARRIER_K);
        if let Some((mean, sigma)) = priors.h_mag_prior {
            lp += gaussian_log_prior(h, mean, sigma);
        }

        // G parameter bounds + optional Gaussian.
        let g = x[1];
        lp += logistic_barrier(
            g,
            priors.g_param_bounds.0,
            priors.g_param_bounds.1,
            BARRIER_K,
        );
        if let Some((mean, sigma)) = priors.g_param_prior {
            lp += gaussian_log_prior(g, mean, sigma);
        }

        // f_sigma at index 2.
        lp += gaussian_log_prior(x[2], 1.5_f64.ln(), 0.3);
        lp += logistic_barrier(x[2], 0.5_f64.ln(), 5.0_f64.ln(), BARRIER_K);

        return lp;
    }

    let (fi, ri) = model.nuisance_param_indices();

    // Diameter bounds.
    lp += logistic_barrier(
        x[0],
        priors.ln_diam_bounds.0,
        priors.ln_diam_bounds.1,
        BARRIER_K,
    );

    // Beaming (NEATM only): bounds + optional Gaussian.
    if model.is_neatm() {
        let ln_eta = x[1];
        lp += logistic_barrier(
            ln_eta,
            priors.ln_beaming_bounds.0,
            priors.ln_beaming_bounds.1,
            BARRIER_K,
        );
        if let Some((mean, sigma)) = priors.ln_beaming_prior {
            lp += gaussian_log_prior(ln_eta, mean, sigma);
        }
    }

    // f_sigma: Gaussian(ln 1.5, 0.3) + bounds [ln 0.5, ln 5.0].
    lp += gaussian_log_prior(x[fi], 1.5_f64.ln(), 0.3);
    lp += logistic_barrier(x[fi], 0.5_f64.ln(), 5.0_f64.ln(), BARRIER_K);

    // R_IR: configurable Gaussian + bounds.
    lp += logistic_barrier(
        x[ri],
        priors.ln_r_ir_bounds.0,
        priors.ln_r_ir_bounds.1,
        BARRIER_K,
    );
    if let Some((mean, sigma)) = priors.ln_r_ir_prior {
        lp += gaussian_log_prior(x[ri], mean, sigma);
    }

    lp
}

/// Evaluate the full log-posterior = log-likelihood + log-prior.
///
/// Returns a value to be **maximized**.
/// Returns `f64::NEG_INFINITY` for infeasible points.
pub(super) fn log_posterior(
    model: Model,
    x: &[f64],
    obs: &[FluxObs],
    hg: &HGParams,
    emissivity: f64,
    priors: &Priors,
) -> f64 {
    let ll = log_likelihood(model, x, obs, hg, emissivity);
    if !ll.is_finite() {
        return f64::NEG_INFINITY;
    }
    let val = ll + log_prior(model, x, priors);
    if val.is_finite() {
        val
    } else {
        f64::NEG_INFINITY
    }
}
