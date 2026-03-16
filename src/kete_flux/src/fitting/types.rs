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

use crate::{BandInfo, ModelResults, albedo_from_h_mag_diam, frm_total_flux, neatm_total_flux};
use nalgebra::Vector3;

/// Which model to fit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Model {
    /// Near-Earth Asteroid Thermal Model -- beaming is a free parameter.
    Neatm,
    /// Fast Rotating Model -- beaming fixed at pi.
    Frm,
    /// HG reflected-light model only -- fits H and G.
    Hg,
}

impl Model {
    /// Number of free parameters.
    ///
    /// NEATM: `[ln_D, ln_beaming, H, G, ln_f_sigma, ln_R_IR]` -> 6.
    /// FRM:   `[ln_D, H, G, ln_f_sigma, ln_R_IR]`          -> 5.
    /// HG:    `[H, G, ln_f_sigma]`                          -> 3.
    #[must_use]
    pub fn dim(self) -> usize {
        match self {
            Self::Neatm => 6,
            Self::Frm => 5,
            Self::Hg => 3,
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

    /// Extract physical parameters from the log-space vector.
    ///
    /// Only valid for thermal models (NEATM / FRM).
    /// Emissivity is not in the vector -- it must be supplied separately.
    pub(crate) fn unpack_thermal_params(self, x: &[f64], emissivity: f64) -> ThermalParams {
        debug_assert!(!self.is_hg(), "unpack is not valid for Hg");
        let diam = x[0].exp();
        match self {
            Self::Neatm => ThermalParams {
                diam,
                beaming: x[1].exp(),
                h_mag: x[2],
                g_param: x[3],
                emissivity,
                f_sigma: x[4].exp(),
                r_ir: x[5].exp(),
            },
            Self::Frm => ThermalParams {
                diam,
                beaming: std::f64::consts::PI,
                h_mag: x[1],
                g_param: x[2],
                emissivity,
                f_sigma: x[3].exp(),
                r_ir: x[4].exp(),
            },
            Self::Hg => unreachable!(),
        }
    }
}

/// Physical parameters extracted from the log-space parameter vector.
///
/// Used for thermal models (NEATM / FRM).
pub(crate) struct ThermalParams {
    pub diam: f64,
    pub beaming: f64,
    pub h_mag: f64,
    pub g_param: f64,
    pub emissivity: f64,
    pub f_sigma: f64,
    pub r_ir: f64,
}

impl Model {
    /// Compute apparent total fluxes for the given geometry.
    pub(crate) fn compute_fluxes(
        self,
        bands: &[BandInfo],
        band_albedos: &[f64],
        beaming: f64,
        g_param: f64,
        h_mag: f64,
        vis_albedo: f64,
        diam: f64,
        emissivity: f64,
        sun2obj: &Vector3<f64>,
        sun2obs: &Vector3<f64>,
    ) -> ModelResults {
        match self {
            Self::Neatm => neatm_total_flux(
                bands,
                band_albedos,
                diam,
                vis_albedo,
                g_param,
                h_mag,
                beaming,
                emissivity,
                sun2obj,
                sun2obs,
            ),
            Self::Frm => frm_total_flux(
                bands,
                band_albedos,
                diam,
                vis_albedo,
                g_param,
                h_mag,
                emissivity,
                sun2obj,
                sun2obs,
            ),
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

/// Configuration for a single fitted parameter's prior.
///
/// Each parameter has:
/// - `bounds`: `(lo, hi)` logistic-barrier hard bounds.
/// - `gaussian`: Optional `(mean, sigma)` Gaussian centering prior.
///
/// When `gaussian` is `Some((mean, sigma))`, the posterior is pulled toward
/// `mean`.  When `gaussian` is `None`, only the hard bounds apply
/// (flat/uniform prior within the bounded region).
///
/// To effectively fix a parameter to a value, set tight bounds around it
/// (e.g., `bounds = (val - 0.001, val + 0.001)`) -- the logistic barrier
/// will constrain samples to that narrow range.
#[derive(Debug, Clone)]
pub struct ParamPrior {
    /// (lo, hi) logistic-barrier bounds.
    pub bounds: (f64, f64),
    /// Optional Gaussian centering prior (mean, sigma).
    /// `None` means flat prior within bounds.
    pub gaussian: Option<(f64, f64)>,
}

impl ParamPrior {
    /// Create a prior with only hard bounds (flat/uniform within the range).
    #[must_use]
    pub fn bounds_only(lo: f64, hi: f64) -> Self {
        Self {
            bounds: (lo, hi),
            gaussian: None,
        }
    }

    /// Create a prior with hard bounds and a Gaussian center.
    #[must_use]
    pub fn with_gaussian(lo: f64, hi: f64, mean: f64, sigma: f64) -> Self {
        Self {
            bounds: (lo, hi),
            gaussian: Some((mean, sigma)),
        }
    }

    /// Evaluate the log-prior contribution for this parameter.
    pub(super) fn log_prob(&self, x: f64) -> f64 {
        let mut lp = logistic_barrier(x, self.bounds.0, self.bounds.1, BARRIER_K);
        if let Some((mean, sigma)) = self.gaussian {
            lp += gaussian_log_prior(x, mean, sigma);
        }
        lp
    }

    /// Midpoint of the bounds, or the Gaussian mean if set.
    pub(crate) fn center(&self) -> f64 {
        self.gaussian
            .map_or(0.5 * (self.bounds.0 + self.bounds.1), |(m, _)| m)
    }
}

/// Prior configuration for model fitting.
///
/// Each fitted parameter is configured via a [`ParamPrior`] with hard bounds
/// and an optional Gaussian center.  The nuisance parameter `f_sigma` has
/// a sensible fixed prior handled internally.
#[derive(Debug, Clone)]
pub struct FluxPriors {
    /// Prior on ln(D) (diameter in km).  Used by NEATM/FRM.
    pub ln_diam: ParamPrior,
    /// Prior on ln(beaming).  Used by NEATM.
    pub ln_beaming: ParamPrior,
    /// Prior on `ln(R_IR)` (IR-to-visible albedo ratio).  Used by NEATM/FRM.
    pub ln_r_ir: ParamPrior,
    /// Prior on H magnitude.
    pub h_mag: ParamPrior,
    /// Prior on G parameter.
    pub g_param: ParamPrior,
    /// Prior on geometric albedo pV (linear scale).
    /// Used by thermal models to penalize infeasible albedos derived from D and H.
    pub pv: ParamPrior,
}

impl Default for FluxPriors {
    /// Sensible defaults:
    /// - D in [0.001, 1000] km (bounds only)
    /// - beaming in [0.5, 3.0], Gaussian (ln 1.0, 0.3)
    /// - `R_IR` in [0.5, 2.0], Gaussian (ln 1.6, 0.3)
    /// - H in [-5, 35] (bounds only)
    /// - G in [-0.3, 0.7], Gaussian (0.2, 0.01)
    fn default() -> Self {
        Self {
            ln_diam: ParamPrior::bounds_only(0.001_f64.ln(), 1000.0_f64.ln()),
            ln_beaming: ParamPrior::with_gaussian(0.5_f64.ln(), 3.0_f64.ln(), 1.0_f64.ln(), 0.3),
            ln_r_ir: ParamPrior::with_gaussian(0.5_f64.ln(), 2.0_f64.ln(), 1.6_f64.ln(), 0.3),
            h_mag: ParamPrior::bounds_only(-5.0, 35.0),
            g_param: ParamPrior::with_gaussian(-0.3, 0.7, 0.2, 0.01),
            pv: ParamPrior::bounds_only(0.0, 1.0),
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

/// Given a diameter D (km), H magnitude, and `c_hg` constant,
/// compute the visual geometric albedo pV.
///
/// pV = (`c_hg` / D)^2 * 10^(-2H/5)
pub(super) fn pv_from_diameter(diam_km: f64, h_mag: f64, c_hg: f64) -> f64 {
    albedo_from_h_mag_diam(h_mag, diam_km, c_hg)
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
    c_hg: f64,
    emissivity: f64,
) -> Option<ForwardModelResult> {
    if model.is_hg() {
        return evaluate_hg_forward_model(x, obs, c_hg);
    }

    let tp = model.unpack_thermal_params(x, emissivity);

    let pv = pv_from_diameter(tp.diam, tp.h_mag, c_hg);
    if !pv.is_finite() || pv <= 0.0 || pv > 2.0 {
        return None;
    }

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
        let band_albedos: Vec<f64> = bands.iter().map(|_| tp.r_ir * pv).collect();
        let result = model.compute_fluxes(
            &bands,
            &band_albedos,
            tp.beaming,
            tp.g_param,
            tp.h_mag,
            pv,
            tp.diam,
            tp.emissivity,
            &obs[anchor].sun2obj,
            &obs[anchor].sun2obs,
        );
        let rf = result.reflected_fraction();
        for (local, &obs_idx) in indices.iter().enumerate() {
            model_fluxes[obs_idx] = result.fluxes[local];
            reflected_frac[obs_idx] = rf[local];
        }
    }

    Some(ForwardModelResult {
        model_fluxes,
        reflected_frac,
        f_sigma: tp.f_sigma,
    })
}

/// HG reflected-light-only forward model.
///
/// Parameters: `x = [H, G, ln_f_sigma]`.
/// Computes reflected flux via `hg_apparent_flux` per observation.
fn evaluate_hg_forward_model(x: &[f64], obs: &[FluxObs], c_hg: f64) -> Option<ForwardModelResult> {
    let h = x[0];
    let g = x[1];
    let f_sigma = x[2].exp();

    // Choose D so that pV = 1;
    // the product D^2 * pV = c_hg^2 * 10^(-2H/5) is determined by H alone.
    let diam = c_hg * 10_f64.powf(-h / 5.0);
    if !diam.is_finite() || diam <= 0.0 {
        return None;
    }

    let n = obs.len();
    let mut model_fluxes = Vec::with_capacity(n);
    for ob in obs {
        let flux =
            crate::hg_apparent_flux(g, diam, &ob.sun2obj, &ob.sun2obs, ob.band.wavelength, 1.0)
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
    c_hg: f64,
    emissivity: f64,
) -> f64 {
    let Some(fwd) = evaluate_forward_model(model, x, obs, c_hg, emissivity) else {
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
/// All models share H, G, and `f_sigma` priors.  Thermal models add
/// diameter, beaming, emissivity, and `R_IR` priors.  When a `pv` prior
/// is present, the derived albedo `pV(D, H)` is also penalized.
pub(super) fn log_prior(model: Model, x: &[f64], priors: &FluxPriors, c_hg: f64) -> f64 {
    let mut lp = 0.0;

    if model.is_hg() {
        lp += priors.h_mag.log_prob(x[0]);
        lp += priors.g_param.log_prob(x[1]);

        // f_sigma at index 2.
        lp += gaussian_log_prior(x[2], 1.5_f64.ln(), 0.3);
        lp += logistic_barrier(x[2], 0.5_f64.ln(), 5.0_f64.ln(), BARRIER_K);

        return lp;
    }

    // ----- Thermal (NEATM / FRM) -----
    // Indices: NEATM [ln_D, ln_beaming, H, G, ln_fs, ln_rir]
    //          FRM   [ln_D, H, G, ln_fs, ln_rir]
    let h_idx: usize = if model.is_neatm() { 2 } else { 1 };
    let g_idx = h_idx + 1;
    let f_idx = g_idx + 1;
    let r_idx = f_idx + 1;

    // Diameter.
    lp += priors.ln_diam.log_prob(x[0]);

    // Beaming (NEATM only).
    if model.is_neatm() {
        lp += priors.ln_beaming.log_prob(x[1]);
    }

    // H magnitude.
    lp += priors.h_mag.log_prob(x[h_idx]);

    // G parameter.
    lp += priors.g_param.log_prob(x[g_idx]);

    // f_sigma: Gaussian(ln 1.5, 0.3) + bounds [ln 0.5, ln 5.0].
    lp += gaussian_log_prior(x[f_idx], 1.5_f64.ln(), 0.3);
    lp += logistic_barrier(x[f_idx], 0.5_f64.ln(), 5.0_f64.ln(), BARRIER_K);

    // R_IR.
    lp += priors.ln_r_ir.log_prob(x[r_idx]);

    // pV prior (derived from D and H).
    let diam = x[0].exp();
    let h = x[h_idx];
    let pv = pv_from_diameter(diam, h, c_hg);
    if pv.is_finite() && pv > 0.0 {
        lp += priors.pv.log_prob(pv);
    } else {
        return f64::NEG_INFINITY;
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
    c_hg: f64,
    emissivity: f64,
    priors: &FluxPriors,
) -> f64 {
    let ll = log_likelihood(model, x, obs, c_hg, emissivity);
    if !ll.is_finite() {
        return f64::NEG_INFINITY;
    }
    let val = ll + log_prior(model, x, priors, c_hg);
    if val.is_finite() {
        val
    } else {
        f64::NEG_INFINITY
    }
}
