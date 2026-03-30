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

//! MCMC posterior sampling via NUTS for model fitting,
//! plus parallel batch fitting.

use super::types::{FluxObs, FluxPriors, Model};
use kete_core::errors::{Error, KeteResult};
use kete_stats::fitting::{NelderMeadResult, nelder_mead};
use nalgebra::{DMatrix, DVector};
use nuts_rs::rand::SeedableRng;
use nuts_rs::{
    Chain, CpuLogpFunc, CpuMath, CpuMathError, DiagGradNutsSettings, LogpError, Settings,
};
use rayon::prelude::*;
use std::collections::HashMap;

/// Build a negative log-posterior closure for Nelder-Mead minimization.
fn build_neg_log_posterior(
    model: Model,
    obs: &[FluxObs],
    c_hg: f64,
    emissivity: f64,
    priors: &FluxPriors,
) -> impl Fn(&[f64]) -> f64 {
    let obs = obs.to_vec();
    let priors = priors.clone();
    move |x: &[f64]| -> f64 {
        let lp = model.log_posterior(x, &obs, c_hg, emissivity, &priors);
        if lp.is_finite() { -lp } else { f64::MAX }
    }
}

/// Nelder-Mead seed result (kept minimal -- only fields MCMC needs).
struct NelderMeadSeed {
    /// Raw parameter vector at the best NM point.
    params: Vec<f64>,
}

/// Multi-start Nelder-Mead to find a good starting point for MCMC.
fn nelder_mead_seed(
    model: Model,
    obs: &[FluxObs],
    c_hg: f64,
    emissivity: f64,
    priors: &FluxPriors,
) -> KeteResult<NelderMeadSeed> {
    let objective = build_neg_log_posterior(model, obs, c_hg, emissivity, priors);

    // Derive seed values for H, G from priors.
    let h0 = priors.h_mag.center();
    let g0 = priors.g_param.center();

    let (starts, scale): (Vec<Vec<f64>>, Vec<f64>) = match model {
        Model::Neatm => {
            // [D, beaming, H, G, f_sigma, R_IR]
            let mid_d = priors.diameter.center();
            let mid_beaming = priors.beaming.center();
            let mid_r_ir = priors.r_ir.center();
            (
                vec![
                    vec![mid_d, 1.0, h0, g0, 1.5, mid_r_ir],
                    vec![mid_d, mid_beaming, h0, g0, 1.0, mid_r_ir],
                    vec![1.0, 1.5, h0, g0, 1.0, mid_r_ir],
                    vec![20.0, 0.8, h0, g0, 1.0, mid_r_ir],
                    vec![50.0, 1.0, h0, g0, 2.0, mid_r_ir],
                ],
                vec![5.0, 0.3, 1.0, 0.05, 0.2, 0.2],
            )
        }
        Model::Frm => {
            // [D, H, G, f_sigma, R_IR]
            let mid_d = priors.diameter.center();
            let mid_r_ir = priors.r_ir.center();
            (
                vec![
                    vec![mid_d, h0, g0, 1.5, mid_r_ir],
                    vec![mid_d, h0, g0, 1.0, mid_r_ir],
                    vec![1.0, h0, g0, 1.0, mid_r_ir],
                    vec![20.0, h0, g0, 1.0, mid_r_ir],
                    vec![50.0, h0, g0, 2.0, mid_r_ir],
                ],
                vec![5.0, 1.0, 0.05, 0.2, 0.2],
            )
        }
        Model::Hg => {
            // Two-phase approach for HG: first find H, G with f_sigma
            // fixed at 1.0, then free all three.  This prevents the
            // degenerate equilibrium where f_sigma inflates to absorb
            // a wrong H value, producing uniform residuals near 1.0.
            let mid_g = priors.g_param.center();
            let hg_starts_2d = vec![
                vec![h0, g0],
                vec![h0 - 2.0, g0],
                vec![h0 + 2.0, mid_g],
                vec![h0 - 4.0, g0],
                vec![h0 + 4.0, mid_g],
            ];
            let scale_2d = vec![1.0, 0.05];

            // Phase 1: optimize H, G with f_sigma = 1.0.
            let objective_2d = |x2: &[f64]| -> f64 {
                let x3 = [x2[0], x2[1], 1.0];
                objective(&x3)
            };

            let mut best_2d: Option<(Vec<f64>, f64)> = None;
            for start in &hg_starts_2d {
                if let Ok(NelderMeadResult { point, value, .. }) =
                    nelder_mead(objective_2d, start, &scale_2d, 1e-7, 2_000)
                    && best_2d.as_ref().is_none_or(|b| value < b.1)
                {
                    best_2d = Some((point, value));
                }
            }

            // Phase 2: free f_sigma, starting from the phase-1 best.
            let hg_starts_3d: Vec<Vec<f64>> = if let Some((ref best, _)) = best_2d {
                vec![
                    vec![best[0], best[1], 1.0],
                    vec![best[0], best[1], 0.5],
                    vec![best[0], best[1], 2.0],
                    vec![h0, g0, 1.0],
                ]
            } else {
                vec![
                    vec![h0, g0, 1.0],
                    vec![h0 - 2.0, g0, 1.0],
                    vec![h0 + 2.0, mid_g, 1.0],
                ]
            };

            (hg_starts_3d, vec![1.0, 0.05, 0.2])
        }
    };

    let mut best: Option<(Vec<f64>, f64)> = None;
    for start in &starts {
        if let Ok(NelderMeadResult { point, value, .. }) =
            nelder_mead(&objective, start, &scale, 1e-7, 2_000)
            && best.as_ref().is_none_or(|b| value < b.1)
        {
            best = Some((point, value));
        }
    }

    let (params, value) = best.ok_or_else(|| {
        Error::Convergence("Nelder-Mead failed to find a feasible starting point".into())
    })?;
    if value >= f64::MAX {
        return Err(Error::Convergence(
            "Nelder-Mead converged to an infeasible point (log-posterior = -inf)".into(),
        ));
    }
    Ok(NelderMeadSeed { params })
}

/// Step size for finite-difference gradients.
const FINITE_DIFF_STEP: f64 = 1e-5;

/// Fallback gradient value when finite-differences produce non-finite or zero
/// results. Must be nonzero because nuts-rs rejects zero gradient components
/// during initialization (`array_all_finite_and_nonzero`).
const GRAD_FALLBACK: f64 = 1e-10;

/// Result of a full MCMC fit.
#[derive(Debug, Clone)]
pub struct FitResult {
    /// Which model was fit.
    /// Determines the draw column layout:
    /// NEATM: `[D, pV, beaming, H, G, R_IR, f_sigma]`,
    /// FRM:   `[D, pV, H, G, R_IR, f_sigma]`,
    /// HG:    `[H, G, f_sigma]`.
    pub model: Model,

    /// Raw posterior draws.  Column layout depends on `model`.
    pub draws: Vec<Vec<f64>>,

    /// Whether each draw was divergent.
    pub divergent: Vec<bool>,

    /// Total number of divergent transitions.
    pub n_divergent: usize,

    /// Reduced chi-squared at the MAP point using **inflated** uncertainties:
    /// `(1 / dof) * sum ((obs - model) / (f_sigma * sigma_i))^2` where
    /// `dof = nobs - nparams`.  A value near 1.0 indicates the model
    /// explains the data at the level of measurement noise.
    pub reduced_chi2: f64,

    /// Number of non-upper-limit observations.
    pub nobs: usize,

    /// Model fluxes at the MAP point, one per observation.
    pub best_fit_fluxes: Vec<f64>,

    /// Standardized residuals at the MAP point: `(obs - model) / (f_sigma * sigma_i)`.
    pub best_fit_residuals: Vec<f64>,

    /// Reflected-light fraction at the MAP point, one per observation.
    pub best_fit_reflected_frac: Vec<f64>,
}

impl FitResult {
    /// Column names for the posterior draw vectors.
    #[must_use]
    pub fn column_names(&self) -> &'static [&'static str] {
        self.model.draw_column_names()
    }
}

#[derive(Debug)]
struct RecoverableError;

impl LogpError for RecoverableError {
    fn is_recoverable(&self) -> bool {
        true
    }
}

impl std::fmt::Display for RecoverableError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("infeasible point")
    }
}

impl std::error::Error for RecoverableError {}

/// Log-posterior for MCMC.
///
/// The sampler operates in whitened "xi-space"; physical parameters are
/// recovered via `x = seed + L * xi`.
struct Posterior {
    obs: Vec<FluxObs>,
    c_hg: f64,
    emissivity: f64,
    priors: FluxPriors,
    /// Center of the whitening transform (best NM point).
    seed_vec: DVector<f64>,
    /// Cholesky-like factor: `x = seed + L * xi`.
    whiten_l: DMatrix<f64>,
    model: Model,
}

impl Posterior {
    /// Convert whitened xi -> physical parameter vector x.
    fn xi_to_x(&self, xi: &[f64]) -> Vec<f64> {
        let xi = DVector::from_column_slice(xi);
        let x = &self.seed_vec + &self.whiten_l * &xi;
        x.as_slice().to_vec()
    }

    /// Evaluate the log-posterior at physical parameter vector `x`.
    fn eval_logp(&self, x: &[f64]) -> f64 {
        self.model
            .log_posterior(x, &self.obs, self.c_hg, self.emissivity, &self.priors)
    }
}

impl nuts_rs::HasDims for Posterior {
    fn dim_sizes(&self) -> HashMap<String, u64> {
        let mut m = HashMap::new();
        let _ = m.insert("dim".to_string(), self.seed_vec.len() as u64);
        m
    }
}

impl CpuLogpFunc for Posterior {
    type LogpError = RecoverableError;
    type FlowParameters = ();
    type ExpandedVector = Vec<f64>;

    fn dim(&self) -> usize {
        self.seed_vec.len()
    }

    fn logp(&mut self, position: &[f64], gradient: &mut [f64]) -> Result<f64, Self::LogpError> {
        let x = self.xi_to_x(position);
        let lp = self.eval_logp(&x);
        if !lp.is_finite() {
            return Err(RecoverableError);
        }

        // Finite-difference gradient in xi-space.
        // Each dimension is independent -- evaluate all perturbed points in parallel.
        let d = self.dim();
        let this = &*self;
        let pos = position.to_vec();
        let grad_vals: Vec<f64> = (0..d)
            .into_par_iter()
            .map(|j| {
                let mut buf = pos.clone();
                buf[j] += FINITE_DIFF_STEP;
                let lp_plus = this.eval_logp(&this.xi_to_x(&buf));
                buf[j] -= 2.0 * FINITE_DIFF_STEP;
                let lp_minus = this.eval_logp(&this.xi_to_x(&buf));
                let g = if lp_plus.is_finite() && lp_minus.is_finite() {
                    (lp_plus - lp_minus) / (2.0 * FINITE_DIFF_STEP)
                } else if lp_plus.is_finite() {
                    (lp_plus - lp) / FINITE_DIFF_STEP
                } else if lp_minus.is_finite() {
                    (lp - lp_minus) / FINITE_DIFF_STEP
                } else {
                    GRAD_FALLBACK
                };
                if g.is_finite() && g != 0.0 {
                    g
                } else {
                    GRAD_FALLBACK
                }
            })
            .collect();
        gradient.copy_from_slice(&grad_vals);
        Ok(lp)
    }

    fn expand_vector<R: rand::Rng + ?Sized>(
        &mut self,
        _rng: &mut R,
        array: &[f64],
    ) -> Result<Self::ExpandedVector, CpuMathError> {
        Ok(array.to_vec())
    }
}

/// Run one NUTS chain and return `(draws, divergent_flags)`.
fn run_chain(
    posterior: Posterior,
    num_tune: u64,
    num_draws: u64,
    maxdepth: u64,
    target_accept: f64,
    chain_seed: u64,
) -> Result<(Vec<Vec<f64>>, Vec<bool>), String> {
    use rand::distr::Uniform;
    use rand::prelude::Distribution;

    let d = posterior.dim();

    let mut settings = DiagGradNutsSettings {
        num_tune,
        num_draws,
        maxdepth,
        seed: chain_seed,
        num_chains: 1,
        ..DiagGradNutsSettings::default()
    };
    settings.adapt_options.step_size_settings.target_accept = target_accept;

    let math = CpuMath::new(posterior);
    let mut rng = rand::rngs::SmallRng::seed_from_u64(chain_seed);
    let mut sampler = settings.new_chain(chain_seed, math, &mut rng);

    // Try initializing at the origin (MAP point), then small perturbations
    // if the gradient is problematic at the exact MAP.
    let mut init: Vec<f64> = vec![0.0; d];
    let mut initialized = false;
    for attempt in 0..5 {
        if sampler.set_position(&init).is_ok() {
            initialized = true;
            break;
        }
        // Perturb away from the exact MAP to avoid boundary gradient issues.
        let scale = 0.01 * f64::from(1_i32 << attempt);
        let uniform = Uniform::new(-scale, scale).expect("valid range");
        init = (0..d).map(|_| uniform.sample(&mut rng)).collect();
    }
    if !initialized {
        return Err("NUTS init failed after retries: bad gradient at MAP point".into());
    }

    let total = num_tune + num_draws;
    let mut draws = Vec::with_capacity(num_draws as usize);
    let mut divergent = Vec::with_capacity(num_draws as usize);

    for _ in 0..total {
        let Ok((position, progress)) = sampler.draw() else {
            break;
        };
        if progress.tuning {
            continue;
        }
        draws.push(position.as_ref().to_vec());
        divergent.push(progress.diverging);
    }

    Ok((draws, divergent))
}

/// Minimum allowed whitening scale (prevents near-zero step sizes).
const MIN_WHITEN_SCALE: f64 = 0.01;

/// Estimate diagonal whitening scales from the finite-difference Hessian of
/// the objective function at the MAP point.
///
/// Returns a vector of scales `s_j = 1 / sqrt(|H_jj|)`, floored at
/// `MIN_WHITEN_SCALE`.  Falls back to `fallback` if any Hessian diagonal
/// element is non-finite.
fn hessian_whitening_scales(
    objective: &dyn Fn(&[f64]) -> f64,
    map_point: &[f64],
    fallback: &[f64],
) -> Vec<f64> {
    let d = map_point.len();
    let f0 = objective(map_point);
    if !f0.is_finite() {
        return fallback.to_vec();
    }

    // step for Hessian FD (larger than gradient FD_EPS)
    let eps = 1e-4;
    let mut scales = Vec::with_capacity(d);
    let mut buf = map_point.to_vec();

    for j in 0..d {
        let orig = buf[j];
        buf[j] = orig + eps;
        let fp = objective(&buf);
        buf[j] = orig - eps;
        let fm = objective(&buf);
        buf[j] = orig;

        let h_jj = (fp - 2.0 * f0 + fm) / (eps * eps);
        if h_jj.is_finite() && h_jj.abs() > 1e-30 {
            scales.push((1.0 / h_jj.abs().sqrt()).max(MIN_WHITEN_SCALE));
        } else {
            // Axis is flat or numerically unstable -- use fallback.
            return fallback.to_vec();
        }
    }

    scales
}

/// Fit a model using NUTS MCMC and return posterior summaries.
///
/// Internally runs multi-start Nelder-Mead to find the MAP, builds a
/// whitened posterior, and runs `num_chains` independent NUTS chains.
///
/// `c_hg` is the H-D-pV conversion factor. `emissivity` is a fixed
/// thermal emissivity (not fitted).
///
/// # Errors
/// - If Nelder-Mead fails to find a feasible starting point.
/// - If all MCMC chains fail to produce valid draws.
pub fn fit_mcmc(
    model: Model,
    obs: &[FluxObs],
    c_hg: f64,
    emissivity: f64,
    priors: &FluxPriors,
    num_chains: usize,
    num_tune: usize,
    num_draws: usize,
) -> KeteResult<FitResult> {
    // 1. Multi-start NM seed.
    let nm = nelder_mead_seed(model, obs, c_hg, emissivity, priors)?;
    let seed = nm.params;

    let seed_vec = DVector::from_column_slice(&seed);

    // MAP diagnostics at the NM seed.
    let (reduced_chi2, nobs, best_fit_fluxes, best_fit_residuals, best_fit_reflected_frac) = {
        let params = model.unpack(&seed, emissivity, c_hg);
        let fwd = model.evaluate_forward_model(&params, obs);
        let mut chi2 = 0.0;
        let mut n = 0_usize;
        let mut residuals = Vec::with_capacity(obs.len());
        for (i, ob) in obs.iter().enumerate() {
            let sigma = params.f_sigma * ob.sigma;
            let r = if sigma > 0.0 {
                (ob.flux - fwd.model_fluxes[i]) / sigma
            } else {
                0.0
            };
            residuals.push(r);
            if !ob.is_upper_limit {
                chi2 += r * r;
                n += 1;
            }
        }
        let dof = n.saturating_sub(model.dim());
        let reduced = if dof > 0 { chi2 / dof as f64 } else { f64::NAN };
        (reduced, n, fwd.model_fluxes, residuals, fwd.reflected_frac)
    };

    // Diagonal whitening from FD Hessian of the objective.
    let d = model.dim();
    let fallback: Vec<f64> = std::iter::once(0.3)
        .chain(std::iter::repeat_n(0.15, d - 1))
        .collect();
    let objective = build_neg_log_posterior(model, obs, c_hg, emissivity, priors);
    let scales = hessian_whitening_scales(&objective, &seed, &fallback);
    let whiten_l = DMatrix::from_diagonal(&DVector::from_column_slice(&scales));

    // 2. Run chains in parallel.
    let chain_outcomes: Vec<Result<(Vec<Vec<f64>>, Vec<bool>), String>> = (0..num_chains)
        .into_par_iter()
        .map(|chain_idx| {
            let posterior = Posterior {
                obs: obs.to_vec(),
                c_hg,
                emissivity,
                priors: priors.clone(),
                seed_vec: seed_vec.clone(),
                whiten_l: whiten_l.clone(),
                model,
            };
            let (xi_draws, div) = run_chain(
                posterior,
                num_tune as u64,
                num_draws as u64,
                10,
                0.8,
                chain_idx as u64,
            )?;
            let (phys, filtered_div): (Vec<Vec<f64>>, Vec<bool>) = xi_draws
                .iter()
                .zip(div)
                .map(|(xi, d)| {
                    let xi_dv = DVector::from_column_slice(xi);
                    let x = &seed_vec + &whiten_l * &xi_dv;
                    let p = model.unpack(x.as_slice(), emissivity, c_hg);
                    (p.to_draw_row(model), d)
                })
                .unzip();
            Ok((phys, filtered_div))
        })
        .collect();

    let mut all_draws: Vec<Vec<f64>> = Vec::new();
    let mut all_divergent: Vec<bool> = Vec::new();
    let mut chain_errors: Vec<String> = Vec::new();
    for outcome in chain_outcomes {
        match outcome {
            Ok((draws, div)) => {
                all_draws.extend(draws);
                all_divergent.extend(div);
            }
            Err(e) => chain_errors.push(e),
        }
    }

    if all_draws.is_empty() {
        let detail = if chain_errors.is_empty() {
            "all chains produced zero draws".into()
        } else {
            format!(
                "{}/{} chains failed: {}",
                chain_errors.len(),
                num_chains,
                chain_errors[0]
            )
        };
        return Err(Error::Convergence(format!(
            "MCMC failed to produce valid draws ({detail})"
        )));
    }

    let n_divergent = all_divergent.iter().filter(|&&d| d).count();

    Ok(FitResult {
        model,
        draws: all_draws,
        divergent: all_divergent,
        n_divergent,
        reduced_chi2,
        nobs,
        best_fit_fluxes,
        best_fit_residuals,
        best_fit_reflected_frac,
    })
}

/// Inputs for a single MCMC fit, bundled for batch processing.
#[derive(Debug, Clone)]
pub struct FitTask {
    /// Which model to fit.
    pub model: Model,
    /// Observations for this object.
    pub obs: Vec<FluxObs>,
    /// Relationship constant for D-H-pV conversion (km).
    pub c_hg: f64,
    /// Fixed thermal emissivity (not fitted).
    pub emissivity: f64,
    /// Prior/bound configuration (includes H, G priors).
    pub priors: FluxPriors,
    /// Number of MCMC chains.
    pub num_chains: usize,
    /// Number of tuning (warmup) draws per chain.
    pub num_tune: usize,
    /// Number of posterior draws per chain.
    pub num_draws: usize,
}

/// Fit many objects in parallel using rayon.
///
/// Each task is independent and dispatched to a rayon thread.  Returns one
/// `KeteResult<FitResult>` per task, in the same order as the input.
#[must_use]
pub fn fit_batch(tasks: &[FitTask]) -> Vec<KeteResult<FitResult>> {
    tasks
        .par_iter()
        .map(|t| {
            fit_mcmc(
                t.model,
                &t.obs,
                t.c_hg,
                t.emissivity,
                &t.priors,
                t.num_chains,
                t.num_tune,
                t.num_draws,
            )
        })
        .collect()
}
