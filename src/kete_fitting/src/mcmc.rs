//! NUTS MCMC sampling for non-Gaussian orbit posteriors.
//!
//! Provides [`nuts_sample`], which runs one NUTS chain per orbital mode
//! (seed) and pools the draws into a single [`OrbitSamples`] collection.
//! Chains are run in parallel via Rayon.
//!
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

use crate::diff_correction::{OrbitFit, StmObs, stm_sweep};
use crate::obs::Observation;
use kete_core::frames::Equatorial;
use kete_core::prelude::{Error, KeteResult, State};
use kete_core::propagation::NonGravModel;
use nalgebra::{DMatrix, DVector};
use nuts_rs::rand::SeedableRng;
use nuts_rs::{
    Chain, CpuLogpFunc, CpuMath, CpuMathError, DiagGradNutsSettings, LogpError, Settings,
};
use rayon::prelude::*;
use std::collections::HashMap;
use std::fmt;

/// Posterior orbit samples from NUTS MCMC.
#[derive(Debug, Clone)]
pub struct OrbitSamples {
    /// Designator of the object being fitted.
    pub desig: String,
    /// Common reference epoch (JD, TDB).
    pub epoch: f64,
    /// Draws: `[total_draws][6 + Np]`.
    ///
    /// Each inner vector is `[x, y, z, vx, vy, vz, ng_params...]` in the
    /// Equatorial frame at `epoch`.
    pub draws: Vec<Vec<f64>>,
    /// Seed index (0-based) that generated each draw.
    pub chain_id: Vec<usize>,
    /// True if the draw was a divergent transition.
    pub divergent: Vec<bool>,
    /// Log-posterior at each draw.
    pub logp: Vec<f64>,
}

/// Error returned by [`OrbitalPosterior::logp`] when propagation fails.
#[derive(Debug)]
struct PropagationError {
    msg: String,
    recoverable: bool,
}

impl fmt::Display for PropagationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.msg)
    }
}

impl std::error::Error for PropagationError {}

impl LogpError for PropagationError {
    fn is_recoverable(&self) -> bool {
        self.recoverable
    }
}

// OrbitalPosterior -- implements CpuLogpFunc

/// Log-posterior density over orbital states, parameterized in a whitened
/// coordinate system centered on the MAP.
struct OrbitalPosterior {
    /// MAP state at the reference epoch.
    map_state: State<Equatorial>,
    /// Lower Cholesky factor of the (regularized) MAP covariance, D x D.
    chol_l: DMatrix<f64>,
    /// MAP state vector (position ++ velocity ++ non-grav params), D-vector.
    map_vec: DVector<f64>,
    /// Observations (time-sorted).
    obs: Vec<Observation>,
    /// Inclusion mask (all true -- we include every obs).
    included: Vec<bool>,
    /// Whether to include extended (asteroid) perturbers.
    include_asteroids: bool,
    /// Non-gravitational model (if any).
    non_grav: Option<NonGravModel>,
    /// Student-t degrees of freedom (`f64::INFINITY` = Gaussian).
    student_nu: f64,
    /// Parameter dimension: 6 + Np.
    dim: usize,
}

impl OrbitalPosterior {
    /// Transform whitened coordinates back to physical state.
    fn xi_to_state(&self, xi: &[f64]) -> (State<Equatorial>, Option<NonGravModel>) {
        let xi_vec = DVector::from_column_slice(xi);
        let x = &self.map_vec + &self.chol_l * &xi_vec;

        let mut state = self.map_state.clone();
        state.pos = [x[0], x[1], x[2]].into();
        state.vel = [x[3], x[4], x[5]].into();

        let ng = self.non_grav.as_ref().map(|model| {
            let mut m = model.clone();
            let np = m.n_free_params();
            let mut params = vec![0.0; np];
            for k in 0..np {
                params[k] = x[6 + k];
            }
            m.set_free_params(&params);
            m
        });

        (state, ng)
    }

    /// Compute logp and gradient from an STM sweep result.
    fn logp_from_sweep(&self, sweep: &[StmObs], grad_xi: &mut [f64]) -> f64 {
        let d = self.dim;
        let mut grad_x = DVector::<f64>::zeros(d);
        let mut logp = 0.0;
        let nu = self.student_nu;
        let gaussian = nu.is_infinite();

        for entry in sweep {
            let m = entry.residual.len();
            // H_epoch = H_local * Phi_cum  (m x D)
            let h_epoch = &entry.h_local * &entry.phi_cum;

            for k in 0..m {
                let r = entry.residual[k];
                // weights = 1/sigma^2
                let sigma2 = 1.0 / entry.weights[k];

                if gaussian {
                    // Gaussian: logp += -r^2 / (2 sigma^2)
                    logp += -0.5 * r * r / sigma2;
                    // d(logp)/d(x) needs the chain rule: d(r)/d(x) = -H,
                    // so d(logp)/d(x) = -d(logp)/d(r) * H = (r / sigma^2) * H
                    let dl_dx_factor = r / sigma2;
                    for j in 0..d {
                        grad_x[j] += h_epoch[(k, j)] * dl_dx_factor;
                    }
                } else {
                    // Student-t: logp += -(nu+1)/2 * ln(1 + r^2/(nu*sigma^2))
                    let s = r * r / (nu * sigma2);
                    logp += -0.5 * (nu + 1.0) * (1.0 + s).ln();
                    // d(logp)/d(r) = -(nu+1)*r / (nu*sigma^2 + r^2)
                    // d(logp)/d(x) = -d(logp)/d(r) * H = (nu+1)*r / (nu*sigma^2 + r^2) * H
                    let dl_dx_factor = (nu + 1.0) * r / (nu * sigma2 + r * r);
                    for j in 0..d {
                        grad_x[j] += h_epoch[(k, j)] * dl_dx_factor;
                    }
                }
            }
        }

        // Transform gradient: grad_xi = L^T * grad_x
        let g = self.chol_l.transpose() * &grad_x;
        for j in 0..d {
            grad_xi[j] = g[j];
        }

        logp
    }
}

impl nuts_rs::HasDims for OrbitalPosterior {
    fn dim_sizes(&self) -> HashMap<String, u64> {
        let mut m = HashMap::new();
        let _ = m.insert("dim".to_string(), self.dim as u64);
        m
    }
}

impl CpuLogpFunc for OrbitalPosterior {
    type LogpError = PropagationError;
    type FlowParameters = ();
    type ExpandedVector = Vec<f64>;

    fn dim(&self) -> usize {
        self.dim
    }

    fn logp(&mut self, position: &[f64], gradient: &mut [f64]) -> Result<f64, Self::LogpError> {
        let (trial_state, trial_ng) = self.xi_to_state(position);

        let sweep = stm_sweep(
            &trial_state,
            &self.obs,
            &self.included,
            self.include_asteroids,
            trial_ng.as_ref(),
        )
        .map_err(|e| PropagationError {
            msg: format!("STM sweep failed: {e}"),
            recoverable: true,
        })?;

        let lp = self.logp_from_sweep(&sweep, gradient);
        Ok(lp)
    }

    fn expand_vector<R: rand::Rng + ?Sized>(
        &mut self,
        _rng: &mut R,
        array: &[f64],
    ) -> Result<Self::ExpandedVector, CpuMathError> {
        // Transform from whitened xi back to physical coordinates for storage.
        let xi_vec = DVector::from_column_slice(array);
        let x = &self.map_vec + &self.chol_l * &xi_vec;
        Ok(x.as_slice().to_vec())
    }
}

// nuts_sample -- public entry point

/// Run NUTS MCMC sampling over orbital posteriors.
///
/// One chain per seed is run in parallel.  All seeds must share the same
/// reference epoch.
///
/// The non-gravitational model (if any) is taken from each seed's
/// `OrbitFit::non_grav`, which already contains the fitted parameter values
/// that the covariance was linearized around.
///
/// Chains are automatically spread across available CPU cores.  When there
/// are fewer seeds than cores, each seed spawns multiple sub-chains (each
/// with its own RNG seed and tuning phase).  The `chain_id` in the returned
/// [`OrbitSamples`] identifies the seed (orbital mode), not the sub-chain.
///
/// `num_draws` is the **total** number of posterior draws returned across
/// all seeds.  Each seed receives `num_draws / n_seeds` draws (remainder
/// goes to the first seeds), which are then split across sub-chains.
///
/// # Arguments
/// * `seeds` -- Converged `OrbitFit` results, one per orbital mode.
/// * `obs` -- Observations (any order; sorted internally).
/// * `include_asteroids` -- Whether to include extended (asteroid) perturbers.
/// * `num_draws` -- Total posterior draws across all seeds.
/// * `num_tune` -- Tuning (warmup) steps per sub-chain.  Because sampling
///   uses whitened coordinates, 50 is typically sufficient.
/// * `student_nu` -- Student-t degrees of freedom (`f64::INFINITY` for Gaussian).
///
/// # Errors
/// Returns an error if `seeds` is empty or epochs differ.
pub fn nuts_sample(
    seeds: &[OrbitFit],
    obs: &[Observation],
    include_asteroids: bool,
    num_draws: usize,
    num_tune: usize,
    student_nu: f64,
) -> KeteResult<OrbitSamples> {
    if seeds.is_empty() {
        return Err(Error::ValueError("No seeds provided".into()));
    }

    // All seeds must share the same reference epoch.
    let epoch = seeds[0].uncertain_state.state.epoch.jd;
    for (i, seed) in seeds.iter().enumerate().skip(1) {
        if (seed.uncertain_state.state.epoch.jd - epoch).abs() > 1e-12 {
            return Err(Error::ValueError(format!(
                "Seed {i} epoch ({}) differs from seed 0 epoch ({epoch})",
                seed.uncertain_state.state.epoch.jd
            )));
        }
    }

    // Sort observations once.
    let mut sorted_obs = obs.to_vec();
    sorted_obs.sort_by(|a, b| {
        a.epoch()
            .jd
            .partial_cmp(&b.epoch().jd)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Distribute num_draws across seeds, then sub-chains across cores.
    let n_cores = std::thread::available_parallelism()
        .map(std::num::NonZero::get)
        .unwrap_or(1);
    let n_seeds = seeds.len();
    let chains_per_seed = (n_cores / n_seeds).max(1);

    // Divide total draws among seeds (remainder to the first seeds).
    let draws_base_per_seed = num_draws / n_seeds;
    let draws_extra_seeds = num_draws % n_seeds;

    // Build a flat task list: (seed_index, draws_for_chain, rng_seed).
    let mut tasks: Vec<(usize, usize, u64)> = Vec::new();
    for seed_idx in 0..n_seeds {
        let seed_draws = draws_base_per_seed + usize::from(seed_idx < draws_extra_seeds);
        let base = seed_draws / chains_per_seed;
        let extra = seed_draws % chains_per_seed;
        for sub in 0..chains_per_seed {
            let draws = base + usize::from(sub < extra);
            if draws == 0 {
                continue;
            }
            let rng_seed = (seed_idx * chains_per_seed + sub) as u64;
            tasks.push((seed_idx, draws, rng_seed));
        }
    }

    // Run all chains in parallel.
    let chain_results: Vec<(usize, KeteResult<(Vec<Vec<f64>>, Vec<bool>, Vec<f64>)>)> = tasks
        .par_iter()
        .map(|&(seed_idx, draws, rng_seed)| {
            let result = run_single_chain(
                &seeds[seed_idx],
                &sorted_obs,
                include_asteroids,
                draws,
                num_tune,
                student_nu,
                rng_seed,
            );
            (seed_idx, result)
        })
        .collect();

    // Collect results.  chain_id reflects the seed index (orbital mode).
    let mut all_draws = Vec::new();
    let mut all_chain_id = Vec::new();
    let mut all_divergent = Vec::new();
    let mut all_logp = Vec::new();

    for (seed_idx, result) in chain_results {
        let (draws, divergent, logp_vals) = result?;
        let n = draws.len();
        all_draws.extend(draws);
        all_chain_id.extend(std::iter::repeat_n(seed_idx, n));
        all_divergent.extend(divergent);
        all_logp.extend(logp_vals);
    }

    Ok(OrbitSamples {
        desig: seeds[0].uncertain_state.state.desig.to_string(),
        epoch,
        draws: all_draws,
        chain_id: all_chain_id,
        divergent: all_divergent,
        logp: all_logp,
    })
}

/// Run a single NUTS chain for one seed.
fn run_single_chain(
    seed: &OrbitFit,
    sorted_obs: &[Observation],
    include_asteroids: bool,
    num_draws: usize,
    num_tune: usize,
    student_nu: f64,
    chain_idx: u64,
) -> KeteResult<(Vec<Vec<f64>>, Vec<bool>, Vec<f64>)> {
    // Use the non-grav model from the seed (if any).  This is the model
    // that differential correction fitted and whose parameter values the
    // covariance was linearized around.
    let non_grav = seed.uncertain_state.non_grav.as_ref();
    let np = non_grav.map_or(0, NonGravModel::n_free_params);
    let d = 6 + np;

    // Build the MAP vector and Cholesky factor locally so we can use them
    // for the xi -> x transform when storing draws.
    let pos: [f64; 3] = seed.uncertain_state.state.pos.into();
    let vel: [f64; 3] = seed.uncertain_state.state.vel.into();
    let mut map_vec = DVector::<f64>::zeros(d);
    for i in 0..3 {
        map_vec[i] = pos[i];
        map_vec[3 + i] = vel[i];
    }
    if let Some(ng) = non_grav {
        let params = ng.get_free_params();
        for k in 0..np {
            map_vec[6 + k] = params[k];
        }
    }
    let chol_l = regularized_cholesky(&seed.uncertain_state.cov_matrix, np)?;

    let posterior = OrbitalPosterior {
        map_state: seed.uncertain_state.state.clone(),
        chol_l: chol_l.clone(),
        map_vec: map_vec.clone(),
        obs: sorted_obs.to_vec(),
        included: vec![true; sorted_obs.len()],
        include_asteroids,
        non_grav: non_grav.cloned(),
        student_nu,
        dim: d,
    };

    // Configure NUTS.
    let settings = DiagGradNutsSettings {
        num_tune: num_tune as u64,
        num_draws: num_draws as u64,
        maxdepth: 6,
        seed: chain_idx,
        num_chains: 1,
        ..DiagGradNutsSettings::default()
    };

    let math = CpuMath::new(posterior);

    let mut rng = rand::rngs::SmallRng::seed_from_u64(chain_idx);
    let mut sampler = settings.new_chain(chain_idx, math, &mut rng);

    // Initialize at xi = 0 (the MAP).
    let init = vec![0.0_f64; d];
    sampler
        .set_position(&init)
        .map_err(|e| Error::ValueError(format!("NUTS init failed: {e}")))?;

    let total_draws = num_tune as u64 + num_draws as u64;
    let mut draws = Vec::with_capacity(num_draws);
    let mut divergent = Vec::with_capacity(num_draws);
    let mut logp_vals = Vec::with_capacity(num_draws);

    for _ in 0..total_draws {
        let (position, progress) = sampler
            .draw()
            .map_err(|e| Error::ValueError(format!("NUTS draw failed: {e}")))?;

        // Skip tuning draws.
        if progress.tuning {
            continue;
        }

        // position is in xi-space; transform to physical coords for storage.
        let xi = position.as_ref();
        let xi_vec = DVector::from_column_slice(xi);
        let x = &map_vec + &chol_l * &xi_vec;
        draws.push(x.as_slice().to_vec());
        divergent.push(progress.diverging);
        logp_vals.push(f64::NAN);
    }

    Ok((draws, divergent, logp_vals))
}

// Cholesky regularization

/// Compute the lower Cholesky factor of a regularized covariance matrix.
///
/// Eigenvalues below a relative threshold (1e-14 * the largest eigenvalue)
/// are raised to that threshold.  This bounds the condition number at ~1e7,
/// keeping the whitened coordinate system well-conditioned for NUTS without
/// distorting well-determined directions.
///
/// For fully degenerate matrices (e.g. `from_state` with zero covariance)
/// a tiny absolute floor of 1e-30 prevents division by zero.
fn regularized_cholesky(cov: &DMatrix<f64>, np: usize) -> KeteResult<DMatrix<f64>> {
    let d = cov.nrows();
    assert_eq!(
        d,
        6 + np,
        "covariance dimension must equal 6 + n_nongrav_params"
    );

    // Eigendecompose.
    let eigen = cov.clone().symmetric_eigen();

    // Relative floor: bound the condition number so the whitened space
    // is well-scaled.  Only truly degenerate (near-zero) eigenvalues are
    // raised; well-determined directions keep their actual variance.
    let max_eig = eigen.eigenvalues.iter().copied().fold(0.0_f64, f64::max);
    let min_eigenvalue = (max_eig * 1e-14).max(1e-30);

    let mut eigenvalues = eigen.eigenvalues.clone();
    for i in 0..d {
        if eigenvalues[i] < min_eigenvalue {
            eigenvalues[i] = min_eigenvalue;
        }
    }

    // Reconstruct: C_reg = V * diag(lambda_floored) * V^T
    let v = &eigen.eigenvectors;
    let lambda_diag = DMatrix::from_diagonal(&eigenvalues);
    let c_reg = v * lambda_diag * v.transpose();

    // Cholesky factor.
    let chol = c_reg.clone().cholesky().ok_or_else(|| {
        Error::ValueError("Cholesky factorization failed on regularized covariance".into())
    })?;

    Ok(chol.l())
}
