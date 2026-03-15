//! MCMC orbit uncertainty estimation from observations.
//!
//! Provides [`fit_orbit_mcmc`], which estimates the range of orbits
//! consistent with a set of observations by running parallel MCMC chains.
//! Each converged differential-correction fit gets its own chain, and the
//! results are pooled into a single [`OrbitSamples`] collection.
//!
//! Sampling uses whitened Cartesian coordinates centered on the fit state,
//! with the DC covariance as the whitening factor.
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

use crate::obs::AstrometricObservation;
use crate::orbit_fitting::{StmObs, accumulate_normal_equations, stm_sweep};
use kete_core::constants::GMS;
use kete_core::frames::Equatorial;
use kete_core::prelude::{Error, KeteResult, State};
use kete_core::propagation::{NonGravModel, propagate_two_body};
use nalgebra::{DMatrix, DVector};
use nuts_rs::rand::SeedableRng;
use nuts_rs::{
    Chain, CpuLogpFunc, CpuMath, CpuMathError, DiagGradNutsSettings, LogpError, Settings,
};
use rayon::prelude::*;
use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

/// Student-t degrees of freedom for the MCMC likelihood.
///
/// `nu = 5` is heavy-tailed enough that 3-5 sigma outlier observations are
/// automatically down-weighted (their contribution to the log-likelihood
/// plateaus instead of growing quadratically), yet light-tailed enough that
/// NUTS still gets a strong gradient signal for efficient adaptation.
///
/// * `nu = 3-4`: maximum outlier robustness, but the likelihood surface is
///   very flat and NUTS mixes slowly with more divergences.
/// * `nu = 5`: standard "robust default" in Bayesian regression (the
///   recommendation from the Stan development team and `brms`).
/// * `nu >= 10`: barely distinguishable from Gaussian -- single outliers
///   can still dominate the posterior.
/// * `nu = infinity`: pure Gaussian likelihood.
const STUDENT_NU: f64 = 5.0;

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
    /// Fit index (0-based) that generated each draw.
    pub chain_id: Vec<usize>,
    /// True if the draw was a divergent transition.
    pub divergent: Vec<bool>,
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

/// Minimum barycentric distance (AU) before penalty ramps up.
const PRIOR_R_MIN: f64 = 0.01;
/// Maximum barycentric distance (AU) before penalty ramps up.
const PRIOR_R_MAX: f64 = 1000.0;
/// Steepness of the logistic barrier.
const PRIOR_K: f64 = 100.0;

/// Smooth physical prior: penalizes unphysical orbits with differentiable
/// logistic barriers so the gradient is always well-defined.
///
/// Penalties:
///   - barycentric distance below `PRIOR_R_MIN` or above `PRIOR_R_MAX`
///   - orbital eccentricity `e >= 1` (unbound / hyperbolic orbits)
///
/// The eccentricity barrier uses `e^2` (via the eccentricity vector) rather
/// than `e` to avoid a `1/e` singularity at circular orbits.  Because
/// `e^2 < 1 <=> e < 1`, the barrier `log sigma(K*(1 - e^2))` enforces bound
/// orbits while remaining smooth everywhere.
///
/// Returns `(log_prior, grad_prior)` where `grad_prior` has 6 elements
/// `[d/dx, d/dy, d/dz, d/dvx, d/dvy, d/dvz]`.
fn physical_prior(pos: &[f64; 3], vel: &[f64; 3]) -> (f64, [f64; 6]) {
    let mut grad = [0.0; 6];
    let mut lp = 0.0;

    let px = pos[0];
    let py = pos[1];
    let pz = pos[2];
    let vx = vel[0];
    let vy = vel[1];
    let vz = vel[2];

    let r2 = px * px + py * py + pz * pz;
    let r = r2.sqrt();
    let v2 = vx * vx + vy * vy + vz * vz;

    if r < 1e-15 {
        return (-1e10, [0.0; 6]);
    }

    // r_min barrier: log(sigmoid(K * (r - r_min)))
    let z_min = PRIOR_K * (r - PRIOR_R_MIN);
    let (lp_min, dlp_dr_min) = log_sigmoid_with_grad(z_min, PRIOR_K);
    lp += lp_min;

    // r_max barrier: log(sigmoid(K * (r_max - r)))
    let z_max = PRIOR_K * (PRIOR_R_MAX - r);
    let (lp_max, dlp_dz_max) = log_sigmoid_with_grad(z_max, PRIOR_K);
    lp += lp_max;
    let dlp_dr_max = -dlp_dz_max;

    // Distance gradient (position only).
    let dlp_dr = dlp_dr_min + dlp_dr_max;
    let inv_r = 1.0 / r;
    grad[0] = dlp_dr * px * inv_r;
    grad[1] = dlp_dr * py * inv_r;
    grad[2] = dlp_dr * pz * inv_r;

    // Eccentricity barrier: log(sigmoid(K * (1 - e^2)))
    //
    // The eccentricity vector for a Keplerian orbit about the Sun:
    //   e_vec = ((v^2 - mu/r)*pos - (pos*vel)*vel) / mu
    //
    // Using e^2 = |e_vec|^2 avoids a 1/e singularity at e = 0.
    let mu = GMS;
    let inv_mu = 1.0 / mu;
    let a_coeff = v2 - mu / r;
    let rdotv = px * vx + py * vy + pz * vz;

    let ex = (a_coeff * px - rdotv * vx) * inv_mu;
    let ey = (a_coeff * py - rdotv * vy) * inv_mu;
    let ez = (a_coeff * pz - rdotv * vz) * inv_mu;
    let e_sq = ex * ex + ey * ey + ez * ez;

    let z_ecc = PRIOR_K * (1.0 - e_sq);
    let (lp_ecc, dlp_df_ecc) = log_sigmoid_with_grad(z_ecc, PRIOR_K);
    lp += lp_ecc;

    // Gradient of e^2 w.r.t. the 6 state variables.
    //
    // d(e_vec_i)/d(r_j) = r_i*r_j/r^3 + (a_coeff/mu)*delta_ij - v_i*v_j/mu
    // d(e_vec_i)/d(v_j) = (2*v_j*r_i - r_j*v_i - (r*v)*delta_ij) / mu
    // d(e^2)/d(.) = 2 * Sum_i e_vec_i * d(e_vec_i)/d(.)
    //
    // dlp/d(state) = dlp/d(1-e^2) * d(1-e^2)/d(state) = -dlp_df_ecc * d(e^2)/d(state)
    let e_vec = [ex, ey, ez];
    let pos_arr = [px, py, pz];
    let vel_arr = [vx, vy, vz];
    let inv_r3 = 1.0 / (r * r2);
    let a_over_mu = a_coeff * inv_mu;

    for j in 0..3 {
        let mut de_sq_dr = 0.0;
        let mut de_sq_dv = 0.0;
        for i in 0..3 {
            let kron = if i == j { 1.0 } else { 0.0 };

            let de_dr = pos_arr[i] * pos_arr[j] * inv_r3 + a_over_mu * kron
                - vel_arr[i] * vel_arr[j] * inv_mu;
            de_sq_dr += e_vec[i] * de_dr;

            let de_dv =
                (2.0 * vel_arr[j] * pos_arr[i] - pos_arr[j] * vel_arr[i] - rdotv * kron) * inv_mu;
            de_sq_dv += e_vec[i] * de_dv;
        }
        grad[j] += -dlp_df_ecc * 2.0 * de_sq_dr;
        grad[3 + j] += -dlp_df_ecc * 2.0 * de_sq_dv;
    }

    (lp, grad)
}

/// Compute `log(sigmoid(z))` and `d(log sigmoid(z))/d(x)` for `z = k*x`.
///
/// Returns `(lp, dlp_dx)` where `dlp_dx = sigma(-z) * k`.  When the caller
/// has `z = k * f(x)`, multiply the second return by `df/dx` to get the
/// total derivative.
fn log_sigmoid_with_grad(z: f64, k: f64) -> (f64, f64) {
    let lp = if z > 20.0 {
        0.0
    } else if z < -20.0 {
        z
    } else {
        -(1.0 + (-z).exp()).ln()
    };

    let sig_neg_z = if z > 20.0 {
        (-z).exp()
    } else if z < -20.0 {
        1.0
    } else {
        1.0 / (1.0 + z.exp())
    };

    (lp, sig_neg_z * k)
}

// ---------------------------------------------------------------------------

/// Log-posterior density over orbital states, parameterized in whitened
/// Cartesian coordinates centered on the seed state.
struct OrbitalPosterior {
    /// Seed state at the reference epoch.
    seed_state: State<Equatorial>,
    /// Whitening factor (sqrt-covariance), D x D.
    whiten_l: DMatrix<f64>,
    /// Seed vector `[x, y, z, vx, vy, vz, ng_params...]`, D-vector.
    seed_vec: DVector<f64>,
    /// Observations (time-sorted, shared across chains).
    obs: Arc<[AstrometricObservation]>,
    /// Inclusion mask for observations (from DC outlier rejection).
    included: Vec<bool>,
    /// Whether to include extended (asteroid) perturbers.
    include_asteroids: bool,
    /// Non-gravitational model (if any).
    non_grav: Option<NonGravModel>,
}

impl OrbitalPosterior {
    /// Transform whitened coordinates back to Cartesian + non-grav vector.
    fn xi_to_cart(&self, xi: &[f64]) -> DVector<f64> {
        let xi_vec = DVector::from_column_slice(xi);
        &self.seed_vec + &self.whiten_l * &xi_vec
    }

    /// Build a `State` (and optional `NonGravModel`) from the full
    /// Cartesian parameter vector.
    fn vec_to_state(
        &self,
        cart_full: &DVector<f64>,
    ) -> (State<Equatorial>, Option<NonGravModel>, [f64; 3], [f64; 3]) {
        let pos: [f64; 3] = cart_full.as_slice()[..3].try_into().unwrap();
        let vel: [f64; 3] = cart_full.as_slice()[3..6].try_into().unwrap();

        let mut state = self.seed_state.clone();
        state.pos = pos.into();
        state.vel = vel.into();

        let ng = self.non_grav.as_ref().map(|model| {
            let mut m = model.clone();
            let np = m.n_free_params();
            let mut params = vec![0.0; np];
            for k in 0..np {
                params[k] = cart_full[6 + k];
            }
            m.set_free_params(&params);
            m
        });

        (state, ng, pos, vel)
    }

    /// Compute logp and gradient from an STM sweep result.
    fn logp_from_sweep(
        &self,
        sweep: &[StmObs],
        pos: &[f64; 3],
        vel: &[f64; 3],
        grad_xi: &mut [f64],
    ) -> f64 {
        let d = self.seed_vec.len();
        let mut grad_cart = DVector::<f64>::zeros(d);
        let mut logp = 0.0;
        let nu = STUDENT_NU;
        let gaussian = nu.is_infinite();

        for entry in sweep {
            let m = entry.residual.len();
            let h_epoch = &entry.h_local * &entry.phi_cum;

            for k in 0..m {
                let r = entry.residual[k];
                let sigma2 = 1.0 / entry.weights[k];

                if gaussian {
                    logp += -0.5 * r * r / sigma2;
                    let dl_dx_factor = r / sigma2;
                    for j in 0..d {
                        grad_cart[j] += h_epoch[(k, j)] * dl_dx_factor;
                    }
                } else {
                    let s = r * r / (nu * sigma2);
                    logp += -0.5 * (nu + 1.0) * (1.0 + s).ln();
                    let dl_dx_factor = (nu + 1.0) * r / (nu * sigma2 + r * r);
                    for j in 0..d {
                        grad_cart[j] += h_epoch[(k, j)] * dl_dx_factor;
                    }
                }
            }
        }

        // Physical prior (operates on Cartesian state).
        let (lp_prior, grad_prior) = physical_prior(pos, vel);
        logp += lp_prior;
        for i in 0..6 {
            grad_cart[i] += grad_prior[i];
        }

        // Transform to whitened coordinates: grad_xi = L^T * grad_cart.
        let g = self.whiten_l.transpose() * &grad_cart;
        for j in 0..d {
            grad_xi[j] = g[j];
        }

        logp
    }
}

impl nuts_rs::HasDims for OrbitalPosterior {
    fn dim_sizes(&self) -> HashMap<String, u64> {
        let mut m = HashMap::new();
        let _ = m.insert("dim".to_string(), self.seed_vec.len() as u64);
        m
    }
}

impl CpuLogpFunc for OrbitalPosterior {
    type LogpError = PropagationError;
    type FlowParameters = ();
    type ExpandedVector = Vec<f64>;

    fn dim(&self) -> usize {
        self.seed_vec.len()
    }

    fn logp(&mut self, position: &[f64], gradient: &mut [f64]) -> Result<f64, Self::LogpError> {
        let cart_full = self.xi_to_cart(position);
        let (trial_state, trial_ng, pos, vel) = self.vec_to_state(&cart_full);

        // Hard wall: reject unbound (hyperbolic) proposals.
        // Two-body energy: E = v^2/2 - mu/r.  Bound <=> E < 0 <=> v^2 < 2*mu/r.
        let r2 = pos[0] * pos[0] + pos[1] * pos[1] + pos[2] * pos[2];
        let v2 = vel[0] * vel[0] + vel[1] * vel[1] + vel[2] * vel[2];
        let r = r2.sqrt();
        if r < 1e-15 || v2 >= 2.0 * GMS / r {
            return Err(PropagationError {
                msg: "unbound orbit".into(),
                recoverable: true,
            });
        }

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

        let lp = self.logp_from_sweep(&sweep, &pos, &vel, gradient);
        Ok(lp)
    }

    fn expand_vector<R: rand::Rng + ?Sized>(
        &mut self,
        _rng: &mut R,
        array: &[f64],
    ) -> Result<Self::ExpandedVector, CpuMathError> {
        let cart_full = self.xi_to_cart(array);
        Ok(cart_full.as_slice().to_vec())
    }
}

/// Build a whitening Cholesky factor from the seed state via a single-pass
/// linearization.  If the STM sweep or information matrix inversion fails,
/// fall back to a diagonal heuristic.
fn build_cholesky(
    seed: &State<Equatorial>,
    obs: &[AstrometricObservation],
    include_asteroids: bool,
    non_grav: Option<&NonGravModel>,
) -> DMatrix<f64> {
    let np = non_grav.map_or(0, NonGravModel::n_free_params);
    let included = vec![true; obs.len()];

    if let Ok((info_mat, _, _)) =
        accumulate_normal_equations(seed, obs, &included, include_asteroids, non_grav)
        && let Some(chol) = sqrt_cov_from_info(&info_mat)
    {
        return chol;
    }

    diagonal_heuristic_whiten_cart(seed, np)
}

/// Compute a whitening factor directly from the information matrix.
///
/// One eigendecomposition of `info` yields eigenvectors `V` and eigenvalues
/// `e_i`.  The covariance is `V * diag(1/e_i) * V^T`, so its square root
/// is `L = V * diag(1/sqrt(e_i))`.  Eigenvalues below `1e-14 * max(e)`
/// are raised to that threshold to cap the condition number.
///
/// Returns `None` if the matrix is fully degenerate.
fn sqrt_cov_from_info(info: &DMatrix<f64>) -> Option<DMatrix<f64>> {
    let eigen = info.clone().symmetric_eigen();
    let max_eig = eigen.eigenvalues.iter().copied().fold(0.0_f64, f64::max);
    if max_eig < 1e-30 {
        return None;
    }
    let threshold = max_eig * 1e-14;
    let d = info.nrows();
    let mut scale = DVector::<f64>::zeros(d);
    for i in 0..d {
        let e = eigen.eigenvalues[i].max(threshold);
        scale[i] = 1.0 / e.sqrt();
    }
    Some(&eigen.eigenvectors * DMatrix::from_diagonal(&scale))
}

/// Diagonal heuristic in Cartesian space.
///
/// Position uncertainties are set to 1% of the current heliocentric
/// distance, velocity uncertainties to 1% of the current speed.
fn diagonal_heuristic_whiten_cart(seed: &State<Equatorial>, np: usize) -> DMatrix<f64> {
    let d = 6 + np;
    let pos: [f64; 3] = seed.pos.into();
    let vel: [f64; 3] = seed.vel.into();
    let r = (pos[0] * pos[0] + pos[1] * pos[1] + pos[2] * pos[2])
        .sqrt()
        .max(0.1);
    let v = (vel[0] * vel[0] + vel[1] * vel[1] + vel[2] * vel[2])
        .sqrt()
        .max(1e-4);

    let mut l = DMatrix::<f64>::zeros(d, d);
    for i in 0..3 {
        l[(i, i)] = 0.01 * r;
    }
    for i in 3..6 {
        l[(i, i)] = 0.01 * v;
    }
    for i in 6..d {
        l[(i, i)] = 1e-10;
    }
    l
}

// fit_orbit_mcmc -- public entry point

/// Estimate orbit uncertainty from observations using MCMC.
///
/// Given one or more candidate states (seeds) and a set of observations,
/// this produces a collection of plausible orbits that are statistically
/// consistent with the data.  Parallel chains are run automatically.
///
/// All seeds must share the same reference epoch.
///
/// A single non-gravitational model (if any) is shared across all chains.
///
/// When there are fewer seeds than CPU cores, each seed spawns multiple
/// independent sub-chains with dispersed starting points.  The `chain_id`
/// in the returned [`OrbitSamples`] identifies the seed (orbital mode),
/// not the sub-chain.
///
/// `num_draws` is the **total** number of orbit samples returned across
/// all seeds.  Each seed receives `num_draws / n_seeds` samples (remainder
/// goes to the first seeds).
///
/// # Arguments
/// * `seeds` -- Candidate orbital states (e.g. from IOD), one per mode.
///   Seeds at different epochs are automatically propagated to the first
///   seed's epoch via two-body.
/// * `obs` -- Observations (any order; sorted internally).
/// * `include_asteroids` -- Whether to include asteroid perturbers.
/// * `num_draws` -- Total orbit samples across all seeds.
/// * `num_tune` -- Warmup steps per sub-chain (default 500).  These are
///   discarded after adaptation.
/// * `non_grav` -- Optional shared non-gravitational model.
/// * `maxdepth` -- Maximum sampler tree depth (default 10).  Higher values
///   allow more thorough exploration at greater cost.
/// * `target_accept` -- Target acceptance probability for step-size
///   adaptation (default 0.8).  Lowering this (e.g. 0.6) makes the sampler
///   take larger steps, which helps in poorly constrained situations.
///
/// # Errors
/// Returns an error if `seeds` is empty or two-body propagation fails.
pub fn fit_orbit_mcmc(
    seeds: &[State<Equatorial>],
    obs: &[AstrometricObservation],
    include_asteroids: bool,
    num_draws: usize,
    num_tune: usize,
    non_grav: Option<&NonGravModel>,
    maxdepth: u64,
    target_accept: f64,
) -> KeteResult<OrbitSamples> {
    if seeds.is_empty() {
        return Err(Error::ValueError("No seeds provided".into()));
    }

    // Propagate all seeds to the first seed's epoch if needed.
    let epoch = seeds[0].epoch;
    let seeds: Vec<State<Equatorial>> = seeds
        .iter()
        .map(|s| {
            if (s.epoch.jd - epoch.jd).abs() > 1e-12 {
                propagate_two_body(s, epoch)
            } else {
                Ok(s.clone())
            }
        })
        .collect::<KeteResult<Vec<_>>>()?;

    // Sort observations once and share across chains.
    let mut sorted_obs = obs.to_vec();
    sorted_obs.sort_by(|a, b| {
        a.epoch()
            .jd
            .partial_cmp(&b.epoch().jd)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let sorted_obs: Arc<[AstrometricObservation]> = sorted_obs.into();

    // Distribute num_draws across seeds, then sub-chains across cores.
    let n_cores = std::thread::available_parallelism()
        .map(std::num::NonZero::get)
        .unwrap_or(1);
    let n_seeds = seeds.len();
    let chains_per_seed = (n_cores / n_seeds).max(1);

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

    // Pre-compute Cholesky factors for each seed (serial, fast).
    let chol_factors: Vec<DMatrix<f64>> = seeds
        .iter()
        .map(|seed| build_cholesky(seed, &sorted_obs, include_asteroids, non_grav))
        .collect();

    // Run all chains in parallel.
    let chain_results: Vec<(usize, KeteResult<(Vec<Vec<f64>>, Vec<bool>)>)> = tasks
        .par_iter()
        .map(|&(seed_idx, draws, rng_seed)| {
            let result = run_single_chain(
                &seeds[seed_idx],
                &chol_factors[seed_idx],
                &sorted_obs,
                include_asteroids,
                non_grav,
                draws,
                num_tune,
                maxdepth,
                target_accept,
                rng_seed,
            );
            (seed_idx, result)
        })
        .collect();

    // Collect results.
    let mut all_draws = Vec::new();
    let mut all_chain_id = Vec::new();
    let mut all_divergent = Vec::new();

    for (seed_idx, result) in chain_results {
        let (draws, divergent) = result?;
        let n = draws.len();
        all_draws.extend(draws);
        all_chain_id.extend(std::iter::repeat_n(seed_idx, n));
        all_divergent.extend(divergent);
    }

    Ok(OrbitSamples {
        desig: seeds[0].desig.to_string(),
        epoch: epoch.jd,
        draws: all_draws,
        chain_id: all_chain_id,
        divergent: all_divergent,
    })
}

/// Run a single NUTS chain for one seed.
fn run_single_chain(
    seed: &State<Equatorial>,
    whiten_l: &DMatrix<f64>,
    sorted_obs: &Arc<[AstrometricObservation]>,
    include_asteroids: bool,
    non_grav: Option<&NonGravModel>,
    num_draws: usize,
    num_tune: usize,
    maxdepth: u64,
    target_accept: f64,
    chain_idx: u64,
) -> KeteResult<(Vec<Vec<f64>>, Vec<bool>)> {
    let np = non_grav.map_or(0, NonGravModel::n_free_params);
    let d = 6 + np;

    // Seed vector in Cartesian coordinates.
    let pos: [f64; 3] = seed.pos.into();
    let vel: [f64; 3] = seed.vel.into();
    let mut seed_vec = DVector::<f64>::zeros(d);
    seed_vec[0] = pos[0];
    seed_vec[1] = pos[1];
    seed_vec[2] = pos[2];
    seed_vec[3] = vel[0];
    seed_vec[4] = vel[1];
    seed_vec[5] = vel[2];
    if let Some(ng) = non_grav {
        let params = ng.get_free_params();
        for k in 0..np {
            seed_vec[6 + k] = params[k];
        }
    }

    // If the seed orbit is hyperbolic, project velocity to make it
    // marginally bound and fall back to the diagonal heuristic.
    let r = (pos[0] * pos[0] + pos[1] * pos[1] + pos[2] * pos[2]).sqrt();
    let v = (vel[0] * vel[0] + vel[1] * vel[1] + vel[2] * vel[2]).sqrt();
    let v_esc = (2.0 * GMS / r.max(1e-15)).sqrt();
    let whiten_l = if v >= v_esc {
        let scale = 0.95 * v_esc / v;
        seed_vec[3] *= scale;
        seed_vec[4] *= scale;
        seed_vec[5] *= scale;
        diagonal_heuristic_whiten_cart(seed, np)
    } else {
        whiten_l.clone()
    };

    let posterior = OrbitalPosterior {
        seed_state: seed.clone(),
        whiten_l: whiten_l.clone(),
        seed_vec: seed_vec.clone(),
        obs: Arc::clone(sorted_obs),
        included: vec![true; sorted_obs.len()],
        include_asteroids,
        non_grav: non_grav.cloned(),
    };

    let mut settings = DiagGradNutsSettings {
        num_tune: num_tune as u64,
        num_draws: num_draws as u64,
        maxdepth,
        seed: chain_idx,
        num_chains: 1,
        ..DiagGradNutsSettings::default()
    };
    settings.adapt_options.step_size_settings.target_accept = target_accept;

    let math = CpuMath::new(posterior);
    let mut rng = rand::rngs::SmallRng::seed_from_u64(chain_idx);
    let mut sampler = settings.new_chain(chain_idx, math, &mut rng);

    // Initialize at a random draw from the whitening distribution
    // (standard normal in xi-space ~= covariance sample in Cartesian).
    // This disperses sub-chains across the prior, improving exploration
    // of elongated or multi-modal posteriors.
    let init: Vec<f64> = {
        use rand::distr::Uniform;
        use rand::prelude::Distribution;
        let uniform = Uniform::new(-1.0, 1.0).unwrap();
        (0..d).map(|_| uniform.sample(&mut rng)).collect()
    };
    sampler
        .set_position(&init)
        .map_err(|e| Error::ValueError(format!("NUTS init failed: {e}")))?;

    let total_draws = num_tune as u64 + num_draws as u64;
    let mut draws = Vec::with_capacity(num_draws);
    let mut divergent = Vec::with_capacity(num_draws);

    for _ in 0..total_draws {
        let (position, progress) = sampler
            .draw()
            .map_err(|e| Error::ValueError(format!("NUTS draw failed: {e}")))?;

        if progress.tuning {
            continue;
        }

        // Convert from whitened xi-coordinates back to Cartesian.
        let xi = DVector::from_column_slice(position.as_ref());
        let cart = &seed_vec + &whiten_l * &xi;
        draws.push(cart.as_slice().to_vec());
        divergent.push(progress.diverging);
    }

    Ok((draws, divergent))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn physical_prior_nominal_orbit_no_penalty() {
        // ~1 AU circular orbit: well inside allowed bounds, e ~= 0.03.
        let pos = [1.0, 0.0, 0.0];
        // ~circular speed at 1 AU (AU/day)
        let vel = [0.0, 0.017, 0.0];
        let (lp, grad) = physical_prior(&pos, &vel);
        // logp should be modest (eccentricity barrier contributes a small term
        // even for a valid orbit because e > 0 but not near 1).
        assert!(lp > -1.0, "logp = {lp}, expected > -1 for nominal orbit");
        // gradient should be finite.
        assert!(
            grad.iter().all(|g| g.is_finite()),
            "gradient must be finite"
        );
    }

    #[test]
    fn physical_prior_too_close_penalized() {
        // r = 0.001 AU -- well below PRIOR_R_MIN = 0.01.
        let pos = [0.001, 0.0, 0.0];
        let vel = [0.0, 0.01, 0.0];
        let (lp, grad) = physical_prior(&pos, &vel);
        assert!(lp < -1.0, "logp = {lp}, expected penalty for r << r_min");
        assert!(grad[0].is_finite(), "gradient must be finite");
    }

    #[test]
    fn physical_prior_too_far_penalized() {
        // r = 5000 AU -- well above PRIOR_R_MAX.
        let pos = [5000.0, 0.0, 0.0];
        let vel = [0.0, 1e-5, 0.0];
        let (lp, grad) = physical_prior(&pos, &vel);
        assert!(
            lp < -10.0,
            "logp = {lp}, expected large penalty for r >> r_max"
        );
        assert!(grad[0].is_finite(), "gradient must be finite");
    }

    #[test]
    fn physical_prior_hyperbolic_penalized() {
        // r = 1 AU, v = 0.1 AU/day (>> v_esc ~= 0.024). Eccentricity >> 1.
        let pos = [1.0, 0.0, 0.0];
        let vel = [0.0, 0.1, 0.0];
        let (lp, grad) = physical_prior(&pos, &vel);
        assert!(lp < -5.0, "logp = {lp}, expected penalty for e >> 1");
        assert!(grad[4].is_finite(), "velocity gradient must be finite");
    }

    #[test]
    fn physical_prior_zero_radius() {
        let pos = [0.0, 0.0, 0.0];
        let vel = [0.0, 0.0, 0.0];
        let (lp, grad) = physical_prior(&pos, &vel);
        assert!(lp < -1e9, "logp = {lp}, expected huge penalty for r=0");
        // gradient should be finite (we return early with zeros).
        assert!(grad.iter().all(|g| g.is_finite()));
    }

    #[test]
    fn sqrt_cov_from_info_identity() {
        // info = I_6 => cov = I_6 => L = I_6.
        let info = DMatrix::<f64>::identity(6, 6);
        let l = sqrt_cov_from_info(&info).expect("should not be None");
        // L * L^T should be close to I (the covariance).
        let cov = &l * l.transpose();
        for i in 0..6 {
            for j in 0..6 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (cov[(i, j)] - expected).abs() < 1e-12,
                    "cov[({i},{j})] = {}, expected {expected}",
                    cov[(i, j)]
                );
            }
        }
    }

    #[test]
    fn sqrt_cov_from_info_scaled() {
        // info = diag(4, 100, 1, 1, 1, 1) => cov = diag(1/4, 1/100, 1, ...).
        let mut info = DMatrix::<f64>::identity(6, 6);
        info[(0, 0)] = 4.0;
        info[(1, 1)] = 100.0;
        let l = sqrt_cov_from_info(&info).expect("should not be None");
        let cov = &l * l.transpose();
        assert!(
            (cov[(0, 0)] - 0.25).abs() < 1e-12,
            "cov[0,0] = {}",
            cov[(0, 0)]
        );
        assert!(
            (cov[(1, 1)] - 0.01).abs() < 1e-12,
            "cov[1,1] = {}",
            cov[(1, 1)]
        );
    }

    #[test]
    fn sqrt_cov_from_info_degenerate_returns_none() {
        let info = DMatrix::<f64>::zeros(6, 6);
        assert!(sqrt_cov_from_info(&info).is_none());
    }

    // ------------------------------------------------------------------
    // Integration test: full logp gradient via whitened Cartesian
    // ------------------------------------------------------------------

    #[test]
    fn logp_gradient_matches_finite_difference() {
        use kete_core::desigs::Desig;

        // Construct a minimal OrbitalPosterior with zero observations.
        // logp = physical_prior only (no likelihood term).
        let seed_pos = [1.5, 0.3, -0.1];
        let seed_vel = [0.002, -0.014, 0.001];
        let seed_state: State<Equatorial> = State::new(
            Desig::Empty,
            2460000.5.into(),
            seed_pos.into(),
            seed_vel.into(),
            0,
        );

        let d = 6;
        let mut seed_vec = DVector::<f64>::zeros(d);
        seed_vec[0] = seed_pos[0];
        seed_vec[1] = seed_pos[1];
        seed_vec[2] = seed_pos[2];
        seed_vec[3] = seed_vel[0];
        seed_vec[4] = seed_vel[1];
        seed_vec[5] = seed_vel[2];

        // Identity whitening: xi == Cartesian deviations.
        let whiten_l = DMatrix::<f64>::identity(d, d);
        let obs: Arc<[AstrometricObservation]> = Vec::new().into();

        let mut posterior = OrbitalPosterior {
            seed_state,
            whiten_l,
            seed_vec,
            obs: Arc::clone(&obs),
            included: vec![],
            include_asteroids: false,
            non_grav: None,
        };

        // Evaluate at a small offset from the seed.
        let xi = [0.001, -0.0005, 0.002, 0.0001, -0.0001, 0.0003];
        let mut grad = [0.0_f64; 6];
        let lp = posterior
            .logp(&xi, &mut grad)
            .expect("logp should succeed with no observations");

        assert!(lp.is_finite(), "logp = {lp}");
        assert!(grad.iter().all(|g| g.is_finite()), "gradient not finite");

        // Finite-difference check on each component.
        let eps = 1e-6;
        for i in 0..6 {
            let mut xi_p = xi;
            let mut xi_m = xi;
            xi_p[i] += eps;
            xi_m[i] -= eps;

            let mut g_dummy = [0.0_f64; 6];
            let lp_p = posterior.logp(&xi_p, &mut g_dummy).unwrap();
            let lp_m = posterior.logp(&xi_m, &mut g_dummy).unwrap();
            let fd = (lp_p - lp_m) / (2.0 * eps);

            assert!(
                (grad[i] - fd).abs() < 1e-4,
                "grad[{i}]: analytic={} fd={} diff={}",
                grad[i],
                fd,
                (grad[i] - fd).abs()
            );
        }
    }

    #[test]
    fn logp_gradient_nontrivial_whitening() {
        use kete_core::desigs::Desig;

        // Same as above but with a realistic, correlated whitening matrix.
        let seed_pos = [1.5, 0.3, -0.1];
        let seed_vel = [0.002, -0.014, 0.001];
        let seed_state: State<Equatorial> = State::new(
            Desig::Empty,
            2460000.5.into(),
            seed_pos.into(),
            seed_vel.into(),
            0,
        );

        let d = 6;
        let mut seed_vec = DVector::<f64>::zeros(d);
        seed_vec[0] = seed_pos[0];
        seed_vec[1] = seed_pos[1];
        seed_vec[2] = seed_pos[2];
        seed_vec[3] = seed_vel[0];
        seed_vec[4] = seed_vel[1];
        seed_vec[5] = seed_vel[2];

        // Build a whitening matrix from a synthetic information matrix.
        // The info matrix is the inverse of the covariance.
        let mut info = DMatrix::<f64>::zeros(d, d);
        info[(0, 0)] = 1e4;
        info[(1, 1)] = 5e3;
        info[(2, 2)] = 1e4;
        info[(3, 3)] = 1e8;
        info[(4, 4)] = 5e7;
        info[(5, 5)] = 1e8;
        // Off-diagonal coupling.
        info[(0, 3)] = 5e5;
        info[(3, 0)] = 5e5;
        info[(1, 4)] = -3e5;
        info[(4, 1)] = -3e5;

        let whiten_l = sqrt_cov_from_info(&info).expect("info matrix should be valid");
        let obs: Arc<[AstrometricObservation]> = Vec::new().into();

        let mut posterior = OrbitalPosterior {
            seed_state,
            whiten_l,
            seed_vec,
            obs: Arc::clone(&obs),
            included: vec![],
            include_asteroids: false,
            non_grav: None,
        };

        // Evaluate at a moderate offset in whitened space.
        let xi = [0.5, -0.3, 0.1, 0.2, -0.7, 0.4];
        let mut grad = [0.0_f64; 6];
        let lp = posterior.logp(&xi, &mut grad).expect("logp should succeed");

        assert!(lp.is_finite(), "logp = {lp}");
        assert!(grad.iter().all(|g| g.is_finite()), "gradient not finite");

        let eps = 1e-6;
        for i in 0..6 {
            let mut xi_p = xi;
            let mut xi_m = xi;
            xi_p[i] += eps;
            xi_m[i] -= eps;

            let mut g_dummy = [0.0_f64; 6];
            let lp_p = posterior.logp(&xi_p, &mut g_dummy).unwrap();
            let lp_m = posterior.logp(&xi_m, &mut g_dummy).unwrap();
            let fd = (lp_p - lp_m) / (2.0 * eps);

            assert!(
                (grad[i] - fd).abs() < 1e-3,
                "grad[{i}]: analytic={} fd={} diff={}",
                grad[i],
                fd,
                (grad[i] - fd).abs()
            );
        }
    }
}
