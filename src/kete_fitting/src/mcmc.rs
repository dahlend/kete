//! MCMC orbit uncertainty estimation from observations.
//!
//! Provides [`fit_orbit_mcmc`], which estimates the range of orbits
//! consistent with a set of observations by running parallel MCMC chains.
//! Each candidate orbital state (seed) gets its own chain, and the results
//! are pooled into a single [`OrbitSamples`] collection.
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

use crate::obs::Observation;
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

/// Minimum heliocentric distance (AU) before penalty ramps up.
const PRIOR_R_MIN: f64 = 0.01;
/// Maximum heliocentric distance (AU) before penalty ramps up.
const PRIOR_R_MAX: f64 = 1000.0;
/// Steepness of the logistic barrier.
const PRIOR_K: f64 = 100.0;

/// Smooth physical prior: penalizes unphysical orbits with differentiable
/// logistic barriers so the gradient is always well-defined.
///
/// Penalties:
///   - heliocentric distance below `PRIOR_R_MIN` or above `PRIOR_R_MAX`
///   - speed exceeding local escape speed `v_esc = sqrt(2 * GMS / r)`
///
/// Returns `(log_prior, grad_prior)` where `grad_prior` is a D-vector.
fn physical_prior(x: &DVector<f64>) -> (f64, DVector<f64>) {
    let d = x.len();
    let mut grad = DVector::<f64>::zeros(d);
    let mut lp = 0.0;

    let px = x[0];
    let py = x[1];
    let pz = x[2];
    let vx = x[3];
    let vy = x[4];
    let vz = x[5];

    let r2 = px * px + py * py + pz * pz;
    let r = r2.sqrt();
    let v2 = vx * vx + vy * vy + vz * vz;
    let v = v2.sqrt();

    if r < 1e-15 {
        return (-1e10, grad);
    }

    // r_min barrier: log(sigmoid(K * (r - r_min)))
    let z_min = PRIOR_K * (r - PRIOR_R_MIN);
    let (lp_min, dlp_dr_min) = log_sigmoid_with_grad(z_min, PRIOR_K);
    lp += lp_min;

    // r_max barrier: log(sigmoid(K * (r_max - r)))
    let z_max = PRIOR_K * (PRIOR_R_MAX - r);
    let (lp_max, dlp_dz_max) = log_sigmoid_with_grad(z_max, PRIOR_K);
    lp += lp_max;
    // dz_max/dr = -K
    let dlp_dr_max = -dlp_dz_max;

    // escape speed barrier: log(sigmoid(K * (v_esc - v)))
    let v_esc = (2.0 * GMS / r).sqrt();
    let z_esc = PRIOR_K * (v_esc - v);
    let (lp_esc, dlp_dz_esc) = log_sigmoid_with_grad(z_esc, PRIOR_K);
    lp += lp_esc;

    let dv_esc_dr = -GMS / (r2 * v_esc);
    let dlp_dr_esc = dlp_dz_esc * dv_esc_dr;

    let dlp_dr = dlp_dr_min + dlp_dr_max + dlp_dr_esc;
    let inv_r = 1.0 / r;
    grad[0] += dlp_dr * px * inv_r;
    grad[1] += dlp_dr * py * inv_r;
    grad[2] += dlp_dr * pz * inv_r;

    if v > 1e-15 {
        let dlp_dv = -dlp_dz_esc;
        let inv_v = 1.0 / v;
        grad[3] += dlp_dv * vx * inv_v;
        grad[4] += dlp_dv * vy * inv_v;
        grad[5] += dlp_dv * vz * inv_v;
    }

    (lp, grad)
}

/// Compute `log(sigmoid(z))` and `sigmoid(-z) * k`.
///
/// Returns `(lp, d(lp)/d(outer))` where the caller supplies `df/d(outer)`
/// separately.
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

/// Log-posterior density over orbital states, parameterized in a whitened
/// coordinate system centered on the seed state.
struct OrbitalPosterior {
    /// Seed state at the reference epoch.
    seed_state: State<Equatorial>,
    /// Lower Cholesky factor of the mass matrix, D x D.
    chol_l: DMatrix<f64>,
    /// Seed state vector (position ++ velocity ++ non-grav params), D-vector.
    seed_vec: DVector<f64>,
    /// Observations (time-sorted, shared across chains).
    obs: Arc<[Observation]>,
    /// Inclusion mask (all true).
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
    /// Transform whitened coordinates back to physical state vector.
    fn xi_to_x(&self, xi: &[f64]) -> DVector<f64> {
        let xi_vec = DVector::from_column_slice(xi);
        &self.seed_vec + &self.chol_l * &xi_vec
    }

    /// Transform whitened coordinates to a State + optional `NonGravModel`.
    fn xi_to_state(&self, xi: &[f64]) -> (State<Equatorial>, Option<NonGravModel>) {
        let x = self.xi_to_x(xi);

        let mut state = self.seed_state.clone();
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
    fn logp_from_sweep(&self, sweep: &[StmObs], xi: &[f64], grad_xi: &mut [f64]) -> f64 {
        let d = self.dim;
        let mut grad_x = DVector::<f64>::zeros(d);
        let mut logp = 0.0;
        let nu = self.student_nu;
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
                        grad_x[j] += h_epoch[(k, j)] * dl_dx_factor;
                    }
                } else {
                    let s = r * r / (nu * sigma2);
                    logp += -0.5 * (nu + 1.0) * (1.0 + s).ln();
                    let dl_dx_factor = (nu + 1.0) * r / (nu * sigma2 + r * r);
                    for j in 0..d {
                        grad_x[j] += h_epoch[(k, j)] * dl_dx_factor;
                    }
                }
            }
        }

        // Add physical prior.
        let x = self.xi_to_x(xi);
        let (lp_prior, grad_prior) = physical_prior(&x);
        logp += lp_prior;
        grad_x += &grad_prior;

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

        let lp = self.logp_from_sweep(&sweep, position, gradient);
        Ok(lp)
    }

    fn expand_vector<R: rand::Rng + ?Sized>(
        &mut self,
        _rng: &mut R,
        array: &[f64],
    ) -> Result<Self::ExpandedVector, CpuMathError> {
        let x = self.xi_to_x(array);
        Ok(x.as_slice().to_vec())
    }
}

/// Build a whitening Cholesky factor from the seed state via a single-pass
/// linearization.  If the STM sweep or information matrix inversion fails,
/// fall back to a diagonal heuristic.
fn build_cholesky(
    seed: &State<Equatorial>,
    obs: &[Observation],
    include_asteroids: bool,
    non_grav: Option<&NonGravModel>,
) -> DMatrix<f64> {
    let np = non_grav.map_or(0, NonGravModel::n_free_params);
    let included = vec![true; obs.len()];

    // Try single-pass linearization.
    if let Ok((info_mat, _, _)) =
        accumulate_normal_equations(seed, obs, &included, include_asteroids, non_grav)
        && let Some(chol) = cholesky_from_info(&info_mat)
    {
        return chol;
    }

    // Fallback: diagonal heuristic.
    diagonal_heuristic_cholesky(seed, np)
}

/// Compute a whitening factor directly from the information matrix.
///
/// One eigendecomposition of `info` yields eigenvectors `V` and eigenvalues
/// `e_i`.  The covariance is `V * diag(1/e_i) * V^T`, so its square root
/// is `L = V * diag(1/sqrt(e_i))`.  Eigenvalues below `1e-14 * max(e)`
/// are raised to that threshold to cap the condition number.
///
/// Returns `None` if the matrix is fully degenerate.
fn cholesky_from_info(info: &DMatrix<f64>) -> Option<DMatrix<f64>> {
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

/// Diagonal heuristic: 1% of heliocentric distance for position,
/// 1% of orbital speed for velocity.
fn diagonal_heuristic_cholesky(seed: &State<Equatorial>, np: usize) -> DMatrix<f64> {
    let d = 6 + np;
    let pos: [f64; 3] = seed.pos.into();
    let vel: [f64; 3] = seed.vel.into();

    let r = (pos[0] * pos[0] + pos[1] * pos[1] + pos[2] * pos[2])
        .sqrt()
        .max(0.1);
    let v = (vel[0] * vel[0] + vel[1] * vel[1] + vel[2] * vel[2])
        .sqrt()
        .max(1e-4);

    let pos_sigma = 0.01 * r;
    let vel_sigma = 0.01 * v;

    let mut l = DMatrix::<f64>::zeros(d, d);
    for i in 0..3 {
        l[(i, i)] = pos_sigma;
    }
    for i in 3..6 {
        l[(i, i)] = vel_sigma;
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
/// independent sub-chains.  The `chain_id` in the returned
/// [`OrbitSamples`] identifies the seed (orbital mode), not the sub-chain.
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
/// * `student_nu` -- Student-t degrees of freedom (`f64::INFINITY` for
///   Gaussian).  Lower values down-weight outlier observations.
/// * `non_grav` -- Optional shared non-gravitational model.
/// * `maxdepth` -- Maximum sampler tree depth (default 10).  Higher values
///   allow more thorough exploration at greater cost.
///
/// # Errors
/// Returns an error if `seeds` is empty or two-body propagation fails.
pub fn fit_orbit_mcmc(
    seeds: &[State<Equatorial>],
    obs: &[Observation],
    include_asteroids: bool,
    num_draws: usize,
    num_tune: usize,
    student_nu: f64,
    non_grav: Option<&NonGravModel>,
    maxdepth: u64,
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
    let sorted_obs: Arc<[Observation]> = sorted_obs.into();

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
                student_nu,
                maxdepth,
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
    chol_l: &DMatrix<f64>,
    sorted_obs: &Arc<[Observation]>,
    include_asteroids: bool,
    non_grav: Option<&NonGravModel>,
    num_draws: usize,
    num_tune: usize,
    student_nu: f64,
    maxdepth: u64,
    chain_idx: u64,
) -> KeteResult<(Vec<Vec<f64>>, Vec<bool>)> {
    let np = non_grav.map_or(0, NonGravModel::n_free_params);
    let d = 6 + np;

    let pos: [f64; 3] = seed.pos.into();
    let vel: [f64; 3] = seed.vel.into();
    let mut seed_vec = DVector::<f64>::zeros(d);
    for i in 0..3 {
        seed_vec[i] = pos[i];
        seed_vec[3 + i] = vel[i];
    }
    if let Some(ng) = non_grav {
        let params = ng.get_free_params();
        for k in 0..np {
            seed_vec[6 + k] = params[k];
        }
    }

    let posterior = OrbitalPosterior {
        seed_state: seed.clone(),
        chol_l: chol_l.clone(),
        seed_vec: seed_vec.clone(),
        obs: Arc::clone(sorted_obs),
        included: vec![true; sorted_obs.len()],
        include_asteroids,
        non_grav: non_grav.cloned(),
        student_nu,
        dim: d,
    };

    let settings = DiagGradNutsSettings {
        num_tune: num_tune as u64,
        num_draws: num_draws as u64,
        maxdepth,
        seed: chain_idx,
        num_chains: 1,
        ..DiagGradNutsSettings::default()
    };

    let math = CpuMath::new(posterior);
    let mut rng = rand::rngs::SmallRng::seed_from_u64(chain_idx);
    let mut sampler = settings.new_chain(chain_idx, math, &mut rng);

    let init = vec![0.0_f64; d];
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

        let xi = position.as_ref();
        let xi_vec = DVector::from_column_slice(xi);
        let x = &seed_vec + chol_l * &xi_vec;
        draws.push(x.as_slice().to_vec());
        divergent.push(progress.diverging);
    }

    Ok((draws, divergent))
}

#[cfg(test)]
mod tests {
    use super::*;
    use kete_core::desigs::Desig;

    fn make_state(pos: [f64; 3], vel: [f64; 3], jd: f64) -> State<Equatorial> {
        State::new(Desig::Empty, jd.into(), pos.into(), vel.into(), 0)
    }

    #[test]
    fn physical_prior_nominal_orbit_no_penalty() {
        // ~1 AU circular orbit: well inside allowed bounds.
        let mut x = DVector::<f64>::zeros(6);
        // 1 AU along x
        x[0] = 1.0;
        // ~circular speed at 1 AU (AU/day)
        x[4] = 0.017;
        let (lp, grad) = physical_prior(&x);
        // logp should be modest (escape-speed barrier contributes a small term
        // even for a valid orbit because v < v_esc but not by a large margin).
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
        let mut x = DVector::<f64>::zeros(6);
        x[0] = 0.001;
        x[4] = 0.01;
        let (lp, grad) = physical_prior(&x);
        assert!(lp < -1.0, "logp = {lp}, expected penalty for r << r_min");
        assert!(grad[0].is_finite(), "gradient must be finite");
    }

    #[test]
    fn physical_prior_too_far_penalized() {
        // r = 5000 AU -- well above PRIOR_R_MAX.
        let mut x = DVector::<f64>::zeros(6);
        x[0] = 5000.0;
        x[4] = 1e-5;
        let (lp, grad) = physical_prior(&x);
        assert!(
            lp < -10.0,
            "logp = {lp}, expected large penalty for r >> r_max"
        );
        assert!(grad[0].is_finite(), "gradient must be finite");
    }

    #[test]
    fn physical_prior_escape_speed_penalized() {
        // r = 1 AU, v_esc ~ 0.024 AU/day. Set v = 0.1 AU/day (>>v_esc).
        let mut x = DVector::<f64>::zeros(6);
        x[0] = 1.0;
        x[4] = 0.1;
        let (lp, grad) = physical_prior(&x);
        assert!(lp < -5.0, "logp = {lp}, expected penalty for v >> v_esc");
        assert!(grad[4].is_finite(), "velocity gradient must be finite");
    }

    #[test]
    fn physical_prior_zero_radius() {
        let x = DVector::<f64>::zeros(6);
        let (lp, grad) = physical_prior(&x);
        assert!(lp < -1e9, "logp = {lp}, expected huge penalty for r=0");
        // gradient should be finite (we return early with zeros).
        assert!(grad.iter().all(|g| g.is_finite()));
    }

    #[test]
    fn cholesky_from_info_identity() {
        // info = I_6 => cov = I_6 => L = I_6.
        let info = DMatrix::<f64>::identity(6, 6);
        let l = cholesky_from_info(&info).expect("should not be None");
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
    fn cholesky_from_info_scaled() {
        // info = diag(4, 100, 1, 1, 1, 1) => cov = diag(1/4, 1/100, 1, ...).
        let mut info = DMatrix::<f64>::identity(6, 6);
        info[(0, 0)] = 4.0;
        info[(1, 1)] = 100.0;
        let l = cholesky_from_info(&info).expect("should not be None");
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
    fn cholesky_from_info_degenerate_returns_none() {
        let info = DMatrix::<f64>::zeros(6, 6);
        assert!(cholesky_from_info(&info).is_none());
    }

    #[test]
    fn diagonal_heuristic_shape_and_values() {
        let seed = make_state([2.0, 0.0, 0.0], [0.0, 0.01, 0.0], 2451545.0);
        let l = diagonal_heuristic_cholesky(&seed, 0);
        assert_eq!(l.nrows(), 6);
        assert_eq!(l.ncols(), 6);
        // Position sigma: 0.01 * r = 0.01 * 2.0 = 0.02
        assert!((l[(0, 0)] - 0.02).abs() < 1e-14);
        assert!((l[(1, 1)] - 0.02).abs() < 1e-14);
        assert!((l[(2, 2)] - 0.02).abs() < 1e-14);
        // Velocity sigma: 0.01 * v = 0.01 * 0.01 = 0.0001
        assert!((l[(3, 3)] - 0.0001).abs() < 1e-14);
        // Off-diagonals zero.
        assert!((l[(0, 1)]).abs() < 1e-30);
    }

    #[test]
    fn diagonal_heuristic_with_nongrav_params() {
        let seed = make_state([1.0, 0.0, 0.0], [0.0, 0.017, 0.0], 2451545.0);
        let l = diagonal_heuristic_cholesky(&seed, 3);
        assert_eq!(l.nrows(), 9);
        assert_eq!(l.ncols(), 9);
        for i in 6..9 {
            assert!((l[(i, i)] - 1e-10).abs() < 1e-25);
        }
    }
}
