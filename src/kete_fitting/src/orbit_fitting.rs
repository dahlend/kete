//! Batch least-squares orbit fitting using differential correction.
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
use crate::uncertain_state::UncertainState;
use kete_core::frames::Equatorial;
use kete_core::prelude::{Error, KeteResult, State};
use kete_core::propagation::{
    NonGravModel, compute_state_transition, light_time_correct, propagate_n_body_spk,
};
use nalgebra::{DMatrix, DVector};

/// Result of orbit determination via batch least squares.
#[derive(Debug, Clone)]
pub struct OrbitFit {
    /// Core uncertain orbit (state + covariance + `non_grav` template).
    pub uncertain_state: UncertainState,

    /// Post-fit residuals for every observation (time-sorted).
    /// Each entry has as many elements as the measurement dimension
    /// of that observation.  Excluded observations have `NaN`
    /// residuals.
    pub residuals: Vec<DVector<f64>>,

    /// All input observations (time-sorted).
    pub observations: Vec<Observation>,

    /// Per-observation inclusion mask (time-sorted, same length as
    /// `observations`).  `true` means the observation was used in
    /// the final fit; `false` means it was rejected as an outlier.
    pub included: Vec<bool>,

    /// Reduced weighted RMS of residuals (included observations only).
    /// Divided by degrees of freedom (`n_measurements` - `n_params`).
    pub rms: f64,

    /// Whether the solver achieved strict convergence (correction norm
    /// dropped below `tol`). When `false` the fit is the best found
    /// within the iteration limit but may not be fully converged.
    pub converged: bool,
}

/// Fit an orbit to observations using iterative least squares.
///
/// Refines an initial orbital state guess to best match the observations,
/// and estimates the uncertainty of the result via a covariance matrix.
/// Can automatically identify and reject outlier observations.
///
/// The input `initial_state` **must** be SSB-centered (`center_id == 0`).
/// All internal propagation uses SSB coordinates.
///
/// Observations are fitted in progressively wider time windows
/// centered on the reference epoch: ±30, ±60, ±180, and ±360 days,
/// followed by a final pass that includes the full arc.  Each stage
/// bootstraps from the previous converged solution.  Windows that
/// contain fewer than 4 observations are skipped automatically.
/// The final pass re-evaluates all observations for outlier rejection
/// (if enabled).
///
/// Outlier rejection is controlled by `max_reject_passes`.  When zero,
/// no rejection is performed and the fit uses all observations.
///
/// When `auto_sigma` is true the effective rejection threshold is scaled
/// per pass by a robust estimate (MAD-based) of the actual residual
/// scatter, so it adapts to the data rather than relying on stated
/// uncertainties being correct.
///
/// # Arguments
/// * `initial_state` - Initial guess for the object state at the reference
///   epoch.
/// * `obs` - Observations (any order; they are sorted internally).
/// * `include_asteroids` - When true, include asteroid masses in the force model.
/// * `non_grav` - Optional non-gravitational model.
/// * `max_iter` - Maximum iterations per convergence pass.
/// * `tol` - Convergence tolerance on the state correction norm (AU for
///   position, AU/day for velocity).
/// * `chi2_threshold` - Per-observation chi-squared threshold for outlier
///   rejection.  Only used when `max_reject_passes > 0`.
/// * `max_reject_passes` - Maximum outlier-rejection cycles.  Set to 0 to
///   disable rejection entirely.
/// * `auto_sigma` - When true, adaptively rescale the rejection threshold
///   based on actual residual scatter.
///
/// # Errors
/// Fails if any internal propagation or solve fails.
pub fn fit_orbit(
    initial_state: &State<Equatorial>,
    obs: &[Observation],
    include_asteroids: bool,
    non_grav: Option<&NonGravModel>,
    max_iter: usize,
    tol: f64,
    chi2_threshold: f64,
    max_reject_passes: usize,
    auto_sigma: bool,
) -> KeteResult<OrbitFit> {
    if obs.is_empty() {
        return Err(Error::ValueError("No observations provided".into()));
    }
    let sorted: Vec<Observation> = sort_by_epoch(obs)
        .into_iter()
        .filter(|o| {
            let s = o.observer();
            let pos_ok: bool = s.pos.into_iter().all(|v: f64| v.is_finite());
            let vel_ok: bool = s.vel.into_iter().all(|v: f64| v.is_finite());
            pos_ok && vel_ok
        })
        .collect();
    if sorted.is_empty() {
        return Err(Error::ValueError(
            "No observations with finite observer states".into(),
        ));
    }
    let ref_jd = initial_state.epoch.jd;

    // Fixed window radii (days) centered on the reference epoch.
    // Each stage bootstraps from the previous converged solution.
    // Windows with fewer than 4 observations are skipped automatically.
    let windows: Vec<f64> = vec![30.0, 60.0, 180.0, 360.0, f64::INFINITY];

    let mut state = initial_state.clone();
    let ng = non_grav.cloned();
    let mut prev_n_in_window: usize = 0;

    // Expansion stages: converge + reject on each window.
    // Non-grav parameters are frozen during expansion because short arcs
    // have almost no sensitivity to them; fitting them here would produce
    // wildly wrong values that poison subsequent stages.  The original
    // non-grav values are preserved and used in the final full-arc pass.
    for &radius in &windows[..windows.len() - 1] {
        let included = select_obs_within_window(&sorted, ref_jd, radius);
        let n_in_window = included.iter().filter(|&&v| v).count();
        if n_in_window < 4 || n_in_window == prev_n_in_window {
            // Too few observations, or this window includes the same
            // set as the previous one -- skip.
            continue;
        }
        prev_n_in_window = n_in_window;
        if let Ok(result) = solve_with_rejection(
            &state,
            &sorted,
            &included,
            include_asteroids,
            None,
            max_iter,
            tol,
            chi2_threshold,
            max_reject_passes,
            auto_sigma,
        ) {
            state = result.uncertain_state.state.clone();
        }
        // On error: keep previous state, try the next wider window.
    }

    // Final full-arc pass: re-include all observations and reject anew.
    let included = vec![true; sorted.len()];
    solve_with_rejection(
        &state,
        &sorted,
        &included,
        include_asteroids,
        ng,
        max_iter,
        tol,
        chi2_threshold,
        max_reject_passes,
        auto_sigma,
    )
}

/// Build a boolean inclusion mask for observations within +/-`dt_days` of
/// `ref_jd`.
fn select_obs_within_window(sorted_obs: &[Observation], ref_jd: f64, dt_days: f64) -> Vec<bool> {
    sorted_obs
        .iter()
        .map(|ob| (ob.epoch().jd - ref_jd).abs() <= dt_days)
        .collect()
}

/// Converge + outlier-reject on a subset defined by `included`.
///
/// First converges, then batch-rejects all observations whose
/// per-observation chi-squared exceeds the (possibly rescaled) threshold.
///
/// When `auto_sigma` is true, the threshold is multiplied by a robust
/// variance scale factor estimated from the MAD of normalized residuals.
/// This makes rejection adaptive to the actual data scatter.
fn solve_with_rejection(
    initial_state: &State<Equatorial>,
    sorted_obs: &[Observation],
    included: &[bool],
    include_asteroids: bool,
    non_grav: Option<NonGravModel>,
    max_iter: usize,
    tol: f64,
    chi2_threshold: f64,
    max_reject_passes: usize,
    auto_sigma: bool,
) -> KeteResult<OrbitFit> {
    let mut fit = iterate_to_convergence(
        initial_state,
        sorted_obs,
        included,
        include_asteroids,
        non_grav,
        max_iter,
        tol,
    )?;

    // Batch rejection loop: reject all outliers per pass, then re-converge.
    let np = fit
        .uncertain_state
        .non_grav
        .as_ref()
        .map_or(0, NonGravModel::n_free_params);
    let min_included = (6 + np).max(4);

    for _ in 0..max_reject_passes {
        let n_included = fit.included.iter().filter(|&&inc| inc).count();
        if n_included <= min_included {
            break;
        }

        // When auto_sigma is enabled, estimate the robust variance scale
        // factor from the MAD of per-component normalized residuals
        // (r / sigma) across included observations.  The effective
        // threshold becomes chi2_threshold * scale^2, adapting to the
        // actual scatter in the data.
        let effective_threshold = if auto_sigma {
            let mut abs_norm: Vec<f64> = Vec::new();
            for (i, res) in fit.residuals.iter().enumerate() {
                if !fit.included[i] {
                    continue;
                }
                let w = fit.observations[i].weights();
                for (r, wi) in res.iter().zip(w.iter()) {
                    abs_norm.push((r * r * wi).sqrt().abs());
                }
            }
            abs_norm.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            // For zero-mean data X ~ N(0, k), the MAD equals
            // median(|X|) = k * 0.6745, so 1.4826 * median(|X|)
            // recovers k.  We work with |r/sigma| which are already
            // the absolute normalized residuals.
            // Floor at 1.0 so we never tighten beyond the
            // user-specified threshold.
            let robust_sigma = if abs_norm.is_empty() {
                1.0
            } else {
                let median_abs = abs_norm[abs_norm.len() / 2];
                (1.4826 * median_abs).max(1.0)
            };
            chi2_threshold * robust_sigma * robust_sigma
        } else {
            chi2_threshold
        };

        // Compute per-observation chi^2 for all included observations,
        // then reject the worst first (largest chi^2) up to budget.
        let mut obs_chi2: Vec<(usize, f64)> = fit
            .residuals
            .iter()
            .enumerate()
            .filter(|&(i, _)| fit.included[i])
            .map(|(i, res)| {
                let w = fit.observations[i].weights();
                let chi2: f64 = res.iter().zip(w.iter()).map(|(r, wi)| r * r * wi).sum();
                (i, chi2)
            })
            .filter(|&(_, chi2)| chi2 > effective_threshold)
            .collect();
        obs_chi2.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let budget = n_included - min_included;
        let rejected_any = !obs_chi2.is_empty();
        for &(i, _) in obs_chi2.iter().take(budget) {
            fit.included[i] = false;
        }

        if !rejected_any {
            break;
        }

        fit = iterate_to_convergence(
            &fit.uncertain_state.state,
            sorted_obs,
            &fit.included,
            include_asteroids,
            fit.uncertain_state.non_grav.clone(),
            max_iter,
            tol,
        )?;
    }

    // When auto_sigma is enabled, rescale the covariance by the reduced
    // chi-squared of the included observations.  This inflates the
    // covariance to reflect the actual data scatter when the stated
    // sigmas are incorrect (a posteriori variance scaling).
    if auto_sigma {
        let n_included = fit.included.iter().filter(|&&inc| inc).count();
        let n_params = 6 + fit
            .uncertain_state
            .non_grav
            .as_ref()
            .map_or(0, NonGravModel::n_free_params);
        let n_measurements: usize = fit
            .residuals
            .iter()
            .zip(fit.included.iter())
            .filter(|&(_, &inc)| inc)
            .map(|(r, _)| r.len())
            .sum();
        let dof = n_measurements.saturating_sub(n_params);
        if dof > 0 && n_included > n_params {
            let chi2_total: f64 = fit
                .residuals
                .iter()
                .enumerate()
                .filter(|&(i, _)| fit.included[i])
                .map(|(i, res)| {
                    let w = fit.observations[i].weights();
                    res.iter()
                        .zip(w.iter())
                        .map(|(r, wi)| r * r * wi)
                        .sum::<f64>()
                })
                .sum();
            let chi2_reduced = chi2_total / dof as f64;
            // Only inflate, never shrink -- if chi2_reduced < 1 the
            // stated sigmas are already conservative.
            if chi2_reduced > 1.0 {
                fit.uncertain_state.cov_matrix *= chi2_reduced;
            }
        }
    }

    Ok(fit)
}

/// Return observations sorted by epoch (ascending).
fn sort_by_epoch(obs: &[Observation]) -> Vec<Observation> {
    let mut sorted = obs.to_vec();
    sorted.sort_by(|a, b| {
        a.epoch()
            .jd
            .partial_cmp(&b.epoch().jd)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    sorted
}

/// Number of free non-grav parameters (0 when `None`).
fn n_nongrav_params(ng: Option<&NonGravModel>) -> usize {
    ng.map_or(0, NonGravModel::n_free_params)
}

/// Run the iterative convergence loop with adaptive Levenberg-Marquardt
/// damping and step-size limiting.
///
/// Each iteration re-linearizes at the current state, solves the damped
/// normal equations `(N + lambda * diag(N)) dx = b`, limits the step magnitude,
/// and moves forward unconditionally.  Re-linearizing every iteration is
/// essential: the step limiter caps *magnitude* but not *direction*, so
/// recycling a stale Jacobian would repeatedly propose the same capped
/// step and stall.
///
/// Lambda is adjusted heuristically: decreased when chi-squared improves,
/// increased when it worsens.  This steers the solver between
/// Gauss-Newton (fast near the solution) and steepest descent (safe far
/// from it).
///
/// Unlike a naive "apply and hope" loop, this uses proper LM step
/// acceptance: a trial correction is only accepted when chi^2 improves.
/// On rejection the solver increases lambda and re-solves from the same
/// linearization point (no repropagation).  This guarantees that
/// `state_epoch` is always the best state seen.
fn iterate_to_convergence(
    initial_state: &State<Equatorial>,
    obs: &[Observation],
    included: &[bool],
    include_asteroids: bool,
    mut non_grav: Option<NonGravModel>,
    max_iter: usize,
    tol: f64,
) -> KeteResult<OrbitFit> {
    let mut state_epoch = initial_state.clone();
    // Start with non-zero damping when fitting non-grav parameters.
    // Their information-matrix entries are often orders of magnitude
    // smaller than the orbital entries, so an undamped first step can
    // produce enormous non-grav corrections that poison the fit.
    let np = n_nongrav_params(non_grav.as_ref());
    let mut lambda = if np > 0 { 1e-4 } else { 0.0 };

    // Linearize at the initial state.
    let Ok((mut info_mat, mut rhs_vec, mut chi2)) = accumulate_normal_equations(
        &state_epoch,
        obs,
        included,
        include_asteroids,
        non_grav.as_ref(),
    ) else {
        // Can't even linearize the initial state -- return it as-is.
        return Ok(make_non_converged_result(
            &state_epoch,
            obs,
            included,
            include_asteroids,
            non_grav,
        ));
    };

    for _ in 0..max_iter {
        let dx = solve_damped(&info_mat, &rhs_vec, lambda)?;
        let dx = limit_correction(dx);

        let converged = dx.norm() < tol;

        // Build trial state.
        let mut trial_state = state_epoch.clone();
        let mut trial_ng = non_grav.clone();
        apply_correction(&mut trial_state, &dx, &mut trial_ng);

        // Reject unphysical trial states without repropagating.
        let r = trial_state.pos.norm();
        let v = trial_state.vel.norm();
        if !r.is_finite() || !v.is_finite() || !(1e-4..=1e4).contains(&r) {
            lambda = if lambda < 1e-6 { 1.0 } else { lambda * 10.0 };
            if lambda > 1e12 {
                break;
            }
            continue;
        }

        // Linearize at the trial state.
        let trial_sweep = stm_sweep(
            &trial_state,
            obs,
            included,
            include_asteroids,
            trial_ng.as_ref(),
        );

        if let Ok(sweep) = trial_sweep {
            let (new_info, new_rhs, new_chi2, sweep_residuals) =
                accumulate_from_sweep(&sweep, trial_ng.as_ref());
            if new_chi2 <= chi2 {
                // Accept step: chi^2 improved (or stayed equal).
                state_epoch = trial_state;
                non_grav = trial_ng;
                info_mat = new_info;
                rhs_vec = new_rhs;
                chi2 = new_chi2;
                lambda *= 0.1;

                if converged {
                    let covariance = svd_pseudo_inverse(&info_mat, 1e-14)?;

                    // Build per-obs residuals from the sweep (included
                    // observations only have meaningful values; excluded
                    // get NaN since they are never used downstream).
                    let mut residuals = Vec::with_capacity(obs.len());
                    let mut sweep_idx = 0;
                    for (i, observation) in obs.iter().enumerate() {
                        if included[i] {
                            residuals.push(sweep_residuals[sweep_idx].clone());
                            sweep_idx += 1;
                        } else {
                            residuals.push(DVector::from_element(
                                observation.measurement_dim(),
                                f64::NAN,
                            ));
                        }
                    }

                    let n_params = 6 + n_nongrav_params(non_grav.as_ref());
                    let rms = weighted_rms(&residuals, obs, included, n_params);
                    let uncertain_state = UncertainState::new(state_epoch, covariance, non_grav)?;
                    return Ok(OrbitFit {
                        uncertain_state,
                        residuals,
                        observations: obs.to_vec(),
                        included: included.to_vec(),
                        rms,
                        converged: true,
                    });
                }
            } else {
                // Reject step: increase damping and re-solve from
                // the same linearization point.
                lambda = if lambda < 1e-6 { 1.0 } else { lambda * 10.0 };
                if lambda > 1e12 {
                    break;
                }
            }
        } else {
            // Propagation failed at trial state -- reject and damp.
            lambda = if lambda < 1e-6 { 1.0 } else { lambda * 10.0 };
            if lambda > 1e12 {
                break;
            }
        }
    }

    // Did not converge -- return the best accepted state.
    Ok(make_non_converged_result(
        &state_epoch,
        obs,
        included,
        include_asteroids,
        non_grav,
    ))
}

/// Build an `OrbitFit` with `converged: false` for the given state.
///
/// Propagation may fail for the current state (e.g. the initial guess
/// cannot reach all observation epochs).  In that case we return a
/// zeroed covariance and NaN residuals so that the caller still gets a
/// valid `OrbitFit` instead of a hard error.
fn make_non_converged_result(
    state: &State<Equatorial>,
    obs: &[Observation],
    included: &[bool],
    include_asteroids: bool,
    non_grav: Option<NonGravModel>,
) -> OrbitFit {
    let n_params = 6 + n_nongrav_params(non_grav.as_ref());

    // Try to compute residuals and covariance; fall back to placeholders
    // if propagation fails.
    let (covariance, residuals, rms) = if let Ok((info_mat, _, _)) =
        accumulate_normal_equations(state, obs, included, include_asteroids, non_grav.as_ref())
    {
        let cov = svd_pseudo_inverse(&info_mat, 1e-14)
            .unwrap_or_else(|_| DMatrix::zeros(n_params, n_params));
        if let Ok(res) = compute_residuals(state, obs, include_asteroids, non_grav.as_ref()) {
            let r = weighted_rms(&res, obs, included, n_params);
            (cov, res, r)
        } else {
            let nan_res: Vec<DVector<f64>> = obs
                .iter()
                .map(|o| DVector::from_element(o.weights().len(), f64::NAN))
                .collect();
            (cov, nan_res, f64::INFINITY)
        }
    } else {
        let nan_res: Vec<DVector<f64>> = obs
            .iter()
            .map(|o| DVector::from_element(o.weights().len(), f64::NAN))
            .collect();
        (DMatrix::zeros(n_params, n_params), nan_res, f64::INFINITY)
    };

    // Construct UncertainState; cannot fail here because we control
    // the covariance dimensions.
    let uncertain_state =
        UncertainState::new(state.clone(), covariance, non_grav).unwrap_or_else(|_| {
            UncertainState {
                state: state.clone(),
                cov_matrix: DMatrix::zeros(n_params, n_params),
                non_grav: None,
            }
        });

    OrbitFit {
        uncertain_state,
        residuals,
        observations: obs.to_vec(),
        included: included.to_vec(),
        rms,
        converged: false,
    }
}

/// Result of the STM sweep at a single observation epoch.
///
/// Contains the cumulative state transition matrix, the observation
/// residual, and the local geometric Jacobian needed to build either
/// normal equations (batch least squares) or log-posterior gradients
/// (MCMC sampling).
#[derive(Debug, Clone)]
pub struct StmObs {
    /// Cumulative STM from the reference epoch to this observation, 6 x D.
    pub phi_cum: DMatrix<f64>,
    /// Observation residual (observed - computed), m-vector.
    pub residual: DVector<f64>,
    /// Local geometric partial derivatives, m x 6.
    pub h_local: DMatrix<f64>,
    /// Weight vector (1/sigma^2 per measurement component), m-vector.
    pub weights: DVector<f64>,
}

/// Propagate the epoch state through observations in time order, computing
/// the chained STM, residuals, and local Jacobians at each included
/// observation.
///
/// Excluded observations (where `included[i]` is `false`) are still
/// propagated through so the STM chain remains valid, but no `StmObs`
/// entry is emitted for them.
///
/// The returned vector contains one `StmObs` per *included* observation
/// (not per input observation), in time-sorted order.
///
/// # Errors
/// Returns an error if propagation or observation evaluation fails.
///
/// # Panics
/// Panics if the observer state position has zero norm.
pub fn stm_sweep(
    state_epoch: &State<Equatorial>,
    obs: &[Observation],
    included: &[bool],
    include_asteroids: bool,
    non_grav: Option<&NonGravModel>,
) -> KeteResult<Vec<StmObs>> {
    debug_assert!(
        obs.windows(2).all(|w| w[0].epoch().jd <= w[1].epoch().jd),
        "stm_sweep: observations must be sorted by epoch"
    );
    let np = n_nongrav_params(non_grav);
    let d = 6 + np;

    // Cumulative STM: 6 x D, initialized to [I_6 | 0_{6 x Np}].
    let mut phi_cum = DMatrix::<f64>::zeros(6, d);
    for i in 0..6 {
        phi_cum[(i, i)] = 1.0;
    }

    let mut state_cur = state_epoch.clone();
    let mut results = Vec::new();

    for (i, observation) in obs.iter().enumerate() {
        let obs_epoch = observation.epoch();

        // Propagate from current state to observation epoch via STM.
        if (obs_epoch.jd - state_cur.epoch.jd).abs() > 1e-12 {
            let (new_state, phi_k) = compute_state_transition(
                &state_cur,
                obs_epoch,
                include_asteroids,
                non_grav.cloned(),
            )?;

            // phi_k is 6 x (6 + Np).
            // phi_state is the 6 x 6 state block.
            let phi_state = phi_k.columns(0, 6).clone_owned();

            // Chain the state block: Phi_cum[:, 0:6] = Phi_state * Phi_cum[:, 0:6]
            let new_state_cols = &phi_state * phi_cum.columns(0, 6);
            phi_cum.columns_mut(0, 6).copy_from(&new_state_cols);

            // Chain the parameter block (if any):
            // Phi_cum[:, 6:] = Phi_state * Phi_cum[:, 6:] + Phi_param
            if np > 0 {
                // phi_param is the 6 x Np parameter sensitivity block.
                let phi_param = phi_k.columns(6, np).clone_owned();
                let new_param_cols = &phi_state * phi_cum.columns(6, np) + &phi_param;
                phi_cum.columns_mut(6, np).copy_from(&new_param_cols);
            }

            state_cur = new_state;
        }

        // Skip excluded observations, but still propagate through them
        // so the STM chain stays correct.
        if !included[i] {
            continue;
        }

        // Apply two-body light-time correction once;
        // use the corrected state for both residual and partials.
        let obs_state = observation.observer();
        let obj_lt = light_time_correct(&state_cur, &obs_state.pos)?;

        let residual = observation.residual_from_corrected(&obj_lt);

        // Local geometric partials (m x 6).
        let h_local = observation.partials(&obj_lt);

        // Weight vector.
        let weights = observation.weights();

        results.push(StmObs {
            phi_cum: phi_cum.clone(),
            residual,
            h_local,
            weights,
        });
    }

    Ok(results)
}

/// Accumulate the weighted normal equations for one linearization pass.
///
/// Returns `(info_mat, rhs_vec, chi2)` where `info_mat` is the
/// (6+Np) x (6+Np) information matrix, `rhs_vec` is the right-hand
/// side, and `chi2` is the current weighted sum of squared residuals.
///
/// # Errors
/// Returns an error if the underlying STM sweep fails.
pub fn accumulate_normal_equations(
    state_epoch: &State<Equatorial>,
    obs: &[Observation],
    included: &[bool],
    include_asteroids: bool,
    non_grav: Option<&NonGravModel>,
) -> KeteResult<(DMatrix<f64>, DVector<f64>, f64)> {
    let sweep = stm_sweep(state_epoch, obs, included, include_asteroids, non_grav)?;
    let (n_mat, b_vec, chi2, _) = accumulate_from_sweep(&sweep, non_grav);
    Ok((n_mat, b_vec, chi2))
}

/// Accumulate normal equations from a pre-computed STM sweep.
///
/// Returns `(info_mat, rhs_vec, chi2, residuals)` where `residuals`
/// contains one entry per included observation (matching the sweep).
fn accumulate_from_sweep(
    sweep: &[StmObs],
    non_grav: Option<&NonGravModel>,
) -> (DMatrix<f64>, DVector<f64>, f64, Vec<DVector<f64>>) {
    let np = n_nongrav_params(non_grav);
    let d = 6 + np;

    let mut n_mat = DMatrix::<f64>::zeros(d, d);
    let mut b_vec = DVector::<f64>::zeros(d);
    let mut chi2 = 0.0;
    let mut residuals = Vec::with_capacity(sweep.len());

    for entry in sweep {
        let m = entry.residual.len();

        // Map to epoch: H_epoch = H_local * Phi_cum  (m x D).
        let h_epoch = &entry.h_local * &entry.phi_cum;

        // Accumulate chi-squared.
        for k in 0..m {
            chi2 += entry.residual[k] * entry.residual[k] * entry.weights[k];
        }

        residuals.push(entry.residual.clone());

        // Accumulate normal matrix and RHS via weighted outer products:
        //   N += H^T W H,  b += H^T W r
        // Build sqrt(W) * H and sqrt(W) * r for efficient rank-m update.
        // hw is m x d, wr is m x 1.
        let mut hw = h_epoch.clone();
        let mut wr = entry.residual.clone();
        for k in 0..m {
            let sw = entry.weights[k].sqrt();
            for j in 0..d {
                hw[(k, j)] *= sw;
            }
            wr[k] *= sw;
        }
        // N += (sqrt(W) H)^T (sqrt(W) H)  =  H^T W H
        n_mat += hw.transpose() * &hw;
        // b += (sqrt(W) H)^T (sqrt(W) r)  =  H^T W r
        b_vec += hw.transpose() * &wr;
    }

    (n_mat, b_vec, chi2, residuals)
}

/// Solve `(N + lambda * diag(N)) * dx = b` via SVD.
///
/// When `lambda > 0` the diagonal of N is augmented, pulling the solution
/// toward a steepest-descent step and stabilising poorly-constrained
/// directions.
fn solve_damped(
    n_mat: &DMatrix<f64>,
    b_vec: &DVector<f64>,
    lambda: f64,
) -> KeteResult<DVector<f64>> {
    let mut n_work = n_mat.clone();
    if lambda > 0.0 {
        for i in 0..n_work.nrows() {
            n_work[(i, i)] += lambda * n_mat[(i, i)].abs().max(1e-15);
        }
    }
    let svd = n_work.svd(true, true);
    svd.solve(b_vec, 1e-14)
        .map_err(|_| Error::ValueError("SVD solve failed on damped normal matrix".into()))
}

/// Cap position and velocity corrections to prevent wild jumps.
///
/// Position is limited to 0.5 AU and velocity to 0.005 AU/day per
/// iteration.  Non-grav parameters are not capped here -- the
/// Levenberg-Marquardt damping (especially the non-zero initial lambda
/// set when `np > 0`) regulates them instead, since their typical
/// magnitudes span many orders of magnitude (1e-12 to 1e-1).
fn limit_correction(mut dx: DVector<f64>) -> DVector<f64> {
    // AU
    const MAX_POS: f64 = 0.5;
    // AU/day
    const MAX_VEL: f64 = 0.005;

    let pos_norm = (dx[0] * dx[0] + dx[1] * dx[1] + dx[2] * dx[2]).sqrt();
    if pos_norm > MAX_POS {
        let s = MAX_POS / pos_norm;
        for v in dx.rows_mut(0, 3).iter_mut() {
            *v *= s;
        }
    }

    let vel_norm = (dx[3] * dx[3] + dx[4] * dx[4] + dx[5] * dx[5]).sqrt();
    if vel_norm > MAX_VEL {
        let s = MAX_VEL / vel_norm;
        for v in dx.rows_mut(3, 3).iter_mut() {
            *v *= s;
        }
    }

    dx
}

/// Apply a state correction vector to the epoch state and (optionally)
/// non-grav parameters.
fn apply_correction(
    state: &mut State<Equatorial>,
    dx: &DVector<f64>,
    non_grav: &mut Option<NonGravModel>,
) {
    let pos: [f64; 3] = state.pos.into();
    state.pos = [pos[0] + dx[0], pos[1] + dx[1], pos[2] + dx[2]].into();

    let vel: [f64; 3] = state.vel.into();
    state.vel = [vel[0] + dx[3], vel[1] + dx[4], vel[2] + dx[5]].into();

    // Apply non-grav parameter corrections from dx[6..].
    if let Some(ng) = non_grav.as_mut() {
        let np = ng.n_free_params();
        let mut params = ng.get_free_params();
        for k in 0..np {
            params[k] += dx[6 + k];
        }
        ng.set_free_params(&params);
    }
}

/// SVD-based pseudo-inverse, robust to near-singular matrices.
///
/// Singular values below `eps * sigma_max` are treated as zero.
fn svd_pseudo_inverse(mat: &DMatrix<f64>, eps: f64) -> KeteResult<DMatrix<f64>> {
    let svd = mat.clone().svd(true, true);
    let sigma_max = svd.singular_values.max();
    let thr = eps * sigma_max;
    let u = svd
        .u
        .as_ref()
        .ok_or_else(|| Error::ValueError("SVD failed (no U)".into()))?;
    let vt = svd
        .v_t
        .as_ref()
        .ok_or_else(|| Error::ValueError("SVD failed (no V^T)".into()))?;
    let n = svd.singular_values.len();
    let mut s_inv = DMatrix::<f64>::zeros(n, n);
    for i in 0..n {
        let si = svd.singular_values[i];
        if si > thr {
            s_inv[(i, i)] = 1.0 / si;
        }
    }
    Ok(vt.transpose() * s_inv * u.transpose())
}

/// Compute post-fit residuals for all observations (time-sorted order).
///
/// Uses the 6-dim `propagate_n_body_spk` (not the 60-dim STM
/// integrator) so this is ~5x cheaper than an STM sweep.
fn compute_residuals(
    state_epoch: &State<Equatorial>,
    obs: &[Observation],
    include_asteroids: bool,
    non_grav: Option<&NonGravModel>,
) -> KeteResult<Vec<DVector<f64>>> {
    let mut residuals = Vec::with_capacity(obs.len());
    let mut state_cur = state_epoch.clone();

    for observation in obs {
        let obs_epoch = observation.epoch();

        // Propagate to observation epoch (6-dim, no STM).
        if (obs_epoch.jd - state_cur.epoch.jd).abs() > 1e-12 {
            state_cur =
                propagate_n_body_spk(state_cur, obs_epoch, include_asteroids, non_grav.cloned())?;
        }

        let obs_state = observation.observer();
        let obj_lt = light_time_correct(&state_cur, &obs_state.pos)?;
        let res = observation.residual_from_corrected(&obj_lt);
        residuals.push(res);
    }

    Ok(residuals)
}

/// Compute reduced weighted RMS of residuals for included observations.
///
/// Uses degrees of freedom (`n_measurements` - `n_params`) as divisor so the
/// value is comparable regardless of the number of observations.
fn weighted_rms(
    residuals: &[DVector<f64>],
    obs: &[Observation],
    included: &[bool],
    n_params: usize,
) -> f64 {
    let mut sum = 0.0;
    let mut n_meas: usize = 0;
    for (i, res) in residuals.iter().enumerate() {
        if !included[i] {
            continue;
        }
        let w = obs[i].weights();
        for (r, wi) in res.iter().zip(w.iter()) {
            sum += r * r * wi;
            n_meas += 1;
        }
    }
    let dof = n_meas.saturating_sub(n_params);
    if dof > 0 {
        (sum / dof as f64).sqrt()
    } else if n_meas > 0 {
        (sum / n_meas as f64).sqrt()
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kete_core::constants::GMS;
    use kete_core::desigs::Desig;
    use kete_core::propagation::propagate_n_body_spk;
    use kete_core::time::{TDB, Time};

    /// Helper: build a simple state.
    fn make_state(pos: [f64; 3], vel: [f64; 3], jd: f64) -> State<Equatorial> {
        State::new(Desig::Empty, jd.into(), pos.into(), vel.into(), 0)
    }

    /// Generate synthetic optical observations, optionally with a non-grav model.
    ///
    /// Uses the full N-body SPK propagator so that the physics model is
    /// consistent with the batch least-squares solver.
    fn synth_observations(
        true_state: &State<Equatorial>,
        epochs: &[f64],
        observer_pos_fn: impl Fn(f64) -> ([f64; 3], [f64; 3]),
        sigma: f64,
        non_grav: Option<&NonGravModel>,
    ) -> Vec<Observation> {
        let mut observations = Vec::new();
        for &jd in epochs {
            let (obs_pos, obs_vel) = observer_pos_fn(jd);
            let observer = make_state(obs_pos, obs_vel, jd);

            let obj_at = propagate_n_body_spk(
                true_state.clone(),
                Time::<TDB>::new(jd),
                false,
                non_grav.cloned(),
            )
            .unwrap();

            let obj_lt = light_time_correct(&obj_at, &observer.pos).unwrap();
            let (ra, dec) = (obj_lt.pos - observer.pos).to_ra_dec();

            observations.push(Observation::Optical {
                observer,
                ra,
                dec,
                sigma_ra: sigma,
                sigma_dec: sigma,
            });
        }
        observations
    }

    /// Earth-like observer on a circular orbit at 1 AU with slight inclination.
    fn earth_observer(jd: f64) -> ([f64; 3], [f64; 3]) {
        // ~0.0172 AU/day
        let v_earth = (GMS / 1.0_f64).sqrt();
        let period = 2.0 * std::f64::consts::PI / v_earth;
        let t = (jd - 2460000.5) / period * 2.0 * std::f64::consts::PI;
        let incl: f64 = 0.05;
        let pos = [t.cos(), t.sin() * incl.cos(), t.sin() * incl.sin()];
        let vel = [
            -v_earth * t.sin(),
            v_earth * t.cos() * incl.cos(),
            v_earth * t.cos() * incl.sin(),
        ];
        (pos, vel)
    }

    #[test]
    fn test_fit_orbit_two_body() {
        // True orbit: circular at 1.5 AU.
        let r = 1.5;
        let v = (GMS / r).sqrt();
        let true_state = make_state([r, 0.0, 0.0], [0.0, v, 0.0], 2460000.5);

        // Generate 10 observations over 60 days.
        let epochs: Vec<f64> = (0..10).map(|i| 2460000.5 + f64::from(i) * 6.0).collect();
        // ~0.2 arcsec
        let sigma = 1e-6;
        let observations = synth_observations(&true_state, &epochs, earth_observer, sigma, None);

        // Perturbed initial state (5% error in position, 3% in velocity).
        let perturbed = make_state([r * 1.05, 0.0, 0.0], [0.0, v * 0.97, 0.0], 2460000.5);

        let fit = fit_orbit(
            &perturbed,
            &observations,
            false,
            None,
            20,
            1e-8,
            9.0,
            0,
            false,
        )
        .unwrap();

        // Check that the fit converged near the true state.
        let pos_err = (fit.uncertain_state.state.pos - true_state.pos).norm();
        let vel_err = (fit.uncertain_state.state.vel - true_state.vel).norm();

        // Should recover position to < 1e-4 AU and velocity to < 1e-5 AU/day.
        assert!(pos_err < 1e-4, "Position error {pos_err:.6e} too large");
        assert!(vel_err < 1e-5, "Velocity error {vel_err:.6e} too large");

        // RMS should be very small (near-perfect synthetic data).
        assert!(fit.rms < 1e-3, "Weighted RMS {:.6e} too large", fit.rms);

        // Covariance should be positive definite (check diagonal > 0).
        for i in 0..6 {
            assert!(
                fit.uncertain_state.cov_matrix[(i, i)] > 0.0,
                "Covariance diagonal [{i},{i}] = {} not positive",
                fit.uncertain_state.cov_matrix[(i, i)]
            );
        }
    }

    #[test]
    fn test_fit_orbit_elliptical() {
        // Moderately eccentric orbit: a = 2.0, r_peri = 1.4, e ~ 0.3.
        let a = 2.0;
        let r_peri = 1.4;
        let v_peri = (GMS * (2.0 / r_peri - 1.0 / a)).sqrt();
        let true_state = make_state([r_peri, 0.0, 0.0], [0.0, v_peri, 0.0], 2460000.5);

        // 8 observations over 40 days.
        let epochs: Vec<f64> = (0..8).map(|i| 2460000.5 + f64::from(i) * 5.0).collect();
        let sigma = 1e-6;
        let observations = synth_observations(&true_state, &epochs, earth_observer, sigma, None);

        // Perturbed initial state.
        let perturbed = make_state(
            [r_peri * 1.03, 0.0, 0.005],
            [0.0, v_peri * 0.98, 0.0],
            2460000.5,
        );

        let fit = fit_orbit(
            &perturbed,
            &observations,
            false,
            None,
            20,
            1e-8,
            9.0,
            0,
            false,
        )
        .unwrap();

        let pos_err = (fit.uncertain_state.state.pos - true_state.pos).norm();

        assert!(
            pos_err < 1e-3,
            "Position error {pos_err:.6e} too large for elliptical orbit"
        );
    }

    #[test]
    fn test_outlier_rejection() {
        // True orbit: circular at 1.5 AU.
        let r = 1.5;
        let v = (GMS / r).sqrt();
        let true_state = make_state([r, 0.0, 0.0], [0.0, v, 0.0], 2460000.5);

        let epochs: Vec<f64> = (0..10).map(|i| 2460000.5 + f64::from(i) * 6.0).collect();
        let sigma = 1e-6;
        let mut observations =
            synth_observations(&true_state, &epochs, earth_observer, sigma, None);

        // Corrupt observation 3 with a large offset (100x sigma).
        if let Observation::Optical { ref mut ra, .. } = observations[3] {
            *ra += 100.0 * sigma;
        }

        let fit = fit_orbit(
            // Start from true state to ensure convergence.
            &true_state,
            &observations,
            false,
            None,
            20,
            1e-8,
            9.0,
            3,
            false,
        )
        .unwrap();

        // At least one observation should have been rejected.
        let n_total = 10;
        let n_included = fit.included.iter().filter(|&&v| v).count();
        let n_rejected = n_total - n_included;
        assert!(
            n_rejected >= 1,
            "Expected at least 1 rejection, got {n_rejected}"
        );
    }

    #[test]
    fn test_nongrav_jpl_comet_fitting() {
        // Circular orbit at 1.5 AU with a tangential non-grav force (a2).
        let r = 1.5;
        let v = (GMS / r).sqrt();
        let true_state = make_state([r, 0.0, 0.0], [0.0, v, 0.0], 2460000.5);
        // AU/day^2, large enough to be detectable
        let true_a2 = 1e-8;
        let true_ng = NonGravModel::new_jpl_comet_default(0.0, true_a2, 0.0);

        // Generate 15 observations over 90 days with the non-grav model.
        let epochs: Vec<f64> = (0..15).map(|i| 2460000.5 + f64::from(i) * 6.0).collect();
        // Tight observations.
        let sigma = 1e-7;
        let observations =
            synth_observations(&true_state, &epochs, earth_observer, sigma, Some(&true_ng));

        // Start from true state + non-grav model with a2=0 and fit.
        let init_ng = NonGravModel::new_jpl_comet_default(0.0, 0.0, 0.0);

        let fit = fit_orbit(
            &true_state,
            &observations,
            false,
            Some(&init_ng),
            30,
            1e-10,
            9.0,
            0,
            false,
        )
        .unwrap();

        // The fitted non-grav model should exist and have a2 close to true_a2.
        let fitted_ng = fit
            .uncertain_state
            .non_grav
            .as_ref()
            .expect("non_grav should be present");
        let fitted_params = fitted_ng.get_free_params();
        let a2_err = (fitted_params[1] - true_a2).abs();
        assert!(
            a2_err < true_a2 * 0.1,
            "a2 error {a2_err:.6e} too large (true={true_a2:.6e}, fitted={:.6e})",
            fitted_params[1]
        );

        // Covariance should be 9x9.
        assert_eq!(
            fit.uncertain_state.cov_matrix.nrows(),
            9,
            "Expected 9x9 covariance"
        );
        assert_eq!(
            fit.uncertain_state.cov_matrix.ncols(),
            9,
            "Expected 9x9 covariance"
        );

        // RMS should be small.
        assert!(fit.rms < 1e-3, "Weighted RMS {:.6e} too large", fit.rms);
    }

    #[test]
    fn test_nongrav_dust_fitting() {
        // Object at 1.2 AU with dust model (beta).
        let r = 1.2;
        let v = (GMS / r).sqrt();
        let true_state = make_state([r, 0.0, 0.0], [0.0, v, 0.0], 2460000.5);
        let true_beta = 0.001;
        let true_ng = NonGravModel::new_dust(true_beta);

        // 15 observations over 90 days.
        let epochs: Vec<f64> = (0..15).map(|i| 2460000.5 + f64::from(i) * 6.0).collect();
        let sigma = 1e-7;
        let observations =
            synth_observations(&true_state, &epochs, earth_observer, sigma, Some(&true_ng));

        // Start from true state with beta=0.
        let init_ng = NonGravModel::new_dust(0.0);

        let fit = fit_orbit(
            &true_state,
            &observations,
            false,
            Some(&init_ng),
            30,
            1e-10,
            9.0,
            0,
            false,
        )
        .unwrap();

        let fitted_ng = fit
            .uncertain_state
            .non_grav
            .as_ref()
            .expect("non_grav should be present");
        let fitted_params = fitted_ng.get_free_params();
        let beta_err = (fitted_params[0] - true_beta).abs();
        assert!(
            beta_err < true_beta * 0.1,
            "beta error {beta_err:.6e} too large (true={true_beta:.6e}, fitted={:.6e})",
            fitted_params[0]
        );

        // Covariance should be 7x7.
        assert_eq!(
            fit.uncertain_state.cov_matrix.nrows(),
            7,
            "Expected 7x7 covariance"
        );
        assert_eq!(
            fit.uncertain_state.cov_matrix.ncols(),
            7,
            "Expected 7x7 covariance"
        );

        assert!(fit.rms < 1e-3, "Weighted RMS {:.6e} too large", fit.rms);
    }

    #[test]
    fn test_gradual_fit_long_arc() {
        // 2-year arc with a perturbed initial state.
        // The gradual fitting should converge where a direct full-arc
        // fit from the same initial guess would struggle.
        let r = 2.0;
        let v = (GMS / r).sqrt();
        let true_state = make_state([r, 0.0, 0.0], [0.0, v, 0.0], 2460000.5);

        // 40 observations over ~720 days.
        let epochs: Vec<f64> = (0..40).map(|i| 2460000.5 + f64::from(i) * 18.0).collect();
        let sigma = 1e-6;
        let observations = synth_observations(&true_state, &epochs, earth_observer, sigma, None);

        // Perturb initial state by 10% position and 5% velocity.
        let perturbed = make_state([r * 1.10, 0.0, 0.0], [0.0, v * 0.95, 0.0], 2460000.5);

        let fit = fit_orbit(
            &perturbed,
            &observations,
            false,
            None,
            50,
            1e-8,
            9.0,
            3,
            false,
        )
        .unwrap();

        let pos_err = (fit.uncertain_state.state.pos - true_state.pos).norm();
        assert!(
            pos_err < 1e-3,
            "Gradual long-arc: pos error {pos_err:.6e} too large"
        );
        assert!(
            fit.converged,
            "Gradual long-arc should converge, rms={:.6e}",
            fit.rms
        );
    }

    #[test]
    fn test_gradual_fit_rejection_reinclusion() {
        // Verify that observations rejected in early windows are
        // re-evaluated in the final pass.
        let r = 1.8;
        let v = (GMS / r).sqrt();
        let true_state = make_state([r, 0.0, 0.0], [0.0, v, 0.0], 2460000.5);

        // 20 observations over 400 days.
        let epochs: Vec<f64> = (0..20).map(|i| 2460000.5 + f64::from(i) * 20.0).collect();
        let sigma = 1e-6;
        let mut observations =
            synth_observations(&true_state, &epochs, earth_observer, sigma, None);

        // Corrupt one observation near the end of the arc (beyond the
        // seed window).  With a bad initial guess this might look like an
        // outlier during early windows but should be correctly handled
        // in the final pass.
        if let Observation::Optical { ref mut ra, .. } = observations[18] {
            *ra += 50.0 * sigma;
        }

        let fit = fit_orbit(
            &true_state,
            &observations,
            false,
            None,
            50,
            1e-8,
            9.0,
            5,
            false,
        )
        .unwrap();

        // The corrupted observation should be rejected.
        let n_total = 20;
        let n_included = fit.included.iter().filter(|&&v| v).count();
        let n_rejected = n_total - n_included;
        assert!(
            n_rejected >= 1,
            "Expected at least 1 rejection, got {n_rejected}"
        );

        // Orbit should still be good.
        let pos_err = (fit.uncertain_state.state.pos - true_state.pos).norm();
        assert!(
            pos_err < 1e-3,
            "Rejection re-inclusion: pos error {pos_err:.6e} too large"
        );
    }
}
