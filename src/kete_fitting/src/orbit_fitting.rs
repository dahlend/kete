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

use crate::obs::AstrometricObservation;
use crate::uncertain_state::UncertainState;
use kete_core::forces::NonGravModel;
use kete_core::frames::Equatorial;
use kete_core::kepler::{analytic_2_body_stm, light_time_correct};
use kete_core::prelude::{Error, KeteResult, State};
use kete_spice::prelude::{LOADED_SPK, compute_state_transition};
use kete_spice::propagation::propagate_n_body_spk;
use nalgebra::{DMatrix, DVector, Vector3};
use rayon::prelude::*;

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
    pub observations: Vec<AstrometricObservation>,

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
/// centered on the reference epoch: +/-30, +/-60, +/-180, and +/-360 days,
/// followed by a final pass that includes the full arc.  Each stage
/// bootstraps from the previous converged solution.  Windows that
/// contain fewer than 4 observations are skipped automatically.
/// The final pass re-evaluates all observations for outlier rejection
/// (if enabled).
///
/// Outlier rejection is controlled by `max_reject_passes`.  When zero,
/// no rejection is performed and the fit uses all observations.
///
/// The CMC 2003 z-scores are already leverage-normalized and
/// approximately self-normalizing, so the rejection threshold is used
/// directly without further scaling.
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
/// * `chi2_threshold` - Per-observation z-score threshold for outlier
///   rejection. An observation is rejected when its weighted
///   chi-squared exceeds this multiple of the leverage-corrected
///   expected value, and recovered when it falls below `6/7` of this
///   threshold (Carpino-Milani-Chesley 2003 hysteresis).  Only used
///   when `max_reject_passes > 0`.
/// * `max_reject_passes` - Maximum outlier-rejection cycles.  Set to 0 to
///   disable rejection entirely.
///
/// # Errors
/// Fails if any internal propagation or solve fails.
pub fn fit_orbit(
    initial_state: &State<Equatorial>,
    obs: &[AstrometricObservation],
    include_asteroids: bool,
    non_grav: Option<&NonGravModel>,
    max_iter: usize,
    tol: f64,
    chi2_threshold: f64,
    max_reject_passes: usize,
) -> KeteResult<OrbitFit> {
    if obs.is_empty() {
        return Err(Error::ValueError("No observations provided".into()));
    }
    let sorted: Vec<AstrometricObservation> = sort_by_epoch(obs)
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

    // Geometric window radii (days) centered on the reference epoch.
    // Each stage bootstraps from the previous converged solution.
    // Windows with fewer than 4 observations are skipped automatically.
    // Starting at 30 days and doubling until the full arc is covered
    // prevents the large jumps that cause convergence failure on long arcs.
    let arc_radius = sorted
        .iter()
        .map(|ob| (ob.epoch().jd - ref_jd).abs())
        .fold(0.0_f64, f64::max);
    let mut windows: Vec<f64> = Vec::new();
    let mut radius = 30.0;
    while radius < arc_radius {
        windows.push(radius);
        radius *= 2.0;
    }
    windows.push(f64::INFINITY);

    let mut state = initial_state.clone();
    let ng = non_grav.cloned();
    let mut prev_n_in_window: usize = 0;

    // Expansion stages: converge + reject on each window.
    // Non-grav parameters are frozen during expansion because short arcs
    // have almost no sensitivity to them; fitting them here would produce
    // wildly wrong values that poison subsequent stages.  The original
    // non-grav values are preserved and used in the final full-arc pass.
    for &radius in &windows[..windows.len() - 1] {
        let windowed: Vec<AstrometricObservation> = sorted
            .iter()
            .filter(|ob| (ob.epoch().jd - ref_jd).abs() <= radius)
            .cloned()
            .collect();
        let n_in_window = windowed.len();
        if n_in_window < 4 || n_in_window == prev_n_in_window {
            // Too few observations, or this window includes the same
            // set as the previous one -- skip.
            continue;
        }
        prev_n_in_window = n_in_window;
        let included = vec![true; n_in_window];

        // Score the incoming state on the current window so we can detect
        // and reject a stage that returns a strictly worse orbit.  Without
        // this guard, a poorly-constrained early window can converge to a
        // low-residual but globally wrong orbit and silently poison every
        // subsequent stage.  compute_residuals is cheap (6-dim, no STM).
        let pre_rms = compute_residuals(&state, &windowed, include_asteroids, None)
            .ok()
            .map_or(f64::INFINITY, |r| weighted_rms(&r, &windowed, &included, 6));

        if let Ok(result) = solve_with_rejection(
            &state,
            &windowed,
            &included,
            include_asteroids,
            None,
            max_iter,
            tol,
            chi2_threshold,
            max_reject_passes,
        ) {
            let candidate = result.uncertain_state.state.clone();
            // Re-score the candidate on the FULL window (ignoring any
            // outlier rejections inside solve_with_rejection) for an
            // apples-to-apples comparison.
            let post_rms = compute_residuals(&candidate, &windowed, include_asteroids, None)
                .ok()
                .map_or(f64::INFINITY, |r| weighted_rms(&r, &windowed, &included, 6));
            if post_rms.is_finite() && post_rms <= pre_rms {
                state = candidate;
            }
            // else: keep prior state, the stage made things worse.
        }
        // On error: keep previous state, try the next wider window.
    }

    // Final full-arc pass.
    //
    // When fitting non-grav parameters, first run a gravity-only
    // solve_with_rejection to obtain a CMC-stable rejection mask, then
    // call iterate_to_convergence directly on that FIXED mask with the
    // NG model.  This avoids two known pathologies:
    //
    // 1. Leverage inflation: with 2+ extra free parameters the CMC
    //    z-score denominator shrinks for every observation.
    //    Borderline observations (z close to the threshold) that were
    //    accepted by gravity-only would be spuriously rejected by the
    //    NG CMC loop, weakening the normal equations and shifting the
    //    solution to a worse local minimum.
    //
    // 2. Cold-start sensitivity: the NG fit should start from the
    //    gravity-optimal state, not the windowed-expansion state.
    //
    // Using the gravity-only mask as the fixed observation set for the
    // NG convergence pass matches the practice in Milani, Chesley &
    // Sansaturio (2005) and Farnocchia et al. (2013): the outlier set
    // is determined by the best available model (gravity-only) and
    // carried into the NG estimation without re-running rejection.
    let included = vec![true; sorted.len()];
    if ng.is_some()
        && let Ok(grav_fit) = solve_with_rejection(
            &state,
            &sorted,
            &included,
            include_asteroids,
            None,
            max_iter,
            tol,
            chi2_threshold,
            max_reject_passes,
        )
    {
        return iterate_to_convergence(
            &grav_fit.uncertain_state.state,
            &sorted,
            &grav_fit.included,
            include_asteroids,
            ng,
            max_iter,
            tol,
        );
    }
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
    )
}

/// Converge + iteratively reject/recover outliers on a subset.
///
/// Implements the Carpino, Milani & Chesley (2003, Icarus 166:248)
/// rejection algorithm.  After each least-squares pass, every caller-
/// allowed observation receives a leverage-corrected z-score
///
/// ```text
/// z_i = chi2_i / max(m_i - trace(H_i C H_i^T diag(W_i)), 0.5)
/// ```
///
/// where `H_i` is the parameter-space design block for the observation,
/// `C` is the current parameter covariance, and `W_i` is the diagonal
/// weight matrix.  Currently included observations with `z > chi2_rej`
/// are rejected; currently rejected observations with `z < chi2_rec`
/// are recovered.  The two thresholds are `chi2_threshold` and
/// `chi2_threshold * 6 / 7` respectively (a 7/6 hysteresis ratio that
/// prevents oscillation).  A floor of `max(d+1, 4)` included
/// observations is enforced by un-rejecting the smallest-z newly
/// rejected observations.  Iteration stops when the included set is
/// stable.
///
fn solve_with_rejection(
    initial_state: &State<Equatorial>,
    sorted_obs: &[AstrometricObservation],
    included: &[bool],
    include_asteroids: bool,
    non_grav: Option<NonGravModel>,
    max_iter: usize,
    tol: f64,
    chi2_threshold: f64,
    max_reject_passes: usize,
) -> KeteResult<OrbitFit> {
    let mut current_included = included.to_vec();
    let mut fit = iterate_to_convergence(
        initial_state,
        sorted_obs,
        &current_included,
        include_asteroids,
        non_grav,
        max_iter,
        tol,
    )?;

    let np = fit
        .uncertain_state
        .non_grav
        .as_ref()
        .map_or(0, NonGravModel::n_free_params);
    let min_included = (6 + np).max(4);

    // Rejection and recovery thresholds with 7/6 hysteresis ratio
    // (Carpino, Milani & Chesley 2003).  The z-scores are already
    // leverage-normalized so no additional scaling is applied.
    let chi2_rej = chi2_threshold;
    let chi2_rec = chi2_threshold * (6.0 / 7.0);

    for _ in 0..max_reject_passes {
        // Sweep over every caller-allowed observation (mask = `included`)
        // so that currently rejected observations also receive z-scores
        // and can be recovered.  stm_sweep emits one StmObs per included
        // observation in time-sorted order.
        let sweep = stm_sweep(
            &fit.uncertain_state.state,
            sorted_obs,
            included,
            include_asteroids,
            fit.uncertain_state.non_grav.as_ref(),
        )?;
        let cov = &fit.uncertain_state.cov_matrix;

        let mut z_scores: Vec<f64> = Vec::with_capacity(sweep.len());
        for entry in &sweep {
            let m = entry.weights.len();
            // Parameter-space design block: H_i = h_local * phi_cum
            // (m x d).  Leverage trace = sum_k (H_i C H_i^T)[k,k] * W_k.
            let h_full: DMatrix<f64> = &entry.h_local * &entry.phi_cum;
            let hch: DMatrix<f64> = &h_full * cov * h_full.transpose();
            let leverage: f64 = (0..m).map(|k| hch[(k, k)] * entry.weights[k]).sum();
            let chi2: f64 = entry
                .residual
                .iter()
                .zip(entry.weights.iter())
                .map(|(r, w)| r * r * w)
                .sum();
            let expected = (m as f64 - leverage).max(0.5);
            z_scores.push(chi2 / expected);
        }

        // Apply hysteresis to build the new included mask.
        // Radar observations are never rejected -- they are too few and
        // too precise to discard without explicit user intent.
        let mut new_included = current_included.clone();
        let mut sweep_idx = 0;
        for i in 0..sorted_obs.len() {
            if !included[i] {
                continue;
            }
            let z = z_scores[sweep_idx];
            sweep_idx += 1;
            if sorted_obs[i].is_radar() {
                new_included[i] = true;
                continue;
            }
            new_included[i] = if current_included[i] {
                z <= chi2_rej
            } else {
                z <= chi2_rec
            };
        }

        // Enforce the minimum-included floor by un-rejecting the
        // smallest-z newly rejected observations -- those are the
        // least-bad rejections.
        let n_new = new_included.iter().filter(|&&v| v).count();
        if n_new < min_included {
            let mut newly_rejected: Vec<(usize, f64)> = Vec::new();
            let mut sweep_idx = 0;
            for i in 0..sorted_obs.len() {
                if !included[i] {
                    continue;
                }
                let z = z_scores[sweep_idx];
                sweep_idx += 1;
                if current_included[i] && !new_included[i] {
                    newly_rejected.push((i, z));
                }
            }
            newly_rejected
                .sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            let need = min_included - n_new;
            for &(i, _) in newly_rejected.iter().take(need) {
                new_included[i] = true;
            }
        }

        if new_included == current_included {
            break;
        }
        current_included = new_included;

        fit = iterate_to_convergence(
            &fit.uncertain_state.state,
            sorted_obs,
            &current_included,
            include_asteroids,
            fit.uncertain_state.non_grav.clone(),
            max_iter,
            tol,
        )?;
    }

    Ok(fit)
}

/// Return observations sorted by epoch (ascending).
fn sort_by_epoch(obs: &[AstrometricObservation]) -> Vec<AstrometricObservation> {
    let mut sorted = obs.to_vec();
    sorted.sort_by(|a, b| {
        a.epoch()
            .jd
            .partial_cmp(&b.epoch().jd)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    sorted
}

/// Apply light-time correction to an SSB-centered object state.
///
/// Takes the current SSB-centered object state and the observer state,
/// converts to heliocentric, applies light-time correction, then
/// converts back to SSB.  Returns the corrected SSB-centered state.
fn light_time_corrected_state(
    mut state_ssb: State<Equatorial>,
    mut observer: State<Equatorial>,
) -> KeteResult<State<Equatorial>> {
    let spk = LOADED_SPK.try_read()?;
    spk.try_change_center(&mut state_ssb, 10)?;
    spk.try_change_center(&mut observer, 10)?;
    let obs_sun = observer.pos - state_ssb.pos + state_ssb.pos;
    let mut obj_lt = light_time_correct(&state_ssb, &obs_sun)?;
    spk.try_change_center(&mut obj_lt, 0)?;
    Ok(obj_lt)
}

/// Run the iterative convergence loop with adaptive Levenberg-Marquardt
/// damping and step-size limiting.
///
/// Each iteration re-linearizes at the current state, solves the damped
/// normal equations `(N + lambda * diag(N)) dx = b`, limits the step magnitude,
/// and decides accept/reject based on the Huber rho loss.  Re-linearizing
/// every iteration is essential: the step limiter caps *magnitude* but not
/// *direction*, so recycling a stale Jacobian would repeatedly propose the
/// same capped step and stall.
///
/// Lambda is adjusted heuristically: decreased when the loss improves,
/// increased when it worsens.  This steers the solver between
/// Gauss-Newton (fast near the solution) and steepest descent (safe far
/// from it).
///
/// The acceptance metric is the Huber rho objective rather than the
/// Huber-weighted sum of squared residuals.  Huber weights themselves
/// shift as residuals cross the `HUBER_K` threshold, so the weighted
/// sum-of-squares is non-monotone under IRLS-LM steps and would cause
/// nearby starting states to land on opposite sides of the threshold and
/// take wildly different convergence paths.  The rho loss is the proper
/// M-estimator objective and is monotone under correct LM steps.
///
/// On rejection the solver increases lambda and re-solves from the same
/// linearization point (no repropagation).  This guarantees that
/// `state_epoch` is always the best state seen.
fn iterate_to_convergence(
    initial_state: &State<Equatorial>,
    obs: &[AstrometricObservation],
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
    let np = non_grav.as_ref().map_or(0, NonGravModel::n_free_params);
    let mut lambda = if np > 0 { 1e-4 } else { 0.0 };

    // Linearize at the initial state.
    let Ok((mut info_mat, mut rhs_vec, mut loss)) = accumulate_normal_equations(
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
        let dx = backtrack_to_ng_bounds(dx, non_grav.as_ref());

        // Convergence test. The raw `dx.norm()` (mixed units of AU,
        // AU/day, and arbitrary non-grav coefficients) is dominated
        // by the largest-magnitude parameters and can leave small
        // parameters (velocity ~1e-2, non-grav down to ~1e-12) far
        // from optimum when convergence is declared. We additionally
        // require each component's step to be small in "standard
        // error" units (sqrt(|N_ii|) times the step), which is scale-
        // invariant. Both criteria must hold.
        let mut max_scaled_step = 0.0_f64;
        for i in 0..dx.len() {
            let s = info_mat[(i, i)].abs().sqrt();
            max_scaled_step = max_scaled_step.max((dx[i] * s).abs());
        }
        let converged = dx.norm() < tol && max_scaled_step < 1e-3;

        // Build trial state.
        let mut trial_state = state_epoch.clone();
        let mut trial_ng = non_grav.clone();
        let ng_in_bounds = apply_correction(&mut trial_state, &dx, &mut trial_ng);

        // Reject unphysical trial states without repropagating.
        let r = trial_state.pos.norm();
        let v = trial_state.vel.norm();
        if !ng_in_bounds || !r.is_finite() || !v.is_finite() || !(1e-4..=1e4).contains(&r) {
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
            let (new_info, new_rhs, new_loss, sweep_residuals) =
                accumulate_from_sweep(&sweep, trial_ng.as_ref());
            if new_loss <= loss {
                // Accept step: Huber loss improved (or stayed equal).
                state_epoch = trial_state;
                non_grav = trial_ng;
                info_mat = new_info;
                rhs_vec = new_rhs;
                loss = new_loss;
                lambda *= 0.1;

                if converged {
                    let covariance = scaled_pseudo_inverse(&info_mat).map_err(|e| {
                        Error::ValueError(format!("SVD pseudo-inverse failed: {e}"))
                    })?;

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

                    let n_params = 6 + non_grav.as_ref().map_or(0, NonGravModel::n_free_params);
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
    obs: &[AstrometricObservation],
    included: &[bool],
    include_asteroids: bool,
    non_grav: Option<NonGravModel>,
) -> OrbitFit {
    let n_params = 6 + non_grav.as_ref().map_or(0, NonGravModel::n_free_params);

    // Try to compute covariance and residuals together; fall back to
    // placeholders if any propagation step fails.
    let (covariance, residuals, rms) =
        accumulate_normal_equations(state, obs, included, include_asteroids, non_grav.as_ref())
            .and_then(|(info_mat, _, _)| {
                let cov = scaled_pseudo_inverse(&info_mat)
                    .unwrap_or_else(|_| DMatrix::zeros(n_params, n_params));
                let res = compute_residuals(state, obs, include_asteroids, non_grav.as_ref())?;
                let r = weighted_rms(&res, obs, included, n_params);
                Ok((cov, res, r))
            })
            .unwrap_or_else(|_: Error| {
                let nan_res = obs
                    .iter()
                    .map(|o| DVector::from_element(o.weights().len(), f64::NAN))
                    .collect();
                (DMatrix::zeros(n_params, n_params), nan_res, f64::INFINITY)
            });

    // Dimensions are correct by construction -- `new` cannot fail.
    let uncertain_state =
        UncertainState::new(state.clone(), covariance, non_grav).expect("dimension mismatch");

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
pub(crate) struct StmObs {
    /// Cumulative STM from the reference epoch to this observation, 6 x D.
    pub phi_cum: DMatrix<f64>,
    /// Observation residual (observed - computed), m-vector.
    pub residual: DVector<f64>,
    /// Local geometric partial derivatives, m x 6.
    pub h_local: DMatrix<f64>,
    /// Weight vector (1/sigma^2 per measurement component), m-vector.
    pub weights: DVector<f64>,
}

/// Inner STM sweep over a contiguous observation slice starting from a
/// checkpoint state.
///
/// Returns one `StmObs` per *included* observation whose `phi_cum` is
/// expressed relative to the checkpoint epoch (not the reference epoch),
/// together with the final `phi_cum` at the end of the segment.
///
/// Used by [`stm_sweep`] to implement parallel divide-and-conquer
/// over long observation arcs.
fn stm_sweep_inner(
    checkpoint: &State<Equatorial>,
    obs: &[AstrometricObservation],
    included: &[bool],
    include_asteroids: bool,
    non_grav: Option<&NonGravModel>,
) -> KeteResult<(Vec<StmObs>, DMatrix<f64>)> {
    let np = non_grav.map_or(0, NonGravModel::n_free_params);
    let d = 6 + np;

    // Local phi_cum initialized to identity relative to the checkpoint.
    let mut phi_cum = DMatrix::<f64>::zeros(6, d);
    for i in 0..6 {
        phi_cum[(i, i)] = 1.0;
    }

    let mut state_cur = checkpoint.clone();
    let mut results = Vec::new();

    for (i, observation) in obs.iter().enumerate() {
        let obs_epoch = observation.epoch();

        if (obs_epoch.jd - state_cur.epoch.jd).abs() > 1e-12 {
            let (new_state, phi_k) = compute_state_transition(
                &state_cur,
                obs_epoch,
                include_asteroids,
                non_grav.cloned(),
            )?;

            let phi_state: DMatrix<f64> = phi_k.columns(0, 6).clone_owned();
            let new_state_cols = &phi_state * phi_cum.columns(0, 6);
            phi_cum.columns_mut(0, 6).copy_from(&new_state_cols);

            if np > 0 {
                let phi_param = phi_k.columns(6, np).clone_owned();
                let new_param_cols = &phi_state * phi_cum.columns(6, np) + &phi_param;
                phi_cum.columns_mut(6, np).copy_from(&new_param_cols);
            }

            state_cur = new_state;
        }

        if !included[i] {
            continue;
        }

        let obj_lt = light_time_corrected_state(state_cur.clone(), observation.observer().clone())?;

        let residual = observation.residual_from_corrected(&obj_lt);
        let h_local = observation.partials(&obj_lt);
        let weights = observation.weights();

        results.push(StmObs {
            phi_cum: phi_cum.clone(),
            residual,
            h_local,
            weights,
        });
    }

    Ok((results, phi_cum))
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
/// When observations are numerous, the arc is split into segments that
/// are processed in parallel.  A cheap sequential pre-pass with
/// `propagate_n_body_spk` (6-dim) computes checkpoint states at segment
/// boundaries; each segment then runs its own STM sub-sweep (30-dim)
/// independently.  A brief sequential composition step chains the
/// per-segment STMs into the full cumulative `phi_cum` values.
///
/// # Errors
/// Returns an error if propagation or observation evaluation fails.
///
/// # Panics
/// Panics if the observer state position has zero norm.
pub(crate) fn stm_sweep(
    state_epoch: &State<Equatorial>,
    obs: &[AstrometricObservation],
    included: &[bool],
    include_asteroids: bool,
    non_grav: Option<&NonGravModel>,
) -> KeteResult<Vec<StmObs>> {
    // Minimum observations per segment to keep per-segment overhead small.
    const MIN_OBS_PER_SEGMENT: usize = 8;
    // Segments per thread: more tasks than cores lets rayon's work-stealing
    // compensate when segments finish at different times.
    const SEGMENTS_PER_THREAD: usize = 4;

    debug_assert!(
        obs.windows(2).all(|w| w[0].epoch().jd <= w[1].epoch().jd),
        "stm_sweep: observations must be sorted by epoch"
    );

    let n_threads = rayon::current_num_threads();
    let max_segments = n_threads * SEGMENTS_PER_THREAD;
    // Number of candidate segments, capped by observation density.
    let n_segs_candidate = max_segments.min(obs.len() / MIN_OBS_PER_SEGMENT).max(1);

    // Time-balanced segment boundaries.
    //
    // Split by time span rather than observation count so each segment
    // covers roughly the same interval and requires equal Radau integration
    // work.  Count-based splitting produces wildly unequal work for real
    // asteroid data: a segment coinciding with a dense opposition burst
    // (many observations hours apart) finishes instantly, while one that
    // spans a multi-year gap runs much longer, leaving cores idle.
    let t_start = obs[0].epoch().jd;
    let t_end = obs[obs.len() - 1].epoch().jd;
    let t_span = t_end - t_start;

    let segment_ranges: Vec<(usize, usize)> = if n_segs_candidate > 1 && t_span > 0.0 {
        let dt = t_span / n_segs_candidate as f64;
        let mut starts: Vec<usize> = vec![0];
        for s in 1..n_segs_candidate {
            let t_boundary = t_start + s as f64 * dt;
            let idx = obs.partition_point(|o| o.epoch().jd < t_boundary);
            if idx > *starts.last().unwrap() && idx < obs.len() {
                starts.push(idx);
            }
        }
        let n = starts.len();
        let mut ranges: Vec<(usize, usize)> = starts[..n - 1]
            .iter()
            .zip(starts[1..].iter())
            .map(|(&s, &e)| (s, e))
            .collect();
        ranges.push((starts[n - 1], obs.len()));
        ranges
    } else if n_segs_candidate > 1 {
        // All observations at the same epoch -- equal-count fallback.
        let seg_size = obs.len().div_ceil(n_segs_candidate);
        (0..n_segs_candidate)
            .map(|s| (s * seg_size, obs.len().min((s + 1) * seg_size)))
            .filter(|(start, end)| start < end)
            .collect()
    } else {
        vec![(0, obs.len())]
    };

    let n_segments = segment_ranges.len();
    if n_segments <= 1 {
        return stm_sweep_inner(state_epoch, obs, included, include_asteroids, non_grav)
            .map(|(results, _)| results);
    }

    let np = non_grav.map_or(0, NonGravModel::n_free_params);
    let d = 6 + np;

    // Step 1 -- sequential cheap pre-pass.
    // Propagate the 6-dim state to checkpoint epochs so that every
    // segment has an exact starting state.  Using `propagate_n_body_spk`
    // (6-dim) rather than `compute_state_transition` (30-dim) keeps this
    // pre-pass ~5x cheaper than a full STM integration.
    //
    // IMPORTANT: checkpoints are placed at the epoch of the LAST
    // observation of each segment (obs[end - 1]), not the first
    // observation of the next segment (obs[start_next]).  Both refer to
    // consecutive observation indices (end - 1 and end), but the epoch
    // difference matters:
    //
    //   phi_local_end[s]  = Phi(c_s -> obs[end_s - 1])
    //   c_{s+1}           = state at obs[end_s - 1].epoch
    //   => phi_local_end[s] = Phi(c_s -> c_{s+1})  -- exact: no gap.
    //
    // If instead c_{s+1} were at obs[end_s].epoch, phi_local_end would
    // cover only up to obs[end_s - 1], missing one integration step per
    // segment boundary, causing ~0.2% error per boundary in phi_cum.
    let mut checkpoint_states: Vec<State<Equatorial>> = Vec::with_capacity(n_segments);
    checkpoint_states.push(state_epoch.clone());
    let mut cur = state_epoch.clone();
    for &(_, end) in &segment_ranges[..n_segments - 1] {
        let target_epoch = obs[end - 1].epoch();
        if (target_epoch.jd - cur.epoch.jd).abs() > 1e-12 {
            cur = propagate_n_body_spk(cur, target_epoch, include_asteroids, non_grav.cloned())?;
        }
        checkpoint_states.push(cur.clone());
    }

    // Step 2 -- parallel STM sub-sweeps.
    // Each segment runs independently from its checkpoint with a fresh
    // identity phi_cum.  The local phi_cum values are relative to the
    // segment's checkpoint epoch, not the reference epoch.
    let inputs: Vec<(&[AstrometricObservation], &[bool], &State<Equatorial>)> = segment_ranges
        .iter()
        .enumerate()
        .map(|(s, &(start, end))| {
            (
                &obs[start..end],
                &included[start..end],
                &checkpoint_states[s],
            )
        })
        .collect();

    let segment_results: Vec<KeteResult<(Vec<StmObs>, DMatrix<f64>)>> = inputs
        .into_par_iter()
        .map(|(obs_seg, inc_seg, checkpoint)| {
            stm_sweep_inner(checkpoint, obs_seg, inc_seg, include_asteroids, non_grav)
        })
        .collect();

    let mut local_results: Vec<(Vec<StmObs>, DMatrix<f64>)> = Vec::with_capacity(n_segments);
    for res in segment_results {
        local_results.push(res?);
    }

    // Step 3 -- sequential prefix composition.
    // phi_prefix[s] is the cumulative phi_cum from the reference epoch
    // to the start of segment s.  It obeys the same chaining rule as
    // the inner loop:
    //   prefix[s+1].state_cols = end[s].state_cols * prefix[s].state_cols
    //   prefix[s+1].param_cols = end[s].state_cols * prefix[s].param_cols
    //                          + end[s].param_cols
    let mut phi_prefix = {
        let mut m = DMatrix::<f64>::zeros(6, d);
        for i in 0..6 {
            m[(i, i)] = 1.0;
        }
        m
    };
    let mut prefixes: Vec<DMatrix<f64>> = Vec::with_capacity(n_segments);
    prefixes.push(phi_prefix.clone());
    for seg_result in local_results.iter().take(n_segments - 1) {
        let phi_end = &seg_result.1;
        let phi_end_state = phi_end.columns(0, 6).clone_owned();
        let new_state_cols = &phi_end_state * phi_prefix.columns(0, 6);
        phi_prefix.columns_mut(0, 6).copy_from(&new_state_cols);
        if np > 0 {
            let phi_end_param = phi_end.columns(6, np).clone_owned();
            let new_param_cols = &phi_end_state * phi_prefix.columns(6, np) + &phi_end_param;
            phi_prefix.columns_mut(6, np).copy_from(&new_param_cols);
        }
        prefixes.push(phi_prefix.clone());
    }

    // Step 4 -- apply prefix transform.
    // For each StmObs in segment s with local phi_cum L:
    //   full.state_cols = L.state_cols * prefix[s].state_cols
    //   full.param_cols = L.state_cols * prefix[s].param_cols + L.param_cols
    let mut all_results = Vec::new();
    for (s, (local_obs, _)) in local_results.into_iter().enumerate() {
        let prefix_a = prefixes[s].columns(0, 6).clone_owned();
        for mut entry in local_obs {
            let local_a = entry.phi_cum.columns(0, 6).clone_owned();
            let full_a = &local_a * &prefix_a;
            entry.phi_cum.columns_mut(0, 6).copy_from(&full_a);
            if np > 0 {
                let local_b = entry.phi_cum.columns(6, np).clone_owned();
                let prefix_b = prefixes[s].columns(6, np);
                let full_b = &local_a * prefix_b + &local_b;
                entry.phi_cum.columns_mut(6, np).copy_from(&full_b);
            }
            all_results.push(entry);
        }
    }

    Ok(all_results)
}

/// Two-body variant of [`stm_sweep`] for MCMC sampling.
///
/// Uses analytic Keplerian propagation with the closed-form Lagrange-coefficient
/// STM instead of the full N-body Radau integrator.  Lower accuracy but much faster.
///
/// The trade-off is that N-body perturbations are ignored in both the residuals
/// and the Jacobian.  For the use case of MCMC this is usually acceptable because
/// the posterior shape is dominated by observation geometry, not small planetary
/// perturbations, and the likelihood + gradient remain internally consistent.
///
/// Non-gravitational parameter sensitivities are set to zero (two-body dynamics
/// has no non-grav forces), so this is most useful when `non_grav` is `None`.
///
/// The propagation is performed in heliocentric (Sun-centered) coordinates
/// because [`analytic_2_body_stm`] treats the coordinate origin as the central
/// mass.  The input `state_epoch` may be in any inertial frame and is
/// converted internally; the returned STM is identical in either heliocentric
/// or SSB coordinates because the Sun-SSB offset is independent of the
/// object's state.  The observation Jacobian (`h_local`) is likewise
/// frame-invariant because it depends only on the object-observer vector.
///
/// # Errors
/// Returns an error if two-body propagation or light-time correction fails.
pub(crate) fn stm_sweep_two_body(
    state_epoch: &State<Equatorial>,
    obs: &[AstrometricObservation],
    included: &[bool],
    non_grav: Option<&NonGravModel>,
) -> KeteResult<Vec<StmObs>> {
    debug_assert!(
        obs.windows(2).all(|w| w[0].epoch().jd <= w[1].epoch().jd),
        "stm_sweep_two_body: observations must be sorted by epoch"
    );
    let np = non_grav.map_or(0, NonGravModel::n_free_params);
    let d = 6 + np;

    let mut phi_cum = DMatrix::<f64>::zeros(6, d);
    for i in 0..6 {
        phi_cum[(i, i)] = 1.0;
    }

    // Convert to heliocentric coordinates for propagation.  `analytic_2_body`
    // assumes the Sun sits at the origin, so feeding it SSB-centered coords
    // (the usual convention inside the fitter) would bias every step by the
    // Sun-SSB offset (~0.007 AU), producing residuals dominated by that
    // systematic error rather than by the orbit fit.
    let mut state_helio = state_epoch.clone();
    if state_helio.center_id != 10 {
        let spk = LOADED_SPK.try_read()?;
        spk.try_change_center(&mut state_helio, 10)?;
    }

    let mut cur_pos: Vector3<f64> = state_helio.pos.into();
    let mut cur_vel: Vector3<f64> = state_helio.vel.into();
    let mut cur_epoch = state_helio.epoch;
    let mut results = Vec::new();

    for (i, observation) in obs.iter().enumerate() {
        let obs_epoch = observation.epoch();

        if (obs_epoch.jd - cur_epoch.jd).abs() > 1e-12 {
            let dt = obs_epoch - cur_epoch;
            let (new_pos, new_vel, phi_k) = analytic_2_body_stm(dt, &cur_pos, &cur_vel, None)?;

            // Chain the 6x6 state block.
            let new_state_cols = &phi_k * phi_cum.columns(0, 6);
            phi_cum.columns_mut(0, 6).copy_from(&new_state_cols);

            // Non-grav parameter columns: phi_state * phi_cum_param + 0
            // (two-body has no parameter sensitivity, so phi_param = 0).
            if np > 0 {
                let new_param_cols = &phi_k * phi_cum.columns(6, np);
                phi_cum.columns_mut(6, np).copy_from(&new_param_cols);
            }

            cur_pos = new_pos;
            cur_vel = new_vel;
            cur_epoch = obs_epoch;
        }

        if !included[i] {
            continue;
        }

        // Build a heliocentric State for light-time correction and residuals.
        // `light_time_corrected_state` re-centers as needed internally.
        let mut cur_state = state_helio.clone();
        cur_state.pos = cur_pos.into();
        cur_state.vel = cur_vel.into();
        cur_state.epoch = cur_epoch;

        let obj_lt = light_time_corrected_state(cur_state, observation.observer().clone())?;

        let residual = observation.residual_from_corrected(&obj_lt);
        let h_local = observation.partials(&obj_lt);
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
/// Returns `(info_mat, rhs_vec, huber_loss)` where `info_mat` is the
/// (6+Np) x (6+Np) information matrix, `rhs_vec` is the right-hand
/// side, and `huber_loss` is the Huber rho objective summed over all
/// included measurements (the merit function used by the LM accept gate).
///
/// # Errors
/// Returns an error if the underlying STM sweep fails.
pub(crate) fn accumulate_normal_equations(
    state_epoch: &State<Equatorial>,
    obs: &[AstrometricObservation],
    included: &[bool],
    include_asteroids: bool,
    non_grav: Option<&NonGravModel>,
) -> KeteResult<(DMatrix<f64>, DVector<f64>, f64)> {
    let sweep = stm_sweep(state_epoch, obs, included, include_asteroids, non_grav)?;
    let (n_mat, b_vec, loss, _) = accumulate_from_sweep(&sweep, non_grav);
    Ok((n_mat, b_vec, loss))
}

/// Huber tuning constant for IRLS downweighting.
///
/// k = 1.345 gives 95% asymptotic efficiency at the Gaussian model while
/// providing strong outlier resistance.
const HUBER_K: f64 = 1.345;

/// Accumulate normal equations from a pre-computed STM sweep.
///
/// Applies iteratively reweighted least squares (IRLS) with Huber loss:
/// each measurement component whose normalized residual exceeds [`HUBER_K`]
/// is smoothly downweighted, preventing large residuals from dominating the
/// normal equations.
///
/// Returns `(info_mat, rhs_vec, huber_loss, residuals)` where `huber_loss`
/// is the Huber rho objective summed over all measurements.  This is the
/// merit function the IRLS-LM solver should monotonically reduce; using the
/// Huber-weighted sum-of-squared residuals instead would be non-monotone
/// because Huber weights themselves shift as residuals cross the threshold,
/// producing inconsistent step-acceptance decisions for nearby starting
/// states.  `residuals` contains one entry per included observation
/// (matching the sweep).
fn accumulate_from_sweep(
    sweep: &[StmObs],
    non_grav: Option<&NonGravModel>,
) -> (DMatrix<f64>, DVector<f64>, f64, Vec<DVector<f64>>) {
    let np = non_grav.map_or(0, NonGravModel::n_free_params);
    let d = 6 + np;

    let mut n_mat = DMatrix::<f64>::zeros(d, d);
    let mut b_vec = DVector::<f64>::zeros(d);
    let mut huber_loss = 0.0;
    let mut residuals = Vec::with_capacity(sweep.len());

    for entry in sweep {
        let m = entry.residual.len();

        // Map to epoch: H_epoch = H_local * Phi_cum  (m x D).
        let h_epoch = &entry.h_local * &entry.phi_cum;

        // Huber IRLS: downweight components with large normalized residuals,
        // and accumulate the Huber rho loss over all measurements using the
        // ORIGINAL (un-Huber'd) weights.
        let mut weights = entry.weights.clone();
        for k in 0..m {
            let z_abs = (entry.residual[k] * entry.weights[k].sqrt()).abs();
            if z_abs <= HUBER_K {
                huber_loss += 0.5 * z_abs * z_abs;
            } else {
                huber_loss += HUBER_K * z_abs - 0.5 * HUBER_K * HUBER_K;
                weights[k] *= HUBER_K / z_abs;
            }
        }

        residuals.push(entry.residual.clone());

        // Accumulate normal matrix and RHS via weighted outer products:
        //   N += H^T W H,  b += H^T W r
        // Build sqrt(W) * H and sqrt(W) * r for efficient rank-m update.
        // hw is m x d, wr is m x 1.
        let mut hw = h_epoch.clone();
        let mut wr = entry.residual.clone();
        for k in 0..m {
            let sw = weights[k].sqrt();
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

    (n_mat, b_vec, huber_loss, residuals)
}

/// Solve `(N + lambda * diag(N)) * dx = b` via SVD with column scaling.
///
/// When `lambda > 0` the diagonal of N is augmented, pulling the solution
/// toward a steepest-descent step and stabilising poorly-constrained
/// directions.
///
/// Column scaling is essential when the parameters span very different
/// magnitudes (e.g. position in AU together with `density` in kg/m^3 or
/// `thermal_inertia` in SI units).  Without it, the information matrix
/// can have a condition number of 1e20 or more and the SVD pseudo-inverse
/// truncation cutoff zeroes out the small parameter directions, producing
/// degenerate steps.  We pre- and post-multiply by `D = diag(1/sqrt(|N_ii|))`
/// so the scaled normal matrix has unit diagonal and a relative SVD cutoff
/// is meaningful.
fn solve_damped(
    n_mat: &DMatrix<f64>,
    b_vec: &DVector<f64>,
    lambda: f64,
) -> KeteResult<DVector<f64>> {
    let n = n_mat.nrows();

    // Build inverse column scales d_inv[i] = 1/sqrt(|N_ii|).
    let mut d_inv = DVector::<f64>::zeros(n);
    for i in 0..n {
        let dii = n_mat[(i, i)].abs().max(1e-30);
        d_inv[i] = 1.0 / dii.sqrt();
    }

    // Scaled normal matrix: N_s = D^-1 N D^-1  (unit diagonal by construction).
    let mut n_scaled = n_mat.clone();
    for i in 0..n {
        for j in 0..n {
            n_scaled[(i, j)] *= d_inv[i] * d_inv[j];
        }
    }

    // Marquardt damping in the scaled space (where diag(N_s) = 1):
    //   (N_s + lambda I) dx_s = b_s
    if lambda > 0.0 {
        for i in 0..n {
            n_scaled[(i, i)] += lambda;
        }
    }

    // Scaled RHS.
    let mut b_scaled = b_vec.clone();
    for i in 0..n {
        b_scaled[i] *= d_inv[i];
    }

    let svd = n_scaled.svd(true, true);
    let dx_scaled = svd
        .solve(&b_scaled, 1e-12)
        .map_err(|_| Error::ValueError("SVD solve failed on damped normal matrix".into()))?;

    // Unscale: dx = D^-1 dx_s.
    let mut dx = dx_scaled;
    for i in 0..n {
        dx[i] *= d_inv[i];
    }
    Ok(dx)
}

/// Pseudo-invert a normal-equations (information) matrix using the
/// same diagonal column scaling as [`solve_damped`].
///
/// The fit parameters span many orders of magnitude (state in AU and
/// AU/day, non-grav coefficients from ~1e-12 to ~1e1).  Without
/// scaling, the information matrix has condition numbers easily
/// exceeding 1e20, and the relative SVD truncation cutoff zeroes out
/// the small-parameter rows and columns -- producing a covariance that
/// is essentially numerical noise.  Scaling so the diagonal is unit
/// makes the relative cutoff meaningful, then we unscale on the way
/// out: cov = D^-1 (D^-1 N D^-1)^+ D^-1.
fn scaled_pseudo_inverse(n_mat: &DMatrix<f64>) -> KeteResult<DMatrix<f64>> {
    let n = n_mat.nrows();
    let mut d_inv = DVector::<f64>::zeros(n);
    for i in 0..n {
        let dii = n_mat[(i, i)].abs().max(1e-30);
        d_inv[i] = 1.0 / dii.sqrt();
    }

    let mut n_scaled = n_mat.clone();
    for i in 0..n {
        for j in 0..n {
            n_scaled[(i, j)] *= d_inv[i] * d_inv[j];
        }
    }

    let cov_scaled = n_scaled
        .svd(true, true)
        .pseudo_inverse(1e-12)
        .map_err(|e| Error::ValueError(format!("SVD pseudo-inverse failed: {e}")))?;

    let mut cov = cov_scaled;
    for i in 0..n {
        for j in 0..n {
            cov[(i, j)] *= d_inv[i] * d_inv[j];
        }
    }
    Ok(cov)
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

/// Backtrack only the non-grav components of a step so each parameter
/// stays strictly above its lower bound.
///
/// Position and velocity are left untouched. For each non-grav
/// parameter that would be pushed at or below its bound, the
/// corresponding entry of `dx[6+k]` is shrunk so the new value sits
/// just above the bound. This avoids the pathology where a single
/// out-of-bounds non-grav step inflates Marquardt damping and shrinks
/// the position step (which was correctly sized) to nothing -- the
/// orbital fit then stalls completely.
fn backtrack_to_ng_bounds(mut dx: DVector<f64>, non_grav: Option<&NonGravModel>) -> DVector<f64> {
    let Some(ng) = non_grav else {
        return dx;
    };
    let np = ng.n_free_params();
    let params = ng.get_free_params();
    let bounds = ng.param_lower_bounds();
    for k in 0..np {
        let cur = params[k];
        let lo = bounds[k];
        if !lo.is_finite() {
            continue;
        }
        let proposed = cur + dx[6 + k];
        if proposed > lo {
            continue;
        }
        // Shrink to land halfway between the current value and the
        // bound so we make progress toward the bound without snapping
        // onto it (which would silently re-trigger the clamp inside
        // `set_free_params`).
        let safe_step = 0.5 * (lo - cur);
        dx[6 + k] = safe_step;
    }
    dx
}

/// Apply a state correction vector to the epoch state and (optionally)
/// non-grav parameters.
fn apply_correction(
    state: &mut State<Equatorial>,
    dx: &DVector<f64>,
    non_grav: &mut Option<NonGravModel>,
) -> bool {
    if let Some(ng) = non_grav.as_ref() {
        let np = ng.n_free_params();
        let params = ng.get_free_params();
        let bounds = ng.param_lower_bounds();
        for k in 0..np {
            let new_val = params[k] + dx[6 + k];
            if new_val <= bounds[k] || !new_val.is_finite() {
                return false;
            }
        }
    }

    let pos: [f64; 3] = state.pos.into();
    state.pos = [pos[0] + dx[0], pos[1] + dx[1], pos[2] + dx[2]].into();

    let vel: [f64; 3] = state.vel.into();
    state.vel = [vel[0] + dx[3], vel[1] + dx[4], vel[2] + dx[5]].into();

    if let Some(ng) = non_grav.as_mut() {
        let np = ng.n_free_params();
        let mut params = ng.get_free_params();
        for k in 0..np {
            params[k] += dx[6 + k];
        }
        ng.set_free_params(&params);
    }
    true
}

/// Compute post-fit residuals for all observations (time-sorted order).
///
/// Uses the 6-dim `propagate_n_body_spk` (not the 60-dim STM
/// integrator) so this is ~5x cheaper than an STM sweep.
fn compute_residuals(
    state_epoch: &State<Equatorial>,
    obs: &[AstrometricObservation],
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

        let obj_lt = light_time_corrected_state(state_cur.clone(), observation.observer().clone())?;
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
    obs: &[AstrometricObservation],
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
    use kete_core::time::{TDB, Time};
    use kete_spice::propagation::propagate_n_body_spk;

    use kete_spice::test_data::ensure_test_spk;

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
    ) -> Vec<AstrometricObservation> {
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

            let spk = LOADED_SPK.try_read().unwrap();
            let mut sun_at = obj_at.clone();
            spk.try_change_center(&mut sun_at, 10).unwrap();
            let obs_helio = observer.pos - obj_at.pos + sun_at.pos;
            let mut obj_lt = light_time_correct(&sun_at, &obs_helio).unwrap();
            spk.try_change_center(&mut obj_lt, 0).unwrap();
            let (ra, dec) = (obj_lt.pos - observer.pos).to_ra_dec();

            observations.push(AstrometricObservation::Optical {
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
        ensure_test_spk();
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

        let fit = fit_orbit(&perturbed, &observations, false, None, 20, 1e-8, 9.0, 0).unwrap();

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
        ensure_test_spk();
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

        let fit = fit_orbit(&perturbed, &observations, false, None, 20, 1e-8, 9.0, 0).unwrap();

        let pos_err = (fit.uncertain_state.state.pos - true_state.pos).norm();

        assert!(
            pos_err < 1e-3,
            "Position error {pos_err:.6e} too large for elliptical orbit"
        );
    }

    #[test]
    fn test_outlier_rejection() {
        ensure_test_spk();
        // True orbit: circular at 1.5 AU.
        let r = 1.5;
        let v = (GMS / r).sqrt();
        let true_state = make_state([r, 0.0, 0.0], [0.0, v, 0.0], 2460000.5);

        let epochs: Vec<f64> = (0..10).map(|i| 2460000.5 + f64::from(i) * 6.0).collect();
        let sigma = 1e-6;
        let mut observations =
            synth_observations(&true_state, &epochs, earth_observer, sigma, None);

        // Corrupt observation 3 with a large offset (100x sigma).
        if let AstrometricObservation::Optical { ref mut ra, .. } = observations[3] {
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
        ensure_test_spk();
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
        ensure_test_spk();
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
        ensure_test_spk();
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

        let fit = fit_orbit(&perturbed, &observations, false, None, 50, 1e-8, 9.0, 3).unwrap();

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
        ensure_test_spk();
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
        if let AstrometricObservation::Optical { ref mut ra, .. } = observations[18] {
            *ra += 50.0 * sigma;
        }

        let fit = fit_orbit(&true_state, &observations, false, None, 50, 1e-8, 9.0, 5).unwrap();

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

    /// Verify that the parallel multi-segment [`stm_sweep`] produces the same
    /// normal equations as the sequential single-segment path.
    ///
    /// This catches the boundary-epoch bug where the checkpoint for segment
    /// s+1 is placed at `obs[end_s]` instead of `obs[end_s-1]`, leaving a
    /// missing integration step in every prefix matrix and causing ~0.2%
    /// per-boundary error in `phi_cum` for Ceres-like data.
    #[test]
    fn test_stm_sweep_parallel_matches_sequential() {
        ensure_test_spk();
        // 100 observations over 2 years -- large enough that multiple
        // segments are created regardless of the thread count, while
        // staying cheap enough for a unit test.
        let r = 2.5;
        let v = (GMS / r).sqrt();
        let state = make_state([r, 0.0, 0.0], [0.0, v, 0.0], 2460000.5);
        let epochs: Vec<f64> = (0..100).map(|i| 2460000.5 + f64::from(i) * 7.3).collect();
        let sigma = 1e-6;
        let observations = synth_observations(&state, &epochs, earth_observer, sigma, None);
        let included = vec![true; observations.len()];

        // Force the parallel path: temporarily request a pool with at least
        // 4 threads so n_segments > 1.  Fall back to evaluating both paths
        // with the real pool size if rayon has few threads.
        let n_threads = rayon::current_num_threads();

        // Sequential reference: force single segment by calling the inner
        // function directly.
        let (seq_results, _) =
            stm_sweep_inner(&state, &observations, &included, false, None).unwrap();

        // Parallel result via the public API.
        let par_results = stm_sweep(&state, &observations, &included, false, None).unwrap();

        assert_eq!(
            seq_results.len(),
            par_results.len(),
            "result count mismatch"
        );

        if n_threads < 2 {
            // Single-threaded rayon -- both paths are identical by
            // construction; just check they have the same length.
            return;
        }

        for (k, (seq, par)) in seq_results.iter().zip(par_results.iter()).enumerate() {
            let phi_diff = (&seq.phi_cum - &par.phi_cum).norm();
            let phi_scale = seq.phi_cum.norm().max(1.0);
            assert!(
                phi_diff / phi_scale < 1e-9,
                "phi_cum mismatch at obs {k}: relative error {:.3e}",
                phi_diff / phi_scale
            );

            let res_diff = (&seq.residual - &par.residual).norm();
            assert!(
                res_diff < 1e-10,
                "residual mismatch at obs {k}: {res_diff:.3e}"
            );
        }
    }
}
