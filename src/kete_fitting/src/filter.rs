//! Extended Kalman Filter and RTS smoother for orbit fitting.
//!
//! An alternative to the batch DC pipeline in [`crate::orbit_fitting`].
//! Processes observations sequentially in time order, needing only
//! short-arc STM propagations between consecutive observations.
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
use crate::orbit_fitting::OrbitFit;
use crate::uncertain_state::UncertainState;
use kete_core::forces::NonGravModel;
use kete_core::frames::{CenterBody, Equatorial, SSB};
use kete_core::kepler::light_time_correct;
use kete_core::prelude::{Error, KeteResult, State};
use kete_spice::prelude::{LOADED_SPK, compute_state_transition};
use nalgebra::{DMatrix, DVector};

/// Stored quantities at each observation epoch for the RTS backward pass.
struct FilterEpoch {
    /// Filtered (post-update) state vector.
    x_filtered: DVector<f64>,
    /// Filtered covariance.
    p_filtered: DMatrix<f64>,
    /// Predicted (pre-update) state vector.
    x_predicted: DVector<f64>,
    /// Predicted covariance.
    p_predicted: DMatrix<f64>,
    /// STM from the *previous* epoch to this one.
    phi: DMatrix<f64>,
}

/// Number of free non-grav parameters (0 when `None`).
fn n_nongrav_params(ng: Option<&NonGravModel>) -> usize {
    ng.map_or(0, NonGravModel::n_free_params)
}

/// Pack a [`State`] (and optional non-grav params) into a state vector.
fn state_to_vec<C: CenterBody>(state: &State<Equatorial, C>, non_grav: Option<&NonGravModel>) -> DVector<f64> {
    let np = n_nongrav_params(non_grav);
    let dim = 6 + np;
    let mut xv = DVector::zeros(dim);
    xv[0] = state.pos[0];
    xv[1] = state.pos[1];
    xv[2] = state.pos[2];
    xv[3] = state.vel[0];
    xv[4] = state.vel[1];
    xv[5] = state.vel[2];
    if let Some(ng) = non_grav {
        let fp = ng.get_free_params();
        for (idx, &val) in fp.iter().enumerate() {
            xv[6 + idx] = val;
        }
    }
    xv
}

/// Unpack a state vector back into a [`State`] (and update non-grav params).
fn vec_to_state<C: CenterBody + Clone>(
    xv: &DVector<f64>,
    template: &State<Equatorial, C>,
    non_grav: &mut Option<NonGravModel>,
) -> State<Equatorial, C> {
    let mut state = template.clone();
    state.pos = [xv[0], xv[1], xv[2]].into();
    state.vel = [xv[3], xv[4], xv[5]].into();
    if let Some(ng) = non_grav.as_mut() {
        let np = ng.n_free_params();
        let params: Vec<f64> = xv.rows(6, np).iter().copied().collect();
        ng.set_free_params(&params);
    }
    state
}

/// Build a process noise matrix for the prediction step.
///
/// Applies noise only to velocity components (unmodeled acceleration).
/// `Q_vv = q * dt * I_3x3`.  Position noise is zero.
fn build_process_noise(dim: usize, spectral_q: f64, dt: f64) -> DMatrix<f64> {
    let mut qmat = DMatrix::zeros(dim, dim);
    let q_vel = spectral_q * dt.abs();
    for idx in 3..6 {
        qmat[(idx, idx)] = q_vel;
    }
    qmat
}

/// Expand a 6 x `dim` STM from `compute_state_transition` into a `dim` x `dim`
/// matrix by padding with identity for the non-grav parameter block (parameters
/// propagate unchanged, already handled in the sensitivity columns).
fn expand_phi(phi_6xd: &DMatrix<f64>, dim: usize) -> DMatrix<f64> {
    let np = dim - 6;
    let mut phi = DMatrix::zeros(dim, dim);
    phi.view_mut((0, 0), (6, 6))
        .copy_from(&phi_6xd.view((0, 0), (6, 6)));
    if np > 0 {
        phi.view_mut((0, 6), (6, np))
            .copy_from(&phi_6xd.view((0, 6), (6, np)));
    }
    for idx in 0..np {
        phi[(6 + idx, 6 + idx)] = 1.0;
    }
    phi
}

/// Expand an `m x 6` local Jacobian into `m x dim` by appending zero columns
/// for non-grav parameters (optical/radar measurements do not depend
/// directly on non-grav params at the observation epoch).
fn expand_h(h_local: &DMatrix<f64>, dim: usize) -> DMatrix<f64> {
    let meas_dim = h_local.nrows();
    let mut h_full = DMatrix::zeros(meas_dim, dim);
    h_full.view_mut((0, 0), (meas_dim, 6)).copy_from(h_local);
    h_full
}

/// Apply light-time correction and change center for an SSB-centered state.
///
/// Returns the light-time-corrected state in SSB coordinates.
fn apply_light_time(
    obj_state: &State<Equatorial, SSB>,
    observer: &State<Equatorial, SSB>,
) -> KeteResult<State<Equatorial, SSB>> {
    let spk = LOADED_SPK.try_read()?;
    let sun_state = spk.try_to_sun(obj_state.clone().into())?;
    let obs_sun = spk.try_to_sun(observer.clone().into())?.pos;
    let obj_lt_sun = light_time_correct(&sun_state, &obs_sun)?;
    spk.try_to_ssb(obj_lt_sun.into())
}

/// Store the predicted state as the filtered state (observation skipped)
/// and push it onto the epoch list.
fn push_skipped_epoch(
    epochs: &mut Vec<FilterEpoch>,
    accepted_flags: &mut Vec<bool>,
    xv_pred: &DVector<f64>,
    cov_pred: &DMatrix<f64>,
    phi: DMatrix<f64>,
) {
    epochs.push(FilterEpoch {
        x_filtered: xv_pred.clone(),
        p_filtered: cov_pred.clone(),
        x_predicted: xv_pred.clone(),
        p_predicted: cov_pred.clone(),
        phi,
    });
    accepted_flags.push(false);
}

/// Maximum number of forward-backward iterations for the IEKF.
const MAX_ITERATIONS: usize = 10;

/// Convergence threshold: stop iterating when the state change (in AU)
/// between successive iterations is below this value.
const CONVERGENCE_THRESHOLD: f64 = 1e-12;

/// Factor by which the previous iteration's smoothed covariance is
/// inflated before re-use.  Accounts for linearization error so that
/// subsequent passes don't overcorrect on the first few observations.
const COVARIANCE_INFLATION: f64 = 4.0;

/// Fit an orbit using an Iterated Extended Kalman Filter with RTS smoother.
///
/// Processes observations sequentially, needing only short-arc STM
/// propagations between consecutive observations. The RTS backward
/// pass then produces a smoothed estimate. Multiple forward-backward
/// iterations are performed until convergence, which handles the
/// nonlinearity of the observation model.
///
/// A conservative initial covariance is constructed automatically:
/// 1 AU^2 for position, 0.01 (AU/day)^2 for velocity, and 1.0 for
/// any non-gravitational parameters.
///
/// # Arguments
/// * `initial_state` -- Initial guess (SSB-centered Equatorial).
/// * `obs` -- Observations (any order; sorted internally).
/// * `include_asteroids` -- Include asteroid masses in the force model.
/// * `non_grav` -- Optional non-gravitational model.
/// * `chi2_gate` -- Per-observation innovation gate threshold. Observations
///   whose Mahalanobis distance exceeds this are skipped. Typical value:
///   16.0 (chi-squared with 2 dof at ~99.7%).
/// * `process_noise_q` -- Process noise spectral density (AU^2/day^3).
///   Set to 0.0 or very small (1e-30) for well-modeled orbits.
///
/// # Errors
/// Fails if propagation or SPK queries fail.
pub fn fit_orbit_filter(
    initial_state: &State<Equatorial, SSB>,
    obs: &[AstrometricObservation],
    include_asteroids: bool,
    non_grav: Option<&NonGravModel>,
    chi2_gate: f64,
    process_noise_q: f64,
) -> KeteResult<OrbitFit> {
    if obs.is_empty() {
        return Err(Error::ValueError("No observations provided".into()));
    }

    let np = n_nongrav_params(non_grav);
    let dim = 6 + np;

    // Build a conservative initial covariance: generous enough to cover
    // the uncertainty from any reasonable IOD seed.
    let mut initial_covariance = DMatrix::zeros(dim, dim);
    for idx in 0..3 {
        initial_covariance[(idx, idx)] = 1.0; // 1 AU^2
    }
    for idx in 3..6 {
        initial_covariance[(idx, idx)] = 0.01; // (0.1 AU/day)^2
    }
    for idx in 6..dim {
        initial_covariance[(idx, idx)] = 1.0;
    }

    let sorted = sort_by_epoch(obs);
    if sorted.is_empty() {
        return Err(Error::ValueError(
            "No observations with finite observer states".into(),
        ));
    }

    let mut ng = non_grav.cloned();
    let mut iter_state: State<Equatorial, SSB> = initial_state.clone();

    // Outer iteration loop: re-linearize around the smoothed state.
    // After each pass the smoothed covariance (inflated) is used as the
    // initial covariance for the next pass (IEKFm variant), preventing
    // excessively large Kalman gains once the state is close to the truth.
    let mut best_epochs = Vec::new();
    let mut best_accepted = Vec::new();
    let mut best_xv_smoothed = Vec::new();
    let mut best_cov_smoothed = Vec::new();

    // Covariance used at the start of each forward pass. Starts as the
    // default initial covariance and is updated to an inflated version
    // of the smoothed covariance after each iteration (IEKFm).
    let mut iter_cov = initial_covariance;

    for iteration in 0..MAX_ITERATIONS {
        // Disable innovation gating during intermediate iterations.
        // Gating is applied only on the final pass for outlier rejection.
        let iter_gate = if iteration + 1 == MAX_ITERATIONS {
            chi2_gate
        } else {
            f64::INFINITY
        };

        let (epochs, accepted_flags, xv_fwd, cov_fwd) = forward_pass(
            &iter_state,
            &iter_cov,
            &sorted,
            include_asteroids,
            &mut ng,
            non_grav,
            iter_gate,
            process_noise_q,
            dim,
        );

        // At least some observations must have been assimilated.
        let n_accepted = accepted_flags.iter().filter(|&&a| a).count();
        if n_accepted == 0 {
            return Err(Error::ValueError(
                "Filter could not process any observations -- the initial state \
                 may be too far from the truth."
                    .into(),
            ));
        }

        let (xv_smoothed, cov_smoothed) = backward_pass(&epochs, xv_fwd, cov_fwd, dim);

        // The smoothed state at the first epoch becomes the new initial state.
        let new_xv = &xv_smoothed[0];
        let change = (new_xv - state_to_vec(&iter_state, non_grav)).norm();

        // Update initial state for next iteration.
        iter_state = vec_to_state(new_xv, &iter_state, &mut ng);

        // Update covariance for the next iteration: inflate the smoothed
        // covariance so subsequent passes start with a tighter (but still
        // conservative) estimate instead of the original large prior.
        iter_cov = &cov_smoothed[0] * COVARIANCE_INFLATION;
        // Enforce symmetry.
        iter_cov = (&iter_cov + iter_cov.transpose()) * 0.5;

        best_epochs = epochs;
        best_accepted = accepted_flags;
        best_xv_smoothed = xv_smoothed;
        best_cov_smoothed = cov_smoothed;

        if change < CONVERGENCE_THRESHOLD && iteration + 1 < MAX_ITERATIONS {
            // Converged early -- do one final pass with the gate enabled.
            let (final_epochs, final_accepted, final_xv, final_cov) = forward_pass(
                &iter_state,
                &iter_cov,
                &sorted,
                include_asteroids,
                &mut ng,
                non_grav,
                chi2_gate,
                process_noise_q,
                dim,
            );

            let (final_xv_sm, final_cov_sm) =
                backward_pass(&final_epochs, final_xv, final_cov, dim);

            best_epochs = final_epochs;
            best_accepted = final_accepted;
            best_xv_smoothed = final_xv_sm;
            best_cov_smoothed = final_cov_sm;
            break;
        }
    }

    // -- Build result ------------------------------------------------
    build_result(
        &sorted,
        initial_state,
        &best_epochs,
        &best_accepted,
        &best_xv_smoothed,
        &best_cov_smoothed,
        &mut ng,
        non_grav,
        dim,
    )
}

/// Single forward EKF pass over all observations.
#[expect(clippy::too_many_arguments, reason = "filter configuration parameters")]
fn forward_pass(
    initial_state: &State<Equatorial, SSB>,
    initial_covariance: &DMatrix<f64>,
    sorted: &[AstrometricObservation],
    include_asteroids: bool,
    ng: &mut Option<NonGravModel>,
    non_grav: Option<&NonGravModel>,
    chi2_gate: f64,
    process_noise_q: f64,
    dim: usize,
) -> (Vec<FilterEpoch>, Vec<bool>, DVector<f64>, DMatrix<f64>) {
    let mut xv = state_to_vec(initial_state, non_grav);
    let mut cov = initial_covariance.clone();
    let mut state_cur: State<Equatorial, SSB> = initial_state.clone();
    let mut epochs: Vec<FilterEpoch> = Vec::with_capacity(sorted.len());
    let mut accepted_flags: Vec<bool> = Vec::with_capacity(sorted.len());

    for observation in sorted {
        let obs_epoch = observation.epoch();
        let dt = obs_epoch.jd - state_cur.epoch.jd;

        // -- Prediction --------------------------------------------
        // EKF: propagate the state nonlinearly; use the STM only for
        // the covariance prediction.
        let (phi_full, xv_pred, cov_pred, new_state) = if dt.abs() > 1e-12 {
            let ssb_result = compute_state_transition(&state_cur, obs_epoch, include_asteroids, ng.clone()).ok();
            if let Some((propagated_ssb, phi_6xd)) = ssb_result {
                let phi = expand_phi(&phi_6xd, dim);
                let qmat = build_process_noise(dim, process_noise_q, dt);
                let xv_pred = state_to_vec(&propagated_ssb, non_grav);
                let mut cov_pred = &phi * &cov * phi.transpose() + qmat;
                // Enforce symmetry after prediction.
                cov_pred = (&cov_pred + cov_pred.transpose()) * 0.5;
                (phi, xv_pred, cov_pred, propagated_ssb)
            } else {
                // Propagation failed -- skip this observation.
                // State remains at the current epoch; the next observation
                // will attempt a longer propagation from here.
                let phi = DMatrix::identity(dim, dim);
                push_skipped_epoch(&mut epochs, &mut accepted_flags, &xv, &cov, phi);
                continue;
            }
        } else {
            let phi = DMatrix::identity(dim, dim);
            (phi, xv.clone(), cov.clone(), state_cur.clone())
        };

        // -- Measurement update ------------------------------------
        let Ok(observer_state) = observation.observer() else {
            push_skipped_epoch(
                &mut epochs,
                &mut accepted_flags,
                &xv_pred,
                &cov_pred,
                phi_full,
            );
            xv = xv_pred;
            cov = cov_pred;
            state_cur = new_state;
            continue;
        };
        let Ok(obj_lt_ssb) = apply_light_time(&new_state, &observer_state) else {
            // Light-time correction failed -- skip this observation
            // but advance the state to the predicted epoch.
            push_skipped_epoch(
                &mut epochs,
                &mut accepted_flags,
                &xv_pred,
                &cov_pred,
                phi_full,
            );
            xv = xv_pred;
            cov = cov_pred;
            state_cur = new_state;
            continue;
        };
        let Ok(residual) = observation.residual_from_corrected(&obj_lt_ssb) else {
            push_skipped_epoch(
                &mut epochs,
                &mut accepted_flags,
                &xv_pred,
                &cov_pred,
                phi_full,
            );
            xv = xv_pred;
            cov = cov_pred;
            state_cur = new_state;
            continue;
        };
        let Ok(h_local) = observation.partials(&obj_lt_ssb) else {
            push_skipped_epoch(
                &mut epochs,
                &mut accepted_flags,
                &xv_pred,
                &cov_pred,
                phi_full,
            );
            xv = xv_pred;
            cov = cov_pred;
            state_cur = new_state;
            continue;
        };
        let h_full = expand_h(&h_local, dim);

        // Measurement noise R is the inverse of the base information
        // matrix (which carries any RA/Dec correlation).  We invert via
        // pseudo-inverse for robustness; the matrix is small and
        // well-conditioned for all realistic sigma combinations.
        let w_base = observation.base_weight_matrix();
        let Some(r_mat) = w_base.clone().try_inverse() else {
            push_skipped_epoch(
                &mut epochs,
                &mut accepted_flags,
                &xv_pred,
                &cov_pred,
                phi_full,
            );
            xv = xv_pred;
            cov = cov_pred;
            state_cur = new_state;
            continue;
        };

        // Innovation covariance: S = H P_pred H^T + R
        let innov_cov = &h_full * &cov_pred * h_full.transpose() + &r_mat;

        // Cholesky decomposition of S for gate test and Kalman gain.
        let Some(innov_chol) = innov_cov.clone().cholesky() else {
            push_skipped_epoch(
                &mut epochs,
                &mut accepted_flags,
                &xv_pred,
                &cov_pred,
                phi_full,
            );
            xv = xv_pred;
            cov = cov_pred;
            state_cur = new_state;
            continue;
        };

        let innov_inv = innov_chol.inverse();
        let gate_val = (&residual.transpose() * &innov_inv * &residual)[(0, 0)];

        if gate_val > chi2_gate {
            push_skipped_epoch(
                &mut epochs,
                &mut accepted_flags,
                &xv_pred,
                &cov_pred,
                phi_full,
            );
            xv = xv_pred;
            cov = cov_pred;
            state_cur = new_state;
            continue;
        }

        // Kalman gain: K = P_pred H^T S^{-1}
        let k_gain = &cov_pred * h_full.transpose() * &innov_inv;

        // State update
        let xv_upd = &xv_pred + &k_gain * &residual;

        // Joseph-form covariance update for numerical stability:
        // P = (I - KH) P_pred (I - KH)^T + K R K^T
        let ikh = DMatrix::identity(dim, dim) - &k_gain * &h_full;
        let mut cov_upd =
            &ikh * &cov_pred * ikh.transpose() + &k_gain * &r_mat * k_gain.transpose();

        // Enforce symmetry to prevent numerical drift.
        cov_upd = (&cov_upd + cov_upd.transpose()) * 0.5;

        epochs.push(FilterEpoch {
            x_filtered: xv_upd.clone(),
            p_filtered: cov_upd.clone(),
            x_predicted: xv_pred,
            p_predicted: cov_pred,
            phi: phi_full,
        });
        accepted_flags.push(true);

        xv = xv_upd;
        cov = cov_upd;
        state_cur = vec_to_state(&xv, &new_state, ng);
    }

    (epochs, accepted_flags, xv, cov)
}

/// RTS backward smoother pass.
fn backward_pass(
    epochs: &[FilterEpoch],
    xv_final: DVector<f64>,
    cov_final: DMatrix<f64>,
    dim: usize,
) -> (Vec<DVector<f64>>, Vec<DMatrix<f64>>) {
    let n_obs = epochs.len();
    let mut xv_smooth = xv_final;
    let mut cov_smooth = cov_final;

    let mut xv_smoothed: Vec<DVector<f64>> = vec![DVector::zeros(dim); n_obs];
    let mut cov_smoothed: Vec<DMatrix<f64>> = vec![DMatrix::zeros(dim, dim); n_obs];
    xv_smoothed[n_obs - 1] = xv_smooth.clone();
    cov_smoothed[n_obs - 1] = cov_smooth.clone();

    for idx in (0..n_obs - 1).rev() {
        let ep_next = &epochs[idx + 1];

        // Smoother gain: G_k = P_f_k * Phi_{k+1}^T * P_pred_{k+1}^{-1}
        let Some(pred_chol) = ep_next.p_predicted.clone().cholesky() else {
            xv_smoothed[idx] = epochs[idx].x_filtered.clone();
            cov_smoothed[idx] = epochs[idx].p_filtered.clone();
            continue;
        };
        let pred_inv = pred_chol.inverse();

        let gain = &epochs[idx].p_filtered * ep_next.phi.transpose() * &pred_inv;

        xv_smooth = &epochs[idx].x_filtered + &gain * (&xv_smooth - &ep_next.x_predicted);

        // Joseph-form smoother covariance for numerical stability:
        // P_s = (I - G*Phi) * P_f * (I - G*Phi)^T + G * P_s_next * G^T
        let igp = DMatrix::identity(dim, dim) - &gain * &ep_next.phi;
        cov_smooth = &igp * &epochs[idx].p_filtered * igp.transpose()
            + &gain * &cov_smooth * gain.transpose();

        // Enforce symmetry.
        cov_smooth = (&cov_smooth + cov_smooth.transpose()) * 0.5;

        xv_smoothed[idx] = xv_smooth.clone();
        cov_smoothed[idx] = cov_smooth.clone();
    }

    (xv_smoothed, cov_smoothed)
}

/// Build the final [`OrbitFit`] result from smoothed states.
#[expect(
    clippy::too_many_arguments,
    reason = "assembles final result from filter outputs"
)]
fn build_result(
    sorted: &[AstrometricObservation],
    initial_state: &State<Equatorial, SSB>,
    epochs: &[FilterEpoch],
    accepted_flags: &[bool],
    xv_smoothed: &[DVector<f64>],
    cov_smoothed: &[DMatrix<f64>],
    ng: &mut Option<NonGravModel>,
    non_grav: Option<&NonGravModel>,
    dim: usize,
) -> KeteResult<OrbitFit> {
    let n_obs = epochs.len();

    // Report the smoothed state at the first observation epoch
    // (where the smoother has the best estimate).
    let final_x = &xv_smoothed[0];
    let final_p = &cov_smoothed[0];

    let mut result_state = vec_to_state(final_x, initial_state, ng);
    result_state.epoch = sorted[0].epoch();

    let uncertain = UncertainState {
        state: result_state.into(),
        cov_matrix: final_p.clone(),
        non_grav: ng.clone(),
    };

    // -- Compute final residuals from smoothed states --------------
    let mut residuals: Vec<DVector<f64>> = Vec::with_capacity(n_obs);
    let mut chi2_sum = 0.0;
    let mut n_meas = 0_usize;

    for (idx, observation) in sorted.iter().enumerate() {
        if !accepted_flags[idx] {
            let md = observation.measurement_dim();
            residuals.push(DVector::from_element(md, f64::NAN));
            continue;
        }

        let xs = &xv_smoothed[idx];
        let mut recon_state = vec_to_state(xs, initial_state, &mut non_grav.cloned());
        recon_state.epoch = observation.epoch();

        let observer_state = observation.observer()?;
        let obj_lt_ssb = apply_light_time(&recon_state, &observer_state)?;
        let res = observation.residual_from_corrected(&obj_lt_ssb)?;
        // Use the full base weight matrix so correlated-axis observations
        // contribute correctly to chi^2.
        let w_base = observation.base_weight_matrix();
        let wr = &w_base * &res;
        chi2_sum += res.dot(&wr);
        n_meas += res.len();
        residuals.push(res);
    }

    let dof = if n_meas > dim { n_meas - dim } else { 1 };
    let rms = (chi2_sum / dof as f64).sqrt();

    Ok(OrbitFit {
        uncertain_state: uncertain,
        residuals,
        observations: sorted.to_vec(),
        included: accepted_flags.to_vec(),
        rms,
        converged: true,
    })
}

/// Sort observations by epoch, filtering out observations whose observer
/// state cannot be resolved or contains non-finite components.
fn sort_by_epoch(obs: &[AstrometricObservation]) -> Vec<AstrometricObservation> {
    let mut sorted: Vec<AstrometricObservation> = obs
        .iter()
        .filter(|ob| {
            let Ok(st) = ob.observer() else { return false };
            let pos_ok: bool = st.pos.into_iter().all(|val: f64| val.is_finite());
            let vel_ok: bool = st.vel.into_iter().all(|val: f64| val.is_finite());
            pos_ok && vel_ok
        })
        .cloned()
        .collect();
    sorted.sort_by(|aa, bb| {
        aa.epoch()
            .jd
            .partial_cmp(&bb.epoch().jd)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    sorted
}

#[cfg(test)]
mod tests {
    use super::*;
    use kete_core::constants::GMS;
    use kete_core::desigs::Desig;
    use kete_core::frames::Equatorial;
    use kete_core::kepler::light_time_correct;
    use kete_core::prelude::State;
    use kete_core::time::{TDB, Time};
    use kete_spice::prelude::LOADED_SPK;
    use kete_spice::propagation::propagate_n_body_spk;
    use kete_spice::test_data::ensure_test_spk;

    /// Build a simple SSB-centered state.
    fn make_state(pos: [f64; 3], vel: [f64; 3], jd: f64) -> State<Equatorial, SSB> {
        State { desig: Desig::Empty, epoch: jd.into(), pos: pos.into(), vel: vel.into(), center: SSB }
    }

    /// Earth-like observer on a circular orbit at 1 AU with slight inclination.
    fn earth_observer(jd: f64) -> ([f64; 3], [f64; 3]) {
        let v_earth = (GMS / 1.0_f64).sqrt();
        let period = 2.0 * std::f64::consts::PI / v_earth;
        let angle = (jd - 2_460_000.5) / period * 2.0 * std::f64::consts::PI;
        let incl: f64 = 0.05;
        let pos = [
            angle.cos(),
            angle.sin() * incl.cos(),
            angle.sin() * incl.sin(),
        ];
        let vel = [
            -v_earth * angle.sin(),
            v_earth * angle.cos() * incl.cos(),
            v_earth * angle.cos() * incl.sin(),
        ];
        (pos, vel)
    }

    /// Generate synthetic optical observations using N-body propagation.
    fn synth_observations(
        true_state: &State<Equatorial, SSB>,
        epochs: &[f64],
        sigma: f64,
    ) -> Vec<AstrometricObservation> {
        let mut observations = Vec::new();
        for &jd in epochs {
            let (obs_pos, obs_vel) = earth_observer(jd);
            let observer = make_state(obs_pos, obs_vel, jd);

            let obj_at =
                propagate_n_body_spk(true_state.clone(), Time::<TDB>::new(jd), false, None)
                    .unwrap();

            let spk = LOADED_SPK.try_read().unwrap();
            let sun_at = spk.try_to_sun(obj_at.clone().into()).unwrap();
            let obs_helio = observer.pos - obj_at.pos + sun_at.pos;
            let obj_lt_sun = light_time_correct(&sun_at, &obs_helio).unwrap();
            let obj_lt = spk.try_to_ssb(obj_lt_sun.into()).unwrap();
            let (ra, dec) = (obj_lt.pos - observer.pos).to_ra_dec();

            observations.push(AstrometricObservation::Optical {
                observer,
                ra,
                dec,
                sigma_ra: sigma,
                sigma_dec: sigma,
                sigma_corr: 0.0,
                time_sigma: 0.0,
                is_occultation: false,
            });
        }
        observations
    }

    /// Providing zero observations returns an error.
    #[test]
    fn test_empty_obs() {
        let state = make_state([1.0, 0.0, 0.0], [0.0, 0.01, 0.0], 2_451_545.0);
        let result = fit_orbit_filter(&state, &[], true, None, 16.0, 0.0);
        assert!(result.is_err());
    }

    /// EKF+RTS should recover a circular orbit from synthetic observations
    /// when started from a slightly perturbed initial state.
    #[test]
    fn test_filter_circular_orbit() {
        ensure_test_spk();

        let radius = 1.5;
        let vel = (GMS / radius).sqrt();
        let true_state = make_state([radius, 0.0, 0.0], [0.0, vel, 0.0], 2_460_000.5);

        // 20 observations over 120 days.
        let epochs: Vec<f64> = (0..20).map(|i| 2_460_000.5 + f64::from(i) * 6.0).collect();
        let sigma = 1e-6; // ~0.2 arcsec
        let observations = synth_observations(&true_state, &epochs, sigma);

        // Start from a 2% position perturbation.
        let perturbed = make_state(
            [radius * 1.02, 0.0, 0.0],
            [0.0, vel * 0.98, 0.0],
            2_460_000.5,
        );

        let fit = fit_orbit_filter(
            &perturbed,
            &observations,
            false,
            None,
            100.0, // generous gate
            0.0,   // no process noise
        )
        .unwrap();

        let n_accepted = fit.included.iter().filter(|&&v| v).count();

        // Should recover the orbit well.
        let pos_err = (fit.uncertain_state.state.pos - true_state.pos).norm();
        let vel_err = (fit.uncertain_state.state.vel - true_state.vel).norm();

        assert!(pos_err < 1e-3, "Position error {pos_err:.6e} too large");
        assert!(vel_err < 1e-4, "Velocity error {vel_err:.6e} too large");

        // All observations should be accepted.
        assert_eq!(n_accepted, 20, "Expected all 20 observations accepted");

        // Covariance diagonal should be positive.
        for idx in 0..6 {
            assert!(
                fit.uncertain_state.cov_matrix[(idx, idx)] > 0.0,
                "Cov diagonal [{idx},{idx}] not positive"
            );
        }
    }

    /// EKF+RTS on a longer arc (2 years) with a larger perturbation.
    #[test]
    fn test_filter_long_arc() {
        ensure_test_spk();

        let radius = 2.0;
        let vel = (GMS / radius).sqrt();
        let true_state = make_state([radius, 0.0, 0.0], [0.0, vel, 0.0], 2_460_000.5);

        // 30 observations over 720 days.
        let epochs: Vec<f64> = (0..30).map(|i| 2_460_000.5 + f64::from(i) * 24.0).collect();
        let sigma = 1e-6;
        let observations = synth_observations(&true_state, &epochs, sigma);

        // 3% position perturbation, 2% velocity.
        let perturbed = make_state(
            [radius * 1.03, 0.0, 0.0],
            [0.0, vel * 0.98, 0.0],
            2_460_000.5,
        );

        let fit = fit_orbit_filter(
            &perturbed,
            &observations,
            false,
            None,
            400.0, // generous gate
            0.0,
        )
        .unwrap();

        let pos_err = (fit.uncertain_state.state.pos - true_state.pos).norm();

        assert!(
            pos_err < 1e-2,
            "Long-arc position error {pos_err:.6e} too large"
        );
    }
}
