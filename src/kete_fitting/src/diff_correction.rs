//! Batch least-squares differential correction with chained STM propagation.
//!
//! The solver accumulates normal equations at the reference epoch by chaining
//! the state transition matrix forward through the sorted observation sequence.
//! This avoids STM inversion and gives the same result as a sequential
//! information filter at the same computational cost.

use crate::obs::{Observation, two_body_lt_state};
use kete_core::constants::GravParams;
use kete_core::frames::Equatorial;
use kete_core::prelude::{Error, KeteResult, State};
use kete_core::propagation::{NonGravModel, compute_state_transition};
use nalgebra::{DMatrix, DVector};

/// Result of orbit determination via batch least squares.
#[derive(Debug, Clone)]
pub struct OrbitFit {
    /// Best-fit state at the reference epoch.
    pub state: State<Equatorial>,

    /// Covariance matrix at the reference epoch, (6+Np) x (6+Np).
    /// When non-grav parameters are fitted, Np > 0 and the lower-right
    /// block contains the formal uncertainties of the non-grav parameters.
    pub covariance: DMatrix<f64>,

    /// Post-fit residuals in time-sorted order. Each entry has as many
    /// elements as the measurement dimension of that observation.
    pub residuals: Vec<DVector<f64>>,

    /// Whether each observation was included (true) or rejected by
    /// outlier gating (false). Time-sorted order.
    pub included: Vec<bool>,

    /// Reduced weighted RMS of residuals (included observations only).
    /// Divided by degrees of freedom (`n_measurements` - `n_params`).
    pub rms: f64,

    /// Fitted non-gravitational model (if any). When non-grav parameters
    /// are included in the solve-for state, this contains the updated
    /// model with fitted parameter values.
    pub non_grav: Option<NonGravModel>,

    /// Whether the solver achieved strict convergence (correction norm
    /// dropped below `tol`). When `false` the fit is the best found
    /// within the iteration limit but may not be fully converged.
    pub converged: bool,
}

/// Run batch least-squares differential correction.
///
/// # Arguments
/// * `initial_state` - Initial guess for the object state at the reference
///   epoch. The epoch of this state is the reference epoch for all
///   normal-equation accumulation.
/// * `obs` - Observations (any order; they are sorted internally).
/// * `massive_obj` - Gravitating bodies for STM propagation.
/// * `non_grav` - Optional non-gravitational model.
/// * `max_iter` - Maximum number of differential-correction iterations.
/// * `tol` - Convergence tolerance on the state correction norm (AU for
///   position, AU/day for velocity).
///
/// # Errors
/// Fails if propagation fails or the normal matrix is singular.
pub fn differential_correction(
    initial_state: &State<Equatorial>,
    obs: &[Observation],
    massive_obj: &[GravParams],
    non_grav: Option<&NonGravModel>,
    max_iter: usize,
    tol: f64,
) -> KeteResult<OrbitFit> {
    if obs.is_empty() {
        return Err(Error::ValueError("No observations provided".into()));
    }
    let sorted = sort_by_epoch(obs);
    let included = vec![true; sorted.len()];
    let fit = iterate_to_convergence(
        initial_state,
        &sorted,
        &included,
        massive_obj,
        non_grav.cloned(),
        max_iter,
        tol,
    )?;
    if !fit.converged {
        return Err(Error::Convergence(format!(
            "Differential correction did not converge in {max_iter} iterations"
        )));
    }
    Ok(fit)
}

/// Run arc-expanding differential correction with chi-squared outlier rejection.
///
/// For arcs longer than 180 days, progressively wider time windows are
/// fitted around the reference epoch so that each stage bootstraps from
/// the previous converged solution.  The final pass fits the full arc
/// and re-evaluates all observations for outlier rejection.
///
/// Short arcs (<= 180 days) skip the expansion and go straight to a
/// single full-arc fit with rejection.
///
/// # Errors
/// Fails if any internal propagation or solve fails.
pub fn differential_correction_with_rejection(
    initial_state: &State<Equatorial>,
    obs: &[Observation],
    massive_obj: &[GravParams],
    non_grav: Option<&NonGravModel>,
    max_iter: usize,
    tol: f64,
    chi2_threshold: f64,
    max_reject_passes: usize,
) -> KeteResult<OrbitFit> {
    if obs.is_empty() {
        return Err(Error::ValueError("No observations provided".into()));
    }
    let sorted = sort_by_epoch(obs);
    let ref_jd = initial_state.epoch.jd;

    // Compute arc span.  The `obs.is_empty()` guard above ensures these
    // are safe.
    let jd_first = sorted[0].epoch().jd;
    let jd_last = sorted[sorted.len() - 1].epoch().jd;
    let arc_span = jd_last - jd_first;

    // Build adaptive window schedule.
    // Short arcs: just fit everything. Medium: seed +/-90, then full.
    // Long (>720 d): seed +/-90, intermediate +/-half_arc, full.
    let windows: Vec<f64> = if arc_span <= 180.0 {
        vec![f64::INFINITY]
    } else if arc_span <= 720.0 {
        vec![90.0, f64::INFINITY]
    } else {
        vec![90.0, arc_span / 2.0, f64::INFINITY]
    };

    let mut state = initial_state.clone();
    let mut ng = non_grav.cloned();

    // Expansion stages: converge + reject on each window.
    for &radius in &windows[..windows.len() - 1] {
        let included = select_obs_within_window(&sorted, ref_jd, radius);
        let n_in_window = included.iter().filter(|&&v| v).count();
        if n_in_window < 4 {
            continue; // too few observations in this window
        }
        if let Ok(fit) = solve_with_rejection(
            &state,
            &sorted,
            &included,
            massive_obj,
            ng.clone(),
            max_iter,
            tol,
            chi2_threshold,
            max_reject_passes,
        ) {
            state = fit.state;
            ng = fit.non_grav;
        }
        // On error: keep previous state, try the next wider window.
    }

    // Final full-arc pass: re-include all observations and reject anew.
    let included = vec![true; sorted.len()];
    solve_with_rejection(
        &state,
        &sorted,
        &included,
        massive_obj,
        ng,
        max_iter,
        tol,
        chi2_threshold,
        max_reject_passes,
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
/// First converges using `solve_once`, then iteratively rejects the single
/// worst outlier exceeding `chi2_threshold` and re-converges.
fn solve_with_rejection(
    initial_state: &State<Equatorial>,
    sorted_obs: &[Observation],
    included: &[bool],
    massive_obj: &[GravParams],
    non_grav: Option<NonGravModel>,
    max_iter: usize,
    tol: f64,
    chi2_threshold: f64,
    max_reject_passes: usize,
) -> KeteResult<OrbitFit> {
    let mut fit = iterate_to_convergence(
        initial_state,
        sorted_obs,
        included,
        massive_obj,
        non_grav,
        max_iter,
        tol,
    )?;

    // Rejection loop: remove one outlier at a time from the included set.
    let np = fit.non_grav.as_ref().map_or(0, NonGravModel::n_free_params);
    let min_included = (6 + np).max(4);

    for _ in 0..max_reject_passes {
        let n_included = fit.included.iter().filter(|&&inc| inc).count();
        if n_included <= min_included {
            break;
        }

        // Find the single worst included observation by chi-squared.
        let mut worst_idx = None;
        let mut worst_chi2 = chi2_threshold;
        for (i, res) in fit.residuals.iter().enumerate() {
            if !fit.included[i] {
                continue;
            }
            let w = sorted_obs[i].weights();
            let chi2: f64 = res.iter().zip(w.iter()).map(|(r, wi)| r * r * wi).sum();
            if chi2 > worst_chi2 {
                worst_chi2 = chi2;
                worst_idx = Some(i);
            }
        }

        let Some(idx) = worst_idx else {
            break;
        };
        fit.included[idx] = false;

        fit = iterate_to_convergence(
            &fit.state,
            sorted_obs,
            &fit.included,
            massive_obj,
            fit.non_grav.clone(),
            max_iter,
            tol,
        )?;
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
/// Each iteration re-linearises at the current state, solves the damped
/// normal equations `(N + lambda * diag(N)) dx = b`, limits the step magnitude,
/// and moves forward unconditionally.  Re-linearising every iteration is
/// essential: the step limiter caps *magnitude* but not *direction*, so
/// recycling a stale Jacobian would repeatedly propose the same capped
/// step and stall.
///
/// Lambda is adjusted heuristically: decreased when chi-squared improves,
/// increased when it worsens.  This steers the solver between
/// Gauss-Newton (fast near the solution) and steepest descent (safe far
/// from it).
fn iterate_to_convergence(
    initial_state: &State<Equatorial>,
    obs: &[Observation],
    included: &[bool],
    massive_obj: &[GravParams],
    mut non_grav: Option<NonGravModel>,
    max_iter: usize,
    tol: f64,
) -> KeteResult<OrbitFit> {
    let mut state_epoch = initial_state.clone();
    let mut lambda = 0.0_f64;
    let mut prev_chi2 = f64::MAX;

    // Cache from the last iteration so we don't have to re-propagate
    // the entire arc when the loop exhausts max_iter.
    let mut last_info_mat = None;

    for _iter in 0..max_iter {
        let (info_mat, rhs_vec, chi2) = accumulate_normal_equations(
            &state_epoch,
            obs,
            included,
            massive_obj,
            non_grav.as_ref(),
        )?;

        // Adaptive LM damping: relax on improvement, tighten on worsening.
        if prev_chi2 < f64::MAX {
            if chi2 < prev_chi2 {
                lambda *= 0.1;
            } else {
                lambda = if lambda < 1e-6 { 1.0 } else { lambda * 10.0 };
            }
        }
        prev_chi2 = chi2;

        let dx = solve_damped(&info_mat, &rhs_vec, lambda)?;
        let dx = limit_correction(dx);

        let converged = dx.norm() < tol;

        // Save the information matrix *before* applying the correction,
        // since it was linearised at the current state_epoch.
        last_info_mat = Some(info_mat);

        apply_correction(&mut state_epoch, &dx, &mut non_grav);

        if converged {
            // Re-compute residuals at the newly corrected state.
            let covariance = svd_pseudo_inverse(last_info_mat.as_ref().unwrap(), 1e-14)?;
            let residuals = compute_residuals(&state_epoch, obs, massive_obj, non_grav.as_ref())?;
            let n_params = 6 + n_nongrav_params(non_grav.as_ref());
            let rms = weighted_rms(&residuals, obs, included, n_params);
            return Ok(OrbitFit {
                state: state_epoch,
                covariance,
                residuals,
                included: included.to_vec(),
                rms,
                non_grav,
                converged: true,
            });
        }
    }

    // Did not converge -- return best-effort result with converged=false.
    // This allows callers (e.g. the arc-expanding loop) to use the
    // partially-converged state as a seed for the next stage.
    //
    // Reuse the cached information matrix from the last iteration instead
    // of re-computing it (saves a full arc propagation).
    let n_params = 6 + n_nongrav_params(non_grav.as_ref());
    let info_mat = match last_info_mat {
        Some(m) => m,
        None => {
            // max_iter == 0: never entered the loop.
            accumulate_normal_equations(
                &state_epoch,
                obs,
                included,
                massive_obj,
                non_grav.as_ref(),
            )?
            .0
        }
    };
    let covariance = svd_pseudo_inverse(&info_mat, 1e-14)?;
    let residuals = compute_residuals(&state_epoch, obs, massive_obj, non_grav.as_ref())?;
    let rms = weighted_rms(&residuals, obs, included, n_params);
    Ok(OrbitFit {
        state: state_epoch,
        covariance,
        residuals,
        included: included.to_vec(),
        rms,
        non_grav,
        converged: false,
    })
}

/// Accumulate the weighted normal equations for one linearisation pass.
///
/// Returns `(info_mat, rhs_vec, chi2)` where `info_mat` is the
/// (6+Np) x (6+Np) information matrix, `rhs_vec` is the right-hand
/// side, and `chi2` is the current weighted sum of squared residuals.
fn accumulate_normal_equations(
    state_epoch: &State<Equatorial>,
    obs: &[Observation],
    included: &[bool],
    massive_obj: &[GravParams],
    non_grav: Option<&NonGravModel>,
) -> KeteResult<(DMatrix<f64>, DVector<f64>, f64)> {
    let np = n_nongrav_params(non_grav);
    let d = 6 + np;

    let mut n_mat = DMatrix::<f64>::zeros(d, d);
    let mut b_vec = DVector::<f64>::zeros(d);
    let mut chi2 = 0.0;

    // Cumulative STM: 6 x D.
    // Initialized to [I_6 | 0_{6 x Np}].
    let mut phi_cum = DMatrix::<f64>::zeros(6, d);
    for i in 0..6 {
        phi_cum[(i, i)] = 1.0;
    }

    let mut state_cur = state_epoch.clone();

    for (i, observation) in obs.iter().enumerate() {
        let obs_epoch = observation.epoch();

        // Propagate from current state to observation epoch via STM.
        if (obs_epoch.jd - state_cur.epoch.jd).abs() > 1e-12 {
            let (new_state, phi_k) =
                compute_state_transition(&state_cur, obs_epoch, massive_obj, non_grav.cloned())?;

            // phi_k is 6 x (6 + Np).
            let phi_state = phi_k.columns(0, 6).clone_owned(); // 6 x 6

            // Chain the state block: Phi_cum[:, 0:6] = Phi_state * Phi_cum[:, 0:6]
            let new_state_cols = &phi_state * phi_cum.columns(0, 6);
            phi_cum.columns_mut(0, 6).copy_from(&new_state_cols);

            // Chain the parameter block (if any):
            // Phi_cum[:, 6:] = Phi_state * Phi_cum[:, 6:] + Phi_param
            if np > 0 {
                let phi_param = phi_k.columns(6, np).clone_owned(); // 6 x Np
                let new_param_cols = &phi_state * phi_cum.columns(6, np) + &phi_param;
                phi_cum.columns_mut(6, np).copy_from(&new_param_cols);
            }

            state_cur = new_state;
        }

        // Skip excluded observations from the normal equations, but still
        // propagate through them so the STM chain stays correct.
        if !included[i] {
            continue;
        }

        // Apply two-body light-time correction.
        let obs_pos = observation.observer();
        let obj_lt = two_body_lt_state(&state_cur, obs_pos)?;

        let (residual, _predicted) = observation.residual(&state_cur)?;

        // Local geometric partials (m x 6).
        let h_local = observation.partials(&obj_lt);

        // Map to epoch: H_epoch = H_local * Phi_cum  (m x D).
        let h_epoch = &h_local * &phi_cum;

        // Weight vector.
        let w = observation.weights();

        // Accumulate chi-squared.
        let m = observation.measurement_dim();
        for k in 0..m {
            chi2 += residual[k] * residual[k] * w[k];
        }

        // Accumulate normal matrix and RHS via weighted outer products:
        //   N += H^T W H,  b += H^T W r
        // Build sqrt(W) * H and sqrt(W) * r for efficient rank-m update.
        let mut hw = h_epoch.clone(); // m x d
        let mut wr = residual.clone(); // m x 1
        for k in 0..m {
            let sw = w[k].sqrt();
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

    Ok((n_mat, b_vec, chi2))
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
/// Position is limited to 0.5 AU and velocity to 0.005 AU/day per iteration.
/// Non-grav parameters (indices 6..) are left uncapped since the LM damping
/// already regulates them.
fn limit_correction(mut dx: DVector<f64>) -> DVector<f64> {
    const MAX_POS: f64 = 0.5; // AU
    const MAX_VEL: f64 = 0.005; // AU/day

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
/// NOTE: This reuses `compute_state_transition` for propagation even
/// though only the state (not the STM) is needed.  The STM integrator
/// carries a 30-dim augmented state vs 6-dim for plain N-body, making
/// this ~5x more expensive than necessary.  A dedicated propagator
/// accepting a custom mass list would eliminate this overhead.
fn compute_residuals(
    state_epoch: &State<Equatorial>,
    obs: &[Observation],
    massive_obj: &[GravParams],
    non_grav: Option<&NonGravModel>,
) -> KeteResult<Vec<DVector<f64>>> {
    let mut residuals = Vec::with_capacity(obs.len());
    let mut state_cur = state_epoch.clone();

    for observation in obs {
        let obs_epoch = observation.epoch();

        // Propagate to observation epoch.
        if (obs_epoch.jd - state_cur.epoch.jd).abs() > 1e-12 {
            let (new_state, _phi) =
                compute_state_transition(&state_cur, obs_epoch, massive_obj, non_grav.cloned())?;
            state_cur = new_state;
        }

        let (res, _pred) = observation.residual(&state_cur)?;
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
    use kete_core::constants::{GMS, GravParams};
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

            let obj_lt = two_body_lt_state(&obj_at, &observer).unwrap();
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
        let v_earth = (GMS / 1.0_f64).sqrt(); // ~0.0172 AU/day
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
    fn test_differential_correction_two_body() {
        // True orbit: circular at 1.5 AU.
        let r = 1.5;
        let v = (GMS / r).sqrt();
        let true_state = make_state([r, 0.0, 0.0], [0.0, v, 0.0], 2460000.5);

        // Generate 10 observations over 60 days.
        let epochs: Vec<f64> = (0..10).map(|i| 2460000.5 + f64::from(i) * 6.0).collect();
        let sigma = 1e-6; // ~0.2 arcsec
        let observations = synth_observations(&true_state, &epochs, earth_observer, sigma, None);

        // Perturbed initial state (5% error in position, 3% in velocity).
        let perturbed = make_state([r * 1.05, 0.0, 0.0], [0.0, v * 0.97, 0.0], 2460000.5);

        let massive = GravParams::planets();

        let fit =
            differential_correction(&perturbed, &observations, &massive, None, 20, 1e-8).unwrap();

        // Check that the fit converged near the true state.
        let pos_err = (fit.state.pos - true_state.pos).norm();
        let vel_err = (fit.state.vel - true_state.vel).norm();

        // Should recover position to < 1e-4 AU and velocity to < 1e-5 AU/day.
        assert!(pos_err < 1e-4, "Position error {pos_err:.6e} too large");
        assert!(vel_err < 1e-5, "Velocity error {vel_err:.6e} too large");

        // RMS should be very small (near-perfect synthetic data).
        assert!(fit.rms < 1e-3, "Weighted RMS {:.6e} too large", fit.rms);

        // Covariance should be positive definite (check diagonal > 0).
        for i in 0..6 {
            assert!(
                fit.covariance[(i, i)] > 0.0,
                "Covariance diagonal [{i},{i}] = {} not positive",
                fit.covariance[(i, i)]
            );
        }
    }

    #[test]
    fn test_differential_correction_elliptical() {
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

        let massive = GravParams::planets();

        let fit =
            differential_correction(&perturbed, &observations, &massive, None, 20, 1e-8).unwrap();

        let pos_err = (fit.state.pos - true_state.pos).norm();

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

        let massive = GravParams::planets();

        let fit = differential_correction_with_rejection(
            &true_state, // start from true state to ensure convergence
            &observations,
            &massive,
            None,
            20,
            1e-8,
            9.0, // chi2 threshold
            3,
        )
        .unwrap();

        // At least one observation should have been rejected.
        let n_rejected = fit.included.iter().filter(|&&inc| !inc).count();
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
        let true_a2 = 1e-8; // AU/day^2, large enough to be detectable
        let true_ng = NonGravModel::new_jpl_comet_default(0.0, true_a2, 0.0);

        // Generate 15 observations over 90 days with the non-grav model.
        let epochs: Vec<f64> = (0..15).map(|i| 2460000.5 + f64::from(i) * 6.0).collect();
        let sigma = 1e-7; // tight observations
        let observations =
            synth_observations(&true_state, &epochs, earth_observer, sigma, Some(&true_ng));

        // Start from true state + non-grav model with a2=0 and fit.
        let init_ng = NonGravModel::new_jpl_comet_default(0.0, 0.0, 0.0);
        let massive = GravParams::planets();

        let fit = differential_correction(
            &true_state,
            &observations,
            &massive,
            Some(&init_ng),
            30,
            1e-10,
        )
        .unwrap();

        // The fitted non-grav model should exist and have a2 close to true_a2.
        let fitted_ng = fit.non_grav.as_ref().expect("non_grav should be present");
        let fitted_params = fitted_ng.get_free_params();
        let a2_err = (fitted_params[1] - true_a2).abs();
        assert!(
            a2_err < true_a2 * 0.1,
            "a2 error {a2_err:.6e} too large (true={true_a2:.6e}, fitted={:.6e})",
            fitted_params[1]
        );

        // Covariance should be 9x9.
        assert_eq!(fit.covariance.nrows(), 9, "Expected 9x9 covariance");
        assert_eq!(fit.covariance.ncols(), 9, "Expected 9x9 covariance");

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
        let massive = GravParams::planets();

        let fit = differential_correction(
            &true_state,
            &observations,
            &massive,
            Some(&init_ng),
            30,
            1e-10,
        )
        .unwrap();

        let fitted_ng = fit.non_grav.as_ref().expect("non_grav should be present");
        let fitted_params = fitted_ng.get_free_params();
        let beta_err = (fitted_params[0] - true_beta).abs();
        assert!(
            beta_err < true_beta * 0.1,
            "beta error {beta_err:.6e} too large (true={true_beta:.6e}, fitted={:.6e})",
            fitted_params[0]
        );

        // Covariance should be 7x7.
        assert_eq!(fit.covariance.nrows(), 7, "Expected 7x7 covariance");
        assert_eq!(fit.covariance.ncols(), 7, "Expected 7x7 covariance");

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

        let massive = GravParams::planets();

        let fit = differential_correction_with_rejection(
            &perturbed,
            &observations,
            &massive,
            None,
            50,
            1e-8,
            9.0,
            3,
        )
        .unwrap();

        let pos_err = (fit.state.pos - true_state.pos).norm();
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

        let massive = GravParams::planets();

        let fit = differential_correction_with_rejection(
            &true_state,
            &observations,
            &massive,
            None,
            50,
            1e-8,
            9.0,
            5,
        )
        .unwrap();

        // The corrupted observation should be rejected.
        // Sort order matches input (already sorted by epoch).
        let n_rejected = fit.included.iter().filter(|&&inc| !inc).count();
        assert!(
            n_rejected >= 1,
            "Expected at least 1 rejection, got {n_rejected}"
        );

        // Orbit should still be good.
        let pos_err = (fit.state.pos - true_state.pos).norm();
        assert!(
            pos_err < 1e-3,
            "Rejection re-inclusion: pos error {pos_err:.6e} too large"
        );
    }
}
