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

    /// Weighted RMS of residuals (included observations only).
    pub rms: f64,

    /// Fitted non-gravitational model (if any). When non-grav parameters
    /// are included in the solve-for state, this contains the updated
    /// model with fitted parameter values.
    pub non_grav: Option<NonGravModel>,
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
    solve_once(
        initial_state,
        &sorted,
        &included,
        massive_obj,
        non_grav.cloned(),
        max_iter,
        tol,
    )
}

/// Run differential correction with chi-squared outlier rejection.
///
/// First converges using all observations, then rejects outliers and
/// re-converges. The `chi2_threshold` controls the rejection threshold
/// (default suggestion: 9.0 for optical, 8.0 for 1-D radar).
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
    let included = vec![true; sorted.len()];

    // First pass: converge with all observations.
    let mut fit = solve_once(
        initial_state,
        &sorted,
        &included,
        massive_obj,
        non_grav.cloned(),
        max_iter,
        tol,
    )?;

    // Rejection passes.
    for _ in 0..max_reject_passes {
        let mut any_rejected = false;
        for (i, res) in fit.residuals.iter().enumerate() {
            if !fit.included[i] {
                continue;
            }
            let w = sorted[i].weights();
            let chi2: f64 = res.iter().zip(w.iter()).map(|(r, wi)| r * r * wi).sum();
            if chi2 > chi2_threshold {
                fit.included[i] = false;
                any_rejected = true;
            }
        }
        if !any_rejected {
            break;
        }

        // Re-solve from current best state with updated inclusion mask.
        fit = solve_once(
            &fit.state,
            &sorted,
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

/// Run the iterative convergence loop.
///
/// Iterates `one_iteration` -> `apply_correction` -> check tolerance.
/// On convergence, computes the covariance and post-fit residuals.
fn solve_once(
    initial_state: &State<Equatorial>,
    obs: &[Observation],
    included: &[bool],
    massive_obj: &[GravParams],
    mut non_grav: Option<NonGravModel>,
    max_iter: usize,
    tol: f64,
) -> KeteResult<OrbitFit> {
    let mut state_epoch = initial_state.clone();

    for _iter in 0..max_iter {
        let (dx, n_mat) =
            one_iteration(&state_epoch, obs, included, massive_obj, non_grav.as_ref())?;
        apply_correction(&mut state_epoch, &dx, &mut non_grav);

        if dx.norm() < tol {
            let covariance = svd_pseudo_inverse(&n_mat, 1e-14)?;
            let residuals = compute_residuals(&state_epoch, obs, massive_obj, non_grav.as_ref())?;
            let rms = weighted_rms(&residuals, obs, included);
            return Ok(OrbitFit {
                state: state_epoch,
                covariance,
                residuals,
                included: included.to_vec(),
                rms,
                non_grav,
            });
        }
    }

    Err(Error::Convergence(format!(
        "Differential correction did not converge in {max_iter} iterations"
    )))
}

/// Perform one iteration of the batch least squares.
///
/// Returns `(dx, N)` where `dx` is the D-dimensional state correction vector
/// (D = 6 + Np) and `N` is the normal matrix (for covariance on convergence).
fn one_iteration(
    state_epoch: &State<Equatorial>,
    obs: &[Observation],
    included: &[bool],
    massive_obj: &[GravParams],
    non_grav: Option<&NonGravModel>,
) -> KeteResult<(DVector<f64>, DMatrix<f64>)> {
    let np = n_nongrav_params(non_grav);
    let d = 6 + np;

    let mut n_mat = DMatrix::<f64>::zeros(d, d);
    let mut b_vec = DVector::<f64>::zeros(d);

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

        // Accumulate normal equations: N += H^T W H, b += H^T W r.
        // W is diagonal, stored as a DVector.
        let m = observation.measurement_dim();
        for ii in 0..d {
            for jj in 0..d {
                for k in 0..m {
                    n_mat[(ii, jj)] += h_epoch[(k, ii)] * w[k] * h_epoch[(k, jj)];
                }
            }
            for k in 0..m {
                b_vec[ii] += h_epoch[(k, ii)] * w[k] * residual[k];
            }
        }
    }

    // Solve: dx = N^{-1} * b via SVD (robust to near-singular N).
    let svd = n_mat.clone().svd(true, true);
    let dx = svd
        .solve(&b_vec, 1e-14)
        .map_err(|_| Error::ValueError("SVD solve failed on normal matrix".into()))?;

    Ok((dx, n_mat))
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

/// Compute weighted RMS of residuals for included observations.
fn weighted_rms(residuals: &[DVector<f64>], obs: &[Observation], included: &[bool]) -> f64 {
    let mut sum = 0.0;
    let mut count = 0.0;
    for (i, res) in residuals.iter().enumerate() {
        if !included[i] {
            continue;
        }
        let w = obs[i].weights();
        for (r, wi) in res.iter().zip(w.iter()) {
            sum += r * r * wi;
            count += 1.0;
        }
    }
    if count > 0.0 {
        (sum / count).sqrt()
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

    /// Generate synthetic optical observations with an optional non-grav model.
    ///
    /// Uses the full N-body SPK propagator so that the physics model is
    /// consistent with the batch least-squares solver.
    fn synth_observations_ng(
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

    /// Generate synthetic optical observations for a given true state.
    ///
    /// Uses the full N-body SPK propagator so that the physics model is
    /// consistent with the batch least-squares solver (which chains the
    /// variational STM inside the same integrator).
    fn synth_observations(
        true_state: &State<Equatorial>,
        epochs: &[f64],
        observer_pos_fn: impl Fn(f64) -> ([f64; 3], [f64; 3]),
        sigma: f64,
    ) -> Vec<Observation> {
        let mut observations = Vec::new();
        for &jd in epochs {
            let (obs_pos, obs_vel) = observer_pos_fn(jd);
            let observer = make_state(obs_pos, obs_vel, jd);

            // Propagate true object to this epoch via N-body SPK (same physics
            // as the solver) so there is no model mismatch.
            let obj_at =
                propagate_n_body_spk(true_state.clone(), Time::<TDB>::new(jd), false, None)
                    .unwrap();

            // Apply two-body light-time correction (consistent with solver).
            let obj_lt = two_body_lt_state(&obj_at, &observer).unwrap();

            // Compute RA/Dec.
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
        let observations = synth_observations(&true_state, &epochs, earth_observer, sigma);

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
        let observations = synth_observations(&true_state, &epochs, earth_observer, sigma);

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
        let mut observations = synth_observations(&true_state, &epochs, earth_observer, sigma);

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
            synth_observations_ng(&true_state, &epochs, earth_observer, sigma, Some(&true_ng));

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
            synth_observations_ng(&true_state, &epochs, earth_observer, sigma, Some(&true_ng));

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
}
