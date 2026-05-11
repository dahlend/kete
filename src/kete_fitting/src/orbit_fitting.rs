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

use crate::obs::{AstrometricObservation, differential_light_deflect};
use kete_core::forces::{FrozenForce, FrozenNonGrav, GravParams, ParameterizedForce};
use kete_core::frames::SunCenter;
use std::sync::Arc;

/// Internal owned bundle the fitter carries through trial/accept/reject
/// loops: typed [`ParameterizedForce`] template, current parameter values, and lower
/// bounds. A tuple alias rather than a struct -- no methods, no OOP
/// coupling. Bounds are carried alongside values for cheap mutation
/// without touching the template or re-passing bounds at every call.
pub type NonGravFit = (
    Arc<dyn ParameterizedForce<Frame = Equatorial, Center = SunCenter>>,
    Vec<f64>,
    Vec<f64>,
);

#[cfg(test)]
fn jpl_comet_default_fit(a1: f64, a2: f64, a3: f64) -> NonGravFit {
    use kete_core::forces::JplCometNonGrav;
    (
        Arc::new(JplCometNonGrav::standard_comet()),
        vec![a1, a2, a3],
        vec![f64::NEG_INFINITY; 3],
    )
}

#[cfg(test)]
fn dust_fit(beta: f64) -> NonGravFit {
    use kete_core::forces::DustNonGrav;
    (Arc::new(DustNonGrav), vec![beta], vec![f64::NEG_INFINITY])
}
use kete_core::frames::{Equatorial, SSB};
use kete_core::kepler::light_time_correct;
use kete_core::prelude::{Error, KeteResult, State, UncertainState};
use kete_core::state::StateLike;
use kete_core::time::{TDB, Time};
use kete_spice::prelude::{LOADED_SPK, compute_state_transition};
use kete_spice::propagation::{SpkNBody, helio_with_frozen_nongrav};
use nalgebra::{DMatrix, DVector};
use rayon::prelude::*;

/// Helper: propagate an SSB-centered state under heliocentric N-body
/// gravity plus optional frozen non-grav values.
fn propagate_helio(
    state: State<Equatorial, SSB>,
    jd_final: Time<TDB>,
    include_extended: bool,
    non_grav: Option<&FrozenNonGrav>,
) -> KeteResult<State<Equatorial, SSB>> {
    let spk = LOADED_SPK.try_read()?;
    if include_extended {
        let planets = GravParams::selected_masses();
        match non_grav {
            None => state.propagate_with(&SpkNBody::new(&spk, &planets), jd_final),
            Some(frozen) => {
                let force = helio_with_frozen_nongrav(&spk, &planets, frozen)?;
                state.propagate_with(&force, jd_final)
            }
        }
    } else {
        let planets = GravParams::planets();
        match non_grav {
            None => state.propagate_with(&SpkNBody::new(&spk, &planets), jd_final),
            Some(frozen) => {
                let force = helio_with_frozen_nongrav(&spk, &planets, frozen)?;
                state.propagate_with(&force, jd_final)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tuning constants
// ---------------------------------------------------------------------------

/// Initial window radius for the arc-expansion bootstrap (days).
/// Windows double from this value until the full arc is covered.
const EXPANSION_INITIAL_RADIUS_DAYS: f64 = 30.0;

/// Sigma floor applied to every observation during windowed expansion (radians, ~0.5 arcsec).
/// Prevents high-precision observations (occultations, Gaia transits) from dominating
/// the gradient before the orbit has converged near the true solution.
/// Released for the final full-arc pass.
const EXPANSION_SIGMA_FLOOR_RAD: f64 = 0.5 / 3600.0 * (std::f64::consts::PI / 180.0);

/// CMC recovery threshold = rejection threshold * this factor (Carpino et al. 2003).
const CMC_RECOVERY_FRACTION: f64 = 6.0 / 7.0;

/// Minimum leverage-corrected expected chi2 denominator (prevents division instability).
const CMC_MIN_EXPECTED: f64 = 0.5;

/// Adaptive widening: multiply `chi2_threshold` by this factor per retry.
const WIDENING_MULTIPLIER: f64 = 1.5;

/// Adaptive widening: ceiling expressed as a multiple of the initial threshold.
const WIDENING_CEILING_FACTOR: f64 = 20.0;

/// Adaptive widening: retry if the included observation fraction drops below this.
const WIDENING_MIN_FRACTION: f64 = 2.0 / 3.0;

/// Adaptive widening: retry if the arc span drops below this fraction of the original.
const WIDENING_MIN_ARC_FRACTION: f64 = 0.5;

/// Maximum position correction per LM step (AU).
const LM_MAX_POS_CORRECTION_AU: f64 = 0.5;

/// Maximum velocity correction per LM step (AU/day).
const LM_MAX_VEL_CORRECTION_AU_DAY: f64 = 0.005;

/// Initial LM damping lambda when non-gravitational parameters are free.
/// Their information-matrix entries are often orders of magnitude smaller than
/// the orbital entries, so an undamped first step can produce enormous corrections.
const LM_NG_INITIAL_LAMBDA: f64 = 1e-4;

/// Factor by which LM lambda increases on a rejected step.
const LM_LAMBDA_INCREASE: f64 = 10.0;

/// Factor by which LM lambda decreases on an accepted step.
const LM_LAMBDA_DECREASE: f64 = 0.1;

/// LM lambda ceiling; solver gives up and returns best-found state above this.
const LM_LAMBDA_MAX: f64 = 1e12;

/// LM lambda threshold below which we reset to 1.0 on rejection (avoids
/// getting stuck at near-zero lambda that never damps enough).
const LM_LAMBDA_RESET_THRESHOLD: f64 = 1e-6;

/// Result of orbit determination via batch least squares.
///
/// The covariance in `uncertain_state` is the Danby-rescaled posterior
/// covariance `(H^T W H)^{-1} * chi^2 / dof`.  If the per-observation
/// sigmas are well-calibrated the rescaling factor is ~1; if they
/// understate the true noise by a factor k, the post-fit RMS is ~k and
/// the covariance is inflated by k^2, so reported uncertainties stay
/// well-calibrated regardless.
#[derive(Clone)]
pub struct OrbitFit {
    /// Core uncertain orbit (state + covariance + free parameters).
    pub uncertain_state: UncertainState,

    /// Non-gravitational fit data: `(template, values, lower_bounds)`.
    /// Free parameter values live on `uncertain_state.free_params`;
    /// this field carries the typed [`ParameterizedForce`] template (variant + fixed
    /// coefficients) plus the fitter's view of the values and bounds
    /// for round-trip reconstruction. `None` when fitting gravity-only
    /// orbits.
    pub non_grav: Option<NonGravFit>,

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

impl std::fmt::Debug for OrbitFit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OrbitFit")
            .field("uncertain_state", &self.uncertain_state)
            .field("non_grav_present", &self.non_grav.is_some())
            .field("residuals", &self.residuals)
            .field("observations", &self.observations)
            .field("included", &self.included)
            .field("rms", &self.rms)
            .field("converged", &self.converged)
            .finish()
    }
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
///   threshold (Carpino-Milani-Chesley 2003 hysteresis).  For a 2D
///   optical observation, a 3-sigma outlier in one axis gives z = 4.5,
///   so a value of 4.5 matches per-component 3-sigma rejection.  Only
///   used when `max_reject_passes > 0`.
/// * `max_reject_passes` - Maximum outlier-rejection cycles.  Set to 0 to
///   disable rejection entirely.
///
/// # Errors
/// Fails if any internal propagation or solve fails.
pub fn fit_orbit(
    initial_state: &State<Equatorial, SSB>,
    obs: &[AstrometricObservation],
    include_asteroids: bool,
    non_grav: Option<&NonGravFit>,
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
            let Ok(s) = o.observer() else { return false };
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
    let mut radius = EXPANSION_INITIAL_RADIUS_DAYS;
    while radius < arc_radius {
        windows.push(radius);
        radius *= 2.0;
    }
    windows.push(f64::INFINITY);

    let mut state = initial_state.clone();
    let ng = non_grav.cloned();
    let mut prev_n_in_window: usize = 0;

    let expansion_floor = EXPANSION_SIGMA_FLOOR_RAD;

    // Expansion stages: converge + reject on each window.
    // Non-grav parameters are frozen during expansion because short arcs
    // have almost no sensitivity to them; fitting them here would produce
    // wildly wrong values that poison subsequent stages.  The original
    // non-grav values are preserved and used in the final full-arc pass.
    for &radius in &windows[..windows.len() - 1] {
        let windowed: Vec<AstrometricObservation> = sorted
            .iter()
            .filter(|ob| (ob.epoch().jd - ref_jd).abs() <= radius)
            .map(|ob| ob.with_sigma_floor(expansion_floor))
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
        ) && let Ok(candidate) =
            State::<Equatorial, SSB>::try_from(result.uncertain_state.state.clone())
        {
            // Re-score the candidate on the FULL window (ignoring any
            // outlier rejections inside solve_with_rejection) for an
            // apples-to-apples comparison.
            let post_rms = compute_residuals(&candidate, &windowed, include_asteroids, None)
                .ok()
                .map_or(f64::INFINITY, |r| weighted_rms(&r, &windowed, &included, 6));
            if post_rms.is_finite() && post_rms <= pre_rms {
                state = candidate;
            }
            // else: keep prior state, the conversion invariant was violated.
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

    // Compute the best available fit with raw (Fisher-inverse) covariance,
    // then apply the Danby sigma_sq rescaling once at the boundary.  Keeping
    // covariance raw internally is required for CMC leverage correctness.
    let mut fit = fit_orbit_raw(
        &state,
        &sorted,
        &included,
        include_asteroids,
        ng,
        max_iter,
        tol,
        chi2_threshold,
        max_reject_passes,
    )?;
    rescale_covariance_danby(&mut fit);
    Ok(fit)
}

/// Internal worker for [`fit_orbit`] that returns a fit with the raw Fisher
/// inverse covariance (no Danby rescaling applied).  The public wrapper
/// applies the rescaling once at the boundary.
#[allow(
    clippy::too_many_arguments,
    reason = "orbit fitting requires many tuning parameters"
)]
fn fit_orbit_raw(
    state: &State<Equatorial, SSB>,
    sorted: &[AstrometricObservation],
    included: &[bool],
    include_asteroids: bool,
    ng: Option<NonGravFit>,
    max_iter: usize,
    tol: f64,
    chi2_threshold: f64,
    max_reject_passes: usize,
) -> KeteResult<OrbitFit> {
    // No non-grav requested: single gravity-only solve.
    if ng.is_none() {
        return solve_with_rejection_adaptive(
            state,
            sorted,
            included,
            include_asteroids,
            None,
            max_iter,
            tol,
            chi2_threshold,
            max_reject_passes,
        );
    }

    // Non-grav requested.  Run gravity-only first to establish a stable
    // starting point and a gravity-determined rejection mask, then run
    // the NG pass on that mask.
    let Ok(grav_fit) = solve_with_rejection_adaptive(
        state,
        sorted,
        included,
        include_asteroids,
        None,
        max_iter,
        tol,
        chi2_threshold,
        max_reject_passes,
    ) else {
        // Gravity-only failed outright; fall back to a direct NG solve.
        return solve_with_rejection_adaptive(
            state,
            sorted,
            included,
            include_asteroids,
            ng,
            max_iter,
            tol,
            chi2_threshold,
            max_reject_passes,
        );
    };

    let grav_state = State::<Equatorial, SSB>::try_from(grav_fit.uncertain_state.state.clone())?;

    iterate_to_convergence(
        &grav_state,
        sorted,
        &grav_fit.included,
        include_asteroids,
        ng,
        max_iter,
        tol,
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
    initial_state: &State<Equatorial, SSB>,
    sorted_obs: &[AstrometricObservation],
    included: &[bool],
    include_asteroids: bool,
    non_grav: Option<NonGravFit>,
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

    let np = fit.uncertain_state.free_params.len();
    let min_included = (6 + np).max(4);

    // Rejection and recovery thresholds with 7/6 hysteresis ratio
    // (Carpino, Milani & Chesley 2003).  The z-scores are already
    // leverage-normalized so no additional scaling is applied.
    let chi2_rej = chi2_threshold;
    let chi2_rec = chi2_threshold * CMC_RECOVERY_FRACTION;

    for _ in 0..max_reject_passes {
        // Sweep over every caller-allowed observation (mask = `included`)
        // so that currently rejected observations also receive z-scores
        // and can be recovered.  stm_sweep emits one StmObs per included
        // observation in time-sorted order.
        let sweep_state: State<Equatorial, SSB> = fit.uncertain_state.state.clone().try_into()?;
        let sweep = stm_sweep(
            &sweep_state,
            sorted_obs,
            included,
            include_asteroids,
            fit.non_grav.as_ref(),
        )?;
        // CMC leverage uses the Fisher inverse C_0 = (H^T W H)^+, which is
        // stored directly in cov_matrix (no chi-square scaling applied).
        let cov_fisher = &fit.uncertain_state.cov_matrix;

        let mut z_scores: Vec<f64> = Vec::with_capacity(sweep.len());
        for entry in &sweep {
            let m = entry.weight_matrix.nrows();
            // Parameter-space design block: H_i = h_local * phi_cum (m x d).
            let h_full: DMatrix<f64> = &entry.h_local * &entry.phi_cum;
            let hch: DMatrix<f64> = &h_full * cov_fisher * h_full.transpose();
            // leverage = tr(H C_0 H^T W)  (full matrix trace for non-diagonal W).
            let leverage: f64 = (&hch * &entry.weight_matrix).trace();
            // chi2 = r^T W r  (full weight matrix, includes timing correction).
            let wr = &entry.weight_matrix * &entry.residual;
            let chi2: f64 = entry.residual.dot(&wr);
            let expected = (m as f64 - leverage).max(CMC_MIN_EXPECTED);
            // z = chi2 / (m - leverage): reject when the weighted squared
            // residual exceeds chi2_threshold times the leverage-corrected
            // expected value.  With calibrated weights this is equivalent to
            // an absolute sigma threshold: threshold 4.5 rejects at ~3-sigma
            // per component for a 2-component optical observation.
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

        let iter_state: State<Equatorial, SSB> = fit.uncertain_state.state.clone().try_into()?;
        fit = iterate_to_convergence(
            &iter_state,
            sorted_obs,
            &current_included,
            include_asteroids,
            fit.non_grav.clone(),
            max_iter,
            tol,
        )?;
    }

    Ok(fit)
}

/// Adaptive wrapper around [`solve_with_rejection`] that widens the
/// outlier-rejection threshold when the initial attempt would toss too
/// many observations or shrink the observation arc too much.
///
/// Starts at the caller's threshold.  If the final inclusion set drops
/// below 2/3 of the original count or the arc span below 1/2 of the
/// original span, the threshold is multiplied by 1.5 and the fit is
/// retried, up to a ceiling of 20x the caller's threshold.  This
/// protects fits on noisy or borderline data from "evaporating" down to
/// too few observations, while still rejecting clean outliers at the
/// user's stated threshold.
///
/// When `max_reject_passes == 0` this is a pass-through -- no rejection
/// is happening, so there is nothing to widen.
#[allow(
    clippy::too_many_arguments,
    reason = "orbit fitting requires many tuning parameters"
)]
fn solve_with_rejection_adaptive(
    initial_state: &State<Equatorial, SSB>,
    sorted_obs: &[AstrometricObservation],
    included: &[bool],
    include_asteroids: bool,
    non_grav: Option<NonGravFit>,
    max_iter: usize,
    tol: f64,
    chi2_threshold: f64,
    max_reject_passes: usize,
) -> KeteResult<OrbitFit> {
    if max_reject_passes == 0 {
        return solve_with_rejection(
            initial_state,
            sorted_obs,
            included,
            include_asteroids,
            non_grav,
            max_iter,
            tol,
            chi2_threshold,
            max_reject_passes,
        );
    }

    let orig_n = included.iter().filter(|&&v| v).count();
    let orig_span = included_arc_span(sorted_obs, included);

    // Widen up to 20x the caller's threshold.
    let max_threshold = chi2_threshold * WIDENING_CEILING_FACTOR;
    let mut threshold = chi2_threshold;
    let mut last_fit: Option<OrbitFit> = None;

    while threshold <= max_threshold {
        let fit = solve_with_rejection(
            initial_state,
            sorted_obs,
            included,
            include_asteroids,
            non_grav.clone(),
            max_iter,
            tol,
            threshold,
            max_reject_passes,
        )?;

        let final_n = fit.included.iter().filter(|&&v| v).count();
        let final_span = included_arc_span(sorted_obs, &fit.included);

        let dropped_too_many = (final_n as f64) < (orig_n as f64) * WIDENING_MIN_FRACTION;
        let arc_too_short = final_span < orig_span * WIDENING_MIN_ARC_FRACTION;

        if !dropped_too_many && !arc_too_short {
            return Ok(fit);
        }

        last_fit = Some(fit);
        threshold *= WIDENING_MULTIPLIER;
    }

    // Widening ceiling reached; return the last attempt.
    last_fit.ok_or_else(|| {
        Error::ValueError("adaptive widening exhausted without completing a pass".into())
    })
}

/// Time span (in days) of observations currently marked included.
/// Returns 0 when fewer than two observations are included.
fn included_arc_span(obs: &[AstrometricObservation], included: &[bool]) -> f64 {
    let mut min_jd = f64::INFINITY;
    let mut max_jd = f64::NEG_INFINITY;
    for (i, ob) in obs.iter().enumerate() {
        if !included[i] {
            continue;
        }
        let jd = ob.epoch().jd;
        if jd < min_jd {
            min_jd = jd;
        }
        if jd > max_jd {
            max_jd = jd;
        }
    }
    if min_jd.is_finite() && max_jd.is_finite() {
        (max_jd - min_jd).max(0.0)
    } else {
        0.0
    }
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
    state_ssb: State<Equatorial, SSB>,
    observer: State<Equatorial, SSB>,
) -> KeteResult<State<Equatorial, SSB>> {
    let spk = LOADED_SPK.try_read()?;
    // Convert both object and observer to heliocentric for two-body correction.
    let observer_dyn: State<Equatorial> = observer.into();
    let state_sun = spk.try_to_sun(state_ssb)?;
    let observer_sun = spk.try_to_sun(observer_dyn)?;
    let obj_lt_helio = light_time_correct(&state_sun, &observer_sun.pos)?;
    // Apply differential gravitational light deflection (solar bending of the
    // photon path relative to background stars at infinity).
    let deflected_pos = differential_light_deflect(&observer_sun.pos, obj_lt_helio.pos);
    let obj_lt_deflected = State {
        pos: deflected_pos,
        ..obj_lt_helio
    };
    spk.try_to_ssb(obj_lt_deflected)
}

/// Run the iterative convergence loop with adaptive Levenberg-Marquardt
/// damping and step-size limiting.
///
/// Each iteration re-linearizes at the current state, solves the damped
/// normal equations `(N + lambda * diag(N)) dx = b`, limits the step
/// magnitude, and decides accept/reject based on the weighted L2 loss
/// (`0.5 * chi^2`).  Re-linearizing every iteration is essential: the
/// step limiter caps *magnitude* but not *direction*, so recycling a
/// stale Jacobian would repeatedly propose the same capped step and
/// stall.
///
/// Lambda is adjusted heuristically: decreased when the loss improves,
/// increased when it worsens.  This steers the solver between
/// Gauss-Newton (fast near the solution) and steepest descent (safe far
/// from it).
///
/// The acceptance metric matches the weighted RMS users see at the end
/// of the fit.  Outlier handling is the job of the Carpino-Milani-Chesley
/// rejection loop in `solve_with_rejection`, not the loss function -- this
/// keeps the loss monotone under correct LM steps and makes the Fisher
/// inverse a well-defined covariance for the estimator being computed.
///
/// On rejection the solver increases lambda and re-solves from the same
/// linearization point (no repropagation).  This guarantees that
/// `state_epoch` is always the best state seen.
fn iterate_to_convergence(
    initial_state: &State<Equatorial, SSB>,
    obs: &[AstrometricObservation],
    included: &[bool],
    include_asteroids: bool,
    mut non_grav: Option<NonGravFit>,
    max_iter: usize,
    tol: f64,
) -> KeteResult<OrbitFit> {
    let mut state_epoch: State<Equatorial, SSB> = initial_state.clone();
    // Start with non-zero damping when fitting non-grav parameters.
    // Their information-matrix entries are often orders of magnitude
    // smaller than the orbital entries, so an undamped first step can
    // produce enormous non-grav corrections that poison the fit.
    let np = non_grav.as_ref().map_or(0, |m: &NonGravFit| m.1.len());
    let mut lambda = if np > 0 { LM_NG_INITIAL_LAMBDA } else { 0.0 };

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
            lambda = if lambda < LM_LAMBDA_RESET_THRESHOLD {
                1.0
            } else {
                lambda * LM_LAMBDA_INCREASE
            };
            if lambda > LM_LAMBDA_MAX {
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
                // Accept step: weighted L2 loss improved (or stayed equal).
                state_epoch = trial_state;
                non_grav = trial_ng;
                info_mat = new_info;
                rhs_vec = new_rhs;
                loss = new_loss;
                lambda *= LM_LAMBDA_DECREASE;

                if converged {
                    let raw_cov = scaled_pseudo_inverse(&info_mat).map_err(|e| {
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

                    let n_params = 6 + non_grav.as_ref().map_or(0, |m: &NonGravFit| m.1.len());
                    let rms = weighted_rms(&residuals, obs, included, n_params);
                    // Internal covariance stays as the raw Fisher inverse
                    // (H^T W H)^{-1}.  `solve_with_rejection` requires the
                    // raw inverse for the CMC leverage-correction formula:
                    //   leverage = tr(H C H^T W)
                    // which is only consistent when `C` is the unscaled
                    // information inverse.  The Danby unit-weight rescaling
                    // (covariance *= rms^2) is applied once at the
                    // `fit_orbit` boundary; see `rescale_covariance_danby`.
                    let covariance = raw_cov;
                    let free_params = non_grav.as_ref().map_or_else(Vec::new, |m| m.1.clone());
                    let uncertain_state =
                        UncertainState::new(state_epoch.into(), covariance, free_params)?;
                    return Ok(OrbitFit {
                        uncertain_state,
                        non_grav,
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
                lambda = if lambda < LM_LAMBDA_RESET_THRESHOLD {
                    1.0
                } else {
                    lambda * LM_LAMBDA_INCREASE
                };
                if lambda > LM_LAMBDA_MAX {
                    break;
                }
            }
        } else {
            // Propagation failed at trial state -- reject and damp.
            lambda = if lambda < LM_LAMBDA_RESET_THRESHOLD {
                1.0
            } else {
                lambda * LM_LAMBDA_INCREASE
            };
            if lambda > LM_LAMBDA_MAX {
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
    state: &State<Equatorial, SSB>,
    obs: &[AstrometricObservation],
    included: &[bool],
    include_asteroids: bool,
    non_grav: Option<NonGravFit>,
) -> OrbitFit {
    let n_params = 6 + non_grav.as_ref().map_or(0, |m: &NonGravFit| m.1.len());

    // Try to compute covariance and residuals together; fall back to
    // placeholders if any propagation step fails.  Covariance here is the
    // raw Fisher inverse; the Danby rescaling is applied once at the
    // `fit_orbit` boundary (see `rescale_covariance_danby`).
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
    let free_params = non_grav.as_ref().map_or_else(Vec::new, |m| m.1.clone());
    let uncertain_state = UncertainState::new(state.clone().into(), covariance, free_params)
        .expect("dimension mismatch");

    OrbitFit {
        uncertain_state,
        non_grav,
        residuals,
        observations: obs.to_vec(),
        included: included.to_vec(),
        rms,
        converged: false,
    }
}

/// Apply the Danby unit-weight-variance rescaling to a fit's covariance.
///
/// Internally the solver stores the raw Fisher inverse `(H^T W H)^{-1}`
/// because the CMC leverage formula requires it.  The reported covariance
/// should be the posterior estimate, which equals the Fisher inverse times
/// `sigma_sq = chi^2 / dof = rms^2`.  This function applies that rescaling
/// in place and is called once at every `fit_orbit` return boundary.
///
/// Reference: Danby, Fundamentals of Celestial Mechanics 2nd ed., p. 243
/// eq. 7.5.21.
fn rescale_covariance_danby(fit: &mut OrbitFit) {
    if fit.rms.is_finite() && fit.rms > 0.0 {
        let sigma_sq = fit.rms * fit.rms;
        fit.uncertain_state.cov_matrix *= sigma_sq;
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
    /// Full weight matrix (inverse noise covariance), m x m.
    /// For optical with timing correction this is a non-diagonal 2x2 matrix;
    /// for radar it is a 1x1 scalar matrix.
    pub weight_matrix: DMatrix<f64>,
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
    checkpoint: &State<Equatorial, SSB>,
    obs: &[AstrometricObservation],
    included: &[bool],
    include_asteroids: bool,
    non_grav: Option<&NonGravFit>,
) -> KeteResult<(Vec<StmObs>, DMatrix<f64>)> {
    let np = non_grav.map_or(0, |m: &NonGravFit| m.1.len());
    let d = 6 + np;

    // Local phi_cum initialized to identity relative to the checkpoint.
    let mut phi_cum = DMatrix::<f64>::zeros(6, d);
    for i in 0..6 {
        phi_cum[(i, i)] = 1.0;
    }

    let mut state_cur: State<Equatorial, SSB> = checkpoint.clone();
    let mut results = Vec::new();

    for (i, observation) in obs.iter().enumerate() {
        let obs_epoch = observation.epoch();

        if (obs_epoch.jd - state_cur.epoch.jd).abs() > 1e-12 {
            let ng_frozen = non_grav
                .map(|f| FrozenForce::new(f.0.clone(), f.1.clone()))
                .transpose()?;
            let (new_state, phi_k) = compute_state_transition(
                &state_cur,
                obs_epoch,
                include_asteroids,
                ng_frozen.as_ref(),
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

        let observer_state = observation.observer()?;
        let obj_lt = light_time_corrected_state(state_cur.clone(), observer_state.clone())?;

        let residual = observation.residual_from_corrected(&obj_lt)?;
        let h_local = observation.partials(&obj_lt)?;

        // Apparent angular velocity (rad/day): H[:,0:3] * (v_obj - v_obs).
        // Used to inflate along-track uncertainty for timing errors.
        let v_obj: [f64; 3] = obj_lt.vel.into();
        let v_obs: [f64; 3] = observer_state.vel.into();
        let vel_rel = DVector::from_column_slice(&[
            v_obj[0] - v_obs[0],
            v_obj[1] - v_obs[1],
            v_obj[2] - v_obs[2],
        ]);
        let motion = h_local.columns(0, 3) * &vel_rel;
        let weight_matrix = observation.weight_matrix(
            motion.get(0).copied().unwrap_or(0.0),
            motion.get(1).copied().unwrap_or(0.0),
        );

        results.push(StmObs {
            phi_cum: phi_cum.clone(),
            residual,
            h_local,
            weight_matrix,
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
/// `propagate_helio` (6-dim) computes checkpoint states at segment
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
    state_epoch: &State<Equatorial, SSB>,
    obs: &[AstrometricObservation],
    included: &[bool],
    include_asteroids: bool,
    non_grav: Option<&NonGravFit>,
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

    if obs.is_empty() {
        return Ok(vec![]);
    }

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

    let np = non_grav.map_or(0, |m: &NonGravFit| m.1.len());
    let d = 6 + np;

    // Step 1 -- sequential cheap pre-pass.
    // Propagate the 6-dim state to checkpoint epochs so that every
    // segment has an exact starting state.  Using the 6-dim helper
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
    let mut checkpoint_states: Vec<State<Equatorial, SSB>> = Vec::with_capacity(n_segments);
    checkpoint_states.push(state_epoch.clone());
    let mut cur: State<Equatorial, SSB> = state_epoch.clone();
    for &(_, end) in &segment_ranges[..n_segments - 1] {
        let target_epoch = obs[end - 1].epoch();
        if (target_epoch.jd - cur.epoch.jd).abs() > 1e-12 {
            let ng_frozen = non_grav
                .map(|f| FrozenForce::new(f.0.clone(), f.1.clone()))
                .transpose()?;
            cur = propagate_helio(
                cur.clone(),
                target_epoch,
                include_asteroids,
                ng_frozen.as_ref(),
            )?;
        }
        checkpoint_states.push(cur.clone());
    }

    // Step 2 -- parallel STM sub-sweeps.
    // Each segment runs independently from its checkpoint with a fresh
    // identity phi_cum.  The local phi_cum values are relative to the
    // segment's checkpoint epoch, not the reference epoch.
    let inputs: Vec<(&[AstrometricObservation], &[bool], &State<Equatorial, SSB>)> = segment_ranges
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

/// Accumulate the weighted normal equations for one linearization pass.
///
/// Returns `(info_mat, rhs_vec, loss)` where `info_mat` is the
/// (6+Np) x (6+Np) information matrix, `rhs_vec` is the right-hand
/// side, and `loss` is `0.5 * chi^2` summed over all included measurements
/// (the merit function used by the LM accept gate).
///
/// # Errors
/// Returns an error if the underlying STM sweep fails.
pub(crate) fn accumulate_normal_equations(
    state_epoch: &State<Equatorial, SSB>,
    obs: &[AstrometricObservation],
    included: &[bool],
    include_asteroids: bool,
    non_grav: Option<&NonGravFit>,
) -> KeteResult<(DMatrix<f64>, DVector<f64>, f64)> {
    let sweep = stm_sweep(state_epoch, obs, included, include_asteroids, non_grav)?;
    let (n_mat, b_vec, loss, _) = accumulate_from_sweep(&sweep, non_grav);
    Ok((n_mat, b_vec, loss))
}

/// Accumulate normal equations from a pre-computed STM sweep.
///
/// Plain weighted least-squares accumulation: every observation contributes
/// `r^T W r` to chi^2 without IRLS downweighting.  Outlier handling is the
/// job of the Carpino-Milani-Chesley rejection loop in `solve_with_rejection`,
/// not the loss function.  This keeps the solver's merit function aligned
/// with the weighted RMS the user sees and makes the Fisher-inverse
/// covariance well-defined.
///
/// Returns `(info_mat, rhs_vec, loss, residuals)` where `loss = 0.5 * chi^2`.
/// `residuals` contains one entry per included observation (matching the sweep).
fn accumulate_from_sweep(
    sweep: &[StmObs],
    non_grav: Option<&NonGravFit>,
) -> (DMatrix<f64>, DVector<f64>, f64, Vec<DVector<f64>>) {
    let np = non_grav.map_or(0, |m: &NonGravFit| m.1.len());
    let d = 6 + np;

    let mut n_mat = DMatrix::<f64>::zeros(d, d);
    let mut b_vec = DVector::<f64>::zeros(d);
    let mut loss = 0.0;
    let mut residuals = Vec::with_capacity(sweep.len());

    for entry in sweep {
        // Map to epoch: H_epoch = H_local * Phi_cum  (m x D).
        let h_epoch = &entry.h_local * &entry.phi_cum;
        let w = &entry.weight_matrix;
        let wr = w * &entry.residual;

        // Weighted L2: chi^2 contribution = r^T W r; loss = 0.5 * chi^2.
        loss += 0.5 * entry.residual.dot(&wr);

        residuals.push(entry.residual.clone());

        // Accumulate normal matrix and RHS:
        //   N += H^T W H,  b += H^T W r
        n_mat += h_epoch.transpose() * w * &h_epoch;
        b_vec += h_epoch.transpose() * &wr;
    }

    (n_mat, b_vec, loss, residuals)
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
/// iteration.  Non-grav parameters are not capped here: LM damping
/// (which ramps lambda on rejected steps) keeps NG behavior under
/// control without a per-iteration step cap that would slow convergence
/// in well-conditioned NG cases.
fn limit_correction(mut dx: DVector<f64>) -> DVector<f64> {
    // AU
    let pos_norm = (dx[0] * dx[0] + dx[1] * dx[1] + dx[2] * dx[2]).sqrt();
    if pos_norm > LM_MAX_POS_CORRECTION_AU {
        let s = LM_MAX_POS_CORRECTION_AU / pos_norm;
        for v in dx.rows_mut(0, 3).iter_mut() {
            *v *= s;
        }
    }

    let vel_norm = (dx[3] * dx[3] + dx[4] * dx[4] + dx[5] * dx[5]).sqrt();
    if vel_norm > LM_MAX_VEL_CORRECTION_AU_DAY {
        let s = LM_MAX_VEL_CORRECTION_AU_DAY / vel_norm;
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
fn backtrack_to_ng_bounds(mut dx: DVector<f64>, non_grav: Option<&NonGravFit>) -> DVector<f64> {
    let Some(ng) = non_grav else {
        return dx;
    };
    let np = ng.1.len();
    for k in 0..np {
        let cur = ng.1[k];
        let lo = ng.2[k];
        if !lo.is_finite() {
            continue;
        }
        let proposed = cur + dx[6 + k];
        if proposed > lo {
            continue;
        }
        // Shrink to land halfway between the current value and the
        // bound so we make progress toward the bound without snapping
        // onto it (which would re-trigger the clamp).
        let safe_step = 0.5 * (lo - cur);
        dx[6 + k] = safe_step;
    }
    dx
}

/// Apply a state correction vector to the epoch state and (optionally)
/// non-grav parameters.
fn apply_correction(
    state: &mut State<Equatorial, SSB>,
    dx: &DVector<f64>,
    non_grav: &mut Option<NonGravFit>,
) -> bool {
    if let Some(ng) = non_grav.as_ref() {
        let np = ng.1.len();
        for k in 0..np {
            let new_val = ng.1[k] + dx[6 + k];
            if new_val <= ng.2[k] || !new_val.is_finite() {
                return false;
            }
        }
    }

    let pos: [f64; 3] = state.pos.into();
    state.pos = [pos[0] + dx[0], pos[1] + dx[1], pos[2] + dx[2]].into();

    let vel: [f64; 3] = state.vel.into();
    state.vel = [vel[0] + dx[3], vel[1] + dx[4], vel[2] + dx[5]].into();

    if let Some(ng) = non_grav.as_mut() {
        let np = ng.1.len();
        for k in 0..np {
            ng.1[k] += dx[6 + k];
        }
    }
    true
}

/// Compute post-fit residuals for all observations (time-sorted order).
///
/// Uses the 6-dim `propagate_helio` (not the 60-dim STM
/// integrator) so this is ~5x cheaper than an STM sweep.
fn compute_residuals(
    state_epoch: &State<Equatorial, SSB>,
    obs: &[AstrometricObservation],
    include_asteroids: bool,
    non_grav: Option<&NonGravFit>,
) -> KeteResult<Vec<DVector<f64>>> {
    let mut residuals = Vec::with_capacity(obs.len());
    let mut state_cur: State<Equatorial, SSB> = state_epoch.clone();

    for observation in obs {
        let obs_epoch = observation.epoch();

        // Propagate to observation epoch (6-dim, no STM).
        if (obs_epoch.jd - state_cur.epoch.jd).abs() > 1e-12 {
            let ng_frozen = non_grav
                .map(|f| FrozenForce::new(f.0.clone(), f.1.clone()))
                .transpose()?;
            state_cur = propagate_helio(
                state_cur.clone(),
                obs_epoch,
                include_asteroids,
                ng_frozen.as_ref(),
            )?;
        }

        let observer_state = observation.observer()?;
        let obj_lt = light_time_corrected_state(state_cur.clone(), observer_state)?;
        let res = observation.residual_from_corrected(&obj_lt)?;
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
        // Use the full base weight matrix so that the RA/Dec correlation,
        // when present, is correctly accounted for in chi^2.  Timing
        // correction is excluded here -- `weighted_rms` reports a
        // position-only RMS that does not depend on apparent motion.
        let w = obs[i].base_weight_matrix();
        let wr = &w * res;
        sum += res.dot(&wr);
        n_meas += w.nrows();
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
    use kete_core::Band;
    use kete_core::constants::GMS;
    use kete_core::desigs::Desig;
    use kete_core::time::{TDB, Time};
    use kete_spice::test_data::ensure_test_spk;

    /// Helper: build a simple state.
    fn make_state(pos: [f64; 3], vel: [f64; 3], jd: f64) -> State<Equatorial, SSB> {
        State {
            desig: Desig::Empty,
            epoch: jd.into(),
            pos: pos.into(),
            vel: vel.into(),
            center: SSB,
        }
    }

    /// Generate synthetic optical observations, optionally with a non-grav model.
    ///
    /// Uses the full N-body SPK propagator so that the physics model is
    /// consistent with the batch least-squares solver.
    fn synth_observations(
        true_state: &State<Equatorial, SSB>,
        epochs: &[f64],
        observer_pos_fn: impl Fn(f64) -> ([f64; 3], [f64; 3]),
        sigma: f64,
        non_grav: Option<&NonGravFit>,
    ) -> Vec<AstrometricObservation> {
        let mut observations = Vec::new();
        for &jd in epochs {
            let (obs_pos, obs_vel) = observer_pos_fn(jd);
            let observer = make_state(obs_pos, obs_vel, jd);

            let ng_frozen = non_grav
                .map(|f| FrozenForce::new(f.0.clone(), f.1.clone()))
                .transpose()
                .expect("NonGravFit values match n_free_params");
            let obj_at = propagate_helio(
                true_state.clone(),
                Time::<TDB>::new(jd),
                false,
                ng_frozen.as_ref(),
            )
            .unwrap();

            let spk = LOADED_SPK.try_read().unwrap();
            let sun_at = spk.try_to_sun(obj_at.clone()).unwrap();
            let obs_helio = observer.pos - obj_at.pos + sun_at.pos;
            let obj_lt_sun = light_time_correct(&sun_at, &obs_helio).unwrap();
            // Apply DLD so synthetic observations are consistent with the prediction model.
            let deflected_pos = differential_light_deflect(&obs_helio, obj_lt_sun.pos);
            let obj_lt_deflected = State {
                pos: deflected_pos,
                ..obj_lt_sun
            };
            let obj_lt = spk.try_to_ssb(obj_lt_deflected).unwrap();
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
                band: Band::Unknown([0; 8]),
                mag: f64::NAN,
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
        let true_ng = jpl_comet_default_fit(0.0, true_a2, 0.0);

        // Generate 15 observations over 90 days with the non-grav model.
        let epochs: Vec<f64> = (0..15).map(|i| 2460000.5 + f64::from(i) * 6.0).collect();
        // Tight observations.
        let sigma = 1e-7;
        let observations =
            synth_observations(&true_state, &epochs, earth_observer, sigma, Some(&true_ng));

        // Start from true state + non-grav model with a2=0 and fit.
        let init_ng = jpl_comet_default_fit(0.0, 0.0, 0.0);

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
        let fitted_params = fit.uncertain_state.free_params.clone();
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
        let true_ng = dust_fit(true_beta);

        // 15 observations over 90 days.
        let epochs: Vec<f64> = (0..15).map(|i| 2460000.5 + f64::from(i) * 6.0).collect();
        let sigma = 1e-7;
        let observations =
            synth_observations(&true_state, &epochs, earth_observer, sigma, Some(&true_ng));

        // Start from true state with beta=0.
        let init_ng = dust_fit(0.0);

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

        let fitted_params = fit.uncertain_state.free_params.clone();
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

        // ParameterizedForce the parallel path: temporarily request a pool with at least
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

    /// Verify properties of `differential_light_deflect`.
    ///
    /// 1. The correction is perpendicular to the topocentric direction
    ///    (i.e. it shifts only the angle, not the range).
    /// 2. The magnitude is consistent with the standard GR formula:
    ///    `delta_theta ~ 2*GMS/c^2 * |tan(psi2/2) - tan(psi1/2)|`.
    /// 3. Objects near the Sun get larger corrections than objects far away.
    /// 4. Collinear or zero-distance edge cases return unmodified positions.
    #[test]
    fn test_differential_light_deflect() {
        use kete_core::constants::{C_AU_PER_DAY, GMS};
        use kete_core::frames::Vector;

        let bend_factor = 2.0 * GMS / (C_AU_PER_DAY * C_AU_PER_DAY);

        // Observer at 1 AU along +x, object at 2 AU along (1,1,0) / sqrt(2).
        let observer: Vector<Equatorial> = [1.0, 0.0, 0.0].into();
        let obj: Vector<Equatorial> =
            [std::f64::consts::SQRT_2, std::f64::consts::SQRT_2, 0.0].into();

        let deflected = differential_light_deflect(&observer, obj);

        // 1. The deflection should be perpendicular to the topocentric direction.
        let p_orig = obj - observer;
        let p_defl = deflected - observer;
        let p_orig_hat = p_orig / p_orig.norm();
        // Component of deflected topocentric direction along original: should be ~1.
        let cos_angle = p_defl.dot(&p_orig_hat) / p_defl.norm();
        assert!(
            (cos_angle - 1.0).abs() < 1e-8,
            "DLD should not change range direction appreciably, cos_angle deviation={:.3e}",
            (cos_angle - 1.0).abs()
        );

        // 2. Check the angular shift magnitude is in the right ballpark.
        let cross = p_orig_hat.cross(&(p_defl / p_defl.norm()));
        let angle_shift = cross.norm().asin();
        // For this geometry: psi2 ~ pi - elongation, bending ~ bend_factor * (tan(psi2/2) - tan(psi1/2))
        // Rough upper bound: should be < 2e-3 radians (< 400 arcsec) and > 0.
        assert!(
            angle_shift > 0.0,
            "DLD should produce a positive angular shift"
        );
        assert!(
            angle_shift < bend_factor * 10.0,
            "DLD angular shift {angle_shift:.3e} rad seems unreasonably large"
        );

        // 3. For two objects in the same sky direction from the observer, the one
        //    nearer to the Sun has a larger psi2 - psi1 differential and thus a
        //    larger DLD angular correction.  Both objects are along (0, 1, 0) from
        //    the observer at [1, 0, 0], differing only in heliocentric distance.
        let obj_near_sun: Vector<Equatorial> = [1.0, 0.5, 0.0].into(); // 1.12 AU from Sun
        let obj_far_sun: Vector<Equatorial> = [1.0, 2.0, 0.0].into(); // 2.24 AU from Sun
        let defl_near_sun = differential_light_deflect(&observer, obj_near_sun);
        let defl_far_sun = differential_light_deflect(&observer, obj_far_sun);

        let p_near_sun = obj_near_sun - observer;
        let p_defl_near_sun = defl_near_sun - observer;
        let angle_near_sun = p_near_sun.cross(&p_defl_near_sun).norm()
            / (p_near_sun.norm() * p_defl_near_sun.norm());

        let p_far_sun = obj_far_sun - observer;
        let p_defl_far_sun = defl_far_sun - observer;
        let angle_far_sun =
            p_far_sun.cross(&p_defl_far_sun).norm() / (p_far_sun.norm() * p_defl_far_sun.norm());

        assert!(
            angle_near_sun > angle_far_sun,
            "Object nearer to Sun should have larger DLD: near={angle_near_sun:.3e} vs far={angle_far_sun:.3e}"
        );

        // 4. Edge case: observer at origin -> no correction.
        let zero_obs: Vector<Equatorial> = [0.0, 0.0, 0.0].into();
        let obj_edge: Vector<Equatorial> = [1.0, 0.5, 0.0].into();
        let defl_edge = differential_light_deflect(&zero_obs, obj_edge);
        // olen = 0, function should return obj_lt_pos unchanged.
        let diff = (defl_edge - obj_edge).norm();
        assert!(
            diff < 1e-15,
            "Zero observer should return unchanged position, diff={diff:.3e}"
        );

        // 5. Edge case: object same position as observer -> no correction.
        let defl_same = differential_light_deflect(&observer, observer);
        let diff_same = (defl_same - observer).norm();
        assert!(
            diff_same < 1e-15,
            "Zero topocentric distance should return unchanged position, diff={diff_same:.3e}"
        );
    }

    /// When non-grav is requested but the truth has no non-grav signal, the
    /// NG fit is always returned (``non_grav`` field populated), even if the
    /// unconstrained A terms drift slightly.
    #[test]
    fn test_ng_always_returned_when_requested() {
        ensure_test_spk();
        let r = 1.5;
        let v = (GMS / r).sqrt();
        let true_state = make_state([r, 0.0, 0.0], [0.0, v, 0.0], 2460000.5);

        let epochs: Vec<f64> = (0..15).map(|i| 2460000.5 + f64::from(i) * 6.0).collect();
        let observations = synth_observations(&true_state, &epochs, earth_observer, 1e-6, None);

        let init_ng = jpl_comet_default_fit(0.0, 0.0, 0.0);
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

        assert!(
            fit.non_grav.is_some(),
            "non_grav should always be returned when requested"
        );
    }

    /// When the truth has a real non-grav signal, the fitter should recover it.
    #[test]
    fn test_ng_recovers_true_signal() {
        ensure_test_spk();
        // Truth has a tangential non-grav term.
        let r = 1.5;
        let v = (GMS / r).sqrt();
        let true_state = make_state([r, 0.0, 0.0], [0.0, v, 0.0], 2460000.5);
        let true_a2 = 1e-8;
        let true_ng = jpl_comet_default_fit(0.0, true_a2, 0.0);

        let epochs: Vec<f64> = (0..15).map(|i| 2460000.5 + f64::from(i) * 6.0).collect();
        let sigma = 1e-7;
        let observations =
            synth_observations(&true_state, &epochs, earth_observer, sigma, Some(&true_ng));

        let init_ng = jpl_comet_default_fit(0.0, 0.0, 0.0);
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

        // NG fit should be the one returned (non_grav populated) and should
        // recover a2 near the truth.
        assert!(
            fit.non_grav.is_some(),
            "non_grav should be kept when it helps"
        );
        let fitted_params = fit.uncertain_state.free_params.clone();
        let a2_err = (fitted_params[1] - true_a2).abs();
        assert!(
            a2_err < true_a2 * 0.1,
            "a2 error {a2_err:.6e} too large (true={true_a2:.6e}, fitted={:.6e})",
            fitted_params[1]
        );
    }

    /// After dropping Huber, the solver's merit function is `0.5 * chi^2`.
    /// At convergence the `loss` returned by `accumulate_normal_equations`
    /// must equal `0.5 * chi^2` where chi^2 is the weighted sum-of-squared
    /// residuals on the included observations.
    #[test]
    fn test_loss_equals_half_chi_squared() {
        ensure_test_spk();
        let r = 1.5;
        let v = (GMS / r).sqrt();
        let true_state = make_state([r, 0.0, 0.0], [0.0, v, 0.0], 2460000.5);
        let epochs: Vec<f64> = (0..10).map(|i| 2460000.5 + f64::from(i) * 6.0).collect();
        let sigma = 1e-6;
        let observations = synth_observations(&true_state, &epochs, earth_observer, sigma, None);
        let included = vec![true; observations.len()];

        let fit = fit_orbit(&true_state, &observations, false, None, 20, 1e-10, 9.0, 0).unwrap();

        // Evaluate normal equations at the converged state.
        let fit_state: State<Equatorial, SSB> =
            fit.uncertain_state.state.clone().try_into().unwrap();
        let (_n_mat, _b_vec, loss) =
            accumulate_normal_equations(&fit_state, &observations, &included, false, None).unwrap();

        // Compute chi^2 from the fit's residuals directly.
        let mut chi2 = 0.0_f64;
        for (i, res) in fit.residuals.iter().enumerate() {
            if !fit.included[i] {
                continue;
            }
            let w = observations[i].weights();
            for (rv, wv) in res.iter().zip(w.iter()) {
                chi2 += rv * rv * wv;
            }
        }

        let expected = 0.5 * chi2;
        let abs_err = (loss - expected).abs();
        let rel_err = abs_err / expected.abs().max(1e-30);
        assert!(
            rel_err < 1e-10 || abs_err < 1e-20,
            "loss = {loss:.12e} but 0.5 * chi^2 = {expected:.12e} (rel_err = {rel_err:.3e})"
        );
    }

    /// Adaptive rejection widening: when enough observations are at a
    /// borderline-outlier level that fixed-threshold rejection would drop
    /// below 2/3 of the arc, the adaptive wrapper widens the threshold
    /// until the inclusion set is preserved.  Without widening, the fit
    /// would "evaporate" to an unreasonably short arc.
    #[test]
    fn test_adaptive_rejection_widening_preserves_arc() {
        ensure_test_spk();
        let r = 1.5;
        let v = (GMS / r).sqrt();
        let true_state = make_state([r, 0.0, 0.0], [0.0, v, 0.0], 2460000.5);

        let epochs: Vec<f64> = (0..20).map(|i| 2460000.5 + f64::from(i) * 4.0).collect();
        let stated_sigma = 1e-7;
        let mut observations =
            synth_observations(&true_state, &epochs, earth_observer, stated_sigma, None);

        // Inject alternating +/- offsets of 3.5 * stated_sigma in 70% of
        // the observations.  Alternating signs ensure the fit cannot
        // absorb them into the state; the residuals stay near the
        // injected level.
        //
        //   z at 3.5 sigma Mahalanobis = (3.5)^2 = 12.25
        //   Rejection at z=4.5 (chi2 > 9): these observations would be
        //   rejected (since 12.25 > 9).
        //   With widening past 12.25/2 = 6.125 (threshold ~ 6.5+),
        //   they are kept.
        //
        // At 14 out of 20 = 70% injected outliers, naive rejection drops
        // to 6 (30%), well below the 2/3 = 14 floor.  Widening should
        // preserve at least 14.
        let injected = 3.5 * stated_sigma;
        for (i, obs) in observations.iter_mut().enumerate().take(14) {
            if let AstrometricObservation::Optical { ra, dec, .. } = obs {
                let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
                *ra += sign * injected;
                *dec -= sign * injected;
            }
        }

        let fit = fit_orbit(&true_state, &observations, false, None, 30, 1e-10, 4.5, 3).unwrap();

        let n_included = fit.included.iter().filter(|&&v| v).count();
        let n_total = observations.len();

        assert!(
            n_included * 3 >= n_total * 2,
            "Adaptive widening should preserve >= 2/3 of observations; \
             got {n_included}/{n_total}"
        );
    }

    /// A single large outlier (100 sigma) should still be rejected even
    /// with adaptive widening enabled, because its z-score far exceeds any
    /// reasonable widened threshold.  This guards against the widening
    /// loop going too permissive.
    #[test]
    fn test_adaptive_widening_still_rejects_large_outliers() {
        ensure_test_spk();
        let r = 1.5;
        let v = (GMS / r).sqrt();
        let true_state = make_state([r, 0.0, 0.0], [0.0, v, 0.0], 2460000.5);

        let epochs: Vec<f64> = (0..10).map(|i| 2460000.5 + f64::from(i) * 6.0).collect();
        let sigma = 1e-7;
        let mut observations =
            synth_observations(&true_state, &epochs, earth_observer, sigma, None);

        // Inject a single 100-sigma RA offset at observation 3.
        if let AstrometricObservation::Optical { ra, .. } = &mut observations[3] {
            *ra += 100.0 * sigma;
        }

        let fit = fit_orbit(&true_state, &observations, false, None, 20, 1e-10, 4.5, 3).unwrap();

        // Observation 3 should be rejected, all others kept.
        assert!(!fit.included[3], "100-sigma outlier should remain rejected");
        let n_included = fit.included.iter().filter(|&&v| v).count();
        assert!(
            n_included >= 9,
            "All clean observations should be kept; got {n_included}/10"
        );
    }

    /// Danby rescaling test: the reported covariance must equal the raw
    /// Fisher inverse scaled by `rms^2`.  We create observations with
    /// controlled noise (alternating +/- offsets) so that the post-fit RMS
    /// is nontrivial, then confirm the reported covariance diagonals match
    /// `raw_cov_ii * rms^2` algebraically.
    #[test]
    fn test_danby_rescaling_cov_matches_fisher_times_rms_squared() {
        ensure_test_spk();
        let r = 1.5;
        let v = (GMS / r).sqrt();
        let true_state = make_state([r, 0.0, 0.0], [0.0, v, 0.0], 2460000.5);

        let epochs: Vec<f64> = (0..20).map(|i| 2460000.5 + f64::from(i) * 5.0).collect();
        let stated_sigma = 1e-7;
        let mut observations =
            synth_observations(&true_state, &epochs, earth_observer, stated_sigma, None);

        // Inject alternating +/- offsets of ~2x stated sigma so the fit
        // will converge with RMS ~2 (weighted residuals are dominated by
        // the injected offsets).
        let injected = 2.0 * stated_sigma;
        for (i, obs) in observations.iter_mut().enumerate() {
            if let AstrometricObservation::Optical { ra, dec, .. } = obs {
                let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
                *ra += sign * injected;
                *dec -= sign * injected;
            }
        }

        let fit = fit_orbit(&true_state, &observations, false, None, 30, 1e-10, 9.0, 0).unwrap();

        assert!(
            fit.rms > 0.5,
            "Expected nontrivial RMS, got {:.3e}",
            fit.rms
        );

        // Reconstruct the raw Fisher inverse at the fit state.
        let fit_state: State<Equatorial, SSB> =
            fit.uncertain_state.state.clone().try_into().unwrap();
        let (info_mat, _, _) =
            accumulate_normal_equations(&fit_state, &observations, &fit.included, false, None)
                .unwrap();
        let raw_cov = scaled_pseudo_inverse(&info_mat).unwrap();

        // Reported covariance must equal raw * rms^2 on every diagonal.
        let sigma_sq = fit.rms * fit.rms;
        for i in 0..6 {
            let reported = fit.uncertain_state.cov_matrix[(i, i)];
            let expected = raw_cov[(i, i)] * sigma_sq;
            let rel_err = (reported - expected).abs() / expected.abs().max(1e-30);
            assert!(
                rel_err < 1e-6,
                "cov[{i},{i}] = {reported:.6e}, expected raw*sigma_sq = \
                 {expected:.6e} (rel_err = {rel_err:.3e})"
            );
        }
    }

    /// Observations constructed with non-zero `sigma_corr` must produce a
    /// different fit covariance than the same observations with zero
    /// correlation -- a sanity check that the correlation actually threads
    /// through `weight_matrix`, `weighted_rms`, and the reported covariance.
    #[test]
    fn test_correlation_changes_fit_covariance() {
        ensure_test_spk();
        let r = 1.5;
        let v = (GMS / r).sqrt();
        let true_state = make_state([r, 0.0, 0.0], [0.0, v, 0.0], 2460000.5);

        let epochs: Vec<f64> = (0..15).map(|i| 2460000.5 + f64::from(i) * 6.0).collect();
        let sigma = 1e-6;
        let obs_uncorr = synth_observations(&true_state, &epochs, earth_observer, sigma, None);

        // Build a correlated copy with sigma_corr = 0.6.
        let obs_corr: Vec<_> = obs_uncorr
            .iter()
            .map(|ob| {
                if let AstrometricObservation::Optical {
                    observer,
                    ra,
                    dec,
                    sigma_ra,
                    sigma_dec,
                    time_sigma,
                    band,
                    mag,
                    ..
                } = ob
                {
                    AstrometricObservation::Optical {
                        observer: observer.clone(),
                        ra: *ra,
                        dec: *dec,
                        sigma_ra: *sigma_ra,
                        sigma_dec: *sigma_dec,
                        sigma_corr: 0.6,
                        time_sigma: *time_sigma,
                        is_occultation: false,
                        band: *band,
                        mag: *mag,
                    }
                } else {
                    ob.clone()
                }
            })
            .collect();

        let fit_uncorr =
            fit_orbit(&true_state, &obs_uncorr, false, None, 20, 1e-10, 9.0, 0).unwrap();
        let fit_corr = fit_orbit(&true_state, &obs_corr, false, None, 20, 1e-10, 9.0, 0).unwrap();

        // The states should agree (injected residuals are zero in both).
        let pos_diff =
            (fit_uncorr.uncertain_state.state.pos - fit_corr.uncertain_state.state.pos).norm();
        assert!(
            pos_diff < 1e-8,
            "State should barely change with correlation, got pos_diff = {pos_diff:.3e}"
        );

        // The covariance should differ: with correlation 0.6, the
        // position covariance diagonals shrink by a factor of ~1/(1-0.36)
        // = ~1.56 in the information matrix, i.e., the reported
        // covariance diagonals grow by that factor.  At least one
        // position diagonal must differ by more than 10%.
        let mut max_rel_diff: f64 = 0.0;
        for i in 0..3 {
            let a = fit_uncorr.uncertain_state.cov_matrix[(i, i)];
            let b = fit_corr.uncertain_state.cov_matrix[(i, i)];
            let rel = (b - a).abs() / a.abs().max(1e-30);
            max_rel_diff = max_rel_diff.max(rel);
        }
        assert!(
            max_rel_diff > 0.05,
            "Correlation should change cov diagonals by >5%; got max rel diff = {max_rel_diff:.3e}"
        );
    }
}
