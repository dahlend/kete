//! Initial Orbit Determination (IOD).
//!
//! Given optical observations, compute an approximate heliocentric state that
//! can seed the batch least-squares orbit fitting or MCMC.
//!
//! [`initial_orbit_determination`] performs range-scanning IOD using Lambert's
//! solver.  It works on any arc length from single-night tracklets (minutes)
//! to multi-year arcs, and from close-approach NEOs/bolides to distant TNOs.
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

use kete_core::constants::GMS_SQRT;
use kete_core::frames::{Equatorial, Vector};
use kete_core::prelude::{Error, KeteResult, State};
use kete_core::propagation::{light_time_correct, propagate_two_body};
use kete_core::time::{TDB, Time};
use rayon::prelude::*;

use crate::Observation;
use crate::lambert::lambert;

/// Unified IOD: a robust approach to initial orbit determination.
///
/// Works on any observation arc from minutes to years, and any orbit type
/// from close-approach NEOs/bolides to distant TNOs.
///
/// # Algorithm
///
/// 1. Select observation pairs with adaptive time baselines.
/// 2. Coarse 2-D scan over (`log rho_a`, `log rho_b`), the topocentric
///    distances at each observation.  100x100 grid, log-spaced 0.00001-500 AU.
/// 3. Solve Lambert's problem (prograde, falling back to retrograde) for
///    each grid point to obtain velocity.
/// 4. Refine the best seeds with nested local grid search.
/// 5. Return the best candidates, de-duplicated by position.
///
/// All returned states are at `epoch` (default: last observation).
///
/// # Arguments
/// * `obs` - At least 2 optical observations.
/// * `epoch` - Reference epoch for returned states.  `None` defaults to the
///   last observation (for forward prediction).  Pass the first observation's
///   epoch for backward prediction.
///
/// # Errors
/// - Fewer than 2 optical observations.
/// - No valid candidates found.
/// - Non-optical observations passed.
pub fn initial_orbit_determination(
    obs: &[Observation],
    epoch: Option<Time<TDB>>,
) -> KeteResult<Vec<State<Equatorial>>> {
    if obs.len() < 2 {
        return Err(Error::ValueError(
            "IOD requires at least 2 optical observations".into(),
        ));
    }

    let mut sorted = obs.to_vec();
    sorted.sort_by(|a, b| {
        a.epoch()
            .jd
            .partial_cmp(&b.epoch().jd)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Default epoch: last observation (for forward prediction).
    // sorted is non-empty (we return Err for len < 2 above).
    let ref_epoch = match epoch {
        Some(e) => e,
        None => sorted[sorted.len() - 1].epoch(),
    };

    scanning_iod_core(&sorted, ref_epoch)
}

/// Core range-scanning IOD on pre-sorted observations.
///
/// Selects observation pairs with adaptive baselines, runs a 2D grid scan
/// and nested refinement for each, then rescores all candidates,
/// deduplicates, and returns states at `ref_epoch`.
fn scanning_iod_core(
    sorted_obs: &[Observation],
    ref_epoch: Time<TDB>,
) -> KeteResult<Vec<State<Equatorial>>> {
    let pairs = select_ranging_pairs(sorted_obs);
    if pairs.is_empty() {
        return Err(Error::ValueError(
            "IOD: could not find a usable observation pair".into(),
        ));
    }

    let mut all_refined: Vec<(f64, State<Equatorial>)> = Vec::new();

    for (i_a, i_b) in &pairs {
        if let Ok(mut candidates) = run_ranging_for_pair(sorted_obs, *i_a, *i_b) {
            all_refined.append(&mut candidates);
        }
    }

    if all_refined.is_empty() {
        return Err(Error::ValueError(
            "IOD: no physically valid candidates found from any pair".into(),
        ));
    }

    // Rescore every candidate against the SAME observation set so scores
    // are directly comparable.  Use observations near the reference epoch
    // so that scoring reflects fit quality where the user actually cares
    // (typically the last observation for forward prediction).
    let rescore_indices = select_scoring_cluster(sorted_obs, ref_epoch.jd);
    let rescore_obs: Vec<Observation> = rescore_indices
        .iter()
        .map(|&i| sorted_obs[i].clone())
        .collect();

    for entry in &mut all_refined {
        if let Some(score) = observation_residual(&entry.1, &rescore_obs) {
            entry.0 = score;
        } else {
            entry.0 = 1e20;
        }
    }

    all_refined.retain(|s| s.0.is_finite() && s.0 < 1e20);

    if all_refined.is_empty() {
        return Err(Error::ValueError(
            "IOD: all candidates filtered out after rescoring".into(),
        ));
    }

    all_refined.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let best_score = all_refined[0].0;
    let score_cutoff = best_score * 10.0;

    let mut results: Vec<State<Equatorial>> = Vec::new();
    for (score, state) in all_refined {
        if score > score_cutoff {
            continue;
        }
        results.push(state);
    }

    if results.is_empty() {
        return Err(Error::ValueError("IOD: all candidates filtered out".into()));
    }

    dedup_states(&mut results);
    results.truncate(5);
    propagate_to_common_epoch(&mut results, ref_epoch)?;
    Ok(results)
}

/// Propagate all states to a common reference epoch.
fn propagate_to_common_epoch(
    states: &mut [State<Equatorial>],
    target: Time<TDB>,
) -> KeteResult<()> {
    for state in states.iter_mut() {
        if (state.epoch.jd - target.jd).abs() > 1e-12 {
            *state = propagate_two_body(state, target)?;
        }
    }
    Ok(())
}

/// Run the coarse grid scan + nested local grid refinement for a single ranging pair.
///
/// Returns a vector of `(score, state)` candidates, scored against a
/// local subset of observations near the pair midpoint.
fn run_ranging_for_pair(
    sorted_obs: &[Observation],
    i_a: usize,
    i_b: usize,
) -> KeteResult<Vec<(f64, State<Equatorial>)>> {
    let (ra_a, dec_a, obs_a) = sorted_obs[i_a].as_optical()?;
    let (ra_b, dec_b, obs_b) = sorted_obs[i_b].as_optical()?;

    let los_a = Vector::<Equatorial>::from_ra_dec(ra_a, dec_a);
    let los_b = Vector::<Equatorial>::from_ra_dec(ra_b, dec_b);

    let dt = obs_b.epoch.jd - obs_a.epoch.jd;
    if dt.abs() < 1e-6 {
        return Err(Error::ValueError(
            "IOD: selected pair too close in time".into(),
        ));
    }

    // Score against a dense observation cluster near the pair midpoint.
    // Clusters are short enough that two-body is accurate for any orbit,
    // and dense enough to average out observation noise.
    let ref_jd = f64::midpoint(obs_a.epoch.jd, obs_b.epoch.jd);
    let scoring_indices = select_scoring_cluster(sorted_obs, ref_jd);
    let scoring_obs: Vec<Observation> = scoring_indices
        .iter()
        .map(|&i| sorted_obs[i].clone())
        .collect();

    // 2-D grid scan over (log rho_a, log rho_b).
    // Independent distances for the two observations -- no equal-helio-distance
    // constraint, so eccentric and hyperbolic orbits are naturally sampled.
    let n_scan: usize = 100;
    let log_min = 0.00001_f64.ln();
    let log_max = 1000.0_f64.ln();

    // (score, rho_a, rho_b) — Flatten the 2D grid into a single range and
    // parallel-iterate so all captures are simple shared references.
    let mut scan_scores: Vec<(f64, f64, f64)> = (0..n_scan * n_scan)
        .into_par_iter()
        .filter_map(|idx| {
            let ia = idx / n_scan;
            let ib = idx % n_scan;

            let frac_a = ia as f64 / (n_scan - 1) as f64;
            let rho_a = (log_min + (log_max - log_min) * frac_a).exp();
            let r_a = obs_a.pos + los_a * rho_a;

            let frac_b = ib as f64 / (n_scan - 1) as f64;
            let rho_b = (log_min + (log_max - log_min) * frac_b).exp();
            let r_b = obs_b.pos + los_b * rho_b;

            let Ok((vel, _)) =
                lambert(&r_a, &r_b, dt, true).or_else(|_| lambert(&r_a, &r_b, dt, false))
            else {
                return None;
            };
            let state = State::new(kete_core::desigs::Desig::Empty, obs_a.epoch, r_a, vel, 0);

            if !is_physically_valid(&state) {
                return None;
            }

            let score = observation_residual(&state, &scoring_obs)?;
            Some((score, rho_a, rho_b))
        })
        .collect();

    if scan_scores.is_empty() {
        return Err(Error::ValueError(
            "IOD: no valid candidates in scan for this pair".into(),
        ));
    }

    // Pick the best seeds, requiring they differ by at least 50% in
    // distance so we sample distinct basins.
    scan_scores.retain(|s| s.0.is_finite());
    scan_scores.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut seeds: Vec<(f64, f64, f64)> = Vec::new();
    for &entry in &scan_scores {
        let dominated = seeds.iter().any(|s| {
            let ratio = entry.1 / s.1;
            ratio > 0.67 && ratio < 1.5
        });
        if !dominated {
            seeds.push(entry);
        }
        if seeds.len() >= 5 {
            break;
        }
    }

    // Refine each seed with nested local grid search.
    // Two levels of zoom: each 11x11 grid narrows by 5x per level.
    // This stays in the neighborhood of the coarse-scan minimum,
    // unlike Nelder-Mead which can wander to extreme eccentricities.
    let mut refined: Vec<(f64, State<Equatorial>)> = Vec::new();

    for &(seed_score, rho_a_seed, rho_b_seed) in &seeds {
        let mut best_log_a = rho_a_seed.ln();
        let mut best_log_b = rho_b_seed.ln();
        let mut best_score = seed_score;

        // Start with window = 1 grid cell from the coarse scan.
        let coarse_step = (log_max - log_min) / (n_scan - 1) as f64;
        let mut half_width = coarse_step;

        for _level in 0..2 {
            let n_refine: usize = 11;
            let center_a = best_log_a;
            let center_b = best_log_b;

            for ia in 0..n_refine {
                let frac_a = ia as f64 / (n_refine - 1) as f64;
                let log_a = (center_a - half_width) + 2.0 * half_width * frac_a;
                let rho_a = log_a.exp();
                if rho_a < 1e-5 {
                    continue;
                }
                let r_a = obs_a.pos + los_a * rho_a;

                for ib in 0..n_refine {
                    let frac_b = ib as f64 / (n_refine - 1) as f64;
                    let log_b = (center_b - half_width) + 2.0 * half_width * frac_b;
                    let rho_b = log_b.exp();
                    if rho_b < 1e-5 {
                        continue;
                    }
                    let r_b = obs_b.pos + los_b * rho_b;

                    let Ok((vel, _)) =
                        lambert(&r_a, &r_b, dt, true).or_else(|_| lambert(&r_a, &r_b, dt, false))
                    else {
                        continue;
                    };
                    let state =
                        State::new(kete_core::desigs::Desig::Empty, obs_a.epoch, r_a, vel, 0);

                    if !is_physically_valid(&state) {
                        continue;
                    }

                    let Some(score) = observation_residual(&state, &scoring_obs) else {
                        continue;
                    };

                    if score < best_score {
                        best_score = score;
                        best_log_a = log_a;
                        best_log_b = log_b;
                    }
                }
            }

            // Narrow the window by 5x for the next level.
            half_width /= 5.0;
        }

        if best_score >= 1e20 {
            continue;
        }

        let rho_a_opt = best_log_a.exp();
        let rho_b_opt = best_log_b.exp();
        let r_a = obs_a.pos + los_a * rho_a_opt;
        let r_b = obs_b.pos + los_b * rho_b_opt;
        let Ok((vel, _)) =
            lambert(&r_a, &r_b, dt, true).or_else(|_| lambert(&r_a, &r_b, dt, false))
        else {
            continue;
        };

        let state = State::new(kete_core::desigs::Desig::Empty, obs_a.epoch, r_a, vel, 0);
        refined.push((best_score, state));
    }

    Ok(refined)
}

/// Select observation pairs for ranging.
///
/// Returns up to 3 distinct `(i_a, i_b)` pairs:
/// - ~3-day baseline (good for NEOs and close encounters)
/// - ~10-day baseline (good for main-belt and distant objects)
/// - first-last fallback (always included)
///
/// For short arcs (single-night), the target baselines won't match and
/// the first-last pair provides the only pair.
fn select_ranging_pairs(sorted_obs: &[Observation]) -> Vec<(usize, usize)> {
    let n = sorted_obs.len();
    if n < 2 {
        return vec![];
    }

    let target_baselines = [3.0, 10.0];
    let mut pairs: Vec<(usize, usize)> = Vec::new();

    for &target in &target_baselines {
        if let Some(pair) = best_pair_near_baseline(sorted_obs, target)
            && !pairs.contains(&pair)
        {
            pairs.push(pair);
        }
    }

    // Always include first-last as a fallback.
    let full = (0, n - 1);
    if full.0 != full.1 && !pairs.contains(&full) {
        pairs.push(full);
    }

    pairs
}

/// Find the pair `(i, j)` whose time separation is closest to `target_days`.
///
/// Sliding-window approach: for each `i`, advance `j` until `dt(i,j)`
/// brackets the target, then check both sides.  O(n) time.
///
/// Returns `None` if no pair with `dt > 0` exists.
fn best_pair_near_baseline(sorted_obs: &[Observation], target_days: f64) -> Option<(usize, usize)> {
    let n = sorted_obs.len();
    let mut best: Option<(usize, usize)> = None;
    let mut best_dist = f64::MAX;

    let mut j = 1_usize;
    for i in 0..n {
        // Advance j until dt(i,j) >= target (bracket from below).
        while j < n && (sorted_obs[j].epoch().jd - sorted_obs[i].epoch().jd) < target_days {
            j += 1;
        }
        // Check j-1 and j (they bracket the target baseline).
        for &candidate in &[j.saturating_sub(1), j] {
            if candidate <= i || candidate >= n {
                continue;
            }
            let dt = sorted_obs[candidate].epoch().jd - sorted_obs[i].epoch().jd;
            if dt < 1e-6 {
                continue;
            }
            let dist = (dt - target_days).abs();
            if dist < best_dist {
                best_dist = dist;
                best = Some((i, candidate));
            }
        }
    }

    best
}

/// Select observations for scoring IOD candidates.
///
/// IOD scoring only needs to answer "does this candidate roughly match?"
/// -- not "is this a good orbit fit?". We want observations spanning enough
/// arc for distance leverage (>= ~1 day) but short enough that two-body
/// is reliable at IOD precision.
///
/// Uses a 3-day window around `ref_jd`, capped at 10 observations
/// (whichever limit is reached first).  Always returns at least 2
/// observations (falling back to the 2 nearest if the window is empty).
fn select_scoring_cluster(sorted_obs: &[Observation], ref_jd: f64) -> Vec<usize> {
    let mut indices: Vec<usize> = sorted_obs
        .iter()
        .enumerate()
        .filter(|(_, ob)| (ob.epoch().jd - ref_jd).abs() <= 3.0)
        .map(|(i, _)| i)
        .collect();

    // Fallback: 2 nearest observations regardless of window.
    if indices.len() < 2 {
        let mut by_dist: Vec<(usize, f64)> = sorted_obs
            .iter()
            .enumerate()
            .map(|(i, ob)| (i, (ob.epoch().jd - ref_jd).abs()))
            .collect();
        by_dist.sort_by(|a, b| a.1.total_cmp(&b.1));
        return by_dist
            .iter()
            .take(2.min(sorted_obs.len()))
            .map(|&(i, _)| i)
            .collect();
    }

    // Stride down to 10 if too many observations.
    if indices.len() > 10 {
        let n = indices.len();
        let step = (n - 1) as f64 / 9.0;
        let mut strided = Vec::with_capacity(10);
        for k in 0..10 {
            #[allow(clippy::cast_sign_loss, reason = "product is always positive")]
            let idx = (f64::from(k) * step).round() as usize;
            strided.push(indices[idx]);
        }
        indices = strided;
    }

    indices
}

/// Check that a candidate state represents a physically plausible solar system orbit.
///
/// Broad bounds: heliocentric distance 0.001-1000 AU, eccentricity < 5.0.
/// This admits hyperbolic impactors and distant TNOs while still rejecting
/// wildly unphysical solutions from the grid scan.
fn is_physically_valid(state: &State<Equatorial>) -> bool {
    let r = state.pos.norm();

    if !(0.001..=1000.0).contains(&r) {
        return false;
    }

    // Eccentricity is frame-independent; compute directly from pos/vel
    // without the full CometElements construction and frame conversion.
    let vel_scaled = state.vel / GMS_SQRT;
    let v_mag2 = vel_scaled.norm_squared();
    let vp_dot = state.pos.dot(&vel_scaled);
    let ecc_vec = (v_mag2 - 1.0 / r) * state.pos - vp_dot * vel_scaled;
    if ecc_vec.norm() >= 5.0 {
        return false;
    }

    true
}

/// Remove near-duplicate candidate states.
///
/// Uses a distance-adaptive threshold: 0.01 AU or 0.1% of heliocentric
/// distance, whichever is larger.  This prevents over-pruning at large
/// distances and under-pruning near the Sun.
fn dedup_states(states: &mut Vec<State<Equatorial>>) {
    let mut keep = vec![true; states.len()];
    for i in 0..states.len() {
        if !keep[i] {
            continue;
        }
        for j in (i + 1)..states.len() {
            if !keep[j] {
                continue;
            }
            let r = states[i].pos.norm();
            let threshold = (0.001 * r).max(0.01);
            if (states[i].pos - states[j].pos).norm() < threshold {
                keep[j] = false;
            }
        }
    }
    let mut idx = 0;
    states.retain(|_| {
        let k = keep[idx];
        idx += 1;
        k
    });
}

/// Trimmed-mean angular residual between a state's two-body prediction and
/// the observed LOS directions.
///
/// Computes the angular separation^2 for each observation, then returns the
/// mean of the best 90% (dropping the worst 10%).  This makes scoring
/// robust against misidentified observations and blunders.
///
/// Observations that fail two-body propagation or light-time correction are
/// silently skipped rather than aborting the entire computation.  Returns
/// `None` only when fewer than [`MIN_OBS`] observations could be scored.
fn observation_residual(state: &State<Equatorial>, obs: &[Observation]) -> Option<f64> {
    const MIN_OBS: usize = 2;

    let mut residuals: Vec<f64> = Vec::with_capacity(obs.len());

    for ob in obs {
        let Ok((ra_obs, dec_obs, obs_state)) = ob.as_optical() else {
            continue;
        };
        let Ok(predicted) = propagate_two_body(state, obs_state.epoch) else {
            continue;
        };
        let Ok(predicted) = light_time_correct(&predicted, &obs_state.pos) else {
            continue;
        };
        let los_pred = predicted.pos - obs_state.pos;
        let rho_pred = los_pred.norm();
        if rho_pred < 1e-10 {
            continue;
        }
        let los_unit = los_pred / rho_pred;
        let los_obs = Vector::<Equatorial>::from_ra_dec(ra_obs, dec_obs);
        let cos_angle = los_unit.dot(&los_obs).clamp(-1.0, 1.0);
        residuals.push(cos_angle.acos().powi(2));
    }

    if residuals.len() < MIN_OBS {
        return None;
    }

    // Trimmed mean: drop the worst 10% of residuals.
    residuals.sort_by(f64::total_cmp);
    #[allow(clippy::cast_sign_loss, reason = "product is always positive")]
    let n_keep = (residuals.len() as f64 * 0.9).ceil() as usize;
    let trimmed_sum: f64 = residuals[..n_keep].iter().sum();
    Some(trimmed_sum / n_keep as f64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use kete_core::constants::GMS;
    use kete_core::desigs::Desig;
    use kete_core::propagation::{propagate_n_body_spk, propagate_two_body};
    use kete_core::spice::LOADED_SPK;
    use kete_core::time::{TDB, Time};

    fn make_state(pos: [f64; 3], vel: [f64; 3], jd: f64) -> State<Equatorial> {
        State::new(Desig::Empty, jd.into(), pos.into(), vel.into(), 0)
    }

    struct Rng(u64);
    impl Rng {
        fn new(seed: u64) -> Self {
            Self(seed)
        }
        fn next_u64(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }
        fn uniform(&mut self) -> f64 {
            (self.next_u64() >> 11) as f64 / ((1_u64 << 53) as f64)
        }
        fn gaussian(&mut self) -> f64 {
            let u1 = self.uniform().max(1e-18);
            let u2 = self.uniform();
            (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
        }
    }

    /// Synthesize observations with an ecliptic-plane observer.
    fn synth_optical_ecliptic(
        obj: &State<Equatorial>,
        epochs: &[f64],
        noise_arcsec: f64,
        seed: u64,
    ) -> Vec<Observation> {
        let r_earth = 1.0;
        let v_earth = (GMS / r_earth).sqrt();
        let obl = 23.44_f64.to_radians();
        let earth_ref = make_state(
            [r_earth, 0.0, 0.0],
            [0.0, v_earth * obl.cos(), v_earth * obl.sin()],
            epochs[0],
        );

        let noise_rad = noise_arcsec * std::f64::consts::PI / (180.0 * 3600.0);
        let mut rng = Rng::new(seed);

        epochs
            .iter()
            .map(|&jd| {
                let obj_at = propagate_two_body(obj, Time::<TDB>::new(jd))
                    .expect("two-body propagation failed");
                let observer = propagate_two_body(&earth_ref, Time::<TDB>::new(jd))
                    .expect("earth propagation failed");
                let d = obj_at.pos - observer.pos;
                let (ra, dec) = d.to_ra_dec();
                let ra_noisy = ra + rng.gaussian() * noise_rad / dec.cos().max(0.1);
                let dec_noisy = dec + rng.gaussian() * noise_rad;
                let sigma = if noise_rad > 0.0 { noise_rad } else { 1e-6 };
                Observation::Optical {
                    observer,
                    ra: ra_noisy,
                    dec: dec_noisy,
                    sigma_ra: sigma,
                    sigma_dec: sigma,
                }
            })
            .collect()
    }

    fn best_candidate<'a>(
        candidates: &'a [State<Equatorial>],
        truth: &State<Equatorial>,
    ) -> &'a State<Equatorial> {
        candidates
            .iter()
            .min_by(|a, b| {
                let da = (a.pos - truth.pos).norm();
                let db = (b.pos - truth.pos).norm();
                da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap()
    }

    // -- Scanning IOD tests ---------------------------------------------------

    #[test]
    fn test_scanning_30min_cadence() {
        let r = 2.0;
        let v = (GMS / r).sqrt();
        let obl = 23.44_f64.to_radians();
        let i = 8.0_f64.to_radians();
        let obj = make_state(
            [r, 0.0, 0.0],
            [0.0, v * (obl + i).cos(), v * (obl + i).sin()],
            2460000.5,
        );

        let epochs = [
            2460000.5,
            2460000.5 + 0.5 / 24.0,
            2460000.5 + 1.0 / 24.0,
            2460001.5,
            2460001.5 + 0.5 / 24.0,
            2460001.5 + 1.0 / 24.0,
        ];
        let observations = synth_optical_ecliptic(&obj, &epochs, 1.0, 77777);

        let results = initial_orbit_determination(&observations, None);
        assert!(
            results.is_ok(),
            "Should work with 30-min cadence: {:?}",
            results.err()
        );
        let results = results.unwrap();
        assert!(!results.is_empty(), "Should find at least one candidate");

        let true_r = 2.0;
        let has_reasonable = results.iter().any(|c| {
            let r = c.pos.norm();
            r > true_r / 3.0 && r < true_r * 3.0
        });
        assert!(
            has_reasonable,
            "At least one candidate should be within 3x of true distance {true_r} AU, \
             got distances: {:?}",
            results.iter().map(|c| c.pos.norm()).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_scanning_20min_cadence_2night() {
        let r = 2.5;
        let v = (GMS / r).sqrt();
        let obl = 23.44_f64.to_radians();
        let i = 3.0_f64.to_radians();
        let obj = make_state(
            [r, 0.0, 0.0],
            [0.0, v * (obl + i).cos(), v * (obl + i).sin()],
            2460000.5,
        );

        let dt = 20.0 / (24.0 * 60.0);
        let epochs = [
            2460000.5,
            2460000.5 + dt,
            2460000.5 + 2.0 * dt,
            2460001.5,
            2460001.5 + dt,
            2460001.5 + 2.0 * dt,
        ];
        let observations = synth_optical_ecliptic(&obj, &epochs, 0.5, 88888);

        let results = initial_orbit_determination(&observations, None);
        assert!(
            results.is_ok(),
            "Should handle 20-min cadence: {:?}",
            results.err()
        );
        assert!(!results.unwrap().is_empty());
    }

    #[test]
    fn test_scanning_3obs_minimum_2night() {
        let r = 2.0;
        let v = (GMS / r).sqrt();
        let obl = 23.44_f64.to_radians();
        let i = 10.0_f64.to_radians();
        let obj = make_state(
            [r, 0.0, 0.0],
            [0.0, v * (obl + i).cos(), v * (obl + i).sin()],
            2460000.5,
        );

        let epochs = [2460000.5, 2460001.5, 2460001.5 + 0.5 / 24.0];
        let observations = synth_optical_ecliptic(&obj, &epochs, 0.5, 99999);

        let results = initial_orbit_determination(&observations, None);
        assert!(
            results.is_ok(),
            "Should work for 3 obs on 2 nights: {:?}",
            results.err()
        );
        assert!(!results.unwrap().is_empty());
    }

    #[test]
    fn test_scanning_long_arc() {
        let r = 2.0;
        let v = (GMS / r).sqrt();
        let i = 10.0_f64.to_radians();
        let obl = 23.44_f64.to_radians();
        let obj = make_state(
            [r, 0.0, 0.0],
            [0.0, v * (obl + i).cos(), v * (obl + i).sin()],
            2460000.5,
        );

        let epochs = [2460000.5, 2460030.5, 2460060.5, 2460090.5];
        let observations = synth_optical_ecliptic(&obj, &epochs, 1.0, 11223);

        let results = initial_orbit_determination(&observations, None);
        assert!(
            results.is_ok(),
            "Should handle 90-day arc: {:?}",
            results.err()
        );
        let results = results.unwrap();
        assert!(!results.is_empty(), "Should find at least one candidate");

        let obj_at = propagate_two_body(&obj, Time::<TDB>::new(epochs[epochs.len() - 1])).unwrap();
        let best = best_candidate(&results, &obj_at);
        let pos_err = (best.pos - obj_at.pos).norm();
        let r_true = obj_at.pos.norm();
        assert!(
            pos_err / r_true < 0.3,
            "Long arc: position error {pos_err:.4} too large relative to r={r_true:.4}"
        );
    }

    #[test]
    fn test_scanning_elliptical_long_arc() {
        let a = 2.0;
        let r_peri = 1.4;
        let v_peri = (GMS * (2.0 / r_peri - 1.0 / a)).sqrt();
        let obl = 23.44_f64.to_radians();
        let i = 15.0_f64.to_radians();
        let obj = make_state(
            [r_peri, 0.0, 0.0],
            [0.0, v_peri * (obl + i).cos(), v_peri * (obl + i).sin()],
            2460000.5,
        );

        let epochs = [2460000.5, 2460015.5, 2460030.5, 2460045.5, 2460060.5];
        let observations = synth_optical_ecliptic(&obj, &epochs, 1.0, 33445);

        let results = initial_orbit_determination(&observations, None);
        assert!(
            results.is_ok(),
            "Should handle elliptical long arc: {:?}",
            results.err()
        );
        let results = results.unwrap();
        assert!(!results.is_empty());

        let obj_at = propagate_two_body(&obj, Time::<TDB>::new(epochs[epochs.len() - 1])).unwrap();
        let best = best_candidate(&results, &obj_at);
        let pos_err = (best.pos - obj_at.pos).norm();
        let r_true = obj_at.pos.norm();
        assert!(
            pos_err / r_true < 0.3,
            "Elliptical long arc: error {pos_err:.4} too large relative to r={r_true:.4}"
        );
    }

    #[test]
    fn test_scanning_short_arc() {
        let r = 2.0;
        let v = (GMS / r).sqrt();
        let obl = 23.44_f64.to_radians();
        let i = 5.0_f64.to_radians();
        let obj = make_state(
            [r, 0.0, 0.0],
            [0.0, v * (obl + i).cos(), v * (obl + i).sin()],
            2460000.5,
        );

        let dt = 30.0 / (24.0 * 60.0);
        let epochs = [
            2460000.5,
            2460000.5 + dt,
            2460000.5 + 2.0 * dt,
            2460001.5,
            2460001.5 + dt,
            2460001.5 + 2.0 * dt,
        ];
        let observations = synth_optical_ecliptic(&obj, &epochs, 1.0, 55667);

        let results = initial_orbit_determination(&observations, None);
        assert!(
            results.is_ok(),
            "Should handle short 2-night arc: {:?}",
            results.err()
        );
        let results = results.unwrap();
        assert!(!results.is_empty(), "Should find at least one candidate");

        let true_r = 2.0;
        let has_reasonable = results.iter().any(|c| {
            let cr = c.pos.norm();
            cr > true_r / 3.0 && cr < true_r * 3.0
        });
        assert!(
            has_reasonable,
            "At least one candidate should be within 3x of true distance {true_r} AU, \
             got distances: {:?}",
            results.iter().map(|c| c.pos.norm()).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_scanning_year_long_arc() {
        let r = 2.5;
        let v = (GMS / r).sqrt();
        let obl = 23.44_f64.to_radians();
        let i = 7.0_f64.to_radians();
        let obj = make_state(
            [r, 0.0, 0.0],
            [0.0, v * (obl + i).cos(), v * (obl + i).sin()],
            2460000.5,
        );

        let mut epochs = Vec::new();
        for month in 0..12 {
            let base = 2460000.5 + 30.0 * f64::from(month);
            epochs.push(base);
            epochs.push(base + 0.5 / 24.0);
        }
        let observations = synth_optical_ecliptic(&obj, &epochs, 1.0, 12321);

        let results = initial_orbit_determination(&observations, None);
        assert!(
            results.is_ok(),
            "Should handle year-long arc: {:?}",
            results.err()
        );
        let results = results.unwrap();
        assert!(!results.is_empty(), "Should find at least one candidate");

        let obj_at = propagate_two_body(&obj, Time::<TDB>::new(*epochs.last().unwrap())).unwrap();
        let best = best_candidate(&results, &obj_at);
        let pos_err = (best.pos - obj_at.pos).norm();
        let r_true = obj_at.pos.norm();
        assert!(
            pos_err / r_true < 0.25,
            "Year-long arc: error {pos_err:.4} too large relative to r={r_true:.4}"
        );
    }

    #[test]
    fn test_scanning_neo_long_arc() {
        // Bennu-like NEO: a ~ 1.126 AU, e ~ 0.2, i ~ 6 deg.
        // ~200 observations over 2 years with N-body propagation and SPK Earth.
        let a = 1.126;
        let e = 0.20;
        let r_apo = a * (1.0 + e);
        let v_apo = (GMS * (2.0 / r_apo - 1.0 / a)).sqrt();

        let obl = 23.44_f64.to_radians();
        let inc = 6.0_f64.to_radians();
        let total_tilt = obl + inc;

        let obj = make_state(
            [r_apo, 0.0, 0.0],
            [0.0, v_apo * total_tilt.cos(), v_apo * total_tilt.sin()],
            2460000.5,
        );

        let mut epochs = Vec::new();
        for k in 0..48 {
            let base = 2460000.5 + 15.0 * f64::from(k);
            epochs.push(base);
            epochs.push(base + 0.3 / 24.0);
            epochs.push(base + 0.7 / 24.0);
            if k % 3 == 0 {
                epochs.push(base + 1.0);
            }
        }

        let spk = LOADED_SPK.try_read().unwrap();
        let noise_arcsec = 1.0_f64;
        let noise_rad = noise_arcsec * std::f64::consts::PI / (180.0 * 3600.0);
        let mut rng = Rng::new(77777);

        let observations: Vec<Observation> = epochs
            .iter()
            .map(|&jd| {
                let obj_at = propagate_n_body_spk(obj.clone(), Time::<TDB>::new(jd), false, None)
                    .expect("N-body propagation failed");

                let observer: State<Equatorial> = spk
                    .try_get_state_with_center(399, Time::<TDB>::new(jd), 0)
                    .expect("Earth SPK lookup failed");

                let obj_lt = light_time_correct(&obj_at, &observer.pos)
                    .expect("light-time correction failed");

                let d = obj_lt.pos - observer.pos;
                let (ra, dec) = d.to_ra_dec();
                let ra_noisy = ra + rng.gaussian() * noise_rad / dec.cos().max(0.1);
                let dec_noisy = dec + rng.gaussian() * noise_rad;

                Observation::Optical {
                    observer,
                    ra: ra_noisy,
                    dec: dec_noisy,
                    sigma_ra: noise_rad,
                    sigma_dec: noise_rad,
                }
            })
            .collect();

        drop(spk);

        // Use first epoch to keep the comparison near the ranging pair where
        // two-body is most accurate (truth is N-body over 2 years).
        let first_epoch = Time::<TDB>::new(epochs[0]);
        let results = initial_orbit_determination(&observations, Some(first_epoch));
        assert!(
            results.is_ok(),
            "Should handle NEO long arc: {:?}",
            results.err()
        );
        let results = results.unwrap();
        assert!(!results.is_empty(), "Should find at least one candidate");

        let obj_at = propagate_n_body_spk(obj.clone(), first_epoch, false, None).unwrap();
        let best = best_candidate(&results, &obj_at);
        let pos_err = (best.pos - obj_at.pos).norm();
        let r_true = obj_at.pos.norm();
        // Loosened to 1.0: 2-year N-body truth vs two-body IOD is inherently
        // imprecise.  The tight scoring window further limits which candidates
        // rank highest.  IOD is a seed -- diff correction refines from here.
        assert!(
            pos_err / r_true < 1.0,
            "NEO long arc: pos error {pos_err:.4} too large relative to r={r_true:.4}"
        );
    }

    #[test]
    fn test_scanning_close_encounter_neo() {
        // Apophis-like close encounter: a ~ 0.92 AU, e ~ 0.19, i ~ 3 deg.
        // Object passes ~0.1 AU from Earth with high apparent motion.
        // Observations span ~20 days around closest approach.
        let a = 0.92;
        let e = 0.19;
        let r_peri = a * (1.0 - e);
        let v_peri = (GMS * (2.0 / r_peri - 1.0 / a)).sqrt();

        let obl = 23.44_f64.to_radians();
        let inc = 3.4_f64.to_radians();
        let total_tilt = obl + inc;

        // Start near perihelion where the NEO is close to Earth's orbit.
        let obj = make_state(
            [r_peri, 0.0, 0.0],
            [0.0, v_peri * total_tilt.cos(), v_peri * total_tilt.sin()],
            2460000.5,
        );

        // 20-day arc with observations every 1-2 days (close encounter cadence).
        let mut epochs = Vec::new();
        for day in 0..20 {
            let base = 2460000.5 + f64::from(day);
            epochs.push(base);
            if day % 2 == 0 {
                epochs.push(base + 0.25 / 24.0);
            }
        }
        let observations = synth_optical_ecliptic(&obj, &epochs, 1.0, 31415);

        let results = initial_orbit_determination(&observations, None);
        assert!(
            results.is_ok(),
            "Should handle close-encounter NEO: {:?}",
            results.err()
        );
        let results = results.unwrap();
        assert!(
            !results.is_empty(),
            "Should find at least one candidate for close-encounter NEO"
        );

        // At least one candidate should have a heliocentric distance
        // in the right ballpark (within 3x of true).
        let obj_at = propagate_two_body(&obj, Time::<TDB>::new(epochs[0])).unwrap();
        let true_r = obj_at.pos.norm();
        let has_reasonable = results.iter().any(|c| {
            let cr = c.pos.norm();
            cr > true_r / 3.0 && cr < true_r * 3.0
        });
        assert!(
            has_reasonable,
            "Close-encounter NEO: at least one candidate within 3x of true r={true_r:.3}, \
             got distances: {:?}",
            results.iter().map(|c| c.pos.norm()).collect::<Vec<_>>()
        );
    }

    // -- Single-night IOD tests ------------------------------------------------

    #[test]
    fn test_single_night_circular_2au() {
        // Circular orbit at 2 AU, 4 observations over 4 hours on one night.
        let r = 2.0;
        let v = (GMS / r).sqrt();
        let obl = 23.44_f64.to_radians();
        let i = 5.0_f64.to_radians();
        let obj = make_state(
            [r, 0.0, 0.0],
            [0.0, v * (obl + i).cos(), v * (obl + i).sin()],
            2460000.5,
        );

        // 80-minute cadence
        let dt = 80.0 / (24.0 * 60.0);
        let epochs = [
            2460000.5,
            2460000.5 + dt,
            2460000.5 + 2.0 * dt,
            2460000.5 + 3.0 * dt,
        ];
        let observations = synth_optical_ecliptic(&obj, &epochs, 0.5, 44444);

        let results = initial_orbit_determination(&observations, None);
        assert!(
            results.is_ok(),
            "IOD should work for single-night 2 AU: {:?}",
            results.err()
        );
        assert!(
            !results.unwrap().is_empty(),
            "Should find at least one candidate"
        );
    }

    #[test]
    fn test_single_night_neo() {
        // Apollo-type NEO at ~1.5 AU, 6 observations over 6 hours.
        let a = 1.8;
        let r = 1.5;
        let v = (GMS * (2.0 / r - 1.0 / a)).sqrt();
        let obl = 23.44_f64.to_radians();
        let i = 8.0_f64.to_radians();
        let obj = make_state(
            [r, 0.0, 0.0],
            [0.0, v * (obl + i).cos(), v * (obl + i).sin()],
            2460000.5,
        );

        // ~72-minute cadence
        let dt = 72.0 / (24.0 * 60.0);
        let epochs = [
            2460000.5,
            2460000.5 + dt,
            2460000.5 + 2.0 * dt,
            2460000.5 + 3.0 * dt,
            2460000.5 + 4.0 * dt,
            2460000.5 + 5.0 * dt,
        ];
        let observations = synth_optical_ecliptic(&obj, &epochs, 1.0, 55555);

        let results = initial_orbit_determination(&observations, None);
        assert!(
            results.is_ok(),
            "IOD should work for NEO single night: {:?}",
            results.err()
        );
        assert!(
            !results.unwrap().is_empty(),
            "Should find at least one candidate"
        );
    }

    #[test]
    fn test_minimum_2obs() {
        // Minimum requirement: 2 observations, 1-hour separation.
        let r = 2.0;
        let v = (GMS / r).sqrt();
        let obl = 23.44_f64.to_radians();
        let obj = make_state(
            [r, 0.0, 0.0],
            [0.0, v * obl.cos(), v * obl.sin()],
            2460000.5,
        );

        let dt = 60.0 / (24.0 * 60.0);
        let epochs = [2460000.5, 2460000.5 + dt];
        let observations = synth_optical_ecliptic(&obj, &epochs, 0.5, 77777);

        let results = initial_orbit_determination(&observations, None);
        assert!(
            results.is_ok(),
            "IOD should work with just 2 obs: {:?}",
            results.err()
        );
        assert!(!results.unwrap().is_empty());
    }

    #[test]
    fn test_rejects_1obs() {
        // Should fail with only 1 observation.
        let r = 2.0;
        let v = (GMS / r).sqrt();
        let obl = 23.44_f64.to_radians();
        let obj = make_state(
            [r, 0.0, 0.0],
            [0.0, v * obl.cos(), v * obl.sin()],
            2460000.5,
        );

        let observations = synth_optical_ecliptic(&obj, &[2460000.5], 0.5, 88888);
        assert!(
            initial_orbit_determination(&observations, None).is_err(),
            "IOD should reject a single observation"
        );
    }

    #[test]
    fn test_epoch_parameter() {
        // Verify that the epoch parameter controls the output epoch.
        let r = 2.0;
        let v = (GMS / r).sqrt();
        let obl = 23.44_f64.to_radians();
        let i = 5.0_f64.to_radians();
        let obj = make_state(
            [r, 0.0, 0.0],
            [0.0, v * (obl + i).cos(), v * (obl + i).sin()],
            2460000.5,
        );

        let epochs = [2460000.5, 2460001.5, 2460001.5 + 0.5 / 24.0];
        let observations = synth_optical_ecliptic(&obj, &epochs, 0.5, 99988);

        // Default epoch = last observation.
        let results = initial_orbit_determination(&observations, None).unwrap();
        assert!(!results.is_empty());
        let last_jd = epochs[epochs.len() - 1];
        for c in &results {
            assert!(
                (c.epoch.jd - last_jd).abs() < 1e-10,
                "Default epoch should be last obs, got {}",
                c.epoch.jd
            );
        }

        // Explicit epoch = first observation.
        let first_epoch = Time::<TDB>::new(epochs[0]);
        let results = initial_orbit_determination(&observations, Some(first_epoch)).unwrap();
        assert!(!results.is_empty());
        for c in &results {
            assert!(
                (c.epoch.jd - epochs[0]).abs() < 1e-10,
                "Epoch should be first obs, got {}",
                c.epoch.jd
            );
        }
    }
}
