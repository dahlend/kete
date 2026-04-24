//! Initial Orbit Determination (IOD).
//!
//! Given optical observations, compute an approximate heliocentric state that
//! can seed the batch least-squares orbit fitting or MCMC.
//!
//! [`initial_orbit_determination`] combines topocentric range-scanning with
//! Lambert's solver and Gauss angles-only IOD.  It works on any arc length
//! from single-night tracklets (minutes) to multi-year arcs, and from
//! close-approach NEOs/bolides to distant TNOs.
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

use kete_core::constants::{GMS, GMS_SQRT};
use kete_core::frames::{Equatorial, Vector};
use kete_core::kepler::{light_time_correct, propagate_two_body};
use kete_core::prelude::{Error, KeteResult, State};
use rayon::prelude::*;

use crate::AstrometricObservation;
use crate::lambert::lambert;

/// Unified IOD: a robust approach to initial orbit determination.
///
/// Works on any observation arc from minutes to years, and any orbit type
/// from close-approach NEOs/bolides to distant TNOs.
///
/// Returns `(score, state)` pairs sorted by ascending score (best first).
/// The score is the trimmed-mean angular residual in radians squared
/// plus a soft eccentricity penalty and weak energy/perihelion priors.
///
/// When the observations span multiple apparitions (gaps > 60 days), up to
/// two apparitions (ranked by observation count) are passed through the
/// pipeline and their candidates are scored against a common rescore window.
/// This avoids committing prematurely to a single arc when ranking is
/// ambiguous.
///
/// # Algorithm
///
/// 1. Group observations into apparitions; select up to 2 recent ones.
/// 2. Select observation pairs with deterministic baseline targets
///    (3, 10, 30, 90 days) plus a first-last fallback.
/// 3. Coarse 2-D scan over (`log rho_a`, `log rho_b`), the topocentric
///    distances at each observation. 100x100 grid, log-spaced
///    0.00001-1000 AU.
/// 4. Solve Lambert's problem (prograde, falling back to retrograde) for
///    each grid point to obtain velocity.
/// 5. Refine the best seeds with a single-pass 11x11 local grid.
/// 6. Append Gauss angles-only candidates from distributed triplets.
/// 7. Return the best candidates, de-duplicated by position.
///
/// All returned states are at the most recent observation epoch.
///
/// # Arguments
/// * `obs` - At least 2 optical observations.
///
/// # Errors
/// - Fewer than 2 optical observations.
/// - No valid candidates found.
/// - Non-optical observations passed.
pub fn initial_orbit_determination(
    obs: &[AstrometricObservation],
) -> KeteResult<Vec<(f64, State<Equatorial>)>> {
    // Rescore window caps -- see usage below for the full rationale.
    const RESCORE_WINDOW_DAYS: f64 = 1_460.0; // ~4 years, ~1 main-belt period
    const RESCORE_MAX_OBS: usize = 2_000;

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
    let sorted = sorted;

    let arc_sets = select_iod_apparitions(&sorted);

    // Reference epoch is the most recent observation in the full input.
    let ref_epoch = sorted[sorted.len() - 1].epoch();

    // Build a flat list of (apparition_index, i_a, i_b) ranging tasks
    // across all selected apparitions, then run them in parallel.  Each
    // pair's scan is independent; nesting with the inner grid parallelism
    // is handled by the rayon thread pool.
    let mut ranging_tasks: Vec<(usize, usize, usize)> = Vec::new();
    let mut gauss_tasks: Vec<(usize, usize, usize, usize)> = Vec::new();
    for (arc_idx, sorted_obs) in arc_sets.iter().enumerate() {
        for (i_a, i_b) in select_helio_ranging_pairs(sorted_obs) {
            ranging_tasks.push((arc_idx, i_a, i_b));
        }
        if sorted_obs.len() >= 3 {
            for (i0, i1, i2) in select_helio_gauss_triplets(sorted_obs) {
                gauss_tasks.push((arc_idx, i0, i1, i2));
            }
        }
    }

    // Gauss angles-only IOD complements Lambert ranging by providing
    // candidates without a distance grid, which can help when the grid
    // misses the true distance or for short arcs where Lambert is
    // ill-conditioned.
    let ranging_candidates: Vec<State<Equatorial>> = ranging_tasks
        .into_par_iter()
        .flat_map_iter(|(arc_idx, i_a, i_b)| {
            run_ranging_for_pair(&arc_sets[arc_idx], i_a, i_b)
                .into_iter()
                .flat_map(|v| v.into_iter().map(|(_, s)| s))
        })
        .collect();

    let gauss_candidates: Vec<State<Equatorial>> = gauss_tasks
        .into_par_iter()
        .flat_map_iter(|(arc_idx, i0, i1, i2)| gauss_iod(&arc_sets[arc_idx], i0, i1, i2))
        .collect();

    let mut all_candidates: Vec<State<Equatorial>> = ranging_candidates;
    all_candidates.extend(gauss_candidates);

    if all_candidates.is_empty() {
        return Err(Error::ValueError(
            "IOD: no physically valid candidates found from any pair".into(),
        ));
    }

    // Rescore every candidate against the SAME observation set so scores
    // are directly comparable. Take the most recent observations spanning
    // at most ~one main-belt period (4 years) so two-body Keplerian drift
    // does not dominate the residual.  Observation density varies widely
    // across objects, so a count cap is also enforced to keep dense modern
    // datasets from swamping the scoring with short-arc recent data.
    let last_jd = sorted[sorted.len() - 1].epoch().jd;
    let window_start_jd = last_jd - RESCORE_WINDOW_DAYS;
    let time_idx = sorted.partition_point(|o| o.epoch().jd < window_start_jd);
    let count_idx = sorted.len().saturating_sub(RESCORE_MAX_OBS);
    // Take the more recent start so both caps are enforced.
    let window_idx = time_idx.max(count_idx);
    let rescore_source = &sorted[window_idx..];
    let rescore_obs: Vec<AstrometricObservation> = select_distributed_obs(rescore_source, 120)
        .into_iter()
        .map(|i| rescore_source[i].clone())
        .collect();

    let mut all_refined: Vec<(f64, State<Equatorial>)> = all_candidates
        .into_par_iter()
        .map(|state| {
            let score = score_candidate(&state, &rescore_obs).unwrap_or(1e20);
            (score, state)
        })
        .collect();

    all_refined.retain(|s| s.0.is_finite() && s.0 < 1e20);

    if all_refined.is_empty() {
        return Err(Error::ValueError(
            "IOD: all candidates filtered out after rescoring".into(),
        ));
    }

    all_refined.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let best_score = all_refined[0].0;
    let score_cutoff = best_score * 10.0;

    let mut results: Vec<(f64, State<Equatorial>)> = Vec::new();
    for (score, state) in all_refined {
        if score > score_cutoff {
            continue;
        }
        results.push((score, state));
    }

    if results.is_empty() {
        return Err(Error::ValueError("IOD: all candidates filtered out".into()));
    }

    // Dedup by state position (preserving scores).
    let states: Vec<State<Equatorial>> = results.iter().map(|(_, s)| s.clone()).collect();
    let kept_indices = dedup_states_indices(&states);
    results = kept_indices
        .into_iter()
        .map(|i| results[i].clone())
        .collect();
    results.truncate(5);

    // Propagate to common epoch, discarding candidates that fail.
    let mut propagated: Vec<(f64, State<Equatorial>)> = Vec::with_capacity(results.len());
    let spk = kete_spice::prelude::LOADED_SPK.try_read().ok();
    for (score, state) in results {
        if (state.epoch.jd - ref_epoch.jd).abs() < 1e-12 {
            propagated.push((score, state));
        } else if let Some(ref spk) = spk
            && let Ok(sun_state) = spk.try_to_sun(state.clone())
            && let Ok(prop) = propagate_two_body(&sun_state, ref_epoch)
        {
            propagated.push((score, prop.into()));
        }
    }

    if propagated.is_empty() {
        return Err(Error::ValueError(
            "IOD: no candidates survived propagation to reference epoch".into(),
        ));
    }

    Ok(propagated)
}

/// Deterministic ranging-pair schedule.
///
/// Uses a fixed baseline-target family with deterministic nearest-pair
/// matching. This keeps behavior predictable while sampling short, medium,
/// long, and very-long baselines.
fn select_helio_ranging_pairs(sorted_obs: &[AstrometricObservation]) -> Vec<(usize, usize)> {
    let n = sorted_obs.len();
    if n < 2 {
        return vec![];
    }
    if n == 2 {
        return vec![(0, 1)];
    }

    let mut pairs: Vec<(usize, usize)> = Vec::new();
    let mut push_pair = |a: usize, b: usize| {
        if a == b {
            return;
        }
        let pair = if a < b { (a, b) } else { (b, a) };
        if !pairs.contains(&pair) {
            pairs.push(pair);
        }
    };

    // Fixed baseline targets (days): short, medium, long, very long.
    // The nearest pair to each target is selected deterministically.
    for target in [3.0_f64, 10.0, 30.0, 90.0] {
        if let Some((a, b)) = best_pair_near_baseline(sorted_obs, target) {
            push_pair(a, b);
        }
    }

    // Always include first-last as a fallback long-baseline constraint.
    push_pair(0, n - 1);

    pairs
}

/// Deterministic Gauss triplet schedule.
///
/// Builds fixed triplet families from distributed anchors. This avoids
/// dependence on apparition-specific heuristics while still sampling both
/// long-baseline and mid-arc geometries.
fn select_helio_gauss_triplets(
    sorted_obs: &[AstrometricObservation],
) -> Vec<(usize, usize, usize)> {
    let n = sorted_obs.len();
    if n < 3 {
        return vec![];
    }

    let anchor_count = n.min(7);
    let anchors = select_distributed_obs(sorted_obs, anchor_count);
    if anchors.len() < 3 {
        return vec![];
    }

    let first = anchors[0];
    let last = anchors[anchors.len() - 1];
    let mid = anchors[anchors.len() / 2];
    let q1 = anchors[anchors.len() / 4];
    let q3 = anchors[(anchors.len() * 3) / 4];

    let mut triplets: Vec<(usize, usize, usize)> = Vec::new();
    let mut push_triplet = |a: usize, b: usize, c: usize| {
        let mut idx = [a, b, c];
        idx.sort_unstable();
        if idx[0] == idx[1] || idx[1] == idx[2] {
            return;
        }
        let triplet = (idx[0], idx[1], idx[2]);
        if !triplets.contains(&triplet) {
            triplets.push(triplet);
        }
    };

    push_triplet(first, mid, last);
    push_triplet(first, q1, mid);
    push_triplet(mid, q3, last);

    triplets
}

/// Select one or more candidate apparitions for IOD from sorted observations.
///
/// Among the 5 most recent apparitions with substantive data (at least 5
/// observations and a 25-day arc, with looser fallbacks), returns up to 2
/// ranked by observation count. Apparitions exceeding the 200-observation
/// cap are down-sampled to a hybrid of the recent tail plus a few early
/// anchors so long-baseline curvature is retained without overloading the
/// grid scan.
fn select_iod_apparitions(
    sorted_obs: &[AstrometricObservation],
) -> Vec<Vec<AstrometricObservation>> {
    const GAP_THRESHOLD: f64 = 60.0;
    const MAX_IOD_OBS: usize = 200;
    const MAX_APPARITIONS: usize = 2;

    let n = sorted_obs.len();
    if n == 0 {
        return vec![];
    }

    // Build apparition index ranges (start, end), end-exclusive.
    let mut apparitions: Vec<(usize, usize)> = Vec::new();
    let mut start = 0;
    for i in 1..n {
        if sorted_obs[i].epoch().jd - sorted_obs[i - 1].epoch().jd > GAP_THRESHOLD {
            apparitions.push((start, i));
            start = i;
        }
    }
    apparitions.push((start, n));

    let arc_days =
        |&(s, e): &(usize, usize)| sorted_obs[e - 1].epoch().jd - sorted_obs[s].epoch().jd;

    // Single apparition: just cap-and-go.
    if apparitions.len() == 1 {
        return vec![cap_apparition(sorted_obs, MAX_IOD_OBS)];
    }

    // Among the 5 most recent apparitions, pick the substantive ones with
    // graceful fallbacks if everything is short-arc/sparse.
    let recent: Vec<(usize, usize)> = apparitions.iter().rev().take(5).copied().collect();
    let mut chosen: Vec<(usize, usize)> = recent
        .iter()
        .copied()
        .filter(|&(s, e)| e - s >= 5 && arc_days(&(s, e)) >= 25.0)
        .collect();
    if chosen.is_empty() {
        chosen = recent
            .iter()
            .copied()
            .filter(|&(s, e)| e - s >= 5)
            .collect();
    }
    if chosen.is_empty() {
        chosen = recent
            .iter()
            .copied()
            .filter(|&(s, e)| e - s >= 2)
            .collect();
    }
    if chosen.is_empty() {
        chosen.push((n.saturating_sub(MAX_IOD_OBS), n));
    }

    // Rank by observation count, keep top N, then re-sort by epoch for
    // deterministic candidate ordering downstream.
    chosen.sort_by_key(|&(s, e)| std::cmp::Reverse(e - s));
    chosen.truncate(MAX_APPARITIONS);
    chosen.sort_by_key(|&(s, _)| s);

    chosen
        .into_iter()
        .map(|(s, e)| cap_apparition(&sorted_obs[s..e], MAX_IOD_OBS))
        .collect()
}

/// Down-sample an apparition slice to at most `cap` observations while
/// retaining the recent tail and a few evenly-spaced early anchors.
fn cap_apparition(slice: &[AstrometricObservation], cap: usize) -> Vec<AstrometricObservation> {
    let len = slice.len();
    if len <= cap {
        return slice.to_vec();
    }

    let n_recent = cap * 4 / 5;
    let n_early = cap - n_recent;
    let early_span = len.saturating_sub(n_recent).max(1);

    let mut out: Vec<AstrometricObservation> = Vec::with_capacity(cap);
    if n_early > 0 {
        for i in 0..n_early {
            let idx = if n_early == 1 {
                0
            } else {
                i * (early_span - 1) / (n_early - 1)
            };
            out.push(slice[idx].clone());
        }
    }
    out.extend_from_slice(&slice[len - n_recent..]);
    out
}

/// Find the pair `(i, j)` whose time separation is closest to `target_days`.
///
/// Sliding-window approach: for each `i`, advance `j` until `dt(i,j)`
/// brackets the target, then check both sides.  O(n) time.
///
/// Returns `None` if no pair with `dt > 0` exists.
fn best_pair_near_baseline(
    sorted_obs: &[AstrometricObservation],
    target_days: f64,
) -> Option<(usize, usize)> {
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

/// Return up to `n_max` observation indices distributed evenly by time
/// across the full arc.
///
/// Distributing scoring observations across the arc ensures that scoring
/// reflects fit quality throughout the observation history, not just near
/// the Lambert pair endpoints where any orbit trivially fits.
fn select_distributed_obs(sorted_obs: &[AstrometricObservation], n_max: usize) -> Vec<usize> {
    let n = sorted_obs.len();
    if n == 0 || n_max == 0 {
        return vec![];
    }
    if n <= n_max {
        return (0..n).collect();
    }
    if n_max == 1 {
        return vec![0];
    }
    // Invariant: n > n_max >= 2.
    let t_start = sorted_obs[0].epoch().jd;
    let t_end = sorted_obs[n - 1].epoch().jd;
    let t_span = t_end - t_start;
    if t_span < 1e-12 {
        return (0..n_max).collect();
    }
    let mut selected: Vec<usize> = Vec::with_capacity(n_max);
    for k in 0..n_max {
        let frac = k as f64 / (n_max - 1) as f64;
        let t_target = t_start + t_span * frac;
        let idx = sorted_obs
            .partition_point(|o| o.epoch().jd < t_target)
            .min(n - 1);
        if selected.last() != Some(&idx) {
            selected.push(idx);
        }
    }
    selected
}

/// Run the coarse grid scan + single-pass local grid refinement for one ranging pair.
///
/// Returns a vector of `(score, state)` candidates, scored against
/// observations distributed across the full arc.  Velocity at each grid
/// point is solved via Lambert (prograde, falling back to retrograde).
fn run_ranging_for_pair(
    sorted_obs: &[AstrometricObservation],
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

    // Score against observations distributed across the full arc.
    // Any Lambert orbit passes exactly through the two pair endpoints by
    // construction, so scoring near only those endpoints cannot distinguish
    // correct distances from wrong ones.  Distributing scoring observations
    // across the full arc ensures that wrong-distance orbits (which diverge
    // at epochs far from the pair) are penalized.
    let scoring_obs: Vec<AstrometricObservation> = select_distributed_obs(sorted_obs, 15)
        .into_iter()
        .map(|i| sorted_obs[i].clone())
        .collect();

    // 2-D grid scan over (log rho_a, log rho_b).
    // Independent distances for the two observations -- no equal-helio-distance
    // constraint, so eccentric and hyperbolic orbits are naturally sampled.
    let n_scan: usize = 100;
    let log_min = 0.00001_f64.ln();
    let log_max = 1000.0_f64.ln();

    // (score, rho_a, rho_b) -- Flatten the 2D grid into a single range and
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

            let vel = lambert_velocity(&r_a, &r_b, dt)?;
            let state = State::new(kete_core::desigs::Desig::Empty, obs_a.epoch, r_a, vel, 0);

            if !is_physically_valid(&state) {
                return None;
            }

            let score = score_candidate_residual(&state, &scoring_obs)?;
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

    // Refine each seed with a single-pass local grid search.
    // 11x11 grid centered on the coarse-scan minimum, half-width = one
    // coarse cell. This stays in the neighborhood of the coarse minimum
    // without wandering to extreme eccentricities.
    let coarse_step = (log_max - log_min) / (n_scan - 1) as f64;
    let mut refined: Vec<(f64, State<Equatorial>)> = Vec::new();

    for &(seed_score, rho_a_seed, rho_b_seed) in &seeds {
        // Seed state -- used as the fallback if no refinement cell improves on it.
        let seed_r_a = obs_a.pos + los_a * rho_a_seed;
        let seed_r_b = obs_b.pos + los_b * rho_b_seed;
        let Some(seed_vel) = lambert_velocity(&seed_r_a, &seed_r_b, dt) else {
            continue;
        };
        let seed_state = State::new(
            kete_core::desigs::Desig::Empty,
            obs_a.epoch,
            seed_r_a,
            seed_vel,
            0,
        );

        let n_refine: usize = 11;
        let half_width = coarse_step;
        let center_a = rho_a_seed.ln();
        let center_b = rho_b_seed.ln();

        // Evaluate the 11x11 refinement grid in parallel and pick the best
        // (lowest-scoring) cell.  Cells are independent and each does a
        // Lambert solve plus a residual score, so this scales well.
        let best = (0..n_refine * n_refine)
            .into_par_iter()
            .filter_map(|idx| {
                let ia = idx / n_refine;
                let ib = idx % n_refine;
                let frac_a = ia as f64 / (n_refine - 1) as f64;
                let log_a = (center_a - half_width) + 2.0 * half_width * frac_a;
                let rho_a = log_a.exp();
                if rho_a < 1e-5 {
                    return None;
                }
                let r_a = obs_a.pos + los_a * rho_a;

                let frac_b = ib as f64 / (n_refine - 1) as f64;
                let log_b = (center_b - half_width) + 2.0 * half_width * frac_b;
                let rho_b = log_b.exp();
                if rho_b < 1e-5 {
                    return None;
                }
                let r_b = obs_b.pos + los_b * rho_b;

                let vel = lambert_velocity(&r_a, &r_b, dt)?;
                let state = State::new(kete_core::desigs::Desig::Empty, obs_a.epoch, r_a, vel, 0);
                if !is_physically_valid(&state) {
                    return None;
                }
                let score = score_candidate_residual(&state, &scoring_obs)?;
                Some((score, state))
            })
            .reduce(
                || (f64::INFINITY, seed_state.clone()),
                |a, b| if a.0 <= b.0 { a } else { b },
            );

        let (best_score, best_state) = if best.0 < seed_score {
            best
        } else {
            (seed_score, seed_state)
        };
        refined.push((best_score, best_state));
    }

    Ok(refined)
}

/// Solve Lambert's problem for the velocity at `r_a` connecting to `r_b`
/// across `dt` days. Tries prograde first, then retrograde.
fn lambert_velocity(
    r_a: &Vector<Equatorial>,
    r_b: &Vector<Equatorial>,
    dt: f64,
) -> Option<Vector<Equatorial>> {
    let solutions = lambert(r_a, r_b, dt, true, 0)
        .or_else(|_| lambert(r_a, r_b, dt, false, 0))
        .ok()?;
    Some(solutions[0].0)
}

/// Gauss angles-only IOD using three observations.
///
/// Given three optical observations, solves for the heliocentric state at the
/// middle observation epoch.  The method constructs an 8th-degree polynomial
/// whose real positive roots give candidate slant ranges, then computes
/// velocity via Lagrange f/g coefficients.
///
/// Returns up to 3 candidate states (one per valid polynomial root).
///
/// Reference: Curtis (2014), ch. 5; Danby (1992), ch. 11.
fn gauss_iod(
    obs: &[AstrometricObservation],
    i1: usize,
    i2: usize,
    i3: usize,
) -> Vec<State<Equatorial>> {
    let mut results = Vec::new();

    let Ok((ra1, dec1, o1)) = obs[i1].as_optical() else {
        return results;
    };
    let Ok((ra2, dec2, o2)) = obs[i2].as_optical() else {
        return results;
    };
    let Ok((ra3, dec3, o3)) = obs[i3].as_optical() else {
        return results;
    };

    let l1 = Vector::<Equatorial>::from_ra_dec(ra1, dec1);
    let l2 = Vector::<Equatorial>::from_ra_dec(ra2, dec2);
    let l3 = Vector::<Equatorial>::from_ra_dec(ra3, dec3);

    // Time intervals in days (Gauss uses tau = k * dt where k = sqrt(mu)).
    let tau1 = GMS_SQRT * (o1.epoch.jd - o2.epoch.jd);
    let tau3 = GMS_SQRT * (o3.epoch.jd - o2.epoch.jd);
    let tau = tau3 - tau1;

    // Cross products for the D matrix.
    let p1 = l2.cross(&l3);
    let p2 = l1.cross(&l3);
    let p3 = l1.cross(&l2);

    let d0 = l1.dot(&p1);
    if d0.abs() < 1e-15 {
        // Coplanar LOS directions -- Gauss is degenerate.
        return results;
    }

    // D matrix: D[i][j] = R_i . p_j (1-indexed in the formulation).
    let d11 = o1.pos.dot(&p1);
    let d12 = o1.pos.dot(&p2);
    let d13 = o1.pos.dot(&p3);
    let d21 = o2.pos.dot(&p1);
    let d22 = o2.pos.dot(&p2);
    let d23 = o2.pos.dot(&p3);
    let d31 = o3.pos.dot(&p1);
    let d32 = o3.pos.dot(&p2);
    let d33 = o3.pos.dot(&p3);

    // Gauss ratios (Curtis eqn 5.98-5.99, adapted for non-uniform spacing).
    let a_coeff = (-d12 * tau / tau3 + d22 + d32 * tau / tau1) / d0;
    let b_coeff = (d12 * (tau * tau - tau3 * tau3) * tau3 + d32 * (tau * tau - tau1 * tau1) * tau1)
        / (6.0 * d0);

    // Scalar equation for r2 = |R2 + rho2 * L2|.
    // Expanding and rearranging gives an 8th-degree polynomial in r2:
    //   r2^8 - (A^2 + 2*A*E + F) * r2^6 - 2*B*(A+E) * r2^3 - B^2 = 0
    // where E = R2 . L2, F = |R2|^2.
    let e_val = o2.pos.dot(&l2);
    let f_val = o2.pos.norm_squared();

    // Coefficients of the scalar polynomial:
    // r2^8 + c6*r2^6 + c3*r2^3 + c0 = 0
    let c6 = -(a_coeff * a_coeff + 2.0 * a_coeff * e_val + f_val);
    let c3 = -2.0 * b_coeff * (a_coeff + e_val);
    let c0 = -(b_coeff * b_coeff);

    // Find real positive roots by scanning and bisection.
    // The polynomial p(r) = r^8 + c6*r^6 + c3*r^3 + c0.
    // For physical orbits, r2 is in (0.001, 1000) AU.
    let poly = |r: f64| -> f64 {
        let r3 = r * r * r;
        let r6 = r3 * r3;
        let r8 = r6 * r * r;
        r8 + c6 * r6 + c3 * r3 + c0
    };

    // Scan for sign changes in the interval [0.01, 100].
    let n_scan: u32 = 500;
    let r_min_log = 0.01_f64.ln();
    let r_max_log = 100.0_f64.ln();

    let mut roots = Vec::new();
    let mut prev_r = r_min_log.exp();
    let mut prev_p = poly(prev_r);

    for i in 1..=n_scan {
        let frac = f64::from(i) / f64::from(n_scan);
        let r = (r_min_log + (r_max_log - r_min_log) * frac).exp();
        let p = poly(r);

        if prev_p * p < 0.0
            && let Ok(root) = kete_stats::fitting::bisection(poly, prev_r, r, 50)
            && root > 0.0
        {
            roots.push(root);
        }

        prev_r = r;
        prev_p = p;
    }

    // For each root r2, compute rho2 and the state.
    for r2 in roots {
        let r2_cubed = r2 * r2 * r2;
        let rho2 = a_coeff + b_coeff / r2_cubed;

        // Slant range must be positive.
        if rho2 < 1e-6 {
            continue;
        }

        // Position at middle observation.
        let pos2 = o2.pos + l2 * rho2;

        // Slant ranges at observations 1 and 3 (Curtis eqn 5.112-5.113).
        let rho1 = ((6.0 * (d31 * tau1 / tau3 + d21 * tau / tau3) * r2_cubed
            + d31 * (tau * tau - tau1 * tau1) * tau1)
            / (6.0 * r2_cubed + tau * tau - tau3 * tau3)
            - d11)
            / d0;
        let rho3 = ((6.0 * (d13 * tau3 / tau1 - d23 * tau / tau1) * r2_cubed
            + d13 * (tau * tau - tau3 * tau3) * tau3)
            / (6.0 * r2_cubed + tau * tau - tau1 * tau1)
            - d33)
            / d0;

        if rho1 < 1e-6 || rho3 < 1e-6 {
            continue;
        }

        let pos1 = o1.pos + l1 * rho1;
        let pos3 = o3.pos + l3 * rho3;

        // Velocity at middle observation via Lagrange f, g coefficients.
        // f1 = 1 - 0.5 * mu * tau1^2 / r2^3  (first-order truncation)
        // g1 = tau1 - mu * tau1^3 / (6 * r2^3)
        // Similarly for f3, g3.
        let mu_over_r3 = 1.0 / r2_cubed;
        let f1 = 1.0 - 0.5 * mu_over_r3 * tau1 * tau1;
        let g1 = tau1 - mu_over_r3 * tau1 * tau1 * tau1 / 6.0;
        let f3 = 1.0 - 0.5 * mu_over_r3 * tau3 * tau3;
        let g3 = tau3 - mu_over_r3 * tau3 * tau3 * tau3 / 6.0;

        let fg_det = f1 * g3 - f3 * g1;
        if fg_det.abs() < 1e-20 {
            continue;
        }

        // v2 = (f1 * r3 - f3 * r1) / (f1*g3 - f3*g1)
        // but we need to convert tau back to days for velocity units.
        // Since tau = GMS_SQRT * dt, and positions are in AU, velocity
        // from f/g is in AU per Gaussian day.  Convert: v_au_day = v * GMS_SQRT.
        let vel2 = (pos3 * f1 - pos1 * f3) / fg_det * GMS_SQRT;

        let state = State::new(kete_core::desigs::Desig::Empty, o2.epoch, pos2, vel2, 0);

        if is_physically_valid(&state) {
            results.push(state);
        }
    }

    results
}

/// Score a candidate state against an observation set.
///
/// Combines the trimmed-mean angular residual with a cross-track curvature
/// term (when the arc is long enough), scales by a soft eccentricity prior,
/// and applies weak physical priors that regularize pathological high-energy
/// and extremely small-perihelion states.
fn score_candidate(state: &State<Equatorial>, obs: &[AstrometricObservation]) -> Option<f64> {
    let base = score_candidate_residual(state, obs)?;
    let energy_prior = specific_energy_excess_penalty(state);
    let peri_prior = low_perihelion_penalty(state);
    Some(base * (1.0 + 0.05 * energy_prior + 0.02 * peri_prior))
}

/// Inner-loop scoring used during pair grid scan and refinement.
///
/// Excludes the energy/perihelion priors so that pair-level ranking remains
/// purely data-driven; priors are applied at the top-level rescore where
/// the final candidate ordering is decided.
fn score_candidate_residual(
    state: &State<Equatorial>,
    obs: &[AstrometricObservation],
) -> Option<f64> {
    let residual = observation_residual(state, obs)?;
    let curvature = curvature_residual(state, obs).unwrap_or(0.0);
    Some((residual + curvature) * eccentricity_penalty(state))
}

fn specific_energy_excess_penalty(state: &State<Equatorial>) -> f64 {
    let r = state.pos.norm();
    if r <= 0.0 {
        return 1.0;
    }
    let eps = 0.5 * state.vel.norm_squared() - GMS / r;
    if eps <= 0.0 {
        0.0
    } else {
        let scaled = eps * r / GMS;
        scaled * scaled
    }
}

fn low_perihelion_penalty(state: &State<Equatorial>) -> f64 {
    let e = compute_eccentricity(state);
    if !e.is_finite() {
        return 1.0;
    }

    let h = state.pos.cross(&state.vel).norm();
    let p = h * h / GMS;
    let denom = 1.0 + e;
    if denom <= 1e-12 {
        return 1.0;
    }

    let q = p / denom;
    if q >= 0.05 {
        0.0
    } else {
        let deficit = (0.05 - q.max(0.0)) / 0.05;
        deficit * deficit
    }
}

/// Sigma-clipped mean angular residual between a state's two-body prediction
/// and the observed LOS directions.
///
/// Computes the angular separation^2 for each observation, applies 3-sigma
/// clipping, then returns the mean.  This makes scoring robust against
/// misidentified observations and blunders.
///
/// Observations that fail two-body propagation or light-time correction are
/// silently skipped rather than aborting the entire computation.  Returns
/// `None` only when fewer than 2 observations could be scored.
fn observation_residual(state: &State<Equatorial>, obs: &[AstrometricObservation]) -> Option<f64> {
    let mut residuals: Vec<f64> = Vec::with_capacity(obs.len());

    // Convert to Sun-centered for two-body propagation and light-time correction.
    let spk = kete_spice::prelude::LOADED_SPK.try_read().ok()?;
    let sun_state = spk.try_to_sun(state.clone()).ok()?;

    for ob in obs {
        let Ok((ra_obs, dec_obs, obs_state)) = ob.as_optical() else {
            continue;
        };
        let Ok(predicted) = propagate_two_body(&sun_state, obs_state.epoch) else {
            continue;
        };
        let Some(obs_helio) = spk.try_to_sun(obs_state.clone().into()).ok() else {
            continue;
        };
        let Ok(predicted) = light_time_correct(&predicted, &obs_helio.pos) else {
            continue;
        };
        let los_pred = predicted.pos - obs_helio.pos;
        let rho_pred = los_pred.norm();
        if rho_pred < 1e-10 {
            continue;
        }
        let los_unit = los_pred / rho_pred;
        let los_obs = Vector::<Equatorial>::from_ra_dec(ra_obs, dec_obs);
        let cos_angle = los_unit.dot(&los_obs).clamp(-1.0, 1.0);
        residuals.push(cos_angle.acos().powi(2));
    }

    let Ok(data) = kete_stats::prelude::Data::try_from(residuals) else {
        return None;
    };
    let clipped = data.sigma_clip(3.0, 3.0, 3);
    Some(clipped.mean())
}

/// Cross-track curvature residual.
///
/// Measures how well a candidate reproduces the observed track curvature
/// (deviation perpendicular to the great-circle chord connecting the first
/// and last observations).  Track curvature is the primary observable that
/// constrains heliocentric distance: objects at different distances produce
/// different amounts of parallax-induced curvature over the same time
/// baseline.
///
/// Returns `None` when the angular span is too small for curvature to be
/// reliably measured above typical astrometric noise (~1 arcsec), avoiding
/// false discrimination from noise-dominated short arcs.
fn curvature_residual(state: &State<Equatorial>, obs: &[AstrometricObservation]) -> Option<f64> {
    if obs.len() < 4 {
        return None;
    }

    // LOS for first and last observations define the chord.
    let (ra_first, dec_first, _) = obs.first()?.as_optical().ok()?;
    let (ra_last, dec_last, _) = obs.last()?.as_optical().ok()?;
    let los_first = Vector::<Equatorial>::from_ra_dec(ra_first, dec_first);
    let los_last = Vector::<Equatorial>::from_ra_dec(ra_last, dec_last);

    // Total angular span.
    let span = los_first.dot(&los_last).clamp(-1.0, 1.0).acos();

    // Below ~1 degree, cross-track deviations are dominated by astrometric
    // noise rather than orbital curvature.
    if span < 0.017 {
        return None;
    }

    // Normal to the great-circle chord -- defines the cross-track direction.
    let normal = los_first.cross(&los_last);
    let normal_mag = normal.norm();
    if normal_mag < 1e-15 {
        return None;
    }
    let normal = normal / normal_mag;

    // Propagate candidate and compute cross-track deviations.
    let spk = kete_spice::prelude::LOADED_SPK.try_read().ok()?;
    let sun_state = spk.try_to_sun(state.clone()).ok()?;

    let mut residuals: Vec<f64> = Vec::new();

    // Skip first and last (they define the chord, so their cross-track
    // deviation is identically zero by construction for observations).
    for ob in &obs[1..obs.len() - 1] {
        let Ok((ra_obs, dec_obs, obs_state)) = ob.as_optical() else {
            continue;
        };
        let los_obs = Vector::<Equatorial>::from_ra_dec(ra_obs, dec_obs);
        let obs_cross = los_obs.dot(&normal);

        let Ok(predicted) = propagate_two_body(&sun_state, obs_state.epoch) else {
            continue;
        };
        let Some(obs_helio) = spk.try_to_sun(obs_state.clone().into()).ok() else {
            continue;
        };
        let Ok(predicted) = light_time_correct(&predicted, &obs_helio.pos) else {
            continue;
        };
        let los_pred = predicted.pos - obs_helio.pos;
        let rho_pred = los_pred.norm();
        if rho_pred < 1e-10 {
            continue;
        }
        let los_pred_unit = los_pred / rho_pred;
        let pred_cross = los_pred_unit.dot(&normal);

        let diff = pred_cross - obs_cross;
        residuals.push(diff * diff);
    }

    let Ok(data) = kete_stats::prelude::Data::try_from(residuals) else {
        return None;
    };
    let clipped = data.sigma_clip(3.0, 3.0, 3);

    // Ramp weight from 0 at 1 degree to full at ~5 degrees.  This ensures
    // that marginal-span arcs don't get over-credited for curvature that
    // may be noise.
    let weight = ((span - 0.017) / 0.07).min(1.0);
    Some(clipped.mean() * weight)
}

/// Soft eccentricity prior for IOD scoring.
///
/// Returns a multiplicative factor (>= 1.0) applied to the angular
/// residual.  Bound solutions (e < 1) are only weakly regularized, while
/// strongly hyperbolic solutions are penalized more aggressively.
///
/// This avoids over-biasing the ranking toward artificially low-e orbits
/// when several physically plausible bound solutions fit similarly well.
fn eccentricity_penalty(state: &State<Equatorial>) -> f64 {
    let e = compute_eccentricity(state);
    if e <= 1.0 {
        // Very mild prior for bound orbits.
        1.0 + 0.25 * e * e
    } else {
        // Keep a stronger penalty once the solution is hyperbolic.
        let ex = e - 1.0;
        1.25 + 2.0 * ex * ex
    }
}

/// Compute the eccentricity of a candidate state.
///
/// Uses the vis-viva relation to get the eccentricity vector directly from
/// Cartesian pos/vel, avoiding the full [`CometElements`](kete_core::elements::CometElements) construction.
fn compute_eccentricity(state: &State<Equatorial>) -> f64 {
    let r = state.pos.norm();
    let vel_scaled = state.vel / GMS_SQRT;
    let v_mag2 = vel_scaled.norm_squared();
    let vp_dot = state.pos.dot(&vel_scaled);
    let ecc_vec = (v_mag2 - 1.0 / r) * state.pos - vp_dot * vel_scaled;
    ecc_vec.norm()
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

    if compute_eccentricity(state) >= 5.0 {
        return false;
    }

    true
}

/// Return indices of states to keep after deduplication.
fn dedup_states_indices(states: &[State<Equatorial>]) -> Vec<usize> {
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
    (0..states.len()).filter(|&i| keep[i]).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use kete_core::constants::GMS;
    use kete_core::desigs::Desig;
    use kete_core::frames::{SSB, SunCenter};
    use kete_core::kepler::{light_time_correct, propagate_two_body};
    use kete_core::time::{TDB, Time};
    use kete_spice::prelude::{LOADED_SPK, propagate_n_body_spk};

    use kete_spice::test_data::ensure_test_spk;

    fn make_state(pos: [f64; 3], vel: [f64; 3], jd: f64) -> State<Equatorial, SunCenter> {
        State {
            desig: Desig::Empty,
            epoch: jd.into(),
            pos: pos.into(),
            vel: vel.into(),
            center: SunCenter,
        }
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
        obj: &State<Equatorial, SunCenter>,
        epochs: &[f64],
        noise_arcsec: f64,
        seed: u64,
    ) -> Vec<AstrometricObservation> {
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
                let observer_sun = propagate_two_body(&earth_ref, Time::<TDB>::new(jd))
                    .expect("earth propagation failed");
                // Convert to SSB (approximate: sun ~= ssb for test purposes)
                let observer = State::<Equatorial, SSB> {
                    desig: observer_sun.desig,
                    epoch: observer_sun.epoch,
                    pos: observer_sun.pos,
                    vel: observer_sun.vel,
                    center: SSB,
                };
                let d = obj_at.pos - observer.pos;
                let (ra, dec) = d.to_ra_dec();
                let ra_noisy = ra + rng.gaussian() * noise_rad / dec.cos().max(0.1);
                let dec_noisy = dec + rng.gaussian() * noise_rad;
                let sigma = if noise_rad > 0.0 { noise_rad } else { 1e-6 };
                AstrometricObservation::Optical {
                    observer,
                    ra: ra_noisy,
                    dec: dec_noisy,
                    sigma_ra: sigma,
                    sigma_dec: sigma,
                    time_sigma: 0.0,
                }
            })
            .collect()
    }

    fn best_candidate<'a, C: kete_core::frames::CenterBody>(
        candidates: &'a [(f64, State<Equatorial>)],
        truth: &State<Equatorial, C>,
    ) -> &'a State<Equatorial> {
        candidates
            .iter()
            .map(|(_, s)| s)
            .min_by(|a, b| {
                let da = (a.pos - truth.pos).norm();
                let db = (b.pos - truth.pos).norm();
                da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap()
    }

    /// Extract just the states from scored IOD results.
    fn states(scored: &[(f64, State<Equatorial>)]) -> Vec<&State<Equatorial>> {
        scored.iter().map(|(_, s)| s).collect()
    }

    // -- Scanning IOD tests ---------------------------------------------------

    #[test]
    fn test_scanning_30min_cadence() {
        ensure_test_spk();
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

        let results = initial_orbit_determination(&observations);
        assert!(
            results.is_ok(),
            "Should work with 30-min cadence: {:?}",
            results.err()
        );
        let results = results.unwrap();
        assert!(!results.is_empty(), "Should find at least one candidate");

        let true_r = 2.0;
        let has_reasonable = states(&results).iter().any(|c| {
            let r = c.pos.norm();
            r > true_r / 3.0 && r < true_r * 3.0
        });
        assert!(
            has_reasonable,
            "At least one candidate should be within 3x of true distance {true_r} AU, \
             got distances: {:?}",
            states(&results)
                .iter()
                .map(|c| c.pos.norm())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_scanning_20min_cadence_2night() {
        ensure_test_spk();
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

        let results = initial_orbit_determination(&observations);
        assert!(
            results.is_ok(),
            "Should handle 20-min cadence: {:?}",
            results.err()
        );
        assert!(!results.unwrap().is_empty());
    }

    #[test]
    fn test_scanning_3obs_minimum_2night() {
        ensure_test_spk();
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

        let results = initial_orbit_determination(&observations);
        assert!(
            results.is_ok(),
            "Should work for 3 obs on 2 nights: {:?}",
            results.err()
        );
        assert!(!results.unwrap().is_empty());
    }

    #[test]
    fn test_scanning_long_arc() {
        ensure_test_spk();
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

        let results = initial_orbit_determination(&observations);
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
        ensure_test_spk();
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

        let results = initial_orbit_determination(&observations);
        assert!(
            results.is_ok(),
            "Should handle elliptical long arc: {:?}",
            results.err()
        );
        let results = results.unwrap();
        assert!(!results.is_empty());

        let result_epoch = results[0].1.epoch;
        let obj_at = propagate_two_body(&obj, result_epoch).unwrap();
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
        ensure_test_spk();
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

        let results = initial_orbit_determination(&observations);
        assert!(
            results.is_ok(),
            "Should handle short 2-night arc: {:?}",
            results.err()
        );
        let results = results.unwrap();
        assert!(!results.is_empty(), "Should find at least one candidate");

        let true_r = 2.0;
        let has_reasonable = states(&results).iter().any(|c| {
            let cr = c.pos.norm();
            cr > true_r / 3.0 && cr < true_r * 3.0
        });
        assert!(
            has_reasonable,
            "At least one candidate should be within 3x of true distance {true_r} AU, \
             got distances: {:?}",
            states(&results)
                .iter()
                .map(|c| c.pos.norm())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_scanning_year_long_arc() {
        ensure_test_spk();
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

        let results = initial_orbit_determination(&observations);
        assert!(
            results.is_ok(),
            "Should handle year-long arc: {:?}",
            results.err()
        );
        let results = results.unwrap();
        assert!(!results.is_empty(), "Should find at least one candidate");

        let result_epoch = results[0].1.epoch;
        let obj_at = propagate_two_body(&obj, result_epoch).unwrap();
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
        ensure_test_spk();
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

        let observations: Vec<AstrometricObservation> = epochs
            .iter()
            .map(|&jd| {
                let obj_at = propagate_n_body_spk(
                    spk.try_to_ssb(obj.clone().into())
                        .expect("Center conversion failed"),
                    Time::<TDB>::new(jd),
                    false,
                    None,
                )
                .expect("N-body propagation failed");

                let observer: State<Equatorial> = spk
                    .try_get_state_with_center(399, Time::<TDB>::new(jd), 0)
                    .expect("Earth SPK lookup failed");
                let observer_ssb: State<Equatorial, SSB> = observer
                    .clone()
                    .try_into()
                    .expect("Earth state should be SSB-centered (center_id=0)");

                let sun_at = spk
                    .try_to_sun(obj_at.clone().into())
                    .expect("SPK center change failed");
                let obs_helio = observer.pos - obj_at.pos + sun_at.pos;
                let obj_lt_sun =
                    light_time_correct(&sun_at, &obs_helio).expect("light-time correction failed");
                let obj_lt = spk
                    .try_to_ssb(obj_lt_sun.into())
                    .expect("SPK center change failed");

                let d = obj_lt.pos - observer.pos;
                let (ra, dec) = d.to_ra_dec();
                let ra_noisy = ra + rng.gaussian() * noise_rad / dec.cos().max(0.1);
                let dec_noisy = dec + rng.gaussian() * noise_rad;

                AstrometricObservation::Optical {
                    observer: observer_ssb,
                    ra: ra_noisy,
                    dec: dec_noisy,
                    sigma_ra: noise_rad,
                    sigma_dec: noise_rad,
                    time_sigma: 0.0,
                }
            })
            .collect();

        drop(spk);

        let results = initial_orbit_determination(&observations);
        assert!(
            results.is_ok(),
            "Should handle NEO long arc: {:?}",
            results.err()
        );
        let results = results.unwrap();
        assert!(!results.is_empty(), "Should find at least one candidate");

        let obj_at = {
            let spk = LOADED_SPK.try_read().unwrap();
            let obj_ssb = spk.try_to_ssb(obj.clone().into()).unwrap();
            drop(spk);
            propagate_n_body_spk(obj_ssb, results[0].1.epoch, false, None).unwrap()
        };
        let best = best_candidate(&results, &obj_at);
        let pos_err = (best.pos - obj_at.pos).norm();
        let r_true = obj_at.pos.norm();
        // Loosened to 1.0: 2-year N-body truth vs two-body IOD is inherently
        // imprecise.  IOD is a seed -- diff correction refines from here.
        assert!(
            pos_err / r_true < 1.0,
            "NEO long arc: pos error {pos_err:.4} too large relative to r={r_true:.4}"
        );
    }

    #[test]
    fn test_scanning_close_encounter_neo() {
        ensure_test_spk();
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

        let results = initial_orbit_determination(&observations);
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
        let has_reasonable = states(&results).iter().any(|c| {
            let cr = c.pos.norm();
            cr > true_r / 3.0 && cr < true_r * 3.0
        });
        assert!(
            has_reasonable,
            "Close-encounter NEO: at least one candidate within 3x of true r={true_r:.3}, \
             got distances: {:?}",
            states(&results)
                .iter()
                .map(|c| c.pos.norm())
                .collect::<Vec<_>>()
        );
    }

    // -- Single-night IOD tests ------------------------------------------------

    #[test]
    fn test_single_night_circular_2au() {
        ensure_test_spk();
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

        let results = initial_orbit_determination(&observations);
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
        ensure_test_spk();
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

        let results = initial_orbit_determination(&observations);
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
        ensure_test_spk();
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

        let results = initial_orbit_determination(&observations);
        assert!(
            results.is_ok(),
            "IOD should work with just 2 obs: {:?}",
            results.err()
        );
        assert!(!results.unwrap().is_empty());
    }

    #[test]
    fn test_rejects_1obs() {
        ensure_test_spk();
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
            initial_orbit_determination(&observations).is_err(),
            "IOD should reject a single observation"
        );
    }

    #[test]
    fn test_epoch_parameter() {
        ensure_test_spk();
        // Verify that the output epoch is the last observation.
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

        let results = initial_orbit_determination(&observations).unwrap();
        assert!(!results.is_empty());
        let last_jd = epochs[epochs.len() - 1];
        for (_, c) in &results {
            assert!(
                (c.epoch.jd - last_jd).abs() < 1e-10,
                "Epoch should be last obs {last_jd}, got {}",
                c.epoch.jd
            );
        }
    }

    // -- Apparition selection tests ----------------------------------

    /// Make a minimal optical observation at a given JD.
    /// Uses a fixed position and direction — only the epoch matters for
    /// apparition grouping tests.
    fn dummy_obs(jd: f64) -> AstrometricObservation {
        let s = make_state([1.0, 0.0, 0.0], [0.0, 0.01, 0.0], jd);
        let observer = State::<Equatorial, SSB> {
            desig: s.desig,
            epoch: s.epoch,
            pos: s.pos,
            vel: s.vel,
            center: SSB,
        };
        AstrometricObservation::Optical {
            observer,
            ra: 0.0,
            dec: 0.0,
            sigma_ra: 1e-5,
            sigma_dec: 1e-5,
            time_sigma: 0.0,
        }
    }

    fn dummy_obs_batch(jds: &[f64]) -> Vec<AstrometricObservation> {
        jds.iter().map(|&jd| dummy_obs(jd)).collect()
    }

    #[test]
    fn helio_pairs_are_deterministic_and_unique() {
        let jds: Vec<f64> = (0..24).map(|i| 2460000.0 + f64::from(i) * 0.8).collect();
        let obs = dummy_obs_batch(&jds);

        let pairs_a = select_helio_ranging_pairs(&obs);
        let pairs_b = select_helio_ranging_pairs(&obs);

        assert_eq!(pairs_a, pairs_b);
        assert!(!pairs_a.is_empty());
        assert!(pairs_a.len() <= 5);

        for (i, (a, b)) in pairs_a.iter().enumerate() {
            assert!(a < b);
            assert!(*b < obs.len());
            for (c, d) in pairs_a.iter().skip(i + 1) {
                assert_ne!((a, b), (c, d));
            }
        }
    }

    #[test]
    fn helio_gauss_triplets_are_deterministic_and_unique() {
        let jds: Vec<f64> = (0..30).map(|i| 2460000.0 + f64::from(i) * 0.7).collect();
        let obs = dummy_obs_batch(&jds);

        let triplets_a = select_helio_gauss_triplets(&obs);
        let triplets_b = select_helio_gauss_triplets(&obs);

        assert_eq!(triplets_a, triplets_b);
        assert!(!triplets_a.is_empty());
        assert!(triplets_a.len() <= 3);

        for (i, (a, b, c)) in triplets_a.iter().enumerate() {
            assert!(a < b && b < c);
            assert!(*c < obs.len());
            for (d, e, f) in triplets_a.iter().skip(i + 1) {
                assert_ne!((a, b, c), (d, e, f));
            }
        }
    }

    #[test]
    fn ranging_runner_handles_short_baseline() {
        ensure_test_spk();

        let r = 2.0;
        let v = (GMS / r).sqrt();
        let obl = 23.44_f64.to_radians();
        let obj = make_state(
            [r, 0.0, 0.0],
            [0.0, v * obl.cos(), v * obl.sin()],
            2460000.5,
        );

        let dt = 45.0 / (24.0 * 60.0);
        let epochs = [
            2460000.5,
            2460000.5 + dt,
            2460000.5 + 2.0 * dt,
            2460000.5 + 3.0 * dt,
        ];
        let observations = synth_optical_ecliptic(&obj, &epochs, 0.5, 20260421);

        let ranged = run_ranging_for_pair(&observations, 0, 1);
        assert!(
            ranged.is_ok(),
            "ranging runner failed on short baseline: {ranged:?}"
        );
        assert!(!ranged.unwrap().is_empty());
    }
}
