//! Initial Orbit Determination (IOD).
//!
//! Given optical observations, compute an approximate heliocentric state that
//! can seed the batch least-squares differential corrector.
//!
//! The scanning method works on any observation arc from days to years.  It
//! scans topocentric distance on a log-spaced grid, identifies candidate
//! basins, and refines each with Nelder-Mead optimisation in 2-D.

use kete_core::constants::GMS;
use kete_core::frames::{Equatorial, Vector};
use kete_core::prelude::{Error, KeteResult, State};
use kete_core::propagation::propagate_two_body;

use crate::Observation;

// --- Public entry point ------------------------------------------------------

/// Range-scanning IOD: a robust approach to initial orbit determination.
///
/// Works on any observation arc from days to years.  The algorithm:
///
/// 1. Select a pair of observations with ideal time separation (~3-30 days).
/// 2. Coarse 1-D scan of topocentric distance (log-scale, 200 points).
/// 3. Take the top candidates from the scan as seed basins.
/// 4. Refine each with Nelder-Mead in 2-D (`log rho_a`, `log rho_b`).
/// 5. Return the best candidates, de-duplicated by position.
///
/// Returns all physically valid candidate states (SSB-centered, Equatorial).
///
/// # Errors
/// - Fewer than 3 optical observations.
/// - No valid candidates found.
/// - Non-optical observations passed.
pub fn initial_orbit_determination(obs: &[Observation]) -> KeteResult<Vec<State<Equatorial>>> {
    if obs.len() < 3 {
        return Err(Error::ValueError(
            "IOD requires at least 3 optical observations".into(),
        ));
    }

    let mut sorted = obs.to_vec();
    sorted.sort_by(|a, b| {
        a.epoch()
            .jd
            .partial_cmp(&b.epoch().jd)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    scanning_iod_core(&sorted)
}

// --- Core algorithm ----------------------------------------------------------

/// Core range-scanning IOD on pre-sorted observations.
fn scanning_iod_core(sorted_obs: &[Observation]) -> KeteResult<Vec<State<Equatorial>>> {
    let n = sorted_obs.len();
    if n < 3 {
        return Err(Error::ValueError(
            "IOD requires at least 3 observations".into(),
        ));
    }

    // 1. Select a good pair for ranging.
    let (i_a, i_b) = select_ranging_pair(sorted_obs);
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

    // 2. Build a scoring subset near the ranging pair epoch.
    let ref_jd = f64::midpoint(obs_a.epoch.jd, obs_b.epoch.jd);
    let scoring_indices = select_scoring_subset(sorted_obs, 20, ref_jd, 90.0);
    let scoring_obs: Vec<Observation> = scoring_indices
        .iter()
        .map(|&i| sorted_obs[i].clone())
        .collect();

    // 3. Coarse 1-D grid scan (log-spaced).
    let n_scan: usize = 200;
    let log_min = 0.005_f64.ln();
    let log_max = 120.0_f64.ln();

    let mut scan_scores: Vec<(f64, f64, f64)> = Vec::new(); // (score, rho_a, rho_b)

    for i in 0..n_scan {
        let frac = i as f64 / (n_scan - 1) as f64;
        let rho = (log_min + (log_max - log_min) * frac).exp();

        let r_a = obs_a.pos + los_a * rho;
        let r_helio = r_a.norm();

        let Some(rho_b) = rho_for_helio_distance(&obs_b.pos, &los_b, r_helio) else {
            continue;
        };
        if rho_b < 1e-5 {
            continue;
        }

        let r_b = obs_b.pos + los_b * rho_b;
        let vel = (r_b - r_a) / dt;

        let state = State::new(kete_core::desigs::Desig::Empty, obs_a.epoch, r_a, vel, 0);

        if !is_physically_valid(&state) {
            continue;
        }

        let Some(score) = observation_residual(&state, &scoring_obs) else {
            continue;
        };

        scan_scores.push((score, rho, rho_b));
    }

    if scan_scores.is_empty() {
        return Err(Error::ValueError(
            "IOD: no physically valid candidates found".into(),
        ));
    }

    // 4. Pick the best seeds from the scan.
    //    Sort by score and take up to 5, requiring they differ by at least 50%
    //    in distance so we sample distinct basins.
    scan_scores.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

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

    // 5. Refine each seed with Nelder-Mead in 2-D (log rho_a, log rho_b).
    let objective = |x: &[f64]| -> f64 {
        let rho_a = x[0].exp();
        let rho_b_val = x[1].exp();

        if rho_a < 1e-5 || rho_b_val < 1e-5 {
            return 1e20;
        }

        let r_a = obs_a.pos + los_a * rho_a;
        let r_b = obs_b.pos + los_b * rho_b_val;
        let vel = (r_b - r_a) / dt;

        let state = State::new(kete_core::desigs::Desig::Empty, obs_a.epoch, r_a, vel, 0);

        if !is_physically_valid(&state) {
            return 1e20;
        }

        observation_residual(&state, &scoring_obs).unwrap_or(1e20)
    };

    let mut refined: Vec<(f64, State<Equatorial>)> = Vec::new();

    for (_, rho_a, rho_b_val) in &seeds {
        let log_rho_a = rho_a.ln();
        let log_rho_b = rho_b_val.ln();

        let scale_a = (log_rho_a.abs() * 0.1).max(0.1);
        let scale_b = (log_rho_b.abs() * 0.1).max(0.1);

        let nm_result = kete_stats::fitting::nelder_mead(
            objective,
            &[log_rho_a, log_rho_b],
            &[scale_a, scale_b],
            1e-14,
            500,
        );

        let (best_log_a, best_log_b, best_score) = match nm_result {
            Ok(res) => (res.point[0], res.point[1], res.value),
            Err(_) => (log_rho_a, log_rho_b, objective(&[log_rho_a, log_rho_b])),
        };

        if best_score >= 1e20 {
            continue;
        }

        let rho_a_opt = best_log_a.exp();
        let rho_b_opt = best_log_b.exp();
        let r_a = obs_a.pos + los_a * rho_a_opt;
        let r_b = obs_b.pos + los_b * rho_b_opt;
        let vel = (r_b - r_a) / dt;

        let state = State::new(kete_core::desigs::Desig::Empty, obs_a.epoch, r_a, vel, 0);
        refined.push((best_score, state));
    }

    if refined.is_empty() {
        return Err(Error::ValueError(
            "IOD: refinement produced no valid candidates".into(),
        ));
    }

    // 6. Score-filter and de-duplicate.
    refined.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    let best_score = refined[0].0;
    let score_cutoff = best_score * 10.0;

    let mut results: Vec<State<Equatorial>> = Vec::new();
    for (score, state) in refined {
        if score > score_cutoff {
            continue;
        }
        results.push(state);
    }

    if results.is_empty() {
        return Err(Error::ValueError("IOD: all candidates filtered out".into()));
    }

    dedup_states(&mut results);
    results.truncate(3);
    Ok(results)
}

// --- Helpers -----------------------------------------------------------------

/// Select a pair of observation indices with time separation in the ideal
/// range for finite-difference velocity estimation.
///
/// Prefers a baseline of 3-30 days.  Falls back to the widest pair if none
/// exists in that range.  Returns `(i_a, i_b)` with `i_a < i_b`.
fn select_ranging_pair(sorted_obs: &[Observation]) -> (usize, usize) {
    let n = sorted_obs.len();
    let ideal_min = 3.0_f64;
    let ideal_max = 30.0_f64;

    let mut best = (0_usize, n - 1);
    let mut best_score = f64::MAX;

    let mut j = 0_usize;
    for i in 0..n {
        if j <= i {
            j = i + 1;
        }
        while j < n && (sorted_obs[j].epoch().jd - sorted_obs[i].epoch().jd) < ideal_min {
            j += 1;
        }
        if j >= n {
            break;
        }
        let dt = sorted_obs[j].epoch().jd - sorted_obs[i].epoch().jd;
        let score = if dt <= ideal_max {
            (dt - 10.0).abs()
        } else {
            100.0 + dt
        };
        if score < best_score {
            best_score = score;
            best = (i, j);
        }
    }

    if best_score >= 100.0 {
        best = (0, n - 1);
    }

    best
}

/// Select a well-distributed scoring subset of up to `max_n` observations
/// near a reference epoch.
fn select_scoring_subset(
    sorted_obs: &[Observation],
    max_n: usize,
    ref_jd: f64,
    max_dt_days: f64,
) -> Vec<usize> {
    let mut nearby: Vec<usize> = sorted_obs
        .iter()
        .enumerate()
        .filter(|(_, ob)| (ob.epoch().jd - ref_jd).abs() <= max_dt_days)
        .map(|(i, _)| i)
        .collect();

    if nearby.len() < 3 {
        let mut by_dist: Vec<(usize, f64)> = sorted_obs
            .iter()
            .enumerate()
            .map(|(i, ob)| (i, (ob.epoch().jd - ref_jd).abs()))
            .collect();
        by_dist.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        nearby = by_dist
            .iter()
            .take(3.min(sorted_obs.len()))
            .map(|&(i, _)| i)
            .collect();
        nearby.sort_unstable();
    }

    let n = nearby.len();
    if n <= max_n {
        return nearby;
    }

    let mut indices = Vec::with_capacity(max_n);
    indices.push(nearby[0]);
    let step = (n - 1) as f64 / (max_n - 1) as f64;
    for k in 1..max_n - 1 {
        #[allow(clippy::cast_sign_loss, reason = "step is always positive")]
        let idx = (k as f64 * step).round() as usize;
        let obs_idx = nearby[idx];
        if obs_idx != *indices.last().unwrap() {
            indices.push(obs_idx);
        }
    }
    if *indices.last().unwrap() != nearby[n - 1] {
        indices.push(nearby[n - 1]);
    }
    indices
}

/// Check that a candidate state represents a physically plausible solar system orbit.
fn is_physically_valid(state: &State<Equatorial>) -> bool {
    let r = state.pos.norm();
    let v = state.vel.norm();

    if !(0.05..=500.0).contains(&r) {
        return false;
    }
    if v > 0.06 {
        return false;
    }

    // Require bound (elliptical) orbit.
    let energy = 0.5 * v * v - GMS / r;
    energy < 0.0
}

/// Remove near-duplicate candidate states (position within 0.01 AU).
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
            if (states[i].pos - states[j].pos).norm() < 0.01 {
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

/// Compute the positive topocentric distance rho such that
/// `|R_obs + rho * L_hat| = r_target`.
fn rho_for_helio_distance(
    r_obs: &Vector<Equatorial>,
    los: &Vector<Equatorial>,
    r_target: f64,
) -> Option<f64> {
    let b = 2.0 * r_obs.dot(los);
    let c = r_obs.dot(r_obs) - r_target * r_target;
    let disc = b * b - 4.0 * c;
    if disc < 0.0 {
        return None;
    }
    let sqrt_d = disc.sqrt();
    let rho_plus = (-b + sqrt_d) * 0.5;
    let rho_minus = (-b - sqrt_d) * 0.5;
    match (rho_plus > 0.0, rho_minus > 0.0) {
        (true, true) => Some(rho_plus.min(rho_minus)),
        (true, false) => Some(rho_plus),
        (false, true) => Some(rho_minus),
        _ => None,
    }
}

/// Total angular residual (sum of squared angular errors in radians) between
/// a state's two-body prediction and the observed LOS directions.
fn observation_residual(state: &State<Equatorial>, obs: &[Observation]) -> Option<f64> {
    let mut total = 0.0;
    for ob in obs {
        let (ra_obs, dec_obs, obs_state) = ob.as_optical().ok()?;
        let predicted = propagate_two_body(state, obs_state.epoch).ok()?;
        let los_pred = predicted.pos - obs_state.pos;
        let rho_pred = los_pred.norm();
        if rho_pred < 1e-10 {
            return None;
        }
        let los_unit = los_pred / rho_pred;
        let los_obs = Vector::<Equatorial>::from_ra_dec(ra_obs, dec_obs);
        let cos_angle = los_unit.dot(&los_obs).clamp(-1.0, 1.0);
        total += cos_angle.acos().powi(2);
    }
    Some(total)
}

// --- Tests -------------------------------------------------------------------

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
                da.partial_cmp(&db).unwrap()
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

        let results = initial_orbit_determination(&observations);
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

        let obj_at = propagate_two_body(&obj, Time::<TDB>::new(2460000.5)).unwrap();
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

        let results = initial_orbit_determination(&observations);
        assert!(
            results.is_ok(),
            "Should handle elliptical long arc: {:?}",
            results.err()
        );
        let results = results.unwrap();
        assert!(!results.is_empty());

        let obj_at = propagate_two_body(&obj, Time::<TDB>::new(epochs[0])).unwrap();
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

        let results = initial_orbit_determination(&observations);
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

        let results = initial_orbit_determination(&observations);
        assert!(
            results.is_ok(),
            "Should handle year-long arc: {:?}",
            results.err()
        );
        let results = results.unwrap();
        assert!(!results.is_empty(), "Should find at least one candidate");

        let obj_at = propagate_two_body(&obj, Time::<TDB>::new(epochs[0])).unwrap();
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

                let obj_lt = crate::obs::two_body_lt_state(&obj_at, &observer)
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

        let results = initial_orbit_determination(&observations);
        assert!(
            results.is_ok(),
            "Should handle NEO long arc: {:?}",
            results.err()
        );
        let results = results.unwrap();
        assert!(!results.is_empty(), "Should find at least one candidate");

        let obj_at =
            propagate_n_body_spk(obj.clone(), Time::<TDB>::new(epochs[0]), false, None).unwrap();
        let best = best_candidate(&results, &obj_at);
        let pos_err = (best.pos - obj_at.pos).norm();
        let r_true = obj_at.pos.norm();
        assert!(
            pos_err / r_true < 0.30,
            "NEO long arc: pos error {pos_err:.4} too large relative to r={r_true:.4}"
        );
    }
}
