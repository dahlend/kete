// SPDX-FileCopyrightText: 2026 Dar Dahlen
// SPDX-License-Identifier: BSD-3-Clause

//! Trajectory analysis that needs body states or N-body propagation.
//!
//! Complements [`analysis`](crate::analysis) (B-plane, orbital elements) with
//! functions that query an [`Ephemeris`](crate::ephemeris::Ephemeris).

use super::n_body::NBody;
use crate::elements::CometElements;
use crate::ephemeris::Ephemeris;
use crate::errors::{Error, KeteResult};
use crate::frames::{Ecliptic, Equatorial};
use crate::state::State;
use crate::time::{TDB, Time};
use nalgebra::Vector3;

/// Find the epoch and distance of closest approach between two objects.
///
/// Both objects are propagated using full N-body mechanics over the search
/// window, with body states from `ephem`. If either state's designation
/// corresponds to a body that `ephem` covers, its ephemeris is used directly
/// instead of N-body propagation.
///
/// A coarse grid scan followed by golden-section refinement locates the
/// minimum separation.
///
/// # Errors
/// Returns an error if the states have different center IDs or the time
/// window is non-positive.
pub fn closest_approach<E: Ephemeris>(
    ephem: &E,
    state_a: &State<Equatorial>,
    state_b: &State<Equatorial>,
    jd_start: Time<TDB>,
    jd_end: Time<TDB>,
    include_extended: bool,
) -> KeteResult<(Time<TDB>, f64)> {
    if state_a.center_id() != state_b.center_id() {
        return Err(Error::ValueError(
            "Both states must share the same center_id".into(),
        ));
    }

    let span = (jd_end - jd_start).elapsed;
    if span <= 0.0 {
        return Err(Error::ValueError("jd_end must be after jd_start".into()));
    }

    // Adaptive sample count: at least 20 samples per orbital period of the
    // shorter-period object, minimum 200 total. The period is heliocentric. A
    // state the ephemeris cannot move to the Sun only loses the adaptive count.
    let period = |state: &State<Equatorial>| {
        let mut sun_state = state.clone();
        if ephem.try_change_center(&mut sun_state, 10).is_err() {
            return f64::NAN;
        }
        CometElements::from_state(&sun_state.into_frame::<Ecliptic>())
            .map_or(f64::NAN, |elem| elem.orbital_period())
    };
    let min_period = period(state_a).min(period(state_b));
    #[allow(clippy::cast_sign_loss, reason = "always positive by construction")]
    let n_samples = if min_period.is_finite() && min_period > 0.0 {
        ((span / min_period) * 20.0).ceil().max(200.0) as usize
    } else {
        200
    };
    let dt = span / n_samples as f64;

    let mut cur_a = state_at_time(state_a, jd_start, ephem, include_extended)?;
    let mut cur_b = state_at_time(state_b, jd_start, ephem, include_extended)?;

    let mut best_idx = 0;
    let mut best_dist = (Vector3::from(cur_a.pos) - Vector3::from(cur_b.pos)).norm();
    let mut prev_a = cur_a.clone();
    let mut prev_b = cur_b.clone();

    for i in 1..=n_samples {
        let t = jd_start + i as f64 * dt;
        let old_a = cur_a.clone();
        let old_b = cur_b.clone();
        cur_a = state_at_time(&cur_a, t, ephem, include_extended)?;
        cur_b = state_at_time(&cur_b, t, ephem, include_extended)?;
        let d = (Vector3::from(cur_a.pos) - Vector3::from(cur_b.pos)).norm();
        if d < best_dist {
            best_dist = d;
            best_idx = i;
            prev_a = old_a;
            prev_b = old_b;
        }
    }

    // Bracket the coarse minimum and refine with golden-section search, on offsets
    // in days from jd_start.
    let lo_idx = best_idx.saturating_sub(1);
    let hi_idx = (best_idx + 1).min(n_samples);
    let lo_off = lo_idx as f64 * dt;
    let hi_off = hi_idx as f64 * dt;
    let tol = 1e-10; // ~0.01 ms

    let ref_a = &prev_a;
    let ref_b = &prev_b;
    let mut inner_err: Option<Error> = None;
    let dist_at = |off: f64| -> f64 {
        if inner_err.is_some() {
            return f64::NAN;
        }
        let t = jd_start + off;
        let (sa, sb) = match (
            state_at_time(ref_a, t, ephem, include_extended),
            state_at_time(ref_b, t, ephem, include_extended),
        ) {
            (Ok(a), Ok(b)) => (a, b),
            (Err(e), _) | (_, Err(e)) => {
                inner_err = Some(e);
                return f64::NAN;
            }
        };
        (Vector3::from(sa.pos) - Vector3::from(sb.pos)).norm_squared()
    };

    let best_off = kete_stats::fitting::golden_section_search(dist_at, lo_off, hi_off, tol)
        .map_err(|_| {
            inner_err.unwrap_or_else(|| {
                Error::ValueError("Golden-section search failed to converge".into())
            })
        })?;

    let final_jd = jd_start + best_off;
    let sa = state_at_time(ref_a, final_jd, ephem, include_extended)?;
    let sb = state_at_time(ref_b, final_jd, ephem, include_extended)?;
    Ok((
        final_jd,
        (Vector3::from(sa.pos) - Vector3::from(sb.pos)).norm(),
    ))
}

/// Get the state of a body at a given time, from the ephemeris if possible,
/// otherwise propagating with N-body.
///
/// If the state's designation maps to a NAIF ID that `ephem` covers at
/// `time`, its ephemeris is used directly. Otherwise the state is propagated
/// forward under N-body gravity.
fn state_at_time<E: Ephemeris>(
    state: &State<Equatorial>,
    time: Time<TDB>,
    ephem: &E,
    include_extended: bool,
) -> KeteResult<State<Equatorial>> {
    let center = state.center_id();
    if let Some(id) = state.desig.clone().naif_id()
        && let Ok(st) = ephem.try_get_state_with_center(id, time, center)
    {
        return Ok(st);
    }
    let ssb = ephem.try_to_ssb(state.clone())?;
    let ssb_result = ssb.propagate_with(&NBody::new(ephem, include_extended), time)?;
    let mut result: State<Equatorial> = ssb_result.into();
    if center != 0 {
        ephem.try_change_center(&mut result, center)?;
    }
    Ok(result)
}
