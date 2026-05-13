//! Trajectory analysis tools requiring SPICE ephemeris data.
//!
//! Complements `kete_core::analysis` (B-plane, orbital elements) with
//! functions that need live SPK lookups or N-body propagation.

use kete_core::elements::CometElements;
use kete_core::errors::Error;
use kete_core::frames::{Ecliptic, Equatorial};
use kete_core::prelude::KeteResult;
use kete_core::state::{State, StateLike};
use kete_core::time::{TDB, Time};
use nalgebra::Vector3;

use super::spk_n_body::SpkNBody;
use crate::spk::{LOADED_SPK, SpkCollection};

/// Find the epoch and distance of closest approach between two objects.
///
/// Both objects are propagated using full N-body mechanics over the search
/// window. If either state's designation corresponds to a body available in
/// the loaded SPK kernels, the SPK ephemeris is used directly instead of
/// N-body propagation.
///
/// A coarse grid scan followed by golden-section refinement locates the
/// minimum separation.
///
/// # Errors
/// Returns an error if the states have different center IDs or the time
/// window is non-positive.
pub fn closest_approach(
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

    let span = jd_end.jd - jd_start.jd;
    if span <= 0.0 {
        return Err(Error::ValueError("jd_end must be after jd_start".into()));
    }

    let spk = LOADED_SPK.try_read().map_err(Error::from)?;

    // Adaptive sample count: at least 20 samples per orbital period of the
    // shorter-period object, minimum 200 total.
    let elem_a = CometElements::from_state(&state_a.clone().into_frame::<Ecliptic>());
    let elem_b = CometElements::from_state(&state_b.clone().into_frame::<Ecliptic>());
    let min_period = elem_a.orbital_period().min(elem_b.orbital_period());
    #[allow(clippy::cast_sign_loss, reason = "always positive by construction")]
    let n_samples = if min_period.is_finite() && min_period > 0.0 {
        ((span / min_period) * 20.0).ceil().max(200.0) as usize
    } else {
        200
    };
    let dt = span / n_samples as f64;

    let mut cur_a = state_at_time(state_a, jd_start, &spk, include_extended)?;
    let mut cur_b = state_at_time(state_b, jd_start, &spk, include_extended)?;

    let mut best_idx = 0;
    let mut best_dist = (Vector3::from(cur_a.pos) - Vector3::from(cur_b.pos)).norm();
    let mut prev_a = cur_a.clone();
    let mut prev_b = cur_b.clone();

    for i in 1..=n_samples {
        let t: Time<TDB> = (jd_start.jd + i as f64 * dt).into();
        let old_a = cur_a.clone();
        let old_b = cur_b.clone();
        cur_a = state_at_time(&cur_a, t, &spk, include_extended)?;
        cur_b = state_at_time(&cur_b, t, &spk, include_extended)?;
        let d = (Vector3::from(cur_a.pos) - Vector3::from(cur_b.pos)).norm();
        if d < best_dist {
            best_dist = d;
            best_idx = i;
            prev_a = old_a;
            prev_b = old_b;
        }
    }

    // Bracket the coarse minimum and refine with golden-section search.
    // Use offsets from jd_start to avoid precision loss on large JD values.
    let lo_idx = best_idx.saturating_sub(1);
    let hi_idx = (best_idx + 1).min(n_samples);
    let base = jd_start.jd;
    let lo_off = jd_start.jd + lo_idx as f64 * dt - base;
    let hi_off = jd_start.jd + hi_idx as f64 * dt - base;
    let tol = 1e-10; // ~0.01 ms

    let ref_a = &prev_a;
    let ref_b = &prev_b;
    let mut inner_err: Option<Error> = None;
    let dist_at = |off: f64| -> f64 {
        if inner_err.is_some() {
            return f64::NAN;
        }
        let t: Time<TDB> = (base + off).into();
        let (sa, sb) = match (
            state_at_time(ref_a, t, &spk, include_extended),
            state_at_time(ref_b, t, &spk, include_extended),
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

    let final_jd: Time<TDB> = (base + best_off).into();
    let sa = state_at_time(ref_a, final_jd, &spk, include_extended)?;
    let sb = state_at_time(ref_b, final_jd, &spk, include_extended)?;
    Ok((
        final_jd,
        (Vector3::from(sa.pos) - Vector3::from(sb.pos)).norm(),
    ))
}

/// Get the state of a body at a given time, using the SPK if possible,
/// otherwise propagating with N-body.
///
/// If the state's designation maps to a NAIF ID that the SPK can serve at
/// `time`, the SPK ephemeris is used directly. Otherwise the state is
/// propagated forward under N-body gravity.
fn state_at_time(
    state: &State<Equatorial>,
    time: Time<TDB>,
    spk: &SpkCollection,
    include_extended: bool,
) -> KeteResult<State<Equatorial>> {
    let center = state.center_id();
    if let Some(id) = state.desig.clone().naif_id()
        && let Ok(st) = spk.try_get_state_with_center(id, time, center)
    {
        return Ok(st);
    }
    let ssb = spk.try_to_ssb(state.clone())?;
    let ssb_result = ssb.propagate_with(&SpkNBody::new(include_extended), time)?;
    let mut result: State<Equatorial> = ssb_result.into();
    if center != 0 {
        spk.try_change_center(&mut result, center)?;
    }
    Ok(result)
}
