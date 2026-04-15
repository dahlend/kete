//! SPK repacker - converts arbitrary SPK segments to compact output forms.
//!
//! Two output types are supported:
//!
//! - **Type 2** (Chebyshev position polynomials) — best for slow-moving objects
//!   (asteroids, planets, deep-space missions).  Excellent compression for smooth
//!   orbits.
//!
//! - **Type 13** (Hermite interpolation, unequal time steps) — best for fast
//!   orbiters (LEO, MEO) or any object whose source data is already dense.
//!   Stores position + velocity at each node; Hermite interpolation naturally
//!   follows the orbital dynamics between nodes.
//!
//! The output frame is fixed as Equatorial J2000 (`frame_id = 1`).
//!
//! # Boundary strategy
//!
//! Rather than probing the source at many times to discover where coverage
//! exists, the repacker reads the raw segment boundaries directly from the
//! loaded [`super::SpkCollection`].  Queries are clamped to known-good coverage,
//! eliminating micro-gap failures and all the tolerance/retry/skip machinery
//! that entails.

use crate::interpolation::{chebyshev_fit, hermite_interpolation};
use crate::{jd_to_spice_jd, spice_jd_to_jd};
use kete_core::constants::AU_KM;
use kete_core::errors::Error;
use kete_core::frames::Equatorial;
use kete_core::prelude::KeteResult;
use kete_core::time::{TDB, Time};
use rayon::prelude::*;

use super::SpkArray;
use super::type2::SpkSegmentType2;
use super::type13::SpkSegmentType13;

// ── Type 2 constants ────────────────────────────────────────────────────────

/// Minimum step size for Type 2: 0.1 day (in SPICE seconds).
const S_MIN: f64 = 0.1 * 86400.0;

/// Default maximum step size for Type 2: 30 days (in SPICE seconds).
const S_MAX_DEFAULT: f64 = 30.0 * 86400.0;

/// Number of probe intervals used per binary-search iteration.
const N_PROBES: usize = 20;

/// Maximum retry attempts if the full-pass error exceeds the threshold.
const MAX_RETRIES: usize = 5;

/// Step shrink factor applied on each retry.
const RETRY_SHRINK: f64 = 0.75;

/// Minimum number of intervals per rayon work unit.
const RAYON_MIN_LEN: usize = 4;

/// Maximum recursion depth for subrange splitting.
const MAX_SPLIT_DEPTH: usize = 20;

// ── Type 13 constants ───────────────────────────────────────────────────────

/// Minimum node spacing for Type 13: 60 seconds.
const T13_S_MIN: f64 = 60.0;

/// Default maximum node spacing for Type 13: 1 day (in SPICE seconds).
const T13_S_MAX_DEFAULT: f64 = 86400.0;

/// Number of validation points between each pair of neighboring nodes.
const T13_N_VALIDATE: usize = 5;

// ── Segment-domain helpers ──────────────────────────────────────────────────

/// Buffer (SPICE seconds) to shrink each segment boundary inward, ensuring
/// that queries after JD round-trip stay within the segment.  100 us gives
/// ~2x margin over the ~50 us f64 round-trip error at typical JD values.
const BOUNDARY_BUFFER: f64 = 1e-4;

/// Gap tolerance for merging adjacent segments.  Segments separated by less
/// than this are treated as contiguous.
const GAP_TOLERANCE: f64 = 1.0;

// ═════════════════════════════════════════════════════════════════════════════
//  Type 2 repacker — Chebyshev polynomial position fitting
// ═════════════════════════════════════════════════════════════════════════════

/// Repack all source segments for `object_id` into compact Type 2 Chebyshev
/// SPK arrays.
///
/// See module-level docs for the overall strategy.
///
/// # Errors
/// Returns an error if `degree` is outside `[1, 27]` or if the underlying
/// segment evaluation fails.
pub fn repack_to_type2(
    source: &super::SpkCollection,
    object_id: i32,
    center_id: i32,
    threshold_km: f64,
    degree: usize,
    explicit_ranges: Option<&[(f64, f64)]>,
) -> KeteResult<Vec<SpkArray>> {
    if !(1..=27).contains(&degree) {
        return Err(Error::ValueError(format!(
            "Chebyshev degree must be in [1, 27], got {degree}"
        )));
    }

    let raw = raw_boundaries(source, object_id, explicit_ranges)?;
    let segments = buffer_and_sort(&raw, BOUNDARY_BUFFER);
    let runs = merge_continuous_runs(
        &segments,
        GAP_TOLERANCE,
        source,
        object_id,
        center_id,
        threshold_km,
    );
    if runs.is_empty() {
        return Err(Error::Bounds(format!(
            "No SPK coverage found for NAIF ID {object_id}"
        )));
    }

    let mut arrays = Vec::with_capacity(runs.len());
    for &(t_start, t_end) in &runs {
        if t_end - t_start < S_MIN / 2.0 {
            continue;
        }
        arrays.extend(fit_subrange(
            source,
            object_id,
            center_id,
            t_start,
            t_end,
            degree,
            threshold_km,
            &segments,
            0,
        )?);
    }

    if arrays.is_empty() {
        return Err(Error::ValueError(format!(
            "Repacking NAIF {object_id}: all ranges too short for Type 2."
        )));
    }
    Ok(arrays)
}

/// Try to fit a Type 2 sub-range, recursively splitting on failure.
fn fit_subrange(
    source: &super::SpkCollection,
    object_id: i32,
    center_id: i32,
    t_start: f64,
    t_end: f64,
    degree: usize,
    threshold_km: f64,
    segments: &[(f64, f64)],
    depth: usize,
) -> KeteResult<Vec<SpkArray>> {
    let range = t_end - t_start;
    if range < S_MIN / 2.0 {
        return Ok(Vec::new());
    }

    // Try direct fit.
    if let Ok(step) = find_step_size(
        source,
        object_id,
        center_id,
        t_start,
        t_end,
        threshold_km,
        degree,
        segments,
    ) {
        let jd_start = spice_jd_to_jd(t_start);
        let jd_end = spice_jd_to_jd(t_end);
        if let Ok(array) = fit_range(
            source,
            object_id,
            center_id,
            t_start,
            t_end,
            step,
            degree,
            threshold_km,
            jd_start,
            jd_end,
            segments,
        ) {
            return Ok(vec![array]);
        }
    }

    // At max recursion depth, force-fit at minimum step size (no validation)
    // rather than returning empty and creating a gap.
    if depth > MAX_SPLIT_DEPTH {
        return force_fit(
            source, object_id, center_id, t_start, t_end, degree, segments,
        )
        .map(|a| vec![a]);
    }

    // Split in half and recurse.
    let t_mid = f64::midpoint(t_start, t_end);
    let mut arrays = fit_subrange(
        source,
        object_id,
        center_id,
        t_start,
        t_mid,
        degree,
        threshold_km,
        segments,
        depth + 1,
    )?;
    arrays.extend(fit_subrange(
        source,
        object_id,
        center_id,
        t_mid,
        t_end,
        degree,
        threshold_km,
        segments,
        depth + 1,
    )?);
    Ok(arrays)
}

/// Force-fit a Type 2 sub-range at minimum step size without validation.
/// Used as a last resort when recursive splitting has exhausted its depth
/// budget, ensuring no gaps in the output.
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "n_records bounded; exact in f64."
)]
fn force_fit(
    source: &super::SpkCollection,
    object_id: i32,
    center_id: i32,
    t_start: f64,
    t_end: f64,
    degree: usize,
    segments: &[(f64, f64)],
) -> KeteResult<SpkArray> {
    let range = t_end - t_start;
    let step = S_MIN.min(range);
    let n_records = ((range / step).ceil().max(1.0)) as usize;
    let actual_step = range / n_records as f64;
    let half_step = actual_step / 2.0;

    let results: Vec<KeteResult<(Vec<f64>, f64)>> = (0..n_records)
        .into_par_iter()
        .with_min_len(RAYON_MIN_LEN)
        .map(|i| {
            let t_mid = t_start + half_step + i as f64 * actual_step;
            fit_interval(
                source, object_id, center_id, t_mid, half_step, degree, segments,
            )
        })
        .collect();

    let n_coef = degree + 1;
    let ninrec = 3 * n_coef;
    let mut cdata = Vec::with_capacity(n_records * ninrec);
    for result in results {
        let (coeffs, _) = result?;
        cdata.extend_from_slice(&coeffs);
    }

    let jd_start = spice_jd_to_jd(t_start);
    let jd_end = spice_jd_to_jd(t_end);
    SpkSegmentType2::new_array(
        object_id,
        center_id,
        1,
        &cdata,
        n_records,
        t_start,
        actual_step,
        degree,
        jd_start,
        jd_end,
        &format!("REPACK {object_id}"),
    )
}

/// Binary-search for the largest step size (SPICE seconds) that keeps the
/// position error below `threshold_km`.
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "n_records bounded; exact in f64."
)]
fn find_step_size(
    source: &super::SpkCollection,
    object_id: i32,
    center_id: i32,
    t_start: f64,
    t_end: f64,
    threshold_km: f64,
    degree: usize,
    segments: &[(f64, f64)],
) -> KeteResult<f64> {
    let range = t_end - t_start;
    if range < S_MIN / 2.0 {
        return Err(Error::ValueError(format!(
            "Time range ({:.4} days) is too short to repack.",
            range / 86400.0,
        )));
    }

    let s_max = (range / 4.0).clamp(S_MIN, S_MAX_DEFAULT);

    if probe_max_error(
        source, object_id, center_id, t_start, t_end, s_max, degree, segments,
    )? <= threshold_km
    {
        return Ok(s_max);
    }
    if probe_max_error(
        source, object_id, center_id, t_start, t_end, S_MIN, degree, segments,
    )? > threshold_km
    {
        return Err(Error::ValueError(format!(
            "Cannot meet {threshold_km} km threshold for NAIF {object_id} with \
             degree {degree} even at minimum step ({:.4} days).",
            S_MIN / 86400.0
        )));
    }

    let mut lo = S_MIN;
    let mut hi = s_max;
    for _ in 0..25 {
        if (hi - lo) / lo < 1e-4 {
            break;
        }
        let mid = f64::midpoint(lo, hi);
        if probe_max_error(
            source, object_id, center_id, t_start, t_end, mid, degree, segments,
        )? <= threshold_km
        {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    Ok(lo)
}

/// Evaluate the maximum Chebyshev fit error over [`N_PROBES`] evenly-spaced
/// intervals for a candidate step size.
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "n_records bounded; exact in f64."
)]
fn probe_max_error(
    source: &super::SpkCollection,
    object_id: i32,
    center_id: i32,
    t_start: f64,
    t_end: f64,
    step: f64,
    degree: usize,
    segments: &[(f64, f64)],
) -> KeteResult<f64> {
    let n_records = ((t_end - t_start) / step).ceil().max(1.0) as usize;
    let actual_step = (t_end - t_start) / n_records as f64;
    let half_step = actual_step / 2.0;

    let sqrt_probes = (n_records as f64).sqrt().ceil() as usize;
    let n_probes = N_PROBES.max(sqrt_probes).min(n_records);
    let mut max_err = 0.0_f64;

    for i in 0..n_probes {
        let rec_idx = if n_probes == 1 {
            0
        } else {
            i * (n_records - 1) / (n_probes - 1)
        };
        let t_mid = t_start + half_step + rec_idx as f64 * actual_step;
        let (_, err) = fit_interval(
            source, object_id, center_id, t_mid, half_step, degree, segments,
        )?;
        max_err = max_err.max(err);
    }
    Ok(max_err)
}

/// Fit all intervals in a time range, producing one SPK Type 2 array.
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "n_records bounded; exact in f64."
)]
fn fit_range(
    source: &super::SpkCollection,
    object_id: i32,
    center_id: i32,
    t_start: f64,
    t_end: f64,
    initial_step: f64,
    degree: usize,
    threshold_km: f64,
    jd_start: Time<TDB>,
    jd_end: Time<TDB>,
    segments: &[(f64, f64)],
) -> KeteResult<SpkArray> {
    let mut step = initial_step;

    for _ in 0..MAX_RETRIES {
        let n_records = ((t_end - t_start) / step).ceil().max(1.0) as usize;
        let actual_step = (t_end - t_start) / n_records as f64;
        let half_step = actual_step / 2.0;

        let results: Vec<KeteResult<(Vec<f64>, f64)>> = (0..n_records)
            .into_par_iter()
            .with_min_len(RAYON_MIN_LEN)
            .map(|i| {
                let t_mid = t_start + half_step + i as f64 * actual_step;
                fit_interval(
                    source, object_id, center_id, t_mid, half_step, degree, segments,
                )
            })
            .collect();

        let n_coef = degree + 1;
        let ninrec = 3 * n_coef;
        let mut cdata = Vec::with_capacity(n_records * ninrec);
        let mut max_err = 0.0_f64;

        for result in results {
            let (coeffs, err) = result?;
            cdata.extend_from_slice(&coeffs);
            max_err = max_err.max(err);
        }

        if max_err <= threshold_km {
            return SpkSegmentType2::new_array(
                object_id,
                center_id,
                1, // Equatorial J2000
                &cdata,
                n_records,
                t_start,
                actual_step,
                degree,
                jd_start,
                jd_end,
                &format!("REPACK {object_id}"),
            );
        }

        let new_step = step * RETRY_SHRINK;
        if new_step < S_MIN {
            break;
        }
        step = new_step;
    }

    Err(Error::ValueError(format!(
        "Repacking NAIF {object_id}: could not meet {threshold_km} km threshold \
         after {MAX_RETRIES} attempts (final step {:.3} days, degree {degree}).",
        step / 86400.0
    )))
}

/// Fit one Chebyshev interval: sample at Gauss-Lobatto nodes, fit, validate.
#[allow(
    clippy::cast_precision_loss,
    reason = "Indices bounded by 3*(degree+1) <= 84; exact in f64."
)]
fn fit_interval(
    source: &super::SpkCollection,
    object_id: i32,
    center_id: i32,
    t_mid: f64,
    half_len: f64,
    degree: usize,
    segments: &[(f64, f64)],
) -> KeteResult<(Vec<f64>, f64)> {
    let n = degree + 1;
    let d_f = degree as f64;

    let mut xs = vec![0.0_f64; n];
    let mut ys = vec![0.0_f64; n];
    let mut zs = vec![0.0_f64; n];

    for k in 0..n {
        let tau = (std::f64::consts::PI * k as f64 / d_f).cos();
        let t = t_mid + half_len * tau;
        let state = safe_query(source, object_id, center_id, t, segments)?;
        let pos_au: [f64; 3] = state.pos.into();
        xs[k] = pos_au[0] * AU_KM;
        ys[k] = pos_au[1] * AU_KM;
        zs[k] = pos_au[2] * AU_KM;
    }

    let cdata = chebyshev_fit(&xs, &ys, &zs);

    let m_val = 3 * n;
    let mut max_err = 0.0_f64;

    for i in 0..m_val {
        let tau = -1.0 + 2.0 * (i as f64 + 0.5) / m_val as f64;
        let t = t_mid + half_len * tau;
        let state = safe_query(source, object_id, center_id, t, segments)?;
        let pos_au: [f64; 3] = state.pos.into();
        let px = pos_au[0] * AU_KM;
        let py = pos_au[1] * AU_KM;
        let pz = pos_au[2] * AU_KM;

        let (val, _) = crate::interpolation::chebyshev_evaluate_both(
            tau,
            &cdata[..n],
            &cdata[n..2 * n],
            &cdata[2 * n..],
        )?;

        let err = ((val[0] - px).powi(2) + (val[1] - py).powi(2) + (val[2] - pz).powi(2)).sqrt();
        if err > max_err {
            max_err = err;
        }
    }

    Ok((cdata, max_err))
}

// ═════════════════════════════════════════════════════════════════════════════
//  Type 13 repacker — Hermite interpolation with adaptive node spacing
// ═════════════════════════════════════════════════════════════════════════════

/// Repack source segments for `object_id` into Type 13 Hermite SPK arrays.
///
/// # Errors
/// Returns an error if `degree` is even, outside `[1, 27]`, or if the
/// underlying segment evaluation fails.
pub fn repack_to_type13(
    source: &super::SpkCollection,
    object_id: i32,
    center_id: i32,
    threshold_km: f64,
    degree: usize,
    explicit_ranges: Option<&[(f64, f64)]>,
) -> KeteResult<Vec<SpkArray>> {
    if !(1..=27).contains(&degree) || degree.is_multiple_of(2) {
        return Err(Error::ValueError(format!(
            "Type 13 degree must be odd and in [1, 27], got {degree}"
        )));
    }

    let raw = raw_boundaries(source, object_id, explicit_ranges)?;
    let segments = buffer_and_sort(&raw, BOUNDARY_BUFFER);
    let runs = merge_continuous_runs(
        &segments,
        GAP_TOLERANCE,
        source,
        object_id,
        center_id,
        threshold_km,
    );
    if runs.is_empty() {
        return Err(Error::Bounds(format!(
            "No SPK coverage found for NAIF ID {object_id}"
        )));
    }

    let mut arrays = Vec::with_capacity(runs.len());
    for &(t_start, t_end) in &runs {
        if t_end - t_start < T13_S_MIN / 2.0 {
            continue;
        }
        arrays.extend(t13_fit_subrange(
            source,
            object_id,
            center_id,
            t_start,
            t_end,
            degree,
            threshold_km,
            &segments,
            0,
        )?);
    }

    if arrays.is_empty() {
        return Err(Error::ValueError(format!(
            "Repacking NAIF {object_id}: all ranges too short for Type 13."
        )));
    }
    Ok(arrays)
}

/// Try to fit a Type 13 sub-range, recursively splitting on failure.
fn t13_fit_subrange(
    source: &super::SpkCollection,
    object_id: i32,
    center_id: i32,
    t_start: f64,
    t_end: f64,
    degree: usize,
    threshold_km: f64,
    segments: &[(f64, f64)],
    depth: usize,
) -> KeteResult<Vec<SpkArray>> {
    let range = t_end - t_start;
    if range < T13_S_MIN / 2.0 {
        return Ok(Vec::new());
    }

    // Try direct fit.
    if let Ok(step) = t13_find_step_size(
        source,
        object_id,
        center_id,
        t_start,
        t_end,
        threshold_km,
        degree,
        segments,
    ) && let Ok(array) = t13_fit_range(
        source,
        object_id,
        center_id,
        t_start,
        t_end,
        step,
        degree,
        threshold_km,
        segments,
    ) {
        return Ok(vec![array]);
    }

    // At max recursion depth, force-fit at minimum spacing (no validation)
    // rather than returning empty and creating a gap.
    if depth > MAX_SPLIT_DEPTH {
        return t13_force_fit(
            source, object_id, center_id, t_start, t_end, degree, segments,
        )
        .map(|a| vec![a]);
    }

    // Split in half and recurse.
    let t_mid = f64::midpoint(t_start, t_end);
    let mut arrays = t13_fit_subrange(
        source,
        object_id,
        center_id,
        t_start,
        t_mid,
        degree,
        threshold_km,
        segments,
        depth + 1,
    )?;
    arrays.extend(t13_fit_subrange(
        source,
        object_id,
        center_id,
        t_mid,
        t_end,
        degree,
        threshold_km,
        segments,
        depth + 1,
    )?);
    Ok(arrays)
}

/// Force-fit a Type 13 sub-range at minimum node spacing without validation.
/// Used as a last resort when recursive splitting has exhausted its depth
/// budget, ensuring no gaps in the output.
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "n_nodes bounded by range/T13_S_MIN; exact in f64."
)]
fn t13_force_fit(
    source: &super::SpkCollection,
    object_id: i32,
    center_id: i32,
    t_start: f64,
    t_end: f64,
    degree: usize,
    segments: &[(f64, f64)],
) -> KeteResult<SpkArray> {
    let range = t_end - t_start;
    let min_nodes = degree.div_ceil(2) + 1;
    let step = T13_S_MIN.min(range / min_nodes as f64);
    let n_nodes = ((range / step).ceil() as usize + 1).max(min_nodes);
    let actual_step = range / (n_nodes - 1) as f64;

    let states: Vec<KeteResult<(Time<TDB>, [f64; 3], [f64; 3])>> = (0..n_nodes)
        .into_par_iter()
        .with_min_len(RAYON_MIN_LEN)
        .map(|i| {
            let t = t_start + i as f64 * actual_step;
            let state = safe_query(source, object_id, center_id, t, segments)?;
            let jd = state.epoch;
            let pos_au: [f64; 3] = state.pos.into();
            let vel_au: [f64; 3] = state.vel.into();
            Ok((
                jd,
                [pos_au[0] * AU_KM, pos_au[1] * AU_KM, pos_au[2] * AU_KM],
                [
                    vel_au[0] * AU_KM / 86400.0,
                    vel_au[1] * AU_KM / 86400.0,
                    vel_au[2] * AU_KM / 86400.0,
                ],
            ))
        })
        .collect();

    let mut sampled = Vec::with_capacity(n_nodes);
    for r in states {
        sampled.push(r?);
    }

    SpkSegmentType13::new_array(
        object_id,
        center_id,
        1,
        &sampled,
        degree as u32,
        &format!("REPACK {object_id}"),
    )
}

/// Binary-search for the largest node spacing that keeps interpolation error
/// below `threshold_km`.
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "n_nodes bounded by range/T13_S_MIN; exact in f64."
)]
fn t13_find_step_size(
    source: &super::SpkCollection,
    object_id: i32,
    center_id: i32,
    t_start: f64,
    t_end: f64,
    threshold_km: f64,
    degree: usize,
    segments: &[(f64, f64)],
) -> KeteResult<f64> {
    let range = t_end - t_start;
    if range < T13_S_MIN / 2.0 {
        return Err(Error::ValueError(format!(
            "Time range ({range:.4}s) too short for Type 13.",
        )));
    }

    let s_max = (range / 4.0).clamp(T13_S_MIN, T13_S_MAX_DEFAULT);

    if t13_probe_max_error(
        source, object_id, center_id, t_start, t_end, s_max, degree, segments,
    )? <= threshold_km
    {
        return Ok(s_max);
    }
    if t13_probe_max_error(
        source, object_id, center_id, t_start, t_end, T13_S_MIN, degree, segments,
    )? > threshold_km
    {
        return Err(Error::ValueError(format!(
            "Cannot meet {threshold_km} km threshold for NAIF {object_id} (Type 13) \
             with degree {degree} even at minimum spacing ({T13_S_MIN:.1}s).",
        )));
    }

    let mut lo = T13_S_MIN;
    let mut hi = s_max;
    for _ in 0..25 {
        if (hi - lo) / lo < 1e-4 {
            break;
        }
        let mid = f64::midpoint(lo, hi);
        if t13_probe_max_error(
            source, object_id, center_id, t_start, t_end, mid, degree, segments,
        )? <= threshold_km
        {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    Ok(lo)
}

/// Probe Hermite interpolation error at evenly-spaced gaps for a candidate
/// node spacing.
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "n_nodes and indices bounded; exact in f64."
)]
fn t13_probe_max_error(
    source: &super::SpkCollection,
    object_id: i32,
    center_id: i32,
    t_start: f64,
    t_end: f64,
    step: f64,
    degree: usize,
    segments: &[(f64, f64)],
) -> KeteResult<f64> {
    let range = t_end - t_start;
    let n_nodes = ((range / step).ceil() as usize + 1).max(2);
    let actual_step = range / (n_nodes - 1) as f64;

    let n_gaps = n_nodes - 1;
    let sqrt_probes = (n_gaps as f64).sqrt().ceil() as usize;
    let n_probes = N_PROBES.max(sqrt_probes).min(n_gaps);
    let window_size = degree.div_ceil(2);
    let mut max_err = 0.0_f64;
    let boundaries = segment_boundary_times(segments);

    for probe_i in 0..n_probes {
        let gap_idx = if n_probes == 1 {
            0
        } else {
            probe_i * (n_gaps - 1) / (n_probes - 1)
        };

        let win_start = gap_idx.saturating_sub(window_size / 2);
        let win_end = (win_start + window_size).min(n_nodes);
        let win_start = if win_end == n_nodes {
            n_nodes.saturating_sub(window_size)
        } else {
            win_start
        };

        // Skip probe windows that straddle a source segment boundary —
        // nodes from different segments may have small discontinuities
        // that produce misleading error estimates.
        let t_win_lo = t_start + win_start as f64 * actual_step;
        let t_win_hi = t_start + (win_end - 1) as f64 * actual_step;
        if spans_boundary(t_win_lo, t_win_hi, &boundaries) {
            continue;
        }

        let mut times = Vec::with_capacity(win_end - win_start);
        let mut px = Vec::with_capacity(win_end - win_start);
        let mut py = Vec::with_capacity(win_end - win_start);
        let mut pz = Vec::with_capacity(win_end - win_start);
        let mut vx = Vec::with_capacity(win_end - win_start);
        let mut vy = Vec::with_capacity(win_end - win_start);
        let mut vz = Vec::with_capacity(win_end - win_start);

        for idx in win_start..win_end {
            let t = t_start + idx as f64 * actual_step;
            let state = safe_query(source, object_id, center_id, t, segments)?;
            let pos_au: [f64; 3] = state.pos.into();
            let vel_au: [f64; 3] = state.vel.into();
            times.push(t);
            px.push(pos_au[0] * AU_KM);
            py.push(pos_au[1] * AU_KM);
            pz.push(pos_au[2] * AU_KM);
            vx.push(vel_au[0] * AU_KM / 86400.0);
            vy.push(vel_au[1] * AU_KM / 86400.0);
            vz.push(vel_au[2] * AU_KM / 86400.0);
        }

        let t_mid = t_start + (gap_idx as f64 + 0.5) * actual_step;
        let state = safe_query(source, object_id, center_id, t_mid, segments)?;
        let orig_pos: [f64; 3] = state.pos.into();

        let (ix, _) = hermite_interpolation(&times, &px, &vx, t_mid);
        let (iy, _) = hermite_interpolation(&times, &py, &vy, t_mid);
        let (iz, _) = hermite_interpolation(&times, &pz, &vz, t_mid);

        let err = ((ix - orig_pos[0] * AU_KM).powi(2)
            + (iy - orig_pos[1] * AU_KM).powi(2)
            + (iz - orig_pos[2] * AU_KM).powi(2))
        .sqrt();

        max_err = max_err.max(err);
    }

    Ok(max_err)
}

/// Sample states at evenly-spaced nodes and build a Type 13 array.
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "n_nodes bounded by range/T13_S_MIN; indices exact in f64."
)]
fn t13_fit_range(
    source: &super::SpkCollection,
    object_id: i32,
    center_id: i32,
    t_start: f64,
    t_end: f64,
    initial_step: f64,
    degree: usize,
    threshold_km: f64,
    segments: &[(f64, f64)],
) -> KeteResult<SpkArray> {
    let mut step = initial_step;

    for _ in 0..MAX_RETRIES {
        let n_nodes = ((t_end - t_start) / step).ceil() as usize + 1;
        let n_nodes = n_nodes.max(2);
        let actual_step = (t_end - t_start) / (n_nodes - 1) as f64;

        let states: Vec<KeteResult<(Time<TDB>, [f64; 3], [f64; 3])>> = (0..n_nodes)
            .into_par_iter()
            .with_min_len(RAYON_MIN_LEN)
            .map(|i| {
                let t = t_start + i as f64 * actual_step;
                let state = safe_query(source, object_id, center_id, t, segments)?;
                let jd = state.epoch;
                let pos_au: [f64; 3] = state.pos.into();
                let vel_au: [f64; 3] = state.vel.into();
                Ok((
                    jd,
                    [pos_au[0] * AU_KM, pos_au[1] * AU_KM, pos_au[2] * AU_KM],
                    [
                        vel_au[0] * AU_KM / 86400.0,
                        vel_au[1] * AU_KM / 86400.0,
                        vel_au[2] * AU_KM / 86400.0,
                    ],
                ))
            })
            .collect();

        let mut sampled = Vec::with_capacity(n_nodes);
        for r in states {
            sampled.push(r?);
        }

        let array = SpkSegmentType13::new_array(
            object_id,
            center_id,
            1, // Equatorial J2000
            &sampled,
            degree as u32,
            &format!("REPACK {object_id}"),
        )?;

        let max_err = t13_validate_sampled(
            source,
            object_id,
            center_id,
            degree,
            &sampled,
            threshold_km,
            segments,
        )?;

        if max_err <= threshold_km {
            return Ok(array);
        }

        let new_step = step * RETRY_SHRINK;
        if new_step < T13_S_MIN {
            break;
        }
        step = new_step;
    }

    Err(Error::ValueError(format!(
        "Repacking NAIF {object_id} (Type 13): could not meet {threshold_km} km \
         threshold after {MAX_RETRIES} attempts (final step {step:.1}s, degree {degree}).",
    )))
}

/// Validate sampled nodes by checking Hermite interpolation error at sub-node
/// points against the source.
#[allow(
    clippy::cast_precision_loss,
    reason = "j in [0, T13_N_VALIDATE]; exact in f64."
)]
fn t13_validate_sampled(
    source: &super::SpkCollection,
    object_id: i32,
    center_id: i32,
    degree: usize,
    sampled: &[(Time<TDB>, [f64; 3], [f64; 3])],
    threshold_km: f64,
    segments: &[(f64, f64)],
) -> KeteResult<f64> {
    let n_nodes = sampled.len();
    let window_size = degree.div_ceil(2);
    let mut max_err = 0.0_f64;
    let boundaries = segment_boundary_times(segments);

    let spice_times: Vec<f64> = sampled
        .iter()
        .map(|(jd, _, _)| jd_to_spice_jd(*jd))
        .collect();

    for i in 0..n_nodes - 1 {
        let t0 = spice_times[i];
        let t1 = spice_times[i + 1];

        // Skip intervals that straddle a source segment boundary.
        // The source may have a small discontinuity there that the
        // Hermite interpolation smoothly bridges; comparing against
        // the discontinuous source would be an unfair test.
        if spans_boundary(t0, t1, &boundaries) {
            continue;
        }

        let win_start = i.saturating_sub(window_size / 2);
        let win_end = (win_start + window_size).min(n_nodes);
        let win_start = if win_end == n_nodes {
            n_nodes.saturating_sub(window_size)
        } else {
            win_start
        };

        let times = &spice_times[win_start..win_end];
        let px: Vec<f64> = (win_start..win_end).map(|k| sampled[k].1[0]).collect();
        let py: Vec<f64> = (win_start..win_end).map(|k| sampled[k].1[1]).collect();
        let pz: Vec<f64> = (win_start..win_end).map(|k| sampled[k].1[2]).collect();
        let vx: Vec<f64> = (win_start..win_end).map(|k| sampled[k].2[0]).collect();
        let vy: Vec<f64> = (win_start..win_end).map(|k| sampled[k].2[1]).collect();
        let vz: Vec<f64> = (win_start..win_end).map(|k| sampled[k].2[2]).collect();

        for j in 1..=T13_N_VALIDATE {
            let frac = j as f64 / (T13_N_VALIDATE + 1) as f64;
            let t = t0 + frac * (t1 - t0);

            let orig = safe_query(source, object_id, center_id, t, segments)?;
            let orig_pos: [f64; 3] = orig.pos.into();

            let (ix, _) = hermite_interpolation(times, &px, &vx, t);
            let (iy, _) = hermite_interpolation(times, &py, &vy, t);
            let (iz, _) = hermite_interpolation(times, &pz, &vz, t);

            let err = ((ix - orig_pos[0] * AU_KM).powi(2)
                + (iy - orig_pos[1] * AU_KM).powi(2)
                + (iz - orig_pos[2] * AU_KM).powi(2))
            .sqrt();

            max_err = max_err.max(err);
            if max_err > threshold_km * 2.0 {
                return Ok(max_err);
            }
        }
    }

    Ok(max_err)
}

// ── Segment-domain helpers ──────────────────────────────────────────────────

/// Sort raw `(start, end)` ranges and shrink each inward by `buffer` to
/// protect against JD round-trip imprecision.
fn buffer_and_sort(raw: &[(f64, f64)], buffer: f64) -> Vec<(f64, f64)> {
    let mut sorted: Vec<(f64, f64)> = raw.to_vec();
    sorted.sort_by(|a, b| a.0.total_cmp(&b.0));
    sorted
        .into_iter()
        .filter_map(|(s, e)| {
            let sb = s + buffer;
            let eb = e - buffer;
            if sb < eb { Some((sb, eb)) } else { None }
        })
        .collect()
}

/// Merge sorted segments into contiguous runs, but *only* when the data is
/// actually continuous across the boundary.  Two segments are merged when:
///   1. They overlap or abut (gap <= 0), OR
///   2. The gap is smaller than `gap_tolerance` AND velocity-extrapolated
///      position from the first segment matches the second within
///      `continuity_km`.
///
/// Segments that are close in time but have a real data discontinuity
/// (different navigation solutions) stay separate, preventing the fitter
/// from having to bridge a jump it cannot represent.
fn merge_continuous_runs(
    sorted_segments: &[(f64, f64)],
    gap_tolerance: f64,
    source: &super::SpkCollection,
    object_id: i32,
    center_id: i32,
    continuity_km: f64,
) -> Vec<(f64, f64)> {
    if sorted_segments.is_empty() {
        return Vec::new();
    }
    let mut merged = Vec::with_capacity(sorted_segments.len());
    let mut cur = sorted_segments[0];
    for &(s, e) in &sorted_segments[1..] {
        let gap = s - cur.1;
        if gap <= 0.0 {
            // Overlapping or exactly abutting — always merge.
            cur.1 = cur.1.max(e);
        } else if gap <= gap_tolerance
            && is_boundary_continuous(
                source,
                object_id,
                center_id,
                cur.1,
                s,
                sorted_segments,
                continuity_km,
            )
            .unwrap_or(false)
        {
            cur.1 = cur.1.max(e);
        } else {
            merged.push(cur);
            cur = (s, e);
        }
    }
    merged.push(cur);
    merged
}

/// Check whether source data is continuous across a segment boundary by
/// extrapolating position from the end of one segment to the start of the
/// next using the endpoint velocity.  Returns `true` if the mismatch is
/// below `threshold_km`.
fn is_boundary_continuous(
    source: &super::SpkCollection,
    object_id: i32,
    center_id: i32,
    t_before: f64,
    t_after: f64,
    segments: &[(f64, f64)],
    threshold_km: f64,
) -> KeteResult<bool> {
    let state_a = safe_query(source, object_id, center_id, t_before, segments)?;
    let state_b = safe_query(source, object_id, center_id, t_after, segments)?;
    let dt_days = (t_after - t_before) / 86400.0;
    let pos_a: [f64; 3] = state_a.pos.into();
    let vel_a: [f64; 3] = state_a.vel.into();
    let pos_b: [f64; 3] = state_b.pos.into();
    let err_au_sq = (0..3)
        .map(|i| (pos_a[i] + vel_a[i] * dt_days - pos_b[i]).powi(2))
        .sum::<f64>();
    Ok(err_au_sq.sqrt() * AU_KM <= threshold_km)
}

/// Clamp `t` to the nearest time covered by `segments` (must be sorted by
/// start time and pre-buffered).
///
/// If `t` is already inside a segment, returns it unchanged.  Otherwise
/// returns the nearest segment edge.
fn clamp_to_coverage(t: f64, segments: &[(f64, f64)]) -> f64 {
    // Binary search: find the last segment whose start <= t.
    let idx = segments.partition_point(|&(s, _)| s <= t);
    if idx > 0 && t <= segments[idx - 1].1 {
        return t; // inside segment[idx-1]
    }
    // In a gap (or outside all segments).  Pick closest edge.
    let mut best = t;
    let mut best_dist = f64::INFINITY;
    if idx > 0 {
        let d = (t - segments[idx - 1].1).abs();
        if d < best_dist {
            best = segments[idx - 1].1;
            best_dist = d;
        }
    }
    if idx < segments.len() {
        let d = (segments[idx].0 - t).abs();
        if d < best_dist {
            best = segments[idx].0;
        }
    }
    best
}

/// Collect all segment start/end times as a sorted, deduped boundary list.
fn segment_boundary_times(segments: &[(f64, f64)]) -> Vec<f64> {
    let mut b = Vec::with_capacity(segments.len() * 2);
    for &(s, e) in segments {
        b.push(s);
        b.push(e);
    }
    b.sort_by(f64::total_cmp);
    b.dedup();
    b
}

/// Does the open interval `(t0, t1)` contain any value from `boundaries`
/// (which must be sorted)?
fn spans_boundary(t0: f64, t1: f64, boundaries: &[f64]) -> bool {
    let idx = boundaries.partition_point(|&b| b <= t0);
    idx < boundaries.len() && boundaries[idx] < t1
}

/// Query the source at SPICE-second time `t`, clamping to known segment
/// coverage so that micro-gaps between abutting source segments never cause
/// failures.
fn safe_query(
    source: &super::SpkCollection,
    object_id: i32,
    center_id: i32,
    t: f64,
    segments: &[(f64, f64)],
) -> KeteResult<kete_core::state::State<Equatorial>> {
    let tc = clamp_to_coverage(t, segments);
    let jd = spice_jd_to_jd(tc);
    source.try_get_state_with_center::<Equatorial>(object_id, jd, center_id)
}

/// Obtain raw segment boundaries: from `explicit_ranges` if provided,
/// otherwise from the source's loaded segments.
fn raw_boundaries(
    source: &super::SpkCollection,
    object_id: i32,
    explicit_ranges: Option<&[(f64, f64)]>,
) -> KeteResult<Vec<(f64, f64)>> {
    let raw = match explicit_ranges {
        Some(r) => r.to_vec(),
        None => source.segment_boundaries(object_id),
    };
    if raw.is_empty() {
        return Err(Error::Bounds(format!(
            "No SPK coverage found for NAIF ID {object_id}"
        )));
    }
    Ok(raw)
}

// ═════════════════════════════════════════════════════════════════════════════
//  Tests
// ═════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    /// Integration test: repack `20000042.bsp` (Type 21) to Type 2, verify
    /// the repacked file is smaller and positions match within 0.5 km at 100
    /// evenly-spaced epochs.
    #[test]
    fn repack_20000042_roundtrip() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap();
        let bsp_path = root.join("docs/data/20000042.bsp");
        let original_size = std::fs::metadata(&bsp_path).unwrap().len() as usize;

        // Load into a standalone SpkCollection.
        let mut spk = super::super::SpkCollection::default();
        spk.load_file(bsp_path.to_str().unwrap()).unwrap();

        // Discover the object ID and its center from the loaded segments.
        let object_id = *spk.segments.keys().next().expect("No segments found");
        let info = spk.available_info(object_id);
        assert!(!info.is_empty(), "No coverage for object {object_id}");
        let center_id = info[0].2;

        // Repack with 0.5 km threshold, degree 15.
        let threshold_km = 0.5;
        let arrays = repack_to_type2(&spk, object_id, center_id, threshold_km, 15, None).unwrap();
        assert!(!arrays.is_empty(), "Repack produced no arrays");

        // Write repacked arrays to an in-memory buffer.
        let mut daf = crate::daf::DafFile::new_spk("repack test", "");
        for array in arrays {
            daf.arrays.push(array.daf);
        }
        let mut buf = Cursor::new(Vec::new());
        daf.write_to(&mut buf).unwrap();
        let written = buf.into_inner();

        assert!(
            written.len() < original_size,
            "Repacked ({} bytes) should be smaller than original ({} bytes)",
            written.len(),
            original_size,
        );

        // Reload repacked segments.
        let repacked_daf = crate::daf::DafFile::from_buffer(Cursor::new(&written)).unwrap();
        let mut repacked_spk = super::super::SpkCollection::default();
        for daf_array in repacked_daf.arrays {
            let seg: SpkArray = daf_array.try_into().unwrap();
            repacked_spk
                .segments
                .entry(seg.object_id)
                .or_default()
                .push(seg.try_into().unwrap());
        }

        // Compare positions at 100 evenly-spaced epochs across the repacked coverage.
        let repacked_info = repacked_spk.available_info(object_id);
        let (jd_start, jd_end, ..) = repacked_info[0];
        let mut max_err_km = 0.0_f64;
        for i in 0..100 {
            #[allow(clippy::cast_precision_loss, reason = "i in [0,99]; exact in f64.")]
            let frac = f64::from(i) / 99.0;
            let jd = Time::<TDB>::new(jd_start.jd + frac * (jd_end.jd - jd_start.jd));

            let orig = spk
                .try_get_state_with_center::<Equatorial>(object_id, jd, center_id)
                .unwrap();
            let repacked = repacked_spk
                .try_get_state_with_center::<Equatorial>(object_id, jd, center_id)
                .unwrap();

            let op: [f64; 3] = orig.pos.into();
            let rp: [f64; 3] = repacked.pos.into();
            let err_km =
                ((op[0] - rp[0]).powi(2) + (op[1] - rp[1]).powi(2) + (op[2] - rp[2]).powi(2))
                    .sqrt()
                    * AU_KM;
            max_err_km = max_err_km.max(err_km);
            assert!(
                err_km < threshold_km,
                "Position error {err_km:.6} km exceeds threshold at JD {}",
                jd.jd
            );
        }
        eprintln!(
            "repack_20000042: repacked {} -> {} bytes ({:.1}% reduction), max error {max_err_km:.4} km",
            original_size,
            written.len(),
            (1.0 - written.len() as f64 / original_size as f64) * 100.0
        );
    }

    /// Integration test: repack `20000042.bsp` to Type 13, verify positions
    /// match within 0.5 km at 100 evenly-spaced epochs.
    #[test]
    fn repack_20000042_type13_roundtrip() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap();
        let bsp_path = root.join("docs/data/20000042.bsp");
        let original_size = std::fs::metadata(&bsp_path).unwrap().len() as usize;

        let mut spk = super::super::SpkCollection::default();
        spk.load_file(bsp_path.to_str().unwrap()).unwrap();

        let object_id = *spk.segments.keys().next().expect("No segments found");
        let info = spk.available_info(object_id);
        assert!(!info.is_empty(), "No coverage for object {object_id}");
        let center_id = info[0].2;

        let threshold_km = 0.5;
        let arrays = repack_to_type13(&spk, object_id, center_id, threshold_km, 7, None).unwrap();
        assert!(!arrays.is_empty(), "Repack produced no arrays");

        // Write repacked arrays to an in-memory buffer.
        let mut daf = crate::daf::DafFile::new_spk("repack type13 test", "");
        for array in arrays {
            daf.arrays.push(array.daf);
        }
        let mut buf = Cursor::new(Vec::new());
        daf.write_to(&mut buf).unwrap();
        let written = buf.into_inner();

        // Reload and verify positions.
        let repacked_daf = crate::daf::DafFile::from_buffer(Cursor::new(&written)).unwrap();
        let mut repacked_spk = super::super::SpkCollection::default();
        for daf_array in repacked_daf.arrays {
            let seg: SpkArray = daf_array.try_into().unwrap();
            repacked_spk
                .segments
                .entry(seg.object_id)
                .or_default()
                .push(seg.try_into().unwrap());
        }

        let repacked_info = repacked_spk.available_info(object_id);
        let (jd_start, jd_end, ..) = repacked_info[0];
        let mut max_err_km = 0.0_f64;
        for i in 0..100 {
            #[allow(clippy::cast_precision_loss, reason = "i in [0,99]; exact in f64.")]
            let frac = f64::from(i) / 99.0;
            let jd = Time::<TDB>::new(jd_start.jd + frac * (jd_end.jd - jd_start.jd));

            let orig = spk
                .try_get_state_with_center::<Equatorial>(object_id, jd, center_id)
                .unwrap();
            let repacked = repacked_spk
                .try_get_state_with_center::<Equatorial>(object_id, jd, center_id)
                .unwrap();

            let op: [f64; 3] = orig.pos.into();
            let rp: [f64; 3] = repacked.pos.into();
            let err_km =
                ((op[0] - rp[0]).powi(2) + (op[1] - rp[1]).powi(2) + (op[2] - rp[2]).powi(2))
                    .sqrt()
                    * AU_KM;
            max_err_km = max_err_km.max(err_km);
            assert!(
                err_km < threshold_km,
                "Position error {err_km:.6} km exceeds threshold at JD {}",
                jd.jd
            );
        }
        eprintln!(
            "repack_type13: repacked {} -> {} bytes ({:.1}% reduction), max error {max_err_km:.4} km",
            original_size,
            written.len(),
            (1.0 - written.len() as f64 / original_size as f64) * 100.0
        );
    }
}
