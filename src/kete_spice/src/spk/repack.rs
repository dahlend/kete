//! SPK repacker. It converts loaded SPK segments into compact output segments.
//!
//! The repacker writes two output types:
//!
//! - **Type 2** (Chebyshev position polynomials) -- for slow-moving objects
//!   with smooth orbits, such as asteroids, planets, and deep-space missions.
//!
//! - **Type 13** (Hermite interpolation, unequal time steps) -- for fast
//!   orbiters (LEO, MEO), or for any object with dense source data. Type 13
//!   stores the position and the velocity at each node.
//!
//! The repacker compares every output record with the source at sample points.
//! A record fails if its position error exceeds `threshold_km`. On success,
//! the output covers the source coverage, except source segments shorter than
//! `2 * BOUNDARY_BUFFER`. The output bridges a gap of up to `GAP_TOLERANCE`
//! between segments where the data is continuous. Where the source jumps
//! between segments, the output has a cut at the jump and does not bridge it.
//!
//! The output frame is Equatorial J2000 (`frame_id = 1`).

use crate::interpolation::{chebyshev_fit, hermite_interpolation};
use crate::spice_jd_to_jd;
use kete_core::constants::AU_KM;
use kete_core::errors::Error;
use kete_core::frames::Equatorial;
use kete_core::prelude::KeteResult;
use rayon::prelude::*;

use super::SpkArray;
use super::type2::SpkSegmentType2;
use super::type13::SpkSegmentType13;

// Type 2 constants

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

// Type 13 constants

/// Minimum node spacing for Type 13: 60 seconds.
const T13_S_MIN: f64 = 60.0;

/// Default maximum node spacing for Type 13: 1 day (in SPICE seconds).
const T13_S_MAX_DEFAULT: f64 = 86400.0;

/// Number of validation points between each pair of neighboring nodes.
const T13_N_VALIDATE: usize = 5;

// -- Segment-domain helpers --------------------------------------------------

/// Buffer (SPICE seconds) to shrink each segment boundary inward, ensuring
/// that queries after JD round-trip stay within the segment.  100 us gives
/// ~2x margin over the ~50 us f64 round-trip error at typical JD values.
const BOUNDARY_BUFFER: f64 = 1e-4;

/// Gap tolerance for merging adjacent segments.  Segments separated by less
/// than this are treated as contiguous.
const GAP_TOLERANCE: f64 = 1.0;

/// Offset (SPICE seconds) on each side of a source segment boundary at which
/// the repacker compares the source to find a jump in the data.
const JUMP_PROBE: f64 = 1.0;

/// Repack all source segments for `object_id` into Type 2 Chebyshev SPK
/// arrays.
///
/// `center_id` is the center of the output states. `threshold_km` is the
/// maximum position error of the output, in km. `degree` is the degree of the
/// Chebyshev polynomials. `explicit_ranges` holds `(start, end)` ranges in TDB
/// seconds from J2000. If given, these ranges replace the loaded segment
/// boundaries of `object_id` as the coverage to repack. The module docs
/// describe the coverage of the output and the cuts at jumps.
///
/// # Errors
/// - `Error::ValueError` if `degree` is outside `[1, 27]`.
/// - `Error::Bounds` if `object_id` has no coverage, or no range longer than
///   `2 * BOUNDARY_BUFFER`.
/// - `Error::ValueError` if a part of the coverage cannot be fit within
///   `threshold_km`. A source query that fails inside a part also gives this
///   error.
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

    let (segments, runs) =
        continuous_runs(source, object_id, center_id, threshold_km, explicit_ranges)?;
    fit_runs(&runs, &segments, |t_start, t_end, run_segments| {
        fit_subrange(
            source,
            object_id,
            center_id,
            t_start,
            t_end,
            degree,
            threshold_km,
            run_segments,
            0,
        )
    })
}

/// Fit a Type 2 sub-range, and split it in half recursively on failure.
///
/// On success, arrays that meet `threshold_km` cover every part of the
/// sub-range. `segments` are the query segments of the run. `depth` is the
/// recursion depth.
///
/// # Errors
/// `Error::ValueError` if a part shorter than `S_MIN`, or a part at
/// `MAX_SPLIT_DEPTH`, cannot be fit. The error names that part.
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

    // Try a direct fit. A range too short for the step search is one record.
    let step = if range < S_MIN / 2.0 {
        Ok(range)
    } else {
        find_step_size(
            source,
            object_id,
            center_id,
            t_start,
            t_end,
            threshold_km,
            degree,
            segments,
        )
    };
    if let Ok(step) = step
        && let Ok(array) = fit_range(
            source,
            object_id,
            center_id,
            t_start,
            t_end,
            step,
            degree,
            threshold_km,
            segments,
        )
    {
        return Ok(vec![array]);
    }

    // Stop at the depth limit, or when the halves would be shorter than the
    // minimum step. Such halves cannot fit better than this range.
    if range < S_MIN || depth >= MAX_SPLIT_DEPTH {
        return Err(unfit_error(
            object_id,
            2,
            threshold_km,
            degree,
            t_start,
            t_end,
        ));
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

/// Return the error for a part of a run that cannot be fit to the threshold.
///
/// `output_type` is the SPK type of the output. The message gives the part as
/// a range of JD.
fn unfit_error(
    object_id: i32,
    output_type: i32,
    threshold_km: f64,
    degree: usize,
    t_start: f64,
    t_end: f64,
) -> Error {
    Error::ValueError(format!(
        "Repacking NAIF {object_id} (Type {output_type}): could not meet the \
         {threshold_km} km threshold with degree {degree} between JD {:.6} and {:.6}.",
        spice_jd_to_jd(t_start).jd,
        spice_jd_to_jd(t_end).jd,
    ))
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
                t_start,
                t_end,
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

/// Repack all source segments for `object_id` into Type 13 Hermite SPK arrays.
///
/// `center_id` is the center of the output states. `threshold_km` is the
/// maximum position error of the output, in km. `degree` is the degree of the
/// Hermite polynomials. `explicit_ranges` holds `(start, end)` ranges in TDB
/// seconds from J2000. If given, these ranges replace the loaded segment
/// boundaries of `object_id` as the coverage to repack. The module docs
/// describe the coverage of the output and the cuts at jumps.
///
/// # Errors
/// - `Error::ValueError` if `degree` is even or outside `[1, 27]`.
/// - `Error::Bounds` if `object_id` has no coverage, or no range longer than
///   `2 * BOUNDARY_BUFFER`.
/// - `Error::ValueError` if a part of the coverage cannot be fit within
///   `threshold_km`. A source query that fails inside a part also gives this
///   error.
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

    let (segments, runs) =
        continuous_runs(source, object_id, center_id, threshold_km, explicit_ranges)?;
    fit_runs(&runs, &segments, |t_start, t_end, run_segments| {
        t13_fit_subrange(
            source,
            object_id,
            center_id,
            t_start,
            t_end,
            degree,
            threshold_km,
            run_segments,
            0,
        )
    })
}

/// Fit a Type 13 sub-range, and split it in half recursively on failure.
///
/// On success, arrays that meet `threshold_km` cover every part of the
/// sub-range. `segments` are the query segments of the run. `depth` is the
/// recursion depth.
///
/// # Errors
/// `Error::ValueError` if a part shorter than `T13_S_MIN`, or a part at
/// `MAX_SPLIT_DEPTH`, cannot be fit. The error names that part.
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

    // Try a direct fit. A range too short for the step search uses the fewest
    // nodes.
    let step = if range < T13_S_MIN / 2.0 {
        Ok(range)
    } else {
        t13_find_step_size(
            source,
            object_id,
            center_id,
            t_start,
            t_end,
            threshold_km,
            degree,
            segments,
        )
    };
    if let Ok(step) = step
        && let Ok(array) = t13_fit_range(
            source,
            object_id,
            center_id,
            t_start,
            t_end,
            step,
            degree,
            threshold_km,
            segments,
        )
    {
        return Ok(vec![array]);
    }

    // Stop at the depth limit, or when the halves would be shorter than the
    // minimum spacing. Such halves cannot fit better than this range.
    if range < T13_S_MIN || depth >= MAX_SPLIT_DEPTH {
        return Err(unfit_error(
            object_id,
            13,
            threshold_km,
            degree,
            t_start,
            t_end,
        ));
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
        let n_nodes = n_nodes.max(degree.div_ceil(2)).max(2);
        let actual_step = (t_end - t_start) / (n_nodes - 1) as f64;

        let states: Vec<KeteResult<(f64, [f64; 3], [f64; 3])>> = (0..n_nodes)
            .into_par_iter()
            .with_min_len(RAYON_MIN_LEN)
            .map(|i| {
                let t = t_start + i as f64 * actual_step;
                let state = safe_query(source, object_id, center_id, t, segments)?;
                let pos_au: [f64; 3] = state.pos.into();
                let vel_au: [f64; 3] = state.vel.into();
                Ok((
                    t,
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

        let segment = SpkSegmentType13::try_from(array)?;
        let max_err = t13_validate_sampled(
            source,
            object_id,
            center_id,
            &segment,
            &sampled,
            threshold_km,
            segments,
        )?;

        if max_err <= threshold_km {
            return Ok(segment.array);
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

/// Return the maximum position error (km) of a Type 13 segment against the
/// source.
///
/// The function evaluates `segment` with the same code that reads the file. It
/// evaluates at `T13_N_VALIDATE` points between each pair of nodes in
/// `sampled`. It returns early when the error exceeds `2 * threshold_km`.
///
/// # Errors
/// Returns the error of a source query that fails.
#[allow(
    clippy::cast_precision_loss,
    reason = "j in [0, T13_N_VALIDATE]; exact in f64."
)]
fn t13_validate_sampled(
    source: &super::SpkCollection,
    object_id: i32,
    center_id: i32,
    segment: &SpkSegmentType13,
    sampled: &[(f64, [f64; 3], [f64; 3])],
    threshold_km: f64,
    segments: &[(f64, f64)],
) -> KeteResult<f64> {
    let mut max_err = 0.0_f64;

    for pair in sampled.windows(2) {
        let (t0, t1) = (pair[0].0, pair[1].0);

        for j in 1..=T13_N_VALIDATE {
            let frac = j as f64 / (T13_N_VALIDATE + 1) as f64;
            let t = t0 + frac * (t1 - t0);

            let orig = safe_query(source, object_id, center_id, t, segments)?;
            let orig_pos: [f64; 3] = orig.pos.into();
            let (pos, _) = segment.try_get_pos_vel(t);

            let err = (0..3)
                .map(|k| ((pos[k] - orig_pos[k]) * AU_KM).powi(2))
                .sum::<f64>()
                .sqrt();

            max_err = max_err.max(err);
            if max_err > threshold_km * 2.0 {
                return Ok(max_err);
            }
        }
    }

    Ok(max_err)
}

// -- Segment-domain helpers --------------------------------------------------

/// Shrink each raw `(start, end)` range inward by `buffer`, then merge the
/// overlapping ranges.
///
/// The buffer protects queries from JD round-trip imprecision at the range
/// edges. The result is sorted and disjoint, which [`clamp_to_coverage`]
/// requires: a time inside a range nested in another is still covered by the
/// outer range.
fn buffer_and_sort(raw: &[(f64, f64)], buffer: f64) -> Vec<(f64, f64)> {
    let mut sorted: Vec<(f64, f64)> = raw
        .iter()
        .map(|&(s, e)| (s + buffer, e - buffer))
        .filter(|(s, e)| s < e)
        .collect();
    sorted.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut merged: Vec<(f64, f64)> = Vec::with_capacity(sorted.len());
    for (s, e) in sorted {
        match merged.last_mut() {
            Some(last) if s <= last.1 => last.1 = last.1.max(e),
            _ => merged.push((s, e)),
        }
    }
    merged
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
    query_segments: &[(f64, f64)],
) -> Vec<(f64, f64)> {
    if sorted_segments.is_empty() {
        return Vec::new();
    }
    let mut merged = Vec::with_capacity(sorted_segments.len());
    let mut cur = sorted_segments[0];
    for &(s, e) in &sorted_segments[1..] {
        let gap = s - cur.1;
        if gap <= 0.0 {
            // Overlapping or exactly abutting -- always merge.
            cur.1 = cur.1.max(e);
        } else if gap <= gap_tolerance
            && is_boundary_continuous(
                source,
                object_id,
                center_id,
                cur.1,
                s,
                query_segments,
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

/// Check whether the source data is continuous between two nearby times.
///
/// The function moves the position at `t_before` to `t_after` with the mean of
/// the two velocities. It returns `true` if the mismatch with the position at
/// `t_after` is at most `threshold_km`.
///
/// # Errors
/// Returns the error of a source query that fails.
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
    let vel_b: [f64; 3] = state_b.vel.into();
    let err_au_sq = (0..3)
        .map(|i| (pos_a[i] + f64::midpoint(vel_a[i], vel_b[i]) * dt_days - pos_b[i]).powi(2))
        .sum::<f64>();
    Ok(err_au_sq.sqrt() * AU_KM <= threshold_km)
}

/// Clamp `t` to the nearest time covered by `segments`, which must be sorted
/// and disjoint, as [`buffer_and_sort`] returns them.
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

/// Query the source at SPICE-second time `t`.
///
/// The function first clamps the time into `segments`. These segments are
/// shrunk inward by `BOUNDARY_BUFFER`, so that the query, made as a JD, stays
/// inside the source coverage. If the clamp moves the time, the function moves
/// the clamped state back to `t` with linear motion. This covers the buffer at
/// a segment edge and small gaps between abutting source segments.
///
/// # Errors
/// Returns the error of the SPK query, such as `Error::Bounds` if no loaded
/// segment covers the clamped time.
fn safe_query(
    source: &super::SpkCollection,
    object_id: i32,
    center_id: i32,
    t: f64,
    segments: &[(f64, f64)],
) -> KeteResult<kete_core::state::State<Equatorial>> {
    let tc = clamp_to_coverage(t, segments);
    let mut state =
        source.try_get_state_with_center::<Equatorial>(object_id, spice_jd_to_jd(tc), center_id)?;
    if tc != t {
        state.pos += state.vel * ((t - tc) / 86400.0);
    }
    Ok(state)
}

/// Return the source coverage of `object_id` as query segments and as runs.
///
/// The query segments are the source segments, shrunk inward by
/// `BOUNDARY_BUFFER`. The runs are ranges over which the source is continuous
/// to `threshold_km`. Runs use the time bounds of the source. A run continues
/// across a small gap where the data is continuous. At a segment boundary
/// where the source jumps, one run ends and the next run starts.
///
/// # Errors
/// `Error::Bounds` if `object_id` has no coverage, or no range longer than
/// `2 * BOUNDARY_BUFFER`.
fn continuous_runs(
    source: &super::SpkCollection,
    object_id: i32,
    center_id: i32,
    threshold_km: f64,
    explicit_ranges: Option<&[(f64, f64)]>,
) -> KeteResult<(Vec<(f64, f64)>, Vec<(f64, f64)>)> {
    let mut raw = raw_boundaries(source, object_id, explicit_ranges)?;
    raw.retain(|&(s, e)| e - s > 2.0 * BOUNDARY_BUFFER);
    raw.sort_by(|a, b| a.0.total_cmp(&b.0));
    let segments = buffer_and_sort(&raw, BOUNDARY_BUFFER);
    let merged = merge_continuous_runs(
        &raw,
        GAP_TOLERANCE,
        source,
        object_id,
        center_id,
        threshold_km,
        &segments,
    );
    if merged.is_empty() {
        return Err(Error::Bounds(format!(
            "No SPK coverage found for NAIF ID {object_id}"
        )));
    }

    let mut boundaries: Vec<f64> = raw.iter().flat_map(|&(s, e)| [s, e]).collect();
    boundaries.sort_by(f64::total_cmp);
    boundaries.dedup();

    let mut runs = Vec::with_capacity(merged.len());
    for (lo, hi) in merged {
        let mut start = lo;
        for &t in &boundaries {
            if t - JUMP_PROBE <= start || t + JUMP_PROBE >= hi {
                continue;
            }
            let continuous = is_boundary_continuous(
                source,
                object_id,
                center_id,
                t - JUMP_PROBE,
                t + JUMP_PROBE,
                &segments,
                threshold_km,
            )
            .unwrap_or(false);
            if !continuous {
                runs.push((start, t));
                start = t;
            }
        }
        runs.push((start, hi));
    }
    Ok((segments, runs))
}

/// Fit each run with `fit`, and return all the arrays.
///
/// The function gives `fit` the bounds of the run and the query segments,
/// clipped to the run shrunk by `BOUNDARY_BUFFER`. Thus queries for a run
/// clamp into the run itself, and at a cut each side reads its own data.
///
/// # Errors
/// Returns the first error from `fit`.
fn fit_runs(
    runs: &[(f64, f64)],
    segments: &[(f64, f64)],
    fit: impl Fn(f64, f64, &[(f64, f64)]) -> KeteResult<Vec<SpkArray>>,
) -> KeteResult<Vec<SpkArray>> {
    let mut arrays = Vec::with_capacity(runs.len());
    for &(t_start, t_end) in runs {
        let (lo, hi) = (t_start + BOUNDARY_BUFFER, t_end - BOUNDARY_BUFFER);
        let clipped: Vec<(f64, f64)> = segments
            .iter()
            .map(|&(a, b)| (a.max(lo), b.min(hi)))
            .filter(|(a, b)| a < b)
            .collect();
        arrays.extend(fit(t_start, t_end, &clipped)?);
    }
    Ok(arrays)
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

// =============================================================================
//  Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// A time inside an outer range stays covered when a shorter range is
    /// nested in it, including when the two ranges share a start.
    #[test]
    fn nested_ranges_stay_covered() {
        for raw in [[(0.0, 100.0), (10.0, 20.0)], [(0.0, 100.0), (0.0, 20.0)]] {
            let segments = buffer_and_sort(&raw, 1.0);
            assert_eq!(segments, vec![(1.0, 99.0)]);
            assert_eq!(clamp_to_coverage(50.0, &segments), 50.0);
        }
        // Disjoint ranges stay separate, and a gap clamps to the nearest edge.
        let segments = buffer_and_sort(&[(0.0, 10.0), (20.0, 30.0)], 1.0);
        assert_eq!(segments, vec![(1.0, 9.0), (21.0, 29.0)]);
        assert_eq!(clamp_to_coverage(14.0, &segments), 9.0);
    }
    use kete_core::time::{TDB, Time};
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

    /// The output arrays cover the source coverage without gaps.
    ///
    /// A threshold that the fit cannot meet gives an error, not a partial
    /// result.
    #[test]
    fn repack_covers_everything_or_errors() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap();
        let bsp_path = root.join("docs/data/20000042.bsp");
        let mut spk = super::super::SpkCollection::default();
        spk.load_file(bsp_path.to_str().unwrap()).unwrap();
        let object_id = *spk.segments.keys().next().expect("No segments found");
        let center_id = spk.available_info(object_id)[0].2;

        for output_type in [2, 13] {
            let repack = |threshold_km: f64, degree: usize| {
                if output_type == 2 {
                    repack_to_type2(&spk, object_id, center_id, threshold_km, degree, None)
                } else {
                    repack_to_type13(&spk, object_id, center_id, threshold_km, degree, None)
                }
            };

            let mut arrays = repack(0.5, 7).unwrap();
            arrays.sort_by(|a, b| a.jds_start.total_cmp(&b.jds_start));
            for pair in arrays.windows(2) {
                assert!(
                    pair[1].jds_start <= pair[0].jds_end,
                    "type {output_type}: gap from {} to {}",
                    pair[0].jds_end,
                    pair[1].jds_start
                );
            }
            let source = spk.segment_boundaries(object_id);
            let start = source.iter().map(|x| x.0).fold(f64::INFINITY, f64::min);
            let end = source.iter().map(|x| x.1).fold(f64::NEG_INFINITY, f64::max);
            assert_eq!(arrays[0].jds_start, start, "type {output_type}");
            assert_eq!(arrays[arrays.len() - 1].jds_end, end, "type {output_type}");

            assert!(repack(1e-9, 1).is_err(), "type {output_type}");
        }
    }

    /// The output does not bridge a jump between two source segments.
    ///
    /// The output has a cut at the jump, and each side reproduces its own
    /// segment. The test checks two segments that abut, and a later segment
    /// that overlaps an earlier one.
    #[test]
    fn repack_cuts_at_a_jump() {
        for second_start in [86400.0, 43200.0] {
            repack_cuts_at_a_jump_at(second_start);
        }
    }

    fn repack_cuts_at_a_jump_at(jump: f64) {
        use crate::spk::SpkSegmentType18;
        let fixed = |x_km: f64, t0: f64| {
            let records = [x_km, 0.0, 0.0, 0.0, 0.0, 0.0];
            let records = [records, records].concat();
            SpkSegmentType18::new_array(
                1000,
                10,
                1,
                &records,
                &[t0, t0 + 86400.0],
                1,
                2,
                t0,
                t0 + 86400.0,
                "fixed",
            )
            .unwrap()
        };
        let mut daf = crate::daf::DafFile::new_spk("jump", "");
        daf.arrays.push(fixed(1.0e8, 0.0).daf);
        daf.arrays.push(fixed(1.0e8 + 1000.0, jump).daf);
        let mut buf = Cursor::new(Vec::new());
        daf.write_to(&mut buf).unwrap();
        let mut spk = super::super::SpkCollection::default();
        spk.load_from_reader(Cursor::new(buf.into_inner())).unwrap();

        for output_type in [2, 13] {
            let arrays = if output_type == 2 {
                repack_to_type2(&spk, 1000, 10, 0.01, 7, None).unwrap()
            } else {
                repack_to_type13(&spk, 1000, 10, 0.01, 7, None).unwrap()
            };
            let mut out = crate::daf::DafFile::new_spk("out", "");
            for array in arrays {
                assert!(
                    array.jds_end <= jump || array.jds_start >= jump,
                    "type {output_type}: array spans the jump at {jump}"
                );
                out.arrays.push(array.daf);
            }
            let mut buf = Cursor::new(Vec::new());
            out.write_to(&mut buf).unwrap();
            let mut repacked = super::super::SpkCollection::default();
            repacked
                .load_from_reader(Cursor::new(buf.into_inner()))
                .unwrap();
            for (t, x_km) in [
                (1000.0, 1.0e8),
                (jump - 400.0, 1.0e8),
                (jump + 600.0, 1.0e8 + 1000.0),
            ] {
                let state = repacked
                    .try_get_state_with_center::<Equatorial>(1000, spice_jd_to_jd(t), 10)
                    .unwrap();
                assert!(
                    (state.pos[0] * AU_KM - x_km).abs() < 0.01,
                    "type {output_type} jump={jump} t={t}"
                );
            }
        }
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
