// SPDX-FileCopyrightText: 2026 Dar Dahlen
// SPDX-FileCopyrightText: 2025 California Institute of Technology
// SPDX-License-Identifier: BSD-3-Clause

//! SPK Segment Type 9 - Lagrange Interpolation (Unequal Time Steps).
//!
//! <https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/FORTRAN/req/spk.html#Type%209:%20Lagrange%20Interpolation%20---%20Unequal%20Time%20Steps>

use super::SpkArray;
use crate::interpolation::lagrange_interpolation;
use kete_core::constants::AU_KM;
use kete_core::errors::Error;
use kete_core::prelude::KeteResult;
use kete_core::time::{TDB, Time};

/// Lagrange Interpolation (Uneven Time Steps)
///
/// This uses a collection of individual positions/velocities and interpolates between
/// them using Lagrange interpolation.
/// <https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/FORTRAN/req/spk.html#Type%209:%20Lagrange%20Interpolation%20---%20Unequal%20Time%20Steps>
#[derive(Debug)]
pub struct SpkSegmentType9 {
    pub(crate) array: SpkArray,
    poly_degree: usize,
    n_records: usize,
}

impl SpkSegmentType9 {
    /// Create a Type 9 (Lagrange interpolation, unequal time steps) SPK array.
    ///
    /// `object_id` is the NAIF ID of the body, and `center_id` is the NAIF ID
    /// of the center body. `frame_id` is the NAIF frame ID. Each entry of
    /// `states` is `(epoch, [x, y, z], [vx, vy, vz])`. The epoch is in TDB
    /// seconds from J2000, the position is in km, and the velocity is in km/s.
    /// The segment covers the first epoch to the last epoch. `degree` is the
    /// Lagrange polynomial degree. `segment_name` is the name stored in the DAF
    /// name record, which holds at most 40 characters.
    ///
    /// # Errors
    /// Returns [`Error::ValueError`] in these cases:
    /// - `states` is empty.
    /// - `degree` is outside `[1, 27]`.
    /// - `states` holds fewer than `degree + 1` entries.
    /// - The epochs are not strictly increasing.
    pub fn new_array(
        object_id: i32,
        center_id: i32,
        frame_id: i32,
        states: &[(f64, [f64; 3], [f64; 3])],
        degree: u32,
        segment_name: &str,
    ) -> KeteResult<SpkArray> {
        let (Some(first), Some(last)) = (states.first(), states.last()) else {
            return Err(Error::ValueError("Type 9: need at least one state.".into()));
        };
        let (jds_start, jds_end) = (first.0, last.0);

        let n = states.len();
        if !(1..=27).contains(&degree) {
            return Err(Error::ValueError(
                "Type 9: degree must be in [1, 27].".into(),
            ));
        }
        if n <= degree as usize {
            return Err(Error::ValueError(
                "Type 9: need at least degree + 1 states.".into(),
            ));
        }
        for w in states.windows(2) {
            if w[1].0 <= w[0].0 {
                return Err(Error::ValueError(
                    "Type 9: epochs must be strictly increasing.".into(),
                ));
            }
        }

        // Layout: [6*n states] [n epochs] [directory] [degree] [n]
        let n_dir = if n > 100 { (n - 1) / 100 } else { 0 };
        let mut data = Vec::with_capacity(7 * n + n_dir + 2);
        for &(_, pos, vel) in states {
            data.extend_from_slice(&pos);
            data.extend_from_slice(&vel);
        }
        for &(epoch, _, _) in states {
            data.push(epoch);
        }
        for i in 1..=n_dir {
            data.push(states[i * 100 - 1].0);
        }
        data.push(f64::from(degree));
        data.push(n as f64);

        Ok(SpkArray::new(
            object_id,
            center_id,
            frame_id,
            9,
            jds_start,
            jds_end,
            data,
            segment_name.to_string(),
        ))
    }

    #[inline(always)]
    fn get_record(&self, idx: usize) -> Type9RecordView<'_> {
        unsafe {
            let rec = self.array.daf.data.get_unchecked(idx * 6..(idx + 1) * 6);
            Type9RecordView {
                pos: rec[0..3].try_into().unwrap(),
                vel: rec[3..6].try_into().unwrap(),
            }
        }
    }

    #[inline(always)]
    fn get_times(&self) -> &[f64] {
        unsafe {
            self.array
                .daf
                .data
                .get_unchecked(self.n_records * 6..self.n_records * 7)
        }
    }

    #[inline(always)]
    pub(crate) fn try_get_pos_vel(&self, time: Time<TDB>) -> ([f64; 3], [f64; 3]) {
        let jds = time.j2000_seconds();
        let times = self.get_times();
        let window_size = self.poly_degree + 1;
        let start_idx = window_start(times, jds, window_size);
        let offset = time.j2000_seconds_minus(times[start_idx]);

        let mut pos = [0.0; 3];
        let mut vel = [0.0; 3];
        for idx in 0..3 {
            let mut p: Box<[f64]> = (0..window_size)
                .map(|i| self.get_record(i + start_idx).pos[idx])
                .collect();
            let mut dp: Box<[f64]> = (0..window_size)
                .map(|i| self.get_record(i + start_idx).vel[idx])
                .collect();
            let p =
                lagrange_interpolation(&times[start_idx..start_idx + window_size], &mut p, offset);
            let v =
                lagrange_interpolation(&times[start_idx..start_idx + window_size], &mut dp, offset);
            pos[idx] = p / AU_KM;
            vel[idx] = v / AU_KM * 86400.;
        }

        (pos, vel)
    }
}

/// Type 9 Record View
/// A view into a record of type 9, provided mainly for clarity to the underlying
/// data structure.
struct Type9RecordView<'a> {
    pos: &'a [f64; 3],
    vel: &'a [f64; 3],
}

impl TryFrom<SpkArray> for SpkSegmentType9 {
    type Error = Error;

    fn try_from(array: SpkArray) -> KeteResult<Self> {
        let (poly_degree, n_records) = read_control(&array, "9")?;
        let window_size = poly_degree + 1;
        if window_size > n_records {
            return Err(Error::IOError(format!(
                "SPK Type 9: polynomial degree ({poly_degree}) requires at least \
                 {window_size} records, but only {n_records} present"
            )));
        }

        Ok(Self {
            array,
            poly_degree,
            n_records,
        })
    }
}

/// Read and check the two control words at the end of a type 9 or 13 segment.
///
/// The control words are `[value, N]`. `value` is the polynomial degree for
/// type 9, and the window size minus 1 for type 13. `N` is the number of
/// states. `kind` names the segment type in error messages. The function
/// returns `(value, N)`.
///
/// # Errors
/// Returns [`Error::IOError`] in these cases:
/// - The array holds fewer than 2 values.
/// - A control word is not a non-negative whole number.
/// - `value` is not less than `N`.
/// - The array holds fewer than `7 * N + 2` values.
#[allow(
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation,
    reason = "Values are checked to be non-negative whole numbers first."
)]
pub(in crate::spk) fn read_control(array: &SpkArray, kind: &str) -> KeteResult<(usize, usize)> {
    let len = array.daf.len();
    if len < 2 {
        return Err(Error::IOError(format!(
            "SPK Type {kind}: segment is truncated."
        )));
    }
    let value = array.daf[len - 2];
    let n_records = array.daf[len - 1];
    let valid = |x: f64| x.is_finite() && x >= 0.0 && x.fract() == 0.0;
    if !(valid(value) && valid(n_records)) {
        return Err(Error::IOError(format!(
            "SPK Type {kind}: invalid control words [{value}, {n_records}]."
        )));
    }
    let (value, n_records) = (value as usize, n_records as usize);
    // A window must fit in the states, so the degree (type 9) or the window
    // size minus 1 (type 13) must be less than the number of states.
    if value >= n_records || n_records.checked_mul(7).is_none_or(|x| x + 2 > len) {
        return Err(Error::IOError(format!(
            "SPK Type {kind}: segment holds {len} values, too few for {n_records} states."
        )));
    }
    Ok((value, n_records))
}

/// Return the first index of the interpolation window for SPK types 9 and 13.
///
/// This follows the SPKR09 window rule, which SPICE also uses for type 13.
/// `times` holds the state epochs in increasing order. An even window has the
/// same number of epochs on each side of the interval that contains `jds`. An
/// odd window centers on the nearest epoch. On a tie, it centers on the later
/// epoch. Near either end of the segment, the window shifts to stay inside it.
/// `window_size` must be in `1..=times.len()`.
///
/// # Panics
/// With overflow checks on, panics if `window_size` is 0 or greater than
/// `times.len()`. The callers pass a window size that [`read_control`] checked.
pub(in crate::spk) fn window_start(times: &[f64], jds: f64, window_size: usize) -> usize {
    let n = times.len();
    let n_before = times.partition_point(|&t| t < jds);
    // Last epoch strictly before jds, or the first epoch if there is none.
    let low = n_before.max(1) - 1;
    let anchor = if window_size.is_multiple_of(2)
        || n_before == 0
        || low + 1 >= n
        || (jds - times[low]).abs() < (jds - times[low + 1]).abs()
    {
        low
    } else {
        low + 1
    };
    anchor
        .saturating_sub((window_size - 1) / 2)
        .min(n - window_size)
}

#[cfg(test)]
mod tests {
    use super::window_start;

    /// Check window placement against the SPKR09 rule, with epochs every 60 s.
    #[test]
    fn window_start_matches_spkr09() {
        let times: Vec<f64> = (0..11).map(|i| f64::from(i) * 60.0).collect();

        // An even window splits evenly around the containing interval. The
        // nearer end of the interval does not change the window.
        assert_eq!(window_start(&times, 310.0, 8), 2);
        assert_eq!(window_start(&times, 350.0, 8), 2);
        assert_eq!(window_start(&times, 300.0, 8), 1);
        assert_eq!(window_start(&times, 310.0, 4), 4);

        // An odd window centers on the nearest epoch. On a tie, it uses the
        // later epoch.
        assert_eq!(window_start(&times, 310.0, 5), 3);
        assert_eq!(window_start(&times, 330.0, 5), 4);
        assert_eq!(window_start(&times, 300.0, 5), 3);

        // At either end, the window shifts to stay inside the segment.
        assert_eq!(window_start(&times, 0.0, 8), 0);
        assert_eq!(window_start(&times, 10.0, 5), 0);
        assert_eq!(window_start(&times, 600.0, 8), 3);
        assert_eq!(window_start(&times, 600.0, 11), 0);
    }
}
