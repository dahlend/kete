//! SPK Segment Type 19 - ESOC/DDID Piecewise Interpolation.
//!
//! A type 19 segment holds a sequence of mini-segments and a set of
//! interpolation intervals. Each interval selects one mini-segment. Each
//! mini-segment has the ESOC/DDID packet layout of a type 18 segment, and can
//! also use subtype 2. One type 19 segment can hold an ephemeris that would
//! otherwise need many type 18 segments.
//!
//! <https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/FORTRAN/req/spk.html#Type%2019:%20ESOC/DDID%20Piecewise%20Interpolation>

use super::SpkArray;
use super::type18::{PacketSeries, TYPE19_MAX_DEGREE, check_packet_layout, packet_size};
use kete_core::errors::Error;
use kete_core::prelude::KeteResult;
use kete_core::time::{TDB, Time};

/// Type 19 Segment
///
/// The segment holds `n_intervals` mini-segments. After them come the interval
/// bounds, the bound directory, the mini-segment pointers, and a trailer of
/// two values. This struct stores only the offsets and the boundary flag. A
/// request selects one mini-segment. The reader evaluates it with the same
/// `PacketSeries` code as a type 18 segment.
///
/// Each mini-segment can have its own subtype and interpolation degree. The
/// reader therefore reads the mini-segment control words on each request and
/// does not cache them.
#[derive(Debug)]
pub struct SpkSegmentType19 {
    pub(crate) array: SpkArray,

    /// Number of interpolation intervals. Each interval has one mini-segment.
    n_intervals: usize,

    /// Index of the first of the `n_intervals + 1` interval bound times.
    bounds_idx: usize,

    /// Index of the first of the `n_intervals + 1` mini-segment pointers.
    pointers_idx: usize,

    /// If true, a request exactly on a shared interval bound uses the later
    /// interval. If false, it uses the earlier interval.
    select_last: bool,
}

impl SpkSegmentType19 {
    #[inline(always)]
    fn bounds(&self) -> &[f64] {
        // SAFETY: `try_from` placed `bounds_idx` so that the `n_intervals + 1`
        // bounds and the bound directory end at `pointers_idx`, inside the
        // array.
        unsafe {
            self.array
                .daf
                .data
                .get_unchecked(self.bounds_idx..self.bounds_idx + self.n_intervals + 1)
        }
    }

    #[inline(always)]
    fn pointers(&self) -> &[f64] {
        // SAFETY: `try_from` placed `pointers_idx` so that the
        // `n_intervals + 1` pointers end 2 values before the end of the array.
        unsafe {
            self.array
                .daf
                .data
                .get_unchecked(self.pointers_idx..self.pointers_idx + self.n_intervals + 1)
        }
    }

    /// Return the index of the interpolation interval that covers `jds`.
    ///
    /// Interval `i` spans `bounds[i]` to `bounds[i + 1]`. For a request on a
    /// shared bound, `select_last` selects the interval. A request before the
    /// first bound uses the first interval. A request after the last bound uses
    /// the last interval.
    fn interval_index(&self, jds: f64) -> usize {
        let bounds = self.bounds();
        let mut idx = bounds.partition_point(|&b| b <= jds).saturating_sub(1);

        // `partition_point` counts an exact match, so it gives the later
        // interval. Step back when the file selects the earlier interval.
        if !self.select_last && idx > 0 && bounds[idx] == jds {
            idx -= 1;
        }
        idx.min(self.n_intervals - 1)
    }

    /// Read the mini-segment of an interval as a [`PacketSeries`].
    ///
    /// `interval` must be less than `n_intervals`.
    ///
    /// # Errors
    /// Returns [`Error::IOError`] in these cases:
    /// - A pointer of the interval is less than 1, or the pointers do not give
    ///   a range inside the array.
    /// - The mini-segment holds fewer than 3 values.
    /// - A control word is not a non-negative whole number.
    /// - [`check_packet_layout`] rejects the control words, with a largest
    ///   degree of 27.
    ///
    /// Returns [`Error::ValueError`] if the subtype is not 0, 1, or 2.
    #[allow(
        clippy::cast_sign_loss,
        clippy::cast_possible_truncation,
        reason = "Pointers are 1-based indices stored as f64; correct for a valid file."
    )]
    fn mini_segment(&self, interval: usize) -> KeteResult<PacketSeries<'_>> {
        let pointers = self.pointers();
        let out_of_range = || Error::IOError("Type 19: mini-segment pointer out of range.".into());
        let start = (pointers[interval] as usize)
            .checked_sub(1)
            .ok_or_else(out_of_range)?;
        let end = (pointers[interval + 1] as usize)
            .checked_sub(1)
            .ok_or_else(out_of_range)?;
        let data = self
            .array
            .daf
            .data
            .get(start..end)
            .ok_or_else(out_of_range)?;

        if data.len() < 3 {
            return Err(Error::IOError("Type 19: mini-segment truncated.".into()));
        }
        let len = data.len();
        let control = [data[len - 3], data[len - 2], data[len - 1]];
        if control
            .iter()
            .any(|x| !(x.is_finite() && *x >= 0.0 && x.fract() == 0.0))
        {
            return Err(Error::IOError(format!(
                "Type 19: invalid mini-segment control words {control:?}."
            )));
        }
        let [subtype, window_size, n_records] = control.map(|x| x as usize);
        let record_size = packet_size(subtype).ok_or_else(|| {
            Error::ValueError(format!(
                "SPK Segment Type 19 does not support subtype {subtype}."
            ))
        })?;

        check_packet_layout(
            subtype,
            n_records,
            window_size,
            record_size,
            len - 3,
            TYPE19_MAX_DEGREE,
        )?;

        Ok(PacketSeries {
            packets: &data[..n_records * record_size],
            epochs: &data[n_records * record_size..n_records * (record_size + 1)],
            subtype,
            window_size,
            record_size,
        })
    }

    /// Return the position in AU and the velocity in AU/day at `jds`.
    ///
    /// `jds` is in TDB seconds from J2000.
    ///
    /// # Errors
    /// Returns the errors of `mini_segment` for the interval that covers `jds`.
    /// These are [`Error::IOError`] for a malformed mini-segment, and
    /// [`Error::ValueError`] for an unsupported subtype.
    #[inline(always)]
    pub(crate) fn try_get_pos_vel(&self, time: Time<TDB>) -> KeteResult<([f64; 3], [f64; 3])> {
        let jds = time.j2000_seconds();
        Ok(self
            .mini_segment(self.interval_index(jds))?
            .try_get_pos_vel(time))
    }
}

impl TryFrom<SpkArray> for SpkSegmentType19 {
    type Error = Error;

    // The cast truncates a fractional interval count. It maps a negative or NaN
    // count to 0, which the check below rejects.
    #[allow(
        clippy::cast_sign_loss,
        clippy::cast_possible_truncation,
        reason = "This is correct as long as the file is correct."
    )]
    fn try_from(array: SpkArray) -> KeteResult<Self> {
        let len = array.daf.len();
        if len < 4 {
            return Err(Error::IOError("SPK Segment Type 19 is truncated.".into()));
        }
        let count = array.daf[len - 1];
        let select_last = array.daf[len - 2] == 1.0;

        // The count must fit in the array, which also keeps the offset sums
        // below from overflowing.
        if !(count.is_finite() && count >= 1.0 && count.fract() == 0.0 && count < len as f64) {
            return Err(Error::IOError(format!(
                "SPK Segment Type 19 has an invalid interval count {count}."
            )));
        }
        let n_intervals = count as usize;

        // From the end, the array holds the trailer, the n + 1 pointers, the
        // bound directory, and the n + 1 bounds. The directory holds every
        // 100th bound except the final bound. It therefore has n / 100
        // entries.
        let pointers_idx = len
            .checked_sub(2 + n_intervals + 1)
            .ok_or_else(|| Error::IOError("SPK Segment Type 19 is truncated.".into()))?;
        let bounds_idx = pointers_idx
            .checked_sub(n_intervals / 100 + n_intervals + 1)
            .ok_or_else(|| Error::IOError("SPK Segment Type 19 is truncated.".into()))?;

        Ok(Self {
            array,
            n_intervals,
            bounds_idx,
            pointers_idx,
            select_last,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kete_core::constants::AU_KM;

    /// An interval count that is not a whole number within the array is an
    /// error, not a panic.
    #[test]
    fn invalid_interval_count_is_an_error() {
        for count in [1e300, f64::NAN, 0.0, 1.5, -1.0] {
            let array = SpkArray::new(
                1,
                0,
                1,
                19,
                0.0,
                1.0,
                vec![0.0, 0.0, 1.0, count],
                "x".into(),
            );
            assert!(SpkSegmentType19::try_from(array).is_err(), "count {count}");
        }
    }

    /// Return the state of a straight-line motion at `seconds`.
    ///
    /// Every subtype reproduces straight-line motion to rounding error.
    fn state_at(seconds: f64) -> ([f64; 3], [f64; 3]) {
        let vel = [1.5, -0.25, 0.75];
        (
            [
                1.0e8 + vel[0] * seconds,
                -2.0e8 + vel[1] * seconds,
                3.0e7 + vel[2] * seconds,
            ],
            vel,
        )
    }

    /// Build one mini-segment: packets, epochs, directory, subtype, window, and
    /// count.
    fn mini_segment(subtype: usize, window: usize, epochs: &[f64]) -> Vec<f64> {
        let mut data = Vec::new();
        for &t in epochs {
            let (pos, vel) = state_at(t);
            data.extend_from_slice(&pos);
            data.extend_from_slice(&vel);
            if subtype == 0 {
                // Subtype 0 holds the velocity twice. One copy is the
                // derivative for the position fit. The other copy is the
                // velocity to interpolate.
                data.extend_from_slice(&vel);
                data.extend_from_slice(&[0.0; 3]);
            }
        }
        data.extend_from_slice(epochs);
        for i in 1..=(epochs.len() - 1) / 100 {
            data.push(epochs[i * 100 - 1]);
        }
        data.push(subtype as f64);
        data.push(window as f64);
        data.push(epochs.len() as f64);
        data
    }

    /// Build two mini-segments that meet at `split`, with the given boundary
    /// flag.
    fn segment(subtype: usize, select_last: bool) -> SpkSegmentType19 {
        let split = 600.0;
        let first: Vec<f64> = (0..11).map(|i| 60.0 * f64::from(i)).collect();
        let second: Vec<f64> = (10..21).map(|i| 60.0 * f64::from(i)).collect();

        let a = mini_segment(subtype, 4, &first);
        let b = mini_segment(subtype, 4, &second);

        let mut data = Vec::new();
        data.extend_from_slice(&a);
        data.extend_from_slice(&b);
        // Interval bounds, an empty directory, and 1-based pointers.
        data.extend_from_slice(&[0.0, split, 1200.0]);
        data.push(1.0);
        data.push(a.len() as f64 + 1.0);
        data.push((a.len() + b.len()) as f64 + 1.0);
        data.push(if select_last { 1.0 } else { 0.0 });
        data.push(2.0);

        SpkArray::new(
            1_000_012,
            10,
            1,
            19,
            0.0,
            86400.0,
            data,
            "type 19 test".into(),
        )
        .try_into()
        .unwrap()
    }

    fn assert_state(seg: &SpkSegmentType19, seconds: f64) {
        let (pos, vel) = seg
            .try_get_pos_vel(Time::from_j2000_seconds(seconds))
            .unwrap();
        let (want_pos, want_vel) = state_at(seconds);
        for idx in 0..3 {
            let dp = (pos[idx] * AU_KM - want_pos[idx]).abs();
            let dv = (vel[idx] * AU_KM / 86400.0 - want_vel[idx]).abs();
            assert!(dp < 1e-4, "pos[{idx}] off by {dp:e} km at t={seconds}");
            assert!(dv < 1e-9, "vel[{idx}] off by {dv:e} km/s at t={seconds}");
        }
    }

    #[test]
    fn every_subtype_recovers_linear_motion() {
        for subtype in [0, 1, 2] {
            let seg = segment(subtype, true);
            for &t in &[30.0, 150.0, 599.0, 601.0, 900.0, 1170.0] {
                assert_state(&seg, t);
            }
        }
    }

    /// Check that the flag selects the interval at the shared bound.
    ///
    /// Both mini-segments describe the same motion. The state is therefore
    /// continuous across the shared bound for either flag value.
    #[test]
    fn boundary_flag_selects_an_interval() {
        let split = 600.0;
        for select_last in [true, false] {
            let seg = segment(1, select_last);
            assert_eq!(seg.interval_index(split), usize::from(select_last));
            assert_eq!(seg.interval_index(300.0), 0);
            assert_eq!(seg.interval_index(900.0), 1);
            assert_state(&seg, split);
        }
    }

    /// Check that a request at or past the final bound uses the last interval.
    ///
    /// This keeps the pointer index inside the pointer array.
    #[test]
    fn request_at_the_final_bound_is_clamped() {
        let seg = segment(1, true);
        assert_eq!(seg.interval_index(1200.0), 1);
        assert_eq!(seg.interval_index(1e9), 1);
        assert_state(&seg, 1200.0);
    }

    #[test]
    fn unsupported_subtype_is_rejected() {
        let mut seg = segment(1, true);
        // The subtype of the first mini-segment is 3 values before its end.
        #[allow(
            clippy::cast_sign_loss,
            reason = "pointers are positive by construction"
        )]
        let a_len = seg.pointers()[1] as usize - 1;
        seg.array.daf.data[a_len - 3] = 7.0;
        assert!(seg.try_get_pos_vel(Time::from_j2000_seconds(30.0)).is_err());
    }
}
