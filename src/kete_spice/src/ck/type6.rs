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

//! CK Segment Type 6 - MEX/Rosetta attitude, piecewise interpolation.
//!
//! A type 6 segment holds a sequence of mini-segments and a set of intervals.
//! Each interval selects one mini-segment. A mini-segment has the layout of a
//! type 5 segment without the interpolation intervals. Thus one type 6 segment
//! can hold data that needs several type 5 segments.
//!
//! <https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/ck.html#Data%20Type%206>

use super::array::stored_count;
use super::type5::{PacketSeries, check_window, packet_size};
use super::{CkArray, instrument_frame};
use kete_core::errors::{Error, KeteResult};
use kete_core::frames::NonInertialFrame;
use kete_core::time::{TDB, Time};
use nalgebra::UnitQuaternion;

/// Piecewise pointing data made of type 5 style mini-segments.
///
/// The mini-segments occupy the start of the segment. The interval bounds, the
/// bound directory, the mini-segment pointers, and a trailer follow them. The
/// struct holds only the offsets. A request resolves to one mini-segment. The
/// reader reads that mini-segment as a packet series, as for a type 5
/// interval.
///
/// Each mini-segment can have its own subtype and window size. Thus the reader
/// reads them for each request and does not cache them.
#[derive(Debug)]
pub struct CkSegmentType6 {
    pub(in crate::ck) array: CkArray,

    /// Number of mini-segments, which equals the number of intervals.
    n_intervals: usize,

    /// Index of the first of the `n_intervals + 1` interval bound times.
    bounds_idx: usize,

    /// Index of the first of the `n_intervals + 1` mini-segment pointers.
    pointers_idx: usize,

    /// Selection rule for a request exactly on a shared interval bound.
    ///
    /// If true, the request uses the later interval. If false, it uses the
    /// earlier interval.
    select_last: bool,
}

impl CkSegmentType6 {
    #[inline(always)]
    fn bounds(&self) -> &[f64] {
        unsafe {
            self.array
                .daf
                .data
                .get_unchecked(self.bounds_idx..self.bounds_idx + self.n_intervals + 1)
        }
    }

    #[inline(always)]
    fn pointers(&self) -> &[f64] {
        unsafe {
            self.array
                .daf
                .data
                .get_unchecked(self.pointers_idx..self.pointers_idx + self.n_intervals + 1)
        }
    }

    /// Return whether the segment holds pointing at `tick`.
    ///
    /// The rules follow SPICE CKR06 with zero tolerance. The tick must lie
    /// between the first and the last interval bound. The tick must also be no
    /// later than the last packet epoch of the mini-segment for its interval. A
    /// mini-segment can end before its interval ends. The gap after it holds no
    /// pointing. A malformed mini-segment reports pointing, so that the read
    /// returns the error.
    pub(in crate::ck) fn has_data_at(&self, tick: f64) -> bool {
        let bounds = self.bounds();
        if !(bounds[0] <= tick && tick <= bounds[self.n_intervals]) {
            return false;
        }
        self.mini_segment(self.interval_index(tick))
            .map_or(true, |series| {
                series.epochs.last().is_some_and(|&t| tick <= t)
            })
    }

    /// Return the index of the interval that covers the request tick.
    ///
    /// Interval `i` spans `bounds[i]` to `bounds[i + 1]`. The `select_last`
    /// flag resolves a request on a shared bound. A request outside every
    /// interval clamps to the nearest interval. [`Self::has_data_at`] and
    /// [`Self::get_quaternion_at_tick`] reject such a request.
    fn interval_index(&self, tick: f64) -> usize {
        let bounds = self.bounds();
        let mut idx = bounds.partition_point(|&b| b <= tick).saturating_sub(1);

        // partition_point counts an exact match, so an exact match lands on
        // the later interval. Step back when the file selects the earlier one.
        if !self.select_last && idx > 0 && bounds[idx] == tick {
            idx -= 1;
        }
        idx.min(self.n_intervals - 1)
    }

    /// Return the mini-segment of an interval as a type 5 style packet series.
    ///
    /// `interval` is the interval index. The stored pointers are 1-based.
    ///
    /// # Errors
    /// - [`Error::IOError`] if a pointer is out of range, if the mini-segment
    ///   has fewer than 4 values, or if it is shorter than its packet count.
    /// - [`Error::IOError`] if the stored record count, window size, or subtype
    ///   is not a valid whole number, if the record count is zero, or if the
    ///   window size is odd or less than 2.
    /// - [`Error::ValueError`] if the subtype is not supported, or if the clock
    ///   rate is not finite and positive.
    #[allow(
        clippy::cast_sign_loss,
        clippy::cast_possible_truncation,
        reason = "Pointers are positive offsets stored as f64; correct for a valid file."
    )]
    fn mini_segment(&self, interval: usize) -> KeteResult<PacketSeries<'_>> {
        let pointers = self.pointers();
        let out_of_range =
            || Error::IOError("CK Type 6: mini-segment pointer out of range.".into());
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

        if data.len() < 4 {
            return Err(Error::IOError("CK Type 6: mini-segment truncated.".into()));
        }
        let len = data.len();
        let n_records = stored_count(data[len - 1], len, "record count")?;
        let window_size = stored_count(data[len - 2], len, "window size")?;
        let subtype = stored_count(data[len - 3], len, "subtype")?;
        let seconds_per_tick = data[len - 4];
        check_window(window_size)?;
        let rec_size = packet_size(subtype).ok_or_else(|| {
            Error::ValueError(format!(
                "CK Segment Type 6 does not support subtype {subtype}."
            ))
        })?;

        if n_records == 0 || len < n_records * (rec_size + 1) {
            return Err(Error::IOError(
                "CK Type 6: mini-segment shorter than its packet count.".into(),
            ));
        }
        if !(seconds_per_tick.is_finite() && seconds_per_tick > 0.0) {
            return Err(Error::ValueError(
                "CK Segment Type 6 mini-segment has a non-positive clock rate.".into(),
            ));
        }

        Ok(PacketSeries {
            packets: &data[..n_records * rec_size],
            epochs: &data[n_records * rec_size..n_records * (rec_size + 1)],
            subtype,
            // Keep the nominal size even when the mini-segment is shorter. The
            // evaluator truncates the window to the available packets.
            window_size: window_size.max(1),
            rec_size,
            seconds_per_tick,
            has_rates: self.array.produces_angular_rates,
        })
    }

    pub(crate) fn try_get_orientation(
        &self,
        time: Time<TDB>,
        tick: f64,
    ) -> KeteResult<(Time<TDB>, NonInertialFrame)> {
        let (quaternion, rates) = self.get_quaternion_at_tick(tick)?;

        let frame = instrument_frame(
            time,
            quaternion.to_rotation_matrix(),
            rates,
            self.array.reference_frame_id,
        );

        Ok((time, frame))
    }

    /// Return the interpolated attitude and angular velocity (radians per second) at
    /// an encoded clock tick.
    ///
    /// # Errors
    /// - [`Error::Bounds`] if the segment holds no pointing at `tick`.
    /// - The errors of [`Self::mini_segment`] if the mini-segment for `tick` is
    ///   malformed.
    pub(crate) fn get_quaternion_at_tick(
        &self,
        tick: f64,
    ) -> KeteResult<(UnitQuaternion<f64>, Option<[f64; 3]>)> {
        if !self.has_data_at(tick) {
            return Err(Error::Bounds(
                "CK type 6 segment has no pointing at the requested time.".into(),
            ));
        }
        Ok(self.mini_segment(self.interval_index(tick))?.eval(tick))
    }
}

impl TryFrom<CkArray> for CkSegmentType6 {
    type Error = Error;

    #[allow(
        clippy::cast_sign_loss,
        clippy::cast_possible_truncation,
        reason = "Counts come from the file and are correct for a valid file."
    )]
    fn try_from(array: CkArray) -> Result<Self, Self::Error> {
        let len = array.daf.len();
        if len < 4 {
            return Err(Error::Bounds("CK Segment Type 6 is truncated.".into()));
        }
        let n_intervals = array.daf[len - 1] as usize;
        let select_last = array.daf[len - 2] == 1.0;

        if n_intervals == 0 {
            return Err(Error::Bounds(
                "CK Segment Type 6 contains no intervals.".into(),
            ));
        }

        // From the end, the segment holds the trailer, n+1 pointers, the bound
        // directory, and n+1 bounds. The directory holds every 100th bound. It
        // drops the last entry when the bound count is a multiple of 100. Thus
        // it has n / 100 entries.
        let pointers_idx = len
            .checked_sub(2 + n_intervals + 1)
            .ok_or_else(|| Error::Bounds("CK Segment Type 6 is truncated.".into()))?;
        let bounds_idx = pointers_idx
            .checked_sub(n_intervals / 100 + n_intervals + 1)
            .ok_or_else(|| Error::Bounds("CK Segment Type 6 is truncated.".into()))?;

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

    const AXIS: [f64; 3] = [0.3, -0.5, 0.81];
    const SPIN: f64 = 2.0e-4;
    const TICK_RATE: f64 = 1.52587891e-05;

    fn unit_axis() -> [f64; 3] {
        let n = (AXIS[0] * AXIS[0] + AXIS[1] * AXIS[1] + AXIS[2] * AXIS[2]).sqrt();
        [AXIS[0] / n, AXIS[1] / n, AXIS[2] / n]
    }

    /// Return the same analytic attitude that the type 5 tests use.
    fn attitude(seconds: f64) -> ([f64; 4], [f64; 4], [f64; 3]) {
        let u = unit_axis();
        let ang = 0.1 + SPIN * seconds;
        let (s, c) = (ang / 2.0).sin_cos();
        let q = [c, s * u[0], s * u[1], s * u[2]];
        let dq = [
            -SPIN / 2.0 * s,
            SPIN / 2.0 * c * u[0],
            SPIN / 2.0 * c * u[1],
            SPIN / 2.0 * c * u[2],
        ];
        let av = [-SPIN * u[0], -SPIN * u[1], -SPIN * u[2]];
        (q, dq, av)
    }

    fn tick_of(seconds: f64) -> f64 {
        1.0e12 + seconds / TICK_RATE
    }

    /// Build one mini-segment.
    ///
    /// The layout is packets, tags, directory, rate, subtype, window, and
    /// count.
    fn mini_segment(subtype: usize, window: usize, seconds: &[f64]) -> Vec<f64> {
        let mut data = Vec::new();
        for &t in seconds {
            let (q, dq, av) = attitude(t);
            match subtype {
                0 => {
                    data.extend_from_slice(&q);
                    data.extend_from_slice(&dq);
                }
                1 => data.extend_from_slice(&q),
                2 => {
                    data.extend_from_slice(&q);
                    data.extend_from_slice(&dq);
                    data.extend_from_slice(&av);
                    data.extend_from_slice(&[0.0; 3]);
                }
                _ => {
                    data.extend_from_slice(&q);
                    data.extend_from_slice(&av);
                }
            }
        }
        let tags: Vec<f64> = seconds.iter().map(|&t| tick_of(t)).collect();
        data.extend_from_slice(&tags);
        for i in 1..=(tags.len() - 1) / 100 {
            data.push(tags[i * 100 - 1]);
        }
        data.extend_from_slice(&[
            TICK_RATE,
            subtype as f64,
            window as f64,
            seconds.len() as f64,
        ]);
        data
    }

    /// Build a segment of two mini-segments that meet at 600 s.
    ///
    /// `select_last` sets the selection rule at the shared bound.
    fn segment(subtype: usize, select_last: bool) -> CkSegmentType6 {
        let first: Vec<f64> = (0..11).map(|i| 60.0 * f64::from(i)).collect();
        segment_with_first(subtype, select_last, &first)
    }

    /// Build a segment of two intervals that meet at 600 s.
    ///
    /// The first mini-segment holds packets at the times in `first`, in
    /// seconds. The second holds packets from 600 s to 1200 s.
    fn segment_with_first(subtype: usize, select_last: bool, first: &[f64]) -> CkSegmentType6 {
        let second: Vec<f64> = (10..21).map(|i| 60.0 * f64::from(i)).collect();
        let a = mini_segment(subtype, 6, first);
        let b = mini_segment(subtype, 6, &second);

        let mut data = Vec::new();
        data.extend_from_slice(&a);
        data.extend_from_slice(&b);
        // Interval bounds, then an empty directory, then 1-based pointers.
        data.extend_from_slice(&[tick_of(0.0), tick_of(600.0), tick_of(1200.0)]);
        data.push(1.0);
        data.push(a.len() as f64 + 1.0);
        data.push((a.len() + b.len()) as f64 + 1.0);
        data.push(if select_last { 1.0 } else { 0.0 });
        data.push(2.0);

        CkArray::new(
            -226000,
            1,
            6,
            true,
            tick_of(0.0),
            tick_of(1200.0),
            data,
            "type 6 test".into(),
        )
        .try_into()
        .unwrap()
    }

    /// At a stored tag, the interpolation reproduces that packet.
    ///
    /// The test covers the tags of both mini-segments.
    #[test]
    fn every_subtype_reproduces_its_packets() {
        for subtype in [0, 1, 2, 3] {
            let seg = segment(subtype, true);
            for i in 0..=20 {
                let seconds = 60.0 * f64::from(i);
                let (quat, _) = seg.get_quaternion_at_tick(tick_of(seconds)).unwrap();
                let (want, _, _) = attitude(seconds);
                let got = quat.into_inner();
                let err = (got.w - want[0])
                    .abs()
                    .max((got.i - want[1]).abs())
                    .max((got.j - want[2]).abs())
                    .max((got.k - want[3]).abs());
                assert!(err < 1e-11, "subtype {subtype} tag {i}: error {err:e}");
            }
        }
    }

    #[test]
    fn angular_velocity_matches_the_analytic_value() {
        let (_, _, want) = attitude(0.0);
        for subtype in [0, 1, 2, 3] {
            let seg = segment(subtype, true);
            // Inside the second mini-segment, between two tags.
            let (_, rates) = seg.get_quaternion_at_tick(tick_of(870.0)).unwrap();
            let rates = rates.expect("segment declares angular rates");
            for idx in 0..3 {
                let err = (rates[idx] - want[idx]).abs();
                assert!(err < 1e-10, "subtype {subtype} axis {idx}: error {err:e}");
            }
        }
    }

    /// The flag selects the interval at a shared bound.
    ///
    /// Both mini-segments describe the same attitude. Thus the attitude at the
    /// bound is the same for either flag value.
    #[test]
    fn boundary_flag_selects_an_interval() {
        for select_last in [true, false] {
            let seg = segment(1, select_last);
            let bound = tick_of(600.0);
            assert_eq!(seg.interval_index(bound), usize::from(select_last));
            assert_eq!(seg.interval_index(tick_of(300.0)), 0);
            assert_eq!(seg.interval_index(tick_of(900.0)), 1);
            let (quat, _) = seg.get_quaternion_at_tick(bound).unwrap();
            let (want, _, _) = attitude(600.0);
            assert!((quat.into_inner().w - want[0]).abs() < 1e-11);
        }
    }

    /// A request past the final bound clamps to the last interval.
    ///
    /// The index stays inside the pointer array. The segment holds no pointing
    /// at such a request.
    #[test]
    fn request_past_the_final_bound_is_clamped() {
        let seg = segment(1, true);
        assert_eq!(seg.interval_index(tick_of(1200.0)), 1);
        assert_eq!(seg.interval_index(tick_of(1e6)), 1);
        assert!(seg.get_quaternion_at_tick(tick_of(1200.0)).is_ok());
        assert!(seg.get_quaternion_at_tick(tick_of(1260.0)).is_err());
    }

    /// A mini-segment that ends before its interval leaves a gap.
    ///
    /// The gap holds no pointing. The reader does not extrapolate into it.
    #[test]
    fn gap_after_the_last_packet_has_no_pointing() {
        let first: Vec<f64> = (0..9).map(|i| 60.0 * f64::from(i)).collect();
        let seg = segment_with_first(1, true, &first);
        assert!(seg.has_data_at(tick_of(480.0)));
        assert!(seg.get_quaternion_at_tick(tick_of(480.0)).is_ok());
        assert!(!seg.has_data_at(tick_of(540.0)));
        assert!(seg.get_quaternion_at_tick(tick_of(540.0)).is_err());
        assert!(seg.has_data_at(tick_of(600.0)));
    }

    #[test]
    fn unsupported_subtype_is_rejected() {
        let mut seg = segment(1, true);
        // The subtype of the first mini-segment is the third value from its
        // end.
        #[allow(
            clippy::cast_sign_loss,
            reason = "pointers are positive by construction"
        )]
        let a_len = seg.pointers()[1] as usize - 1;
        seg.array.daf.data[a_len - 3] = 9.0;
        assert!(seg.get_quaternion_at_tick(tick_of(30.0)).is_err());
    }
}
