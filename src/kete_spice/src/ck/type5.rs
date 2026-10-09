// SPDX-FileCopyrightText: 2026 Dar Dahlen
// SPDX-License-Identifier: BSD-3-Clause

//! CK Segment Type 5 - MEX/Rosetta attitude file interpolation.
//!
//! The segment stores attitude as a series of quaternion packets. Each packet
//! has an SCLK time tag, and the tags need not be evenly spaced. Interpolation
//! intervals partition the packets, as in type 3. Inside an interval, the
//! reader interpolates a sliding window of packets. The window is centered on
//! the request time. The interval bounds truncate the window.
//!
//! <https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/ck.html#Data%20Type%205>

use super::array::stored_count;
use super::{CkArray, instrument_frame};
use crate::interpolation::{hermite_interpolation, lagrange_interpolation_both};
use crate::text::sclk::Sclk;
use kete_core::errors::{Error, KeteResult};
use kete_core::frames::NonInertialFrame;
use kete_core::time::{TDB, Time};
use nalgebra::{Quaternion, Unit, UnitQuaternion};

/// Return the number of values in a type 5 packet of the given subtype.
///
/// The function returns `None` for an unsupported subtype. The subtypes are:
///
/// - 0: quaternion and quaternion derivative. Hermite interpolation.
/// - 1: quaternion only. Lagrange interpolation. The angular velocity comes
///   from the derivative of the polynomial.
/// - 2: quaternion, quaternion derivative, angular velocity, and angular
///   velocity derivative. Hermite interpolation.
/// - 3: quaternion and angular velocity. Lagrange interpolation.
pub(in crate::ck) const fn packet_size(subtype: usize) -> Option<usize> {
    match subtype {
        0 => Some(8),
        1 => Some(4),
        2 => Some(14),
        3 => Some(7),
        _ => None,
    }
}

/// Continuous pointing data interpolated with a sliding polynomial window.
///
/// The time tags are encoded SCLK ticks. The file stores derivatives per
/// second. The interpolation abscissae are ticks, so the reader multiplies the
/// derivatives by `seconds_per_tick` before the interpolation. It divides the
/// interpolated quaternion derivative by `seconds_per_tick` to return it per
/// second.
#[derive(Debug)]
pub struct CkSegmentType5 {
    pub(in crate::ck) array: CkArray,

    n_records: usize,
    n_intervals: usize,
    rec_size: usize,
    subtype: usize,
    window_size: usize,

    /// Nominal spacecraft clock rate, in seconds per tick.
    seconds_per_tick: f64,

    time_start_idx: usize,
    interval_start_idx: usize,
}

impl CkSegmentType5 {
    #[inline(always)]
    fn record_times(&self) -> &[f64] {
        unsafe {
            self.array
                .daf
                .data
                .get_unchecked(self.time_start_idx..self.time_start_idx + self.n_records)
        }
    }

    #[inline(always)]
    fn interval_starts(&self) -> &[f64] {
        unsafe {
            self.array
                .daf
                .data
                .get_unchecked(self.interval_start_idx..self.interval_start_idx + self.n_intervals)
        }
    }

    /// Return whether `tick` falls inside one of the interpolation intervals.
    ///
    /// An interval runs from its start time to the last record before the next
    /// interval starts. The segment holds no pointing between intervals.
    pub(in crate::ck) fn has_data_at(&self, tick: f64) -> bool {
        let starts = self.interval_starts();
        let idx = starts.partition_point(|&x| x <= tick);
        if idx == 0 {
            return false;
        }
        let times = self.record_times();
        let end = if idx < self.n_intervals {
            times.partition_point(|&x| x < starts[idx])
        } else {
            self.n_records
        };
        end > 0 && tick <= times[end - 1]
    }

    /// Return the range of record indices in the interval that contains `tick`.
    ///
    /// The range is `(begin, end)`, with `end` exclusive. Interval `i` runs
    /// from its start time to the last record before the start of interval
    /// `i + 1`. Thus the ranges partition the records.
    /// [`super::CkCollection::try_get_pointing`] checks [`Self::has_data_at`]
    /// first, so a tick from there is always inside an interval.
    fn interval_records(&self, tick: f64) -> (usize, usize) {
        let times = self.record_times();
        if self.n_intervals == 1 {
            return (0, self.n_records);
        }
        let starts = self.interval_starts();
        let idx = starts.partition_point(|&x| x <= tick).saturating_sub(1);

        let begin = times.partition_point(|&x| x < starts[idx]);
        let end = if idx + 1 < self.n_intervals {
            times.partition_point(|&x| x < starts[idx + 1])
        } else {
            self.n_records
        };
        (begin, end.max(begin + 1))
    }

    pub(crate) fn try_get_orientation(
        &self,
        time: Time<TDB>,
        tick: f64,
        sclk: &Sclk,
    ) -> KeteResult<(Time<TDB>, NonInertialFrame)> {
        let (time, quaternion, rates) = self.get_quaternion_at_time(time, tick, sclk)?;

        let frame = instrument_frame(
            time,
            quaternion.to_rotation_matrix(),
            rates,
            self.array.reference_frame_id,
        );

        Ok((time, frame))
    }

    /// Return the interpolated attitude and angular velocity at `time`.
    ///
    /// The function returns the time, the quaternion of the C-matrix, and the
    /// angular velocity in radians per second. The time is the record time for
    /// a segment with one record. Otherwise it is the request time.
    ///
    /// # Errors
    /// [`Error::Bounds`] if the segment has one record, and its time is outside
    /// the clock `sclk`.
    ///
    /// # Panics
    /// Panics under the same condition as [`Self::get_quaternion_at_tick`].
    pub(crate) fn get_quaternion_at_time(
        &self,
        time: Time<TDB>,
        tick: f64,
        sclk: &Sclk,
    ) -> KeteResult<(Time<TDB>, UnitQuaternion<f64>, Option<[f64; 3]>)> {
        let (quat, rates) = self.get_quaternion_at_tick(tick);
        if self.n_records == 1 {
            Ok((sclk.tick_to_time(self.record_times()[0])?, quat, rates))
        } else {
            Ok((time, quat, rates))
        }
    }

    /// Return the interpolated attitude at an encoded clock tick.
    ///
    /// # Panics
    /// Panics if the segment has more than one interval, and `tick` is at or
    /// after an interval start that is later than the last record time. Only a
    /// malformed segment has such an interval start. [`Self::has_data_at`] is
    /// false for such a tick.
    pub(crate) fn get_quaternion_at_tick(
        &self,
        tick: f64,
    ) -> (UnitQuaternion<f64>, Option<[f64; 3]>) {
        let (begin, end) = self.interval_records(tick);
        PacketSeries {
            packets: &self.array.daf.data[begin * self.rec_size..end * self.rec_size],
            epochs: &self.record_times()[begin..end],
            subtype: self.subtype,
            window_size: self.window_size,
            rec_size: self.rec_size,
            seconds_per_tick: self.seconds_per_tick,
            has_rates: self.array.produces_angular_rates,
        }
        .eval(tick)
    }
}

/// A series of type 5 style attitude packets with their clock ticks.
///
/// One type 5 interpolation interval and one type 6 mini-segment share this
/// layout. Both segment types use the interpolation here. The angular velocity
/// is in radians per second.
pub(in crate::ck) struct PacketSeries<'a> {
    pub packets: &'a [f64],
    pub epochs: &'a [f64],
    pub subtype: usize,
    pub window_size: usize,
    pub rec_size: usize,
    pub seconds_per_tick: f64,
    pub has_rates: bool,
}

impl PacketSeries<'_> {
    #[inline(always)]
    fn packet(&self, idx: usize) -> &[f64] {
        unsafe {
            self.packets
                .get_unchecked(idx * self.rec_size..(idx + 1) * self.rec_size)
        }
    }

    /// Return the first index and the size of the interpolation window.
    ///
    /// The window is centered on the request time. The request time sits
    /// between the two central tags. The window holds at most half the nominal
    /// size on each side of the request time. Near either end of the series,
    /// the window is truncated, not shifted. Thus it can hold fewer packets
    /// than the nominal size.
    fn window(&self, tick: f64) -> (usize, usize) {
        let n = self.epochs.len();
        let half = self.window_size / 2;
        // `low` and `high` bracket the request time, as in the SPICE reader: `low`
        // is the last tag before the request and `high` the next. A request on the
        // first tag uses the first two tags, so the window still reaches past it.
        let before = self.epochs.partition_point(|&e| e < tick);
        let (low, high) = if before == 0 {
            (0, 1.min(n - 1))
        } else {
            (before - 1, before)
        };
        // Use half of the nominal window size on each side, truncated at the ends
        // of the series.
        let left = half.min(low + 1);
        let right = half.min(n - high);
        let first = low + 1 - left;
        (first, (left + right).min(n - first).max(1))
    }

    /// Return the per-packet signs that make the window quaternions consistent.
    ///
    /// Each entry is +1 or -1. With the signs applied, successive quaternions
    /// in the window have a non-negative dot product.
    fn window_signs(&self, start: usize, size: usize) -> Box<[f64]> {
        let mut signs = vec![1.0; size].into_boxed_slice();
        for i in 1..size {
            let prev = self.packet(start + i - 1);
            let cur = self.packet(start + i);
            let dot: f64 = (0..4).map(|c| prev[c] * cur[c]).sum();
            signs[i] = if dot * signs[i - 1] < 0.0 { -1.0 } else { 1.0 };
        }
        signs
    }

    pub(in crate::ck) fn eval(&self, tick: f64) -> (UnitQuaternion<f64>, Option<[f64; 3]>) {
        let mut quat = [0.0; 4];
        let mut dquat = [0.0; 4];
        let mut rates = [0.0; 3];

        if self.epochs.len() == 1 {
            // A single packet holds the attitude and its derivatives directly.
            // A subtype without a stored derivative has a zero rate.
            let packet = self.packet(0);
            quat.copy_from_slice(&packet[..4]);
            match self.subtype {
                0 => dquat.copy_from_slice(&packet[4..8]),
                2 => {
                    dquat.copy_from_slice(&packet[4..8]);
                    rates.copy_from_slice(&packet[8..11]);
                }
                3 => rates.copy_from_slice(&packet[4..7]),
                _ => (),
            }
        } else {
            let (start, size) = self.window(tick);
            let times = &self.epochs[start..start + size];

            // The file stores derivatives per second. The abscissae are ticks.
            let rate = self.seconds_per_tick;

            match self.subtype {
                0 | 2 => {
                    // As for the other subtypes, a stored packet can carry the
                    // opposite sign of its neighbors. A quaternion and its
                    // derivative flip together, so the interpolation stays exact.
                    // The SPICE reader raises an error here instead.
                    let signs = self.window_signs(start, size);
                    for idx in 0..4 {
                        let q: Box<[f64]> = (0..size)
                            .map(|i| self.packet(start + i)[idx] * signs[i])
                            .collect();
                        let dq: Box<[f64]> = (0..size)
                            .map(|i| self.packet(start + i)[idx + 4] * rate * signs[i])
                            .collect();
                        let (v, dv) = hermite_interpolation(times, &q, &dq, tick - times[0]);
                        quat[idx] = v;
                        dquat[idx] = dv / rate;
                    }
                    if self.subtype == 2 {
                        for (idx, out) in rates.iter_mut().enumerate() {
                            let a: Box<[f64]> =
                                (0..size).map(|i| self.packet(start + i)[idx + 8]).collect();
                            let da: Box<[f64]> = (0..size)
                                .map(|i| self.packet(start + i)[idx + 11] * rate)
                                .collect();
                            *out = hermite_interpolation(times, &a, &da, tick - times[0]).0;
                        }
                    }
                }
                1 | 3 => {
                    // q and -q are the same attitude. These subtypes put no sign
                    // restriction on the stored packets, so a window can contain a
                    // sign flip. An interpolation across a sign flip does not give
                    // a valid attitude. Thus the code makes the signs consistent
                    // first.
                    let signs = self.window_signs(start, size);
                    for idx in 0..4 {
                        let mut q: Box<[f64]> = (0..size)
                            .map(|i| self.packet(start + i)[idx] * signs[i])
                            .collect();
                        let (v, dv) = lagrange_interpolation_both(times, &mut q, tick - times[0]);
                        quat[idx] = v;
                        dquat[idx] = dv / rate;
                    }
                    if self.subtype == 3 {
                        for (idx, out) in rates.iter_mut().enumerate() {
                            let mut a: Box<[f64]> =
                                (0..size).map(|i| self.packet(start + i)[idx + 4]).collect();
                            *out = lagrange_interpolation_both(times, &mut a, tick - times[0]).0;
                        }
                    }
                }
                _ => unreachable!(),
            }
        }

        let quat = Quaternion::new(quat[0], quat[1], quat[2], quat[3]);
        let norm = quat.norm();
        let unit = Unit::new_normalize(quat);

        let rates = if self.has_rates {
            Some(if self.subtype == 2 || self.subtype == 3 {
                rates
            } else {
                // Only the unit quaternion holds attitude. Thus the code uses
                // the derivative of q/|q|, not the derivative of q.
                let dq = Quaternion::new(dquat[0], dquat[1], dquat[2], dquat[3]);
                let dot = quat.dot(&dq) / (norm * norm);
                let dunit = (dq - quat * dot) / norm;
                angular_velocity(&unit, &dunit)
            })
        } else {
            None
        };

        (unit, rates)
    }
}

/// Return the angular velocity from a unit quaternion and its time derivative.
///
/// The result is the vector part of `-2 * conj(q) * dq`. SPICE `qdq2av`
/// computes the same quantity. The unit is radians per time unit of `dq`.
fn angular_velocity(q: &UnitQuaternion<f64>, dq: &Quaternion<f64>) -> [f64; 3] {
    let product = q.conjugate().into_inner() * *dq;
    [-2.0 * product.i, -2.0 * product.j, -2.0 * product.k]
}

/// Check that a CK type 5 or type 6 window size is even and at least 2.
///
/// # Errors
/// [`Error::IOError`] if the window size is odd or less than 2.
pub(in crate::ck) fn check_window(window_size: usize) -> KeteResult<()> {
    if window_size < 2 || !window_size.is_multiple_of(2) {
        return Err(Error::IOError(format!(
            "CK window size must be even and at least 2, found {window_size}."
        )));
    }
    Ok(())
}

impl TryFrom<CkArray> for CkSegmentType5 {
    type Error = Error;

    fn try_from(array: CkArray) -> Result<Self, Self::Error> {
        let len = array.daf.len();
        if len < 5 {
            return Err(Error::Bounds("CK Segment Type 5 is truncated.".into()));
        }
        let n_records = stored_count(array.daf[len - 1], len, "record count")?;
        let n_intervals = stored_count(array.daf[len - 2], len, "interval count")?;
        let window_size = stored_count(array.daf[len - 3], len, "window size")?;
        let subtype = stored_count(array.daf[len - 4], len, "subtype")?;
        let seconds_per_tick = array.daf[len - 5];
        check_window(window_size)?;

        if n_records == 0 {
            return Err(Error::Bounds(
                "CK File does not contain any records.".into(),
            ));
        }
        if n_intervals == 0 {
            return Err(Error::Bounds(
                "CK File does not contain any intervals of records.".into(),
            ));
        }
        let rec_size = packet_size(subtype).ok_or_else(|| {
            Error::ValueError(format!(
                "CK Segment Type 5 does not support subtype {subtype}."
            ))
        })?;
        if !(seconds_per_tick.is_finite() && seconds_per_tick > 0.0) {
            return Err(Error::ValueError(
                "CK Segment Type 5 has a non-positive clock rate.".into(),
            ));
        }

        let time_dir_size = (n_records - 1) / 100;
        let interval_dir_size = (n_intervals - 1) / 100;

        // Layout: [n*rec_size packets][n times][time dir]
        //         [n_intervals starts][interval dir]
        //         [rate, subtype, window, n_intervals, n]
        let expected =
            n_records * rec_size + n_records + time_dir_size + n_intervals + interval_dir_size + 5;
        if expected != len {
            return Err(Error::Bounds(format!(
                "CK Segment Type 5 not formatted correctly. Expected {expected} values, found {len}."
            )));
        }

        let time_start_idx = n_records * rec_size;
        let interval_start_idx = time_start_idx + n_records + time_dir_size;

        Ok(Self {
            array,
            n_records,
            n_intervals,
            rec_size,
            subtype,
            window_size: window_size.max(1),
            seconds_per_tick,
            time_start_idx,
            interval_start_idx,
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

    /// Return the quaternion, its derivative, and the angular velocity.
    ///
    /// The attitude is a rotation about a constant axis. Its angle increases
    /// linearly with `seconds`. Derivatives are per second. The angular
    /// velocity of this attitude is `-SPIN * axis`. That is the vector part of
    /// `-2 * conj(q) * dq`.
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

    /// Build a type 5 segment with one interval and `n` packets.
    ///
    /// The packets are one minute apart. The data follows the layout that
    /// `try_from` reads.
    fn segment(subtype: usize, window: usize, n: usize) -> CkSegmentType5 {
        let rec_size = packet_size(subtype).unwrap();
        let mut data: Vec<f64> = Vec::new();
        let mut ticks: Vec<f64> = Vec::new();
        for i in 0..n {
            let seconds = 60.0 * i as f64;
            ticks.push(1.0e12 + seconds / TICK_RATE);
            let (q, dq, av) = attitude(seconds);
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
        assert_eq!(data.len(), n * rec_size);
        data.extend_from_slice(&ticks);
        for i in 1..=(n - 1) / 100 {
            data.push(ticks[i * 100 - 1]);
        }
        data.push(ticks[0]);
        data.extend_from_slice(&[TICK_RATE, subtype as f64, window as f64, 1.0, n as f64]);

        let array = CkArray::new(
            -226000,
            1,
            5,
            true,
            ticks[0],
            *ticks.last().unwrap(),
            data,
            "type 5 test".into(),
        );
        array.try_into().unwrap()
    }

    /// Compare a real Rosetta segment with the result of the SPICE toolkit.
    ///
    /// The segment has subtype 1, 14 packets, and a nominal window of 10. Thus
    /// the window is centered, not truncated. The reference quaternion comes
    /// from `ckgpav` on the same file at the same encoded tick.
    #[test]
    fn real_rosetta_segment_matches_cspice() {
        // Segment 26440 of RATT_DV_223_01_01____00302.BC, copied verbatim.
        let data: Vec<f64> = vec![
            0.9058548559676056,
            -0.20966544835954948,
            -0.12677143189787743,
            0.3455378181019547,
            0.9068243519999534,
            -0.2052347616349985,
            -0.12934027899093115,
            0.3447018704141857,
            0.9093734554625186,
            -0.19284277297128877,
            -0.13653482350256174,
            0.34235920519832275,
            0.9138513520985015,
            -0.16760987053339582,
            -0.15106010222170355,
            0.33758477910635704,
            0.9184458338152195,
            -0.1332743988415864,
            -0.17052208404414426,
            0.3310851911730904,
            0.9221589156571363,
            -0.08274763069948401,
            -0.1984883067111904,
            0.32152473619707916,
            0.9216505365732763,
            -0.015408157762950962,
            -0.23443763559781675,
            0.3088071762829507,
            0.9130741026569581,
            0.06525927623050716,
            -0.2753543517691789,
            0.2936271290000916,
            0.8994905924020717,
            0.13210906859463115,
            -0.30732475085351696,
            0.28109671945444886,
            0.8856004656328235,
            0.18176967160737947,
            -0.32983217990282154,
            0.27181305129670363,
            0.874488539806264,
            0.21497705418223934,
            -0.34424501366296056,
            0.26561255710305354,
            0.865679435641274,
            0.23861551412315327,
            -0.3541709806983688,
            0.26119852135627447,
            0.8614426824516471,
            0.2493275904680596,
            -0.35856706276051536,
            0.2591947510757084,
            0.8603996832920119,
            0.25192037092604597,
            -0.3596110553037436,
            0.25870523884687135,
            24253585116025.742,
            24253590358904.027,
            24253595601782.312,
            24253602155380.17,
            24253608708978.03,
            24253616573295.457,
            24253625748332.457,
            24253636234089.035,
            24253645409126.027,
            24253653273443.46,
            24253659827041.32,
            24253666380639.176,
            24253671623517.465,
            24253675555676.18,
            24253585116025.742,
            1.52587890625e-05,
            1.0,
            10.0,
            1.0,
            14.0,
        ];
        let tick = 24253618080623.316;
        let expected = [
            0.922_487_085_515_535_9,
            -0.072_169_070_279_168_88,
            -0.204_238_061_256_795_17,
            0.319_524_673_047_572_9,
        ];

        let array = CkArray::new(-226000, 1, 5, true, data[56], data[69], data, "real".into());
        let seg: CkSegmentType5 = array.try_into().unwrap();
        let (quat, _) = seg.get_quaternion_at_tick(tick);
        let q = quat.into_inner();
        // q and -q are the same attitude, so compare up to sign.
        let sign = if q.w * expected[0] < 0.0 { -1.0 } else { 1.0 };
        let got = [sign * q.w, sign * q.i, sign * q.j, sign * q.k];
        for idx in 0..4 {
            let err = (got[idx] - expected[idx]).abs();
            assert!(
                err < 1e-14,
                "component {idx}: got {} want {} ({err:e})",
                got[idx],
                expected[idx]
            );
        }
    }

    #[test]
    fn packet_sizes_match_the_spec() {
        assert_eq!(packet_size(0), Some(8));
        assert_eq!(packet_size(1), Some(4));
        assert_eq!(packet_size(2), Some(14));
        assert_eq!(packet_size(3), Some(7));
        assert_eq!(packet_size(4), None);
    }

    #[test]
    fn malformed_segment_is_rejected() {
        let mut seg = segment(1, 4, 8);
        // Claim more packets than the data can hold.
        let len = seg.array.daf.data.len();
        seg.array.daf.data[len - 1] = 99.0;
        let array = CkArray::new(
            -226000,
            1,
            5,
            true,
            0.0,
            1.0,
            seg.array.daf.data.to_vec(),
            "x".into(),
        );
        assert!(CkSegmentType5::try_from(array).is_err());
    }

    /// At a stored time tag, the interpolation reproduces that packet.
    #[test]
    fn reproduces_packets_at_their_own_epochs() {
        for (subtype, window) in [(0, 4), (1, 4), (2, 4), (3, 4)] {
            let seg = segment(subtype, window, 12);
            for i in 0..12 {
                let tick = seg.record_times()[i];
                let (quat, _) = seg.get_quaternion_at_tick(tick);
                let (expected, _, _) = attitude(60.0 * i as f64);
                let got = quat.into_inner();
                let err = (got.w - expected[0])
                    .abs()
                    .max((got.i - expected[1]).abs())
                    .max((got.j - expected[2]).abs())
                    .max((got.k - expected[3]).abs());
                assert!(err < 1e-12, "subtype {subtype} packet {i}: error {err:e}");
            }
        }
    }

    /// The angular velocity matches the analytic value for every subtype.
    ///
    /// Subtypes 2 and 3 store the angular velocity. Subtypes 0 and 1 derive it
    /// from the quaternion.
    #[test]
    fn angular_velocity_matches_the_analytic_value() {
        let (_, _, expected) = attitude(0.0);
        for (subtype, window) in [(0, 6), (1, 6), (2, 6), (3, 6)] {
            let seg = segment(subtype, window, 12);
            // Use a tick midway between two tags, so that the result depends on
            // the interpolation.
            let tick = f64::midpoint(seg.record_times()[5], seg.record_times()[6]);
            let (_, rates) = seg.get_quaternion_at_tick(tick);
            let rates = rates.expect("segment declares angular rates");
            for idx in 0..3 {
                let err = (rates[idx] - expected[idx]).abs();
                assert!(
                    err < 1e-10,
                    "subtype {subtype} axis {idx}: got {} want {} ({err:e})",
                    rates[idx],
                    expected[idx]
                );
            }
        }
    }

    /// A single packet gives its stored rate for subtypes 0, 2 and 3, and a zero
    /// rate for subtype 1, which stores no derivative.
    #[test]
    fn single_packet_rates() {
        let (_, _, stored) = attitude(0.0);
        for subtype in 0..4 {
            let seg = segment(subtype, 2, 1);
            let (_, rates) = seg.get_quaternion_at_tick(seg.record_times()[0]);
            let rates = rates.expect("segment declares angular rates");
            let expected = if subtype == 1 { [0.0; 3] } else { stored };
            for idx in 0..3 {
                assert!(
                    (rates[idx] - expected[idx]).abs() < 1e-12,
                    "subtype {subtype}"
                );
            }
        }
    }

    /// At the first tag the window still reaches past it, as in the SPICE reader.
    /// A Lagrange window of 2 then gives the chord rate, not zero.
    #[test]
    fn first_epoch_rate_uses_two_packets() {
        let (_, _, expected) = attitude(0.0);
        let seg = segment(1, 2, 6);
        let (_, rates) = seg.get_quaternion_at_tick(seg.record_times()[0]);
        let rates = rates.expect("segment declares angular rates");
        for idx in 0..3 {
            assert!(
                (rates[idx] - expected[idx]).abs() < 1e-3 * expected[idx].abs().max(1e-12),
                "axis {idx}: got {} want {}",
                rates[idx],
                expected[idx]
            );
        }
    }

    /// A Hermite packet stored with the opposite sign of its neighbors is the same
    /// attitude, and the interpolation across it is unchanged.
    #[test]
    fn hermite_sign_flip_is_the_same_attitude() {
        for subtype in [0, 2] {
            let reference = segment(subtype, 4, 12);
            let mut flipped = segment(subtype, 4, 12);
            let rec_size = packet_size(subtype).unwrap();
            // Flip the quaternion and its derivative of packet 6.
            for value in &mut flipped.array.daf.data[6 * rec_size..6 * rec_size + 8] {
                *value = -*value;
            }
            let tick = f64::midpoint(reference.record_times()[5], reference.record_times()[6]);
            let (q_ref, _) = reference.get_quaternion_at_tick(tick);
            let (q_flip, _) = flipped.get_quaternion_at_tick(tick);
            let dot = q_ref.into_inner().dot(&q_flip.into_inner()).abs();
            assert!((dot - 1.0).abs() < 1e-12, "subtype {subtype}: |dot| {dot}");
        }
    }

    /// A window wider than the segment is truncated to the available packets.
    ///
    /// The reader does not read past the end of the series.
    #[test]
    fn window_wider_than_the_segment_is_truncated() {
        let seg = segment(1, 10, 3);
        let tick = seg.record_times()[1];
        let (quat, _) = seg.get_quaternion_at_tick(tick);
        let (expected, _, _) = attitude(60.0);
        assert!((quat.into_inner().w - expected[0]).abs() < 1e-12);
    }
}
