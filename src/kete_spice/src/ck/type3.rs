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

use super::array::stored_count;
use super::{CkArray, instrument_frame};
use crate::sclk::LOADED_SCLK;
use kete_core::errors::{Error, KeteResult};
use kete_core::frames::NonInertialFrame;
use kete_core::time::{TDB, Time};
use nalgebra::{Quaternion, Unit, UnitQuaternion};

/// Discrete pointing data with linear interpolation between records.
///
/// The segment holds a set of interpolation intervals. Each interval has a
/// start time and holds one or more records. The reader interpolates only
/// between two records of the same interval.
///
/// Interpolation does not extend past the bounds of an interval. The segment
/// holds no pointing between intervals. A request exactly on a record returns
/// that record. Thus an interval with a single record holds pointing only at
/// the time of that record.
///
/// <https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/ck.html#Data%20Type%203>
#[derive(Debug)]
pub struct CkSegmentType3 {
    pub(in crate::ck) array: CkArray,
    n_intervals: usize,
    n_records: usize,
    rec_size: usize,

    interval_start_idx: usize,
    time_start_idx: usize,
}

impl CkSegmentType3 {
    fn get_record(&self, idx: usize) -> Type3RecordView<'_> {
        unsafe {
            let rec = self
                .array
                .daf
                .data
                .get_unchecked(idx * self.rec_size..(idx + 1) * self.rec_size);
            Type3RecordView {
                quaternion: rec[..4].try_into().unwrap_unchecked(),
                accel: &rec[4..],
            }
        }
    }

    fn interval_starts(&self) -> &[f64] {
        unsafe {
            self.array
                .daf
                .data
                .get_unchecked(self.interval_start_idx..self.interval_start_idx + self.n_intervals)
        }
    }

    fn record_times(&self) -> &[f64] {
        unsafe {
            self.array
                .daf
                .data
                .get_unchecked(self.time_start_idx..self.time_start_idx + self.n_records)
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

    pub(crate) fn try_get_orientation(
        &self,
        time: Time<TDB>,
    ) -> KeteResult<(Time<TDB>, NonInertialFrame)> {
        let (time, quaternion, accel) = self.get_quaternion_at_time(time)?;

        let frame = instrument_frame(
            time,
            quaternion.to_rotation_matrix(),
            accel,
            self.array.reference_frame_id,
        );

        Ok((time, frame))
    }

    /// Return the pointing at the given time, interpolating if necessary.
    ///
    /// The rules follow SPICE CKR03 and CKE03 with zero tolerance. Between two
    /// records of the same interval, the rotation turns at a constant angular
    /// rate. The turn is about the axis of the rotation that carries the first
    /// C-matrix to the second. The angular velocity is interpolated linearly. A
    /// request exactly on a record with no partner in its interval returns that
    /// record.
    ///
    /// The function returns the time, the quaternion of the C-matrix, and the
    /// angular velocity if the records hold it. The time is the request time
    /// when the function interpolates. Otherwise it is the record time.
    ///
    /// # Errors
    /// - [`Error::Bounds`] if the SCLK singleton lock is not available.
    /// - [`Error::ValueError`] if no SCLK clock is loaded for the spacecraft.
    /// - [`Error::Bounds`] if no interval covers the time.
    pub(crate) fn get_quaternion_at_time(
        &self,
        time: Time<TDB>,
    ) -> KeteResult<(Time<TDB>, UnitQuaternion<f64>, Option<[f64; 3]>)> {
        let sclk = LOADED_SCLK
            .try_read()
            .map_err(|_| Error::Bounds("Failed to read SCLK data.".into()))?;
        let naif_id = self.array.naif_id;
        let tick = sclk.try_time_to_tick(naif_id, time)?;

        let times = self.record_times();
        let starts = self.interval_starts();
        let not_covered =
            || Error::Bounds("CK type 3 segment has no pointing at the requested time.".into());

        let single = |idx: usize| -> KeteResult<_> {
            let (quat, rates) = self.get_record(idx).into();
            let t = sclk.try_tick_to_time(naif_id, times[idx])?;
            Ok((t, Unit::from_quaternion(quat), rates))
        };

        // Number of records at or before the request.
        let n_before = times.partition_point(|&x| x <= tick);
        if n_before == 0 {
            return Err(not_covered());
        }
        if n_before == self.n_records {
            return if tick == times[self.n_records - 1] {
                single(self.n_records - 1)
            } else {
                Err(not_covered())
            };
        }
        let (left, right) = (n_before - 1, n_before);

        // The following record belongs to the same interval only if it comes
        // before the next interval starts.
        let next_start = starts
            .get(starts.partition_point(|&x| x <= tick))
            .copied()
            .unwrap_or(f64::INFINITY);
        if times[right] >= next_start {
            return if tick == times[left] {
                single(left)
            } else {
                Err(not_covered())
            };
        }

        let frac = (tick - times[left]) / (times[right] - times[left]);
        let (q0, rate0) = self.get_record(left).into();
        let (q1, rate1) = self.get_record(right).into();
        let (q0, q1) = (Unit::from_quaternion(q0), Unit::from_quaternion(q1));
        // Use the shortest rotation from the first record to the second. It
        // does not depend on the sign of either stored quaternion.
        let quaternion = q0 * (q0.inverse() * q1).powf(frac);

        let rates = match (rate0, rate1) {
            (Some(a), Some(b)) => Some(std::array::from_fn(|i| a[i] * (1.0 - frac) + b[i] * frac)),
            _ => None,
        };
        Ok((time, quaternion, rates))
    }

    /// Build a CK Type 3 data array (discrete pointing with linear interpolation).
    ///
    /// Records contain either 7 values (with angular velocity) or 4 values (without):
    /// - With rates: `[q0, q1, q2, q3, av1, av2, av3]`
    /// - Without rates: `[q0, q1, q2, q3]`
    ///
    /// # Arguments
    /// * `records`           - Flat slice of `n * rec_size` pointing values.
    /// * `record_times`      - n SCLK times, one per record.
    /// * `interval_starts`   - m SCLK interval start times (<= n, defines interpolation regions).
    /// * `has_angular_rates` - Whether records include angular velocity (7 vs 4 values).
    ///
    /// # Errors
    /// Returns an error if there are no records or intervals, or the records
    /// length is inconsistent with the expected per-record size.
    fn build_data(
        records: &[f64],
        record_times: &[f64],
        interval_starts: &[f64],
        has_angular_rates: bool,
    ) -> KeteResult<Vec<f64>> {
        let n = record_times.len();
        let m = interval_starts.len();
        let rec_size: usize = if has_angular_rates { 7 } else { 4 };

        if n == 0 {
            return Err(Error::ValueError(
                "CK Type 3: need at least one record.".into(),
            ));
        }
        if m == 0 {
            return Err(Error::ValueError(
                "CK Type 3: need at least one interval.".into(),
            ));
        }
        if records.len() != n * rec_size {
            return Err(Error::ValueError(format!(
                "CK Type 3: records length ({}) must be n ({}) * rec_size ({})",
                records.len(),
                n,
                rec_size
            )));
        }

        // Layout: [n*rec_size records][n record_times][time_dir]
        //         [m interval_starts][interval_dir][m][n]
        let time_dir_size = if n > 100 { (n - 1) / 100 } else { 0 };
        let interval_dir_size = if m > 100 { (m - 1) / 100 } else { 0 };

        let total = n * rec_size + n + time_dir_size + m + interval_dir_size + 2;
        let mut data = Vec::with_capacity(total);

        // Pointing records
        data.extend_from_slice(records);

        // Record times
        data.extend_from_slice(record_times);

        // Record time directory
        for i in 1..=time_dir_size {
            data.push(record_times[(i * 100 - 1).min(n - 1)]);
        }

        // Interval start times
        data.extend_from_slice(interval_starts);

        // Interval directory
        for i in 1..=interval_dir_size {
            data.push(interval_starts[(i * 100 - 1).min(m - 1)]);
        }

        // Trailer: n_intervals, n_records
        data.push(m as f64);
        data.push(n as f64);

        Ok(data)
    }

    /// Create a Type 3 (discrete pointing, linear interpolation) CK array.
    ///
    /// # Arguments
    /// * `instrument_id`      - NAIF instrument ID.
    /// * `reference_frame_id` - Reference frame ID.
    /// * `records`            - Flat slice of pointing values.
    /// * `record_times`       - n SCLK times for each record.
    /// * `interval_starts`    - m SCLK interval start times.
    /// * `has_angular_rates`  - Whether records include angular velocity.
    /// * `segment_name`       - Name stored in the DAF name record (max 40 chars).
    ///
    /// # Errors
    /// Returns an error if the data builder rejects the inputs.
    #[allow(
        clippy::missing_panics_doc,
        reason = "build_data validates non-empty slices before the unwrap is reached"
    )]
    pub fn new_array(
        instrument_id: i32,
        reference_frame_id: i32,
        records: &[f64],
        record_times: &[f64],
        interval_starts: &[f64],
        has_angular_rates: bool,
        segment_name: &str,
    ) -> KeteResult<CkArray> {
        let data = Self::build_data(records, record_times, interval_starts, has_angular_rates)?;
        let tick_start = record_times[0];
        let tick_end = *record_times.last().unwrap();
        Ok(CkArray::new(
            instrument_id,
            reference_frame_id,
            3,
            has_angular_rates,
            tick_start,
            tick_end,
            data,
            segment_name.to_string(),
        ))
    }
}

struct Type3RecordView<'a> {
    quaternion: &'a [f64; 4],
    accel: &'a [f64],
}

impl From<Type3RecordView<'_>> for (Quaternion<f64>, Option<[f64; 3]>) {
    fn from(record: Type3RecordView<'_>) -> Self {
        let quaternion = match record.quaternion {
            &[a, b, c, d] => Quaternion::new(a, b, c, d),
        };

        let accel = match record.accel {
            &[a, b, c] => Some([a, b, c]),
            _ => None,
        };
        (quaternion, accel)
    }
}

impl TryFrom<CkArray> for CkSegmentType3 {
    type Error = Error;

    #[allow(
        clippy::cast_sign_loss,
        reason = "cast should work except when file is incorrectly formatted"
    )]
    fn try_from(array: CkArray) -> Result<Self, Self::Error> {
        let len = array.daf.len();
        if len < 2 {
            return Err(Error::Bounds("CK Segment Type 3 is truncated.".into()));
        }
        let n_records = stored_count(array.daf[len - 1], len, "record count")?;
        let n_intervals = stored_count(array.daf[len - 2], len, "interval count")?;

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

        let rec_size = if array.produces_angular_rates { 7 } else { 4 };

        // Times are also broken up into a 'directory' of every 100th time.
        // This calculates the size of the directory.
        let time_dir_size = (n_records - 1) / 100;

        // interval times are also broken up into a 'directory' of every 100th
        // interval start time. This calculates the size of the directory.
        let interval_dir_size = (n_intervals - 1) / 100;

        // there are n_records
        let mut expected_size = n_records * rec_size;
        // 2 lists of times + 2 numbers at the end
        expected_size += n_intervals + n_records + 2;
        // 2 directories
        expected_size += time_dir_size + interval_dir_size;

        if expected_size != array.daf.len() {
            return Err(Error::Bounds(
                "CK File not formatted correctly. Number of records found in file don't match expected."
                    .into(),
            ));
        }

        let time_start_idx = n_records * rec_size;
        let interval_start_idx = time_start_idx + n_records + time_dir_size;

        Ok(Self {
            array,
            n_intervals,
            n_records,
            rec_size,
            interval_start_idx,
            time_start_idx,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::daf::DafFile;

    #[test]
    fn ck_type3_basic() {
        // 3 records with angular rates (7 each)
        let records = vec![
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.707, 0.707, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0,
            1.0, 0.0, 0.0, 0.0, 0.2,
        ];
        let times = vec![100.0, 200.0, 300.0];
        let intervals = vec![100.0, 300.0];
        let data = CkSegmentType3::build_data(&records, &times, &intervals, true).unwrap();
        // 21 records + 3 times + 0 time_dir + 2 intervals + 0 interval_dir + 2 = 28
        assert_eq!(data.len(), 28);
        // Last two: n_intervals=2, n_records=3
        assert_eq!(data[data.len() - 2], 2.0);
        assert_eq!(data[data.len() - 1], 3.0);
    }

    #[test]
    fn ck_type3_round_trip() {
        use std::io::Cursor;

        let records = vec![
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.707, 0.707, 0.0, 0.0, 0.1, 0.0, 0.0,
        ];
        let times = vec![100.0, 200.0];
        let intervals = vec![100.0, 200.0];

        let mut daf = DafFile::new_ck("test ck", "ck round trip");
        let ck_arr =
            CkSegmentType3::new_array(-12345, 1, &records, &times, &intervals, true, "Test CK Seg")
                .unwrap();
        daf.arrays.push(ck_arr.daf);

        let mut buf = Cursor::new(Vec::new());
        daf.write_to(&mut buf).unwrap();

        let bytes = buf.into_inner();
        let daf = DafFile::from_buffer(Cursor::new(&bytes)).unwrap();
        assert_eq!(daf.daf_type, crate::daf::DAFType::Ck);
        assert_eq!(daf.arrays.len(), 1);

        let ck: CkArray = daf.arrays.into_iter().next().unwrap().try_into().unwrap();
        assert_eq!(ck.instrument_id, -12345);
        assert_eq!(ck.segment_type, 3);
        assert!(ck.produces_angular_rates);
    }
}
