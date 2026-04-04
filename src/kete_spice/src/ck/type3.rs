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

use super::CkArray;
use crate::sclk::LOADED_SCLK;
use kete_core::errors::{Error, KeteResult};
use kete_core::frames::NonInertialFrame;
use kete_core::time::{TDB, Time};
use nalgebra::{Quaternion, Rotation3, Unit, UnitQuaternion};

/// Discrete pointing data with linear interpolation between.
///
/// This segment type is broken up into intervals, each with a beginning and
/// end. One or more data points may be contained within each intervals. Linear
/// interpolation may be performed within a intervals.
///
/// Queries may include a user supplied tolerance on the requested time.
///
/// Interpolation does not extend past the bounds of an interval, the closest
/// point may be returned, provided it is within the specified tolerance.
///
/// Single points of data are allowed (no interpolation as long as it is within
/// the tolerance).
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

    /// Get the list of times inside of the interval.
    ///
    /// This queries the directory of interval start times, then uses the
    /// start and stop of the matching interval to find the list of times
    /// in the SCLK time directory.
    ///
    /// If the time requested is not within any interval, return the closest
    /// interval, and the start index of the associated clock times.
    fn get_times_in_interval(&self, time_sclk: f64) -> (&[f64], usize) {
        // first, check if the time is inside a known interval
        let interval_starts = self.interval_starts();
        if self.n_intervals == 1 {
            // If there is only one interval, return its times
            return (self.record_times(), 0);
        }
        let mut interval_idx = interval_starts.partition_point(|&x| x <= time_sclk);

        // if the interval_index is the last one, or the second to last one, return the last interval
        if interval_idx >= self.n_intervals - 1 {
            interval_idx = self.n_intervals - 2;
        }
        let interval_start_time = interval_starts[interval_idx];
        let interval_stop_time = interval_starts[interval_idx + 1];

        // find the start and stop index in the time directory uing the interval times
        let record_times = self.record_times();
        let start_idx = record_times.partition_point(|&x| x < interval_start_time);
        let stop_idx = record_times.partition_point(|&x| x <= interval_stop_time);
        (&record_times[start_idx..stop_idx], start_idx)
    }

    pub(crate) fn try_get_orientation(
        &self,
        time: Time<TDB>,
    ) -> KeteResult<(Time<TDB>, NonInertialFrame)> {
        let (time, quaternion, accel) = self.get_quaternion_at_time(time)?;

        let mut rates: [f64; 3] = accel.unwrap_or_default();
        for x in &mut rates {
            *x *= 86400.0;
        }
        let rotation_rate = Rotation3::from_scaled_axis(rates.into());

        let frame = NonInertialFrame::from_rotations(
            time,
            quaternion.to_rotation_matrix().inverse(),
            Some(rotation_rate.inverse().into_inner()),
            self.array.reference_frame_id,
            self.array.instrument_id,
        );

        Ok((time, frame))
    }

    /// Return the record at the given time, interpolating if necessary.
    ///
    /// This will return the best effort record, along with the time of
    /// the record. If the requested time is outside of any interval, this
    /// will return the closest record.
    pub(crate) fn get_quaternion_at_time(
        &self,
        time: Time<TDB>,
    ) -> KeteResult<(Time<TDB>, UnitQuaternion<f64>, Option<[f64; 3]>)> {
        let sclk = LOADED_SCLK
            .try_read()
            .map_err(|_| Error::Bounds("Failed to read SCLK data.".into()))?;
        let tick = sclk.try_time_to_tick(self.array.naif_id, time)?;

        // If there is only one record, return it immediately.
        if self.n_records == 1 {
            let record = self.get_record(0);
            let t = sclk.try_tick_to_time(self.array.naif_id, self.record_times()[0])?;
            let (quat, accel) = record.into();
            return Ok((t, Unit::from_quaternion(quat), accel));
        }

        let (interval_times, start_idx) = self.get_times_in_interval(tick);

        // find the closest two times in the interval
        let mut idx = interval_times.partition_point(|&x| x <= tick);
        if interval_times.len() == idx {
            // if the index is after the end of the interval, return the last record
            let record = self.get_record(idx - 1);
            let (quat, accel) = record.into();
            let t = sclk.try_tick_to_time(self.array.naif_id, *(interval_times.last().unwrap()))?;
            Ok((t, Unit::from_quaternion(quat), accel))
        } else if idx == 0 {
            // if the index is before the beginning of the interval, return the first record
            let record = self.get_record(0);
            let (quat, accel) = record.into();
            let t =
                sclk.try_tick_to_time(self.array.naif_id, *(interval_times.first().unwrap()))?;
            Ok((t, Unit::from_quaternion(quat), accel))
        } else {
            // otherwise, we have two records to interpolate between
            idx -= 1;
            let t1 = interval_times[idx];
            let t2 = interval_times[idx + 1];
            let (q0, acc0) = self.get_record(start_idx + idx).into();
            let (q1, acc1) = self.get_record(start_idx + idx + 1).into();
            let dt = (tick - t1) / (t2 - t1);
            let quaternion = q0.lerp(&q1, dt);

            let accel: Option<[f64; 3]> = acc0.map(|acc0| {
                acc0.iter()
                    .zip(acc1.unwrap())
                    .map(|(a1, a2)| a1 * (1.0 - dt) + a2 * dt)
                    .collect::<Vec<_>>()
                    .try_into()
                    .unwrap()
            });

            Ok((time, Unit::from_quaternion(quaternion), accel))
        }
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
        let n_records = array.daf[array.daf.len() - 1] as usize;
        let n_intervals = array.daf[array.daf.len() - 2] as usize;

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
