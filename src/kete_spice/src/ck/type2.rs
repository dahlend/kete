// SPDX-FileCopyrightText: 2026 Dar Dahlen
// SPDX-License-Identifier: BSD-3-Clause

use super::{CkArray, instrument_frame};
use kete_core::errors::{Error, KeteResult};
use kete_core::frames::NonInertialFrame;
use kete_core::time::{TDB, Time};
use nalgebra::{Quaternion, Rotation3, Unit, Vector3};

/// Discrete pointing data.
///
/// The segment holds a set of intervals. The angular velocity is constant
/// inside each interval. Each interval has one record of 8 values. The first 4
/// values are the quaternion at the interval start. The next 3 values are the
/// angular velocity in radians per second, expressed in the reference frame.
/// The last value is the number of seconds per SCLK tick.
///
/// <https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/ck.html#Data%20Type%202>
#[derive(Debug)]
pub struct CkSegmentType2 {
    pub(in crate::ck) array: CkArray,

    n_records: usize,

    time_start_idx: usize,
}

impl CkSegmentType2 {
    fn get_record(&self, idx: usize) -> (Quaternion<f64>, [f64; 3], f64) {
        unsafe {
            let rec = self.array.daf.data.get_unchecked(idx * 8..(idx + 1) * 8);
            let quaternion = Quaternion::new(rec[0], rec[1], rec[2], rec[3]);
            let angular_velocity = [rec[4], rec[5], rec[6]];
            let seconds_per_tick = rec[7];
            (quaternion, angular_velocity, seconds_per_tick)
        }
    }

    fn time_stops(&self) -> &[f64] {
        let start = self.time_start_idx + self.n_records;
        unsafe {
            self.array
                .daf
                .data
                .get_unchecked(start..start + self.n_records)
        }
    }

    /// Return whether `tick` falls inside one of the pointing intervals.
    ///
    /// Type 2 intervals are disjoint and can leave gaps. The segment holds no
    /// pointing in a gap.
    pub(in crate::ck) fn has_data_at(&self, tick: f64) -> bool {
        let idx = self.time_starts().partition_point(|&x| x <= tick);
        idx > 0 && tick <= self.time_stops()[idx - 1]
    }

    fn time_starts(&self) -> &[f64] {
        unsafe {
            self.array
                .daf
                .data
                .get_unchecked(self.time_start_idx..self.time_start_idx + self.n_records)
        }
    }

    pub(crate) fn try_get_orientation(
        &self,
        time: Time<TDB>,
        tick: f64,
    ) -> KeteResult<(Time<TDB>, NonInertialFrame)> {
        // get the time of the last record and its index
        let time_starts = self.time_starts();
        let (record_time, record_idx) = if self.n_records == 1 {
            // If there is only one interval, return its times
            (self.time_starts()[0], 0)
        } else {
            // The interval to use is the last one that starts at or before the
            // tick.
            let interval_idx = time_starts.partition_point(|&x| x <= tick);
            if interval_idx == 0 {
                // The tick is before the first interval. The dt check below
                // rejects it.
                (time_starts[0], 0)
            } else {
                let idx = interval_idx - 1;
                (time_starts[idx], idx)
            }
        };
        let (quaternion, angular_velocity, seconds_per_tick) = self.get_record(record_idx);

        let dt = tick - record_time;

        if dt < 0.0 {
            return Err(Error::Bounds(format!(
                "Requested clock tick {tick} is before the start of the segment."
            )));
        }
        // The instrument turns about the angular velocity vector through the
        // angle |w| times the elapsed seconds. The vector is in the reference
        // frame, so the turn multiplies the C-matrix on the right.
        let elapsed = Vector3::from(angular_velocity) * (dt * seconds_per_tick);
        let c_matrix = Unit::from_quaternion(quaternion).to_rotation_matrix()
            * Rotation3::from_scaled_axis(-elapsed);

        let frame = instrument_frame(
            time,
            c_matrix,
            Some(angular_velocity),
            self.array.reference_frame_id,
        );
        Ok((time, frame))
    }

    /// Build the data of a CK type 2 array.
    ///
    /// Type 2 is discrete pointing with no interpolation. `records` is a flat
    /// slice of `n * 8` values, one record of 8 values per interval. Each
    /// record is `[q0, q1, q2, q3, av1, av2, av3, seconds_per_tick]`.
    /// `start_times` and `stop_times` hold the `n` SCLK start and stop times of
    /// the intervals.
    ///
    /// # Errors
    /// [`Error::ValueError`] if `start_times` is empty, if the length of
    /// `records` is not `n * 8`, or if the length of `stop_times` is not `n`.
    fn build_data(
        records: &[f64],
        start_times: &[f64],
        stop_times: &[f64],
    ) -> KeteResult<Vec<f64>> {
        let n = start_times.len();
        if n == 0 {
            return Err(Error::ValueError(
                "CK Type 2: need at least one record.".into(),
            ));
        }
        if records.len() != n * 8 {
            return Err(Error::ValueError(format!(
                "CK Type 2: records length ({}) must be n ({}) * 8",
                records.len(),
                n
            )));
        }
        if stop_times.len() != n {
            return Err(Error::ValueError(
                "CK Type 2: stop_times length must match start_times.".into(),
            ));
        }

        // Layout: [8n pointing records][n start_times][n stop_times][directory]
        // Directory: one entry per 100 start times
        let dir_size = if n > 100 { (n - 1) / 100 } else { 0 };
        let mut data = Vec::with_capacity(10 * n + dir_size);

        data.extend_from_slice(records);
        data.extend_from_slice(start_times);
        data.extend_from_slice(stop_times);
        for i in 1..=dir_size {
            data.push(start_times[(i * 100 - 1).min(n - 1)]);
        }

        Ok(data)
    }

    /// Create a Type 2 (discrete pointing, no interpolation) CK array.
    ///
    /// # Arguments
    /// * `instrument_id`      - NAIF instrument ID.
    /// * `reference_frame_id` - Reference frame ID.
    /// * `records`            - Flat slice of `n * 8` pointing values.
    /// * `start_times`        - n SCLK interval start times.
    /// * `stop_times`         - n SCLK interval stop times.
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
        start_times: &[f64],
        stop_times: &[f64],
        segment_name: &str,
    ) -> KeteResult<CkArray> {
        let data = Self::build_data(records, start_times, stop_times)?;
        let tick_start = start_times[0];
        let tick_end = *stop_times.last().unwrap();
        Ok(CkArray::new(
            instrument_id,
            reference_frame_id,
            2,
            true,
            tick_start,
            tick_end,
            data,
            segment_name.to_string(),
        ))
    }
}

impl TryFrom<CkArray> for CkSegmentType2 {
    type Error = Error;

    fn try_from(array: CkArray) -> Result<Self, Self::Error> {
        // Each interval has a record of 8 values, a start time and a stop time.
        // A directory holds every 100th start time. The array therefore holds
        // 10 n + (n - 1) / 100 values for n records.
        let array_len = array.daf.len();
        let layout = |n: usize| 10 * n + n.saturating_sub(1) / 100;
        let n_records = (array_len.saturating_sub(array_len / 1000 + 1) / 10..=array_len / 10)
            .find(|&n| layout(n) == array_len)
            .unwrap_or(0);
        let dir_size = n_records.saturating_sub(1) / 100;

        if array_len != (n_records * 10 + dir_size) {
            return Err(Error::Bounds(
                "CK File is not formatted correctly, directory size of segments appear incorrect."
                    .into(),
            ));
        }
        if n_records == 0 {
            return Err(Error::Bounds(
                "CK File does not contain any records.".into(),
            ));
        }

        let time_start_idx = n_records * 8;

        Ok(Self {
            array,
            n_records,
            time_start_idx,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ck_type2_basic() {
        // 2 pointing records, each 8 values
        let records = vec![
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.01, 0.707, 0.707, 0.0, 0.0, 0.0, 0.1, 0.0, 0.02,
        ];
        let start_times = vec![100.0, 200.0];
        let stop_times = vec![200.0, 300.0];
        let data = CkSegmentType2::build_data(&records, &start_times, &stop_times).unwrap();
        // 16 + 2 + 2 + 0 dir = 20
        assert_eq!(data.len(), 20);
    }

    #[test]
    fn ck_type2_round_trip() {
        // Build -> TryFrom round-trip to verify time_start_idx correctness
        let records = vec![
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.01, 0.707, 0.707, 0.0, 0.0, 0.0, 0.1, 0.0, 0.02,
        ];
        let start_times = vec![100.0, 200.0];
        let stop_times = vec![200.0, 300.0];

        let array =
            CkSegmentType2::new_array(-12345, 1, &records, &start_times, &stop_times, "test")
                .unwrap();
        let seg = CkSegmentType2::try_from(array).unwrap();

        assert_eq!(seg.n_records, 2);
        assert_eq!(seg.time_starts(), &[100.0, 200.0]);
    }
}
