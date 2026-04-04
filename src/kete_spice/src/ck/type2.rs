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
use nalgebra::{Quaternion, Rotation3, Unit};

/// Discrete pointing data.
///
/// This segment type is broken up into intervals, during each interval the
/// rotation rate is constant. Each interval has a defined orientation saved
/// as a quaternion, and then a vector defining the axis of rotation, then
/// the last value is the angular rate of rotation in SCLK ticks per second.
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
            let accel: [f64; 3] = rec[4..7].try_into().unwrap();
            let angular_rate = rec[7];
            (quaternion, accel, angular_rate)
        }
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
    ) -> KeteResult<(Time<TDB>, NonInertialFrame)> {
        let sclk = LOADED_SCLK
            .try_read()
            .map_err(|_| Error::Bounds("Failed to read SCLK data.".into()))?;
        let tick = sclk.try_time_to_tick(self.array.naif_id, time)?;

        // get the time of the last record and its index
        let time_starts = self.time_starts();
        let (record_time, record_idx) = if self.n_records == 1 {
            // If there is only one interval, return its times
            (self.time_starts()[0], 0)
        } else {
            let interval_idx = time_starts.partition_point(|&x| x <= tick);
            if interval_idx >= self.n_records - 1 {
                // If the index is the last one, return the last record
                (time_starts[self.n_records - 1], self.n_records - 1)
            } else if interval_idx == 0 {
                // If the index is before the beginning of the interval, return the first record
                (time_starts[0], 0)
            } else {
                // Otherwise, we have a valid index
                let idx = interval_idx - 1;
                (time_starts[idx], idx)
            }
        };
        let (quaternion, mut accel_vec, rate) = self.get_record(record_idx);

        let dt = tick - record_time;

        if dt < 0.0 {
            return Err(Error::Bounds(format!(
                "Requested time {record_idx} is before the start of the segment."
            )));
        }
        let mut rotation = Unit::from_quaternion(quaternion).to_rotation_matrix();

        for x in &mut accel_vec {
            *x *= 86400.0 * dt * rate;
        }
        let rates = Rotation3::from_scaled_axis(accel_vec.into());
        rotation *= rates;

        let frame = NonInertialFrame::from_rotations(
            time,
            rotation.inverse(),
            None,
            self.array.reference_frame_id,
            self.array.instrument_id,
        );
        Ok((time, frame))
    }

    /// Build a CK Type 2 data array (discrete pointing, no interpolation).
    ///
    /// Each pointing record contains 8 values:
    /// `[q0, q1, q2, q3, av1, av2, av3, angular_rate]`
    ///
    /// # Arguments
    /// * `records`     - Flat slice of `n * 8` pointing record values.
    /// * `start_times` - n SCLK start times for each interval.
    /// * `stop_times`  - n SCLK stop times for each interval.
    ///
    /// # Errors
    /// Returns an error if there are no records, the records length is not `n * 8`,
    /// or start and stop times lengths differ.
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
        // each pointing record is 8 numbers long, along with a start and stop time
        // and a directory of every 100th time.
        let array_len = array.daf.len();
        let mut n_records = array.daf.len() / 10;
        let mut dir_size = if n_records > 0 {
            (n_records - 1) / 100
        } else {
            0
        };

        // n_records will be an over estimate, as it is also counting the directory
        n_records -= ((n_records * 10 + dir_size) - array_len) / 10;
        dir_size = if n_records > 0 {
            (n_records - 1) / 100
        } else {
            0
        };
        // probably dont need the second time, but better safe than sorry
        n_records -= ((n_records * 10 + dir_size) - array_len) / 10;
        dir_size = if n_records > 0 {
            (n_records - 1) / 100
        } else {
            0
        };

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
        // Build → TryFrom round-trip to verify time_start_idx correctness
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
