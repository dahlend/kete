// SPDX-FileCopyrightText: 2026 Dar Dahlen
// SPDX-FileCopyrightText: 2025 California Institute of Technology
// SPDX-License-Identifier: BSD-3-Clause

//! SPK Segment Type 21 - Extended Modified Difference Arrays.
//!
//! Type 21 is the variable-coefficient generalization of Type 1, supporting
//! arbitrary numbers of coefficients per record.
//!
//! <https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/spk.html#Type%2021:%20Extended%20Modified%20Difference%20Arrays>

use super::SpkArray;
use super::type1::difference_orders;
use kete_core::constants::AU_KM;
use kete_core::errors::Error;
use kete_core::prelude::KeteResult;
use kete_core::time::{TDB, Time};

/// Largest number of difference coefficients per component.
///
/// This is the MAXTRM limit of the SPICE type 21 format.
const MAX_DIM: usize = 25;

/// Extended Modified Difference Arrays
///
/// <https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/spk.html#Type%2021:%20Extended%20Modified%20Difference%20Arrays>
///
#[derive(Debug)]
pub struct SpkSegmentType21 {
    pub(crate) array: SpkArray,
    n_coef: usize,
    n_records: usize,
    record_len: usize,
}

impl SpkSegmentType21 {
    /// Create a Type 21 (Extended Modified Difference Arrays) SPK array.
    ///
    /// `object_id` is the NAIF ID of the body, and `center_id` is the NAIF ID
    /// of the center body. `frame_id` is the NAIF frame ID. `records` is a flat
    /// slice of `4 * n_coef + 11` difference-line values per record. `epochs`
    /// holds one epoch per record, in TDB seconds from J2000. `n_coef` is the
    /// number of difference coefficients per component. `jds_start` and
    /// `jds_end` are the segment start and end, in TDB seconds from J2000.
    /// `segment_name` is the name stored in the DAF name record, which holds at
    /// most 40 characters.
    ///
    /// The reader rejects a segment with an `n_coef` of 0 or greater than 25.
    /// This function does not check that limit.
    ///
    /// # Errors
    /// Returns [`Error::ValueError`] if `epochs` is empty, or if the length of
    /// `records` is not `(4 * n_coef + 11) * epochs.len()`.
    pub fn new_array(
        object_id: i32,
        center_id: i32,
        frame_id: i32,
        records: &[f64],
        epochs: &[f64],
        n_coef: usize,
        jds_start: f64,
        jds_end: f64,
        segment_name: &str,
    ) -> KeteResult<SpkArray> {
        let record_len = 4 * n_coef + 11;
        let n = epochs.len();
        if n == 0 {
            return Err(Error::ValueError(
                "Type 21: need at least one record.".into(),
            ));
        }
        if records.len() != n * record_len {
            return Err(Error::ValueError(format!(
                "Type 21: records length ({}) must be n ({}) * record_len ({})",
                records.len(),
                n,
                record_len
            )));
        }
        // Layout: [n*record_len records][n epochs][every 100th epoch]
        //         [n_coef][n_records]
        let mut data = Vec::with_capacity(n * record_len + n + n / 100 + 2);
        data.extend_from_slice(records);
        data.extend_from_slice(epochs);
        data.extend(epochs.iter().skip(99).step_by(100));
        data.push(n_coef as f64);
        data.push(n as f64);

        Ok(SpkArray::new(
            object_id,
            center_id,
            frame_id,
            21,
            jds_start,
            jds_end,
            data,
            segment_name.to_string(),
        ))
    }

    #[inline(always)]
    fn get_record(&self, idx: usize) -> &[f64] {
        // SAFETY: `try_from` checked that the array holds `n_records` records
        // of `record_len` values, followed by the epochs. The only caller
        // clamps `idx` to `n_records - 1`.
        unsafe {
            self.array
                .daf
                .data
                .get_unchecked(idx * self.record_len..(idx + 1) * self.record_len)
        }
    }

    #[inline(always)]
    fn get_times(&self) -> &[f64] {
        // SAFETY: `try_from` checked that the array holds at least
        // `n_records * (record_len + 1) + 2` values, so the epochs lie inside
        // it.
        unsafe {
            self.array.daf.data.get_unchecked(
                self.n_records * self.record_len..self.n_records * (self.record_len + 1),
            )
        }
    }

    #[inline(always)]
    #[allow(
        clippy::cast_sign_loss,
        reason = "This is correct as long as the file is correct."
    )]
    pub(crate) fn try_get_pos_vel(&self, time: Time<TDB>) -> KeteResult<([f64; 3], [f64; 3])> {
        let jds = time.j2000_seconds();
        // Records are laid out as so:
        //
        // Size      Description
        // ----------------------
        // 1          Reference Epoch for the difference line
        // n_coef     Step size function vector
        // 6          Reference state - x, vx, y, vy, z, vz  (interleaved order)
        // 3*n_coef   Modified divided difference arrays
        // 1          Maximum integration order plus 1
        // 3          Integration order array
        // total: 11 + 4*n_coef

        // we need to find the first record which has a time greater than or equal
        // to the target jd.

        // A time after the epoch of the last record uses the last record.
        let start_idx = self
            .get_times()
            .partition_point(|&t| t < jds)
            .min(self.n_records - 1);

        let record = self.get_record(start_idx);

        let ref_time = record[0];

        let func_vec = &record[1..=self.n_coef];
        let ref_state = &record[self.n_coef + 1..self.n_coef + 7];

        let divided_diff_array = &record[self.n_coef + 7..4 * self.n_coef + 7];

        let (kq_max1, kq) = difference_orders(
            record[4 * self.n_coef + 7],
            &record[4 * self.n_coef + 8..4 * self.n_coef + 11],
            self.n_coef,
        )?;

        // in the spice code ref_time is in seconds from j2000
        let dt = time.j2000_seconds_minus(ref_time);

        let mut fc = [0.0; MAX_DIM];
        let mut wc = [0.0; MAX_DIM];

        let mut tp = dt;
        for (idx, f) in func_vec.iter().take(kq_max1 - 2).enumerate() {
            if *f == 0.0 {
                // don't divide by 0 below, file was built incorrectly.
                return Err(Error::IOError(
                    "SPK File contains segments of type 21 has invalid contents.".into(),
                ));
            }

            fc[idx] = tp / f;
            wc[idx] = dt / f;
            tp = dt + f;
        }

        let mut w = [0.0; MAX_DIM + 2];
        for (idx, w) in w.iter_mut().take(kq_max1).enumerate() {
            *w = (idx as f64 + 1.0).recip();
        }

        let mut ks = kq_max1 - 1;
        let mut jx = 0;
        let mut ks1 = ks - 1;

        while ks >= 2 {
            jx += 1;
            for j in 0..jx {
                w[j + ks] = fc[j] * w[j + ks1] - wc[j] * w[j + ks];
            }
            ks = ks1;
            ks1 -= 1;
        }

        // position interpolation
        let pos = std::array::from_fn(|idx| {
            let sum: f64 = (1..=kq[idx])
                .rev()
                .map(|j| divided_diff_array[idx * self.n_coef + j - 1] * w[j + ks - 1])
                .sum();
            (ref_state[2 * idx] + dt * (sum * dt + ref_state[2 * idx + 1])) / AU_KM
        });

        // Recompute W for velocities
        for j in 0..jx {
            w[j + ks] = fc[j] * w[j + ks1] - wc[j] * w[j + ks];
        }
        ks -= 1;

        // velocity interpolation
        let vel = std::array::from_fn(|idx| {
            let sum: f64 = (1..=kq[idx])
                .rev()
                .map(|j| divided_diff_array[idx * self.n_coef + j - 1] * w[j + ks - 1])
                .sum();
            (ref_state[2 * idx + 1] + dt * sum) / AU_KM * 86400.0
        });

        Ok((pos, vel))
    }
}

impl TryFrom<SpkArray> for SpkSegmentType21 {
    type Error = Error;

    #[allow(
        clippy::cast_sign_loss,
        clippy::cast_possible_truncation,
        reason = "Values are checked to be positive whole numbers first."
    )]
    fn try_from(array: SpkArray) -> KeteResult<Self> {
        let len = array.daf.len();
        let (n_coef, n_records) = if len < 2 {
            (0.0, 0.0)
        } else {
            (array.daf[len - 2], array.daf[len - 1])
        };
        let count = |x: f64| x.is_finite() && x >= 1.0 && x.fract() == 0.0;
        if !(count(n_coef) && count(n_records)) {
            return Err(Error::IOError(format!(
                "SPK Type 21: invalid control words [{n_coef}, {n_records}]."
            )));
        }
        let (n_coef, n_records) = (n_coef as usize, n_records as usize);
        if n_coef > MAX_DIM {
            return Err(Error::IOError(format!(
                "SPK Type 21: {n_coef} difference coefficients exceeds the limit of {MAX_DIM}."
            )));
        }
        let record_len = 4 * n_coef + 11;

        if n_records
            .checked_mul(record_len + 1)
            .is_none_or(|x| x + 2 > len)
        {
            return Err(Error::IOError(format!(
                "SPK Type 21: data length ({len}) too short for {n_records} records of \
                 length {record_len}"
            )));
        }

        Ok(Self {
            array,
            n_coef,
            n_records,
            record_len,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const N_COEF: usize = 3;

    /// Build one difference line with no difference terms.
    ///
    /// SPKE21 evaluates this line as linear motion from its reference state.
    /// The state is x = 1000 km, moving at 1 km/s.
    fn linear_record(epoch: f64) -> Vec<f64> {
        let mut record = vec![0.0; 4 * N_COEF + 11];
        record[0] = epoch;
        record[1..=N_COEF].fill(1.0);
        record[N_COEF + 1] = 1000.0;
        record[N_COEF + 2] = 1.0;
        record[4 * N_COEF + 7] = 2.0;
        record
    }

    fn segment(epochs: &[f64], jds_end: f64) -> SpkSegmentType21 {
        let records: Vec<f64> = epochs.iter().flat_map(|&t| linear_record(t)).collect();
        SpkSegmentType21::new_array(
            1000,
            10,
            1,
            &records,
            epochs,
            N_COEF,
            epochs[0] - 100.0,
            jds_end,
            "type 21",
        )
        .unwrap()
        .try_into()
        .unwrap()
    }

    #[test]
    fn type21_evaluates_and_clamps_to_the_last_record() {
        let seg = segment(&[0.0, 100.0], 200.0);
        let (pos, vel) = seg.try_get_pos_vel(Time::from_j2000_seconds(90.0)).unwrap();
        assert!((pos[0] * AU_KM - 990.0).abs() < 1e-9);
        assert!((vel[0] * AU_KM / 86400.0 - 1.0).abs() < 1e-12);
        let (pos, _) = seg
            .try_get_pos_vel(Time::from_j2000_seconds(150.0))
            .unwrap();
        assert!((pos[0] * AU_KM - 1050.0).abs() < 1e-9);
    }

    /// Check that the writer repeats every 100th epoch in a directory, as
    /// SPKW21 does.
    #[test]
    fn type21_writes_the_epoch_directory() {
        let epochs: Vec<f64> = (0..250).map(f64::from).collect();
        let seg = segment(&epochs, 249.0);
        let data = &seg.array.daf.data;
        let record_len = 4 * N_COEF + 11;
        assert_eq!(data.len(), 250 * (record_len + 1) + 2 + 2);
        assert_eq!(data[250 * (record_len + 1)], 99.0);
        assert_eq!(data[250 * (record_len + 1) + 1], 199.0);
    }

    /// Check that the reader rejects more than 25 difference coefficients per
    /// component.
    #[test]
    fn type21_rejects_too_many_coefficients() {
        let seg = segment(&[0.0, 100.0], 100.0);
        let mut data = seg.array.daf.data.to_vec();
        let n = data.len();
        data[n - 2] = 2.0_f64.powi(62);
        let array = SpkArray::new(1000, 10, 1, 21, 0.0, 100.0, data, "bad".into());
        assert!(SpkSegmentType21::try_from(array).is_err());
    }
}
