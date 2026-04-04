//! SPK Segment Type 21 - Extended Modified Difference Arrays.
//!
//! Type 21 is the variable-coefficient generalisation of Type 1, supporting
//! arbitrary numbers of coefficients per record.
//!
//! <https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/spk.html#Type%2021:%20Extended%20Modified%20Difference%20Arrays>

use super::SpkArray;
use kete_core::constants::AU_KM;
use kete_core::errors::Error;
use kete_core::prelude::KeteResult;
use kete_core::time::{TDB, Time};

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
    /// Create a Type 21 (Extended Modified Difference Arrays) SPK array from raw records.
    ///
    /// # Arguments
    /// * `object_id`    - NAIF ID of the body.
    /// * `center_id`    - NAIF ID of the center body.
    /// * `frame_id`     - NAIF frame ID.
    /// * `records`      - Flat slice of `n * (4*n_coef+11)` pre-computed values.
    /// * `epochs`       - n epoch values (SPICE seconds from J2000).
    /// * `n_coef`       - Number of coefficients per component.
    /// * `jd_start`     - Segment start epoch.
    /// * `jd_end`       - Segment end epoch.
    /// * `segment_name` - Name stored in the DAF name record (max 40 chars).
    ///
    /// # Errors
    /// Returns an error if `records` length is not `n * (4 * n_coef + 11)` where
    /// `n = epochs.len()`.
    pub fn new_array(
        object_id: i32,
        center_id: i32,
        frame_id: i32,
        records: &[f64],
        epochs: &[f64],
        n_coef: usize,
        jd_start: Time<TDB>,
        jd_end: Time<TDB>,
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
        // Layout: [n*record_len records][n epochs][n_coef][n_records]
        let mut data = Vec::with_capacity(n * record_len + n + 2);
        data.extend_from_slice(records);
        data.extend_from_slice(epochs);
        data.push(n_coef as f64);
        data.push(n as f64);

        Ok(SpkArray::new(
            object_id,
            center_id,
            frame_id,
            21,
            jd_start,
            jd_end,
            data,
            segment_name.to_string(),
        ))
    }

    #[inline(always)]
    fn get_record(&self, idx: usize) -> &[f64] {
        unsafe {
            self.array
                .daf
                .data
                .get_unchecked(idx * self.record_len..(idx + 1) * self.record_len)
        }
    }

    #[inline(always)]
    fn get_times(&self) -> &[f64] {
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
    pub(crate) fn try_get_pos_vel(&self, jds: f64) -> KeteResult<([f64; 3], [f64; 3])> {
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

        let start_idx = self
            .get_times()
            .binary_search_by(|probe| probe.total_cmp(&jds))
            .unwrap_or_else(|c| c);

        let record = self.get_record(start_idx);

        let ref_time = record[0];

        let func_vec = &record[1..=self.n_coef];
        let ref_state = &record[self.n_coef + 1..self.n_coef + 7];

        let divided_diff_array = &record[self.n_coef + 7..4 * self.n_coef + 7];

        let kq_max1 = record[4 * self.n_coef + 7] as usize;
        let kq = &record[4 * self.n_coef + 8..4 * self.n_coef + 11];

        // in the spice code ref_time is in seconds from j2000
        let dt = jds - ref_time;

        let mut fc = Vec::<f64>::with_capacity(self.n_coef);
        let mut wc = Vec::<f64>::with_capacity(self.n_coef);

        let mut tp = dt;
        for f in func_vec.iter().take(kq_max1 - 2) {
            if *f == 0.0 {
                // don't divide by 0 below, file was built incorrectly.
                return Err(Error::IOError(
                    "SPK File contains segments of type 21 has invalid contents.".into(),
                ));
            }

            fc.push(tp / f);
            wc.push(dt / f);
            tp = dt + f;
        }

        let mut w: Box<[f64]> = (0..kq_max1).map(|x| (x as f64 + 1.0).recip()).collect();

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
            let sum: f64 = (1..=(kq[idx] as usize))
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
            let sum: f64 = (1..=(kq[idx] as usize))
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
        reason = "This is correct as long as the file is correct."
    )]
    fn try_from(array: SpkArray) -> KeteResult<Self> {
        let n_records = array.daf[array.daf.len() - 1] as usize;
        let n_coef = array.daf[array.daf.len() - 2] as usize;
        let record_len = 4 * n_coef + 11;

        if array.daf.data.len() < n_records * record_len + n_records + 2 {
            return Err(Error::IOError(format!(
                "SPK Type 21: data length ({}) too short for {} records of length {}",
                array.daf.data.len(),
                n_records,
                record_len
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
