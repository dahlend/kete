//! SPK Segment Type 1 - Modified Difference Arrays.
//!
//! Type 1 segments store pre-computed difference-line records as produced by
//! JPL orbit determination software. Each record is exactly 71 f64 values.
//!
//! <https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/spk.html#Type%201:%20Modified%20Difference%20Arrays>

use super::SpkArray;
use kete_core::constants::AU_KM;
use kete_core::errors::Error;
use kete_core::prelude::KeteResult;
use kete_core::time::{TDB, Time};

/// Modified Difference Arrays
///
/// <https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/spk.html#Type%201:%20Modified%20Difference%20Arrays>
///
// This format might be derived from works related to this paper:
// Recurrence Relations for Computing With Modified Divided Differences*
// Fred Krogh 1979
#[derive(Debug)]
pub struct SpkSegmentType1 {
    pub(crate) array: SpkArray,

    n_records: usize,
}

#[allow(
    clippy::cast_sign_loss,
    reason = "This is correct as long as the file is correct."
)]
impl SpkSegmentType1 {
    #[inline(always)]
    fn get_record(&self, idx: usize) -> &[f64] {
        unsafe { self.array.daf.data.get_unchecked(idx * 71..(idx + 1) * 71) }
    }

    #[inline(always)]
    fn get_times(&self) -> &[f64] {
        unsafe {
            self.array
                .daf
                .data
                .get_unchecked(self.n_records * 71..(self.n_records * 72))
        }
    }

    #[inline(always)]
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

        let func_vec = &record[1..16];
        let ref_state = &record[16..22];

        let divided_diff_array = &record[22..67];

        let kq_max1 = record[67] as usize;
        let kq = &record[68..71];

        // in the spice code ref_time is in seconds from j2000
        let dt = jds - ref_time;

        let mut fc = [0.0; 15];
        let mut wc = [0.0; 15];

        let mut tp = dt;
        for idx in 0..(kq_max1 - 2) {
            let f = func_vec[idx];
            if f == 0.0 {
                // don't divide by 0 below, file was built incorrectly.
                Err(Error::IOError(
                    "SPK File containing segments of type 1 has invalid contents.".into(),
                ))?;
            }

            fc[idx] = tp / f;
            wc[idx] = dt / f;
            tp = dt + f;
        }

        let mut w: Box<[f64]> = { (0..kq_max1).map(|x| (x as f64 + 1.0).recip()).collect() };

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
                .map(|j| divided_diff_array[15 * idx + j - 1] * w[j + ks - 1])
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
                .map(|j| divided_diff_array[15 * idx + j - 1] * w[j + ks - 1])
                .sum();
            (ref_state[2 * idx + 1] + dt * sum) / AU_KM * 86400.0
        });
        Ok((pos, vel))
    }

    /// Create a Type 1 (Modified Difference Arrays) SPK array from raw records.
    ///
    /// # Arguments
    /// * `object_id`    - NAIF ID of the body.
    /// * `center_id`    - NAIF ID of the center body.
    /// * `frame_id`     - NAIF frame ID.
    /// * `records`      - Flat slice of `n * 71` pre-computed difference-line values.
    /// * `epochs`       - n epoch values (SPICE seconds from J2000).
    /// * `jd_start`     - Segment start epoch.
    /// * `jd_end`       - Segment end epoch.
    /// * `segment_name` - Name stored in the DAF name record (max 40 chars).
    ///
    /// # Errors
    /// Returns an error if `records` length is not `n * 71` where `n = epochs.len()`.
    pub fn new_array(
        object_id: i32,
        center_id: i32,
        frame_id: i32,
        records: &[f64],
        epochs: &[f64],
        jd_start: Time<TDB>,
        jd_end: Time<TDB>,
        segment_name: &str,
    ) -> KeteResult<SpkArray> {
        let n = epochs.len();
        if n == 0 {
            return Err(Error::ValueError(
                "Type 1: need at least one record.".into(),
            ));
        }
        if records.len() != n * 71 {
            return Err(Error::ValueError(format!(
                "Type 1: records length ({}) must be n ({}) * 71",
                records.len(),
                n
            )));
        }
        // Layout: [n*71 records][n epochs][n_records]
        let mut data = Vec::with_capacity(72 * n + 1);
        data.extend_from_slice(records);
        data.extend_from_slice(epochs);
        data.push(n as f64);

        Ok(SpkArray::new(
            object_id,
            center_id,
            frame_id,
            1,
            jd_start,
            jd_end,
            data,
            segment_name.to_string(),
        ))
    }
}

impl TryFrom<SpkArray> for SpkSegmentType1 {
    type Error = Error;

    #[allow(
        clippy::cast_sign_loss,
        reason = "This is correct as long as the file is correct."
    )]
    fn try_from(array: SpkArray) -> KeteResult<Self> {
        let n_records = array.daf[array.daf.len() - 1] as usize;
        if array.daf.data.len() < n_records * 71 + n_records + 1 {
            return Err(Error::IOError(format!(
                "SPK Type 1: data length ({}) too short for {} records",
                array.daf.data.len(),
                n_records
            )));
        }
        Ok(Self { array, n_records })
    }
}
