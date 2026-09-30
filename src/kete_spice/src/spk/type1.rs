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
        // SAFETY: `try_from` checked that the array holds `n_records` records
        // of 71 values, followed by the epochs. The only caller clamps `idx` to
        // `n_records - 1`.
        unsafe { self.array.daf.data.get_unchecked(idx * 71..(idx + 1) * 71) }
    }

    #[inline(always)]
    fn get_times(&self) -> &[f64] {
        // SAFETY: `try_from` checked that the array holds at least
        // `72 * n_records + 1` values, so the epochs lie inside it.
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

        // A time after the epoch of the last record uses the last record.
        let start_idx = self
            .get_times()
            .partition_point(|&t| t < jds)
            .min(self.n_records - 1);

        let record = self.get_record(start_idx);

        let ref_time = record[0];

        let func_vec = &record[1..16];
        let ref_state = &record[16..22];

        let divided_diff_array = &record[22..67];

        let (kq_max1, kq) = difference_orders(record[67], &record[68..71], 15)?;

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

        let mut w = [0.0; 17];
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
            let sum: f64 = (1..=kq[idx])
                .rev()
                .map(|j| divided_diff_array[15 * idx + j - 1] * w[j + ks - 1])
                .sum();
            (ref_state[2 * idx + 1] + dt * sum) / AU_KM * 86400.0
        });
        Ok((pos, vel))
    }

    /// Create a Type 1 (Modified Difference Arrays) SPK array from raw records.
    ///
    /// `object_id` is the NAIF ID of the body, and `center_id` is the NAIF ID
    /// of the center body. `frame_id` is the NAIF frame ID. `records` is a flat
    /// slice of 71 difference-line values per record. `epochs` holds one epoch
    /// per record, in TDB seconds from J2000. `jds_start` and `jds_end` are the
    /// segment start and end, in TDB seconds from J2000. `segment_name` is the
    /// name stored in the DAF name record, which holds at most 40 characters.
    ///
    /// # Errors
    /// Returns [`Error::ValueError`] if `epochs` is empty, or if the length of
    /// `records` is not `71 * epochs.len()`.
    pub fn new_array(
        object_id: i32,
        center_id: i32,
        frame_id: i32,
        records: &[f64],
        epochs: &[f64],
        jds_start: f64,
        jds_end: f64,
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
        // Layout: [n*71 records][n epochs][every 100th epoch][n_records]
        let mut data = Vec::with_capacity(72 * n + n / 100 + 1);
        data.extend_from_slice(records);
        data.extend_from_slice(epochs);
        data.extend(epochs.iter().skip(99).step_by(100));
        data.push(n as f64);

        Ok(SpkArray::new(
            object_id,
            center_id,
            frame_id,
            1,
            jds_start,
            jds_end,
            data,
            segment_name.to_string(),
        ))
    }
}

impl TryFrom<SpkArray> for SpkSegmentType1 {
    type Error = Error;

    #[allow(
        clippy::cast_sign_loss,
        clippy::cast_possible_truncation,
        reason = "The count is checked to be a positive whole number first."
    )]
    fn try_from(array: SpkArray) -> KeteResult<Self> {
        let len = array.daf.len();
        let n_records = if len == 0 { 0.0 } else { array.daf[len - 1] };
        if !(n_records.is_finite() && n_records >= 1.0 && n_records.fract() == 0.0)
            || (n_records as usize)
                .checked_mul(72)
                .is_none_or(|x| x + 1 > len)
        {
            return Err(Error::IOError(format!(
                "SPK Type 1: data length ({len}) inconsistent with {n_records} records"
            )));
        }
        Ok(Self {
            array,
            n_records: n_records as usize,
        })
    }
}

/// Read and check the integration orders of a type 1 or 21 difference line.
///
/// `kq_max1` is the maximum integration order plus 1. `kq` holds the three
/// integration orders, one per component. `max_dim` is the number of difference
/// coefficients per component. The function returns the orders as integers.
///
/// # Errors
/// Returns [`Error::IOError`] in these cases:
/// - `kq_max1` is not a whole number in `[2, max_dim + 2]`.
/// - `kq` does not hold exactly 3 values.
/// - A value in `kq` is not a whole number in `[0, min(max_dim, kq_max1 - 1)]`.
#[allow(
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation,
    reason = "Values are checked to be whole numbers in range first."
)]
pub(in crate::spk) fn difference_orders(
    kq_max1: f64,
    kq: &[f64],
    max_dim: usize,
) -> KeteResult<(usize, [usize; 3])> {
    let whole = |x: f64, hi: usize| x.fract() == 0.0 && (0.0..=hi as f64).contains(&x);
    if !whole(kq_max1, max_dim + 2) || kq_max1 < 2.0 {
        return Err(Error::IOError(format!(
            "SPK difference line has invalid maximum order {kq_max1}."
        )));
    }
    let kq_max1 = kq_max1 as usize;
    if kq.len() != 3 || kq.iter().any(|&k| !whole(k, max_dim.min(kq_max1 - 1))) {
        return Err(Error::IOError(format!(
            "SPK difference line has invalid integration orders {kq:?}."
        )));
    }
    Ok((kq_max1, [kq[0] as usize, kq[1] as usize, kq[2] as usize]))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build one difference line with no difference terms.
    ///
    /// SPKE01 evaluates this line as linear motion from its reference state.
    /// The state is x = 1000 km, moving at 1 km/s.
    fn linear_record(epoch: f64) -> Vec<f64> {
        let mut record = vec![0.0; 71];
        record[0] = epoch;
        record[1..16].fill(1.0);
        record[16] = 1000.0;
        record[17] = 1.0;
        record[67] = 2.0;
        record
    }

    fn segment(epochs: &[f64], jds_end: f64) -> SpkSegmentType1 {
        let records: Vec<f64> = epochs.iter().flat_map(|&t| linear_record(t)).collect();
        SpkSegmentType1::new_array(
            1000,
            10,
            1,
            &records,
            epochs,
            epochs[0] - 100.0,
            jds_end,
            "type 1",
        )
        .unwrap()
        .try_into()
        .unwrap()
    }

    #[test]
    fn type1_evaluates_and_clamps_to_the_last_record() {
        let seg = segment(&[0.0, 100.0], 200.0);
        let (pos, vel) = seg.try_get_pos_vel(90.0).unwrap();
        assert!((pos[0] * AU_KM - 990.0).abs() < 1e-9);
        assert!((vel[0] * AU_KM / 86400.0 - 1.0).abs() < 1e-12);
        // A time past the last epoch uses the last record.
        let (pos, _) = seg.try_get_pos_vel(150.0).unwrap();
        assert!((pos[0] * AU_KM - 1050.0).abs() < 1e-9);
    }

    /// Check that the writer repeats every 100th epoch in a directory, as
    /// SPKW01 does.
    #[test]
    fn type1_writes_the_epoch_directory() {
        let epochs: Vec<f64> = (0..250).map(f64::from).collect();
        let seg = segment(&epochs, 249.0);
        let data = &seg.array.daf.data;
        assert_eq!(data.len(), 250 * 72 + 2 + 1);
        assert_eq!(data[250 * 72], 99.0);
        assert_eq!(data[250 * 72 + 1], 199.0);
    }

    #[test]
    fn type1_rejects_invalid_orders() {
        assert!(difference_orders(2.0, &[0.0, 0.0, 0.0], 15).is_ok());
        assert!(difference_orders(17.0, &[15.0, 15.0, 15.0], 15).is_ok());
        assert!(difference_orders(1.0, &[0.0, 0.0, 0.0], 15).is_err());
        assert!(difference_orders(18.0, &[0.0, 0.0, 0.0], 15).is_err());
        assert!(difference_orders(3.0, &[3.0, 0.0, 0.0], 15).is_err());
        assert!(difference_orders(3.0, &[f64::NAN, 0.0, 0.0], 15).is_err());
    }

    #[test]
    fn type1_rejects_inconsistent_trailer() {
        let seg = segment(&[0.0, 100.0], 100.0);
        let mut data = seg.array.daf.data.to_vec();
        let n = data.len();
        data[n - 1] = 3.0;
        let array = SpkArray::new(1000, 10, 1, 1, 0.0, 100.0, data, "bad".into());
        assert!(SpkSegmentType1::try_from(array).is_err());
    }
}
