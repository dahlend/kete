//! SPK Segment Type 2 - Chebyshev Polynomials (Position Only).
//!
//! <https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/spk.html#Type%202:%20Chebyshev%20position%20only>

use super::SpkArray;
use crate::interpolation::chebyshev_evaluate_both;
use kete_core::constants::AU_KM;
use kete_core::errors::Error;
use kete_core::prelude::KeteResult;
use kete_core::time::{TDB, Time};

/// Chebyshev Polynomials (Position Only)
///
/// <https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/spk.html#Type%202:%20Chebyshev%20position%20only>
///
#[derive(Debug)]
pub struct SpkSegmentType2 {
    pub(crate) array: SpkArray,
    jds_step: f64,
    n_coef: usize,
    record_len: usize,
}

/// Type 2 Record View
/// A view into a record of type 2, provided mainly for clarity to the underlying
/// data structure.
struct Type2RecordView<'a> {
    t_mid: &'a f64,
    t_step: &'a f64,

    x_coef: &'a [f64],
    y_coef: &'a [f64],
    z_coef: &'a [f64],
}

impl SpkSegmentType2 {
    #[inline(always)]
    fn get_record(&self, idx: usize) -> Type2RecordView<'_> {
        unsafe {
            let vals = self
                .array
                .daf
                .data
                .get_unchecked(idx * self.record_len..(idx + 1) * self.record_len);

            Type2RecordView {
                t_mid: vals.get_unchecked(0),
                t_step: vals.get_unchecked(1),
                x_coef: vals.get_unchecked(2..(self.n_coef + 2)),
                y_coef: vals.get_unchecked((self.n_coef + 2)..(2 * self.n_coef + 2)),
                z_coef: vals.get_unchecked((2 * self.n_coef + 2)..(3 * self.n_coef + 2)),
            }
        }
    }

    #[inline(always)]
    pub(crate) fn try_get_pos_vel(&self, jds: f64) -> KeteResult<([f64; 3], [f64; 3])> {
        let jds_start = self.array.jds_start;

        #[allow(
            clippy::cast_sign_loss,
            reason = "This is correct as long as the file is correct."
        )]
        let record_index = ((jds - jds_start) / self.jds_step).floor() as usize;
        let record = self.get_record(record_index);

        let t_step = record.t_step;

        let t = (jds - record.t_mid) / t_step;

        let t_step_scaled = 86400.0 / t_step / AU_KM;

        let (p, v) = chebyshev_evaluate_both(t, record.x_coef, record.y_coef, record.z_coef)?;
        Ok((
            [p[0] / AU_KM, p[1] / AU_KM, p[2] / AU_KM],
            [
                v[0] * t_step_scaled,
                v[1] * t_step_scaled,
                v[2] * t_step_scaled,
            ],
        ))
    }

    /// Create a Type 2 (Chebyshev position only, fixed intervals) SPK array.
    ///
    /// # Arguments
    /// * `object_id`    - NAIF ID of the body.
    /// * `center_id`    - NAIF ID of the center body.
    /// * `frame_id`     - NAIF frame ID.
    /// * `cdata`        - Flat Chebyshev coefficients, `(polydg+1)*3` values per record.
    /// * `n_records`    - Number of records.
    /// * `btime`        - Begin time of first interval (SPICE seconds from J2000).
    /// * `intlen`       - Length of each interval (seconds). Must be > 0.
    /// * `polydg`       - Polynomial degree, in `[0, 27]`.
    /// * `jd_start`     - Segment start epoch.
    /// * `jd_end`       - Segment end epoch.
    /// * `segment_name` - Name stored in the DAF name record (max 40 chars).
    ///
    /// # Errors
    /// Returns an error if `polydg` is outside `[0, 27]` or `cdata` length is
    /// inconsistent with `n_records` and `polydg`.
    pub fn new_array(
        object_id: i32,
        center_id: i32,
        frame_id: i32,
        cdata: &[f64],
        n_records: usize,
        btime: f64,
        intlen: f64,
        polydg: usize,
        jd_start: Time<TDB>,
        jd_end: Time<TDB>,
        segment_name: &str,
    ) -> KeteResult<SpkArray> {
        let data = build_type2_data(cdata, n_records, btime, intlen, polydg)?;
        Ok(SpkArray::new(
            object_id,
            center_id,
            frame_id,
            2,
            jd_start,
            jd_end,
            data,
            segment_name.to_string(),
        ))
    }
}

impl TryFrom<SpkArray> for SpkSegmentType2 {
    type Error = Error;

    fn try_from(array: SpkArray) -> Result<Self, Self::Error> {
        #[allow(
            clippy::cast_sign_loss,
            reason = "This is correct as long as the file is correct."
        )]
        let record_len = array.daf[array.daf.len() - 2] as usize;
        let jds_step = array.daf[array.daf.len() - 3];

        let n_coef = (record_len - 2) / 3;

        if 3 * n_coef + 2 != record_len {
            return Err(Error::ValueError("File incorrectly formatted, found number of Chebyshev coefficients doesn't match expected".into()));
        }

        Ok(Self {
            array,
            jds_step,
            n_coef,
            record_len,
        })
    }
}

/// Build an SPK Type 2 data array (Chebyshev position only, fixed intervals).
///
/// # Arguments
/// * `cdata`  - Flat Chebyshev coefficients. Per record: `(polydg+1)*3` values
///   arranged as `[X_0..X_polydg, Y_0..Y_polydg, Z_0..Z_polydg]`.
/// * `n`      - Number of records.
/// * `btime`  - Begin time of first interval (SPICE seconds from J2000).
/// * `intlen` - Length of each interval (seconds). Must be > 0.
/// * `polydg` - Polynomial degree, in `[0, 27]`.
///
/// # Errors
/// Returns an error if the degree is out of range, interval length is
/// non-positive, or the coefficient data length is inconsistent.
pub(crate) fn build_type2_data(
    cdata: &[f64],
    n: usize,
    btime: f64,
    intlen: f64,
    polydg: usize,
) -> KeteResult<Vec<f64>> {
    if polydg > 27 {
        return Err(Error::ValueError(
            "Type 2: polydg must be in [0, 27].".into(),
        ));
    }
    if intlen <= 0.0 {
        return Err(Error::ValueError("Type 2: intlen must be positive.".into()));
    }
    let ninrec = (polydg + 1) * 3;
    if cdata.len() != ninrec * n {
        return Err(Error::ValueError(format!(
            "Type 2: cdata length {} != ninrec({}) * n({})",
            cdata.len(),
            ninrec,
            n
        )));
    }

    let rsize = (ninrec + 2) as f64;
    let radius = intlen / 2.0;
    // Layout: [n records of (mid, radius, coeffs)] [btime, intlen, rsize, n]
    let mut data = Vec::with_capacity(n * (ninrec + 2) + 4);

    for i in 0..n {
        let mid = btime + radius + (i as f64) * intlen;
        data.push(mid);
        data.push(radius);
        data.extend_from_slice(&cdata[i * ninrec..(i + 1) * ninrec]);
    }
    data.push(btime);
    data.push(intlen);
    data.push(rsize);
    data.push(n as f64);

    Ok(data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn type2_basic() {
        // polydg=1 -> ninrec = 2*3 = 6 coeffs per record
        let cdata = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 1 record
        let data = build_type2_data(&cdata, 1, 0.0, 100.0, 1).unwrap();
        // 1 record of (mid, radius, 6 coeffs) + 4 trailer = 12
        assert_eq!(data.len(), (6 + 2) + 4);
        // mid = 0 + 50 = 50, radius = 50
        assert_eq!(data[0], 50.0);
        assert_eq!(data[1], 50.0);
        // coeffs
        assert_eq!(&data[2..8], &cdata[..]);
        // trailer: btime, intlen, rsize, n
        assert_eq!(data[8], 0.0);
        assert_eq!(data[9], 100.0);
        assert_eq!(data[10], 8.0); // ninrec + 2 = 6 + 2
        assert_eq!(data[11], 1.0);
    }

    #[test]
    fn type2_validation() {
        assert!(build_type2_data(&[], 1, 0.0, 100.0, 1).is_err()); // wrong cdata len
        assert!(build_type2_data(&[0.0; 6], 1, 0.0, -1.0, 1).is_err()); // negative intlen
        assert!(build_type2_data(&[0.0; 6], 1, 0.0, 100.0, 28).is_err()); // polydg > 27
    }
}
