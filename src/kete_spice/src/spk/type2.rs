// SPDX-FileCopyrightText: 2026 Dar Dahlen
// SPDX-FileCopyrightText: 2025 California Institute of Technology
// SPDX-License-Identifier: BSD-3-Clause

//! SPK Segment Type 2 - Chebyshev Polynomials (Position Only).
//!
//! <https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/spk.html#Type%202:%20Chebyshev%20position%20only>

use super::SpkArray;
use crate::interpolation::{ChebyshevLayout, chebyshev_evaluate_both};
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
    layout: ChebyshevLayout,
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
        let record_len = self.layout.record_len;
        let n_coef = self.layout.n_coef;
        // SAFETY: `ChebyshevLayout::from_array` checked that the array holds
        // `n_records` records of `record_len` values, with
        // `record_len = 3 * n_coef + 2`. The only caller takes `idx` from
        // `record_index`, which returns at most `n_records - 1`.
        unsafe {
            let vals = self
                .array
                .daf
                .data
                .get_unchecked(idx * record_len..(idx + 1) * record_len);

            Type2RecordView {
                t_mid: vals.get_unchecked(0),
                t_step: vals.get_unchecked(1),
                x_coef: vals.get_unchecked(2..(n_coef + 2)),
                y_coef: vals.get_unchecked((n_coef + 2)..(2 * n_coef + 2)),
                z_coef: vals.get_unchecked((2 * n_coef + 2)..(3 * n_coef + 2)),
            }
        }
    }

    #[inline(always)]
    pub(crate) fn try_get_pos_vel(&self, time: Time<TDB>) -> KeteResult<([f64; 3], [f64; 3])> {
        let jds = time.j2000_seconds();
        let record_index = self.layout.record_index(jds);
        let record = self.get_record(record_index);

        let t_step = record.t_step;

        let t = time.j2000_seconds_minus(*record.t_mid) / t_step;

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
    /// `object_id` is the NAIF ID of the body, and `center_id` is the NAIF ID
    /// of the center body. `frame_id` is the NAIF frame ID. `cdata` holds the
    /// flat Chebyshev coefficients, `3 * (polydg + 1)` values per record.
    /// `n_records` is the number of records. `btime` is the start of the first
    /// interval, in TDB seconds from J2000. `intlen` is the length of each
    /// interval, in seconds. `polydg` is the polynomial degree. `jds_start` and
    /// `jds_end` are the segment start and end, in TDB seconds from J2000.
    /// `segment_name` is the name stored in the DAF name record, which holds at
    /// most 40 characters.
    ///
    /// # Errors
    /// Returns [`Error::ValueError`] in these cases:
    /// - `polydg` is greater than 27.
    /// - `intlen` is zero or negative.
    /// - The length of `cdata` is not `3 * (polydg + 1) * n_records`.
    pub fn new_array(
        object_id: i32,
        center_id: i32,
        frame_id: i32,
        cdata: &[f64],
        n_records: usize,
        btime: f64,
        intlen: f64,
        polydg: usize,
        jds_start: f64,
        jds_end: f64,
        segment_name: &str,
    ) -> KeteResult<SpkArray> {
        let data = build_type2_data(cdata, n_records, btime, intlen, polydg)?;
        Ok(SpkArray::new(
            object_id,
            center_id,
            frame_id,
            2,
            jds_start,
            jds_end,
            data,
            segment_name.to_string(),
        ))
    }
}

impl TryFrom<SpkArray> for SpkSegmentType2 {
    type Error = Error;

    fn try_from(array: SpkArray) -> Result<Self, Self::Error> {
        let layout = ChebyshevLayout::from_array(&array.daf, 3)?;
        Ok(Self { array, layout })
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

    /// Build three degree 0 records that hold the constants 1, 2, and 3 km.
    fn three_constant_records(jds_start: f64, jds_end: f64) -> SpkSegmentType2 {
        let cdata = [1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0];
        let array = SpkSegmentType2::new_array(
            1000, 10, 1, &cdata, 3, 0.0, 100.0, 0, jds_start, jds_end, "test",
        )
        .unwrap();
        array.try_into().unwrap()
    }

    #[test]
    fn type2_record_located_from_init() {
        // The segment starts part way through the first record, as in a kernel
        // cut from a larger one. The reader must still locate records from
        // INIT.
        let seg = three_constant_records(50.0, 300.0);
        for (jds, expected) in [(50.0, 1.0), (99.0, 1.0), (120.0, 2.0), (199.0, 2.0)] {
            let (p, v) = seg.try_get_pos_vel(Time::from_j2000_seconds(jds)).unwrap();
            assert!(
                (p[0] * AU_KM - expected).abs() < 1e-9,
                "jds={jds}: {}",
                p[0]
            );
            assert_eq!(v, [0.0; 3]);
        }
        // A time exactly at the end of the last record uses the last record.
        let (p, _) = seg
            .try_get_pos_vel(Time::from_j2000_seconds(300.0))
            .unwrap();
        assert!((p[0] * AU_KM - 3.0).abs() < 1e-9);
    }

    #[test]
    fn type2_rejects_inconsistent_layout() {
        let seg = three_constant_records(0.0, 300.0);
        let mut data = seg.array.daf.data.to_vec();
        let n = data.len();
        let make = |data: Vec<f64>| {
            SpkSegmentType2::try_from(SpkArray::new(
                1000,
                10,
                1,
                2,
                0.0,
                300.0,
                data,
                "bad".into(),
            ))
        };
        data[n - 1] = 0.0;
        assert!(make(data.clone()).is_err());
        data[n - 1] = 4.0;
        assert!(make(data.clone()).is_err());
        data[n - 1] = 3.0;
        data[n - 2] = 1.0;
        assert!(make(data).is_err());
        assert!(make(vec![0.0; 3]).is_err());
    }

    #[test]
    fn type2_validation() {
        assert!(build_type2_data(&[], 1, 0.0, 100.0, 1).is_err()); // wrong cdata len
        assert!(build_type2_data(&[0.0; 6], 1, 0.0, -1.0, 1).is_err()); // negative intlen
        assert!(build_type2_data(&[0.0; 6], 1, 0.0, 100.0, 28).is_err()); // polydg > 27
    }
}
