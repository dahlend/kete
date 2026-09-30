//! SPK Segment Type 3 - Chebyshev Polynomials (Position & Velocity).
//!
//! <https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/FORTRAN/req/spk.html#Type%203:%20Chebyshev%20position%20and%20velocity>

use super::SpkArray;
use crate::interpolation::{ChebyshevLayout, chebyshev_evaluate};
use kete_core::constants::AU_KM;
use kete_core::errors::Error;
use kete_core::prelude::KeteResult;

/// Type 3 - Chebyshev Polynomials (Position & Velocity)
///
/// <https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/FORTRAN/req/spk.html#Type%203:%20Chebyshev%20position%20and%20velocity>
///
#[derive(Debug)]
pub struct SpkSegmentType3 {
    pub(crate) array: SpkArray,
    layout: ChebyshevLayout,
}

/// Type 3 Record View
/// A view into a record of type 3, provided mainly for clarity to the underlying
/// data structure.
struct Type3RecordView<'a> {
    t_mid: &'a f64,
    t_step: &'a f64,

    x_coef: &'a [f64],
    y_coef: &'a [f64],
    z_coef: &'a [f64],

    vx_coef: &'a [f64],
    vy_coef: &'a [f64],
    vz_coef: &'a [f64],
}

impl SpkSegmentType3 {
    #[inline(always)]
    fn get_record(&self, idx: usize) -> Type3RecordView<'_> {
        let record_len = self.layout.record_len;
        let n_coef = self.layout.n_coef;
        // SAFETY: `ChebyshevLayout::from_array` checked that the array holds
        // `n_records` records of `record_len` values, with
        // `record_len = 6 * n_coef + 2`. The only caller takes `idx` from
        // `record_index`, which returns at most `n_records - 1`.
        unsafe {
            let vals = self
                .array
                .daf
                .data
                .get_unchecked(idx * record_len..(idx + 1) * record_len);

            Type3RecordView {
                t_mid: vals.get_unchecked(0),
                t_step: vals.get_unchecked(1),
                x_coef: vals.get_unchecked(2..(n_coef + 2)),
                y_coef: vals.get_unchecked((n_coef + 2)..(2 * n_coef + 2)),
                z_coef: vals.get_unchecked((2 * n_coef + 2)..(3 * n_coef + 2)),
                vx_coef: vals.get_unchecked((3 * n_coef + 2)..(4 * n_coef + 2)),
                vy_coef: vals.get_unchecked((4 * n_coef + 2)..(5 * n_coef + 2)),
                vz_coef: vals.get_unchecked((5 * n_coef + 2)..(6 * n_coef + 2)),
            }
        }
    }

    #[inline(always)]
    pub(crate) fn try_get_pos_vel(&self, jds: f64) -> KeteResult<([f64; 3], [f64; 3])> {
        let record_index = self.layout.record_index(jds);
        let record = self.get_record(record_index);

        let t_step = record.t_step;

        let t = (jds - record.t_mid) / t_step;

        let t_scaled = 86400.0 / AU_KM;

        let p = chebyshev_evaluate(t, record.x_coef, record.y_coef, record.z_coef)?;
        let v = chebyshev_evaluate(t, record.vx_coef, record.vy_coef, record.vz_coef)?;
        Ok((
            [p[0] / AU_KM, p[1] / AU_KM, p[2] / AU_KM],
            [v[0] * t_scaled, v[1] * t_scaled, v[2] * t_scaled],
        ))
    }

    /// Create a Type 3 (Chebyshev position and velocity) SPK array.
    ///
    /// `object_id` is the NAIF ID of the body, and `center_id` is the NAIF ID
    /// of the center body. `frame_id` is the NAIF frame ID. `cdata` holds the
    /// flat Chebyshev coefficients, `6 * (polydg + 1)` values per record.
    /// `n_records` is the number of records. `btime` is the start of the first
    /// interval, in TDB seconds from J2000. `intlen` is the fixed length of
    /// each interval, in seconds. `polydg` is the polynomial degree.
    /// `jds_start` and `jds_end` are the segment start and end, in TDB seconds
    /// from J2000. `segment_name` is the name stored in the DAF name record,
    /// which holds at most 40 characters.
    ///
    /// # Errors
    /// Returns [`Error::ValueError`] in these cases:
    /// - `polydg` is greater than 27.
    /// - `intlen` is zero or negative.
    /// - The length of `cdata` is not `6 * (polydg + 1) * n_records`.
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
        if polydg > 27 {
            return Err(Error::ValueError(
                "Type 3: polydg must be in [0, 27].".into(),
            ));
        }
        if intlen <= 0.0 {
            return Err(Error::ValueError("Type 3: intlen must be positive.".into()));
        }
        let ninrec = (polydg + 1) * 6;
        if cdata.len() != ninrec * n_records {
            return Err(Error::ValueError(format!(
                "Type 3: cdata length {} != ninrec({}) * n({})",
                cdata.len(),
                ninrec,
                n_records
            )));
        }

        let rsize = (ninrec + 2) as f64;
        let radius = intlen / 2.0;
        let mut data = Vec::with_capacity(n_records * (ninrec + 2) + 4);
        for i in 0..n_records {
            let mid = btime + radius + (i as f64) * intlen;
            data.push(mid);
            data.push(radius);
            data.extend_from_slice(&cdata[i * ninrec..(i + 1) * ninrec]);
        }
        data.push(btime);
        data.push(intlen);
        data.push(rsize);
        data.push(n_records as f64);

        Ok(SpkArray::new(
            object_id,
            center_id,
            frame_id,
            3,
            jds_start,
            jds_end,
            data,
            segment_name.to_string(),
        ))
    }
}

impl TryFrom<SpkArray> for SpkSegmentType3 {
    type Error = Error;
    fn try_from(array: SpkArray) -> KeteResult<Self> {
        let layout = ChebyshevLayout::from_array(&array.daf, 6)?;
        Ok(Self { array, layout })
    }
}
