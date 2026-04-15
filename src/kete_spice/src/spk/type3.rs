//! SPK Segment Type 3 - Chebyshev Polynomials (Position & Velocity).
//!
//! <https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/FORTRAN/req/spk.html#Type%203:%20Chebyshev%20position%20and%20velocity>

use super::SpkArray;
use crate::interpolation::chebyshev_evaluate;
use kete_core::constants::AU_KM;
use kete_core::errors::Error;
use kete_core::prelude::KeteResult;
use kete_core::time::{TDB, Time};

/// Type 3 - Chebyshev Polynomials (Position & Velocity)
///
/// <https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/FORTRAN/req/spk.html#Type%203:%20Chebyshev%20position%20and%20velocity>
///
#[derive(Debug)]
pub struct SpkSegmentType3 {
    pub(crate) array: SpkArray,
    jds_step: f64,
    n_coef: usize,
    n_records: usize,
    record_len: usize,
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
        unsafe {
            let vals = self
                .array
                .daf
                .data
                .get_unchecked(idx * self.record_len..(idx + 1) * self.record_len);

            Type3RecordView {
                t_mid: vals.get_unchecked(0),
                t_step: vals.get_unchecked(1),
                x_coef: vals.get_unchecked(2..(self.n_coef + 2)),
                y_coef: vals.get_unchecked((self.n_coef + 2)..(2 * self.n_coef + 2)),
                z_coef: vals.get_unchecked((2 * self.n_coef + 2)..(3 * self.n_coef + 2)),
                vx_coef: vals.get_unchecked((3 * self.n_coef + 2)..(4 * self.n_coef + 2)),
                vy_coef: vals.get_unchecked((4 * self.n_coef + 2)..(5 * self.n_coef + 2)),
                vz_coef: vals.get_unchecked((5 * self.n_coef + 2)..(6 * self.n_coef + 2)),
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
        // Clamp to the last record when jds lands exactly on the segment end boundary.
        let record_index =
            (((jds - jds_start) / self.jds_step).floor() as usize).min(self.n_records - 1);
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

    /// Create a Type 3 (Chebyshev position and velocity, fixed intervals) SPK array.
    ///
    /// # Arguments
    /// * `object_id`    - NAIF ID of the body.
    /// * `center_id`    - NAIF ID of the center body.
    /// * `frame_id`     - NAIF frame ID.
    /// * `cdata`        - Flat Chebyshev coefficients, `(polydg+1)*6` values per record.
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
            jd_start,
            jd_end,
            data,
            segment_name.to_string(),
        ))
    }
}

impl TryFrom<SpkArray> for SpkSegmentType3 {
    type Error = Error;
    fn try_from(array: SpkArray) -> KeteResult<Self> {
        #[allow(
            clippy::cast_sign_loss,
            reason = "This is correct as long as the file is correct."
        )]
        let record_len = array.daf[array.daf.len() - 2] as usize;
        let jds_step = array.daf[array.daf.len() - 3];
        #[allow(
            clippy::cast_sign_loss,
            reason = "This is correct as long as the file is correct."
        )]
        let n_records = array.daf[array.daf.len() - 1] as usize;

        let n_coef = (record_len - 2) / 6;

        if 6 * n_coef + 2 != record_len {
            return Err(Error::ValueError("File incorrectly formatted, found number of Chebyshev coefficients doesn't match expected".into()));
        }

        Ok(Self {
            array,
            jds_step,
            n_coef,
            n_records,
            record_len,
        })
    }
}
