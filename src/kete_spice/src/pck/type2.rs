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

use super::PckArray;
use crate::interpolation::chebyshev_evaluate_both;
use crate::spice_jd_to_jd;
use crate::spk::type2::build_type2_data;
use kete_core::errors::Error;
use kete_core::frames::NonInertialFrame;
use kete_core::prelude::KeteResult;
use kete_core::time::{TDB, Time};

/// Chebyshev polynomials (Euler angles only)
///
/// <https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/pck.html#Binary%20PCK%20Kernel>
///
#[derive(Debug)]
pub struct PckSegmentType2 {
    pub(in crate::pck) array: PckArray,
    jd_step: f64,
    n_coef: usize,
    record_len: usize,
}

impl PckSegmentType2 {
    fn get_record(&self, idx: usize) -> &[f64] {
        unsafe {
            self.array
                .daf
                .data
                .get_unchecked(idx * self.record_len..(idx + 1) * self.record_len)
        }
    }

    /// Return the stored orientation, along with the rate of change of the orientation.
    pub(in crate::pck) fn try_get_orientation(&self, jds: f64) -> KeteResult<NonInertialFrame> {
        // Records in the segment contain information about the central position of the
        // north pole, as well as the position of the prime meridian. These values for
        // type 2 segments are stored as chebyshev polynomials of the first kind, in
        // essentially the exact same format as the Type 2 SPK segments.
        // Records for this type are structured as so:
        // - time at midpoint of record.
        // - (length of time record is valid for) / 2.0
        // - N Chebyshev polynomial coefficients for ra
        // - N Chebyshev polynomial coefficients for dec
        // - N Chebyshev polynomial coefficients for w
        //
        // Rate of change for each of these values can be calculated by using the
        // derivative of chebyshev of the first kind, which is done below.
        let jds_start = self.array.jds_start;
        #[allow(
            clippy::cast_sign_loss,
            reason = "safe as long as file is correctly formatted."
        )]
        let record_index = ((jds - jds_start) / self.jd_step).floor() as usize;
        let record = self.get_record(record_index);
        let t_mid = record[0];
        let t_step = record[1];
        let t = (jds - t_mid) / t_step;

        let ra_coef = &record[2..(self.n_coef + 2)];
        let dec_coef = &record[(self.n_coef + 2)..(2 * self.n_coef + 2)];
        let w_coef = &record[(2 * self.n_coef + 2)..(3 * self.n_coef + 2)];

        let ([ra, dec, w], [ra_der, dec_der, w_der]) =
            chebyshev_evaluate_both(t, ra_coef, dec_coef, w_coef)?;

        // rem_euclid is equivalent to the modulo operator, so this maps w to [0, 2pi]
        let w = w.rem_euclid(std::f64::consts::TAU);

        let time = spice_jd_to_jd(jds);
        let frame = NonInertialFrame::from_euler::<'Z', 'X', 'Z'>(
            time,
            [ra, dec, w],
            [
                // convert to radians per day
                ra_der / t_step * 86400.0,
                dec_der / t_step * 86400.0,
                w_der / t_step * 86400.0,
            ],
            self.array.reference_frame_id,
            1,
        );

        Ok(frame)
    }

    /// Create a Type 2 (Chebyshev Euler angles, fixed intervals) PCK array.
    ///
    /// # Arguments
    /// * `frame_id`          - Body-fixed frame ID (e.g., 3000 for Earth).
    /// * `reference_frame_id`- Reference inertial frame (e.g., 17 for Ecliptic).
    /// * `cdata`             - Flat Chebyshev coefficients, `(polydg+1)*3` values per record
    ///   arranged as `[RA_0..RA_d, DEC_0..DEC_d, W_0..W_d]`.
    /// * `n_records`         - Number of records.
    /// * `btime`             - Begin time of first interval (SPICE seconds from J2000).
    /// * `intlen`            - Length of each interval (seconds). Must be > 0.
    /// * `polydg`            - Polynomial degree, in `[0, 27]`.
    /// * `jd_start`          - Segment start epoch.
    /// * `jd_end`            - Segment end epoch.
    /// * `segment_name`      - Name stored in the DAF name record (max 40 chars).
    ///
    /// # Errors
    /// Returns an error if the data builder rejects the inputs.
    pub fn new_array(
        frame_id: i32,
        reference_frame_id: i32,
        cdata: &[f64],
        n_records: usize,
        btime: f64,
        intlen: f64,
        polydg: usize,
        jd_start: Time<TDB>,
        jd_end: Time<TDB>,
        segment_name: &str,
    ) -> KeteResult<PckArray> {
        let data = build_type2_data(cdata, n_records, btime, intlen, polydg)?;
        Ok(PckArray::new(
            frame_id,
            reference_frame_id,
            2,
            jd_start,
            jd_end,
            data,
            segment_name.to_string(),
        ))
    }
}

impl TryFrom<PckArray> for PckSegmentType2 {
    type Error = Error;

    #[allow(
        clippy::cast_sign_loss,
        reason = "cast should work except when file is incorrectly formatted"
    )]
    fn try_from(array: PckArray) -> Result<Self, Self::Error> {
        let n_records = array.daf[array.daf.len() - 1] as usize;
        let record_len = array.daf[array.daf.len() - 2] as usize;
        let jd_step = array.daf[array.daf.len() - 3];

        let n_coef = (record_len - 2) / 3;

        // Type 2 layout: [n_records * record_len] [btime, intlen, rsize, n]
        let expected_len = record_len * n_records + 4;

        if expected_len != array.daf.len() {
            Err(Error::IOError(format!(
                "PCK type 2 format error: expected data length {expected_len}, found {}.",
                array.daf.len()
            )))?;
        }

        Ok(Self {
            array,
            jd_step,
            n_coef,
            record_len,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::daf::DafFile;

    #[test]
    fn pck_type2_round_trip() {
        use std::io::Cursor;

        let polydg = 2;
        let ninrec = (polydg + 1) * 3; // 9
        let n = 3;
        let btime = 0.0;
        let intlen = 86400.0;
        let cdata: Vec<f64> = (0..ninrec * n).map(|i| i as f64 * 0.01).collect();
        let jd_start: Time<TDB> = 2451545.0.into();
        let jd_end: Time<TDB> = (2451545.0 + 3.0).into();

        let mut daf = DafFile::new_pck("test pck", "pck round trip test");
        let pck_arr = PckSegmentType2::new_array(
            3000,
            17,
            &cdata,
            n,
            btime,
            intlen,
            polydg,
            jd_start,
            jd_end,
            "Earth orientation",
        )
        .unwrap();
        daf.arrays.push(pck_arr.daf);

        let mut buf = Cursor::new(Vec::new());
        daf.write_to(&mut buf).unwrap();

        let bytes = buf.into_inner();
        let daf = DafFile::from_buffer(Cursor::new(&bytes)).unwrap();
        assert_eq!(daf.daf_type, crate::daf::DAFType::Pck);
        assert_eq!(daf.arrays.len(), 1);
        assert_eq!(daf.n_doubles, 2);
        assert_eq!(daf.n_ints, 5);

        let pck: PckArray = daf.arrays.into_iter().next().unwrap().try_into().unwrap();
        assert_eq!(pck.frame_id, 3000);
    }
}
