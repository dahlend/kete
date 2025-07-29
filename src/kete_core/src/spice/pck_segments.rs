//! Most users should interface with `pck.rs`, not this module.
//!
//! PCK Files are collections of `Segments`, which are ranges of times where the state
//! of an object is recorded. These segments are typically made up of many individual
//! `Records`, with an associated maximum and minimum time where they are valid for.
//!
//! There are unique structs for each possible segment type, not all are currently
//! supported.
//!
//! <https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/pck.html>
//!
//! There is a lot of repetition in this file, as many of the segment types have very
//! similar internal structures.
//!
// BSD 3-Clause License
//
// Copyright (c) 2025, Dar Dahlen
// Copyright (c) 2025, California Institute of Technology
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

use super::jd_to_spice_jd;
use super::{PckArray, interpolation::*};
use crate::errors::Error;
use crate::frames::NonInertialFrame;
use crate::prelude::KeteResult;
use crate::time::{TDB, Time};
use std::fmt::Debug;

#[derive(Debug)]
pub(in crate::spice) enum PckSegment {
    Type2(PckSegmentType2),
}

impl From<PckSegment> for PckArray {
    fn from(value: PckSegment) -> Self {
        match value {
            PckSegment::Type2(seg) => seg.array,
        }
    }
}

impl TryFrom<PckArray> for PckSegment {
    type Error = Error;

    fn try_from(array: PckArray) -> Result<Self, Self::Error> {
        match array.segment_type {
            2 => Ok(Self::Type2(array.try_into()?)),
            v => Err(Error::IOError(format!(
                "PCK Segment type {v:?} not supported."
            ))),
        }
    }
}

impl<'a> From<&'a PckSegment> for &'a PckArray {
    fn from(value: &'a PckSegment) -> Self {
        match value {
            PckSegment::Type2(seg) => &seg.array,
        }
    }
}

impl PckSegment {
    /// Return the [`NonInertialFrame`] at the specified JD. If the requested time is not within
    /// the available range, this will fail.
    pub(in crate::spice) fn try_get_orientation(
        &self,
        center_id: i32,
        jd: f64,
    ) -> KeteResult<NonInertialFrame> {
        let arr_ref: &PckArray = self.into();

        if center_id != arr_ref.frame_id {
            Err(Error::DAFLimits(
                "Center ID is not present in this record.".into(),
            ))?;
        }

        let jds = jd_to_spice_jd(jd);

        if jds < arr_ref.jds_start || jds > arr_ref.jds_end {
            Err(Error::DAFLimits("JD is not present in this record.".into()))?;
        }
        if arr_ref.reference_frame_id != 17 {
            Err(Error::ValueError(format!(
                "PCK frame ID {} is not supported. Only 17 (Ecliptic) is supported.",
                arr_ref.reference_frame_id
            )))?;
        }

        match &self {
            Self::Type2(v) => v.try_get_orientation(jds),
        }
    }
}

/// Chebyshev polynomials (Euler angles only)
///
/// <https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/spk.html#Type%201:%20Modified%20Difference%20Arrays>
///
#[derive(Debug)]
pub(in crate::spice) struct PckSegmentType2 {
    array: PckArray,
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
    fn try_get_orientation(&self, jds: f64) -> KeteResult<NonInertialFrame> {
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

        let time = Time::<TDB>::new(jd_to_spice_jd(jds));
        let frame = NonInertialFrame::from_euler::<'Z', 'X', 'Z'>(
            time,
            [ra, dec, w],
            [
                ra_der / t_step * 86400.0, // convert to radians per day
                dec_der / t_step * 86400.0,
                w_der / t_step * 86400.0,
            ],
            self.array.reference_frame_id,
            1,
        );

        Ok(frame)
    }
}

impl TryFrom<PckArray> for PckSegmentType2 {
    type Error = Error;

    fn try_from(array: PckArray) -> Result<Self, Self::Error> {
        let n_records = array.daf[array.daf.len() - 1] as usize;
        let record_len = array.daf[array.daf.len() - 2] as usize;
        let jd_step = array.daf[array.daf.len() - 3];

        let n_coef = (record_len - 2) / 3;

        if n_records * record_len + 4 != array.daf.len() {
            return Err(Error::IOError(
                "PCK File not formatted correctly. Number records found in file dont match expected number."
                    .into(),
            ));
        }

        Ok(Self {
            array,
            jd_step,
            n_coef,
            record_len,
        })
    }
}
