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
use super::type2::PckSegmentType2;
use crate::jd_to_spice_jd;
use kete_core::errors::Error;
use kete_core::frames::NonInertialFrame;
use kete_core::prelude::KeteResult;
use kete_core::time::{TDB, Time};
use std::fmt::Debug;

#[derive(Debug)]
pub(in crate::pck) enum PckSegment {
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
    pub(in crate::pck) fn try_get_orientation(
        &self,
        center_id: i32,
        epoch: Time<TDB>,
    ) -> KeteResult<NonInertialFrame> {
        let arr_ref: &PckArray = self.into();

        if center_id != arr_ref.frame_id {
            Err(Error::Bounds(
                "Center ID is not present in this record.".into(),
            ))?;
        }

        let jds = jd_to_spice_jd(epoch);

        if jds < arr_ref.jds_start || jds > arr_ref.jds_end {
            Err(Error::Bounds("JD is not present in this record.".into()))?;
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
