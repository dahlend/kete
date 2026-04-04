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

use super::CkArray;
use super::type2::CkSegmentType2;
use super::type3::CkSegmentType3;
use kete_core::errors::{Error, KeteResult};
use kete_core::frames::NonInertialFrame;
use kete_core::time::{TDB, Time};

#[derive(Debug)]
pub(crate) enum CkSegment {
    Type2(CkSegmentType2),
    Type3(CkSegmentType3),
}

impl CkSegment {
    pub(crate) fn try_get_orientation(
        &self,
        instrument_id: i32,
        time: Time<TDB>,
    ) -> KeteResult<(Time<TDB>, NonInertialFrame)> {
        let arr_ref: &CkArray = self.into();
        if arr_ref.instrument_id != instrument_id {
            return Err(Error::Bounds(format!(
                "Instrument ID is not present in this record. {}",
                arr_ref.instrument_id
            )));
        }

        match self {
            Self::Type3(seg) => seg.try_get_orientation(time),
            Self::Type2(seg) => seg.try_get_orientation(time),
        }
    }
}

impl<'a> From<&'a CkSegment> for &'a CkArray {
    fn from(value: &'a CkSegment) -> Self {
        match value {
            CkSegment::Type3(seg) => &seg.array,
            CkSegment::Type2(seg) => &seg.array,
        }
    }
}

impl From<CkSegment> for CkArray {
    fn from(value: CkSegment) -> Self {
        match value {
            CkSegment::Type3(seg) => seg.array,
            CkSegment::Type2(seg) => seg.array,
        }
    }
}

impl TryFrom<CkArray> for CkSegment {
    type Error = Error;

    fn try_from(array: CkArray) -> Result<Self, Self::Error> {
        match array.segment_type {
            2 => Ok(Self::Type2(array.try_into()?)),
            3 => Ok(Self::Type3(array.try_into()?)),
            v => Err(Error::IOError(format!(
                "CK Segment type {v:?} not supported.",
            ))),
        }
    }
}
