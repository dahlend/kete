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

use super::SpkArray;
use super::type1::SpkSegmentType1;
use super::type2::SpkSegmentType2;
use super::type3::SpkSegmentType3;
use super::type9::SpkSegmentType9;
use super::type10::SpkSegmentType10;
use super::type13::SpkSegmentType13;
use super::type18::SpkSegmentType18;
use super::type21::SpkSegmentType21;
use crate::daf::DafArray;
use crate::jd_to_spice_jd;
use kete_core::errors::Error;
use kete_core::frames::{Ecliptic, Equatorial, FK4, Galactic, InertialFrame};
use kete_core::prelude::{Desig, KeteResult};
use kete_core::state::State;
use kete_core::time::{TDB, Time};
use std::fmt::Debug;

#[derive(Debug)]
pub(in crate::spk) enum SpkSegment {
    Type1(SpkSegmentType1),
    Type2(SpkSegmentType2),
    Type3(SpkSegmentType3),
    Type9(SpkSegmentType9),
    Type10(SpkSegmentType10),
    Type13(SpkSegmentType13),
    Type18(SpkSegmentType18),
    Type21(SpkSegmentType21),
}

impl TryFrom<SpkArray> for SpkSegment {
    type Error = Error;

    fn try_from(array: SpkArray) -> Result<Self, Self::Error> {
        match array.segment_type {
            1 => Ok(Self::Type1(array.try_into()?)),
            2 => Ok(Self::Type2(array.try_into()?)),
            3 => Ok(Self::Type3(array.try_into()?)),
            9 => Ok(Self::Type9(array.try_into()?)),
            10 => Ok(Self::Type10(array.try_into()?)),
            13 => Ok(Self::Type13(array.try_into()?)),
            18 => Ok(Self::Type18(array.try_into()?)),
            21 => Ok(Self::Type21(array.try_into()?)),
            v => Err(Error::IOError(format!(
                "SPK Segment type {v} not supported. Please submit a github issue!",
            ))),
        }
    }
}

impl<'a> From<&'a SpkSegment> for &'a SpkArray {
    fn from(segment: &'a SpkSegment) -> Self {
        match segment {
            SpkSegment::Type1(seg) => &seg.array,
            SpkSegment::Type2(seg) => &seg.array,
            SpkSegment::Type3(v) => &v.array,
            SpkSegment::Type9(v) => &v.array,
            SpkSegment::Type10(v) => &v.array.array,
            SpkSegment::Type13(v) => &v.array,
            SpkSegment::Type18(v) => &v.array,
            SpkSegment::Type21(v) => &v.array,
        }
    }
}

impl From<SpkSegment> for DafArray {
    fn from(segment: SpkSegment) -> Self {
        match segment {
            SpkSegment::Type1(seg) => seg.array.daf,
            SpkSegment::Type2(seg) => seg.array.daf,
            SpkSegment::Type3(v) => v.array.daf,
            SpkSegment::Type9(v) => v.array.daf,
            SpkSegment::Type10(v) => v.array.array.daf,
            SpkSegment::Type13(v) => v.array.daf,
            SpkSegment::Type18(v) => v.array.daf,
            SpkSegment::Type21(v) => v.array.daf,
        }
    }
}

impl SpkSegment {
    /// Return the [`State`] object at the specified JD. If the requested time is
    /// not within the available range, this will fail.
    #[inline(always)]
    pub(in crate::spk) fn try_get_state<T: InertialFrame>(
        &self,
        jd: Time<TDB>,
    ) -> KeteResult<State<T>> {
        let arr_ref: &SpkArray = self.into();

        let jds = jd_to_spice_jd(jd);

        // this is faster than calling contains, probably because the || instead of &&
        if jds < arr_ref.jds_start || jds > arr_ref.jds_end {
            return Err(Error::Bounds(
                "JD is not present in this record.".to_string(),
            ));
        }

        let (pos, vel) = match &self {
            Self::Type1(v) => v.try_get_pos_vel(jds)?,
            Self::Type2(v) => v.try_get_pos_vel(jds)?,
            Self::Type3(v) => v.try_get_pos_vel(jds)?,
            Self::Type9(v) => v.try_get_pos_vel(jds),
            Self::Type10(v) => v.try_get_pos_vel(jds),
            Self::Type13(v) => v.try_get_pos_vel(jds),
            Self::Type18(v) => v.try_get_pos_vel(jds),
            Self::Type21(v) => v.try_get_pos_vel(jds)?,
        };

        match arr_ref.frame_id {
            1 => Ok(State::<Equatorial>::new(
                Desig::Naif(arr_ref.object_id),
                jd,
                pos.into(),
                vel.into(),
                arr_ref.center_id,
            )
            .into_frame()),
            3 => Ok(State::<FK4>::new(
                Desig::Naif(arr_ref.object_id),
                jd,
                pos.into(),
                vel.into(),
                arr_ref.center_id,
            )
            .into_frame()),
            13 => Ok(State::<Galactic>::new(
                Desig::Naif(arr_ref.object_id),
                jd,
                pos.into(),
                vel.into(),
                arr_ref.center_id,
            )
            .into_frame()),
            17 => Ok(State::<Ecliptic>::new(
                Desig::Naif(arr_ref.object_id),
                jd,
                pos.into(),
                vel.into(),
                arr_ref.center_id,
            )
            .into_frame()),
            _ => Err(Error::ValueError(format!(
                "Frame {:?} not supported. Please submit a github issue! Please include the SPK file!",
                arr_ref.frame_id
            )))?,
        }
    }
}
