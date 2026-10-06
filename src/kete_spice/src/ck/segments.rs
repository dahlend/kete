// SPDX-FileCopyrightText: 2026 Dar Dahlen
// SPDX-License-Identifier: BSD-3-Clause

use super::CkArray;
use super::type2::CkSegmentType2;
use super::type3::CkSegmentType3;
use super::type5::CkSegmentType5;
use super::type6::CkSegmentType6;
use crate::text::sclk::Sclk;
use kete_core::errors::{Error, KeteResult};
use kete_core::frames::NonInertialFrame;
use kete_core::time::{TDB, Time};

#[derive(Debug)]
pub(crate) enum CkSegment {
    Type2(CkSegmentType2),
    Type3(CkSegmentType3),
    Type5(CkSegmentType5),
    Type6(CkSegmentType6),
}

impl CkSegment {
    /// The pointing at `time`, whose tick `tick` on the segment's clock `sclk` the
    /// caller has already computed.
    pub(crate) fn try_get_orientation(
        &self,
        instrument_id: i32,
        time: Time<TDB>,
        tick: f64,
        sclk: &Sclk,
    ) -> KeteResult<(Time<TDB>, NonInertialFrame)> {
        let arr_ref: &CkArray = self.into();
        if arr_ref.instrument_id != instrument_id {
            return Err(Error::Bounds(format!(
                "Instrument ID is not present in this record. {}",
                arr_ref.instrument_id
            )));
        }

        match self {
            Self::Type3(seg) => seg.try_get_orientation(time, tick, sclk),
            Self::Type2(seg) => seg.try_get_orientation(time, tick),
            Self::Type5(seg) => seg.try_get_orientation(time, tick, sclk),
            Self::Type6(seg) => seg.try_get_orientation(time, tick),
        }
    }
}

impl CkSegment {
    /// Return whether the segment holds pointing at the clock tick `tick`.
    ///
    /// A segment can span `tick` and hold no pointing there. Every supported
    /// type can leave gaps between intervals inside the segment bounds.
    pub(crate) fn has_data_at(&self, tick: f64) -> bool {
        match self {
            Self::Type2(seg) => seg.has_data_at(tick),
            Self::Type3(seg) => seg.has_data_at(tick),
            Self::Type5(seg) => seg.has_data_at(tick),
            Self::Type6(seg) => seg.has_data_at(tick),
        }
    }
}

impl<'a> From<&'a CkSegment> for &'a CkArray {
    fn from(value: &'a CkSegment) -> Self {
        match value {
            CkSegment::Type3(seg) => &seg.array,
            CkSegment::Type2(seg) => &seg.array,
            CkSegment::Type5(seg) => &seg.array,
            CkSegment::Type6(seg) => &seg.array,
        }
    }
}

impl From<CkSegment> for CkArray {
    fn from(value: CkSegment) -> Self {
        match value {
            CkSegment::Type3(seg) => seg.array,
            CkSegment::Type2(seg) => seg.array,
            CkSegment::Type5(seg) => seg.array,
            CkSegment::Type6(seg) => seg.array,
        }
    }
}

impl TryFrom<CkArray> for CkSegment {
    type Error = Error;

    fn try_from(array: CkArray) -> Result<Self, Self::Error> {
        match array.segment_type {
            2 => Ok(Self::Type2(array.try_into()?)),
            3 => Ok(Self::Type3(array.try_into()?)),
            5 => Ok(Self::Type5(array.try_into()?)),
            6 => Ok(Self::Type6(array.try_into()?)),
            v => Err(Error::IOError(format!(
                "CK Segment type {v:?} not supported.",
            ))),
        }
    }
}
