// SPDX-FileCopyrightText: 2026 Dar Dahlen
// SPDX-License-Identifier: BSD-3-Clause

use super::PckArray;
use super::type2::PckSegmentType2;
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

        let jds = epoch.j2000_seconds();

        if jds < arr_ref.jds_start || jds > arr_ref.jds_end {
            Err(Error::Bounds("JD is not present in this record.".into()))?;
        }

        match &self {
            Self::Type2(v) => v.try_get_orientation(epoch),
        }
    }
}
