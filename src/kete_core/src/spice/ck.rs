//! Loading and reading of states from JPL CK kernel files.
//!
//! CKs are intended to be loaded into a singleton which is accessible via the
//! [`LOADED_CK`] object defined below. This singleton is wrapped in a
//! [`crossbeam::sync::ShardedLock`], meaning before its use it must by unwrapped.
//! A vast majority of intended use cases will only be the read case.
//!
// BSD 3-Clause License
//
// Copyright (c) 2025, Dar Dahlen
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

use crate::{
    errors::{Error, KeteResult},
    frames::NonInertialFrame,
    time::{TDB, Time},
};

use super::{CkArray, DAFType, DafFile, LOADED_SCLK, ck_segments::CkSegment};
use crossbeam::sync::ShardedLock;

/// A collection of segments.
#[derive(Debug, Default)]
pub struct CkCollection {
    /// Collection of CK file information
    pub(crate) segments: Vec<CkSegment>,
}

impl CkCollection {
    /// Given an CK filename, load all the segments present inside of it.
    /// These segments are added to the CK singleton in memory.
    ///
    /// # Errors
    /// [`Error::IOError`] if the file is not a CK formatted file.
    ///
    pub fn load_file(&mut self, filename: &str) -> KeteResult<()> {
        let file = DafFile::from_file(filename)?;
        if !matches!(file.daf_type, DAFType::Ck) {
            return Err(Error::IOError(format!(
                "File {filename:?} is not a CK formatted file."
            )))?;
        }

        for array in file.arrays {
            let ck_array: CkArray = array.try_into()?;
            let segment: CkSegment = ck_array.try_into()?;
            self.segments.push(segment);
        }
        Ok(())
    }

    /// Clear all loaded CK kernels.
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// Get the closest record to the given JD for the specified instrument ID.
    ///
    /// # Errors
    /// [`Error::DAFLimits`] if the instrument ID does not have a record for the target JD.
    ///
    pub fn try_get_frame(
        &self,
        jd: f64,
        instrument_id: i32,
    ) -> KeteResult<(Time<TDB>, NonInertialFrame)> {
        let time = Time::<TDB>::new(jd);
        let sclk = LOADED_SCLK.try_read().unwrap();
        let spice_id = instrument_id / 1000;
        let tick = sclk.try_time_to_tick(spice_id, time)?;

        for segment in &self.segments {
            let array: &CkArray = segment.into();
            if (array.instrument_id == instrument_id) & array.contains(tick) {
                return segment.try_get_orientation(instrument_id, time);
            }
        }

        Err(Error::DAFLimits(format!(
            "Instrument ({instrument_id}) does not have an CK record for the target JD."
        )))?
    }

    /// Return a list of all loaded instrument ids.
    pub fn loaded_instruments(&self) -> Vec<i32> {
        self.segments
            .iter()
            .map(|s| {
                let array: &CkArray = s.into();
                array.instrument_id
            })
            .collect::<Vec<i32>>()
            .into_iter()
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect()
    }

    /// For a given NAIF ID, return all increments of time which are currently loaded.
    pub fn available_info(&self, instrument_id: i32) -> Vec<(i32, i32, i32, f64, f64)> {
        self.segments
            .iter()
            .filter_map(|s| {
                let array: &CkArray = s.into();
                if array.instrument_id == instrument_id {
                    Some((
                        array.instrument_id,
                        array.reference_frame_id,
                        array.segment_type,
                        array.tick_start,
                        array.tick_end,
                    ))
                } else {
                    None
                }
            })
            .collect()
    }
}

/// CK singleton.
/// This is a lock protected [`CkCollection`], and must be `.try_read().unwrapped()` for any
/// read-only cases.
pub static LOADED_CK: std::sync::LazyLock<ShardedLock<CkCollection>> =
    std::sync::LazyLock::new(|| {
        let singleton = CkCollection::default();
        ShardedLock::new(singleton)
    });
