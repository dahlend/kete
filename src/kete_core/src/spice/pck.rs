//! Loading and reading of states from JPL PCK kernel files.
//!
//! PCKs are intended to be loaded into a singleton which is accessible via the
//! [`LOADED_PCK`] object defined below. This singleton is wrapped in a
//! [`crossbeam::sync::ShardedLock`], meaning before its use it must by unwrapped.
//! A vast majority of intended use cases will only be the read case.
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

use std::collections::HashSet;
use std::fs;

use super::PckArray;
use super::daf::{DAFType, DafFile};
use super::pck_segments::PckSegment;
use crate::cache::cache_path;
use crate::errors::{Error, KeteResult};
use crate::frames::NonInertialFrame;
use crossbeam::sync::ShardedLock;

/// A collection of segments.
#[derive(Debug, Default)]
pub struct PckCollection {
    /// Collection of PCK file information
    segments: Vec<PckSegment>,
}

impl PckCollection {
    /// Given an PCK filename, load all the segments present inside of it.
    /// These segments are added to the PCK singleton in memory.
    pub fn load_file(&mut self, filename: &str) -> KeteResult<()> {
        let file = DafFile::from_file(filename)?;
        if !matches!(file.daf_type, DAFType::Pck) {
            return Err(Error::IOError(format!(
                "File {filename:?} is not a PCK formatted file."
            )))?;
        }

        for array in file.arrays {
            let pck_array: PckArray = array.try_into()?;
            let segment: PckSegment = pck_array.try_into()?;
            self.segments.push(segment);
        }
        Ok(())
    }

    /// Get the raw orientation from the loaded PCK files.
    /// This orientation will have the frame of what was originally present in the file.
    pub fn try_get_orientation(&self, id: i32, jd: f64) -> KeteResult<NonInertialFrame> {
        for segment in self.segments.iter() {
            let array: &PckArray = segment.into();
            if (array.frame_id == id) & array.contains(jd) {
                return segment.try_get_orientation(id, jd);
            }
        }

        Err(Error::DAFLimits(format!(
            "Object ({id}) does not have an PCK record for the target JD."
        )))?
    }

    /// Delete all segments in the PCK singleton, equivalent to unloading all files.
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// Return a list of all loaded segments in the PCK singleton.
    /// This is a list of the center NAIF IDs of the segments.
    pub fn loaded_objects(&self) -> Vec<i32> {
        let loaded: HashSet<i32> = self
            .segments
            .iter()
            .map(|x| Into::<&PckArray>::into(x).frame_id)
            .collect();
        loaded.into_iter().collect()
    }

    /// Load the core files.
    pub fn load_core(&mut self) -> KeteResult<()> {
        let cache = cache_path("kernels/core")?;
        self.load_directory(cache)?;
        Ok(())
    }

    /// Load files in the cache directory.
    pub fn load_cache(&mut self) -> KeteResult<()> {
        let cache = cache_path("kernels")?;
        self.load_directory(cache)?;
        Ok(())
    }

    /// Load all PCK files from a directory.
    pub fn load_directory(&mut self, directory: String) -> KeteResult<()> {
        fs::read_dir(&directory)?.for_each(|entry| {
            let entry = entry.unwrap();
            let path = entry.path();
            if path.is_file() {
                let filename = path.to_str().unwrap();
                if filename.to_lowercase().ends_with(".bpc") {
                    if let Err(err) = self.load_file(filename) {
                        eprintln!("Failed to load PCK file {filename}: {err}");
                    }
                }
            }
        });
        Ok(())
    }
}

/// PCK singleton.
/// This is a lock protected [`PckCollection`], and must be `.try_read().unwrapped()` for any
/// read-only cases.
pub static LOADED_PCK: std::sync::LazyLock<ShardedLock<PckCollection>> =
    std::sync::LazyLock::new(|| {
        let mut singleton = PckCollection::default();
        let _ = singleton.load_core();
        ShardedLock::new(singleton)
    });
