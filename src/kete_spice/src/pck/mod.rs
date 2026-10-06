// SPDX-FileCopyrightText: 2026 Dar Dahlen
// SPDX-License-Identifier: BSD-3-Clause

//! Loading and reading of orientations from binary PCK kernel files.
//!
//! The [`LOADED_PCK`] singleton holds the loaded PCK segments. A
//! [`crossbeam::sync::ShardedLock`] protects it, so a caller must acquire the
//! lock before use. Most uses need only a read lock.

mod array;
mod segments;
/// PCK Type 2: Chebyshev polynomials (Euler angles).
pub mod type2;

pub use array::PckArray;
pub use type2::PckSegmentType2;

use std::collections::HashSet;
use std::fs;

use crate::daf::{DAFType, DafFile};
use crate::prepend_by_precedence;
use crossbeam::sync::ShardedLock;
use kete_core::cache::cache_path;
use kete_core::errors::{Error, KeteResult};
use kete_core::frames::NonInertialFrame;
use kete_core::time::{TDB, Time};
use segments::PckSegment;

/// A collection of segments.
#[derive(Debug, Default)]
pub struct PckCollection {
    /// Collection of PCK file information
    segments: Vec<PckSegment>,
}

impl PckCollection {
    /// Load all the segments of a PCK file into this collection, ahead of those already
    /// loaded.
    ///
    /// # Errors
    /// May fail if there are IO or Parsing errors.
    pub fn load_file(&mut self, filename: &str) -> KeteResult<()> {
        let file = DafFile::from_file(filename)?;
        if !matches!(file.daf_type, DAFType::Pck) {
            Err(Error::IOError(format!(
                "File {filename:?} is not a PCK formatted file."
            )))?;
        }

        let mut segments = Vec::with_capacity(file.arrays.len());
        for array in file.arrays {
            let pck_array: PckArray = array.try_into()?;
            segments.push(PckSegment::try_from(pck_array)?);
        }
        // A later file, and a later segment within a file, takes precedence.
        prepend_by_precedence(&mut self.segments, segments, |seg| {
            let arr: &PckArray = seg.into();
            arr.frame_id
        });
        Ok(())
    }

    /// The PCK frame with class ID `class_id` at `jd`, relative to the reference frame
    /// stored in the segment.
    ///
    /// # Errors
    /// Fails when no loaded segment holds `class_id` at `jd`.
    pub fn try_get_orientation(
        &self,
        class_id: i32,
        jd: Time<TDB>,
    ) -> KeteResult<NonInertialFrame> {
        for segment in &self.segments {
            let array: &PckArray = segment.into();
            if (array.frame_id == class_id) & array.contains(jd) {
                return segment.try_get_orientation(class_id, jd);
            }
        }

        Err(Error::Bounds(format!(
            "No loaded PCK segment holds class ID {class_id} at JD {}.",
            jd.jd()
        )))
    }

    /// Whether any loaded segment, at any time, holds the class ID `class_id`.
    #[must_use]
    pub fn has_frame(&self, class_id: i32) -> bool {
        self.segments
            .iter()
            .any(|segment| Into::<&PckArray>::into(segment).frame_id == class_id)
    }

    /// Delete all segments in the PCK singleton, equivalent to unloading all files.
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// Return the frame IDs of all loaded segments, each ID once.
    ///
    /// The order of the IDs is not defined.
    #[must_use]
    pub fn loaded_objects(&self) -> Vec<i32> {
        let loaded: HashSet<i32> = self
            .segments
            .iter()
            .map(|x| Into::<&PckArray>::into(x).frame_id)
            .collect();
        loaded.into_iter().collect()
    }

    /// Load the files in the core kernel cache directory.
    ///
    /// # Errors
    /// Fails when the cache directory cannot be found or read. A file that fails to
    /// parse is reported with ``eprintln`` and skipped.
    pub fn load_core(&mut self) -> KeteResult<()> {
        let cache = cache_path("kernels/core")?;
        self.load_directory(&cache)?;
        Ok(())
    }

    /// Load all PCK files in a directory, in sorted filename order.
    ///
    /// # Errors
    /// Fails when the directory cannot be read. A file that fails to parse is reported
    /// with ``eprintln`` and skipped.
    pub fn load_directory(&mut self, directory: &str) -> KeteResult<()> {
        // A later file takes precedence, so the load order matters. `read_dir`
        // does not define an order. A sorted order gives the same result on
        // every machine.
        let mut files: Vec<_> = fs::read_dir(directory)?
            .filter_map(|entry| entry.ok().map(|e| e.path()))
            .filter(|path| path.is_file())
            .collect();
        files.sort();
        for path in files {
            if let Some(filename) = path.to_str()
                && filename.to_lowercase().ends_with(".bpc")
                && let Err(err) = self.load_file(filename)
            {
                eprintln!("Failed to load PCK file {filename}: {err}");
            }
        }
        Ok(())
    }
}

/// PCK singleton.
///
/// A [`ShardedLock`] protects the [`PckCollection`]. Use `.try_read()` for
/// read-only access. The first access loads the core kernel files, and it
/// ignores a failure to load them.
pub static LOADED_PCK: std::sync::LazyLock<ShardedLock<PckCollection>> =
    std::sync::LazyLock::new(|| {
        let mut singleton = PckCollection::default();
        let _ = singleton.load_core();
        ShardedLock::new(singleton)
    });
