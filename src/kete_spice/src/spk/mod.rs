//! Loading and reading of states from JPL SPK kernel files.
//!
//! SPK data loads into a singleton, the [`LOADED_SPK`] static. A
//! [`crossbeam::sync::ShardedLock`] wraps the singleton, so a caller must
//! acquire the lock before use. Most uses need only read access.
//!
//! Example:
//! ```
//!     use kete_spice::spk::LOADED_SPK;
//!     use kete_core::frames::Ecliptic;
//!
//!     // get a read-only reference to the [`SpkCollection`]
//!     let singleton = LOADED_SPK.try_read().unwrap();
//!
//!     // get the state of 399 (Earth)
//!     let state = singleton.try_get_state::<Ecliptic>(399, 2451545.0.into());
//! ```

mod array;
pub mod repack;
pub(crate) mod segments;
pub mod type1;
pub mod type10;
pub mod type13;
pub mod type18;
pub mod type19;
pub mod type2;
pub mod type21;
pub mod type3;
pub mod type9;

pub use array::SpkArray;
pub use repack::repack_to_type2;
pub use repack::repack_to_type13;
pub use type1::SpkSegmentType1;
pub use type2::SpkSegmentType2;
pub use type3::SpkSegmentType3;
pub use type9::SpkSegmentType9;
pub use type13::SpkSegmentType13;
pub use type18::SpkSegmentType18;
pub use type19::SpkSegmentType19;
pub use type21::SpkSegmentType21;
// BSD 3-Clause License
//
// Copyright (c) 2026, Dar Dahlen
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

use crate::daf::DAFType;
use crate::daf::DafFile;
use crate::prepend_by_precedence;
use crate::spice_jd_to_jd;
use kete_core::cache::cache_path;
use kete_core::desigs::{NaifId, naif_ids_from_name};
use kete_core::errors::Error;
use kete_core::frames::{DynCenter, InertialFrame, SSB, SunCenter};
use kete_core::prelude::KeteResult;
use kete_core::state::State;
use kete_core::time::{TDB, Time};
use segments::SpkSegment;
use std::collections::{HashMap, HashSet};
use std::fs;

use crossbeam::sync::ShardedLock;

/// Maximum number of bodies in a chain of segment centers, including the first
/// body.
const MAX_CHAIN: usize = 8;

/// A chain of bodies, each paired with the segment that leads to the next body.
///
/// [`SpkCollection::center_chain`] fills the chain. The last body has no
/// segment.
type CenterChain<'a> = [(i32, Option<&'a SpkSegment>); MAX_CHAIN];

/// A collection of SPK segments.
#[derive(Debug, Default)]
pub struct SpkCollection {
    // This collection is split into two parts, the planet segments and the rest of the
    // segments. This is done to allow the planet segments to be accessed quickly,
    // as they are by far the most commonly used. Somewhat surprisingly, the
    // planet segments perform much better as a vector than as a hashmap, by about 40%
    // in typical usage. Putting everything in a vector destroys performance for
    // items further down the vector.
    /// Planet segments specifically for speed.
    planet_segments: Vec<SpkSegment>,

    /// Collection of SPK Segment information.
    segments: HashMap<i32, Vec<SpkSegment>>,

    /// Cache of all loaded NAIF IDs.
    naif_ids: HashMap<String, NaifId>,
}

impl SpkCollection {
    /// Get the raw state from the loaded SPK files.
    /// This state will have the center and frame of whatever was originally loaded
    /// into the file.
    ///
    /// # Errors
    /// Fails when the id or jd is not found in the [`SpkCollection`].
    #[inline(always)]
    pub fn try_get_state<T: InertialFrame>(&self, id: i32, jd: Time<TDB>) -> KeteResult<State<T>> {
        self.best_segment(id, jd)
            .ok_or_else(|| {
                Error::Bounds(format!(
                    "Object ({id}) does not have an SPK record for the target JD."
                ))
            })?
            .try_get_state(jd)
    }

    /// Return the highest precedence segment for `id` that covers `jd`.
    ///
    /// Return `None` if no loaded segment for `id` covers `jd`. The solar
    /// system barycenter (ID 0) is the origin of every ephemeris and has no
    /// segment. Thus the function returns `None` for ID 0 without a lookup.
    #[inline(always)]
    fn best_segment(&self, id: i32, jd: Time<TDB>) -> Option<&SpkSegment> {
        if id == 0 {
            return None;
        }
        // The loader stores IDs 0 to 1000 only in the planet segments.
        if (0..=1000).contains(&id) {
            return self.planet_segments.iter().find(|segment| {
                let arr_ref: &SpkArray = (*segment).into();
                arr_ref.object_id == id && arr_ref.contains(jd)
            });
        }
        self.segments.get(&id)?.iter().find(|segment| {
            let arr_ref: &SpkArray = (*segment).into();
            arr_ref.contains(jd)
        })
    }

    /// Load a state from the file, then attempt to change the center to the center id
    /// specified.
    ///
    /// # Errors
    /// Fails when the id or jd is not found in the [`SpkCollection`].
    #[inline(always)]
    pub fn try_get_state_with_center<T: InertialFrame>(
        &self,
        id: i32,
        jd: Time<TDB>,
        center: i32,
    ) -> KeteResult<State<T>> {
        let mut state = self.try_get_state(id, jd)?;
        if state.center_id() != center {
            self.try_change_center(&mut state, center)?;
        }
        Ok(state)
    }

    /// Change the center of a state with the loaded SPK segments.
    ///
    /// The function changes `state` in place so that its center is
    /// `new_center`. It uses the highest precedence segments that cover the
    /// epoch of the state. From each of the two centers, it follows the centers
    /// of these segments until the two chains share a body.
    ///
    /// # Errors
    /// - `Error::Bounds` if the segments that cover the epoch do not connect
    ///   the center of the state to `new_center`.
    /// - `Error::ValueError` if the segments form a cycle, or if a chain does
    ///   not end within `MAX_CHAIN` bodies.
    /// - The error of a segment evaluation that fails on the path.
    pub fn try_change_center<T: InertialFrame>(
        &self,
        state: &mut State<T>,
        new_center: i32,
    ) -> KeteResult<()> {
        let old_center = state.center_id();
        if old_center == new_center {
            return Ok(());
        }
        let epoch = state.epoch;

        // The common cases need no chains: one center is the parent of the
        // other, or the two centers share a parent.
        let old_segment = self.best_segment(old_center, epoch);
        let old_parent = old_segment.map(|segment| Into::<&SpkArray>::into(segment).center_id);
        if let Some(segment) = old_segment
            && old_parent == Some(new_center)
        {
            return state.try_change_center(segment.try_get_state(epoch)?);
        }
        if let Some(segment) = self.best_segment(new_center, epoch) {
            let new_parent = Into::<&SpkArray>::into(segment).center_id;
            if new_parent == old_center {
                return state.try_change_center(segment.try_get_state(epoch)?);
            }
            if let Some(old_segment) = old_segment
                && old_parent == Some(new_parent)
            {
                state.try_change_center(old_segment.try_get_state(epoch)?)?;
                return state.try_change_center(segment.try_get_state(epoch)?);
            }
        }

        let mut from: CenterChain<'_> = [(0, None); MAX_CHAIN];
        let mut to: CenterChain<'_> = [(0, None); MAX_CHAIN];
        let n_from = self.center_chain(old_center, epoch, new_center, &mut from)?;
        let (up, down) = if from[n_from - 1].0 == new_center {
            (n_from - 1, 0)
        } else {
            let n_to = self.center_chain(new_center, epoch, old_center, &mut to)?;
            from[..n_from]
                .iter()
                .enumerate()
                .find_map(|(i, (body, _))| {
                    to[..n_to]
                        .iter()
                        .position(|(b, _)| b == body)
                        .map(|j| (i, j))
                })
                .ok_or_else(|| {
                    Error::Bounds(format!(
                        "SPK files are missing information to be able to map from obj \
                         {old_center} to obj {new_center} at JD {}.",
                        epoch.jd
                    ))
                })?
        };

        // Move up from the old center to the shared body, then down to the new
        // center. Every body before the end of a chain has a segment.
        for (_, segment) in &from[..up] {
            if let Some(segment) = segment {
                state.try_change_center(segment.try_get_state(epoch)?)?;
            }
        }
        for (_, segment) in to[..down].iter().rev() {
            if let Some(segment) = segment {
                state.try_change_center(segment.try_get_state(epoch)?)?;
            }
        }
        Ok(())
    }

    /// Fill `chain` with a chain of bodies and return the number of bodies.
    ///
    /// The chain starts at `id`. The function pairs each body with its highest
    /// precedence segment that covers `jd`. The center of that segment is the
    /// next body. The chain ends at `stop`, or at a body with no segment that
    /// covers `jd`. The last body has no segment.
    ///
    /// # Errors
    /// `Error::ValueError` if the segments form a cycle, or if the chain does
    /// not end within [`MAX_CHAIN`] bodies.
    fn center_chain<'a>(
        &'a self,
        id: i32,
        jd: Time<TDB>,
        stop: i32,
        chain: &mut CenterChain<'a>,
    ) -> KeteResult<usize> {
        let mut len = 0;
        let mut body = id;
        while len < MAX_CHAIN {
            if body == stop {
                chain[len] = (body, None);
                return Ok(len + 1);
            }
            let segment = self.best_segment(body, jd);
            chain[len] = (body, segment);
            len += 1;
            let Some(segment) = segment else {
                return Ok(len);
            };
            body = Into::<&SpkArray>::into(segment).center_id;
            if chain[..len].iter().any(|(b, _)| *b == body) {
                break;
            }
        }
        Err(Error::ValueError(format!(
            "SPK segments covering JD {} form a cycle, or a chain of more than \
             {MAX_CHAIN} centers, from object {id}.",
            jd.jd
        )))
    }

    /// Change the center of a state to the Solar System Barycenter, returning a
    /// typed `State<T, SSB>` that carries the SSB guarantee at compile time.
    ///
    /// # Errors
    /// Fails when the required NAIF lookups are not available in the loaded SPKs.
    pub fn try_to_ssb<T: InertialFrame>(
        &self,
        state: impl Into<State<T, DynCenter>>,
    ) -> KeteResult<State<T, SSB>> {
        let mut state: State<T, DynCenter> = state.into();
        if state.center_id() != 0 {
            self.try_change_center(&mut state, 0)?;
        }
        State::<T, SSB>::try_from(state)
    }

    /// Change the center of a state to the Sun, returning a typed `State<T, SunCenter>`
    /// that carries the Sun-centered guarantee at compile time.
    ///
    /// # Errors
    /// Fails when the required NAIF lookups are not available in the loaded SPKs.
    pub fn try_to_sun<T: InertialFrame>(
        &self,
        state: impl Into<State<T, DynCenter>>,
    ) -> KeteResult<State<T, SunCenter>> {
        let mut state: State<T, DynCenter> = state.into();
        if state.center_id() != 10 {
            self.try_change_center(&mut state, 10)?;
        }
        State::<T, SunCenter>::try_from(state)
    }

    /// For a given NAIF ID, return all increments of time which are currently loaded.
    #[must_use]
    pub fn available_info(&self, id: i32) -> Vec<(Time<TDB>, Time<TDB>, i32, i32, i32)> {
        let mut segment_info = Vec::<(Time<TDB>, Time<TDB>, i32, i32, i32)>::new();
        if let Some(segments) = self.segments.get(&id) {
            for segment in segments {
                let spk_array_ref: &SpkArray = segment.into();
                let jds_start = spk_array_ref.jds_start;
                let jds_end = spk_array_ref.jds_end;
                segment_info.push((
                    spice_jd_to_jd(jds_start),
                    spice_jd_to_jd(jds_end),
                    spk_array_ref.center_id,
                    spk_array_ref.frame_id,
                    spk_array_ref.segment_type,
                ));
            }
        }

        self.planet_segments.iter().for_each(|segment| {
            let spk_array_ref: &SpkArray = segment.into();
            if spk_array_ref.object_id == id {
                let jds_start = spk_array_ref.jds_start;
                let jds_end = spk_array_ref.jds_end;
                segment_info.push((
                    spice_jd_to_jd(jds_start),
                    spice_jd_to_jd(jds_end),
                    spk_array_ref.center_id,
                    spk_array_ref.frame_id,
                    spk_array_ref.segment_type,
                ));
            }
        });
        if segment_info.is_empty() {
            return segment_info;
        }

        segment_info.sort_by(|a, b| (a.0.jd).total_cmp(&b.0.jd));

        let mut avail_times = Vec::<(Time<TDB>, Time<TDB>, i32, i32, i32)>::new();

        let mut cur_segment = segment_info[0];
        for segment in segment_info.iter().skip(1) {
            // if the segments are overlapped or nearly overlapped, join them together
            // 1e-8 is approximately a millisecond
            if cur_segment.1.jd <= (segment.0.jd - 1e-8) {
                avail_times.push(cur_segment);
                cur_segment = *segment;
            } else {
                cur_segment.1.jd = segment.1.jd.max(cur_segment.1.jd);
            }
        }
        avail_times.push(cur_segment);

        avail_times
    }

    /// Return the raw `(jds_start, jds_end)` SPICE-second boundaries for every
    /// loaded segment of the given object, sorted by start time.
    ///
    /// These are the exact segment boundaries as stored in the SPK files,
    /// without any merging or buffering.  Useful for the repacker to know
    /// precisely where source data exists.
    #[must_use]
    pub fn segment_boundaries(&self, id: i32) -> Vec<(f64, f64)> {
        let mut bounds = Vec::new();
        for seg in &self.planet_segments {
            let arr: &SpkArray = seg.into();
            if arr.object_id == id {
                bounds.push((arr.jds_start, arr.jds_end));
            }
        }
        if let Some(segs) = self.segments.get(&id) {
            for seg in segs {
                let arr: &SpkArray = seg.into();
                bounds.push((arr.jds_start, arr.jds_end));
            }
        }
        bounds.sort_by(|a, b| a.0.total_cmp(&b.0));
        bounds
    }

    /// Return a hash set of all unique identifies loaded in the SPKs.
    /// If include centers is true, then this additionally includes the IDs for the
    /// center IDs. For example, if ``include_centers`` is false, then `0` will never
    /// be included in the loaded objects set, as 0 is a privileged position at the
    /// barycenter of the solar system. It is not typically defined in relation to
    /// anything else.
    #[must_use]
    pub fn loaded_objects(&self, include_centers: bool) -> HashSet<i32> {
        let mut found = HashSet::new();

        for seg in &self.planet_segments {
            let spk_array_ref: &SpkArray = seg.into();
            let _ = found.insert(spk_array_ref.object_id);
            if include_centers {
                let _ = found.insert(spk_array_ref.center_id);
            }
        }

        self.segments.iter().for_each(|(obj_id, segs)| {
            let _ = found.insert(*obj_id);
            if include_centers {
                for seg in segs {
                    let spk_array_ref: &SpkArray = seg.into();
                    let _ = found.insert(spk_array_ref.center_id);
                }
            }
        });
        found
    }

    /// Load all the segments of an SPK file into this collection.
    ///
    /// `filename` is the path of the file. For overlapping segments, a segment
    /// from a file loaded later takes precedence. Within a file, a segment
    /// stored later takes precedence. If the file fails to load, the collection
    /// does not change.
    ///
    /// # Errors
    /// - `Error::IOError` if the file cannot be read, is not an SPK file, or
    ///   holds a segment of an unsupported type.
    /// - The error of the segment parser if a segment is malformed.
    pub fn load_file(&mut self, filename: &str) -> KeteResult<()> {
        let file = DafFile::from_file(filename)?;
        self.load_daf(file, filename)
    }

    /// Load SPK segments from any `Read + Seek` source (e.g. an in-memory buffer).
    ///
    /// # Errors
    /// Loading files may fail for a number of reasons, likely incorrect formatted
    /// content.
    pub fn load_from_reader<R: std::io::Read + std::io::Seek>(
        &mut self,
        reader: R,
    ) -> KeteResult<()> {
        let file = DafFile::from_buffer(reader)?;
        self.load_daf(file, "<buffer>")
    }

    fn load_daf(&mut self, file: DafFile, source: &str) -> KeteResult<()> {
        if !matches!(file.daf_type, DAFType::Spk) {
            Err(Error::IOError(format!(
                "File {source:?} is not a SPK formatted file."
            )))?;
        }
        // Parse the whole file before the collection changes. Thus a file that
        // fails partway through does not change the collection.
        let mut planets = Vec::new();
        let mut others: HashMap<i32, Vec<SpkSegment>> = HashMap::new();
        for daf_array in file.arrays {
            let segment: SpkArray = daf_array.try_into()?;
            let id = segment.object_id;
            if (0..=1000).contains(&id) {
                planets.push(segment.try_into()?);
            } else {
                others.entry(id).or_default().push(segment.try_into()?);
            }
        }

        // Later files, and later segments within a file, take precedence. They
        // go first because the readers return the first match.
        prepend_by_precedence(&mut self.planet_segments, planets, |seg| {
            let arr: &SpkArray = seg.into();
            arr.object_id
        });
        for (id, segs) in others {
            let _ = self
                .segments
                .entry(id)
                .or_default()
                .splice(0..0, segs.into_iter().rev());
        }
        Ok(())
    }

    /// Delete all segments in the SPK singleton, equivalent to unloading all files.
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// Load the core files.
    ///
    /// # Errors
    /// May fail if there are IO or Parsing errors.
    pub fn load_core(&mut self) -> KeteResult<()> {
        let cache = cache_path("kernels/core")?;
        self.load_directory(&cache)?;
        Ok(())
    }

    /// Load files in the cache directory.
    ///
    /// # Errors
    /// May fail if there are IO or Parsing errors.
    pub fn load_cache(&mut self) -> KeteResult<()> {
        let cache = cache_path("kernels")?;
        self.load_directory(&cache)?;
        Ok(())
    }

    /// Load all SPK files from a directory.
    ///
    /// # Errors
    /// May fail if there are IO or Parsing errors.
    pub fn load_directory(&mut self, directory: &str) -> KeteResult<()> {
        // Later files take precedence, so the load order matters. `read_dir`
        // does not specify an order. Sorted order gives the same result on
        // every machine.
        let mut files: Vec<_> = fs::read_dir(directory)?
            .filter_map(|entry| entry.ok().map(|e| e.path()))
            .filter(|path| path.is_file())
            .collect();
        files.sort();
        for path in files {
            if let Some(filename) = path.to_str()
                && filename.to_lowercase().ends_with(".bsp")
                && let Err(err) = self.load_file(filename)
            {
                eprintln!("Failed to load SPK file {filename}: {err}");
            }
        }
        Ok(())
    }

    /// Try to get the unique loaded NAIF ID for the given name.
    ///
    /// If there are multiple ids which match, but one of them is an exact match,
    /// that one is returned.
    ///
    /// # Errors
    /// If no ID is found, an error is returned.
    /// If multiple IDs are found, an error is returned.
    pub fn try_id_from_name(&mut self, name: &str) -> KeteResult<NaifId> {
        // check first for cache hit with a read only lock

        if let Some(id) = self.naif_ids.get(name.to_lowercase().as_str()) {
            return Ok(id.clone());
        }

        let mut loaded_ids = self
            .planet_segments
            .iter()
            .map(|x| {
                let arr: &SpkArray = x.into();
                arr.object_id
            })
            .collect::<HashSet<i32>>();
        loaded_ids.extend(self.segments.keys().copied());

        let mut ids = naif_ids_from_name(name);
        // remove any IDs which are not loaded in the SPK files.
        ids.retain(|id| loaded_ids.contains(&id.id) || id.id == 0);

        if ids.is_empty() {
            return Err(Error::ValueError(format!(
                "No NAIF ID found for name: {name}"
            )));
        } else if ids.len() == 1 {
            #[allow(
                clippy::missing_panics_doc,
                reason = "Length is 1, unwrap always possible."
            )]
            let id = ids.first().unwrap().clone();
            let _ = self.naif_ids.insert(name.to_lowercase(), id.clone());
            return Ok(id);
        }

        // check if any of the returned names match exactly
        for id in &ids {
            if id.name.to_lowercase() == name.to_lowercase() {
                let _ = self.naif_ids.insert(name.to_lowercase(), id.clone());
                return Ok(id.clone());
            }
        }

        Err(Error::ValueError(format!(
            "Multiple NAIF IDs found for name '{}':\n{}",
            name,
            ids.iter()
                .map(|id| id.name.clone())
                .collect::<Vec<String>>()
                .join(",\n")
        )))
    }
}

/// SPK singleton.
///
/// A lock protects this [`SpkCollection`]. Use `.try_read()` for read-only
/// access.
pub static LOADED_SPK: std::sync::LazyLock<ShardedLock<SpkCollection>> =
    std::sync::LazyLock::new(|| {
        let mut singleton = SpkCollection::default();
        let _ = singleton.load_core();
        ShardedLock::new(singleton)
    });

#[cfg(test)]
mod tests {
    use super::*;
    use crate::daf::DafFile;
    use kete_core::constants::AU_KM;
    use kete_core::frames::Equatorial;
    use std::io::Cursor;

    /// Return a type 18 segment that holds the object at `x_km` from the Sun.
    ///
    /// The segment covers one day from J2000.
    fn fixed_segment(object_id: i32, x_km: f64, name: &str) -> SpkArray {
        fixed_segment_about(object_id, 10, x_km, name)
    }

    /// Return a type 18 segment that holds the object at `x_km` from
    /// `center_id`.
    ///
    /// The segment covers one day from J2000.
    fn fixed_segment_about(object_id: i32, center_id: i32, x_km: f64, name: &str) -> SpkArray {
        let records = [x_km, 0.0, 0.0, 0.0, 0.0, 0.0, x_km, 0.0, 0.0, 0.0, 0.0, 0.0];
        SpkSegmentType18::new_array(
            object_id,
            center_id,
            1,
            &records,
            &[0.0, 86400.0],
            1,
            2,
            0.0,
            86400.0,
            name,
        )
        .unwrap()
    }

    fn file_of(segments: Vec<SpkArray>) -> Vec<u8> {
        let mut daf = DafFile::new_spk("precedence test", "");
        for seg in segments {
            daf.arrays.push(seg.daf);
        }
        let mut buf = Cursor::new(Vec::new());
        daf.write_to(&mut buf).unwrap();
        buf.into_inner()
    }

    fn x_km(spk: &SpkCollection, id: i32) -> f64 {
        let state: State<Equatorial> = spk.try_get_state(id, Time::new(2451545.5)).unwrap();
        state.pos[0] * AU_KM
    }

    /// Overlapping segments resolve to the file loaded last.
    ///
    /// The test checks a planet ID and a non-planet ID, because the collection
    /// stores the two kinds separately.
    #[test]
    fn the_file_loaded_last_wins() {
        for id in [5, 2_000_001] {
            let older = file_of(vec![fixed_segment(id, 1.0, "older")]);
            let newer = file_of(vec![fixed_segment(id, 2.0, "newer")]);

            let mut spk = SpkCollection::default();
            spk.load_from_reader(Cursor::new(&older)).unwrap();
            spk.load_from_reader(Cursor::new(&newer)).unwrap();
            assert_eq!(x_km(&spk, id), 2.0, "id {id}");

            let mut spk = SpkCollection::default();
            spk.load_from_reader(Cursor::new(&newer)).unwrap();
            spk.load_from_reader(Cursor::new(&older)).unwrap();
            assert_eq!(x_km(&spk, id), 1.0, "id {id}");
        }
    }

    /// Within one file, overlapping segments resolve to the segment stored
    /// last.
    #[test]
    fn the_segment_stored_last_wins_within_a_file() {
        for id in [5, 2_000_001] {
            let file = file_of(vec![
                fixed_segment(id, 1.0, "first"),
                fixed_segment(id, 2.0, "second"),
            ]);
            let mut spk = SpkCollection::default();
            spk.load_from_reader(Cursor::new(&file)).unwrap();
            assert_eq!(x_km(&spk, id), 2.0, "id {id}");
        }
    }

    /// Change the center from an object that orbits a spacecraft to the SSB.
    ///
    /// A spacecraft is not a barycenter. The path to the SSB goes through the
    /// segment of the spacecraft, then through the segment of its center.
    #[test]
    fn change_center_from_a_spacecraft_reaches_the_ssb() {
        let file = file_of(vec![
            fixed_segment_about(6, 0, 1000.0, "saturn bary"),
            fixed_segment_about(-82, 6, 5.0, "spacecraft"),
            fixed_segment_about(2_000_001, -82, 1.0, "object"),
        ]);
        let mut spk = SpkCollection::default();
        spk.load_from_reader(Cursor::new(&file)).unwrap();

        let jd = Time::new(2451545.5);
        let state: State<Equatorial> = spk.try_get_state_with_center(2_000_001, jd, 0).unwrap();
        assert_eq!(state.center_id(), 0);
        assert!((state.pos[0] * AU_KM - 1006.0).abs() < 1e-6);

        let state: State<Equatorial> = spk.try_get_state_with_center(-82, jd, 0).unwrap();
        assert_eq!(state.center_id(), 0);
        assert!((state.pos[0] * AU_KM - 1005.0).abs() < 1e-6);
    }

    /// A center change follows only the segments that cover the epoch.
    ///
    /// The test includes a segment that links the same bodies at a different
    /// time. The center change must not use it.
    #[test]
    fn change_center_uses_segments_covering_the_epoch() {
        let records = [50.0, 0.0, 0.0, 0.0, 0.0, 0.0, 50.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let later = SpkSegmentType18::new_array(
            3,
            10,
            1,
            &records,
            &[5.0 * 86400.0, 6.0 * 86400.0],
            1,
            2,
            5.0 * 86400.0,
            6.0 * 86400.0,
            "emb from sun later",
        )
        .unwrap();
        let file = file_of(vec![
            fixed_segment_about(10, 0, 1.0, "sun"),
            fixed_segment_about(3, 0, 100.0, "emb"),
            later,
            fixed_segment_about(399, 3, 2.0, "earth"),
        ]);
        let mut spk = SpkCollection::default();
        spk.load_from_reader(Cursor::new(&file)).unwrap();

        let state: State<Equatorial> = spk
            .try_get_state_with_center(399, Time::new(2451545.5), 10)
            .unwrap();
        assert_eq!(state.center_id(), 10);
        assert!((state.pos[0] * AU_KM - 101.0).abs() < 1e-6);

        let state: State<Equatorial> = spk
            .try_get_state_with_center(10, Time::new(2451545.5), 399)
            .unwrap();
        assert_eq!(state.center_id(), 399);
        assert!((state.pos[0] * AU_KM + 101.0).abs() < 1e-6);
    }
}
