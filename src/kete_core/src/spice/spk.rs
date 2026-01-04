//! Loading and reading of states from JPL SPK kernel files.
//!
//! SPKs are intended to be loaded into a singleton which is accessible via the
//! [`LOADED_SPK`] function defined below. This singleton is wrapped in a
//! [`crossbeam::sync::ShardedLock`], meaning before its use it must by unwrapped.
//! A vast majority of intended use cases will only be the read case.
//!
//! Here is a small worked example:
//! ```
//!     use kete_core::spice::LOADED_SPK;
//!     use kete_core::frames::Ecliptic;
//!
//!     // get a read-only reference to the [`SpkCollection`]
//!     let singleton = LOADED_SPK.try_read().unwrap();
//!
//!     // get the state of 399 (Earth)
//!     let state = singleton.try_get_state::<Ecliptic>(399, 2451545.0.into());
//! ```
//!
//!
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

use super::daf::DafFile;
use super::{DAFType, SpkArray, spice_jd_to_jd, spk_segments::SpkSegment};
use super::{NaifId, naif_ids_from_name};
use crate::cache::cache_path;
use crate::errors::Error;
use crate::frames::InertialFrame;
use crate::prelude::KeteResult;
use crate::state::State;
use crate::time::{TDB, Time};
use pathfinding::prelude::dijkstra;
use std::collections::{HashMap, HashSet};
use std::fs;

use crossbeam::sync::ShardedLock;

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

    /// Cache for the pathfinding algorithm between different segments.
    map_cache: HashMap<(i32, i32), Vec<i32>>,

    /// Map from object id to all connected pairs.
    nodes: HashMap<i32, HashSet<(i32, i32)>>,

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
        for segment in &self.planet_segments {
            let arr_ref: &SpkArray = segment.into();
            if arr_ref.object_id == id && arr_ref.contains(jd) {
                return segment.try_get_state(jd);
            }
        }
        if let Some(segments) = self.segments.get(&id) {
            for segment in segments {
                let arr_ref: &SpkArray = segment.into();
                if arr_ref.contains(jd) {
                    return segment.try_get_state(jd);
                }
            }
        }
        Err(Error::Bounds(format!(
            "Object ({id}) does not have an SPK record for the target JD."
        )))
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
        if state.center_id != center {
            self.try_change_center(&mut state, center)?;
        }
        Ok(state)
    }

    /// Use the data loaded in the SPKs to change the center ID of the provided state.
    ///
    /// # Errors
    /// Fails when the id or jd is not found in the [`SpkCollection`].
    pub fn try_change_center<T: InertialFrame>(
        &self,
        state: &mut State<T>,
        new_center: i32,
    ) -> KeteResult<()> {
        match (state.center_id, new_center) {
            (a, b) if a == b => (),
            (i, 0) if i <= 10 => {
                state.try_change_center(self.try_get_state(i, state.epoch)?)?;
            }
            (0, 10) => {
                let next = self.try_get_state(10, state.epoch)?;
                state.try_change_center(next)?;
            }
            (i, 10) if i < 10 => {
                state.try_change_center(self.try_get_state(i, state.epoch)?)?;
                state.try_change_center(self.try_get_state(10, state.epoch)?)?;
            }
            (10, i) if (i > 1) & (i < 10) => {
                state.try_change_center(self.try_get_state(10, state.epoch)?)?;
                state.try_change_center(self.try_get_state(i, state.epoch)?)?;
            }
            _ => {
                let path = self.find_path(state.center_id, new_center)?;
                for intermediate in path {
                    let next = self.try_get_state(intermediate, state.epoch)?;
                    state.try_change_center(next)?;
                }
            }
        }
        Ok(())
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

    /// Given a NAIF ID, and a target NAIF ID, find the intermediate SPICE Segments
    /// which need to be loaded to find a path from one object to the other.
    /// Use Dijkstra plus the known segments to calculate a path.
    fn find_path(&self, start: i32, goal: i32) -> KeteResult<Vec<i32>> {
        // first we check to see if the cache contains the lookup we need.
        if let Some(path) = self.map_cache.get(&(start, goal)) {
            return Ok(path.clone());
        }

        // not in the cache, manually compute
        let nodes = &self.nodes;
        let result = dijkstra(
            &(start, i32::MIN),
            |&current| match nodes.get(&current.0) {
                Some(set) => set.iter().map(|p| (*p, 1_i32)).collect(),
                None => Vec::<((i32, i32), i32)>::new(),
            },
            |&p| p.0 == goal,
        );
        if let Some((v, _)) = result {
            Ok(v.iter().skip(1).map(|x| x.1).collect())
        } else {
            Err(Error::Bounds(format!(
                "SPK files are missing information to be able to map from obj {start} to obj {goal}"
            )))
        }
    }

    /// Return all mappings from one object to another.
    ///
    /// These mappings are used to be able to change the center ID from whatever is saved in
    /// the spks to any possible combination.
    fn build_mapping(&mut self) {
        static PRECACHE: &[i32] = &[0, 10, 399];

        fn update_nodes(segment: &SpkSegment, nodes: &mut HashMap<i32, HashSet<(i32, i32)>>) {
            let array_ref: &SpkArray = segment.into();
            if let std::collections::hash_map::Entry::Vacant(e) = nodes.entry(array_ref.object_id) {
                let mut set = HashSet::new();
                let _ = set.insert((array_ref.center_id, array_ref.object_id));
                let _ = e.insert(set);
            } else {
                let _ = nodes
                    .get_mut(&array_ref.object_id)
                    .unwrap()
                    .insert((array_ref.center_id, array_ref.object_id));
            }
            if let std::collections::hash_map::Entry::Vacant(e) = nodes.entry(array_ref.center_id) {
                let mut set = HashSet::new();
                let _ = set.insert((array_ref.object_id, array_ref.object_id));
                let _ = e.insert(set);
            } else {
                let _ = nodes
                    .get_mut(&array_ref.center_id)
                    .unwrap()
                    .insert((array_ref.object_id, array_ref.object_id));
            }
        }

        let mut nodes: HashMap<i32, HashSet<(i32, i32)>> = HashMap::new();

        self.planet_segments
            .iter()
            .for_each(|x| update_nodes(x, &mut nodes));

        for segs in self.segments.values() {
            for x in segs {
                update_nodes(x, &mut nodes);
            }
        }

        let loaded = self.loaded_objects(true);

        for &start in &loaded {
            for &goal in PRECACHE {
                let key = (start, goal);

                if self.map_cache.contains_key(&key) {
                    continue;
                }

                let result = dijkstra(
                    &(start, -100_i32),
                    |&current| match nodes.get(&current.0) {
                        Some(set) => set.iter().map(|p| (*p, 1_i32)).collect(),
                        None => Vec::<((i32, i32), i32)>::new(),
                    },
                    |&p| p.0 == goal,
                );

                if let Some((v, _)) = result {
                    let v: Vec<i32> = v.iter().skip(1).map(|x| x.1).collect();
                    let _ = self.map_cache.insert(key, v);
                }
            }
        }

        self.nodes = nodes;
    }

    /// Given an SPK filename, load all the segments present inside of it.
    /// These segments are added to the SPK singleton in memory.
    ///
    /// # Errors
    /// Loading files may fail for a number of reasons, including incorrect formatted
    /// files or IO errors.
    pub fn load_file(&mut self, filename: &str) -> KeteResult<()> {
        let file = DafFile::from_file(filename)?;

        if !matches!(file.daf_type, DAFType::Spk) {
            Err(Error::IOError(format!(
                "File {filename:?} is not a PCK formatted file."
            )))?;
        }
        for daf_array in file.arrays {
            let segment: SpkArray = daf_array.try_into()?;
            if (segment.object_id >= 0) && (segment.object_id <= 1000) {
                self.planet_segments.push(segment.try_into()?);
            } else {
                self.segments
                    .entry(segment.object_id)
                    .or_default()
                    .push(segment.try_into()?);
            }
        }
        self.build_mapping();
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
        fs::read_dir(directory)?.for_each(|entry| {
            if let Ok(entry) = entry
                && entry.path().is_file()
                && let Some(filename) = entry.path().to_str()
                && filename.to_lowercase().ends_with(".bsp")
                && let Err(err) = self.load_file(filename)
            {
                eprintln!("Failed to load SPK file {filename}: {err}");
            }
        });
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
                .map(|id| id.name.to_string())
                .collect::<Vec<String>>()
                .join(",\n")
        )))
    }
}

/// SPK singleton.
/// This is a lock protected [`SpkCollection`], and must be `.try_read().unwrapped()` for any
/// read-only cases.
pub static LOADED_SPK: std::sync::LazyLock<ShardedLock<SpkCollection>> =
    std::sync::LazyLock::new(|| {
        let mut singleton = SpkCollection::default();
        let _ = singleton.load_core();
        ShardedLock::new(singleton)
    });
