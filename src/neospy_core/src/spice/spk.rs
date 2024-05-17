/// Loading and reading of states from JPL SPK kernel files.
///
/// SPKs are intended to be loaded into a singleton which is accessible via the
/// [`get_spk_singleton`] function defined below. This singleton is wrapped in a RwLock,
/// meaning before its use it must by unwrapped. A vast majority of intended use cases
/// will only be the read case.
///
/// Here is a small worked example:
/// ```
///     use neospy_core::spice::get_spk_singleton;
///     use neospy_core::frames::Frame;
///
///     // get a read-only reference to the [`SegmentCollection`]
///     let singleton = get_spk_singleton().try_read().unwrap();
///
///     // get the state of 399 (Earth) with respect to the Sun (10)
///     let state = singleton.try_get_state(399, 2451545.0, 10, Frame::Ecliptic);
/// ```
///
///
use super::daf::{DAFType, Daf};
use super::spk_segments::*;
use crate::errors::NEOSpyError;
use crate::frames::Frame;
use crate::state::State;
use pathfinding::prelude::dijkstra;
use std::collections::{HashMap, HashSet};

use crossbeam::sync::ShardedLock;
use std::io::{Cursor, Read, Seek};
use std::mem::MaybeUninit;
use std::sync::Once;

const PRELOAD_SPKS: &[&[u8]] = &[
    include_bytes!("../../data/de440s.bsp"),
    include_bytes!("../../data/wise.bsp"),
    include_bytes!("../../data/20000001.bsp"),
    include_bytes!("../../data/20000002.bsp"),
    include_bytes!("../../data/20000004.bsp"),
    include_bytes!("../../data/20000010.bsp"),
    include_bytes!("../../data/20000704.bsp"),
];

/// A collection of segments.
#[derive(Debug)]
pub struct SpkSegmentCollection {
    segments: Vec<SpkSegment>,
    map_cache: HashMap<(isize, isize), Vec<isize>>,

    /// Map from object id to all connected pairs.
    nodes: HashMap<isize, HashSet<(isize, isize)>>,
}

/// Define the SPK singleton structure.
pub type SpkSingleton = ShardedLock<SpkSegmentCollection>;

impl SpkSegmentCollection {
    /// Get the raw state from the loaded SPK files.
    /// This state will have the center and frame of whatever was originally loaded
    /// into the file.
    pub fn try_get_raw_state(&self, id: isize, jd: f64) -> Result<State, NEOSpyError> {
        for segment in self.segments.iter() {
            if id == segment.obj_id && segment.contains(jd) {
                return segment.try_get_state(jd);
            }
        }
        Err(NEOSpyError::DAFLimits(
            format!(
                "Object ({}) does not have an SPK record for the target JD.",
                id
            )
            .to_string(),
        ))
    }

    /// For a given NAIF ID, return all increments of time which are currently loaded.
    pub fn available_info(&self, id: isize) -> Vec<(f64, f64, isize, Frame, usize)> {
        let mut segment_info = Vec::<(f64, f64, isize, Frame, usize)>::new();
        for segment in self.segments.iter() {
            if id == segment.obj_id {
                let jd_range = segment.jd_range();
                segment_info.push((
                    jd_range.0,
                    jd_range.1,
                    segment.center_id,
                    segment.ref_frame,
                    segment.segment_type,
                ))
            }
        }
        if segment_info.is_empty() {
            return segment_info;
        }

        segment_info.sort_by(|a, b| (a.0).total_cmp(&b.0));

        let mut avail_times = Vec::<(f64, f64, isize, Frame, usize)>::new();

        let mut cur_segment = segment_info[0];
        for segment in segment_info.iter().skip(1) {
            // if the segments are overlapped or nearly overlapped, join them together
            // 1e-8 is approximately a millisecond
            if cur_segment.1 <= (segment.0 - 1e-8) {
                avail_times.push(cur_segment);
                cur_segment = *segment;
            } else {
                cur_segment.1 = segment.1.max(cur_segment.1)
            }
        }
        avail_times.push(cur_segment);

        avail_times
    }

    /// Load a state from the file, then attempt to change the center to the center id
    /// specified.
    pub fn try_get_state(
        &self,
        id: isize,
        jd: f64,
        center: isize,
        frame: Frame,
    ) -> Result<State, NEOSpyError> {
        let mut state = self.try_get_raw_state(id, jd)?;
        self.try_change_center(&mut state, center)?;
        state.try_change_frame_mut(frame)?;
        Ok(state)
    }

    /// Use the data loaded in the SPKs to change the center ID of the provided state.
    pub fn try_change_center(
        &self,
        state: &mut State,
        new_center: isize,
    ) -> Result<(), NEOSpyError> {
        if state.center_id == new_center {
            return Ok(());
        }

        let path = self.find_path(state.center_id, new_center)?;

        for intermediate in path {
            let next = self.try_get_raw_state(intermediate, state.jd)?;
            state.try_change_center(next)?;
        }
        Ok(())
    }

    /// Return a hash set of all unique identifies loaded in the SPKs.
    /// If include centers is true, then this additionally includes the IDs for the
    /// center IDs. For example, if include_centers is false, then `0` will never be
    /// included in the loaded objects set, as 0 is a privileged position at the
    /// barycenter of the solar system. It is not typically defined in relation to
    /// anything else.
    pub fn loaded_objects(&self, include_centers: bool) -> HashSet<isize> {
        let mut found = HashSet::new();

        for segment in self.segments.iter() {
            let _ = found.insert(segment.obj_id);
            if include_centers {
                let _ = found.insert(segment.center_id);
            }
        }
        found
    }

    /// Given a NAIF ID, and a target NAIF ID, find the intermediate SPICE Segments
    /// which need to be loaded to find a path from one object to the other.
    /// Use Dijkstra plus the known segments to calculate a path.
    fn find_path(&self, start: isize, goal: isize) -> Result<Vec<isize>, NEOSpyError> {
        // first we check to see if the cache contains the lookup we need.
        if let Some(path) = self.map_cache.get(&(start, goal)) {
            return Ok(path.clone());
        }

        // not in the cache, manually compute
        let nodes = &self.nodes;
        let result = dijkstra(
            &(start, isize::MIN),
            |&current| match nodes.get(&current.0) {
                Some(set) => set.iter().map(|p| (*p, 1_isize)).collect(),
                None => Vec::<((isize, isize), isize)>::new(),
            },
            |&p| p.0 == goal,
        );

        if let Some((v, _)) = result {
            Ok(v.iter().skip(1).map(|x| x.1).collect())
        } else {
            Err(NEOSpyError::DAFLimits(format!(
                "SPK files are missing information to be able to map from obj {} to obj {}",
                start, goal
            )))
        }
    }

    /// Given an SPK filename, load all the segments present inside of it.
    /// These segments are added to the SPK singleton in memory.
    pub fn load_file(&mut self, filename: &str) -> Result<Daf, NEOSpyError> {
        let mut file = std::fs::File::open(filename)?;
        let mut buffer = Vec::new();
        let _ = file.read_to_end(&mut buffer)?;
        let mut buffer = Cursor::new(&buffer);

        let daf = self.load_segments(&mut buffer)?;
        Ok(daf)
    }

    /// Given a reference to a buffer, load all the segments present inside of it.
    /// These segments are added to the SPK singleton in memory.
    pub fn load_segments<T: Read + Seek>(&mut self, mut buffer: T) -> Result<Daf, NEOSpyError> {
        let daf = Daf::try_load_header(&mut buffer)?;
        if daf.daf_type != DAFType::Spk {
            return Err(NEOSpyError::IOError(
                "Attempted to load a DAF file which is not an SPK as an SPK.".into(),
            ));
        }

        let summaries = daf.try_load_summaries(&mut buffer)?;

        for summary in summaries {
            self.segments
                .push(SpkSegment::from_summary(&mut buffer, summary)?);
        }

        Ok(daf)
    }

    /// Return all mappings from one object to another.
    ///
    /// These mappings are used to be able to change the center ID from whatever is saved in
    /// the spks to any possible combination.
    pub fn build_cache(&mut self) {
        static PRECACHE: &[isize] = &[0, 10, 399];

        let mut nodes: HashMap<isize, HashSet<(isize, isize)>> = HashMap::new();

        fn update_nodes(segment: &SpkSegment, nodes: &mut HashMap<isize, HashSet<(isize, isize)>>) {
            if let std::collections::hash_map::Entry::Vacant(e) = nodes.entry(segment.obj_id) {
                let mut set = HashSet::new();
                let _ = set.insert((segment.center_id, segment.obj_id));
                let _ = e.insert(set);
            } else {
                let _ = nodes
                    .get_mut(&segment.obj_id)
                    .unwrap()
                    .insert((segment.center_id, segment.obj_id));
            }
            if let std::collections::hash_map::Entry::Vacant(e) = nodes.entry(segment.center_id) {
                let mut set = HashSet::new();
                let _ = set.insert((segment.obj_id, segment.obj_id));
                let _ = e.insert(set);
            } else {
                let _ = nodes
                    .get_mut(&segment.center_id)
                    .unwrap()
                    .insert((segment.obj_id, segment.obj_id));
            }
        }

        for segment in self.segments.iter() {
            update_nodes(segment, &mut nodes);
        }

        let loaded = self.loaded_objects(true);

        for &start in loaded.iter() {
            for &goal in PRECACHE {
                let key = (start, goal);

                if self.map_cache.contains_key(&key) {
                    continue;
                }

                let result = dijkstra(
                    &(start, -100_isize),
                    |&current| match nodes.get(&current.0) {
                        Some(set) => set.iter().map(|p| (*p, 1_isize)).collect(),
                        None => Vec::<((isize, isize), isize)>::new(),
                    },
                    |&p| p.0 == goal,
                );

                if let Some((v, _)) = result {
                    let v: Vec<isize> = v.iter().skip(1).map(|x| x.1).collect();
                    let _ = self.map_cache.insert(key, v);
                }
            }
        }

        self.nodes = nodes;
    }

    /// Delete all segments in the SPK singleton, equivalent to unloading all files.
    pub fn reset(&mut self) {
        let segments: SpkSegmentCollection = SpkSegmentCollection {
            segments: Vec::new(),
            map_cache: HashMap::new(),
            nodes: HashMap::new(),
        };

        *self = segments;

        for preload in PRELOAD_SPKS {
            let mut de440 = Cursor::new(preload);
            let _ = self.load_segments(&mut de440).unwrap();
        }
        self.build_cache();
    }
}

/// Get the SPK singleton.
/// This is a RwLock protected SPKCollection, and must be `.try_read().unwrapped()` for any
/// read-only cases.
///
/// This singleton starts initialized with preloaded SPK files for the planets.
pub fn get_spk_singleton() -> &'static SpkSingleton {
    // Create an uninitialized static
    static mut SINGLETON: MaybeUninit<SpkSingleton> = MaybeUninit::uninit();
    static ONCE: Once = Once::new();

    unsafe {
        ONCE.call_once(|| {
            let mut segments: SpkSegmentCollection = SpkSegmentCollection {
                segments: Vec::new(),
                map_cache: HashMap::new(),
                nodes: HashMap::new(),
            };
            segments.reset();
            let singleton: SpkSingleton = ShardedLock::new(segments);
            // Store it to the static var, i.e. initialize it
            let _ = SINGLETON.write(singleton);
        });

        // Now we give out a shared reference to the data, which is safe to use
        // concurrently.
        SINGLETON.assume_init_ref()
    }
}
