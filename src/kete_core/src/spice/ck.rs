//! Loading and reading of states from JPL CK kernel files.
//!
//! PCKs are intended to be loaded into a singleton which is accessible via the
//! [`LOADED_CK`] object defined below. This singleton is wrapped in a
//! [`crossbeam::sync::ShardedLock`], meaning before its use it must by unwrapped.
//! A vast majority of intended use cases will only be the read case.
//!

use crate::{
    errors::{Error, KeteResult},
    frames::NonInertialFrame,
    time::{TDB, Time},
};

use super::{CkArray, DAFType, DafFile, LOADED_SCLK, ck_segments::CkSegment};
use crossbeam::sync::ShardedLock;
use lazy_static::lazy_static;

/// A collection of segments.
#[derive(Debug, Default)]
pub struct CkCollection {
    /// Collection of PCK file information
    pub(crate) segments: Vec<CkSegment>,
}

/// Define the PCK singleton structure.
type CkSingleton = ShardedLock<CkCollection>;

impl CkCollection {
    /// Given an CK filename, load all the segments present inside of it.
    /// These segments are added to the PCK singleton in memory.
    pub fn load_file(&mut self, filename: &str) -> KeteResult<()> {
        let file = DafFile::from_file(filename)?;
        if !matches!(file.daf_type, DAFType::Ck) {
            return Err(Error::IOError(format!(
                "File {:?} is not aPCK formatted file.",
                filename
            )))?;
        }

        for array in file.arrays {
            let pck_array: CkArray = array.try_into()?;
            let segment: CkSegment = pck_array.try_into()?;
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
    pub fn try_get_frame(
        &self,
        jd: f64,
        instrument_id: i32,
    ) -> KeteResult<(Time<TDB>, NonInertialFrame)> {
        let time = Time::<TDB>::new(jd);
        let sclk = LOADED_SCLK.try_read().unwrap();
        let spice_id = instrument_id / 1000;
        let tick = sclk.try_time_to_tick(spice_id, time)?;

        for segment in self.segments.iter() {
            let array: &CkArray = segment.into();
            if (array.instrument_id == instrument_id) & array.contains(tick) {
                return segment.try_get_orientation(instrument_id, time);
            }
        }

        Err(Error::DAFLimits(format!(
            "Instrument ({}) does not have an CK record for the target JD.",
            instrument_id
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
                if array.instrument_id != instrument_id {
                    None
                } else {
                    Some((
                        array.instrument_id,
                        array.reference_frame_id,
                        array.segment_type,
                        array.tick_start,
                        array.tick_end,
                    ))
                }
            })
            .collect()
    }
}

lazy_static! {
    /// PCK singleton.
    /// This is a RwLock protected PCKCollection, and must be `.try_read().unwrapped()` for any
    /// read-only cases.
    pub static ref LOADED_CK: CkSingleton = {
        let singleton = CkCollection::default();
        ShardedLock::new(singleton)
    };
}
