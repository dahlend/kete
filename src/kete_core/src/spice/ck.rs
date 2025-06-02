//! Loading and reading of states from JPL CK kernel files.
//!
//! PCKs are intended to be loaded into a singleton which is accessible via the
//! [`LOADED_CK`] object defined below. This singleton is wrapped in a RwLock,
//! meaning before its use it must by unwrapped. A vast majority of intended use cases
//! will only be the read case.
//!

use crate::{
    errors::{Error, KeteResult},
    time::{scales::TDB, Time},
};

use super::{ck_segments::CkSegment, CkArray, DAFType, DafFile};
use crossbeam::sync::ShardedLock;
use lazy_static::lazy_static;

/// A collection of segments.
#[derive(Debug, Default)]
pub struct CkCollection {
    /// Collection of PCK file information
    segments: Vec<CkSegment>,
}

/// Define the PCK singleton structure.
pub type CkSingleton = ShardedLock<CkCollection>;

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

    /// Get the closest record to the given JD for the specified instrument ID.
    ///
    pub fn get_record_at_time(
        &self,
        jd: f64,
        naif_id: i32,
    ) -> KeteResult<(Time<TDB>, [f64; 4], Option<[f64; 3]>)> {
        let time = Time::<TDB>::new(jd);
        return self.segments.first().unwrap().try_get_record(naif_id, time);

        // Err(Error::DAFLimits(
        //     "No segment found for the requested time.".into(),
        // ))
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
