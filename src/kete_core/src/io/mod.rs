//! File IO related tools

pub mod bytes;
#[cfg(feature = "polars")]
pub mod parquet;
pub mod serde_const_arr;

use crate::prelude::{Error, KeteResult};
use bincode::serde::{decode_from_std_read, encode_into_std_write};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufReader, BufWriter};

/// Support for automatic derivation of Save/Load
pub trait FileIO: Serialize
where
    for<'de> Self: Deserialize<'de>,
{
    /// Save into a binary file.
    ///
    /// Binary file formats as used by kete are not guaranteed to be stable in future
    /// versions.
    ///
    fn save(&self, filename: String) -> KeteResult<usize> {
        let mut f = BufWriter::new(File::create(filename)?);
        encode_into_std_write(self, &mut f, bincode::config::legacy())
            .map_err(|_| Error::IOError("Failed to write to file".into()))
    }

    /// Load from a binary file.
    ///
    /// Binary file formats as used by kete are not guaranteed to be stable in future
    /// versions.
    fn load(filename: String) -> KeteResult<Self> {
        let mut f = BufReader::new(File::open(filename)?);
        decode_from_std_read(&mut f, bincode::config::legacy())
            .map_err(|_| Error::IOError("Failed to read from file".into()))
    }

    /// Save a vector of this object into a binary file.
    ///
    /// Binary file formats as used by kete are not guaranteed to be stable in future
    /// versions.
    fn save_vec(vec: &[Self], filename: String) -> KeteResult<()> {
        let mut f = BufWriter::new(File::create(filename)?);

        let _ = encode_into_std_write(vec, &mut f, bincode::config::legacy())
            .map_err(|_| Error::IOError("Failed to write to file".into()))?;
        Ok(())
    }

    /// load a vector of this object into a binary file.
    ///
    /// Binary file formats as used by kete are not guaranteed to be stable in future
    /// versions.
    fn load_vec(filename: String) -> KeteResult<Vec<Self>> {
        let mut f = BufReader::new(File::open(filename)?);

        let res: Vec<Self> = decode_from_std_read(&mut f, bincode::config::legacy())
            .map_err(|_| Error::IOError("Failed to load from file".into()))?;
        Ok(res)
    }
}
