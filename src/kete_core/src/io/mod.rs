//! File IO related tools
// BSD 3-Clause License
//
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
