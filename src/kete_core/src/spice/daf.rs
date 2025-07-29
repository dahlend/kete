//! Support for arbitrary DAF files
//! DAF is a superset which includes SPK and PCK files.
//!
//! DAF files are laid out in 1024 Byte "Records"
//! - The first record is header information about the contents of the file.
//! - The following N records are text comments.
//! - Immediately following the comments there is a Summary Record.
//!
//! These summary records contain the location information for all the contents
//! of the DAF file.
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

use crate::io::bytes::{
    bytes_to_f64, bytes_to_f64_vec, bytes_to_i32, bytes_to_i32_vec, bytes_to_string,
    read_bytes_exact, read_f64_vec, read_str,
};

use crate::errors::{Error, KeteResult};
use std::fmt::Debug;
use std::io::{Cursor, Read, Seek};
use std::ops::Index;
use std::slice::SliceIndex;

use super::jd_to_spice_jd;

/// DAF Files can contain multiple different types of data.
/// This list contains the supported formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DAFType {
    /// An unrecognized DAF type.
    Unrecognized([u8; 3]),

    /// SPK files are planetary and satellite ephemeris data.
    Spk,

    /// PCK Files are planetary and satellite orientation data.
    Pck,

    /// CK files define instrument orientation data.
    Ck,
}

impl From<&str> for DAFType {
    fn from(magic: &str) -> Self {
        match &magic.to_uppercase()[4..7] {
            "SPK" => Self::Spk,
            "PCK" => Self::Pck,
            "CK " => Self::Ck,
            other => Self::Unrecognized(other.as_bytes().try_into().unwrap()),
        }
    }
}

/// DAF files header information.
/// This contains
#[derive(Debug)]
pub struct DafFile {
    /// Magic number within the DAF file corresponds to this DAF type.
    pub daf_type: DAFType,

    /// Number of f64 in each array.
    pub n_doubles: i32,

    /// Number of i32s in each array.
    pub n_ints: i32,

    /// Number of chars in the descriptor string of each array.
    pub n_chars: i32,

    /// Is the file little endian.
    pub little_endian: bool,

    /// Internal Descriptor.
    pub internal_desc: String,

    /// Index of initial summary record.
    /// Note that this is 1 indexed and corresponds to record index
    /// not file byte index.
    pub init_summary_record_index: i32,

    /// Index of final summary record.
    /// Note that this is 1 indexed and corresponds to record index
    /// not file byte index.
    pub final_summary_record_index: i32,

    /// First free address of the file.
    /// Index of initial summary record
    /// Note that this is 1 indexed.
    pub first_free: i32,

    /// FTP Validation string
    pub ftp_validation_str: String,

    /// The comment records.
    /// Each record is trimmed to 1000 chars, as that is what SPKs use internally.
    pub comments: String,

    /// DAF Arrays contained within this file.
    pub arrays: Vec<DafArray>,
}

impl DafFile {
    /// Try to load a single record from the DAF.
    pub fn try_load_record<T: Read + Seek>(file: &mut T, idx: u64) -> KeteResult<Box<[u8]>> {
        let _ = file.seek(std::io::SeekFrom::Start(1024 * (idx - 1)))?;
        read_bytes_exact(file, 1024)
    }

    /// Load the contents of a DAF file.
    pub fn from_buffer<T: Read + Seek>(mut buffer: T) -> KeteResult<Self> {
        let bytes = Self::try_load_record(&mut buffer, 1)?;
        let daf_type: DAFType = bytes_to_string(&bytes[0..8]).as_str().into();

        let little_endian = match bytes_to_string(&bytes[88..96]).to_lowercase().as_str() {
            "ltl-ieee" => true,
            "big-ieee" => false,
            _ => Err(Error::IOError(
                "Expected little or big endian in DAF file, found neither".into(),
            ))?,
        };

        let n_doubles = bytes_to_i32(&bytes[8..12], little_endian)?;
        let n_ints = bytes_to_i32(&bytes[12..16], little_endian)?;
        let n_chars = 8 * (n_doubles + (n_ints + 1) / 2);

        // record index of the first summary record in the file
        // records are 1024 long, and 1 indexed because fortran.
        let init_summary_record_index = bytes_to_i32(&bytes[76..80], little_endian)?;

        // the following values are not used, so are not stored.
        let internal_desc = bytes_to_string(&bytes[16..76]);
        let final_summary_record_index = bytes_to_i32(&bytes[80..84], little_endian)?;
        let first_free = bytes_to_i32(&bytes[84..88], little_endian)?;

        let ftp_validation_str = bytes_to_string(&bytes[966..966 + 28]);

        // after the header, there are comments until the first record index.
        // so read the next (init_summary_record_index-2) records:
        // -1 for fortran indexing
        // -1 for having already read a single record
        let mut comments: Vec<String> = Vec::with_capacity(init_summary_record_index as usize - 2);
        for _ in 0..(init_summary_record_index - 2) {
            // TODO: Check if the 1000 character limit is what other formats use.
            // 1k is used by SPK for sure.
            comments.push(read_str(&mut buffer, 1024)?.chars().take(1000).collect());
        }

        let mut daf = Self {
            daf_type,
            n_doubles,
            n_ints,
            n_chars,
            little_endian,
            internal_desc,
            init_summary_record_index,
            final_summary_record_index,
            first_free,
            ftp_validation_str,
            comments: comments.join(""),
            arrays: Vec::new(),
        };

        daf.try_load_arrays(&mut buffer)?;
        Ok(daf)
    }

    /// Load DAF file from the specified filename.
    pub fn from_file(filename: &str) -> KeteResult<Self> {
        let mut file = std::fs::File::open(filename)?;
        let mut buffer = Vec::new();
        let _ = file.read_to_end(&mut buffer)?;
        let mut buffer = Cursor::new(&buffer);
        Self::from_buffer(&mut buffer)
    }

    /// Load all [`DafArray`] segments from the DAF file.
    /// These are tuples containing a series of f64s and i32s along with arrays of data.
    /// The meaning of these values depends on the particular implementation of the DAF.
    ///
    pub fn try_load_arrays<T: Read + Seek>(&mut self, file: &mut T) -> KeteResult<()> {
        let summary_size = self.n_doubles + (self.n_ints + 1) / 2;

        let mut next_idx = self.init_summary_record_index;
        loop {
            if next_idx == 0 {
                break;
            }
            let bytes = Self::try_load_record(file, next_idx as u64)?;

            next_idx = bytes_to_f64(&bytes[0..8], self.little_endian)? as i32;
            // let prev_idx = bytes_to_f64(&bytes[8..16], daf.little_endian)? as i32;
            let n_summaries = bytes_to_f64(&bytes[16..24], self.little_endian)? as i32;

            for idy in 0..n_summaries {
                let sum_start = (3 * 8 + idy * summary_size * 8) as usize;
                let floats = bytes_to_f64_vec(
                    &bytes[sum_start..sum_start + 8 * self.n_doubles as usize],
                    self.little_endian,
                )?;
                let ints = bytes_to_i32_vec(
                    &bytes[sum_start + 8 * self.n_doubles as usize
                        ..sum_start + (8 * self.n_doubles + 4 * self.n_ints) as usize],
                    self.little_endian,
                )?;

                let array = DafArray::try_load_array(
                    file,
                    floats,
                    ints,
                    self.daf_type,
                    self.little_endian,
                )?;
                self.arrays.push(array);
            }
        }
        Ok(())
    }
}

/// DAF Arrays are f64 arrays of structured data.
///
/// Contents of the structure depends on specific file formats, however they are all
/// made up of floats.
pub struct DafArray {
    /// [`DafArray`] segment summary float information.
    pub summary_floats: Box<[f64]>,

    /// [`DafArray`] segment summary int information.
    pub summary_ints: Box<[i32]>,

    /// Data contained within the array.
    pub data: Box<[f64]>,

    /// The type of DAF array.
    pub daf_type: DAFType,
}

impl Debug for DafArray {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("DafArray({} values)", self.data.len()))
    }
}

impl DafArray {
    /// Try to load an DAF array from summary data
    pub fn try_load_array<T: Read + Seek>(
        buffer: &mut T,
        summary_floats: Box<[f64]>,
        summary_ints: Box<[i32]>,
        daf_type: DAFType,
        little_endian: bool,
    ) -> KeteResult<Self> {
        let n_ints = summary_ints.len();
        if n_ints < 2 {
            Err(Error::IOError("DAF File incorrectly Formatted.".into()))?;
        }

        // From DAF documentation:
        // "The initial and final addresses of an array are always the values of the
        //  final two integer components of the summary for the array. "
        let array_start = summary_ints[n_ints - 2] as u64;
        let array_end = summary_ints[n_ints - 1] as u64;

        if array_end < array_start {
            Err(Error::IOError("DAF File incorrectly Formatted.".into()))?;
        }

        let _ = buffer.seek(std::io::SeekFrom::Start(8 * (array_start - 1)))?;

        let n_floats = (array_end - array_start + 1) as usize;

        let data = read_f64_vec(buffer, n_floats, little_endian)?;

        Ok(Self {
            summary_floats,
            summary_ints,
            data,
            daf_type,
        })
    }

    /// Total length of the array.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Test if array is empty.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

impl<Idx> Index<Idx> for DafArray
where
    Idx: SliceIndex<[f64], Output = f64>,
{
    type Output = f64;

    fn index(&self, idx: Idx) -> &Self::Output {
        self.data.index(idx)
    }
}

/// DAF Array of SPK data.
/// This is a wrapper around the [`DafArray`] which is specific to SPK data.
///
#[derive(Debug)]
pub struct SpkArray {
    /// The internal representation of the DAF array.
    pub daf: DafArray,

    /// JD Time in spice units of seconds from J2000.
    pub jds_start: f64,

    /// JD Time in spice units of seconds from J2000.
    pub jds_end: f64,

    /// The reference NAIF ID for the object in this Array.
    pub object_id: i32,

    /// The reference center NAIF ID for the central body in this Array.
    pub center_id: i32,

    /// The spice frame ID of the array.
    pub frame_id: i32,

    /// The spice segment type.
    pub segment_type: i32,
}

impl SpkArray {
    /// Is the specified JD within the range of this array.
    pub fn contains(&self, jd: f64) -> bool {
        let jds = jd_to_spice_jd(jd);
        (jds >= self.jds_start) && (jds <= self.jds_end)
    }
}

impl TryFrom<DafArray> for SpkArray {
    type Error = Error;

    fn try_from(array: DafArray) -> Result<Self, Self::Error> {
        if array.daf_type != DAFType::Spk {
            Err(Error::IOError("DAF Array is not a SPK array.".into()))?;
        }

        if array.summary_floats.len() != 2 {
            Err(Error::IOError(
                "DAF Array is not a SPK array. Summary of array is incorrectly formatted, incorrect number of floats.".into(),
            ))?;
        }

        if array.summary_ints.len() != 6 {
            Err(Error::IOError("DAF Array is not a SPK array. Summary of array is incorrectly formatted, incorrect number of ints.".into()))?;
        }

        let jds_start = array.summary_floats[0];
        let jds_end = array.summary_floats[1];

        // The last two integers in the summary are the start and end of the array.
        let object_id = array.summary_ints[0];
        let center_id = array.summary_ints[1];
        let frame_id = array.summary_ints[2];
        let segment_type = array.summary_ints[3];

        Ok(Self {
            daf: array,
            jds_start,
            jds_end,
            object_id,
            center_id,
            frame_id,
            segment_type,
        })
    }
}

#[derive(Debug)]

/// DAF Array of PCK data.
/// This is a wrapper around the [`DafArray`] which is specific to PCK data.
pub struct PckArray {
    /// The internal representation of the DAF array.
    pub daf: DafArray,

    /// JD Time in spice units of seconds from J2000.
    pub jds_start: f64,

    /// JD Time in spice units of seconds from J2000.
    pub jds_end: f64,

    /// The ID which identifies this frame.
    pub frame_id: i32,

    /// The inertial reference frame this PCK is defined against.
    pub reference_frame_id: i32,

    /// The spice segment type.
    pub segment_type: i32,
}

impl PckArray {
    /// Is the specified JD within the range of this array.
    pub fn contains(&self, jd: f64) -> bool {
        let jds = jd_to_spice_jd(jd);
        (jds >= self.jds_start) && (jds <= self.jds_end)
    }
}

impl TryFrom<DafArray> for PckArray {
    type Error = Error;

    fn try_from(array: DafArray) -> Result<Self, Self::Error> {
        if array.daf_type != DAFType::Pck {
            Err(Error::IOError("DAF Array is not a PCK array.".into()))?;
        }

        if array.summary_floats.len() != 2 {
            Err(Error::IOError(
                "DAF Array is not a PCK array. Summary of array is incorrectly formatted, incorrect number of floats.".into(),
            ))?;
        }

        if array.summary_ints.len() != 5 {
            Err(Error::IOError("DAF Array is not a PCK array. Summary of array is incorrectly formatted, incorrect number of ints.".into()))?;
        }

        let jds_start = array.summary_floats[0];
        let jds_end = array.summary_floats[1];

        // The last two integers in the summary are the start and end of the array.
        let frame_id = array.summary_ints[0];
        let reference_frame_id = array.summary_ints[1];
        let segment_type = array.summary_ints[2];

        Ok(Self {
            daf: array,
            jds_start,
            jds_end,
            frame_id,
            reference_frame_id,
            segment_type,
        })
    }
}

/// DAF Array of CK data.
/// These are segments of data.
/// This is a wrapper around the [`DafArray`] which is specific to CK data.
#[derive(Debug)]
pub struct CkArray {
    /// The internal representation of the DAF array.
    pub daf: DafArray,

    /// Start SCLK tick time of the spacecraft.
    pub tick_start: f64,

    /// End SCLK tick time of the spacecraft.
    pub tick_end: f64,

    /// Instrument ID
    pub instrument_id: i32,

    /// NAIF ID of the spacecraft.
    pub naif_id: i32,

    /// The spice frame ID of the array.
    /// Called the `Reference` in SPICE documentation.
    pub reference_frame_id: i32,

    /// The spice segment type.
    pub segment_type: i32,

    /// Does this segment produce angular rates.
    pub produces_angular_rates: bool,
}

impl CkArray {
    /// Is the specified SCLK tick within the range of this array.
    pub fn contains(&self, tick: f64) -> bool {
        (tick >= self.tick_start) && (tick <= self.tick_end)
    }
}

impl TryFrom<DafArray> for CkArray {
    type Error = Error;

    fn try_from(array: DafArray) -> Result<Self, Self::Error> {
        if array.daf_type != DAFType::Ck {
            return Err(Error::IOError("DAF Array is not a CK array.".into()));
        }

        if array.summary_floats.len() != 2 {
            return Err(Error::IOError(
                "DAF Array is not a CK array. Summary of array is incorrectly formatted, incorrect number of floats.".into(),
            ));
        }

        if array.summary_ints.len() != 6 {
            return Err(Error::IOError("DAF Array is not a CK array. Summary of array is incorrectly formatted, incorrect number of ints.".into()));
        }

        let tick_start = array.summary_floats[0];
        let tick_end = array.summary_floats[1];

        // The last two integers in the summary are the start and end of the array.
        // Those two values are already contained within the DafArray stored in this
        // object.
        let instrument_id = array.summary_ints[0];
        let naif_id = array.summary_ints[0] / 1000;
        let frame_id = array.summary_ints[1];
        let segment_type = array.summary_ints[2];
        let produces_angular_rates = array.summary_ints[3] == 1;

        Ok(Self {
            daf: array,
            tick_start,
            tick_end,
            instrument_id,
            naif_id,
            reference_frame_id: frame_id,
            segment_type,
            produces_angular_rates,
        })
    }
}
