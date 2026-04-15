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

use kete_core::io::bytes::{
    bytes_to_f64, bytes_to_f64_vec, bytes_to_i32, bytes_to_i32_vec, bytes_to_string,
    f64_vec_to_bytes, i32_to_bytes, i32_vec_to_bytes, read_bytes_exact, read_f64_vec,
    string_to_padded_bytes,
};

use kete_core::errors::{Error, KeteResult};
use std::fmt::Debug;
use std::io::{Cursor, Read, Seek, Write};
use std::ops::Index;
use std::slice::SliceIndex;

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

impl DAFType {
    /// Return the 8-byte DAF file magic header for this type.
    fn magic_bytes(self) -> [u8; 8] {
        match self {
            Self::Spk => *b"DAF/SPK ",
            Self::Pck => *b"DAF/PCK ",
            Self::Ck => *b"DAF/CK  ",
            Self::Unrecognized(x) => {
                let mut buf = *b"DAF/    ";
                buf[4..7].copy_from_slice(&x);
                buf
            }
        }
    }
}

impl TryFrom<&str> for DAFType {
    type Error = Error;

    fn try_from(magic: &str) -> KeteResult<Self> {
        match magic
            .to_uppercase()
            .get(4..7)
            .ok_or(Error::IOError("DAF Magic number not long enough".into()))?
        {
            "SPK" => Ok(Self::Spk),
            "PCK" => Ok(Self::Pck),
            "CK " => Ok(Self::Ck),
            other => Ok(Self::Unrecognized(other.as_bytes().try_into().unwrap())),
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
    /// Reconstructed from the DAF comment area (a flat byte stream across records,
    /// 1000 content bytes per 1024-byte record, \x00 line separators -> newlines,
    /// \x04 EOT marker stripped).
    pub comments: String,

    /// Raw comment record bytes from the original file (1024 bytes per record).
    /// Used for bit-exact reproduction when writing back a file that was read
    /// from disk.
    raw_comment_records: Vec<Box<[u8]>>,

    /// Raw summary/name record pairs from the original file.
    /// Each entry is (`summary_record_bytes`, `name_record_bytes`).
    /// Used for bit-exact reproduction of trailing/padding bytes.
    raw_record_pairs: Vec<(Box<[u8]>, Box<[u8]>)>,

    /// Trailing bytes beyond the computed data boundary in the original file.
    /// Some CSPICE tools leave extra records at the end.
    trailing_bytes: Vec<u8>,

    /// DAF Arrays contained within this file.
    pub arrays: Vec<DafArray>,
}

impl DafFile {
    /// Try to load a single record from the DAF.
    ///
    /// # Errors
    /// [`Error::IOError`] if a seek fails in the file.
    pub fn try_load_record<T: Read + Seek>(file: &mut T, idx: u64) -> KeteResult<Box<[u8]>> {
        let _ = file.seek(std::io::SeekFrom::Start(1024 * (idx - 1)))?;
        read_bytes_exact(file, 1024)
    }

    /// Load the contents of a DAF file.
    ///
    /// # Errors
    /// Fails if there are read errors, or the file is incorrect in some way.
    pub fn from_buffer<T: Read + Seek>(mut buffer: T) -> KeteResult<Self> {
        let bytes = Self::try_load_record(&mut buffer, 1)?;
        let daf_type: DAFType = bytes_to_string(&bytes[0..8]).as_str().try_into()?;

        let little_endian = match bytes_to_string(&bytes[88..96]).to_lowercase().as_str() {
            "ltl-ieee" => true,
            "big-ieee" => false,
            _ => Err(Error::IOError(
                "Expected little or big endian in DAF file, found neither".into(),
            ))?,
        };
        if !little_endian {
            // Swap in memory and re-parse as little-endian.
            let _ = buffer.seek(std::io::SeekFrom::Start(0))?;
            let mut raw = Vec::new();
            let _ = buffer.read_to_end(&mut raw)?;
            let swapped = swap_daf_be_to_le(&raw)?;
            return Self::from_buffer(Cursor::new(swapped));
        }

        let n_doubles = bytes_to_i32(&bytes[8..12])?;
        let n_ints = bytes_to_i32(&bytes[12..16])?;
        let n_chars = 8 * (n_doubles + (n_ints + 1) / 2);

        // record index of the first summary record in the file
        // records are 1024 long, and 1 indexed because fortran.
        let init_summary_record_index = bytes_to_i32(&bytes[76..80])?.abs();

        // the following values are not used, so are not stored.
        let internal_desc = bytes_to_string(&bytes[16..76]);
        let final_summary_record_index = bytes_to_i32(&bytes[80..84])?;
        let first_free = bytes_to_i32(&bytes[84..88])?;

        let ftp_validation_str = bytes_to_string(&bytes[699..699 + 28]);

        // after the header, there are comments until the first record index.
        // so read the next (init_summary_record_index-2) records:
        // -1 for fortran indexing
        // -1 for having already read a single record
        #[allow(clippy::cast_sign_loss, reason = "known to be positive")]
        let n_comment_recs = (init_summary_record_index - 2) as usize;
        let mut raw_comment_records: Vec<Box<[u8]>> = Vec::with_capacity(n_comment_recs);
        let mut comments: Vec<String> = Vec::with_capacity(n_comment_recs);
        for _ in 0..n_comment_recs {
            let rec = read_bytes_exact(&mut buffer, 1024)?;
            // Only the first 1000 bytes of each 1024-byte record carry comment
            // content; the remaining 24 bytes are reserved for Fortran
            // direct-access record overhead and must be ignored.
            comments.push(bytes_to_string(&rec[..1000]));
            raw_comment_records.push(rec);
        }

        // Comment records use \x04 (EOT) as a terminator and \x00 as line
        // separators (converted to newlines by bytes_to_string). Strip
        // the EOT marker and any trailing space padding that follows it.
        let mut joined_comments = comments.join("");
        if let Some(eot_pos) = joined_comments.find('\x04') {
            joined_comments.truncate(eot_pos);
        }
        let joined_comments = joined_comments.clone();

        let mut daf = Self {
            daf_type,
            n_doubles,
            n_ints,
            n_chars,
            internal_desc,
            init_summary_record_index,
            final_summary_record_index,
            first_free,
            ftp_validation_str,
            comments: joined_comments,
            raw_comment_records,
            raw_record_pairs: Vec::new(),
            trailing_bytes: Vec::new(),
            arrays: Vec::new(),
        };

        daf.try_load_arrays(&mut buffer)?;

        // Capture any trailing bytes beyond the computed data boundary.
        // Some CSPICE tools leave extra records at the end of the file.
        if first_free > 1 {
            #[allow(
                clippy::cast_sign_loss,
                reason = "first_free > 1 guaranteed by enclosing if"
            )]
            let last_data_record = (first_free as u64 - 1).div_ceil(128);
            let expected_end = last_data_record * 1024;
            let file_end = buffer.seek(std::io::SeekFrom::End(0))?;
            if file_end > expected_end {
                let _ = buffer.seek(std::io::SeekFrom::Start(expected_end))?;
                let trail_len = (file_end - expected_end) as usize;
                daf.trailing_bytes = read_bytes_exact(&mut buffer, trail_len)?.to_vec();
            }
        }

        Ok(daf)
    }

    /// Construct a new empty DAF file for writing SPK data.
    #[must_use]
    pub fn new_spk(internal_desc: &str, comment: &str) -> Self {
        Self {
            daf_type: DAFType::Spk,
            n_doubles: 2,
            n_ints: 6,
            n_chars: 40,
            internal_desc: internal_desc.to_string(),
            init_summary_record_index: 0,
            final_summary_record_index: 0,
            first_free: 0,
            ftp_validation_str: String::new(),
            comments: comment.to_string(),
            raw_comment_records: Vec::new(),
            raw_record_pairs: Vec::new(),
            trailing_bytes: Vec::new(),
            arrays: Vec::new(),
        }
    }

    /// Construct a new empty DAF file for writing PCK data.
    #[must_use]
    pub fn new_pck(internal_desc: &str, comment: &str) -> Self {
        Self {
            daf_type: DAFType::Pck,
            n_doubles: 2,
            n_ints: 5,
            n_chars: 40,
            internal_desc: internal_desc.to_string(),
            init_summary_record_index: 0,
            final_summary_record_index: 0,
            first_free: 0,
            ftp_validation_str: String::new(),
            comments: comment.to_string(),
            raw_comment_records: Vec::new(),
            raw_record_pairs: Vec::new(),
            trailing_bytes: Vec::new(),
            arrays: Vec::new(),
        }
    }

    /// Construct a new empty DAF file for writing CK data.
    #[must_use]
    pub fn new_ck(internal_desc: &str, comment: &str) -> Self {
        Self {
            daf_type: DAFType::Ck,
            n_doubles: 2,
            n_ints: 6,
            n_chars: 40,
            internal_desc: internal_desc.to_string(),
            init_summary_record_index: 0,
            final_summary_record_index: 0,
            first_free: 0,
            ftp_validation_str: String::new(),
            comments: comment.to_string(),
            raw_comment_records: Vec::new(),
            raw_record_pairs: Vec::new(),
            trailing_bytes: Vec::new(),
            arrays: Vec::new(),
        }
    }

    /// Write the complete DAF file to the provided writer.
    ///
    /// Uses an interleaved layout matching CSPICE: each batch of segments has
    /// its summary/name record pair followed immediately by the segment data,
    /// then padding to a record boundary before the next batch.
    ///
    /// # Errors
    /// Returns an error if writing to the underlying writer fails.
    ///
    /// # Panics
    /// Panics if arrays is non-empty and the last batch has no addresses.
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        clippy::cast_sign_loss,
        reason = "DAF format uses i32 record indices that are always non-negative in valid files"
    )]
    pub fn write_to<W: Write + Seek>(&self, w: &mut W) -> KeteResult<()> {
        // FTP validation string -- identical in every DAF file.
        const FTPSTR: [u8; 28] = [
            0x46, 0x54, 0x50, 0x53, 0x54, 0x52, // FTPSTR
            0x3A, 0x0D, 0x3A, // :CR:
            0x0A, 0x3A, // LF:
            0x0D, 0x0A, 0x3A, // CRLF:
            0x0D, 0x00, 0x3A, // CR NUL:
            0x81, 0x3A, // 0x81:
            0x10, 0xCE, 0x3A, // 0x10 0xCE:
            0x45, 0x4E, 0x44, 0x46, 0x54, 0x50, // ENDFTP
        ];

        // --- Comment area ---
        // Convert comment text: newlines -> \x00 separators, append \x04 EOT.
        let comment_bytes: Vec<u8> = self
            .comments
            .bytes()
            .map(|b| if b == b'\n' { 0x00 } else { b })
            .chain(std::iter::once(0x04))
            .collect();
        // Each record holds 1000 content bytes + 24 null padding.
        // If we read this file from disk, preserve the original comment record
        // count (which may include pre-allocated empty space).
        let n_comment_records = if self.init_summary_record_index > 2 {
            (self.init_summary_record_index - 2) as usize
        } else {
            comment_bytes.len().div_ceil(1000).max(1)
        };

        // --- Interleaved summary/name + data layout ---
        // summary_stride: bytes per summary in the record (word-aligned).
        // NI ints are packed into ceil(NI/2) f64 words, so the stride may
        // exceed the raw byte count (8*ND + 4*NI) by up to 4 bytes.
        let summary_size = (self.n_doubles + (self.n_ints + 1) / 2) as usize;
        let summary_stride = summary_size * 8;
        let summary_bytes = (8 * self.n_doubles + 4 * self.n_ints) as usize;
        // Limited by both the summary record capacity (with 24-byte header)
        // and the name record capacity (n_chars bytes per name, 1024-byte record).
        let max_per_record = ((1024 - 24) / summary_stride).min(1024 / self.n_chars as usize);
        let n_arrays = self.arrays.len();
        let n_batches = if n_arrays == 0 {
            1
        } else {
            n_arrays.div_ceil(max_per_record)
        };

        // Pre-compute the record position and word addresses for each batch.
        // Layout per batch: [summary_record] [name_record] [data...] [pad]
        let mut batch_summary_records: Vec<i32> = Vec::with_capacity(n_batches);
        let mut batch_counts: Vec<usize> = Vec::with_capacity(n_batches);
        let mut all_addresses: Vec<(i32, i32)> = Vec::with_capacity(n_arrays);

        let mut current_record = (n_comment_records + 2) as i32;
        let mut arr_offset = 0;
        for _ in 0..n_batches {
            let count = (n_arrays - arr_offset).min(max_per_record);
            batch_summary_records.push(current_record);
            batch_counts.push(count);

            let name_record = current_record + 1;
            let data_start_word = name_record as usize * 128 + 1;
            let mut next_word = data_start_word;
            for i in 0..count {
                let start = next_word as i32;
                let end = start + self.arrays[arr_offset + i].len() as i32 - 1;
                all_addresses.push((start, end));
                next_word = end as usize + 1;
            }

            // Next batch starts at the next record after the data.
            if count > 0 {
                let last_end_word = all_addresses.last().unwrap().1 as usize;
                current_record = (last_end_word.div_ceil(128) + 1) as i32;
            } else {
                current_record = name_record + 1;
            }
            arr_offset += count;
        }

        let fward = batch_summary_records[0];
        let bward = *batch_summary_records.last().unwrap();
        let free = if let Some(&(_, end)) = all_addresses.last() {
            end + 1
        } else {
            (bward + 1) * 128 + 1
        };

        // ====== Record 1: File record (1024 bytes) ======
        let mut rec = vec![0_u8; 1024];
        // Magic: "DAF/XXX "
        rec[0..8].copy_from_slice(&self.daf_type.magic_bytes());
        // ND, NI
        rec[8..12].copy_from_slice(&i32_to_bytes(self.n_doubles));
        rec[12..16].copy_from_slice(&i32_to_bytes(self.n_ints));
        // Internal filename (60 chars, space-padded)
        let desc_bytes = string_to_padded_bytes(&self.internal_desc, 60);
        rec[16..76].copy_from_slice(&desc_bytes);
        // FWARD, BWARD, FREE
        rec[76..80].copy_from_slice(&i32_to_bytes(fward));
        rec[80..84].copy_from_slice(&i32_to_bytes(bward));
        rec[84..88].copy_from_slice(&i32_to_bytes(free));
        // Endianness string
        let endian_str = "LTL-IEEE";
        rec[88..96].copy_from_slice(endian_str.as_bytes());
        // FTPSTR at offset 699
        rec[699..727].copy_from_slice(&FTPSTR);
        w.write_all(&rec)?;

        // ====== Comment records ======
        if self.raw_comment_records.len() == n_comment_records {
            // Write back the exact bytes we read from the original file.
            for rec in &self.raw_comment_records {
                w.write_all(rec)?;
            }
        } else {
            for i in 0..n_comment_records {
                let mut crec = vec![0_u8; 1024];
                let start = i * 1000;
                let end = (start + 1000).min(comment_bytes.len());
                if start < comment_bytes.len() {
                    let chunk = &comment_bytes[start..end];
                    crec[..chunk.len()].copy_from_slice(chunk);
                    // Space-pad from end of content to byte 1000.
                    for b in &mut crec[chunk.len()..1000] {
                        *b = b' ';
                    }
                }
                // Records fully past the content stay all-null (pre-allocated space).
                // Bytes 1000..1023 stay as null (Fortran overhead area).
                w.write_all(&crec)?;
            }
        }

        // ====== Interleaved summary/name + data batches ======
        let mut arr_idx = 0;
        for batch in 0..n_batches {
            let count = batch_counts[batch];

            // Start from original raw bytes if available (preserves trailing
            // padding bytes for bit-exact reproduction), otherwise zeros.
            let (mut srec, mut nrec) = if batch < self.raw_record_pairs.len() {
                (
                    self.raw_record_pairs[batch].0.to_vec(),
                    self.raw_record_pairs[batch].1.to_vec(),
                )
            } else {
                (vec![0_u8; 1024], vec![0_u8; 1024])
            };

            // --- Summary record header ---
            let next_sr = if batch + 1 < n_batches {
                f64::from(batch_summary_records[batch + 1])
            } else {
                0.0
            };
            let prev_sr = if batch > 0 {
                f64::from(batch_summary_records[batch - 1])
            } else {
                0.0
            };
            srec[0..8].copy_from_slice(f64_vec_to_bytes(&[next_sr]));
            srec[8..16].copy_from_slice(f64_vec_to_bytes(&[prev_sr]));
            srec[16..24].copy_from_slice(f64_vec_to_bytes(&[count as f64]));

            for j in 0..count {
                let (a_start, a_end) = all_addresses[arr_idx];
                let summary = self.arrays[arr_idx].summary_to_bytes(a_start, a_end);
                // Use word-aligned stride for summary placement in record.
                let s_off = 24 + j * summary_stride;
                srec[s_off..s_off + summary_bytes].copy_from_slice(&summary);

                let n_off = j * self.n_chars as usize;
                let name_padded =
                    string_to_padded_bytes(&self.arrays[arr_idx].name, self.n_chars as usize);
                nrec[n_off..n_off + self.n_chars as usize].copy_from_slice(&name_padded);

                arr_idx += 1;
            }

            w.write_all(&srec)?;
            w.write_all(&nrec)?;

            // --- Data for this batch's segments ---
            for j in 0..count {
                w.write_all(self.arrays[arr_idx - count + j].data_to_bytes())?;
            }

            // Pad to record boundary after each batch.
            let pos = w.stream_position()? as usize;
            let remainder = pos % 1024;
            if remainder != 0 {
                let pad = 1024 - remainder;
                w.write_all(&vec![0_u8; pad])?;
            }
        }

        // Write any trailing bytes captured from the original file.
        if !self.trailing_bytes.is_empty() {
            w.write_all(&self.trailing_bytes)?;
        }

        Ok(())
    }

    /// Write this DAF to a file.
    ///
    /// # Errors
    /// Returns an error if the file cannot be created or written.
    pub fn write_file(&self, filename: &str) -> KeteResult<()> {
        let file = std::fs::File::create(filename)?;
        let mut w = std::io::BufWriter::new(file);
        self.write_to(&mut w)
    }

    /// Load DAF file from the specified filename.
    ///
    /// # Errors
    /// Fails if there is a read error or parsing error.
    pub fn from_file(filename: &str) -> KeteResult<Self> {
        let mut file = std::fs::File::open(filename)?;
        let mut buffer = Vec::new();
        let _ = file.read_to_end(&mut buffer)?;
        let mut buffer = Cursor::new(&buffer);
        Self::from_buffer(&mut buffer)
    }

    /// Load all [`DafArray`] segments from the DAF file.
    /// These are tuples containing a series of f64s and i32s along with arrays of
    /// data.
    /// The meaning of these values depends on the particular implementation of the
    /// DAF.
    ///
    /// # Errors
    /// Can fail if file fails to read or byte conversion fails (file incorrectly
    /// formatted).
    ///
    #[allow(
        clippy::cast_sign_loss,
        reason = "cast should work except when file is incorrectly formatted"
    )]
    pub fn try_load_arrays<T: Read + Seek>(&mut self, file: &mut T) -> KeteResult<()> {
        let summary_size = self.n_doubles + (self.n_ints + 1) / 2;

        let mut next_idx = self.init_summary_record_index;
        loop {
            if next_idx == 0 {
                break;
            }
            let current_idx = next_idx;
            let bytes = Self::try_load_record(file, current_idx as u64)?;

            next_idx = bytes_to_f64(&bytes[0..8])? as i32;
            // let prev_idx = bytes_to_f64(&bytes[8..16])? as i32;
            let n_summaries = bytes_to_f64(&bytes[16..24])? as i32;

            // Name record immediately follows each summary record.
            let name_bytes = Self::try_load_record(file, (current_idx + 1) as u64)?;

            // Store raw record pair for bit-exact reproduction.
            self.raw_record_pairs
                .push((bytes.clone(), name_bytes.clone()));

            for idy in 0..n_summaries {
                let sum_start = (3 * 8 + idy * summary_size * 8) as usize;
                let floats =
                    bytes_to_f64_vec(&bytes[sum_start..sum_start + 8 * self.n_doubles as usize])?;
                let ints = bytes_to_i32_vec(
                    &bytes[sum_start + 8 * self.n_doubles as usize
                        ..sum_start + (8 * self.n_doubles + 4 * self.n_ints) as usize],
                )?;

                // Extract segment name from name record.
                let name_start = (idy * self.n_chars) as usize;
                let name_end = (name_start + self.n_chars as usize).min(name_bytes.len());
                let name = bytes_to_string(&name_bytes[name_start..name_end])
                    .trim()
                    .to_string();

                let array = DafArray::try_load_array(file, floats, ints, self.daf_type, name);
                self.arrays.push(array?);
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

    /// Segment descriptor name from the DAF name record.
    pub name: String,
}

impl Debug for DafArray {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("DafArray({} values)", self.data.len()))
    }
}

impl DafArray {
    /// Try to load an DAF array from summary data.
    ///
    /// # Errors
    /// Fails when a file is either incorrectly formatted, or a read error occurs.
    #[allow(
        clippy::cast_sign_loss,
        reason = "cast should work except when file is incorrectly formatted"
    )]
    pub fn try_load_array<T: Read + Seek>(
        buffer: &mut T,
        summary_floats: Box<[f64]>,
        summary_ints: Box<[i32]>,
        daf_type: DAFType,
        name: String,
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

        let data = read_f64_vec(buffer, n_floats)?;

        Ok(Self {
            summary_floats,
            summary_ints,
            data,
            daf_type,
            name,
        })
    }

    /// Total length of the array.
    #[must_use]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Test if array is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Construct a new [`DafArray`] from raw components.
    #[must_use]
    pub fn new(
        summary_floats: Box<[f64]>,
        summary_ints: Box<[i32]>,
        data: Box<[f64]>,
        daf_type: DAFType,
        name: String,
    ) -> Self {
        Self {
            summary_floats,
            summary_ints,
            data,
            daf_type,
            name,
        }
    }

    /// Serialize the summary (floats then ints) as bytes for inclusion in a summary
    /// record. The last two ints in the summary are the 1-indexed start/end
    /// double-precision addresses of the array data in the file; these are provided
    /// by the caller since they depend on file layout.
    #[must_use]
    pub fn summary_to_bytes(&self, array_start: i32, array_end: i32) -> Vec<u8> {
        let n_ints = self.summary_ints.len();
        // Build a copy of summary_ints with the last two replaced by addresses.
        let mut ints = self.summary_ints.to_vec();
        if n_ints >= 2 {
            ints[n_ints - 2] = array_start;
            ints[n_ints - 1] = array_end;
        }
        let mut out = f64_vec_to_bytes(&self.summary_floats).to_vec();
        out.extend_from_slice(i32_vec_to_bytes(&ints));
        out
    }

    /// View the data array as bytes (zero-copy).
    #[must_use]
    pub fn data_to_bytes(&self) -> &[u8] {
        f64_vec_to_bytes(&self.data)
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

/// Convert a big-endian DAF file to a little-endian copy.
///
/// Reads `input_filename`, byte-swaps every numeric field (all f64 and i32
/// values in the file-record header, summary records, and data arrays), updates
/// the endianness marker from `"BIG-IEEE"` to `"LTL-IEEE"`, and writes the
/// result to `output_filename`.
///
/// Many older SPK files distributed by NAIF (e.g. Spitzer) are big-endian.
/// kete can only read little-endian DAF files, so this function is provided
/// as a one-time conversion step.
///
/// # Errors
/// Returns [`Error::IOError`] if:
/// - `input_filename` cannot be read.
/// - The input is not a valid big-endian DAF (wrong endian marker, too small,
///   or invalid header values).
/// - `output_filename` cannot be written.
pub fn convert_daf_big_to_little_endian(
    input_filename: &str,
    output_filename: &str,
) -> KeteResult<()> {
    let bytes = std::fs::read(input_filename)
        .map_err(|e| Error::IOError(format!("Failed to read {input_filename}: {e}")))?;
    let converted = swap_daf_be_to_le(&bytes)?;
    std::fs::write(output_filename, converted)
        .map_err(|e| Error::IOError(format!("Failed to write {output_filename}: {e}")))?;
    Ok(())
}

/// Byte-swap all numeric fields of a big-endian DAF byte buffer, returning a
/// little-endian copy.
///
/// # Errors
/// Returns [`Error::IOError`] if the buffer is not a valid big-endian DAF.
#[allow(
    clippy::cast_sign_loss,
    reason = "header values are validated to be non-negative before casting to usize"
)]
#[allow(
    clippy::missing_panics_doc,
    reason = "slice lengths are validated before .try_into().unwrap() calls"
)]
fn swap_daf_be_to_le(input: &[u8]) -> KeteResult<Vec<u8>> {
    if input.len() < 1024 {
        return Err(Error::IOError("DAF file is too small to be valid.".into()));
    }

    if !input[88..96].eq_ignore_ascii_case(b"big-ieee") {
        return Err(Error::IOError(
            "Input is not a big-endian DAF file (expected 'BIG-IEEE' at byte offset 88).".into(),
        ));
    }

    // Read header values in big-endian before any in-place swapping.
    let n_doubles = i32::from_be_bytes(input[8..12].try_into().unwrap());
    let n_ints = i32::from_be_bytes(input[12..16].try_into().unwrap());
    let init_summary_idx = i32::from_be_bytes(input[76..80].try_into().unwrap()).abs();

    if n_doubles < 0 || n_ints < 2 {
        return Err(Error::IOError(
            "DAF header has invalid ND or NI values.".into(),
        ));
    }
    if init_summary_idx < 2 {
        return Err(Error::IOError(
            "DAF header has invalid FWARD (first summary record index).".into(),
        ));
    }

    let mut buf = input.to_vec();

    // Byte-swap the five i32 fields in the file record.
    swap4(&mut buf, 8); // ND (n_doubles)
    swap4(&mut buf, 12); // NI (n_ints)
    swap4(&mut buf, 76); // FWARD
    swap4(&mut buf, 80); // BWARD
    swap4(&mut buf, 84); // FREE

    // Overwrite the endianness tag.
    buf[88..96].copy_from_slice(b"LTL-IEEE");

    let nd = n_doubles as usize;
    let ni = n_ints as usize;
    // Each summary occupies (ND + ceil(NI/2)) 8-byte words, word-aligned.
    let summary_stride = (nd + ni.div_ceil(2)) * 8;

    // Walk the linked list of summary records.
    let mut current_idx = init_summary_idx as usize;
    while current_idx != 0 {
        let rec_off = 1024 * (current_idx - 1);
        if rec_off + 24 > buf.len() {
            break;
        }

        // Read next-record index and n_summaries from the original BE bytes.
        let next_f = f64::from_be_bytes(input[rec_off..rec_off + 8].try_into().unwrap());
        let n_summ =
            f64::from_be_bytes(input[rec_off + 16..rec_off + 24].try_into().unwrap()) as usize;

        // Swap the three summary-record header f64s.
        swap8(&mut buf, rec_off); // next summary record
        swap8(&mut buf, rec_off + 8); // prev summary record
        swap8(&mut buf, rec_off + 16); // n_summaries

        for j in 0..n_summ {
            let s_off = rec_off + 24 + j * summary_stride;
            let ints_off = s_off + nd * 8;
            if ints_off + ni * 4 > buf.len() {
                break;
            }

            // Swap n_doubles f64s.
            for k in 0..nd {
                swap8(&mut buf, s_off + k * 8);
            }
            // Swap n_ints i32s.
            for k in 0..ni {
                swap4(&mut buf, ints_off + k * 4);
            }

            // Array start/end are the last two ints.  They have already been
            // byte-swapped above, so read them back as little-endian.
            let last_two = ints_off + (ni - 2) * 4;
            let array_start =
                i32::from_le_bytes(buf[last_two..last_two + 4].try_into().unwrap()) as usize;
            let array_end =
                i32::from_le_bytes(buf[last_two + 4..last_two + 8].try_into().unwrap()) as usize;

            if array_start == 0 || array_end < array_start {
                continue;
            }

            // Byte-swap every f64 word in the data segment.
            for word in array_start..=array_end {
                let byte_off = 8 * (word - 1);
                if byte_off + 8 <= buf.len() {
                    swap8(&mut buf, byte_off);
                }
            }
        }

        current_idx = if next_f == 0.0 { 0 } else { next_f as usize };
    }

    Ok(buf)
}

#[inline(always)]
fn swap8(buf: &mut [u8], off: usize) {
    buf[off..off + 8].reverse();
}

#[inline(always)]
fn swap4(buf: &mut [u8], off: usize) {
    buf[off..off + 4].reverse();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn daf_array_new_fields() {
        let floats: Box<[f64]> = vec![1.0, 2.0].into();
        let ints: Box<[i32]> = vec![10, 20, 30, 40, 0, 0].into();
        let data: Box<[f64]> = vec![100.0, 200.0, 300.0].into();
        let arr = DafArray::new(
            floats.clone(),
            ints.clone(),
            data.clone(),
            DAFType::Spk,
            "test segment".into(),
        );
        assert_eq!(&*arr.summary_floats, &*floats);
        assert_eq!(&*arr.summary_ints, &*ints);
        assert_eq!(&*arr.data, &*data);
        assert_eq!(arr.daf_type, DAFType::Spk);
        assert_eq!(arr.name, "test segment");
        assert_eq!(arr.len(), 3);
        assert!(!arr.is_empty());
    }

    #[test]
    fn daf_array_summary_round_trip() {
        let floats: Box<[f64]> = vec![1.5, 2.5].into();
        let ints: Box<[i32]> = vec![10, 20, 30, 40, 0, 0].into();
        let data: Box<[f64]> = vec![99.0].into();
        let arr = DafArray::new(floats, ints, data, DAFType::Spk, String::new());

        let start_addr: i32 = 501;
        let end_addr: i32 = 501;
        let bytes = arr.summary_to_bytes(start_addr, end_addr);

        // Parse back: 2 f64s then 6 i32s
        let f0 = bytes_to_f64(&bytes[0..8]).unwrap();
        let f1 = bytes_to_f64(&bytes[8..16]).unwrap();
        assert_eq!(f0, 1.5);
        assert_eq!(f1, 2.5);

        let parsed_ints = bytes_to_i32_vec(&bytes[16..]).unwrap();
        assert_eq!(parsed_ints[0], 10);
        assert_eq!(parsed_ints[1], 20);
        assert_eq!(parsed_ints[2], 30);
        assert_eq!(parsed_ints[3], 40);
        assert_eq!(parsed_ints[4], start_addr);
        assert_eq!(parsed_ints[5], end_addr);
    }

    #[test]
    fn daf_array_data_round_trip() {
        let data: Box<[f64]> = vec![1.0, 2.0, 3.0, std::f64::consts::PI].into();
        let arr = DafArray::new(
            vec![].into(),
            vec![0, 0].into(),
            data.clone(),
            DAFType::Spk,
            String::new(),
        );

        let bytes = arr.data_to_bytes();
        let back = bytes_to_f64_vec(bytes).unwrap();
        assert_eq!(&*back, &*data);
    }

    #[test]
    fn daf_array_summary_to_bytes() {
        let floats: Box<[f64]> = vec![10.0].into();
        let ints: Box<[i32]> = vec![5, 0, 0].into();
        let arr = DafArray::new(floats, ints, vec![].into(), DAFType::Pck, String::new());

        let bytes = arr.summary_to_bytes(100, 200);

        let f0 = bytes_to_f64(&bytes[0..8]).unwrap();
        assert_eq!(f0, 10.0);

        let parsed_ints = bytes_to_i32_vec(&bytes[8..]).unwrap();
        assert_eq!(parsed_ints[0], 5);
        assert_eq!(parsed_ints[1], 100);
        assert_eq!(parsed_ints[2], 200);
    }

    #[test]
    fn daf_file_write_read_round_trip() {
        // Build a minimal SPK file with one segment containing simple data.
        let mut daf = DafFile::new_spk("test_roundtrip", "a comment\nsecond line");

        // 3 states (6 floats each) + 3 epochs + degree + n = 22 floats
        let data: Vec<f64> = vec![
            // state 0
            1.0, 2.0, 3.0, 0.01, 0.02, 0.03, // state 1
            4.0, 5.0, 6.0, 0.04, 0.05, 0.06, // state 2
            7.0, 8.0, 9.0, 0.07, 0.08, 0.09, // epochs
            0.0, 43200.0, 86400.0, // degree, n
            1.0, 3.0,
        ];
        let arr = DafArray::new(
            vec![0.0, 86400.0].into(),
            vec![399, 10, 1, 9, 0, 0].into(),
            data.clone().into(),
            DAFType::Spk,
            "seg1".into(),
        );
        daf.arrays.push(arr);

        // Write to in-memory buffer.
        let mut buf = Cursor::new(Vec::new());
        daf.write_to(&mut buf).unwrap();

        // Read back.
        buf.set_position(0);
        let read_back = DafFile::from_buffer(&mut buf).unwrap();

        assert_eq!(read_back.daf_type, DAFType::Spk);
        assert_eq!(read_back.n_doubles, 2);
        assert_eq!(read_back.n_ints, 6);
        assert_eq!(read_back.internal_desc.trim(), "test_roundtrip");
        assert_eq!(read_back.comments, "a comment\nsecond line");
        assert_eq!(read_back.arrays.len(), 1);

        let arr = &read_back.arrays[0];
        assert_eq!(&*arr.data, &data[..]);
        assert_eq!(arr.name, "seg1");

        // Verify SPK summary fields.
        let arr2 = read_back.arrays.into_iter().next().unwrap();
        assert_eq!(arr2.summary_ints[0], 399);
        assert_eq!(arr2.summary_ints[1], 10);
        assert_eq!(arr2.summary_ints[2], 1);
        assert_eq!(arr2.summary_ints[3], 9);
    }

    #[test]
    fn daf_file_long_comment_round_trip() {
        // A comment longer than 1000 bytes forces multiple comment records.
        // This verifies that the 24-byte Fortran overhead at the end of each
        // 1024-byte record is not injected into the comment text.
        let long_comment: String = (0..120)
            .map(|i| format!("line {i:04}: abcdefghijklmnopqrstuvwxyz"))
            .collect::<Vec<_>>()
            .join("\n");
        assert!(long_comment.len() > 2000, "comment must span >2 records");

        let mut daf = DafFile::new_spk("long_comment", &long_comment);
        let arr = DafArray::new(
            vec![0.0, 86400.0].into(),
            vec![1, 10, 1, 9, 0, 0].into(),
            vec![1.0, 2.0].into(),
            DAFType::Spk,
            "s".into(),
        );
        daf.arrays.push(arr);

        let mut buf = Cursor::new(Vec::new());
        daf.write_to(&mut buf).unwrap();
        buf.set_position(0);
        let read_back = DafFile::from_buffer(&mut buf).unwrap();
        assert_eq!(read_back.comments, long_comment);
    }

    #[test]
    fn daf_file_empty_comment() {
        let mut daf = DafFile::new_spk("empty_comment", "");
        let arr = DafArray::new(
            vec![0.0, 86400.0].into(),
            vec![1, 10, 1, 9, 0, 0].into(),
            vec![1.0, 2.0].into(),
            DAFType::Spk,
            "s".into(),
        );
        daf.arrays.push(arr);

        let mut buf = Cursor::new(Vec::new());
        daf.write_to(&mut buf).unwrap();
        buf.set_position(0);
        let read_back = DafFile::from_buffer(&mut buf).unwrap();
        assert!(read_back.comments.is_empty());
        assert_eq!(read_back.arrays.len(), 1);
    }

    #[test]
    fn daf_file_multiple_segments() {
        let mut daf = DafFile::new_spk("multi_seg", "");

        for i in 0..5 {
            let data = vec![f64::from(i); 10];
            let arr = DafArray::new(
                vec![0.0, 86400.0].into(),
                vec![i + 100, 10, 1, 9, 0, 0].into(),
                data.into(),
                DAFType::Spk,
                format!("segment {i}"),
            );
            daf.arrays.push(arr);
        }

        let mut buf = Cursor::new(Vec::new());
        daf.write_to(&mut buf).unwrap();
        buf.set_position(0);
        let read_back = DafFile::from_buffer(&mut buf).unwrap();

        assert_eq!(read_back.arrays.len(), 5);
        for (i, arr) in read_back.arrays.iter().enumerate() {
            #[allow(clippy::cast_possible_wrap, reason = "test index always fits i32")]
            let expected_id = i as i32 + 100;
            assert_eq!(arr.summary_ints[0], expected_id);
            assert_eq!(arr.name, format!("segment {i}"));
            assert_eq!(&*arr.data, &vec![f64::from(expected_id - 100); 10][..]);
        }
    }
}
