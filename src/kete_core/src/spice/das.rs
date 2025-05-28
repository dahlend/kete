/// DAS (Direct Access Segregated) File Format
///
/// DAS files are binary files which are designed to be used in simple database
/// applications.
///
/// Most files in SPICE are called by one name for the format, and another name
/// for the actual implementation. This can frequently lead to confusion.
///
/// Below is a list of names of formats related to DAS and the name of their
/// implementations.
///
/// - DLA (Doubly Linked Array) - which itself is a base format for other formats:
///     - DSK - Digital Shape Kernel, .bds files
/// - EK - Event Kernels, .dbk .bdb files
///
use std::io::{Read, Seek};

use crate::{
    errors::Error,
    io::bytes::{bytes_to_i32, bytes_to_string, read_bytes_exact},
};

// DAS record lengths are all 1024 bytes.

pub struct DasContents {
    header: FileRecord,

    comments: Vec<String>,

    reserved_records: Vec<i32>,

    reserved_chars: Vec<u8>,
}

pub struct DirRecord {
    pub prev_addr: i32,
    pub next_addr: i32,
    // Directory records are doubly linked lists
    // the contents of the list are a set of 3 pairs in indices
    // the first pair points to the extent of the char records,
    // the second pair points to the extent of the float records,
    // and the third pair points to the extent of the int records.
    // empty contents are represented by address range 0:0.
    //
    // After the index pairs, there are cluster descriptors.
}

pub struct CharRecord(pub [u8; 1024]);

pub struct FloatRecord(pub [f64; 128]);

pub struct IntRecord(pub [i32; 256]);

pub struct FileRecord {
    /// Text descriptor for the type of file.
    /// Extends from [0..8] in the file.
    pub id_word: [u8; 8],

    /// Internal filename (string of length 60).
    /// This is not recommended to be used, as the comments
    /// section is preferred.
    /// Extends from [8..68] in the file.
    pub internal_filename: [u8; 60],

    ///Number of reserved records in the file.
    /// Extends from [68..72] in the file.
    /// Reserved records are used for future expansions.
    pub n_reserved_records: i32,

    /// Number of reserved characters in the file.
    /// Extends from [72..76] in the file.
    /// Reserved records are used for future expansions.
    pub n_reserved_chars: i32,

    /// Number of comment records in the file.
    /// Extends from [76..80] in the file.
    pub n_comment_records: i32,

    /// Number of comment characters in the file.
    /// Extends from [80..84] in the file.
    pub n_comment_chars: i32,

    /// Text descriptor for the type of file.
    /// Extends from [84..92] in the file.
    pub format_identifier: [u8; 8],

    // There is a gap of nulls from 92 to 996
    /// ftp checksum to validate the contents of the file.
    /// Extends from [966..994] in the file.
    pub ftp_checksum: [u8; 28],
}

impl FileRecord {
    pub fn from_buffer<T: Read + Seek>(buffer: &mut T) -> Result<Self, Error> {
        let _ = buffer.seek(std::io::SeekFrom::Start(0))?;
        let contents: Box<[u8]> = read_bytes_exact(buffer, 1024)?;

        let id_word = contents[0..8].try_into().unwrap();
        let internal_filename = contents[8..68].try_into().unwrap();

        let little_endian = match bytes_to_string(&contents[84..92]).to_lowercase().as_str() {
            "ltl-ieee" => true,
            "big-ieee" => false,
            _ => Err(Error::IOError(
                "Expected little or big endian in DAS file, found neither".into(),
            ))?,
        };

        let n_reserved_records = bytes_to_i32(&contents[68..72], little_endian)?;
        let n_reserved_chars = bytes_to_i32(&contents[72..76], little_endian)?;
        let n_comment_records = bytes_to_i32(&contents[76..80], little_endian)?;
        let n_comment_chars = bytes_to_i32(&contents[80..84], little_endian)?;
        let format_identifier = contents[84..92].try_into().unwrap();
        let ftp_checksum = contents[966..994].try_into().unwrap();

        Ok(FileRecord {
            id_word,
            internal_filename,
            n_reserved_records,
            n_reserved_chars,
            n_comment_records,
            n_comment_chars,
            format_identifier,
            ftp_checksum,
        })
    }
}
