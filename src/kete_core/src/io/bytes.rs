//! Converting to and from bytes
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

use crate::errors::{Error, KeteResult};
use std::io::Read;

/// Read the exact number of specified bytes from the file.
///
/// # Errors
/// Returns an error if fewer than `n_bytes` are available.
pub fn read_bytes_exact<T: Read>(buffer: T, n_bytes: usize) -> KeteResult<Box<[u8]>> {
    let mut bytes = Vec::with_capacity(n_bytes);
    let n_read = buffer.take(n_bytes as u64).read_to_end(&mut bytes)?;
    if n_read != n_bytes {
        Err(Error::IOError("Unexpected end of file.".into()))?;
    }

    Ok(bytes.into())
}

/// Change a collection of bytes into a f64.
///
/// # Errors
/// Returns an error if the slice is not exactly 8 bytes.
pub fn bytes_to_f64(bytes: &[u8], little_endian: bool) -> KeteResult<f64> {
    let bytes: [u8; 8] = bytes
        .try_into()
        .map_err(|_| Error::IOError("File is not correctly formatted".into()))?;
    if little_endian {
        Ok(f64::from_le_bytes(bytes))
    } else {
        Ok(f64::from_be_bytes(bytes))
    }
}

/// Change a collection of bytes into a vector of f64s.
///
/// # Errors
/// Returns an error if the byte length is not a multiple of 8.
pub fn bytes_to_f64_vec(bytes: &[u8], little_endian: bool) -> KeteResult<Box<[f64]>> {
    let byte_len = bytes.len();
    if !byte_len.is_multiple_of(8) {
        Err(Error::IOError("File is not correctly formatted".into()))?;
    }
    let res: Box<[f64]> = (0..byte_len / 8)
        .map(|idx| bytes_to_f64(&bytes[8 * idx..(8 + 8 * idx)], little_endian))
        .collect::<KeteResult<_>>()?;
    Ok(res)
}

/// Change a collection of bytes into a vector of i32s.
///
/// # Errors
/// Returns an error if the byte length is not a multiple of 4.
pub fn bytes_to_i32_vec(bytes: &[u8], little_endian: bool) -> KeteResult<Box<[i32]>> {
    let byte_len = bytes.len();
    if !byte_len.is_multiple_of(4) {
        Err(Error::IOError("File is not correctly formatted".into()))?;
    }
    let res: Box<[i32]> = (0..byte_len / 4)
        .map(|idx| bytes_to_i32(&bytes[4 * idx..(4 + 4 * idx)], little_endian))
        .collect::<KeteResult<_>>()?;
    Ok(res)
}

/// Change a collection of bytes into a i32.
///
/// # Errors
/// Returns an error if the slice is not exactly 4 bytes.
pub fn bytes_to_i32(bytes: &[u8], little_endian: bool) -> KeteResult<i32> {
    let bytes: [u8; 4] = bytes
        .try_into()
        .map_err(|_| Error::IOError("File is not correctly formatted".into()))?;
    if little_endian {
        Ok(i32::from_le_bytes(bytes))
    } else {
        Ok(i32::from_be_bytes(bytes))
    }
}

/// Change a collection of bytes into a String.
#[must_use]
pub fn bytes_to_string(bytes: &[u8]) -> String {
    let mut bytes = bytes.to_vec();
    for x in &mut bytes {
        if x == &0x00 {
            *x = 0x0a;
        }
    }
    String::from_utf8_lossy(&bytes).to_string()
}

/// Read a multiple contiguous f64s from the file.
///
/// # Errors
/// Returns an error if the read or conversion fails.
pub fn read_f64_vec<T: Read>(
    buffer: T,
    n_floats: usize,
    little_endian: bool,
) -> KeteResult<Box<[f64]>> {
    let bytes = read_bytes_exact(buffer, 8 * n_floats)?;
    bytes_to_f64_vec(&bytes, little_endian)
}

/// Read a string of the specified length from the file.
/// 0x00 are replaced with new lines, and new lines are stripped from the end of the
/// string.
///
/// # Errors
/// Returns an error if the read fails.
pub fn read_str<T: Read>(buffer: T, length: usize) -> KeteResult<String> {
    let bytes = read_bytes_exact(buffer, length)?;
    Ok(bytes_to_string(&bytes))
}

/// Encode an f64 as 8 bytes.
#[must_use]
pub fn f64_to_bytes(value: f64, little_endian: bool) -> [u8; 8] {
    if little_endian {
        value.to_le_bytes()
    } else {
        value.to_be_bytes()
    }
}

/// Encode an i32 as 4 bytes.
#[must_use]
pub fn i32_to_bytes(value: i32, little_endian: bool) -> [u8; 4] {
    if little_endian {
        value.to_le_bytes()
    } else {
        value.to_be_bytes()
    }
}

/// Encode a slice of f64s as bytes.
#[must_use]
pub fn f64_vec_to_bytes(values: &[f64], little_endian: bool) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 8);
    for &v in values {
        out.extend_from_slice(&f64_to_bytes(v, little_endian));
    }
    out
}

/// Encode a slice of i32s as bytes.
#[must_use]
pub fn i32_vec_to_bytes(values: &[i32], little_endian: bool) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 4);
    for &v in values {
        out.extend_from_slice(&i32_to_bytes(v, little_endian));
    }
    out
}

/// Write a fixed-width string, padded with spaces to `width` bytes.
/// Truncates if the string is longer than `width`.
#[must_use]
pub fn string_to_padded_bytes(s: &str, width: usize) -> Vec<u8> {
    let s_bytes = s.as_bytes();
    let copy_len = s_bytes.len().min(width);
    let mut out = vec![b' '; width];
    out[..copy_len].copy_from_slice(&s_bytes[..copy_len]);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f64_round_trip_le() {
        let values = [0.0, 1.0, -1.0, f64::MIN, f64::MAX, std::f64::consts::PI];
        for &v in &values {
            let bytes = f64_to_bytes(v, true);
            let back = bytes_to_f64(&bytes, true).unwrap();
            assert_eq!(v, back);
        }
    }

    #[test]
    fn f64_round_trip_be() {
        let values = [0.0, 1.0, -1.0, f64::MIN, f64::MAX, std::f64::consts::PI];
        for &v in &values {
            let bytes = f64_to_bytes(v, false);
            let back = bytes_to_f64(&bytes, false).unwrap();
            assert_eq!(v, back);
        }
    }

    #[test]
    fn i32_round_trip_le() {
        let values = [0, 1, -1, i32::MIN, i32::MAX, 42, -999];
        for &v in &values {
            let bytes = i32_to_bytes(v, true);
            let back = bytes_to_i32(&bytes, true).unwrap();
            assert_eq!(v, back);
        }
    }

    #[test]
    fn i32_round_trip_be() {
        let values = [0, 1, -1, i32::MIN, i32::MAX, 42, -999];
        for &v in &values {
            let bytes = i32_to_bytes(v, false);
            let back = bytes_to_i32(&bytes, false).unwrap();
            assert_eq!(v, back);
        }
    }

    #[test]
    fn f64_vec_round_trip() {
        let values = vec![1.0, 2.0, 3.0, std::f64::consts::E, -0.0];
        for le in [true, false] {
            let bytes = f64_vec_to_bytes(&values, le);
            assert_eq!(bytes.len(), values.len() * 8);
            let back = bytes_to_f64_vec(&bytes, le).unwrap();
            assert_eq!(&*back, &values[..]);
        }
    }

    #[test]
    fn i32_vec_round_trip() {
        let values = vec![10, -20, 0, i32::MAX, i32::MIN];
        for le in [true, false] {
            let bytes = i32_vec_to_bytes(&values, le);
            assert_eq!(bytes.len(), values.len() * 4);
            let back = bytes_to_i32_vec(&bytes, le).unwrap();
            assert_eq!(&*back, &values[..]);
        }
    }

    #[test]
    fn string_to_padded_exact_width() {
        let out = string_to_padded_bytes("hello", 5);
        assert_eq!(&out, b"hello");
    }

    #[test]
    fn string_to_padded_shorter() {
        let out = string_to_padded_bytes("hi", 8);
        assert_eq!(&out, b"hi      ");
    }

    #[test]
    fn string_to_padded_truncates() {
        let out = string_to_padded_bytes("hello world", 5);
        assert_eq!(&out, b"hello");
    }

    #[test]
    fn string_to_padded_empty() {
        let out = string_to_padded_bytes("", 4);
        assert_eq!(&out, b"    ");
    }

    #[test]
    fn string_to_padded_zero_width() {
        let out = string_to_padded_bytes("anything", 0);
        assert!(out.is_empty());
    }
}
