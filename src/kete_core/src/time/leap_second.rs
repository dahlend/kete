//! Leap Second information
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

use itertools::Itertools;
use std::str::FromStr;

use serde::Deserialize;

use crate::prelude::{Error, KeteResult};

/// Leap Second Information
/// This is parsed from the contents of the `leap_second.dat` file.
#[derive(Debug, Deserialize)]
struct LeapSecond {
    ///  MJD
    pub mjd: f64,

    /// Offset from TAI time in fractions of day
    pub tai_m_utc: f64,
}

impl FromStr for LeapSecond {
    type Err = Error;

    /// Load a [`LeapSecond`] from a single string.
    fn from_str(row: &str) -> KeteResult<Self> {
        let (mjd, _, _, _, tai_m_utc) = row.split_whitespace().next_tuple().ok_or(
            Error::IOError("Leap Second file incorrectly formatted.".into()),
        )?;

        Ok(Self {
            mjd: mjd.parse()?,
            tai_m_utc: tai_m_utc.parse::<f64>()? / 86400.0,
        })
    }
}

/// Load the leap second file during compilation.
const PRELOAD_LEAPSECONDS: &[u8] = include_bytes!("../../data/leap_second.dat");

/// Leap second definitions
static LEAP_SECONDS: std::sync::LazyLock<Vec<LeapSecond>> = std::sync::LazyLock::new(|| {
    let mut codes = Vec::new();
    let text = std::str::from_utf8(PRELOAD_LEAPSECONDS)
        .unwrap()
        .split('\n');
    for row in text.filter(|x| !x.starts_with('#') & (!x.trim().is_empty())) {
        let code = LeapSecond::from_str(row).unwrap();
        codes.push(code);
    }
    codes
});

/// Given an MJD return the TAI - UTC offset for that epoch in days.
///
/// TAI - UTC = offset
/// TAI - offset = UTC
/// TAI = offset + UTC
///
/// # Arguments
///
/// * `MJD` - MJD in TAI scaled time.
pub(crate) fn tai_to_utc_offset(mjd: f64) -> f64 {
    match LEAP_SECONDS.binary_search_by(|probe| probe.mjd.total_cmp(&mjd)) {
        Ok(idx) => LEAP_SECONDS[idx].tai_m_utc,
        Err(0) => 0.0,
        Err(idx) => LEAP_SECONDS[idx - 1].tai_m_utc,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_leap_second() {
        {
            let t = &LEAP_SECONDS[0];
            assert!(t.tai_m_utc == 10.0 / 86400.0);
            assert!(t.mjd == 41317.0);
        }
        {
            let t = &LEAP_SECONDS[27];
            assert!(t.tai_m_utc == 37.0 / 86400.0);
            assert!(t.mjd == 57754.0);
        }
    }

    #[test]
    fn test_lookup() {
        assert!(tai_to_utc_offset(0.0) == 0.0);
        assert!(tai_to_utc_offset(41317.0) == 10.0 / 86400.0);
        assert!(tai_to_utc_offset(41317.1) == 10.0 / 86400.0);
        assert!(tai_to_utc_offset(57753.9) == 36.0 / 86400.0);
        assert!(tai_to_utc_offset(57754.0) == 37.0 / 86400.0);
        assert!(tai_to_utc_offset(57755.0) == 37.0 / 86400.0);
    }
}
