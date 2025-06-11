//! Leap Second information
use itertools::Itertools;
use std::str::FromStr;

use lazy_static::lazy_static;
use serde::Deserialize;

use crate::prelude::{Error, KeteResult};

/// Leap Second Information
/// This is parsed from the contents of the `leap_second.dat` file.
#[derive(Debug, Deserialize)]
pub struct LeapSecond {
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

lazy_static! {
    /// Leap second definitions
    pub static ref LEAP_SECONDS: Vec<LeapSecond> = {
        let mut codes = Vec::new();
        let text = std::str::from_utf8(PRELOAD_LEAPSECONDS).unwrap().split('\n');
        for row in text.filter(|x| !x.starts_with('#') & (!x.trim().is_empty())) {
            let code = LeapSecond::from_str(row).unwrap();
            codes.push(code);
        }
        codes
    };
}

/// Given an MJD return the TAI - UTC offset for that epoch in days.
///
/// TAI - UTC = offset
/// TAI - offset = UTC
/// TAI = offset + UTC
///
/// # Arguments
///
/// * `MJD` - MJD in TAI scaled time.
pub fn tai_to_utc_offset(mjd: &f64) -> f64 {
    match LEAP_SECONDS.binary_search_by(|probe| probe.mjd.total_cmp(mjd)) {
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
        assert!(tai_to_utc_offset(&0.0) == 0.0);
        assert!(tai_to_utc_offset(&41317.0) == 10.0 / 86400.0);
        assert!(tai_to_utc_offset(&41317.1) == 10.0 / 86400.0);
        assert!(tai_to_utc_offset(&57753.9) == 36.0 / 86400.0);
        assert!(tai_to_utc_offset(&57754.0) == 37.0 / 86400.0);
        assert!(tai_to_utc_offset(&57755.0) == 37.0 / 86400.0);
    }
}
