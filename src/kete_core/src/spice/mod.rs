//! Spice replacement methods
//! Primarily includes the ability to read the contents of SPK files.
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

mod ck;
mod ck_segments;
mod daf;
mod interpolation;
mod naif_ids;
mod obs_codes;
mod pck;
mod pck_segments;
mod spk;
mod spk_segments;

// slightly higher visibility as it has useful parsing methods
pub(crate) mod sclk;

pub use ck::{CkCollection, LOADED_CK};
pub use daf::{CkArray, DAFType, DafArray, DafFile, PckArray, SpkArray};
pub use naif_ids::{NaifId, naif_ids_from_name, try_name_from_id};
pub use obs_codes::{OBS_CODES, ObsCode, try_obs_code_from_name};
pub use pck::{LOADED_PCK, PckCollection};
pub use sclk::{LOADED_SCLK, SclkCollection};
pub use spk::{LOADED_SPK, SpkCollection};

/// Convert seconds from J2000 into JD.
///
/// # Arguments
/// * `jds_sec` - The number of TDB seconds from J2000.
///
/// # Returns
/// The Julian Date (TDB).
#[inline(always)]
fn spice_jd_to_jd(jds_sec: f64) -> f64 {
    // 86400.0 = 60 * 60 * 24
    jds_sec / 86400.0 + 2451545.0
}

/// Convert TDB JD to seconds from J2000.
#[inline(always)]
fn jd_to_spice_jd(jd: f64) -> f64 {
    // 86400.0 = 60 * 60 * 24
    (jd - 2451545.0) * 86400.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spice_jd_to_jd() {
        {
            let jd_sec = 0.0;
            let jd = spice_jd_to_jd(jd_sec);
            assert_eq!(jd, 2451545.0);
        }
        {
            let jd_sec = 86400.0; // 1 day in seconds
            let jd = spice_jd_to_jd(jd_sec);
            assert_eq!(jd, 2451546.0);
        }
    }

    #[test]
    fn test_jd_to_spice_jd() {
        {
            let jd = 2451545.0;
            let jd_sec = jd_to_spice_jd(jd);
            assert_eq!(jd_sec, 0.0);
        }
        {
            let jd = 2451546.0; // 1 day after J2000
            let jd_sec = jd_to_spice_jd(jd);
            assert_eq!(jd_sec, 86400.0);
        }
    }

    #[test]
    fn test_spice_jd_to_jd_and_back() {
        let jd_sec = 1.0;
        let jd = spice_jd_to_jd(jd_sec);
        let jd_sec_back = jd_to_spice_jd(jd);
        assert!((jd_sec - jd_sec_back).abs() < 1e-5);
    }
}
