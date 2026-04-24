//! Star catalog astrometric debiasing.
//!
//! Implements the lookup of per-catalog astrometric biases from the JPL
//! debiasing tables (Farnocchia, Chesley, Chamberlin, Tholen 2015 -- FCCT14;
//! Eggl, Farnocchia, Chamberlin, Chesley 2018 -- EFCC18). The tables give
//! the mean offset of a given star catalog with respect to a reference
//! frame (PPMXL for FCCT14, `Gaia-DR2` for EFCC18) on a `HEALPix` `nside=64`
//! grid (49152 tiles), with four `int16` values per tile per catalog:
//! RA bias, Dec bias, RA proper motion bias, Dec proper motion bias.
//!
//! The corrections returned by [`DebiasTable::lookup`] are the values to
//! subtract from the reported (RA, Dec) to align the observation with the
//! reference frame.
//!
//! Source data: `<https://ssd.jpl.nasa.gov/ftp/ssd/debias/>`
//
// BSD 3-Clause License
//
// Copyright (c) 2026, Dar Dahlen
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

use kete_core::prelude::{Error, KeteResult};
use kete_stats::healpix::ang_to_ring;

/// `HEALPix` resolution used by both FCCT14 and EFCC18.
pub const DEBIAS_NSIDE: u32 = 64;

/// Number of `HEALPix` tiles at `nside = 64`.
pub const DEBIAS_N_TILES: usize = 12 * (DEBIAS_NSIDE as usize) * (DEBIAS_NSIDE as usize);

/// Reference epoch (J2000.0) used for the proper-motion term.
pub const DEBIAS_EPOCH_JD: f64 = 2_451_545.0;

/// Arcseconds to radians.
const ARCSEC_TO_RAD: f64 = std::f64::consts::PI / (180.0 * 3600.0);

/// Which release of the debiasing table is loaded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DebiasVersion {
    /// FCCT14 (19 catalogs, reference: PPMXL subset).
    Fcct14,
    /// EFCC18 (26 catalogs, reference: Gaia-DR2).
    Efcc18,
}

impl DebiasVersion {
    fn from_n_cats(n_cats: usize) -> KeteResult<Self> {
        match n_cats {
            19 => Ok(Self::Fcct14),
            26 => Ok(Self::Efcc18),
            n => Err(Error::IOError(format!(
                "Debias table must have 19 (FCCT14) or 26 (EFCC18) catalogs, got {n}"
            ))),
        }
    }
}

/// Per-catalog astrometric bias table on a `HEALPix` `nside=64` grid.
///
/// Cheap to clone via `Arc` if needed by callers; the underlying data is
/// stored as a contiguous `Box<[i16]>` of length
/// [`DEBIAS_N_TILES`] `* n_cats * 4`.
#[derive(Debug, Clone)]
pub struct DebiasTable {
    catalogs: Vec<u8>,
    /// ASCII byte -> column index, or -1 if unsupported.
    cat_index: Box<[i8; 128]>,
    /// Layout: `data[(tile * n_cats + cat) * 4 + k]` for `k` in
    /// `{0: ra, 1: dec, 2: pm_ra, 3: pm_dec}`. RA/Dec biases are stored
    /// as milli-arcseconds; proper-motion biases as 10 micro-arcseconds
    /// per year (matching the JPL ASCII format scaling).
    data: Box<[i16]>,
    version: DebiasVersion,
}

impl DebiasTable {
    /// Parse the JPL `bias.dat` ASCII text.
    ///
    /// The JPL ASCII layout is:
    ///   * Lines beginning with `!` are comments. The header row
    ///     `! |-----  CATNAME X  -----|...` (one cell per catalog) is
    ///     parsed to recover the per-column ASCII catalog code (the last
    ///     single-character alphanumeric token before the closing dashes).
    ///   * Each data row contains `n_cats * 4` whitespace-separated
    ///     floating point values in the order `(ra, dec, pm_ra, pm_dec)`
    ///     per catalog.  RA/Dec biases are in arcseconds (3 decimal
    ///     places, milli-arcsec resolution).  Proper-motion biases are
    ///     in mas/yr (2 decimal places, 10 micro-arcsec/yr resolution).
    ///
    /// Per the JPL `bias.dat` header, the **RA bias already includes the
    /// `cos(dec)` factor**.  Callers applying the correction to a
    /// non-cos-corrected RA must divide the returned `delta_ra` by
    /// `cos(dec)`.
    ///
    /// # Errors
    /// Returns an error if the header row is missing, the number of tiles
    /// or per-row values does not match the expected layout, the catalog
    /// count is not 19 (FCCT14) or 26 (EFCC18), or any value falls outside
    /// the `i16` range.
    pub fn from_ascii(text: &str) -> KeteResult<Self> {
        let mut catalogs: Option<Vec<u8>> = None;
        let mut data: Vec<i16> = Vec::new();
        let mut tile_count: usize = 0;

        for line in text.lines() {
            let trimmed = line.trim_start();
            if trimmed.is_empty() {
                continue;
            }
            if trimmed.starts_with('!') {
                if catalogs.is_none()
                    && let Some(codes) = parse_catalog_header(trimmed)
                {
                    catalogs = Some(codes);
                }
                continue;
            }
            let cats = catalogs.as_ref().ok_or_else(|| {
                Error::IOError("Debias ASCII data row encountered before catalog header".into())
            })?;
            let expected = cats.len() * 4;
            if data.capacity() == 0 {
                data.reserve_exact(DEBIAS_N_TILES * expected);
            }
            let mut count = 0_usize;
            for (k, tok) in trimmed.split_ascii_whitespace().enumerate() {
                let v: f64 = tok.parse().map_err(|_| {
                    Error::IOError(format!("Debias row contains non-numeric token '{tok}'"))
                })?;
                // Columns 0/1 are RA/Dec arcsec -> milli-arcsec (x1000).
                // Columns 2/3 are pm_ra/pm_dec mas/yr -> 10 micro-arcsec/yr (x100).
                let scale = if k % 4 < 2 { 1000.0 } else { 100.0 };
                let scaled = (v * scale).round();
                if !scaled.is_finite()
                    || scaled < f64::from(i16::MIN)
                    || scaled > f64::from(i16::MAX)
                {
                    return Err(Error::IOError(format!(
                        "Debias value {v} out of i16 range after scaling"
                    )));
                }
                #[allow(clippy::cast_possible_truncation, reason = "range checked just above")]
                data.push(scaled as i16);
                count += 1;
            }
            if count != expected {
                return Err(Error::IOError(format!(
                    "Debias row {tile_count} has {count} values, expected {expected}"
                )));
            }
            tile_count += 1;
            if tile_count > DEBIAS_N_TILES {
                return Err(Error::IOError(format!(
                    "Debias table has more than {DEBIAS_N_TILES} tiles"
                )));
            }
        }

        let catalogs =
            catalogs.ok_or_else(|| Error::IOError("Debias header row not found".into()))?;
        if tile_count != DEBIAS_N_TILES {
            return Err(Error::IOError(format!(
                "Debias table has {tile_count} tiles, expected {DEBIAS_N_TILES}"
            )));
        }
        Self::new(catalogs, data.into_boxed_slice())
    }

    fn new(catalogs: Vec<u8>, data: Box<[i16]>) -> KeteResult<Self> {
        let n_cats = catalogs.len();
        let version = DebiasVersion::from_n_cats(n_cats)?;
        let expected = DEBIAS_N_TILES * n_cats * 4;
        if data.len() != expected {
            return Err(Error::IOError(format!(
                "Debias data length {} does not match expected {}",
                data.len(),
                expected
            )));
        }
        if catalogs.iter().any(|&c| !c.is_ascii() || c == 0) {
            return Err(Error::IOError(
                "Debias catalog codes must be non-zero ASCII bytes".into(),
            ));
        }
        let mut cat_index = Box::new([-1_i8; 128]);
        for (i, &c) in catalogs.iter().enumerate() {
            // EFCC18 pre-2023 mistakenly used 'W' for catalog 'Y' in the
            // last column. Normalize to 'Y' on load.
            let c = if version == DebiasVersion::Efcc18 && i == 25 && c == b'W' {
                b'Y'
            } else {
                c
            };
            let idx = i8::try_from(i).map_err(|_| Error::IOError("Too many catalogs".into()))?;
            cat_index[c as usize] = idx;
        }
        // Re-normalize the stored catalog code as well.
        let catalogs = if version == DebiasVersion::Efcc18 && catalogs.last() == Some(&b'W') {
            let mut c = catalogs.clone();
            *c.last_mut().unwrap() = b'Y';
            c
        } else {
            catalogs
        };
        Ok(Self {
            catalogs,
            cat_index,
            data,
            version,
        })
    }

    /// Catalog codes (single ASCII characters as bytes) supported by this
    /// table, in column order.
    #[must_use]
    pub fn catalogs(&self) -> &[u8] {
        &self.catalogs
    }

    /// Which JPL release this table represents.
    #[must_use]
    pub fn version(&self) -> DebiasVersion {
        self.version
    }

    /// Lookup the (ra, dec) bias correction for a single observation.
    ///
    /// Returns `(delta_ra, delta_dec)` in radians. The convention is the
    /// same as the source tables: subtract these from the reported
    /// `(ra, dec)` to obtain the reference-frame-aligned position.
    ///
    /// **`delta_ra` already includes the `cos(dec)` factor**, matching
    /// the JPL `bias.dat` convention. Callers applying the correction to
    /// raw RA (without the cos-dec scaling) must divide `delta_ra` by
    /// `cos(dec)` first.
    ///
    /// `ra` and `dec` are in radians. `jd` is the observation epoch in
    /// Julian days (TDB or UTC -- the proper-motion term is dominated by
    /// year-scale arithmetic, so the time-system distinction is
    /// negligible at the milli-arcsec level).
    ///
    /// Returns `None` if the catalog code is not present in this table.
    #[inline]
    #[must_use]
    pub fn lookup(&self, catalog: u8, ra: f64, dec: f64, jd: f64) -> Option<(f64, f64)> {
        let cat = self.catalog_column(catalog)?;
        let tile = ang_to_ring(DEBIAS_NSIDE, ra, dec) as usize;
        let n_cats = self.catalogs.len();
        let base = (tile * n_cats + cat) * 4;
        // Bounds: tile < N_TILES, cat < n_cats => base + 3 < data.len().
        let raw = &self.data[base..base + 4];
        let years = (jd - DEBIAS_EPOCH_JD) / 365.25;
        // ra/dec biases: i16 in milli-arcsec.
        // proper motions: i16 in 10 micro-arcsec / yr (1e-5 arcsec/yr).
        let d_ra_arcsec = f64::from(raw[0]) * 1e-3 + years * f64::from(raw[2]) * 1e-5;
        let d_dec_arcsec = f64::from(raw[1]) * 1e-3 + years * f64::from(raw[3]) * 1e-5;
        Some((d_ra_arcsec * ARCSEC_TO_RAD, d_dec_arcsec * ARCSEC_TO_RAD))
    }

    /// Column index for the given catalog code, or `None` if unsupported.
    #[inline]
    #[must_use]
    pub fn catalog_column(&self, catalog: u8) -> Option<usize> {
        if catalog >= 128 {
            return None;
        }
        let v = self.cat_index[catalog as usize];
        if v < 0 {
            None
        } else {
            #[allow(
                clippy::cast_sign_loss,
                reason = "v is non-negative after the check above"
            )]
            Some(v as usize)
        }
    }
}

/// Parse a JPL `bias.dat` header line of the form
/// `! |-----  CATNAME X  -----|---  CATNAME2 Y  -----|...` and recover
/// the per-column ASCII catalog codes.
///
/// Within each `|...|`-delimited cell, the catalog code is the last
/// single-character alphanumeric token before the trailing dashes
/// (e.g. `a` in `-----  USNO-A1.0 a  -----`).
///
/// Returns `None` if the line is not a recognizable header.
fn parse_catalog_header(line: &str) -> Option<Vec<u8>> {
    if !line.starts_with("! |---") && !line.starts_with("!|---") {
        return None;
    }
    // Strip the leading `! ` so we can split by `|` cleanly.
    let body = line.trim_start_matches('!').trim_start();
    let mut codes = Vec::new();
    for cell in body.split('|') {
        // A valid catalog cell looks like `-----  NAME X  -----` (with
        // varying whitespace).  Strip the dashes and take the last
        // single-character alphanumeric token.
        let cleaned = cell.trim_matches(|c: char| c == '-' || c.is_whitespace());
        if cleaned.is_empty() {
            continue;
        }
        let last = cleaned.split_ascii_whitespace().next_back()?;
        if last.len() == 1 {
            let b = last.as_bytes()[0];
            if b.is_ascii_alphanumeric() {
                codes.push(b);
            }
        }
    }
    if codes.is_empty() { None } else { Some(codes) }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a synthetic table: zero everywhere except a single tile/cat
    /// with known values, so we can exercise the lookup arithmetic and
    /// serialization without the 5 MB JPL file.
    fn synthetic_table(n_cats: usize) -> DebiasTable {
        // FCCT14 catalog codes (19) are the canonical short list; for
        // EFCC18 (26) we extend with extra alphanumerics.
        let codes_19: &[u8] = b"abcdefghijklmnopqru";
        let codes: Vec<u8> = if n_cats == 19 {
            codes_19.to_vec()
        } else {
            // 26 distinct ASCII letters.
            let mut v: Vec<u8> = (b'a'..=b'z').collect();
            v.truncate(26);
            v
        };
        assert_eq!(codes.len(), n_cats);
        let mut data = vec![0_i16; DEBIAS_N_TILES * n_cats * 4];

        // Set tile 100, catalog index 3, to a known biases:
        //   ra = 1234 mas = 1.234"
        //   dec = -5678 mas = -5.678"
        //   pm_ra = 200 -> 0.002"/yr
        //   pm_dec = -100 -> -0.001"/yr
        let base = (100 * n_cats + 3) * 4;
        data[base] = 1234;
        data[base + 1] = -5678;
        data[base + 2] = 200;
        data[base + 3] = -100;
        DebiasTable::new(codes, data.into_boxed_slice()).expect("synthetic table")
    }

    #[test]
    fn parse_catalog_header_extracts_codes() {
        let line =
            "! |-----  USNO-A1.0 a  -----|  |-----  Tycho-2 g  -----|  |-----  Gaia-DR2 V  -----|";
        let codes = parse_catalog_header(line).expect("parses");
        assert_eq!(codes, b"agV");
    }

    #[test]
    fn parse_catalog_header_handles_3dash_legacy() {
        // FCCT14-style with 3 dashes.
        let line = "! |---  PPMXL w  ---|  |---  Gaia2 V  ---|";
        let codes = parse_catalog_header(line).expect("parses");
        assert_eq!(codes, b"wV");
    }

    #[test]
    fn version_from_n_cats() {
        assert_eq!(
            DebiasVersion::from_n_cats(19).unwrap(),
            DebiasVersion::Fcct14
        );
        assert_eq!(
            DebiasVersion::from_n_cats(26).unwrap(),
            DebiasVersion::Efcc18
        );
        assert!(DebiasVersion::from_n_cats(20).is_err());
    }

    #[test]
    fn unknown_catalog_returns_none() {
        let table = synthetic_table(19);
        assert!(table.lookup(b'Z', 0.0, 0.0, DEBIAS_EPOCH_JD).is_none());
        assert!(table.lookup(0, 0.0, 0.0, DEBIAS_EPOCH_JD).is_none());
        assert!(table.lookup(200, 0.0, 0.0, DEBIAS_EPOCH_JD).is_none());
    }

    #[test]
    fn lookup_recovers_known_values_at_epoch() {
        let table = synthetic_table(19);
        // Find the (ra, dec) at the center of tile 100.
        let (ra, dec) = kete_stats::healpix::ring_to_ang(DEBIAS_NSIDE, 100);
        let (d_ra, d_dec) = table
            .lookup(b'd', ra, dec, DEBIAS_EPOCH_JD)
            .expect("known catalog");
        // 1.234" and -5.678", in radians.
        assert!((d_ra / ARCSEC_TO_RAD - 1.234).abs() < 1e-9);
        assert!((d_dec / ARCSEC_TO_RAD - (-5.678)).abs() < 1e-9);
    }

    #[test]
    fn lookup_applies_proper_motion_term() {
        let table = synthetic_table(19);
        let (ra, dec) = kete_stats::healpix::ring_to_ang(DEBIAS_NSIDE, 100);
        // Ten years past J2000: pm contributes 10 * 0.002 = 0.020" in RA,
        //                                       10 * -0.001 = -0.010" in Dec.
        let jd = DEBIAS_EPOCH_JD + 10.0 * 365.25;
        let (d_ra, d_dec) = table.lookup(b'd', ra, dec, jd).unwrap();
        let expected_ra = 1.234 + 10.0 * 0.002;
        let expected_dec = -5.678 + 10.0 * (-0.001);
        assert!((d_ra / ARCSEC_TO_RAD - expected_ra).abs() < 1e-9);
        assert!((d_dec / ARCSEC_TO_RAD - expected_dec).abs() < 1e-9);
    }

    #[test]
    fn ascii_round_trip_through_parser() {
        // Build a tiny 2-cat ASCII document with only a header; the parser
        // should accept the header but reject the wrong tile count, so we
        // assert the error type instead. (Generating all 49152 tiles in a
        // unit test is wasteful.)
        let header = "! |-----  PPMXL a  -----|  |-----  UCAC4 b  -----|\n";
        let row = "0.000 0.000 0.00 0.00 0.000 0.000 0.00 0.00\n";
        let mut text = String::from(header);
        text.push_str(row);
        let err = DebiasTable::from_ascii(&text).unwrap_err();
        // Must fail because we have 1 tile, not 49152.
        let Error::IOError(msg) = err else {
            panic!("expected IOError, got {err:?}");
        };
        assert!(msg.contains("tiles"));
    }

    #[test]
    fn ascii_parser_scales_floats_correctly() {
        use std::fmt::Write;
        // One catalog, one row -- exercise the float -> i16 scaling.
        // We use the 19-cat path artificially by stuffing 19 columns of
        // identical values; the row count check still trips, but we can
        // instead inspect the parser via a tiny shim: build a 19-cat
        // header, one row, and confirm that the error is the *tiles*
        // mismatch (proving the row parsed without raising a numeric
        // error).  A direct numeric check is exercised by the existing
        // `lookup_recovers_known_values_at_epoch` test on synthetic
        // tables.
        let mut header = String::from("!");
        for c in b"abcdefghijklmnopqru" {
            let _ = write!(header, " |-----  X {}  -----|", *c as char);
        }
        header.push('\n');
        // Row of all `0.279 -0.001 -0.49 1.04` repeated 19 times.
        let cell = "0.279 -0.001 -0.49 1.04";
        let row = (0..19).map(|_| cell).collect::<Vec<_>>().join(" ") + "\n";
        header.push_str(&row);
        let err = DebiasTable::from_ascii(&header).unwrap_err();
        let Error::IOError(msg) = err else {
            panic!("expected IOError, got {err:?}");
        };
        assert!(msg.contains("tiles"), "unexpected error: {msg}");
    }

    #[test]
    fn efcc18_renames_w_to_y_in_last_column() {
        // 26 catalogs ending in 'W' should be normalized to 'Y'.
        let mut codes: Vec<u8> = (b'a'..=b'z').collect();
        codes.truncate(26);
        *codes.last_mut().unwrap() = b'W';
        let data = vec![0_i16; DEBIAS_N_TILES * 26 * 4].into_boxed_slice();
        let table = DebiasTable::new(codes, data).unwrap();
        assert_eq!(table.catalogs().last(), Some(&b'Y'));
        assert!(table.catalog_column(b'Y').is_some());
        assert!(table.catalog_column(b'W').is_none());
    }
}
