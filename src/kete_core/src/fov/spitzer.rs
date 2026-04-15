//! # Spitzer Space Telescope FOV definitions.
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

use super::{Contains, FovLike, OnSkyRectangle, SkyPatch};
use crate::fov::FOV;
use crate::frames::Vector;
use crate::prelude::*;
use std::{fmt::Display, str::FromStr};

/// Spitzer instrument bands.
#[derive(PartialEq, Clone, Copy, Debug)]
pub enum SpitzerBand {
    /// IRAC channel 1 (3.6 um).
    Irac1,
    /// IRAC channel 2 (4.5 um).
    Irac2,
    /// IRAC channel 3 (5.8 um).
    Irac3,
    /// IRAC channel 4 (8.0 um).
    Irac4,
    /// MIPS 24 um.
    Mips24,
    /// MIPS 70 um.
    Mips70,
    /// MIPS 160 um.
    Mips160,
    /// IRS Peak-Up Blue (13.3-18.7 um).
    IrsPeakUpBlue,
    /// IRS Peak-Up Red (18.5-26.0 um).
    IrsPeakUpRed,
}

impl Display for SpitzerBand {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Irac1 => f.write_str("IRAC1"),
            Self::Irac2 => f.write_str("IRAC2"),
            Self::Irac3 => f.write_str("IRAC3"),
            Self::Irac4 => f.write_str("IRAC4"),
            Self::Mips24 => f.write_str("MIPS24"),
            Self::Mips70 => f.write_str("MIPS70"),
            Self::Mips160 => f.write_str("MIPS160"),
            Self::IrsPeakUpBlue => f.write_str("IRS Peak-Up Blue"),
            Self::IrsPeakUpRed => f.write_str("IRS Peak-Up Red"),
        }
    }
}

impl FromStr for SpitzerBand {
    type Err = Error;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_uppercase().as_str() {
            "IRAC1" => Ok(Self::Irac1),
            "IRAC2" => Ok(Self::Irac2),
            "IRAC3" => Ok(Self::Irac3),
            "IRAC4" => Ok(Self::Irac4),
            "MIPS24" => Ok(Self::Mips24),
            "MIPS70" => Ok(Self::Mips70),
            "MIPS160" => Ok(Self::Mips160),
            "IRS PEAK-UP BLUE" => Ok(Self::IrsPeakUpBlue),
            "IRS PEAK-UP RED" => Ok(Self::IrsPeakUpRed),
            _ => Err(Error::ValueError(
                "SpitzerBand must be one of ('IRAC1', 'IRAC2', 'IRAC3', 'IRAC4', 'MIPS24', 'MIPS70', 'MIPS160', 'IRS Peak-Up Blue', 'IRS Peak-Up Red')".into(),
            )),
        }
    }
}

/// Spitzer BCD frame FOV, covering both IRAC and MIPS instruments.
#[derive(Debug, Clone)]
pub struct SpitzerFrame {
    /// State of the observer.
    pub(crate) observer: State<Equatorial>,

    /// Patch of sky.
    pub(crate) patch: OnSkyRectangle,

    /// IRSA ``obs_publisher_did`` identifying this BCD plane.
    pub obs_id: Box<str>,

    /// Instrument band.
    pub band: SpitzerBand,

    /// IRSA IBE artifact URI for the BCD FITS file.
    pub artifact_uri: Box<str>,

    /// Exposure duration in seconds.
    pub duration: f64,
}

impl SpitzerFrame {
    /// Create a Spitzer frame from a pointing vector, rotation and explicit FOV size.
    #[must_use]
    pub fn new(
        pointing: Vector<Equatorial>,
        rotation: f64,
        observer: State<Equatorial>,
        obs_id: Box<str>,
        band: SpitzerBand,
        artifact_uri: Box<str>,
        width: f64,
        height: f64,
        duration: f64,
    ) -> Self {
        let patch = OnSkyRectangle::new(pointing, rotation, width, height);
        Self {
            observer,
            patch,
            obs_id,
            band,
            artifact_uri,
            duration,
        }
    }

    /// Create a Spitzer frame from the 4 corners of the FOV.
    #[must_use]
    pub fn from_corners(
        corners: [Vector<Equatorial>; 4],
        observer: State<Equatorial>,
        obs_id: Box<str>,
        band: SpitzerBand,
        artifact_uri: Box<str>,
        duration: f64,
    ) -> Self {
        // 1 arcminute tolerance for the CAOM polygon not forming a perfect rectangle.
        let patch = OnSkyRectangle::from_corners(corners, 60_f64.recip().to_radians());
        Self {
            observer,
            patch,
            obs_id,
            band,
            artifact_uri,
            duration,
        }
    }
}

impl FovLike for SpitzerFrame {
    type ChildFov = Self;

    #[inline]
    fn get_child(&self, index: usize) -> Self::ChildFov {
        assert!(index == 0, "SpitzerFrame FOV only has a single patch");
        self.clone()
    }

    #[inline]
    fn into_fov(self) -> FOV {
        FOV::Spitzer(self)
    }

    #[inline]
    fn observer(&self) -> &State<Equatorial> {
        &self.observer
    }

    #[inline]
    fn contains(&self, obs_to_obj: &Vector<Equatorial>) -> (usize, Contains) {
        (0, self.patch.contains(obs_to_obj))
    }

    #[inline]
    fn n_patches(&self) -> usize {
        1
    }

    #[inline]
    fn pointing(&self) -> KeteResult<Vector<Equatorial>> {
        Ok(self.patch.pointing())
    }

    #[inline]
    fn corners(&self) -> KeteResult<Vec<Vector<Equatorial>>> {
        Ok(self.patch.corners().into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::IRAC_WIDTH;
    use crate::desigs::Desig;
    use crate::frames::Equatorial;
    use crate::time::{TDB, Time};

    fn make_observer() -> State<Equatorial> {
        State::new(
            Desig::Empty,
            Time::<TDB>::new(2451545.0),
            Vector::new([1.0, 0.0, 0.0]),
            Vector::new([0.0, 0.0, 0.0]),
            10,
        )
    }

    #[test]
    fn test_spitzer_band_display_roundtrip() {
        for band in [
            SpitzerBand::Irac1,
            SpitzerBand::Irac2,
            SpitzerBand::Irac3,
            SpitzerBand::Irac4,
            SpitzerBand::Mips24,
            SpitzerBand::Mips70,
            SpitzerBand::Mips160,
            SpitzerBand::IrsPeakUpBlue,
            SpitzerBand::IrsPeakUpRed,
        ] {
            let s = band.to_string();
            let parsed: SpitzerBand = s.parse().unwrap();
            assert_eq!(parsed, band);
        }
    }

    #[test]
    fn test_spitzer_from_pointing() {
        let observer = make_observer();
        let pointing: Vector<Equatorial> = [1.0, 0.0, 0.0].into();
        let fov = SpitzerFrame::new(
            pointing,
            0.0,
            observer,
            "ivo://test/obs_id".into(),
            SpitzerBand::Irac1,
            "".into(),
            IRAC_WIDTH,
            IRAC_WIDTH,
            12.0,
        );
        assert_eq!(fov.band, SpitzerBand::Irac1);
        assert_eq!(&*fov.obs_id, "ivo://test/obs_id");
        assert_eq!(fov.n_patches(), 1);

        // centroid should be inside
        let (_, c) = fov.contains(&[1.0, 0.0, 0.0].into());
        assert!(c.is_inside(), "centroid must be inside");

        // far-off point must be outside
        let (_, c) = fov.contains(&[0.0, 1.0, 0.0].into());
        assert!(!c.is_inside(), "orthogonal point must be outside");
    }

    #[test]
    fn test_spitzer_from_corners() {
        let observer = make_observer();
        // Build a small square centred near [1,0,0]
        let d = 0.0005_f64; // half-width ~0.03 deg, well within IRAC FOV
        let corners: [Vector<Equatorial>; 4] = [
            [1.0, -d, -d].into(),
            [1.0, d, -d].into(),
            [1.0, d, d].into(),
            [1.0, -d, d].into(),
        ];
        let fov = SpitzerFrame::from_corners(
            corners,
            observer,
            "ivo://test/obs2".into(),
            SpitzerBand::Mips24,
            "".into(),
            10.0,
        );
        assert_eq!(fov.band, SpitzerBand::Mips24);

        // centroid inside
        let (_, c) = fov.contains(&[1.0, 0.0, 0.0].into());
        assert!(c.is_inside(), "centroid must be inside corners FOV");

        // far point outside
        let (_, c) = fov.contains(&[1.0, 1.0, 0.0].into());
        assert!(!c.is_inside(), "far point must be outside corners FOV");
    }

    #[test]
    fn test_spitzer_into_fov() {
        let observer = make_observer();
        let pointing: Vector<Equatorial> = [1.0, 0.0, 0.0].into();
        let frame = SpitzerFrame::new(
            pointing,
            0.0,
            observer,
            "test".into(),
            SpitzerBand::Irac3,
            "".into(),
            IRAC_WIDTH,
            IRAC_WIDTH,
            12.0,
        );
        let fov = frame.into_fov();
        assert!(matches!(fov, FOV::Spitzer(_)));
    }
}
