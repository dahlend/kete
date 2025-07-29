//! # PTF Fov definitions.
// BSD 3-Clause License
//
// Copyright (c) 2025, Dar Dahlen
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

use super::patches::closest_inside;
use super::{Contains, FOV, FovLike, OnSkyRectangle, SkyPatch};
use crate::{frames::Vector, prelude::*};
use serde::{Deserialize, Serialize};
use std::{fmt::Display, str::FromStr};

/// PTF Filters used over the course of the survey.
#[derive(PartialEq, Clone, Copy, Debug, Serialize, Deserialize)]
pub enum PTFFilter {
    /// G Band Filter
    G,

    /// R Band Filter
    R,

    /// Hydrogen Alpha 656 nm Filter
    HA656,

    /// Hydrogen Alpha 663nm filter
    HA663,
}

impl Display for PTFFilter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::G => f.write_str("G"),
            Self::R => f.write_str("R"),
            Self::HA656 => f.write_str("HA656"),
            Self::HA663 => f.write_str("HA663"),
        }
    }
}

impl FromStr for PTFFilter {
    type Err = Error;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_uppercase().as_str() {
            "G" => Ok(Self::G),
            "R" => Ok(Self::R),
            "HA656" => Ok(Self::HA656),
            "HA663" => Ok(Self::HA663),
            _ => Err(Error::ValueError(
                "PTF Filter has to be one of ('G', 'R', 'HA656', 'HA663')".into(),
            )),
        }
    }
}

/// PTF frame data, single ccd
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct PtfCcd {
    /// State of the observer
    observer: State<Equatorial>,

    /// Patch of sky
    pub patch: OnSkyRectangle,

    /// Field ID
    pub field: u32,

    /// Which CCID was the frame taken with
    pub ccdid: u8,

    /// Filter
    pub filter: PTFFilter,

    /// Filename of the processed image
    pub filename: Box<str>,

    /// Infobits flag
    pub info_bits: u32,

    /// FWHM seeing conditions
    pub seeing: f32,
}

impl PtfCcd {
    /// Create a Ptf field of view
    pub fn new(
        corners: [Vector<Equatorial>; 4],
        observer: State<Equatorial>,
        field: u32,
        ccdid: u8,
        filter: PTFFilter,
        filename: Box<str>,
        info_bits: u32,
        seeing: f32,
    ) -> Self {
        let patch = OnSkyRectangle::from_corners(corners, 0.0);
        Self {
            patch,
            observer,
            field,
            ccdid,
            filter,
            filename,
            seeing,
            info_bits,
        }
    }
}

impl FovLike for PtfCcd {
    fn get_fov(&self, index: usize) -> FOV {
        if index != 0 {
            panic!("FOV only has a single patch")
        }
        FOV::PtfCcd(self.clone())
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

/// Ptf frame data, full collection of all CCDs
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct PtfField {
    /// Individual CCDs
    ccds: Vec<PtfCcd>,

    /// Observer position
    observer: State<Equatorial>,

    /// Field ID
    pub field: u32,

    /// Filter
    pub filter: PTFFilter,
}

impl PtfField {
    /// Construct a new [`PtfField`] from a list of ccds.
    /// These ccds must be from the same field and having matching value as
    /// appropriate.
    pub fn new(ccds: Vec<PtfCcd>) -> KeteResult<Self> {
        if ccds.is_empty() {
            Err(Error::ValueError("Ptf Field must contains PtfCcd".into()))?;
        }

        let first = ccds.first().unwrap();

        let observer = first.observer().clone();
        let field = first.field;
        let filter = first.filter;

        for ccd in ccds.iter() {
            if ccd.field != field || ccd.filter != filter || ccd.observer().jd != observer.jd {
                Err(Error::ValueError(
                    "All PtfCcds must have matching values except CCD ID etc.".into(),
                ))?;
            }
        }
        Ok(Self {
            ccds,
            observer,
            field,
            filter,
        })
    }
}

impl FovLike for PtfField {
    fn get_fov(&self, index: usize) -> FOV {
        FOV::PtfCcd(self.ccds[index].clone())
    }

    fn observer(&self) -> &State<Equatorial> {
        &self.observer
    }

    fn contains(&self, obs_to_obj: &Vector<Equatorial>) -> (usize, Contains) {
        closest_inside(
            &self
                .ccds
                .iter()
                .map(|x| x.contains(obs_to_obj).1)
                .collect::<Vec<_>>(),
        )
    }

    fn n_patches(&self) -> usize {
        self.ccds.len()
    }

    #[inline]
    fn pointing(&self) -> KeteResult<Vector<Equatorial>> {
        if self.ccds.is_empty() {
            Err(Error::ValueError("ZtfField has no ccd quads".into()))
        } else {
            // return the average pointing of all ccd quads
            Ok(self
                .ccds
                .iter()
                .fold(Vector::new([0.0; 3]), |acc, x| acc + x.pointing().unwrap()))
        }
    }

    #[inline]
    fn corners(&self) -> KeteResult<Vec<Vector<Equatorial>>> {
        if self.ccds.is_empty() {
            Err(Error::ValueError("ZtfField has no ccd quads".into()))
        } else {
            // return all the corners of all ccd quads
            Ok(self
                .ccds
                .iter()
                .flat_map(|x| x.corners().unwrap())
                .collect())
        }
    }
}
