// SPDX-FileCopyrightText: 2026 Dar Dahlen
// SPDX-License-Identifier: BSD-3-Clause

//! # PTF Fov definitions.

use super::FovLike;
use super::fov_like::{patches_corners, patches_pointing};
use crate::fov::FOV;
use crate::geometry::closest_inside;
use crate::geometry::{Contains, SkyPatch, SphericalPolygon};
use crate::{frames::Vector, prelude::*};
use std::{fmt::Display, str::FromStr};

/// PTF Filters used over the course of the survey.
#[derive(PartialEq, Clone, Copy, Debug)]
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
#[derive(Debug, Clone)]
pub struct PtfCcd {
    /// State of the observer
    pub(crate) observer: State<Equatorial>,

    /// Patch of sky
    pub patch: SphericalPolygon,

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
    #[must_use]
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
        let patch = SphericalPolygon::from_corners(corners, 0.0);
        Self {
            observer,
            patch,
            field,
            ccdid,
            filter,
            filename,
            info_bits,
            seeing,
        }
    }
}

impl FovLike for PtfCcd {
    type ChildFov = Self;

    fn get_child(&self, index: usize) -> Self::ChildFov {
        assert!(index == 0, "FOV only has a single patch");
        self.clone()
    }

    #[inline]
    fn into_fov(self) -> FOV {
        FOV::PtfCcd(self)
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
        Ok(self.patch.corners())
    }
}

/// Ptf frame data, full collection of all CCDs
#[derive(Debug, Clone)]
pub struct PtfField {
    /// Individual CCDs
    pub(crate) ccds: Vec<PtfCcd>,

    /// Observer position
    pub(crate) observer: State<Equatorial>,

    /// Field ID
    pub field: u32,

    /// Filter
    pub filter: PTFFilter,
}

impl PtfField {
    /// Construct a new [`PtfField`] from a list of ccds.
    /// These ccds must be from the same field and having matching value as
    /// appropriate.
    ///
    /// # Errors
    /// Construction will fail if no ccds are provided or if they are inconsistent.
    pub fn new(ccds: Vec<PtfCcd>) -> KeteResult<Self> {
        if ccds.is_empty() {
            Err(Error::ValueError("Ptf Field must contains PtfCcd".into()))?;
        }

        #[allow(clippy::missing_panics_doc, reason = "ccds is not empty")]
        let first = ccds.first().unwrap();

        let observer = first.observer().clone();
        let field = first.field;
        let filter = first.filter;

        for ccd in &ccds {
            if ccd.field != field
                || ccd.filter != filter
                || !ccd.observer().epoch.same_instant(&observer.epoch)
            {
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
    type ChildFov = PtfCcd;

    fn get_child(&self, index: usize) -> Self::ChildFov {
        self.ccds[index].clone()
    }

    #[inline]
    fn into_fov(self) -> FOV {
        FOV::PtfField(self)
    }

    fn observer(&self) -> &State<Equatorial> {
        &self.observer
    }

    fn contains(&self, obs_to_obj: &Vector<Equatorial>) -> (usize, Contains) {
        closest_inside(self.ccds.iter().map(|x| x.contains(obs_to_obj).1))
    }

    fn n_patches(&self) -> usize {
        self.ccds.len()
    }

    fn pointing(&self) -> KeteResult<Vector<Equatorial>> {
        patches_pointing(&self.ccds)
    }

    fn corners(&self) -> KeteResult<Vec<Vector<Equatorial>>> {
        patches_corners(&self.ccds)
    }
}
