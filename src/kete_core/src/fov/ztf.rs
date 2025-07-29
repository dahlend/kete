//! # ZTF Fov definitions.
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

use super::patches::closest_inside;
use super::{Contains, FOV, FovLike, OnSkyRectangle, SkyPatch};
use crate::{frames::Vector, prelude::*};
use serde::{Deserialize, Serialize};

/// ZTF frame data, single quad of a single chip
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ZtfCcdQuad {
    /// State of the observer
    observer: State<Equatorial>,

    /// Patch of sky
    patch: OnSkyRectangle,

    /// Field ID
    pub field: u32,

    /// File Frac Day
    /// String representation of the filename for this frame.
    pub filefracday: u64,

    /// Magnitude limit of this frame
    pub maglimit: f64,

    /// Filter ID
    pub fid: u64,

    /// Filter code used for the frame
    pub filtercode: Box<str>,

    /// Image Type Code
    pub imgtypecode: Box<str>,

    /// Which CCID was the frame taken with
    pub ccdid: u8,

    /// Quadrant ID
    pub qid: u8,
}

impl ZtfCcdQuad {
    /// Create a ZTF field of view
    pub fn new(
        corners: [Vector<Equatorial>; 4],
        observer: State<Equatorial>,
        field: u32,
        filefracday: u64,
        ccdid: u8,
        filtercode: Box<str>,
        imgtypecode: Box<str>,
        qid: u8,
        maglimit: f64,
        fid: u64,
    ) -> Self {
        let patch = OnSkyRectangle::from_corners(corners, 0.0);
        Self {
            patch,
            observer,
            field,
            filefracday,
            ccdid,
            filtercode,
            imgtypecode,
            qid,
            maglimit,
            fid,
        }
    }
}

impl FovLike for ZtfCcdQuad {
    fn get_fov(&self, index: usize) -> FOV {
        if index != 0 {
            panic!("FOV only has a single patch")
        }
        FOV::ZtfCcdQuad(self.clone())
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

/// ZTF frame data, single quad of a single chip
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ZtfField {
    /// Individual CCD quads
    ccd_quads: Vec<ZtfCcdQuad>,

    /// Observer position
    observer: State<Equatorial>,

    /// Field ID
    pub field: u32,

    /// Filter ID
    pub fid: u64,

    /// Filter code used for the frame
    pub filtercode: Box<str>,

    /// Image Type Code
    pub imgtypecode: Box<str>,
}

impl ZtfField {
    /// Construct a new [`ZtfField`] from a list of ccd quads.
    /// These ccd quads must be from the same field and having matching value as
    /// appropriate.
    pub fn new(ccd_quads: Vec<ZtfCcdQuad>) -> KeteResult<Self> {
        if ccd_quads.is_empty() {
            Err(Error::ValueError(
                "Ztf Field must contains ZtfCcdQuads".into(),
            ))?;
        }

        let first = ccd_quads.first().unwrap();

        let observer = first.observer().clone();
        let field = first.field;
        let fid = first.fid;
        let filtercode = first.filtercode.clone();
        let imgtypecode = first.imgtypecode.clone();

        for ccd in ccd_quads.iter() {
            if ccd.field != field
                || ccd.fid != fid
                || ccd.filtercode != filtercode
                || ccd.imgtypecode != imgtypecode
                || ccd.observer().jd != observer.jd
            {
                Err(Error::ValueError(
                    "All ZtfCcdQuads must have matching values except CCD ID etc.".into(),
                ))?;
            }
        }
        Ok(Self {
            ccd_quads,
            observer,
            field,
            fid,
            filtercode,
            imgtypecode,
        })
    }
}

impl FovLike for ZtfField {
    fn get_fov(&self, index: usize) -> FOV {
        FOV::ZtfCcdQuad(self.ccd_quads[index].clone())
    }

    fn observer(&self) -> &State<Equatorial> {
        &self.observer
    }

    fn contains(&self, obs_to_obj: &Vector<Equatorial>) -> (usize, Contains) {
        closest_inside(
            &self
                .ccd_quads
                .iter()
                .map(|x| x.contains(obs_to_obj).1)
                .collect::<Vec<_>>(),
        )
    }

    fn n_patches(&self) -> usize {
        self.ccd_quads.len()
    }

    #[inline]
    fn pointing(&self) -> KeteResult<Vector<Equatorial>> {
        if self.ccd_quads.is_empty() {
            Err(Error::ValueError("ZtfField has no ccd quads".into()))
        } else {
            // return the average pointing of all ccd quads
            Ok(self
                .ccd_quads
                .iter()
                .fold(Vector::new([0.0; 3]), |acc, x| acc + x.pointing().unwrap()))
        }
    }

    #[inline]
    fn corners(&self) -> KeteResult<Vec<Vector<Equatorial>>> {
        if self.ccd_quads.is_empty() {
            Err(Error::ValueError("ZtfField has no ccd quads".into()))
        } else {
            // return all the corners of all ccd quads
            Ok(self
                .ccd_quads
                .iter()
                .flat_map(|x| x.corners().unwrap())
                .collect())
        }
    }
}
