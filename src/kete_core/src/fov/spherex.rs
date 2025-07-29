//! # Spherex Fov definitions.
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

use super::{Contains, FOV, FovLike, OnSkyRectangle, SkyPatch};
use crate::fov::patches::closest_inside;
use crate::frames::Vector;
use crate::prelude::*;
use serde::{Deserialize, Serialize};

/// Spherex frame data, both optical assemblies
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SpherexCmos {
    /// State of the observer
    observer: State<Equatorial>,

    /// Patch of sky
    patch: OnSkyRectangle,

    /// uri indicating where the frame is stored in IRSA
    pub uri: Box<str>,

    /// The Plane ID identified from the spherex.plane table
    pub plane_id: Box<str>,
}

impl SpherexCmos {
    /// Create a Spherex fov from corners
    pub fn new(
        corners: [Vector<Equatorial>; 4],
        observer: State<Equatorial>,
        uri: Box<str>,
        plane_id: Box<str>,
    ) -> Self {
        let patch = OnSkyRectangle::from_corners(corners, 0.0);
        Self {
            patch,
            observer,
            uri,
            plane_id,
        }
    }
}

impl FovLike for SpherexCmos {
    #[inline]
    fn get_fov(&self, index: usize) -> FOV {
        if index != 0 {
            panic!("SPHEREx FOV only has a single patch")
        }
        FOV::SpherexCmos(self.clone())
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

/// Spherex frame data, multiple individual CMOS at one instant.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SpherexField {
    /// Individual CMOS quads
    cmos_frames: Vec<SpherexCmos>,

    /// Observer position
    observer: State<Equatorial>,

    /// obsid UUID
    pub obsid: Box<str>,

    /// observationid, also called `obs_id` (not the same as obsid)
    pub observationid: Box<str>,
}

impl SpherexField {
    /// Construct a new [`SpherexField`] from a list of cmos frames.
    /// These cmos frames must be from the same field and having matching value as
    /// appropriate.
    pub fn new(
        cmos_frames: Vec<SpherexCmos>,
        obsid: Box<str>,
        observationid: Box<str>,
    ) -> KeteResult<Self> {
        if cmos_frames.is_empty() {
            Err(Error::ValueError(
                "Spherex Field must contain at least 1 SpherexCMOS".into(),
            ))?;
        }

        let first = cmos_frames.first().unwrap();

        let observer = first.observer().clone();

        for ccd in cmos_frames.iter() {
            if ccd.observer().jd != observer.jd {
                Err(Error::ValueError(
                    "All SpherexCMOS must have matching values times".into(),
                ))?;
            }
        }
        Ok(Self {
            cmos_frames,
            observer,
            obsid,
            observationid,
        })
    }
}

impl FovLike for SpherexField {
    fn get_fov(&self, index: usize) -> FOV {
        FOV::SpherexCmos(self.cmos_frames[index].clone())
    }

    fn observer(&self) -> &State<Equatorial> {
        &self.observer
    }

    fn contains(&self, obs_to_obj: &Vector<Equatorial>) -> (usize, Contains) {
        closest_inside(
            &self
                .cmos_frames
                .iter()
                .map(|x| x.contains(obs_to_obj).1)
                .collect::<Vec<_>>(),
        )
    }

    fn n_patches(&self) -> usize {
        self.cmos_frames.len()
    }

    #[inline]
    fn pointing(&self) -> KeteResult<Vector<Equatorial>> {
        if self.cmos_frames.is_empty() {
            Err(Error::ValueError("SphereField has no cmos frames".into()))
        } else {
            // return the average pointing of all cmos frames
            Ok(self
                .cmos_frames
                .iter()
                .fold(Vector::new([0.0; 3]), |acc, x| acc + x.pointing().unwrap()))
        }
    }

    #[inline]
    fn corners(&self) -> KeteResult<Vec<Vector<Equatorial>>> {
        if self.cmos_frames.is_empty() {
            Err(Error::ValueError("SphereField has no cmos frames".into()))
        } else {
            // return all the corners of all cmos frames
            Ok(self
                .cmos_frames
                .iter()
                .flat_map(|x| x.corners().unwrap())
                .collect())
        }
    }
}
