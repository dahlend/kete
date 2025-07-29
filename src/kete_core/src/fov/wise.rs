//! # WISE Fov definitions.
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

use super::{Contains, FOV, FovLike, OnSkyRectangle, SkyPatch};
use crate::prelude::*;
use crate::{constants::WISE_WIDTH, frames::Vector};
use serde::{Deserialize, Serialize};

/// WISE or NEOWISE frame data, all bands
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct WiseCmos {
    /// State of the observer
    observer: State<Equatorial>,

    /// Patch of sky
    patch: OnSkyRectangle,

    /// Frame number of the fov
    pub frame_num: u64,

    /// Scan ID of the fov
    pub scan_id: Box<str>,
}

impl WiseCmos {
    /// Create a Wise fov
    pub fn new(
        pointing: Vector<Equatorial>,
        rotation: f64,
        observer: State<Equatorial>,
        frame_num: u64,
        scan_id: Box<str>,
    ) -> Self {
        let patch = OnSkyRectangle::new(pointing, rotation, WISE_WIDTH, WISE_WIDTH);
        Self {
            patch,
            observer,
            frame_num,
            scan_id,
        }
    }

    /// Create a Wise fov from corners
    pub fn from_corners(
        corners: [Vector<Equatorial>; 4],
        observer: State<Equatorial>,
        frame_num: u64,
        scan_id: Box<str>,
    ) -> Self {
        let patch = OnSkyRectangle::from_corners(corners, 60_f64.recip().to_radians());
        Self {
            patch,
            observer,
            frame_num,
            scan_id,
        }
    }
}

impl FovLike for WiseCmos {
    #[inline]
    fn get_fov(&self, index: usize) -> FOV {
        if index != 0 {
            panic!("Wise FOV only has a single patch")
        }
        FOV::Wise(self.clone())
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
