//! # Definitions of contiguous field of views
//! These field of views are made up of single contiguous patches of sky, typically single image sensors.
// BSD 3-Clause License
//
// Copyright (c) 2026, Dar Dahlen
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

use std::fmt::Debug;

use serde::{Deserialize, Serialize};

use super::{Contains, FovLike, OnSkyRectangle, SkyPatch, SphericalCone};
use crate::{
    errors::{Error, KeteResult},
    fov::FOV,
    frames::{Equatorial, Vector},
    state::State,
};

/// Generic rectangular FOV
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct GenericRectangle {
    observer: State<Equatorial>,

    /// Patch of sky
    patch: OnSkyRectangle,

    /// Rotation of the FOV.
    pub rotation: f64,
}

impl GenericRectangle {
    /// Create a new Generic Rectangular FOV
    #[must_use]
    pub fn new(
        pointing: Vector<Equatorial>,
        rotation: f64,
        lon_width: f64,
        lat_width: f64,
        observer: State<Equatorial>,
    ) -> Self {
        let patch = OnSkyRectangle::new(pointing, rotation, lon_width, lat_width);
        Self {
            observer,
            patch,
            rotation,
        }
    }

    /// Create a Field of view from a collection of corners.
    #[must_use]
    pub fn from_corners(
        corners: [Vector<Equatorial>; 4],
        observer: State<Equatorial>,
        expand_angle: f64,
    ) -> Self {
        let patch = OnSkyRectangle::from_corners(corners, expand_angle);
        Self {
            patch,
            observer,
            rotation: f64::NAN,
        }
    }

    /// Latitudinal width of the FOV.
    #[inline]
    #[must_use]
    pub fn lat_width(&self) -> f64 {
        self.patch.lat_width()
    }

    /// Longitudinal width of the FOV.
    #[inline]
    #[must_use]
    pub fn lon_width(&self) -> f64 {
        self.patch.lon_width()
    }
}

impl FovLike for GenericRectangle {
    type ChildFov = Self;

    #[inline]
    fn get_child(&self, index: usize) -> Self {
        assert!(index == 0, "FOV only has a single patch");
        self.clone()
    }

    #[inline]
    fn into_fov(self) -> FOV {
        FOV::GenericRectangle(self)
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

/// Generic rectangular FOV
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OmniDirectional {
    observer: State<Equatorial>,
}

impl OmniDirectional {
    /// Create a new Omni-Directional FOV
    #[must_use]
    pub fn new(observer: State<Equatorial>) -> Self {
        Self { observer }
    }
}

impl FovLike for OmniDirectional {
    type ChildFov = Self;

    #[inline]
    fn get_child(&self, index: usize) -> Self {
        assert!(index == 0, "FOV only has a single patch");
        self.clone()
    }

    #[inline]
    fn into_fov(self) -> FOV {
        FOV::OmniDirectional(self)
    }

    #[inline]
    fn observer(&self) -> &State<Equatorial> {
        &self.observer
    }

    #[inline]
    fn contains(&self, _obs_to_obj: &Vector<Equatorial>) -> (usize, Contains) {
        (0, Contains::Inside)
    }

    #[inline]
    fn n_patches(&self) -> usize {
        1
    }

    #[inline]
    fn pointing(&self) -> KeteResult<Vector<Equatorial>> {
        Err(Error::ValueError(
            "OmniDirectional FOV does not have a pointing vector.".into(),
        ))
    }

    #[inline]
    fn corners(&self) -> KeteResult<Vec<Vector<Equatorial>>> {
        Err(Error::ValueError(
            "OmniDirectional FOV does not have corners.".into(),
        ))
    }
}

/// Generic rectangular FOV
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct GenericCone {
    observer: State<Equatorial>,

    /// Patch of sky
    pub patch: SphericalCone,
}

impl GenericCone {
    /// Create a new Generic Conic FOV
    #[must_use]
    pub fn new(pointing: Vector<Equatorial>, angle: f64, observer: State<Equatorial>) -> Self {
        let patch = SphericalCone::new(&pointing, angle);
        Self { observer, patch }
    }

    /// Angle of the cone from the central pointing vector.
    #[inline]
    #[must_use]
    pub fn angle(&self) -> &f64 {
        &self.patch.angle
    }
}

impl FovLike for GenericCone {
    type ChildFov = Self;

    #[inline]
    fn get_child(&self, index: usize) -> Self {
        assert!(index == 0, "FOV only has a single patch");
        self.clone()
    }

    #[inline]
    fn into_fov(self) -> FOV {
        FOV::GenericCone(self)
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
        Err(Error::ValueError(
            "GenericCone does not have corners.".into(),
        ))
    }
}
