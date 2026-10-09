// SPDX-FileCopyrightText: 2026 Dar Dahlen
// SPDX-FileCopyrightText: 2025 California Institute of Technology
// SPDX-License-Identifier: BSD-3-Clause

//! # WISE Fov definitions.

use super::FovLike;
use crate::fov::FOV;
use crate::geometry::{Contains, SkyPatch, SphericalPolygon};
use crate::prelude::*;
use crate::{constants::WISE_WIDTH, frames::Vector};
/// WISE or NEOWISE frame data, all bands
#[derive(Debug, Clone)]
pub struct WiseCmos {
    /// State of the observer
    pub(crate) observer: State<Equatorial>,

    /// Patch of sky
    pub(crate) patch: SphericalPolygon,

    /// Frame number of the fov
    pub frame_num: u64,

    /// Scan ID of the fov
    pub scan_id: Box<str>,
}

impl WiseCmos {
    /// Create a Wise fov
    ///
    /// # Errors
    /// Returns [`Error::ValueError`](crate::errors::Error::ValueError) if
    /// `pointing` is not finite or points at a celestial pole, where the rotation
    /// is undefined. See [`SphericalPolygon::new`](crate::geometry::SphericalPolygon::new).
    pub fn new(
        pointing: Vector<Equatorial>,
        rotation: f64,
        observer: State<Equatorial>,
        frame_num: u64,
        scan_id: Box<str>,
    ) -> KeteResult<Self> {
        let patch = SphericalPolygon::new(pointing, rotation, WISE_WIDTH, WISE_WIDTH)?;
        Ok(Self {
            observer,
            patch,
            frame_num,
            scan_id,
        })
    }

    /// Create a Wise fov from corners
    #[must_use]
    pub fn from_corners(
        corners: [Vector<Equatorial>; 4],
        observer: State<Equatorial>,
        frame_num: u64,
        scan_id: Box<str>,
    ) -> Self {
        let patch = SphericalPolygon::from_corners(corners, 60_f64.recip().to_radians());
        Self {
            observer,
            patch,
            frame_num,
            scan_id,
        }
    }
}

impl FovLike for WiseCmos {
    type ChildFov = Self;

    #[inline]
    fn get_child(&self, index: usize) -> Self::ChildFov {
        assert!(index == 0, "Wise FOV only has a single patch");
        self.clone()
    }

    #[inline]
    fn into_fov(self) -> FOV {
        FOV::Wise(self)
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
