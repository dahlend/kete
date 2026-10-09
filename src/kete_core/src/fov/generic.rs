// SPDX-FileCopyrightText: 2026 Dar Dahlen
// SPDX-FileCopyrightText: 2025 California Institute of Technology
// SPDX-License-Identifier: BSD-3-Clause

//! # Definitions of contiguous field of views
//! These field of views are made up of single contiguous patches of sky, typically single image sensors.

use std::fmt::Debug;

use super::FovLike;
use crate::geometry::{Contains, SkyPatch, SphericalCone, SphericalPolygon};
use crate::{
    errors::{Error, KeteResult},
    fov::FOV,
    frames::{Equatorial, Vector},
    state::State,
};

/// Generic rectangular FOV
#[derive(Debug, Clone)]
pub struct GenericRectangle {
    pub(crate) observer: State<Equatorial>,

    /// Patch of sky
    pub(crate) patch: SphericalPolygon,

    /// Rotation of the FOV.
    pub rotation: f64,
}

impl GenericRectangle {
    /// Create a new Generic Rectangular FOV
    ///
    /// # Errors
    /// Returns [`Error::ValueError`](crate::errors::Error::ValueError) if
    /// `pointing` is not finite or points at a celestial pole, where the rotation
    /// is undefined. See [`SphericalPolygon::new`](crate::geometry::SphericalPolygon::new).
    pub fn new(
        pointing: Vector<Equatorial>,
        rotation: f64,
        lon_width: f64,
        lat_width: f64,
        observer: State<Equatorial>,
    ) -> KeteResult<Self> {
        let patch = SphericalPolygon::new(pointing, rotation, lon_width, lat_width)?;
        Ok(Self {
            observer,
            patch,
            rotation,
        })
    }

    /// Create a Field of view from a collection of corners.
    #[must_use]
    pub fn from_corners(
        corners: [Vector<Equatorial>; 4],
        observer: State<Equatorial>,
        expand_angle: f64,
    ) -> Self {
        let patch = SphericalPolygon::from_corners(corners, expand_angle);
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
        Ok(self.patch.corners())
    }
}

/// Generic rectangular FOV
#[derive(Debug, Clone)]
pub struct OmniDirectional {
    pub(crate) observer: State<Equatorial>,
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

/// Generic polygon FOV, convex or not.
///
/// See [`SphericalPolygon::try_from_corners`] for the conditions on the
/// corners.
#[derive(Debug, Clone)]
pub struct GenericPolygon {
    pub(crate) observer: State<Equatorial>,

    /// Patch of sky.
    pub patch: SphericalPolygon,
}

impl GenericPolygon {
    /// The polygon FOV with `corners`, given in order around it, seen by
    /// `observer`.
    ///
    /// # Errors
    /// [`Error::ValueError`] if the corners do not form a valid polygon.
    pub fn new(corners: &[Vector<Equatorial>], observer: State<Equatorial>) -> KeteResult<Self> {
        Ok(Self {
            observer,
            patch: SphericalPolygon::try_from_corners(corners)?,
        })
    }
}

impl FovLike for GenericPolygon {
    type ChildFov = Self;

    #[inline]
    fn get_child(&self, index: usize) -> Self {
        assert!(index == 0, "FOV only has a single patch");
        self.clone()
    }

    #[inline]
    fn into_fov(self) -> FOV {
        FOV::GenericPolygon(self)
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

/// Generic conic FOV
#[derive(Debug, Clone)]
pub struct GenericCone {
    pub(crate) observer: State<Equatorial>,

    /// Patch of sky
    pub patch: SphericalCone,
}

impl GenericCone {
    /// Create a new Generic Conic FOV, `angle` is in radians.
    #[must_use]
    pub fn new(pointing: Vector<Equatorial>, angle: f64, observer: State<Equatorial>) -> Self {
        let patch = SphericalCone::new(&pointing, angle);
        Self { observer, patch }
    }

    /// Angle of the cone from the central pointing vector in radians.
    #[inline]
    #[must_use]
    pub fn angle(&self) -> f64 {
        self.patch.angle()
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
