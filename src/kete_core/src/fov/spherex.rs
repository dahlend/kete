// SPDX-FileCopyrightText: 2026 Dar Dahlen
// SPDX-License-Identifier: BSD-3-Clause

//! # Spherex Fov definitions.

use super::FovLike;
use crate::fov::FOV;
use crate::frames::Vector;
use crate::geometry::closest_inside;
use crate::geometry::{Contains, SkyPatch, SphericalPolygon};
use crate::prelude::*;
/// Spherex frame data, both optical assemblies
#[derive(Debug, Clone)]
pub struct SpherexCmos {
    /// State of the observer
    pub(crate) observer: State<Equatorial>,

    /// Patch of sky
    pub(crate) patch: SphericalPolygon,

    /// uri indicating where the frame is stored in IRSA
    pub uri: Box<str>,

    /// The Plane ID identified from the spherex.plane table
    pub plane_id: Box<str>,
}

impl SpherexCmos {
    /// Create a Spherex fov from corners
    #[must_use]
    pub fn new(
        corners: [Vector<Equatorial>; 4],
        observer: State<Equatorial>,
        uri: Box<str>,
        plane_id: Box<str>,
    ) -> Self {
        let patch = SphericalPolygon::from_corners(corners, 0.0);
        Self {
            observer,
            patch,
            uri,
            plane_id,
        }
    }
}

impl FovLike for SpherexCmos {
    type ChildFov = Self;

    #[inline]
    fn get_child(&self, index: usize) -> Self::ChildFov {
        assert!(index == 0, "SPHEREx FOV only has a single patch");
        self.clone()
    }

    #[inline]
    fn into_fov(self) -> FOV {
        FOV::SpherexCmos(self)
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

/// Spherex frame data, multiple individual CMOS at one instant.
#[derive(Debug, Clone)]
pub struct SpherexField {
    /// Individual CMOS quads
    pub(crate) cmos_frames: Vec<SpherexCmos>,

    /// Observer position
    pub(crate) observer: State<Equatorial>,

    /// obsid UUID
    pub obsid: Box<str>,

    /// observationid, also called `obs_id` (not the same as obsid)
    pub observationid: Box<str>,
}

impl SpherexField {
    /// Construct a new [`SpherexField`] from a list of cmos frames.
    /// These cmos frames must be from the same field and having matching value as
    /// appropriate.
    ///
    /// # Errors
    /// ``cmos_frames`` must not be empty, and all frames must be consistent with one
    /// another.
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

        #[allow(clippy::missing_panics_doc, reason = "frame is not empty")]
        let first = cmos_frames.first().unwrap();

        let observer = first.observer().clone();

        for ccd in &cmos_frames {
            if !ccd.observer().epoch.same_instant(&observer.epoch) {
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
    type ChildFov = SpherexCmos;

    fn get_child(&self, index: usize) -> Self::ChildFov {
        self.cmos_frames[index].clone()
    }

    #[inline]
    fn into_fov(self) -> FOV {
        FOV::SpherexField(self)
    }

    fn observer(&self) -> &State<Equatorial> {
        &self.observer
    }

    fn contains(&self, obs_to_obj: &Vector<Equatorial>) -> (usize, Contains) {
        closest_inside(self.cmos_frames.iter().map(|x| x.contains(obs_to_obj).1))
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
