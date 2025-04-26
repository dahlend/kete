//! # WISE Fov definitions.
use core::f64;

use super::{Contains, FovLike, OnSkyRectangle, SkyPatch, FOV};
use crate::prelude::*;
use crate::{constants::WISE_WIDTH, frames::Vector};
use serde::{Deserialize, Serialize};

/// WISE or NEOWISE frame data, all bands
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct WiseCmos {
    /// State of the observer
    observer: State<Equatorial>,

    /// Patch of sky
    pub patch: OnSkyRectangle,

    /// Rotation of the FOV.
    pub rotation: f64,

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
            rotation,
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
        let patch = OnSkyRectangle::from_corners(corners);
        Self {
            patch,
            observer,
            frame_num,
            rotation: f64::NAN,
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
}
