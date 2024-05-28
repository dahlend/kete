//! # WISE Fov definitions.
use super::{Contains, FovLike, OnSkyRectangle, SkyPatch, FOV, Frame, NEOSpyError};
use crate::constants::WISE_WIDTH;
use crate::prelude::*;
use nalgebra::Vector3;
use serde::{Deserialize, Serialize};

/// WISE or NEOWISE frame data, all bands
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct WiseCmos {
    /// State of the observer
    observer: State,

    /// Patch of sky
    pub patch: OnSkyRectangle,

    /// Rotation of the FOV.
    pub rotation: f64,

    /// Frame number of the fov
    pub frame_num: usize,

    /// Scan ID of the fov
    pub scan_id: Box<str>,
}

impl WiseCmos {
    /// Create a Wise fov
    pub fn new(
        pointing: Vector3<f64>,
        rotation: f64,
        observer: State,
        frame_num: usize,
        scan_id: Box<str>,
    ) -> Self {
        let patch = OnSkyRectangle::new(pointing, rotation, WISE_WIDTH, WISE_WIDTH, observer.frame);
        Self {
            patch,
            observer,
            frame_num,
            rotation,
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
    fn observer(&self) -> &State {
        &self.observer
    }

    #[inline]
    fn contains(&self, obs_to_obj: &Vector3<f64>) -> (usize, Contains) {
        (0, self.patch.contains(obs_to_obj))
    }

    #[inline]
    fn n_patches(&self) -> usize {
        1
    }

    fn try_frame_change_mut(
        &mut self,
        target_frame: Frame,
    ) -> Result<(), NEOSpyError> {
        self.observer.try_change_frame_mut(target_frame)?;
        self.patch = self.patch.try_frame_change(target_frame)?;
        Ok(())
    }
}
