//! # Field of View like trait
//! This trait defines field of view checks for portions of the sky.
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

use super::{Contains, FOV};

use crate::constants::C_AU_PER_DAY_INV;
use crate::frames::{Equatorial, Vector};
use crate::prelude::*;

/// Field of View like objects.
/// These may contain multiple unique sky patches, so as a result the expected
/// behavior is to return the index as well as the [`Contains`] for the closest
/// sky patch.
pub trait FovLike: Sync + Sized {
    /// Return the FOV of the patch at the specified index.
    /// This will panic if the index is out of allowed bounds.
    fn get_fov(&self, index: usize) -> FOV;

    /// Position of the observer.
    fn observer(&self) -> &State<Equatorial>;

    /// Is the specified vector contained within this [`FovLike`].
    /// A [`Contains`] is returned for each sky patch.
    fn contains(&self, obs_to_obj: &Vector<Equatorial>) -> (usize, Contains);

    /// Number of sky patches contained within this FOV.
    fn n_patches(&self) -> usize;

    /// Get the pointing vector of the FOV.
    ///
    /// # Errors
    /// Some ``FoVs`` may not have a well formed pointing vector.
    fn pointing(&self) -> KeteResult<Vector<Equatorial>>;

    /// Get the corners of the FOV.
    ///
    /// # Errors
    /// Not all ``FoVs`` contain corners, such as a Cone.
    fn corners(&self) -> KeteResult<Vec<Vector<Equatorial>>>;

    /// Check if a static source is visible. This assumes the vector passed in is at an
    /// infinite distance from the observer.
    #[inline]
    fn check_static(&self, pos: &Vector<Equatorial>) -> (usize, Contains) {
        self.contains(pos)
    }

    /// Assuming the object undergoes linear motion, check to see if it is within the
    /// field of view.
    #[inline]
    fn check_linear(&self, state: &State<Equatorial>) -> (usize, Contains, State<Equatorial>) {
        let pos = state.pos;
        let vel = state.vel;
        let obs = self.observer();

        let obs_pos = obs.pos;

        let rel_pos = pos - obs_pos;

        // This also accounts for first order light delay.
        let dt = obs.epoch.jd - state.epoch.jd - rel_pos.norm() * C_AU_PER_DAY_INV;
        let new_pos = pos + vel * dt;
        let new_rel_pos = new_pos - obs_pos;
        let (idx, contains) = self.contains(&new_rel_pos);
        let new_state = State::new(
            state.desig.clone(),
            obs.epoch + dt,
            new_pos,
            vel,
            obs.center_id,
        );
        (idx, contains, new_state)
    }

    /// Given a collection of static positions, return the index of the input vector
    /// which was visible.
    fn check_statics(&self, pos: &[Vector<Equatorial>]) -> Vec<Option<(Vec<usize>, FOV)>> {
        let mut visible: Vec<Vec<usize>> = vec![Vec::new(); self.n_patches()];

        pos.iter().enumerate().for_each(|(vec_idx, p)| {
            if let (patch_idx, Contains::Inside) = self.check_static(p) {
                visible[patch_idx].push(vec_idx);
            }
        });

        visible
            .into_iter()
            .enumerate()
            .map(|(idx, vis_patch)| {
                if vis_patch.is_empty() {
                    None
                } else {
                    Some((vis_patch, self.get_fov(idx)))
                }
            })
            .collect()
    }
}
