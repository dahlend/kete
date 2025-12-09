//! Collections of [`State`] objects are grouped together into a [`SimultaneousStates`].
//! These primarily exist to allow for easy read/write to binary formats.
//
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

use crate::fov::FOV;
use crate::frames::{Equatorial, Vector};
use crate::io::FileIO;
use crate::prelude::{Error, KeteResult, State};
use crate::time::{TDB, Time};
use nalgebra::Vector3;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use serde::{Deserialize, Serialize};

/// Collection of [`State`] at the same time.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct SimultaneousStates {
    /// Collection of states
    pub states: Vec<State<Equatorial>>,

    /// Common JD time of all states
    pub epoch: Time<TDB>,

    /// Center ID of all states.
    pub center_id: i32,

    /// An optional field of view.
    pub fov: Option<FOV>,
}

impl FileIO for SimultaneousStates {}

impl SimultaneousStates {
    /// Create a new Exact `SimultaneousStates`
    /// Simultaneous States occur at the same JD, which is defined by either the time
    /// in the optional fov, or the time of the first state.
    ///
    /// # Errors
    ///
    /// [`Error::ValueError`] possible for multiple reasons:
    /// - Input vector must contain at least one state.
    /// - Center ids of all states must match.
    /// - Epoch times must match unless an FOV is provided.
    pub fn new_exact(states: Vec<State<Equatorial>>, fov: Option<FOV>) -> KeteResult<Self> {
        let Some(state) = states.first() else {
            return Err(Error::ValueError(
                "SimultaneousStates must contain at least one state.".into(),
            ));
        };
        let (mut jd, center_id) = (state.epoch, state.center_id);

        if let Some(f) = &fov {
            jd = f.observer().epoch;
        }

        if states.iter().any(|state| state.center_id != center_id) {
            return Err(Error::ValueError("Center IDs do not match expected".into()));
        }

        if fov.is_none() && states.iter().any(|state| state.epoch != jd) {
            return Err(Error::ValueError(
                "Epoch JDs do not match expected, this is only allowed if there is an associated FOV."
                    .into(),
            ));
        }

        Ok(Self {
            states,
            epoch: jd,
            center_id,
            fov,
        })
    }

    /// Compute RA/Dec along with linearized rates.
    ///
    /// Returns a vector containing [ra, dec, ra' * cos(dec), dec'], all vectors
    /// are automatically cast into the equatorial frame.
    /// The returned RA rate is scaled by cos(dec) so that it is equivalent to a
    /// linear projection onto the observing plane.
    ///
    /// # Errors
    ///
    /// [`Error::ValueError`] possible for multiple reasons:
    /// - FOV is not specified.
    /// - Center id of the FOV state does not match the center ID of the states.
    pub fn ra_dec_with_rates(&self) -> KeteResult<Vec<[f64; 4]>> {
        let Some(fov) = &self.fov else {
            return Err(Error::ValueError(
                "Field of view must be specified for the ra/dec to be computed.".into(),
            ));
        };

        let obs = fov.observer();

        if obs.center_id != self.center_id {
            return Err(Error::ValueError(
                "Field of view center ID does not match the states center ID.".into(),
            ));
        }

        let obs_pos = obs.pos;
        let obs_vel = obs.vel;

        Ok(self
            .states
            .par_iter()
            .with_min_len(100)
            .map(|state| {
                let state = state.clone();
                let pos_diff: Vector3<f64> = (state.pos - obs_pos).into();
                let vel_diff: Vector3<f64> = (state.vel - obs_vel).into();

                let d_ra = (pos_diff.x * vel_diff.y - pos_diff.y * vel_diff.x)
                    / (pos_diff.x.powi(2) + pos_diff.y.powi(2));
                let r2 = pos_diff.norm_squared();

                let d_dec = (vel_diff.z - pos_diff.z * pos_diff.dot(&vel_diff) / r2)
                    / (r2 - pos_diff.z.powi(2)).sqrt();

                let vec: Vector<Equatorial> = pos_diff.into();
                let (ra, dec) = vec.to_ra_dec();

                [ra, dec, d_ra * dec.cos(), d_dec]
            })
            .collect())
    }
}
