//! State vector representations.
//!
//! Keeping track of the location and velocity of an object requires more information
//! than just a position and velocity vector. Because there is no universal coordinate
//! system, positions have to be provided with respect to a reference frame.
//! There are two pieces to this, the basis of the reference frame, and the origin.
//!
//! Bringing this all together, the minimum information to know the state of an object
//! is:
//! - Frame of reference
//! - Origin
//! - Position
//! - Velocity
//! - Time
//! - ID - Some unique identifier for the object so that other objects may reference it.
//!
//! Below is the [`State`] which defines this minimum information.
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

use serde::{Deserialize, Serialize};
use std::fmt::Debug;

use crate::desigs::Desig;
use crate::errors::{Error, KeteResult};
use crate::frames::{InertialFrame, Vector};

/// Exact State of an object.
///
/// This represents the id, position, and velocity of an object with respect to a
/// coordinate frame and a center point.
///
/// This state object assumes no uncertainty in its values.
#[derive(Debug, Serialize, Clone, Deserialize, PartialEq)]
pub struct State<T>
where
    T: InertialFrame,
{
    /// Designation number which corresponds to the object.
    pub desig: Desig,

    /// JD of the object's state in TDB scaled time.
    pub jd: f64,

    /// Position of the object with respect to the `center_id` object, units of AU.
    pub pos: Vector<T>,

    /// Velocity of the object with respect to the `center_id` object, units of AU/Day.
    pub vel: Vector<T>,

    /// Position and velocity are given with respect to the specified `center_id`.
    /// The only privileged center ID is the Solar System Barycenter 0.
    pub center_id: i32,
}

impl<T: InertialFrame> State<T> {
    /// Construct a new State object.
    #[inline(always)]
    pub fn new(desig: Desig, jd: f64, pos: Vector<T>, vel: Vector<T>, center_id: i32) -> Self {
        Self {
            desig,
            jd,
            pos,
            vel,
            center_id,
        }
    }

    /// Construct a new state made of NAN pos and vel vectors but containing the
    /// remaining data. This is primarily useful as a place holder when propagation
    /// has failed and the object needs to be recorded still.
    #[inline(always)]
    pub fn new_nan(desig: Desig, jd: f64, center_id: i32) -> Self {
        Self::new(desig, jd, Vector::new_nan(), Vector::new_nan(), center_id)
    }

    /// Are all values finite.
    pub fn is_finite(&self) -> bool {
        self.pos.is_finite() & self.vel.is_finite() & self.jd.is_finite()
    }

    /// Trade the center ID and ID values, and flip the direction of the position and
    /// velocity vectors.
    #[inline(always)]
    pub fn try_flip_center_id(&mut self) -> KeteResult<()> {
        if let Desig::Naif(mut id) = self.desig {
            std::mem::swap(&mut id, &mut self.center_id);
            self.pos = -self.pos;
            self.vel = -self.vel;
            self.desig = Desig::Naif(id);
            return Ok(());
        }
        Err(Error::ValueError(
            "Flip center ID is only valid for NAIF ids.".into(),
        ))
    }

    /// Mutate the current state and change its center to the center defined in the
    /// provided state.
    /// For example if the current states center id is 2 (Venus), and a state is
    /// provided which represents 2 (Venus) with its center defined as 10 (Sun), then
    /// this changes the current states center to be 10 (the Sun).
    ///
    /// This will flip the center id and ID of the provided state if necessary.
    ///
    /// # Arguments
    ///
    /// * `state` - [`State`] object which defines the new center point.
    #[inline(always)]
    pub fn try_change_center(&mut self, mut state: Self) -> KeteResult<()> {
        if self.jd != state.jd {
            return Err(Error::ValueError("States don't have matching jds.".into()));
        }

        let Desig::Naif(state_id) = state.desig else {
            return Err(Error::ValueError(
                "Changing centers only works on states with NAIF Ids.".into(),
            ));
        };

        // target state does not match at all, error
        if self.center_id != state.center_id && self.center_id != state_id {
            return Err(Error::ValueError(
                "States do not reference one another at all, cannot change center.".into(),
            ));
        }

        // Flip center ID if necessary for the state
        if self.center_id == state.center_id {
            state.try_flip_center_id()?;
        }

        // Now the state is where it is supposed to be, update as required.
        self.center_id = state.center_id;
        self.pos += &state.pos;
        self.vel += &state.vel;
        Ok(())
    }

    /// Attempt to update the designation from a naif id to a name.
    pub fn try_naif_id_to_name(&mut self) {
        self.desig = self.desig.clone().try_naif_id_to_name();
    }

    /// Convert the state into a new frame.
    #[inline(always)]
    pub fn into_frame<B: InertialFrame>(self) -> State<B> {
        let pos = self.pos.into_frame::<B>();
        let vel = self.vel.into_frame::<B>();

        State {
            desig: self.desig,
            jd: self.jd,
            pos,
            vel,
            center_id: self.center_id,
        }
    }
}

#[cfg(test)]
mod tests {

    use crate::frames::{Ecliptic, Equatorial};

    use super::*;

    #[test]
    fn flip_center() {
        let mut a = State::<Equatorial>::new(
            Desig::Naif(1),
            0.0,
            [1.0, 0.0, 0.0].into(),
            [0.0, 1.0, 0.0].into(),
            0,
        );
        a.try_flip_center_id().unwrap();

        let pos: [f64; 3] = a.pos.into();
        let vel: [f64; 3] = a.vel.into();
        assert!(a.center_id == 1);
        assert!(pos == [-1.0, 0.0, 0.0]);
        assert!(vel == [0.0, -1.0, 0.0]);
    }

    #[test]
    fn nan_finite() {
        let a = State::<Equatorial>::new(
            Desig::Naif(1),
            0.0,
            [1.0, 0.0, 0.0].into(),
            [0.0, 1.0, 0.0].into(),
            0,
        );
        assert!(a.is_finite());

        let b = State::<Equatorial>::new_nan(Desig::Empty, 0.0, 1000);
        assert!(!b.is_finite());
    }

    #[test]
    fn naif_name_resolution() {
        let mut a = State::<Ecliptic>::new(
            Desig::Naif(1),
            0.0,
            [1.0, 0.0, 0.0].into(),
            [0.0, 1.0, 0.0].into(),
            0,
        );
        a.try_naif_id_to_name();
        assert!(a.desig == Desig::Name("mercury barycenter".into()));
        assert!(a.desig.full_string() == "Name(\"mercury barycenter\")");
        assert!(a.desig.to_string() == "mercury barycenter");
    }

    #[test]
    fn change_center() {
        let mut a = State::<Ecliptic>::new(
            Desig::Naif(1),
            0.0,
            [1.0, 0.0, 0.0].into(),
            [1.0, 0.0, 0.0].into(),
            0,
        );
        let b = State::<Equatorial>::new(
            Desig::Naif(3),
            0.0,
            [0.0, 1.0, 0.0].into(),
            [0.0, 1.0, 0.0].into(),
            0,
        );
        a.try_change_center(b.into_frame()).unwrap();

        assert!(a.center_id == 3);
        assert!(a.pos[0] == 1.0);
        assert!(a.pos[1] != 0.0);
        assert!(a.pos[2] != 0.0);
        assert!(a.vel[0] == 1.0);

        // try cases which cause errors
        let diff_jd = State::<Equatorial>::new(
            Desig::Naif(3),
            1.0,
            [0.0, 1.0, 0.0].into(),
            [0.0, 1.0, 0.0].into(),
            0,
        );
        assert!(a.try_change_center(diff_jd.into_frame()).is_err());

        let not_naif_id = State::<Equatorial>::new(
            Desig::Empty,
            0.0,
            [0.0, 1.0, 0.0].into(),
            [0.0, 1.0, 0.0].into(),
            0,
        );
        assert!(a.try_change_center(not_naif_id.into_frame()).is_err());

        let no_matching_id = State::<Equatorial>::new(
            Desig::Naif(2),
            0.0,
            [0.0, 1.0, 0.0].into(),
            [0.0, 1.0, 0.0].into(),
            1000000000,
        );
        assert!(a.try_change_center(no_matching_id.into_frame()).is_err());
    }
}
