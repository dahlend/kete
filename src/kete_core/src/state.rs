//! # State vector representations.
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

use crate::desigs::Desig;
use crate::errors::{Error, KeteResult};
use crate::frames::{CenterBody, DynCenter, EarthCenter, InertialFrame, SSB, SunCenter, Vector};
use crate::time::{TDB, Time};

/// Exact State of an object.
///
/// This represents the id, position, and velocity of an object with respect to a
/// coordinate frame and a center point.
///
/// The second type parameter `C` encodes the gravitational center at compile time.
/// The default `DynCenter(i32)` carries the center NAIF id at runtime.  Use `SSB`,
/// `SunCenter`, or `EarthCenter` for states whose center is known at compile time.
///
/// This state object assumes no uncertainty in its values.
#[derive(Debug, Clone, PartialEq)]
#[must_use]
pub struct State<T, C = DynCenter>
where
    T: InertialFrame,
    C: CenterBody,
    DynCenter: From<C>,
{
    /// Designation number which corresponds to the object.
    pub desig: Desig,

    /// Epoch of the object's state in TDB scaled time.
    pub epoch: Time<TDB>,

    /// Position of the object with respect to the center body, units of AU.
    pub pos: Vector<T>,

    /// Velocity of the object with respect to the center body, units of AU/Day.
    pub vel: Vector<T>,

    /// Center body encoding the NAIF id of the reference origin.
    pub center: C,
}

impl<T: InertialFrame, C: CenterBody> State<T, C>
where
    DynCenter: From<C>,
{
    /// NAIF id of the center body.
    #[inline(always)]
    pub fn center_id(&self) -> i32 {
        self.center.center_id()
    }

    /// Are all values finite.
    #[must_use]
    pub fn is_finite(&self) -> bool {
        self.pos.is_finite() & self.vel.is_finite() & self.epoch.jd.is_finite()
    }

    /// Convert the state into a new frame, preserving the center type.
    #[inline(always)]
    pub fn into_frame<B: InertialFrame>(self) -> State<B, C> {
        let pos = self.pos.into_frame::<B>();
        let vel = self.vel.into_frame::<B>();
        State {
            desig: self.desig,
            epoch: self.epoch,
            pos,
            vel,
            center: self.center,
        }
    }
}

impl<T: InertialFrame, C: CenterBody> State<T, C>
where
    DynCenter: From<C>,
{
    /// Construct a new State object.
    ///
    /// # Arguments
    /// * `desig` - Designation number which corresponds to the object.
    /// * `epoch` - Epoch of the object's state in TDB scaled time.
    /// * `pos` - Position of the object with respect to the center body, units of AU.
    /// * `vel` - Velocity of the object with respect to the center body, units of AU/Day.
    /// * `center` - Central body, must implement [`CenterBody`].
    #[inline(always)]
    pub fn new(
        desig: impl Into<Desig>,
        epoch: impl Into<Time<TDB>>,
        pos: impl Into<Vector<T>>,
        vel: impl Into<Vector<T>>,
        center: impl Into<C>,
    ) -> Self {
        Self {
            desig: desig.into(),
            epoch: epoch.into(),
            pos: pos.into(),
            vel: vel.into(),
            center: center.into(),
        }
    }

    /// Construct a new state made of NAN pos and vel vectors.
    ///
    /// This is primarily useful as a place holder when propagation has failed
    /// and the object needs to be recorded still.
    #[inline(always)]
    pub fn new_nan(
        desig: impl Into<Desig>,
        epoch: impl Into<Time<TDB>>,
        center: impl Into<C>,
    ) -> Self {
        Self::new(
            desig,
            epoch,
            Vector::new_nan(),
            Vector::new_nan(),
            center.into(),
        )
    }
}

impl<T: InertialFrame> State<T, DynCenter> {
    /// Trade the center ID and ID values, and flip the direction of the position and
    /// velocity vectors.
    #[inline(always)]
    pub(crate) fn try_flip_center_id(&mut self) -> KeteResult<()> {
        if let Desig::Naif(mut id) = self.desig {
            std::mem::swap(&mut id, &mut self.center.0);
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
    ///
    /// For example if the current states center id is 2 (Venus), and a state is
    /// provided which represents 2 (Venus) with its center defined as 10 (Sun), then
    /// this changes the current states center to be 10 (the Sun).
    ///
    /// This will flip the center id and ID of the provided state if necessary.
    ///
    /// # Arguments
    ///
    /// * `state` - [`State`] object which defines the new center point.
    ///
    /// # Errors
    ///
    /// [`Error::ValueError`] possible for multiple reasons:
    /// - Other state is not at the same instant in time (epoch).
    /// - [`Desig`] is not a Naif ID.
    /// - Center id of the state does not match the ID in the other state.
    #[inline(always)]
    pub fn try_change_center(&mut self, mut state: Self) -> KeteResult<()> {
        if self.epoch != state.epoch {
            return Err(Error::ValueError(
                "States don't have matching epochs.".into(),
            ));
        }

        let Desig::Naif(state_id) = state.desig else {
            return Err(Error::ValueError(
                "Changing centers only works on states with NAIF Ids.".into(),
            ));
        };

        // target state does not match at all, error
        if self.center.0 != state.center.0 && self.center.0 != state_id {
            return Err(Error::ValueError(
                "States do not reference one another at all, cannot change center.".into(),
            ));
        }

        // Flip center ID if necessary for the state
        if self.center.0 == state.center.0 {
            state.try_flip_center_id()?;
        }

        // Now the state is where it is supposed to be, update as required.
        self.center = DynCenter(state.center.0);
        self.pos += &state.pos;
        self.vel += &state.vel;
        Ok(())
    }
}

impl<T: InertialFrame> TryFrom<State<T, DynCenter>> for State<T, SSB> {
    type Error = Error;

    fn try_from(s: State<T, DynCenter>) -> KeteResult<Self> {
        if s.center.0 != 0 {
            return Err(Error::ValueError(format!(
                "Expected SSB-centered state (center_id=0), got center_id={}",
                s.center.0
            )));
        }
        Ok(Self {
            desig: s.desig,
            epoch: s.epoch,
            pos: s.pos,
            vel: s.vel,
            center: SSB,
        })
    }
}

impl<T: InertialFrame> TryFrom<State<T, DynCenter>> for State<T, SunCenter> {
    type Error = Error;

    fn try_from(s: State<T, DynCenter>) -> KeteResult<Self> {
        if s.center.0 != 10 {
            return Err(Error::ValueError(format!(
                "Expected Sun-centered state (center_id=10), got center_id={}",
                s.center.0
            )));
        }
        Ok(Self {
            desig: s.desig,
            epoch: s.epoch,
            pos: s.pos,
            vel: s.vel,
            center: SunCenter,
        })
    }
}

impl<T: InertialFrame> TryFrom<State<T, DynCenter>> for State<T, EarthCenter> {
    type Error = Error;

    fn try_from(s: State<T, DynCenter>) -> KeteResult<Self> {
        if s.center.0 != 399 {
            return Err(Error::ValueError(format!(
                "Expected Earth-centered state (center_id=399), got center_id={}",
                s.center.0
            )));
        }
        Ok(Self {
            desig: s.desig,
            epoch: s.epoch,
            pos: s.pos,
            vel: s.vel,
            center: EarthCenter,
        })
    }
}

impl<T: InertialFrame> From<State<T, SSB>> for State<T, DynCenter> {
    fn from(s: State<T, SSB>) -> Self {
        Self {
            desig: s.desig,
            epoch: s.epoch,
            pos: s.pos,
            vel: s.vel,
            center: DynCenter(0),
        }
    }
}

impl<T: InertialFrame> From<State<T, SunCenter>> for State<T, DynCenter> {
    fn from(s: State<T, SunCenter>) -> Self {
        Self {
            desig: s.desig,
            epoch: s.epoch,
            pos: s.pos,
            vel: s.vel,
            center: DynCenter(10),
        }
    }
}

impl<T: InertialFrame> From<State<T, EarthCenter>> for State<T, DynCenter> {
    fn from(s: State<T, EarthCenter>) -> Self {
        Self {
            desig: s.desig,
            epoch: s.epoch,
            pos: s.pos,
            vel: s.vel,
            center: DynCenter(399),
        }
    }
}

#[cfg(test)]
mod tests {

    use crate::frames::{Ecliptic, Equatorial};

    use super::*;

    #[test]
    fn flip_center() {
        let mut a = State::<Equatorial>::new(1, 0.0, [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], 0);
        a.try_flip_center_id().unwrap();

        let pos: [f64; 3] = a.pos.into();
        let vel: [f64; 3] = a.vel.into();
        assert!(a.center_id() == 1);
        assert!(pos == [-1.0, 0.0, 0.0]);
        assert!(vel == [0.0, -1.0, 0.0]);
    }

    #[test]
    fn nan_finite() {
        let a = State::<Equatorial>::new(None, 0.0, [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], 0);
        assert!(a.is_finite());

        let b = State::<Equatorial>::new_nan(Desig::Empty, 0.0, 1000);
        assert!(!b.is_finite());
    }

    #[test]
    fn ssb_round_trip() {
        let dyn_state = State::<Equatorial>::new(None, 0.0, [1.0, 2.0, 3.0], [0.1, 0.2, 0.3], 0);
        let ssb: State<Equatorial, SSB> = dyn_state.clone().try_into().unwrap();
        let back: State<Equatorial, DynCenter> = ssb.into();
        assert_eq!(dyn_state, back);
    }

    #[test]
    fn ssb_try_from_wrong_center() {
        let bad = State::<Equatorial>::new(
            None,
            0.0,
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            10, // not SSB
        );
        assert!(State::<Equatorial, SSB>::try_from(bad).is_err());
    }

    #[test]
    fn change_center() {
        let mut a = State::<Ecliptic>::new(1, 0.0, [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], 0);
        let b = State::<Equatorial>::new(3, 0.0, [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], 0);
        a.try_change_center(b.into_frame()).unwrap();

        assert!(a.center_id() == 3);
        assert!(a.pos[0] == 1.0);
        assert!(a.pos[1] != 0.0);
        assert!(a.pos[2] != 0.0);
        assert!(a.vel[0] == 1.0);

        // try cases which cause errors
        let diff_jd = State::<Equatorial>::new(3, 1.0, [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], 0);
        assert!(a.try_change_center(diff_jd.into_frame()).is_err());

        let not_naif_id = State::<Equatorial>::new(None, 0.0, [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], 0);
        assert!(a.try_change_center(not_naif_id.into_frame()).is_err());

        let no_matching_id =
            State::<Equatorial>::new(2, 0.0, [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], 1_000_000_000);
        assert!(a.try_change_center(no_matching_id.into_frame()).is_err());
    }
}
