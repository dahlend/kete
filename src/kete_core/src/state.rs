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
//!
use serde::{Deserialize, Serialize};
use std::fmt::{Debug, Display};

use crate::errors::{Error, KeteResult};
use crate::frames::{
    ecliptic_to_equatorial, ecliptic_to_fk4, ecliptic_to_galactic, equatorial_to_ecliptic,
    fk4_to_ecliptic, galactic_to_ecliptic, inertial_to_noninertial, noninertial_to_inertial, Frame,
};
use crate::spice;
use nalgebra::Vector3;

/// Designation for an object.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Hash, Eq)]
pub enum Desig {
    /// Permanent ID, an integer.
    Perm(u64),

    /// Provisional Designation
    Prov(String),

    /// Text name
    Name(String),

    /// NAIF id for the object.
    /// These are used by SPICE kernels for identification.
    Naif(i64),

    /// No id assigned.
    Empty,
}

impl Desig {
    /// Return a full string representation of the designation, including the type.
    pub fn full_string(&self) -> String {
        format!("{:?}", self)
    }
}

impl Display for Desig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&match self {
            Desig::Empty => "".to_string(),
            Desig::Prov(s) => s.clone(),
            Desig::Name(s) => s.clone(),
            Desig::Perm(i) => i.to_string(),
            Desig::Naif(i) => i.to_string(),
        })
    }
}

/// Exact State of an object.
///
/// This represents the id, position, and velocity of an object with respect to a
/// coordinate frame and a center point.
///
/// This state object assumes no uncertainty in its values.
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct State {
    /// Designation number which corresponds to the object.
    pub desig: Desig,

    /// JD of the object's state in TDB scaled time.
    pub jd: f64,

    /// Position of the object with respect to the center_id object, units of AU.
    pub pos: [f64; 3],

    /// Velocity of the object with respect to the center_id object, units of AU/Day.
    pub vel: [f64; 3],

    /// Coordinate frame of the object.
    pub frame: Frame,

    /// Position and velocity are given with respect to the specified center_id.
    /// The only privileged center ID is the Solar System Barycenter 0.
    pub center_id: i64,
}

impl State {
    /// Construct a new State object.
    #[inline(always)]
    pub fn new(
        desig: Desig,
        jd: f64,
        pos: Vector3<f64>,
        vel: Vector3<f64>,
        frame: Frame,
        center_id: i64,
    ) -> Self {
        State {
            desig,
            jd,
            pos: pos.into(),
            vel: vel.into(),
            frame,
            center_id,
        }
    }

    /// Construct a new state made of NAN pos and vel vectors but containing the
    /// remaining data. This is primarily useful as a place holder when propagation
    /// has failed and the object needs to be recorded still.
    #[inline(always)]
    pub fn new_nan(desig: Desig, jd: f64, frame: Frame, center_id: i64) -> Self {
        Self::new(
            desig,
            jd,
            [f64::NAN, f64::NAN, f64::NAN].into(),
            [f64::NAN, f64::NAN, f64::NAN].into(),
            frame,
            center_id,
        )
    }

    /// Are all values finite.
    pub fn is_finite(&self) -> bool {
        !(self.pos.iter().any(|x| !x.is_finite())
            || self.vel.iter().any(|x| !x.is_finite())
            || !self.jd.is_finite())
    }

    /// Mutate the current state from its current [`Frame`] to the new target [`Frame`].
    ///
    /// # Arguments
    ///
    /// * `target_frame` - Target frame from the [`Frame`] enum.
    #[inline(always)]
    pub fn try_change_frame_mut(&mut self, target_frame: Frame) -> KeteResult<()> {
        if self.frame == target_frame {
            return Ok(());
        }

        // All states are moved from the input frame into the ecliptic frame, then from
        // the ecliptic to the final frame. This reduces the amount of code required.

        let new_pos: [f64; 3];
        let new_vel: [f64; 3];
        match self.frame {
            Frame::Equatorial => {
                new_pos = equatorial_to_ecliptic(&self.pos.into()).into();
                new_vel = equatorial_to_ecliptic(&self.vel.into()).into();
            }
            Frame::Ecliptic => {
                new_pos = self.pos;
                new_vel = self.vel;
            }
            Frame::EclipticNonInertial(_, frame_angles) => {
                let (p, v) =
                    noninertial_to_inertial(&frame_angles, &self.pos.into(), &self.vel.into());
                new_pos = p.into();
                new_vel = v.into()
            }
            Frame::FK4 => {
                new_pos = fk4_to_ecliptic(&self.pos.into()).into();
                new_vel = fk4_to_ecliptic(&self.vel.into()).into();
            }
            Frame::Galactic => {
                new_pos = galactic_to_ecliptic(&self.pos.into()).into();
                new_vel = galactic_to_ecliptic(&self.vel.into()).into();
            }
            Frame::Unknown(id) => return Err(Error::UnknownFrame(id as i32)),
        }

        // new_pos and new_vel are now in ecliptic.
        match target_frame {
            Frame::Equatorial => {
                self.pos = ecliptic_to_equatorial(&new_pos.into()).into();
                self.vel = ecliptic_to_equatorial(&new_vel.into()).into();
            }
            Frame::Ecliptic => {
                self.pos = new_pos;
                self.vel = new_vel;
            }
            Frame::EclipticNonInertial(_, frame_angles) => {
                let (p, v) =
                    inertial_to_noninertial(&frame_angles, &new_pos.into(), &new_vel.into());
                self.pos = p.into();
                self.vel = v.into();
            }
            Frame::FK4 => {
                self.pos = ecliptic_to_fk4(&new_pos.into()).into();
                self.vel = ecliptic_to_fk4(&new_vel.into()).into();
            }
            Frame::Galactic => {
                self.pos = ecliptic_to_galactic(&new_pos.into()).into();
                self.vel = ecliptic_to_galactic(&new_vel.into()).into();
            }
            Frame::Unknown(id) => return Err(Error::UnknownFrame(id as i32)),
        }
        self.frame = target_frame;
        Ok(())
    }

    /// Trade the center ID and ID values, and flip the direction of the position and
    /// velocity vectors.
    #[inline(always)]
    pub fn try_flip_center_id(&mut self) -> KeteResult<()> {
        if let Desig::Naif(mut id) = self.desig {
            (id, self.center_id) = (self.center_id, id);
            for i in 0..3 {
                self.pos[i] = -self.pos[i];
                self.vel[i] = -self.vel[i];
            }
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

        let state_id = match state.desig {
            Desig::Naif(id) => id,
            _ => {
                return Err(Error::ValueError(
                    "Changing centers only works on states with NAIF Ids.".into(),
                ))
            }
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

        // Now they match, and we know the center id will change, update target frame
        // if necessary.
        if self.frame != state.frame {
            state.try_change_frame_mut(self.frame)?;
        }

        // Now the state is where it is supposed to be, update as required.
        self.center_id = state.center_id;
        for i in 0..3 {
            self.pos[i] += state.pos[i];
            self.vel[i] += state.vel[i];
        }
        Ok(())
    }

    /// Attempt to update the designation from a naif id to a name.
    pub fn try_naif_id_to_name(&mut self) -> Option<()> {
        if let Desig::Naif(id) = self.desig {
            self.desig = Desig::Name(spice::try_name_from_id(id)?);
            Some(())
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn flip_center() {
        let mut a = State::new(
            Desig::Naif(1),
            0.0,
            [1.0, 0.0, 0.0].into(),
            [0.0, 1.0, 0.0].into(),
            Frame::Ecliptic,
            0,
        );
        a.try_flip_center_id().unwrap();

        assert!(a.center_id == 1);
        assert!(a.pos == [-1.0, 0.0, 0.0]);
        assert!(a.vel == [0.0, -1.0, 0.0]);
    }

    #[test]
    fn nan_finite() {
        let a = State::new(
            Desig::Naif(1),
            0.0,
            [1.0, 0.0, 0.0].into(),
            [0.0, 1.0, 0.0].into(),
            Frame::Ecliptic,
            0,
        );
        assert!(a.is_finite());

        let b = State::new_nan(Desig::Empty, 0.0, Frame::Ecliptic, 1000);
        assert!(!b.is_finite())
    }

    #[test]
    fn desig_strings() {
        assert!(Desig::Empty.to_string() == "");
        assert!(Desig::Naif(100).to_string() == "100");
        assert!(Desig::Name("Foo".into()).to_string() == "Foo");
        assert!(Desig::Perm(123).to_string() == "123");
        assert!(Desig::Prov("Prov".into()).to_string() == "Prov");
    }

    #[test]
    fn naif_name_resolution() {
        let mut a = State::new(
            Desig::Naif(1),
            0.0,
            [1.0, 0.0, 0.0].into(),
            [0.0, 1.0, 0.0].into(),
            Frame::Ecliptic,
            0,
        );
        assert!(a.try_naif_id_to_name().is_some());
        assert!(a.desig == Desig::Name("mercury barycenter".into()));
        assert!(a.desig.full_string() == "Name(\"mercury barycenter\")");
        assert!(a.desig.to_string() == "mercury barycenter");
    }

    #[test]
    fn frame_roundtrip() {
        let mut state = State::new(
            Desig::Naif(1),
            0.0,
            [1.0, 0.0, 0.0].into(),
            [1.0, 0.0, 0.0].into(),
            Frame::Ecliptic,
            0,
        );

        assert!(state.try_change_frame_mut(Frame::Unknown(10000)).is_err());

        let non_inerial = Frame::EclipticNonInertial(123, [1.0, 1.2, 1.4, 1.1, 1.3, 1.5]);

        for frame_a in vec![
            Frame::Ecliptic,
            Frame::Equatorial,
            Frame::FK4,
            Frame::Galactic,
            non_inerial,
        ] {
            let mut test_state = state.clone();
            test_state.try_change_frame_mut(frame_a).unwrap();
            let pos_exp = &test_state.pos;
            let vel_exp = &test_state.vel;
            for frame_b in vec![
                Frame::Ecliptic,
                Frame::Equatorial,
                Frame::FK4,
                Frame::Galactic,
                non_inerial,
            ] {
                let mut test_state_b = test_state.clone();
                test_state_b.try_change_frame_mut(frame_b).unwrap();
                test_state_b.try_change_frame_mut(frame_a).unwrap();
                let pos = &test_state_b.pos;
                let vel = &test_state_b.vel;
                assert!((pos[0] - pos_exp[0]).abs() < 10.0 * f64::EPSILON);
                assert!((pos[1] - pos_exp[1]).abs() < 10.0 * f64::EPSILON);
                assert!((pos[2] - pos_exp[2]).abs() < 10.0 * f64::EPSILON);
                assert!((vel[0] - vel_exp[0]).abs() < 10.0 * f64::EPSILON);
                assert!((vel[1] - vel_exp[1]).abs() < 10.0 * f64::EPSILON);
                assert!((vel[2] - vel_exp[2]).abs() < 10.0 * f64::EPSILON);
            }
        }
    }
    #[test]
    fn change_center() {
        let mut a = State::new(
            Desig::Naif(1),
            0.0,
            [1.0, 0.0, 0.0].into(),
            [1.0, 0.0, 0.0].into(),
            Frame::Ecliptic,
            0,
        );
        let b = State::new(
            Desig::Naif(3),
            0.0,
            [0.0, 1.0, 0.0].into(),
            [0.0, 1.0, 0.0].into(),
            Frame::Equatorial,
            0,
        );
        a.try_change_center(b).unwrap();

        assert!(a.center_id == 3);
        assert!(a.pos[0] == 1.0);
        assert!(a.pos[1] != 0.0);
        assert!(a.pos[2] != 0.0);
        assert!(a.vel[0] == 1.0);

        // try cases which cause errors
        let diff_jd = State::new(
            Desig::Naif(3),
            1.0,
            [0.0, 1.0, 0.0].into(),
            [0.0, 1.0, 0.0].into(),
            Frame::Equatorial,
            0,
        );
        assert!(a.try_change_center(diff_jd).is_err());

        let not_naif_id = State::new(
            Desig::Empty,
            0.0,
            [0.0, 1.0, 0.0].into(),
            [0.0, 1.0, 0.0].into(),
            Frame::Equatorial,
            0,
        );
        assert!(a.try_change_center(not_naif_id).is_err());

        let no_matching_id = State::new(
            Desig::Naif(2),
            0.0,
            [0.0, 1.0, 0.0].into(),
            [0.0, 1.0, 0.0].into(),
            Frame::Equatorial,
            1000000000,
        );
        assert!(a.try_change_center(no_matching_id).is_err());
    }
}
