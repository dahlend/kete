// SPDX-FileCopyrightText: 2026 Dar Dahlen
// SPDX-License-Identifier: BSD-3-Clause

//! Ephemeris providers: states of solar system bodies.
//!
//! The propagation and visibility code in [`propagation`](crate::propagation)
//! and [`fov`](crate::fov) needs the position of other bodies at arbitrary
//! times. It gets them from an [`EphemerisSource`].
//!
//! Orientations of body frames are separate: a
//! [`FrameSource`](crate::frames::FrameSource) provides them, so states and
//! frames can come from different providers.
//!
//! `kete_spice` implements the trait for its loaded kernels. Any other source
//! of states, such as a table or an analytic model, can implement it too.

use crate::errors::KeteResult;
use crate::frames::{Equatorial, SSB, SunCenter};
use crate::state::State;
use crate::time::{TDB, Time};
use nalgebra::Vector3;

/// A source of body states.
///
/// States are equatorial, with positions in AU and velocities in AU/day. A
/// provider returns an error for a body, frame or time that it does not cover.
/// It does not extrapolate.
///
/// Only [`try_get_state_with_center`](Self::try_get_state_with_center) is
/// required. The provided methods derive from it. A provider can override them
/// with a faster route to the same result.
///
/// Threads share a provider by reference, so it is `Sync`. It need not be
/// `Send`, so it can hold read guards.
pub trait EphemerisSource: Sync {
    /// State of body `id` relative to body `center` at `time`.
    ///
    /// A provider serves every body it covers relative to the Solar System
    /// Barycenter (NAIF id 0) at least. The provided methods route through
    /// it.
    ///
    /// # Errors
    /// Fails when the provider does not cover `id` or `center` at `time`.
    fn try_get_state_with_center(
        &self,
        id: i32,
        time: Time<TDB>,
        center: i32,
    ) -> KeteResult<State<Equatorial>>;

    /// Move `state` to be relative to body `center`.
    ///
    /// The default subtracts the state of `center` relative to the barycenter
    /// and adds that of the current center.
    ///
    /// # Errors
    /// Fails when the provider does not cover either center at the epoch of
    /// `state`.
    fn try_change_center(&self, state: &mut State<Equatorial>, center: i32) -> KeteResult<()> {
        let old = state.center_id();
        if old == center {
            return Ok(());
        }
        let (mut pos, mut vel): (Vector3<f64>, Vector3<f64>) = (state.pos.into(), state.vel.into());
        if old != 0 {
            let s = self.try_get_state_with_center(old, state.epoch, 0)?;
            pos += Vector3::from(s.pos);
            vel += Vector3::from(s.vel);
        }
        if center != 0 {
            let s = self.try_get_state_with_center(center, state.epoch, 0)?;
            pos -= Vector3::from(s.pos);
            vel -= Vector3::from(s.vel);
        }
        *state = State::new(state.desig.clone(), state.epoch, pos, vel, center);
        Ok(())
    }

    /// `state` relative to the Solar System Barycenter.
    ///
    /// # Errors
    /// As [`try_change_center`](Self::try_change_center).
    fn try_to_ssb(&self, mut state: State<Equatorial>) -> KeteResult<State<Equatorial, SSB>> {
        self.try_change_center(&mut state, 0)?;
        State::<Equatorial, SSB>::try_from(state)
    }

    /// `state` relative to the Sun.
    ///
    /// # Errors
    /// As [`try_change_center`](Self::try_change_center).
    fn try_to_sun(&self, mut state: State<Equatorial>) -> KeteResult<State<Equatorial, SunCenter>> {
        self.try_change_center(&mut state, 10)?;
        State::<Equatorial, SunCenter>::try_from(state)
    }
}

#[cfg(test)]
pub(crate) mod test_ephemeris {
    //! A small analytic ephemeris for tests that need no kernels.

    use super::{EphemerisSource, KeteResult};
    use crate::desigs::Desig;
    use crate::errors::Error;
    use crate::frames::Equatorial;
    use crate::state::State;
    use crate::time::{TDB, Time};
    use nalgebra::Vector3;

    /// The Sun at the barycenter, at rest, and one body (NAIF id 5) on a
    /// circular orbit of `radius` AU about it in the equatorial plane, with
    /// angular rate `rate` rad/day.
    #[derive(Debug, Clone, Copy)]
    pub(crate) struct SunAndOne {
        pub radius: f64,
        pub rate: f64,
    }

    impl SunAndOne {
        fn ssb_state(&self, id: i32, time: Time<TDB>) -> KeteResult<(Vector3<f64>, Vector3<f64>)> {
            match id {
                0 | 10 => Ok((Vector3::zeros(), Vector3::zeros())),
                5 => {
                    let phase = self.rate * (time.jd() - 2_451_545.0);
                    let (s, c) = phase.sin_cos();
                    Ok((
                        Vector3::new(c, s, 0.0) * self.radius,
                        Vector3::new(-s, c, 0.0) * self.radius * self.rate,
                    ))
                }
                _ => Err(Error::Bounds(format!("No state for {id}."))),
            }
        }
    }

    impl EphemerisSource for SunAndOne {
        fn try_get_state_with_center(
            &self,
            id: i32,
            time: Time<TDB>,
            center: i32,
        ) -> KeteResult<State<Equatorial>> {
            let (p, v) = self.ssb_state(id, time)?;
            let (cp, cv) = self.ssb_state(center, time)?;
            Ok(State::new(Desig::Naif(id), time, p - cp, v - cv, center))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::test_ephemeris::SunAndOne;
    use super::*;
    use crate::desigs::Desig;

    /// The default center change moves a state between centers through the
    /// barycenter, and back.
    #[test]
    fn default_center_change_round_trips() {
        let eph = SunAndOne {
            radius: 5.2,
            rate: 0.0015,
        };
        let time = Time::<TDB>::new(2_451_600.0);
        let start = State::<Equatorial>::new(
            Desig::Empty,
            time,
            [1.0, 2.0, 0.5],
            [0.01, -0.002, 0.003],
            5,
        );
        let body = eph.try_get_state_with_center(5, time, 10).unwrap();
        let mut moved = start.clone();
        eph.try_change_center(&mut moved, 10).unwrap();
        assert_eq!(moved.center_id(), 10);
        assert!(
            (Vector3::from(moved.pos) - Vector3::from(start.pos) - Vector3::from(body.pos)).norm()
                < 1e-15
        );
        eph.try_change_center(&mut moved, 5).unwrap();
        assert!((Vector3::from(moved.pos) - Vector3::from(start.pos)).norm() < 1e-15);
        assert!((Vector3::from(moved.vel) - Vector3::from(start.vel)).norm() < 1e-15);
    }
}
