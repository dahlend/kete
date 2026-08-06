//! `StateLike` trait: state-shape polymorphism for propagation.
//!
//! A `StateLike` knows its frame and center, exposes its epoch, and
//! advances itself under a given [`ParameterizedForce`] to a target epoch, so the
//! propagator boundary is a single trait method.
//!
//! It is implemented for the exact cartesian state shapes only. `UncertainState` and
//! `DiffuseState` are element-native and carry a covariance; see the note at the bottom
//! of this file for why the trait does not fit them.
//!
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

use nalgebra::Vector3;

use super::State;
use crate::errors::KeteResult;
use crate::forces::ParameterizedForce;
use crate::frames::{CenterBody, DynCenter, InertialFrame, Vector};
use crate::integrators::RadauIntegrator;
use crate::time::{TDB, Time};

/// State-shape polymorphism for propagation.
///
/// Implementors describe how their particular state shape advances under
/// a given [`ParameterizedForce`]. [`State`] integrates `accel` directly.
pub trait StateLike {
    /// Inertial frame of the state.
    type Frame: InertialFrame;

    /// Center marker (typically `SSB`, `SunCenter`, `EarthCenter`).
    type Center: Send + Sync + Copy + 'static;

    /// Current epoch of the state.
    fn epoch(&self) -> Time<TDB>;

    /// Advance the state to `to` under the given forces.
    ///
    /// Validates that `forces.n_free_params()` matches whatever fitted
    /// parameters the state carries (zero for plain `State`).
    ///
    /// # Errors
    /// Propagation may fail for several reasons (integrator failure,
    /// missing data referenced by forces, parameter-count mismatch).
    fn propagate_with<Forces>(self, forces: &Forces, to: Time<TDB>) -> KeteResult<Self>
    where
        Forces: ParameterizedForce<Frame = Self::Frame, Center = Self::Center>,
        Self: Sized;
}

impl<F: InertialFrame, C: CenterBody + 'static> StateLike for State<F, C>
where
    DynCenter: From<C>,
{
    type Frame = F;
    type Center = C;

    fn epoch(&self) -> Time<TDB> {
        self.epoch
    }

    fn propagate_with<Forces>(self, forces: &Forces, to: Time<TDB>) -> KeteResult<Self>
    where
        Forces: ParameterizedForce<Frame = F, Center = C>,
    {
        if forces.n_free_params() != 0 {
            return Err(crate::errors::Error::ValueError(format!(
                "State::propagate_with requires a force with zero free parameters \
                 (got {}); freeze any non-grav template first via FrozenForce::new.",
                forces.n_free_params()
            )));
        }

        let pos_init: Vector3<f64> = self.pos.into();
        let vel_init: Vector3<f64> = self.vel.into();

        // Wrap the ParameterizedForce trait into the SecondOrderODE shape the integrator
        // expects. Capture `forces` by reference; metadata is unit type
        // since plain State propagation needs no per-step bookkeeping.
        let ode = |time: Time<TDB>,
                   pos: &Vector3<f64>,
                   vel: &Vector3<f64>,
                   _meta: &mut (),
                   _exact_eval: bool|
         -> KeteResult<Vector3<f64>> {
            let pos_typed = Vector::<F>::new([pos[0], pos[1], pos[2]]);
            let vel_typed = Vector::<F>::new([vel[0], vel[1], vel[2]]);
            let accel = forces.accel(time, &pos_typed, &vel_typed, &[])?;
            Ok(accel.into())
        };

        let (final_pos, final_vel, ()) =
            RadauIntegrator::integrate(&ode, pos_init, vel_init, self.epoch, to, (), Some(3))?;

        Ok(Self {
            desig: self.desig,
            epoch: to,
            pos: Vector::<F>::new([final_pos[0], final_pos[1], final_pos[2]]),
            vel: Vector::<F>::new([final_vel[0], final_vel[1], final_vel[2]]),
            center: self.center,
        })
    }
}

// `UncertainState` and `DiffuseState` deliberately do **not** implement `StateLike`.
//
// The trait fixes `Center` at the type level and `propagate_with` requires the force to
// share it. That works for a bare state, whose center is both what its coordinates are
// relative to and where the ODE is integrated. An element-native uncertain state has
// neither property: its element center is a runtime NAIF id, and its integration center is
// a per-call choice that need not equal it. Propagating a covariance is also basis
// specific in a way propagating a point is not - the same distribution is far better
// described in element coordinates than in cartesian - so a trait that makes the bases look
// interchangeable here would be misleading rather than merely loose.
//
// Nothing needs them: the workspace has no functions generic over `StateLike`, and every
// call site outside this file propagates a plain `State`.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::desigs::Desig;
    use crate::forces::{ParameterizedForce, Sum};
    use crate::frames::{Equatorial, SunCenter};

    /// A trivial central-mass gravity force for testing. Constant GM at
    /// the center body. Holds no free parameters.
    struct TwoBody {
        gm: f64,
    }

    impl ParameterizedForce for TwoBody {
        type Frame = Equatorial;
        type Center = SunCenter;

        fn accel(
            &self,
            _time: Time<TDB>,
            pos: &Vector<Equatorial>,
            _vel: &Vector<Equatorial>,
            _free_params: &[f64],
        ) -> KeteResult<Vector<Equatorial>> {
            let p: Vector3<f64> = (*pos).into();
            let r3 = p.norm().powi(3);
            Ok(Vector::<Equatorial>::new((-p * (self.gm / r3)).into()))
        }
    }

    #[test]
    fn state_propagate_with_two_body_kepler() {
        // Circular orbit at 1 AU around a body with GM = (2*pi/year)^2 = unity in our units.
        // We'll use the kete GMS constant via a simple fact: at r = 1 AU, with circular
        // velocity v = sqrt(GMS), we should return to the start after one period 2*pi/sqrt(GMS).
        let gm = crate::constants::GMS;
        let v_circ = gm.sqrt();
        let period = 2.0 * std::f64::consts::PI / v_circ;
        let start = State::<Equatorial, SunCenter> {
            desig: Desig::Empty,
            epoch: Time::<TDB>::new(0.0),
            pos: Vector::<Equatorial>::new([1.0, 0.0, 0.0]),
            vel: Vector::<Equatorial>::new([0.0, v_circ, 0.0]),
            center: SunCenter,
        };
        let force = TwoBody { gm };
        let final_state = start
            .clone()
            .propagate_with(&force, Time::<TDB>::new(period))
            .unwrap();
        let pos: Vector3<f64> = final_state.pos.into();
        let vel: Vector3<f64> = final_state.vel.into();
        // After one full period, position and velocity should match the start.
        assert!((pos.x - 1.0).abs() < 1e-9, "x = {}", pos.x);
        assert!(pos.y.abs() < 1e-9, "y = {}", pos.y);
        assert!(pos.z.abs() < 1e-12, "z = {}", pos.z);
        assert!(vel.x.abs() < 1e-9, "vx = {}", vel.x);
        assert!((vel.y - v_circ).abs() < 1e-9, "vy = {}", vel.y);
        assert!(vel.z.abs() < 1e-12, "vz = {}", vel.z);
    }

    #[test]
    fn state_propagate_with_sum_two_forces() {
        // TwoBody (gm) + TwoBody (0) summed should equal TwoBody (gm)
        // directly, since the zero force contributes no acceleration.
        let gm = crate::constants::GMS;
        let v_circ = gm.sqrt();
        let dt = 30.0; // 30 days
        let start = State::<Equatorial, SunCenter> {
            desig: Desig::Empty,
            epoch: Time::<TDB>::new(0.0),
            pos: Vector::<Equatorial>::new([1.0, 0.0, 0.0]),
            vel: Vector::<Equatorial>::new([0.0, v_circ, 0.0]),
            center: SunCenter,
        };
        let direct = start
            .clone()
            .propagate_with(&TwoBody { gm }, Time::<TDB>::new(dt))
            .unwrap();
        let sum = Sum::new(TwoBody { gm }, TwoBody { gm: 0.0 });
        let through_sum = start.propagate_with(&sum, Time::<TDB>::new(dt)).unwrap();
        let p1: Vector3<f64> = direct.pos.into();
        let p2: Vector3<f64> = through_sum.pos.into();
        let v1: Vector3<f64> = direct.vel.into();
        let v2: Vector3<f64> = through_sum.vel.into();
        assert!((p1 - p2).norm() < 1e-14);
        assert!((v1 - v2).norm() < 1e-14);
    }
}
