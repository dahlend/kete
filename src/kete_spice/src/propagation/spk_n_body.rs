//! [`SpkNBody`]: SPK-based N-body Newtonian gravity as a [`Force`] impl.
//!
//! Gravity-only. Non-gravitational forces compose via [`Sum`] with
//! [`Recenter`]-wrapped [`ParameterizedForce`] impls (e.g. [`FrozenNonGrav`],
//! [`NonGravMask`]) for the appropriate center body.
//!
//! [`Force`]: kete_core::forces::Force
//! [`Sum`]: kete_core::forces::Sum
//! [`FrozenNonGrav`]: kete_core::forces::FrozenNonGrav
//! [`NonGravMask`]: kete_core::forces::NonGravMask
//! [`Recenter`]: super::recenter::Recenter
//!
// BSD 3-Clause License
//
// Copyright (c) 2026, Dar Dahlen
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

use kete_core::errors::KeteResult;
use kete_core::forces::{
    Force, FrozenNonGrav, GravParams, NonGravMask, ParameterizedForce, Sum, analytical_jacobians,
};
use kete_core::frames::{Equatorial, SSB, Vector};
use kete_core::time::{TDB, Time};
use nalgebra::{Matrix3, Matrix3xX, Vector3};

use crate::propagation::Recenter;
use crate::spk::LOADED_SPK;

/// SPK-based N-body Newtonian gravity [`ParameterizedForce`].
///
/// `Send + Sync`, with the borrow lifetime tied to the `SpkCollection`
/// it references; callers typically hold a read guard from
/// `LOADED_SPK.try_read()` for the lifetime of the propagation.
///
/// `Center = SSB`. Compose with [`Recenter`]-wrapped [`ParameterizedForce`] impls
/// via [`Sum`] to add non-gravitational contributions.
pub struct SpkNBody {
    /// Borrowed slice of massive bodies whose gravity is included.
    /// Borrowed (not owned) so parallel batch propagation does not
    /// reallocate the planet list per task.
    pub massive_obj: Vec<GravParams>,
}

impl SpkNBody {
    /// Build with the given list of massive bodies.
    #[must_use]
    pub fn new(include_extended: bool) -> Self {
        let massive_obj = if include_extended {
            GravParams::known_masses().clone()
        } else {
            GravParams::planets().clone()
        };
        Self {
            massive_obj: massive_obj.clone(),
        }
    }
}

impl std::fmt::Debug for SpkNBody {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SpkNBody")
            .field("n_massive_obj", &self.massive_obj.len())
            .finish()
    }
}

impl ParameterizedForce for SpkNBody {
    type Frame = Equatorial;
    type Center = SSB;

    fn accel(
        &self,
        time: Time<TDB>,
        pos: &Vector<Equatorial>,
        vel: &Vector<Equatorial>,
        _free_params: &[f64],
    ) -> KeteResult<Vector<Equatorial>> {
        let spk = LOADED_SPK.try_read()?;
        let pos_v: Vector3<f64> = (*pos).into();
        let vel_v: Vector3<f64> = (*vel).into();
        let mut accel = Vector3::<f64>::zeros();
        for grav_params in &self.massive_obj {
            let body_state =
                spk.try_get_state_with_center::<Equatorial>(grav_params.naif_id, time, 0)?;
            let rel_pos: Vector3<f64> = pos_v - Vector3::from(body_state.pos);
            let rel_vel: Vector3<f64> = vel_v - Vector3::from(body_state.vel);
            grav_params.add_acceleration(&mut accel, &rel_pos, &rel_vel);
        }
        Ok(Vector::<Equatorial>::new(accel.into()))
    }

    fn jacobians(
        &self,
        time: Time<TDB>,
        pos: &Vector<Equatorial>,
        vel: &Vector<Equatorial>,
        _free_params: &[f64],
    ) -> KeteResult<(Matrix3<f64>, Matrix3<f64>)> {
        let spk = LOADED_SPK.try_read()?;
        let pos_v: Vector3<f64> = (*pos).into();
        let vel_v: Vector3<f64> = (*vel).into();
        let body_states: Vec<(Vector3<f64>, Vector3<f64>)> = self
            .massive_obj
            .iter()
            .map(|g| {
                let s = spk.try_get_state_with_center::<Equatorial>(g.naif_id, time, 0)?;
                Ok((Vector3::from(s.pos), Vector3::from(s.vel)))
            })
            .collect::<KeteResult<_>>()?;
        Ok(analytical_jacobians(
            &pos_v,
            &vel_v,
            &body_states,
            &self.massive_obj,
        ))
    }
}

/// Marker: `SpkNBody` is pure gravity with `n_free_params() == 0`.
impl Force for SpkNBody {}

/// Force model enum to simplify the most common composition pattern:
/// pure N-body gravity with optionally some kind of non-gravitational force.
#[derive(Debug)]
pub enum SpkNonGravs {
    /// Pure N-body gravity.
    Gravity(SpkNBody),
    /// N-body gravity composed with a `Recenter`-wrapped non-grav mask.
    WithNonGrav(Sum<SpkNBody, Recenter<SSB, NonGravMask>>),
    /// N-body gravity composed with a `Recenter`-wrapped frozen non-grav.
    WithFrozenNonGrav(Sum<SpkNBody, Recenter<SSB, FrozenNonGrav>>),
}

impl SpkNonGravs {
    /// Constructor for pure gravity.
    #[must_use]
    pub fn gravity(include_extended: bool) -> Self {
        Self::Gravity(SpkNBody::new(include_extended))
    }

    /// Constructor for gravity with a masked non-grav.
    #[must_use]
    pub fn with_non_grav_mask(include_extended: bool, mask: NonGravMask) -> Self {
        Self::WithNonGrav(Sum::new(
            SpkNBody::new(include_extended),
            Recenter::<SSB, _>::new(mask),
        ))
    }

    /// Constructor for gravity with a frozen non-grav.
    #[must_use]
    pub fn with_frozen_non_grav(include_extended: bool, frozen: FrozenNonGrav) -> Self {
        Self::WithFrozenNonGrav(Sum::new(
            SpkNBody::new(include_extended),
            Recenter::<SSB, _>::new(frozen),
        ))
    }
}

impl ParameterizedForce for SpkNonGravs {
    type Frame = Equatorial;
    type Center = SSB;

    fn n_free_params(&self) -> usize {
        match self {
            Self::Gravity(f) => f.n_free_params(),
            Self::WithNonGrav(f) => f.n_free_params(),
            Self::WithFrozenNonGrav(f) => f.n_free_params(),
        }
    }

    fn free_param_names(&self) -> Vec<&'static str> {
        match self {
            Self::Gravity(f) => f.free_param_names(),
            Self::WithNonGrav(f) => f.free_param_names(),
            Self::WithFrozenNonGrav(f) => f.free_param_names(),
        }
    }

    fn lower_bounds(&self) -> Vec<Option<f64>> {
        match self {
            Self::Gravity(f) => f.lower_bounds(),
            Self::WithNonGrav(f) => f.lower_bounds(),
            Self::WithFrozenNonGrav(f) => f.lower_bounds(),
        }
    }

    fn accel(
        &self,
        time: Time<TDB>,
        pos: &Vector<Equatorial>,
        vel: &Vector<Equatorial>,
        free_params: &[f64],
    ) -> KeteResult<Vector<Equatorial>> {
        match self {
            Self::Gravity(f) => f.accel(time, pos, vel, free_params),
            Self::WithNonGrav(f) => f.accel(time, pos, vel, free_params),
            Self::WithFrozenNonGrav(f) => f.accel(time, pos, vel, free_params),
        }
    }

    fn jacobians(
        &self,
        time: Time<TDB>,
        pos: &Vector<Equatorial>,
        vel: &Vector<Equatorial>,
        free_params: &[f64],
    ) -> KeteResult<(Matrix3<f64>, Matrix3<f64>)> {
        match self {
            Self::Gravity(f) => f.jacobians(time, pos, vel, free_params),
            Self::WithNonGrav(f) => f.jacobians(time, pos, vel, free_params),
            Self::WithFrozenNonGrav(f) => f.jacobians(time, pos, vel, free_params),
        }
    }

    fn parameter_jacobian(
        &self,
        time: Time<TDB>,
        pos: &Vector<Equatorial>,
        vel: &Vector<Equatorial>,
        free_params: &[f64],
    ) -> KeteResult<Matrix3xX<f64>> {
        match self {
            Self::Gravity(f) => f.parameter_jacobian(time, pos, vel, free_params),
            Self::WithNonGrav(f) => f.parameter_jacobian(time, pos, vel, free_params),
            Self::WithFrozenNonGrav(f) => f.parameter_jacobian(time, pos, vel, free_params),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kete_core::desigs::Desig;
    use kete_core::state::State;

    /// Analytical `jacobian_pos` matches FD of `accel` to FD precision.
    #[test]
    fn spk_n_body_jacobian_pos_matches_finite_difference() {
        crate::test_data::ensure_test_spk();
        let force = SpkNBody::new(false);
        let time = Time::<TDB>::new(2_451_545.0);
        let pos = Vector::<Equatorial>::new([1.5, 0.5, 0.1]);
        let vel = Vector::<Equatorial>::new([-0.008, 0.012, 0.001]);

        let (analytical, _) = force.jacobians(time, &pos, &vel, &[]).unwrap();

        let pos_v: Vector3<f64> = pos.into();
        let h = pos_v.norm() * 1e-6;
        let mut fd = Matrix3::<f64>::zeros();
        for j in 0..3 {
            let mut p_plus = pos_v;
            p_plus[j] += h;
            let mut p_minus = pos_v;
            p_minus[j] -= h;
            let a_plus: Vector3<f64> = force
                .accel(time, &Vector::<Equatorial>::new(p_plus.into()), &vel, &[])
                .unwrap()
                .into();
            let a_minus: Vector3<f64> = force
                .accel(time, &Vector::<Equatorial>::new(p_minus.into()), &vel, &[])
                .unwrap()
                .into();
            let col = (a_plus - a_minus) / (2.0 * h);
            for i in 0..3 {
                fd[(i, j)] = col[i];
            }
        }
        let max_err = (analytical - fd).abs().max();
        let analytical_scale = analytical.abs().max();
        assert!(
            max_err < 1e-6 * analytical_scale.max(1e-6),
            "max_err = {max_err}, analytical_scale = {analytical_scale}"
        );
    }

    /// With analytical jacobians, the variational STM matches the
    /// existing `compute_state_transition` to working precision.
    #[test]
    fn spk_n_body_analytical_stm_matches_compute_state_transition() {
        use kete_core::state::propagate_with_stm;

        crate::test_data::ensure_test_spk();

        let start = State::<Equatorial, SSB> {
            desig: Desig::Empty,
            epoch: Time::<TDB>::new(2_451_545.0),
            pos: Vector::<Equatorial>::new([0.5, 1.0, 0.1]),
            vel: Vector::<Equatorial>::new([-0.012, 0.008, 0.001]),
            center: SSB,
        };
        let target = Time::<TDB>::new(2_451_545.0 + 5.0);

        let (legacy_state, legacy_stm) = crate::propagation::compute_state_transition::<
            kete_core::forces::JplCometNonGrav,
        >(&start, target, false, None)
        .unwrap();

        let force = SpkNBody::new(false);
        let pos_init: Vector3<f64> = start.pos.into();
        let vel_init: Vector3<f64> = start.vel.into();
        let (pos_new, vel_new, new_stm) =
            propagate_with_stm(&force, pos_init, vel_init, &[], start.epoch, target).unwrap();

        let legacy_pos: Vector3<f64> = legacy_state.pos.into();
        let legacy_vel: Vector3<f64> = legacy_state.vel.into();
        assert!((pos_new - legacy_pos).norm() < 1e-12);
        assert!((vel_new - legacy_vel).norm() < 1e-12);

        let max_diff = (legacy_stm - new_stm).abs().max();
        assert!(
            max_diff < 1e-10,
            "STM max element diff = {max_diff} (analytical jacobian)"
        );
    }
}
