//! [`Recenter`]: a `ParameterizedForce` adapter that shifts the reference body of
//! the input pos/vel before delegating to an inner `ParameterizedForce`.
//!
//! ## Why this exists
//!
//! Forces are written in the natural reference frame for their physics:
//! gravity from N bodies in SSB-relative coordinates, JPL non-grav and
//! dust SRP in Sun-relative coordinates, atmospheric drag in
//! Earth-relative coordinates, polyhedral asteroid gravity in
//! body-relative coordinates. Composing two forces via
//! [`Sum`](kete_core::forces::Sum) requires both members to share
//! `Center`, which would be impossible without an adapter that bridges
//! between centers.
//!
//! [`Recenter`] is that adapter for **inertial-to-inertial translation**.
//! It looks up the offset between two NAIF bodies via SPK at each call,
//! subtracts it from the input pos/vel, then delegates to the inner
//! `ParameterizedForce`. Body-fixed-frame conversions (which carry rotational
//! components and require PCK data, not just SPK) are out of scope and
//! would need a sibling adapter.
//!
//! ## Construction
//!
//! ```ignore
//! // Sun-relative dust SRP wrapped to participate in an SSB-centered force model.
//! let dust = Recenter::<SSB, _>::new(&spk, DustNonGrav);
//! //                    ^^^                ^^^^^^^^^
//! //                    From type         inner force (Center = SunCenter)
//! ```
//!
//! The integrator hands the wrapper SSB-relative pos/vel; the wrapper
//! converts to Sun-relative and calls the inner force's `accel` with
//! the shifted coordinates.
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

use std::cell::RefCell;
use std::marker::PhantomData;

use kete_core::errors::KeteResult;
use kete_core::forces::{Force, ParameterizedForce};
use kete_core::frames::{CenterBody, Vector};
use kete_core::time::{TDB, Time};
use nalgebra::{Matrix3, Matrix3xX, Vector3};

use crate::spk::SpkCollection;

// Per-thread cache for the most recent Recenter shift lookup.
// Keyed by (from_naif_id, to_naif_id, time.jd) so different Recenter
// instances on the same thread do not collide. Each rayon thread gets its
// own slot, eliminating the cross-thread mutex contention that hurt
// parallel sigma-point evaluation.
thread_local! {
    static SHIFT_CACHE: RefCell<Option<(i32, i32, f64, Vector3<f64>, Vector3<f64>)>> =
        const { RefCell::new(None) };
}

/// `ParameterizedForce` adapter that recenters input pos/vel from `FromC` to
/// `F::Center` via SPK lookup before delegating to the inner force.
///
/// `Center = FromC` (the type the integrator sees); the inner force's
/// `Center` is `F::Center` (where the inner physics is naturally
/// expressed).
///
/// Per integrator step this performs **one** SPK query for
/// `(F::Center)` relative to `FromC`. The offset is reused for the
/// `accel`/`jacobian_pos`/`jacobian_vel`/`parameter_jacobian` quartet
/// because all four methods are called at the same `time` -- but the
/// query is repeated per quartet, since `ParameterizedForce` doesn't expose a
/// per-step context.
pub struct Recenter<'a, FromC, F>
where
    F: ParameterizedForce,
    F::Center: CenterBody,
    FromC: CenterBody,
{
    /// Borrowed reference to the loaded SPK collection.
    pub spk: &'a SpkCollection,
    /// The inner force, written for `Center = F::Center`.
    pub inner: F,
    _phantom: PhantomData<FromC>,
}

impl<'a, FromC, F> Recenter<'a, FromC, F>
where
    F: ParameterizedForce,
    F::Center: CenterBody,
    FromC: CenterBody,
{
    /// Wrap `inner` so it accepts pos/vel relative to `FromC`. The
    /// inner force's `Center` is determined by its impl.
    #[must_use]
    pub fn new(spk: &'a SpkCollection, inner: F) -> Self {
        Self {
            spk,
            inner,
            _phantom: PhantomData,
        }
    }
}

impl<FromC, F> std::fmt::Debug for Recenter<'_, FromC, F>
where
    F: ParameterizedForce + std::fmt::Debug,
    F::Center: CenterBody,
    FromC: CenterBody,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Recenter")
            .field("from_id", &FromC::NAIF_ID)
            .field("to_id", &F::Center::NAIF_ID)
            .field("inner", &self.inner)
            .finish()
    }
}

impl<FromC, F> ParameterizedForce for Recenter<'_, FromC, F>
where
    F: ParameterizedForce,
    F::Center: CenterBody,
    FromC: CenterBody,
{
    type Frame = F::Frame;
    type Center = FromC;

    fn n_free_params(&self) -> usize {
        self.inner.n_free_params()
    }

    fn free_param_names(&self) -> Vec<&'static str> {
        self.inner.free_param_names()
    }

    fn accel(
        &self,
        time: Time<TDB>,
        pos: &Vector<F::Frame>,
        vel: &Vector<F::Frame>,
        free_params: &[f64],
    ) -> KeteResult<Vector<F::Frame>> {
        let (shifted_pos, shifted_vel) = self.shift(time, pos, vel)?;
        self.inner
            .accel(time, &shifted_pos, &shifted_vel, free_params)
    }

    fn jacobians(
        &self,
        time: Time<TDB>,
        pos: &Vector<F::Frame>,
        vel: &Vector<F::Frame>,
        free_params: &[f64],
    ) -> KeteResult<(Matrix3<f64>, Matrix3<f64>)> {
        // The shift is purely a function of time, so d(shifted)/d(input) = I.
        // The jacobians pass through unchanged.
        let (shifted_pos, shifted_vel) = self.shift(time, pos, vel)?;
        self.inner
            .jacobians(time, &shifted_pos, &shifted_vel, free_params)
    }

    fn parameter_jacobian(
        &self,
        time: Time<TDB>,
        pos: &Vector<F::Frame>,
        vel: &Vector<F::Frame>,
        free_params: &[f64],
    ) -> KeteResult<Matrix3xX<f64>> {
        // The shift does not depend on parameters, so parameter
        // jacobians pass through unchanged.
        let (shifted_pos, shifted_vel) = self.shift(time, pos, vel)?;
        self.inner
            .parameter_jacobian(time, &shifted_pos, &shifted_vel, free_params)
    }
}

/// Marker: `Recenter` is a [`Force`] iff its inner force is. Recenter never
/// introduces free parameters; it only shifts the coordinate frame.
impl<FromC, F> Force for Recenter<'_, FromC, F>
where
    F: Force,
    F::Center: CenterBody,
    FromC: CenterBody,
{
}

impl<FromC, F> Recenter<'_, FromC, F>
where
    F: ParameterizedForce,
    F::Center: CenterBody,
    FromC: CenterBody,
{
    /// Look up the `(F::Center - FromC)` offset at `time` from the borrowed SPK
    /// and subtract it from the input `(pos, vel)`. Returns the
    /// `F::Center`-relative coordinates the inner force expects.
    ///
    /// The shift is cached per-thread by `(from_naif_id, to_naif_id, time.jd)`.
    /// Repeated calls within a single integrator substage (FD-jacobian fan-out:
    /// accel + 3 perturbed accels for `jacobian_pos`, etc.) hit the cache
    /// without any lock. Each rayon thread has its own slot, so parallel
    /// sigma-point evaluation incurs no cross-thread contention.
    fn shift(
        &self,
        time: Time<TDB>,
        pos: &Vector<F::Frame>,
        vel: &Vector<F::Frame>,
    ) -> KeteResult<(Vector<F::Frame>, Vector<F::Frame>)> {
        let pos_v: Vector3<f64> = (*pos).into();
        let vel_v: Vector3<f64> = (*vel).into();
        let from_id = FromC::NAIF_ID;
        let to_id = F::Center::NAIF_ID;

        let cached = SHIFT_CACHE.with_borrow(|c| {
            if let Some((cf, ct, cjd, cp, cv)) = c.as_ref()
                && *cf == from_id
                && *ct == to_id
                && *cjd == time.jd
            {
                return Some((*cp, *cv));
            }
            None
        });

        let (shift_pos, shift_vel) = if let Some(hit) = cached {
            hit
        } else {
            let shift = self.spk.try_get_state_with_center::<F::Frame>(
                F::Center::NAIF_ID,
                time,
                FromC::NAIF_ID,
            )?;
            let sp: Vector3<f64> = shift.pos.into();
            let sv: Vector3<f64> = shift.vel.into();
            SHIFT_CACHE.with_borrow_mut(|c| {
                *c = Some((from_id, to_id, time.jd, sp, sv));
            });
            (sp, sv)
        };

        Ok((
            Vector::<F::Frame>::new((pos_v - shift_pos).into()),
            Vector::<F::Frame>::new((vel_v - shift_vel).into()),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kete_core::desigs::Desig;
    use kete_core::forces::{DustNonGrav, JplCometNonGrav};
    use kete_core::frames::{Equatorial, SSB};
    use kete_core::state::State;

    use crate::spk::LOADED_SPK;

    /// `Recenter::<SSB, DustNonGrav>` correctly converts SSB-relative
    /// input into Sun-relative coordinates for the inner force.
    #[test]
    fn recenter_dust_accel_matches_direct_sun_relative() {
        crate::test_data::ensure_test_spk();
        let spk_guard = LOADED_SPK.try_read().unwrap();

        let start = State::<Equatorial, SSB> {
            desig: Desig::Empty,
            epoch: Time::<TDB>::new(2_451_545.0),
            pos: Vector::<Equatorial>::new([0.5, 1.0, 0.1]),
            vel: Vector::<Equatorial>::new([-0.012, 0.008, 0.001]),
            center: SSB,
        };
        let beta = 0.001;
        let dust_force = Recenter::<SSB, _>::new(&spk_guard, DustNonGrav);

        let pos_ssb: Vector3<f64> = start.pos.into();
        let vel_ssb: Vector3<f64> = start.vel.into();
        let sun_state = spk_guard
            .try_get_state_with_center::<Equatorial>(10, start.epoch, 0)
            .unwrap();
        let pos_sun: Vector3<f64> = pos_ssb - Vector3::from(sun_state.pos);
        let vel_sun: Vector3<f64> = vel_ssb - Vector3::from(sun_state.vel);

        let pos_sun_v = Vector::<Equatorial>::new([pos_sun[0], pos_sun[1], pos_sun[2]]);
        let vel_sun_v = Vector::<Equatorial>::new([vel_sun[0], vel_sun[1], vel_sun[2]]);
        let reference_accel: Vector3<f64> = DustNonGrav
            .accel(start.epoch, &pos_sun_v, &vel_sun_v, &[beta])
            .unwrap()
            .into();

        let through_wrapper: Vector3<f64> = dust_force
            .accel(start.epoch, &start.pos, &start.vel, &[beta])
            .unwrap()
            .into();

        assert!(
            (through_wrapper - reference_accel).norm() < 1e-14,
            "Recenter dust output does not match direct Sun-relative dust output:\n  \
             wrapper={through_wrapper:?}\n  reference={reference_accel:?}"
        );
    }

    /// `Recenter::<FromC, F>` exposes the inner force's free params
    /// unchanged.
    #[test]
    fn recenter_free_params_passthrough() {
        crate::test_data::ensure_test_spk();
        let spk_guard = LOADED_SPK.try_read().unwrap();
        let force = Recenter::<SSB, _>::new(&spk_guard, DustNonGrav);
        assert_eq!(force.n_free_params(), 1);
        assert_eq!(force.free_param_names(), vec!["beta"]);
    }

    /// End-to-end: `Sum<SpkNBody, Recenter<SSB, _>>` with `JplCometNonGrav`
    /// composed directly produces a consistent propagation result.
    /// Smoke test of the static composition path through the variational
    /// integrator with non-zero free parameters.
    #[test]
    fn composed_jpl_comet_via_sum_propagates_consistently() {
        use kete_core::forces::Sum;
        use kete_core::state::propagate_with_stm;

        use crate::propagation::SpkNBody;

        crate::test_data::ensure_test_spk();
        let spk_guard = LOADED_SPK.try_read().unwrap();

        let pos_init = Vector3::new(0.5, 1.0, 0.1);
        let vel_init = Vector3::new(-0.012, 0.008, 0.001);
        let epoch = Time::<TDB>::new(2_451_545.0);
        let target = Time::<TDB>::new(2_451_545.0 + 5.0);

        let a1 = 1.0e-8;
        let a2 = 2.0e-9;
        let a3 = -3.0e-10;

        let force = Sum::new(
            SpkNBody::new(&spk_guard, false),
            Recenter::<SSB, _>::new(&spk_guard, JplCometNonGrav::standard_comet()),
        );
        let (pos_f, vel_f, sens_f) =
            propagate_with_stm(&force, pos_init, vel_init, &[a1, a2, a3], epoch, target).unwrap();

        assert!(pos_f.iter().all(|v| v.is_finite()));
        assert!(vel_f.iter().all(|v| v.is_finite()));
        assert_eq!(sens_f.nrows(), 6);
        assert_eq!(sens_f.ncols(), 9);
    }

    /// `Recenter::<FromC, F>` jacobians pass through the inner
    /// force's jacobians unchanged (modulo the shifted evaluation
    /// point). Verifies by direct comparison: jacobian at the wrapper's
    /// SSB-relative input should equal the inner's jacobian at
    /// Sun-relative input.
    #[test]
    fn recenter_jacobians_passthrough() {
        crate::test_data::ensure_test_spk();
        let spk_guard = LOADED_SPK.try_read().unwrap();
        let dust = DustNonGrav;

        let time = Time::<TDB>::new(2_451_545.0);
        let pos_ssb = Vector::<Equatorial>::new([0.5, 1.0, 0.1]);
        let vel_ssb = Vector::<Equatorial>::new([-0.012, 0.008, 0.001]);
        let beta = 0.001;

        let wrapped = Recenter::<SSB, _>::new(&spk_guard, DustNonGrav);
        let (j_pos_wrapped, j_vel_wrapped) = wrapped
            .jacobians(time, &pos_ssb, &vel_ssb, &[beta])
            .unwrap();

        let sun = spk_guard
            .try_get_state_with_center::<Equatorial>(10, time, 0)
            .unwrap();
        let pos_ssb_v: Vector3<f64> = pos_ssb.into();
        let vel_ssb_v: Vector3<f64> = vel_ssb.into();
        let pos_sun = Vector::<Equatorial>::new((pos_ssb_v - Vector3::from(sun.pos)).into());
        let vel_sun = Vector::<Equatorial>::new((vel_ssb_v - Vector3::from(sun.vel)).into());
        let (j_pos_direct, j_vel_direct) =
            dust.jacobians(time, &pos_sun, &vel_sun, &[beta]).unwrap();

        let max_diff = (j_pos_wrapped - j_pos_direct).abs().max();
        assert!(
            max_diff < 1e-14,
            "jacobians passthrough mismatch (da/dr): max diff = {max_diff}"
        );
        let max_diff_v = (j_vel_wrapped - j_vel_direct).abs().max();
        assert!(
            max_diff_v < 1e-14,
            "jacobians passthrough mismatch (da/dv): max diff = {max_diff_v}"
        );
    }
}
