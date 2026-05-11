//! [`Recenter`]: a `ParameterizedForce` adapter that shifts the reference body of
//! the input pos/vel before delegating to an inner `ParameterizedForce`.
//!
//! ## Why this exists
//!
//! Forces are written in the natural reference frame for their physics:
//! gravity from N bodies in SSB-relative coordinates, JPL non-grav and
//! dust SRP in Sun-relative coordinates, atmospheric drag in
//! Earth-relative coordinates, polyhedral asteroid gravity in
//! body-relative coordinates. A [`ForceSet`](kete_core::forces::ForceSet)
//! requires every member to share `Center`, which would be impossible
//! without an adapter that bridges between centers.
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
//! // Sun-relative dust SRP wrapped to participate in an SSB-centered ForceSet.
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

use std::marker::PhantomData;
use std::sync::Mutex;

use kete_core::errors::KeteResult;
use kete_core::forces::{Force, ParameterizedForce};
use kete_core::frames::{NaifBody, Vector};
use kete_core::time::{TDB, Time};
use nalgebra::{Matrix3, Matrix3xX, Vector3};

use crate::spk::SpkCollection;

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
    F::Center: NaifBody,
    FromC: NaifBody,
{
    /// Borrowed SPK reference. Same lifetime contract as [`SpkNBody`]:
    /// the caller holds the read guard for the duration of propagation.
    ///
    /// [`SpkNBody`]: super::spk_n_body::SpkNBody
    pub spk: &'a SpkCollection,
    /// The inner force, written for `Center = F::Center`.
    pub inner: F,
    /// Per-step shift cache. Holds `(time.jd, shift_pos, shift_vel)` for
    /// the most recent SPK lookup. The variational integrator calls
    /// `accel + jacobian_pos + jacobian_vel + parameter_jacobian` at the
    /// same `time` within a substage (FD jacobians multiply this by 4-7
    /// accel calls); caching the shift turns those repeated SPK lookups
    /// into a single lookup per substage. Without this, a Recenter
    /// wrapping an FD-jacobian force pays an SPK query on every call.
    /// Same `Mutex`-only-for-Send+Sync pattern as [`SpkNBody`].
    ///
    /// [`SpkNBody`]: super::spk_n_body::SpkNBody
    cache: Mutex<Option<(f64, Vector3<f64>, Vector3<f64>)>>,
    _phantom: PhantomData<FromC>,
}

impl<'a, FromC, F> Recenter<'a, FromC, F>
where
    F: ParameterizedForce,
    F::Center: NaifBody,
    FromC: NaifBody,
{
    /// Wrap `inner` so it accepts pos/vel relative to `FromC`. The
    /// inner force's `Center` is determined by its impl.
    #[must_use]
    pub fn new(spk: &'a SpkCollection, inner: F) -> Self {
        Self {
            spk,
            inner,
            cache: Mutex::new(None),
            _phantom: PhantomData,
        }
    }
}

impl<FromC, F> std::fmt::Debug for Recenter<'_, FromC, F>
where
    F: ParameterizedForce + std::fmt::Debug,
    F::Center: NaifBody,
    FromC: NaifBody,
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
    F::Center: NaifBody,
    FromC: NaifBody,
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
    F::Center: NaifBody,
    FromC: NaifBody,
{
}

impl<FromC, F> Recenter<'_, FromC, F>
where
    F: ParameterizedForce,
    F::Center: NaifBody,
    FromC: NaifBody,
{
    /// Look up the `(F::Center - FromC)` offset at `time` from SPK and
    /// subtract it from the input `(pos, vel)`. Returns the
    /// `F::Center`-relative coordinates the inner force expects.
    ///
    /// The SPK lookup is cached by `time.jd` -- repeated calls within a
    /// single integrator substage (typical FD-jacobian fan-out: accel +
    /// 3 perturbed accels for `jacobian_pos`, plus more for
    /// `parameter_jacobian`) reuse the cached shift instead of hitting
    /// SPK each time.
    fn shift(
        &self,
        time: Time<TDB>,
        pos: &Vector<F::Frame>,
        vel: &Vector<F::Frame>,
    ) -> KeteResult<(Vector<F::Frame>, Vector<F::Frame>)> {
        let pos_v: Vector3<f64> = (*pos).into();
        let vel_v: Vector3<f64> = (*vel).into();
        let mut guard = self.cache.lock().expect("Recenter cache lock poisoned");
        let (shift_pos, shift_vel) = if let Some((cached_jd, cp, cv)) = guard.as_ref()
            && *cached_jd == time.jd
        {
            (*cp, *cv)
        } else {
            let shift = self.spk.try_get_state_with_center::<F::Frame>(
                F::Center::NAIF_ID,
                time,
                FromC::NAIF_ID,
            )?;
            let sp: Vector3<f64> = shift.pos.into();
            let sv: Vector3<f64> = shift.vel.into();
            *guard = Some((time.jd, sp, sv));
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
    use kete_core::forces::DustNonGrav;
    use kete_core::frames::{Equatorial, SSB};
    use kete_core::state::State;
    use std::sync::Arc;

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

    /// End-to-end: composed `ForceSet` of gravity-only `SpkNBody` plus
    /// Two equivalent JPL Comet routings via `Recenter<SSB, _>` produce
    /// the same propagation result: one wrapping `JplCometNonGrav` as an
    /// `Arc<dyn ParameterizedForce>` (the polymorphic API), the other
    /// wrapping it monomorphized directly. Validates parity between the
    /// two construction paths.
    #[test]
    fn composed_jpl_comet_via_nongrav_model_matches_standalone_force() {
        use kete_core::forces::{ForceSet, GravParams, JplCometNonGrav, ParameterizedForce};
        use kete_core::frames::SunCenter;
        use kete_core::state::propagate_with_stm;

        use crate::propagation::SpkNBody;

        crate::test_data::ensure_test_spk();
        let spk_guard = LOADED_SPK.try_read().unwrap();
        let planets = GravParams::planets();

        let pos_init = Vector3::new(0.5, 1.0, 0.1);
        let vel_init = Vector3::new(-0.012, 0.008, 0.001);
        let epoch = Time::<TDB>::new(2_451_545.0);
        let target = Time::<TDB>::new(2_451_545.0 + 5.0);

        let a1 = 1.0e-8;
        let a2 = 2.0e-9;
        let a3 = -3.0e-10;

        // Path A: ForceSet with the typed JplCometNonGrav wrapped in
        // an `Arc<dyn ParameterizedForce>` (the polymorphic API).
        let template_a: Arc<dyn ParameterizedForce<Frame = Equatorial, Center = SunCenter>> =
            Arc::new(JplCometNonGrav::standard_comet());
        let force_a: ForceSet<'_, Equatorial, SSB> = ForceSet::new()
            .with(Box::new(SpkNBody::new(&spk_guard, &planets)))
            .with(Box::new(Recenter::<SSB, _>::new(&spk_guard, template_a)));
        let (pos_a, vel_a, sens_a) =
            propagate_with_stm(&force_a, pos_init, vel_init, &[a1, a2, a3], epoch, target).unwrap();

        // Path B: ForceSet with the typed JplCometNonGrav passed
        // directly (monomorphized). Should match Path A bit-for-bit.
        let force_b: ForceSet<'_, Equatorial, SSB> = ForceSet::new()
            .with(Box::new(SpkNBody::new(&spk_guard, &planets)))
            .with(Box::new(Recenter::<SSB, _>::new(
                &spk_guard,
                JplCometNonGrav::standard_comet(),
            )));
        let (pos_b, vel_b, sens_b) =
            propagate_with_stm(&force_b, pos_init, vel_init, &[a1, a2, a3], epoch, target).unwrap();

        assert!(
            (pos_a - pos_b).norm() < 1e-12,
            "pos mismatch: enum={pos_a:?}, standalone={pos_b:?}"
        );
        assert!(
            (vel_a - vel_b).norm() < 1e-12,
            "vel mismatch: enum={vel_a:?}, standalone={vel_b:?}"
        );

        // STM tolerance reflects the jacobian-implementation gap: the
        // monolithic SpkNBody path uses analytical-via-internal-FD
        // partials (`nongrav_param_partials` for JplComet RTN-frame
        // basis); the composed path goes through `JplCometNonGrav`'s
        // default forward-FD jacobian. Both are FD-quality, so the STM
        // matches to FD-precision-squared accumulated over the arc,
        // not to round-off. Pos/vel match exactly because both paths
        // call the same `accel` math.
        let stm_diff = (&sens_a - &sens_b).abs().max();
        assert!(
            stm_diff < 1e-3,
            "composed STM mismatch: max element diff = {stm_diff} \
             (expected ~1e-5 from FD-vs-FD jacobian noise)"
        );
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

        // Through wrapper: feed SSB-relative pos/vel.
        let wrapped = Recenter::<SSB, _>::new(&spk_guard, DustNonGrav);
        let (j_pos_wrapped, j_vel_wrapped) = wrapped
            .jacobians(time, &pos_ssb, &vel_ssb, &[beta])
            .unwrap();

        // Direct: shift to Sun-relative manually, call inner.
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
