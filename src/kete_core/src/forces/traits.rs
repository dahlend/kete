//! The force trait: stateless physics pieces that contribute acceleration.
//!
//! [`ParameterizedForce`] is a family of forces parameterized by an `&[f64]` slice.
//! Fitted parameters are passed at every method call; fixed physical constants live as
//! struct fields. It covers both forces with zero free parameters (gravity, or a fully
//! fixed [`ParameterMask`](super::ParameterMask)) and parameterized non-gravitational
//! templates (e.g. `DustNonGrav`, `JplCometNonGrav`).
//!
//! All forces speak AU/day at the API: positions in AU, velocities in
//! AU/day, time in days (TDB), accelerations in AU/day^2.
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

use nalgebra::{Matrix3, Matrix3xX, Vector3};

use crate::errors::KeteResult;
use crate::frames::{CenterBody, InertialFrame, Vector};
use crate::time::{TDB, Time};

/// A single piece of physics contributing acceleration to a body.
///
/// Forces are stateless templates. Fitted parameters are passed to every
/// method as a slice; fixed physical constants live as struct fields.
///
/// `Send + Sync` is required because forces cross thread boundaries
/// during parallel batch propagation.
pub trait ParameterizedForce: Send + Sync {
    /// Inertial frame the force expects positions and velocities in.
    type Frame: InertialFrame;

    /// Center body the position and velocity passed to `accel` are
    /// measured relative to.
    type Center: CenterBody;

    /// Per-integration working storage, passed to every evaluation method.
    ///
    /// An integrator creates one with `Default` for each integration and hands it to
    /// every evaluation, so a force can keep quantities that depend only on time, such as
    /// ephemeris states, across the repeated evaluations at the same time within a step.
    /// Forces with nothing to keep use `()`. A force that wraps another carries the
    /// inner force's `Meta` in its own and passes it on.
    ///
    /// The returned values must not depend on what `meta` holds: a force computes the
    /// same result from an empty one.
    type Meta: Default;

    /// Number of free (fittable) parameters this force exposes.
    fn n_free_params(&self) -> usize {
        0
    }

    /// Names of free parameters in the order they appear in `&[f64]`
    /// arguments. Length must equal `n_free_params()`.
    fn free_param_names(&self) -> Vec<&'static str> {
        Vec::new()
    }

    /// Physical lower bounds for the free parameters, in the same order as
    /// `free_param_names`. Length must equal `n_free_params()`.
    ///
    /// `None` means unbounded below; `Some(v)` means the parameter is
    /// constrained to `p >= v`. Forces declare bounds that reflect physical
    /// reality (e.g. radiation pressure coefficients cannot be negative).
    /// Fitters use these to prevent unphysical steps.
    fn lower_bounds(&self) -> Vec<Option<f64>> {
        vec![None; self.n_free_params()]
    }

    /// Acceleration in AU/day^2.
    ///
    /// Position in AU, velocity in AU/day, time as TDB Julian date,
    /// `free_params` length must equal `n_free_params()`. Implementations
    /// must return `Err` rather than silently fall back when an external
    /// dependency (SPK lookup, table interpolation, etc.) fails;
    /// propagation correctness depends on errors surfacing immediately.
    ///
    /// `meta` is the integration's working storage (see [`Self::Meta`]). `exact_eval` is
    /// true when the state is one the integrator has accepted - the start of the
    /// integration and the end of each accepted step - and false for trial evaluations
    /// within a step and for derivative probes. A force may act on accepted states only,
    /// for example by returning an error for an impact, but the acceleration it returns
    /// must not depend on `exact_eval`.
    ///
    /// A single evaluation outside an integration passes `&mut Default::default()` and
    /// `false`.
    ///
    /// # Errors
    /// Implementations propagate any errors from external lookups or
    /// numerical routines they depend on.
    fn accel(
        &self,
        time: Time<TDB>,
        pos: &Vector<Self::Frame>,
        vel: &Vector<Self::Frame>,
        free_params: &[f64],
        meta: &mut Self::Meta,
        exact_eval: bool,
    ) -> KeteResult<Vector<Self::Frame>>;

    /// Position and velocity derivatives of acceleration:
    /// `(d(accel)/d(pos), d(accel)/d(vel))`, each 3x3.
    ///
    /// Combining the two derivatives into one method lets implementations
    /// that derive both from a shared analysis (the N-body Jacobian, the
    /// non-grav models) compute them in one pass instead of two.
    ///
    /// Default: forward finite differences for **both** derivatives. Forces with cheap
    /// analytical forms should override, and the ones that matter for performance do -
    /// the N-body gravity and [`DustNonGrav`](super::DustNonGrav).
    ///
    /// A force with no velocity dependence pays only the extra evaluations: its
    /// acceleration is bit-identical at the perturbed velocities, so the difference is
    /// exactly zero rather than merely small.
    ///
    /// Differencing both blocks rather than assuming the velocity block is zero costs
    /// three extra evaluations on forces that do not need it, and is correct on the ones
    /// that do. Assuming zero is a silent wrong answer for any velocity-dependent force
    /// that does not override, and it reaches every variational propagation and every
    /// fitted covariance built with that force. A slower default is preferable to a
    /// default that is sometimes wrong.
    ///
    /// # Errors
    /// Forwards errors from the underlying [`accel`](Self::accel) calls.
    fn jacobians(
        &self,
        time: Time<TDB>,
        pos: &Vector<Self::Frame>,
        vel: &Vector<Self::Frame>,
        free_params: &[f64],
        meta: &mut Self::Meta,
    ) -> KeteResult<(Matrix3<f64>, Matrix3<f64>)> {
        let base: Vector3<f64> = self.accel(time, pos, vel, free_params, meta, false)?.into();
        let pos_raw: Vector3<f64> = (*pos).into();
        let vel_raw: Vector3<f64> = (*vel).into();
        let mut da_dr = Matrix3::<f64>::zeros();
        let mut da_dv = Matrix3::<f64>::zeros();
        for axis in 0..3 {
            let h = fd_step(pos_raw[axis]);
            let mut perturbed = pos_raw;
            perturbed[axis] += h;
            let perturbed_vec = Vector::<Self::Frame>::new(perturbed.into());
            let a: Vector3<f64> = self
                .accel(time, &perturbed_vec, vel, free_params, meta, false)?
                .into();
            let col = (a - base) / h;
            da_dr[(0, axis)] = col[0];
            da_dr[(1, axis)] = col[1];
            da_dr[(2, axis)] = col[2];

            let h = fd_step(vel_raw[axis]);
            let mut perturbed = vel_raw;
            perturbed[axis] += h;
            let perturbed_vec = Vector::<Self::Frame>::new(perturbed.into());
            let a: Vector3<f64> = self
                .accel(time, pos, &perturbed_vec, free_params, meta, false)?
                .into();
            let col = (a - base) / h;
            da_dv[(0, axis)] = col[0];
            da_dv[(1, axis)] = col[1];
            da_dv[(2, axis)] = col[2];
        }
        Ok((da_dr, da_dv))
    }

    /// Acceleration and all of its derivatives at one point:
    /// `(accel, d(accel)/d(pos), d(accel)/d(vel), d(accel)/d(free_params))`.
    ///
    /// This is what variational propagation evaluates at every step. The default calls
    /// [`accel`](Self::accel), [`jacobians`](Self::jacobians) and
    /// [`parameter_jacobian`](Self::parameter_jacobian) in turn, so it agrees with them
    /// by construction. A force whose three methods share expensive work - an ephemeris
    /// lookup per massive body, a sum over the facets of a shape model - should override
    /// it to do that work once. `meta` and `exact_eval` are as for
    /// [`accel`](Self::accel).
    ///
    /// # Errors
    /// Forwards errors from the underlying calls.
    #[allow(clippy::type_complexity, reason = "a flat tuple of the four results")]
    fn accel_and_jacobians(
        &self,
        time: Time<TDB>,
        pos: &Vector<Self::Frame>,
        vel: &Vector<Self::Frame>,
        free_params: &[f64],
        meta: &mut Self::Meta,
        exact_eval: bool,
    ) -> KeteResult<(
        Vector<Self::Frame>,
        Matrix3<f64>,
        Matrix3<f64>,
        Matrix3xX<f64>,
    )> {
        let accel = self.accel(time, pos, vel, free_params, meta, exact_eval)?;
        let (da_dr, da_dv) = self.jacobians(time, pos, vel, free_params, meta)?;
        let da_dp = self.parameter_jacobian(time, pos, vel, free_params, meta)?;
        Ok((accel, da_dr, da_dv, da_dp))
    }

    /// Parameter derivative of acceleration: `d(accel)/d(free_params)`,
    /// shape 3 x `n_free_params`.
    ///
    /// Default: forward finite differences perturbing each parameter
    /// slot. Forces with cheap analytical derivatives should override.
    /// Returns a 3x0 matrix when `n_free_params() == 0`.
    ///
    /// # Errors
    /// Forwards errors from the underlying [`accel`](Self::accel) calls.
    fn parameter_jacobian(
        &self,
        time: Time<TDB>,
        pos: &Vector<Self::Frame>,
        vel: &Vector<Self::Frame>,
        free_params: &[f64],
        meta: &mut Self::Meta,
    ) -> KeteResult<Matrix3xX<f64>> {
        let n = self.n_free_params();
        let mut out = Matrix3xX::<f64>::zeros(n);
        if n == 0 {
            return Ok(out);
        }
        let base: Vector3<f64> = self.accel(time, pos, vel, free_params, meta, false)?.into();
        let mut perturbed_params = free_params.to_vec();
        for slot in 0..n {
            let original = perturbed_params[slot];
            let h = fd_step(original);
            perturbed_params[slot] = original + h;
            let a: Vector3<f64> = self
                .accel(time, pos, vel, &perturbed_params, meta, false)?
                .into();
            perturbed_params[slot] = original;
            let col = (a - base) / h;
            out[(0, slot)] = col[0];
            out[(1, slot)] = col[1];
            out[(2, slot)] = col[2];
        }
        Ok(out)
    }
}

/// Pick an FD step: `sqrt(eps) * max(|value|, 1e-3)`.
///
/// The 1e-3 floor keeps the step proportional to the parameter for values
/// above that scale, and falls back to an absolute step of ~1.5e-11 for
/// near-zero parameters. Forces whose parameters are naturally much smaller
/// than 1e-3 (e.g. sub-pico-AU/day non-grav terms) should override
/// `parameter_jacobian` with an analytical form.
fn fd_step(value: f64) -> f64 {
    let scale = value.abs().max(1e-3);
    f64::EPSILON.sqrt() * scale
}
