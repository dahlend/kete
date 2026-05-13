//! Force traits: stateless physics pieces that contribute acceleration.
//!
//! Two traits express the bound-vs-parameterized distinction at the type level:
//!
//! - [`ParameterizedForce`]: a family of forces parameterized by an `&[f64]`
//!   slice. Every concrete force impl implements this trait. Fitted parameters
//!   are passed at every method call; fixed physical constants live as struct
//!   fields. Includes both pure-physics impls with zero free parameters (e.g.
//!   `SpkNBody`) and parameterized non-gravitational templates (e.g.
//!   `DustNonGrav`, `JplCometNonGrav`).
//!
//! - [`Force`]: a marker subtrait of [`ParameterizedForce`] asserting that
//!   `n_free_params() == 0`. Implementors are *exactly defined* -- accel/jacobian
//!   queries need no extra parameters. Used wherever a single concrete force is
//!   required: plain `State` propagation, batch propagation, and any function
//!   that accepts only fully-bound forces.
//!
//! Bound force types (`SpkNBody`, [`FrozenForce`](super::FrozenForce)) implement
//! both traits. Parameterized templates (`DustNonGrav`, [`ParameterMask`](super::ParameterMask))
//! implement only [`ParameterizedForce`].
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
use crate::frames::{InertialFrame, Vector};
use crate::time::{TDB, Time};

/// A single piece of physics contributing acceleration to a body.
///
/// Forces are stateless templates. Fitted parameters are passed to every
/// method as a slice; fixed physical constants live as struct fields.
///
/// `Send + Sync` is required because `Vec<Box<dyn ParameterizedForce<...>>>`
/// instances cross thread boundaries during parallel batch propagation.
///
/// `Center` is a marker type identifying which center body the position
/// and velocity passed to `accel` are measured relative to. It is not
/// required to implement `CenterBody` on the trait (doing so would
/// break dyn-compatibility because `CenterBody` carries a `DynCenter:
/// From<Self>` bound). State/force compatibility is enforced at the
/// `StateLike::propagate_with` boundary, not on the trait itself.
pub trait ParameterizedForce: Send + Sync {
    /// Inertial frame the force expects positions and velocities in.
    type Frame: InertialFrame;

    /// Center body marker. Should be one of the kete center types
    /// (`SSB`, `SunCenter`, `EarthCenter`, `DynCenter`).
    type Center: Send + Sync + Copy + 'static;

    /// Optional downcast handle for runtime variant introspection.
    ///
    /// Default returns `None`; concrete `'static` non-grav impls
    /// (`DustNonGrav`, `JplCometNonGrav`, `FarnocchiaNonGrav`) override
    /// to return `Some(self)`. Allows the Python wrapper to recover the
    /// typed force from an `Arc<dyn ParameterizedForce<...>>`.
    /// Lifetime-borrowed impls (`SpkNBody<'a>`, `Recenter<'a, ...>`)
    /// cannot be `Any` and keep the `None` default.
    fn as_any(&self) -> Option<&(dyn std::any::Any + 'static)> {
        None
    }

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
    /// # Errors
    /// Implementations propagate any errors from external lookups or
    /// numerical routines they depend on.
    fn accel(
        &self,
        time: Time<TDB>,
        pos: &Vector<Self::Frame>,
        vel: &Vector<Self::Frame>,
        free_params: &[f64],
    ) -> KeteResult<Vector<Self::Frame>>;

    /// Position and velocity derivatives of acceleration:
    /// `(d(accel)/d(pos), d(accel)/d(vel))`, each 3x3.
    ///
    /// Combining the two derivatives into one method lets implementations
    /// that derive both from a shared analysis (the N-body Jacobian, the
    /// non-grav models) compute them in one pass instead of two.
    ///
    /// Default: forward finite differences for `d(accel)/d(pos)` and zero
    /// for `d(accel)/d(vel)`. Most gravitational forces have no velocity
    /// dependence; forces that do (GR correction, drag, certain non-grav
    /// models) override.
    ///
    /// # Errors
    /// Forwards errors from the underlying [`accel`](Self::accel) calls.
    fn jacobians(
        &self,
        time: Time<TDB>,
        pos: &Vector<Self::Frame>,
        vel: &Vector<Self::Frame>,
        free_params: &[f64],
    ) -> KeteResult<(Matrix3<f64>, Matrix3<f64>)> {
        let base: Vector3<f64> = self.accel(time, pos, vel, free_params)?.into();
        let pos_raw: Vector3<f64> = (*pos).into();
        let mut da_dr = Matrix3::<f64>::zeros();
        for axis in 0..3 {
            let h = fd_step(pos_raw[axis]);
            let mut perturbed = pos_raw;
            perturbed[axis] += h;
            let perturbed_vec = Vector::<Self::Frame>::new(perturbed.into());
            let a: Vector3<f64> = self.accel(time, &perturbed_vec, vel, free_params)?.into();
            let col = (a - base) / h;
            da_dr[(0, axis)] = col[0];
            da_dr[(1, axis)] = col[1];
            da_dr[(2, axis)] = col[2];
        }
        Ok((da_dr, Matrix3::zeros()))
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
    ) -> KeteResult<Matrix3xX<f64>> {
        let n = self.n_free_params();
        let mut out = Matrix3xX::<f64>::zeros(n);
        if n == 0 {
            return Ok(out);
        }
        let base: Vector3<f64> = self.accel(time, pos, vel, free_params)?.into();
        let mut perturbed_params = free_params.to_vec();
        for slot in 0..n {
            let original = perturbed_params[slot];
            let h = fd_step(original);
            perturbed_params[slot] = original + h;
            let a: Vector3<f64> = self.accel(time, pos, vel, &perturbed_params)?.into();
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

/// Marker subtrait: a [`ParameterizedForce`] with no free parameters.
///
/// Implementors guarantee `n_free_params() == 0` at the type level. This is
/// the trait `state.propagate_with` and other "exactly defined" propagation
/// paths require, and the trait [`FrozenForce`](super::FrozenForce) lifts a
/// parameterized template into.
///
/// `Force` adds no methods -- it is purely a type-level promise. Implementors
/// add `impl Force for X {}` alongside their `ParameterizedForce` impl when
/// their `n_free_params()` is structurally zero.
pub trait Force: ParameterizedForce {}

// Blanket `ParameterizedForce` impls for shared/owned smart pointers around a
// trait object. These let `Arc<dyn ParameterizedForce<...>>` and
// `Box<dyn ParameterizedForce<...>>` be passed as a force to generic APIs
// such as `ForceSet::with(Box<dyn ParameterizedForce>)`,
// `Recenter::<FromC, _>::new(spk, force)`, or `propagate_with_stm(forces, ...)`.
impl<F> ParameterizedForce for std::sync::Arc<F>
where
    F: ParameterizedForce + ?Sized,
{
    type Frame = F::Frame;
    type Center = F::Center;

    fn as_any(&self) -> Option<&(dyn std::any::Any + 'static)> {
        (**self).as_any()
    }

    fn n_free_params(&self) -> usize {
        (**self).n_free_params()
    }

    fn free_param_names(&self) -> Vec<&'static str> {
        (**self).free_param_names()
    }

    fn lower_bounds(&self) -> Vec<Option<f64>> {
        (**self).lower_bounds()
    }

    fn accel(
        &self,
        time: Time<TDB>,
        pos: &Vector<Self::Frame>,
        vel: &Vector<Self::Frame>,
        free_params: &[f64],
    ) -> KeteResult<Vector<Self::Frame>> {
        (**self).accel(time, pos, vel, free_params)
    }

    fn jacobians(
        &self,
        time: Time<TDB>,
        pos: &Vector<Self::Frame>,
        vel: &Vector<Self::Frame>,
        free_params: &[f64],
    ) -> KeteResult<(Matrix3<f64>, Matrix3<f64>)> {
        (**self).jacobians(time, pos, vel, free_params)
    }

    fn parameter_jacobian(
        &self,
        time: Time<TDB>,
        pos: &Vector<Self::Frame>,
        vel: &Vector<Self::Frame>,
        free_params: &[f64],
    ) -> KeteResult<Matrix3xX<f64>> {
        (**self).parameter_jacobian(time, pos, vel, free_params)
    }
}

/// Blanket marker: `Arc<F>` is `Force` iff `F` is.
impl<F> Force for std::sync::Arc<F> where F: Force + ?Sized {}
