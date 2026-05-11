//! [`ParameterMask`]: per-parameter freeze adapter for `ParameterizedForce` impls.
//!
//! Wraps an inner `ParameterizedForce` and replaces a subset of its free parameters
//! with fixed values. Frozen slots disappear from `n_free_params()`,
//! `free_param_names()`, and the `parameter_jacobian` columns; `accel`
//! and the dynamics Jacobians delegate to the inner force after merging
//! frozen and free parameters into the inner's parameter order.
//!
//! Two main use cases:
//! - **Variational template** (all-`None` mask): the wrapper exposes the same
//!   number of free parameters as the inner force, used as the parameterized
//!   template stored on an `UncertainState` or `DiffuseState`. Values come from
//!   the carrying state's `free_params` at integration time.
//! - **Partial freeze**: in orbit fitting, expose only a subset of
//!   parameters (e.g. JPL non-grav `A1` only, leaving `A2`/`A3` at fixed
//!   values). The fitter sees a smaller parameter space and a tighter
//!   covariance.
//!
//! For fully-frozen forces (all parameters baked in, used for plain `State`
//! propagation), use [`FrozenForce`](super::FrozenForce) instead.
//
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

use nalgebra::{Matrix3, Matrix3xX};

use crate::errors::{Error, KeteResult};
use crate::forces::ParameterizedForce;
use crate::frames::Vector;
use crate::time::{TDB, Time};

/// Per-parameter freeze adapter wrapping an inner [`ParameterizedForce`].
///
/// `mask[i]` corresponds to inner parameter `i` (in
/// `inner.free_param_names()` order):
/// - `Some(v)` -- frozen at `v`; absent from this adapter's free-parameter
///   surface.
/// - `None` -- exposed as one of this adapter's free parameters, in mask
///   order.
#[derive(Debug, Clone)]
pub struct ParameterMask<F: ParameterizedForce> {
    /// Wrapped force.
    pub inner: F,

    /// One entry per inner parameter; `Some(v)` freezes, `None` exposes.
    pub mask: Vec<Option<f64>>,
}

impl<F: ParameterizedForce> ParameterMask<F> {
    /// Build with an explicit per-parameter mask.
    ///
    /// # Errors
    /// Returns `ValueError` if `mask.len() != inner.n_free_params()`.
    pub fn new(inner: F, mask: Vec<Option<f64>>) -> KeteResult<Self> {
        if mask.len() != inner.n_free_params() {
            return Err(Error::ValueError(format!(
                "ParameterMask::new: mask length {} does not match inner.n_free_params() {}",
                mask.len(),
                inner.n_free_params()
            )));
        }
        Ok(Self { inner, mask })
    }

    /// Reconstruct the full inner parameter slice by interleaving frozen
    /// values (from the mask) with caller-provided free values.
    ///
    /// Public so callers reconstructing the canonical non-grav entry
    /// (e.g. the Python wrapper recovering variant identity through a
    /// frozen wrapper) can produce the full parameter vector that
    /// `inner` would see at `accel` time.
    ///
    /// # Errors
    /// Returns `ValueError` if `free_params.len() != self.n_free_params()`.
    pub fn merge(&self, free_params: &[f64]) -> KeteResult<Vec<f64>> {
        // Use the mask directly (not the trait method) so this method
        // is callable from inherent impl contexts that don't require
        // `F: 'static`.
        let expected = self.mask.iter().filter(|m| m.is_none()).count();
        if free_params.len() != expected {
            return Err(Error::ValueError(format!(
                "ParameterMask: expected {} free parameters, got {}",
                expected,
                free_params.len()
            )));
        }
        let mut full = Vec::with_capacity(self.mask.len());
        let mut idx = 0;
        for slot in &self.mask {
            if let Some(v) = slot {
                full.push(*v);
            } else {
                full.push(free_params[idx]);
                idx += 1;
            }
        }
        Ok(full)
    }
}

impl<F: ParameterizedForce + 'static> ParameterizedForce for ParameterMask<F> {
    type Frame = F::Frame;
    type Center = F::Center;

    fn n_free_params(&self) -> usize {
        self.mask.iter().filter(|m| m.is_none()).count()
    }

    fn free_param_names(&self) -> Vec<&'static str> {
        self.inner
            .free_param_names()
            .into_iter()
            .zip(&self.mask)
            .filter_map(|(name, slot)| slot.is_none().then_some(name))
            .collect()
    }

    fn accel(
        &self,
        time: Time<TDB>,
        pos: &Vector<F::Frame>,
        vel: &Vector<F::Frame>,
        free_params: &[f64],
    ) -> KeteResult<Vector<F::Frame>> {
        let full = self.merge(free_params)?;
        self.inner.accel(time, pos, vel, &full)
    }

    fn jacobians(
        &self,
        time: Time<TDB>,
        pos: &Vector<F::Frame>,
        vel: &Vector<F::Frame>,
        free_params: &[f64],
    ) -> KeteResult<(Matrix3<f64>, Matrix3<f64>)> {
        let full = self.merge(free_params)?;
        self.inner.jacobians(time, pos, vel, &full)
    }

    fn parameter_jacobian(
        &self,
        time: Time<TDB>,
        pos: &Vector<F::Frame>,
        vel: &Vector<F::Frame>,
        free_params: &[f64],
    ) -> KeteResult<Matrix3xX<f64>> {
        let full = self.merge(free_params)?;
        let inner_jac = self.inner.parameter_jacobian(time, pos, vel, &full)?;
        let n_free = self.n_free_params();
        let mut out = Matrix3xX::<f64>::zeros(n_free);
        let mut out_col = 0;
        for (i, slot) in self.mask.iter().enumerate() {
            if slot.is_none() {
                out.set_column(out_col, &inner_jac.column(i));
                out_col += 1;
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::forces::JplCometNonGrav;
    use crate::frames::{Equatorial, Vector};
    use nalgebra::Vector3;

    fn pos() -> Vector<Equatorial> {
        Vector::<Equatorial>::new([1.5, 0.3, 0.1])
    }
    fn vel() -> Vector<Equatorial> {
        Vector::<Equatorial>::new([-0.005, 0.012, 0.001])
    }
    fn epoch() -> Time<TDB> {
        Time::<TDB>::new(2_451_545.0)
    }

    #[test]
    fn partial_freeze_a1_only() {
        let inner = JplCometNonGrav::standard_comet();
        let mask = vec![None, Some(2.0e-9), Some(-3.0e-10)];
        let masked = ParameterMask::new(inner, mask).unwrap();
        assert_eq!(masked.n_free_params(), 1);
        assert_eq!(masked.free_param_names(), vec!["a1"]);
    }

    #[test]
    fn partial_freeze_accel_matches_full_call() {
        let inner = JplCometNonGrav::standard_comet();
        let mask = vec![None, Some(2.0e-9), Some(-3.0e-10)];
        let masked = ParameterMask::new(inner.clone(), mask).unwrap();

        let a_through: Vector3<f64> = masked
            .accel(epoch(), &pos(), &vel(), &[1.0e-8])
            .unwrap()
            .into();
        let a_direct: Vector3<f64> = inner
            .accel(epoch(), &pos(), &vel(), &[1.0e-8, 2.0e-9, -3.0e-10])
            .unwrap()
            .into();
        assert!((a_through - a_direct).norm() < 1e-15);
    }

    #[test]
    fn partial_freeze_parameter_jacobian_projects_columns() {
        let inner = JplCometNonGrav::standard_comet();
        let mask = vec![None, Some(2.0e-9), Some(-3.0e-10)];
        let masked = ParameterMask::new(inner.clone(), mask).unwrap();

        let inner_jac = inner
            .parameter_jacobian(epoch(), &pos(), &vel(), &[1.0e-8, 2.0e-9, -3.0e-10])
            .unwrap();
        let masked_jac = masked
            .parameter_jacobian(epoch(), &pos(), &vel(), &[1.0e-8])
            .unwrap();

        assert_eq!(masked_jac.ncols(), 1);
        // The single masked column equals inner column 0 (a1).
        for row in 0..3 {
            let diff = (masked_jac[(row, 0)] - inner_jac[(row, 0)]).abs();
            assert!(diff < 1e-15, "row {row} diff = {diff}");
        }
    }

    #[test]
    fn mask_length_mismatch_errors() {
        let res = ParameterMask::new(JplCometNonGrav::standard_comet(), vec![None, None]);
        assert!(res.is_err());
    }
}
