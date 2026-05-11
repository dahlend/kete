//! `ForceSet`: aggregator that composes multiple `ParameterizedForce` impls.
//!
//! A `ForceSet` is itself a `ParameterizedForce`, so composition is recursive. Free
//! parameters concatenate across forces in push order, and the global
//! `&[f64]` slice is sliced per-force using `n_free_params()` as the stride.
//!
//! # Parameter ordering contract
//!
//! The layout of the combined `free_params` slice is determined by the order
//! forces were added via [`ForceSet::with`]: the first force's
//! `n_free_params()` elements come first, then the second force's, and so on.
//! This ordering is a semantic contract. An `UncertainState`'s `free_params`
//! and covariance rows/columns are sized and indexed to match a specific
//! `ForceSet` layout. Reconstructing a `ForceSet` with forces in a different
//! order and using it with a stored `UncertainState` will silently invert the
//! covariance interpretation for those parameters.
//!
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

use crate::errors::{Error, KeteResult};
use crate::forces::traits::ParameterizedForce;
use crate::frames::{InertialFrame, Vector};
use crate::time::{TDB, Time};

/// A composable collection of `ParameterizedForce` impls sharing a frame and center.
///
/// Forces are stored as `Box<dyn ParameterizedForce<...> + 'a>` so the set can mix
/// concrete [`ParameterizedForce`] types and hold borrowed-lifetime forces (e.g.,
/// `SpkNBody<'a>` borrowing a SPK read guard, or `Recenter<'a, ...>`).
/// The set is itself a `ParameterizedForce`, so a `ForceSet` can be nested inside
/// another. Use `'static` when all forces own their data.
pub struct ForceSet<'a, F: InertialFrame, C: Send + Sync + Copy + 'static> {
    /// Forces contributing to the total acceleration. Free parameters
    /// concatenate in iteration order.
    pub forces: Vec<Box<dyn ParameterizedForce<Frame = F, Center = C> + 'a>>,
}

impl<F: InertialFrame, C: Send + Sync + Copy + 'static> std::fmt::Debug for ForceSet<'_, F, C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ForceSet")
            .field("n_forces", &self.forces.len())
            .field(
                "n_free_params",
                &<Self as ParameterizedForce>::n_free_params(self),
            )
            .finish()
    }
}

impl<F: InertialFrame, C: Send + Sync + Copy + 'static> Default for ForceSet<'_, F, C> {
    fn default() -> Self {
        Self { forces: Vec::new() }
    }
}

impl<'a, F: InertialFrame, C: Send + Sync + Copy + 'static> ForceSet<'a, F, C> {
    /// Build an empty force set.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a force to the set. Returns `self` so calls chain.
    #[must_use]
    pub fn with(mut self, force: Box<dyn ParameterizedForce<Frame = F, Center = C> + 'a>) -> Self {
        self.forces.push(force);
        self
    }
}

impl<F: InertialFrame, C: Send + Sync + Copy + 'static> ParameterizedForce for ForceSet<'_, F, C> {
    type Frame = F;
    type Center = C;

    fn n_free_params(&self) -> usize {
        self.forces.iter().map(|f| f.n_free_params()).sum()
    }

    fn free_param_names(&self) -> Vec<&'static str> {
        self.forces
            .iter()
            .flat_map(|f| f.free_param_names())
            .collect()
    }

    fn accel(
        &self,
        time: Time<TDB>,
        pos: &Vector<F>,
        vel: &Vector<F>,
        free_params: &[f64],
    ) -> KeteResult<Vector<F>> {
        check_params(free_params, self.n_free_params())?;
        let mut sum = Vector3::<f64>::zeros();
        let mut comp = Vector3::<f64>::zeros();
        let mut offset = 0;
        for force in &self.forces {
            let n = force.n_free_params();
            let slice = inner_slice(free_params, offset, n);
            let term: Vector3<f64> = force.accel(time, pos, vel, slice)?.into();
            neumaier_add(&mut sum, &mut comp, term);
            offset += n;
        }
        Ok(Vector::<F>::new((sum + comp).into()))
    }

    fn jacobians(
        &self,
        time: Time<TDB>,
        pos: &Vector<F>,
        vel: &Vector<F>,
        free_params: &[f64],
    ) -> KeteResult<(Matrix3<f64>, Matrix3<f64>)> {
        check_params(free_params, self.n_free_params())?;
        let mut sum_dr = Matrix3::<f64>::zeros();
        let mut comp_dr = Matrix3::<f64>::zeros();
        let mut sum_dv = Matrix3::<f64>::zeros();
        let mut comp_dv = Matrix3::<f64>::zeros();
        let mut offset = 0;
        for force in &self.forces {
            let n = force.n_free_params();
            let slice = inner_slice(free_params, offset, n);
            let (term_dr, term_dv) = force.jacobians(time, pos, vel, slice)?;
            neumaier_add_matrix(&mut sum_dr, &mut comp_dr, term_dr);
            neumaier_add_matrix(&mut sum_dv, &mut comp_dv, term_dv);
            offset += n;
        }
        Ok((sum_dr + comp_dr, sum_dv + comp_dv))
    }

    fn parameter_jacobian(
        &self,
        time: Time<TDB>,
        pos: &Vector<F>,
        vel: &Vector<F>,
        free_params: &[f64],
    ) -> KeteResult<Matrix3xX<f64>> {
        check_params(free_params, self.n_free_params())?;
        // Each force's parameter columns are disjoint: F_i's columns are
        // produced only by F_i. Concatenate the per-force 3 x n_i blocks
        // side by side. No cross-force summation needed.
        let total = self.n_free_params();
        let mut out = Matrix3xX::<f64>::zeros(total);
        let mut offset = 0;
        for force in &self.forces {
            let n = force.n_free_params();
            if n == 0 {
                continue;
            }
            let slice = inner_slice(free_params, offset, n);
            let block = force.parameter_jacobian(time, pos, vel, slice)?;
            out.columns_mut(offset, n).copy_from(&block);
            offset += n;
        }
        Ok(out)
    }
}

/// Validate that `free_params` has the correct length for this `ForceSet`.
///
/// Two valid call shapes are accepted:
/// - Empty slice: valid only when all forces have `n_free_params() == 0`
///   (pure gravity path). Each inner force receives `&[]`.
/// - Slice of length `expected`: sliced per-force in push order.
///
/// Any other length is a caller bug that would produce wrong physics
/// (silent wrong offset) or an opaque index panic. Return an error early.
#[inline]
fn check_params(free_params: &[f64], expected: usize) -> KeteResult<()> {
    if !free_params.is_empty() && free_params.len() != expected {
        return Err(Error::ValueError(format!(
            "ForceSet expects {} free parameters, got {}; \
             pass an empty slice to use stored values, or a slice of \
             exactly the right length for variational propagation",
            expected,
            free_params.len()
        )));
    }
    Ok(())
}

/// Slice of `free_params` to forward to a single inner force.
///
/// Precondition: `free_params` is empty or has exactly `self.n_free_params()`
/// elements (enforced by `check_params` at the top of each public method).
#[inline(always)]
fn inner_slice(free_params: &[f64], offset: usize, n: usize) -> &[f64] {
    if free_params.is_empty() {
        &[]
    } else {
        &free_params[offset..offset + n]
    }
}

/// Neumaier-compensated 3D vector addition: `sum += term`, with `comp`
/// accumulating cancellation error.
///
/// After the loop, the corrected sum is `sum + comp`.
fn neumaier_add(sum: &mut Vector3<f64>, comp: &mut Vector3<f64>, term: Vector3<f64>) {
    for axis in 0..3 {
        let s = sum[axis];
        let t = term[axis];
        let new_sum = s + t;
        let lost = if s.abs() >= t.abs() {
            (s - new_sum) + t
        } else {
            (t - new_sum) + s
        };
        sum[axis] = new_sum;
        comp[axis] += lost;
    }
}

/// Element-wise Neumaier compensated addition for 3x3 matrices.
fn neumaier_add_matrix(sum: &mut Matrix3<f64>, comp: &mut Matrix3<f64>, term: Matrix3<f64>) {
    for row in 0..3 {
        for col in 0..3 {
            let s = sum[(row, col)];
            let t = term[(row, col)];
            let new_sum = s + t;
            let lost = if s.abs() >= t.abs() {
                (s - new_sum) + t
            } else {
                (t - new_sum) + s
            };
            sum[(row, col)] = new_sum;
            comp[(row, col)] += lost;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frames::{Equatorial, SunCenter};

    /// Trivial constant force: returns a fixed acceleration regardless of input.
    /// Used for testing summation behavior.
    struct ConstantForce {
        accel: Vector3<f64>,
    }

    impl ParameterizedForce for ConstantForce {
        type Frame = Equatorial;
        type Center = SunCenter;
        fn accel(
            &self,
            _time: Time<TDB>,
            _pos: &Vector<Equatorial>,
            _vel: &Vector<Equatorial>,
            _free_params: &[f64],
        ) -> KeteResult<Vector<Equatorial>> {
            Ok(Vector::<Equatorial>::new(self.accel.into()))
        }
    }

    #[test]
    fn empty_force_set_returns_zero() {
        let set: ForceSet<'_, Equatorial, SunCenter> = ForceSet::new();
        let pos = Vector::<Equatorial>::new([1.0, 0.0, 0.0]);
        let vel = Vector::<Equatorial>::new([0.0, 1.0, 0.0]);
        let a = set.accel(Time::<TDB>::new(0.0), &pos, &vel, &[]).unwrap();
        assert_eq!(
            <Vector<Equatorial> as Into<Vector3<f64>>>::into(a),
            Vector3::zeros()
        );
        assert_eq!(set.n_free_params(), 0);
        assert!(set.free_param_names().is_empty());
    }

    #[test]
    fn force_set_sums_constant_forces() {
        let set: ForceSet<'_, Equatorial, SunCenter> = ForceSet::new()
            .with(Box::new(ConstantForce {
                accel: Vector3::new(1.0, 0.0, 0.0),
            }))
            .with(Box::new(ConstantForce {
                accel: Vector3::new(0.0, 2.0, 0.0),
            }))
            .with(Box::new(ConstantForce {
                accel: Vector3::new(0.0, 0.0, 3.0),
            }));
        let pos = Vector::<Equatorial>::new([1.0, 0.0, 0.0]);
        let vel = Vector::<Equatorial>::new([0.0, 1.0, 0.0]);
        let a: Vector3<f64> = set
            .accel(Time::<TDB>::new(0.0), &pos, &vel, &[])
            .unwrap()
            .into();
        assert_eq!(a, Vector3::new(1.0, 2.0, 3.0));
    }

    #[test]
    fn neumaier_recovers_precision_for_disparate_terms() {
        // Sum 1.0 with 1e6 small terms of size 1e-16: naive sum loses
        // them entirely; compensated sum recovers the small contribution.
        let large = 1.0;
        let small = 1.0e-16;
        let n = 1_000_000;

        let mut naive_sum = Vector3::new(large, 0.0, 0.0);
        for _ in 0..n {
            naive_sum += Vector3::new(small, 0.0, 0.0);
        }

        let mut sum = Vector3::new(large, 0.0, 0.0);
        let mut comp = Vector3::zeros();
        for _ in 0..n {
            neumaier_add(&mut sum, &mut comp, Vector3::new(small, 0.0, 0.0));
        }
        let compensated = sum + comp;

        let expected = large + f64::from(n) * small;
        // Naive sum loses precision; compensated stays close to expected.
        let naive_err = (naive_sum.x - expected).abs();
        let comp_err = (compensated.x - expected).abs();
        assert!(
            comp_err < naive_err * 1e-6,
            "comp_err={comp_err}, naive_err={naive_err}"
        );
    }
}
