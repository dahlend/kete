//! [`ParameterMask`]: a `ParameterizedForce` with each parameter either fixed or free.
//!
//! Wraps an inner `ParameterizedForce` and fixes a subset of its parameters at given
//! values. Fixed slots disappear from `n_free_params()`, `free_param_names()`,
//! `lower_bounds()`, and the `parameter_jacobian` columns; `accel` and the dynamics
//! Jacobians delegate to the inner force after merging fixed and free values into the
//! inner's parameter order.
//!
//! The three cases it covers:
//! - **All free** ([`ParameterMask::all_free`]): the same free parameters as the inner
//!   force. This is the parameterized template stored on an `UncertainState` or
//!   `DiffuseState`; values come from the carrying state's `free_params`.
//! - **Partly fixed**: in orbit fitting, expose only a subset of parameters (e.g. JPL
//!   non-grav `A2` only, leaving `A1`/`A3` at fixed values).
//! - **All fixed** ([`ParameterMask::all_fixed`], [`ParameterMask::fixed_at`]): no free
//!   parameters, which is what propagating a plain `State` requires.
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

use std::borrow::Cow;

use nalgebra::{Matrix3, Matrix3xX};

use crate::errors::{Error, KeteResult};
use crate::forces::ParameterizedForce;
use crate::frames::Vector;
use crate::time::{TDB, Time};

/// An inner [`ParameterizedForce`] with each of its parameters either fixed at a value or
/// left free.
///
/// Free parameters keep the inner force's order.
#[derive(Debug, Clone)]
pub struct ParameterMask<F: ParameterizedForce> {
    /// Wrapped force.
    inner: F,

    /// One entry per inner parameter: the value of a fixed slot, unused for a free one.
    values: Vec<f64>,

    /// Indices of the free slots, ascending.
    free: Vec<usize>,
}

impl<F: ParameterizedForce> ParameterMask<F> {
    /// Build from one entry per inner parameter, in `inner.free_param_names()` order:
    /// `Some(v)` fixes the parameter at `v`, `None` leaves it free.
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
        let free = (0..mask.len()).filter(|&i| mask[i].is_none()).collect();
        let values = mask.into_iter().map(|slot| slot.unwrap_or(0.0)).collect();
        Ok(Self {
            inner,
            values,
            free,
        })
    }

    /// Every parameter of `inner` left free.
    pub fn all_free(inner: F) -> Self {
        let n = inner.n_free_params();
        Self {
            inner,
            values: vec![0.0; n],
            free: (0..n).collect(),
        }
    }

    /// Every parameter of `inner` fixed, at `values` in the inner force's order.
    ///
    /// # Errors
    /// Returns `ValueError` if `values.len() != inner.n_free_params()`.
    pub fn all_fixed(inner: F, values: Vec<f64>) -> KeteResult<Self> {
        if values.len() != inner.n_free_params() {
            return Err(Error::ValueError(format!(
                "ParameterMask::all_fixed: {} values provided but inner force has {} parameters",
                values.len(),
                inner.n_free_params()
            )));
        }
        Ok(Self {
            inner,
            values,
            free: Vec::new(),
        })
    }

    /// The wrapped force.
    pub fn inner(&self) -> &F {
        &self.inner
    }

    /// Replace the wrapped force with one that has the same number of parameters,
    /// keeping which of them are fixed and at what values.
    ///
    /// # Errors
    /// Returns `ValueError` if `inner` has a different number of parameters.
    pub fn set_inner(&mut self, inner: F) -> KeteResult<()> {
        if inner.n_free_params() != self.values.len() {
            return Err(Error::ValueError(format!(
                "ParameterMask::set_inner: new force has {} parameters, expected {}",
                inner.n_free_params(),
                self.values.len()
            )));
        }
        self.inner = inner;
        Ok(())
    }

    /// One entry per inner parameter: `Some(v)` for a parameter fixed at `v`, `None` for
    /// a free one. This is what [`Self::new`] was given.
    pub fn mask(&self) -> Vec<Option<f64>> {
        let mut mask: Vec<Option<f64>> = self.values.iter().copied().map(Some).collect();
        for &i in &self.free {
            mask[i] = None;
        }
        mask
    }

    /// Whether every parameter is fixed, so that there are no free parameters.
    pub fn is_fixed(&self) -> bool {
        self.free.is_empty()
    }

    /// Values of all inner parameters, when every one of them is fixed.
    ///
    /// # Errors
    /// Returns `ValueError` if any parameter is free.
    pub fn fixed_values(&self) -> KeteResult<&[f64]> {
        if !self.is_fixed() {
            return Err(Error::ValueError(format!(
                "ParameterMask: {} of {} parameters are free, but all must be fixed here",
                self.free.len(),
                self.values.len()
            )));
        }
        Ok(&self.values)
    }

    /// The full inner parameter vector: fixed values interleaved with the
    /// caller-provided free values.
    ///
    /// # Errors
    /// Returns `ValueError` if `free_params.len() != self.n_free_params()`.
    pub fn merge(&self, free_params: &[f64]) -> KeteResult<Vec<f64>> {
        self.check_len(free_params)?;
        let mut full = self.values.clone();
        for (&i, &value) in self.free.iter().zip(free_params) {
            full[i] = value;
        }
        Ok(full)
    }

    /// As [`Self::merge`], without a copy in the two common cases: with everything fixed
    /// the stored values are the inner slice, and with nothing fixed the caller's slice is.
    ///
    /// Inlined, with [`Self::check_len`]: they sit on the per-step path of every
    /// propagation with a non-grav, where the `N-Body/Frozen-Dust` benchmark shows the
    /// call overhead.
    #[inline]
    fn full_params<'a>(&'a self, free_params: &'a [f64]) -> KeteResult<Cow<'a, [f64]>> {
        self.check_len(free_params)?;
        if self.free.is_empty() {
            Ok(Cow::Borrowed(&self.values))
        } else if self.free.len() == self.values.len() {
            Ok(Cow::Borrowed(free_params))
        } else {
            self.merge(free_params).map(Cow::Owned)
        }
    }

    #[inline]
    fn check_len(&self, free_params: &[f64]) -> KeteResult<()> {
        if free_params.len() == self.free.len() {
            return Ok(());
        }
        Err(Error::ValueError(format!(
            "ParameterMask: expected {} free parameters, got {}",
            self.free.len(),
            free_params.len()
        )))
    }
}

impl<F: ParameterizedForce + Clone> ParameterMask<F> {
    /// The same force with every parameter fixed: the fixed slots as they are, the free
    /// slots at `free_values`.
    ///
    /// # Errors
    /// Returns `ValueError` if `free_values.len() != self.n_free_params()`.
    pub fn fixed_at(&self, free_values: &[f64]) -> KeteResult<Self> {
        Self::all_fixed(self.inner.clone(), self.merge(free_values)?)
    }
}

impl<F: ParameterizedForce> ParameterizedForce for ParameterMask<F> {
    type Frame = F::Frame;
    type Center = F::Center;
    type Meta = F::Meta;

    fn n_free_params(&self) -> usize {
        self.free.len()
    }

    fn free_param_names(&self) -> Vec<&'static str> {
        let names = self.inner.free_param_names();
        self.free.iter().map(|&i| names[i]).collect()
    }

    fn lower_bounds(&self) -> Vec<Option<f64>> {
        let bounds = self.inner.lower_bounds();
        self.free.iter().map(|&i| bounds[i]).collect()
    }

    fn accel(
        &self,
        time: Time<TDB>,
        pos: &Vector<F::Frame>,
        vel: &Vector<F::Frame>,
        free_params: &[f64],
        meta: &mut Self::Meta,
        exact_eval: bool,
    ) -> KeteResult<Vector<F::Frame>> {
        let full = self.full_params(free_params)?;
        self.inner.accel(time, pos, vel, &full, meta, exact_eval)
    }

    fn jacobians(
        &self,
        time: Time<TDB>,
        pos: &Vector<F::Frame>,
        vel: &Vector<F::Frame>,
        free_params: &[f64],
        meta: &mut Self::Meta,
    ) -> KeteResult<(Matrix3<f64>, Matrix3<f64>)> {
        let full = self.full_params(free_params)?;
        self.inner.jacobians(time, pos, vel, &full, meta)
    }

    fn parameter_jacobian(
        &self,
        time: Time<TDB>,
        pos: &Vector<F::Frame>,
        vel: &Vector<F::Frame>,
        free_params: &[f64],
        meta: &mut Self::Meta,
    ) -> KeteResult<Matrix3xX<f64>> {
        if self.free.is_empty() {
            self.check_len(free_params)?;
            return Ok(Matrix3xX::zeros(0));
        }
        let full = self.full_params(free_params)?;
        let inner_jac = self.inner.parameter_jacobian(time, pos, vel, &full, meta)?;
        if self.free.len() == self.values.len() {
            // Nothing fixed: every inner column is a free column.
            return Ok(inner_jac);
        }
        let mut out = Matrix3xX::<f64>::zeros(self.free.len());
        for (out_col, &i) in self.free.iter().enumerate() {
            out.set_column(out_col, &inner_jac.column(i));
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::forces::{JplCometNonGrav, NonGravKind};
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
            .accel(
                epoch(),
                &pos(),
                &vel(),
                &[1.0e-8],
                &mut Default::default(),
                false,
            )
            .unwrap()
            .into();
        let a_direct: Vector3<f64> = inner
            .accel(
                epoch(),
                &pos(),
                &vel(),
                &[1.0e-8, 2.0e-9, -3.0e-10],
                &mut Default::default(),
                false,
            )
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
            .parameter_jacobian(
                epoch(),
                &pos(),
                &vel(),
                &[1.0e-8, 2.0e-9, -3.0e-10],
                &mut Default::default(),
            )
            .unwrap();
        let masked_jac = masked
            .parameter_jacobian(epoch(), &pos(), &vel(), &[1.0e-8], &mut Default::default())
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

    /// A force whose three parameters each have a different lower bound.
    struct MixedBounds;

    impl ParameterizedForce for MixedBounds {
        type Frame = Equatorial;
        type Center = crate::frames::SunCenter;
        type Meta = ();

        fn n_free_params(&self) -> usize {
            3
        }

        fn free_param_names(&self) -> Vec<&'static str> {
            vec!["p0", "p1", "p2"]
        }

        fn lower_bounds(&self) -> Vec<Option<f64>> {
            vec![Some(0.0), None, Some(-1.0)]
        }

        fn accel(
            &self,
            _time: Time<TDB>,
            _pos: &Vector<Equatorial>,
            _vel: &Vector<Equatorial>,
            _free_params: &[f64],
            _meta: &mut Self::Meta,
            _exact_eval: bool,
        ) -> KeteResult<Vector<Equatorial>> {
            Ok(Vector::new([0.0; 3]))
        }
    }

    /// The mask's bounds line up with its free parameters, not with the inner force's.
    /// A fitter indexes bounds by free-parameter position, so with the first slot frozen
    /// position 0 must be `p1`'s bound and position 1 must be `p2`'s.
    #[test]
    fn partial_freeze_lower_bounds_follow_free_slots() {
        let masked = ParameterMask::new(MixedBounds, vec![Some(0.5), None, None]).unwrap();
        assert_eq!(masked.free_param_names(), vec!["p1", "p2"]);
        assert_eq!(masked.lower_bounds(), vec![None, Some(-1.0)]);

        let masked = ParameterMask::new(MixedBounds, vec![None, Some(0.5), None]).unwrap();
        assert_eq!(masked.lower_bounds(), vec![Some(0.0), Some(-1.0)]);
    }

    /// With every parameter fixed there are no free parameters: the stored values are
    /// used, and values passed by the caller are an error rather than ignored.
    #[test]
    fn all_fixed_has_no_free_parameters() {
        let inner = JplCometNonGrav::standard_comet();
        let values = vec![1.0e-8, 2.0e-9, -3.0e-10];
        let fixed = ParameterMask::all_fixed(inner.clone(), values.clone()).unwrap();
        assert!(fixed.is_fixed());
        assert_eq!(fixed.n_free_params(), 0);
        assert!(fixed.free_param_names().is_empty());
        assert_eq!(fixed.fixed_values().unwrap(), values.as_slice());

        let through: Vector3<f64> = fixed
            .accel(epoch(), &pos(), &vel(), &[], &mut Default::default(), false)
            .unwrap()
            .into();
        let direct: Vector3<f64> = inner
            .accel(
                epoch(),
                &pos(),
                &vel(),
                &values,
                &mut Default::default(),
                false,
            )
            .unwrap()
            .into();
        assert_eq!(through, direct);
        assert_eq!(
            fixed
                .parameter_jacobian(epoch(), &pos(), &vel(), &[], &mut Default::default())
                .unwrap()
                .ncols(),
            0
        );

        assert!(
            fixed
                .accel(
                    epoch(),
                    &pos(),
                    &vel(),
                    &[1.0],
                    &mut Default::default(),
                    false
                )
                .is_err()
        );
        assert!(ParameterMask::all_fixed(inner, vec![1.0]).is_err());
    }

    /// `fixed_at` keeps the fixed slots and fills the free ones, and `mask` reports what
    /// the mask was built from.
    #[test]
    fn fixed_at_fills_the_free_slots() {
        let inner = JplCometNonGrav::standard_comet();
        let mask = vec![Some(1.0e-8), None, Some(-3.0e-10)];
        let masked = ParameterMask::new(inner.clone(), mask.clone()).unwrap();
        assert_eq!(masked.mask(), mask);
        assert!(!masked.is_fixed());
        assert!(masked.fixed_values().is_err());

        let fixed = masked.fixed_at(&[2.0e-9]).unwrap();
        assert_eq!(fixed.fixed_values().unwrap(), &[1.0e-8, 2.0e-9, -3.0e-10]);
        assert!(masked.fixed_at(&[]).is_err());

        let all_free = ParameterMask::all_free(inner);
        assert_eq!(all_free.mask(), vec![None; 3]);
        assert_eq!(all_free.n_free_params(), 3);
    }

    /// The wrapped force can be replaced only by one with the same parameter count.
    #[test]
    fn set_inner_checks_parameter_count() {
        let mut masked =
            ParameterMask::all_free(NonGravKind::JplComet(JplCometNonGrav::standard_comet()));
        let other = JplCometNonGrav::new(1.0, 1.0, 2.0, 1.0, 0.0, 0.0);
        masked.set_inner(NonGravKind::JplComet(other)).unwrap();
        assert!(
            masked
                .set_inner(NonGravKind::Dust(crate::forces::DustNonGrav))
                .is_err()
        );
    }
}
