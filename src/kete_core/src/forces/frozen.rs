//! [`FrozenForce`]: a `ParameterizedForce` impl with all parameters baked in at construction.
//!
//! Unlike [`ParameterMask`], which adapts an inner `ParameterizedForce` by masking some
//! parameters, `FrozenForce` stores a concrete parameter vector alongside the
//! inner force and unconditionally returns `n_free_params() == 0`. The type
//! itself is the guarantee -- no runtime check needed.
//!
//! Use `FrozenForce` wherever you have a known parameter estimate and want to
//! propagate a plain [`State`](crate::prelude::State):
//! - batch parallel propagation (each object has its own values)
//! - covariance samples (each draw is one specific state + parameter set)
//! - orbit-fitter inner loop (evaluate residuals at current estimates)
//!
//! For the variational path -- propagating an [`UncertainState`](crate::state::UncertainState)
//! and computing parameter-sensitivity columns in the STM -- use
//! [`ParameterMask`](super::ParameterMask) with an all-`None` mask instead.

use nalgebra::Matrix3;

use crate::errors::{Error, KeteResult};
use crate::forces::{Force, ParameterizedForce};
use crate::frames::Vector;
use crate::time::{TDB, Time};

/// A `ParameterizedForce` wrapper with all parameters frozen at construction time.
///
/// `n_free_params()` unconditionally returns `0`. The baked-in `values`
/// are forwarded to the inner force at every evaluation.
#[derive(Debug, Clone)]
pub struct FrozenForce<F: ParameterizedForce> {
    /// Wrapped force.
    pub inner: F,
    /// Frozen parameter values, one per `inner.n_free_params()` at construction.
    pub values: Vec<f64>,
}

impl<F: ParameterizedForce> FrozenForce<F> {
    /// Wrap `inner` with the given parameter values frozen in.
    ///
    /// # Errors
    /// Returns `ValueError` if `values.len() != inner.n_free_params()`.
    pub fn new(inner: F, values: Vec<f64>) -> KeteResult<Self> {
        if values.len() != inner.n_free_params() {
            return Err(Error::ValueError(format!(
                "FrozenForce: {} values provided but inner force has {} free parameters",
                values.len(),
                inner.n_free_params()
            )));
        }
        Ok(Self { inner, values })
    }

    /// The frozen parameter values.
    pub fn values(&self) -> &[f64] {
        &self.values
    }
}

impl<F: ParameterizedForce + 'static> ParameterizedForce for FrozenForce<F> {
    type Frame = F::Frame;
    type Center = F::Center;

    // `n_free_params`, `free_param_names`, `parameter_jacobian`: inherit the
    // trait defaults (0, empty, 3x0). Only `accel` and `jacobians` need
    // overrides -- both delegate to the inner force with baked-in values
    // (preserving the inner's analytical jacobian impl when available).

    fn accel(
        &self,
        time: Time<TDB>,
        pos: &Vector<F::Frame>,
        vel: &Vector<F::Frame>,
        _free_params: &[f64],
    ) -> KeteResult<Vector<F::Frame>> {
        self.inner.accel(time, pos, vel, &self.values)
    }

    fn jacobians(
        &self,
        time: Time<TDB>,
        pos: &Vector<F::Frame>,
        vel: &Vector<F::Frame>,
        _free_params: &[f64],
    ) -> KeteResult<(Matrix3<f64>, Matrix3<f64>)> {
        self.inner.jacobians(time, pos, vel, &self.values)
    }
}

/// Marker: a [`FrozenForce`] has its parameters baked in, so it is structurally
/// a [`Force`] (no free parameters).
impl<F: ParameterizedForce + 'static> Force for FrozenForce<F> {}
