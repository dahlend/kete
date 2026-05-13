//! `Sum<A, B>`: additive composition of two [`ParameterizedForce`] impls.
//!
//! `Sum` is the only composition primitive: a force model is either a single
//! force, or `Sum<A, B>` of two forces sharing a frame and center. Free
//! parameters concatenate (A first, B second); accelerations and jacobians
//! sum with Neumaier compensation; parameter-jacobian columns concatenate
//! side by side.
//!
//! ## Parameter ordering contract
//!
//! `Sum<A, B>` always lays out parameters as `[A's params | B's params]`.
//! An `UncertainState`'s `free_params` and covariance rows/columns are
//! sized to match a specific composition; reconstructing a `Sum` with the
//! forces swapped silently inverts the covariance interpretation for the
//! affected parameters. Compose in the same order on both sides.

use nalgebra::{Matrix3, Matrix3xX, Vector3};

use crate::errors::{Error, KeteResult};
use crate::forces::{Force, ParameterizedForce};
use crate::frames::Vector;
use crate::time::{TDB, Time};

/// Additive composition of two [`ParameterizedForce`] impls.
///
/// The two inner forces must share `Frame` and `Center`.
#[derive(Debug, Clone)]
pub struct Sum<A, B>(pub A, pub B);

impl<A, B> Sum<A, B> {
    /// Compose two forces additively.
    pub fn new(a: A, b: B) -> Self {
        Self(a, b)
    }
}

impl<A, B> ParameterizedForce for Sum<A, B>
where
    A: ParameterizedForce,
    B: ParameterizedForce<Frame = A::Frame, Center = A::Center>,
{
    type Frame = A::Frame;
    type Center = A::Center;

    fn n_free_params(&self) -> usize {
        self.0.n_free_params() + self.1.n_free_params()
    }

    fn free_param_names(&self) -> Vec<&'static str> {
        let mut names = self.0.free_param_names();
        names.extend(self.1.free_param_names());
        names
    }

    fn lower_bounds(&self) -> Vec<Option<f64>> {
        let mut bounds = self.0.lower_bounds();
        bounds.extend(self.1.lower_bounds());
        bounds
    }

    fn accel(
        &self,
        time: Time<TDB>,
        pos: &Vector<A::Frame>,
        vel: &Vector<A::Frame>,
        free_params: &[f64],
    ) -> KeteResult<Vector<A::Frame>> {
        let (a_slice, b_slice) = split_params(self, free_params)?;
        let a: Vector3<f64> = self.0.accel(time, pos, vel, a_slice)?.into();
        let b: Vector3<f64> = self.1.accel(time, pos, vel, b_slice)?.into();
        let mut sum = Vector3::<f64>::zeros();
        let mut comp = Vector3::<f64>::zeros();
        neumaier_add(&mut sum, &mut comp, a);
        neumaier_add(&mut sum, &mut comp, b);
        Ok(Vector::<A::Frame>::new((sum + comp).into()))
    }

    fn jacobians(
        &self,
        time: Time<TDB>,
        pos: &Vector<A::Frame>,
        vel: &Vector<A::Frame>,
        free_params: &[f64],
    ) -> KeteResult<(Matrix3<f64>, Matrix3<f64>)> {
        let (a_slice, b_slice) = split_params(self, free_params)?;
        let (a_dr, a_dv) = self.0.jacobians(time, pos, vel, a_slice)?;
        let (b_dr, b_dv) = self.1.jacobians(time, pos, vel, b_slice)?;
        let mut sum_dr = Matrix3::<f64>::zeros();
        let mut comp_dr = Matrix3::<f64>::zeros();
        let mut sum_dv = Matrix3::<f64>::zeros();
        let mut comp_dv = Matrix3::<f64>::zeros();
        neumaier_add_matrix(&mut sum_dr, &mut comp_dr, a_dr);
        neumaier_add_matrix(&mut sum_dr, &mut comp_dr, b_dr);
        neumaier_add_matrix(&mut sum_dv, &mut comp_dv, a_dv);
        neumaier_add_matrix(&mut sum_dv, &mut comp_dv, b_dv);
        Ok((sum_dr + comp_dr, sum_dv + comp_dv))
    }

    fn parameter_jacobian(
        &self,
        time: Time<TDB>,
        pos: &Vector<A::Frame>,
        vel: &Vector<A::Frame>,
        free_params: &[f64],
    ) -> KeteResult<Matrix3xX<f64>> {
        let (a_slice, b_slice) = split_params(self, free_params)?;
        let na = self.0.n_free_params();
        let nb = self.1.n_free_params();
        let mut out = Matrix3xX::<f64>::zeros(na + nb);
        if na > 0 {
            let block_a = self.0.parameter_jacobian(time, pos, vel, a_slice)?;
            out.columns_mut(0, na).copy_from(&block_a);
        }
        if nb > 0 {
            let block_b = self.1.parameter_jacobian(time, pos, vel, b_slice)?;
            out.columns_mut(na, nb).copy_from(&block_b);
        }
        Ok(out)
    }
}

/// Marker: `Sum<A, B>` is a [`Force`] iff both inner forces are.
impl<A, B> Force for Sum<A, B>
where
    A: Force,
    B: Force<Frame = A::Frame, Center = A::Center>,
{
}

/// Split the combined `free_params` slice into A's and B's portions.
///
/// Empty input is valid iff total expected parameters is zero, in which
/// case both forces receive `&[]`.
fn split_params<'a, A, B>(
    sum: &Sum<A, B>,
    free_params: &'a [f64],
) -> KeteResult<(&'a [f64], &'a [f64])>
where
    A: ParameterizedForce,
    B: ParameterizedForce<Frame = A::Frame, Center = A::Center>,
{
    let na = sum.0.n_free_params();
    let nb = sum.1.n_free_params();
    let total = na + nb;
    if free_params.is_empty() {
        if total == 0 {
            return Ok((&[], &[]));
        }
        return Err(Error::ValueError(format!(
            "Sum expects {total} free parameters but received an empty slice",
        )));
    }
    if free_params.len() != total {
        return Err(Error::ValueError(format!(
            "Sum expects {total} free parameters, got {}",
            free_params.len()
        )));
    }
    Ok((&free_params[..na], &free_params[na..]))
}

/// Neumaier-compensated 3D vector addition: `sum += term`, with `comp`
/// accumulating cancellation error.
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
    fn sum_adds_two_constant_forces() {
        let combo = Sum::new(
            ConstantForce {
                accel: Vector3::new(1.0, 0.0, 0.0),
            },
            ConstantForce {
                accel: Vector3::new(0.0, 2.0, 0.0),
            },
        );
        let pos = Vector::<Equatorial>::new([1.0, 0.0, 0.0]);
        let vel = Vector::<Equatorial>::new([0.0, 1.0, 0.0]);
        let a: Vector3<f64> = combo
            .accel(Time::<TDB>::new(0.0), &pos, &vel, &[])
            .unwrap()
            .into();
        assert_eq!(a, Vector3::new(1.0, 2.0, 0.0));
    }
}
