//! Dust grain force: solar radiation pressure + Poynting-Robertson drag.

use nalgebra::{Matrix3, Matrix3xX, Vector3};

use crate::constants::{C_AU_PER_DAY_INV_SQUARED, GMS};
use crate::errors::KeteResult;
use crate::forces::ParameterizedForce;
use crate::frames::{Equatorial, SunCenter, Vector};
use crate::time::{TDB, Time};

/// Dust grain force: solar radiation pressure plus Poynting-Robertson drag.
///
/// `beta` is exposed as a free parameter.
///
/// `accel` expects `pos`/`vel` Sun-relative.
#[derive(Debug, Clone, Default)]
pub struct DustNonGrav;

impl ParameterizedForce for DustNonGrav {
    type Frame = Equatorial;
    type Center = SunCenter;

    fn n_free_params(&self) -> usize {
        1
    }

    fn free_param_names(&self) -> Vec<&'static str> {
        vec!["beta"]
    }

    fn lower_bounds(&self) -> Vec<Option<f64>> {
        vec![Some(0.0)]
    }

    fn accel(
        &self,
        _time: Time<TDB>,
        pos: &Vector<Equatorial>,
        vel: &Vector<Equatorial>,
        free_params: &[f64],
    ) -> KeteResult<Vector<Equatorial>> {
        let beta = free_params[0];
        let pos_v: Vector3<f64> = (*pos).into();
        let vel_v: Vector3<f64> = (*vel).into();
        let pos_norm = pos_v.normalize();
        let r_dot = pos_norm.dot(&vel_v);
        let norm2_inv = pos_v.norm_squared().recip();
        let scaling = GMS * beta * norm2_inv;
        let result = scaling
            * ((1.0 - r_dot * C_AU_PER_DAY_INV_SQUARED) * pos_norm
                - vel_v * C_AU_PER_DAY_INV_SQUARED);
        Ok(Vector::<Equatorial>::new(result.into()))
    }

    fn jacobians(
        &self,
        _time: Time<TDB>,
        pos: &Vector<Equatorial>,
        vel: &Vector<Equatorial>,
        free_params: &[f64],
    ) -> KeteResult<(Matrix3<f64>, Matrix3<f64>)> {
        let pos_v: Vector3<f64> = (*pos).into();
        let vel_v: Vector3<f64> = (*vel).into();
        let beta = free_params[0];
        let r = pos_v.norm();
        let r2 = r * r;
        let d_hat = pos_v / r;
        let cinv2 = C_AU_PER_DAY_INV_SQUARED;
        let r_dot = d_hat.dot(&vel_v);
        let s = GMS * beta / r2;
        let ident = Matrix3::<f64>::identity();
        let inner = (1.0 - r_dot * cinv2) * d_hat - cinv2 * vel_v;
        let dd_hat = (ident - d_hat * d_hat.transpose()) / r;
        let dr_dot_col = (vel_v - r_dot * d_hat) / r;
        let da_dr = (-2.0 * s / r2) * inner * pos_v.transpose()
            + s * (-cinv2 * d_hat * dr_dot_col.transpose() + (1.0 - r_dot * cinv2) * dd_hat);
        let da_dv = -s * cinv2 * (d_hat * d_hat.transpose() + ident);
        Ok((da_dr, da_dv))
    }

    fn parameter_jacobian(
        &self,
        _time: Time<TDB>,
        pos: &Vector<Equatorial>,
        vel: &Vector<Equatorial>,
        _free_params: &[f64],
    ) -> KeteResult<Matrix3xX<f64>> {
        let pos_v: Vector3<f64> = (*pos).into();
        let vel_v: Vector3<f64> = (*vel).into();
        let pos_hat = pos_v.normalize();
        let r_dot = pos_hat.dot(&vel_v);
        let norm2_inv = pos_v.norm_squared().recip();
        let scale = GMS * norm2_inv;
        let partial = scale
            * ((1.0 - r_dot * C_AU_PER_DAY_INV_SQUARED) * pos_hat
                - vel_v * C_AU_PER_DAY_INV_SQUARED);
        let mut out = Matrix3xX::<f64>::zeros(1);
        out[(0, 0)] = partial[0];
        out[(1, 0)] = partial[1];
        out[(2, 0)] = partial[2];
        Ok(out)
    }
}
