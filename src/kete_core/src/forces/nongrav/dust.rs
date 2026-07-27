//! Dust grain force: solar radiation pressure + Poynting-Robertson drag.

use nalgebra::{Matrix3, Matrix3xX, Vector3};

use crate::constants::{C_AU_PER_DAY_INV, GMS};
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
        // Poynting-Robertson + solar-wind drag: first order in v/c (Burns, Lamy
        // & Soter 1979).
        let result =
            scaling * ((1.0 - r_dot * C_AU_PER_DAY_INV) * pos_norm - vel_v * C_AU_PER_DAY_INV);
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
        let cinv = C_AU_PER_DAY_INV;
        let r_dot = d_hat.dot(&vel_v);
        let s = GMS * beta / r2;
        let ident = Matrix3::<f64>::identity();
        let inner = (1.0 - r_dot * cinv) * d_hat - cinv * vel_v;
        let dd_hat = (ident - d_hat * d_hat.transpose()) / r;
        let dr_dot_col = (vel_v - r_dot * d_hat) / r;
        let da_dr = (-2.0 * s / r2) * inner * pos_v.transpose()
            + s * (-cinv * d_hat * dr_dot_col.transpose() + (1.0 - r_dot * cinv) * dd_hat);
        let da_dv = -s * cinv * (d_hat * d_hat.transpose() + ident);
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
        let partial =
            scale * ((1.0 - r_dot * C_AU_PER_DAY_INV) * pos_hat - vel_v * C_AU_PER_DAY_INV);
        let mut out = Matrix3xX::<f64>::zeros(1);
        out[(0, 0)] = partial[0];
        out[(1, 0)] = partial[1];
        out[(2, 0)] = partial[2];
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::semi_major_axis;
    use crate::constants::C_AU_PER_DAY;
    use crate::integrators::RadauIntegrator;
    use crate::time::Time;
    use nalgebra::DVector;
    use std::f64::consts::TAU;

    /// The drag is first order in v/c: at a point with purely tangential
    /// velocity the transverse acceleration is exactly `-beta GMS v / (r^2 c)`.
    /// A 1/c^2 factor (the prior bug) would make this ~173x too small.
    #[test]
    fn drag_magnitude_matches_bls() {
        let beta = 0.1;
        let r = 1.3;
        let v = 0.018; // AU/day, perpendicular to r so r_dot = 0
        let pos = Vector::<Equatorial>::new([r, 0.0, 0.0]);
        let vel = Vector::<Equatorial>::new([0.0, v, 0.0]);
        let accel: Vector3<f64> = DustNonGrav
            .accel(Time::new(0.0), &pos, &vel, &[beta])
            .unwrap()
            .into();

        // Radial component: radiation pressure beta GMS / r^2 (r_dot = 0).
        let radial = beta * GMS / (r * r);
        // Transverse component: the PR drag, -beta GMS v / (r^2 c).
        let transverse = -beta * GMS * v / (r * r * C_AU_PER_DAY);
        println!("drag_magnitude: radial {:.6e} (exp {radial:.6e})", accel.x);
        println!("  transverse {:.6e} (exp {transverse:.6e})", accel.y);
        assert!((accel.x - radial).abs() < 1e-18);
        assert!(
            (accel.y - transverse).abs() < 1e-18,
            "drag off by a factor (units bug?): {} vs {transverse}",
            accel.y
        );
        assert!(accel.z.abs() < 1e-30);
    }

    /// Poynting-Robertson inspiral of a circular grain, driven through the
    /// Radau integrator (which handles the velocity-dependent force natively).
    ///
    /// For a circular orbit `da/dt = -2 beta GMS / (c a)` (Wyatt & Whipple
    /// 1950), so `a^2` decays linearly at rate `d(a^2)/dt = -4 beta GMS / c`,
    /// independent of the radiation-reduced gravity. This is the physical
    /// benchmark the force must reproduce.
    #[test]
    fn pr_inspiral_rate_matches_theory() {
        let beta = 0.02;
        let a0 = 1.0;
        let mu_eff = (1.0 - beta) * GMS; // gravity minus radiation pressure

        // Circular orbit for the reduced gravity: v = sqrt(mu_eff / a0).
        let v0 = (mu_eff / a0).sqrt();
        let accel = |t: Time<TDB>,
                     pos: &DVector<f64>,
                     vel: &DVector<f64>,
                     _m: &mut (),
                     _e: bool|
         -> KeteResult<DVector<f64>> {
            let p = Vector::<Equatorial>::new([pos[0], pos[1], pos[2]]);
            let v = Vector::<Equatorial>::new([vel[0], vel[1], vel[2]]);
            let pv: Vector3<f64> = p.into();
            let r3 = pv.norm().powi(3);
            let grav = -GMS / r3 * pv; // full solar gravity
            let dust: Vector3<f64> = DustNonGrav.accel(t, &p, &v, &[beta])?.into();
            let tot = grav + dust;
            Ok(DVector::from_row_slice(&[tot.x, tot.y, tot.z]))
        };

        let period = TAU * (a0.powi(3) / mu_eff).sqrt();
        let mut a_sq = Vec::new();
        let mut times = Vec::new();
        let mut pos = DVector::from_row_slice(&[a0, 0.0, 0.0]);
        let mut vel = DVector::from_row_slice(&[0.0, v0, 0.0]);
        let n_orbits = 200_i32;
        for k in 0..=n_orbits {
            let p = Vector3::new(pos[0], pos[1], pos[2]);
            let v = Vector3::new(vel[0], vel[1], vel[2]);
            // Osculating semi-major axis under the reduced gravity.
            let a = semi_major_axis(&p, &v, mu_eff);
            times.push(f64::from(k) * period);
            a_sq.push(a * a);
            if k == n_orbits {
                break;
            }
            let t0 = Time::<TDB>::new(times[times.len() - 1]);
            let t1 = Time::<TDB>::new(times[times.len() - 1] + period);
            let (np, nv, ()) =
                RadauIntegrator::integrate(&accel, pos, vel, t0, t1, (), None).unwrap();
            pos = np;
            vel = nv;
        }

        // Slope of a^2 vs t by least squares.
        #[allow(clippy::cast_precision_loss, reason = "small loop count")]
        let n = times.len() as f64;
        let mean_t = times.iter().sum::<f64>() / n;
        let mean_a = a_sq.iter().sum::<f64>() / n;
        let num: f64 = times
            .iter()
            .zip(&a_sq)
            .map(|(t, a)| (t - mean_t) * (a - mean_a))
            .sum();
        let den: f64 = times.iter().map(|t| (t - mean_t).powi(2)).sum();
        let measured = num / den;
        let predicted = -4.0 * beta * GMS / C_AU_PER_DAY;
        let rel = ((measured - predicted) / predicted).abs();

        println!("pr_inspiral: beta = {beta}, a0 = {a0} AU, {n_orbits} orbits");
        println!("  measured d(a^2)/dt  {measured:.6e} AU^2/day");
        println!("  predicted -4bGMS/c  {predicted:.6e} AU^2/day");
        println!("  relative difference {rel:.3e}");
        assert!(rel < 0.01, "PR inspiral rate off by {rel:e}");
    }
}
