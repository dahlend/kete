//! JPL comet non-gravitational force (`a1`, `a2`, `a3` in RTN frame).

use nalgebra::{Matrix3xX, Vector3};

use crate::constants::GMS;
use crate::errors::KeteResult;
use crate::forces::ParameterizedForce;
use crate::frames::{Equatorial, SunCenter, Vector};
use crate::kepler::analytic_2_body;
use crate::time::{TDB, Time};

/// JPL non-gravitational force in radial / tangential / normal frame.
///
/// Exposes `a1`, `a2`, `a3` as free parameters; the `g(r)` shape
/// coefficients are fixed fields.
///
/// `accel` expects `pos`/`vel` Sun-relative.
#[derive(Debug, Clone)]
pub struct JplCometNonGrav {
    /// `g(r) = alpha * (r/r_0)^(-m) * (1 + (r/r_0)^n)^(-k)` coefficient.
    pub alpha: f64,
    /// `g(r)` reference distance in AU.
    pub r_0: f64,
    /// `g(r)` exponent.
    pub m: f64,
    /// `g(r)` exponent.
    pub n: f64,
    /// `g(r)` exponent.
    pub k: f64,
    /// Time delay in days; positions are propagated by `-dt` via two-body
    /// Kepler before evaluating `g(r)` when `dt != 0`.
    pub dt: f64,
}

impl JplCometNonGrav {
    /// Build with all coefficients explicit.
    #[must_use]
    pub fn new(alpha: f64, r_0: f64, m: f64, n: f64, k: f64, dt: f64) -> Self {
        Self {
            alpha,
            r_0,
            m,
            n,
            k,
            dt,
        }
    }

    /// Standard comet drop-off coefficients (Marsden et al.).
    #[must_use]
    pub fn standard_comet() -> Self {
        Self {
            alpha: 0.111_262_042_6,
            r_0: 2.808,
            m: 2.15,
            n: 5.093,
            k: 4.6142,
            dt: 0.0,
        }
    }

    /// Evaluate `g(r)` at the (optionally `dt`-back-propagated) position.
    fn g_r(&self, pos: &Vector3<f64>, vel: &Vector3<f64>) -> KeteResult<f64> {
        let mut pos = *pos;
        if self.dt != 0.0 {
            // Back-propagate by dt to get the perihelion-referenced position
            // for the g(r) scaling.
            match analytic_2_body((-self.dt).into(), &pos, vel, None) {
                Ok((p, _)) => pos = p,
                Err(err) => {
                    // For sigma-point perturbations with large sigma_factor the
                    // perturbed orbit can be unbound, where the Kepler solve may
                    // fail and the delayed-position model is out of its regime
                    // anyway; fall back to the current position (dt=0
                    // approximation) rather than propagating the error up
                    // through the integrator.  For bound orbits a failure is a
                    // genuine convergence problem and must surface loudly.
                    let specific_energy = 0.5 * vel.norm_squared() - GMS / pos.norm();
                    if specific_energy < 0.0 {
                        return Err(err);
                    }
                }
            }
        }
        Ok(self.g_of_r(pos.norm()))
    }

    /// Evaluate `g(r)` at the given heliocentric distance.
    fn g_of_r(&self, r: f64) -> f64 {
        let rr0 = r / self.r_0;
        self.alpha * rr0.powf(-self.m) * (1.0 + rr0.powf(self.n)).powf(-self.k)
    }

    /// Acceleration from the given `a1`/`a2`/`a3`, with `g(r)` evaluated at
    /// the current distance (the `dt` lag is not applied; callers with a
    /// nonzero `dt` must use the [`ParameterizedForce::accel`] path).
    ///
    /// `pos` and `vel` are Sun-relative on any shared inertial axes; the
    /// model is built from dot and cross products only, so it is frame
    /// covariant and the result is returned on the same axes. This is the
    /// shared core used by the Wisdom-Holman map's kick.
    pub(crate) fn accel_no_lag(
        &self,
        pos: &Vector3<f64>,
        vel: &Vector3<f64>,
        a1: f64,
        a2: f64,
        a3: f64,
    ) -> Vector3<f64> {
        let (r_hat, t_hat, n_hat) = rtn_dirs(pos, vel);
        let scale = self.g_of_r(pos.norm());
        r_hat * (scale * a1) + t_hat * (scale * a2) + n_hat * (scale * a3)
    }
}

/// The radial / transverse / normal unit vectors of a Sun-relative state.
fn rtn_dirs(pos: &Vector3<f64>, vel: &Vector3<f64>) -> (Vector3<f64>, Vector3<f64>, Vector3<f64>) {
    let r_hat = pos.normalize();
    let t_hat = (vel - r_hat * vel.dot(&r_hat)).normalize();
    let n_hat = r_hat.cross(&t_hat);
    (r_hat, t_hat, n_hat)
}

impl ParameterizedForce for JplCometNonGrav {
    type Frame = Equatorial;
    type Center = SunCenter;

    fn n_free_params(&self) -> usize {
        3
    }

    fn free_param_names(&self) -> Vec<&'static str> {
        vec!["a1", "a2", "a3"]
    }

    fn accel(
        &self,
        _time: Time<TDB>,
        pos: &Vector<Equatorial>,
        vel: &Vector<Equatorial>,
        free_params: &[f64],
    ) -> KeteResult<Vector<Equatorial>> {
        let pos: Vector3<f64> = (*pos).into();
        let vel: Vector3<f64> = (*vel).into();
        let (r_hat, t_hat, n_hat) = rtn_dirs(&pos, &vel);

        let scale = self.g_r(&pos, &vel)?;

        let [a1, a2, a3] = [free_params[0], free_params[1], free_params[2]];
        let result = r_hat * (scale * a1) + t_hat * (scale * a2) + n_hat * (scale * a3);
        Ok(Vector::<Equatorial>::new(result.into()))
    }

    fn parameter_jacobian(
        &self,
        _time: Time<TDB>,
        pos: &Vector<Equatorial>,
        vel: &Vector<Equatorial>,
        _free_params: &[f64],
    ) -> KeteResult<Matrix3xX<f64>> {
        // The acceleration is linear in (a1, a2, a3): the columns are `g(r)`
        // times the RTN basis, independent of the parameter values.
        let pos: Vector3<f64> = (*pos).into();
        let vel: Vector3<f64> = (*vel).into();
        let (r_hat, t_hat, n_hat) = rtn_dirs(&pos, &vel);

        let scale = self.g_r(&pos, &vel)?;

        let mut out = Matrix3xX::<f64>::zeros(3);
        out.set_column(0, &(r_hat * scale));
        out.set_column(1, &(t_hat * scale));
        out.set_column(2, &(n_hat * scale));
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pos() -> Vector<Equatorial> {
        Vector::<Equatorial>::new([1.5, 0.3, 0.1])
    }
    fn vel() -> Vector<Equatorial> {
        // Bound orbit at ~1.53 AU.
        Vector::<Equatorial>::new([-0.005, 0.012, 0.001])
    }
    fn epoch() -> Time<TDB> {
        Time::<TDB>::new(2_451_545.0)
    }

    fn accel_at(f: &JplCometNonGrav, params: &[f64]) -> Vector3<f64> {
        f.accel(epoch(), &pos(), &vel(), params).unwrap().into()
    }

    /// The velocity derivative must be real, and must be the right one.
    ///
    /// This force builds its radial-transverse-normal basis from the velocity, so
    /// `d(accel)/d(vel)` is nonzero, and a zero block there would reach every variational
    /// propagation and every fitted covariance built with this force.
    ///
    /// The reference is closed form, derived rather than differenced. With no `dt` lag
    /// neither `g(r)` nor the radial direction depends on velocity, and a velocity
    /// perturbation only tilts the transverse and normal directions about the radial axis:
    ///
    /// ```text
    /// w         = vel - r_hat (vel . r_hat)        transverse velocity
    /// d t_hat   = n_hat n_hat^T / |w|
    /// d n_hat   = -t_hat n_hat^T / |w|
    /// d accel   = g (a2 n_hat - a3 t_hat) n_hat^T / |w|
    /// ```
    ///
    /// Rank one, and orthogonal to the radial direction - a velocity change along `r_hat`
    /// leaves the basis alone, which the test checks separately.
    #[test]
    fn velocity_jacobian_matches_closed_form() {
        let force = JplCometNonGrav::standard_comet();
        assert_eq!(force.dt, 0.0, "the closed form below assumes no lag");
        let params = [1e-8, -3e-9, 5e-10];
        let (_, da_dv) = force.jacobians(epoch(), &pos(), &vel(), &params).unwrap();

        let pos_v: Vector3<f64> = pos().into();
        let vel_v: Vector3<f64> = vel().into();
        let (r_hat, t_hat, n_hat) = rtn_dirs(&pos_v, &vel_v);
        let transverse = (vel_v - r_hat * vel_v.dot(&r_hat)).norm();
        let scale = force.g_of_r(pos_v.norm());
        let expected =
            (n_hat * params[1] - t_hat * params[2]) * n_hat.transpose() * (scale / transverse);

        // The library differences forward at `sqrt(eps)` scaled steps, so the comparison
        // is held to that rather than to rounding.
        let relative = (da_dv - expected).norm() / expected.norm();
        println!("d(accel)/d(vel) vs closed form: {relative:e}");
        assert!(
            relative < 1e-6,
            "velocity jacobian {relative:e} does not match the closed form"
        );

        // The block must be non-zero at all, which is the failure a wrong default gives.
        assert!(
            da_dv.norm() > 0.0,
            "velocity jacobian is zero; this force depends on the velocity"
        );

        // A velocity change along the radial direction does not move the basis.
        let radial_response = da_dv * r_hat;
        assert!(
            radial_response.norm() < 1e-6 * da_dv.norm(),
            "radial velocity perturbation should not tilt the basis"
        );
    }

    /// The lag branch also has a velocity derivative, through the two-body
    /// back-propagation inside `g(r)`, and differencing picks it up without needing that
    /// propagation differentiated.
    #[test]
    fn velocity_jacobian_is_nonzero_with_lag() {
        let mut force = JplCometNonGrav::standard_comet();
        force.dt = 30.0;
        let params = [1e-8, -3e-9, 5e-10];
        let (_, da_dv) = force.jacobians(epoch(), &pos(), &vel(), &params).unwrap();
        assert!(da_dv.norm() > 0.0, "velocity jacobian is zero with a lag");
    }

    #[test]
    fn analytic_jacobian_matches_finite_difference() {
        // With and without the dt back-propagation branch.
        for dt in [0.0, 30.0] {
            let mut f = JplCometNonGrav::standard_comet();
            f.dt = dt;
            let params = [1e-8, -3e-9, 5e-10];
            let jac = f
                .parameter_jacobian(epoch(), &pos(), &vel(), &params)
                .unwrap();
            let h = 1e-10;
            for col in 0..3 {
                let analytic = Vector3::new(jac[(0, col)], jac[(1, col)], jac[(2, col)]);
                let mut p = params;
                p[col] += h;
                let a_plus = accel_at(&f, &p);
                p[col] = params[col] - h;
                let a_minus = accel_at(&f, &p);
                let fd = (a_plus - a_minus) / (2.0 * h);
                assert!(
                    (analytic - fd).norm() < 1e-9 * fd.norm(),
                    "dt {dt} column {col}: analytic {analytic:?} vs fd {fd:?}"
                );
            }
        }
    }

    #[test]
    fn zero_params_zero_accel() {
        let f = JplCometNonGrav::standard_comet();
        assert_eq!(accel_at(&f, &[0.0, 0.0, 0.0]).norm(), 0.0);
    }
}
