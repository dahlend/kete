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
        let rr0 = pos.norm() / self.r_0;
        Ok(self.alpha * rr0.powf(-self.m) * (1.0 + rr0.powf(self.n)).powf(-self.k))
    }
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
        let pos_norm = pos.normalize();
        let t_vec = (vel - pos_norm * vel.dot(&pos_norm)).normalize();
        let n_vec = pos_norm.cross(&t_vec);

        let scale = self.g_r(&pos, &vel)?;

        let [a1, a2, a3] = [free_params[0], free_params[1], free_params[2]];
        let result = pos_norm * (scale * a1) + t_vec * (scale * a2) + n_vec * (scale * a3);
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
        let pos_norm = pos.normalize();
        let t_vec = (vel - pos_norm * vel.dot(&pos_norm)).normalize();
        let n_vec = pos_norm.cross(&t_vec);

        let scale = self.g_r(&pos, &vel)?;

        let mut out = Matrix3xX::<f64>::zeros(3);
        out.set_column(0, &(pos_norm * scale));
        out.set_column(1, &(t_vec * scale));
        out.set_column(2, &(n_vec * scale));
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
