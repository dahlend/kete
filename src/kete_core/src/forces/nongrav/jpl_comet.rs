//! JPL comet non-gravitational force (`a1`, `a2`, `a3` in RTN frame).

use nalgebra::Vector3;

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
}

impl ParameterizedForce for JplCometNonGrav {
    type Frame = Equatorial;
    type Center = SunCenter;

    fn as_any(&self) -> Option<&(dyn std::any::Any + 'static)> {
        Some(self)
    }

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
        let mut pos: Vector3<f64> = (*pos).into();
        let vel: Vector3<f64> = (*vel).into();
        let pos_norm = pos.normalize();
        let t_vec = (vel - pos_norm * vel.dot(&pos_norm)).normalize();
        let n_vec = pos_norm.cross(&t_vec);

        if self.dt != 0.0 {
            let (p, _) = analytic_2_body((-self.dt).into(), &pos, &vel, None)?;
            pos = p;
        }

        let rr0 = pos.norm() / self.r_0;
        let scale = self.alpha * rr0.powf(-self.m) * (1.0 + rr0.powf(self.n)).powf(-self.k);

        let [a1, a2, a3] = [free_params[0], free_params[1], free_params[2]];
        let result = pos_norm * (scale * a1) + t_vec * (scale * a2) + n_vec * (scale * a3);
        Ok(Vector::<Equatorial>::new(result.into()))
    }
}
