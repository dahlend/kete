//! Farnocchia et al. 2025 oblate-spheroid radiation + thermal recoil force.

use nalgebra::Vector3;

use crate::constants::{F0_OVER_C_AU_DAY2, SOLAR_FLUX, STEFAN_BOLTZMANN};
use crate::errors::{Error, KeteResult};
use crate::forces::ParameterizedForce;
use crate::frames::{Equatorial, SunCenter, Vector};
use crate::time::{TDB, Time};

/// Compute `A/M` (`m^2 / kg`) from physical inputs (Farnocchia 2025 Eq. 6).
///
/// `density` is in kg/m^3, `diameter` in km, `flattening` is the axis ratio
/// `R_P / R_E` (1.0 for a sphere).
#[must_use]
pub fn a_over_m_from_physical(density: f64, diameter: f64, flattening: f64) -> f64 {
    let r_p = flattening.powf(2.0 / 3.0) * diameter * 500.0;
    3.0 / (4.0 * density * r_p)
}

/// Inverse of [`a_over_m_from_physical`]: solve for bulk density (`kg / m^3`)
/// given `a_over_m`, `diameter` (km), and `flattening`.
#[must_use]
pub fn density_from_a_over_m(a_over_m: f64, diameter: f64, flattening: f64) -> f64 {
    let r_p = flattening.powf(2.0 / 3.0) * diameter * 500.0;
    3.0 / (4.0 * a_over_m * r_p)
}

/// Compute `lambda_0` (dimensionless, Farnocchia 2025 Eq. 12) from
/// physical inputs.
#[must_use]
pub fn lambda_0_from_physical(
    thermal_inertia: f64,
    emissivity: f64,
    absorptivity: f64,
    flattening: f64,
    rotation_period: f64,
) -> f64 {
    let sigma = shape_factors(flattening).2;
    let denom = (emissivity * STEFAN_BOLTZMANN).powf(0.25) * (absorptivity * SOLAR_FLUX).powf(0.75);
    thermal_inertia * sigma.powf(0.75) / denom
        * (std::f64::consts::PI / (2.0 * rotation_period * 3600.0)).sqrt()
}

/// Inverse of [`lambda_0_from_physical`].
#[must_use]
pub fn thermal_inertia_from_lambda_0(
    lambda_0: f64,
    emissivity: f64,
    absorptivity: f64,
    flattening: f64,
    rotation_period: f64,
) -> f64 {
    let sigma = shape_factors(flattening).2;
    let numer = (emissivity * STEFAN_BOLTZMANN).powf(0.25) * (absorptivity * SOLAR_FLUX).powf(0.75);
    lambda_0 * numer
        / (sigma.powf(0.75) * (std::f64::consts::PI / (2.0 * rotation_period * 3600.0)).sqrt())
}

/// Oblate-spheroid shape factors `(psi_X, psi_Z, Sigma)` (Farnocchia 2025).
///
/// `e` is the axis ratio `R_P / R_E`. Sphere limit (`e >= 1`): all factors 1.
/// `Sigma` is also used by the physical-input helpers.
pub(super) fn shape_factors(e: f64) -> (f64, f64, f64) {
    if e >= 1.0 - 1e-9 {
        return (1.0, 1.0, 1.0);
    }
    let e2 = e * e;
    let eta = (1.0 - e2).sqrt();
    let log_term = ((1.0 + eta) / (1.0 - eta)).ln();
    let psi_x = 3.0 * e2 / (4.0 * eta * eta) * ((1.0 + eta * eta) / (2.0 * eta) * log_term - 1.0);
    let psi_z = 3.0 / (2.0 * eta * eta) * (1.0 - e2 / (2.0 * eta) * log_term);
    let sigma = 0.5 * (1.0 + e2 / (2.0 * eta) * log_term);
    (psi_x, psi_z, sigma)
}

/// Farnocchia et al. 2025 oblate-spheroid radiation force.
///
/// `a_over_m` and `lambda_0` are free parameters; `albedo`, `absorptivity`,
/// `flattening`, and `spin_pole` are fixed surface descriptors.
///
/// `accel` expects `pos`/`vel` Sun-relative.
#[derive(Debug, Clone)]
pub struct FarnocchiaNonGrav {
    /// Geometric albedo `a_0` (Lambert approximation). Enters SRP only.
    pub albedo: f64,
    /// `alpha = 1 - A_B`, where `A_B` is the Bond albedo. Multiplies the
    /// thermal terms.
    pub absorptivity: f64,
    /// Axis ratio `e = R_P / R_E` (1.0 for a sphere, < 1 for oblate).
    pub flattening: f64,
    /// Spin pole unit vector in the equatorial frame, pre-normalized.
    pub spin_pole: Vector<Equatorial>,
}

impl FarnocchiaNonGrav {
    /// Build, validating and pre-normalizing the spin pole.
    ///
    /// # Errors
    /// Returns `Error::ValueError` if any of `albedo`, `absorptivity`, or
    /// `flattening` is non-finite or negative, if `flattening > 1`, or if
    /// the spin pole is non-finite or zero.
    pub fn new(
        albedo: f64,
        absorptivity: f64,
        flattening: f64,
        spin_pole: Vector<Equatorial>,
    ) -> KeteResult<Self> {
        for (name, v) in [
            ("albedo", albedo),
            ("absorptivity", absorptivity),
            ("flattening", flattening),
        ] {
            if !v.is_finite() || v < 0.0 {
                return Err(Error::ValueError(format!(
                    "FarnocchiaNonGrav: '{name}' must be finite and >= 0 (got {v})"
                )));
            }
        }
        if flattening > 1.0 {
            return Err(Error::ValueError(format!(
                "FarnocchiaNonGrav: 'flattening' must be <= 1 (got {flattening})"
            )));
        }
        if !spin_pole.is_finite() || spin_pole.norm() == 0.0 {
            return Err(Error::ValueError(
                "FarnocchiaNonGrav: 'spin_pole' must be a finite, non-zero vector".into(),
            ));
        }
        Ok(Self {
            albedo,
            absorptivity,
            flattening,
            spin_pole: spin_pole.normalize(),
        })
    }
}

impl ParameterizedForce for FarnocchiaNonGrav {
    type Frame = Equatorial;
    type Center = SunCenter;

    fn as_any(&self) -> Option<&(dyn std::any::Any + 'static)> {
        Some(self)
    }

    fn n_free_params(&self) -> usize {
        2
    }

    fn free_param_names(&self) -> Vec<&'static str> {
        vec!["a_over_m", "lambda_0"]
    }

    fn accel(
        &self,
        _time: Time<TDB>,
        pos: &Vector<Equatorial>,
        _vel: &Vector<Equatorial>,
        free_params: &[f64],
    ) -> KeteResult<Vector<Equatorial>> {
        let a_over_m = free_params[0];
        let lambda_0 = free_params[1];
        let e = self.flattening;

        let s_hat: Vector3<f64> = self.spin_pole.into();
        let pos_v: Vector3<f64> = (*pos).into();
        let r = pos_v.norm();
        let r_inv = r.recip();
        let r_hat = pos_v * r_inv;
        let g = r_inv * r_inv;

        let (psi_x, psi_z, _sigma) = shape_factors(e);

        let r_dot_s = r_hat.dot(&s_hat);
        let cos_theta_0 = -r_dot_s;
        let sin2_theta_0 = (1.0 - cos_theta_0 * cos_theta_0).max(0.0);
        let j2_theta = (e * e * sin2_theta_0 + cos_theta_0 * cos_theta_0).sqrt();

        let scale = a_over_m * F0_OVER_C_AU_DAY2 * g;

        let four_ninths_a0 = 4.0 / 9.0 * self.albedo;
        let srp_radial = j2_theta + four_ninths_a0 * psi_x;
        let srp_pole = four_ninths_a0 * (psi_z - psi_x) * r_dot_s;
        let mut accel = scale * (srp_radial * r_hat + srp_pole * s_hat);

        if self.absorptivity > 0.0 && lambda_0 > 0.0 {
            let lambda = lambda_0 / j2_theta.powf(0.75) * r.powf(1.5);
            let denom = 1.0 + 2.0 * lambda + 2.0 * lambda * lambda;
            let big_lambda_1 = (1.0 + lambda) / denom;
            let big_lambda_2 = lambda / denom;

            let four_ninths_alpha = 4.0 / 9.0 * self.absorptivity;

            let t1_radial = big_lambda_1 * psi_x;
            let t1_pole = (psi_z - big_lambda_1 * psi_x) * r_dot_s;
            accel += (four_ninths_alpha * scale) * (t1_radial * r_hat + t1_pole * s_hat);

            let t2_coeff = -four_ninths_alpha * scale * big_lambda_2 * psi_x;
            accel += t2_coeff * r_hat.cross(&s_hat);
        }

        Ok(Vector::<Equatorial>::new(accel.into()))
    }
}
