//! Farnocchia et al. 2025 oblate-spheroid radiation + thermal recoil force.

use nalgebra::{Matrix3xX, Vector3};

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

/// Radiation pressure and thermal recoil acceleration in AU/Day^2, expressed
/// on the axes of whatever frame `pos` and `spin_pole` share.
///
/// `pos` is Sun-relative in AU and `spin_pole` must be a unit vector. The model
/// is built entirely from dot and cross products of the position and spin-pole
/// directions, so it is frame covariant: rotating both inputs rotates the
/// result. Callers may therefore work in any inertial frame, provided both
/// inputs are expressed in it.
///
/// This is the shared core of [`FarnocchiaNonGrav::accel`] (which supplies the
/// equatorial frame) and the Wisdom-Holman map's Yarkovsky kick (which
/// pre-rotates the pole into the map's frame to avoid rotating in its inner
/// loop). It is infallible so callers need no error plumbing.
///
/// The term along `r_hat x s_hat`, carrying the thermal lag `Lambda_2`, is the
/// Yarkovsky drift driver; it is not curl-free, which is what lets it do
/// secular work on the semi-major axis.
pub(crate) fn radiation_accel(
    pos: &Vector3<f64>,
    spin_pole: &Vector3<f64>,
    albedo: f64,
    absorptivity: f64,
    flattening: f64,
    a_over_m: f64,
    lambda_0: f64,
) -> Vector3<f64> {
    let e = flattening;
    let r = pos.norm();
    let r_inv = r.recip();
    let r_hat = pos * r_inv;
    let g = r_inv * r_inv;

    let (psi_x, psi_z, _sigma) = shape_factors(e);

    let r_dot_s = r_hat.dot(spin_pole);
    let cos_theta_0 = -r_dot_s;
    let sin2_theta_0 = (1.0 - cos_theta_0 * cos_theta_0).max(0.0);
    let j2_theta = (e * e * sin2_theta_0 + cos_theta_0 * cos_theta_0).sqrt();

    let scale = a_over_m * F0_OVER_C_AU_DAY2 * g;

    let four_ninths_a0 = 4.0 / 9.0 * albedo;
    let srp_radial = j2_theta + four_ninths_a0 * psi_x;
    let srp_pole = four_ninths_a0 * (psi_z - psi_x) * r_dot_s;
    let mut accel = scale * (srp_radial * r_hat + srp_pole * spin_pole);

    // Thermal recoil. The zero-lag limit `lambda_0 = 0` is included: there
    // `Lambda_1 = 1` and `Lambda_2 = 0`, leaving the radial recoil from
    // instantaneous re-emission with no transverse (Yarkovsky) component. This
    // keeps the force continuous in `lambda_0`, which the fitters rely on when
    // the parameter starts at 0. Set `absorptivity` to 0 to disable the thermal
    // terms entirely.
    if absorptivity > 0.0 && lambda_0 >= 0.0 {
        let lambda = lambda_0 / j2_theta.powf(0.75) * r.powf(1.5);
        let denom = 1.0 + 2.0 * lambda + 2.0 * lambda * lambda;
        let big_lambda_1 = (1.0 + lambda) / denom;
        let big_lambda_2 = lambda / denom;

        let four_ninths_alpha = 4.0 / 9.0 * absorptivity;

        let t1_radial = big_lambda_1 * psi_x;
        let t1_pole = (psi_z - big_lambda_1 * psi_x) * r_dot_s;
        accel += (four_ninths_alpha * scale) * (t1_radial * r_hat + t1_pole * spin_pole);

        let t2_coeff = -four_ninths_alpha * scale * big_lambda_2 * psi_x;
        accel += t2_coeff * r_hat.cross(spin_pole);
    }
    accel
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
    let sigma = f64::midpoint(1.0, e2 / (2.0 * eta) * log_term);
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

    fn n_free_params(&self) -> usize {
        2
    }

    fn free_param_names(&self) -> Vec<&'static str> {
        vec!["a_over_m", "lambda_0"]
    }

    fn lower_bounds(&self) -> Vec<Option<f64>> {
        vec![Some(0.0), Some(0.0)]
    }

    fn accel(
        &self,
        _time: Time<TDB>,
        pos: &Vector<Equatorial>,
        _vel: &Vector<Equatorial>,
        free_params: &[f64],
    ) -> KeteResult<Vector<Equatorial>> {
        let accel = radiation_accel(
            &(*pos).into(),
            &self.spin_pole.into(),
            self.albedo,
            self.absorptivity,
            self.flattening,
            free_params[0],
            free_params[1],
        );
        Ok(Vector::<Equatorial>::new(accel.into()))
    }

    fn parameter_jacobian(
        &self,
        _time: Time<TDB>,
        pos: &Vector<Equatorial>,
        _vel: &Vector<Equatorial>,
        free_params: &[f64],
    ) -> KeteResult<Matrix3xX<f64>> {
        let a_over_m = free_params[0];
        let lambda_0 = free_params[1];
        let e = self.flattening;

        // Geometry mirrors `accel`; the FD-consistency test guards the two
        // copies against drifting apart.
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

        // The whole acceleration is linear in `a_over_m`, so its column is the
        // acceleration evaluated per unit `a_over_m`.
        let unit_scale = F0_OVER_C_AU_DAY2 * g;
        let four_ninths_a0 = 4.0 / 9.0 * self.albedo;
        let srp_radial = j2_theta + four_ninths_a0 * psi_x;
        let srp_pole = four_ninths_a0 * (psi_z - psi_x) * r_dot_s;
        let mut d_a_over_m = unit_scale * (srp_radial * r_hat + srp_pole * s_hat);

        let mut d_lambda_0 = Vector3::zeros();
        if self.absorptivity > 0.0 && lambda_0 >= 0.0 {
            // `lambda = lambda_0 * c` with `c` position-only, so
            // `d/d(lambda_0) = c * d/d(lambda)`.
            let c = r.powf(1.5) / j2_theta.powf(0.75);
            let lambda = lambda_0 * c;
            let denom = 1.0 + 2.0 * lambda + 2.0 * lambda * lambda;
            let big_lambda_1 = (1.0 + lambda) / denom;
            let big_lambda_2 = lambda / denom;

            let four_ninths_alpha = 4.0 / 9.0 * self.absorptivity;

            let t1_radial = big_lambda_1 * psi_x;
            let t1_pole = (psi_z - big_lambda_1 * psi_x) * r_dot_s;
            d_a_over_m += (four_ninths_alpha * unit_scale) * (t1_radial * r_hat + t1_pole * s_hat);
            d_a_over_m -=
                (four_ninths_alpha * unit_scale * big_lambda_2 * psi_x) * r_hat.cross(&s_hat);

            // d(Lambda_1)/d(lambda) and d(Lambda_2)/d(lambda).
            let inv_denom2 = (denom * denom).recip();
            let d_big_1 = -(1.0 + 4.0 * lambda + 2.0 * lambda * lambda) * inv_denom2;
            let d_big_2 = (1.0 - 2.0 * lambda * lambda) * inv_denom2;

            d_lambda_0 = (a_over_m * four_ninths_alpha * unit_scale * psi_x * c)
                * (d_big_1 * (r_hat - r_dot_s * s_hat) - d_big_2 * r_hat.cross(&s_hat));
        }

        let mut out = Matrix3xX::<f64>::zeros(2);
        out.set_column(0, &d_a_over_m);
        out.set_column(1, &d_lambda_0);
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn force() -> FarnocchiaNonGrav {
        FarnocchiaNonGrav::new(0.15, 0.9, 0.9, Vector::<Equatorial>::new([0.2, -0.3, 0.93]))
            .unwrap()
    }

    fn pos() -> Vector<Equatorial> {
        Vector::<Equatorial>::new([1.2, -0.4, 0.2])
    }
    fn vel() -> Vector<Equatorial> {
        Vector::<Equatorial>::new([0.003, 0.011, -0.001])
    }
    fn epoch() -> Time<TDB> {
        Time::<TDB>::new(2_451_545.0)
    }

    fn accel_at(f: &FarnocchiaNonGrav, params: &[f64]) -> Vector3<f64> {
        f.accel(epoch(), &pos(), &vel(), params).unwrap().into()
    }

    fn jac_col(f: &FarnocchiaNonGrav, params: &[f64], col: usize) -> Vector3<f64> {
        let jac = f
            .parameter_jacobian(epoch(), &pos(), &vel(), params)
            .unwrap();
        Vector3::new(jac[(0, col)], jac[(1, col)], jac[(2, col)])
    }

    /// The other side of the `d(accel)/d(vel)` default. This model is a radiation force
    /// depending on position and spin pole alone, so a zero velocity derivative is the
    /// right answer, and differencing must produce it **exactly** rather than as noise:
    /// the acceleration is bit-identical at the perturbed velocities, so the differences
    /// cancel to zero rather than to something small.
    ///
    /// This is what lets the trait default difference the velocity block unconditionally
    /// without injecting noise into the forces that do not need it.
    #[test]
    fn velocity_jacobian_is_exactly_zero() {
        let f = force();
        let params = [1.0, 0.5];
        let (da_dr, da_dv) = f.jacobians(epoch(), &pos(), &vel(), &params).unwrap();
        assert!(
            da_dv.iter().all(|entry| *entry == 0.0),
            "a velocity-independent force differenced to a nonzero velocity jacobian: \
             {da_dv:?}"
        );
        assert!(
            da_dr.norm() > 0.0,
            "the position jacobian should not be zero"
        );
    }

    #[test]
    fn analytic_jacobian_matches_finite_difference() {
        let f = force();
        let params = [2e-5, 0.7];
        let steps = [1e-9, 1e-6];
        for col in 0..2 {
            let analytic = jac_col(&f, &params, col);
            let mut p = params;
            p[col] += steps[col];
            let a_plus = accel_at(&f, &p);
            p[col] = params[col] - steps[col];
            let a_minus = accel_at(&f, &p);
            let fd = (a_plus - a_minus) / (2.0 * steps[col]);
            let scale = fd.norm();
            assert!(
                (analytic - fd).norm() < 1e-6 * scale,
                "column {col}: analytic {analytic:?} vs fd {fd:?}"
            );
        }
    }

    #[test]
    fn continuous_at_zero_lag() {
        let f = force();
        let a_zero = accel_at(&f, &[1e-5, 0.0]);
        let a_eps = accel_at(&f, &[1e-5, 1e-12]);
        assert!(
            (a_zero - a_eps).norm() < 1e-9 * a_zero.norm(),
            "zero-lag limit is discontinuous"
        );

        // The zero-lag limit keeps the radial recoil from instantaneous
        // re-emission; disabling thermal entirely (absorptivity = 0) must
        // differ from lambda_0 = 0.
        let f_no_thermal =
            FarnocchiaNonGrav::new(0.15, 0.0, 0.9, Vector::<Equatorial>::new([0.2, -0.3, 0.93]))
                .unwrap();
        let a_srp_only = accel_at(&f_no_thermal, &[1e-5, 0.0]);
        assert!(
            (a_zero - a_srp_only).norm() > 1e-3 * a_zero.norm(),
            "lambda_0 = 0 should retain the radial thermal recoil"
        );
    }

    #[test]
    fn jacobian_well_defined_at_fit_start() {
        let f = force();

        // Both free parameters at the fit's 0 starting point: the a_over_m
        // column is the finite unit acceleration; the lambda_0 column is
        // exactly zero because the whole force scales with a_over_m.
        let col_am = jac_col(&f, &[0.0, 0.0], 0);
        let col_l0 = jac_col(&f, &[0.0, 0.0], 1);
        assert!(col_am.norm().is_finite() && col_am.norm() > 0.0);
        assert_eq!(col_l0.norm(), 0.0);

        // Once a_over_m is off zero the lambda_0 column is finite and matches
        // a forward difference (valid now that the force is continuous at 0).
        let params = [1e-5, 0.0];
        let analytic = jac_col(&f, &params, 1);
        let h = 1e-8;
        let fd = (accel_at(&f, &[1e-5, h]) - accel_at(&f, &params)) / h;
        assert!(
            (analytic - fd).norm() < 1e-6 * analytic.norm(),
            "lambda_0 partial at 0: analytic {analytic:?} vs fd {fd:?}"
        );
    }
}
