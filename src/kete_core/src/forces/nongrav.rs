// BSD 3-Clause License
//
// Copyright (c) 2026, Dar Dahlen
// Copyright (c) 2025, California Institute of Technology
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

use nalgebra::Vector3;
use pathfinding::num_traits::Zero;

use crate::constants::{
    C_AU_PER_DAY_INV_SQUARED, F0_OVER_C_AU_DAY2, GMS, SOLAR_FLUX, STEFAN_BOLTZMANN,
};
use crate::errors::{Error, KeteResult};
use crate::frames::{Equatorial, Vector};

use crate::kepler::analytic_2_body;

/// Non-Gravitational models.
/// These are used during integration to model non-gravitational forces on particles in
/// the solar system.
#[derive(Debug, Clone)]
pub enum NonGravModel {
    /// JPL's non-gravitational forces are modeled as defined on page 139 of the
    /// Comets II textbook.
    ///
    /// This model adds 3 "A" terms to the acceleration which the object feels. These
    /// A terms represent additional radial, tangential, and normal forces on the
    /// object.
    ///
    /// The additional acceleration is:
    /// `accel_additional = A_1 * g(r) * r_vec + A_2 * g(r) * t_vec + A_3 * g(r) * n_vec`
    ///
    /// where `r_vec`, `t_vec`, and `n_vec` are the radial, tangential, and normal unit
    /// vectors.
    ///
    /// The `g(r)` function is defined as:
    /// `g(r) = alpha * (r / r_0)^(-m) * (1 + (r / r_0)^n)^(-k)`
    ///
    /// When `alpha = 1.0`, `n = 0.0`, `k = 0.0`, `r_0 = 1.0`, and `m = 2.0`, this
    /// reduces to a `1/r^2` dependence.
    ///
    JplComet {
        /// Constant for the radial non-gravitational force.
        a1: f64,
        /// Constant for the tangential non-gravitational force.
        a2: f64,
        /// Constant for the normal non-gravitational force.
        a3: f64,
        /// Coefficients for the g(r) function defined above.
        alpha: f64,
        /// Coefficients for the g(r) function defined above.
        r_0: f64,
        /// Coefficients for the g(r) function defined above.
        m: f64,
        /// Coefficients for the g(r) function defined above.
        n: f64,
        /// Coefficients for the g(r) function defined above.
        k: f64,
        /// Time delay for the forces, this is applied by propagating the object to the
        /// specified delay before computing the forces.
        dt: f64,
    },

    /// Dust model, including Solar Radiation Pressure (SRP) and the Poynting-Robertson
    /// effect.
    ///
    /// SRP acts as an effective reduction in the gravitational force of the
    /// Sun, reducing the central acceleration force in the radial direction.
    ///
    /// Poynting-Robertson acts as a drag force, in the opposite direction of motion.
    Dust {
        /// Beta Parameter
        beta: f64,
    },

    /// Physical radiation force model from Farnocchia et al. 2025
    /// ("Radiation Forces and Trajectory of Hayabusa2# Target 1998 KY26",
    /// `ApJL` 993:L9).
    ///
    /// Models the body as an oblate spheroid with a fixed spin pole and
    /// computes solar radiation pressure plus thermal recoil (Yarkovsky)
    /// acceleration from first principles.
    ///
    /// Internally stores only the two fittable parameters in the form used
    /// by the paper: `a_over_m` (Eq. 6) and `lambda_0` (Eq. 12). Fitting on
    /// these directly is far better conditioned than fitting on
    /// `(density, thermal_inertia)`: `A/M` multiplies the entire
    /// acceleration linearly, and `lambda_0` enters the thermal terms only
    /// through the dimensionless lag parameter `lambda`.
    FarnocchiaModel {
        /// Area-to-mass ratio (`m^2 / kg`). Primary fittable parameter
        /// (Eq. 6: `A/M = 3 / (4 * rho * R_P)`).
        a_over_m: f64,

        /// Dimensionless thermal parameter at 1 AU (Eq. 12). Secondary
        /// fittable parameter.
        lambda_0: f64,

        /// Geometric albedo `a_0` (Lambert approximation). Enters SRP only.
        albedo: f64,

        /// Absorptivity `alpha = 1 - A_B`, where `A_B` is the Bond albedo.
        /// Multiplies the thermal terms.
        absorptivity: f64,

        /// Axis ratio `e = R_P / R_E` (`1.0` for a sphere, `< 1` for oblate).
        flattening: f64,

        /// Spin pole unit vector in the equatorial (integration) frame,
        /// fixed in inertial space. The constructor accepts any inertial
        /// frame and converts to equatorial for storage so no per-step
        /// frame conversion is needed during propagation.
        spin_pole: Vector<Equatorial>,
    },
}

impl NonGravModel {
    /// Construct a new non-grav model, manually specifying all parameters.
    /// Consider using the other constructors if this is a simple object.
    #[allow(clippy::too_many_arguments, reason = "Not practical to avoid this")]
    #[must_use]
    pub fn new_jpl(
        a1: f64,
        a2: f64,
        a3: f64,
        alpha: f64,
        r_0: f64,
        m: f64,
        n: f64,
        k: f64,
        dt: f64,
    ) -> Self {
        Self::JplComet {
            a1,
            a2,
            a3,
            alpha,
            r_0,
            m,
            n,
            k,
            dt,
        }
    }

    /// Construct a new non-grav dust model.
    #[must_use]
    pub fn new_dust(beta: f64) -> Self {
        Self::Dust { beta }
    }

    /// Construct a `FarnocchiaModel` from the two fittable parameters
    /// (`a_over_m`, `lambda_0`) and the fixed surface descriptors.
    ///
    /// The fittable parameters are the ones used in Farnocchia et al. 2025
    /// (Eq. 6 and Eq. 12). To compute them from physical surface inputs
    /// (`density`, `thermal_inertia`, etc.), use the free helpers
    /// [`a_over_m_from_physical`] and [`lambda_0_from_physical`].
    ///
    /// # Errors
    /// Returns an error if `a_over_m` is non-finite or non-positive, if
    /// `lambda_0` is non-finite or negative, if any of `albedo`,
    /// `absorptivity`, `flattening` is non-finite or negative, or if
    /// `flattening` is greater than `1`.
    pub fn new_farnocchia(
        a_over_m: f64,
        lambda_0: f64,
        albedo: f64,
        absorptivity: f64,
        flattening: f64,
        spin_pole: Vector<Equatorial>,
    ) -> KeteResult<Self> {
        if !a_over_m.is_finite() || a_over_m <= 0.0 {
            return Err(Error::ValueError(format!(
                "FarnocchiaModel non-grav: 'a_over_m' must be finite and > 0 (got {a_over_m})"
            )));
        }
        for (name, v) in [
            ("lambda_0", lambda_0),
            ("albedo", albedo),
            ("absorptivity", absorptivity),
            ("flattening", flattening),
        ] {
            if !v.is_finite() || v < 0.0 {
                return Err(Error::ValueError(format!(
                    "FarnocchiaModel non-grav: '{name}' must be finite and >= 0 (got {v})"
                )));
            }
        }
        if flattening > 1.0 {
            return Err(Error::ValueError(format!(
                "FarnocchiaModel non-grav: 'flattening' must be <= 1 (got {flattening})"
            )));
        }
        // Spin pole is stored in the equatorial (integration) frame so that
        // pos/vel and spin_pole share a frame inside `add_acceleration`.
        let spin_pole = spin_pole.into_frame::<Equatorial>();
        if !spin_pole.is_finite() || spin_pole.norm() == 0.0 {
            return Err(Error::ValueError(
                "FarnocchiaModel non-grav: 'spin_pole' must be a finite, non-zero vector".into(),
            ));
        }
        // Normalize so the equations only see a unit pole regardless of input.
        let spin_pole = spin_pole.normalize();
        Ok(Self::FarnocchiaModel {
            a_over_m,
            lambda_0,
            albedo,
            absorptivity,
            flattening,
            spin_pole,
        })
    }

    /// Number of free (solvable) parameters in this model.
    ///
    /// For `JplComet`, only finite A terms are counted as free; NaN marks a
    /// term as absent (held at zero). `Dust` always has 1 free parameter.
    /// `FarnocchiaModel` has 2 free parameters (`a_over_m`, `lambda_0`).
    #[must_use]
    pub fn n_free_params(&self) -> usize {
        match self {
            Self::JplComet { a1, a2, a3, .. } => {
                [a1, a2, a3].iter().filter(|v| v.is_finite()).count()
            }
            Self::Dust { .. } => 1,
            Self::FarnocchiaModel { .. } => 2,
        }
    }

    /// Return the free parameters as a vector.
    ///
    /// For `JplComet`, only finite A terms are returned; NaN-valued terms are
    /// excluded.
    #[must_use]
    pub fn get_free_params(&self) -> Vec<f64> {
        match self {
            Self::JplComet { a1, a2, a3, .. } => [*a1, *a2, *a3]
                .into_iter()
                .filter(|v| v.is_finite())
                .collect(),
            Self::Dust { beta } => vec![*beta],
            Self::FarnocchiaModel {
                a_over_m, lambda_0, ..
            } => vec![*a_over_m, *lambda_0],
        }
    }

    /// Names of the free (solvable) parameters.
    ///
    /// The order matches [`get_free_params`](Self::get_free_params) and
    /// [`set_free_params`](Self::set_free_params).
    ///
    /// For `JplComet`, only names for finite A terms are returned.
    #[must_use]
    pub fn param_names(&self) -> Vec<&str> {
        match self {
            Self::JplComet { a1, a2, a3, .. } => {
                const NAMES: [&str; 3] = ["a1", "a2", "a3"];
                [a1, a2, a3]
                    .iter()
                    .zip(NAMES)
                    .filter(|(v, _)| v.is_finite())
                    .map(|(_, name)| name)
                    .collect()
            }
            Self::Dust { .. } => vec!["beta"],
            Self::FarnocchiaModel { .. } => vec!["a_over_m", "lambda_0"],
        }
    }

    /// Lower bounds for the free parameters (matching
    /// [`get_free_params`](Self::get_free_params)).
    ///
    /// `f64::NEG_INFINITY` indicates no lower bound. These are used by
    /// the orbit fitter to reject trial steps that would otherwise be
    /// silently clamped by [`set_free_params`](Self::set_free_params), which would pin
    /// the optimizer at the boundary.
    #[must_use]
    pub fn param_lower_bounds(&self) -> Vec<f64> {
        match self {
            Self::JplComet { a1, a2, a3, .. } => [a1, a2, a3]
                .iter()
                .filter(|v| v.is_finite())
                .map(|_| f64::NEG_INFINITY)
                .collect(),
            Self::Dust { .. } => vec![f64::NEG_INFINITY],
            // a_over_m must remain strictly positive; lambda_0 must
            // remain >= 0 (zero turns off the thermal component).
            Self::FarnocchiaModel { .. } => vec![0.0, 0.0],
        }
    }

    /// Update the free parameters from a slice.
    ///
    /// Only finite (non-NaN) A terms are updated for `JplComet`.
    ///
    /// # Panics
    /// Panics if the slice length does not match `n_free_params()`.
    pub fn set_free_params(&mut self, params: &[f64]) {
        match self {
            Self::JplComet { a1, a2, a3, .. } => {
                let n = [&*a1, &*a2, &*a3].iter().filter(|v| v.is_finite()).count();
                assert!(
                    params.len() == n,
                    "JplComet requires {} params, got {}",
                    n,
                    params.len()
                );
                let mut iter = params.iter();
                for val in [a1, a2, a3] {
                    if val.is_finite() {
                        *val = *iter.next().unwrap();
                    }
                }
            }
            Self::Dust { beta } => {
                assert!(
                    params.len() == 1,
                    "Dust requires 1 param, got {}",
                    params.len()
                );
                *beta = params[0];
            }
            Self::FarnocchiaModel {
                a_over_m, lambda_0, ..
            } => {
                assert!(
                    params.len() == 2,
                    "FarnocchiaModel requires 2 params, got {}",
                    params.len()
                );
                // A/M must remain strictly positive; lambda_0 must remain
                // non-negative (zero turns off the thermal component).
                *a_over_m = params[0].max(1e-30);
                *lambda_0 = params[1].max(0.0);
            }
        }
    }

    /// Construct a new non-grav model which follows the default comet drop-off.
    #[must_use]
    pub fn new_jpl_comet_default(a1: f64, a2: f64, a3: f64) -> Self {
        Self::JplComet {
            a1,
            a2,
            a3,
            alpha: 0.1112620426,
            r_0: 2.808,
            m: 2.15,
            n: 5.093,
            k: 4.6142,
            dt: 0.0,
        }
    }

    /// Compute the non-gravitational acceleration vector when provided the position
    /// and velocity vector with respect to the sun.
    ///
    /// # Panics
    /// Panics when two body propagation fails.
    #[inline(always)]
    pub fn add_acceleration(
        &self,
        accel: &mut Vector3<f64>,
        pos: &Vector3<f64>,
        vel: &Vector3<f64>,
    ) {
        match self {
            Self::Dust { beta } => {
                let pos_norm = pos.normalize();
                let r_dot = &pos_norm.dot(vel);
                let norm2_inv = pos.norm_squared().recip();
                let scaling = GMS * beta * norm2_inv;
                *accel += scaling
                    * ((1.0 - r_dot * C_AU_PER_DAY_INV_SQUARED) * pos_norm
                        - vel * C_AU_PER_DAY_INV_SQUARED);
            }

            Self::JplComet {
                a1,
                a2,
                a3,
                alpha,
                r_0,
                m,
                n,
                k,
                dt,
            } => {
                let mut pos = *pos;
                let pos_norm = pos.normalize();
                let t_vec = (vel - pos_norm * vel.dot(&pos_norm)).normalize();

                // normalized by construction (R x T = normal in angular momentum direction)
                let n_vec = pos_norm.cross(&t_vec);

                if !dt.is_zero() {
                    (pos, _) = analytic_2_body((-dt).into(), &pos, vel, None).unwrap();
                }
                let rr0 = pos.norm() / r_0;
                let scale = alpha * rr0.powf(-m) * (1.0 + rr0.powf(*n)).powf(-k);
                // NaN A terms are absent -- treat as zero.
                let a1_eff = if a1.is_finite() { *a1 } else { 0.0 };
                let a2_eff = if a2.is_finite() { *a2 } else { 0.0 };
                let a3_eff = if a3.is_finite() { *a3 } else { 0.0 };
                *accel += pos_norm * (scale * a1_eff);
                *accel += t_vec * (scale * a2_eff);
                *accel += n_vec * (scale * a3_eff);
            }

            Self::FarnocchiaModel {
                a_over_m,
                lambda_0,
                albedo,
                absorptivity,
                flattening,
                spin_pole,
                ..
            } => {
                let e = *flattening;

                // pos/vel are Sun-relative in the equatorial (integration)
                // frame; `spin_pole` is stored pre-converted to equatorial
                // and pre-normalized at construction.
                let s_hat: Vector3<f64> = (*spin_pole).into();
                let r = pos.norm();
                let r_inv = r.recip();
                let r_hat = pos * r_inv;

                // g(r) = 1/r^2 with r in AU.
                let g = r_inv * r_inv;

                // Shape factors (oblate spheroid with axis ratio e = R_P/R_E).
                let (psi_x, psi_z, _sigma_s) = shape_factors(e);

                // Subsolar colatitude theta_0: cos(theta_0) = -r_hat . s_hat.
                let r_dot_s = r_hat.dot(&s_hat);
                let cos_theta_0 = -r_dot_s;
                let sin2_theta_0 = (1.0 - cos_theta_0 * cos_theta_0).max(0.0);
                // J_2(theta_0) = sqrt(e^2 sin^2 + cos^2). Always >= e for e<=1.
                let j2_theta = (e * e * sin2_theta_0 + cos_theta_0 * cos_theta_0).sqrt();

                // Common scale factor (A/M) * (F_0/c) * g(r), units AU/Day^2.
                let scale = a_over_m * F0_OVER_C_AU_DAY2 * g;

                // SRP (Eq. 5).
                let four_ninths_a0 = 4.0 / 9.0 * albedo;
                let srp_radial = j2_theta + four_ninths_a0 * psi_x;
                let srp_pole = four_ninths_a0 * (psi_z - psi_x) * r_dot_s;
                *accel += scale * (srp_radial * r_hat + srp_pole * s_hat);

                // Thermal terms vanish if the body neither absorbs light
                // (alpha = 0) nor has any thermal lag (lambda_0 = 0).
                if *absorptivity > 0.0 && *lambda_0 > 0.0 {
                    // Thermal lag parameter lambda(r, theta_0). j2_theta > 0
                    // for any e > 0, so the division is safe.
                    let lambda = lambda_0 / j2_theta.powf(0.75) * r.powf(1.5);
                    let denom = 1.0 + 2.0 * lambda + 2.0 * lambda * lambda;
                    let big_lambda_1 = (1.0 + lambda) / denom;
                    let big_lambda_2 = lambda / denom;

                    let four_ninths_alpha = 4.0 / 9.0 * absorptivity;

                    // Thermal T1 (Eq. 7).
                    let t1_radial = big_lambda_1 * psi_x;
                    let t1_pole = (psi_z - big_lambda_1 * psi_x) * r_dot_s;
                    *accel += (four_ninths_alpha * scale) * (t1_radial * r_hat + t1_pole * s_hat);

                    // Thermal T2 / Yarkovsky (Eq. 8).
                    let t2_coeff = -four_ninths_alpha * scale * big_lambda_2 * psi_x;
                    *accel += t2_coeff * r_hat.cross(&s_hat);
                }
            }
        }
    }
}

/// Compute `A/M` (`m^2 / kg`) from physical inputs (Farnocchia 2025 Eq. 6).
///
/// `density` is in kg/m^3, `diameter` in km, `flattening` is the axis ratio
/// `R_P / R_E` (1.0 for a sphere).
#[must_use]
pub fn a_over_m_from_physical(density: f64, diameter: f64, flattening: f64) -> f64 {
    // diameter is in km; convert to m so A/M comes out in m^2/kg.
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

/// Compute `lambda_0` (dimensionless, Eq. 12) from physical inputs.
///
/// `thermal_inertia` is in SI units (J m^-2 K^-1 s^-1/2), `rotation_period`
/// is in hours, `flattening` is the axis ratio `R_P / R_E`.
#[must_use]
pub fn lambda_0_from_physical(
    thermal_inertia: f64,
    emissivity: f64,
    absorptivity: f64,
    flattening: f64,
    rotation_period: f64,
) -> f64 {
    let sigma = sigma_shape(flattening);
    let denom = (emissivity * STEFAN_BOLTZMANN).powf(0.25) * (absorptivity * SOLAR_FLUX).powf(0.75);
    // rotation_period is in hours; convert to seconds.
    thermal_inertia * sigma.powf(0.75) / denom
        * (std::f64::consts::PI / (2.0 * rotation_period * 3600.0)).sqrt()
}

/// Inverse of [`lambda_0_from_physical`]: solve for thermal inertia
/// (SI units) given `lambda_0` and the auxiliary surface inputs.
#[must_use]
pub fn thermal_inertia_from_lambda_0(
    lambda_0: f64,
    emissivity: f64,
    absorptivity: f64,
    flattening: f64,
    rotation_period: f64,
) -> f64 {
    let sigma = sigma_shape(flattening);
    let numer = (emissivity * STEFAN_BOLTZMANN).powf(0.25) * (absorptivity * SOLAR_FLUX).powf(0.75);
    lambda_0 * numer
        / (sigma.powf(0.75) * (std::f64::consts::PI / (2.0 * rotation_period * 3600.0)).sqrt())
}

/// Oblate-spheroid shape factors `(psi_X, psi_Z, Sigma)` from Eqs. 2-4 of
/// Farnocchia et al. 2025. All three approach `1` in the spherical limit
/// `e -> 1`.
fn shape_factors(e: f64) -> (f64, f64, f64) {
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

/// `Sigma` shape factor only (Eq. 4). Used by the `lambda_0` constructor
/// helpers without recomputing the unused `psi_X`/`psi_Z` factors.
fn sigma_shape(e: f64) -> f64 {
    if e >= 1.0 - 1e-9 {
        return 1.0;
    }
    let e2 = e * e;
    let eta = (1.0 - e2).sqrt();
    let log_term = ((1.0 + eta) / (1.0 - eta)).ln();
    0.5 * (1.0 + e2 / (2.0 * eta) * log_term)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: compute the acceleration from a non-grav model at a given pos/vel.
    fn eval(model: &NonGravModel, pos: Vector3<f64>, vel: Vector3<f64>) -> Vector3<f64> {
        let mut accel = Vector3::zeros();
        model.add_acceleration(&mut accel, &pos, &vel);
        accel
    }

    // -- JplComet: direction tests ------------------------------------

    /// Use 1/r^2 model (alpha=1, m=2, n=0, k=0, r0=1) at r=1 AU so g(r)=1,
    /// making the acceleration equal to the A coefficients in the RTN directions.
    fn simple_model(a1: f64, a2: f64, a3: f64) -> NonGravModel {
        NonGravModel::new_jpl(a1, a2, a3, 1.0, 1.0, 2.0, 0.0, 0.0, 0.0)
    }

    /// Circular prograde orbit in the xy-plane at 1 AU:
    ///   R = (1,0,0), T = (0,1,0), N = R x T = (0,0,1)
    fn circular_prograde() -> (Vector3<f64>, Vector3<f64>) {
        (Vector3::new(1.0, 0.0, 0.0), Vector3::new(0.0, 1.0, 0.0))
    }

    #[test]
    fn jpl_a1_radial_direction() {
        let (pos, vel) = circular_prograde();
        let accel = eval(&simple_model(1.0, 0.0, 0.0), pos, vel);
        // A1 only -> acceleration along +x (radial)
        assert!((accel.x - 1.0).abs() < 1e-14);
        assert!(accel.y.abs() < 1e-14);
        assert!(accel.z.abs() < 1e-14);
    }

    #[test]
    fn jpl_a2_tangential_direction() {
        let (pos, vel) = circular_prograde();
        let accel = eval(&simple_model(0.0, 1.0, 0.0), pos, vel);
        // A2 only -> acceleration along +y (tangential)
        assert!(accel.x.abs() < 1e-14);
        assert!((accel.y - 1.0).abs() < 1e-14);
        assert!(accel.z.abs() < 1e-14);
    }

    #[test]
    fn jpl_a3_normal_direction() {
        let (pos, vel) = circular_prograde();
        let accel = eval(&simple_model(0.0, 0.0, 1.0), pos, vel);
        // A3 only -> acceleration along +z (angular momentum / normal)
        assert!(accel.x.abs() < 1e-14);
        assert!(accel.y.abs() < 1e-14);
        assert!((accel.z - 1.0).abs() < 1e-14);
    }

    #[test]
    fn jpl_combined_rtn() {
        let (pos, vel) = circular_prograde();
        let accel = eval(&simple_model(2.0, 3.0, 5.0), pos, vel);
        assert!((accel.x - 2.0).abs() < 1e-14);
        assert!((accel.y - 3.0).abs() < 1e-14);
        assert!((accel.z - 5.0).abs() < 1e-14);
    }

    // -- JplComet: g(r) scaling -------------------------------------

    #[test]
    fn jpl_gr_inverse_square() {
        // 1/r^2 model at r = 2 AU -> g(r) = 1/4
        let model = simple_model(1.0, 0.0, 0.0);
        let pos = Vector3::new(2.0, 0.0, 0.0);
        let vel = Vector3::new(0.0, 1.0, 0.0);
        let accel = eval(&model, pos, vel);
        assert!((accel.x - 0.25).abs() < 1e-14);
    }

    #[test]
    fn jpl_gr_default_comet() {
        // Default comet g(r) at r = r_0 = 2.808:
        // g(r_0) = alpha * 1^(-m) * (1 + 1^n)^(-k) = alpha * 2^(-k)
        let model = NonGravModel::new_jpl_comet_default(1.0, 0.0, 0.0);
        let r0 = 2.808;
        let pos = Vector3::new(r0, 0.0, 0.0);
        let vel = Vector3::new(0.0, 1.0, 0.0);
        let accel = eval(&model, pos, vel);
        let expected = 0.1112620426 * 2.0_f64.powf(-4.6142);
        assert!((accel.x - expected).abs() < 1e-12);
    }

    // -- JplComet: oblique orbit -------------------------------------

    #[test]
    fn jpl_oblique_orbit() {
        // Object at (1, 0, 0) moving at 45 deg out of the ecliptic: vel = (0, 1, 1)/sqrt(2)
        let pos = Vector3::new(1.0, 0.0, 0.0);
        let vel = Vector3::new(0.0, 1.0, 1.0).normalize();
        let model = simple_model(0.0, 0.0, 1.0);
        let accel = eval(&model, pos, vel);

        // T is vel projected perp to R, normalized = (0, 1, 1)/sqrt(2)
        // N = R x T = (1,0,0) x (0,1,1)/sqrt(2) = (0, -1, 1)/sqrt(2)
        let expected_n = Vector3::new(0.0, -1.0, 1.0).normalize();
        assert!((accel - expected_n).norm() < 1e-14);
    }

    // -- Dust model --------------------------------------------------

    #[test]
    fn dust_purely_radial_when_stationary() {
        // v = 0 -> PR term vanishes, only SRP remains: a = GM_s * beta / r^2 * r_hat
        let model = NonGravModel::new_dust(0.5);
        let pos = Vector3::new(1.0, 0.0, 0.0);
        let vel = Vector3::zeros();
        let accel = eval(&model, pos, vel);
        let expected = GMS * 0.5; // r=1, r_hat=(1,0,0)
        assert!((accel.x - expected).abs() < 1e-12);
        assert!(accel.y.abs() < 1e-14);
        assert!(accel.z.abs() < 1e-14);
    }

    #[test]
    fn dust_pr_drag_opposes_velocity() {
        // At r=1, vel tangential -> PR drag component should be negative along vel
        let model = NonGravModel::new_dust(1.0);
        let pos = Vector3::new(1.0, 0.0, 0.0);
        let vel = Vector3::new(0.0, 0.01, 0.0); // slow tangential motion
        let accel = eval(&model, pos, vel);
        // The tangential (y) acceleration should be negative (drag)
        assert!(accel.y < 0.0);
    }

    #[test]
    fn dust_scales_with_beta() {
        let pos = Vector3::new(1.0, 0.0, 0.0);
        let vel = Vector3::new(0.0, 0.01, 0.0);
        let a1 = eval(&NonGravModel::new_dust(1.0), pos, vel);
        let a2 = eval(&NonGravModel::new_dust(2.0), pos, vel);
        assert!((a2 - 2.0 * a1).norm() < 1e-14);
    }

    #[test]
    fn dust_scales_with_inverse_r_squared() {
        let vel = Vector3::zeros();
        let a1 = eval(
            &NonGravModel::new_dust(1.0),
            Vector3::new(1.0, 0.0, 0.0),
            vel,
        );
        let a2 = eval(
            &NonGravModel::new_dust(1.0),
            Vector3::new(2.0, 0.0, 0.0),
            vel,
        );
        // At 2 AU the radial acceleration should be 1/4 of that at 1 AU
        assert!((a2.x / a1.x - 0.25).abs() < 1e-14);
    }

    // -- NaN sentinel for absent A terms ----------------------------

    #[test]
    fn nan_a_terms_excluded_from_free_params() {
        // Only a2 is finite
        let model = NonGravModel::new_jpl(f64::NAN, 1e-8, f64::NAN, 1.0, 1.0, 2.0, 0.0, 0.0, 0.0);
        assert_eq!(model.n_free_params(), 1);
        assert_eq!(model.get_free_params(), vec![1e-8]);
        assert_eq!(model.param_names(), vec!["a2"]);
    }

    #[test]
    fn nan_all_finite_unchanged() {
        let model = NonGravModel::new_jpl(1.0, 2.0, 3.0, 1.0, 1.0, 2.0, 0.0, 0.0, 0.0);
        assert_eq!(model.n_free_params(), 3);
        assert_eq!(model.get_free_params(), vec![1.0, 2.0, 3.0]);
        assert_eq!(model.param_names(), vec!["a1", "a2", "a3"]);
    }

    #[test]
    fn nan_all_nan_zero_params() {
        let model =
            NonGravModel::new_jpl(f64::NAN, f64::NAN, f64::NAN, 1.0, 1.0, 2.0, 0.0, 0.0, 0.0);
        assert_eq!(model.n_free_params(), 0);
        assert!(model.get_free_params().is_empty());
        assert!(model.param_names().is_empty());
    }

    #[test]
    fn set_free_params_respects_nan() {
        let mut model = NonGravModel::new_jpl(0.0, f64::NAN, 0.0, 1.0, 1.0, 2.0, 0.0, 0.0, 0.0);
        // Only a1 and a3 are free
        model.set_free_params(&[5.0, 7.0]);
        assert_eq!(model.get_free_params(), vec![5.0, 7.0]);
        assert_eq!(model.param_names(), vec!["a1", "a3"]);
    }

    #[test]
    fn nan_a_terms_produce_zero_acceleration() {
        let (pos, vel) = circular_prograde();
        let model =
            NonGravModel::new_jpl(f64::NAN, f64::NAN, f64::NAN, 1.0, 1.0, 2.0, 0.0, 0.0, 0.0);
        let accel = eval(&model, pos, vel);
        assert!(accel.x.abs() < 1e-14);
        assert!(accel.y.abs() < 1e-14);
        assert!(accel.z.abs() < 1e-14);
    }

    #[test]
    fn nan_partial_acceleration() {
        // Only a2 active, so only tangential acceleration
        let (pos, vel) = circular_prograde();
        let model = NonGravModel::new_jpl(f64::NAN, 3.0, f64::NAN, 1.0, 1.0, 2.0, 0.0, 0.0, 0.0);
        let accel = eval(&model, pos, vel);
        assert!(accel.x.abs() < 1e-14);
        assert!((accel.y - 3.0).abs() < 1e-14);
        assert!(accel.z.abs() < 1e-14);
    }

    // -- FarnocchiaModel (Farnocchia 2025) --------------------------------

    /// Reference physical parameters used by several tests. Produces an
    /// `a_over_m` of 3 / (4 * density * `R_P`) and a `lambda_0` derived from
    /// the surface thermal inertia.
    fn sphere_physical(
        density: f64,
        thermal_inertia: f64,
        albedo: f64,
        absorptivity: f64,
    ) -> NonGravModel {
        let diameter = 0.011_f64; // km, ~11 m
        let flattening = 1.0_f64;
        let emissivity = 0.9_f64;
        let rotation_period = 1.0_f64; // hours
        let a_over_m = a_over_m_from_physical(density, diameter, flattening);
        let lambda_0 = if absorptivity > 0.0 {
            lambda_0_from_physical(
                thermal_inertia,
                emissivity,
                absorptivity,
                flattening,
                rotation_period,
            )
        } else {
            0.0
        };
        NonGravModel::new_farnocchia(
            a_over_m,
            lambda_0,
            albedo,
            absorptivity,
            flattening,
            Vector::<Equatorial>::new([0.0, 0.0, 1.0]),
        )
        .unwrap()
    }

    #[test]
    fn radiation_free_params_ordering() {
        let m = sphere_physical(2800.0, 130.0, 0.52, 0.71);
        let NonGravModel::FarnocchiaModel {
            a_over_m, lambda_0, ..
        } = m
        else {
            unreachable!()
        };
        assert_eq!(m.n_free_params(), 2);
        assert_eq!(m.get_free_params(), vec![a_over_m, lambda_0]);
        assert_eq!(m.param_names(), vec!["a_over_m", "lambda_0"]);
    }

    #[test]
    fn radiation_set_free_params_clamps_positive() {
        let mut m = sphere_physical(2800.0, 130.0, 0.5, 0.7);
        m.set_free_params(&[-1.0, -1.0]);
        assert_eq!(m.get_free_params(), vec![1e-30, 0.0]);
    }

    #[test]
    fn radiation_shape_factors_spherical_limit() {
        let (px, pz, sig) = shape_factors(1.0);
        assert!((px - 1.0).abs() < 1e-14);
        assert!((pz - 1.0).abs() < 1e-14);
        assert!((sig - 1.0).abs() < 1e-14);
    }

    #[test]
    fn radiation_lambda_functions_limits() {
        // Absorptivity=0 zeros the thermal component. For a sphere at r=1 AU
        // with r_hat . s_hat = 0:
        //   a = (A/M) * F0/c * (1 + 4/9 * a_0) * r_hat
        let model = sphere_physical(2800.0, 130.0, 0.5, 0.0);
        let NonGravModel::FarnocchiaModel { a_over_m, .. } = model else {
            unreachable!()
        };
        let pos = Vector3::new(1.0, 0.0, 0.0);
        let vel = Vector3::zeros();
        let accel = {
            let mut a = Vector3::zeros();
            model.add_acceleration(&mut a, &pos, &vel);
            a
        };
        let expected_x = a_over_m * F0_OVER_C_AU_DAY2 * (1.0 + 4.0 / 9.0 * 0.5);
        assert!((accel.x - expected_x).abs() < 1e-20);
        assert!(accel.y.abs() < 1e-20);
        assert!(accel.z.abs() < 1e-20);
    }

    #[test]
    fn radiation_thermal_lag_vanishes_for_zero_absorptivity() {
        // Verify there is no thermal contribution when absorptivity = 0.
        let pos = Vector3::new(1.0, 0.5, 0.0);
        let vel = Vector3::zeros();
        let m_full = sphere_physical(2800.0, 130.0, 0.5, 0.7);
        let m_srp_only = sphere_physical(2800.0, 130.0, 0.5, 0.0);
        let mut a_full = Vector3::zeros();
        let mut a_srp = Vector3::zeros();
        m_full.add_acceleration(&mut a_full, &pos, &vel);
        m_srp_only.add_acceleration(&mut a_srp, &pos, &vel);
        let diff = a_full - a_srp;
        assert!(diff.norm() > 1e-14);
        // Absorptivity acts on the thermal contribution, but lambda_0 also
        // depends on absorptivity, so we just check the thermal part is
        // non-negligible compared to SRP. More rigorous linearity tests hold
        // when only density is varied.
    }

    #[test]
    fn radiation_scales_linearly_with_one_over_density() {
        // a_over_m is inversely proportional to density; the acceleration
        // scales as 1/density.
        let pos = Vector3::new(1.0, 0.0, 0.0);
        let vel = Vector3::zeros();
        let mut a1 = Vector3::zeros();
        let mut a2 = Vector3::zeros();
        sphere_physical(1000.0, 130.0, 0.5, 0.7).add_acceleration(&mut a1, &pos, &vel);
        sphere_physical(3000.0, 130.0, 0.5, 0.7).add_acceleration(&mut a2, &pos, &vel);
        assert!((a1 - 3.0 * a2).norm() < 1e-20);
    }

    #[test]
    fn radiation_inverse_r_squared() {
        // Isolate SRP by turning off absorptivity; the SRP piece is 1/r^2.
        let vel = Vector3::zeros();
        let mut a_srp1 = Vector3::zeros();
        let mut a_srp2 = Vector3::zeros();
        sphere_physical(2800.0, 130.0, 0.5, 0.0).add_acceleration(
            &mut a_srp1,
            &Vector3::new(1.0, 0.0, 0.0),
            &vel,
        );
        sphere_physical(2800.0, 130.0, 0.5, 0.0).add_acceleration(
            &mut a_srp2,
            &Vector3::new(2.0, 0.0, 0.0),
            &vel,
        );
        assert!((a_srp2.x / a_srp1.x - 0.25).abs() < 1e-12);
        // Full model stays in the xy-plane when s_hat = +z and r_hat in xy.
        let mut a1 = Vector3::zeros();
        let mut a2 = Vector3::zeros();
        sphere_physical(2800.0, 130.0, 0.5, 0.7).add_acceleration(
            &mut a1,
            &Vector3::new(1.0, 0.0, 0.0),
            &vel,
        );
        sphere_physical(2800.0, 130.0, 0.5, 0.7).add_acceleration(
            &mut a2,
            &Vector3::new(2.0, 0.0, 0.0),
            &vel,
        );
        assert!(a1.z.abs() < 1e-20);
        assert!(a2.z.abs() < 1e-20);
    }

    #[test]
    fn radiation_yarkovsky_direction_along_r_cross_s() {
        // On a sphere with r_hat = +x, s_hat = +z: r_hat x s_hat = (0,-1,0),
        // so the T2 direction is -(0,-1,0) = +y.
        let pos = Vector3::new(1.0, 0.0, 0.0);
        let vel = Vector3::zeros();
        let pole = Vector::<Equatorial>::new([0.0, 0.0, 1.0]);
        let a_over_m = a_over_m_from_physical(2800.0, 0.011, 1.0);
        let lambda_0 = lambda_0_from_physical(130.0, 0.9, 0.7, 1.0, 1.0);
        let m_full = NonGravModel::new_farnocchia(a_over_m, lambda_0, 0.0, 0.7, 1.0, pole).unwrap();
        let mut a = Vector3::zeros();
        m_full.add_acceleration(&mut a, &pos, &vel);
        assert!(a.y > 0.0);
    }

    #[test]
    fn radiation_a_over_m_matches_eq6() {
        // For a sphere (flattening=1), R_P = D/2, so A/M = 3 / (4 * rho * R_P).
        // sphere_physical uses diameter = 0.011 km = 11 m.
        let m = sphere_physical(2800.0, 130.0, 0.52, 0.71);
        let NonGravModel::FarnocchiaModel { a_over_m, .. } = m else {
            unreachable!()
        };
        let expected = 3.0 / (2.0 * 2800.0 * 11.0);
        assert!((a_over_m - expected).abs() / expected < 1e-14);
    }

    #[test]
    fn radiation_density_round_trip() {
        let m = sphere_physical(2800.0, 130.0, 0.52, 0.71);
        let NonGravModel::FarnocchiaModel { a_over_m, .. } = m else {
            unreachable!()
        };
        let recovered = density_from_a_over_m(a_over_m, 0.011, 1.0);
        assert!((recovered - 2800.0).abs() < 1e-9);
    }

    #[test]
    fn radiation_thermal_inertia_round_trip() {
        let m = sphere_physical(2800.0, 130.0, 0.52, 0.71);
        let NonGravModel::FarnocchiaModel { lambda_0, .. } = m else {
            unreachable!()
        };
        let recovered = thermal_inertia_from_lambda_0(lambda_0, 0.9, 0.71, 1.0, 1.0);
        assert!((recovered - 130.0).abs() < 1e-9);
    }
}
