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

use crate::constants::{C_AU_PER_DAY_INV_SQUARED, GMS};

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

    /// Number of free (solvable) parameters in this model.
    ///
    /// For `JplComet`, only finite A terms are counted as free; NaN marks a
    /// term as absent (held at zero). `Dust` always has 1 free parameter.
    #[must_use]
    pub fn n_free_params(&self) -> usize {
        match self {
            Self::JplComet { a1, a2, a3, .. } => {
                [a1, a2, a3].iter().filter(|v| v.is_finite()).count()
            }
            Self::Dust { .. } => 1,
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
                // NaN A terms are absent — treat as zero.
                let a1_eff = if a1.is_finite() { *a1 } else { 0.0 };
                let a2_eff = if a2.is_finite() { *a2 } else { 0.0 };
                let a3_eff = if a3.is_finite() { *a3 } else { 0.0 };
                *accel += pos_norm * (scale * a1_eff);
                *accel += t_vec * (scale * a2_eff);
                *accel += n_vec * (scale * a3_eff);
            }
        }
    }
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
}
