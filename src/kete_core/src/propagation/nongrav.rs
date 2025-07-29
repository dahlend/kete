// BSD 3-Clause License
//
// Copyright (c) 2025, Dar Dahlen
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

use super::analytic_2_body;

/// Non-Gravitational models.
/// These are used during integration to model non-gravitational forces on particles in
/// the solar system.
#[derive(Debug, Clone)]
pub enum NonGravModel {
    /// JPL's non-gravitational forces are modeled as defined on page 139 of the Comets II
    /// textbook.
    ///
    /// This model adds 3 "A" terms to the acceleration which the object feels. These
    /// A terms represent additional radial, tangential, and normal forces on the object.
    ///
    /// `accel_additional = A_1 * g(r) * r_vec + A_2 * g(r) * t_vec + A_3 * g(r) * n_vec`
    /// Where `r_vec`, `t_vec`, `n_vec` are the radial, tangential, and normal unit vectors for
    /// the object.
    ///
    /// The g(r) function is defined by the equation:
    /// g(r) = alpha (r / r0) ^ -m * (1 + (r / r0) ^ n) ^ -k
    ///
    /// When alpha=1.0, n=0.0, k=0.0, r0=1.0, and m=2.0, this is equivalent to a 1/r^2
    /// correction.
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
    pub fn new_dust(beta: f64) -> Self {
        Self::Dust { beta }
    }

    /// Construct a new non-grav model which follows the default comet drop-off.
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
                let n_vec = t_vec.cross(&pos_norm).normalize();

                if !dt.is_zero() {
                    (pos, _) = analytic_2_body(-dt, &pos, vel, None).unwrap();
                };
                let rr0 = pos.norm() / r_0;
                let scale = alpha * rr0.powf(-m) * (1.0 + rr0.powf(*n)).powf(-k);
                *accel += pos_norm * (scale * a1);
                *accel += t_vec * (scale * a2);
                *accel += n_vec * (scale * a3);
            }
        }
    }
}
