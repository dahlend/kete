//! Body-fixed rotation models for extended-gravity bodies.
//!
//! A [`RotationModel`] converts between an inertial reference frame
//! (typically ICRF / J2000 equatorial) and the body-fixed frame in which
//! the gravity model is evaluated.  Only simple analytic rotation models
//! are implemented here:
//!
//! - [`RotationModel::Fixed`] - a non-rotating body with a constant
//!   inertial-to-body quaternion.  Useful for tests and for bodies whose
//!   rotation is irrelevant (e.g. very slow rotators evaluated over a
//!   short integration arc).
//! - [`RotationModel::ConstantSpin`] - the IAU "Cartographic Elements"
//!   convention with a fixed pole and linear prime-meridian angle:
//!   `W(t) = W0 + W_dot * (t - epoch)`.
//!
//! PCK-backed rotation (full IAU expansions, nutation/precession terms,
//! arbitrary tabulated body orientations) is intentionally deferred to a
//! later phase that lives in `kete_spice`.
//!
// BSD 3-Clause License
//
// Copyright (c) 2026, Dar Dahlen
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

use std::f64::consts::FRAC_PI_2;

use nalgebra::{UnitQuaternion, Vector3};

/// Analytic rotation model for an extended body.
///
/// Time inputs to all methods are TDB Julian Days.  The caller is
/// responsible for supplying TDB-scaled times (use
/// [`crate::time::Time::tdb`] / similar to convert).
#[derive(Debug, Clone, Copy)]
pub enum RotationModel {
    /// Non-rotating body with a fixed inertial-to-body rotation.
    Fixed {
        /// Inertial -> body-fixed rotation quaternion.
        inertial_to_body: UnitQuaternion<f64>,
    },
    /// IAU Cartographic Elements style constant-spin model.
    ///
    /// Given `pole_ra`, `pole_dec` (the body's north pole right
    /// ascension and declination in the inertial frame, radians) and a
    /// linear prime meridian `W(t) = w0 + w_dot * (t - epoch_jd)`, the
    /// inertial-to-body rotation is the composition (applied right to
    /// left):
    ///
    /// `R = R_z(W) * R_x(pi/2 - dec) * R_z(pi/2 + ra)`.
    ///
    /// `w_dot` is in radians per day.
    ConstantSpin {
        /// Right ascension of the body's north pole, radians.
        pole_ra: f64,
        /// Declination of the body's north pole, radians.
        pole_dec: f64,
        /// Prime meridian angle at epoch, radians.
        w0: f64,
        /// Prime meridian rate, radians per day.
        w_dot: f64,
        /// Epoch (TDB Julian Day) at which `W = w0`.
        epoch_jd: f64,
    },
}

impl RotationModel {
    /// Identity rotation; equivalent to `Fixed { inertial_to_body =
    /// identity }`.  Useful as a default for tests.
    #[must_use]
    pub fn identity() -> Self {
        Self::Fixed {
            inertial_to_body: UnitQuaternion::identity(),
        }
    }

    /// Inertial-to-body-fixed rotation at TDB Julian Day `jd`.
    #[must_use]
    pub fn inertial_to_body(&self, jd: f64) -> UnitQuaternion<f64> {
        match *self {
            Self::Fixed { inertial_to_body } => inertial_to_body,
            Self::ConstantSpin {
                pole_ra,
                pole_dec,
                w0,
                w_dot,
                epoch_jd,
            } => {
                // IAU convention: the inertial-to-body rotation is the
                // PASSIVE composition R_z(W) R_x(pi/2 - dec) R_z(pi/2 + ra).
                // nalgebra's `from_axis_angle` is an ACTIVE rotation, so
                // we negate the angles to get the corresponding passive
                // rotations.  After this, the inertial pole vector
                // (cos(d)cos(ra), cos(d)sin(ra), sin(d)) maps to the
                // body +z axis, and the body's prime meridian sweeps
                // through the inertial frame at rate `w_dot`.
                let w = w0 + w_dot * (jd - epoch_jd);
                let r_z_w = UnitQuaternion::from_axis_angle(&Vector3::z_axis(), -w);
                let r_x =
                    UnitQuaternion::from_axis_angle(&Vector3::x_axis(), -(FRAC_PI_2 - pole_dec));
                let r_z_ra =
                    UnitQuaternion::from_axis_angle(&Vector3::z_axis(), -(FRAC_PI_2 + pole_ra));
                r_z_w * r_x * r_z_ra
            }
        }
    }

    /// Body-fixed-to-inertial rotation at TDB Julian Day `jd`.
    #[must_use]
    pub fn body_to_inertial(&self, jd: f64) -> UnitQuaternion<f64> {
        self.inertial_to_body(jd).inverse()
    }

    /// Angular velocity vector in the inertial frame at `jd`, in
    /// radians per day.  For the [`Self::Fixed`] variant this is the
    /// zero vector.  For [`Self::ConstantSpin`] it is constant in time
    /// (the pole is fixed).
    #[must_use]
    pub fn angular_velocity_inertial(&self, _jd: f64) -> Vector3<f64> {
        match *self {
            Self::Fixed { .. } => Vector3::zeros(),
            Self::ConstantSpin {
                pole_ra,
                pole_dec,
                w_dot,
                ..
            } => {
                // Pole unit vector in inertial coordinates: standard
                // RA/Dec -> Cartesian.
                let cos_d = pole_dec.cos();
                let pole =
                    Vector3::new(cos_d * pole_ra.cos(), cos_d * pole_ra.sin(), pole_dec.sin());
                w_dot * pole
            }
        }
    }

    /// Convert an inertial-frame vector to body-fixed coordinates.
    #[must_use]
    pub fn rotate_to_body(&self, jd: f64, v_inertial: Vector3<f64>) -> Vector3<f64> {
        self.inertial_to_body(jd) * v_inertial
    }

    /// Convert a body-fixed vector back to the inertial frame.
    #[must_use]
    pub fn rotate_to_inertial(&self, jd: f64, v_body: Vector3<f64>) -> Vector3<f64> {
        self.body_to_inertial(jd) * v_body
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::{PI, TAU};

    fn approx_eq_v(a: Vector3<f64>, b: Vector3<f64>, tol: f64) -> bool {
        (a - b).norm() < tol
    }

    #[test]
    fn fixed_identity_is_identity() {
        let r = RotationModel::identity();
        let v = Vector3::new(1.0, 2.0, 3.0);
        assert!(approx_eq_v(r.rotate_to_body(0.0, v), v, 1e-15));
        assert!(approx_eq_v(r.rotate_to_inertial(12345.6, v), v, 1e-15));
        assert!(approx_eq_v(
            r.angular_velocity_inertial(0.0),
            Vector3::zeros(),
            1e-15
        ));
    }

    #[test]
    fn round_trip_constant_spin() {
        let r = RotationModel::ConstantSpin {
            pole_ra: 0.7,
            pole_dec: 1.1,
            w0: 0.3,
            w_dot: 0.05,
            epoch_jd: 2_451_545.0,
        };
        let v = Vector3::new(0.4, -0.7, 0.2);
        for jd in [2_451_545.0, 2_460_000.0, 2_470_000.0] {
            let back = r.rotate_to_inertial(jd, r.rotate_to_body(jd, v));
            assert!(
                approx_eq_v(back, v, 1e-13),
                "round-trip failed at jd={jd}: {back:?} vs {v:?}"
            );
        }
    }

    #[test]
    fn pole_is_invariant_under_spin() {
        // The pole vector, expressed in body coordinates, must coincide
        // with the body z-axis (by construction of the IAU convention),
        // and this must hold at every time (the prime meridian rotates
        // around the pole, which is itself fixed).
        let pole_ra = 0.4;
        let pole_dec = 1.0;
        let r = RotationModel::ConstantSpin {
            pole_ra,
            pole_dec,
            w0: 0.0,
            w_dot: 0.1,
            epoch_jd: 0.0,
        };
        let cos_d = pole_dec.cos();
        let pole_inertial =
            Vector3::new(cos_d * pole_ra.cos(), cos_d * pole_ra.sin(), pole_dec.sin());
        for jd in [0.0_f64, 5.0, 100.0, -42.0] {
            let pole_body = r.rotate_to_body(jd, pole_inertial);
            assert!(
                approx_eq_v(pole_body, Vector3::z(), 1e-12),
                "pole not invariant at jd={jd}: {pole_body:?}"
            );
        }
    }

    #[test]
    fn full_period_returns_to_initial_orientation() {
        // After one rotation period (W advances by 2*pi), the body-fixed
        // representation of any inertial vector must return to its
        // value at epoch.
        let w_dot = 0.2;
        let r = RotationModel::ConstantSpin {
            pole_ra: 0.0,
            pole_dec: 0.5,
            w0: 0.0,
            w_dot,
            epoch_jd: 0.0,
        };
        let period = TAU / w_dot;
        let v = Vector3::new(1.0, 0.0, 0.0);
        let v0 = r.rotate_to_body(0.0, v);
        let v_t = r.rotate_to_body(period, v);
        assert!(approx_eq_v(v0, v_t, 1e-12));
    }

    #[test]
    fn angular_velocity_points_along_pole() {
        let pole_ra = 0.9;
        let pole_dec = -0.3;
        let w_dot = 0.07;
        let r = RotationModel::ConstantSpin {
            pole_ra,
            pole_dec,
            w0: 0.0,
            w_dot,
            epoch_jd: 0.0,
        };
        let cos_d = pole_dec.cos();
        let expected =
            w_dot * Vector3::new(cos_d * pole_ra.cos(), cos_d * pole_ra.sin(), pole_dec.sin());
        let omega = r.angular_velocity_inertial(123.4);
        assert!(approx_eq_v(omega, expected, 1e-15));
    }

    #[test]
    fn equatorial_pole_no_tilt_at_epoch() {
        // Pole at (RA=0, Dec=pi/2) is just the +z axis; with w0 = 0
        // and at epoch, the inertial-to-body rotation should be the
        // identity up to the +pi/2 in RA convention bookkeeping.  The
        // body z-axis should equal the inertial z-axis.
        let r = RotationModel::ConstantSpin {
            pole_ra: 0.0,
            pole_dec: PI / 2.0,
            w0: 0.0,
            w_dot: 0.0,
            epoch_jd: 0.0,
        };
        let pole_body = r.rotate_to_body(0.0, Vector3::z());
        assert!(approx_eq_v(pole_body, Vector3::z(), 1e-12));
    }
}
