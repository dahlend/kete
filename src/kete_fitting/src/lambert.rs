//! Lambert's problem solver.
//!
//! Given two heliocentric position vectors and a transfer time, find the
//! Keplerian orbit that connects them.  Single-revolution solutions only.
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

use kete_core::constants::GMS;
use kete_core::frames::{InertialFrame, Vector};
use kete_core::prelude::{Error, KeteResult};

/// Stumpff function C(z) = (1 - cos(sqrt(z))) / z.
///
/// Handles elliptic (z > 0), parabolic (z ~ 0), and hyperbolic (z < 0) cases.
fn stumpff_c(z: f64) -> f64 {
    if z.abs() < 1e-8 {
        // Taylor: 1/2 - z/24 + z^2/720 - ...
        0.5 - z / 24.0 + z * z / 720.0
    } else if z > 0.0 {
        let sz = z.sqrt();
        (1.0 - sz.cos()) / z
    } else {
        let sz = (-z).sqrt();
        (sz.cosh() - 1.0) / (-z)
    }
}

/// Stumpff function S(z) = (sqrt(z) - sin(sqrt(z))) / sqrt(z)^3.
///
/// Handles elliptic (z > 0), parabolic (z ~ 0), and hyperbolic (z < 0) cases.
fn stumpff_s(z: f64) -> f64 {
    if z.abs() < 1e-8 {
        // Taylor: 1/6 - z/120 + z^2/5040 - ...
        1.0 / 6.0 - z / 120.0 + z * z / 5040.0
    } else if z > 0.0 {
        let sz = z.sqrt();
        (sz - sz.sin()) / (sz * sz * sz)
    } else {
        let sz = (-z).sqrt();
        (sz.sinh() - sz) / (sz * sz * sz)
    }
}

/// Solve Lambert's problem: find the velocity vectors that connect two
/// heliocentric positions `r1` and `r2` via Keplerian (two-body) motion in
/// transfer time `dt`.
///
/// Returns `(v1, v2)` -- the heliocentric velocity at `r1` and `r2`
/// respectively.
///
/// # Arguments
///
/// * `r1` -- Heliocentric position at departure (AU).
/// * `r2` -- Heliocentric position at arrival (AU).
/// * `dt` -- Transfer time in days.  Must be positive.
/// * `prograde` -- If `true`, selects the short-way transfer (transfer
///   angle < 180 deg for orbits with positive angular momentum
///   z-component).  If `false`, selects the long-way transfer.
///
/// # Algorithm
///
/// Universal-variable formulation using Stumpff functions, following the
/// approach in Curtis, *Orbital Mechanics for Engineering Students*.
/// Newton-Raphson iteration on the universal variable `z` (reciprocal
/// semi-major axis), with bisection fallback for robustness.
///
/// # Errors
///
/// * `dt <= 0`
/// * Degenerate geometry (positions at the origin or nearly collinear and
///   `prograde` cannot disambiguate the orbit plane).
/// * Newton-Raphson fails to converge within 50 iterations.
pub fn lambert<T: InertialFrame>(
    r1: &Vector<T>,
    r2: &Vector<T>,
    dt: f64,
    prograde: bool,
) -> KeteResult<(Vector<T>, Vector<T>)> {
    if dt <= 0.0 {
        return Err(Error::ValueError(
            "Lambert: transfer time dt must be positive".into(),
        ));
    }

    let r1_mag = r1.norm();
    let r2_mag = r2.norm();

    if r1_mag < 1e-15 || r2_mag < 1e-15 {
        return Err(Error::ValueError(
            "Lambert: position vectors must be non-zero".into(),
        ));
    }

    // Transfer angle from dot product, disambiguated by cross product.
    let cos_dnu = r1.dot(r2) / (r1_mag * r2_mag);
    let cos_dnu = cos_dnu.clamp(-1.0, 1.0);

    let cross = r1.cross(r2);
    let cross_z = cross[2];

    let dnu = if prograde {
        if cross_z >= 0.0 {
            cos_dnu.acos()
        } else {
            std::f64::consts::TAU - cos_dnu.acos()
        }
    } else if cross_z < 0.0 {
        cos_dnu.acos()
    } else {
        std::f64::consts::TAU - cos_dnu.acos()
    };

    // Auxiliary quantity A.
    let sin_dnu = dnu.sin();
    if sin_dnu.abs() < 1e-14 {
        return Err(Error::ValueError(
            "Lambert: positions are nearly collinear (transfer angle ~ 0 or 180 deg)".into(),
        ));
    }

    let a_coeff = sin_dnu * (r1_mag * r2_mag / (1.0 - cos_dnu)).sqrt();

    // Target value: sqrt(GMS) * dt.
    let sqrt_mu_dt = GMS.sqrt() * dt;

    // y(z) helper.
    let y_of_z = |z: f64| -> f64 {
        let cz = stumpff_c(z);
        if cz.abs() < 1e-30 {
            return f64::MAX;
        }
        r1_mag + r2_mag + a_coeff * (z * stumpff_s(z) - 1.0) / cz.sqrt()
    };

    // F(z) = [y/C]^{3/2} * S + A * sqrt(y) - sqrt(mu) * dt.
    let f_of_z = |z: f64| -> f64 {
        let y = y_of_z(z);
        if y < 0.0 {
            return f64::MAX;
        }
        let cz = stumpff_c(z);
        let sz = stumpff_s(z);
        if cz.abs() < 1e-30 {
            return f64::MAX;
        }
        (y / cz).powf(1.5) * sz + a_coeff * y.sqrt() - sqrt_mu_dt
    };

    // Derivative dF/dz for Newton-Raphson.
    let df_of_z = |z: f64| -> f64 {
        let y = y_of_z(z);
        if y < 1e-30 {
            // Avoid division by zero; nudge Newton step.
            return 1.0;
        }
        let cz = stumpff_c(z);
        let sz = stumpff_s(z);
        if cz.abs() < 1e-30 {
            return 1.0;
        }

        // dy/dz (from differentiating y(z)):
        // For z != 0: dy/dz = A / (4*C^{3/2}) * (S - 3*S'*C/z + ... )
        // Simpler: use the relation  dy/dz = A * sqrt(y) / (2*z)  at z != 0
        // from Curtis eq. 5.43 (simplified form):
        //   dy/dz = ... but numerically easier via finite-difference-free form.
        //
        // Curtis eq. 5.45:
        //   F'(z) = 0                            if z == 0
        //        = [y/C]^{3/2} * (1/(2z)*(C - 3*S/(2*C)) + 3*S^2/(4*C))
        //          + (A/8)*(3*S*sqrt(y)/C + A*sqrt(C/y))

        if z.abs() < 1e-8 {
            // Near z=0, use central difference.
            let eps = 1e-6;
            return (f_of_z(eps) - f_of_z(-eps)) / (2.0 * eps);
        }

        let y_over_c = y / cz;
        let yc32 = y_over_c.powf(1.5);

        let term1 =
            yc32 * (1.0 / (2.0 * z) * (cz - 3.0 * sz / (2.0 * cz)) + 3.0 * sz * sz / (4.0 * cz));
        let term2 = a_coeff / 8.0 * (3.0 * sz * y.sqrt() / cz + a_coeff * (cz / y).sqrt());

        term1 + term2
    };

    // Newton-Raphson with bisection fallback.
    // Initial guess: z = 0 (parabolic).
    let mut z = 0.0;

    // Set initial bracket.  For single-revolution:
    //   z_low  = below the parabolic limit (hyperbolic side)
    //   z_high = (2*pi)^2 is the upper bound for single revolution
    let mut z_low = -4.0 * std::f64::consts::PI * std::f64::consts::PI;
    let mut z_high = 4.0 * std::f64::consts::PI * std::f64::consts::PI;

    // Ensure y(z_low) > 0 by expanding the lower bracket if needed.
    while y_of_z(z_low) < 0.0 && z_low > -1e6 {
        z_low *= 2.0;
    }

    let max_iter = 50;
    for _ in 0..max_iter {
        let fz = f_of_z(z);

        if fz.abs() < 1e-12 {
            break;
        }

        let dfz = df_of_z(z);

        // Newton step.
        let mut z_new = if dfz.abs() > 1e-30 {
            z - fz / dfz
        } else {
            // Derivative vanished -- use bisection.
            0.5 * (z_low + z_high)
        };

        // If Newton step leaves the bracket, fall back to bisection.
        if z_new < z_low || z_new > z_high {
            z_new = 0.5 * (z_low + z_high);
        }

        // Update bracket.
        if fz < 0.0 {
            z_low = z;
        } else {
            z_high = z;
        }

        z = z_new;
    }

    // Check convergence.
    let fz = f_of_z(z);
    if fz.abs() > 1e-6 {
        return Err(Error::Convergence(format!(
            "Lambert: failed to converge after {max_iter} iterations (residual = {fz:.2e})"
        )));
    }

    // Compute Lagrange coefficients.
    let y = y_of_z(z);
    if y < 0.0 {
        return Err(Error::Convergence(
            "Lambert: y(z) < 0 at converged solution".into(),
        ));
    }

    let f = 1.0 - y / r1_mag;
    let g = a_coeff * (y / GMS).sqrt();
    let g_dot = 1.0 - y / r2_mag;

    if g.abs() < 1e-30 {
        return Err(Error::Convergence(
            "Lambert: degenerate Lagrange coefficient g ~ 0".into(),
        ));
    }

    // v1 = (r2 - f * r1) / g
    // v2 = (g_dot * r2 - r1) / g
    let v1 = (*r2 - *r1 * f) / g;
    let v2 = (*r2 * g_dot - *r1) / g;

    Ok((v1, v2))
}

#[cfg(test)]
mod tests {
    use super::*;
    use kete_core::frames::Equatorial;
    use kete_core::prelude::State;
    use kete_core::propagation::propagate_two_body;
    use kete_core::time::{TDB, Time};

    /// Helper: create Vector<Equatorial>.
    fn vec_eq(x: f64, y: f64, z: f64) -> Vector<Equatorial> {
        Vector::new([x, y, z])
    }

    /// Round-trip test: create (r1, v1), propagate dt days via two-body,
    /// then verify Lambert recovers v1 and v2.
    fn round_trip(r1: Vector<Equatorial>, v1: Vector<Equatorial>, dt_days: f64, tol: f64) {
        let epoch: Time<TDB> = 2460000.5_f64.into();
        let s1 = State::new(kete_core::desigs::Desig::Empty, epoch, r1, v1, 0);

        let target: Time<TDB> = (epoch.jd + dt_days).into();
        let s2 = propagate_two_body(&s1, target).expect("two-body propagation failed");

        let (v1_lam, v2_lam) = lambert(&r1, &s2.pos, dt_days, true).expect("Lambert solver failed");

        let dv1 = (v1_lam - v1).norm();
        let dv2 = (v2_lam - s2.vel).norm();

        assert!(
            dv1 < tol,
            "v1 mismatch: err={dv1:.2e} > tol={tol:.2e}\n  expected: [{:.10}, {:.10}, {:.10}]\n  got:      [{:.10}, {:.10}, {:.10}]",
            v1[0],
            v1[1],
            v1[2],
            v1_lam[0],
            v1_lam[1],
            v1_lam[2],
        );
        assert!(dv2 < tol, "v2 mismatch: err={dv2:.2e} > tol={tol:.2e}",);
    }

    #[test]
    fn test_round_trip_circular() {
        // Circular orbit at 1 AU in ecliptic plane (rotated to equatorial).
        let r = 1.0;
        let v = (GMS / r).sqrt();
        let obl = 23.44_f64.to_radians();

        let r1 = vec_eq(r, 0.0, 0.0);
        let v1 = vec_eq(0.0, v * obl.cos(), v * obl.sin());
        round_trip(r1, v1, 30.0, 1e-10);
    }

    #[test]
    fn test_round_trip_elliptic() {
        // Elliptic orbit: a=2 AU, e=0.3, started at perihelion.
        let a = 2.0;
        let e = 0.3;
        let r_peri = a * (1.0 - e);
        let v_peri = (GMS * (2.0 / r_peri - 1.0 / a)).sqrt();
        let obl = 23.44_f64.to_radians();
        let inc = 10.0_f64.to_radians();
        let tilt = obl + inc;

        let r1 = vec_eq(r_peri, 0.0, 0.0);
        let v1 = vec_eq(0.0, v_peri * tilt.cos(), v_peri * tilt.sin());
        round_trip(r1, v1, 50.0, 1e-10);
    }

    #[test]
    fn test_round_trip_high_eccentricity() {
        // Highly elliptic: a=5 AU, e=0.9.
        let a = 5.0;
        let e = 0.9;
        let r_peri = a * (1.0 - e);
        let v_peri = (GMS * (2.0 / r_peri - 1.0 / a)).sqrt();
        let obl = 23.44_f64.to_radians();

        let r1 = vec_eq(r_peri, 0.0, 0.0);
        let v1 = vec_eq(0.0, v_peri * obl.cos(), v_peri * obl.sin());
        round_trip(r1, v1, 20.0, 1e-9);
    }

    #[test]
    fn test_round_trip_hyperbolic() {
        // Hyperbolic orbit: e=1.5, r_peri=0.5 AU.
        let e = 1.5;
        let r_peri = 0.5;
        // Negative for hyperbola.
        let a = r_peri / (e - 1.0);
        let v_peri = (GMS * (2.0 / r_peri - 1.0 / a)).sqrt();
        let obl = 23.44_f64.to_radians();

        let r1 = vec_eq(r_peri, 0.0, 0.0);
        let v1 = vec_eq(0.0, v_peri * obl.cos(), v_peri * obl.sin());
        round_trip(r1, v1, 10.0, 1e-9);
    }

    #[test]
    fn test_round_trip_neo_short_transfer() {
        // NEO-relevant: object at ~0.3 AU, 2-day transfer.
        let a = 1.5;
        // Near perihelion.
        let r = 0.3;
        let v = (GMS * (2.0 / r - 1.0 / a)).sqrt();
        let obl = 23.44_f64.to_radians();

        let r1 = vec_eq(r, 0.0, 0.0);
        let v1 = vec_eq(0.0, v * obl.cos(), v * obl.sin());
        round_trip(r1, v1, 2.0, 1e-10);
    }

    #[test]
    fn test_round_trip_retrograde() {
        // Retrograde orbit (inclination > 90 deg).
        let r = 2.0;
        let v = (GMS / r).sqrt();
        // inclination 150 deg in equatorial frame.
        let inc = 150.0_f64.to_radians();

        let r1 = vec_eq(r, 0.0, 0.0);
        let v1 = vec_eq(0.0, v * inc.cos(), v * inc.sin());

        // For retrograde, cross product z < 0, so use prograde=false.
        let epoch: Time<TDB> = 2460000.5_f64.into();
        let s1 = State::new(kete_core::desigs::Desig::Empty, epoch, r1, v1, 0);
        let target: Time<TDB> = (epoch.jd + 30.0).into();
        let s2 = propagate_two_body(&s1, target).expect("propagation failed");

        let (v1_lam, v2_lam) = lambert(&r1, &s2.pos, 30.0, false).expect("Lambert solver failed");

        let dv1 = (v1_lam - v1).norm();
        let dv2 = (v2_lam - s2.vel).norm();
        assert!(dv1 < 1e-10, "retrograde v1 err: {dv1:.2e}");
        assert!(dv2 < 1e-10, "retrograde v2 err: {dv2:.2e}");
    }

    #[test]
    fn test_hohmann_transfer() {
        // Earth-to-Mars Hohmann transfer: r1=1 AU, r2=1.524 AU.
        // Transfer orbit: a = (r1 + r2) / 2, dt = pi * sqrt(a^3 / GM).
        let r1_mag = 1.0;
        let r2_mag = 1.524;
        let a_transfer: f64 = f64::midpoint(r1_mag, r2_mag);
        let dt = std::f64::consts::PI * (a_transfer.powi(3) / GMS).sqrt();

        let r1 = vec_eq(r1_mag, 0.0, 0.0);

        // For 180-deg transfer the orbit plane is degenerate, but with z=0
        // positions the cross product is zero. Use a slight offset (1e-4 AU ~
        // 15,000 km, negligible vs 1.524 AU).
        let r2 = vec_eq(-r2_mag, 1e-4, 0.0);

        let (v1, _v2) = lambert(&r1, &r2, dt, true).expect("Hohmann failed");

        // Expected departure speed: v1 = sqrt(GM * (2/r1 - 1/a)).
        let v1_expected = (GMS * (2.0 / r1_mag - 1.0 / a_transfer)).sqrt();
        let v1_mag = v1.norm();

        let rel_err = (v1_mag - v1_expected).abs() / v1_expected;
        assert!(
            rel_err < 1e-6,
            "Hohmann v1: expected {v1_expected:.8}, got {v1_mag:.8}, rel_err={rel_err:.2e}"
        );
    }

    #[test]
    fn test_negative_dt_error() {
        let r1 = vec_eq(1.0, 0.0, 0.0);
        let r2 = vec_eq(0.0, 1.5, 0.0);
        assert!(lambert(&r1, &r2, -10.0, true).is_err());
    }

    #[test]
    fn test_zero_dt_error() {
        let r1 = vec_eq(1.0, 0.0, 0.0);
        let r2 = vec_eq(0.0, 1.5, 0.0);
        assert!(lambert(&r1, &r2, 0.0, true).is_err());
    }

    #[test]
    fn test_collinear_error() {
        // r1 and r2 exactly aligned (0-deg transfer): should error.
        let r1 = vec_eq(1.0, 0.0, 0.0);
        let r2 = vec_eq(2.0, 0.0, 0.0);
        assert!(lambert(&r1, &r2, 30.0, true).is_err());
    }

    #[test]
    fn test_round_trip_long_period() {
        // Long-period: a=20 AU, e=0.6, 200-day transfer.
        let a = 20.0;
        let e = 0.6;
        let r_peri = a * (1.0 - e);
        let v_peri = (GMS * (2.0 / r_peri - 1.0 / a)).sqrt();
        let obl = 23.44_f64.to_radians();
        let inc = 20.0_f64.to_radians();
        let tilt = obl + inc;

        let r1 = vec_eq(r_peri, 0.0, 0.0);
        let v1 = vec_eq(0.0, v_peri * tilt.cos(), v_peri * tilt.sin());
        round_trip(r1, v1, 200.0, 1e-9);
    }

    #[test]
    fn test_round_trip_small_angle() {
        // Small transfer angle (~5 deg), 1-day transfer.
        let r = 1.0;
        let v = (GMS / r).sqrt();
        let obl = 23.44_f64.to_radians();
        let r1 = vec_eq(r, 0.0, 0.0);
        let v1 = vec_eq(0.0, v * obl.cos(), v * obl.sin());
        round_trip(r1, v1, 1.0, 1e-10);
    }
}
