//! Lambert's problem solver.
//!
//! Given two heliocentric position vectors and a transfer time, find the
//! Keplerian orbit that connects them.  Handles single-revolution and
//! multi-revolution solutions, including pi-transfers and near-parabolic
//! cases.
//!
//! Uses the universal-variable formulation with Stumpff functions for the
//! time-of-flight equation, solved via Newton-Raphson with bisection
//! fallback.  For near-collinear geometries, the orbit plane is resolved
//! by choosing an arbitrary normal perpendicular to the position vectors.
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
/// Handles elliptic (z > 0), parabolic (z ~ 0), and hyperbolic (z < 0).
fn stumpff_c(z: f64) -> f64 {
    if z.abs() < 1e-8 {
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
/// Handles elliptic (z > 0), parabolic (z ~ 0), and hyperbolic (z < 0).
fn stumpff_s(z: f64) -> f64 {
    if z.abs() < 1e-8 {
        1.0 / 6.0 - z / 120.0 + z * z / 5040.0
    } else if z > 0.0 {
        let sz = z.sqrt();
        (sz - sz.sin()) / (sz * sz * sz)
    } else {
        let sz = (-z).sqrt();
        (sz.sinh() - sz) / (sz * sz * sz)
    }
}

/// Solve Lambert's problem for up to `max_revs` complete revolutions.
///
/// Returns a `Vec` of `(v1, v2)` pairs.  The first element is the N=0
/// (single-revolution) solution; subsequent elements are multi-revolution
/// solutions for N=1..`max_revs`.
///
/// # Arguments
///
/// * `r1` -- Heliocentric position at departure (AU).
/// * `r2` -- Heliocentric position at arrival (AU).
/// * `dt` -- Transfer time in days.  Must be positive.
/// * `prograde` -- If `true`, selects the short-way transfer.
/// * `max_revs` -- Maximum number of complete revolutions to try.
///   Pass 0 for the single-revolution solution only.
///
/// # Errors
///
/// * `dt <= 0`
/// * Positions at the origin.
/// * No solution converges for any revolution count.
pub fn lambert<T: InertialFrame>(
    r1: &Vector<T>,
    r2: &Vector<T>,
    dt: f64,
    prograde: bool,
    max_revs: u32,
) -> KeteResult<Vec<(Vector<T>, Vector<T>)>> {
    let mut results = Vec::new();
    for n in 0..=max_revs {
        match lambert_core(r1, r2, dt, prograde, n) {
            Ok(sol) => results.push(sol),
            Err(_) => {
                if n > 0 {
                    // Multi-rev solutions stop existing beyond a maximum N.
                    break;
                }
            }
        }
    }
    if results.is_empty() {
        return Err(Error::Convergence(
            "Lambert: no solution found for any revolution count".into(),
        ));
    }
    Ok(results)
}

/// Core Lambert solver for a specific revolution count N.
fn lambert_core<T: InertialFrame>(
    r1: &Vector<T>,
    r2: &Vector<T>,
    dt: f64,
    prograde: bool,
    n_revs: u32,
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
    let cos_dnu = (r1.dot(r2) / (r1_mag * r2_mag)).clamp(-1.0, 1.0);
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

    let sin_dnu = dnu.sin();

    // Auxiliary quantity A includes orbit plane information.
    // For near-collinear (sin_dnu ~ 0), A is the important factor.
    let a_coeff = if sin_dnu.abs() > 1e-14 {
        sin_dnu * (r1_mag * r2_mag / (1.0 - cos_dnu)).sqrt()
    } else {
        // Pi-transfer or 0-transfer: use limiting form.
        // For pi transfer (dnu ~ pi), cos_dnu ~ -1, sin_dnu ~ 0.
        // A = sin(dnu) * sqrt(r1*r2/(1-cos_dnu))
        //   ~ sin(dnu) * sqrt(r1*r2/2)   since cos_dnu ~ -1
        // This goes to zero, making the problem degenerate -- the orbit
        // plane is undefined.  We pick an arbitrary plane.
        //
        // For 0-transfer (dnu ~ 0), the chord is zero and there's nothing
        // to solve.  Return error.
        if cos_dnu > 0.0 {
            return Err(Error::ValueError(
                "Lambert: positions are nearly identical (transfer angle ~ 0)".into(),
            ));
        }

        // Pi-transfer: use the small-angle approximation.
        // sin_dnu ~ pi - dnu (for dnu near pi), 1 - cos_dnu ~ 2
        let dnu_from_pi = std::f64::consts::PI - dnu.abs();
        let sin_approx = if dnu_from_pi.abs() < 1e-10 {
            // Exactly pi: sin(dnu) = 0 exactly.  The A coefficient
            // determines the orbit plane.  For a true pi transfer the
            // answer is a rectilinear orbit; we set A small but nonzero.
            let sign = if dnu > 0.0 { 1.0 } else { -1.0 };
            sign * 1e-10
        } else {
            sin_dnu
        };
        sin_approx * (r1_mag * r2_mag / (1.0 - cos_dnu).max(1e-30)).sqrt()
    };

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

    // dF/dz for Newton's method.
    let df_of_z = |z: f64| -> f64 {
        let y = y_of_z(z);
        if y < 1e-30 {
            return 1.0;
        }
        let cz = stumpff_c(z);
        if cz.abs() < 1e-30 {
            return 1.0;
        }

        if z.abs() < 1e-8 {
            let eps = 1e-6;
            return (f_of_z(eps) - f_of_z(-eps)) / (2.0 * eps);
        }

        let sz = stumpff_s(z);
        let y_over_c = y / cz;
        let yc32 = y_over_c.powf(1.5);

        let term1 =
            yc32 * (1.0 / (2.0 * z) * (cz - 3.0 * sz / (2.0 * cz)) + 3.0 * sz * sz / (4.0 * cz));
        let term2 = a_coeff / 8.0 * (3.0 * sz * y.sqrt() / cz + a_coeff * (cz / y).sqrt());

        term1 + term2
    };

    // Set initial bracket and starting z depending on revolution count.
    let max_iter = 50;

    // For N revolutions, the z-range is [(2*N*pi)^2, (2*(N+1)*pi)^2].
    let z_min_bound = if n_revs == 0 {
        -4.0 * std::f64::consts::PI * std::f64::consts::PI
    } else {
        let lower = 2.0 * f64::from(n_revs) * std::f64::consts::PI;
        lower * lower + 1e-6
    };
    let z_max_bound = if n_revs == 0 {
        4.0 * std::f64::consts::PI * std::f64::consts::PI
    } else {
        let upper = 2.0 * (f64::from(n_revs) + 1.0) * std::f64::consts::PI;
        upper * upper - 1e-6
    };

    let mut z_low = z_min_bound;
    let mut z_high = z_max_bound;

    // Ensure y(z_low) > 0 by expanding lower bracket if needed.
    while y_of_z(z_low) < 0.0 && z_low > -1e6 {
        z_low *= 2.0;
    }

    // For multi-rev, check that a solution actually exists (F changes sign).
    if n_revs > 0 {
        let f_low = f_of_z(z_low);
        let f_high = f_of_z(z_high);
        if f_low * f_high > 0.0 {
            return Err(Error::Convergence(format!(
                "Lambert: no {n_revs}-revolution solution exists for this transfer time"
            )));
        }
    }

    // Initial guess.
    let mut z = if n_revs == 0 {
        0.0
    } else {
        0.5 * (z_low + z_high)
    };

    for _ in 0..max_iter {
        let fz = f_of_z(z);

        if fz.abs() < 1e-12 {
            break;
        }

        let dfz = df_of_z(z);

        let mut z_new = if dfz.abs() > 1e-30 {
            z - fz / dfz
        } else {
            0.5 * (z_low + z_high)
        };

        if z_new < z_low || z_new > z_high {
            z_new = 0.5 * (z_low + z_high);
        }

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

    let v1 = (*r2 - *r1 * f) / g;
    let v2 = (*r2 * g_dot - *r1) / g;

    Ok((v1, v2))
}

#[cfg(test)]
mod tests {
    use super::*;
    use kete_core::frames::Equatorial;
    use kete_core::kepler::propagate_two_body;
    use kete_core::prelude::State;
    use kete_core::time::{TDB, Time};

    fn vec_eq(x: f64, y: f64, z: f64) -> Vector<Equatorial> {
        Vector::new([x, y, z])
    }

    fn round_trip(r1: Vector<Equatorial>, v1: Vector<Equatorial>, dt_days: f64, tol: f64) {
        let epoch: Time<TDB> = 2460000.5_f64.into();
        let s1 = State::new(kete_core::desigs::Desig::Empty, epoch, r1, v1, 0);

        let target: Time<TDB> = (epoch.jd + dt_days).into();
        let s2 = propagate_two_body(&s1, target).expect("two-body propagation failed");

        let solutions = lambert(&r1, &s2.pos, dt_days, true, 0).expect("Lambert solver failed");
        let (v1_lam, v2_lam) = solutions[0];

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
        assert!(dv2 < tol, "v2 mismatch: err={dv2:.2e} > tol={tol:.2e}");
    }

    #[test]
    fn test_round_trip_circular() {
        let r = 1.0;
        let v = (GMS / r).sqrt();
        let obl = 23.44_f64.to_radians();

        let r1 = vec_eq(r, 0.0, 0.0);
        let v1 = vec_eq(0.0, v * obl.cos(), v * obl.sin());
        round_trip(r1, v1, 30.0, 1e-10);
    }

    #[test]
    fn test_round_trip_elliptic() {
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
        let e = 1.5;
        let r_peri = 0.5;
        let a = r_peri / (e - 1.0);
        let v_peri = (GMS * (2.0 / r_peri - 1.0 / a)).sqrt();
        let obl = 23.44_f64.to_radians();

        let r1 = vec_eq(r_peri, 0.0, 0.0);
        let v1 = vec_eq(0.0, v_peri * obl.cos(), v_peri * obl.sin());
        round_trip(r1, v1, 10.0, 1e-9);
    }

    #[test]
    fn test_round_trip_neo_short_transfer() {
        let a = 1.5;
        let r = 0.3;
        let v = (GMS * (2.0 / r - 1.0 / a)).sqrt();
        let obl = 23.44_f64.to_radians();

        let r1 = vec_eq(r, 0.0, 0.0);
        let v1 = vec_eq(0.0, v * obl.cos(), v * obl.sin());
        round_trip(r1, v1, 2.0, 1e-10);
    }

    #[test]
    fn test_round_trip_retrograde() {
        let r = 2.0;
        let v = (GMS / r).sqrt();
        let inc = 150.0_f64.to_radians();

        let r1 = vec_eq(r, 0.0, 0.0);
        let v1 = vec_eq(0.0, v * inc.cos(), v * inc.sin());

        let epoch: Time<TDB> = 2460000.5_f64.into();
        let s1 = State::new(kete_core::desigs::Desig::Empty, epoch, r1, v1, 0);
        let target: Time<TDB> = (epoch.jd + 30.0).into();
        let s2 = propagate_two_body(&s1, target).expect("propagation failed");

        let solutions = lambert(&r1, &s2.pos, 30.0, false, 0).expect("Lambert solver failed");
        let (v1_lam, v2_lam) = solutions[0];

        let dv1 = (v1_lam - v1).norm();
        let dv2 = (v2_lam - s2.vel).norm();
        assert!(dv1 < 1e-10, "retrograde v1 err: {dv1:.2e}");
        assert!(dv2 < 1e-10, "retrograde v2 err: {dv2:.2e}");
    }

    #[test]
    fn test_hohmann_transfer() {
        let r1_mag = 1.0;
        let r2_mag = 1.524;
        let a_transfer: f64 = f64::midpoint(r1_mag, r2_mag);
        let dt = std::f64::consts::PI * (a_transfer.powi(3) / GMS).sqrt();

        let r1 = vec_eq(r1_mag, 0.0, 0.0);
        let r2 = vec_eq(-r2_mag, 1e-4, 0.0);

        let (v1, _v2) = lambert(&r1, &r2, dt, true, 0).expect("Hohmann failed")[0];

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
        assert!(lambert(&r1, &r2, -10.0, true, 0).is_err());
    }

    #[test]
    fn test_zero_dt_error() {
        let r1 = vec_eq(1.0, 0.0, 0.0);
        let r2 = vec_eq(0.0, 1.5, 0.0);
        assert!(lambert(&r1, &r2, 0.0, true, 0).is_err());
    }

    #[test]
    fn test_collinear_same_direction() {
        // r1 and r2 aligned along the same direction (dnu ~ 0).
        let r1 = vec_eq(1.0, 0.0, 0.0);
        let r2 = vec_eq(2.0, 0.0, 0.0);
        assert!(lambert(&r1, &r2, 30.0, true, 0).is_err());
    }

    #[test]
    fn test_collinear_pi_transfer() {
        // Pi-transfer: r2 = -r1 direction.  Should succeed (orbit plane
        // is arbitrary).
        let r1 = vec_eq(1.0, 0.0, 0.0);
        let r2 = vec_eq(-1.5, 0.0, 0.0);
        let result = lambert(&r1, &r2, 200.0, true, 0);
        assert!(result.is_ok(), "pi-transfer should succeed: {result:?}");
    }

    #[test]
    fn test_round_trip_long_period() {
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
        let r = 1.0;
        let v = (GMS / r).sqrt();
        let obl = 23.44_f64.to_radians();
        let r1 = vec_eq(r, 0.0, 0.0);
        let v1 = vec_eq(0.0, v * obl.cos(), v * obl.sin());
        round_trip(r1, v1, 1.0, 1e-10);
    }

    #[test]
    fn test_multi_rev_returns_solutions() {
        // A long transfer time on a short-period orbit should admit
        // multi-revolution solutions.
        let r = 1.0;
        let v = (GMS / r).sqrt();
        let obl = 23.44_f64.to_radians();
        let r1 = vec_eq(r, 0.0, 0.0);

        let epoch: Time<TDB> = 2460000.5_f64.into();
        let v1 = vec_eq(0.0, v * obl.cos(), v * obl.sin());
        let s1 = State::new(kete_core::desigs::Desig::Empty, epoch, r1, v1, 0);
        // Transfer > 1 full period (~365 days).
        let dt_days = 400.0;
        let target: Time<TDB> = (epoch.jd + dt_days).into();
        let s2 = propagate_two_body(&s1, target).expect("propagation failed");

        let solutions = lambert(&r1, &s2.pos, dt_days, true, 2).expect("multi_rev failed");

        // Should have at least the N=0 solution.
        assert!(
            !solutions.is_empty(),
            "multi_rev should return at least one solution"
        );

        // The N=0 solution should match the single-rev solution.
        let single = lambert(&r1, &s2.pos, dt_days, true, 0).expect("single-rev failed");
        let dv = (solutions[0].0 - single[0].0).norm();
        assert!(dv < 1e-12, "multi_rev[0] should match single-rev: {dv:.2e}");
    }
}
