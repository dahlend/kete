// SPDX-FileCopyrightText: 2026 Dar Dahlen
// SPDX-License-Identifier: BSD-3-Clause

//! Lambert's problem solver.
//!
//! Given two heliocentric position vectors and a transfer time, find the
//! Keplerian orbit that connects them.  Handles single-revolution and
//! multi-revolution solutions on elliptic, parabolic and hyperbolic orbits,
//! for transfer times from seconds to many orbital periods.
//!
//! Uses the universal-variable formulation with Stumpff functions for the
//! time-of-flight equation, solved by Newton-Raphson inside a bracket which
//! is bisected whenever a Newton step leaves it.  The quantities which vanish
//! on a short arc are built from the chord between the two positions, so the
//! velocity is as accurate as the rounding of the positions allows.  When the
//! positions are on opposite sides of the Sun they do not define an orbit
//! plane, and the plane closest to the reference plane of the frame is used.

use std::f64::consts::{PI, SQRT_2, TAU};

use kete_core::constants::GMS;
use kete_core::frames::{InertialFrame, Vector};
use kete_core::kepler::stumpff_c2_c3;
use kete_core::prelude::{Error, KeteResult};

/// Iteration limit of the root search in `z`.
const MAX_ITER: usize = 60;

/// Relative time-of-flight residual at which the root search stops.
const CONVERGED: f64 = 1e-14;

/// Largest relative time-of-flight residual accepted as a solution.
const ACCEPT: f64 = 1e-9;

/// Solve Lambert's problem for up to `max_revs` complete revolutions.
///
/// Returns a `Vec` of `(v1, v2)` pairs.  The first element is the N=0
/// (single-revolution) solution.  Each revolution count N=1..`max_revs` which
/// admits a solution then adds two pairs, the two orbits which complete N
/// revolutions in the transfer time, ordered by increasing change in
/// eccentric anomaly.
///
/// # Arguments
///
/// * `r1` -- Heliocentric position at departure (AU).
/// * `r2` -- Heliocentric position at arrival (AU).
/// * `dt` -- Transfer time in days.  Must be positive.
/// * `prograde` -- If `true`, selects the orbit whose angular momentum has a
///   non-negative `z` component in the frame of `r1` and `r2`.  If `false`,
///   selects the orbit with a negative `z` component.  This is the short way
///   between the two positions only when the object moves in that direction,
///   otherwise it is the long way around the Sun.  Both are solutions of
///   Lambert's problem, so the wrong choice returns a valid orbit which is
///   not the one the object is on.  The sign is taken in the frame of the
///   inputs.  The poles of the Equatorial and Ecliptic frames are 23.4
///   degrees apart, so an orbit within that angle of polar can be prograde
///   in one frame and retrograde in the other.
/// * `max_revs` -- Maximum number of complete revolutions to try.
///   Pass 0 for the single-revolution solution only.
///
/// # Errors
///
/// * `dt <= 0`
/// * Positions at the origin.
/// * Positions along the same direction from the Sun.
/// * The single-revolution solution does not converge, whatever `max_revs`.
pub fn lambert<T: InertialFrame>(
    r1: &Vector<T>,
    r2: &Vector<T>,
    dt: f64,
    prograde: bool,
    max_revs: u32,
) -> KeteResult<Vec<(Vector<T>, Vector<T>)>> {
    let mut results = lambert_core(r1, r2, dt, prograde, 0)?;
    for n in 1..=max_revs {
        // Multi-rev solutions stop existing beyond a maximum N.
        let Ok(sols) = lambert_core(r1, r2, dt, prograde, n) else {
            break;
        };
        results.extend(sols);
    }
    Ok(results)
}

/// Find the root of a function which increases across the bracket `low` to
/// `high`, from a starting point `z` inside it.
///
/// `func` returns the value, the derivative and the size of the terms of the
/// value.  Newton steps are taken while they land inside the bracket, and the
/// bracket is bisected otherwise.  The bracket is updated with each evaluated
/// point before the next point is chosen.  The search stops when the value is
/// within `tol` of zero, when a Newton step no longer changes `z`, or when the
/// bracket has no floating point number left inside it.  Returns the last
/// point evaluated, with its value and size, so the caller can check it
/// without evaluating it again.
fn find_root(
    func: impl Fn(f64) -> (f64, f64, f64),
    mut low: f64,
    mut high: f64,
    mut z: f64,
    tol: f64,
) -> (f64, f64, f64) {
    for _ in 0..MAX_ITER {
        let (fz, dfz, size) = func(z);
        if fz.abs() <= tol {
            return (z, fz, size);
        }
        if fz < 0.0 {
            low = z;
        } else {
            high = z;
        }
        let mut z_new = z - fz / dfz;
        if z_new == z && dfz.is_finite() {
            // The Newton step is below the resolution of z.
            return (z, fz, size);
        }
        let inside = z_new > low && z_new < high;
        if !inside {
            z_new = f64::midpoint(low, high);
            if z_new <= low || z_new >= high {
                return (z, fz, size);
            }
        }
        z = z_new;
    }
    let (fz, _, size) = func(z);
    (z, fz, size)
}

/// Core Lambert solver for a specific revolution count N.
///
/// Returns one solution for N=0 and two for N>0.
fn lambert_core<T: InertialFrame>(
    r1: &Vector<T>,
    r2: &Vector<T>,
    dt: f64,
    prograde: bool,
    n_revs: u32,
) -> KeteResult<Vec<(Vector<T>, Vector<T>)>> {
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

    // The chord is the only short length in a short arc, and the difference
    // of two nearby vectors is exact.  Every small quantity below is built
    // from it, none from the difference of two rounded lengths.
    let chord = *r2 - *r1;
    let cross = r1.cross(&chord);
    let cross_mag = cross.norm();
    let dot = r1.dot(r2);
    let sin_dnu = cross_mag / (r1_mag * r2_mag);

    // The requested direction of motion decides which way around the Sun the
    // transfer goes.
    let short_way = (cross[2] >= 0.0) == prograde;

    // Unit vector along the angular momentum of the transfer orbit.
    let normal = if sin_dnu > 1e-14 {
        cross * (if short_way { 1.0 } else { -1.0 } / cross_mag)
    } else if dot > 0.0 {
        return Err(Error::ValueError(
            "Lambert: positions are nearly identical (transfer angle ~ 0)".into(),
        ));
    } else {
        // Pi-transfer: the positions are on opposite sides of the Sun and
        // every plane through them holds a solution.  Use the plane closest
        // to the reference plane of the frame.
        let r1_hat = *r1 / r1_mag;
        let mut pole = Vector::<T>::new([0.0, 0.0, 1.0]) - r1_hat * r1_hat[2];
        if pole.norm() < 1e-8 {
            pole = Vector::<T>::new([1.0, 0.0, 0.0]) - r1_hat * r1_hat[0];
        }
        pole * (if prograde { 1.0 } else { -1.0 } / pole.norm())
    };

    // Angle between the positions, in [0, pi].  The transfer angle in the
    // direction of motion is this angle the short way, and 2 pi less this
    // angle the long way.
    let angle = cross_mag.atan2(dot);
    let (sin_half, cos_half) = (0.5 * angle).sin_cos();
    let sin_quarter = (0.25 * angle).sin();
    // 1 - cos(angle / 2)
    let vers_half = 2.0 * sin_quarter * sin_quarter;

    // A = sin(dnu) sqrt(r1 r2 / (1 - cos(dnu))) = sqrt(2 r1 r2) cos(dnu / 2),
    // positive the short way and negative the long way.
    let a_abs = (2.0 * r1_mag * r2_mag).sqrt() * cos_half;
    let a_coeff = if short_way { a_abs } else { -a_abs };

    // The smallest value y can take for z >= 0 is r1 + r2 - sqrt(2) |A|, which
    // is about chord^2 / (4 r) when the positions are close.  It is written
    // here as (sqrt(r2) - sqrt(r1))^2 + 4 sqrt(r1 r2) sin^2(angle / 4).
    let sqrt_r1 = r1_mag.sqrt();
    let sqrt_r2 = r2_mag.sqrt();
    let delta_r = chord.dot(&(*r1 + *r2)) / (r1_mag + r2_mag);
    let delta_sqrt = delta_r / (sqrt_r1 + sqrt_r2);
    let y_floor = delta_sqrt * delta_sqrt + 2.0 * sqrt_r1 * sqrt_r2 * vers_half;

    // Target value: sqrt(GMS) * dt.
    let sqrt_mu_dt = GMS.sqrt() * dt;

    // u(z) = (y(z) - y_floor) / (sqrt(2) |A|).  With (1 - z S) / sqrt(C) =
    // sqrt(2) cos(phi) and sqrt(z) / 2 = N pi + phi, this is 1 - cos(phi) the
    // short way and 1 + cos(phi) the long way, with cosh for negative z.
    let u_of_z = |z: f64| -> f64 {
        if z < 0.0 {
            let half = (0.25 * (-z).sqrt()).sinh();
            if short_way {
                -2.0 * half * half
            } else {
                2.0 + 2.0 * half * half
            }
        } else {
            let (sin, cos) = (0.5 * (0.5 * z.sqrt() - f64::from(n_revs) * PI)).sin_cos();
            if short_way {
                2.0 * sin * sin
            } else {
                2.0 * cos * cos
            }
        }
    };

    // y(z) = r1 + r2 + A (z S - 1) / sqrt(C) = y_floor + sqrt(2) |A| u(z).
    // The short way it is zero at the lower end of the N=0 range.
    let y_of_z = |z: f64| -> f64 { (y_floor + SQRT_2 * a_abs * u_of_z(z)).max(0.0) };

    // Factor by which the rounding error of y(z) exceeds its size.  It is 1
    // except on short way hyperbolic transfers, where the two terms of y
    // cancel as the speed grows.
    let y_cancellation =
        |z: f64| -> f64 { (y_floor + SQRT_2 * a_abs * u_of_z(z).abs()) / y_of_z(z) };

    // F(z) = [y/C]^{3/2} * S + A * sqrt(y) - sqrt(mu) * dt, dF/dz, and the
    // size of the terms summed in F, which sets its rounding error.
    let f_terms_of_z = |z: f64| -> (f64, f64, f64) {
        let y = y_of_z(z);
        let (cz, sz) = stumpff_c2_c3(z);
        let ratio = y / cz;
        let yc32 = ratio * ratio.sqrt();
        let (flight, chord_term) = (yc32 * sz, a_coeff * y.sqrt());
        let fz = flight + chord_term - sqrt_mu_dt;

        // The first form cancels as z goes to 0, where its limit is 1/80.
        let stumpff_term = if z.abs() < 1e-6 {
            1.0 / 80.0
        } else {
            (cz - 1.5 * sz / cz) / (2.0 * z) + 0.75 * sz * sz / cz
        };
        let dfz = yc32 * stumpff_term
            + a_coeff / 8.0 * (3.0 * sz * y.sqrt() / cz + a_coeff * (cz / y).sqrt());
        (fz, dfz, flight + chord_term.abs() + sqrt_mu_dt)
    };

    let tol = CONVERGED * sqrt_mu_dt;

    let roots = if n_revs == 0 {
        // F increases with z from -sqrt(mu) dt at the lower end of the range
        // to infinity at (2 pi)^2.
        let z_high = TAU * TAU;
        let (z_low, z_start) = if short_way && a_abs > 0.0 {
            // y is zero at the lower end.
            let x = 4.0 * (y_floor / (2.0 * SQRT_2 * a_abs)).sqrt().asinh();
            // F is at least A sqrt(y) - sqrt(mu) dt, so the root is at or
            // below the z where that is zero.  On a hyperbolic transfer this
            // is close to the root, and a start at z = 0 is far from it.
            let u_start = ((sqrt_mu_dt / a_abs).powi(2) - y_floor) / (SQRT_2 * a_abs);
            let z_start = if u_start < 0.0 {
                let x_start = 4.0 * (-0.5 * u_start).sqrt().asinh();
                -x_start * x_start
            } else {
                0.0
            };
            (-x * x, z_start)
        } else {
            // y is positive for all z, step down until F is negative.
            let mut z_low = -z_high;
            while f_terms_of_z(z_low).0 > 0.0 && z_low > -1e6 {
                z_low *= 2.0;
            }
            (z_low, 0.0)
        };
        vec![find_root(f_terms_of_z, z_low, z_high, z_start, tol)]
    } else {
        // F is infinite at both ends of the range ((2 N pi)^2, (2 (N+1) pi)^2)
        // and has one minimum between them.  Any point where F is negative
        // separates the two roots, so the search for the minimum, where dF/dz
        // changes sign, stops at the first such point; it runs to the minimum
        // only to show that there is no root.
        let lower = TAU * f64::from(n_revs);
        let upper = TAU * (f64::from(n_revs) + 1.0);
        let (z_low, z_high) = (lower * lower, upper * upper);
        let (mut left, mut right) = (z_low, z_high);
        let mut split = None;
        for _ in 0..MAX_ITER {
            let mid = f64::midpoint(left, right);
            let (f_mid, df_mid, _) = f_terms_of_z(mid);
            if f_mid < 0.0 {
                split = Some(mid);
                break;
            }
            if df_mid < 0.0 {
                left = mid;
            } else {
                right = mid;
            }
        }
        let z_min = split.unwrap_or(f64::midpoint(left, right));
        let f_min = f_terms_of_z(z_min).0;
        if f_min.is_nan() || f_min >= 0.0 {
            return Err(Error::Convergence(format!(
                "Lambert: no {n_revs}-revolution solution exists for this transfer time"
            )));
        }
        // F decreases on the left branch, so search its negative there.
        let negated = |z: f64| -> (f64, f64, f64) {
            let (fz, dfz, size) = f_terms_of_z(z);
            (-fz, -dfz, size)
        };
        vec![
            find_root(negated, z_low, z_min, f64::midpoint(z_low, z_min), tol),
            find_root(
                f_terms_of_z,
                z_min,
                z_high,
                f64::midpoint(z_min, z_high),
                tol,
            ),
        ]
    };

    // In-plane unit vectors perpendicular to each position, along the motion.
    let r1_hat = *r1 / r1_mag;
    let r2_hat = *r2 / r2_mag;
    let t1_hat = normal.cross(&r1_hat);
    let t2_hat = normal.cross(&r2_hat);

    let mut solutions = Vec::with_capacity(roots.len());
    for (z, fz, size) in roots {
        // Check convergence.  The residual is not known better than the
        // rounding of the terms in F.  These exceed the transfer time on long
        // way hyperbolic transfers, and carry the cancellation in y on short
        // way ones.  A NaN residual fails this test.
        let y = y_of_z(z);
        let rounding = f64::EPSILON * size * y_cancellation(z);
        let converged = fz.abs() + rounding <= ACCEPT * sqrt_mu_dt && y > 0.0;
        if !converged {
            return Err(Error::Convergence(format!(
                "Lambert: failed to converge (residual = {fz:.2e})"
            )));
        }

        // The Lagrange coefficients f = 1 - y/r1, g = A sqrt(y/mu) and
        // g_dot = 1 - y/r2 give v1 = (r2 - f r1) / g and v2 = (g_dot r2 - r1) / g.
        // Both are 0/0 at a transfer angle of pi, and subtract nearly equal
        // vectors on a short arc.  The radial and transverse components
        // below are the same velocities with g divided out.
        let u = u_of_z(z);
        let radial = (2.0 * GMS / y).sqrt() * if short_way { 1.0 } else { -1.0 };
        let transverse = sin_half * (2.0 * GMS / y).sqrt();
        let ratio = sqrt_r2 / sqrt_r1;
        let v1 = r1_hat * (radial * (delta_sqrt / sqrt_r1 - ratio * vers_half + u))
            + t1_hat * (transverse * ratio);
        let v2 = r2_hat * (radial * (delta_sqrt / sqrt_r2 + vers_half / ratio - u))
            + t2_hat * (transverse / ratio);
        solutions.push((v1, v2));
    }
    Ok(solutions)
}

#[cfg(test)]
mod tests {
    use super::*;
    use kete_core::frames::{Equatorial, SunCenter};
    use kete_core::kepler::propagate_two_body;
    use kete_core::prelude::State;
    use kete_core::time::{TDB, Time};

    fn round_trip(r1: Vector<Equatorial>, v1: Vector<Equatorial>, dt_days: f64, tol: f64) {
        let epoch: Time<TDB> = 2460000.5_f64.into();
        let s1 = State::<Equatorial, SunCenter> {
            desig: kete_core::desigs::Desig::Empty,
            epoch,
            pos: r1,
            vel: v1,
            center: SunCenter,
        };

        let target: Time<TDB> = epoch + dt_days;
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

        let r1 = Vector::<Equatorial>::new([r, 0.0, 0.0]);
        let v1 = Vector::<Equatorial>::new([0.0, v * obl.cos(), v * obl.sin()]);
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

        let r1 = Vector::<Equatorial>::new([r_peri, 0.0, 0.0]);
        let v1 = Vector::<Equatorial>::new([0.0, v_peri * tilt.cos(), v_peri * tilt.sin()]);
        round_trip(r1, v1, 50.0, 1e-10);
    }

    #[test]
    fn test_round_trip_high_eccentricity() {
        let a = 5.0;
        let e = 0.9;
        let r_peri = a * (1.0 - e);
        let v_peri = (GMS * (2.0 / r_peri - 1.0 / a)).sqrt();
        let obl = 23.44_f64.to_radians();

        let r1 = Vector::<Equatorial>::new([r_peri, 0.0, 0.0]);
        let v1 = Vector::<Equatorial>::new([0.0, v_peri * obl.cos(), v_peri * obl.sin()]);
        round_trip(r1, v1, 20.0, 1e-9);
    }

    #[test]
    fn test_round_trip_hyperbolic() {
        let e = 1.5;
        let r_peri = 0.5;
        let a = r_peri / (e - 1.0);
        let v_peri = (GMS * (2.0 / r_peri - 1.0 / a)).sqrt();
        let obl = 23.44_f64.to_radians();

        let r1 = Vector::<Equatorial>::new([r_peri, 0.0, 0.0]);
        let v1 = Vector::<Equatorial>::new([0.0, v_peri * obl.cos(), v_peri * obl.sin()]);
        round_trip(r1, v1, 10.0, 1e-9);
    }

    #[test]
    fn test_round_trip_neo_short_transfer() {
        let a = 1.5;
        let r = 0.3;
        let v = (GMS * (2.0 / r - 1.0 / a)).sqrt();
        let obl = 23.44_f64.to_radians();

        let r1 = Vector::<Equatorial>::new([r, 0.0, 0.0]);
        let v1 = Vector::<Equatorial>::new([0.0, v * obl.cos(), v * obl.sin()]);
        round_trip(r1, v1, 2.0, 1e-10);
    }

    #[test]
    fn test_round_trip_retrograde() {
        let r = 2.0;
        let v = (GMS / r).sqrt();
        let inc = 150.0_f64.to_radians();

        let r1 = Vector::<Equatorial>::new([r, 0.0, 0.0]);
        let v1 = Vector::<Equatorial>::new([0.0, v * inc.cos(), v * inc.sin()]);

        let epoch: Time<TDB> = 2460000.5_f64.into();
        let s1 = State::<Equatorial, SunCenter> {
            desig: kete_core::desigs::Desig::Empty,
            epoch,
            pos: r1,
            vel: v1,
            center: SunCenter,
        };
        let target: Time<TDB> = epoch + 30.0;
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
        let dt = PI * (a_transfer.powi(3) / GMS).sqrt();

        let r1 = Vector::<Equatorial>::new([r1_mag, 0.0, 0.0]);
        let r2 = Vector::<Equatorial>::new([-r2_mag, 1e-4, 0.0]);

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
        let r1 = Vector::<Equatorial>::new([1.0, 0.0, 0.0]);
        let r2 = Vector::<Equatorial>::new([0.0, 1.5, 0.0]);
        assert!(lambert(&r1, &r2, -10.0, true, 0).is_err());
    }

    #[test]
    fn test_zero_dt_error() {
        let r1 = Vector::<Equatorial>::new([1.0, 0.0, 0.0]);
        let r2 = Vector::<Equatorial>::new([0.0, 1.5, 0.0]);
        assert!(lambert(&r1, &r2, 0.0, true, 0).is_err());
    }

    #[test]
    fn test_collinear_same_direction() {
        // r1 and r2 aligned along the same direction (dnu ~ 0).
        let r1 = Vector::<Equatorial>::new([1.0, 0.0, 0.0]);
        let r2 = Vector::<Equatorial>::new([2.0, 0.0, 0.0]);
        assert!(lambert(&r1, &r2, 30.0, true, 0).is_err());
    }

    #[test]
    fn test_single_rev_failure_is_an_error() {
        // A failure of the single-revolution solution is an error also when
        // more revolutions are asked for, so that the first element is
        // always that solution.
        let r1 = Vector::<Equatorial>::new([1.0, 0.0, 0.0]);
        let r2 = Vector::<Equatorial>::new([0.0, 1.5, 0.0]);
        for max_revs in [0, 1, 3] {
            assert!(lambert(&r1, &r2, -10.0, true, max_revs).is_err());
            let same = lambert(&r1, &(r1 * 2.0), 3000.0, true, max_revs);
            assert!(same.is_err(), "{max_revs}: {same:?}");
        }
    }

    #[test]
    fn test_collinear_pi_transfer() {
        // Pi-transfer: r2 = -r1 direction.  Should succeed (orbit plane
        // is arbitrary).
        let r1 = Vector::<Equatorial>::new([1.0, 0.0, 0.0]);
        let r2 = Vector::<Equatorial>::new([-1.5, 0.0, 0.0]);
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

        let r1 = Vector::<Equatorial>::new([r_peri, 0.0, 0.0]);
        let v1 = Vector::<Equatorial>::new([0.0, v_peri * tilt.cos(), v_peri * tilt.sin()]);
        round_trip(r1, v1, 200.0, 1e-9);
    }

    #[test]
    fn test_round_trip_small_angle() {
        let r = 1.0;
        let v = (GMS / r).sqrt();
        let obl = 23.44_f64.to_radians();
        let r1 = Vector::<Equatorial>::new([r, 0.0, 0.0]);
        let v1 = Vector::<Equatorial>::new([0.0, v * obl.cos(), v * obl.sin()]);
        round_trip(r1, v1, 1.0, 1e-10);
    }

    #[test]
    fn test_multi_rev_returns_solutions() {
        // A long transfer time on a short-period orbit should admit
        // multi-revolution solutions.
        let r = 1.0;
        let v = (GMS / r).sqrt();
        let obl = 23.44_f64.to_radians();
        let r1 = Vector::<Equatorial>::new([r, 0.0, 0.0]);

        let epoch: Time<TDB> = 2460000.5_f64.into();
        let v1 = Vector::<Equatorial>::new([0.0, v * obl.cos(), v * obl.sin()]);
        let s1 = State::<Equatorial, SunCenter> {
            desig: kete_core::desigs::Desig::Empty,
            epoch,
            pos: r1,
            vel: v1,
            center: SunCenter,
        };
        // Transfer > 1 full period (~365 days).
        let dt_days = 400.0;
        let target: Time<TDB> = epoch + dt_days;
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
    /// Exact two-body state `dt_days` after the given one.
    fn two_body(
        r1: Vector<Equatorial>,
        v1: Vector<Equatorial>,
        dt_days: f64,
    ) -> State<Equatorial, SunCenter> {
        let epoch: Time<TDB> = 2460000.5_f64.into();
        let s1 = State::<Equatorial, SunCenter> {
            desig: kete_core::desigs::Desig::Empty,
            epoch,
            pos: r1,
            vel: v1,
            center: SunCenter,
        };
        propagate_two_body(&s1, epoch + dt_days).expect("two-body propagation failed")
    }

    /// Distance by which the orbit from `(r1, v1)` misses `r2` after
    /// `dt_days`, as a fraction of the chord from `r1` to `r2`.
    fn miss(
        r1: Vector<Equatorial>,
        v1: Vector<Equatorial>,
        r2: Vector<Equatorial>,
        dt_days: f64,
    ) -> f64 {
        (two_body(r1, v1, dt_days).pos - r2).norm() / (r2 - r1).norm()
    }

    #[test]
    fn test_round_trip_short_arcs() {
        // Arcs from one second to one day, where the chord is a small
        // fraction of the heliocentric distance.  The positions are rounded
        // to f64, which limits the velocity to about EPSILON * r / chord.
        let inc = 0.4_f64;
        for r in [0.3, 1.0, 2.7, 40.0, 100.0] {
            let v = (GMS / r).sqrt();
            let r1 = Vector::<Equatorial>::new([0.6 * r, 0.8 * r, 0.0]);
            let v1 = Vector::<Equatorial>::new([
                -0.8 * v * inc.cos(),
                0.6 * v * inc.cos(),
                v * inc.sin(),
            ]);
            for dt in [1.0 / 86400.0, 1.0 / 1440.0, 1.0 / 24.0, 1.0] {
                let s2 = two_body(r1, v1, dt);
                let (v1_lam, v2_lam) =
                    lambert(&r1, &s2.pos, dt, true, 0).expect("short arc failed")[0];
                let rounding = f64::EPSILON * r / (s2.pos - r1).norm();
                let err1 = (v1_lam - v1).norm() / v;
                let err2 = (v2_lam - s2.vel).norm() / v;
                println!(
                    "r = {r} AU, dt = {dt:.1e} d: v1 error {err1:.1e}, v2 error {err2:.1e}, \
                     rounding {rounding:.1e}"
                );
                let tol = 10.0 * rounding + 1e-13;
                assert!(err1 < tol, "r = {r}, dt = {dt:e}: v1 error {err1:e}");
                assert!(err2 < tol, "r = {r}, dt = {dt:e}: v2 error {err2:e}");
            }
        }
    }

    #[test]
    fn test_round_trip_distant_arcs() {
        // Arcs of weeks to months at the distance of the outer planets.
        let inc = 0.2_f64;
        for (r, dt) in [(40.0, 20.0), (61.61, 48.72), (100.0, 200.0), (15.0, 7.0)] {
            let v = (GMS / r).sqrt();
            let r1 = Vector::<Equatorial>::new([-0.8 * r, 0.6 * r, 0.0]);
            let v1 = Vector::<Equatorial>::new([
                -0.6 * v * inc.cos(),
                -0.8 * v * inc.cos(),
                v * inc.sin(),
            ]);
            let s2 = two_body(r1, v1, dt);
            let (v1_lam, _) = lambert(&r1, &s2.pos, dt, true, 0).expect("distant arc failed")[0];
            let err = (v1_lam - v1).norm() / v;
            println!("r = {r} AU, dt = {dt} d: v1 error {err:.1e}");
            assert!(err < 1e-11, "r = {r}, dt = {dt}: v1 error {err:e}");
        }
    }

    #[test]
    fn test_round_trip_strongly_hyperbolic() {
        // Speeds of 1, 3 and 30 times the escape speed.
        let r1 = Vector::<Equatorial>::new([1.5, 0.2, -0.3]);
        let dir = Vector::<Equatorial>::new([-0.3, 0.9, 0.2]);
        let escape = (2.0 * GMS / r1.norm()).sqrt();
        for ratio in [1.0, 3.0, 30.0] {
            let v1 = dir * (ratio * escape / dir.norm());
            for dt in [0.01, 1.0, 100.0] {
                let s2 = two_body(r1, v1, dt);
                let (v1_lam, v2_lam) =
                    lambert(&r1, &s2.pos, dt, true, 0).expect("hyperbolic transfer failed")[0];
                let err1 = (v1_lam - v1).norm() / v1.norm();
                let err2 = (v2_lam - s2.vel).norm() / v1.norm();
                println!("{ratio} x escape, dt = {dt} d: v1 error {err1:.1e}, v2 error {err2:.1e}");
                assert!(
                    err1 < 1e-9,
                    "{ratio} x escape, dt = {dt}: v1 error {err1:e}"
                );
                assert!(
                    err2 < 1e-9,
                    "{ratio} x escape, dt = {dt}: v2 error {err2:e}"
                );
            }
        }
    }

    #[test]
    fn test_prograde_selects_angular_momentum_sign() {
        // A retrograde orbit, and a transfer of less than half a revolution.
        // The flag picks the sign of the z angular momentum, so `true`
        // returns the other orbit through the two positions, the long way
        // around the Sun.
        let r = 2.0;
        let v = (GMS / r).sqrt();
        let inc = 150.0_f64.to_radians();
        let r1 = Vector::<Equatorial>::new([r, 0.0, 0.0]);
        let v1 = Vector::<Equatorial>::new([0.0, v * inc.cos(), v * inc.sin()]);
        let dt = 400.0;
        let s2 = two_body(r1, v1, dt);

        let retro = lambert(&r1, &s2.pos, dt, false, 0).expect("retrograde failed")[0].0;
        let pro = lambert(&r1, &s2.pos, dt, true, 0).expect("prograde failed")[0].0;
        assert!((retro - v1).norm() < 1e-12 * v, "retrograde is the orbit");
        assert!(r1.cross(&retro)[2] < 0.0, "retrograde has negative h_z");
        assert!(r1.cross(&pro)[2] > 0.0, "prograde has positive h_z");
        let pro_miss = miss(r1, pro, s2.pos, dt);
        assert!(pro_miss < 1e-9, "prograde misses r2 by {pro_miss:e}");
    }

    #[test]
    fn test_pi_transfer_is_hohmann() {
        // Positions exactly opposite each other.  The plane is the reference
        // plane of the frame, and the speeds are those of the Hohmann ellipse.
        let (r1_mag, r2_mag) = (1.0, 1.524);
        let a_transfer: f64 = f64::midpoint(r1_mag, r2_mag);
        let dt = PI * (a_transfer.powi(3) / GMS).sqrt();
        let speed1 = (GMS * (2.0 / r1_mag - 1.0 / a_transfer)).sqrt();
        let speed2 = (GMS * (2.0 / r2_mag - 1.0 / a_transfer)).sqrt();

        let r1 = Vector::<Equatorial>::new([r1_mag, 0.0, 0.0]);
        let r2 = Vector::<Equatorial>::new([-r2_mag, 0.0, 0.0]);
        for (prograde, sign) in [(true, 1.0), (false, -1.0)] {
            let (v1, v2) = lambert(&r1, &r2, dt, prograde, 0).expect("pi-transfer failed")[0];
            let err1 = (v1 - Vector::new([0.0, sign * speed1, 0.0])).norm();
            let err2 = (v2 - Vector::new([0.0, -sign * speed2, 0.0])).norm();
            assert!(err1 < 1e-14, "v1 error {err1:e}");
            assert!(err2 < 1e-14, "v2 error {err2:e}");
        }
    }

    #[test]
    fn test_multi_rev_recovers_orbit() {
        // An orbit followed for 2.4 periods is one of the two N=2 solutions.
        let a: f64 = 2.0;
        let e = 0.3;
        let r_peri = a * (1.0 - e);
        let v_peri = (GMS * (2.0 / r_peri - 1.0 / a)).sqrt();
        let tilt = 0.5_f64;
        let r1 = Vector::<Equatorial>::new([r_peri, 0.0, 0.0]);
        let v1 = Vector::<Equatorial>::new([0.0, v_peri * tilt.cos(), v_peri * tilt.sin()]);
        let dt = 2.4 * TAU * (a.powi(3) / GMS).sqrt();
        let s2 = two_body(r1, v1, dt);

        let solutions = lambert(&r1, &s2.pos, dt, true, 2).expect("multi-rev failed");
        assert_eq!(solutions.len(), 5, "one N=0, two N=1 and two N=2 solutions");
        for (v1_lam, _) in &solutions {
            let sol_miss = miss(r1, *v1_lam, s2.pos, dt);
            assert!(sol_miss < 1e-9, "solution misses r2 by {sol_miss:e}");
        }
        let closest = solutions[3..]
            .iter()
            .map(|(v1_lam, _)| (*v1_lam - v1).norm() / v_peri)
            .fold(f64::INFINITY, f64::min);
        assert!(
            closest < 1e-12,
            "N=2 solutions miss the orbit by {closest:e}"
        );
    }

    #[test]
    fn test_single_rev_over_many_periods() {
        // The N=0 solution exists for any transfer time.  Over many periods
        // of a circular orbit at r1 it is a much larger orbit.
        let r = 0.48;
        let v = (GMS / r).sqrt();
        let r1 = Vector::<Equatorial>::new([r, 0.0, 0.0]);
        let v1 = Vector::<Equatorial>::new([0.0, v * 0.9, v * 0.19_f64.sqrt()]);
        let dt = 992.6;
        let s2 = two_body(r1, v1, dt);
        let v1_lam = lambert(&r1, &s2.pos, dt, true, 0).expect("long transfer failed")[0].0;
        let sol_miss = miss(r1, v1_lam, s2.pos, dt);
        assert!(sol_miss < 1e-9, "solution misses r2 by {sol_miss:e}");
    }
}
