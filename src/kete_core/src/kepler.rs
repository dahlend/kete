// SPDX-FileCopyrightText: 2026 Dar Dahlen
// SPDX-FileCopyrightText: 2025 California Institute of Technology
// SPDX-License-Identifier: BSD-3-Clause

//! Functions related to two-body motion.
//! These are very fast to compute, however are not very accurate in multi-body systems
//! such as the solar system.

use crate::constants::{C_AU_PER_DAY_INV, GMS};
use crate::errors::Error;
use crate::frames::{InertialFrame, SunCenter, Vector};
use crate::prelude::KeteResult;
use crate::state::State;
use crate::time::{Duration, TDB, Time};
use core::f64;
use nalgebra::{ComplexField, Vector3};
use std::f64::consts::TAU;

/// Propagate a Sun-centered [`State`] with two-body motion about the Sun.
///
/// This ignores the planets, so it is an approximation of the motion.
///
/// # Errors
/// Returns [`Error::Convergence`] if the state is not finite, or if the Kepler
/// solver fails after ten levels of step halving.
pub fn propagate_two_body<T: InertialFrame>(
    state: &State<T, SunCenter>,
    time_final: Time<TDB>,
) -> KeteResult<State<T, SunCenter>> {
    let (pos, vel) = analytic_2_body(
        time_final - state.epoch,
        &state.pos.into(),
        &state.vel.into(),
    )?;

    Ok(State {
        desig: state.desig.clone(),
        epoch: time_final,
        pos: pos.into(),
        vel: vel.into(),
        center: state.center,
    })
}

/// Apply geometric light-time correction to a state.
///
/// The state must be Sun-centered. `observer_pos` is the observer's
/// Sun-centered position in the same frame as the state.
/// Uses two-body backward propagation by the light travel time, iterated up
/// to 3 times until the change in light-travel time is below 1e-12 days.
///
/// # Errors
/// Returns an error if the Kepler solver fails.
pub fn light_time_correct<T: InertialFrame>(
    state: &State<T, SunCenter>,
    observer_pos: &Vector<T>,
) -> KeteResult<State<T, SunCenter>> {
    let mut corrected = state.clone();
    let mut tau = 0.0;

    for _ in 0..3 {
        let dx = corrected.pos - observer_pos;
        let new_tau = dx.norm() * C_AU_PER_DAY_INV;
        if (new_tau - tau).abs() < 1e-12 {
            break;
        }
        tau = new_tau;
        corrected = propagate_two_body(state, state.epoch - tau)?;
    }

    Ok(corrected)
}

/// Propagate an object forward in time by the specified amount assuming only 2 body
/// mechanics.
///
/// If the solver does not converge, this splits the step in halves. It splits
/// at most ten levels deep.
///
/// # Arguments
///
/// * `time` - Time to propagate in days.
/// * `pos` - Starting position, from the center of the sun, in AU.
/// * `vel` - Starting velocity, in AU/Day.
///
/// # Errors
/// Returns [`Error::Convergence`] if the input is not finite, or if the solver
/// still fails after ten levels of halving.
pub fn analytic_2_body(
    time: Duration,
    pos: &Vector3<f64>,
    vel: &Vector3<f64>,
) -> KeteResult<(Vector3<f64>, Vector3<f64>)> {
    two_body_halving(time.elapsed, pos, vel, 0)
}

/// Two-body propagation returning position and velocity increments.
///
/// This is the same computation as [`analytic_2_body`], but it returns the change
/// in position and velocity rather than the new state. The increments are computed
/// without catastrophic cancellation (f-hat / g-dot-hat formulation, WHFAST eqs
/// 37-39), which allows callers to apply them with compensated summation. This is
/// used as the Kepler drift of the Wisdom-Holman symplectic integrator.
///
/// Unlike [`analytic_2_body`], this does not subdivide the time interval when the
/// solver fails to converge, it returns an error immediately.
///
/// # Arguments
///
/// * `time` - Time to propagate in days.
/// * `pos` - Starting position, from the center of the sun, in AU.
/// * `vel` - Starting velocity, in AU/Day.
/// * `mu` - Gravitational parameter of the central body in AU^3/day^2. Pass
///   [`GMS`] for a solar orbit, or a radiation-reduced `(1 - beta) GMS` for a
///   dust grain whose central force is gravity minus radiation pressure.
///
/// # Errors
/// Fails if the input contains non-finite values or the universal Kepler solver
/// does not converge.
pub fn analytic_2_body_delta(
    time: f64,
    pos: &Vector3<f64>,
    vel: &Vector3<f64>,
    mu: f64,
) -> KeteResult<(Vector3<f64>, Vector3<f64>)> {
    let r0 = pos.norm();
    let v0_sq = vel.norm_squared();
    let rv0 = pos.dot(vel);

    if !rv0.is_finite() {
        Err(Error::Convergence("Input included infinity or NAN.".into()))?;
    }
    let (universal_s, beta) = solve_kepler_universal(time, r0, v0_sq, rv0, mu)?;
    let [_, g1, g2, _] = universal_g(universal_s, beta);
    // f-hat / g-dot-hat formulation (WHFAST eqs 37-39).
    // Compute the small corrections separately, so the caller may add the initial
    // values last to avoid catastrophic cancellation when f ~ 1 and g_dot ~ 1.
    let f_hat = -mu * g2 / r0;
    let g = r0 * g1 + rv0 * g2;
    let d_pos = pos * f_hat + vel * g;
    let new_r0 = (pos + d_pos).norm();
    let f_dot = -mu / (new_r0 * r0) * g1;
    let g_dot_hat = -mu * g2 / new_r0;
    let d_vel = pos * f_dot + vel * g_dot_hat;
    Ok((d_pos, d_vel))
}

/// Compute the eccentric anomaly for all orbital classes.
///
/// For `ecc <= 1` this solves Kepler's equation `M = E - e sin(E)`. The result
/// is the eccentric anomaly in `[0, 2 pi)`. At `ecc = 1` this is the limit of
/// the elliptical solution as the eccentricity approaches one.
///
/// For `ecc > 1` this solves the hyperbolic form `M = e sinh(H) - H`. The
/// result is the hyperbolic anomaly.
///
/// The equations are evaluated as `(E - sin E) + (1 - e) sin E` and
/// `(sinh H - H) + (e - 1) sinh H`. For small anomalies the first term comes
/// from its series. This form avoids the cancellation in `E - e sin E` as the
/// eccentricity approaches one.
///
/// # Arguments
///
/// * `ecc` - The eccentricity, must be non-negative.
/// * `mean_anom` - Mean anomaly in radians.
///
/// # Errors
///
/// Returns [`Error::ValueError`] if `ecc` or `mean_anom` is not finite, or if
/// `ecc` is negative. Returns [`Error::Convergence`] if the iteration does not
/// converge in 50 steps.
pub fn compute_eccentric_anomaly(ecc: f64, mean_anom: f64) -> KeteResult<f64> {
    if !ecc.is_finite() || !mean_anom.is_finite() {
        Err(Error::ValueError(
            "Eccentricity and mean anomaly must be finite values".into(),
        ))?;
    }
    if ecc < 0.0 {
        Err(Error::ValueError(
            "Eccentricity must be greater than 0".into(),
        ))?;
    }
    // Each start is the smaller of cbrt(6 |M|) and M + 0.85 e sign(M) for the
    // ellipse, or ln(2 |M| / e + 1.8) sign(M) for the hyperbola. Near unit
    // eccentricity and small M both equations approach M = x^3 / 6, so there
    // the cube root is close to the root.
    if ecc <= 1.0 {
        // M is reduced to [-pi, pi]. Subtracting whole turns leaves a small M
        // exact. At e = 1 the derivative vanishes at E = 0, so M = 0 returns
        // before the iteration.
        let m = mean_anom - TAU * (mean_anom / TAU).round();
        if m == 0.0 {
            return Ok(0.0);
        }
        let mut ecc_anom = (m.abs() + 0.85 * ecc).min((6.0 * m.abs()).cbrt()) * m.signum();
        for _ in 0..50 {
            let (sin_e, cos_e) = ecc_anom.sin_cos();
            let residual = odd_tail(ecc_anom, -1.0) + (1.0 - ecc) * sin_e - m;
            // 1 - e cos(E) as (1 - e) + e (1 - cos(E)), with 1 - cos(E) in a
            // form that does not cancel for small E.
            let one_minus_cos = if cos_e > 0.0 {
                sin_e * sin_e / (1.0 + cos_e)
            } else {
                1.0 - cos_e
            };
            let step = residual / ((1.0 - ecc) + ecc * one_minus_cos);
            ecc_anom -= step;
            if step.abs() <= 4.0 * f64::EPSILON * ecc_anom.abs().max(1.0) {
                return Ok(ecc_anom.rem_euclid(TAU));
            }
        }
    } else {
        let start = (2.0 * mean_anom.abs() / ecc + 1.8).ln();
        let mut hyp_anom = start.min((6.0 * mean_anom.abs()).cbrt()) * mean_anom.signum();
        for _ in 0..50 {
            let (sinh_h, cosh_h) = hyp_anom.sinh_cosh();
            let residual = odd_tail(hyp_anom, 1.0) + (ecc - 1.0) * sinh_h - mean_anom;
            let cosh_minus_one = sinh_h * sinh_h / (cosh_h + 1.0);
            let step = residual / ((ecc - 1.0) + ecc * cosh_minus_one);
            hyp_anom -= step;
            if step.abs() <= 4.0 * f64::EPSILON * hyp_anom.abs().max(1.0) {
                return Ok(hyp_anom);
            }
        }
    }
    Err(Error::Convergence(
        "Failed to solve Kepler's equation".into(),
    ))
}

/// Osculating semi-major axis from a position and velocity about a central body, via the
/// vis-viva relation.
///
///   a = 1 / (2 / r - v^2 / GM)
///
/// Negative for a hyperbolic orbit, infinite for an exactly parabolic one.
///
/// This is the cheap scalar form, taking `mu` explicitly and returning without an error
/// path or an allocation, so it is usable inside an integrator's inner loop.
/// [`CometElements`](crate::elements::CometElements) and [`EquinoctialElements`](crate::elements::EquinoctialElements)
/// expose the same quantity as a method when a full element set is wanted.
///
/// # Arguments
///
/// * `pos` - Position relative to the central body in AU.
/// * `vel` - Velocity relative to the central body in AU/Day.
/// * `mu` - Gravitational parameter of the central body in AU^3/day^2.
#[must_use]
pub fn compute_semi_major(pos: &Vector3<f64>, vel: &Vector3<f64>, mu: f64) -> f64 {
    (2.0 / pos.norm() - vel.norm_squared() / mu).recip()
}

/// Perihelion distance of the osculating orbit from a position and velocity about a
/// central body.
///
///   q = p / (1 + e),  p = |r x v|^2 / GM
///   e = |(v^2 - GM / r) r - (r . v) v| / GM
///
/// Valid for any conic. The eccentricity comes from the eccentricity vector.
/// This keeps q at full precision on a nearly circular orbit, where the energy
/// form `e^2 = 1 + 2 E p / GM` cancels.
///
/// # Arguments
///
/// * `pos` - Position relative to the central body in AU.
/// * `vel` - Velocity relative to the central body in AU/Day.
/// * `mu` - Gravitational parameter of the central body in AU^3/day^2.
#[must_use]
pub fn compute_peri_dist(pos: &Vector3<f64>, vel: &Vector3<f64>, mu: f64) -> f64 {
    let semi_latus = pos.cross(vel).norm_squared() / mu;
    let ecc_vec = (vel.norm_squared() - mu / pos.norm()) * pos - pos.dot(vel) * vel;
    semi_latus / (1.0 + ecc_vec.norm() / mu)
}

/// Compute the Stumpff functions `c2(z)` and `c3(z)` for any `z`.
///
/// For `z > 0`, `c2(z) = (1 - cos(sqrt(z))) / z` and
/// `c3(z) = (sqrt(z) - sin(sqrt(z))) / sqrt(z)^3`. For `z < 0` the functions
/// use `cosh` and `sinh` in place of `cos` and `sin`. At `z = 0` they are `1/2`
/// and `1/6`.
///
/// For `|z| < 1` the closed forms cancel. There the values come from the series
/// `c2 = sum (-z)^k / (2k + 2)!` and `c3 = sum (-z)^k / (2k + 3)!`. Each sum
/// stops when a term no longer changes it.
#[must_use]
pub fn stumpff_c2_c3(z: f64) -> (f64, f64) {
    if z.abs() >= 1.0 {
        let x = z.abs().sqrt();
        return if z > 0.0 {
            let half = (0.5 * x).sin();
            (2.0 * half * half / z, (x - x.sin()) / (z * x))
        } else {
            let half = (0.5 * x).sinh();
            (-2.0 * half * half / z, (x.sinh() - x) / (-z * x))
        };
    }
    let (mut c2, mut c3) = (0.5, 1.0 / 6.0);
    let (mut term2, mut term3) = (0.5, 1.0 / 6.0);
    for k in 0..12_u32 {
        let k = f64::from(k);
        term2 *= -z / ((2.0 * k + 3.0) * (2.0 * k + 4.0));
        term3 *= -z / ((2.0 * k + 4.0) * (2.0 * k + 5.0));
        let (next2, next3) = (c2 + term2, c3 + term3);
        if next2 == c2 && next3 == c3 {
            break;
        }
        (c2, c3) = (next2, next3);
    }
    (c2, c3)
}

/// Propagate as [`analytic_2_body`] does, halving the step on a solver failure.
///
/// `depth` is the number of times the step has been halved.
///
/// # Errors
///
/// Returns [`Error::Convergence`] if the input is not finite, or if `depth`
/// reaches ten.
fn two_body_halving(
    time: f64,
    pos: &Vector3<f64>,
    vel: &Vector3<f64>,
    depth: usize,
) -> KeteResult<(Vector3<f64>, Vector3<f64>)> {
    if depth >= 10 {
        Err(Error::Convergence(
            "Two body recursion depth reached.".into(),
        ))?;
    }
    if !(pos.dot(vel)).is_finite() {
        Err(Error::Convergence("Input included infinity or NAN.".into()))?;
    }
    if let Ok((d_pos, d_vel)) = analytic_2_body_delta(time, pos, vel, GMS) {
        Ok((d_pos + pos, d_vel + vel))
    } else {
        let (inter_pos, inter_vel) = two_body_halving(0.5 * time, pos, vel, depth + 1)?;
        two_body_halving(0.5 * time, &inter_pos, &inter_vel, depth + 1)
    }
}

/// Solve the kepler equation for a universal formulation.
///
/// This finds the universal anomaly `s` at which `r0 G1 + rv0 G2 + mu G3 = dt`.
/// See [`universal_g`] for the functions. The same equation applies to every
/// conic, so the solution is continuous through the parabolic limit.
///
/// # Arguments
///
/// * `dt` - The step size in days.
/// * `r0` - Distance from the central body (AU).
/// * `v0_sq` - Squared speed with respect to the central body (AU^2/day^2).
/// * `rv0` - R vector dotted with the V vector, not normalized.
/// * `mu` - Gravitational parameter of the central body in AU^3/day^2. This is
///   [`GMS`] for solar orbits; a radiation-reduced value `(1 - beta) GMS` for
///   dust grains.
///
/// Returns the universal anomaly and `beta = 2 mu / r0 - v0_sq`.
///
/// # Errors
///
/// Returns [`Error::Convergence`] if the iteration does not converge in 100
/// steps, or if a step is not finite.
pub(crate) fn solve_kepler_universal(
    mut dt: f64,
    r0: f64,
    v0_sq: f64,
    rv0: f64,
    mu: f64,
) -> KeteResult<(f64, f64)> {
    // beta is mu / semi_major
    let beta = 2.0 * mu / r0 - v0_sq;
    let b_sqrt = beta.abs().sqrt();

    if beta > 0.0 {
        // A whole number of periods returns to the start.
        let period = mu * b_sqrt.powi(-3) * TAU;
        if period.is_finite() {
            dt %= period;
        }
    }

    // The parabolic guess is the root of r0 s + rv0 s^2 / 2 + mu s^3 / 6 = dt.
    // It is used only near the parabolic limit. That requires r0 |beta| / mu,
    // which is r0 / |a|, to be small. It also requires |beta s^2| < 1 at the
    // root. Elsewhere the elliptic and hyperbolic guesses are closer and
    // cheaper.
    let parabolic = if r0 * beta.abs() < 1e-3 * mu {
        let a0 = -6.0 * dt / mu;
        let a1 = 6.0 * r0 / mu;
        let a2 = 3.0 * rv0 / mu;
        let p = (3.0 * a1 - a2.powi(2)) / 3.0;
        let q = (9.0 * a1 * a2 - 27.0 * a0 - 2.0 * a2.powi(3)) / 27.0;
        let w = (q / 2.0 + (q.powi(2) / 4.0 + p.powi(3) / 27.0).sqrt()).cbrt();
        w - p / (3.0 * w) - a2 / 3.0
    } else {
        f64::NAN
    };
    let guess = if parabolic.is_finite() && (beta * parabolic * parabolic).abs() < 1.0 {
        parabolic
    } else if beta > 0.0 {
        // One Halley step on the modified Kepler equation starting from the
        // mean anomaly.
        let period = mu * b_sqrt.powi(-3);
        let ec = 1.0 - r0 * beta / mu;
        let es = rv0 * b_sqrt / mu;
        let dm = dt / period;
        let (sin_dm, cos_dm) = dm.sin_cos();
        let num = ec * sin_dm - es * (1.0 - cos_dm);
        let den = 1.0 - ec * cos_dm + es * sin_dm;
        let fpp = ec * sin_dm + es * cos_dm;
        let halley_denom = 2.0 * den * den + num * fpp;
        if halley_denom.abs() > den.abs() * 1e-6 {
            (dm + 2.0 * num * den / halley_denom) / b_sqrt
        } else {
            (dm + num) / b_sqrt
        }
    } else {
        // For large arguments sinh and cosh both approach exp(b s) / 2.
        // Inverting the equation in that form gives
        //
        //   s ~ sign(dt) ln(2 b^3 |dt| / (mu + r0 |beta| + sign(dt) rv0 b)) / b
        //
        // The denominator is positive for every hyperbolic state. This follows
        // from (mu + r0 |beta|)^2 - (r0 v0 b)^2 = mu^2 and |rv0| <= r0 v0.
        // The inversion applies only when the argument of the logarithm exceeds
        // one. Below that, dt / r0 is the better start, because dt = r ds and r
        // is near r0 over a short arc.
        // A long arc from a small distance makes dt / r0 too large. The clamp
        // keeps sinh finite, and the iteration moves down from there.
        let sign = dt.signum();
        let guess_arg =
            2.0 * beta.abs() * b_sqrt * dt.abs() / (mu + r0 * beta.abs() + sign * rv0 * b_sqrt);
        let guess = if guess_arg > 1.0 {
            sign * guess_arg.ln() / b_sqrt
        } else {
            dt / r0
        };
        guess.clamp(-100.0 / b_sqrt, 100.0 / b_sqrt)
    };
    // Halley's method on the time residual. The residual, its derivative (the
    // distance) and its second derivative share one evaluation of the universal
    // functions. The loop takes the last computed step after the residual meets
    // the tolerance, at no extra cost.
    //
    // The tolerance is relative to the terms of the residual. Their rounding
    // grows with the step, so a fixed tolerance in days fails on long arcs.
    //
    // The derivative of the residual is the distance, so the residual increases
    // with s. Each evaluation narrows a bracket on the root. The residual is
    // -dt at s = 0, so the root has the sign of dt. On an elliptical orbit the
    // reduced step is less than one period, so |s| <= 2 pi / sqrt(beta).
    //
    // Far from the root a Halley step can point away from it. A step outside
    // the bracket is replaced by the Newton step, which always points toward
    // the root. If that is also outside, the bracket midpoint replaces it.
    let span = if beta > 0.0 {
        TAU / b_sqrt
    } else {
        f64::INFINITY
    };
    let (mut lo, mut hi) = if dt > 0.0 { (0.0, span) } else { (-span, 0.0) };
    let mut s = guess.clamp(lo, hi);
    for _ in 0..100 {
        let [g0, g1, g2, g3] = universal_g(s, beta);
        let (t1, t2, t3) = (r0 * g1, rv0 * g2, mu * g3);
        let residual = t1 + t2 + t3 - dt;
        let radius = r0 * g0 + rv0 * g1 + mu * g2;
        let radius_rate = rv0 * g0 + (mu - beta * r0) * g1;
        let step = 2.0 * residual * radius / (2.0 * radius * radius - residual * radius_rate);
        let next = s - step;
        let tol = 1e-11 * (t1.abs() + t2.abs() + t3.abs() + dt.abs()).max(1.0);
        if residual.abs() < tol && next.is_finite() {
            return Ok((next, beta));
        }
        if residual > 0.0 {
            hi = s;
        } else {
            lo = s;
        }
        let newton = s - residual / radius;
        s = if next > lo && next < hi {
            next
        } else if newton > lo && newton < hi {
            newton
        } else if lo.is_finite() && hi.is_finite() {
            f64::midpoint(lo, hi)
        } else {
            break;
        };
    }
    Err(Error::Convergence(
        "Failed to solve universal kepler equation".into(),
    ))
}

/// Compute the universal functions `G0` to `G3` at universal anomaly `s`.
///
/// `beta = 2 mu / r - v^2` is the same for every point on the orbit. The
/// functions satisfy `dG(k+1)/ds = Gk` and `dG0/ds = -beta G1`.
///
/// A two-body orbit starts at distance `r0` with `r . v = rv0`. At universal
/// anomaly `s` it reaches time `t = r0 G1 + rv0 G2 + mu G3`. Its distance there
/// is `r = r0 G0 + rv0 G1 + mu G2`.
///
/// The same functions apply to every conic. For `|beta s^2| < 1e-2` all four
/// come from the series of `G2` and `G3`. Otherwise `G0`, `G1` and `G2` come
/// from the sine and cosine of `sqrt(beta) s / 2`. For `beta < 0` the
/// hyperbolic sine and cosine replace them.
///
///  Wisdom, Jack, and David M. Hernandez.
///  "A fast and accurate universal Kepler solver without Stumpff series."
///  Monthly Notices of the Royal Astronomical Society 453.3 (2015): 3015-3023.
///  <https://arxiv.org/abs/1508.02699>
pub(crate) fn universal_g(s: f64, beta: f64) -> [f64; 4] {
    let z = beta * s * s;
    if z.abs() < 1e-2 {
        // The closed forms cancel near the parabolic limit.
        let (c2, c3) = stumpff_c2_c3(z);
        let g2 = s * s * c2;
        let g3 = s * s * s * c3;
        // G0 = 1 - beta G2 and G1 = s - beta G3 follow from the derivatives.
        return [1.0 - beta * g2, s - beta * g3, g2, g3];
    }
    // G0, G1 and G2 share one sine and cosine of the half angle. This keeps the
    // identities between them consistent in rounding, and the energy
    // conservation of the Wisdom-Holman drift depends on those identities.
    // G3 = (s - G1) / beta cancels for |z| < 1, so there it comes from its
    // series.
    let b_sqrt = beta.abs().sqrt();
    let half = 0.5 * b_sqrt * s;
    let (g0, g1, g2) = if beta > 0.0 {
        let (sin_h, cos_h) = half.sin_cos();
        (
            1.0 - 2.0 * sin_h * sin_h,
            2.0 * sin_h * cos_h / b_sqrt,
            2.0 * sin_h * sin_h / beta,
        )
    } else {
        let (sinh_h, cosh_h) = half.sinh_cosh();
        (
            1.0 + 2.0 * sinh_h * sinh_h,
            2.0 * sinh_h * cosh_h / b_sqrt,
            -2.0 * sinh_h * sinh_h / beta,
        )
    };
    let g3 = if z.abs() < 1.0 {
        s * s * s * stumpff_c2_c3(z).1
    } else {
        (s - g1) / beta
    };
    [g0, g1, g2, g3]
}

/// Compute `x - sin(x)` for `sign = -1` or `sinh(x) - x` for `sign = 1`.
///
/// For `|x| <= 0.5` the value comes from the series
/// `x^3 / 3! + sign x^5 / 5! + x^7 / 7! + ...`. The closed form cancels there.
fn odd_tail(x: f64, sign: f64) -> f64 {
    if x.abs() > 0.5 {
        return if sign < 0.0 {
            x - x.sin()
        } else {
            x.sinh() - x
        };
    }
    // The sum stops once a term no longer changes it.
    let x2 = x * x;
    let mut term = x * x2 / 6.0;
    let mut sum = term;
    for k in 1..12_u32 {
        let k = f64::from(k);
        term *= sign * x2 / ((2.0 * k + 2.0) * (2.0 * k + 3.0));
        let next = sum + term;
        if next == sum {
            break;
        }
        sum = next;
    }
    sum
}

#[cfg(test)]
mod tests {
    use std::f64::consts::TAU;

    use super::*;
    use crate::constants::GMS_SQRT;
    use nalgebra::Vector3;

    use super::compute_eccentric_anomaly;

    /// A circular orbit about a reduced central mass `mu = (1-beta) GMS`
    /// (gravity minus radiation pressure) closes after its own period
    /// `2 pi sqrt(a^3 / mu)` and conserves the reduced-gravity energy. This
    /// certifies the `mu` parameter of the universal solver.
    #[test]
    fn test_kepler_reduced_mu() {
        for beta in [0.0, 0.1, 0.3] {
            let mu = (1.0 - beta) * GMS;
            let r = 1.5;
            let v = (mu / r).sqrt(); // circular speed for the reduced gravity
            let pos = Vector3::new(0.0, r, 0.0);
            let vel = Vector3::new(-v, 0.0, 0.0);
            let period = TAU * (r.powi(3) / mu).sqrt();

            let (d_pos, d_vel) = analytic_2_body_delta(period, &pos, &vel, mu).unwrap();
            let end_pos = pos + d_pos;
            let end_vel = vel + d_vel;
            let pos_err = (end_pos - pos).norm();
            let vel_err = (end_vel - vel).norm();

            // Reduced-gravity specific energy is conserved.
            let e0 = 0.5 * v * v - mu / r;
            let e1 = 0.5 * end_vel.norm_squared() - mu / end_pos.norm();
            let e_rel = ((e1 - e0) / e0).abs();

            println!("reduced_mu beta={beta}: pos closure {pos_err:.2e}, energy {e_rel:.2e}");
            assert!(
                pos_err < 1e-8,
                "beta={beta}: orbit did not close: {pos_err:e}"
            );
            assert!(
                vel_err < 1e-8,
                "beta={beta}: velocity did not close: {vel_err:e}"
            );
            assert!(e_rel < 1e-12, "beta={beta}: energy drift {e_rel:e}");
        }
    }

    /// Compare two-body propagation from perihelion to independent references.
    ///
    /// mpmath computed the reference positions to 60 digits. It solved Kepler's
    /// equation in its elliptic, parabolic and hyperbolic forms. The cases
    /// cover eccentricities below, at and above 1, and two perihelion
    /// distances. Times reach ten years before and after perihelion.
    #[test]
    fn two_body_matches_reference_through_parabolic() {
        // (e - 1, q in AU, days from perihelion, x, y in AU)
        let cases: [(f64, f64, f64, f64, f64); 130] = [
            (-1e-1, 0.3, -3650.0, -2.082956925339187, 1.279710510374317),
            (-1e-1, 0.3, -30.0, -0.27297859750136816, -0.7686467011862986),
            (-1e-1, 0.3, 1.0, 0.2983615664582254, 0.043212315251251944),
            (-1e-1, 0.3, 365.0, -4.000923738937227, 1.1783223289232962),
            (-1e-1, 0.3, 3650.0, -2.082956925339187, -1.279710510374317),
            (-1e-1, 2.0, -3650.0, -18.615206464100336, -8.713672531788221),
            (-1e-1, 2.0, -30.0, 1.9670456579996296, -0.5002466005462172),
            (-1e-1, 2.0, 1.0, 1.9999630113958142, 0.01676642871521971),
            (-1e-1, 2.0, 365.0, -0.5250003915809306, 4.240121914555423),
            (-1e-1, 2.0, 3650.0, -18.615206464100336, 8.713672531788221),
            (
                -1e-3,
                0.3,
                -3650.0,
                -24.981822408436088,
                -5.3893706384548405,
            ),
            (-1e-3, 0.3, -30.0, -0.2613736170605537, -0.8201716593292545),
            (-1e-3, 0.3, 1.0, 0.29836200651060885, 0.044323853936829784),
            (-1e-3, 0.3, 365.0, -4.729902012932185, 2.445872021530506),
            (-1e-3, 0.3, 3650.0, -24.981822408436088, 5.3893706384548405),
            (-1e-3, 2.0, -3650.0, -20.22940556336096, -13.295056528187654),
            (-1e-3, 2.0, -30.0, 1.9670720140630924, -0.513117928093777),
            (-1e-3, 2.0, 1.0, 1.9999630114296754, 0.017197691867846417),
            (-1e-3, 2.0, 365.0, -0.472301548455925, 4.444806491157464),
            (-1e-3, 2.0, 3650.0, -20.22940556336096, 13.295056528187654),
            (-1e-4, 0.3, -3650.0, -25.164097809279195, -5.51595068451678),
            (-1e-4, 0.3, -30.0, -0.26127058418125787, -0.8206266038820408),
            (-1e-4, 0.3, 1.0, 0.29836201050891964, 0.04433383102498484),
            (-1e-4, 0.3, 365.0, -4.7349565853046265, 2.456944423652096),
            (-1e-4, 0.3, 3650.0, -25.164097809279195, 5.51595068451678),
            (
                -1e-4,
                2.0,
                -3650.0,
                -20.240678823729297,
                -13.334826946978906,
            ),
            (-1e-4, 2.0, -30.0, 1.9670722532785234, -0.5132334613376888),
            (-1e-4, 2.0, 1.0, 1.9999630114299831, 0.017201562848572306),
            (-1e-4, 2.0, 365.0, -0.4718385224519791, 4.446627655670136),
            (-1e-4, 2.0, 3650.0, -20.240678823729297, 13.334826946978906),
            (-1e-6, 0.3, -3650.0, -25.18404386550317, -5.529876895179806),
            (-1e-6, 0.3, -30.0, -0.2612592533098957, -0.8206766337557924),
            (-1e-6, 0.3, 1.0, 0.29836201094873144, 0.0443349283676296),
            (-1e-6, 0.3, 365.0, -4.735511196271136, 2.4581618542109847),
            (-1e-6, 0.3, 3650.0, -25.18404386550317, 5.529876895179806),
            (
                -1e-6,
                2.0,
                -3650.0,
                -20.241915749860016,
                -13.339199662126637,
            ),
            (-1e-6, 2.0, -30.0, 1.967072279591798, -0.5132461684087519),
            (-1e-6, 2.0, 1.0, 1.999963011430017, 0.01720198860327207),
            (-1e-6, 2.0, 365.0, -0.4717876066275664, 4.446827942073796),
            (-1e-6, 2.0, 3650.0, -20.241915749860016, 13.339199662126637),
            (-1e-8, 0.3, -3650.0, -25.184243222463135, -5.530016159571083),
            (-1e-8, 0.3, -30.0, -0.2612591400039301, -0.8206771340404976),
            (-1e-8, 0.3, 1.0, 0.29836201095312953, 0.04433493934091891),
            (-1e-8, 0.3, 365.0, -4.73551674099026, 2.458174027982282),
            (-1e-8, 0.3, 3650.0, -25.184243222463135, 5.530016159571083),
            (-1e-8, 2.0, -3650.0, -20.241928115991154, -13.33924338724595),
            (-1e-8, 2.0, -30.0, 1.9670722798549305, -0.5132462954778757),
            (-1e-8, 2.0, 1.0, 1.9999630114300173, 0.017201992860765854),
            (-1e-8, 2.0, 365.0, -0.4717870974863676, 4.44682994489612),
            (-1e-8, 2.0, 3650.0, -20.241928115991154, 13.33924338724595),
            (-1e-10, 0.3, -3650.0, -25.18424521602238, -5.530017552215224),
            (-1e-10, 0.3, -30.0, -0.2612591388708707, -0.8206771390433433),
            (-1e-10, 0.3, 1.0, 0.2983620109531735, 0.04433493945065179),
            (-1e-10, 0.3, 365.0, -4.735516796437312, 2.4581741497199414),
            (-1e-10, 0.3, 3650.0, -25.18424521602238, 5.530017552215224),
            (
                -1e-10,
                2.0,
                -3650.0,
                -20.241928239652154,
                -13.339243824496942,
            ),
            (-1e-10, 2.0, -30.0, 1.9670722798575617, -0.5132462967485668),
            (-1e-10, 2.0, 1.0, 1.9999630114300173, 0.017201992903340783),
            (-1e-10, 2.0, 365.0, -0.4717870923949573, 4.446829964924339),
            (-1e-10, 2.0, 3650.0, -20.241928239652154, 13.339243824496942),
            (0.0, 0.3, -3650.0, -25.18424523615934, -5.530017566282336),
            (0.0, 0.3, -30.0, -0.26125913885942564, -0.8206771390938771),
            (0.0, 0.3, 1.0, 0.29836201095317394, 0.0443349394517602),
            (0.0, 0.3, 365.0, -4.735516796997383, 2.4581741509496147),
            (0.0, 0.3, 3650.0, -25.18424523615934, 5.530017566282336),
            (0.0, 2.0, -3650.0, -20.241928240901256, -13.339243828913618),
            (0.0, 2.0, -30.0, 1.9670722798575884, -0.5132462967614021),
            (0.0, 2.0, 1.0, 1.9999630114300173, 0.017201992903770835),
            (0.0, 2.0, 365.0, -0.47178709234352895, 4.4468299651266445),
            (0.0, 2.0, 3650.0, -20.241928240901256, 13.339243828913618),
            (1e-10, 0.3, -3650.0, -25.1842452562963, -5.5300175803494485),
            (1e-10, 0.3, -30.0, -0.26125913884798063, -0.8206771391444109),
            (1e-10, 0.3, 1.0, 0.2983620109531744, 0.044334939452868614),
            (1e-10, 0.3, 365.0, -4.735516797557454, 2.4581741521792884),
            (1e-10, 0.3, 3650.0, -25.1842452562963, 5.5300175803494485),
            (
                1e-10,
                2.0,
                -3650.0,
                -20.241928242150355,
                -13.339243833330295,
            ),
            (1e-10, 2.0, -30.0, 1.9670722798576148, -0.5132462967742373),
            (1e-10, 2.0, 1.0, 1.9999630114300173, 0.017201992904200884),
            (1e-10, 2.0, 365.0, -0.47178709229210053, 4.446829965328949),
            (1e-10, 2.0, 3650.0, -20.241928242150355, 13.339243833330295),
            (1e-8, 0.3, -3650.0, -25.18424724985534, -5.530018972993593),
            (1e-8, 0.3, -30.0, -0.26125913771492126, -0.8206771441472565),
            (1e-8, 0.3, 1.0, 0.2983620109532184, 0.0443349395626015),
            (1e-8, 0.3, 365.0, -4.735516853004504, 2.4581742739169465),
            (1e-8, 0.3, 3650.0, -25.18424724985534, 5.530018972993593),
            (1e-8, 2.0, -3650.0, -20.241928365811347, -13.33924427058128),
            (1e-8, 2.0, -30.0, 1.9670722798602462, -0.5132462980449284),
            (1e-8, 2.0, 1.0, 1.9999630114300173, 0.017201992946775817),
            (1e-8, 2.0, 365.0, -0.4717870872006903, 4.4468299853571684),
            (1e-8, 2.0, 3650.0, -20.241928365811347, 13.33924427058128),
            (1e-6, 0.3, -3650.0, -25.184446604723803, -5.530158237430777),
            (1e-6, 0.3, -30.0, -0.26125902440901116, -0.8206776444316782),
            (1e-6, 0.3, 1.0, 0.2983620109576165, 0.04433495053588803),
            (1e-6, 0.3, 365.0, -4.735522397695542, 2.4581864476774498),
            (1e-6, 0.3, 3650.0, -25.184446604723803, 5.530158237430777),
            (1e-6, 2.0, -3650.0, -20.24194073187926, -13.339287995659541),
            (1e-6, 2.0, -30.0, 1.9670722801233786, -0.5132464251140202),
            (1e-6, 2.0, 1.0, 1.9999630114300178, 0.017201997204268522),
            (1e-6, 2.0, 365.0, -0.4717865780598358, 4.4468319881786496),
            (1e-6, 2.0, 3650.0, -20.24194073187926, 13.339287995659541),
            (1e-4, 0.3, -3650.0, -25.204371745934377, -5.544084907158868),
            (1e-4, 0.3, -30.0, -0.2612476940928062, -0.8207276714707672),
            (1e-4, 0.3, 1.0, 0.29836201139742785, 0.044336047850828154),
            (1e-4, 0.3, 365.0, -4.736076727799788, 2.4594037702959146),
            (1e-4, 0.3, 3650.0, -25.204371745934377, 5.544084907158868),
            (1e-4, 2.0, -3650.0, -20.24317702575818, -13.343660300281359),
            (1e-4, 2.0, -30.0, 1.9670723064365676, -0.5132591318645259),
            (1e-4, 2.0, 1.0, 1.9999630114300515, 0.017202422948218117),
            (1e-4, 2.0, 365.0, -0.47173566567879743, 4.447032266155893),
            (1e-4, 2.0, 3650.0, -20.24317702575818, 13.343660300281359),
            (1e-3, 0.3, -3650.0, -25.384576223888853, -5.670710433751513),
            (1e-3, 0.3, -30.0, -0.26114471617954493, -0.8211823353638013),
            (1e-3, 0.3, 1.0, 0.29836201539569096, 0.04434602219594878),
            (1e-3, 0.3, 365.0, -4.741103491921665, 2.470465485279666),
            (1e-3, 0.3, 3650.0, -25.384576223888853, 5.670710433751513),
            (1e-3, 2.0, -3650.0, -20.25438768678079, -13.38339007297448),
            (1e-3, 2.0, -30.0, 1.9670725456435345, -0.5133746333700843),
            (1e-3, 2.0, 1.0, 1.9999630114303595, 0.017206292864570615),
            (1e-3, 2.0, 365.0, -0.47127298060304135, 4.448852596369788),
            (1e-3, 2.0, 3650.0, -20.25438768678079, 13.38339007297448),
            (1e-1, 0.3, -3650.0, -38.85636199089856, -19.269492983070734),
            (1e-1, 0.3, -30.0, -0.2500923661825647, -0.8698636213624648),
            (1e-1, 0.3, 1.0, 0.2983624549667596, 0.04542983456020304),
            (1e-1, 0.3, 365.0, -5.178005270517754, 3.6337419290663213),
            (1e-1, 0.3, 3650.0, -38.85636199089856, 19.269492983070734),
            (1e-1, 2.0, -3650.0, -21.220055459154437, -17.55774389599646),
            (1e-1, 2.0, -30.0, 1.9670988162154965, -0.5259251527048854),
            (1e-1, 2.0, 1.0, 1.9999630114642204, 0.01762679743631348),
            (1e-1, 2.0, 365.0, -0.42202128336572287, 4.645091825858574),
            (1e-1, 2.0, 3650.0, -21.220055459154437, 17.55774389599646),
        ];
        let mut worst = 0.0_f64;
        for (de, q, dt, x, y) in cases {
            let ecc = 1.0 + de;
            let pos = Vector3::new(q, 0.0, 0.0);
            let vel = Vector3::new(0.0, (GMS * (1.0 + ecc) / q).sqrt(), 0.0);
            let (new_pos, _) = analytic_2_body(dt.into(), &pos, &vel).unwrap();
            let err = (new_pos - Vector3::new(x, y, 0.0)).norm() / new_pos.norm();
            worst = worst.max(err);
            assert!(
                err < 1e-12,
                "e - 1 = {de:e}, q = {q}, dt = {dt}: relative error {err:e}"
            );
        }
        println!("worst relative position error {worst:e}");
    }

    /// Step an eccentric orbit from near perihelion by 1/30 of a period.
    ///
    /// Here the elliptic starting guess is several revolutions from the root.
    /// The bracket on the root brings the iteration back.
    #[test]
    fn eccentric_orbit_near_perihelion_converges() {
        let (r0, rv0, v0, dt) = (
            1.923_213_690_694_382_4,
            -0.018_004_910_928_878_574,
            0.016_941_113_801_842_038,
            656.473_261_551_525_1,
        );
        let (s, beta) = solve_kepler_universal(dt, r0, v0 * v0, rv0, GMS).unwrap();
        let [_, g1, g2, g3] = universal_g(s, beta);
        let residual = r0 * g1 + rv0 * g2 + GMS * g3 - dt;
        assert!(residual.abs() < 1e-9, "time residual {residual:e}");
    }

    /// Propagate from perihelion over arcs of up to a million days.
    ///
    /// The step is not split, so the solver tolerance must hold on long arcs.
    /// mpmath computed the reference positions to 60 digits.
    #[test]
    fn two_body_long_arcs_from_perihelion() {
        // (e, q in AU, days from perihelion, x, y in AU)
        let cases = [
            (
                0.4363,
                38.28,
                150_000.0,
                -61.410_676_693_131_71,
                -53.998_947_412_369_83,
            ),
            (
                0.4363,
                38.28,
                -200_000.0,
                36.361_193_149_350_946,
                14.422_796_063_659_739,
            ),
            (
                0.8496,
                76.2,
                190_000.0,
                -157.896_243_128_816_02,
                225.260_464_060_304_74,
            ),
            (
                1.0,
                2.0,
                1_000_000.0,
                -1_094.170_264_366_351_1,
                93.644_872_336_556_86,
            ),
            (
                1.5,
                1.0,
                -1_000_000.0,
                -8_118.150_427_823_375,
                -9_079.721_930_718_457,
            ),
            (
                1.5,
                30.0,
                1_000_000.0,
                -1_551.695_886_972_828,
                1_834.245_548_346_615,
            ),
        ];
        for (ecc, q, dt, x, y) in cases {
            let pos = Vector3::new(q, 0.0, 0.0);
            let vel = Vector3::new(0.0, (GMS * (1.0 + ecc) / q).sqrt(), 0.0);
            let (d_pos, _) = analytic_2_body_delta(dt, &pos, &vel, GMS).unwrap();
            let new_pos = pos + d_pos;
            let err = (new_pos - Vector3::new(x, y, 0.0)).norm() / new_pos.norm();
            assert!(
                err < 1e-12,
                "e = {ecc}, q = {q}, dt = {dt}: relative error {err:e}"
            );
        }
    }

    /// Propagate hyperbolic orbits backward through perihelion.
    ///
    /// From these states a Halley step far from the root points away from it.
    /// The step is not split. mpmath computed the reference positions to 50
    /// digits.
    #[test]
    fn hyperbolic_backward_through_perihelion() {
        // (days, position, velocity, final x, final y)
        let cases = [
            (
                -1_197.202_044_812_637,
                [358.599_030_380_909_6, 0.0, 0.0],
                [0.024_429_641_627_670_41, 0.000_335_923_323_090_553_35, 0.0],
                [329.350_068_821_857_06, -0.402_167_389_275_016_44],
            ),
            (
                -0.384_959_174_850_017_6,
                [0.181_430_980_393_227_58, 0.0, 0.0],
                [0.210_457_872_449_530_57, 0.038_725_768_212_143_664, 0.0],
                [0.099_445_999_866_462_05, -0.014_864_734_178_756_086],
            ),
            (
                -20_000.0,
                [50.0, 0.0, 0.0],
                [0.015_615_956_330_919_17, 0.000_257_687_212_602_714_23, 0.0],
                [98.175_619_498_344_66, 246.131_376_113_767_4],
            ),
        ];
        for (dt, pos, vel, [x, y]) in cases {
            let pos = Vector3::from(pos);
            let vel = Vector3::from(vel);
            let (d_pos, _) = analytic_2_body_delta(dt, &pos, &vel, GMS).unwrap();
            let new_pos = pos + d_pos;
            let err = (new_pos - Vector3::new(x, y, 0.0)).norm() / new_pos.norm();
            assert!(err < 1e-12, "dt = {dt}: relative error {err:e}");
        }
    }

    #[test]
    fn test_kepler_circular() {
        for r in [0.2, 0.5, 1.0, 2.0] {
            let pos = Vector3::new(0.0, r, 0.0);
            let vel = Vector3::new(-GMS_SQRT / r.sqrt(), 0.0, 0.0);
            let year = TAU / GMS_SQRT * r.powf(3.0 / 2.0);
            let res = analytic_2_body(year.into(), &pos, &vel).unwrap();
            assert!((res.0 - pos).norm() < 1e-8);
            assert!((res.1 - vel).norm() < 1e-8);

            // go backwards
            let res = analytic_2_body((-year).into(), &pos, &vel).unwrap();
            assert!((res.0 - pos).norm() < 1e-8);
            assert!((res.1 - vel).norm() < 1e-8);
        }
    }

    #[test]
    fn test_compute_eccentric_anomaly_hyperbolic() {
        for mean_anom in -100..100 {
            let mean_anom = f64::from(mean_anom);
            assert!(
                compute_eccentric_anomaly(2.0, mean_anom).is_ok(),
                "Mean Anom: {mean_anom}"
            );
        }
    }

    #[test]
    fn test_kepler_parabolic() {
        let pos = Vector3::new(0.0, 2.0, 0.0);
        let vel = Vector3::new(GMS_SQRT, 0.0, 0.0);
        let year = -TAU / GMS_SQRT;
        let res = analytic_2_body(year.into(), &pos, &vel).unwrap();
        let pos_exp = Vector3::new(-4.448805955479905, -0.4739843046525608, 0.0);
        let vel_exp = -Vector3::new(-0.00768983428326951, -0.008552645144187791, 0.0);
        assert!((res.0 - pos_exp).norm() < 1e-8);
        assert!((res.1 - vel_exp).norm() < 1e-8);

        // go backwards
        let res = analytic_2_body((-year).into(), &pos_exp, &vel_exp).unwrap();
        assert!((res.0 - pos).norm() < 1e-8);
        assert!((res.1 - vel).norm() < 1e-8);
    }

    #[test]
    fn test_kepler_hyperbolic() {
        let pos = Vector3::new(0.0, 3.0, 0.0);
        let vel = Vector3::new(-GMS_SQRT, 0.0, 0.0);
        let year = TAU / GMS_SQRT;
        let res = analytic_2_body(year.into(), &pos, &vel).unwrap();
        let pos_exp = Vector3::new(-5.556785268950049, 1.6076633958058089, 0.0);
        let vel_exp = Vector3::new(-0.013061655543084886, -0.005508140023183166, 0.0);
        assert!((res.0 - pos_exp).norm() < 1e-8);
        assert!((res.1 - vel_exp).norm() < 1e-8);

        // go backwards
        let res = analytic_2_body((-year).into(), &pos_exp, &vel_exp).unwrap();
        assert!((res.0 - pos).norm() < 1e-8);
        assert!((res.1 - vel).norm() < 1e-8);
    }

    /// Solve Kepler's equation near and far from unit eccentricity.
    ///
    /// The test substitutes each root back into the equation.
    #[test]
    fn test_eccentric_anomaly_roundtrip() {
        for ecc in [
            0.0,
            0.3,
            0.9,
            1.0 - 1e-6,
            1.0 - 1e-10,
            1.0,
            1.0 + 1e-10,
            1.0 + 1e-6,
            1.5,
            4.0,
        ] {
            for mean_anom in [-3.0, -1e-6, 0.0, 1e-30, 1e-9, 1e-3, 0.5, 3.0, 20.0] {
                let anom = compute_eccentric_anomaly(ecc, mean_anom).unwrap();
                let back = if ecc <= 1.0 {
                    let m = anom - ecc * anom.sin();
                    m - TAU * ((m - mean_anom) / TAU).round()
                } else {
                    ecc * anom.sinh() - anom
                };
                assert!(
                    (back - mean_anom).abs() <= 1e-12 * mean_anom.abs().max(1e-3),
                    "e = {ecc}, M = {mean_anom}: E = {anom} gives M = {back}"
                );
            }
        }
    }

    /// Measure energy error growth over many orbits to detect solver bias.
    ///
    /// A biased solver will show linear (or worse) energy drift, while an
    /// unbiased solver should show only random-walk (~sqrt(N)) growth.
    /// We propagate a Jupiter-like orbit forward in fixed time steps and
    /// check that the relative energy error remains small.
    #[test]
    fn test_kepler_bias() {
        // Jupiter-like: a = 5.2 AU, e = 0.048
        let a = 5.2;
        let ecc = 0.048;
        let r0 = a * (1.0 - ecc); // perihelion
        let v0 = (GMS * (2.0 / r0 - 1.0 / a)).sqrt();
        let pos = Vector3::new(r0, 0.0, 0.0);
        let vel = Vector3::new(0.0, v0, 0.0);

        let energy_0 = 0.5 * v0 * v0 - GMS / r0;

        let dt = 200.0; // days per step
        let n_steps = 10_000;

        let mut p = pos;
        let mut v = vel;
        for _ in 0..n_steps {
            let (np, nv) = analytic_2_body(dt.into(), &p, &v).unwrap();
            p = np;
            v = nv;
        }

        let energy_final = 0.5 * v.norm_squared() - GMS / p.norm();
        let rel_err = ((energy_final - energy_0) / energy_0).abs();

        // Also test with a higher eccentricity orbit (e=0.5)
        let a2 = 2.5;
        let ecc2 = 0.5;
        let r0_2 = a2 * (1.0 - ecc2);
        let v0_2 = (GMS * (2.0 / r0_2 - 1.0 / a2)).sqrt();
        let pos2 = Vector3::new(r0_2, 0.0, 0.0);
        let vel2 = Vector3::new(0.0, v0_2, 0.0);
        let energy_02 = 0.5 * v0_2 * v0_2 - GMS / r0_2;

        let mut p2 = pos2;
        let mut v2 = vel2;
        for _ in 0..n_steps {
            let (np, nv) = analytic_2_body(dt.into(), &p2, &v2).unwrap();
            p2 = np;
            v2 = nv;
        }
        let energy_f2 = 0.5 * v2.norm_squared() - GMS / p2.norm();
        let rel_err2 = ((energy_f2 - energy_02) / energy_02).abs();

        // The energy error after 10k steps is a random walk of rounding errors.
        // Its size is about sqrt(10^4) times a few ulp.
        assert!(
            rel_err < 1e-13,
            "e=0.048: Energy drift too large after {n_steps} steps: {rel_err:.3e}"
        );
        assert!(
            rel_err2 < 1e-13,
            "e=0.5: Energy drift too large after {n_steps} steps: {rel_err2:.3e}"
        );
    }
}
