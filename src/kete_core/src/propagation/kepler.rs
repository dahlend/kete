//! Functions related to two-body motion.
//! These are very fast to compute, however are not very accurate in multi-body systems
//! such as the solar system.
//!
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

use crate::constants::{GMS, GMS_SQRT};
use crate::errors::Error;
use crate::fitting::newton_raphson;
use crate::frames::InertialFrame;
use crate::prelude::{CometElements, KeteResult};
use crate::state::State;
use argmin::core::{CostFunction, Error as ArgminErr, Executor};
use argmin::solver::neldermead::NelderMead;
use core::f64;
use nalgebra::{ComplexField, Vector3};
use std::f64::consts::TAU;

/// How close to ecc=1 do we assume the orbit is parabolic
pub const PARABOLIC_ECC_LIMIT: f64 = 1e-4;

/// Compute the eccentric anomaly for all orbital classes.
///
/// # Arguments
///
/// * `ecc` - The eccentricity, must be non-negative.
/// * `mean_anomaly` - Mean anomaly.
/// * `peri_dist` - Perihelion Distance in AU, only used for parabolic orbits.
///
pub fn compute_eccentric_anomaly(ecc: f64, mean_anom: f64, peri_dist: f64) -> KeteResult<f64> {
    match ecc {
        ecc if !ecc.is_finite() => Err(Error::ValueError(
            "Eccentricity must be a finite value".into(),
        ))?,
        ecc if ecc < 0.0 => Err(Error::ValueError(
            "Eccentricity must be greater than 0".into(),
        ))?,
        ecc if ecc < 1e-6 => Ok(mean_anom),
        ecc if ecc < 1.0 - PARABOLIC_ECC_LIMIT => {
            // Elliptical
            let f = |ecc_anom: f64| -ecc * ecc_anom.sin() + ecc_anom - mean_anom.rem_euclid(TAU);
            let d = |ecc_anom: f64| 1.0 - 1.0 * ecc * ecc_anom.cos();
            Ok(newton_raphson(f, d, mean_anom.rem_euclid(TAU), 1e-11)?.rem_euclid(TAU))
        }
        ecc if ecc < 1.0 + PARABOLIC_ECC_LIMIT => {
            // Parabolic
            // Find the zero point of -mean_anom + peri_dist * ecc_anom + ecc_anom.powi(3) / 6.0
            // this is a simple cubic equation which can be solved analytically.
            let p = 2.0 * peri_dist;
            let q = 6.0 * mean_anom;

            let w = (0.5 * (q + (q.powi(2) + 4.0 * p.powi(3)).sqrt())).cbrt();
            Ok(w - p / w)
        }
        ecc => {
            // Hyperbolic
            let f = |ecc_anom: f64| (ecc * ecc_anom.sinh() - ecc_anom - mean_anom);
            let d = |ecc_anom: f64| -1.0 + ecc * ecc_anom.cosh();
            let guess = (mean_anom / ecc).asinh();
            Ok(newton_raphson(f, d, guess, 1e-11)?)
        }
    }
}

/// Compute the true anomaly for all orbital classes.
///
/// # Arguments
///
/// * `ecc` - The eccentricity, must be non-negative.
/// * `mean_anomaly` - Mean anomaly, between 0 and 2 pi.
/// * `peri_dist` - Perihelion Distance in AU, only used for parabolic orbits.
///
pub fn compute_true_anomaly(ecc: f64, mean_anom: f64, peri_dist: f64) -> KeteResult<f64> {
    let ecc_anom = compute_eccentric_anomaly(ecc, mean_anom, peri_dist)?;

    let anom = match ecc {
        e if (e - 1.0).abs() < PARABOLIC_ECC_LIMIT => {
            (ecc_anom / (2.0 * peri_dist).sqrt()).atan() * 2.0
        }
        e if e < 1.0 => (((1.0 + ecc) / (1.0 - ecc)).sqrt() * (ecc_anom / 2.0).tan()).atan() * 2.0,
        e if e > 1.0 => {
            ((-(1.0 + ecc) / (1.0 - ecc)).sqrt() * (ecc_anom / 2.0).tanh()).atan() * 2.0
        }
        // Other cases are when the checks above fail, should be only NAN and INF
        _ => f64::NAN,
    };

    Ok(anom.rem_euclid(TAU))
}

/// Compute eccentric anomaly from the true anomaly for all orbital classes.
///
/// # Arguments
///
/// * `ecc` - The eccentricity, must be non-negative.
/// * `true_anomaly` - true anomaly, between 0 and 2 pi.
/// * `peri_dist` - Perihelion Distance in AU, only used for parabolic orbits.
///
pub fn eccentric_anomaly_from_true(ecc: f64, true_anom: f64, peri_dist: f64) -> KeteResult<f64> {
    let ecc_anom = match ecc {
        ecc if !ecc.is_finite() => Err(Error::ValueError(
            "Eccentricity must be a finite value".into(),
        ))?,
        ecc if ecc < 0.0 => Err(Error::ValueError(
            "Eccentricity must be greater than 0".into(),
        ))?,
        e if e < 1e-6 => true_anom,
        e if e < 1.0 - PARABOLIC_ECC_LIMIT => {
            let (sin_true, cos_true) = true_anom.sin_cos();
            let t = (1.0 + ecc * cos_true).recip();
            ((1.0 - ecc.powi(2)).sqrt() * sin_true * t).atan2((ecc + cos_true) * t)
        }
        e if e < 1.0 + PARABOLIC_ECC_LIMIT => (0.5 * true_anom).tan() * (2.0 * peri_dist).sqrt(),
        _ => {
            let v = (-(1.0 - ecc) / (1.0 + ecc)).sqrt();
            ((true_anom * 0.5).tan() * v).atanh() * 2.0
        }
    };
    Ok(ecc_anom.rem_euclid(TAU))
}

// Beta value used below to define a parabolic orbit.
const PARABOLIC_BETA: f64 = 1e-10;

/// This function is used by the kepler universal solver below.
/// They are defined by the referenced paper.
///
///  Wisdom, Jack, and David M. Hernandez.
///  "A fast and accurate universal Kepler solver without Stumpff series."
///  Monthly Notices of the Royal Astronomical Society 453.3 (2015): 3015-3023.
///  <https://arxiv.org/abs/1508.02699>
fn g_1(s: f64, beta: f64) -> f64 {
    // the limit of this equation as beta approaches 0 is s
    if beta.abs() < PARABOLIC_BETA {
        return s;
    }
    let beta_sqrt = beta.abs().sqrt();
    if beta >= 0.0 {
        (beta_sqrt * s).sin() / beta_sqrt
    } else {
        (beta_sqrt * s).sinh() / beta_sqrt
    }
}

/// This function is used by the kepler universal solver below.
/// They are defined by the referenced paper.
///
///  Wisdom, Jack, and David M. Hernandez.
///  "A fast and accurate universal Kepler solver without Stumpff series."
///  Monthly Notices of the Royal Astronomical Society 453.3 (2015): 3015-3023.
///  <https://arxiv.org/abs/1508.02699>
fn g_2(s: f64, beta: f64) -> f64 {
    // the limit of this equation as beta approaches 0 is s^2 / 2
    if beta.abs() < PARABOLIC_BETA {
        return s.powi(2) / 2.0;
    }
    let beta_sqrt = beta.abs().sqrt() * s;
    if beta >= 0.0 {
        2.0 * (beta_sqrt / 2.0).sin().powf(2.0) / beta
        // (1.0 - beta_sqrt.cos()) / beta
    } else {
        (1.0 - (beta_sqrt).cosh()) / beta
    }
}

/// Solve the kepler equation for a universal formulation.
///
/// This roughly follows the outline of the algorithm described here:
///
///  Wisdom, Jack, and David M. Hernandez.
///  "A fast and accurate universal Kepler solver without Stumpff series."
///  Monthly Notices of the Royal Astronomical Society 453.3 (2015): 3015-3023.
///  <https://arxiv.org/abs/1508.02699>
///
/// # Arguments
///
/// * `dt` - The step size in days.
/// * `r0` - Distance from the central body (AU).
/// * `v0` - Velocity with respect to the central body (AU/day)
/// * `rv0` - R vector dotted with the V vector, not normalized.
///
fn solve_kepler_universal(mut dt: f64, r0: f64, v0: f64, rv0: f64) -> KeteResult<(f64, f64)> {
    // beta is GMS / semi_major
    let beta = 2.0 * GMS / r0 - v0.powi(2);
    let b_sqrt = beta.abs().sqrt();

    let res = {
        if beta >= PARABOLIC_BETA {
            let period = GMS * b_sqrt.powi(-3);
            dt %= period * TAU;
            // elliptical orbits
            let f = |x: f64| {
                let g1 = (b_sqrt * x).sin() / b_sqrt;
                let g2 = 2.0 * (b_sqrt * x / 2.0).sin().powf(2.0) / beta;
                let g3 = (x - g1) / beta;
                r0 * g1 + rv0 * g2 + GMS * g3 - dt
            };
            let d = |x: f64| {
                let (g2d, g1d) = (b_sqrt * x).sin_cos();
                let g3d = (1.0 - g1d) / beta;
                r0 * g1d + rv0 * g2d + GMS * g3d
            };
            // Significantly better initial guess.
            let guess = b_sqrt.recip() * dt / period;
            newton_raphson(f, d, guess, 1e-11)
        } else if beta.abs() < PARABOLIC_BETA {
            // This is for parabolic orbits.
            // solve a cubic of the form:
            // x^3 + a2 * x^2 + a1 * x + a0 = 0
            let a0 = -6.0 * dt / GMS;
            let a1 = 6.0 * r0 / GMS;
            let a2 = 3.0 * rv0 / GMS;
            let p = (3.0 * a1 - a2.powi(2)) / 3.0;
            let q = (9.0 * a1 * a2 - 27.0 * a0 - 2.0 * a2.powi(3)) / 27.0;
            let w = (q / 2.0 + (q.powi(2) / 4.0 + p.powi(3) / 27.0).sqrt()).powf(1.0 / 3.0);
            let ans = w - p / (3.0 * w) - a2 / 3.0;
            Ok(ans)
        } else {
            // hyperbolic orbits
            let f = |x: f64| {
                let g1 = (b_sqrt * x).sinh() / b_sqrt;
                let g2 = (1.0 - (b_sqrt * x).cosh()) / beta;
                let g3 = (x - g1) / beta;
                r0 * g1 + rv0 * g2 + GMS * g3 - dt
            };
            let d = |x: f64| {
                let (g2d, g1d) = (b_sqrt * x).sinh_cosh();
                let g3d = (1.0 - g1d) / beta;
                r0 * g1d + rv0 * g2d + GMS * g3d
            };
            newton_raphson(f, d, GMS_SQRT * beta.abs() * dt, 1e-11)
        }
    }
    .map_err(|_| Error::Convergence("Failed to solve universal kepler equation".into()))?;
    Ok((res, beta))
}

/// Propagate an object forward in time by the specified amount assuming only 2 body
/// mechanics.
///
/// This will recursively call itself if it fails to converge, attempting to compute for
/// half intervals of time.
///
/// This roughly follows the outline of the algorithm described here:
///
///  Wisdom, Jack, and David M. Hernandez.
///  "A fast and accurate universal Kepler solver without Stumpff series."
///  Monthly Notices of the Royal Astronomical Society 453.3 (2015): 3015-3023.
///  <https://arxiv.org/abs/1508.02699>
///
/// # Arguments
///
/// * `time` - Time to propagate in days.
/// * `pos` - Starting position, from the center of the sun, in AU.
/// * `vel` - Starting velocity, in AU/Day.
/// * `depth` - Current Recursion depth, this should be called with `None` initially.
pub fn analytic_2_body(
    time: f64,
    pos: &Vector3<f64>,
    vel: &Vector3<f64>,
    depth: Option<usize>,
) -> KeteResult<(Vector3<f64>, Vector3<f64>)> {
    let mut depth = depth.unwrap_or(0);
    if depth >= 10 {
        Err(Error::Convergence(
            "Two body recursion depth reached.".into(),
        ))?;
    } else {
        depth += 1;
    }
    if time.abs() < 1e-10 {
        return Ok((*pos, *vel));
    }
    let r0 = pos.norm();
    let v0 = vel.norm();
    let rv0 = pos.dot(vel);

    if rv0.is_nan() || rv0.is_infinite() {
        Err(Error::Convergence("Input included infinity or NAN.".into()))?;
    }

    match solve_kepler_universal(time, r0, v0, rv0) {
        Ok((universal_s, beta)) => {
            let g1 = g_1(universal_s, beta);
            let g2 = g_2(universal_s, beta);
            let f = 1.0 - GMS / r0 * g2;
            let g = r0 * g1 + rv0 * g2;
            let new_pos = pos * f + vel * g;
            let new_r0 = new_pos.norm();
            let f_dot = -GMS / (new_r0 * r0) * g1;
            let g_dot = 1.0 - (GMS / new_r0) * g2;
            let new_vel = pos * f_dot + vel * g_dot;
            Ok((new_pos, new_vel))
        }
        Err(_) => {
            let (inter_pos, inter_vel) = analytic_2_body(0.5 * time, pos, vel, Some(depth))?;
            analytic_2_body(time * 0.5, &inter_pos, &inter_vel, Some(depth))
        }
    }
}

/// Propagate a [`State`] object using the analytic two body equations of motion.
///
/// Orbits are calculated where the central mass is located at (0, 0, 0) and has the
/// mass of the Sun. It is important that the appropriate center ID is chosen, it is
/// recommended to use a center of the Sun (10) for this computation.
///
/// This is the fastest method for getting a relatively good estimate of the orbits.
pub fn propagate_two_body<T: InertialFrame>(
    state: &State<T>,
    jd_final: f64,
) -> KeteResult<State<T>> {
    let (pos, vel) = analytic_2_body(
        jd_final - state.jd,
        &state.pos.into(),
        &state.vel.into(),
        None,
    )?;

    Ok(State::new(
        state.desig.clone(),
        jd_final,
        pos.into(),
        vel.into(),
        state.center_id,
    ))
}

struct MoidCost<T: InertialFrame> {
    state_a: State<T>,

    state_b: State<T>,
}

impl<T: InertialFrame> CostFunction for MoidCost<T> {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, param: &Self::Param) -> Result<Self::Output, ArgminErr> {
        let dt_a = param.first().unwrap();
        let dt_b = param.last().unwrap();

        let s0 = propagate_two_body(&self.state_a, self.state_a.jd + dt_a)?;
        let s1 = propagate_two_body(&self.state_b, self.state_b.jd + dt_b)?;

        Ok((Vector3::from(s0.pos) - Vector3::from(s1.pos)).norm())
    }
}

/// Compute the MOID between two states in au.
/// MOID = Minimum Orbital Intersection Distance
pub fn moid<T: InertialFrame>(mut state_a: State<T>, mut state_b: State<T>) -> KeteResult<f64> {
    const N_STEPS: i32 = 50;

    let elements_a = CometElements::from_state(&state_a.clone().into_frame());
    state_a = propagate_two_body(&state_a, elements_a.peri_time)?;
    let elements_b = CometElements::from_state(&state_b.clone().into_frame());
    state_b = propagate_two_body(&state_b, elements_b.peri_time)?;

    let state_a_step_size = match elements_a.orbital_period() {
        p if p.is_finite() => p / N_STEPS as f64,
        _ => 300.0 / N_STEPS as f64,
    };
    let state_b_step_size = match elements_b.orbital_period() {
        p if p.is_finite() => p / N_STEPS as f64,
        _ => 300.0 / N_STEPS as f64,
    };

    let mut states_b: Vec<State<_>> = Vec::with_capacity(N_STEPS as usize);
    let mut states_a: Vec<State<_>> = Vec::with_capacity(N_STEPS as usize);

    for idx in (-N_STEPS)..N_STEPS {
        states_a.push(propagate_two_body(
            &state_a,
            state_a.jd + idx as f64 * state_a_step_size,
        )?);
        states_b.push(propagate_two_body(
            &state_b,
            state_b.jd + idx as f64 * state_b_step_size,
        )?);
    }
    let mut best = (f64::INFINITY, state_a.clone(), state_b.clone());
    for s0 in &states_a {
        for s1 in &states_b {
            let d = (Vector3::from(s0.pos) - Vector3::from(s1.pos)).norm();
            if d < best.0 {
                best = (d, s0.clone(), s1.clone());
            }
        }
    }

    let cost = MoidCost {
        state_a: best.1,
        state_b: best.2,
    };

    let solver = NelderMead::new(vec![vec![-15.0, -15.0], vec![15.0, -15.0], vec![0.0, 15.0]]);

    let res = Executor::new(cost, solver)
        .configure(|state| state.max_iters(1000))
        .run()
        .unwrap();

    Ok(res.state().get_best_cost())
}

#[cfg(test)]
mod tests {
    use std::f64::consts::TAU;

    use super::*;
    use nalgebra::Vector3;

    use super::compute_eccentric_anomaly;

    #[test]
    fn test_kepler_circular() {
        for r in [0.2, 0.5, 1.0, 2.0] {
            let pos = Vector3::new(0.0, r, 0.0);
            let vel = Vector3::new(-GMS_SQRT / r.sqrt(), 0.0, 0.0);
            let year = TAU / GMS_SQRT * r.powf(3.0 / 2.0);
            let res = analytic_2_body(year, &pos, &vel, None).unwrap();
            assert!((res.0 - pos).norm() < 1e-8);
            assert!((res.1 - vel).norm() < 1e-8);

            // go backwards
            let res = analytic_2_body(-year, &pos, &vel, None).unwrap();
            assert!((res.0 - pos).norm() < 1e-8);
            assert!((res.1 - vel).norm() < 1e-8);
        }
    }

    #[test]
    fn test_compute_eccentric_anom_hyperbolic() {
        for mean_anom in -100..100 {
            let mean_anom = mean_anom as f64;
            assert!(
                compute_eccentric_anomaly(2.0, mean_anom, 0.1).is_ok(),
                "Mean Anom: {mean_anom}"
            );
        }
    }

    #[test]
    fn test_kepler_parabolic() {
        let pos = Vector3::new(0.0, 2.0, 0.0);
        let vel = Vector3::new(GMS_SQRT, 0.0, 0.0);
        let year = -TAU / GMS_SQRT;
        let res = analytic_2_body(year, &pos, &vel, None).unwrap();
        let pos_exp = Vector3::new(-4.448805955479905, -0.4739843046525608, 0.0);
        let vel_exp = -Vector3::new(-0.00768983428326951, -0.008552645144187791, 0.0);
        assert!((res.0 - pos_exp).norm() < 1e-8);
        assert!((res.1 - vel_exp).norm() < 1e-8);

        // go backwards
        let res = analytic_2_body(-year, &pos_exp, &vel_exp, None).unwrap();
        assert!((res.0 - pos).norm() < 1e-8);
        assert!((res.1 - vel).norm() < 1e-8);
    }

    #[test]
    fn test_kepler_hyperbolic() {
        let pos = Vector3::new(0.0, 3.0, 0.0);
        let vel = Vector3::new(-GMS_SQRT, 0.0, 0.0);
        let year = TAU / GMS_SQRT;
        let res = analytic_2_body(year, &pos, &vel, None).unwrap();
        let pos_exp = Vector3::new(-5.556785268950049, 1.6076633958058089, 0.0);
        let vel_exp = Vector3::new(-0.013061655543084886, -0.005508140023183166, 0.0);
        assert!((res.0 - pos_exp).norm() < 1e-8);
        assert!((res.1 - vel_exp).norm() < 1e-8);

        // go backwards
        let res = analytic_2_body(-year, &pos_exp, &vel_exp, None).unwrap();
        assert!((res.0 - pos).norm() < 1e-8);
        assert!((res.1 - vel).norm() < 1e-8);
    }

    #[test]
    fn test_true_anomaly() {
        let a = compute_true_anomaly(0.5, 3.211, 0.1).unwrap();
        let b = compute_true_anomaly(0.5, 3.211 + TAU, 0.1).unwrap();
        assert!((a - b).abs() < 1e-11);

        let ecc_anom = compute_eccentric_anomaly(0.5, 3.211, 0.1).unwrap();
        let c = eccentric_anomaly_from_true(0.5, a, 0.1).unwrap();
        assert!((c - ecc_anom).abs() < 1e-11);

        let a = compute_true_anomaly(0.0, 3.211, 0.1).unwrap();
        let b = compute_true_anomaly(0.0, 3.211 + TAU, 0.1).unwrap();
        assert!((a - b).abs() < 1e-11);

        let ecc_anom = compute_eccentric_anomaly(0.0, 3.211, 0.1).unwrap();
        let c = eccentric_anomaly_from_true(0.0, a, 0.1).unwrap();
        assert!((c - ecc_anom).abs() < 1e-11);

        let a = compute_true_anomaly(1.0, 3.211, 0.1).unwrap();
        let ecc_anom = compute_eccentric_anomaly(1.0, 3.211, 0.1).unwrap();
        let c = eccentric_anomaly_from_true(1.0, a, 0.1).unwrap();
        assert!((c - ecc_anom).abs() < 1e-11);

        let a = compute_true_anomaly(1.5, 3.211, 0.1).unwrap();
        let ecc_anom = compute_eccentric_anomaly(1.5, 3.211, 0.1).unwrap();
        let c = eccentric_anomaly_from_true(1.5, a, 0.1).unwrap();
        assert!((c - ecc_anom).abs() < 1e-11);
    }
}
