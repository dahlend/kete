/// # Picard-Chebyshev Numerical Integrator
///
/// First Order ODE Integrator.
///
/// This is based on a whole slew of papers, though this implementation leans heavily
/// on the algorithm described in:
///
/// "Surfing Chaotic Perturbations in Interplanetary Multi-Flyby Trajectories:
/// Augmented Picard-Chebyshev Integration for Parallel and GPU Computing
/// Architectures", 2022, `<https://doi.org/10.2514/6.2022-1275>`
///
/// Though that paper appears to have a typo in its `A` matrix definition, so some of
/// the thesis written by Darin Koblick "Parallel High-Precision Orbit Propagation
/// Using The Modified Picard-Chebyshev Method", 2012
/// was used to correct the matrix definition.
///
/// This is not a GPU implementation, though some small simplifications were applied to
/// the mathematics. This implementation is designed to reduce the total number of
/// SPICE kernel calls to a minimum, as it was found that the Radau integration time
/// was dominated by these queries.
///
/// Since this integrator fits Chebyshev polynomials at the same time that it performs
/// the integration, the integrator is designed to record the polynomial coefficients.
/// These are stored in the [`PicardStep`], which exposes a function allowing the
/// user to query the state of the system at any point between the start of the
/// integration and the end.
///
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
//
use std::f64::consts::PI;

use itertools::Itertools;
use nalgebra::{SMatrix, SVector};

use crate::errors::{Error, KeteResult};
use crate::propagation::util::FirstOrderODE;
use crate::time::{TDB, Time};

/// Initialization function which fills the integrators initial guess.
type PicardInitFunc<'a, const DIM: usize, const N: usize> =
    &'a dyn Fn(&[Time<TDB>; N], &SVector<f64, DIM>) -> SMatrix<f64, DIM, N>;

/// Second-order ODE function type for the Picard integrator.
/// y'' = F(time, y, y', metadata, `exact_eval`)
type PicardSecondOrderODE<'a, MType, const DIM: usize> =
    &'a dyn Fn(
        Time<TDB>,
        &SVector<f64, DIM>,
        &SVector<f64, DIM>,
        &mut MType,
        bool,
    ) -> KeteResult<SVector<f64, DIM>>;

/// Initialization function for second-order Picard integration.
/// Returns (position, velocity) guesses at all N Chebyshev nodes.
type PicardInitFunc2<'a, const DIM: usize, const N: usize> =
    &'a dyn Fn(
        &[Time<TDB>; N],
        &SVector<f64, DIM>,
        &SVector<f64, DIM>,
    ) -> (SMatrix<f64, DIM, N>, SMatrix<f64, DIM, N>);

/// Initialization function for the picard integrator where the first value is repeated
#[must_use]
pub fn dumb_picard_init<const DIM: usize, const N: usize>(
    _times: &[Time<TDB>; N],
    initial_pos: &SVector<f64, DIM>,
) -> SMatrix<f64, DIM, N> {
    let mut init: SMatrix<f64, DIM, N> = SMatrix::zeros();
    for idx in 0..N {
        init.set_column(idx, initial_pos);
    }
    init
}

/// Initialization function for second-order picard integrator where the initial
/// values are repeated for all nodes.
#[must_use]
pub fn dumb_picard_init_second_order<const DIM: usize, const N: usize>(
    _times: &[Time<TDB>; N],
    initial_pos: &SVector<f64, DIM>,
    initial_vel: &SVector<f64, DIM>,
) -> (SMatrix<f64, DIM, N>, SMatrix<f64, DIM, N>) {
    let mut pos_init: SMatrix<f64, DIM, N> = SMatrix::zeros();
    let mut vel_init: SMatrix<f64, DIM, N> = SMatrix::zeros();
    for idx in 0..N {
        pos_init.set_column(idx, initial_pos);
        vel_init.set_column(idx, initial_vel);
    }
    (pos_init, vel_init)
}

//                                                   ______
//                                      ___.--------'------`---------.____
//                                _.---'----------------------------------`---.__
//                              .'___=]===========================================
// ,-----------------------..__/.'         >--.______        _______.---'
// ]====================<==||(__)        .'          `------'
// `-----------------------`' ----.___--/
//      /       /---'                 `/
//     /_______(______________________/      Boldly going where many numerical
//     `-------------.--------------.'       integrators have gone before.
//                    \________|_.-'
//
/// Picard-Chebyshev integrator for solving first order ODEs.
///
/// This integrator fits Chebyshev polynomials to the provided first order ODEs.
/// These polynomials can be retrieved and queried at any point between the start and
/// end point of the integration. This is designed so that an object may have its orbit
/// integrated any length of time, and there is a very fast lookup for the object's
/// state at any point along the integration.
///
/// This integrator has a set of matrices which are pre-computed and stored.
/// It is recommended to use one of the pre-build integrators [`PC15`] or [`PC25`] to
/// avoid having to reconstruct these matrices every time the integrator is used.
///
/// The mathematics for this integrator can be found in a number of papers, but
/// `<https://doi.org/10.2514/6.2022-1275>` is a good place to start.
///
/// The primary entry-point for the integrator is the [`PicardIntegrator::integrate`]
/// function. See that function for more details on its use.
///
#[derive(Debug, Clone)]
pub struct PicardIntegrator<const N: usize, const NM1: usize> {
    // These parameters names are taken directly from the paper cited above.
    // SA represents the matrix product of S and A.
    c: SMatrix<f64, N, N>,

    a: SMatrix<f64, N, NM1>,

    sa: SVector<f64, N>,

    tau: [f64; N],
}

/// Preconstructed [`PicardIntegrator`] for a 14th order polynomial
pub static PC15: std::sync::LazyLock<PicardIntegrator<15, 14>> =
    std::sync::LazyLock::new(PicardIntegrator::default);

/// Preconstructed [`PicardIntegrator`] for a 24th order polynomial
pub static PC25: std::sync::LazyLock<PicardIntegrator<25, 24>> =
    std::sync::LazyLock::new(PicardIntegrator::default);

/// A single stop of the Picard Integrator
#[derive(Debug, Clone)]
pub struct PicardStep<const N: usize, const DIM: usize> {
    /// Chebyshev polynomial coefficients of the first type
    b: SMatrix<f64, DIM, N>,

    /// Final result
    pub y: SMatrix<f64, DIM, N>,

    /// Start time
    pub t0: Time<TDB>,

    /// End time
    pub t1: Time<TDB>,
}

/// A single step of the second-order Picard Integrator
#[derive(Debug, Clone)]
pub struct PicardStepSecondOrder<const N: usize, const DIM: usize> {
    /// Chebyshev polynomial coefficients for position
    b_pos: SMatrix<f64, DIM, N>,

    /// Chebyshev polynomial coefficients for velocity
    b_vel: SMatrix<f64, DIM, N>,

    /// Position values at each Chebyshev node
    pub pos: SMatrix<f64, DIM, N>,

    /// Velocity values at each Chebyshev node
    pub vel: SMatrix<f64, DIM, N>,

    /// Start time
    pub t0: Time<TDB>,

    /// End time
    pub t1: Time<TDB>,
}

impl<'a, const N: usize, const NM1: usize> PicardIntegrator<N, NM1> {
    /// Integrate from the start point to the end.
    ///
    /// # Arguments
    ///
    /// * `func` - First order ODE function which accepts time, state of the system, a
    ///   metadata class, and a bool value to indicate if the provided state is
    ///   believed to be an exact evaluated position.
    /// * `init_func` - A function which can initialize the starting guess for the
    ///   integrator. If this is unavailable, then [`dumb_picard_init`] can be used as
    ///   a basic initializer.
    /// * `t0` - Start time of the integration.
    /// * `t1` - End time of the integration.
    /// * `step_size` - Initial step size of the integration, a rough guess is all that
    ///   is required, as the integrator will find its own valid step size. This speeds
    ///   up the initial search for the valid step size.
    /// * `metadata` - An initial packet of metadata which will be passed through to
    ///   the `func` during evaluation, this may include information such as masses,
    ///   non-gravitational forces, record keeping of close encounters, etc. This
    ///   metadata is mutated in place during integration.
    ///
    /// # Returns
    ///  
    ///   Returns the expected state of the system at the final time step `t1`.
    ///    
    /// # Errors
    /// Integration can fail for a number of reasons, including convergence failing or
    /// function evals failing.
    pub fn integrate<const DIM: usize, MType>(
        &self,
        func: FirstOrderODE<'a, MType, DIM>,
        init_func: PicardInitFunc<'a, DIM, N>,
        initial_pos: SVector<f64, DIM>,
        t0: Time<TDB>,
        t1: Time<TDB>,
        step_size: f64,
        metadata: &mut MType,
    ) -> KeteResult<SVector<f64, DIM>> {
        let t0 = t0.jd;
        let t1 = t1.jd;
        let mut cur_t0 = t0;
        if (t0 - t1).abs() < 10.0 * f64::EPSILON {
            return Ok(initial_pos);
        }

        let mut cur_stepsize = step_size;
        let mut cur_t1 = t0 + cur_stepsize;
        if (t0 - t1).abs() < step_size.abs() {
            cur_stepsize = t1 - t0;
            cur_t1 = t1;
        }

        // Track the state at the start of each segment
        let mut seg_state = initial_pos;

        loop {
            // Compute init guess with correct times for the current segment
            let times = self.evaluation_times(cur_t0.into(), cur_t1.into());
            let cur_init = init_func(&times, &seg_state);

            let step = self.step(func, cur_t0.into(), cur_t1.into(), cur_init, metadata);
            match step {
                Ok(res) => {
                    // Estimate truncation error from tail Chebyshev coefficients
                    let tail_err = {
                        let mut t = 0.0_f64;
                        for d in 0..DIM {
                            t = t.max(res.b[(d, N - 1)].abs());
                            t = t.max(res.b[(d, N - 2)].abs());
                        }
                        t
                    };
                    let scale = res.y.abs().max().max(1.0);
                    let rel_trunc = tail_err / scale;

                    if rel_trunc > 1e-14 && cur_t1 != t1 {
                        let factor = (1e-14 / rel_trunc)
                            .powf(1.0 / (N as f64 - 1.0))
                            .clamp(0.25, 0.9);
                        cur_stepsize *= factor;
                        cur_t1 = cur_t0 + cur_stepsize;
                        if cur_stepsize.abs() < 1e-10 {
                            return Err(Error::Convergence(format!(
                                "Failed to converge, step sizes became too small. {}",
                                cur_stepsize.abs()
                            )));
                        }
                        continue;
                    }

                    if cur_t1 == t1 {
                        return Ok(res.y.column(N - 1).into());
                    }

                    seg_state = res.y.column(N - 1).into();
                    cur_t0 = cur_t1;

                    // Adaptive step growth based on truncation quality
                    let factor = if rel_trunc > 0.0 {
                        (1e-14 / rel_trunc)
                            .powf(1.0 / (N as f64 - 1.0))
                            .clamp(1.0, 2.0)
                    } else {
                        2.0
                    };
                    cur_stepsize *= factor;

                    cur_t1 = cur_t0 + cur_stepsize;
                    if (cur_stepsize.is_sign_positive() && cur_t1 > t1)
                        || (cur_stepsize.is_sign_negative() && cur_t1 < t1)
                    {
                        cur_t1 = t1;
                    }
                }
                Err(Error::Impact(idx, jd)) => {
                    return Err(Error::Impact(idx, jd));
                }
                Err(_) => {
                    cur_stepsize *= 0.7;
                    cur_t1 = cur_t0 + cur_stepsize;
                }
            }
            if cur_stepsize.abs() < 1e-10 {
                return Err(Error::Convergence(format!(
                    "Failed to converge, step sizes became too small. {}",
                    cur_stepsize.abs()
                )));
            }
        }
    }

    /// Single fallible step of the integrator
    ///
    /// This returns a single [`PicardStep`] which may be used to evaluate the state
    /// of the system at any point between the start time `t0` and end `t1`.
    ///
    fn step<const DIM: usize, MType>(
        &self,
        func: FirstOrderODE<'a, MType, DIM>,
        t0: Time<TDB>,
        t1: Time<TDB>,
        mut cur_pos: SMatrix<f64, DIM, N>,
        metadata: &mut MType,
    ) -> KeteResult<PicardStep<N, DIM>> {
        let mut b: SMatrix<f64, DIM, N> = SMatrix::zeros();
        let mut f: SMatrix<f64, DIM, N> = SMatrix::zeros();
        let times = self.evaluation_times(t0, t1);
        let w2 = (t1 - t0).elapsed / 2.0;
        let mut error: f64 = f64::MAX;
        let mut last_error = f64::MAX;

        // Pin the boundary condition at t0 so it cannot drift during iteration
        let bc_pos: SVector<f64, DIM> = cur_pos.column(0).into();

        // when the answer converges to within tolerance, do one last evaluation and
        // trigger the metadata exact evaluation, allowing the `func` to update the
        // metadata with exact values.
        let mut final_iteration = false;

        for _ in 0..150 {
            for ((&time, y), mut f_row) in times
                .iter()
                .zip(cur_pos.column_iter())
                .zip(f.column_iter_mut())
            {
                // during the final iteration, let `func` know that we should update
                // the metadata
                f_row.set_column(0, &(w2 * func(time, &y.into(), metadata, final_iteration)?));
            }

            b.set_column(0, &(f * self.sa + 2.0 * bc_pos));

            b.fixed_view_mut::<DIM, NM1>(0, 1)
                .iter_mut()
                .zip((f * self.a).into_iter())
                .for_each(|(x, &n)| *x = n);

            let last_pos = cur_pos;
            cur_pos = b * self.c;

            // diverging, quit quickly
            if error > last_error && error > 1.0 {
                return Err(Error::Convergence(
                    "Integrator solution is diverging.".into(),
                ));
            }

            // if we are on the final evaluation, return the answer.
            if final_iteration {
                return Ok(PicardStep {
                    b,
                    y: cur_pos,
                    t0,
                    t1,
                });
            }

            // if we have converged, then trigger the final iteration
            // Scale the threshold by the solution magnitude so that convergence
            // remains achievable for large position values (e.g. 30 AU).
            let tol = 2.0 * f64::EPSILON * cur_pos.abs().max().max(1.0);
            if error < tol {
                final_iteration = true;
            }

            last_error = error;
            error = (last_pos - cur_pos).abs().max();
        }

        Err(Error::Convergence("Integrator failed to converge.".into()))
    }

    /// Given the start and stop times of a desired integration, return the actual
    /// times where the function will be evaluated starting at t0 and ending at t1.
    fn evaluation_times(&self, t0: Time<TDB>, t1: Time<TDB>) -> [Time<TDB>; N] {
        let w2 = (t1 - t0).elapsed / 2.0;
        let w1 = f64::midpoint(t1.jd, t0.jd);
        let mut times: [Time<TDB>; N] = [0.0.into(); N];
        times
            .iter_mut()
            .zip(self.tau)
            .for_each(|(x, tau_v)| x.jd = w2 * tau_v + w1);
        times
    }

    /// Integrate a second-order ODE from the start point to the end.
    ///
    /// # Arguments
    ///
    /// * `func` - Second order ODE function y'' = F(time, y, y', metadata, `exact_eval`).
    /// * `init_func` - A function which can initialize the starting guess for the
    ///   integrator, returning (position, velocity) guesses at all Chebyshev nodes.
    ///   If unavailable, [`dumb_picard_init_second_order`] can be used.
    /// * `initial_pos` - Initial position vector.
    /// * `initial_vel` - Initial velocity vector.
    /// * `t0` - Start time of the integration.
    /// * `t1` - End time of the integration.
    /// * `step_size` - Initial step size guess.
    /// * `metadata` - Metadata passed through to `func` during evaluation.
    ///
    /// # Returns
    ///
    ///   Returns (position, velocity) at the final time step `t1`.
    ///
    /// # Errors
    /// Integration can fail due to convergence failure or function evaluation errors.
    pub fn integrate_second_order<const DIM: usize, MType>(
        &self,
        func: PicardSecondOrderODE<'a, MType, DIM>,
        init_func: PicardInitFunc2<'a, DIM, N>,
        initial_pos: SVector<f64, DIM>,
        initial_vel: SVector<f64, DIM>,
        t0: Time<TDB>,
        t1: Time<TDB>,
        step_size: f64,
        metadata: &mut MType,
    ) -> KeteResult<(SVector<f64, DIM>, SVector<f64, DIM>)> {
        let t0 = t0.jd;
        let t1 = t1.jd;
        let mut cur_t0 = t0;
        if (t0 - t1).abs() < 10.0 * f64::EPSILON {
            return Ok((initial_pos, initial_vel));
        }

        let mut cur_stepsize = step_size;
        let mut cur_t1 = t0 + cur_stepsize;
        if (t0 - t1).abs() < step_size.abs() {
            cur_stepsize = t1 - t0;
            cur_t1 = t1;
        }

        // Track the state at the start of each segment
        let mut seg_pos = initial_pos;
        let mut seg_vel = initial_vel;

        loop {
            // Compute init guess with correct times for the current segment
            let times = self.evaluation_times(cur_t0.into(), cur_t1.into());
            let (cur_pos_init, cur_vel_init) = init_func(&times, &seg_pos, &seg_vel);

            let step = self.step_second_order(
                func,
                cur_t0.into(),
                cur_t1.into(),
                cur_pos_init,
                cur_vel_init,
                metadata,
            );
            match step {
                Ok(res) => {
                    // Estimate truncation error from tail Chebyshev coefficients.
                    // Check both position and velocity polynomials -- near
                    // perihelion the velocity polynomial is often under-resolved
                    // even when position looks fine.
                    let rel_trunc = {
                        let mut pos_tail = 0.0_f64;
                        let mut vel_tail = 0.0_f64;
                        for d in 0..DIM {
                            pos_tail = pos_tail.max(res.b_pos[(d, N - 1)].abs());
                            pos_tail = pos_tail.max(res.b_pos[(d, N - 2)].abs());
                            vel_tail = vel_tail.max(res.b_vel[(d, N - 1)].abs());
                            vel_tail = vel_tail.max(res.b_vel[(d, N - 2)].abs());
                        }
                        let pos_scale = res.pos.abs().max().max(1.0);
                        let vel_scale = res.vel.abs().max().max(1.0);
                        (pos_tail / pos_scale).max(vel_tail / vel_scale)
                    };

                    // If truncation error is too large, reject and shrink
                    // (unless this is the final step, which we must accept)
                    if rel_trunc > 1e-14 && cur_t1 != t1 {
                        let factor = (1e-14 / rel_trunc)
                            .powf(1.0 / (N as f64 - 1.0))
                            .clamp(0.25, 0.9);
                        cur_stepsize *= factor;
                        cur_t1 = cur_t0 + cur_stepsize;
                        if cur_stepsize.abs() < 1e-10 {
                            return Err(Error::Convergence(format!(
                                "Failed to converge, step sizes became too small. {}",
                                cur_stepsize.abs()
                            )));
                        }
                        continue;
                    }

                    // Accept step
                    if cur_t1 == t1 {
                        return Ok((res.pos.column(N - 1).into(), res.vel.column(N - 1).into()));
                    }

                    seg_pos = res.pos.column(N - 1).into();
                    seg_vel = res.vel.column(N - 1).into();
                    cur_t0 = cur_t1;

                    // Adaptive step growth based on truncation quality
                    let factor = if rel_trunc > 0.0 {
                        (1e-14 / rel_trunc)
                            .powf(1.0 / (N as f64 - 1.0))
                            .clamp(1.0, 2.0)
                    } else {
                        2.0
                    };
                    cur_stepsize *= factor;

                    cur_t1 = cur_t0 + cur_stepsize;
                    if (cur_stepsize.is_sign_positive() && cur_t1 > t1)
                        || (cur_stepsize.is_sign_negative() && cur_t1 < t1)
                    {
                        cur_t1 = t1;
                    }
                }
                Err(Error::Impact(idx, jd)) => {
                    return Err(Error::Impact(idx, jd));
                }
                Err(_) => {
                    cur_stepsize *= 0.7;
                    cur_t1 = cur_t0 + cur_stepsize;
                }
            }
            if cur_stepsize.abs() < 1e-10 {
                return Err(Error::Convergence(format!(
                    "Failed to converge, step sizes became too small. {}",
                    cur_stepsize.abs()
                )));
            }
        }
    }

    /// Single fallible step of the second-order integrator.
    ///
    /// Returns a [`PicardStepSecondOrder`] which may be used to evaluate position
    /// and velocity at any point between `t0` and `t1`.
    fn step_second_order<const DIM: usize, MType>(
        &self,
        func: PicardSecondOrderODE<'a, MType, DIM>,
        t0: Time<TDB>,
        t1: Time<TDB>,
        mut cur_pos: SMatrix<f64, DIM, N>,
        mut cur_vel: SMatrix<f64, DIM, N>,
        metadata: &mut MType,
    ) -> KeteResult<PicardStepSecondOrder<N, DIM>> {
        let mut b_pos: SMatrix<f64, DIM, N> = SMatrix::zeros();
        let mut b_vel: SMatrix<f64, DIM, N> = SMatrix::zeros();
        let mut accel: SMatrix<f64, DIM, N> = SMatrix::zeros();
        let times = self.evaluation_times(t0, t1);
        let w2 = (t1 - t0).elapsed / 2.0;
        let mut error: f64 = f64::MAX;
        let mut last_error = f64::MAX;
        let mut final_iteration = false;

        // Pin boundary conditions at t0
        let bc_pos: SVector<f64, DIM> = cur_pos.column(0).into();
        let bc_vel: SVector<f64, DIM> = cur_vel.column(0).into();

        for _ in 0..150 {
            // Evaluate accelerations at all nodes
            for ((&time, (pos_col, vel_col)), mut accel_col) in times
                .iter()
                .zip(cur_pos.column_iter().zip(cur_vel.column_iter()))
                .zip(accel.column_iter_mut())
            {
                accel_col.set_column(
                    0,
                    &(w2 * func(
                        time,
                        &pos_col.into(),
                        &vel_col.into(),
                        metadata,
                        final_iteration,
                    )?),
                );
            }

            // First antidifferentiation: accel -> vel
            b_vel.set_column(0, &(accel * self.sa + 2.0 * bc_vel));
            b_vel
                .fixed_view_mut::<DIM, NM1>(0, 1)
                .iter_mut()
                .zip((accel * self.a).into_iter())
                .for_each(|(x, &n)| *x = n);
            let last_vel = cur_vel;
            cur_vel = b_vel * self.c;

            // Second antidifferentiation: vel -> pos
            let vel_scaled = w2 * cur_vel;
            b_pos.set_column(0, &(vel_scaled * self.sa + 2.0 * bc_pos));
            b_pos
                .fixed_view_mut::<DIM, NM1>(0, 1)
                .iter_mut()
                .zip((vel_scaled * self.a).into_iter())
                .for_each(|(x, &n)| *x = n);

            let last_pos = cur_pos;
            cur_pos = b_pos * self.c;

            // diverging, quit quickly
            if error > last_error && error > 1.0 {
                return Err(Error::Convergence(
                    "Integrator solution is diverging.".into(),
                ));
            }

            if final_iteration {
                return Ok(PicardStepSecondOrder {
                    b_pos,
                    b_vel,
                    pos: cur_pos,
                    vel: cur_vel,
                    t0,
                    t1,
                });
            }

            let pos_tol = 2.0 * f64::EPSILON * cur_pos.abs().max().max(1.0);
            let vel_tol = 2.0 * f64::EPSILON * cur_vel.abs().max().max(1.0);
            let vel_error = (last_vel - cur_vel).abs().max();
            if error < pos_tol && vel_error < vel_tol {
                final_iteration = true;
            }

            last_error = error;
            error = (last_pos - cur_pos).abs().max().max(vel_error);
        }

        Err(Error::Convergence("Integrator failed to converge.".into()))
    }
}

impl<const N: usize, const DIM: usize> PicardStep<N, DIM> {
    /// Evaluate the fitted integration solution at the specified time.
    /// This will fail if the requested time is outside of the integration bounds.
    ///
    /// # Errors
    /// Evaluation may fail if ``t`` is outside of bounds.
    pub fn evaluate(&self, t: f64) -> KeteResult<[f64; DIM]> {
        let w1 = (self.t0.jd + self.t1.jd) * 0.5;
        let w2 = (self.t1 - self.t0).elapsed * 0.5;
        let tau_time = ((t - w1) * w2).acos();
        if tau_time.is_nan() {
            return Err(Error::Bounds(
                "Queried time it outside of the fitted time span".into(),
            ));
        }
        Ok(chebyshev_eval(tau_time, &self.b.transpose().into()))
    }
}

impl<const N: usize, const DIM: usize> PicardStepSecondOrder<N, DIM> {
    /// Evaluate the fitted integration solution at the specified time.
    /// Returns (position, velocity).
    ///
    /// # Errors
    /// Evaluation may fail if ``t`` is outside of bounds.
    pub fn evaluate(&self, t: f64) -> KeteResult<([f64; DIM], [f64; DIM])> {
        let w1 = (self.t0.jd + self.t1.jd) * 0.5;
        let w2 = (self.t1 - self.t0).elapsed * 0.5;
        let tau_time = ((t - w1) * w2).acos();
        if tau_time.is_nan() {
            return Err(Error::Bounds(
                "Queried time is outside of the fitted time span".into(),
            ));
        }
        let pos = chebyshev_eval(tau_time, &self.b_pos.transpose().into());
        let vel = chebyshev_eval(tau_time, &self.b_vel.transpose().into());
        Ok((pos, vel))
    }
}

impl<const N: usize, const NM1: usize> Default for PicardIntegrator<N, NM1> {
    fn default() -> Self {
        // Combining the cos(k*arccos(-cos(...))) simplifies to this equation
        // This is used throughout the integrator's construction
        fn cheb_t(k: f64, j: f64, n: f64) -> f64 {
            (k * (j + n - 1.0) * PI / (n - 1.0)).cos()
        }

        // NM1 must be equal to N-1, I wish I could do this with const generics
        // but that would require switching to nightly rust and I wont do that.
        // So this compile time check is done instead.
        const { assert!(N - 1 == NM1, "NM1 must be 1 less than N.") };

        let n = N as f64;

        // Construct the C matrix
        let mut c = SMatrix::<f64, N, N>::zeros();
        // note that calling cheb_t with k=0 evaluates to 1, and in the paper this
        // gets divided by 2, resulting in 0.5.
        c.fill_column(0, 0.5);

        for j in 0..N {
            for k in 1..N {
                c[(j, k)] = cheb_t(k as f64, j as f64, n);
            }
        }

        // Construct the A matrix from `<https://doi.org/10.2514/6.2022-1275>`
        // Note that the paper appears to have typos in the indexing, I suspect
        // there should be a k in several places where there is currently a j.
        // This implementation is a blend of that paper, and the original math
        // that the paper cites.
        let mut a = SMatrix::<f64, NM1, N>::zeros();
        for k in 0..N - 2 {
            for j in 0..N {
                a[(k, j)] = (cheb_t(k as f64, j as f64, n) - cheb_t((k + 2) as f64, j as f64, n))
                    / (((N - 1) * (k + 1)) as f64);
            }
        }

        for j in 0..N {
            a[(N - 2, j)] = cheb_t(n - 2.0, j as f64, n) / ((N - 1).pow(2) as f64);
        }

        a[(N - 2, 0)] /= 2.0;
        a[(N - 2, N - 1)] /= 2.0;

        // Now we construct the SA Vector, in the original paper they keep S as a
        // separate vector and multiply it against A over and over, whereas we are just
        // doing it once and saving it.
        #[allow(clippy::cast_possible_wrap, reason = "cast cant fail.")]
        let s: SVector<f64, NM1> =
            SVector::from_iterator((0..N).map(|idx| 2.0 * (-1_f64).powi(idx as i32)));

        let sa: SVector<f64, N> = a.transpose() * s;

        let tau: [f64; N] = (0..N)
            .map(|j| -(j as f64 * PI / (n - 1.0)).cos())
            .collect_vec()
            .try_into()
            .unwrap();

        Self {
            a: a.transpose(),
            sa,
            c: c.transpose(),
            tau,
        }
    }
}

/// Evaluate chebyshev polynomials of the first type.
#[inline(always)]
fn chebyshev_eval<const R: usize, const C: usize>(x: f64, coef: &[[f64; R]; C]) -> [f64; C] {
    const { assert!(R > 2, "Dimension R must be greater than 2.") };
    const { assert!(C > 0, "Dimension C must be greater than 0.") };

    let mut val = [0.0; C];

    for idx in 0..C {
        val[idx] = coef[idx][0] + coef[idx][1] * x;
    }

    let mut second_t = 1.0;
    let mut last_t = x;
    let mut next_t;

    let x2: f64 = 2.0 * x;
    #[allow(
        clippy::needless_range_loop,
        reason = "idy indexes into the inner dimension of coef[idx][idy]"
    )]
    for idy in 2..R {
        next_t = x2 * last_t - second_t;
        for idx in 0..C {
            val[idx] += coef[idx][idy] * next_t;
        }
        second_t = last_t;
        last_t = next_t;
    }

    val
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exponential_decay_test() {
        // integrate a first order ode of the form f' = -c  f'(0) = k
        // where c = [1, 0.5, 0.1] and k = [1, 2, 5]
        // This has the analytic solution of f=k * exp(-c*t)

        // define the f' function
        #[allow(clippy::unnecessary_wraps, reason = "Required for integrator")]
        fn func(
            _: Time<TDB>,
            vals: &SVector<f64, 3>,
            _: &mut (),
            _: bool,
        ) -> KeteResult<SVector<f64, 3>> {
            let v: SVector<f64, 3> = [1.0, 0.5, 0.1].into();
            Ok(-vals.component_mul(&v))
        }

        // integrate with the initial conditions
        let p = &PC15;
        let t1 = 5.0.into();
        let res = p
            .integrate(
                &func,
                &dumb_picard_init,
                [1.0, 2.0, 5.0].into(),
                0.0.into(),
                t1,
                0.1,
                &mut (),
            )
            .unwrap();

        // test against the analytic solution
        assert!((res[0] - (1.0 * (-t1.jd).exp())).abs() < 1e-15);
        assert!((res[1] - (2.0 * (-0.5 * t1.jd).exp())).abs() < 1e-15);
        assert!(
            (res[2] - (5.0 * (-0.1 * t1.jd).exp())).abs() < 1e-14,
            "{}",
            (res[2] - (5.0 * (-0.1 * t1.jd).exp())).abs()
        );
    }

    #[test]
    fn harmonic_oscillator_test() {
        // Integrate a second order ODE: y'' = -y, y(0) = [1, 0, 0], y'(0) = [0, 1, 0]
        // Analytic: y = [cos(t), sin(t), 0], y' = [-sin(t), cos(t), 0]
        #[allow(clippy::unnecessary_wraps, reason = "Required for integrator")]
        fn accel(
            _: Time<TDB>,
            pos: &SVector<f64, 3>,
            _vel: &SVector<f64, 3>,
            _: &mut (),
            _: bool,
        ) -> KeteResult<SVector<f64, 3>> {
            Ok(-pos)
        }

        let p = &PC15;
        let t_final = 3.0;
        let (pos, vel) = p
            .integrate_second_order(
                &accel,
                &dumb_picard_init_second_order,
                [1.0, 0.0, 0.0].into(),
                [0.0, 1.0, 0.0].into(),
                0.0.into(),
                t_final.into(),
                0.1,
                &mut (),
            )
            .unwrap();

        assert!(
            (pos[0] - t_final.cos()).abs() < 1e-14,
            "pos[0] error: {}",
            (pos[0] - t_final.cos()).abs()
        );
        assert!(
            (pos[1] - t_final.sin()).abs() < 1e-14,
            "pos[1] error: {}",
            (pos[1] - t_final.sin()).abs()
        );
        assert!(pos[2].abs() < 1e-15, "pos[2] error: {}", pos[2].abs());
        assert!(
            (vel[0] - (-t_final.sin())).abs() < 1e-14,
            "vel[0] error: {}",
            (vel[0] - (-t_final.sin())).abs()
        );
        assert!(
            (vel[1] - t_final.cos()).abs() < 1e-14,
            "vel[1] error: {}",
            (vel[1] - t_final.cos()).abs()
        );
        assert!(vel[2].abs() < 1e-15, "vel[2] error: {}", vel[2].abs());
    }
}
