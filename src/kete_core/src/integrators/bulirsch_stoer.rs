//! Gragg-Bulirsch-Stoer Extrapolation Integrator.
//! This solves a second-order initial value problem using modified midpoint
//! (Stoermer form) with Richardson extrapolation.
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
use crate::errors::Error;
use crate::integrators::util::SecondOrderODE;
use crate::prelude::KeteResult;
use crate::time::{TDB, Time};
use nalgebra::allocator::Allocator;
use nalgebra::{DefaultAllocator, Dim, Matrix, OVector, U1};

/// Result type returned by [`BulirschStoerIntegrator::integrate`].
type BSResult<MType, D> = KeteResult<(OVector<f64, D>, OVector<f64, D>, MType)>;

// Relative tolerance target for the local extrapolation error estimate.
// The increment-based modified midpoint lowers the roundoff floor, allowing
// a tighter tolerance.
const RTOL: f64 = 1e-14;

// Floor value added to scaling denominator to prevent division by zero when
// state components are near zero.
const EPS_FLOOR: f64 = f64::EPSILON * f64::EPSILON;

// Maximum number of extrapolation columns (effective order = 2 * K_MAX).
const K_MAX: usize = 8;

// Minimum extrapolation column index allowed for step acceptance. With j-indexing
// from 0, effective order at column j is 2*(j+1), so MIN_J=3 forces order >= 8
// and MIN_J=4 forces order >= 10. The default acceptance criterion (column-to-
// column change below tolerance) tends to accept too eagerly at low columns on
// smooth orbits, leaving a global error floor set by truncation at order 4-8.
// Forcing a higher floor pushes BS toward Radau-like accuracy on smooth problems
// at the cost of a few extra force evaluations per step.
const MIN_J: usize = 4;

// Safety factor applied when estimating the next step size.
const SAFETY: f64 = 0.9;

// Minimum allowed step size in days.
const MIN_STEP: f64 = 0.005;

// Initial step size in days.
const INIT_STEP: f64 = 0.1;

// Minimum ratio for step size change per step. The reciprocal (10.0)
// is the maximum growth factor, allowing rapid ramp-up to cruising speed.
const MIN_RATIO: f64 = 0.1;

// Maximum number of consecutive step rejections before raising an error.
const MAX_STEP_REJECT: usize = 10;

// Bulirsch sub-step sequence. Each entry is the number of sub-steps used for
// the modified midpoint at that extrapolation level.
const N_SEQ: [usize; K_MAX] = [2, 4, 6, 8, 12, 16, 24, 32];

/// Gragg-Bulirsch-Stoer extrapolation integrator for second-order ODEs.
///
/// This method uses the modified midpoint rule (Stoermer form) at progressively
/// finer sub-step counts, then extrapolates toward zero step size via the
/// Aitken-Neville polynomial algorithm. Step size adapts automatically.
///
/// Compensated (Kahan) summation is used for the state update to reduce
/// roundoff accumulation from O(N) to approximately O(sqrt(N)).
///
/// References:
/// - Hairer, Noersett, Wanner: "Solving Ordinary Differential Equations I",
///   Chapter II.9 (Extrapolation Methods).
/// - Stoer, Bulirsch: "Introduction to Numerical Analysis", Chapter 7.
/// - Gragg (1965): "On Extrapolation Algorithms for Ordinary Initial Value
///   Problems", SIAM J. Numer. Anal.
#[allow(missing_debug_implementations, reason = "No debug impl needed")]
pub struct BulirschStoerIntegrator<'a, MType, D: Dim>
where
    DefaultAllocator: Allocator<D, U1>,
{
    func: SecondOrderODE<'a, MType, D>,
    metadata: MType,

    final_time: Time<TDB>,

    cur_time: Time<TDB>,
    cur_state: OVector<f64, D>,
    cur_state_der: OVector<f64, D>,
    cur_state_der_der: OVector<f64, D>,

    /// Number of leading dimensions used for error estimation and step-size
    /// control. Defaults to the full state dimension. For variational / STM
    /// propagation, set this to 3 so that large STM elements do not
    /// artificially shrink the step size.
    control_dim: usize,

    // Kahan compensated summation error accumulators.
    comp_state: OVector<f64, D>,
    comp_state_der: OVector<f64, D>,
    comp_time: f64,
}

impl<'a, MType, D: Dim> BulirschStoerIntegrator<'a, MType, D>
where
    DefaultAllocator: Allocator<D, U1>,
{
    fn new(
        func: SecondOrderODE<'a, MType, D>,
        state_init: OVector<f64, D>,
        state_der_init: OVector<f64, D>,
        time_init: Time<TDB>,
        final_time: Time<TDB>,
        metadata: MType,
    ) -> KeteResult<Self> {
        let (dim, _) = state_init.shape_generic();
        if state_init.len() != state_der_init.len() {
            return Err(Error::ValueError(
                "Input vectors must be the same length".into(),
            ));
        }
        let full_dim = state_init.len();
        let mut res = Self {
            func,
            metadata,
            final_time,
            cur_time: time_init,
            cur_state: state_init,
            cur_state_der: state_der_init,
            cur_state_der_der: Matrix::zeros_generic(dim, U1),
            control_dim: full_dim,
            comp_state: Matrix::zeros_generic(dim, U1),
            comp_state_der: Matrix::zeros_generic(dim, U1),
            comp_time: 0.0,
        };
        res.cur_state_der_der = (res.func)(
            time_init,
            &res.cur_state,
            &res.cur_state_der,
            &mut res.metadata,
            true,
        )?;
        Ok(res)
    }

    /// Integrate from the initial time to the final time.
    ///
    /// # Errors
    /// Returns an error if the integrator fails to converge, encounters an
    /// impact, or the force function itself returns an error.
    pub fn integrate(
        func: SecondOrderODE<'a, MType, D>,
        state_init: OVector<f64, D>,
        state_der_init: OVector<f64, D>,
        time_init: Time<TDB>,
        final_time: Time<TDB>,
        metadata: MType,
        control_dim: Option<usize>,
    ) -> BSResult<MType, D> {
        let mut integrator = Self::new(
            func,
            state_init,
            state_der_init,
            time_init,
            final_time,
            metadata,
        )?;

        if (final_time - time_init).elapsed.abs() < 1e-10 {
            return Ok((
                integrator.cur_state,
                integrator.cur_state_der,
                integrator.metadata,
            ));
        }

        let mut next_step_size: f64 =
            INIT_STEP.copysign((integrator.final_time - integrator.cur_time).elapsed);

        integrator.control_dim = control_dim.unwrap_or(integrator.control_dim);
        if integrator.control_dim > integrator.cur_state.len() {
            return Err(Error::ValueError(format!(
                "control_dim ({}) exceeds state dimension ({})",
                integrator.control_dim,
                integrator.cur_state.len(),
            )));
        }

        let mut step_failures: usize = 0;
        loop {
            if (integrator.cur_time - integrator.final_time).elapsed.abs() <= next_step_size.abs() {
                next_step_size = (integrator.final_time - integrator.cur_time).elapsed;
            }
            match integrator.step(next_step_size) {
                Ok(s) => {
                    next_step_size = s;
                    if (integrator.cur_time - integrator.final_time).elapsed.abs() < 1e-12 {
                        return Ok((
                            integrator.cur_state,
                            integrator.cur_state_der,
                            integrator.metadata,
                        ));
                    }
                    step_failures = 0;
                }
                Err(error) => match error {
                    Error::Bounds(_) | Error::Impact(_, _) | Error::OutOfMemory => {
                        return Err(error);
                    }
                    Error::Convergence(_)
                    | Error::ValueError(_)
                    | Error::UnknownFrame(_)
                    | Error::IOError(_)
                    | Error::LockFailed => {
                        step_failures += 1;
                        next_step_size *= 0.7;
                        if step_failures > MAX_STEP_REJECT {
                            return Err(Error::Convergence(
                                "Bulirsch-Stoer failed to converge.".into(),
                            ));
                        }
                    }
                },
            }
            if next_step_size.abs() < MIN_STEP {
                next_step_size = MIN_STEP.copysign(next_step_size);
            }
        }
    }

    /// Perform one modified midpoint integration with `n_sub` sub-steps
    /// (Stoermer form for 2nd-order ODEs).
    ///
    /// Returns the macro-step increments `(D_pos, D_vel)` rather than absolute
    /// states, where `D_pos = y_smooth - cur_state` and
    /// `D_vel = ydot_smooth - cur_state_der`. Working in increment form
    /// throughout (both the per-substep `delta = y_k - y_{k-1}` and the
    /// running macro-step quantities) keeps every accumulator on numbers of
    /// order `h * v_0` or smaller, which is the key to making the downstream
    /// extrapolation tableau roundoff-clean.
    ///
    /// The velocity increment is built from a cancellation-free recurrence
    /// `dvel_{k+1} = dvel_k + h * f_k` starting from `dvel_1 = h/2 * a_0`,
    /// which is algebraically equivalent to the standard reconstruction
    /// `(y_n - y_{n-1})/h - v_0` but avoids subtracting two numbers of
    /// magnitude `~v_0`.
    fn modified_midpoint(
        &mut self,
        h_total: f64,
        n_sub: usize,
    ) -> KeteResult<(OVector<f64, D>, OVector<f64, D>)> {
        let h = h_total / n_sub as f64;
        let h2 = h * h;
        let (dim, _) = self.cur_state.shape_generic();

        // Per-substep position increment delta = y_k - y_{k-1}.
        //   delta_1 = h * v_0 + h^2/2 * a_0
        let mut delta = &self.cur_state_der * h + &self.cur_state_der_der * (h2 * 0.5);

        // Running macro-step increments tracked relative to (cur_state, cur_state_der):
        //   d_cur  = y_cur  - cur_state                 (starts as delta_1)
        //   d_prev = y_prev - cur_state                 (starts at 0)
        //   dvel   = (y_cur - y_prev)/h - cur_state_der (starts as h/2 * a_0)
        let mut d_cur = delta.clone();
        let mut d_prev: OVector<f64, D> = Matrix::zeros_generic(dim, U1);
        let mut dvel = &self.cur_state_der_der * (h * 0.5);

        // Absolute y_cur is still required for the force evaluation.
        let mut y_cur = &self.cur_state + &d_cur;

        // Velocity passed to the force at sub-step 1 (= delta_1 / h, the
        // standard half-step estimate). For pure gravity this is unused.
        let mut vel_est = &self.cur_state_der + &dvel;

        for k in 1..n_sub {
            let t_k = (self.cur_time.jd + k as f64 * h).into();
            let f_k = (self.func)(t_k, &y_cur, &vel_est, &mut self.metadata, false)?;

            // delta_{k+1} = delta_k + h^2 * f_k
            delta += &f_k * h2;

            // dvel_{k+1} = dvel_k + h * f_k (cancellation-free)
            dvel += &f_k * h;

            // Advance position increments: d_next = d_cur + delta_{k+1}.
            let d_next = &d_cur + &delta;

            // Central-difference velocity at y_k for the next sub-step's force
            // evaluation. Algebraically (y_{k+1} - y_{k-1})/(2h), assembled
            // from increments to keep cancellation contained.
            vel_est = &self.cur_state_der + (&d_next - &d_prev) / (2.0 * h);

            d_prev = d_cur;
            d_cur = d_next;
            y_cur = &self.cur_state + &d_cur;
        }

        // Final function evaluation at y_n for Gragg smoothing.
        let t_n = (self.cur_time.jd + h_total).into();
        let f_n = (self.func)(t_n, &y_cur, &vel_est, &mut self.metadata, false)?;

        // Gragg-smoothed increments:
        //   y_smooth      = y_n + f_n * h^2/4         => D_pos = d_cur + f_n * h^2/4
        //   ydot_smooth   = (y_n - y_{n-1})/h + f_n*h/2
        //                                              => D_vel = dvel + f_n * h/2
        let d_smooth = &d_cur + &f_n * (h2 * 0.25);
        let ddot_smooth = &dvel + &f_n * (h * 0.5);

        Ok((d_smooth, ddot_smooth))
    }

    /// Attempt one macro-step of size `step_size`.
    ///
    /// Builds the extrapolation tableau column by column, checking the error
    /// estimate after each column. The tableau stores macro-step
    /// **increments** rather than absolute states, which puts the
    /// Aitken-Neville recurrence on numbers of order `h * v` and keeps the
    /// `prev - prev_prev` cancellation off the position-scale floor. On
    /// acceptance, the state is updated using compensated summation and the
    /// suggested next step size is returned.
    fn step(&mut self, step_size: f64) -> KeteResult<f64> {
        let cd = self.control_dim;

        // Extrapolation tableau (increments, not absolute states). Row j stores
        // the most recently computed extrapolation for sub-step count N_SEQ[j].
        // We only need two "active" rows at a time for the Aitken-Neville
        // recurrence, but keeping the full triangle makes the indexing
        // straightforward and K_MAX is small (8).
        let mut tab_pos: Vec<Vec<OVector<f64, D>>> = Vec::with_capacity(K_MAX);
        let mut tab_vel: Vec<Vec<OVector<f64, D>>> = Vec::with_capacity(K_MAX);

        for j in 0..K_MAX {
            let n_sub = N_SEQ[j];
            let (dpos_j, dvel_j) = self.modified_midpoint(step_size, n_sub)?;

            // Start this row of the tableau with the raw midpoint increment.
            let mut row_pos = Vec::with_capacity(j + 1);
            let mut row_vel = Vec::with_capacity(j + 1);
            row_pos.push(dpos_j);
            row_vel.push(dvel_j);

            // Aitken-Neville extrapolation across previous rows. Linear in the
            // inputs, so extrapolating increments yields the increment of the
            // extrapolation - same answer, but the `prev - prev_prev`
            // subtraction now happens on small numbers.
            for k in 1..=j {
                let ratio_sq = (N_SEQ[j] as f64 / N_SEQ[j - k] as f64).powi(2);
                let denom = ratio_sq - 1.0;

                let prev_pos = &row_pos[k - 1];
                let prev_prev_pos = &tab_pos[j - 1][k - 1];
                let delta_pos = prev_pos - prev_prev_pos;
                let extrap_pos = prev_pos + &delta_pos / denom;

                let prev_vel = &row_vel[k - 1];
                let prev_prev_vel = &tab_vel[j - 1][k - 1];
                let delta_vel = prev_vel - prev_prev_vel;
                let extrap_vel = prev_vel + &delta_vel / denom;

                row_pos.push(extrap_pos);
                row_vel.push(extrap_vel);
            }

            // Check error estimate (need at least 2 columns, and at least
            // MIN_J columns to enforce the order floor for acceptance).
            // Position error controls step acceptance. Velocity error
            // influences step-size growth: large velocity extrapolation
            // error (indicating under-resolved energy) prevents aggressive
            // step growth without forcing rejections on short arcs.
            if j >= MIN_J {
                let cur_dpos = &row_pos[j];
                let prev_dpos = &row_pos[j - 1];

                let mut err_max: f64 = 0.0;
                for i in 0..cd {
                    // Scale uses the absolute position |cur_state + cur_dpos|;
                    // the error is |cur_dpos - prev_dpos| (a difference of
                    // small numbers, so cancellation here is bounded by the
                    // truncation error itself).
                    let abs_pos_i = self.cur_state[i] + cur_dpos[i];
                    let scale = RTOL * abs_pos_i.abs() + EPS_FLOOR;
                    let err_i = (cur_dpos[i] - prev_dpos[i]).abs() / scale;
                    if err_i > err_max {
                        err_max = err_i;
                    }
                }

                if err_max < 1.0 {
                    // Step accepted on position accuracy.
                    // Compute velocity error for step-size control.
                    let cur_dvel = &row_vel[j];
                    let prev_dvel = &row_vel[j - 1];
                    let h_abs = step_size.abs();
                    let mut vel_err: f64 = 0.0;
                    for i in 0..cd {
                        let abs_pos_i = self.cur_state[i] + cur_dpos[i];
                        let vel_scale = RTOL * abs_pos_i.abs() / h_abs + EPS_FLOOR;
                        let err_v = (cur_dvel[i] - prev_dvel[i]).abs() / vel_scale;
                        if err_v > vel_err {
                            vel_err = err_v;
                        }
                    }
                    // Use the larger of position and velocity error for
                    // step-size prediction.
                    let sizing_err = err_max.max(vel_err);
                    // Step accepted. The accepted row entries ARE the
                    // increments - apply directly via compensated (Kahan)
                    // summation, no second subtraction needed.
                    let accepted_dpos = &row_pos[j];
                    let accepted_dvel = &row_vel[j];

                    // Kahan summation for position.
                    for i in 0..self.cur_state.len() {
                        let y_pos = accepted_dpos[i] - self.comp_state[i];
                        let t_pos = self.cur_state[i] + y_pos;
                        self.comp_state[i] = (t_pos - self.cur_state[i]) - y_pos;
                        self.cur_state[i] = t_pos;
                    }

                    // Kahan summation for velocity.
                    for i in 0..self.cur_state_der.len() {
                        let y_vel = accepted_dvel[i] - self.comp_state_der[i];
                        let t_vel = self.cur_state_der[i] + y_vel;
                        self.comp_state_der[i] = (t_vel - self.cur_state_der[i]) - y_vel;
                        self.cur_state_der[i] = t_vel;
                    }

                    let y_t = step_size - self.comp_time;
                    let t_t = self.cur_time.jd + y_t;
                    self.comp_time = (t_t - self.cur_time.jd) - y_t;
                    self.cur_time.jd = t_t;

                    // Re-evaluate acceleration at the accepted state.
                    self.cur_state_der_der = (self.func)(
                        self.cur_time,
                        &self.cur_state,
                        &self.cur_state_der,
                        &mut self.metadata,
                        true,
                    )?;

                    // Estimate next step size.
                    // sizing_err is the combined pos+vel scaled error.
                    // Order is 2*(j+1) since j is 0-indexed.
                    //   H_next = H * (1/err)^(1/(2k+1)) * safety
                    // where k = j+1 (1-indexed column count).
                    let order = 2 * (j + 1);
                    let next =
                        step_size * (1.0 / sizing_err).powf(1.0 / (order as f64 + 1.0)) * SAFETY;

                    let clamped = next.abs().clamp(
                        step_size.abs() * MIN_RATIO,
                        step_size.abs() * MIN_RATIO.recip(),
                    );

                    // Preserve sign of step direction.
                    let result = clamped.copysign(step_size);

                    tab_pos.push(row_pos);
                    tab_vel.push(row_vel);
                    let _ = (tab_pos, tab_vel); // consumed
                    return Ok(result);
                }
            }

            tab_pos.push(row_pos);
            tab_vel.push(row_vel);
        }

        // All K_MAX columns exhausted without acceptance.
        Err(Error::Convergence(
            "Bulirsch-Stoer step failed to converge".into(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use nalgebra::Vector3;

    use super::*;
    use crate::integrators::stress_tests::{CentralAccelMeta, central_accel};
    use crate::kepler::analytic_2_body;

    /// Two-body orbit validated against the analytic Kepler solution.
    #[test]
    fn two_body_vs_analytic() {
        let init_pos = Vector3::new(0.46937657, -0.8829981, 0.0);
        let init_vel = Vector3::new(0.01518942, 0.00807426, 0.0);

        let (exact_pos, exact_vel) =
            analytic_2_body(1000.0_f64.into(), &init_pos, &init_vel, None).unwrap();

        let (bs_pos, bs_vel, _) = BulirschStoerIntegrator::integrate(
            &central_accel,
            init_pos,
            init_vel,
            0.0.into(),
            1000.0.into(),
            CentralAccelMeta::default(),
            None,
        )
        .unwrap();

        for i in 0..3 {
            assert!(
                (bs_pos[i] - exact_pos[i]).abs() < 5e-11,
                "pos[{i}] error vs analytic: {:.2e}",
                (bs_pos[i] - exact_pos[i]).abs()
            );
            assert!(
                (bs_vel[i] - exact_vel[i]).abs() < 5e-12,
                "vel[{i}] error vs analytic: {:.2e}",
                (bs_vel[i] - exact_vel[i]).abs()
            );
        }
    }

    /// Test that a circular orbit conserves energy over 100k days.
    #[test]
    fn energy_conservation() {
        let gms = crate::constants::GMS;
        let v_circ = gms.sqrt();
        let (pos, vel, _) = BulirschStoerIntegrator::integrate(
            &central_accel,
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(0.0, v_circ, 0.0),
            0.0.into(),
            100_000.0.into(),
            CentralAccelMeta::default(),
            None,
        )
        .unwrap();

        let energy_init = 0.5 * v_circ * v_circ - gms;
        let energy_final = 0.5 * vel.norm_squared() - gms / pos.norm();
        let rel_err = ((energy_final - energy_init) / energy_init).abs();
        assert!(rel_err < 5e-11, "Energy conservation error: {rel_err:.2e}");
    }

    /// Test backward integration recovers the initial state.
    #[test]
    fn backward_integration() {
        let init_pos = Vector3::new(0.46937657, -0.8829981, 0.0);
        let init_vel = Vector3::new(0.01518942, 0.00807426, 0.0);

        let (mid_pos, mid_vel, _) = BulirschStoerIntegrator::integrate(
            &central_accel,
            init_pos,
            init_vel,
            0.0.into(),
            500.0.into(),
            CentralAccelMeta::default(),
            None,
        )
        .unwrap();

        let (final_pos, final_vel, _) = BulirschStoerIntegrator::integrate(
            &central_accel,
            mid_pos,
            mid_vel,
            500.0.into(),
            0.0.into(),
            CentralAccelMeta::default(),
            None,
        )
        .unwrap();

        for i in 0..3 {
            assert!(
                (final_pos[i] - init_pos[i]).abs() < 1e-10,
                "Backward pos[{i}] error: {:.2e}",
                (final_pos[i] - init_pos[i]).abs()
            );
            assert!(
                (final_vel[i] - init_vel[i]).abs() < 1e-10,
                "Backward vel[{i}] error: {:.2e}",
                (final_vel[i] - init_vel[i]).abs()
            );
        }
    }

    /// Test with a highly eccentric orbit (e ~ 0.9).
    #[test]
    fn eccentric_orbit() {
        let gms = crate::constants::GMS;
        let r_peri = 0.1;
        let a = 1.0;
        let v_peri = (gms * (2.0 / r_peri - 1.0 / a)).sqrt();

        let (pos, vel, _) = BulirschStoerIntegrator::integrate(
            &central_accel,
            Vector3::new(r_peri, 0.0, 0.0),
            Vector3::new(0.0, v_peri, 0.0),
            0.0.into(),
            365.25.into(),
            CentralAccelMeta::default(),
            None,
        )
        .unwrap();

        let energy_init = 0.5 * v_peri * v_peri - gms / r_peri;
        let energy_final = 0.5 * vel.norm_squared() - gms / pos.norm();
        let rel_err = ((energy_final - energy_init) / energy_init).abs();
        assert!(
            rel_err < 1e-11,
            "Eccentric orbit energy error: {rel_err:.2e}"
        );
    }

    /// Near-zero integration interval returns the initial state unchanged.
    #[test]
    fn zero_interval() {
        let (pos, vel, _) = BulirschStoerIntegrator::integrate(
            &central_accel,
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(0.0, 0.01, 0.0),
            0.0.into(),
            0.0.into(),
            CentralAccelMeta::default(),
            None,
        )
        .unwrap();
        assert_eq!(pos[0], 1.0);
        assert_eq!(vel[1], 0.01);
    }
}
