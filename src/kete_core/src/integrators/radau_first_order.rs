//! Gauss-Radau Spacing Numerical Integrator for first-order systems.
//!
//! # Scheme
//!
//! Solves `y' = F(t, y)` for `y` of runtime dimension `D`.
//!
//! Over one step of size `H` from `t_0`, with `s = (t - t_0) / H` in `[0, 1]`, the
//! right-hand side is approximated by a degree-7 polynomial in `s`:
//!
//! ```text
//! F(t_0 + s H) = F_0 + b_0 s + b_1 s^2 + ... + b_6 s^7
//! ```
//!
//! and the solution follows by a single integration:
//!
//! ```text
//! y(t_0 + s H) = y_0  +  H [ F_0 s  +  sum_k b_k s^(k+2) / (k+2) ]
//!
//! y(t_0 + H)   = y_0  +  H ( F_0  +  b . U ),        U_k = 1 / (k + 2)
//! ```
//!
//! This is the same machinery as the second-order [`RadauIntegrator`] with the second
//! integral dropped. `U` is the first-integral coefficient vector that integrator already
//! uses for its velocity update; the Gauss-Radau node spacings, the divided-difference
//! ladder that produces `g`, and the fixed lower-triangular `C` that converts `g -> b` are
//! all independent of the order form and are shared from that module.
//!
//! The `b` coefficients are implicit - each node's `y` depends on them - and are found by
//! fixed-point iteration, at most [`MAX_SWEEPS`] sweeps per step.
//!
//! # Error control
//!
//! `b_k` scales as `H^k`, so the last coefficient `b_6` is the step's local error
//! indicator and the step-size exponent is `1/7` in both order forms.
//!
//! The denominator is per-component and taken from the step itself:
//!
//! ```text
//! scale_i    = max over the step's node evaluations of |F_i|
//! ratio      = max_i |b6_i| / scale_i        over the first `control_dim` components
//! ```
//!
//! which is dimensionless, needs nothing from the caller, and holds every component to the
//! same relative tolerance regardless of its magnitude or units. That matters more here
//! than in the second-order form: a first-order system mixes units by construction -
//! stacking `(velocity, acceleration)` spans the ratio between them - and the element-space
//! right-hand sides this integrator exists to serve are smaller still. Components with
//! `scale_i = 0` are skipped: if `F_i` vanishes at every node then every divided difference
//! vanishes and `b_i` is exactly zero, so they carry no information.
//!
//! [`RadauIntegrator`] takes the same idea but a cheaper scale, `max(|F_i|)` over just the
//! two evaluations bracketing the step, which needs no accumulation and no per-sweep
//! allocation. That form is slightly looser on a problem stiff enough to push the corrector
//! past its contraction limit, where it can accept an under-converged step unless
//! [`SWEEP_TOL`] is tightened. Stiffness of that order does not arise in the second-order
//! integrator's gravitational problems, so the two definitions are each kept where they are
//! used rather than unified.
//!
//! # Predictor
//!
//! The corrector starts from the previous step's `b` rather than from zero. Because `b`
//! lives in the step-normalized variable `s = (t - t_0) / H`, that guess is at the wrong
//! scale as soon as the step size moves, and [`Self::predict`] rescales it exactly,
//! `b_k -> q^(k+1) b_k` for `q = H_new / H_old`.
//!
//! [`RadauIntegrator`] omits this: it does not earn its complexity on the second-order
//! form. It does matter here, and two things make this form more sensitive. Its
//! convergence test is per-component relative with no absolute floor,
//! so it will not tolerate a mis-scaled guess the way a floored test does. And a mis-scaled
//! guess is badly wrong when the step moves: the controller may grow the step by the
//! `1/MIN_RATIO` clamp, and `b_6` then starts off by `q^7` - worse than starting from zero.
//!
//! # Order
//!
//! The eight Gauss-Radau nodes make the converged corrector a collocation method whose
//! quadrature is exact to degree `2n - 2 = 14`, giving order `2n - 1 = 15` and a local
//! error of `H^16`. That argument never references the order form, so the first-order form
//! does not lose order relative to the second-order one;
//! `convergence_order_quadrature` and `convergence_order_ode` in the test module exercise
//! the local-error exponent of a single step, which rises toward that value as the step
//! shrinks until f64 rounding closes the window.
//!
//! This form does need a smaller step than the second-order one to hold the same relative
//! tolerance, and the mechanism is a polynomial degree rather than an order: the
//! second-order form represents the velocity as the integral of the degree-7 acceleration
//! polynomial, which is degree 8, and the position as its second integral, degree 9. The
//! first-order form fits an independent degree-7 polynomial to each of `(v, a)`, so its
//! velocity carries one fewer degree of freedom on the same node set.
//!
//! # Validity and limits
//!
//! - Step size is clamped below at [`MIN_STEP`] days and its per-step change to
//!   `[MIN_RATIO, 1/MIN_RATIO]`.
//! - At most [`MAX_SWEEPS`] corrector sweeps; on failure the caller retries at `0.7 H`,
//!   up to 10 consecutive failures before giving up.
//! - Not symplectic. Energy drift on a Hamiltonian problem is secular, not bounded.
//! - Compensated (Kahan) summation on the state and on time, so roundoff accumulates as
//!   about `sqrt(N) eps` rather than `N eps`.
//!
//! # References
//!
//! E. Everhart (1985), 'An efficient integrator that uses Gauss-Radau spacings',
//! in A. Carusi and G. B. Valsecchi (eds.), Dynamics of Comets: Their Origin and
//! Evolution, Astrophysics and Space Science Library vol. 115, D. Reidel.
//!
//! E. Everhart (1974), 'Implicit single-sequence methods for integrating orbits',
//! Celestial Mechanics 10, 35-55.
//
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
use crate::integrators::radau::{
    C_MAT, EPSILON, GAUSS_RADAU_SPACINGS, MIN_RATIO, MIN_STEP, U_POW_TABLE, U_VEC,
};
use crate::integrators::util::FirstOrderODEDyn;
use crate::prelude::KeteResult;
use crate::time::{TDB, Time};
use itertools::izip;
use nalgebra::allocator::Allocator;
use nalgebra::{DefaultAllocator, Dim, Matrix, OMatrix, OVector, U1, U7};

/// Integrator will return a result of this type.
type RadauFirstOrderResult<MType, D> = KeteResult<(OVector<f64, D>, MType)>;

/// Maximum corrector sweeps before a step is declared non-convergent.
const MAX_SWEEPS: usize = 10;

/// Relative tolerance on the change in `b_6` between corrector sweeps.
const SWEEP_TOL: f64 = 1e-14;

/// Gauss-Radau Spacing Numerical Integrator for first-order systems.
///
/// Solves `y' = F(t, y)`. See the module documentation for the scheme, the error-control
/// definition, and the order.
#[allow(missing_debug_implementations, reason = "No debug impl needed")]
pub struct RadauFirstOrder<'a, MType, D: Dim>
where
    DefaultAllocator: Allocator<D, U1> + Allocator<D, U7>,
{
    func: FirstOrderODEDyn<'a, MType, D>,
    metadata: MType,

    final_time: Time<TDB>,

    cur_time: Time<TDB>,
    cur_state: OVector<f64, D>,
    /// `F(cur_time, cur_state)`, the zeroth coefficient of the step's polynomial.
    cur_state_der: OVector<f64, D>,

    cur_b: OMatrix<f64, D, U7>,
    g_scratch: OMatrix<f64, D, U7>,

    state_scratch: OVector<f64, D>,
    b_scratch: OVector<f64, D>,
    eval_scratch: OVector<f64, D>,
    /// Per-component `max |F_i|` over the current step's node evaluations.
    scale_scratch: OVector<f64, D>,

    /// Number of leading dimensions used for convergence and step-size control.
    /// Defaults to the full state dimension `D`. For variational propagation set this
    /// to the physical rows only, so large variational entries do not shrink the step.
    control_dim: usize,

    // Kahan compensated summation error accumulators.
    comp_state: OVector<f64, D>,
    comp_time: f64,

    /// Size of the last attempted step, used to rescale `cur_b` when the step size
    /// changes.  Zero before the first step.  Tracks attempts rather than accepted
    /// steps so a shrink-and-retry rescales from the size that was actually tried.
    last_step_size: f64,
}

impl<'a, MType, D: Dim> RadauFirstOrder<'a, MType, D>
where
    DefaultAllocator: Allocator<D, U1> + Allocator<D, U7>,
{
    /// Integrate `y' = F(t, y)` from the initial time to the final time.
    ///
    /// `control_dim` restricts convergence and step-size control to the first `n`
    /// components; `None` uses all of them.
    ///
    /// # Errors
    /// Integration may fail if the function returns an error, if a step fails to
    /// converge repeatedly, or if the state diverges to non-finite values.
    pub fn integrate(
        func: FirstOrderODEDyn<'a, MType, D>,
        state_init: OVector<f64, D>,
        time_init: Time<TDB>,
        final_time: Time<TDB>,
        metadata: MType,
        control_dim: Option<usize>,
    ) -> RadauFirstOrderResult<MType, D> {
        let mut integrator = Self::new(func, state_init, time_init, final_time, metadata)?;
        if (final_time - time_init).elapsed.abs() < 1e-10 {
            return Ok((integrator.cur_state, integrator.metadata));
        }
        integrator.control_dim = control_dim.unwrap_or(integrator.control_dim);
        if integrator.control_dim > integrator.cur_state.len() {
            return Err(Error::ValueError(format!(
                "control_dim ({}) exceeds state dimension ({})",
                integrator.control_dim,
                integrator.cur_state.len(),
            )))?;
        }

        // First step guess. The controller reaches the right size within a few steps
        // (it may grow by 1/MIN_RATIO per step), so this only needs to be safe, not
        // sharp. The 1/3 exponent is deliberately lower order than the 1/7 the
        // controller uses, to avoid overshooting on the first step.
        let mut next_step_size: f64 = {
            let f0_norm = integrator
                .cur_state_der
                .rows(0, integrator.control_dim)
                .amax();
            let h0 = if f0_norm > 0.0 {
                (EPSILON / f0_norm).powf(1.0 / 3.0).min(0.1)
            } else {
                0.1
            };
            h0.copysign((integrator.final_time - integrator.cur_time).elapsed)
        };

        // Relative tolerance scaled by the JD magnitude. An absolute tolerance finer
        // than the ULP of `cur_time.jd` (about 5.5e-10 at JD ~2.5e6) could never be
        // satisfied and the loop would not terminate.
        let convergence_tol = {
            let scale = integrator
                .cur_time
                .jd
                .abs()
                .max(integrator.final_time.jd.abs());
            (scale * 1e-13).max(1e-12)
        };

        let mut step_failures = 0;
        loop {
            // A non-finite step size makes every comparison below false, so the
            // early-exit and floor branches would never fire and the loop would spin.
            // This usually means the state diverged during the previous step.
            if !next_step_size.is_finite() {
                return Err(Error::Convergence(
                    "Radau produced non-finite step size (state likely diverged).".into(),
                ));
            }
            if (integrator.cur_time - integrator.final_time).elapsed.abs() <= next_step_size.abs() {
                next_step_size = (integrator.final_time - integrator.cur_time).elapsed;
            }
            match integrator.step(next_step_size) {
                Ok(s) => {
                    next_step_size = s;
                    if (integrator.cur_time - integrator.final_time).elapsed.abs() < convergence_tol
                    {
                        return Ok((integrator.cur_state, integrator.metadata));
                    }
                    step_failures = 0;
                }
                Err(error) => match error {
                    Error::Bounds(_) | Error::Impact(_, _) | Error::OutOfMemory => Err(error)?,
                    Error::Convergence(_)
                    | Error::ValueError(_)
                    | Error::UnknownFrame(_)
                    | Error::IOError(_)
                    | Error::LockFailed => {
                        step_failures += 1;
                        next_step_size *= 0.7;
                        if step_failures > 10 {
                            Err(Error::Convergence("Radau failed to converge.".into()))?;
                        }
                    }
                },
            }
            if next_step_size.abs() < MIN_STEP {
                next_step_size = MIN_STEP.copysign(next_step_size);
            }
        }
    }

    fn new(
        func: FirstOrderODEDyn<'a, MType, D>,
        state_init: OVector<f64, D>,
        time_init: Time<TDB>,
        final_time: Time<TDB>,
        metadata: MType,
    ) -> KeteResult<Self> {
        let (dim, _) = state_init.shape_generic();
        let full_dim = state_init.len();
        let mut res = Self {
            func,
            metadata,
            final_time,
            cur_time: time_init,
            cur_state: state_init,
            cur_state_der: Matrix::zeros_generic(dim, U1),
            cur_b: Matrix::zeros_generic(dim, U7),
            g_scratch: Matrix::zeros_generic(dim, U7),
            b_scratch: Matrix::zeros_generic(dim, U1),
            state_scratch: Matrix::zeros_generic(dim, U1),
            eval_scratch: Matrix::zeros_generic(dim, U1),
            scale_scratch: Matrix::zeros_generic(dim, U1),
            control_dim: full_dim,
            comp_state: Matrix::zeros_generic(dim, U1),
            comp_time: 0.0,
            last_step_size: 0.0,
        };
        res.cur_state_der = (res.func)(time_init, &res.cur_state, &mut res.metadata, true)?;
        Ok(res)
    }

    /// Attempt a single integration step of size `step_size`.
    ///
    /// Returns the recommended next step size on success. Failure means the corrector
    /// did not converge within [`MAX_SWEEPS`], or the ODE function itself failed.
    fn step(&mut self, step_size: f64) -> KeteResult<f64> {
        self.predict(step_size);
        self.g_scratch.fill(0.0);
        self.state_scratch.fill(0.0);
        self.eval_scratch.set_column(0, &self.cur_state_der);
        // The node evaluations have not happened yet, so seed the per-component scale
        // with |F_0|, a genuine evaluation at the left endpoint node.
        self.scale_scratch.set_column(0, &self.cur_state_der.abs());

        for _ in 0..MAX_SWEEPS {
            self.b_scratch.set_column(0, &self.cur_b.column(6));

            for (idj, gauss_radau_frac) in GAUSS_RADAU_SPACINGS.iter().enumerate().skip(1) {
                let u_pow = &U_POW_TABLE[idj - 1];
                let h1 = gauss_radau_frac * step_size;

                // Predict y at this node from the current b, which is the single
                // integral of the polynomial evaluated at s = gauss_radau_frac.
                izip!(
                    self.state_scratch.iter_mut(),
                    self.cur_state.iter(),
                    self.cur_state_der.iter(),
                    self.cur_b.row_iter(),
                )
                .for_each(|(out, state, der, b)| {
                    *out = state + h1 * (der + b.dot(u_pow));
                });

                self.eval_scratch.set_column(
                    0,
                    &(self.func)(
                        (self.cur_time.jd + gauss_radau_frac * step_size).into(),
                        &self.state_scratch,
                        &mut self.metadata,
                        false,
                    )?,
                );

                izip!(self.scale_scratch.iter_mut(), self.eval_scratch.iter())
                    .for_each(|(scale, eval)| *scale = scale.max(eval.abs()));

                let diff = &self.eval_scratch - &self.cur_state_der;

                // Divided differences, equation (4) in Everhart. The lookup tables and
                // switch statements of the original were far slower than this.
                self.g_scratch.set_column(idj - 1, &{
                    let mut gk = diff / *gauss_radau_frac;
                    for (idz, gr_step) in GAUSS_RADAU_SPACINGS.iter().enumerate().take(idj).skip(1)
                    {
                        gk = (gk - self.g_scratch.column(idz - 1)) / (gauss_radau_frac - gr_step);
                    }
                    gk
                });
            }

            self.g_scratch.mul_to(&C_MAT, &mut self.cur_b);

            let b_diff = (self.cur_b.column(6) - &self.b_scratch).abs();
            if self.scaled_max(&b_diff) < SWEEP_TOL {
                return self.accept(step_size);
            }
        }
        Err(Error::Convergence(
            "Radau first-order step failed to converge".into(),
        ))?
    }

    /// Rescale the carried-over `b` for a change of step size (Everhart's predictor).
    ///
    /// `b` are the coefficients of the right-hand side in the step-normalized variable
    /// `s = (t - t_0) / H`.  Changing the step to `H' = q H` means `s = q s'`, so the same
    /// polynomial re-expressed in `s'` has coefficients
    ///
    /// ```text
    /// b_k' = q^(k+1) b_k
    /// ```
    ///
    /// This is an exact change of variable, not an approximation: it costs 7 scalar
    /// multiplies per component and leaves the corrector a strictly better starting guess
    /// whenever the step size moved.  Without it the previous step's `b` is reused at the
    /// wrong scale, which for a growing step understates every coefficient.
    fn predict(&mut self, step_size: f64) {
        if self.last_step_size == 0.0 {
            self.last_step_size = step_size;
            return;
        }
        let q = step_size / self.last_step_size;
        self.last_step_size = step_size;
        if (q - 1.0).abs() < f64::EPSILON {
            return;
        }
        let mut q_pow = 1.0;
        for idx in 0..7 {
            q_pow *= q;
            self.cur_b.column_mut(idx).scale_mut(q_pow);
        }
    }

    /// Largest `|v_i| / scale_i` over the controlled components.
    ///
    /// Components whose right-hand side vanished at every node of this step are skipped:
    /// their divided differences, and therefore `b_i`, are exactly zero, so they carry
    /// no error information and would otherwise be a 0/0.
    fn scaled_max(&self, vals: &OVector<f64, D>) -> f64 {
        izip!(
            vals.iter().take(self.control_dim),
            self.scale_scratch.iter().take(self.control_dim),
        )
        .filter(|(_, scale)| **scale > 0.0)
        .fold(0.0_f64, |acc, (val, scale)| acc.max(val / scale))
    }

    /// Commit a converged step and return the recommended next step size.
    fn accept(&mut self, step_size: f64) -> KeteResult<f64> {
        izip!(
            self.cur_state.iter_mut(),
            self.cur_state_der.iter(),
            self.cur_b.row_iter(),
            self.comp_state.iter_mut(),
        )
        .for_each(|(state, der, b, comp)| {
            let delta = step_size * (der + b.dot(&*U_VEC));
            let y = delta - *comp;
            let t = *state + y;
            *comp = (t - *state) - y;
            *state = t;
        });

        let y_t = step_size - self.comp_time;
        let t_t = self.cur_time.jd + y_t;
        self.comp_time = (t_t - self.cur_time.jd) - y_t;
        self.cur_time.jd = t_t;

        self.cur_state_der = (self.func)(self.cur_time, &self.cur_state, &mut self.metadata, true)?;

        // b_6 scales as H^7, hence the 1/7 exponent. The worst-resolved controlled
        // component drives the step.
        let error_ratio = self.scaled_max(&self.cur_b.column(6).abs().into_owned());
        if error_ratio <= 0.0 {
            // Every controlled component is exactly polynomial of degree < 7 over this
            // step, so there is no error signal. Grow at the clamp.
            return Ok(step_size / MIN_RATIO);
        }
        Ok(step_size
            * (EPSILON / error_ratio)
                .powf(1.0 / 7.0)
                .clamp(MIN_RATIO, MIN_RATIO.recip()))
    }
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::unnecessary_wraps,
        reason = "test right-hand sides must match the FirstOrderODEDyn signature"
    )]

    use super::*;
    use crate::constants::GMS;
    use crate::integrators::RadauIntegrator;
    use crate::integrators::stress_tests::{CentralAccelMeta, central_accel};
    use nalgebra::{Vector3, Vector6};

    /// Counts right-hand side evaluations.
    #[derive(Debug, Default, Clone)]
    struct Evals(usize);

    /// Rate constant of the two order-measurement problems.  Exponentials are used
    /// rather than sinusoids because the error constant of the quadrature carries a high
    /// derivative of the integrand: for `sin` that derivative changes sign with `h`, so
    /// the local error is not a clean power law and the measured exponent oscillates.
    /// All derivatives of `exp` share one sign, so the power law is clean.
    const RATE_TEST: f64 = 3.0;

    /// Strength of the state feedback in [`manufactured`].  Kept small so the corrector
    /// stays comfortably inside its contraction region over the whole step ladder (see
    /// [`corrector_contraction_limit`]), leaving the time variation, set by
    /// [`RATE_TEST`], as what drives the truncation error.
    const FEEDBACK: f64 = 0.01;

    /// Manufactured solution: `y' = a exp(a t) + eps (y - exp(a t))`, exact `y = exp(a t)`
    /// for `y(0) = 1`.  The `eps` term vanishes on the exact solution, so it changes
    /// nothing about the answer but forces the corrector to actually iterate on `y`.
    fn manufactured(
        time: Time<TDB>,
        state: &Vector6<f64>,
        meta: &mut Evals,
        _exact: bool,
    ) -> KeteResult<Vector6<f64>> {
        meta.0 += 1;
        let truth = (RATE_TEST * time.jd).exp();
        let mut out = Vector6::zeros();
        out[0] = RATE_TEST * truth + FEEDBACK * (state[0] - truth);
        Ok(out)
    }

    /// Two-body motion as a first-order system, `y = (pos, vel)`.
    fn two_body(
        _time: Time<TDB>,
        state: &Vector6<f64>,
        meta: &mut Evals,
        _exact: bool,
    ) -> KeteResult<Vector6<f64>> {
        meta.0 += 1;
        let pos = state.fixed_rows::<3>(0);
        let accel = pos * (-GMS * pos.norm().powi(-3));
        let mut out = Vector6::zeros();
        out.fixed_rows_mut::<3>(0)
            .copy_from(&state.fixed_rows::<3>(3));
        out.fixed_rows_mut::<3>(3).copy_from(&accel);
        Ok(out)
    }

    /// State on a conic of semi-latus rectum `p` at true anomaly `nu`, inclined.
    fn conic_state(p: f64, ecc: f64, nu: f64) -> Vector6<f64> {
        let r = p / (1.0 + ecc * nu.cos());
        let (c, s) = (0.6_f64.cos(), 0.6_f64.sin());
        let pos = Vector3::new(r * nu.cos(), r * nu.sin() * c, r * nu.sin() * s);
        let vs = (GMS / p).sqrt();
        let (vx, vy) = (-vs * nu.sin(), vs * (ecc + nu.cos()));
        let vel = Vector3::new(vx, vy * c, vy * s);
        let mut out = Vector6::zeros();
        out.fixed_rows_mut::<3>(0).copy_from(&pos);
        out.fixed_rows_mut::<3>(3).copy_from(&vel);
        out
    }

    /// `y' = a exp(a t)`, exact `y(h) = exp(a h) - 1` for `y(0) = 0`.  The right-hand
    /// side does not depend on `y`, so the corrector is exact after one sweep and the
    /// step is limited only by the rounding floor.  This isolates the quadrature, which
    /// is where the order comes from.
    fn quadrature(
        time: Time<TDB>,
        _state: &Vector6<f64>,
        meta: &mut Evals,
        _exact: bool,
    ) -> KeteResult<Vector6<f64>> {
        meta.0 += 1;
        let mut out = Vector6::zeros();
        out[0] = RATE_TEST * (RATE_TEST * time.jd).exp();
        Ok(out)
    }

    /// Local error of one step of size `h`, bypassing the adaptive driver.  Returns
    /// `None` when the corrector does not converge at that step size.
    fn local_error(
        func: FirstOrderODEDyn<'_, Evals, nalgebra::U6>,
        y0: Vector6<f64>,
        h: f64,
        exact: impl Fn(f64) -> f64,
    ) -> Option<f64> {
        let mut integrator =
            RadauFirstOrder::new(func, y0, 0.0.into(), h.into(), Evals::default()).ok()?;
        let _next = integrator.step(h).ok()?;
        let truth = exact(h);
        Some((integrator.cur_state[0] - truth).abs() / truth.abs().max(1.0))
    }

    /// Measure the local-error exponent and check it against a 15th-order method.
    ///
    /// Local error of one step scales as `H^(p+1)`, so `p = 15` must read as 16. This rules
    /// out the alternative explanation for the first-order form's higher cost, that the
    /// scheme drops to order 8 when written this way.
    fn assert_order_16(
        name: &str,
        func: FirstOrderODEDyn<'_, Evals, nalgebra::U6>,
        y0: Vector6<f64>,
        steps: &[f64],
        exact: impl Fn(f64) -> f64 + Copy,
    ) {
        let errs: Vec<Option<f64>> = steps
            .iter()
            .map(|&h| local_error(func, y0, h, exact))
            .collect();

        let mut clean = Vec::new();
        for idx in 1..steps.len() {
            let (Some(prev), Some(cur)) = (errs[idx - 1], errs[idx]) else {
                println!(
                    "{name}: h {:6.3} corrector did not converge",
                    steps[idx - 1]
                );
                continue;
            };
            let order = (prev / cur).ln() / (steps[idx - 1] / steps[idx]).ln();
            println!(
                "{name}: h {:6.3} -> {:6.3}   err {prev:9.3e} -> {cur:9.3e}   order {order:6.2}",
                steps[idx - 1],
                steps[idx],
            );
            // Only pairs whose smaller error is clear of the rounding floor say
            // anything about truncation; below ~1e-15 this reads f64 noise.
            if cur > 1e-15 {
                clean.push(order);
            }
        }

        assert!(
            !clean.is_empty(),
            "{name}: no step pair above the rounding floor",
        );
        // The measured exponent rises toward its asymptote as h shrinks, and the
        // asymptotic regime only just opens before f64 rounding closes it, so 15 and 16
        // are not separable here.  What is separable, and what this asserts, is that the
        // exponent is nowhere near 8: every usable pair reads 12 or more, and the finest
        // one reads 14 or more.  An order collapse would break both.
        for order in &clean {
            assert!(
                (12.0..=17.0).contains(order),
                "{name}: local order {order:.2} is far from a 15th-order method",
            );
        }
        let finest = clean.last().copied().unwrap_or_default();
        assert!(
            finest >= 14.0,
            "{name}: finest usable pair reads order {finest:.2}, expected >= 14",
        );
    }

    /// Order measured on pure quadrature, where the corrector converges immediately so
    /// the step is limited only by the rounding floor.
    #[test]
    fn convergence_order_quadrature() {
        assert_order_16(
            "quadrature",
            &quadrature,
            Vector6::zeros(),
            &[3.0, 2.5, 2.0, 1.6, 1.3, 1.05, 0.85],
            |h| (RATE_TEST * h).exp() - 1.0,
        );
    }

    /// Order measured on a genuine ODE, where the predicted state feeds back into the
    /// right-hand side and the corrector has to iterate.
    #[test]
    fn convergence_order_ode() {
        let mut y0 = Vector6::zeros();
        y0[0] = 1.0;
        assert_order_16(
            "manufactured",
            &manufactured,
            y0,
            &[3.0, 2.5, 2.0, 1.6, 1.3, 1.05, 0.85],
            |h| (RATE_TEST * h).exp(),
        );
    }

    /// The corrector is a fixed-point iteration, not a Newton solve, so it converges
    /// only while `h |dF/dy|` is below order 1.  For `y' = y` that caps the step near
    /// `h = 1` regardless of how much error would be acceptable there.  This is a
    /// property of the scheme worth pinning: on a stiff problem the step is limited by
    /// the corrector, not by accuracy, and the driver's response is to shrink and retry.
    #[test]
    fn corrector_contraction_limit() {
        fn stiff(
            _time: Time<TDB>,
            state: &Vector6<f64>,
            meta: &mut Evals,
            _exact: bool,
        ) -> KeteResult<Vector6<f64>> {
            meta.0 += 1;
            Ok(*state)
        }

        let mut y0 = Vector6::zeros();
        y0[0] = 1.0;
        let converges = |h: f64| {
            RadauFirstOrder::new(&stiff, y0, 0.0.into(), h.into(), Evals::default())
                .is_ok_and(|mut i| i.step(h).is_ok())
        };

        // `dF/dy = 1` here, so h is exactly the contraction parameter.  Find where the
        // corrector gives up rather than asserting a value: the threshold is set by how
        // far SWEEP_TOL is from the initial b6 and by MAX_SWEEPS, not by contraction
        // alone, so it is well below the naive `h |dF/dy| < 1`.
        let limit = (1..=40)
            .map(|i| f64::from(i) * 0.05)
            .take_while(|h| converges(*h))
            .last()
            .unwrap_or(0.0);
        println!("corrector converges up to h |dF/dy| = {limit:.2}");
        assert!(
            (0.05..1.0).contains(&limit),
            "corrector contraction limit {limit:.2} is outside the expected band",
        );

        // Past it, the adaptive driver must still get the right answer by shrinking.
        let (got, ev) =
            RadauFirstOrder::integrate(&stiff, y0, 0.0.into(), 5.0.into(), Evals::default(), None)
                .unwrap();
        let err = ((got[0] - 5.0_f64.exp()) / 5.0_f64.exp()).abs();
        println!(
            "stiff driver over 5 e-foldings: rel err {err:9.3e}  evals {}",
            ev.0
        );
        assert!(err < 1e-13, "driver failed to recover: {err:e}");
    }

    /// One full period of a Kepler orbit returns to its starting state.  The initial
    /// condition is its own exact reference, so this carries no reference-integrator
    /// error.
    ///
    /// The evaluation budget guards the predictor. Without the `b` rescaling of
    /// [`RadauFirstOrder::predict`] each of these costs several times as much, so a budget
    /// set well above the working cost catches the predictor silently ceasing to work
    /// without being brittle about small changes.
    #[test]
    fn kepler_period_return() {
        // (eccentricity, evaluation budget)
        let cases = [(0.0, 6_000), (0.2, 10_000), (0.5, 15_000), (0.9, 23_000)];
        for (ecc, budget) in cases {
            let semi_major: f64 = 1.0;
            let period = std::f64::consts::TAU * (semi_major.powi(3) / GMS).sqrt();
            // Start at aphelion, the smoothest point of the orbit.
            let y0 = conic_state(semi_major * (1.0 - ecc * ecc), ecc, std::f64::consts::PI);

            let (got, evals) = RadauFirstOrder::integrate(
                &two_body,
                y0,
                0.0.into(),
                period.into(),
                Evals::default(),
                None,
            )
            .unwrap();

            let pos_err = (got.fixed_rows::<3>(0) - y0.fixed_rows::<3>(0)).norm()
                / y0.fixed_rows::<3>(0).norm();
            let vel_err = (got.fixed_rows::<3>(3) - y0.fixed_rows::<3>(3)).norm()
                / y0.fixed_rows::<3>(3).norm();
            println!(
                "e {ecc:4.2}   pos {pos_err:9.3e}   vel {vel_err:9.3e}   evals {}",
                evals.0
            );
            assert!(pos_err < 1e-11, "e={ecc}: position error {pos_err:e}");
            assert!(vel_err < 1e-11, "e={ecc}: velocity error {vel_err:e}");
            assert!(
                evals.0 < budget,
                "e={ecc}: {} evaluations exceeds the budget of {budget}, which suggests \
                 the predictor is no longer helping",
                evals.0,
            );
        }
    }

    /// Conserved quantities of the two-body problem over 100 orbits.  The method is not
    /// symplectic, so drift is expected to be secular; this pins how large it is.
    #[test]
    fn two_body_invariants() {
        let period = std::f64::consts::TAU * GMS.sqrt().recip();
        let y0 = conic_state(1.0 - 0.3 * 0.3, 0.3, std::f64::consts::PI);

        let energy = |y: &Vector6<f64>| {
            let (r, v) = (y.fixed_rows::<3>(0), y.fixed_rows::<3>(3));
            v.norm_squared() / 2.0 - GMS / r.norm()
        };
        let ang_mom = |y: &Vector6<f64>| {
            let (r, v) = (y.fixed_rows::<3>(0), y.fixed_rows::<3>(3));
            r.cross(&v).norm()
        };

        let (got, _) = RadauFirstOrder::integrate(
            &two_body,
            y0,
            0.0.into(),
            (100.0 * period).into(),
            Evals::default(),
            None,
        )
        .unwrap();

        let d_energy = ((energy(&got) - energy(&y0)) / energy(&y0)).abs();
        let d_ang_mom = ((ang_mom(&got) - ang_mom(&y0)) / ang_mom(&y0)).abs();
        println!("100 orbits: dE/E {d_energy:9.3e}   dL/L {d_ang_mom:9.3e}");
        assert!(d_energy < 1e-12, "energy drift {d_energy:e}");
        assert!(d_ang_mom < 1e-12, "angular momentum drift {d_ang_mom:e}");
    }

    /// Cross-check against the second-order integrator on the same physical problem.
    /// The two share no code below the Gauss-Radau constants.
    #[test]
    fn agrees_with_second_order() {
        let y0 = conic_state(1.0 - 0.25 * 0.25, 0.25, 0.7);
        let (first, _) = RadauFirstOrder::integrate(
            &two_body,
            y0,
            0.0.into(),
            900.0.into(),
            Evals::default(),
            None,
        )
        .unwrap();

        let (pos, vel, _) = RadauIntegrator::integrate(
            &central_accel,
            Vector3::from(y0.fixed_rows::<3>(0)),
            Vector3::from(y0.fixed_rows::<3>(3)),
            0.0.into(),
            900.0.into(),
            CentralAccelMeta::default(),
            Some(3),
        )
        .unwrap();

        let pos_err = (first.fixed_rows::<3>(0) - pos).norm() / pos.norm();
        let vel_err = (first.fixed_rows::<3>(3) - vel).norm() / vel.norm();
        println!("vs second order: pos {pos_err:9.3e}  vel {vel_err:9.3e}");
        assert!(pos_err < 1e-11 && vel_err < 1e-11);
    }

    /// A slow component of magnitude 1 alongside a fast component of magnitude 1e-8.
    ///
    /// The fast component's right-hand side is `5e-8`, far below the `1e-6` absolute
    /// floor the second-order integrator adds to its denominator, so under that rule it
    /// would be invisible to the step-size controller and the step would be set by the
    /// slow component alone.  The per-component scale of this integrator has to resolve
    /// both.  This is the discriminating test for that design choice.
    #[test]
    fn mixed_scale_error_control() {
        const OMEGA: f64 = 5.0;
        const AMP: f64 = 1e-8;

        fn mixed(
            _time: Time<TDB>,
            state: &Vector6<f64>,
            meta: &mut Evals,
            _exact: bool,
        ) -> KeteResult<Vector6<f64>> {
            meta.0 += 1;
            let mut out = Vector6::zeros();
            out[0] = -0.01 * state[0];
            out[1] = OMEGA * state[2];
            out[2] = -OMEGA * state[1];
            Ok(out)
        }

        let mut y0 = Vector6::zeros();
        y0[0] = 1.0;
        y0[1] = AMP;

        let t_end = 10.0;
        let (got, evals) = RadauFirstOrder::integrate(
            &mixed,
            y0,
            0.0.into(),
            t_end.into(),
            Evals::default(),
            None,
        )
        .unwrap();

        let slow_err = ((got[0] - (-0.01 * t_end).exp()) / (-0.01 * t_end).exp()).abs();
        let fast_err = (got[1] - AMP * (OMEGA * t_end).cos()).abs() / AMP;
        let fast_err2 = (got[2] + AMP * (OMEGA * t_end).sin()).abs() / AMP;
        println!(
            "slow {slow_err:9.3e}   fast {fast_err:9.3e} {fast_err2:9.3e}   evals {}",
            evals.0
        );
        assert!(slow_err < 1e-12, "slow component error {slow_err:e}");
        assert!(
            fast_err < 1e-10 && fast_err2 < 1e-10,
            "fast small-amplitude component under-resolved: {fast_err:e} {fast_err2:e}",
        );
    }

    /// Evaluation counts for the two order forms as perihelion sharpens, at fixed period so
    /// the arc is one full orbit and the initial state is an exact reference.
    ///
    /// `cargo test -p kete_core --release --lib radau_first_order -- --ignored --nocapture`
    #[test]
    #[ignore = "reports a table, run explicitly"]
    fn order_form_cost_vs_eccentricity() {
        let semi_major: f64 = 1.0;
        let period = std::f64::consts::TAU * (semi_major.powi(3) / GMS).sqrt();
        println!("  e      q(AU)   2nd order D=3         1st order D=6");
        for ecc in [0.0, 0.5, 0.9, 0.99, 0.999] {
            let y0 = conic_state(semi_major * (1.0 - ecc * ecc), ecc, std::f64::consts::PI);
            let pos0 = Vector3::from(y0.fixed_rows::<3>(0));
            let vel0 = Vector3::from(y0.fixed_rows::<3>(3));

            let second = RadauIntegrator::integrate(
                &central_accel,
                pos0,
                vel0,
                0.0.into(),
                period.into(),
                CentralAccelMeta::default(),
                Some(3),
            );
            let first = RadauFirstOrder::integrate(
                &two_body,
                y0,
                0.0.into(),
                period.into(),
                Evals::default(),
                None,
            );

            let fmt_second = second.map_or_else(
                |_| "FAILED".to_string(),
                |(p, _, m)| format!("{:6} ev, {:8.2e}", m.eval_count, (p - pos0).norm()),
            );
            let fmt_first = first.map_or_else(
                |_| "FAILED".to_string(),
                |(y, m)| {
                    format!(
                        "{:6} ev, {:8.2e}",
                        m.0,
                        (y.fixed_rows::<3>(0) - pos0).norm()
                    )
                },
            );
            println!(
                "{ecc:6.3} {:8.5}   {fmt_second}   {fmt_first}",
                semi_major * (1.0 - ecc)
            );
        }
    }
}
