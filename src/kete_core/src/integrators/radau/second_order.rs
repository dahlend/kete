// SPDX-FileCopyrightText: 2026 Dar Dahlen
// SPDX-FileCopyrightText: 2025 California Institute of Technology
// SPDX-License-Identifier: BSD-3-Clause

//! Gauss-Radau Spacing Numerical Integrator
//! This solves a second-order initial value problem.
use crate::errors::Error;
use crate::integrators::util::SecondOrderODE;
use crate::prelude::KeteResult;
use crate::time::{TDB, Time};
use itertools::izip;
use nalgebra::Matrix;
use nalgebra::allocator::Allocator;
use nalgebra::{DefaultAllocator, Dim, OMatrix, OVector, RowSVector, U1, U7};

use super::{
    BPredictor, C_MAT, EPSILON, GAUSS_RADAU_SPACINGS, MIN_RATIO, MIN_STEP, U_POW_TABLE, U_VEC,
};

/// Integrator will return a result of this type.
type RadauResult<MType, D> = KeteResult<(OVector<f64, D>, OVector<f64, D>, MType)>;

/// Number of values stored per component per step: the state, its first and second
/// derivatives at the start of the step, and the seven `b` coefficients.
const DENSE_STRIDE: usize = 10;

/// Margin, in days, by which a query may fall outside the integrated span and still be
/// evaluated. The integrator ends within about 1e-12 day of its target, so this keeps
/// the exact target time queryable.
const DENSE_EDGE_TOL: f64 = 1e-9;

/// Dense output of one [`RadauIntegrator::integrate`] call: every accepted step, which
/// gives the state and its derivative at any time in the integrated span from the
/// integrator's own polynomial, without integrating again.
///
/// Over a step starting at `epoch` with signed length `h`, for
/// `s = (t - epoch) / h` in `[0, 1]`:
///
/// ```text
/// x(t) = x0 + s h v0 + (s h)^2 (a0 / 2 + sum_k b_k s^(k+1) / ((k + 2)(k + 3)))
/// v(t) = v0 + s h (a0 + sum_k b_k s^(k+1) / (k + 2))
/// ```
///
/// with `x0`, `v0`, `a0` the state, its derivative and its second derivative at
/// `epoch`, and `b` the step's seven Gauss-Radau coefficients.
///
/// If the integration fails part way, for example on an impact, the steps accepted
/// before the failure are kept, so the trajectory runs up to it.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct RadauDense {
    /// Number of components of the integrated state.
    n_comp: usize,
    /// Start of each step, in integration order.
    epochs: Vec<Time<TDB>>,
    /// Signed length of each step in days.
    step_sizes: Vec<f64>,
    /// For each step, for each component: `[x0, v0, a0, b0..b6]`.
    coeffs: Vec<f64>,
}

impl RadauDense {
    /// An empty dense output, to pass to [`RadauIntegrator::integrate`].
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of steps held.
    #[must_use]
    pub fn n_steps(&self) -> usize {
        self.epochs.len()
    }

    /// Where the integration started, or `None` if empty.
    #[must_use]
    pub fn start(&self) -> Option<Time<TDB>> {
        self.epochs.first().copied()
    }

    /// Where the integration ended, or `None` if empty.
    #[must_use]
    pub fn end(&self) -> Option<Time<TDB>> {
        Some(*self.epochs.last()? + *self.step_sizes.last()?)
    }

    /// State and its derivative at `time`.
    ///
    /// # Errors
    /// `Error::Bounds` if `time` is outside the integrated span.
    pub fn evaluate(&self, time: Time<TDB>) -> KeteResult<(Vec<f64>, Vec<f64>)> {
        let out_of_span = || {
            Error::Bounds(format!(
                "JD {} is outside the dense output, which covers JD {} to {}.",
                time.jd(),
                self.start().map_or(f64::NAN, |t| t.jd()),
                self.end().map_or(f64::NAN, |t| t.jd()),
            ))
        };
        let (start, end) = self.start().zip(self.end()).ok_or_else(out_of_span)?;
        let backward = self.step_sizes[0] < 0.0;
        let (lo, hi) = if backward { (end, start) } else { (start, end) };
        if (lo - time).elapsed > DENSE_EDGE_TOL || (time - hi).elapsed > DENSE_EDGE_TOL {
            Err(out_of_span())?;
        }
        // Steps whose start has not passed `time`, in the direction of integration.
        let n_before = if backward {
            self.epochs.partition_point(|epoch| *epoch >= time)
        } else {
            self.epochs.partition_point(|epoch| *epoch <= time)
        };
        Ok(self.evaluate_step(n_before.saturating_sub(1), time))
    }

    /// Append an accepted step: its start, signed length, the state and its first and
    /// second derivatives at the start, and its `b` coefficients.
    fn push<D: Dim>(
        &mut self,
        epoch: Time<TDB>,
        step_size: f64,
        state: &OVector<f64, D>,
        state_der: &OVector<f64, D>,
        state_der_der: &OVector<f64, D>,
        b: &OMatrix<f64, D, U7>,
    ) where
        DefaultAllocator: Allocator<D, U1> + Allocator<D, U7>,
    {
        self.n_comp = state.len();
        self.epochs.push(epoch);
        self.step_sizes.push(step_size);
        self.coeffs.reserve(self.n_comp * DENSE_STRIDE);
        for idx in 0..self.n_comp {
            self.coeffs
                .extend_from_slice(&[state[idx], state_der[idx], state_der_der[idx]]);
            self.coeffs.extend((0..7).map(|k| b[(idx, k)]));
        }
    }

    /// Evaluate the polynomial of step `step` at `time`.
    fn evaluate_step(&self, step: usize, time: Time<TDB>) -> (Vec<f64>, Vec<f64>) {
        let step_size = self.step_sizes[step];
        let s = (time - self.epochs[step]).elapsed / step_size;
        let h1 = s * step_size;
        // Powers s^(k+1) folded with the W and U weights, shared by every component.
        let (w_vec, u_vec) = (&*W_VEC, &*U_VEC);
        let mut w = [0.0; 7];
        let mut u = [0.0; 7];
        let mut s_pow = s;
        for k in 0..7 {
            w[k] = w_vec[k] * s_pow;
            u[k] = u_vec[k] * s_pow;
            s_pow *= s;
        }
        let base = step * self.n_comp * DENSE_STRIDE;
        self.coeffs[base..base + self.n_comp * DENSE_STRIDE]
            .as_chunks::<DENSE_STRIDE>()
            .0
            .iter()
            .map(|c| {
                let (mut bw, mut bu) = (0.0, 0.0);
                for k in 0..7 {
                    bw += c[3 + k] * w[k];
                    bu += c[3 + k] * u[k];
                }
                (
                    c[0] + h1 * c[1] + h1 * h1 * (c[2] / 2.0 + bw),
                    c[1] + h1 * (c[2] + bu),
                )
            })
            .unzip()
    }
}

// initialize W
static W_VEC: std::sync::LazyLock<RowSVector<f64, 7>> = std::sync::LazyLock::new(|| {
    let mut w = RowSVector::<f64, 7>::zeros();
    for (idx, e) in w.iter_mut().enumerate() {
        *e = (((idx + 2) * (idx + 3)) as f64).recip();
    }
    w
});

// Precomputed w_pow and u_pow tables for each Gauss-Radau substep.
// W_POW_TABLE[j][k] = h_{j+1}^{k+1} * W_VEC[k]
// where h_{j+1} = GAUSS_RADAU_SPACINGS[j+1].
// Eliminates per-iteration powi calls and SVector construction.
static W_POW_TABLE: std::sync::LazyLock<[RowSVector<f64, 7>; 7]> = std::sync::LazyLock::new(|| {
    let w = &*W_VEC;
    let mut table = [RowSVector::<f64, 7>::zeros(); 7];
    for (j, h) in GAUSS_RADAU_SPACINGS.iter().enumerate().skip(1) {
        let mut hp = *h;
        for k in 0..7 {
            table[j - 1][k] = hp * w[k];
            hp *= h;
        }
    }
    table
});

/// Gauss-Radau Spacing Numerical Integrator
/// This solves a second-order initial value problem.
///
/// References:
/// E. Everhart (1985), 'An efficient integrator that uses Gauss-Radau spacings',
/// A. Carusi and G. B. Valsecchi (eds.),
/// Dynamics of Comets: Their Origin and Evolution (proceedings),
/// Astrophysics and Space Science Library, vol. 115, D. Reidel Publishing Company
///
/// E. Everhart (1974), 'Implicit single-sequence methods for integrating orbits',
/// Celestial Mechanics, vol. 10, no. 1, pp. 35-55
///
/// This uses the 15th-order integrator as seen in the original RADAU code, however
/// many changes and improvements have been made. Some variable names have been chosen
/// to match the original Fortran implementation.
///
/// Each step starts from `b` extrapolated from the previous accepted step, see
/// `BPredictor`.
///
/// Compensated (Kahan) summation is used for the state update to reduce
/// roundoff accumulation from O(N) to approximately O(sqrt(N)).
///
/// # Error control
///
/// Convergence and step size are driven by `max(|b6_i| / scale_i)` over the first
/// `control_dim` components, where `scale_i` is that component's own magnitude, taken as
/// the larger of the two right-hand side evaluations bracketing the step. The criterion
/// is therefore relative and dimensionless.
///
/// The scale is purely multiplicative, with no additive floor. An additive term would be
/// absolute, in AU/day^2, and the solar monopole falls below any such floor at a finite
/// heliocentric distance - `sqrt(GMS / 1e-6) = 17.2 AU` for a floor of `1e-6` - past which
/// the control would stop being relative and the step would grow unchecked. See
/// `accuracy_is_independent_of_heliocentric_distance`.
#[allow(missing_debug_implementations, reason = "No debug impl needed")]
pub struct RadauIntegrator<'a, MType, D: Dim>
where
    DefaultAllocator: Allocator<D, U1> + Allocator<D, U7>,
{
    func: SecondOrderODE<'a, MType, D>,
    metadata: MType,

    final_time: Time<TDB>,

    cur_time: Time<TDB>,
    cur_state: OVector<f64, D>,
    cur_state_der: OVector<f64, D>,
    cur_state_der_der: OVector<f64, D>,

    cur_b: OMatrix<f64, D, U7>,
    g_scratch: OMatrix<f64, D, U7>,
    predictor: BPredictor<D>,

    state_scratch: OVector<f64, D>,
    state_der_scratch: OVector<f64, D>,
    b_scratch: OVector<f64, D>,
    eval_scratch: OVector<f64, D>,

    /// State, derivative, and evaluation at each node from the most recent evaluation
    /// in the current step attempt, so a node whose state has not changed is not
    /// evaluated again.
    node_state: OMatrix<f64, D, U7>,
    node_state_der: OMatrix<f64, D, U7>,
    node_eval: OMatrix<f64, D, U7>,
    node_valid: [bool; 7],

    /// Number of leading dimensions used for convergence and step-size control.
    /// Defaults to the full state dimension `D`.  For variational / STM
    /// propagation set this to 3 (physical accelerations only) so that the
    /// large STM elements do not artificially shrink step-size.
    control_dim: usize,

    // Kahan compensated summation error accumulators.
    comp_state: OVector<f64, D>,
    comp_state_der: OVector<f64, D>,
    comp_time: f64,
}

impl<'a, MType, D: Dim> RadauIntegrator<'a, MType, D>
where
    DefaultAllocator: Allocator<D, U1> + Allocator<D, U7>,
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
            Err(Error::ValueError(
                "Input vectors must be the same length".into(),
            ))?;
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
            cur_b: Matrix::zeros_generic(dim, U7),
            g_scratch: Matrix::zeros_generic(dim, U7),
            predictor: BPredictor::new(dim),
            b_scratch: Matrix::zeros_generic(dim, U1),
            state_scratch: Matrix::zeros_generic(dim, U1),
            state_der_scratch: Matrix::zeros_generic(dim, U1),
            eval_scratch: Matrix::zeros_generic(dim, U1),
            node_state: Matrix::zeros_generic(dim, U7),
            node_state_der: Matrix::zeros_generic(dim, U7),
            node_eval: Matrix::zeros_generic(dim, U7),
            node_valid: [false; 7],
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

    /// Integrate the functions from the initial time to the final time.
    ///
    /// `dense`, when given, is replaced by every accepted step of this integration, so
    /// that the solution can be evaluated at any time in the span afterwards (see
    /// [`RadauDense`]). If the integration fails part way, it keeps the steps accepted
    /// before the failure.
    ///
    /// # Errors
    /// Integration may fail for a number of reasons, either the function fails, or
    /// convergence of the integrator fails.
    pub fn integrate(
        func: SecondOrderODE<'a, MType, D>,
        state_init: OVector<f64, D>,
        state_der_init: OVector<f64, D>,
        time_init: Time<TDB>,
        final_time: Time<TDB>,
        metadata: MType,
        control_dim: Option<usize>,
        mut dense: Option<&mut RadauDense>,
    ) -> RadauResult<MType, D> {
        if let Some(dense) = dense.as_deref_mut() {
            *dense = RadauDense::default();
        }
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
        // Allow callers to control convergence using a subset of dimensions.
        integrator.control_dim = control_dim.unwrap_or(integrator.control_dim);
        if integrator.control_dim > integrator.cur_state.len() {
            Err(Error::ValueError(format!(
                "control_dim ({}) exceeds state dimension ({})",
                integrator.control_dim,
                integrator.cur_state.len(),
            )))?;
        }

        let mut next_step_size = integrator.initial_step_size()?;
        let mut first_step = true;

        // The last step is sized to land on `final_time`, which `Time` resolves to
        // about 1e-16 day at any epoch, so the loop ends within this tolerance of it.
        let convergence_tol = 1e-12;

        let mut step_failures = 0;
        loop {
            // Defensive non-finite check: if `next_step_size` is NaN/Inf, the
            // comparison `(cur_time - final_time).abs() <= NaN.abs()` is false
            // (NaN propagates), so the early-exit and step-size-floor branches
            // below would never fire and the loop would spin forever.  Fail
            // fast instead.  Non-finite state at this point typically means a
            // particle landed at a gravitational singularity (rel_pos = 0)
            // during the previous step, producing 0/0 = NaN in the
            // acceleration.
            if !next_step_size.is_finite() {
                return Err(Error::Convergence(
                    "Radau produced non-finite step size (state likely diverged).".into(),
                ));
            }
            if (integrator.cur_time - integrator.final_time).elapsed.abs() <= next_step_size.abs() {
                next_step_size = (integrator.final_time - integrator.cur_time).elapsed;
            }
            match integrator.step(next_step_size, first_step, dense.as_deref_mut()) {
                Ok(StepOutcome::Retry(s)) => {
                    // The first step came from an estimate rather than from the
                    // controller, and its error was above target: redo it once at the size
                    // the controller asks for. That size is then treated like every later
                    // one, since the error estimate can sit above target at any step size
                    // when a component's acceleration is near zero.
                    next_step_size = s;
                    first_step = false;
                }
                Ok(StepOutcome::Accepted(s)) => {
                    first_step = false;
                    next_step_size = s;
                    if (integrator.cur_time - integrator.final_time).elapsed.abs() < convergence_tol
                    {
                        // Taylor the state onto the target epoch. Returning it at
                        // whatever sub-tolerance time the loop stopped at costs a meter
                        // of along-track position.
                        let dt = (integrator.final_time - integrator.cur_time).elapsed
                            + integrator.comp_time;
                        for idx in 0..integrator.cur_state.len() {
                            let der = integrator.cur_state_der[idx];
                            let der_der = integrator.cur_state_der_der[idx];
                            integrator.cur_state[idx] += der * dt + 0.5 * der_der * dt * dt;
                            integrator.cur_state_der[idx] += der_der * dt;
                        }
                        return Ok((
                            integrator.cur_state,
                            integrator.cur_state_der,
                            integrator.metadata,
                        ));
                    }
                    step_failures = 0;
                }
                // Only a failed step is retried at a smaller size. Any other error comes
                // from the force model or the ephemeris, where a smaller step gives the
                // same error, so it is returned as is.
                Err(error) => match error {
                    Error::Convergence(_) => {
                        step_failures += 1;
                        next_step_size *= 0.7;
                        if step_failures > 10 {
                            Err(Error::Convergence("Radau failed to converge.".into()))?;
                        }
                    }
                    Error::Bounds(_)
                    | Error::Impact(_, _)
                    | Error::ValueError(_)
                    | Error::IOError(_)
                    | Error::LockFailed => Err(error)?,
                },
            }
            if next_step_size.abs() < MIN_STEP {
                next_step_size = MIN_STEP.copysign(next_step_size);
            }
        }
    }

    /// Size of the first step, signed toward `final_time`.
    ///
    /// The controller holds `|b_6| / |a|` near `EPSILON`, and `b_6` scales as `(h / tau)^7`
    /// for a problem that changes on a timescale `tau`, so a step the controller would
    /// choose is about `EPSILON^(1/7) tau`. `tau` is taken as the shorter of
    ///
    /// ```text
    /// |v| / |a|        how long the velocity takes to change by itself
    /// |a| / |da/dt|    how long the acceleration takes to change by itself
    /// ```
    ///
    /// over the first `control_dim` components, with `da/dt` from one extra evaluation
    /// a small fraction of `|v| / |a|` ahead. Both are ratios of the state's own
    /// quantities, so the estimate does not depend on the units. The step is half the
    /// estimate: the first step starts from `b = 0` rather than from a prediction, and a
    /// first step whose error is still above target is redone once (see [`Self::step`]).
    ///
    /// Falls back to 0.1 when `|v|` or `|a|` is zero, where neither timescale exists.
    fn initial_step_size(&mut self) -> KeteResult<f64> {
        let dir = (self.final_time - self.cur_time).elapsed.signum();
        let cd = self.control_dim;
        let v_norm = self.cur_state_der.rows(0, cd).norm();
        let a_norm = self.cur_state_der_der.rows(0, cd).norm();
        if v_norm == 0.0 || a_norm == 0.0 {
            return Ok(0.1 * dir);
        }
        let tau_v = v_norm / a_norm;

        let dt = 1e-3 * tau_v;
        let pos = &self.cur_state
            + &self.cur_state_der * (dir * dt)
            + &self.cur_state_der_der * (0.5 * dt * dt);
        let vel = &self.cur_state_der + &self.cur_state_der_der * (dir * dt);
        let accel = (self.func)(
            self.cur_time + dir * dt,
            &pos,
            &vel,
            &mut self.metadata,
            false,
        )?;
        let jerk = (accel.rows(0, cd) - self.cur_state_der_der.rows(0, cd)).norm() / dt;
        let tau_a = if jerk > 0.0 {
            a_norm / jerk
        } else {
            f64::INFINITY
        };

        let h0 = 0.5 * EPSILON.powf(1.0 / 7.0) * tau_v.min(tau_a);
        Ok(h0.max(MIN_STEP) * dir)
    }

    /// Attempt a single integration step of size `step_size`.
    ///
    /// Returns the recommended next step size on success.  Failure can occur
    /// if the step size is too large for convergence, or if the ODE function
    /// itself returns an error.
    ///
    /// When `first_step` is set, a converged step whose error estimate is above target is
    /// not accepted: the state is left unchanged and the size the controller asks for is
    /// returned as [`StepOutcome::Retry`]. The caller redoes the step once at that size,
    /// and from then on every step size comes from the controller and is accepted once
    /// converged.
    ///
    /// A node whose state and derivative are bit-identical to its previous evaluation in
    /// this step attempt reuses that evaluation instead of calling the function again,
    /// which gives the same result.
    fn step(
        &mut self,
        step_size: f64,
        first_step: bool,
        dense: Option<&mut RadauDense>,
    ) -> KeteResult<StepOutcome> {
        self.predictor.predict(step_size, &mut self.cur_b);
        self.g_scratch.fill(0.0);
        self.node_valid = [false; 7];
        self.state_scratch.fill(0.0);
        self.state_der_scratch.fill(0.0);
        self.eval_scratch.set_column(0, &self.cur_state_der_der);

        for _ in 0..10 {
            self.b_scratch.set_column(0, &self.cur_b.column(6));
            // Calculate b and g
            #[allow(clippy::cast_possible_wrap, reason = "idx does not exceed 8")]
            for (idj, gauss_radau_frac) in GAUSS_RADAU_SPACINGS.iter().enumerate().skip(1) {
                // the sample point at the Guass-Radau spacings.
                // Update each parameter using the current B as a guess to estimate the
                // state of the integrator at the current time + the Gauss-Radau spacing.

                let w_pow = &W_POW_TABLE[idj - 1];
                let u_pow = &U_POW_TABLE[idj - 1];
                let h1 = gauss_radau_frac * step_size;
                let h2 = h1 * h1;

                izip!(
                    self.state_scratch.iter_mut(),
                    self.cur_state.iter(),
                    self.cur_state_der.iter(),
                    self.cur_state_der_der.iter(),
                    self.cur_b.row_iter(),
                )
                .for_each(|(out, state, der, derder, b)| {
                    *out = state + h1 * der + h2 * (derder / 2.0 + b.dot(w_pow));
                });

                izip!(
                    self.state_der_scratch.iter_mut(),
                    self.cur_state_der.iter(),
                    self.cur_state_der_der.iter(),
                    self.cur_b.row_iter(),
                )
                .for_each(|(out, der, derder, b)| {
                    *out = der + h1 * (derder + b.dot(u_pow));
                });

                // Evaluate the function at this new intermediate state, unless the
                // node has not moved since its last evaluation in this step attempt.
                let node = idj - 1;
                if self.node_valid[node]
                    && self.node_state.column(node) == self.state_scratch
                    && self.node_state_der.column(node) == self.state_der_scratch
                {
                    self.eval_scratch
                        .set_column(0, &self.node_eval.column(node));
                } else {
                    self.eval_scratch.set_column(
                        0,
                        &(self.func)(
                            self.cur_time + gauss_radau_frac * step_size,
                            &self.state_scratch,
                            &self.state_der_scratch,
                            &mut self.metadata,
                            false,
                        )?,
                    );
                    self.node_state.set_column(node, &self.state_scratch);
                    self.node_state_der
                        .set_column(node, &self.state_der_scratch);
                    self.node_eval.set_column(node, &self.eval_scratch);
                    self.node_valid[node] = true;
                }

                let diff = &self.eval_scratch - &self.cur_state_der_der;

                // Use the result of that evaluation to update the current G and B
                // matrices for the next gauss spacing.

                // This is equivalent to equation (4) in everhart's paper.
                // The lookup tables and switch statements he uses were performing
                // ~100x slower than this implementation.
                self.g_scratch.set_column(node, &{
                    let mut gk = diff / *gauss_radau_frac;

                    for (idz, gr_step) in GAUSS_RADAU_SPACINGS.iter().enumerate().take(idj).skip(1)
                    {
                        gk = (gk - self.g_scratch.column(idz - 1)) / (gauss_radau_frac - gr_step);
                    }
                    gk
                });
            }

            // Update B from G via the C matrix.
            self.g_scratch.mul_to(&C_MAT, &mut self.cur_b);

            // Convergence and step-size control use only the first
            // `control_dim` components.  For variational propagation this
            // restricts the norms to the physical accelerations, preventing
            // large STM elements from artificially shrinking the step.
            let cd = self.control_dim;

            // Per-component scale, taken from the two right-hand side evaluations that
            // bracket this step: `cur_state_der_der` at its start and `eval_scratch` at
            // the last Gauss-Radau node.  Dividing by the component's own magnitude
            // keeps the criterion relative and dimensionless whatever the units.
            //
            // The scale carries no additive floor.  Such a floor is absolute, in
            // AU/day^2, and is negligible for a heliocentric object inside a few AU but
            // exceeds the solar monopole itself beyond `sqrt(GMS/f)` - 17.2 AU for
            // `f = 1e-6` - at which point the control stops being relative and the step
            // grows without bound.  See
            // `accuracy_is_independent_of_heliocentric_distance`.
            //
            // A component whose right-hand side vanishes at both ends contributes
            // nothing: its divided differences, and so its `b`, are exactly zero, and
            // `0 / MIN_POSITIVE` is zero rather than a NaN.
            //
            // Both ratios are accumulated in one pass, before the state update, since
            // the update overwrites `cur_state_der_der`.  Everything here is scalar so
            // no per-sweep temporaries are allocated.
            let mut sweep_ratio = 0.0_f64;
            let mut error_ratio = 0.0_f64;
            {
                let b6 = self.cur_b.column(6);
                for idx in 0..cd {
                    let scale = self.eval_scratch[idx]
                        .abs()
                        .max(self.cur_state_der_der[idx].abs())
                        .max(f64::MIN_POSITIVE);
                    sweep_ratio = sweep_ratio.max((b6[idx] - self.b_scratch[idx]).abs() / scale);
                    error_ratio = error_ratio.max(b6[idx].abs() / scale);
                }
            }

            // This is using the convergence criterion as defined in
            // https://arxiv.org/pdf/1409.4779.pdf  equation (8)
            if sweep_ratio < 1e-14 {
                if first_step && error_ratio > EPSILON {
                    return Ok(StepOutcome::Retry(
                        step_size * 0.9 * (EPSILON / error_ratio).powf(1.0 / 7.0),
                    ));
                }
                if let Some(dense) = dense {
                    dense.push(
                        self.cur_time,
                        step_size,
                        &self.cur_state,
                        &self.cur_state_der,
                        &self.cur_state_der_der,
                        &self.cur_b,
                    );
                }
                let ss = step_size * step_size;
                for idx in 0..self.cur_state.len() {
                    unsafe {
                        let delta_state = self.cur_state_der.get_unchecked(idx) * step_size
                            + ss * (self.cur_state_der_der.get_unchecked(idx) * 0.5
                                + self.cur_b.row(idx).dot(&W_VEC));
                        let y_pos = delta_state - self.comp_state[idx];
                        let t_pos = self.cur_state[idx] + y_pos;
                        self.comp_state[idx] = (t_pos - self.cur_state[idx]) - y_pos;
                        self.cur_state[idx] = t_pos;

                        let delta_der = step_size
                            * (self.cur_state_der_der.get_unchecked(idx)
                                + self.cur_b.row(idx).dot(&U_VEC));
                        let y_vel = delta_der - self.comp_state_der[idx];
                        let t_vel = self.cur_state_der[idx] + y_vel;
                        self.comp_state_der[idx] = (t_vel - self.cur_state_der[idx]) - y_vel;
                        self.cur_state_der[idx] = t_vel;
                    }
                }
                let y_t = step_size - self.comp_time;
                let t_t = self.cur_time + y_t;
                self.comp_time = (t_t - self.cur_time).elapsed - y_t;
                self.cur_time = t_t;
                self.cur_state_der_der = (self.func)(
                    self.cur_time,
                    &self.cur_state,
                    &self.cur_state_der,
                    &mut self.metadata,
                    true,
                )?;
                self.predictor.accept(step_size, &self.cur_b);
                // Step-size controller: the component-wise ratio computed above,
                // max(|b6_i| / scale_i), lets the worst-resolved component drive the
                // step size.
                return Ok(StepOutcome::Accepted(
                    step_size
                        * (EPSILON / error_ratio)
                            .powf(1.0 / 7.0)
                            .clamp(MIN_RATIO, MIN_RATIO.recip()),
                ));
            }
        }
        Err(Error::Convergence("Radau step failed to converge".into()))?
    }
}

/// Result of one converged step attempt.
enum StepOutcome {
    /// The step was taken; holds the recommended next step size.
    Accepted(f64),
    /// The step was not taken; holds the size to retry it at.
    Retry(f64),
}

#[cfg(test)]
mod tests {
    use nalgebra::Vector3;

    use super::*;
    use crate::integrators::stress_tests::{CentralAccelMeta, central_accel};

    /// On a circular orbit both timescales of the first-step estimate are `1 / n`, so the
    /// first step is `0.5 EPSILON^(1/7) / n` at any radius.
    #[test]
    fn initial_step_follows_the_orbital_timescale() {
        use crate::constants::GMS;
        for radius in [1.0, 30.0] {
            let speed = (GMS / radius).sqrt();
            let mut integrator = RadauIntegrator::new(
                &central_accel,
                Vector3::new(radius, 0.0, 0.0),
                Vector3::new(0.0, speed, 0.0),
                0.0.into(),
                1000.0.into(),
                CentralAccelMeta::default(),
            )
            .unwrap();
            let mean_motion = speed / radius;
            let expected = 0.5 * EPSILON.powf(1.0 / 7.0) / mean_motion;
            let h0 = integrator.initial_step_size().unwrap();
            assert!(
                (h0 / expected - 1.0).abs() < 1e-2,
                "radius {radius}: first step {h0} differs from {expected}"
            );
        }
    }

    /// `x'' = 1 + c t^8` has no jerk at `t = 0`, so the first-step estimate only sees
    /// `|v| / |a|` and overshoots where the `t^8` term takes over. The first step must be
    /// redone smaller, and the result must still match `x = v0 t + t^2 / 2 + c t^10 / 90`.
    #[test]
    fn first_step_is_redone_when_its_error_is_above_target() {
        use nalgebra::Vector1;
        const C: f64 = 1e-20;
        const V0: f64 = 1e4;
        let accel = |time: Time<TDB>,
                     _pos: &Vector1<f64>,
                     _vel: &Vector1<f64>,
                     evals: &mut Vec<(f64, bool)>,
                     exact_eval: bool|
         -> KeteResult<Vector1<f64>> {
            evals.push((time.jd(), exact_eval));
            Ok(Vector1::new(1.0 + C * time.jd().powi(8)))
        };
        let t_final = 1000.0;
        let (pos, _vel, evals) = RadauIntegrator::integrate(
            &accel,
            Vector1::new(0.0),
            Vector1::new(V0),
            0.0.into(),
            t_final.into(),
            Vec::new(),
            None,
            None,
        )
        .unwrap();

        // The first exact evaluation after the initial one marks the first accepted step;
        // an evaluation beyond it before that point belongs to a redone attempt.
        let first_accept = evals
            .iter()
            .skip(1)
            .find(|(_, exact)| *exact)
            .map(|(t, _)| *t)
            .unwrap();
        let furthest_before = evals
            .iter()
            .skip(1)
            .take_while(|(_, exact)| !*exact)
            .map(|(t, _)| *t)
            .fold(0.0_f64, f64::max);
        assert!(
            furthest_before > first_accept,
            "no oversized first attempt: furthest {furthest_before}, accepted {first_accept}"
        );

        let exact = V0 * t_final + 0.5 * t_final.powi(2) + C * t_final.powi(10) / 90.0;
        assert!(
            ((pos[0] - exact) / exact).abs() < 1e-12,
            "position {} differs from {exact}",
            pos[0]
        );
    }

    #[test]
    fn basic_two_body() {
        let (pos, vel, _meta) = RadauIntegrator::integrate(
            &central_accel,
            Vector3::new(0.46937657, -0.8829981, 0.),
            Vector3::new(0.01518942, 0.00807426, 0.),
            0.0.into(),
            1000.0.into(),
            CentralAccelMeta::default(),
            None,
            None,
        )
        .unwrap();
        assert!((pos[0] + 0.916350120888658).abs() < 1e-8);
        assert!((pos[1] + 0.4003771936559588).abs() < 1e-8);
        assert_eq!(pos[2], 0.0);

        assert!((vel[0] - 0.006887328686018099).abs() < 1e-8);
        assert!((vel[1] + 0.01576315407302832).abs() < 1e-8);
        assert_eq!(vel[2], 0.0);
    }

    /// Accuracy must not depend on heliocentric distance.
    ///
    /// An additive floor in the error-control denominator is absolute, in AU/day^2, and
    /// the solar monopole falls below one of `1e-6` at `sqrt(GMS / 1e-6) = 17.2 AU`. Beyond
    /// that the denominator would stop tracking the acceleration and the control would
    /// become absolute, so a distant orbit would come back far outside the tolerance the
    /// same integrator holds at 1 AU while using *fewer* evaluations, the step having grown
    /// unchecked. This is the row that would catch it.
    ///
    /// Each row integrates exactly one period from aphelion, where the initial state is
    /// its own exact reference. The inclined rows are there because a per-component
    /// denominator could in principle over-tighten on a near-zero `a_z`; they cost the
    /// same as the `i = 10 deg` row, so it does not.
    #[test]
    fn accuracy_is_independent_of_heliocentric_distance() {
        use crate::constants::GMS;
        let cases: [(f64, f64, f64); 9] = [
            (1.0, 0.0, 0.0),
            (5.0, 0.0, 0.0),
            (30.0, 0.0, 0.0),
            (100.0, 0.0, 0.0),
            (5.0, 0.8, 0.0),
            (30.0, 0.967, 0.0),
            (1.0, 0.0, 0.001),
            (2.5, 0.1, 10.0),
            (2.5, 0.1, 0.01),
        ];
        for (semi_major, ecc, incl) in cases {
            let period = std::f64::consts::TAU * (semi_major.powi(3) / GMS).sqrt();
            let r_aph = semi_major * (1.0 + ecc);
            let p = semi_major * (1.0 - ecc * ecc);
            let (ci, si) = (incl.to_radians().cos(), incl.to_radians().sin());
            let pos0 = Vector3::new(-r_aph, 0.0, 0.0);
            let vy = -(GMS / p).sqrt() * (1.0 - ecc);
            let vel0 = Vector3::new(0.0, vy * ci, vy * si);

            let (pos, _vel, meta) = RadauIntegrator::integrate(
                &central_accel,
                pos0,
                vel0,
                0.0.into(),
                period.into(),
                CentralAccelMeta::default(),
                Some(3),
                None,
            )
            .unwrap();

            let err = (pos - pos0).norm() / r_aph;
            println!(
                "a={semi_major:6.1} e={ecc:5.3} i={incl:6.3}   {:8} evals   {err:9.2e}",
                meta.eval_count,
            );
            assert!(
                err < 1e-12,
                "a={semi_major} e={ecc} i={incl}: one-period return error {err:e} \
                 exceeds 1e-12; error control is degrading with distance",
            );
        }
    }

    /// Two neighboring trajectories separate by what the dynamics says, not by a
    /// fixed floor set by where each happened to stop in time.
    ///
    /// The loop ends once `cur_time` is within `convergence_tol` of the target and
    /// then reports the state at the target epoch, so two trajectories that reached
    /// it through different step sequences are compared at the same time. A
    /// leftover time offset would become an along-track difference of
    /// `velocity * dt` that differencing does not cancel.
    ///
    /// The check: the response to a tiny offset must stay proportional to that
    /// offset. A floor shows up as the ratio blowing up for the smallest ones.

    #[test]
    fn nearby_trajectories_separate_proportionally() {
        let pos = Vector3::new(1.0, 0.0, 0.0);
        let vel = Vector3::new(0.0, 0.01720209895, 0.0);
        // Epoch chosen at a realistic Julian date: the effect scales with the ULP
        // of the time variable, so it is invisible near JD 0.
        let (t0, t1) = (2451545.0, 2451745.0);

        let run = |offset: f64| {
            let (p, _v, _m) = RadauIntegrator::integrate(
                &central_accel,
                Vector3::new(pos[0] + offset, pos[1], pos[2]),
                vel,
                t0.into(),
                t1.into(),
                CentralAccelMeta::default(),
                None,
                None,
            )
            .unwrap();
            p
        };

        let base = run(0.0);
        // Reference slope from an offset large enough to be well resolved.
        let big = 1e-8;
        let slope = (run(big) - base).norm() / big;

        for k in 1..=8_u32 {
            let offset = f64::from(k) * 1e-14;
            let ratio = (run(offset) - base).norm() / (slope * offset);
            println!("offset {offset:e} AU: separation is {ratio:.2}x the linear response");
            assert!(
                ratio < 5.0,
                "offset {offset:e} separated {ratio:.1}x the linear response, \
                 which means a fixed floor dominates rather than the dynamics"
            );
        }
    }

    /// Integrate three periods of an eccentric (e = 0.5) heliocentric orbit, forward or
    /// backward, into `dense`.
    fn eccentric_dense_run(
        direction: f64,
        dense: &mut RadauDense,
    ) -> (Vector3<f64>, Vector3<f64>, Time<TDB>, f64) {
        use crate::constants::GMS;
        let (a, ecc) = (1.0_f64, 0.5_f64);
        let peri = a * (1.0 - ecc);
        let pos0 = Vector3::new(peri, 0.0, 0.0);
        let vel0 = Vector3::new(0.0, (GMS * (1.0 + ecc) / peri).sqrt(), 0.0);
        let span = direction * 3.0 * std::f64::consts::TAU * (a.powi(3) / GMS).sqrt();
        let t0 = Time::<TDB>::new(2_451_545.0);
        let _ = RadauIntegrator::integrate(
            &central_accel,
            pos0,
            vel0,
            t0,
            t0 + span,
            CentralAccelMeta::default(),
            None,
            Some(dense),
        )
        .unwrap();
        (pos0, vel0, t0, span)
    }

    /// The trajectory covers exactly the integrated span, forward and backward, and
    /// refuses queries outside it. Integrating again into the same dense output
    /// replaces it.
    #[test]
    fn dense_output_covers_the_span() {
        let mut dense = RadauDense::new();
        for direction in [1.0, -1.0] {
            let (_, _, t0, span) = eccentric_dense_run(direction, &mut dense);
            assert!(dense.n_steps() > 0);
            assert!((dense.start().unwrap() - t0).elapsed.abs() < 1e-12);
            assert!((dense.end().unwrap() - (t0 + span)).elapsed.abs() < 1e-9);
            assert!(dense.evaluate(t0).is_ok() && dense.evaluate(t0 + span).is_ok());
            assert!(matches!(
                dense.evaluate(t0 - direction * 1.0),
                Err(Error::Bounds(_))
            ));
            assert!(matches!(
                dense.evaluate(t0 + span + direction * 1.0),
                Err(Error::Bounds(_))
            ));
            println!(
                "dense_output_covers_the_span: direction {direction}, {} steps over {:.1} days",
                dense.n_steps(),
                span.abs()
            );
        }
    }

    /// Each step's polynomial at its end reproduces the state the next step starts
    /// from, so the dense output is continuous in the state and its derivative.
    #[test]
    fn dense_output_is_continuous() {
        let mut dense = RadauDense::new();
        let _ = eccentric_dense_run(1.0, &mut dense);
        let (mut worst_pos, mut worst_vel) = (0.0_f64, 0.0_f64);
        for step in 1..dense.n_steps() {
            let boundary = dense.epochs[step];
            let (pos_a, vel_a) = dense.evaluate_step(step - 1, boundary);
            let (pos_b, vel_b) = dense.evaluate_step(step, boundary);
            let (pa, pb) = (Vector3::from_vec(pos_a), Vector3::from_vec(pos_b));
            let (va, vb) = (Vector3::from_vec(vel_a), Vector3::from_vec(vel_b));
            worst_pos = worst_pos.max((pa - pb).norm() / pb.norm());
            worst_vel = worst_vel.max((va - vb).norm() / vb.norm());
        }
        println!("dense_output_is_continuous: worst jump pos {worst_pos:e}, vel {worst_vel:e}");
        assert!(worst_pos < 1e-14, "position jump {worst_pos:e}");
        assert!(worst_vel < 1e-14, "velocity jump {worst_vel:e}");
    }

    /// Inside the steps, forward and backward, the dense output is within an order of
    /// magnitude of the integrator's accuracy at the step boundaries, against the
    /// analytic two-body solution. Gauss-Radau collocation is most accurate at the end
    /// of each step, so the interior is somewhat worse.
    #[test]
    fn dense_output_matches_the_analytic_orbit() {
        use crate::kepler::analytic_2_body;
        for direction in [1.0, -1.0] {
            let mut dense = RadauDense::new();
            let (pos0, vel0, t0, _) = eccentric_dense_run(direction, &mut dense);
            let error_at = |time: Time<TDB>| {
                let (pos, _) = dense.evaluate(time).unwrap();
                let (exact, _) = analytic_2_body(time - t0, &pos0, &vel0, None).unwrap();
                (Vector3::from_column_slice(&pos) - exact).norm()
            };
            let mut boundary = 0.0_f64;
            let mut interior = 0.0_f64;
            for (epoch, step_size) in dense.epochs.iter().zip(&dense.step_sizes) {
                boundary = boundary.max(error_at(*epoch));
                for frac in [0.13, 0.37, 0.5, 0.71, 0.94] {
                    interior = interior.max(error_at(*epoch + frac * step_size));
                }
            }
            println!(
                "dense_output_matches_the_analytic_orbit: direction {direction}, worst error \
                 at step boundaries {boundary:.3e} AU, inside steps {interior:.3e} AU"
            );
            assert!(
                interior < 10.0 * boundary.max(1e-15),
                "interior error {interior:e} is more than 10x the boundary error {boundary:e}"
            );
        }
    }
}
