//! Gauss-Radau Spacing Numerical Integrator
//! This solves a second-order initial value problem.
// BSD 3-Clause License
//
// Copyright (c) 2026, Dar Dahlen
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
use crate::errors::Error;
use crate::integrators::util::SecondOrderODE;
use crate::prelude::KeteResult;
use crate::time::{TDB, Time};
use itertools::izip;
use nalgebra::Matrix;
use nalgebra::allocator::Allocator;
use nalgebra::{DefaultAllocator, Dim, OMatrix, OVector, RowSVector, SMatrix, U1, U7};

/// Integrator will return a result of this type.
type RadauResult<MType, D> = KeteResult<(OVector<f64, D>, OVector<f64, D>, MType)>;

pub(crate) const GAUSS_RADAU_SPACINGS: [f64; 8] = [
    0.0,
    0.05626256053692215,
    0.18024069173689236,
    0.3526247171131696,
    0.5471536263305554,
    0.7342101772154105,
    0.8853209468390958,
    0.9775206135612875,
];

// initialize W
static W_VEC: std::sync::LazyLock<RowSVector<f64, 7>> = std::sync::LazyLock::new(|| {
    let mut w = RowSVector::<f64, 7>::zeros();
    for (idx, e) in w.iter_mut().enumerate() {
        *e = (((idx + 2) * (idx + 3)) as f64).recip();
    }
    w
});

// initialize U
pub(crate) static U_VEC: std::sync::LazyLock<RowSVector<f64, 7>> = std::sync::LazyLock::new(|| {
    let mut u = RowSVector::<f64, 7>::zeros();
    for (idx, e) in u.iter_mut().enumerate() {
        *e = ((idx + 2) as f64).recip();
    }
    u
});

// initialize C
pub(crate) static C_MAT: std::sync::LazyLock<SMatrix<f64, 7, 7>> = std::sync::LazyLock::new(|| {
    let mut c = SMatrix::<f64, 7, 7>::identity();
    for idx in 0..7 {
        if idx > 0 {
            c[(idx, 0)] = -GAUSS_RADAU_SPACINGS[idx] * c[(idx - 1, 0)];
        }
        for idy in 1..idx {
            c[(idx, idy)] = c[(idx - 1, idy - 1)] - GAUSS_RADAU_SPACINGS[idx] * c[(idx - 1, idy)];
        }
    }
    c
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

pub(crate) static U_POW_TABLE: std::sync::LazyLock<[RowSVector<f64, 7>; 7]> =
    std::sync::LazyLock::new(|| {
        let u = &*U_VEC;
        let mut table = [RowSVector::<f64, 7>::zeros(); 7];
        for (j, h) in GAUSS_RADAU_SPACINGS.iter().enumerate().skip(1) {
            let mut hp = *h;
            for k in 0..7 {
                table[j - 1][k] = hp * u[k];
                hp *= h;
            }
        }
        table
    });

pub(crate) const MIN_RATIO: f64 = 0.25;
pub(crate) const EPSILON: f64 = 1e-6;
pub(crate) const MIN_STEP: f64 = 0.00005;

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
/// to match the original Fortran implementation. After some experimentation it was
/// found that the correction and prediction steps turned out to help to such a small
/// degree in general that they were not worth the added complexity.
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

    state_scratch: OVector<f64, D>,
    state_der_scratch: OVector<f64, D>,
    b_scratch: OVector<f64, D>,
    eval_scratch: OVector<f64, D>,

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
            b_scratch: Matrix::zeros_generic(dim, U1),
            state_scratch: Matrix::zeros_generic(dim, U1),
            state_der_scratch: Matrix::zeros_generic(dim, U1),
            eval_scratch: Matrix::zeros_generic(dim, U1),
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
    ) -> RadauResult<MType, D> {
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

        let mut next_step_size: f64 = {
            // Estimate a reasonable first step from the initial acceleration.
            // h0 = min(0.1, (epsilon / |a0|)^(1/3)) keeps the first step's
            // cubic truncation term O(epsilon).  The 1/3 exponent is
            // deliberately conservative (lower order than the 1/7 used by
            // the step-size controller) so the very first step doesn't
            // overshoot on extreme orbits like sun-grazers.
            let a0_norm = integrator
                .cur_state_der_der
                .rows(0, integrator.control_dim)
                .amax();
            let h0 = if a0_norm > 0.0 {
                (EPSILON / a0_norm).powf(1.0 / 3.0).min(0.1)
            } else {
                0.1
            };
            h0.copysign((integrator.final_time - integrator.cur_time).elapsed)
        };

        // Convergence tolerance scales with the JD magnitude to stay above
        // f64 precision.  Absolute tolerance of 1e-12 is finer than the ULP
        // of cur_time.jd around modern epochs (JD ~2.5e6 has ULP ~5.5e-10),
        // so an absolute check could never be satisfied and the loop would
        // never terminate.  Use a relative tolerance scaled by max(|cur|,
        // |final|) with an absolute floor for the JD=0 case.
        let convergence_tol = {
            let scale = integrator
                .cur_time
                .jd
                .abs()
                .max(integrator.final_time.jd.abs());
            // Absolute floor of 1e-12 covers the small-JD limit; relative
            // factor of 1e-13 leaves ~3 orders of magnitude of headroom
            // above f64 ULP at any reasonable JD.
            (scale * 1e-13).max(1e-12)
        };

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
            match integrator.step(next_step_size) {
                Ok(s) => {
                    next_step_size = s;
                    if (integrator.cur_time - integrator.final_time).elapsed.abs() < convergence_tol
                    {
                        return Ok((
                            integrator.cur_state,
                            integrator.cur_state_der,
                            integrator.metadata,
                        ));
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

    /// Attempt a single integration step of size `step_size`.
    ///
    /// Returns the recommended next step size on success.  Failure can occur
    /// if the step size is too large for convergence, or if the ODE function
    /// itself returns an error.
    ///
    fn step(&mut self, step_size: f64) -> KeteResult<f64> {
        self.g_scratch.fill(0.0);
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

                // Evaluate the function at this new intermediate state.
                self.eval_scratch.set_column(
                    0,
                    &(self.func)(
                        (self.cur_time.jd + gauss_radau_frac * step_size).into(),
                        &self.state_scratch,
                        &self.state_der_scratch,
                        &mut self.metadata,
                        false,
                    )?,
                );

                let diff = &self.eval_scratch - &self.cur_state_der_der;

                // Use the result of that evaluation to update the current G and B
                // matrices for the next gauss spacing.

                // This is equivalent to equation (4) in everhart's paper.
                // The lookup tables and switch statements he uses were performing
                // ~100x slower than this implementation.
                self.g_scratch.set_column(idj - 1, &{
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
                let t_t = self.cur_time.jd + y_t;
                self.comp_time = (t_t - self.cur_time.jd) - y_t;
                self.cur_time.jd = t_t;
                self.cur_state_der_der = (self.func)(
                    self.cur_time,
                    &self.cur_state,
                    &self.cur_state_der,
                    &mut self.metadata,
                    true,
                )?;
                // Step-size controller: the component-wise ratio computed above,
                // max(|b6_i| / scale_i), lets the worst-resolved component drive the
                // step size.
                return Ok(step_size
                    * (EPSILON / error_ratio)
                        .powf(1.0 / 7.0)
                        .clamp(MIN_RATIO, MIN_RATIO.recip()));
            }
        }
        Err(Error::Convergence("Radau step failed to converge".into()))?
    }
}

#[cfg(test)]
mod tests {
    use nalgebra::Vector3;

    use super::*;
    use crate::integrators::stress_tests::{CentralAccelMeta, central_accel};

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
}
