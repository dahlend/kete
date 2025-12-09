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
// Copyright (c) 2025, Dar Dahlen
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

        let mut times = self.evaluation_times(cur_t0.into(), cur_t1.into());
        let mut initial_pos = init_func(&times, &initial_pos);

        let mut has_failed = false;
        loop {
            let step = self.step(func, cur_t0.into(), cur_t1.into(), initial_pos, metadata);
            match step {
                Ok(res) => {
                    if cur_t1 == t1 {
                        return Ok(res.y.column(N - 1).into());
                    }
                    times = self.evaluation_times(cur_t0.into(), cur_t1.into());
                    initial_pos = init_func(&times, &res.y.column(N - 1).into());

                    cur_t0 = cur_t1;
                    cur_t1 += cur_stepsize;
                    if (cur_stepsize.is_sign_positive() && cur_t1 > t1)
                        || (cur_stepsize.is_sign_negative() && cur_t1 < t1)
                    {
                        cur_t1 = t1;
                    }
                    if !has_failed {
                        cur_stepsize *= 1.5;
                    }
                }
                Err(Error::Impact(idx, jd)) => {
                    return Err(Error::Impact(idx, jd));
                }
                Err(_) => {
                    has_failed = true;
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
        let mut error: f64 = 100.0;
        let mut last_error = 50.0;

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

            b.set_column(0, &(f * self.sa + 2.0 * cur_pos.column(0)));

            b.fixed_view_mut::<DIM, NM1>(0, 1)
                .iter_mut()
                .zip((f * self.a).into_iter())
                .for_each(|(x, &n)| *x = n);

            let last_pos = cur_pos;
            cur_pos = b * self.c;

            // diverging, quit quickly
            if error > 1000.0 && last_error > error {
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
            if error < 2.0 * f64::EPSILON {
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
}
