/// # Picard-Chebyshev Numerical Integrator
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
use std::f64::consts::PI;

use itertools::Itertools;
use nalgebra::{SMatrix, SVector};

use crate::errors::{Error, KeteResult};

/// Function will be of the form y' = F(t, y, metadata)
/// This is the first-order general solver.
type PicardFunc<'a, MType, const DIM: usize> =
    &'a dyn Fn(f64, &SVector<f64, DIM>, &mut MType, bool) -> KeteResult<SVector<f64, DIM>>;

/// Container for the pre-computed matrices used by the picard-chebyschev integrator.
#[derive(Debug, Clone)]
#[allow(unused, reason = "debugging")]
pub struct PicardCoefficients<const N: usize, const NM1: usize> {
    c: SMatrix<f64, N, N>,

    a: SMatrix<f64, N, NM1>,

    sa: SVector<f64, N>,

    tau: [f64; N],
}

/// Evaluate chebyshev polynomials of the first type.
#[inline(always)]
pub(crate) fn chebyshev_eval<const R: usize, const C: usize>(
    x: f64,
    coef: &[[f64; R]; C],
) -> [f64; C] {
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

#[allow(unused, reason = "debugging")]
impl<'a, const N: usize, const NM1: usize> PicardCoefficients<N, NM1> {
    /// foo
    pub fn test(&self) -> [f64; N] {
        self.sa.into()
    }

    /// Given the start and stop times of a desired integration, return the actual
    /// times where the function will be evaluated starting at t0 and ending at t1.
    pub fn sample_points(&self, t0: f64, t1: f64) -> [f64; N] {
        let w2 = (t1 - t0) / 2.0;
        let w1 = (t1 + t0) / 2.0;
        let mut times = self.tau;
        times.iter_mut().for_each(|mut x| *x = w2 * (*x) + w1);
        times
    }

    /// Integrate from the start point to the end.
    pub fn integrate<const DIM: usize, MType>(
        &self,
        func: PicardFunc<'a, MType, DIM>,
        t0: f64,
        t1: f64,
        mut initial_pos: SMatrix<f64, DIM, N>,
        metadata: &mut MType,
    ) -> KeteResult<SVector<f64, DIM>> {
        let mut cur_t0 = t0;
        let mut cur_t1 = t1;
        let mut cur_stepsize = t1 - t0;
        let mut n_steps = 0;
        loop {
            let step = self.step(func, cur_t0, cur_t1, initial_pos, metadata);
            match step {
                Ok(res) => {
                    initial_pos = res.y;
                    for d in 0..DIM {
                        for elem in 0..N {
                            initial_pos[(d, elem)] *=
                                initial_pos[(d, NM1)] / initial_pos[(0, elem)];
                        }
                    }
                    if cur_t1 == t1 {
                        return Ok(initial_pos.column(N - 1).into());
                    }
                    cur_t0 = cur_t1;
                    cur_t1 += cur_stepsize;
                    if (cur_stepsize.is_sign_positive() && cur_t1 > t1)
                        || (cur_stepsize.is_sign_negative() && cur_t1 < t1)
                    {
                        cur_t1 = t1;
                    }
                    cur_stepsize *= 1.3;
                    n_steps += 1;
                }
                Err(m) => {
                    cur_stepsize *= 0.75;
                    cur_t1 = cur_t0 + cur_stepsize;
                }
            }
            if cur_stepsize.abs() < 1e-5 {
                return Err(Error::Convergence(
                    "Failed to converge, step sizes became too small.".into(),
                ));
            }
        }
        Err(Error::Convergence("Failed to converge.".into()))
    }

    /// Single fallible step of the integrator
    pub fn step<const DIM: usize, MType>(
        &self,
        func: PicardFunc<'a, MType, DIM>,
        t0: f64,
        t1: f64,
        mut initial_pos: SMatrix<f64, DIM, N>,
        metadata: &mut MType,
    ) -> KeteResult<PicardStep<N, DIM>> {
        let mut b: SMatrix<f64, DIM, N> = SMatrix::zeros();
        let mut last_pos: SMatrix<f64, DIM, N> = SMatrix::zeros();
        let mut f: SMatrix<f64, DIM, N> = SMatrix::zeros();
        let times = self.sample_points(t0, t1);
        let w2 = (t1 - t0) / 2.0;
        let mut error = 100.0;
        let mut last_error = 50.0;
        for idx in 0..30 {
            for ((&time, y), mut f_row) in times
                .iter()
                .zip(initial_pos.column_iter())
                .zip(f.column_iter_mut())
            {
                f_row.set_column(0, &(w2 * func(time, &y.into(), metadata, false)?));
            }

            b.set_column(0, &(f * self.sa + 2.0 * initial_pos.column(0)));

            b.fixed_view_mut::<DIM, NM1>(0, 1)
                .iter_mut()
                .zip((f * self.a).into_iter())
                .for_each(|(x, &n)| *x = n);

            last_pos = initial_pos;
            initial_pos = b * self.c;
            last_error = error;
            error = (last_pos - initial_pos).abs().max();
            if error > 1000.0 && last_error > error {
                return Err(Error::Convergence(format!(
                    "Error diverging. error = {error}"
                )));
            }
            if error < 10.0 * f64::EPSILON {
                return Ok(PicardStep {
                    b,
                    y: initial_pos,
                    prev_y: last_pos,
                    t0,
                    t1,
                    error,
                });
            }
        }
        Err(Error::Convergence(format!(
            "Failed to converge. error = {error}"
        )))
    }
}

impl<const N: usize, const NM1: usize> Default for PicardCoefficients<N, NM1> {
    fn default() -> Self {
        // NM1 must be equal to N-1, I wish I could do this with const generics
        // but that would require switching to nightly rust and I wont do that.
        // So this compile time check is done instead.
        const { assert!(N - 1 == NM1, "NM1 must be 1 less than N.") };

        // Combining the cos(k*arccos(-cos(...))) simplifies to this equation
        // This is used throughout the integrator's construction
        fn cheb_t(k: f64, j: f64, n: f64) -> f64 {
            (k * (j + n - 1.0) * PI / (n - 1.0)).cos()
        }

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

        // Now we construct the SA Vector, in the original paper they keep S as a separate
        // vector and multiply it against A over and over, whereas we are just doing it once
        // and saving it.
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

/// `PicardCoefficients` for a 14th order fit
pub static PC15: std::sync::LazyLock<PicardCoefficients<15, 14>> =
    std::sync::LazyLock::new(PicardCoefficients::default);

/// `PicardCoefficients` for a 24th order fit
pub static PC25: std::sync::LazyLock<PicardCoefficients<25, 24>> =
    std::sync::LazyLock::new(PicardCoefficients::default);

/// A single stop of the Picard Integrator
#[derive(Debug, Clone)]
#[allow(unused, reason = "debugging")]
pub struct PicardStep<const N: usize, const DIM: usize> {
    b: SMatrix<f64, DIM, N>,

    /// Final result
    pub y: SMatrix<f64, DIM, N>,

    prev_y: SMatrix<f64, DIM, N>,

    t0: f64,

    t1: f64,

    error: f64,
}

impl<const N: usize, const DIM: usize> PicardStep<N, DIM> {
    /// Evaluate the fitted integration solution at the specified time.
    /// This will fail if the requested time is outside of the integration bounds.
    pub fn sample(&self, t: f64) -> KeteResult<[f64; DIM]> {
        let w1 = (self.t0 + self.t1) * 0.5;
        let w2 = (self.t1 - self.t0) * 0.5;
        let tau_time = ((t - w1) * w2).acos();
        if tau_time.is_nan() {
            return Err(Error::ExceedsLimits(
                "Queried time it outside of the fitted time span".into(),
            ));
        }
        Ok(chebyshev_eval(tau_time, &self.b.transpose().into()))
    }
}
