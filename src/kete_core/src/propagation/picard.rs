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

/// Container for the pre-computed matrices used by the picard-chebyschev integrator.
#[derive(Debug, Clone)]
#[allow(unused, reason = "debugging")]
pub struct PicardCoefficients<const N: usize, const NM1: usize> {
    c: SMatrix<f64, N, N>,

    a: SMatrix<f64, NM1, N>,

    sa: SVector<f64, N>,

    tau: [f64; N],
}

impl<const N: usize, const NM1: usize> PicardCoefficients<N, NM1> {
    /// foo
    pub fn test(&self) -> [f64; N] {
        self.sa.into()
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

        Self { a, sa, c, tau }
    }
}

/// `PicardCoefficients` for a 14th order fit
pub static PC15: std::sync::LazyLock<PicardCoefficients<15, 14>> =
    std::sync::LazyLock::new(PicardCoefficients::default);

/// `PicardCoefficients` for a 24th order fit
pub static PC25: std::sync::LazyLock<PicardCoefficients<25, 24>> =
    std::sync::LazyLock::new(PicardCoefficients::default);

// def exponential_decay(y, t):
//     return -y
// y = np.array([[1], [2], [10]] * np.exp(-0*np.array(ts)))
// SA = S@A
// B = np.zeros([N, 3])
// last = None
// for _ in range(50):
//     F = w2 * np.array([exponential_decay(y, t0) for y, t0 in zip(y, ts)])

//     B[0] = ((SA @ F.T) + 2*y[:, 0])
//     B[1:] = A @ F.T
//     last = y
//     y = (C @ B).T
