// SPDX-FileCopyrightText: 2026 Dar Dahlen
// SPDX-License-Identifier: BSD-3-Clause

//! Gauss-Radau spacing integrators: [`RadauIntegrator`] for second-order and
//! [`RadauFirstOrder`] for first-order systems, with the node spacings, coefficient
//! tables, step control constants and `b` predictor they share.
mod first_order;
mod second_order;

pub use first_order::RadauFirstOrder;
pub use second_order::{RadauDense, RadauIntegrator};

use nalgebra::allocator::Allocator;
use nalgebra::{DefaultAllocator, Dim, Matrix, OMatrix, RowSVector, SMatrix, U7};

const GAUSS_RADAU_SPACINGS: [f64; 8] = [
    0.0,
    0.05626256053692215,
    0.18024069173689236,
    0.3526247171131696,
    0.5471536263305554,
    0.7342101772154105,
    0.8853209468390958,
    0.9775206135612875,
];

// initialize U
static U_VEC: std::sync::LazyLock<RowSVector<f64, 7>> = std::sync::LazyLock::new(|| {
    let mut u = RowSVector::<f64, 7>::zeros();
    for (idx, e) in u.iter_mut().enumerate() {
        *e = ((idx + 2) as f64).recip();
    }
    u
});

// initialize C
static C_MAT: std::sync::LazyLock<SMatrix<f64, 7, 7>> = std::sync::LazyLock::new(|| {
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

static U_POW_TABLE: std::sync::LazyLock<[RowSVector<f64, 7>; 7]> = std::sync::LazyLock::new(|| {
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

/// Binomial coefficients `C(n, k)` for `n, k <= 7`, used by the `b` predictor.
static BINOMIAL: std::sync::LazyLock<[[f64; 8]; 8]> = std::sync::LazyLock::new(|| {
    let mut c = [[0.0; 8]; 8];
    for n in 0..8 {
        c[n][0] = 1.0;
        for k in 1..=n {
            c[n][k] = c[n - 1][k - 1] + c[n - 1][k];
        }
    }
    c
});

const MIN_RATIO: f64 = 0.25;
const EPSILON: f64 = 1e-6;
const MIN_STEP: f64 = 0.00005;

/// Starting `b` for each step of the Gauss-Radau integrators, extrapolated from the last
/// accepted step (Everhart 1985).
///
/// Both integrators write the right-hand side over a step as
/// `F(s) = F_0 + sum_k b_k s^(k+1)` for `s` in `[0, 1]`. With
/// `q = step_size / last_step_size`, the last step's polynomial re-expanded about the end
/// of that step, in units of the new step, has coefficients
///
/// ```text
/// e_k = q^(k+1) * sum_{j >= k} C(j+1, k+1) b_j
/// ```
///
/// The prediction is `e_k` plus the error of the previous prediction, `last_b - last_e`.
/// It is computed from the last accepted step, so a retry after a failed attempt predicts
/// from the same converged `b` at the retried size. Before the first accepted step, and
/// when the step grows by more than a factor of 20 so that the extrapolation is no longer
/// meaningful, the step starts from zero instead.
///
/// A better starting `b` reduces the number of corrector sweeps a step needs; it does not
/// change the converged solution beyond the convergence tolerance.
struct BPredictor<D: Dim>
where
    DefaultAllocator: Allocator<D, U7>,
{
    /// Prediction the current step attempt started from.
    cur_e: OMatrix<f64, D, U7>,
    /// Converged `b` of the last accepted step, and the prediction it started from.
    last_b: OMatrix<f64, D, U7>,
    last_e: OMatrix<f64, D, U7>,
    /// Size of the last accepted step, zero before the first.
    last_step_size: f64,
}

impl<D: Dim> BPredictor<D>
where
    DefaultAllocator: Allocator<D, U7>,
{
    fn new(dim: D) -> Self {
        Self {
            cur_e: Matrix::zeros_generic(dim, U7),
            last_b: Matrix::zeros_generic(dim, U7),
            last_e: Matrix::zeros_generic(dim, U7),
            last_step_size: 0.0,
        }
    }

    /// Set `b` to the predicted starting value for a step of `step_size`.
    fn predict(&mut self, step_size: f64, b: &mut OMatrix<f64, D, U7>) {
        let q = if self.last_step_size == 0.0 {
            f64::INFINITY
        } else {
            step_size / self.last_step_size
        };
        if q.abs() > 20.0 {
            b.fill(0.0);
            self.cur_e.fill(0.0);
            return;
        }
        let mut q_pow = [q; 7];
        for k in 1..7 {
            q_pow[k] = q_pow[k - 1] * q;
        }
        for row in 0..b.nrows() {
            for k in 0..7 {
                let mut sum = 0.0;
                for j in k..7 {
                    sum += BINOMIAL[j + 1][k + 1] * self.last_b[(row, j)];
                }
                let e = q_pow[k] * sum;
                b[(row, k)] = e + (self.last_b[(row, k)] - self.last_e[(row, k)]);
                self.cur_e[(row, k)] = e;
            }
        }
    }

    /// Record the converged `b` of an accepted step of `step_size`.
    fn accept(&mut self, step_size: f64, b: &OMatrix<f64, D, U7>) {
        self.last_b.copy_from(b);
        self.last_e.copy_from(&self.cur_e);
        self.last_step_size = step_size;
    }
}
