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

use crate::fitting::{ConvergenceError, FittingResult};

/// Result of Nelder-Mead optimization.
#[derive(Debug, Clone)]
pub struct NelderMeadResult {
    /// The point that minimizes the objective function.
    pub point: Vec<f64>,

    /// The objective function value at the optimum.
    pub value: f64,

    /// Number of function evaluations performed.
    pub func_evals: usize,
}

/// Minimize a scalar objective function using the Nelder-Mead simplex method.
///
/// This is a derivative-free optimizer well-suited to low-dimensional problems
/// (typically <= 10 parameters). It maintains a simplex of `n+1` vertices in
/// `n`-dimensional space and iteratively transforms it via reflection,
/// expansion, contraction, and shrink operations.
///
/// # Arguments
///
/// * `func` -- Objective function mapping `&[f64]` -> `f64`. Must return
///   finite values for all evaluations in the search region.
/// * `initial` -- Starting point (length `n`).
/// * `scale` -- Per-dimension step sizes used to build the initial simplex.
///   Each vertex `i` (for `i = 1..=n`) is formed by adding `scale[i-1]` to
///   dimension `i-1` of `initial`. Should reflect the expected scale of
///   variation in each parameter.
/// * `atol` -- Absolute tolerance on the simplex diameter. The solver
///   terminates when the range of function values across the simplex drops
///   below `atol`.
/// * `max_iter` -- Maximum number of iterations (each iteration involves
///   1-`n+1` function evaluations).
///
/// # Returns
///
/// [`NelderMeadResult`] containing the best point, value, and evaluation count.
///
/// # Errors
///
/// Returns [`ConvergenceError::Iterations`] if `max_iter` is reached without
/// convergence, or [`ConvergenceError::NonFinite`] if the objective returns
/// a non-finite value.
///
/// # Example
///
/// ```
/// use kete_stats::fitting::nelder_mead;
///
/// // Rosenbrock function -- minimum at (1, 1).
/// let rosenbrock = |x: &[f64]| {
///     let a = 1.0 - x[0];
///     let b = x[1] - x[0] * x[0];
///     a * a + 100.0 * b * b
/// };
///
/// let result = nelder_mead(rosenbrock, &[-1.0, -1.0], &[0.5, 0.5], 1e-12, 5000).unwrap();
/// assert!((result.point[0] - 1.0).abs() < 1e-4);
/// assert!((result.point[1] - 1.0).abs() < 1e-4);
/// assert!(result.value < 1e-8);
/// ```
///
/// # Panics
///
/// Panics if `initial` and `scale` have different lengths.
pub fn nelder_mead(
    func: impl Fn(&[f64]) -> f64,
    initial: &[f64],
    scale: &[f64],
    atol: f64,
    max_iter: usize,
) -> FittingResult<NelderMeadResult> {
    let n = initial.len();
    assert_eq!(scale.len(), n, "scale must have the same length as initial");

    // Standard Nelder-Mead coefficients.
    let alpha = 1.0; // reflection
    let gamma = 2.0; // expansion
    let rho = 0.5; // contraction
    let sigma = 0.5; // shrink

    let mut func_evals: usize = 0;

    // Evaluate with tracking and NaN guard.
    let eval = |x: &[f64], evals: &mut usize| -> FittingResult<f64> {
        *evals += 1;
        let v = func(x);
        if v.is_finite() {
            Ok(v)
        } else {
            Err(ConvergenceError::NonFinite)
        }
    };

    // Build initial simplex: vertex 0 = initial, vertex i = initial + scale[i-1] on axis i-1.
    let mut simplex: Vec<Vec<f64>> = Vec::with_capacity(n + 1);
    simplex.push(initial.to_vec());
    for i in 0..n {
        let mut v = initial.to_vec();
        v[i] += scale[i];
        simplex.push(v);
    }

    // Evaluate all vertices.
    let mut values: Vec<f64> = Vec::with_capacity(n + 1);
    for v in &simplex {
        values.push(eval(v, &mut func_evals)?);
    }

    for _iter in 0..max_iter {
        // Sort by function value.
        let mut order: Vec<usize> = (0..=n).collect();
        order.sort_by(|&a, &b| values[a].partial_cmp(&values[b]).unwrap());
        let sorted_simplex: Vec<Vec<f64>> = order.iter().map(|&i| simplex[i].clone()).collect();
        let sorted_values: Vec<f64> = order.iter().map(|&i| values[i]).collect();
        simplex = sorted_simplex;
        values = sorted_values;

        // Check convergence: range of values across the simplex.
        let f_best = values[0];
        let f_worst = values[n];
        if (f_worst - f_best).abs() < atol {
            return Ok(NelderMeadResult {
                point: simplex[0].clone(),
                value: values[0],
                func_evals,
            });
        }

        // Centroid of all vertices except the worst.
        let centroid = centroid_excluding_last(&simplex);

        // Reflect.
        let reflected = transform(&centroid, &simplex[n], alpha);
        let f_reflected = eval(&reflected, &mut func_evals)?;

        if f_reflected < values[0] {
            // Try expansion.
            let expanded = transform(&centroid, &simplex[n], gamma);
            let f_expanded = eval(&expanded, &mut func_evals)?;
            if f_expanded < f_reflected {
                simplex[n] = expanded;
                values[n] = f_expanded;
            } else {
                simplex[n] = reflected;
                values[n] = f_reflected;
            }
        } else if f_reflected < values[n - 1] {
            // Reflected is better than second-worst; accept.
            simplex[n] = reflected;
            values[n] = f_reflected;
        } else {
            // Contraction.
            let (contracted, base_val) = if f_reflected < values[n] {
                // Outside contraction.
                (interpolate(&centroid, &reflected, rho), f_reflected)
            } else {
                // Inside contraction.
                (interpolate(&centroid, &simplex[n], rho), values[n])
            };
            let f_contracted = eval(&contracted, &mut func_evals)?;

            if f_contracted < base_val {
                simplex[n] = contracted;
                values[n] = f_contracted;
            } else {
                // Shrink the entire simplex toward the best vertex.
                let best = simplex[0].clone();
                for i in 1..=n {
                    for (sij, &bj) in simplex[i].iter_mut().zip(best.iter()) {
                        *sij = bj + sigma * (*sij - bj);
                    }
                    values[i] = eval(&simplex[i], &mut func_evals)?;
                }
            }
        }
    }

    // Return best found, but flag non-convergence.
    Err(ConvergenceError::Iterations)
}

/// Centroid of all simplex vertices except the last (worst).
fn centroid_excluding_last(simplex: &[Vec<f64>]) -> Vec<f64> {
    let n = simplex.len() - 1; // number of dimensions
    let mut c = vec![0.0; simplex[0].len()];
    for v in &simplex[..n] {
        for (ci, vi) in c.iter_mut().zip(v.iter()) {
            *ci += vi;
        }
    }
    let inv_n = 1.0 / n as f64;
    for ci in &mut c {
        *ci *= inv_n;
    }
    c
}

/// Reflect/expand: centroid + factor * (centroid - worst).
fn transform(centroid: &[f64], worst: &[f64], factor: f64) -> Vec<f64> {
    centroid
        .iter()
        .zip(worst.iter())
        .map(|(&c, &w)| c + factor * (c - w))
        .collect()
}

/// Interpolate: centroid + factor * (point - centroid).
fn interpolate(centroid: &[f64], point: &[f64], factor: f64) -> Vec<f64> {
    centroid
        .iter()
        .zip(point.iter())
        .map(|(&c, &p)| c + factor * (p - c))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quadratic_1d() {
        // f(x) = (x - 3)^2, minimum at x = 3.
        let f = |x: &[f64]| (x[0] - 3.0) * (x[0] - 3.0);
        let result = nelder_mead(f, &[0.0], &[1.0], 1e-14, 500).unwrap();
        assert!(
            (result.point[0] - 3.0).abs() < 1e-6,
            "got {}",
            result.point[0]
        );
        assert!(result.value < 1e-12);
    }

    #[test]
    fn test_quadratic_2d() {
        // f(x,y) = (x-1)^2 + (y-2)^2, minimum at (1, 2).
        let f = |x: &[f64]| (x[0] - 1.0).powi(2) + (x[1] - 2.0).powi(2);
        let result = nelder_mead(f, &[5.0, -3.0], &[1.0, 1.0], 1e-14, 1000).unwrap();
        assert!(
            (result.point[0] - 1.0).abs() < 1e-5,
            "x={}",
            result.point[0]
        );
        assert!(
            (result.point[1] - 2.0).abs() < 1e-5,
            "y={}",
            result.point[1]
        );
        assert!(result.value < 1e-10);
    }

    #[test]
    fn test_rosenbrock() {
        let rosenbrock = |x: &[f64]| {
            let a = 1.0 - x[0];
            let b = x[1] - x[0] * x[0];
            a * a + 100.0 * b * b
        };
        let result = nelder_mead(rosenbrock, &[-1.0, -1.0], &[0.5, 0.5], 1e-14, 10000).unwrap();
        assert!(
            (result.point[0] - 1.0).abs() < 1e-4,
            "x={}",
            result.point[0]
        );
        assert!(
            (result.point[1] - 1.0).abs() < 1e-4,
            "y={}",
            result.point[1]
        );
    }

    #[test]
    fn test_nonfinite_returns_error() {
        let f = |x: &[f64]| if x[0] > 0.5 { f64::NAN } else { x[0] * x[0] };
        let result = nelder_mead(f, &[0.0], &[1.0], 1e-10, 100);
        assert!(result.is_err());
    }
}
