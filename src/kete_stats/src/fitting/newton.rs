// SPDX-FileCopyrightText: 2026 Dar Dahlen
// SPDX-FileCopyrightText: 2025 California Institute of Technology
// SPDX-License-Identifier: BSD-3-Clause

use crate::fitting::{ConvergenceError, FittingResult};

/// Solve root using the Newton-Raphson method.
///
/// This accepts a two functions, the first being a single input function for which
/// the root is desired. The second function being the derivative of the first with
/// respect to the input variable.
///
/// ```
///     use kete_stats::fitting::newton_raphson;
///     let f = |x: f64| { 1.0 * x * x - 1.0 };
///     let d = |x| { 2.0 * x };
///     let root = newton_raphson(f, d, 0.0, 1e-10).unwrap();
///     assert!((root - 1.0).abs() < 1e-12);
/// ```
///
/// # Errors
///
/// [`ConvergenceError`] may be returned in the following cases:
///     - Any function evaluation return a non-finite value.
///     - Derivative is zero but not converged.
///     - Failed to converge within 100 iterations.
#[inline(always)]
#[allow(
    clippy::missing_panics_doc,
    reason = "By construction this cannot panic."
)]
pub fn newton_raphson<T>(
    func: impl Fn(T) -> T,
    der: impl Fn(T) -> T,
    start: T,
    atol: T,
) -> FittingResult<T>
where
    T: num_traits::Float + num_traits::ToPrimitive + num_traits::NumAssignOps,
{
    let mut x = start;

    let eps = T::epsilon() * T::from(1000.0).unwrap();
    let half = T::from(0.5).unwrap();

    // if the starting position has derivative of 0, nudge it a bit.
    if der(x).abs() < eps {
        x += T::from(0.1).unwrap();
    }

    let mut f_eval: T;
    let mut d_eval: T;
    for _ in 0..100 {
        f_eval = func(x);
        if f_eval.abs() < atol {
            return Ok(x);
        }
        d_eval = der(x);

        // Derivative is 0, cannot solve
        if d_eval.abs() < eps {
            Err(ConvergenceError::ZeroDerivative)?;
        }

        if !d_eval.is_finite() || !f_eval.is_finite() {
            Err(ConvergenceError::NonFinite)?;
        }

        // 0.5 reduces the step size to slow down the rate of convergence.
        x -= half * f_eval / d_eval;

        d_eval = der(x);
        if d_eval.abs() < T::from(1e-3).unwrap() {
            f_eval = func(x);
        }
        x -= half * f_eval / d_eval;
    }
    Err(ConvergenceError::Iterations)?
}

#[cfg(test)]
mod tests {
    use crate::fitting::newton_raphson;

    #[test]
    fn test_newton_raphson() {
        let f = |x| 1.0 * x * x - 1.0;
        let d = |x| 2.0 * x;

        let root: f64 = newton_raphson(f, d, 0.0, 1e-10).unwrap();
        assert!((root - 1.0).abs() < 1e-12);
    }
}
