// SPDX-FileCopyrightText: 2026 Dar Dahlen
// SPDX-FileCopyrightText: 2025 California Institute of Technology
// SPDX-License-Identifier: BSD-3-Clause

//! # Halley's method
//!
//! Third order root finding algorithm.
//! This is the next order method of newton-raphson.

use crate::fitting::{ConvergenceError, FittingResult};

/// Solve root using Halley's method.
///
/// This accepts a three functions, the first being a single input function for which
/// the root is desired. The second function being the derivative of the first with
/// respect to the input variable. The third is the second derivative.
///
/// ```
///     use kete_stats::fitting::halley;
///     let f = |x: f64| { 1.0 * x * x - 1.0 };
///     let d = |x| { 2.0 * x };
///     let dd = |_| { 2.0};
///     let root = halley(f, d, dd, 0.0, 1e-10).unwrap();
///     assert!((root - 1.0).abs() < 1e-12);
///
///     // Same but with f32
///     let f = |x: f32| { 1.0 * x * x - 1.0 };
///     let d = |x| { 2.0 * x };
///     let dd = |_| { 2.0};
///     let root = halley(f, d, dd, 0.0, 1e-10).unwrap();
///     assert!((root - 1.0).abs() < 1e-12);
/// ```
///
/// # Arguments
/// * `func` - Function for which the root is desired.
/// * `der` - Derivative of the function.
/// * `sec_der` - Second derivative of the function.
/// * `start` - Initial guess for the root.
/// * `atol` - Absolute tolerance for convergence.
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
pub fn halley<T>(
    func: impl Fn(T) -> T,
    der: impl Fn(T) -> T,
    sec_der: impl Fn(T) -> T,
    start: T,
    atol: T,
) -> FittingResult<T>
where
    T: num_traits::Float + num_traits::ToPrimitive + num_traits::NumAssignOps,
{
    let mut x = start;

    let eps = T::epsilon() * T::from(1000.0).unwrap();
    let two = T::from(2.0).unwrap();

    // if the starting position has derivative of 0, nudge it a bit.
    if der(x).abs() < eps {
        x += T::from(0.1).unwrap();
    }

    let mut f_eval: T;
    let mut d_eval: T;
    let mut d_d_eval: T;
    let mut step: T;
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

        d_d_eval = sec_der(x);

        if !d_d_eval.is_finite() || !d_eval.is_finite() || !f_eval.is_finite() {
            Err(ConvergenceError::NonFinite)?;
        }
        step = f_eval / d_eval;
        step = step / (T::one() - step * d_d_eval / (two * d_eval));

        x -= step;
    }
    Err(ConvergenceError::Iterations)?
}

#[cfg(test)]
mod tests {
    use crate::fitting::halley;

    #[test]
    fn test_haley() {
        let f = |x: f64| 1.0 * x * x - 1.0;
        let d = |x| 2.0 * x;
        let dd = |_| 2.0;
        let root = halley(f, d, dd, 0.0, 1e-10).unwrap();
        assert!((root - 1.0).abs() < 1e-12);
    }
}
