//! # Halley's method
//!
//! Third order root finding algorithm.
//! This is the next order method of newton-raphson.
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
