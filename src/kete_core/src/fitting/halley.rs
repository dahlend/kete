//! # Halley's method
//!
//! Third order root finding algorithm.
//! This is the next order method of newton-raphson.
//
// BSD 3-Clause License
//
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

use crate::{errors::Error, prelude::KeteResult};

/// Solve root using Halley's method.
///
/// This accepts a three functions, the first being a single input function for which
/// the root is desired. The second function being the derivative of the first with
/// respect to the input variable. The third is the third derivative.
///
/// ```
///     use kete_core::fitting::halley;
///     let f = |x| { 1.0 * x * x - 1.0 };
///     let d = |x| { 2.0 * x };
///     let dd = |_| { 2.0};
///     let root = halley(f, d, dd, 0.0, 1e-10).unwrap();
///     assert!((root - 1.0).abs() < 1e-12);
/// ```
///
#[inline(always)]
pub fn halley<Func, Der, SecDer>(
    func: Func,
    der: Der,
    sec_der: SecDer,
    start: f64,
    atol: f64,
) -> KeteResult<f64>
where
    Func: Fn(f64) -> f64,
    Der: Fn(f64) -> f64,
    SecDer: Fn(f64) -> f64,
{
    let mut x = start;

    // if the starting position has derivative of 0, nudge it a bit.
    if der(x).abs() < 1e-12 {
        x += 0.1;
    }

    let mut f_eval: f64;
    let mut d_eval: f64;
    let mut dd_eval: f64;
    let mut step: f64;
    for _ in 0..100 {
        f_eval = func(x);
        if f_eval.abs() < atol {
            return Ok(x);
        }
        d_eval = der(x);

        // Derivative is 0, cannot solve
        if d_eval.abs() < 1e-12 {
            Err(Error::Convergence(
                "Halley's root finding failed to converge due to zero derivative.".into(),
            ))?;
        }

        dd_eval = sec_der(x);

        if !dd_eval.is_finite() || !d_eval.is_finite() || !f_eval.is_finite() {
            Err(Error::Convergence(
                "Halley root finding failed to converge due to non-finite evaluations".into(),
            ))?;
        }
        step = f_eval / d_eval;
        step = step / (1.0 - step * dd_eval / (2.0 * d_eval));

        x -= step;
    }
    Err(Error::Convergence(
        "Halley's root finding hit iteration limit without converging.".into(),
    ))?
}

#[cfg(test)]
mod tests {
    use crate::fitting::halley;

    #[test]
    fn test_haley() {
        let f = |x| 1.0 * x * x - 1.0;
        let d = |x| 2.0 * x;
        let dd = |_| 2.0;
        let root = halley(f, d, dd, 0.0, 1e-10).unwrap();
        assert!((root - 1.0).abs() < 1e-12);
    }
}
