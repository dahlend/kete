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

/// Find the minimum of a unimodal function on `[lo, hi]` using golden-section search.
///
/// Returns the x value at the minimum. The function must be unimodal on the
/// interval (exactly one local minimum).
///
/// ```
///     use kete_stats::fitting::golden_section_search;
///     let f = |x: f64| (x - 3.0) * (x - 3.0);
///     let x_min = golden_section_search(f, 0.0, 5.0, 1e-10).unwrap();
///     assert!((x_min - 3.0).abs() < 1e-9);
/// ```
///
/// # Errors
///
/// [`ConvergenceError`] may be returned in the following cases:
///     - Any function evaluation returns a non-finite value.
///     - Failed to converge within 200 iterations.
pub fn golden_section_search(
    mut func: impl FnMut(f64) -> f64,
    mut lo: f64,
    mut hi: f64,
    atol: f64,
) -> FittingResult<f64> {
    let gr = 0.5 * (5.0_f64.sqrt() - 1.0);

    let mut c = hi - gr * (hi - lo);
    let mut d = lo + gr * (hi - lo);

    for _ in 0..200 {
        if (hi - lo).abs() <= atol {
            return Ok(0.5 * (lo + hi));
        }

        let fc = func(c);
        let fd = func(d);

        if !fc.is_finite() || !fd.is_finite() {
            return Err(ConvergenceError::NonFinite);
        }

        if fc < fd {
            hi = d;
        } else {
            lo = c;
        }

        c = hi - gr * (hi - lo);
        d = lo + gr * (hi - lo);
    }

    Err(ConvergenceError::Iterations)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quadratic_minimum() {
        let f = |x: f64| (x - 3.0) * (x - 3.0);
        let x_min = golden_section_search(f, 0.0, 5.0, 1e-12).unwrap();
        assert!((x_min - 3.0).abs() < 1e-10, "Expected 3.0, got {x_min:.15}");
    }

    #[test]
    fn test_cos_minimum() {
        // cos(x) has minimum at pi on [0, 2*pi]
        let f = |x: f64| x.cos();
        let x_min = golden_section_search(f, 0.0, 2.0 * std::f64::consts::PI, 1e-12).unwrap();
        assert!(
            (x_min - std::f64::consts::PI).abs() < 1e-7,
            "Expected pi, got {x_min:.15}"
        );
    }

    #[test]
    fn test_narrow_bracket() {
        let f = |x: f64| (x - 1.0).powi(4);
        let x_min = golden_section_search(f, 0.99, 1.01, 1e-14).unwrap();
        assert!((x_min - 1.0).abs() < 1e-12, "Expected 1.0, got {x_min:.15}");
    }

    #[test]
    fn test_non_finite_returns_error() {
        let f = |x: f64| if x > 2.5 { f64::NAN } else { (x - 2.0).powi(2) };
        let result = golden_section_search(f, 0.0, 5.0, 1e-10);
        assert!(result.is_err());
    }
}
