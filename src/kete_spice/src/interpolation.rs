//! Interpolation methods used by Spice SPK Files.
//!
//! It is unlikely to be useful outside of reading these files.
//!
// BSD 3-Clause License
//
// Copyright (c) 2026, Dar Dahlen
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

use kete_core::{errors::Error, prelude::KeteResult};
use nalgebra::DVector;

/// Given a list of chebyshev polynomial coefficients, compute the value of the function
/// and it's derivative.
///
/// This is useful for reading values out of JPL SPK format, be aware though that time
/// scaling is also important for that particular use case.
///
/// This evaluates the coefficients at a single point of time, but for 3 sets of
/// coefficients at once. This is specifically done for performance reasons.
///
/// # Arguments
///
/// * `t`       - Time at which to evaluate the chebyshev polynomials.
/// * `coefx`    - Slice of coefficients of the chebyshev polynomials.
/// * `coefy`    - Slice of coefficients of the chebyshev polynomials.
/// * `coefz`    - Slice of coefficients of the chebyshev polynomials.
///
#[inline(always)]
pub(crate) fn chebyshev_evaluate_both(
    t: f64,
    coefx: &[f64],
    coefy: &[f64],
    coefz: &[f64],
) -> KeteResult<([f64; 3], [f64; 3])> {
    let n_coef = coefx.len();

    if n_coef < 2 {
        Err(Error::IOError(
            "File not formatted correctly. Chebyshev polynomial must be greater than order 2."
                .into(),
        ))?;
    }
    let x2 = 2.0 * t;

    let mut val = [
        coefx[0] + coefx[1] * t,
        coefy[0] + coefy[1] * t,
        coefz[0] + coefz[1] * t,
    ];
    let mut second_t = 1.0;
    let mut last_t = t;
    let mut next_t;

    // The derivative of the first kind is defined by the recurrence relation:
    // d T_i / dx = i * U_{i-1}
    let mut second_u = 1.0;
    let mut last_u = x2;
    let mut next_u;

    let mut der_val = [coefx[1], coefy[1], coefz[1]];

    for (idx, ((x, y), z)) in coefx.iter().zip(coefy).zip(coefz).enumerate().skip(2) {
        next_t = x2 * last_t - second_t;
        val[0] += x * next_t;
        val[1] += y * next_t;
        val[2] += z * next_t;

        second_t = last_t;
        last_t = next_t;

        next_u = x2 * last_u - second_u;
        der_val[0] += x * last_u * (idx as f64);
        der_val[1] += y * last_u * (idx as f64);
        der_val[2] += z * last_u * (idx as f64);

        second_u = last_u;
        last_u = next_u;
    }

    Ok((val, der_val))
}

/// Given a list of chebyshev polynomial coefficients, compute the value of the function
/// at the specified `x`.
///
/// This is useful for reading values out of JPL SPK format, be aware though that time
/// scaling is also important for that particular use case.
///
/// This evaluates the coefficients at a single point of time, but for 3 sets of
/// coefficients at once. This is specifically done for performance reasons.
///
/// # Arguments
///
/// * `t`       - Time at which to evaluate the chebyshev polynomials.
/// * `coefx`    - Slice of coefficients of the chebyshev polynomials.
/// * `coefy`    - Slice of coefficients of the chebyshev polynomials.
/// * `coefz`    - Slice of coefficients of the chebyshev polynomials.
///
#[inline(always)]
pub(crate) fn chebyshev_evaluate(
    t: f64,
    coefx: &[f64],
    coefy: &[f64],
    coefz: &[f64],
) -> KeteResult<[f64; 3]> {
    let n_coef = coefx.len();

    if n_coef < 2 {
        Err(Error::IOError(
            "File not formatted correctly. Chebyshev polynomial must be greater than order 2."
                .into(),
        ))?;
    }
    let x2 = 2.0 * t;

    let mut val = [
        coefx[0] + coefx[1] * t,
        coefy[0] + coefy[1] * t,
        coefz[0] + coefz[1] * t,
    ];
    let mut second_t = 1.0;
    let mut last_t = t;
    let mut next_t;

    for ((x, y), z) in coefx.iter().zip(coefy).zip(coefz).skip(2) {
        next_t = x2 * last_t - second_t;
        val[0] += x * next_t;
        val[1] += y * next_t;
        val[2] += z * next_t;

        second_t = last_t;
        last_t = next_t;
    }

    Ok(val)
}

/// Interpolate using Hermite interpolation.
///
/// # Arguments
///
/// * `times` - Times where the function `f` is evaluated at.
/// * `y_vals` - The values of the function `f` at the specified times.
/// * `dy` - The values of the derivative of the function `f`.
/// * `eval_time` - Time at which to evaluate the interpolation function.
#[inline(always)]
pub(crate) fn hermite_interpolation(
    times: &[f64],
    y: &[f64],
    dy: &[f64],
    eval_time: f64,
) -> (f64, f64) {
    debug_assert_eq!(times.len(), y.len(), "Input lengths must match");
    debug_assert_eq!(times.len(), dy.len(), "Input lengths must match");

    let n = y.len();

    let mut work = DVector::<f64>::zeros(2 * y.len());
    let mut d_work = DVector::<f64>::zeros(2 * y.len());
    for (idx, (y0, dy0)) in y.iter().zip(dy).enumerate() {
        work[2 * idx] = *y0;
        work[2 * idx + 1] = *dy0;
    }

    for idx in 1..n {
        let c1 = times[idx] - eval_time;
        let c2 = eval_time - times[idx - 1];
        let denom = times[idx] - times[idx - 1];

        let prev = 2 * idx - 2;
        let cur = prev + 1;
        let next = cur + 1;

        d_work[prev] = work[cur];
        d_work[cur] = (work[next] - work[prev]) / denom;

        let tmp = work[cur] * (eval_time - times[idx - 1]) + work[prev];
        work[cur] = (c1 * work[prev] + c2 * work[next]) / denom;
        work[prev] = tmp;
    }

    d_work[2 * n - 2] = work[2 * n - 1];
    work[2 * n - 2] += work[2 * n - 1] * (eval_time - times[n - 1]);

    for idj in 2..(2 * n) {
        for idi in 1..=(2 * n - idj) {
            let xi = idi.div_ceil(2);
            let xij = (idi + idj).div_ceil(2);
            let c1 = times[xij - 1] - eval_time;
            let c2 = eval_time - times[xi - 1];
            let denom = times[xij - 1] - times[xi - 1];

            d_work[idi - 1] =
                (c1 * d_work[idi - 1] + c2 * d_work[idi] + work[idi] - work[idi - 1]) / denom;
            work[idi - 1] = (c1 * work[idi - 1] + c2 * work[idi]) / denom;
        }
    }
    (work[0], d_work[0])
}

/// Fit three Chebyshev series from samples at the d+1 Gauss-Lobatto nodes.
///
/// Given d+1 function samples at the Chebyshev nodes of the second kind
/// (`tau_k = cos(pi*k/d)` for `k=0,...,d`), computes coefficients `c_j` such that
/// `sum_{j=0}^{d} c_j * T_j(tau_k) == samples[k]` for all k.
///
/// Samples must be provided in k=0,...,d order (tau from +1 down to -1).
/// Returns `[X_0...X_d, Y_0...Y_d, Z_0...Z_d]` in the same units as the inputs.
///
/// Uses the DCT-I inversion formula with half-weighted endpoints (k=0 and k=d).
/// The output is compatible with [`chebyshev_evaluate_both`] and [`chebyshev_evaluate`].
#[allow(
    clippy::cast_precision_loss,
    reason = "Indices at most 27; all values are exactly representable as f64."
)]
pub(crate) fn chebyshev_fit(x_samples: &[f64], y_samples: &[f64], z_samples: &[f64]) -> Vec<f64> {
    debug_assert_eq!(
        x_samples.len(),
        y_samples.len(),
        "sample slices must have equal length"
    );
    debug_assert_eq!(
        x_samples.len(),
        z_samples.len(),
        "sample slices must have equal length"
    );
    debug_assert!(
        x_samples.len() >= 2,
        "need at least 2 samples (degree >= 1)"
    );
    let n = x_samples.len(); // = degree + 1
    let d = n - 1;
    let inv_d = 1.0 / d as f64;

    let mut out = vec![0.0_f64; 3 * n];

    for j in 0..n {
        let mut sx = 0.0_f64;
        let mut sy = 0.0_f64;
        let mut sz = 0.0_f64;

        for k in 0..n {
            // DCT-I: half-weight on the two endpoints (k=0 and k=d).
            let weight = if k == 0 || k == d { 0.5 } else { 1.0 };
            let cos_val = (std::f64::consts::PI * (j * k) as f64 * inv_d).cos();
            sx += weight * x_samples[k] * cos_val;
            sy += weight * y_samples[k] * cos_val;
            sz += weight * z_samples[k] * cos_val;
        }

        // j=0 and j=d: scale by 1/d; interior j: scale by 2/d.
        let scale = if j == 0 || j == d { inv_d } else { 2.0 * inv_d };
        out[j] = sx * scale;
        out[n + j] = sy * scale;
        out[2 * n + j] = sz * scale;
    }

    out
}

/// Interpolate using lagrange interpolation.
///
/// # Arguments
///
/// * `times` - Times where the function `f` is evaluated at.
/// * `y_vals` - The values of the function `f` at the specified times.
/// * `eval_time` - Time at which to evaluate the interpolation function.
pub(crate) fn lagrange_interpolation(x: &[f64], y: &mut [f64], eval_time: f64) -> f64 {
    debug_assert_eq!(x.len(), y.len(), "Input lengths must match");

    // implementation of newton interpolation
    for idx in 1..x.len() {
        for idy in idx..x.len() {
            y[idy] = (y[idy] - y[idx - 1]) / (x[idy] - x[idx - 1]);
        }
    }
    let deg = x.len() - 1;
    let mut val = y[deg];
    for k in 1..=deg {
        val = y[deg - k] + (eval_time - x[deg - k]) * val;
    }
    val
}

#[cfg(test)]
mod tests {
    use super::{chebyshev_evaluate_both, chebyshev_fit, lagrange_interpolation};

    #[test]
    fn test_chebyshev_fit_degree1() {
        // Degree 1: linear, two Gauss-Lobatto nodes at tau=+1 and tau=-1.
        // x_samples[0] = f(+1) = 3, x_samples[1] = f(-1) = 7
        // => c0 = (3+7)/2 = 5, c1 = (3-7)/2 = -2
        let xs = vec![3.0, 7.0];
        let ys = vec![0.0, 0.0];
        let zs = vec![0.0, 0.0];
        let c = chebyshev_fit(&xs, &ys, &zs);
        assert!((c[0] - 5.0).abs() < 1e-14, "c0={}", c[0]);
        assert!((c[1] - (-2.0)).abs() < 1e-14, "c1={}", c[1]);

        let (v1, _) = chebyshev_evaluate_both(1.0, &c[..2], &c[2..4], &c[4..]).unwrap();
        assert!((v1[0] - 3.0).abs() < 1e-14);
        let (v2, _) = chebyshev_evaluate_both(-1.0, &c[..2], &c[2..4], &c[4..]).unwrap();
        assert!((v2[0] - 7.0).abs() < 1e-14);
    }

    #[test]
    fn test_chebyshev_fit_roundtrip() {
        // Fit a cubic polynomial and verify round-trip at all nodes and at an interior point.
        use std::f64::consts::PI;
        let degree = 8;
        let n = degree + 1;
        let f = |t: f64| 1.0 + 2.0 * t - 3.0 * t * t + 0.5 * t * t * t;

        #[allow(
            clippy::cast_precision_loss,
            reason = "integer index cast to f64 for trigonometric node computation"
        )]
        let taus: Vec<f64> = (0..n)
            .map(|k| (PI * k as f64 / degree as f64).cos())
            .collect();
        let xs: Vec<f64> = taus.iter().map(|&t| f(t)).collect();
        let ys: Vec<f64> = taus.iter().map(|&t| f(t) * 2.0).collect();
        let zs: Vec<f64> = taus.iter().map(|&t| -f(t)).collect();

        let c = chebyshev_fit(&xs, &ys, &zs);

        // Residual at every fitting node must be near floating-point rounding.
        for (k, &tau) in taus.iter().enumerate() {
            let (val, _) =
                chebyshev_evaluate_both(tau, &c[..n], &c[n..2 * n], &c[2 * n..]).unwrap();
            assert!(
                (val[0] - xs[k]).abs() < 1e-12,
                "x node residual k={k}: {}",
                val[0] - xs[k]
            );
            assert!(
                (val[1] - ys[k]).abs() < 1e-12,
                "y node residual k={k}: {}",
                val[1] - ys[k]
            );
            assert!(
                (val[2] - zs[k]).abs() < 1e-12,
                "z node residual k={k}: {}",
                val[2] - zs[k]
            );
        }

        // Also check at a non-node point.
        let t_test = 0.3_f64;
        let (val, _) = chebyshev_evaluate_both(t_test, &c[..n], &c[n..2 * n], &c[2 * n..]).unwrap();
        assert!((val[0] - f(t_test)).abs() < 1e-12);
        assert!((val[1] - 2.0 * f(t_test)).abs() < 1e-12);
        assert!((val[2] - (-f(t_test))).abs() < 1e-12);
    }

    #[test]
    fn test_lagrange_interpolation() {
        let times = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let y = times.clone();

        for v in 0..100 {
            let eval_time = f64::from(v) / 100. * 9.0;
            let interp = lagrange_interpolation(&times, &mut y.clone(), eval_time);
            assert!((interp - eval_time).abs() < 1e-12);
        }

        let y1: Vec<_> = times
            .iter()
            .map(|x| x + 1.75 * x.powi(2) - 3.0 * x.powi(3) - 11.0 * x.powi(4))
            .collect();

        for v in 0..100 {
            let x = f64::from(v) / 100. * 9.0;
            let expected = x + 1.75 * x.powi(2) - 3.0 * x.powi(3) - 11.0 * x.powi(4);
            let interp = lagrange_interpolation(&times, &mut y1.clone(), x);
            assert!(
                (interp - expected).abs() < 1e-10,
                "x={} interp={} expected={} diff={}",
                x,
                interp,
                expected,
                interp - expected
            );
        }
    }
}
