// SPDX-FileCopyrightText: 2026 Dar Dahlen
// SPDX-License-Identifier: BSD-3-Clause

//! Small dense linear algebra for the fits.

/// Dot product of two vectors of three numbers.
pub(crate) fn dot(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Sum of the squares of the residuals.
pub(crate) fn chi2<'a>(residuals: impl IntoIterator<Item = &'a [f64; 3]>) -> f64 {
    residuals.into_iter().map(|r| dot(r, r)).sum()
}

/// The solution of a symmetric positive definite linear system, solved
/// scaled to a unit diagonal for its condition, with `damping` added to that
/// unit diagonal. `None` if it is not finite, as for a singular system.
pub(crate) fn scaled_solve<const N: usize>(
    matrix: &[[f64; N]; N],
    rhs: [f64; N],
    damping: f64,
) -> Option<[f64; N]> {
    let scale: [f64; N] = std::array::from_fn(|i| 1.0 / matrix[i][i].sqrt());
    let mut scaled = [[0.0; N]; N];
    for i in 0..N {
        for j in 0..N {
            scaled[i][j] = matrix[i][j] * scale[i] * scale[j];
        }
        scaled[i][i] += damping;
    }
    let out = solve(scaled, std::array::from_fn(|i| rhs[i] * scale[i]))?;
    Some(std::array::from_fn(|i| out[i] * scale[i]))
}

/// The inverse of a symmetric positive definite matrix, by
/// [`scaled_solve`].
pub(crate) fn inverse<const N: usize>(matrix: &[[f64; N]; N]) -> Option<[[f64; N]; N]> {
    let mut out = [[0.0; N]; N];
    for j in 0..N {
        let mut unit = [0.0; N];
        unit[j] = 1.0;
        let column = scaled_solve(matrix, unit, 0.0)?;
        for i in 0..N {
            out[i][j] = column[i];
        }
    }
    Some(out)
}

/// The solution of a symmetric positive definite linear system, by Gaussian
/// elimination. `None` if it is not finite, as for a singular system.
fn solve<const N: usize>(mut matrix: [[f64; N]; N], mut rhs: [f64; N]) -> Option<[f64; N]> {
    for col in 0..N {
        let pivot = matrix[col];
        for row in col + 1..N {
            let factor = matrix[row][col] / pivot[col];
            for (value, above) in matrix[row].iter_mut().zip(&pivot).skip(col) {
                *value -= factor * above;
            }
            rhs[row] -= factor * rhs[col];
        }
    }
    let mut out = [0.0; N];
    for row in (0..N).rev() {
        let rest: f64 = (row + 1..N).map(|k| matrix[row][k] * out[k]).sum();
        out[row] = (rhs[row] - rest) / matrix[row][row];
    }
    out.iter().all(|x| x.is_finite()).then_some(out)
}
