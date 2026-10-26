// SPDX-FileCopyrightText: 2026 Dar Dahlen
// SPDX-License-Identifier: BSD-3-Clause

//! Small dense linear algebra for the fits.

use nalgebra::{Cholesky, SMatrix, SVector, Vector3};

/// Sum of the squares of the residuals.
pub(crate) fn chi2<'a>(residuals: impl IntoIterator<Item = &'a Vector3<f64>>) -> f64 {
    residuals.into_iter().map(Vector3::norm_squared).sum()
}

/// The solution of a symmetric positive definite linear system, solved
/// scaled to a unit diagonal for its condition, with `damping` added to that
/// unit diagonal. `None` if it is not positive definite or not finite.
pub(crate) fn scaled_solve<const N: usize>(
    matrix: &SMatrix<f64, N, N>,
    rhs: &SVector<f64, N>,
    damping: f64,
) -> Option<SVector<f64, N>> {
    let (scaled, scale) = scaled(matrix, damping);
    let out = Cholesky::new(scaled)?
        .solve(&rhs.component_mul(&scale))
        .component_mul(&scale);
    out.iter().all(|x| x.is_finite()).then_some(out)
}

/// The inverse of a symmetric positive definite matrix, scaled as by
/// [`scaled_solve`]. `None` if it is not positive definite or not finite.
pub(crate) fn inverse<const N: usize>(matrix: &SMatrix<f64, N, N>) -> Option<SMatrix<f64, N, N>> {
    let (scaled, scale) = scaled(matrix, 0.0);
    let inverse = Cholesky::new(scaled)?.inverse();
    let out: SMatrix<f64, N, N> = SMatrix::from_fn(|i, j| inverse[(i, j)] * scale[i] * scale[j]);
    out.iter().all(|x| x.is_finite()).then_some(out)
}

/// `matrix` scaled to a unit diagonal, with `damping` added to it, and the
/// scale of each row and column, one over the square root of its diagonal.
fn scaled<const N: usize>(
    matrix: &SMatrix<f64, N, N>,
    damping: f64,
) -> (SMatrix<f64, N, N>, SVector<f64, N>) {
    let scale = matrix.diagonal().map(|x| 1.0 / x.sqrt());
    let mut scaled = SMatrix::from_fn(|i, j| matrix[(i, j)] * scale[i] * scale[j]);
    for i in 0..N {
        scaled[(i, i)] += damping;
    }
    (scaled, scale)
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Matrix3;

    #[test]
    fn solve_and_inverse() {
        // A badly scaled symmetric positive definite matrix, as the normal
        // matrix of position and velocity is.
        let matrix = Matrix3::new(4e6, 2e3, 1.0, 2e3, 3.0, 1e-3, 1.0, 1e-3, 2e-6);
        let rhs = Vector3::new(1.0, -2.0, 3e-3);
        let solved = scaled_solve(&matrix, &rhs, 0.0).unwrap();
        assert!((matrix * solved - rhs).norm() < 1e-8 * rhs.norm());
        let inverted = inverse(&matrix).unwrap();
        assert!((matrix * inverted - Matrix3::identity()).norm() < 1e-8);

        // Neither singular nor indefinite matrices have a solution.
        let singular = Matrix3::new(1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0);
        assert!(scaled_solve(&singular, &rhs, 0.0).is_none());
        let indefinite = Matrix3::new(1.0, 2.0, 0.0, 2.0, 1.0, 0.0, 0.0, 0.0, 1.0);
        assert!(scaled_solve(&indefinite, &rhs, 0.0).is_none());
        assert!(inverse(&indefinite).is_none());
    }
}
