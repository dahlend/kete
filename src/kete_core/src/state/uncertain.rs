//! Uncertain state representation: a best-fit Cartesian state plus a
//! covariance matrix that may span both the 6-element state and any
//! fitted force parameters.
//!
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

use crate::elements::CometElements;
use crate::frames::{CenterBody, DynCenter, Equatorial, InertialFrame};
use crate::prelude::{Error, KeteResult, State};
use nalgebra::DMatrix;
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};

/// A best-fit state together with a covariance matrix and zero or more
/// fitted free parameters.
///
/// The covariance is `(6 + Np) x (6 + Np)` where `Np = free_params.len()`.
/// Rows/cols 0-5 correspond to `[x, y, z, vx, vy, vz]`; rows/cols 6.. are
/// the free parameters in the same order as `free_params`. The semantic
/// labels of the free parameters live with the [`Force`] impls that
/// produce them; the state itself stores values only.
///
#[derive(Debug, Clone)]
pub struct UncertainState<F = Equatorial, C = DynCenter>
where
    F: InertialFrame,
    C: CenterBody,
    DynCenter: From<C>,
{
    /// Best-fit Cartesian state. Provides desig, epoch, and the 6
    /// position/velocity values.
    pub state: State<F, C>,

    /// Covariance matrix, `(6 + Np) x (6 + Np)`.
    pub cov_matrix: DMatrix<f64>,

    /// Current best estimates of the fitted parameters, length `Np`.
    /// `Np` may be zero (no fitted parameters), the typical case for a
    /// pure orbit-determination state.
    pub free_params: Vec<f64>,
}

impl<F, C> UncertainState<F, C>
where
    F: InertialFrame,
    C: CenterBody,
    DynCenter: From<C>,
{
    /// Construct an `UncertainState` from a Cartesian state, covariance,
    /// and free-parameter vector.
    ///
    /// # Errors
    /// Returns an error if `cov_matrix` is not
    /// `(6 + free_params.len()) x (6 + free_params.len())`.
    pub fn new(
        state: State<F, C>,
        cov_matrix: DMatrix<f64>,
        free_params: Vec<f64>,
    ) -> KeteResult<Self> {
        let expected = 6 + free_params.len();
        if cov_matrix.nrows() != expected || cov_matrix.ncols() != expected {
            return Err(Error::ValueError(format!(
                "Covariance matrix must be {expected}x{expected}, \
                 got {}x{}",
                cov_matrix.nrows(),
                cov_matrix.ncols()
            )));
        }
        Ok(Self {
            state,
            cov_matrix,
            free_params,
        })
    }
}

// `from_cometary` is constrained to the default `<Equatorial, DynCenter>`
// shape because it converts from `Ecliptic` and uses i32-based `State::new`.
// `sample` is generic and lives below.
impl UncertainState<Equatorial, DynCenter> {
    /// Construct an `UncertainState` from cometary orbital elements and
    /// a covariance expressed in element space.
    ///
    /// The cometary-element covariance is transformed to a Cartesian
    /// covariance via the numerically evaluated Jacobian
    /// `J = d(x,y,z,vx,vy,vz) / d(e,q,tp,node,w,i)`.
    ///
    /// When the covariance is larger than 6x6 (i.e. includes free
    /// parameters), the off-diagonal cross-terms are transformed by `J`
    /// and the parameter block is left unchanged.
    ///
    /// # Arguments
    /// * `elements` -- Cometary orbital elements with desig and epoch.
    /// * `cov_elem` -- Covariance in element space, `(6+Np) x (6+Np)`.
    ///   Row/column ordering:
    ///   0. eccentricity (dimensionless)
    ///   1. `peri_dist` (AU)
    ///   2. `peri_time` (JD, TDB)
    ///   3. `lon_of_ascending` (**radians**)
    ///   4. `peri_arg` (**radians**)
    ///   5. inclination (**radians**)
    ///   6. free parameters (if any)
    ///
    ///   Angular elements must be in radians, matching the units stored
    ///   in [`CometElements`].  If your source covariance is in degrees
    ///   (e.g. JPL Horizons), scale angular rows/columns by `pi/180`
    ///   before calling this function.
    /// * `free_params` -- Initial free-parameter values, length `Np`.
    ///   `Np = 0` is the no-parameter case.
    ///
    /// # Errors
    /// Returns an error if element-to-state conversion fails or if the
    /// covariance dimensions are inconsistent.
    pub fn from_cometary(
        elements: &CometElements,
        cov_elem: &DMatrix<f64>,
        free_params: Vec<f64>,
    ) -> KeteResult<Self> {
        let np = free_params.len();
        let expected = 6 + np;
        if cov_elem.nrows() != expected || cov_elem.ncols() != expected {
            return Err(Error::ValueError(format!(
                "Element covariance must be {expected}x{expected}, \
                 got {}x{}",
                cov_elem.nrows(),
                cov_elem.ncols()
            )));
        }

        // Nominal Cartesian state (Ecliptic -> Equatorial).
        let state_ecl = elements.try_to_state()?;
        let state: State<Equatorial> = state_ecl.into_frame();

        // Numerically compute the 6x6 Jacobian via finite differences.
        let jac = cometary_to_cartesian_jacobian(elements)?;

        // Transform the orbital-element covariance block.
        let c_elem_6x6 = cov_elem.view((0, 0), (6, 6));
        let c_cart = &jac * c_elem_6x6 * jac.transpose();

        if np == 0 {
            return Self::new(state, c_cart, free_params);
        }

        // Full (6+Np)x(6+Np) covariance with transformed blocks.
        let mut cov_cart = DMatrix::zeros(expected, expected);

        // Upper-left: Cartesian 6x6.
        cov_cart.view_mut((0, 0), (6, 6)).copy_from(&c_cart);

        // Off-diagonal: J * C_cross_elem  (6xNp block).
        let cross_elem = cov_elem.view((0, 6), (6, np));
        let cross_cart = &jac * cross_elem;
        cov_cart.view_mut((0, 6), (6, np)).copy_from(&cross_cart);
        cov_cart
            .view_mut((6, 0), (np, 6))
            .copy_from(&cross_cart.transpose());

        // Lower-right: parameter block unchanged.
        cov_cart
            .view_mut((6, 6), (np, np))
            .copy_from(&cov_elem.view((6, 6), (np, np)));

        Self::new(state, cov_cart, free_params)
    }
}

impl<F, C> UncertainState<F, C>
where
    F: InertialFrame,
    C: CenterBody,
    DynCenter: From<C>,
{
    /// Number of fitted free parameters (`free_params.len()`).
    #[must_use]
    pub fn n_free_params(&self) -> usize {
        self.free_params.len()
    }

    /// Draw random samples from the covariance distribution.
    ///
    /// Returns a vector of `(State, Vec<f64>)` pairs, where the second
    /// element is a perturbed copy of `free_params` with the same
    /// length. Perturbations are drawn from the multivariate normal
    /// defined by `cov_matrix`.
    ///
    /// # Arguments
    /// * `n_samples` -- Number of samples to draw.
    /// * `seed` -- Optional RNG seed for reproducibility.
    ///
    /// # Errors
    /// Returns an error if the covariance is not positive-semidefinite.
    pub fn sample(
        &self,
        n_samples: usize,
        seed: Option<u64>,
    ) -> KeteResult<Vec<(State<F, C>, Vec<f64>)>> {
        let n = self.cov_matrix.nrows();

        // Decompose using eigenvalues to handle positive semi-definite
        // matrices (e.g. when some parameters have zero variance).
        // C = V * diag(d) * V^T  ->  L = V * diag(sqrt(max(d,0)))
        // so that L * z produces samples in the non-null subspace.
        let sym = nalgebra::SymmetricEigen::new(self.cov_matrix.clone());
        let l = {
            let sqrt_diag = DMatrix::from_diagonal(
                &sym.eigenvalues
                    .map(|v| if v > 0.0 { v.sqrt() } else { 0.0 }),
            );
            &sym.eigenvectors * sqrt_diag
        };

        // Build RNG.
        let mut rng = match seed {
            Some(s) => rand::rngs::StdRng::seed_from_u64(s),
            None => rand::rngs::StdRng::from_seed(rand::random()),
        };

        // Nominal values: [x, y, z, vx, vy, vz, <free params...>]
        let np = self.free_params.len();
        let mut nominal = vec![
            self.state.pos[0],
            self.state.pos[1],
            self.state.pos[2],
            self.state.vel[0],
            self.state.vel[1],
            self.state.vel[2],
        ];
        nominal.extend(self.free_params.iter().copied());

        let mut results = Vec::with_capacity(n_samples);

        for _ in 0..n_samples {
            // Draw z ~ N(0, I) and compute delta = L * z.
            let z = nalgebra::DVector::from_fn(n, |_, _| StandardNormal.sample(&mut rng));
            let delta = &l * z;

            // Perturbed state.
            let sampled_state = State {
                desig: self.state.desig.clone(),
                epoch: self.state.epoch,
                pos: crate::frames::Vector::<F>::new([
                    nominal[0] + delta[0],
                    nominal[1] + delta[1],
                    nominal[2] + delta[2],
                ]),
                vel: crate::frames::Vector::<F>::new([
                    nominal[3] + delta[3],
                    nominal[4] + delta[4],
                    nominal[5] + delta[5],
                ]),
                center: self.state.center,
            };

            // Perturbed free params.
            let sampled_params: Vec<f64> = (0..np).map(|i| nominal[6 + i] + delta[6 + i]).collect();

            results.push((sampled_state, sampled_params));
        }

        Ok(results)
    }
}

/// Compute the 6x6 Jacobian `d(x,y,z,vx,vy,vz) / d(e,q,tp,node,w,i)`
/// by central finite differences on `CometElements::try_to_state()`.
///
/// The element ordering is: eccentricity, `peri_dist`, `peri_time`,
/// `lon_of_ascending`, `peri_arg`, inclination.
fn cometary_to_cartesian_jacobian(elements: &CometElements) -> KeteResult<DMatrix<f64>> {
    let mut jac = DMatrix::zeros(6, 6);

    // Central differences are optimal at h ~ eps^(1/3) * scale.
    // Most elements use their own magnitude (floored at 1.0 for near-zero
    // angles).  peri_time is special: its JD value is ~2.5e6, but orbit
    // sensitivity is per-day, so we use an absolute step of eps^(1/3) days.
    // ~6.06e-6
    let eps3 = f64::EPSILON.cbrt();
    let rel = |v: f64| eps3 * v.abs().max(1.0);
    let steps = [
        // eccentricity (dimensionless)
        rel(elements.eccentricity),
        // peri_dist (AU)
        rel(elements.peri_dist),
        // peri_time (days, absolute)
        eps3,
        // lon_of_ascending (rad)
        rel(elements.lon_of_ascending),
        // peri_arg (rad)
        rel(elements.peri_arg),
        // inclination (rad)
        rel(elements.inclination),
    ];

    for col in 0..6 {
        let h = steps[col];

        // For eccentricity near zero, a central difference would perturb
        // to negative e (which is unphysical).  Fall back to a forward
        // difference in that case (O(h) instead of O(h^2), but still
        // adequate for covariance transformation).
        let forward_only = col == 0 && elements.eccentricity < 2.0 * h;

        if forward_only {
            let elem_plus = perturb_element(elements, col, h);
            let state_plus: State<Equatorial> = elem_plus.try_to_state()?.into_frame();
            let state_nom: State<Equatorial> = elements.try_to_state()?.into_frame();

            let inv_h = 1.0 / h;
            for row in 0..3 {
                jac[(row, col)] = (state_plus.pos[row] - state_nom.pos[row]) * inv_h;
            }
            for row in 0..3 {
                jac[(row + 3, col)] = (state_plus.vel[row] - state_nom.vel[row]) * inv_h;
            }
        } else {
            let elem_plus = perturb_element(elements, col, h);
            let elem_minus = perturb_element(elements, col, -h);

            let state_plus: State<Equatorial> = elem_plus.try_to_state()?.into_frame();
            let state_minus: State<Equatorial> = elem_minus.try_to_state()?.into_frame();

            let inv_2h = 1.0 / (2.0 * h);
            for row in 0..3 {
                jac[(row, col)] = (state_plus.pos[row] - state_minus.pos[row]) * inv_2h;
            }
            for row in 0..3 {
                jac[(row + 3, col)] = (state_plus.vel[row] - state_minus.vel[row]) * inv_2h;
            }
        }
    }

    Ok(jac)
}

/// Return a copy of `elements` with the `col`-th element perturbed by `delta`.
///
/// Column mapping: 0=eccentricity, 1=`peri_dist`, 2=`peri_time`,
/// 3=`lon_of_ascending`, 4=`peri_arg`, 5=inclination.
fn perturb_element(elements: &CometElements, col: usize, delta: f64) -> CometElements {
    let mut e = elements.clone();
    match col {
        0 => e.eccentricity += delta,
        1 => e.peri_dist += delta,
        2 => e.peri_time = (e.peri_time.jd + delta).into(),
        3 => e.lon_of_ascending += delta,
        4 => e.peri_arg += delta,
        5 => e.inclination += delta,
        _ => unreachable!("column index must be 0..6"),
    }
    e
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::GMS_SQRT;
    use crate::frames::Ecliptic;
    use crate::prelude::Desig;
    use crate::time::Time;

    /// Helper: build a simple Earth-like state for testing.
    fn test_state() -> State<Equatorial> {
        State::new(
            Desig::Name("Test".into()),
            // J2000.0
            2451545.0,
            [1.0, 0.0, 0.0],
            // ~1 AU circular
            [0.0, 0.01720209895, 0.0],
            10,
        )
    }

    #[test]
    fn test_new_validates_dimensions() {
        let state = test_state();
        let cov_6x6 = DMatrix::identity(6, 6) * 1e-8;
        let result = UncertainState::new(state.clone(), cov_6x6, vec![]);
        assert!(result.is_ok());

        // Wrong size should fail.
        let cov_7x7 = DMatrix::identity(7, 7) * 1e-8;
        let result = UncertainState::new(state, cov_7x7, vec![]);
        assert!(result.is_err());
    }

    #[test]
    fn test_new_with_free_params_validates_dimensions() {
        let state = test_state();
        // 3 free params -> need 9x9.
        let cov_9x9 = DMatrix::identity(9, 9) * 1e-8;
        let result = UncertainState::new(state.clone(), cov_9x9, vec![1e-8, 2e-8, 3e-8]);
        assert!(result.is_ok());

        // 6x6 with 3 free params should fail.
        let cov_6x6 = DMatrix::identity(6, 6) * 1e-8;
        let result = UncertainState::new(state, cov_6x6, vec![1e-8, 2e-8, 3e-8]);
        assert!(result.is_err());
    }

    #[test]
    fn test_n_free_params() {
        let state = test_state();
        let cov = DMatrix::identity(6, 6) * 1e-8;
        let us = UncertainState::new(state.clone(), cov, vec![]).unwrap();
        assert_eq!(us.n_free_params(), 0);

        let cov = DMatrix::identity(9, 9) * 1e-8;
        let us = UncertainState::new(state, cov, vec![1.0, 2.0, 3.0]).unwrap();
        assert_eq!(us.n_free_params(), 3);
    }

    #[test]
    fn test_sample_no_free_params() {
        let state = test_state();
        let cov = DMatrix::identity(6, 6) * 1e-12;
        let us = UncertainState::new(state.clone(), cov, vec![]).unwrap();
        let samples = us.sample(100, Some(42)).unwrap();
        assert_eq!(samples.len(), 100);
        for (s, params) in &samples {
            assert!(params.is_empty());
            // Samples should be close to nominal with tiny covariance.
            assert!((s.pos[0] - state.pos[0]).abs() < 1e-3);
        }
    }

    #[test]
    fn test_sample_with_free_params() {
        let state = test_state();
        let nominal_params = vec![1e-8, 2e-8, 3e-8];
        let cov = DMatrix::identity(9, 9) * 1e-16;
        let us = UncertainState::new(state, cov, nominal_params.clone()).unwrap();
        let samples = us.sample(10, Some(42)).unwrap();
        for (_, params) in &samples {
            assert_eq!(params.len(), 3);
            // Tiny covariance -> sampled params close to nominal.
            for (sampled, nominal) in params.iter().zip(nominal_params.iter()) {
                assert!((sampled - nominal).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn test_sample_zero_covariance_returns_nominal() {
        let state = test_state();
        // Zero covariance (positive semi-definite, all eigenvalues zero).
        let cov = DMatrix::zeros(6, 6);
        let us = UncertainState::new(state.clone(), cov, vec![]).unwrap();
        let samples = us.sample(5, Some(42)).unwrap();
        assert_eq!(samples.len(), 5);
        // Every sample should equal the nominal state exactly.
        for (s, _) in &samples {
            for i in 0..3 {
                assert_eq!(s.pos[i], state.pos[i]);
                assert_eq!(s.vel[i], state.vel[i]);
            }
        }
    }

    #[test]
    fn test_from_cometary_round_trip() {
        // Build a state, convert to cometary elements, then round-trip
        // through from_cometary with an identity-like covariance.
        let state_eq = test_state();
        let state_ecl: State<Ecliptic> = state_eq.clone().into_frame();
        let elements = CometElements::from_state(&state_ecl);

        // Tiny diagonal covariance in element space.
        let cov_elem = DMatrix::identity(6, 6) * 1e-20;
        let us = UncertainState::from_cometary(&elements, &cov_elem, vec![]).unwrap();

        // The recovered state should match the original.
        for i in 0..3 {
            assert!(
                (us.state.pos[i] - state_eq.pos[i]).abs() < 1e-10,
                "pos[{i}] mismatch: {} vs {}",
                us.state.pos[i],
                state_eq.pos[i]
            );
            assert!(
                (us.state.vel[i] - state_eq.vel[i]).abs() < 1e-10,
                "vel[{i}] mismatch: {} vs {}",
                us.state.vel[i],
                state_eq.vel[i]
            );
        }
    }

    #[test]
    fn test_from_cometary_with_free_params() {
        let state_eq = test_state();
        let state_ecl: State<Ecliptic> = state_eq.into_frame();
        let elements = CometElements::from_state(&state_ecl);

        let cov_elem = DMatrix::identity(9, 9) * 1e-20;
        let us =
            UncertainState::from_cometary(&elements, &cov_elem, vec![1e-8, 2e-8, 3e-8]).unwrap();

        assert_eq!(us.cov_matrix.nrows(), 9);
        assert_eq!(us.cov_matrix.ncols(), 9);
        assert_eq!(us.free_params.len(), 3);
    }

    /// Validate the Jacobian by comparing `J * delta_elem` against the
    /// actual Cartesian-space change for a known perturbation.
    ///
    /// Uses a general elliptical orbit (e=0.3, q=1.5 AU, i=20 deg) with
    /// no special symmetries so every Jacobian column is exercised.
    #[test]
    fn test_jacobian_accuracy() {
        let epoch = Time::new(2460000.5);
        let elements = CometElements {
            desig: Desig::Empty,
            epoch,
            eccentricity: 0.3,
            peri_dist: 1.5,
            // 100 days before epoch
            peri_time: Time::new(2459900.5),
            // 45 deg
            lon_of_ascending: std::f64::consts::FRAC_PI_4,
            // 60 deg
            peri_arg: std::f64::consts::FRAC_PI_3,
            inclination: 20.0_f64.to_radians(),
            center_id: 10,
            gm_sqrt: GMS_SQRT,
        };

        let jac = cometary_to_cartesian_jacobian(&elements).unwrap();
        let nominal: State<Equatorial> = elements.try_to_state().unwrap().into_frame();

        // Test each column: apply a perturbation ~1e-4, predict the
        // Cartesian change with J, and compare against the true change.
        let elem_names = ["e", "q", "tp", "Omega", "omega", "i"];
        let perturbation = 1e-4;

        for col in 0..6 {
            let perturbed = perturb_element(&elements, col, perturbation);
            let state_p: State<Equatorial> = perturbed.try_to_state().unwrap().into_frame();

            for row in 0..6 {
                let (predicted, actual) = if row < 3 {
                    (
                        jac[(row, col)] * perturbation,
                        state_p.pos[row] - nominal.pos[row],
                    )
                } else {
                    (
                        jac[(row, col)] * perturbation,
                        state_p.vel[row - 3] - nominal.vel[row - 3],
                    )
                };

                // Allow 1% relative error (O(h^2) from linearity) or
                // 1e-14 absolute (for rows where the derivative is near
                // zero and float noise dominates).
                let err = (predicted - actual).abs();
                let tol = 0.01 * actual.abs() + 1e-14;
                assert!(
                    err < tol,
                    "Jacobian[{row},{}] (d cart / d {}): \
                     predicted={predicted:.6e}, actual={actual:.6e}, err={err:.2e}",
                    col,
                    elem_names[col]
                );
            }
        }
    }

    /// Same Jacobian test but for an equatorial orbit (`lon_of_ascending`
    /// near zero, `peri_arg` near zero) to exercise the step-size floor.
    #[test]
    fn test_jacobian_equatorial_orbit() {
        let epoch = Time::new(2460000.5);
        let elements = CometElements {
            desig: Desig::Empty,
            epoch,
            eccentricity: 0.05,
            peri_dist: 1.0,
            peri_time: Time::new(2459950.5),
            // nearly zero
            lon_of_ascending: 1e-6,
            // nearly zero
            peri_arg: 1e-6,
            // nearly equatorial
            inclination: 1e-4,
            center_id: 10,
            gm_sqrt: GMS_SQRT,
        };

        let jac = cometary_to_cartesian_jacobian(&elements).unwrap();
        let nominal: State<Equatorial> = elements.try_to_state().unwrap().into_frame();

        let perturbation = 1e-4;
        for col in 0..6 {
            let perturbed = perturb_element(&elements, col, perturbation);
            let state_p: State<Equatorial> = perturbed.try_to_state().unwrap().into_frame();

            for row in 0..6 {
                let (predicted, actual) = if row < 3 {
                    (
                        jac[(row, col)] * perturbation,
                        state_p.pos[row] - nominal.pos[row],
                    )
                } else {
                    (
                        jac[(row, col)] * perturbation,
                        state_p.vel[row - 3] - nominal.vel[row - 3],
                    )
                };

                let err = (predicted - actual).abs();
                let tol = 0.01 * actual.abs() + 1e-14;
                assert!(
                    err < tol,
                    "Equatorial Jacobian[{row},{col}]: \
                     predicted={predicted:.6e}, actual={actual:.6e}, err={err:.2e}"
                );
            }
        }
    }
}
