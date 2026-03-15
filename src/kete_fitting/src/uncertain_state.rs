//! Uncertain state representation: a best-fit Cartesian state plus a
//! covariance matrix that may span both the 6-element state and any
//! fitted non-gravitational parameters.
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

use kete_core::elements::CometElements;
use kete_core::frames::Equatorial;
use kete_core::prelude::{Error, KeteResult, State};
use kete_core::propagation::NonGravModel;
use nalgebra::DMatrix;
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};

/// A best-fit state together with a covariance matrix.
///
/// The covariance is always expressed in the Cartesian frame
/// (x, y, z, vx, vy, vz) followed by any non-grav free parameters
/// in the order given by [`NonGravModel::param_names`].
///
/// The total matrix size is `(6 + Np) x (6 + Np)` where `Np` is the
/// number of free non-grav parameters (0, 1, or 3).
#[derive(Debug, Clone)]
pub struct UncertainState {
    /// Best-fit state (Sun-centered Equatorial Cartesian).
    /// Provides desig, epoch, and the 6 position/velocity values.
    pub state: State<Equatorial>,

    /// Covariance matrix, (6 + Np) x (6 + Np).
    /// Rows/cols 0-5 correspond to [x, y, z, vx, vy, vz].
    /// Rows/cols 6.. correspond to `non_grav.param_names()` order.
    pub cov_matrix: DMatrix<f64>,

    /// [`NonGravModel`] with both fitted and fixed parameter values.
    /// When sampling, this is cloned and the free params are
    /// overwritten with sampled values via `set_free_params()`.
    /// Also defines the covariance column layout beyond col 5.
    pub non_grav: Option<NonGravModel>,
}

impl UncertainState {
    /// Construct an `UncertainState` directly from a Cartesian state and
    /// covariance matrix.
    ///
    /// # Errors
    /// Returns an error if `cov_matrix` dimensions do not match the
    /// expected `(6 + n_free_params) x (6 + n_free_params)`.
    pub fn new(
        state: State<Equatorial>,
        cov_matrix: DMatrix<f64>,
        non_grav: Option<NonGravModel>,
    ) -> KeteResult<Self> {
        let np = non_grav.as_ref().map_or(0, NonGravModel::n_free_params);
        let expected = 6 + np;
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
            non_grav,
        })
    }

    /// Construct an `UncertainState` from cometary orbital elements and
    /// a covariance expressed in element space.
    ///
    /// The cometary-element covariance is transformed to a Cartesian
    /// covariance via the numerically evaluated Jacobian
    /// `J = d(x,y,z,vx,vy,vz) / d(e,q,tp,node,w,i)`.
    ///
    /// When the covariance is larger than 6x6 (i.e. includes non-grav
    /// parameters), the off-diagonal cross-terms are transformed by `J`
    /// and the non-grav block is left unchanged.
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
    ///   6. non-grav free parameters (if any)
    ///
    ///   Angular elements must be in radians, matching the units stored
    ///   in [`CometElements`].  If your source covariance is in degrees
    ///   (e.g. JPL Horizons), scale angular rows/columns by `pi/180`
    ///   before calling this function.
    /// * `non_grav` -- Optional non-gravitational model.
    ///
    /// # Errors
    /// Returns an error if element-to-state conversion fails or if the
    /// covariance dimensions are inconsistent.
    pub fn from_cometary(
        elements: &CometElements,
        cov_elem: &DMatrix<f64>,
        non_grav: Option<NonGravModel>,
    ) -> KeteResult<Self> {
        let np = non_grav.as_ref().map_or(0, NonGravModel::n_free_params);
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
            return Self::new(state, c_cart, non_grav);
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

        // Lower-right: non-grav block unchanged.
        cov_cart
            .view_mut((6, 6), (np, np))
            .copy_from(&cov_elem.view((6, 6), (np, np)));

        Self::new(state, cov_cart, non_grav)
    }

    /// Draw random samples from the covariance distribution.
    ///
    /// Returns a vector of `(State, Option<NonGravModel>)` pairs.
    /// Perturbations to the state position and velocity are drawn from
    /// the multivariate normal defined by `cov_matrix`.
    ///
    /// # Arguments
    /// * `n_samples` -- Number of samples to draw.
    /// * `seed` -- Optional RNG seed for reproducibility.
    ///
    /// # Errors
    /// Returns an error if the covariance is not positive-definite.
    pub fn sample(
        &self,
        n_samples: usize,
        seed: Option<u64>,
    ) -> KeteResult<Vec<(State<Equatorial>, Option<NonGravModel>)>> {
        let n = self.cov_matrix.nrows();

        // Decompose using eigenvalues to handle positive semi-definite
        // matrices (e.g. when some non-grav params have zero variance).
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

        // Nominal values: [x, y, z, vx, vy, vz, <nongrav params...>]
        let mut nominal = vec![
            self.state.pos[0],
            self.state.pos[1],
            self.state.pos[2],
            self.state.vel[0],
            self.state.vel[1],
            self.state.vel[2],
        ];
        if let Some(ref ng) = self.non_grav {
            nominal.extend(ng.get_free_params());
        }

        let mut results = Vec::with_capacity(n_samples);

        for _ in 0..n_samples {
            // Draw z ~ N(0, I) and compute delta = L * z.
            let z = nalgebra::DVector::from_fn(n, |_, _| StandardNormal.sample(&mut rng));
            let delta = &l * z;

            // Perturbed state.
            let sampled_state = State::new(
                self.state.desig.clone(),
                self.state.epoch,
                [
                    nominal[0] + delta[0],
                    nominal[1] + delta[1],
                    nominal[2] + delta[2],
                ]
                .into(),
                [
                    nominal[3] + delta[3],
                    nominal[4] + delta[4],
                    nominal[5] + delta[5],
                ]
                .into(),
                self.state.center_id,
            );

            // Perturbed non-grav (if any).
            let sampled_ng = self.non_grav.as_ref().map(|ng| {
                let mut ng_clone = ng.clone();
                let np = ng.n_free_params();
                let params: Vec<f64> = (0..np).map(|i| nominal[6 + i] + delta[6 + i]).collect();
                ng_clone.set_free_params(&params);
                ng_clone
            });

            results.push((sampled_state, sampled_ng));
        }

        Ok(results)
    }

    /// Names of all parameters represented in the covariance matrix,
    /// in row/column order.
    ///
    /// Always starts with `["x", "y", "z", "vx", "vy", "vz"]`, followed
    /// by the non-grav parameter names if present.
    #[must_use]
    pub fn param_names(&self) -> Vec<&str> {
        let mut names = vec!["x", "y", "z", "vx", "vy", "vz"];
        if let Some(ref ng) = self.non_grav {
            names.extend(ng.param_names());
        }
        names
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
    use kete_core::prelude::Desig;
    use kete_core::time::Time;

    /// Helper: build a simple Earth-like state for testing.
    fn test_state() -> State<Equatorial> {
        State::new(
            Desig::Name("Test".into()),
            // J2000.0
            Time::new(2451545.0),
            [1.0, 0.0, 0.0].into(),
            // ~1 AU circular
            [0.0, 0.01720209895, 0.0].into(),
            10,
        )
    }

    #[test]
    fn test_new_validates_dimensions() {
        let state = test_state();
        let cov_6x6 = DMatrix::identity(6, 6) * 1e-8;
        let result = UncertainState::new(state.clone(), cov_6x6, None);
        assert!(result.is_ok());

        // Wrong size should fail.
        let cov_7x7 = DMatrix::identity(7, 7) * 1e-8;
        let result = UncertainState::new(state, cov_7x7, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_new_with_nongrav_validates_dimensions() {
        let state = test_state();
        let ng = NonGravModel::new_jpl_comet_default(1e-8, 2e-8, 3e-8);
        // JplComet has 3 free params, so need 9x9.
        let cov_9x9 = DMatrix::identity(9, 9) * 1e-8;
        let result = UncertainState::new(state.clone(), cov_9x9, Some(ng.clone()));
        assert!(result.is_ok());

        // 6x6 should fail with JplComet.
        let cov_6x6 = DMatrix::identity(6, 6) * 1e-8;
        let result = UncertainState::new(state, cov_6x6, Some(ng));
        assert!(result.is_err());
    }

    #[test]
    fn test_param_names_no_nongrav() {
        let state = test_state();
        let cov = DMatrix::identity(6, 6) * 1e-8;
        let us = UncertainState::new(state, cov, None).unwrap();
        assert_eq!(us.param_names(), vec!["x", "y", "z", "vx", "vy", "vz"]);
    }

    #[test]
    fn test_param_names_jpl_comet() {
        let state = test_state();
        let ng = NonGravModel::new_jpl_comet_default(0.0, 0.0, 0.0);
        let cov = DMatrix::identity(9, 9) * 1e-8;
        let us = UncertainState::new(state, cov, Some(ng)).unwrap();
        assert_eq!(
            us.param_names(),
            vec!["x", "y", "z", "vx", "vy", "vz", "a1", "a2", "a3"]
        );
    }

    #[test]
    fn test_param_names_dust() {
        let state = test_state();
        let ng = NonGravModel::new_dust(0.01);
        let cov = DMatrix::identity(7, 7) * 1e-8;
        let us = UncertainState::new(state, cov, Some(ng)).unwrap();
        assert_eq!(
            us.param_names(),
            vec!["x", "y", "z", "vx", "vy", "vz", "beta"]
        );
    }

    #[test]
    fn test_sample_no_nongrav() {
        let state = test_state();
        let cov = DMatrix::identity(6, 6) * 1e-12;
        let us = UncertainState::new(state.clone(), cov, None).unwrap();
        let samples = us.sample(100, Some(42)).unwrap();
        assert_eq!(samples.len(), 100);
        for (s, ng) in &samples {
            assert!(ng.is_none());
            // Samples should be close to nominal with tiny covariance.
            assert!((s.pos[0] - state.pos[0]).abs() < 1e-3);
        }
    }

    #[test]
    fn test_sample_with_nongrav_preserves_fixed_params() {
        let state = test_state();
        let custom_alpha = 0.5;
        let ng = NonGravModel::new_jpl(
            1e-8,
            2e-8,
            3e-8,
            custom_alpha,
            2.808,
            2.15,
            5.093,
            4.6142,
            0.0,
        );
        let cov = DMatrix::identity(9, 9) * 1e-16;
        let us = UncertainState::new(state, cov, Some(ng)).unwrap();
        let samples = us.sample(10, Some(42)).unwrap();
        for (_, ng_opt) in &samples {
            let ng = ng_opt.as_ref().unwrap();
            // The fixed alpha should be preserved through sampling.
            match ng {
                NonGravModel::JplComet { alpha, .. } => {
                    assert!(
                        (*alpha - custom_alpha).abs() < f64::EPSILON,
                        "alpha was {alpha}, expected {custom_alpha}"
                    );
                }
                NonGravModel::Dust { .. } => panic!("Expected JplComet variant"),
            }
        }
    }

    #[test]
    fn test_sample_zero_covariance_returns_nominal() {
        let state = test_state();
        // Zero covariance (positive semi-definite, all eigenvalues zero).
        let cov = DMatrix::zeros(6, 6);
        let us = UncertainState::new(state.clone(), cov, None).unwrap();
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
        use kete_core::frames::Ecliptic;

        // Build a state, convert to cometary elements, then round-trip
        // through from_cometary with an identity-like covariance.
        let state_eq = test_state();
        let state_ecl: State<Ecliptic> = state_eq.clone().into_frame();
        let elements = CometElements::from_state(&state_ecl);

        // Tiny diagonal covariance in element space.
        let cov_elem = DMatrix::identity(6, 6) * 1e-20;
        let us = UncertainState::from_cometary(&elements, &cov_elem, None).unwrap();

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
    fn test_from_cometary_with_nongrav() {
        use kete_core::frames::Ecliptic;

        let state_eq = test_state();
        let state_ecl: State<Ecliptic> = state_eq.into_frame();
        let elements = CometElements::from_state(&state_ecl);

        let ng = NonGravModel::new_jpl_comet_default(1e-8, 2e-8, 3e-8);
        let cov_elem = DMatrix::identity(9, 9) * 1e-20;
        let us = UncertainState::from_cometary(&elements, &cov_elem, Some(ng)).unwrap();

        assert_eq!(us.cov_matrix.nrows(), 9);
        assert_eq!(us.cov_matrix.ncols(), 9);
        assert!(us.non_grav.is_some());
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
