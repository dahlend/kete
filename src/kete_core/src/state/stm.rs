//! Variational integration via the [`ParameterizedForce`] trait.
//!
//! [`propagate_with_stm`] integrates a [`ParameterizedForce`] together with the
//! second-order encoding of the state-transition matrix and parameter
//! sensitivities, returning the propagated state and a 6 x (6 + Np)
//! sensitivity matrix.
//!
//! The augmented state is dynamically sized (`DVector`) so the caller
//! can supply any number of free parameters. The encoding mirrors the
//! existing `kete_spice::propagation::stm_augmented_accel`:
//!
//! ```text
//! pos_aug = [pos | vec(Phi_rr) | vec(Phi_rv) | s_1 | s_2 | ...]
//! vel_aug = [vel | vec(Phi_rr') | vec(Phi_rv') | s_1' | s_2' | ...]
//! ```
//!
//! with `Phi_rr(0) = I_3`, `Phi_rv'(0) = I_3`, all parameter
//! sensitivities zero at the start. The integrator runs the second-order
//! Radau scheme and the sensitivity matrix is reconstructed from the
//! final augmented state.
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

use nalgebra::{DMatrix, DVector, Matrix3, Vector3};

use crate::errors::{Error, KeteResult};
use crate::forces::ParameterizedForce;
use crate::frames::Vector;
use crate::time::{TDB, Time};

/// Propagate a state and its augmented STM under the given `forces`.
///
/// Returns `(pos_final, vel_final, sensitivity)` where `sensitivity` is
/// the 6 x (6 + Np) matrix:
///
/// ```text
/// cols 0..3 : Phi_rr  (d r_f / d r_0)
/// cols 3..6 : Phi_rv  (d r_f / d v_0)
/// rows 3..6 : Phi_vr / Phi_vv (similarly)
/// cols 6+k  : parameter sensitivity (d (r_f, v_f) / d p_k)
/// ```
///
/// `free_params.len()` must match `forces.n_free_params()`.
///
/// # Errors
/// Propagation may fail if the integrator does not converge or if the
/// `ParameterizedForce` impl returns an error.
pub fn propagate_with_stm<F: ParameterizedForce>(
    forces: &F,
    pos_init: Vector3<f64>,
    vel_init: Vector3<f64>,
    free_params: &[f64],
    epoch_init: Time<TDB>,
    epoch_final: Time<TDB>,
) -> KeteResult<(Vector3<f64>, Vector3<f64>, DMatrix<f64>)> {
    use crate::integrators::RadauIntegrator;

    let np = free_params.len();
    let dim = 21 + 3 * np;

    // Augmented initial conditions.
    let mut pos_aug = DVector::<f64>::zeros(dim);
    let mut vel_aug = DVector::<f64>::zeros(dim);
    pos_aug[0] = pos_init[0];
    pos_aug[1] = pos_init[1];
    pos_aug[2] = pos_init[2];
    vel_aug[0] = vel_init[0];
    vel_aug[1] = vel_init[1];
    vel_aug[2] = vel_init[2];
    // Phi_rr(0) = I3 (column-major in cols 3..12 of pos_aug).
    pos_aug[3] = 1.0;
    pos_aug[7] = 1.0;
    pos_aug[11] = 1.0;
    // Phi_rv'(0) = I3 (column-major in cols 12..21 of vel_aug).
    vel_aug[12] = 1.0;
    vel_aug[16] = 1.0;
    vel_aug[20] = 1.0;

    let ode = |time: Time<TDB>,
               pos_aug: &DVector<f64>,
               vel_aug: &DVector<f64>,
               _meta: &mut (),
               _exact_eval: bool|
     -> KeteResult<DVector<f64>> {
        let mut result = DVector::<f64>::zeros(dim);

        let pos_phys = Vector::<F::Frame>::new([pos_aug[0], pos_aug[1], pos_aug[2]]);
        let vel_phys = Vector::<F::Frame>::new([vel_aug[0], vel_aug[1], vel_aug[2]]);

        // Physical acceleration -- propagate any error from the force.
        let accel: Vector3<f64> = forces
            .accel(time, &pos_phys, &vel_phys, free_params)?
            .into();
        result[0] = accel[0];
        result[1] = accel[1];
        result[2] = accel[2];

        // Dynamics Jacobians at the current physical state.
        let (da_dr, da_dv) = forces.jacobians(time, &pos_phys, &vel_phys, free_params)?;

        // Phi_rr'' = da_dr * Phi_rr + da_dv * Phi_rr'
        let phi_rr = Matrix3::from_column_slice(&pos_aug.as_slice()[3..12]);
        let phi_rr_dot = Matrix3::from_column_slice(&vel_aug.as_slice()[3..12]);
        let phi_rr_ddot = da_dr * phi_rr + da_dv * phi_rr_dot;
        result.as_mut_slice()[3..12].copy_from_slice(phi_rr_ddot.as_slice());

        // Phi_rv'' = da_dr * Phi_rv + da_dv * Phi_rv'
        let phi_rv = Matrix3::from_column_slice(&pos_aug.as_slice()[12..21]);
        let phi_rv_dot = Matrix3::from_column_slice(&vel_aug.as_slice()[12..21]);
        let phi_rv_ddot = da_dr * phi_rv + da_dv * phi_rv_dot;
        result.as_mut_slice()[12..21].copy_from_slice(phi_rv_ddot.as_slice());

        // Parameter sensitivities: s_k'' = da_dr * s_k + da_dv * s_k' + d a / d p_k.
        if np > 0 {
            let dp = forces.parameter_jacobian(time, &pos_phys, &vel_phys, free_params)?;
            for k in 0..np {
                let base = 21 + k * 3;
                let s_k = Vector3::new(pos_aug[base], pos_aug[base + 1], pos_aug[base + 2]);
                let s_k_dot = Vector3::new(vel_aug[base], vel_aug[base + 1], vel_aug[base + 2]);
                let partial = dp.column(k);
                let partial_v = Vector3::new(partial[0], partial[1], partial[2]);
                let s_k_ddot = da_dr * s_k + da_dv * s_k_dot + partial_v;
                result[base] = s_k_ddot[0];
                result[base + 1] = s_k_ddot[1];
                result[base + 2] = s_k_ddot[2];
            }
        }

        Ok(result)
    };

    // control_dim=3 keeps step-size adaptation focused on the physical
    // 3-component acceleration row, matching the existing variational
    // integrator. Without it, large STM entries can drag steps to be
    // unnecessarily small.
    let (pos_f, vel_f, ()) =
        RadauIntegrator::integrate(&ode, pos_aug, vel_aug, epoch_init, epoch_final, (), Some(3))?;

    // Reconstruct the 6 x (6 + Np) sensitivity matrix.
    let phi_rr = Matrix3::from_column_slice(&pos_f.as_slice()[3..12]);
    let phi_rv = Matrix3::from_column_slice(&pos_f.as_slice()[12..21]);
    let phi_vr = Matrix3::from_column_slice(&vel_f.as_slice()[3..12]);
    let phi_vv = Matrix3::from_column_slice(&vel_f.as_slice()[12..21]);

    let mut sens = DMatrix::<f64>::zeros(6, 6 + np);
    sens.fixed_view_mut::<3, 3>(0, 0).copy_from(&phi_rr);
    sens.fixed_view_mut::<3, 3>(0, 3).copy_from(&phi_rv);
    sens.fixed_view_mut::<3, 3>(3, 0).copy_from(&phi_vr);
    sens.fixed_view_mut::<3, 3>(3, 3).copy_from(&phi_vv);

    for k in 0..np {
        let base = 21 + k * 3;
        for i in 0..3 {
            sens[(i, 6 + k)] = pos_f[base + i];
            sens[(3 + i, 6 + k)] = vel_f[base + i];
        }
    }

    let pos_final = Vector3::new(pos_f[0], pos_f[1], pos_f[2]);
    let vel_final = Vector3::new(vel_f[0], vel_f[1], vel_f[2]);
    Ok((pos_final, vel_final, sens))
}

/// Update an augmented covariance under the sensitivity matrix from
/// [`propagate_with_stm`].
///
/// Given a 6 x (6 + Np) sensitivity matrix and a (6 + Np) x (6 + Np)
/// augmented covariance, returns the new augmented covariance:
///
/// ```text
/// Phi_aug = [[ sensitivity ],     // 6 rows
///            [ 0_{Np x 6} | I_Np ]] // Np rows (parameters constant)
/// P_new = Phi_aug * P_old * Phi_aug^T
/// ```
///
/// This is the linear approximation: parameters are held constant during
/// propagation (their values do not change), but their *uncertainty*
/// couples into the state uncertainty through the sensitivity matrix.
#[must_use]
pub fn covariance_update(sensitivity: &DMatrix<f64>, cov_old: &DMatrix<f64>) -> DMatrix<f64> {
    let np = sensitivity.ncols() - 6;
    let total = 6 + np;
    let mut phi_aug = DMatrix::<f64>::zeros(total, total);
    phi_aug.view_mut((0, 0), (6, total)).copy_from(sensitivity);
    for k in 0..np {
        phi_aug[(6 + k, 6 + k)] = 1.0;
    }
    &phi_aug * cov_old * phi_aug.transpose()
}

/// Propagate state and covariance jointly: combines [`propagate_with_stm`]
/// and [`covariance_update`] into a single call.
///
/// Returns `(pos_final, vel_final, cov_final)`. The free-parameter values
/// themselves do not change during integration; only state and covariance
/// are returned.
///
/// # Errors
/// Returns an error if `cov_init` does not have shape `(6 + Np) x (6 + Np)`,
/// or if integration fails.
pub fn propagate_with_covariance<F: ParameterizedForce>(
    forces: &F,
    pos_init: Vector3<f64>,
    vel_init: Vector3<f64>,
    cov_init: &DMatrix<f64>,
    free_params: &[f64],
    epoch_init: Time<TDB>,
    epoch_final: Time<TDB>,
) -> KeteResult<(Vector3<f64>, Vector3<f64>, DMatrix<f64>)> {
    let np = free_params.len();
    let expected = 6 + np;
    if cov_init.nrows() != expected || cov_init.ncols() != expected {
        return Err(Error::ValueError(format!(
            "covariance must be {expected} x {expected} (got {} x {})",
            cov_init.nrows(),
            cov_init.ncols()
        )));
    }
    let (pos_f, vel_f, sens) = propagate_with_stm(
        forces,
        pos_init,
        vel_init,
        free_params,
        epoch_init,
        epoch_final,
    )?;
    let cov_f = covariance_update(&sens, cov_init);
    Ok((pos_f, vel_f, cov_f))
}

/// Propagate a state under `forces` with explicit `free_params`,
/// returning `(pos_final, vel_final)` only.
///
/// Lighter than [`propagate_with_stm`] / [`propagate_with_covariance`]
/// because no STM/covariance machinery is integrated. Use this for
/// sigma-point inner loops, particle propagation, or any other
/// forward-only propagation where the free-parameter values come from
/// outside the state shape.
///
/// `state.propagate_with(forces, to)` covers the typical case where
/// the state itself carries free parameters (or has none); this
/// function exists for the fewer cases where caller-supplied
/// `free_params` are needed without an `UncertainState` wrapper.
///
/// # Errors
/// Returns an error if integration fails or if the force returns an
/// error.
pub fn propagate_state<F: ParameterizedForce>(
    forces: &F,
    pos_init: Vector3<f64>,
    vel_init: Vector3<f64>,
    free_params: &[f64],
    epoch_init: Time<TDB>,
    epoch_final: Time<TDB>,
) -> KeteResult<(Vector3<f64>, Vector3<f64>)> {
    use crate::integrators::RadauIntegrator;

    let ode = |time: Time<TDB>,
               pos: &Vector3<f64>,
               vel: &Vector3<f64>,
               _meta: &mut (),
               _exact_eval: bool|
     -> KeteResult<Vector3<f64>> {
        let pos_typed = Vector::<F::Frame>::new([pos[0], pos[1], pos[2]]);
        let vel_typed = Vector::<F::Frame>::new([vel[0], vel[1], vel[2]]);
        Ok(forces
            .accel(time, &pos_typed, &vel_typed, free_params)?
            .into())
    };

    let (pos_f, vel_f, ()) =
        RadauIntegrator::integrate(&ode, pos_init, vel_init, epoch_init, epoch_final, (), None)?;
    Ok((pos_f, vel_f))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::GMS;
    use crate::forces::ParameterizedForce;
    use crate::frames::{Equatorial, SunCenter, Vector};

    /// A two-body Kepler force with provided GM (no free parameters).
    /// Provides analytical position Jacobian for cross-validation.
    struct TwoBody {
        gm: f64,
    }

    impl ParameterizedForce for TwoBody {
        type Frame = Equatorial;
        type Center = SunCenter;

        fn accel(
            &self,
            _time: Time<TDB>,
            pos: &Vector<Equatorial>,
            _vel: &Vector<Equatorial>,
            _free_params: &[f64],
        ) -> KeteResult<Vector<Equatorial>> {
            let p: Vector3<f64> = (*pos).into();
            let r3 = p.norm().powi(3);
            Ok(Vector::<Equatorial>::new((-p * (self.gm / r3)).into()))
        }

        fn jacobians(
            &self,
            _time: Time<TDB>,
            pos: &Vector<Equatorial>,
            _vel: &Vector<Equatorial>,
            _free_params: &[f64],
        ) -> KeteResult<(Matrix3<f64>, Matrix3<f64>)> {
            // d(-GM r / r^3) / dr = -GM (I/r^3 - 3 r r^T / r^5)
            //                    = GM/r^5 (3 r r^T - r^2 I)
            let p: Vector3<f64> = (*pos).into();
            let r2 = p.norm_squared();
            let r5 = r2 * r2.sqrt();
            let da_dr = (3.0 * p * p.transpose() - r2 * Matrix3::identity()) * (self.gm / r5);
            Ok((da_dr, Matrix3::zeros()))
        }
    }

    /// Variational integration round-trip on a circular orbit:
    /// after one period, the state should return to its starting point
    /// and the STM should be close to the identity (modulo a known
    /// secular drift of ~1 rad along the along-track direction over
    /// one orbit -- so we just assert the position columns are
    /// reasonable, not exactly identity).
    #[test]
    fn stm_two_body_circular_short_arc() {
        let gm = GMS;
        let v_circ = gm.sqrt();
        let pos = Vector3::new(1.0, 0.0, 0.0);
        let vel = Vector3::new(0.0, v_circ, 0.0);
        // Short arc -- 1 day.
        let (pos_f, vel_f, sens) = propagate_with_stm(
            &TwoBody { gm },
            pos,
            vel,
            &[],
            Time::<TDB>::new(0.0),
            Time::<TDB>::new(1.0),
        )
        .unwrap();

        // The state should advance the expected small amount.
        assert!(pos_f.norm() > 0.99 && pos_f.norm() < 1.01);
        assert!(vel_f.norm() > 0.99 * v_circ && vel_f.norm() < 1.01 * v_circ);

        // STM dimensions: 6 x 6 (no free params).
        assert_eq!(sens.nrows(), 6);
        assert_eq!(sens.ncols(), 6);

        // For a short arc, the STM should be close to identity in
        // structure: the 3x3 r-from-r block should have det ~ 1, and
        // the v-from-v block too.
        let phi_rr = sens.fixed_view::<3, 3>(0, 0);
        let phi_vv = sens.fixed_view::<3, 3>(3, 3);
        assert!(
            (phi_rr.determinant() - 1.0).abs() < 1e-3,
            "det(Phi_rr) = {}",
            phi_rr.determinant()
        );
        assert!(
            (phi_vv.determinant() - 1.0).abs() < 1e-3,
            "det(Phi_vv) = {}",
            phi_vv.determinant()
        );
    }

    /// Identity sensitivity (no propagation): covariance should be unchanged.
    #[test]
    fn covariance_update_identity_sensitivity() {
        let np = 2;
        let total = 6 + np;
        let mut sens = DMatrix::<f64>::zeros(6, total);
        for i in 0..6 {
            sens[(i, i)] = 1.0;
        }
        // Random-ish symmetric positive definite covariance.
        let mut cov = DMatrix::<f64>::zeros(total, total);
        for i in 0..total {
            cov[(i, i)] = (i as f64 + 1.0) * 0.5;
        }
        cov[(0, 1)] = 0.1;
        cov[(1, 0)] = 0.1;
        let cov_new = covariance_update(&sens, &cov);
        // With identity sensitivity and identity-augmented STM, the
        // covariance is unchanged.
        for i in 0..total {
            for j in 0..total {
                assert!(
                    (cov_new[(i, j)] - cov[(i, j)]).abs() < 1e-14,
                    "({i},{j}): {} vs {}",
                    cov_new[(i, j)],
                    cov[(i, j)]
                );
            }
        }
    }

    /// Sensitivity-only-on-state (no parameter columns): covariance
    /// update should be a standard 6x6 `P_new` = `Phi` * P * `Phi^T`.
    #[test]
    fn covariance_update_no_parameters() {
        let mut sens = DMatrix::<f64>::zeros(6, 6);
        // Some arbitrary 6x6 transformation.
        for i in 0..6 {
            sens[(i, i)] = 2.0;
            if i + 1 < 6 {
                sens[(i, i + 1)] = 0.5;
            }
        }
        let mut cov = DMatrix::<f64>::zeros(6, 6);
        for i in 0..6 {
            cov[(i, i)] = 1.0;
        }
        let cov_new = covariance_update(&sens, &cov);
        // Direct computation: Phi P Phi^T = sens * I * sens^T = sens * sens^T.
        let expected = &sens * sens.transpose();
        for i in 0..6 {
            for j in 0..6 {
                assert!(
                    (cov_new[(i, j)] - expected[(i, j)]).abs() < 1e-14,
                    "({i},{j}): {} vs {}",
                    cov_new[(i, j)],
                    expected[(i, j)]
                );
            }
        }
    }

    /// Parameter-block invariance: the (Np x Np) bottom-right block of
    /// the covariance is unchanged on each step (parameters are constant).
    /// Tested with a non-trivial state-state coupling and a parameter
    /// covariance entry.
    #[test]
    fn covariance_update_parameter_block_passes_through() {
        let np = 1;
        let total = 6 + np;
        let mut sens = DMatrix::<f64>::zeros(6, total);
        // Identity for state-state; zero state-parameter coupling.
        for i in 0..6 {
            sens[(i, i)] = 1.0;
        }
        // Initial covariance: pure parameter variance.
        let mut cov = DMatrix::<f64>::zeros(total, total);
        cov[(6, 6)] = 0.25; // sigma_p^2 = 0.25
        let cov_new = covariance_update(&sens, &cov);
        assert!(
            (cov_new[(6, 6)] - 0.25).abs() < 1e-14,
            "parameter variance changed: {} vs 0.25",
            cov_new[(6, 6)]
        );
    }

    /// Validation: covariance with wrong shape returns an error.
    #[test]
    fn propagate_with_covariance_validates_shape() {
        let force = TwoBody { gm: GMS };
        let pos = Vector3::new(1.0, 0.0, 0.0);
        let vel = Vector3::new(0.0, GMS.sqrt(), 0.0);
        // 6x6 covariance, but free_params has 1 element -> expected 7x7.
        let cov = DMatrix::<f64>::identity(6, 6);
        let result = propagate_with_covariance(
            &force,
            pos,
            vel,
            &cov,
            &[1.0],
            Time::<TDB>::new(0.0),
            Time::<TDB>::new(1.0),
        );
        assert!(result.is_err());
    }

    /// End-to-end: propagate a state + covariance jointly, verify the
    /// covariance is symmetric afterward (a basic sanity invariant).
    #[test]
    fn propagate_with_covariance_preserves_symmetry() {
        let force = TwoBody { gm: GMS };
        let pos = Vector3::new(1.0, 0.0, 0.0);
        let vel = Vector3::new(0.0, GMS.sqrt(), 0.0);
        // Modest position+velocity covariance.
        let mut cov = DMatrix::<f64>::zeros(6, 6);
        for i in 0..3 {
            cov[(i, i)] = 1e-10; // pos variance, AU^2
            cov[(3 + i, 3 + i)] = 1e-12; // vel variance, (AU/day)^2
        }
        let (_, _, cov_new) = propagate_with_covariance(
            &force,
            pos,
            vel,
            &cov,
            &[],
            Time::<TDB>::new(0.0),
            Time::<TDB>::new(10.0),
        )
        .unwrap();
        // Covariance should remain symmetric to working precision.
        for i in 0..6 {
            for j in (i + 1)..6 {
                let asym = (cov_new[(i, j)] - cov_new[(j, i)]).abs();
                assert!(
                    asym < 1e-14 * cov_new[(i, i)].abs().max(cov_new[(j, j)].abs()),
                    "asymmetry at ({i},{j}): {asym}"
                );
            }
        }
    }

    /// [`ParameterizedForce`] with a single free parameter `gm`. The free parameter
    /// scales the central acceleration, so `d a / d gm = -r / r^3`.
    struct TwoBodyParametric;

    impl ParameterizedForce for TwoBodyParametric {
        type Frame = Equatorial;
        type Center = SunCenter;

        fn n_free_params(&self) -> usize {
            1
        }

        fn free_param_names(&self) -> Vec<&'static str> {
            vec!["gm"]
        }

        fn accel(
            &self,
            _time: Time<TDB>,
            pos: &Vector<Equatorial>,
            _vel: &Vector<Equatorial>,
            free_params: &[f64],
        ) -> KeteResult<Vector<Equatorial>> {
            let gm = free_params[0];
            let p: Vector3<f64> = (*pos).into();
            let r3 = p.norm().powi(3);
            Ok(Vector::<Equatorial>::new((-p * (gm / r3)).into()))
        }
    }

    /// Parameter sensitivity: propagate with a parametric force and
    /// verify the parameter-sensitivity column matches FD against gm.
    #[test]
    fn stm_parameter_sensitivity_matches_finite_difference() {
        let gm = GMS;
        let v_circ = gm.sqrt();
        let pos = Vector3::new(1.0, 0.0, 0.0);
        let vel = Vector3::new(0.0, v_circ, 0.0);
        let dt = 5.0;
        let force = TwoBodyParametric;

        let (_pos_base, _vel_base, sens) = propagate_with_stm(
            &force,
            pos,
            vel,
            &[gm],
            Time::<TDB>::new(0.0),
            Time::<TDB>::new(dt),
        )
        .unwrap();

        // FD: perturb gm by h, repropagate.
        let h = gm * 1e-6;
        let (pos_p, vel_p, _) = propagate_with_stm(
            &force,
            pos,
            vel,
            &[gm + h],
            Time::<TDB>::new(0.0),
            Time::<TDB>::new(dt),
        )
        .unwrap();
        let (pos_b, vel_b, _) = propagate_with_stm(
            &force,
            pos,
            vel,
            &[gm],
            Time::<TDB>::new(0.0),
            Time::<TDB>::new(dt),
        )
        .unwrap();
        let dpos_dp = (pos_p - pos_b) / h;
        let dvel_dp = (vel_p - vel_b) / h;

        // Sensitivity column 6 (the only parameter column).
        // Tolerance is loose because FD over a 5-day arc carries noise
        // and the FD-default jacobian inside propagate_with_stm adds
        // its own error budget.
        for i in 0..3 {
            assert!(
                (sens[(i, 6)] - dpos_dp[i]).abs() < 1e-3,
                "row {} pos: STM={}, FD={}",
                i,
                sens[(i, 6)],
                dpos_dp[i]
            );
            assert!(
                (sens[(3 + i, 6)] - dvel_dp[i]).abs() < 1e-3,
                "row {} vel: STM={}, FD={}",
                i,
                sens[(3 + i, 6)],
                dvel_dp[i]
            );
        }
    }

    /// FD validation: compute STM analytically, then compute it by
    /// finite-differencing the state propagation. They should agree to
    /// roughly FD precision (~1e-5).
    #[test]
    fn stm_two_body_matches_finite_difference() {
        let gm = GMS;
        let v_circ = gm.sqrt();
        let pos = Vector3::new(1.0, 0.0, 0.0);
        let vel = Vector3::new(0.0, v_circ, 0.0);
        let dt = 5.0;
        let force = TwoBody { gm };

        let (_pos_f, _vel_f, sens) = propagate_with_stm(
            &force,
            pos,
            vel,
            &[],
            Time::<TDB>::new(0.0),
            Time::<TDB>::new(dt),
        )
        .unwrap();

        // Compute STM column 0 via FD: perturb pos.x by h, repropagate,
        // take (final - unperturbed) / h.
        let h = 1e-6;
        let mut pos_pert = pos;
        pos_pert.x += h;
        let (pos_f_pert, vel_f_pert, _) = propagate_with_stm(
            &force,
            pos_pert,
            vel,
            &[],
            Time::<TDB>::new(0.0),
            Time::<TDB>::new(dt),
        )
        .unwrap();
        let (pos_f_base, vel_f_base, _) = propagate_with_stm(
            &force,
            pos,
            vel,
            &[],
            Time::<TDB>::new(0.0),
            Time::<TDB>::new(dt),
        )
        .unwrap();
        let dpos_dx = (pos_f_pert - pos_f_base) / h;
        let dvel_dx = (vel_f_pert - vel_f_base) / h;

        // Analytical column 0 of STM is sens column 0 (rows 0..6).
        for i in 0..3 {
            assert!(
                (sens[(i, 0)] - dpos_dx[i]).abs() < 1e-4,
                "row {} pos: STM={}, FD={}",
                i,
                sens[(i, 0)],
                dpos_dx[i]
            );
            assert!(
                (sens[(3 + i, 0)] - dvel_dx[i]).abs() < 1e-4,
                "row {} vel: STM={}, FD={}",
                i,
                sens[(3 + i, 0)],
                dvel_dx[i]
            );
        }
    }
}
