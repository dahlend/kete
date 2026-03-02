//! Jacobian and variational equation machinery for STM computation.
//!
//! This module provides:
//! - Finite-difference state Jacobians (`da/dr`, `da/dv`) of the full force model.
//! - Analytical non-gravitational parameter partials (`da/dp_k`).
//! - An augmented second-order acceleration function for use with the Radau integrator.
//!
//! The augmented state has dimension 30 (maximum), laid out as:
//!   \[0..3\]   physical position
//!   \[3..12\]  `Phi_rr` (3x3, column-major)
//!   \[12..21\] `Phi_rv` (3x3, column-major)
//!   \[21..30\] up to 3 parameter sensitivity vectors `s_k` (3 elements each)
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
//
use crate::constants::GMS;
use crate::frames::Equatorial;
use crate::prelude::KeteResult;
use crate::propagation::nongrav::NonGravModel;
use crate::propagation::{AccelSPKMeta, spk_accel};
use crate::spice::LOADED_SPK;
use crate::time::{TDB, Time};
use nalgebra::{Matrix3, SVector, Vector3};

use super::analytic_2_body;

/// Perturbation size for finite-difference Jacobians.
const EPS: f64 = 1e-7;

/// Compute da/dr and da/dv via central finite differences of [`spk_accel`].
///
/// This automatically captures contributions from all forces (N-body gravity, GR, J2,
/// non-gravitational). The 12 perturbed evaluations use `exact_eval = false` to avoid
/// polluting close-approach metadata.
fn spk_accel_jacobians(
    time: Time<TDB>,
    pos: &Vector3<f64>,
    vel: &Vector3<f64>,
    meta: &mut AccelSPKMeta<'_>,
) -> KeteResult<(Matrix3<f64>, Matrix3<f64>)> {
    let saved_ca = meta.close_approach;
    let mut da_dr = Matrix3::<f64>::zeros();
    let mut da_dv = Matrix3::<f64>::zeros();
    let inv_2eps = 0.5 / EPS;

    for i in 0..3 {
        let mut pos_p = *pos;
        let mut pos_m = *pos;
        pos_p[i] += EPS;
        pos_m[i] -= EPS;
        let a_p = spk_accel(time, &pos_p, vel, meta, false)?;
        let a_m = spk_accel(time, &pos_m, vel, meta, false)?;
        da_dr.set_column(i, &((a_p - a_m) * inv_2eps));
    }

    for i in 0..3 {
        let mut vel_p = *vel;
        let mut vel_m = *vel;
        vel_p[i] += EPS;
        vel_m[i] -= EPS;
        let a_p = spk_accel(time, pos, &vel_p, meta, false)?;
        let a_m = spk_accel(time, pos, &vel_m, meta, false)?;
        da_dv.set_column(i, &((a_p - a_m) * inv_2eps));
    }

    meta.close_approach = saved_ca;
    Ok((da_dr, da_dv))
}

/// Compute analytical partial derivatives of the non-gravitational acceleration
/// with respect to each free parameter.
///
/// Returns up to 3 vectors (one per free parameter). The returned count matches
/// the number of free parameters in the model.
fn nongrav_param_partials(
    model: &NonGravModel,
    pos: &Vector3<f64>,
    vel: &Vector3<f64>,
) -> Vec<Vector3<f64>> {
    match model {
        NonGravModel::JplComet {
            alpha,
            r_0,
            m,
            n,
            k,
            dt,
            ..
        } => {
            let mut eval_pos = *pos;
            let pos_hat = pos.normalize();
            let t_hat = (vel - pos_hat * vel.dot(&pos_hat)).normalize();
            let n_hat = t_hat.cross(&pos_hat); // perpendicular unit vecs -> already unit length

            if *dt != 0.0 {
                (eval_pos, _) = analytic_2_body((-dt).into(), pos, vel, None).unwrap();
            }
            let rr0 = eval_pos.norm() / r_0;
            let scale = alpha * rr0.powf(-m) * (1.0 + rr0.powf(*n)).powf(-k);
            vec![pos_hat * scale, t_hat * scale, n_hat * scale]
        }
        NonGravModel::Dust { .. } => {
            let pos_hat = pos.normalize();
            let r_dot = pos_hat.dot(vel);
            let norm2_inv = pos.norm_squared().recip();
            let scale = GMS * norm2_inv;
            let partial = scale
                * ((1.0 - r_dot * crate::constants::C_AU_PER_DAY_INV_SQUARED) * pos_hat
                    - vel * crate::constants::C_AU_PER_DAY_INV_SQUARED);
            vec![partial]
        }
    }
}

/// Number of free non-grav parameters for a given model.
pub(crate) fn n_params(model: Option<&NonGravModel>) -> usize {
    match model {
        None => 0,
        Some(NonGravModel::JplComet { .. }) => 3,
        Some(NonGravModel::Dust { .. }) => 1,
    }
}

/// Augmented second-order acceleration function for STM + parameter sensitivities.
///
/// Dimension is fixed at 30 (supports up to 3 free non-grav parameters).
/// Unused elements remain zero.
///
/// State layout (30 elements each for `pos_aug` and `vel_aug`):
///   \[0..3\]   object position / velocity
///   \[3..12\]  `Phi_rr` (3x3 col-major) / `Phi_rr'`
///   \[12..21\] `Phi_rv` (3x3 col-major) / `Phi_rv'`
///   \[21..24\] `s_1` / `s_1'`  (parameter sensitivity 1)
///   \[24..27\] `s_2` / `s_2'`  (parameter sensitivity 2)
///   \[27..30\] `s_3` / `s_3'`  (parameter sensitivity 3)
pub(crate) fn stm_augmented_accel(
    time: Time<TDB>,
    pos_aug: &SVector<f64, 30>,
    vel_aug: &SVector<f64, 30>,
    meta: &mut AccelSPKMeta<'_>,
    exact_eval: bool,
) -> KeteResult<SVector<f64, 30>> {
    let mut result = SVector::<f64, 30>::zeros();
    let pos: Vector3<f64> = pos_aug.fixed_rows::<3>(0).into();
    let vel: Vector3<f64> = vel_aug.fixed_rows::<3>(0).into();

    // Physical acceleration
    let accel = spk_accel(time, &pos, &vel, meta, exact_eval)?;
    result.fixed_rows_mut::<3>(0).copy_from(&accel);

    // State Jacobians via finite differences
    let (da_dr, da_dv) = spk_accel_jacobians(time, &pos, &vel, meta)?;

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

    // Parameter sensitivities: s_k'' = da_dr * s_k + da_dv * s_k' + da/dp_k
    // Non-grav partials must use Sun-relative pos/vel, matching spk_accel internals.
    if let Some(model) = meta.non_grav_model.as_ref() {
        let spk = &LOADED_SPK.try_read()?;
        let sun_state = spk.try_get_state_with_center::<Equatorial>(10, time, 0)?;
        let rel_pos = pos - Vector3::from(sun_state.pos);
        let rel_vel = vel - Vector3::from(sun_state.vel);
        let partials = nongrav_param_partials(model, &rel_pos, &rel_vel);
        for (k, partial_k) in partials.iter().enumerate() {
            let base = 21 + k * 3;
            let s_k: Vector3<f64> = pos_aug.fixed_rows::<3>(base).into();
            let s_k_dot: Vector3<f64> = vel_aug.fixed_rows::<3>(base).into();
            let s_k_ddot = da_dr * s_k + da_dv * s_k_dot + partial_k;
            result.fixed_rows_mut::<3>(base).copy_from(&s_k_ddot);
        }
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::GravParams;
    use crate::frames::Equatorial;
    use crate::prelude::Desig;
    use crate::propagation::propagate_n_body_spk;
    use crate::propagation::state_transition::compute_state_transition;
    use crate::state::State;

    /// Helper: create a test state at ~1 AU from the Sun (solar-system barycenter centered).
    fn test_state() -> State<Equatorial> {
        State::new(
            Desig::Name("Test".into()),
            2451545.0.into(), // J2000.0
            [1.0, 0.0, 0.0].into(),
            [0.0, 0.01720209895, 0.0].into(), // ~circular at 1 AU
            0,
        )
    }

    #[test]
    fn stm_n_body_finite_difference_validation() {
        // Validate the variational STM against finite-difference-of-trajectory.
        let state = test_state();
        let jd_final = (2451545.0 + 30.0).into(); // 30 days
        let planets = GravParams::planets();

        let (_final_state, sens) =
            compute_state_transition(&state, jd_final, &planets, None).unwrap();

        // Build STM via finite differences of Radau propagations
        let eps = 1e-6;

        for col in 0..6 {
            let mut pos_p: [f64; 3] = state.pos.into();
            let mut vel_p: [f64; 3] = state.vel.into();
            let mut pos_m: [f64; 3] = state.pos.into();
            let mut vel_m: [f64; 3] = state.vel.into();
            if col < 3 {
                pos_p[col] += eps;
                pos_m[col] -= eps;
            } else {
                vel_p[col - 3] += eps;
                vel_m[col - 3] -= eps;
            }
            let state_p = State::new(
                Desig::Name("P".into()),
                state.epoch,
                pos_p.into(),
                vel_p.into(),
                0,
            );
            let state_m = State::new(
                Desig::Name("M".into()),
                state.epoch,
                pos_m.into(),
                vel_m.into(),
                0,
            );
            let res_p = propagate_n_body_spk(state_p, jd_final, false, None).unwrap();
            let res_m = propagate_n_body_spk(state_m, jd_final, false, None).unwrap();

            let vec_p: Vec<f64> = res_p.pos.into_iter().chain(res_p.vel.into_iter()).collect();
            let vec_m: Vec<f64> = res_m.pos.into_iter().chain(res_m.vel.into_iter()).collect();

            for row in 0..6 {
                let fd = (vec_p[row] - vec_m[row]) / (2.0 * eps);
                let var = sens[(row, col)];
                let abs_err = (fd - var).abs();
                let scale = fd.abs().max(1e-10);
                assert!(
                    abs_err / scale < 1e-3,
                    "STM mismatch at ({}, {}): variational={:.10e}, fd={:.10e}, rel_err={:.4e}",
                    row,
                    col,
                    var,
                    fd,
                    abs_err / scale
                );
            }
        }
    }

    #[test]
    fn stm_determinant_conservative() {
        // For conservative forces (no non-grav), det(STM) should be ~1.
        let state = test_state();
        let jd_final = (2451545.0 + 30.0).into();
        let planets = GravParams::planets();

        let (_final_state, sens) =
            compute_state_transition(&state, jd_final, &planets, None).unwrap();

        // Extract the 6x6 STM
        let stm = sens.fixed_view::<6, 6>(0, 0);
        let det = stm.determinant();
        assert!(
            (det - 1.0).abs() < 1e-4,
            "STM determinant should be ~1 for conservative forces, got {det}"
        );
    }

    #[test]
    fn stm_jpl_comet_param_sensitivity() {
        // Validate parameter sensitivity columns for JplComet model via finite diffs.
        let a1 = 1e-8;
        let a2 = 1e-9;
        let a3 = 1e-10;
        let model = NonGravModel::new_jpl_comet_default(a1, a2, a3);

        let state = test_state();
        let jd_final = (2451545.0 + 30.0).into();
        let planets = GravParams::planets();

        let (_final_state, sens) =
            compute_state_transition(&state, jd_final, &planets, Some(model.clone())).unwrap();

        // Finite-difference test for each A parameter
        // Use a moderate perturbation; the FD accuracy is limited by the nonlinearity
        // of the trajectory w.r.t. the non-grav parameters over 30 days.
        let eps_a = 1e-11;
        let a_vals = [a1, a2, a3];
        for k in 0..3 {
            let mut a_p = a_vals;
            let mut a_m = a_vals;
            a_p[k] += eps_a;
            a_m[k] -= eps_a;

            let model_p = NonGravModel::new_jpl_comet_default(a_p[0], a_p[1], a_p[2]);
            let model_m = NonGravModel::new_jpl_comet_default(a_m[0], a_m[1], a_m[2]);

            let res_p =
                propagate_n_body_spk(state.clone(), jd_final, false, Some(model_p)).unwrap();
            let res_m =
                propagate_n_body_spk(state.clone(), jd_final, false, Some(model_m)).unwrap();

            let vec_p: Vec<f64> = res_p.pos.into_iter().chain(res_p.vel.into_iter()).collect();
            let vec_m: Vec<f64> = res_m.pos.into_iter().chain(res_m.vel.into_iter()).collect();

            for row in 0..6 {
                let fd = (vec_p[row] - vec_m[row]) / (2.0 * eps_a);
                let var = sens[(row, 6 + k)];
                let abs_err = (fd - var).abs();
                let scale = fd.abs().max(var.abs()).max(1e-10);
                assert!(
                    abs_err / scale < 1e-2,
                    "Param sensitivity mismatch for A{} at row {}: var={:.8e}, fd={:.8e}, rel={:.4e}",
                    k + 1,
                    row,
                    var,
                    fd,
                    abs_err / scale
                );
            }
        }
    }

    #[test]
    fn stm_dust_param_sensitivity() {
        // Validate parameter sensitivity column for the Dust (beta) model via FD.
        let beta = 0.01;
        let model = NonGravModel::new_dust(beta);

        let state = test_state();
        let jd_final = (2451545.0 + 30.0).into();
        let planets = GravParams::planets();

        let (_final_state, sens) =
            compute_state_transition(&state, jd_final, &planets, Some(model.clone())).unwrap();

        // Sensitivity matrix should be 6x7 (6 state + 1 beta parameter)
        assert_eq!(sens.ncols(), 7, "Expected 6+1 columns for Dust model");

        // Finite-difference perturbation of beta
        let eps_beta = 1e-6;
        let model_p = NonGravModel::new_dust(beta + eps_beta);
        let model_m = NonGravModel::new_dust(beta - eps_beta);

        let res_p = propagate_n_body_spk(state.clone(), jd_final, false, Some(model_p)).unwrap();
        let res_m = propagate_n_body_spk(state.clone(), jd_final, false, Some(model_m)).unwrap();

        let vec_p: Vec<f64> = res_p.pos.into_iter().chain(res_p.vel).collect();
        let vec_m: Vec<f64> = res_m.pos.into_iter().chain(res_m.vel).collect();

        for row in 0..6 {
            let fd = (vec_p[row] - vec_m[row]) / (2.0 * eps_beta);
            let var = sens[(row, 6)]; // column 6 = beta sensitivity
            let abs_err = (fd - var).abs();
            let scale = fd.abs().max(var.abs()).max(1e-10);
            assert!(
                abs_err / scale < 1e-2,
                "Dust beta sensitivity mismatch at row {}: var={:.8e}, fd={:.8e}, rel={:.4e}",
                row,
                var,
                fd,
                abs_err / scale
            );
        }
    }

    #[test]
    fn stm_long_arc_90_day() {
        // Validate STM over a 90-day arc against finite-difference-of-trajectory.
        let state = test_state();
        let jd_final = (2451545.0 + 90.0).into(); // 90 days
        let planets = GravParams::planets();

        let (_final_state, sens) =
            compute_state_transition(&state, jd_final, &planets, None).unwrap();

        // Finite-difference validation of each STM column
        let eps = 1e-6;

        for col in 0..6 {
            let mut pos_p: [f64; 3] = state.pos.into();
            let mut vel_p: [f64; 3] = state.vel.into();
            let mut pos_m: [f64; 3] = state.pos.into();
            let mut vel_m: [f64; 3] = state.vel.into();
            if col < 3 {
                pos_p[col] += eps;
                pos_m[col] -= eps;
            } else {
                vel_p[col - 3] += eps;
                vel_m[col - 3] -= eps;
            }
            let state_p = State::new(
                Desig::Name("P".into()),
                state.epoch,
                pos_p.into(),
                vel_p.into(),
                0,
            );
            let state_m = State::new(
                Desig::Name("M".into()),
                state.epoch,
                pos_m.into(),
                vel_m.into(),
                0,
            );
            let res_p = propagate_n_body_spk(state_p, jd_final, false, None).unwrap();
            let res_m = propagate_n_body_spk(state_m, jd_final, false, None).unwrap();

            let vec_p: Vec<f64> = res_p.pos.into_iter().chain(res_p.vel).collect();
            let vec_m: Vec<f64> = res_m.pos.into_iter().chain(res_m.vel).collect();

            for row in 0..6 {
                let fd = (vec_p[row] - vec_m[row]) / (2.0 * eps);
                let var = sens[(row, col)];
                let abs_err = (fd - var).abs();
                let scale = fd.abs().max(1e-10);
                // Relax tolerance to 1% for a longer arc; FD accuracy degrades
                // over long arcs due to trajectory divergence.
                assert!(
                    abs_err / scale < 1e-2,
                    "Long-arc STM mismatch at ({}, {}): var={:.10e}, fd={:.10e}, rel={:.4e}",
                    row,
                    col,
                    var,
                    fd,
                    abs_err / scale
                );
            }
        }

        // Determinant check: should still be ~1 for conservative forces
        let stm = sens.fixed_view::<6, 6>(0, 0);
        let det = stm.determinant();
        assert!(
            (det - 1.0).abs() < 1e-3,
            "Long-arc STM determinant should be ~1, got {det}"
        );
    }
}
