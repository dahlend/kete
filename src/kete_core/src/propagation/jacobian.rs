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
use crate::constants::{C_AU_PER_DAY_INV_SQUARED, EARTH_J2, GMS, JUPITER_J2, SUN_J2};
use crate::frames::{Ecliptic, Equatorial, InertialFrame};
use crate::prelude::KeteResult;
use crate::propagation::nongrav::NonGravModel;
use crate::propagation::{AccelSPKMeta, spk_accel_cached};
use crate::spice::LOADED_SPK;
use crate::time::{TDB, Time};
use nalgebra::{Matrix3, SVector, Vector3};

use super::analytic_2_body;

/// Perturbation size for finite-difference Jacobians.
const EPS: f64 = 1e-7;

/// Compute the analytical J2 oblateness Jacobian `da_J2/dd` in the body's
/// pole-aligned frame.
///
/// Arguments:
/// - `d`: relative position in the body's pole-aligned frame
/// - `radius`: equatorial radius of the body (AU)
/// - `j2`: J2 coefficient
/// - `mass`: GM of the body (AU^3/day^2)
fn j2_jacobian(d: &Vector3<f64>, radius: f64, j2: f64, mass: f64) -> Matrix3<f64> {
    let d = *d;
    let r = d.norm();
    let r2 = r * r;
    let z = d.z;

    // lam = 3/2 * J2 * GM * Re^2 / r^5
    let lambda = 1.5 * j2 * mass * (radius / r).powi(2) / (r2 * r);

    // Z = 5 z_hat^2 = 5z^2/r^2
    let big_z = 5.0 * z * z / r2;

    // a/lam = [dx(Z-1), dy(Z-1), dz(Z-3)]
    let a_norm = Vector3::new(
        d.x * (big_z - 1.0),
        d.y * (big_z - 1.0),
        d.z * (big_z - 3.0),
    );

    // F = diag(Z-1, Z-1, Z-3)
    let f_diag = Matrix3::from_diagonal(&Vector3::new(big_z - 1.0, big_z - 1.0, big_z - 3.0));

    // dZ/dd = (10z/r^2)(e_hat_z - z d/r^2)  (column vector)
    let e_z = Vector3::new(0.0, 0.0, 1.0);
    let dz_dd = (10.0 * z / r2) * (e_z - (z / r2) * d);

    // da/dd = lam (-5/r^2 * (a/lam) d^T  +  F  +  d (dZ/dd)^T)
    lambda * (-5.0 / r2 * a_norm * d.transpose() + f_diag + d * dz_dd.transpose())
}

/// Analytical Dust (SRP + Poynting-Robertson) Jacobians `da/dr` and `da/dv`.
///
/// Position and velocity are Sun-relative.
fn dust_jacobians(
    pos: &Vector3<f64>,
    vel: &Vector3<f64>,
    beta: f64,
) -> (Matrix3<f64>, Matrix3<f64>) {
    let pos = *pos;
    let vel = *vel;
    let r = pos.norm();
    let r2 = r * r;
    let d_hat = pos / r;
    let cinv2 = C_AU_PER_DAY_INV_SQUARED;
    let r_dot = d_hat.dot(&vel);
    let s = GMS * beta / r2;
    let ident = Matrix3::<f64>::identity();

    // inner = a_dust / s = (1 - r_dot cinv2) d_hat - cinv2 v
    let inner = (1.0 - r_dot * cinv2) * d_hat - cinv2 * vel;

    // dd_hat/dd = (I - d_hat d_hat^T) / r
    let dd_hat = (ident - d_hat * d_hat.transpose()) / r;

    // dr_dot/dd = ((v - r_dot d_hat) / r)^T  (column vector; transposed in the outer product)
    let dr_dot_col = (vel - r_dot * d_hat) / r;

    // da/dd = (-2s/r^2) inner pos^T  +  s (-cinv2 d_hat (dr_dot/dd) + (1-r_dot cinv2)(dd_hat/dd))
    let da_dr = (-2.0 * s / r2) * inner * pos.transpose()
        + s * (-cinv2 * d_hat * dr_dot_col.transpose() + (1.0 - r_dot * cinv2) * dd_hat);

    // da/dv = -s cinv2 (d_hat d_hat^T + I)
    let da_dv = -s * cinv2 * (d_hat * d_hat.transpose() + ident);

    (da_dr, da_dv)
}

/// Compute non-gravitational Jacobians via targeted finite differences.
///
/// Used for the JPL Comet model whose RTN-frame derivatives are complex.
/// Only 12 evaluations of [`NonGravModel::add_acceleration`] are needed (no SPK
/// lookups), so the cost is negligible.
fn nongrav_jacobians_fd(
    model: &NonGravModel,
    pos: &Vector3<f64>,
    vel: &Vector3<f64>,
) -> (Matrix3<f64>, Matrix3<f64>) {
    let inv_2eps = 0.5 / EPS;
    let mut da_dr = Matrix3::<f64>::zeros();
    let mut da_dv = Matrix3::<f64>::zeros();

    for i in 0..3 {
        let mut pos_p = *pos;
        let mut pos_m = *pos;
        pos_p[i] += EPS;
        pos_m[i] -= EPS;
        let mut a_p = Vector3::zeros();
        let mut a_m = Vector3::zeros();
        model.add_acceleration(&mut a_p, &pos_p, vel);
        model.add_acceleration(&mut a_m, &pos_m, vel);
        da_dr.set_column(i, &((a_p - a_m) * inv_2eps));
    }

    for i in 0..3 {
        let mut vel_p = *vel;
        let mut vel_m = *vel;
        vel_p[i] += EPS;
        vel_m[i] -= EPS;
        let mut a_p = Vector3::zeros();
        let mut a_m = Vector3::zeros();
        model.add_acceleration(&mut a_p, pos, &vel_p);
        model.add_acceleration(&mut a_m, pos, &vel_m);
        da_dv.set_column(i, &((a_p - a_m) * inv_2eps));
    }

    (da_dr, da_dv)
}

/// Compute analytical `da/dr` and `da/dv` for the full force model.
///
/// Includes contributions from:
/// - Newtonian N-body gravity (all massive bodies)
/// - General relativity correction (Sun, Jupiter)
/// - J2 oblateness (Sun, Jupiter, Earth)
/// - Non-gravitational forces (Dust: analytical; JPL Comet: targeted FD)
fn analytical_jacobians(
    pos: &Vector3<f64>,
    vel: &Vector3<f64>,
    cached_states: &[(Vector3<f64>, Vector3<f64>)],
    meta: &AccelSPKMeta<'_>,
) -> (Matrix3<f64>, Matrix3<f64>) {
    let pos = *pos;
    let vel = *vel;
    let mut da_dr = Matrix3::<f64>::zeros();
    let mut da_dv = Matrix3::<f64>::zeros();
    let ident = Matrix3::<f64>::identity();

    for (grav_params, (body_pos, body_vel)) in meta.massive_obj.iter().zip(cached_states) {
        let d = pos - body_pos;
        let v = vel - body_vel;
        let r = d.norm();
        let r2 = r * r;
        let r3 = r2 * r;
        let r5 = r2 * r3;
        let mass = grav_params.mass;

        // 1. Newtonian point-mass: da/dr = -GM/r^5 (r^2I - 3 d d^T)
        da_dr -= (mass / r5) * (r2 * ident - 3.0 * d * d.transpose());

        match grav_params.naif_id {
            5 | 10 => {
                // 2. GR correction (Sec 3.2)
                let cinv2 = C_AU_PER_DAY_INV_SQUARED;
                let kappa = mass * cinv2 / r3;
                let v2 = v.norm_squared();
                let big_c = 4.0 * mass / r - v2;
                let big_r = 4.0 * d.dot(&v);
                let a_gr = big_c * d + big_r * v;

                // da_GR/dr
                da_dr += (-3.0 * kappa / r2) * a_gr * d.transpose()
                    + kappa
                        * ((-4.0 * mass / r3) * d * d.transpose()
                            + big_c * ident
                            + 4.0 * v * v.transpose());

                // da_GR/dv
                da_dv +=
                    kappa * (-2.0 * d * v.transpose() + 4.0 * v * d.transpose() + big_r * ident);

                // 3. J2 oblateness (Sec 3.3  -  ecliptic frame for Sun/Jupiter)
                let j2_val = if grav_params.naif_id == 10 {
                    SUN_J2
                } else {
                    JUPITER_J2
                };
                let d_ec = Ecliptic::from_equatorial(d);
                let j2_jac = j2_jacobian(&d_ec, f64::from(grav_params.radius), j2_val, mass);
                // Rotate back: R * J * R^T
                let rot = *Ecliptic::rotation_to_equatorial().matrix();
                da_dr += rot * j2_jac * rot.transpose();
            }
            399 => {
                // J2 for Earth (Sec 3.3  -  equatorial frame directly)
                da_dr += j2_jacobian(&d, f64::from(grav_params.radius), EARTH_J2, mass);
            }
            _ => {}
        }
    }

    // Non-gravitational forces (Sec 3.4)
    if let Some(model) = &meta.non_grav_model {
        let sun_idx = meta
            .massive_obj
            .iter()
            .position(|g| g.naif_id == 10)
            .expect("Sun must be in massive_obj for non-grav models");
        let (sun_pos, sun_vel) = &cached_states[sun_idx];
        let rel_pos = pos - sun_pos;
        let rel_vel = vel - sun_vel;
        let (ng_dr, ng_dv) = match model {
            NonGravModel::Dust { beta } => dust_jacobians(&rel_pos, &rel_vel, *beta),
            NonGravModel::JplComet { .. } => nongrav_jacobians_fd(model, &rel_pos, &rel_vel),
        };
        da_dr += ng_dr;
        da_dv += ng_dv;
    }

    (da_dr, da_dv)
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
                * ((1.0 - r_dot * C_AU_PER_DAY_INV_SQUARED) * pos_hat
                    - vel * C_AU_PER_DAY_INV_SQUARED);
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

    // Cache planet states once  -  reused by the base acceleration evaluation,
    // the analytical Jacobians, and the non-grav parameter partials.
    let cached_states: Vec<(Vector3<f64>, Vector3<f64>)> = {
        let spk = &LOADED_SPK.try_read()?;
        meta.massive_obj
            .iter()
            .map(|g| {
                let state = spk.try_get_state_with_center::<Equatorial>(g.naif_id, time, 0)?;
                Ok((Vector3::from(state.pos), Vector3::from(state.vel)))
            })
            .collect::<KeteResult<_>>()?
    };

    // Physical acceleration
    let accel = spk_accel_cached(time, &pos, &vel, &cached_states, meta, exact_eval)?;
    result.fixed_rows_mut::<3>(0).copy_from(&accel);

    // State Jacobians via analytical expressions (gravity, GR, J2, non-grav)
    let (da_dr, da_dv) = analytical_jacobians(&pos, &vel, &cached_states, meta);

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
        // Find the Sun (NAIF ID 10) in the cached states.
        let sun_idx = meta
            .massive_obj
            .iter()
            .position(|g| g.naif_id == 10)
            .expect("Sun must be in massive_obj for non-grav models");
        let (sun_pos, sun_vel) = &cached_states[sun_idx];
        let rel_pos = pos - sun_pos;
        let rel_vel = vel - sun_vel;
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

    /// Compute da/dr and da/dv via central finite differences of [`spk_accel_cached`].
    ///
    /// This automatically captures contributions from all forces (N-body gravity, GR, J2,
    /// non-gravitational). The 12 perturbed evaluations use `exact_eval = false` to avoid
    /// polluting close-approach metadata.
    ///
    /// `cached_states` must contain pre-fetched `(pos, vel)` for each massive body,
    /// avoiding redundant SPK lookups across the 12 perturbations.
    ///
    /// Retained as a test-only reference for validating `analytical_jacobians`.
    fn spk_accel_jacobians(
        time: Time<TDB>,
        pos: &Vector3<f64>,
        vel: &Vector3<f64>,
        cached_states: &[(Vector3<f64>, Vector3<f64>)],
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
            let a_p = spk_accel_cached(time, &pos_p, vel, cached_states, meta, false)?;
            let a_m = spk_accel_cached(time, &pos_m, vel, cached_states, meta, false)?;
            da_dr.set_column(i, &((a_p - a_m) * inv_2eps));
        }

        for i in 0..3 {
            let mut vel_p = *vel;
            let mut vel_m = *vel;
            vel_p[i] += EPS;
            vel_m[i] -= EPS;
            let a_p = spk_accel_cached(time, pos, &vel_p, cached_states, meta, false)?;
            let a_m = spk_accel_cached(time, pos, &vel_m, cached_states, meta, false)?;
            da_dv.set_column(i, &((a_p - a_m) * inv_2eps));
        }

        meta.close_approach = saved_ca;
        Ok((da_dr, da_dv))
    }

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

        let (_final_state, sens) = compute_state_transition(&state, jd_final, false, None).unwrap();

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

        let (_final_state, sens) = compute_state_transition(&state, jd_final, false, None).unwrap();

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

        let (_final_state, sens) =
            compute_state_transition(&state, jd_final, false, Some(model.clone())).unwrap();

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

        let (_final_state, sens) =
            compute_state_transition(&state, jd_final, false, Some(model.clone())).unwrap();

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

        let (_final_state, sens) = compute_state_transition(&state, jd_final, false, None).unwrap();

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

    /// Compare analytical Jacobians against the FD reference at a given state.
    fn check_jacobians_match(non_grav: Option<NonGravModel>, tol: f64) {
        let state = test_state();
        let time = state.epoch;
        let pos: Vector3<f64> = state.pos.into();
        let vel: Vector3<f64> = state.vel.into();
        let planets = GravParams::planets();

        let cached_states: Vec<(Vector3<f64>, Vector3<f64>)> = {
            let spk = &LOADED_SPK.try_read().unwrap();
            planets
                .iter()
                .map(|g| {
                    let s = spk
                        .try_get_state_with_center::<Equatorial>(g.naif_id, time, 0)
                        .unwrap();
                    (Vector3::from(s.pos), Vector3::from(s.vel))
                })
                .collect()
        };

        let mut meta = AccelSPKMeta {
            close_approach: None,
            non_grav_model: non_grav,
            massive_obj: &planets,
        };

        let (fd_dr, fd_dv) =
            spk_accel_jacobians(time, &pos, &vel, &cached_states, &mut meta).unwrap();
        let (an_dr, an_dv) = analytical_jacobians(&pos, &vel, &cached_states, &meta);

        // FD round-off noise is ~eps_machine * |a| / EPS ~= 3e-13.
        // For Jacobian elements at or below this floor (e.g. GR da/dv ~ 1e-12),
        // FD accuracy is poor.  Use combined absolute + relative criterion:
        //   |err| < max(scale * rel_tol, abs_tol)
        let abs_tol = 1e-12;

        for i in 0..3 {
            for j in 0..3 {
                // da/dr
                let fd = fd_dr[(i, j)];
                let an = an_dr[(i, j)];
                let abs_err = (fd - an).abs();
                let scale = fd.abs().max(an.abs());
                let threshold = (scale * tol).max(abs_tol);
                assert!(
                    abs_err < threshold,
                    "da_dr[{i},{j}]: analytical={an:.10e}, fd={fd:.10e}, err={abs_err:.4e}, thr={threshold:.4e}"
                );
                // da/dv
                let fd = fd_dv[(i, j)];
                let an = an_dv[(i, j)];
                let abs_err = (fd - an).abs();
                let scale = fd.abs().max(an.abs());
                let threshold = (scale * tol).max(abs_tol);
                assert!(
                    abs_err < threshold,
                    "da_dv[{i},{j}]: analytical={an:.10e}, fd={fd:.10e}, err={abs_err:.4e}, thr={threshold:.4e}"
                );
            }
        }
    }

    #[test]
    fn analytical_vs_fd_gravity_only() {
        check_jacobians_match(None, 5e-6);
    }

    #[test]
    fn analytical_vs_fd_dust() {
        check_jacobians_match(Some(NonGravModel::new_dust(0.01)), 5e-6);
    }

    #[test]
    fn analytical_vs_fd_jpl_comet() {
        check_jacobians_match(
            Some(NonGravModel::new_jpl_comet_default(1e-8, 1e-9, 1e-10)),
            5e-6,
        );
    }
}
