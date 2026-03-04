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

use crate::constants::GravParams;
use crate::frames::Equatorial;
use crate::prelude::{KeteResult, State};
use crate::propagation::AccelSPKMeta;
use crate::propagation::jacobian::{n_params, stm_augmented_accel};
use crate::propagation::nongrav::NonGravModel;
use crate::propagation::radau::RadauIntegrator;
use crate::spice::LOADED_SPK;
use crate::time::{TDB, Time};
use nalgebra::{DMatrix, Matrix3, SVector, Vector3};

/// Compute the state transition matrix and optional parameter sensitivities using the
/// Radau 15th-order integrator with full N-body physics.
///
/// The input state may be centered on any body; it is automatically re-centered
/// to the solar system barycenter (SSB) for integration and restored to the
/// original center on output.
///
/// Returns the propagated [`State`] and a 6x(6+N) sensitivity matrix where N is
/// the number of free non-gravitational parameters (0 for none, 1 for `Dust`, 3 for
/// `JplComet`). Column ordering is:
///
/// ```text
/// cols 0-5  : 6x6 state transition matrix  d(r_f, v_f) / d(r_0, v_0)
/// col  6+k  : parameter sensitivity        d(r_f, v_f) / dp_k
/// ```
///
/// The returned state preserves the designation and `center_id` of the input.
///
/// # Errors
/// Fails when SPK queries fail or integration does not converge.
pub fn compute_state_transition(
    state: &State<Equatorial>,
    jd: Time<TDB>,
    massive_obj: &[GravParams],
    non_grav_model: Option<NonGravModel>,
) -> KeteResult<(State<Equatorial>, DMatrix<f64>)> {
    let np = n_params(non_grav_model.as_ref());
    let original_center = state.center_id;

    // Re-center to SSB for integration (acceleration functions query body
    // positions relative to center=0).
    let mut ssb_state = state.clone();
    if original_center != 0 {
        let spk = &LOADED_SPK.try_read()?;
        spk.try_change_center(&mut ssb_state, 0)?;
    }

    // Build initial augmented state (30-dim, unused elements stay zero)
    let mut pos_aug = SVector::<f64, 30>::zeros();
    let mut vel_aug = SVector::<f64, 30>::zeros();

    // Physical position and velocity (SSB-centered)
    pos_aug
        .fixed_rows_mut::<3>(0)
        .copy_from(&Vector3::from(ssb_state.pos));
    vel_aug
        .fixed_rows_mut::<3>(0)
        .copy_from(&Vector3::from(ssb_state.vel));

    // Phi_rr(0) = I3 (elements 3..12, column-major)
    pos_aug[3] = 1.0;
    pos_aug[7] = 1.0;
    pos_aug[11] = 1.0;

    // Phi_rv'(0) = I3 (elements 12..21 of vel_aug, column-major)
    vel_aug[12] = 1.0;
    vel_aug[16] = 1.0;
    vel_aug[20] = 1.0;

    let metadata = AccelSPKMeta {
        close_approach: None,
        non_grav_model,
        massive_obj,
    };

    let (pos_f, vel_f, _meta) = RadauIntegrator::integrate(
        &stm_augmented_accel,
        pos_aug,
        vel_aug,
        ssb_state.epoch,
        jd,
        metadata,
        Some(3),
    )?;

    let mut final_state = State::new(
        state.desig.clone(),
        jd,
        [pos_f[0], pos_f[1], pos_f[2]].into(),
        [vel_f[0], vel_f[1], vel_f[2]].into(),
        0, // SSB-centered after integration
    );

    // Restore the original center if needed.
    if original_center != 0 {
        let spk = &LOADED_SPK.try_read()?;
        spk.try_change_center(&mut final_state, original_center)?;
    }

    // Build the 6x(6+N) sensitivity matrix
    let ncols = 6 + np;
    let mut sens = DMatrix::<f64>::zeros(6, ncols);

    // 6x6 STM from the four 3x3 blocks
    let phi_rr = Matrix3::from_column_slice(&pos_f.as_slice()[3..12]);
    let phi_rv = Matrix3::from_column_slice(&pos_f.as_slice()[12..21]);
    let phi_vr = Matrix3::from_column_slice(&vel_f.as_slice()[3..12]);
    let phi_vv = Matrix3::from_column_slice(&vel_f.as_slice()[12..21]);

    sens.fixed_view_mut::<3, 3>(0, 0).copy_from(&phi_rr);
    sens.fixed_view_mut::<3, 3>(0, 3).copy_from(&phi_rv);
    sens.fixed_view_mut::<3, 3>(3, 0).copy_from(&phi_vr);
    sens.fixed_view_mut::<3, 3>(3, 3).copy_from(&phi_vv);

    // Parameter sensitivity columns (if any)
    for k in 0..np {
        let base = 21 + k * 3;
        for i in 0..3 {
            sens[(i, 6 + k)] = pos_f[base + i]; // dr_f/dp_k
            sens[(3 + i, 6 + k)] = vel_f[base + i]; // dv_f/dp_k
        }
    }

    Ok((final_state, sens))
}
