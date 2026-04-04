//! State Transition matrix computation
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

use kete_core::forces::{GravParams, NonGravModel};
use kete_core::frames::Equatorial;
use kete_core::integrators::RadauIntegrator;
use kete_core::prelude::{KeteResult, State};
use kete_core::time::{TDB, Time};

use crate::jacobian::{n_params, stm_augmented_accel};
use crate::propagation::AccelSPKMeta;
use crate::spk::LOADED_SPK;
use nalgebra::{DMatrix, Matrix3, SVector, Vector3};

/// Compute the state transition matrix and optional parameter sensitivities using the
/// Radau 15th-order integrator with full N-body physics.
///
/// The input state **must** be centered on the solar system barycenter (SSB,
/// `center_id == 0`).  The returned state is also SSB-centered.  Callers that
/// work in a different center must convert before calling and convert back after.
///
/// When `include_asteroids` is `true`, the force model includes asteroid
/// masses from [`GravParams::selected_masses()`]; otherwise only the
/// planets and Moon from [`GravParams::planets()`] are used.
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
/// # Errors
/// Returns an error if `state.center_id != 0`, or if SPK queries fail or
/// integration does not converge.
pub fn compute_state_transition(
    state: &State<Equatorial>,
    jd: Time<TDB>,
    include_asteroids: bool,
    non_grav_model: Option<NonGravModel>,
) -> KeteResult<(State<Equatorial>, DMatrix<f64>)> {
    let np = n_params(non_grav_model.as_ref());

    if state.center_id != 0 {
        return Err(kete_core::errors::Error::ValueError(
            "compute_state_transition requires an SSB-centered state (center_id == 0)".into(),
        ));
    }

    // Build initial augmented state (30-dim, unused elements stay zero)
    let mut pos_aug = SVector::<f64, 30>::zeros();
    let mut vel_aug = SVector::<f64, 30>::zeros();

    // Physical position and velocity (SSB-centered)
    pos_aug
        .fixed_rows_mut::<3>(0)
        .copy_from(&Vector3::from(state.pos));
    vel_aug
        .fixed_rows_mut::<3>(0)
        .copy_from(&Vector3::from(state.vel));

    // Phi_rr(0) = I3 (elements 3..12, column-major)
    pos_aug[3] = 1.0;
    pos_aug[7] = 1.0;
    pos_aug[11] = 1.0;

    // Phi_rv'(0) = I3 (elements 12..21 of vel_aug, column-major)
    vel_aug[12] = 1.0;
    vel_aug[16] = 1.0;
    vel_aug[20] = 1.0;

    let mass_list = if include_asteroids {
        GravParams::selected_masses().to_vec()
    } else {
        GravParams::planets()
    };

    let spk = &LOADED_SPK.try_read()?;
    let metadata = AccelSPKMeta {
        non_grav_model,
        massive_obj: &mass_list,
        spk,
    };

    let (pos_f, vel_f, _meta) = RadauIntegrator::integrate(
        &stm_augmented_accel,
        pos_aug,
        vel_aug,
        state.epoch,
        jd,
        metadata,
        Some(3),
    )?;

    let final_state = State::new(
        state.desig.clone(),
        jd,
        [pos_f[0], pos_f[1], pos_f[2]].into(),
        [vel_f[0], vel_f[1], vel_f[2]].into(),
        // SSB-centered
        0,
    );

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
            // dr_f/dp_k
            sens[(i, 6 + k)] = pos_f[base + i];
            // dv_f/dp_k
            sens[(3 + i, 6 + k)] = vel_f[base + i];
        }
    }

    Ok((final_state, sens))
}
