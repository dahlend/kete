// BSD 3-Clause License
//
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

use crate::constants::GMS_SQRT;
use crate::frames::Equatorial;
use crate::prelude::{KeteResult, State};
use crate::propagation::{CentralAccelMeta, RK45Integrator, central_accel, central_accel_grad};
use nalgebra::{Const, Matrix6, SVector, U1, U6, Vector3};

fn stm_ivp_eqn(
    jd: f64,
    state: &SVector<f64, 42>,
    meta: &mut CentralAccelMeta,
    exact_eval: bool,
) -> KeteResult<SVector<f64, 42>> {
    let mut res = SVector::<f64, 42>::zeros();

    // first 6 values of the state are pos and vel respectively.
    let pos = Vector3::new(state[0], state[1], state[2]);
    let vel = Vector3::new(state[3], state[4], state[5]);
    let accel = central_accel(jd, &pos, &vel, meta, exact_eval)?;

    // the derivative of pos is the velocity, and the derivative of vel is the acceleration
    // set those as appropriate for the output state
    res.fixed_rows_mut::<3>(0).set_column(0, &vel);
    res.fixed_rows_mut::<3>(3).set_column(0, &accel);

    // the remainder of res is the state transition matrix calculation.
    let mut stm = Matrix6::<f64>::zeros();
    stm.fixed_view_mut::<3, 3>(0, 3)
        .set_diagonal(&Vector3::repeat(1.0));

    let grad = central_accel_grad(0.0, &pos, &vel, meta);
    let mut view = stm.fixed_view_mut::<3, 3>(3, 0);
    view.set_row(0, &grad.row(0));
    view.set_row(1, &grad.row(1));
    view.set_row(2, &grad.row(2));

    let vec_reshape = state
        .fixed_rows::<36>(6)
        .into_owned()
        .reshape_generic(U6, U6);
    res.rows_mut(6, 36).set_column(
        0,
        &(stm * vec_reshape)
            .into_owned()
            .reshape_generic(Const::<36>, U1),
    );

    Ok(res)
}

/// Compute a state transition matrix assuming only 2-body mechanics.
///
/// This uses a Runge-Kutta 4/5 algorithm.
pub fn compute_state_transition(
    state: &mut State<Equatorial>,
    jd: f64,
    central_mass: f64,
) -> ([[f64; 3]; 2], Matrix6<f64>) {
    let meta = CentralAccelMeta {
        mass_scaling: central_mass,
        ..Default::default()
    };

    let mut initial_state = SVector::<f64, 42>::zeros();

    initial_state.rows_mut(6, 36).set_column(
        0,
        &Matrix6::<f64>::identity().reshape_generic(Const::<36>, U1),
    );

    initial_state
        .fixed_rows_mut::<3>(0)
        .set_column(0, &state.pos.into());
    initial_state
        .fixed_rows_mut::<3>(3)
        .set_column(0, &(Vector3::from(state.vel) / GMS_SQRT));
    let rad = RK45Integrator::integrate(
        &stm_ivp_eqn,
        initial_state,
        state.jd * GMS_SQRT,
        jd * GMS_SQRT,
        meta,
        1e-12,
    )
    .unwrap();

    let vec_reshape = rad
        .0
        .fixed_rows::<36>(6)
        .into_owned()
        .reshape_generic(U6, U6)
        .transpose();

    let scaling_a = Matrix6::<f64>::from_diagonal(
        &[
            1.0,
            1.0,
            1.0,
            1.0 / GMS_SQRT,
            1.0 / GMS_SQRT,
            1.0 / GMS_SQRT,
        ]
        .into(),
    );
    let scaling_b =
        Matrix6::<f64>::from_diagonal(&[1.0, 1.0, 1.0, GMS_SQRT, GMS_SQRT, GMS_SQRT].into());
    (
        [
            rad.0.fixed_rows::<3>(0).into(),
            (rad.0.fixed_rows::<3>(3) * GMS_SQRT).into(),
        ],
        scaling_a * vec_reshape * scaling_b,
    )
}
