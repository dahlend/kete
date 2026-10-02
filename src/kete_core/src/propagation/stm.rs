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

use super::n_body::NBody;
use crate::ephemeris::Ephemeris;
use crate::errors::KeteResult;
use crate::forces::ParameterizedForce;
use crate::frames::{Equatorial, SSB, SunCenter};
use crate::state::{State, propagate_with_stm};
use crate::time::{TDB, Time};
use nalgebra::DMatrix;

/// Compute the state transition matrix and optional parameter sensitivities using the
/// Radau 15th-order integrator with full N-body physics, body states from `ephem`.
///
/// The input state must be typed as `State<Equatorial, SSB>`, enforcing at compile
/// time that the center is the solar system barycenter.  The returned state is also
/// SSB-centered.
///
/// When `include_extended` is `true`, the force model includes asteroid
/// masses from `GravParams::selected_masses()`; otherwise only the
/// planets and Moon from `GravParams::planets()` are used.
///
/// When `non_grav` is `Some((force, free_params))`, the state is propagated under
/// gravity plus `force` evaluated at `free_params`, and the STM gains one
/// parameter-sensitivity column `d(r_f, v_f) / d p_k` per free parameter of `force`.
///
/// Returns the propagated [`State`] and a 6x(6+N) sensitivity matrix where N is
/// the number of free parameters of `force`, 0 without one. Column ordering is:
///
/// ```text
/// cols 0-5  : 6x6 state transition matrix  d(r_f, v_f) / d(r_0, v_0)
/// col  6+k  : parameter sensitivity        d(r_f, v_f) / dp_k
/// ```
///
/// # Errors
/// Returns an error if an ephemeris query fails or integration does not converge.
pub fn compute_state_transition<E, F>(
    ephem: &E,
    state: &State<Equatorial, SSB>,
    jd: Time<TDB>,
    include_extended: bool,
    non_grav: Option<(&F, &[f64])>,
) -> KeteResult<(State<Equatorial, SSB>, DMatrix<f64>)>
where
    E: Ephemeris,
    F: ParameterizedForce<Frame = Equatorial, Center = SunCenter> + Clone,
{
    let (non_grav, free_params) = non_grav.map_or((None, &[][..]), |(force, params)| {
        (Some(force.clone()), params)
    });
    let (pos_f, vel_f, sens) = propagate_with_stm(
        &NBody::with_non_grav(ephem, include_extended, non_grav),
        state.pos.into(),
        state.vel.into(),
        free_params,
        state.epoch,
        jd,
    )?;

    let final_state = State {
        desig: state.desig.clone(),
        epoch: jd,
        pos: pos_f.into(),
        vel: vel_f.into(),
        center: SSB,
    };

    Ok((final_state, sens))
}
