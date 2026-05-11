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

use kete_core::forces::{ForceSet, FrozenNonGrav, GravParams, ParameterMask};
use kete_core::frames::{Equatorial, SSB};
use kete_core::prelude::{KeteResult, State};
use kete_core::state::propagate_with_stm;
use kete_core::time::{TDB, Time};

use super::recenter::Recenter;
use super::spk_n_body::SpkNBody;
use crate::spk::LOADED_SPK;
use nalgebra::DMatrix;

/// Compute the state transition matrix and optional parameter sensitivities using the
/// Radau 15th-order integrator with full N-body physics.
///
/// The input state must be typed as `State<Equatorial, SSB>`, enforcing at compile
/// time that the center is the solar system barycenter.  The returned state is also
/// SSB-centered.
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
/// Returns an error if SPK queries fail or integration does not converge.
pub fn compute_state_transition(
    state: &State<Equatorial, SSB>,
    jd: Time<TDB>,
    include_asteroids: bool,
    non_grav: Option<&FrozenNonGrav>,
) -> KeteResult<(State<Equatorial, SSB>, DMatrix<f64>)> {
    let spk = LOADED_SPK.try_read()?;

    let planets = if include_asteroids {
        GravParams::selected_masses()
    } else {
        GravParams::planets()
    };

    // Non-grav (when present) composes via `ForceSet` with `Recenter<SSB, _>`
    // wrapping an all-None variational mask. Values are extracted from the
    // frozen input mask and passed as `free_params` to `propagate_with_stm`
    // so the STM gains parameter-sensitivity columns.
    //
    // Gravity-only path stays bare (`SpkNBody` directly) -- the pure n-body
    // hot path, bit-identical to trajectory-only propagation.
    let (pos_f, vel_f, sens) = match non_grav {
        None => propagate_with_stm(
            &SpkNBody::new(&spk, &planets),
            state.pos.into(),
            state.vel.into(),
            &[],
            state.epoch,
            jd,
        )?,
        Some(frozen) => {
            let values = frozen.values();
            let variational = ParameterMask::new(frozen.inner.clone(), vec![None; values.len()])?;
            let force_set: ForceSet<'_, Equatorial, SSB> = ForceSet::new()
                .with(Box::new(SpkNBody::new(&spk, &planets)))
                .with(Box::new(Recenter::<SSB, _>::new(&spk, variational)));
            propagate_with_stm(
                &force_set,
                state.pos.into(),
                state.vel.into(),
                values,
                state.epoch,
                jd,
            )?
        }
    };

    let final_state = State {
        desig: state.desig.clone(),
        epoch: jd,
        pos: pos_f.into(),
        vel: vel_f.into(),
        center: SSB,
    };

    Ok((final_state, sens))
}
