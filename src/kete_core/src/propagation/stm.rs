// SPDX-FileCopyrightText: 2026 Dar Dahlen
// SPDX-License-Identifier: BSD-3-Clause

//! State Transition matrix computation

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
