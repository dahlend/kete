//! # Propagation
//! The motion of objects (represented by a [`State`]) as a function of time.
//! There are multiple levels of precision available, each with different pros/cons
//! (usually performance related).
// BSD 3-Clause License
//
// Copyright (c) 2025, Dar Dahlen
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

use crate::constants::GravParams;
use crate::errors::Error;
use crate::frames::Equatorial;
use crate::prelude::{Desig, KeteResult};
use crate::spice::LOADED_SPK;
use crate::state::State;
use nalgebra::{DVector, Vector3};

mod acceleration;
mod kepler;
mod nongrav;
mod radau;
mod runge_kutta;
mod state_transition;

// expose the public methods in spk to the outside world.
pub use acceleration::{
    AccelSPKMeta, AccelVecMeta, CentralAccelMeta, accel_grad, central_accel, central_accel_grad,
    spk_accel, vec_accel,
};
pub use kepler::{
    PARABOLIC_ECC_LIMIT, analytic_2_body, compute_eccentric_anomaly, compute_true_anomaly,
    eccentric_anomaly_from_true, moid, propagate_two_body,
};
pub use nongrav::NonGravModel;
pub use radau::RadauIntegrator;
pub use runge_kutta::RK45Integrator;
pub use state_transition::compute_state_transition;

/// Using the Radau 15th order integrator, integrate the position and velocity of an
/// object assuming two body mechanics with a central object located at 0, 0 with the
/// mass of the sun.
///
/// This is primarily intended for testing the numerical integrator, it is strongly
/// recommended to use the Kepler analytic equation to do actual two body calculations.
pub fn propagate_two_body_radau(dt: f64, pos: &[f64; 3], vel: &[f64; 3]) -> ([f64; 3], [f64; 3]) {
    let res = RadauIntegrator::integrate(
        &central_accel,
        Vector3::from(*pos),
        Vector3::from(*vel),
        0.0,
        dt,
        CentralAccelMeta::default(),
    )
    .unwrap();
    (res.0.into(), res.1.into())
}

/// Propagate the state of an object, only considering linear motion.
///
/// This is a very poor approximation over more than a few minutes/hours, however it
/// is very fast.
pub fn propagate_linear(state: &State<Equatorial>, jd_final: f64) -> KeteResult<State<Equatorial>> {
    let dt = jd_final - state.jd;
    let mut pos: Vector3<f64> = state.pos.into();
    pos.iter_mut()
        .zip(state.vel)
        .for_each(|(p, v)| *p += v * dt);

    Ok(State::new(
        state.desig.clone(),
        jd_final,
        pos.into(),
        state.vel,
        state.center_id,
    ))
}

/// Propagate an object using full N-Body physics with the Radau 15th order integrator.
pub fn propagate_n_body_spk(
    mut state: State<Equatorial>,
    jd_final: f64,
    include_extended: bool,
    non_grav_model: Option<NonGravModel>,
) -> KeteResult<State<Equatorial>> {
    let center = state.center_id;
    let spk = &LOADED_SPK.try_read().unwrap();
    spk.try_change_center(&mut state, 0)?;

    let mass_list = {
        if include_extended {
            &GravParams::selected_masses()
        } else {
            &GravParams::planets()
        }
    };

    let metadata = AccelSPKMeta {
        close_approach: None,
        non_grav_model,
        massive_obj: mass_list,
    };

    let (pos, vel, _meta) = {
        RadauIntegrator::integrate(
            &spk_accel,
            state.pos.into(),
            state.vel.into(),
            state.jd,
            jd_final,
            metadata,
        )?
    };

    let mut new_state = State::new(state.desig.to_owned(), jd_final, pos.into(), vel.into(), 0);
    spk.try_change_center(&mut new_state, center)?;
    Ok(new_state)
}

/// Propagate an object using two body mechanics.
/// This is a brute force way to solve the kepler equations of motion as it uses Radau
/// as an integrator.
///
/// It is *strongly recommended* to use the `kepler.rs` code for this, as
/// it will be much more computationally efficient.
pub fn propagation_central(state: &State<Equatorial>, jd_final: f64) -> KeteResult<[[f64; 3]; 2]> {
    let pos: Vector3<f64> = state.pos.into();
    let vel: Vector3<f64> = state.vel.into();
    let (pos, vel, _meta) = RadauIntegrator::integrate(
        &central_accel,
        pos,
        vel,
        state.jd,
        jd_final,
        CentralAccelMeta::default(),
    )?;
    Ok([pos.into(), vel.into()])
}

/// Propagate using n-body mechanics but skipping SPK queries.
/// This will propagate all planets and the Moon, so it may vary from SPK states slightly.
#[allow(clippy::type_complexity, reason = "Not practical to avoid this")]
pub fn propagate_n_body_vec(
    states: Vec<State<Equatorial>>,
    jd_final: f64,
    planet_states: Option<Vec<State<Equatorial>>>,
    non_gravs: Vec<Option<NonGravModel>>,
) -> KeteResult<(Vec<State<Equatorial>>, Vec<State<Equatorial>>)> {
    if states.is_empty() {
        Err(Error::ValueError(
            "State vector is empty, propagation cannot continue".into(),
        ))?;
    }

    if non_gravs.len() != states.len() {
        Err(Error::ValueError(
            "Number of non-grav models doesnt match the number of provided objects.".into(),
        ))?;
    }

    let jd_init = states.first().unwrap().jd;

    let mut pos: Vec<f64> = Vec::new();
    let mut vel: Vec<f64> = Vec::new();
    let mut desigs: Vec<Desig> = Vec::new();

    let planet_states = planet_states.unwrap_or_else(|| {
        let spk = &LOADED_SPK.try_read().unwrap();
        let mut planet_states = Vec::new();
        for obj in GravParams::simplified_planets() {
            let planet = spk
                .try_get_state_with_center::<Equatorial>(obj.naif_id, jd_init, 10)
                .expect("Failed to find state for the provided initial jd");
            planet_states.push(planet);
        }
        planet_states
    });

    if planet_states.len() != GravParams::simplified_planets().len() {
        Err(Error::ValueError(
            "Input planet states must contain the correct number of states.".into(),
        ))?;
    }
    if planet_states.first().unwrap().jd != jd_init {
        Err(Error::ValueError(
            "Planet states JD must match JD of input state.".into(),
        ))?;
    }
    for planet_state in planet_states {
        pos.append(&mut planet_state.pos.into());
        vel.append(&mut planet_state.vel.into());
        desigs.push(planet_state.desig);
    }

    for state in states {
        if jd_init != state.jd {
            Err(Error::ValueError(
                "All input states must have the same JD".into(),
            ))?;
        }
        if state.center_id != 10 {
            Err(Error::ValueError(
                "Center of all states must be 10 (the Sun).".into(),
            ))?;
        }
        pos.append(&mut state.pos.into());
        vel.append(&mut state.vel.into());
        desigs.push(state.desig);
    }

    let meta = AccelVecMeta {
        non_gravs,
        massive_obj: &GravParams::simplified_planets(),
    };

    let (pos, vel, _) = {
        RadauIntegrator::integrate(
            &vec_accel,
            DVector::from(pos),
            DVector::from(vel),
            jd_init,
            jd_final,
            meta,
        )?
    };
    let sun_pos = pos.fixed_rows::<3>(0);
    let sun_vel = vel.fixed_rows::<3>(0);
    let mut all_states: Vec<State<_>> = Vec::new();
    for (idx, desig) in desigs.into_iter().enumerate() {
        let pos = pos.fixed_rows::<3>(idx * 3) - sun_pos;
        let vel = vel.fixed_rows::<3>(idx * 3) - sun_vel;
        let state = State::new(desig, jd_final, pos.into(), vel.into(), 10);
        all_states.push(state);
    }
    let final_states = all_states.split_off(GravParams::simplified_planets().len());
    Ok((final_states, all_states))
}
