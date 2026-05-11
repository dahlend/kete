//! Batched N-body propagation over a packed `(planets | objects)` vector.
//!
//! [`vec_accel`] is the SPK-free ODE function: planets and objects are
//! integrated together as a single packed `DVector`, with massive bodies
//! occupying the leading `N` slots and test particles following.
//! [`propagate_n_body_vec`] is the high-level entry point that builds the
//! initial vector from an input list of states and returns the final
//! states at `jd_final`.

use kete_core::errors::Error;
use kete_core::forces::{FrozenNonGrav, GravParams, ParameterizedForce};
use kete_core::frames::{Equatorial, SunCenter};
use kete_core::integrators::RadauIntegrator;
use kete_core::prelude::{Desig, KeteResult};
use kete_core::state::State;
use kete_core::time::{TDB, Time};
use nalgebra::DVector;

use crate::spk::LOADED_SPK;

/// Metadata for [`vec_accel`]: the SPK-free bulk N-body ODE function.
pub struct AccelVecMeta<'a> {
    /// Per-object frozen non-gravitational force (values baked in), or `None`.
    pub non_gravs: Vec<Option<FrozenNonGrav>>,
    /// Massive bodies providing gravity, same order as the leading slots
    /// in the pos/vel vectors.
    pub massive_obj: &'a [GravParams],
}

impl std::fmt::Debug for AccelVecMeta<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AccelVecMeta")
            .field("n_objects", &self.non_gravs.len())
            .field("n_massive", &self.massive_obj.len())
            .finish()
    }
}

/// Compute the accel on a packed `(planets | objects)` pos/vel vector.
///
/// The first `N` objects in the vector are the massive bodies listed in
/// `meta.massive_obj` (in order). Objects beyond index `N` are test
/// particles subject to gravity from all massive bodies plus optional
/// non-gravitational forces.
///
/// # Errors
/// Fails on an impact.
pub fn vec_accel<D: nalgebra::Dim>(
    time: Time<TDB>,
    pos: &nalgebra::OVector<f64, D>,
    vel: &nalgebra::OVector<f64, D>,
    meta: &mut AccelVecMeta<'_>,
    exact_eval: bool,
) -> KeteResult<nalgebra::OVector<f64, D>>
where
    nalgebra::DefaultAllocator:
        nalgebra::allocator::Allocator<D> + nalgebra::allocator::Allocator<D, nalgebra::U2>,
{
    use kete_core::frames::Vector;
    use nalgebra::{U1, Vector3};
    use std::ops::AddAssign;

    let n_objects = pos.len() / 3;
    let n_massive = meta.massive_obj.len();
    let (dim, _) = pos.shape_generic();
    let mut accel = nalgebra::OVector::<f64, D>::zeros_generic(dim, U1);
    let mut accel_working = Vector3::zeros();

    for idx in 0..n_objects {
        let pos_idx = pos.fixed_rows::<3>(idx * 3);
        let vel_idx = vel.fixed_rows::<3>(idx * 3);
        for (idy, grav_params) in meta.massive_obj.iter().enumerate() {
            if idx == idy {
                continue;
            }
            accel_working.fill(0.0);
            let radius = grav_params.radius;
            let pos_idy = pos.fixed_rows::<3>(idy * 3);
            let vel_idy = vel.fixed_rows::<3>(idy * 3);
            let rel_pos = pos_idx - pos_idy;
            let rel_vel = vel_idx - vel_idy;
            if exact_eval & (rel_pos.norm() as f32 <= radius) {
                Err(Error::Impact(grav_params.naif_id, time))?;
            }
            grav_params.add_acceleration(&mut accel_working, &rel_pos, &rel_vel);
            if (grav_params.naif_id == 10)
                && (idx >= n_massive)
                && let Some(frozen) = &meta.non_gravs[idx - n_massive]
            {
                let pos_vec = Vector::<Equatorial>::new([rel_pos[0], rel_pos[1], rel_pos[2]]);
                let vel_vec = Vector::<Equatorial>::new([rel_vel[0], rel_vel[1], rel_vel[2]]);
                let ng_accel = frozen.accel(time, &pos_vec, &vel_vec, &[])?;
                let ng_v3: Vector3<f64> = ng_accel.into();
                accel_working += ng_v3;
            }
            accel.fixed_rows_mut::<3>(idx * 3).add_assign(accel_working);
        }
    }
    Ok(accel)
}

/// Propagate using n-body mechanics but skipping SPK queries.
/// This will propagate all planets and the Moon, so it may vary from SPK states slightly.
///
/// # Errors
/// Propagation may fail for a number of reasons, including lacking SPK information,
/// numerical singularities, or slow convergence of the integrator. Also returns an
/// error if `states` is empty, if `non_gravs.len()` does not match `states.len()`,
/// or if any `state.epoch` differs from the first state's epoch.
///
/// # Panics
/// Cannot panic in practice: the `.unwrap()` calls on `states.first()` and
/// `planet_states.first()` are gated by earlier length checks; `planet_states`
/// is matched to `simplified_planets()` which is non-empty by construction.
pub fn propagate_n_body_vec(
    states: Vec<State<Equatorial, SunCenter>>,
    jd_final: Time<TDB>,
    planet_states: Option<Vec<State<Equatorial>>>,
    non_gravs: Vec<Option<FrozenNonGrav>>,
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

    #[allow(clippy::missing_panics_doc, reason = "not possible by construction.")]
    let jd_init = states.first().unwrap().epoch;

    let mut pos: Vec<f64> = Vec::new();
    let mut vel: Vec<f64> = Vec::new();
    let mut desigs: Vec<Desig> = Vec::new();
    let spk = &LOADED_SPK.try_read()?;

    let planets = GravParams::simplified_planets();
    let planet_states = if let Some(ps) = planet_states {
        ps
    } else {
        let mut planet_states = Vec::new();
        for obj in &*planets {
            let planet = spk.try_get_state_with_center::<Equatorial>(obj.naif_id, jd_init, 10)?;
            planet_states.push(planet);
        }
        planet_states
    };

    if planet_states.len() != planets.len() {
        Err(Error::ValueError(
            "Input planet states must contain the correct number of states.".into(),
        ))?;
    }
    if planet_states.first().unwrap().epoch != jd_init {
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
        if jd_init != state.epoch {
            Err(Error::ValueError(
                "All input states must have the same JD".into(),
            ))?;
        }
        pos.append(&mut state.pos.into());
        vel.append(&mut state.vel.into());
        desigs.push(state.desig);
    }

    let meta = AccelVecMeta {
        non_gravs,
        massive_obj: &planets,
    };

    let (pos, vel, _) = {
        RadauIntegrator::integrate(
            &vec_accel,
            DVector::from(pos),
            DVector::from(vel),
            jd_init,
            jd_final,
            meta,
            None,
        )?
    };
    let sun_pos = pos.fixed_rows::<3>(0);
    let sun_vel = vel.fixed_rows::<3>(0);
    let mut all_states: Vec<State<_>> = Vec::new();
    for (idx, desig) in desigs.into_iter().enumerate() {
        let pos = pos.fixed_rows::<3>(idx * 3) - sun_pos;
        let vel = vel.fixed_rows::<3>(idx * 3) - sun_vel;
        let state = State::new(desig, jd_final, pos, vel, 10);
        all_states.push(state);
    }
    let final_states = all_states.split_off(planets.len());
    Ok((final_states, all_states))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::propagation::SpkNBody;
    use itertools::Itertools;
    use kete_core::forces::ParameterizedForce;
    use kete_core::frames::Vector;

    #[test]
    fn check_accelerations_equal() {
        crate::test_data::ensure_test_spk();
        let spk = &LOADED_SPK.try_read().unwrap();
        let jd = 2451545.0.into();
        let mut pos: Vec<f64> = Vec::new();
        let mut vel: Vec<f64> = Vec::new();

        let planets = GravParams::planets();
        for obj in &*planets {
            let planet = spk
                .try_get_state_with_center::<Equatorial>(obj.naif_id, jd, 0)
                .unwrap();
            pos.append(&mut planet.pos.into());
            vel.append(&mut planet.vel.into());
        }

        pos.append(&mut [0.0, 0.0, 0.5].into());
        vel.append(&mut [0.0, 0.0, 1.0].into());

        let accel = vec_accel(
            jd,
            &pos.into(),
            &vel.into(),
            &mut AccelVecMeta {
                non_gravs: vec![None],
                massive_obj: &planets,
            },
            false,
        )
        .unwrap()
        .iter()
        .copied()
        .skip(planets.len() * 3)
        .collect_vec();

        let accel2 = SpkNBody::new(spk, &planets)
            .accel(
                jd,
                &Vector::<Equatorial>::new([0.0, 0.0, 0.5]),
                &Vector::<Equatorial>::new([0.0, 0.0, 1.0]),
                &[],
            )
            .unwrap();
        assert!((accel[0] - accel2[0]).abs() < 1e-10);
        assert!((accel[1] - accel2[1]).abs() < 1e-10);
        assert!((accel[2] - accel2[2]).abs() < 1e-10);
    }
}
