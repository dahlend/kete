//! SPK-dependent propagation functions.
//!
//! These functions require loaded SPICE kernels (SPK files) to query planet states.
//! Pure math propagation functions remain in `kete_core::propagation`.

use kete_core::constants::GravParams;
use kete_core::errors::Error;
use kete_core::frames::Equatorial;
use kete_core::prelude::{Desig, KeteResult, SimultaneousStates};
use kete_core::propagation::{
    AccelVecMeta, NonGravModel, PC15, RadauIntegrator, analytic_2_body, vec_accel,
};
use kete_core::state::State;
use kete_core::time::{TDB, Time};

use crate::spice::{LOADED_SPK, SpkCollection};

use itertools::Itertools;
use nalgebra::{DVector, SMatrix, SVector, Vector3};
use rayon::prelude::*;

/// Metadata for the [`spk_accel`] function.
#[derive(Debug)]
pub struct AccelSPKMeta<'a> {
    /// The non-gravitational forces.
    /// If this is not provided, only standard gravitational model is applied.
    /// If this are provided, then the effects of the Non-Grav terms are added.
    pub non_grav_model: Option<NonGravModel>,

    /// The list of massive objects to apply during SPK computation.
    /// This list contains the ID of the object in the SPK along with the mass and
    /// radius of the object. Mass is given in fractions of solar mass and radius is
    /// in AU.
    pub massive_obj: &'a [GravParams],

    /// Reference to the loaded SPK collection.
    /// Stored here so the lock is acquired once and reused for the entire integration.
    pub spk: &'a SpkCollection,
}

/// Compute the accel on an object which experiences acceleration due to all massive
/// objects contained within the Spice Kernel SPKs. This uses the planets and the Moon,
/// and applies a General Relativity correction for the Sun and Jupiter.
///
/// Whatever objects are present in the metadata struct will be used as sources of mass.
/// These objects' states are queried from the loaded SPICE kernels, so this will fail
/// if the SPICE kernels are not present/loaded.
///
/// Typically this relies on DE440s, which contains the years ~1800-2200.
///
/// Metadata:
///     Metadata here records the closest approach to any planet. This value is updated
///     every time the function is called.
///
/// # Arguments
///
/// * `time` - Time of the evaluation in JD in TDB scaled time multiplied by
///   `SUN_GMS_SQRT`.
/// * `pos` - A vector which defines the position with respect to the SSB in AU.
/// * `vel` - A vector which defines the velocity with respect to the SSB in AU/Day
///   multiplied by `SUN_GMS_SQRT`.
/// * `meta` - Metadata object [`AccelSPKMeta`] which records values at each integration
///   step.
///
/// # Errors
/// This is actually infallible, but must have this signature for the integrator.
pub fn spk_accel(
    time: Time<TDB>,
    pos: &Vector3<f64>,
    vel: &Vector3<f64>,
    meta: &mut AccelSPKMeta<'_>,
    exact_eval: bool,
) -> KeteResult<Vector3<f64>> {
    let mut accel = Vector3::<f64>::zeros();

    let spk = meta.spk;

    for grav_params in meta.massive_obj {
        let id = grav_params.naif_id;
        let radius = grav_params.radius;
        let state = spk.try_get_state_with_center::<Equatorial>(id, time, 0)?;
        let rel_pos: Vector3<f64> = pos - Vector3::from(state.pos);
        let rel_vel: Vector3<f64> = vel - Vector3::from(state.vel);

        if exact_eval {
            let r = rel_pos.norm();

            if r as f32 <= radius {
                Err(Error::Impact(id, time))?;
            }
        }
        grav_params.add_acceleration(&mut accel, &rel_pos, &rel_vel);

        if grav_params.naif_id == 10
            && let Some(non_grav) = &meta.non_grav_model
        {
            non_grav.add_acceleration(&mut accel, &rel_pos, &rel_vel);
        }
    }
    Ok(accel)
}

/// Like [`spk_accel`], but uses pre-fetched planet states instead of querying SPK.
///
/// `cached_states` must contain one `(pos, vel)` pair per entry in
/// `meta.massive_obj`, in the same order, already SSB-centered.
///
/// This avoids redundant SPK interpolations when the same planet states are needed
/// for multiple evaluations at the same time (e.g. finite-difference Jacobians).
pub(crate) fn spk_accel_cached(
    time: Time<TDB>,
    pos: &Vector3<f64>,
    vel: &Vector3<f64>,
    cached_states: &[(Vector3<f64>, Vector3<f64>)],
    meta: &mut AccelSPKMeta<'_>,
    exact_eval: bool,
) -> KeteResult<Vector3<f64>> {
    let mut accel = Vector3::<f64>::zeros();

    for (grav_params, (body_pos, body_vel)) in meta.massive_obj.iter().zip(cached_states) {
        let radius = grav_params.radius;
        let rel_pos: Vector3<f64> = pos - body_pos;
        let rel_vel: Vector3<f64> = vel - body_vel;

        if exact_eval {
            let r = rel_pos.norm();

            if r as f32 <= radius {
                Err(Error::Impact(grav_params.naif_id, time))?;
            }
        }
        grav_params.add_acceleration(&mut accel, &rel_pos, &rel_vel);

        if grav_params.naif_id == 10
            && let Some(non_grav) = &meta.non_grav_model
        {
            non_grav.add_acceleration(&mut accel, &rel_pos, &rel_vel);
        }
    }
    Ok(accel)
}

/// Convert the second order ODE acceleration function into a first order.
/// This allows the second order ODE to be used with the picard integrator.
///
/// The `state_vec` is made up of concatenated position and velocity vectors.
/// Otherwise this is just a thin wrapper over the [`spk_accel`] function.
///
/// # Errors
/// Fails when SPK queries fail.
pub fn spk_accel_first_order(
    time: Time<TDB>,
    state_vec: &SVector<f64, 6>,
    meta: &mut AccelSPKMeta<'_>,
    exact_eval: bool,
) -> KeteResult<SVector<f64, 6>> {
    let pos: Vector3<f64> = state_vec.fixed_rows::<3>(0).into();
    let vel: Vector3<f64> = state_vec.fixed_rows::<3>(3).into();
    let accel = spk_accel(time, &pos, &vel, meta, exact_eval)?;
    let mut res = SVector::<f64, 6>::zeros();
    res.fixed_rows_mut::<3>(0).set_column(0, &vel);
    res.fixed_rows_mut::<3>(3).set_column(0, &accel);
    Ok(res)
}

/// Propagate an object using full N-Body physics with the Radau 15th order integrator.
///
/// # Errors
/// Propagation may fail for a number of reasons, including missing SPK data,
/// integration near singularities, or impacts.
pub fn propagate_n_body_spk(
    mut state: State<Equatorial>,
    jd_final: Time<TDB>,
    include_extended: bool,
    non_grav_model: Option<NonGravModel>,
) -> KeteResult<State<Equatorial>> {
    let center = state.center_id;
    let spk = &LOADED_SPK.try_read()?;
    spk.try_change_center(&mut state, 0)?;

    let mass_list = {
        if include_extended {
            &GravParams::selected_masses()
        } else {
            &GravParams::planets()
        }
    };

    let metadata = AccelSPKMeta {
        non_grav_model,
        massive_obj: mass_list,
        spk,
    };

    let (pos, vel, _meta) = {
        RadauIntegrator::integrate(
            &spk_accel,
            state.pos.into(),
            state.vel.into(),
            state.epoch,
            jd_final,
            metadata,
            None,
        )?
    };

    let mut new_state = State::new(state.desig.clone(), jd_final, pos.into(), vel.into(), 0);
    spk.try_change_center(&mut new_state, center)?;
    Ok(new_state)
}

/// Propagate the provided [`Vec<State<Equatorial>>`] using N body mechanics to the
/// specified times, no approximations are made, this can be very CPU intensive.
///
/// This uses rayon to use as many cores as are available.
///
/// This does not compute light delay, however it does include corrections for general
/// relativity due to the Sun.
///
/// # Errors
///
/// Integration may fail for a large number of reasons. This function accepts a
/// ``suppress_errors`` parameter which may be used to continue propagation as long
/// as possible for the most objects as possible.
pub fn propagation_n_body_spk_par(
    states: Vec<State<Equatorial>>,
    jd: Time<TDB>,
    include_asteroids: bool,
    non_gravs: Option<Vec<Option<NonGravModel>>>,
    suppress_errors: bool,
) -> KeteResult<SimultaneousStates> {
    let non_gravs = non_gravs.unwrap_or(vec![None; states.len()]);

    if states.len() != non_gravs.len() {
        Err(Error::ValueError(
            "non_gravs must be the same length as states.".into(),
        ))?;
    }

    let res: KeteResult<Vec<_>> = states
        .into_iter()
        .zip(non_gravs)
        .collect_vec()
        .into_par_iter()
        .with_min_len(10)
        .map(|(state, model)| {
            let center = state.center_id;
            let desig = state.desig.clone();

            if !state.is_finite() {
                if !suppress_errors {
                    Err(Error::ValueError("Input state contains NaNs.".into()))?;
                }
                return Ok(State::<Equatorial>::new_nan(desig, jd, center));
            }
            match propagate_n_body_spk(state, jd, include_asteroids, model) {
                Ok(state) => Ok(state),
                Err(er) => {
                    if suppress_errors {
                        Ok(State::<Equatorial>::new_nan(desig, jd, center))
                    } else {
                        Err(er)?
                    }
                }
            }
        })
        .collect();

    SimultaneousStates::new_exact(res?, None)
}

/// Initialization function for the picard integrator which initializes the state
/// using two body mechanics.
fn picard_two_body_init<const N: usize>(
    times: &[Time<TDB>; N],
    init_pos: &SVector<f64, 6>,
) -> SMatrix<f64, 6, N> {
    let pos: Vector3<f64> = init_pos.fixed_rows::<3>(0).into();
    let vel: Vector3<f64> = init_pos.fixed_rows::<3>(3).into();
    let t0 = times[0];

    let mut res: SMatrix<f64, 6, N> = SMatrix::zeros();
    res.fixed_rows_mut::<3>(0).set_column(0, &pos);
    res.fixed_rows_mut::<3>(3).set_column(0, &vel);

    for (idx, t) in times.iter().enumerate().skip(1) {
        let dt = *t - t0;
        let (p, v) = analytic_2_body(dt, &pos, &vel, None).unwrap();

        res.fixed_rows_mut::<3>(0).set_column(idx, &p);
        res.fixed_rows_mut::<3>(3).set_column(idx, &v);
    }
    res
}

/// Propagate an object using the Picard integrator with full N-Body physics.
///
/// # Errors
/// Propagation may fail for a number of reasons, including missing SPK data,
/// integration near singularities, or impacts.
pub fn propagate_picard_n_body_spk(
    mut state: State<Equatorial>,
    jd_final: Time<TDB>,
    include_extended: bool,
    non_grav_model: Option<NonGravModel>,
) -> KeteResult<State<Equatorial>> {
    let center = state.center_id;
    let spk = &LOADED_SPK.try_read()?;
    spk.try_change_center(&mut state, 0)?;

    let mass_list = {
        if include_extended {
            &GravParams::selected_masses()
        } else {
            &GravParams::planets()
        }
    };

    let mut metadata = AccelSPKMeta {
        non_grav_model,
        massive_obj: mass_list,
        spk,
    };

    let integrator = &PC15;

    let mut state_vec = SVector::<f64, 6>::zeros();
    state_vec
        .fixed_rows_mut::<3>(0)
        .set_column(0, &state.pos.into());
    state_vec
        .fixed_rows_mut::<3>(3)
        .set_column(0, &state.vel.into());

    let final_state_vec = {
        integrator.integrate(
            &spk_accel_first_order,
            &picard_two_body_init,
            state_vec,
            state.epoch,
            jd_final,
            1.0,
            &mut metadata,
        )?
    };

    let pos: Vector3<f64> = final_state_vec.fixed_rows::<3>(0).into();
    let vel: Vector3<f64> = final_state_vec.fixed_rows::<3>(3).into();

    let mut new_state = State::new(state.desig.clone(), jd_final, pos.into(), vel.into(), 0);
    spk.try_change_center(&mut new_state, center)?;
    Ok(new_state)
}

/// Propagate using n-body mechanics but skipping SPK queries.
/// This will propagate all planets and the Moon, so it may vary from SPK states slightly.
///
/// # Errors
/// Propagation may fail for a number of reasons, including lacking SPK information,
/// numerical singularities, or slow convergence of the integrator.
///
/// # Panics
/// Panics if planet states not provided, and cannot find the state in loaded SPKs.
pub fn propagate_n_body_vec(
    states: Vec<State<Equatorial>>,
    jd_final: Time<TDB>,
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

    #[allow(clippy::missing_panics_doc, reason = "not possible by construction.")]
    let jd_init = states.first().unwrap().epoch;

    let mut pos: Vec<f64> = Vec::new();
    let mut vel: Vec<f64> = Vec::new();
    let mut desigs: Vec<Desig> = Vec::new();
    let spk = &LOADED_SPK.try_read()?;

    let planet_states = if let Some(ps) = planet_states {
        ps
    } else {
        let mut planet_states = Vec::new();
        for obj in GravParams::simplified_planets() {
            let planet = spk.try_get_state_with_center::<Equatorial>(obj.naif_id, jd_init, 10)?;
            planet_states.push(planet);
        }
        planet_states
    };

    if planet_states.len() != GravParams::simplified_planets().len() {
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
            None,
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

#[cfg(test)]
mod tests {
    use itertools::Itertools;

    use super::*;
    use kete_core::propagation::{AccelVecMeta, vec_accel};

    #[test]
    fn check_accelerations_equal() {
        let spk = &LOADED_SPK.try_read().unwrap();
        let jd = 2451545.0.into();
        let mut pos: Vec<f64> = Vec::new();
        let mut vel: Vec<f64> = Vec::new();

        for obj in GravParams::planets() {
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
                massive_obj: &GravParams::planets(),
            },
            false,
        )
        .unwrap()
        .iter()
        .copied()
        .skip(GravParams::planets().len() * 3)
        .collect_vec();

        let accel2 = spk_accel(
            jd,
            &[0.0, 0.0, 0.5].into(),
            &[0.0, 0.0, 1.0].into(),
            &mut AccelSPKMeta {
                non_grav_model: None,
                massive_obj: &GravParams::planets(),
                spk,
            },
            false,
        )
        .unwrap();
        assert!((accel[0] - accel2[0]).abs() < 1e-10);
        assert!((accel[1] - accel2[1]).abs() < 1e-10);
        assert!((accel[2] - accel2[2]).abs() < 1e-10);
    }
}
