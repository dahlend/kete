//! SPK-dependent propagation functions.
//!
//! These functions require loaded SPICE kernels (SPK files) to query planet states.
//! Pure math integrators and force models remain in `kete_core::integrators`,
//! `kete_core::forces`, and `kete_core::kepler`.

use kete_core::elements::CometElements;
use kete_core::errors::Error;
use kete_core::forces::GravParams;
use kete_core::forces::{AccelVecMeta, NonGravModel, vec_accel};
use kete_core::frames::{Ecliptic, Equatorial};
use kete_core::integrators::{PC15, RadauIntegrator};
use kete_core::kepler::analytic_2_body;
use kete_core::prelude::{Desig, KeteResult, SimultaneousStates};
use kete_core::state::State;
use kete_core::time::{TDB, Time};

use crate::spk::{LOADED_SPK, SpkCollection};

use itertools::Itertools;
use nalgebra::{DVector, SMatrix, Vector3};
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

/// Initialization function for the second-order picard integrator which initializes
/// the state using two body mechanics.
fn picard_two_body_init_second_order<const N: usize>(
    times: &[Time<TDB>; N],
    init_pos: &Vector3<f64>,
    init_vel: &Vector3<f64>,
) -> (SMatrix<f64, 3, N>, SMatrix<f64, 3, N>) {
    let t0 = times[0];

    let mut pos_mat: SMatrix<f64, 3, N> = SMatrix::zeros();
    let mut vel_mat: SMatrix<f64, 3, N> = SMatrix::zeros();
    pos_mat.set_column(0, init_pos);
    vel_mat.set_column(0, init_vel);

    for (idx, t) in times.iter().enumerate().skip(1) {
        let dt = *t - t0;
        let (p, v) = analytic_2_body(dt, init_pos, init_vel, None).unwrap();
        pos_mat.set_column(idx, &p);
        vel_mat.set_column(idx, &v);
    }
    (pos_mat, vel_mat)
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

    let (pos, vel) = integrator.integrate_second_order(
        &spk_accel,
        &picard_two_body_init_second_order,
        state.pos.into(),
        state.vel.into(),
        state.epoch,
        jd_final,
        1.0,
        &mut metadata,
    )?;

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

/// Get the state of a body at a given time, using the SPK if possible,
/// otherwise propagating with N-body.
fn state_at_time(
    state: &State<Equatorial>,
    time: Time<TDB>,
    spk: &SpkCollection,
    spk_id: Option<i32>,
    center: i32,
    include_extended: bool,
) -> KeteResult<State<Equatorial>> {
    if let Some(id) = spk_id {
        spk.try_get_state_with_center(id, time, center)
    } else {
        propagate_n_body_spk(state.clone(), time, include_extended, None)
    }
}

/// Find the epoch and distance of closest approach between two objects.
///
/// Both objects are propagated using full N-body mechanics over the search
/// window. If either state's designation corresponds to a body available in
/// the loaded SPK kernels, the SPK ephemeris is used directly instead of
/// N-body propagation.
///
/// A coarse grid scan followed by golden-section refinement locates the
/// minimum separation.
///
/// # Errors
/// Returns an error if the states have different center IDs or the time
/// window is non-positive.
pub fn closest_approach(
    state_a: &State<Equatorial>,
    state_b: &State<Equatorial>,
    jd_start: Time<TDB>,
    jd_end: Time<TDB>,
    include_extended: bool,
) -> KeteResult<(Time<TDB>, f64)> {
    if state_a.center_id != state_b.center_id {
        return Err(Error::ValueError(
            "Both states must share the same center_id".into(),
        ));
    }
    let center = state_a.center_id;

    let span = jd_end.jd - jd_start.jd;
    if span <= 0.0 {
        return Err(Error::ValueError("jd_end must be after jd_start".into()));
    }

    // If a state represents a body in the SPK, look it up directly instead of
    // propagating. This avoids the self-impact bug (propagating Earth detects
    // distance 0 to SPICE body 399) and is more accurate for major bodies.
    let spk = LOADED_SPK.try_read().map_err(Error::from)?;
    let spk_id_a = state_a.desig.clone().naif_id().filter(|id| {
        spk.try_get_state_with_center::<Equatorial>(*id, jd_start, center)
            .is_ok()
    });
    let spk_id_b = state_b.desig.clone().naif_id().filter(|id| {
        spk.try_get_state_with_center::<Equatorial>(*id, jd_start, center)
            .is_ok()
    });

    // Adaptive sample count based on orbital periods.
    let elem_a = CometElements::from_state(&state_a.clone().into_frame::<Ecliptic>());
    let elem_b = CometElements::from_state(&state_b.clone().into_frame::<Ecliptic>());
    let min_period = elem_a.orbital_period().min(elem_b.orbital_period());
    #[allow(clippy::cast_sign_loss, reason = "always positive by construction")]
    let n_samples = if min_period.is_finite() && min_period > 0.0 {
        ((span / min_period) * 20.0).ceil().max(200.0) as usize
    } else {
        200
    };
    let dt = span / n_samples as f64;

    // Get both objects at jd_start.
    let mut cur_a = state_at_time(state_a, jd_start, &spk, spk_id_a, center, include_extended)?;
    let mut cur_b = state_at_time(state_b, jd_start, &spk, spk_id_b, center, include_extended)?;

    // Coarse search: step through the time window, tracking the best index and
    // the state one step before it (used as the reference for refinement).
    let mut best_idx = 0;
    let mut best_dist = (Vector3::from(cur_a.pos) - Vector3::from(cur_b.pos)).norm();
    let mut prev_a = cur_a.clone();
    let mut prev_b = cur_b.clone();

    for i in 1..=n_samples {
        let t: Time<TDB> = (jd_start.jd + i as f64 * dt).into();
        let old_a = cur_a.clone();
        let old_b = cur_b.clone();
        cur_a = state_at_time(&cur_a, t, &spk, spk_id_a, center, include_extended)?;
        cur_b = state_at_time(&cur_b, t, &spk, spk_id_b, center, include_extended)?;
        let d = (Vector3::from(cur_a.pos) - Vector3::from(cur_b.pos)).norm();
        if d < best_dist {
            best_dist = d;
            best_idx = i;
            prev_a = old_a;
            prev_b = old_b;
        }
    }

    // Bracket the minimum: one step on each side.
    let lo_idx = best_idx.saturating_sub(1);
    let hi_idx = (best_idx + 1).min(n_samples);
    let lo = jd_start.jd + lo_idx as f64 * dt;
    let hi = jd_start.jd + hi_idx as f64 * dt;

    // Golden-section search to refine.
    // Work in offsets from jd_start to avoid precision loss on large JD values.
    let tol = 1e-10; // ~0.01 ms
    let base = jd_start.jd;
    let lo_off = lo - base;
    let hi_off = hi - base;

    let ref_a = &prev_a;
    let ref_b = &prev_b;

    // Capture any propagation error from inside the closure.
    let mut inner_err: Option<Error> = None;
    let dist_at = |off: f64| -> f64 {
        if inner_err.is_some() {
            return f64::NAN;
        }
        let t: Time<TDB> = (base + off).into();
        let (sa, sb) = match (
            state_at_time(ref_a, t, &spk, spk_id_a, center, include_extended),
            state_at_time(ref_b, t, &spk, spk_id_b, center, include_extended),
        ) {
            (Ok(a), Ok(b)) => (a, b),
            (Err(e), _) | (_, Err(e)) => {
                inner_err = Some(e);
                return f64::NAN;
            }
        };
        (Vector3::from(sa.pos) - Vector3::from(sb.pos)).norm_squared()
    };

    let best_off = kete_stats::fitting::golden_section_search(dist_at, lo_off, hi_off, tol)
        .map_err(|_| {
            inner_err.unwrap_or_else(|| {
                Error::ValueError("Golden-section search failed to converge".into())
            })
        })?;

    let final_jd: Time<TDB> = (base + best_off).into();
    let sa = state_at_time(ref_a, final_jd, &spk, spk_id_a, center, include_extended)?;
    let sb = state_at_time(ref_b, final_jd, &spk, spk_id_b, center, include_extended)?;
    let final_dist = (Vector3::from(sa.pos) - Vector3::from(sb.pos)).norm();

    Ok((final_jd, final_dist))
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;

    use super::*;
    use kete_core::forces::{AccelVecMeta, vec_accel};

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
