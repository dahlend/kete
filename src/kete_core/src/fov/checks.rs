//! Visibility checks that need body states: N-body propagation to the observer epoch,
//! or objects looked up by NAIF id, from an [`Ephemeris`].

use super::{FovLike, check_linear, check_two_body};
use crate::constants::C_AU_PER_DAY_INV;
use crate::desigs::Desig;
use crate::ephemeris::Ephemeris;
use crate::errors::{Error, KeteResult};
use crate::forces::NonGravMask;
use crate::frames::{Equatorial, SSB, SunCenter};
use crate::geometry::Contains;
use crate::kepler::light_time_correct;
use crate::propagation::NBody;
use crate::state::{SimultaneousStates, State};
use crate::time::{TDB, Time};

use rayon::prelude::*;

/// Assuming the object undergoes n-body motion, check to see if it is within the
/// field of view.
///
/// If a non-gravitational model is provided, it is added to the gravitational
/// force model during the propagation. Body states come from `ephem`.
///
/// # Errors
/// Errors can occur for numerous reasons, typically from numerical integration failing.
pub fn check_n_body<E: Ephemeris, F: FovLike>(
    ephem: &E,
    fov: &F,
    state: State<Equatorial, SSB>,
    non_grav: Option<&NonGravMask>,
    include_extended: bool,
) -> KeteResult<(usize, Contains, State<Equatorial>)> {
    let obs = fov.observer();

    let force = NBody::with_non_grav(ephem, include_extended, non_grav.cloned());
    let exact_state = state.propagate_with(&force, obs.epoch)?;
    let sun_state = ephem.try_to_sun(exact_state.into())?;

    let final_state = light_time_correct(&sun_state, &obs.pos)?;
    let rel_pos = final_state.pos - obs.pos;

    let (idx, contains) = fov.contains(&rel_pos);

    Ok((idx, contains, final_state.into()))
}

/// Look up objects by NAIF ID in `ephem` and check which are in the FOV.
///
/// The position of each object comes from `ephem` at the light-time corrected
/// epoch. The result has one entry per patch of `fov`. An entry holds the
/// Sun-centered states seen in that patch, or `None` if the patch has no
/// object. An object is reported as not visible if its lookup fails, for
/// example outside the ephemeris coverage.
///
/// # Panics
/// Panics if `fov` is inconsistent: `contains` returns a patch index of
/// `n_patches` or more, or `get_child` panics for an index below `n_patches`.
pub fn check_ephemeris<E: Ephemeris, F: FovLike>(
    ephem: &E,
    fov: &F,
    obj_ids: &[i32],
) -> Vec<Option<SimultaneousStates>> {
    let obs = fov.observer();

    let mut visible: Vec<Vec<State<_>>> = vec![Vec::new(); fov.n_patches()];

    let states: Vec<_> = obj_ids
        .into_par_iter()
        .with_min_len(100)
        .filter_map(|&obj_id| {
            // Load the state at the observation epoch for an initial position estimate.
            let state = ephem
                .try_get_state_with_center(obj_id, obs.epoch, 10)
                .ok()?;
            let mut corrected: State<Equatorial, SunCenter> = state.try_into().ok()?;
            // Light-time correct by querying the ephemeris at the emission epoch
            // directly. This handles all objects (including the Sun at r0=0) without
            // two-body propagation.
            let mut tau = 0.0_f64;
            for _ in 0..3 {
                let new_tau = (corrected.pos - obs.pos).norm() * C_AU_PER_DAY_INV;
                if (new_tau - tau).abs() < 1e-12 {
                    break;
                }
                tau = new_tau;
                let state = ephem
                    .try_get_state_with_center(obj_id, obs.epoch - tau, 10)
                    .ok()?;
                corrected = state.try_into().ok()?;
            }
            let rel_pos = corrected.pos - obs.pos;
            let (idx, contains) = fov.contains(&rel_pos);
            match contains {
                Contains::Inside => Some((idx, corrected.into())),
                Contains::Outside(_) => None,
            }
        })
        .collect();

    for (patch_idx, state) in states {
        visible[patch_idx].push(state);
    }

    visible
        .into_iter()
        .enumerate()
        .map(|(idx, states_patch)| {
            SimultaneousStates::new_exact(states_patch, Some(fov.get_child(idx).into_fov())).ok()
        })
        .collect()
}

/// Check which states are in the FOV at the observer epoch.
///
/// The result has one entry per patch of `fov`. An entry holds the Sun-centered
/// states seen in that patch, at the time light left the object. An entry is
/// `None` if the patch has no object.
///
/// The checks become progressively more exact. A state without a
/// non-gravitational model, and less than `dt_limit` days from the observer
/// epoch, gets a linear check and then a two-body check. Every other state gets
/// a two-body check and then an n-body propagation. `include_asteroids` adds
/// the registered asteroid masses to the n-body force model. A pre-filter
/// rejects a state only if the state is outside the FOV by more than twice the
/// distance it moves relative to the observer in `dt_limit`.
///
/// `non_gravs` is either empty or holds one entry per state. An empty
/// `non_gravs` means that no state has a non-gravitational model. The linear
/// and two-body checks do not include non-gravitational accelerations. Thus a
/// state with a model always takes the n-body path, and the two-body check is
/// only a coarse pre-filter. This pre-filter assumes that the
/// non-gravitational deviation between the state epoch and the observer epoch
/// is small compared to the pre-filter distance.
///
/// Body states come from `ephem`. A state is reported as not visible if a
/// center change or a propagation fails, for example outside the ephemeris
/// coverage.
///
/// # Errors
/// Returns [`Error::ValueError`] if `non_gravs` is not empty and does not have
/// one entry per state.
///
/// # Panics
/// Panics if `fov` is inconsistent: `contains` returns a patch index of
/// `n_patches` or more, or `get_child` panics for an index below `n_patches`.
pub fn check_visible<E: Ephemeris, F: FovLike>(
    ephem: &E,
    fov: &F,
    states: &[State<Equatorial>],
    non_gravs: &[Option<NonGravMask>],
    dt_limit: f64,
    include_asteroids: bool,
) -> KeteResult<Vec<Option<SimultaneousStates>>> {
    if !(non_gravs.is_empty() || non_gravs.len() == states.len()) {
        Err(Error::ValueError(format!(
            "non_gravs must be empty or have one entry per state, found {} entries for \
             {} states.",
            non_gravs.len(),
            states.len()
        )))?;
    }
    let obs_state = fov.observer();

    // The linear check compares positions directly, so each state moves to the
    // center of the observer. States usually share a center and an epoch. Thus
    // the offset between the two centers is kept for reuse by the next state.
    let mut center_offset: Option<(i32, Time<TDB>, State<Equatorial>)> = None;

    let final_states: Vec<(usize, State<Equatorial>)> = states
        .iter()
        .enumerate()
        .filter_map(|(idx, state)| {
            let non_grav = non_gravs.get(idx).and_then(Option::as_ref);

            if non_grav.is_none() && (state.epoch - obs_state.epoch).elapsed.abs() < dt_limit {
                let offset = match &center_offset {
                    Some((center, epoch, offset))
                        if *center == state.center_id() && *epoch == state.epoch =>
                    {
                        offset
                    }
                    _ => {
                        let mut offset = State::<Equatorial>::new(
                            Desig::Empty,
                            state.epoch,
                            [0.0; 3],
                            [0.0; 3],
                            state.center_id(),
                        );
                        ephem
                            .try_change_center(&mut offset, obs_state.center_id())
                            .ok()?;
                        &center_offset
                            .insert((state.center_id(), state.epoch, offset))
                            .2
                    }
                };
                let relative = State::<Equatorial>::new(
                    state.desig.clone(),
                    state.epoch,
                    state.pos + offset.pos,
                    state.vel + offset.vel,
                    obs_state.center_id(),
                );
                let max_dist = (relative.vel - obs_state.vel).norm() * dt_limit * 2.0;
                let (_, contains, _) = check_linear(fov, &relative);
                if let Contains::Outside(dist) = contains
                    && dist > max_dist
                {
                    return None;
                }
                let sun_state = ephem.try_to_sun(state.clone()).ok()?;
                let (idx, contains, state) = check_two_body(fov, &sun_state).ok()?;
                match contains {
                    Contains::Inside => Some((idx, state.into())),
                    Contains::Outside(_) => None,
                }
            } else {
                let sun_state = ephem.try_to_sun(state.clone()).ok()?;
                let max_dist = (sun_state.vel - obs_state.vel).norm() * dt_limit * 2.0;
                let (_, contains, _) = check_two_body(fov, &sun_state).ok()?;
                if let Contains::Outside(dist) = contains
                    && dist > max_dist
                {
                    return None;
                }
                let ssb_state = ephem.try_to_ssb(state.clone()).ok()?;
                let (idx, contains, state) =
                    check_n_body(ephem, fov, ssb_state, non_grav, include_asteroids).ok()?;
                match contains {
                    Contains::Inside => Some((idx, state)),
                    Contains::Outside(_) => None,
                }
            }
        })
        .collect();

    let mut detector_states = vec![Vec::<State<_>>::new(); fov.n_patches()];
    for (idx, state) in final_states {
        detector_states[idx].push(state);
    }

    Ok(detector_states
        .into_iter()
        .enumerate()
        .map(|(idx, states)| {
            SimultaneousStates::new_exact(states, Some(fov.get_child(idx).into_fov())).ok()
        })
        .collect())
}
