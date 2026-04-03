//! FOV SPICE-dependent visibility checks.
//!
//! These functions provide SPK-dependent visibility checking for FOV types.
//! The `FovLike` trait and FOV types remain in `kete_core`.

use kete_core::fov::{Contains, FovLike, check_linear, check_two_body};
use kete_core::frames::Equatorial;
use kete_core::kepler::light_time_correct;
use kete_core::prelude::{KeteResult, SimultaneousStates, State};

use crate::propagation::propagate_n_body_spk;
use crate::spice::LOADED_SPK;

use rayon::prelude::*;

/// Assuming the object undergoes n-body motion, check to see if it is within the
/// field of view.
///
/// # Errors
/// Errors can occur for numerous reasons, typically from numerical integration failing.
pub fn check_n_body<F: FovLike>(
    fov: &F,
    state: &State<Equatorial>,
    include_asteroids: bool,
) -> KeteResult<(usize, Contains, State<Equatorial>)> {
    let obs = fov.observer();

    let exact_state = propagate_n_body_spk(state.clone(), obs.epoch, include_asteroids, None)?;
    let mut sun_state = exact_state.clone();
    if sun_state.center_id != 10 {
        let spk = LOADED_SPK.try_read()?;
        spk.try_change_center(&mut sun_state, 10)?;
    }

    let dist = (exact_state.pos - obs.pos).norm();
    let final_state = light_time_correct(&sun_state, dist)?;
    let rel_pos = final_state.pos - obs.pos;

    let (idx, contains) = fov.contains(&rel_pos);

    Ok((idx, contains, final_state))
}

/// Given an object ID, attempt to load the object from the SPKs and check visibility.
/// This will fail silently if the object is not found.
///
/// # Panics
///
/// - Panics if the SPK read lock is poisoned.
/// - Panics if the Fov cannot be converted into an FOV enum.
///
pub fn check_spks<F: FovLike>(fov: &F, obj_ids: &[i32]) -> Vec<Option<SimultaneousStates>> {
    let obs = fov.observer();
    let spk = &LOADED_SPK.try_read().unwrap();

    let mut visible: Vec<Vec<State<_>>> = vec![Vec::new(); fov.n_patches()];

    let states: Vec<_> = obj_ids
        .into_par_iter()
        .filter_map(|&obj_id| {
            let state = spk.try_get_state_with_center(obj_id, obs.epoch, 10).ok()?;
            match check_two_body(fov, &state) {
                Ok((idx, Contains::Inside, state)) => Some((idx, state)),
                _ => None,
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

/// Given a list of states, check to see if the objects are visible at the desired time.
///
/// Only the final observed states are returned, if the object was not seen it will not
/// be returned.
///
/// This does progressively more exact checks. For objects close in time (< `dt_limit`),
/// linear then two-body checks are used. For objects further in time, two-body then
/// n-body propagation is used.
///
/// # Panics
///
/// - Panics if the SPK read lock is poisoned.
/// - Panics if the Fov cannot be converted into an FOV enum.
pub fn check_visible<F: FovLike>(
    fov: &F,
    states: &[State<Equatorial>],
    dt_limit: f64,
    include_asteroids: bool,
) -> Vec<Option<SimultaneousStates>> {
    let obs_state = fov.observer();

    let final_states: Vec<(usize, State<Equatorial>)> = states
        .iter()
        .filter_map(|state: &State<_>| {
            let max_dist = (state.vel - obs_state.vel).norm() * dt_limit * 2.0;
            let mut sun_state = state.clone();
            if sun_state.center_id != 10 {
                let spk = LOADED_SPK.try_read().ok()?;
                spk.try_change_center(&mut sun_state, 10).ok()?;
            }

            if (state.epoch - obs_state.epoch).elapsed.abs() < dt_limit {
                let (_, contains, _) = check_linear(fov, state);
                if let Contains::Outside(dist) = contains
                    && dist > max_dist
                {
                    return None;
                }
                let (idx, contains, state) = check_two_body(fov, &sun_state).ok()?;
                match contains {
                    Contains::Inside => Some((idx, state)),
                    Contains::Outside(_) => None,
                }
            } else {
                let (_, contains, _) = check_two_body(fov, &sun_state).ok()?;
                if let Contains::Outside(dist) = contains
                    && dist > max_dist
                {
                    return None;
                }
                let (idx, contains, state) = check_n_body(fov, state, include_asteroids).ok()?;
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

    detector_states
        .into_iter()
        .enumerate()
        .map(|(idx, states)| {
            SimultaneousStates::new_exact(states, Some(fov.get_child(idx).into_fov())).ok()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use kete_core::constants::{self, GMS_SQRT};
    use kete_core::desigs::Desig;
    use kete_core::fov::{GenericRectangle, OmniDirectional};
    use kete_core::state::State;

    use crate::propagation::propagate_n_body_spk;

    #[test]
    fn test_check_rectangle_visible() {
        let circular = State::new(
            Desig::Empty,
            2451545.0.into(),
            [0.0, 1., 0.0].into(),
            [-GMS_SQRT, 0.0, 0.0].into(),
            10,
        );
        let circular_back = State::new(
            Desig::Empty,
            2451545.0.into(),
            [1.0, 0.0, 0.0].into(),
            [0.0, GMS_SQRT, 0.0].into(),
            10,
        );

        for offset in [-10.0_f64, -5.0, 0.0, 5.0, 10.0] {
            let off_state = propagate_n_body_spk(
                circular_back.clone(),
                circular_back.epoch - offset,
                false,
                None,
            )
            .unwrap();

            let vec = circular_back.pos - circular.pos;

            let fov = GenericRectangle::new(vec, 0.0001, 0.01, 0.01, circular.clone());
            assert!(check_two_body(&fov, &off_state).is_ok());
            assert!(check_n_body(&fov, &off_state, false).is_ok());

            assert!(
                check_visible(&fov, &[off_state], 6.0, false)
                    .first()
                    .unwrap()
                    .is_some()
            );
        }
    }

    /// Test the light delay computations for the different checks
    #[test]
    fn test_check_omni_visible() {
        // Build an observer, and check the observability of ceres with different
        // offsets from the observer time.
        // this will exercise the position, velocity, and time offsets due to light delay.
        let spk = &LOADED_SPK.read().unwrap();
        let observer = State::new(
            Desig::Empty,
            2451545.0.into(),
            [0.0, 1., 0.0].into(),
            [-GMS_SQRT, 0.0, 0.0].into(),
            10,
        );

        for offset in [-10.0, -5.0, 0.0, 5.0, 10.0] {
            let ceres = spk
                .try_get_state_with_center(20000001, observer.epoch + offset, 10)
                .unwrap();

            let fov = OmniDirectional::new(observer.clone());

            // Check two body approximation calculation
            let two_body = check_two_body(&fov, &ceres);
            assert!(two_body.is_ok());
            let (_, _, two_body) = two_body.unwrap();
            let dist = (two_body.pos - observer.pos).norm();
            assert!(
                (observer.epoch.jd - two_body.epoch.jd - dist * constants::C_AU_PER_DAY_INV).abs()
                    < 1e-6
            );
            let ceres_exact = spk
                .try_get_state_with_center(20000001, two_body.epoch, 10)
                .unwrap();
            // check that we are within about 150km - not bad for 2 body
            assert!((two_body.pos - ceres_exact.pos).norm() < 1e-6);

            // Check n body approximation calculation
            let n_body = check_n_body(&fov, &ceres, false);
            assert!(n_body.is_ok());
            let (_, _, n_body) = n_body.unwrap();
            assert!(
                (observer.epoch.jd - n_body.epoch.jd - dist * constants::C_AU_PER_DAY_INV).abs()
                    < 1e-6
            );
            let ceres_exact = spk
                .try_get_state_with_center(20000001, n_body.epoch, 10)
                .unwrap();
            // check that we are within about 150m
            assert!((n_body.pos - ceres_exact.pos).norm() < 1e-9);

            // Check spk queries
            let spk_check = &check_spks(&fov, &[20000001])[0];
            assert!(spk_check.is_some());
            let spk_check = &spk_check.as_ref().unwrap().states[0];
            assert!(
                (observer.epoch.jd - spk_check.epoch.jd - dist * constants::C_AU_PER_DAY_INV).abs()
                    < 1e-6
            );
            let ceres_exact = spk
                .try_get_state_with_center(20000001, spk_check.epoch, 10)
                .unwrap();
            // check that we are within about 150 micron
            assert!((spk_check.pos - ceres_exact.pos).norm() < 1e-12);

            assert!(
                check_visible(&fov, &[ceres], 6.0, false)
                    .first()
                    .unwrap()
                    .is_some()
            );
        }
    }
}
