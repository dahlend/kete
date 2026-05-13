//! FOV SPICE-dependent visibility checks.
//!
//! These functions provide SPK-dependent visibility checking for FOV types.
//! The `FovLike` trait and FOV types remain in `kete_core`.

use kete_core::constants::C_AU_PER_DAY_INV;
use kete_core::fov::{Contains, FovLike, check_linear, check_two_body};
use kete_core::frames::{Equatorial, SSB, SunCenter};
use kete_core::kepler::light_time_correct;
use kete_core::prelude::{KeteResult, SimultaneousStates, State};
use kete_core::state::StateLike;

use crate::propagation::SpkNBody;
use crate::spk::LOADED_SPK;

use rayon::prelude::*;

/// Assuming the object undergoes n-body motion, check to see if it is within the
/// field of view.
///
/// # Errors
/// Errors can occur for numerous reasons, typically from numerical integration failing.
pub fn check_n_body<F: FovLike>(
    fov: &F,
    state: State<Equatorial, SSB>,
    include_extended: bool,
) -> KeteResult<(usize, Contains, State<Equatorial>)> {
    let obs = fov.observer();

    let spk = LOADED_SPK.try_read()?;
    let exact_state = state.propagate_with(&SpkNBody::new(&spk, include_extended), obs.epoch)?;
    let sun_state = spk.try_to_sun(exact_state)?;

    let final_state = light_time_correct(&sun_state, &obs.pos)?;
    let rel_pos = final_state.pos - obs.pos;

    let (idx, contains) = fov.contains(&rel_pos);

    Ok((idx, contains, final_state.into()))
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
            // Load the state at the observation epoch for an initial position estimate.
            let state = spk.try_get_state_with_center(obj_id, obs.epoch, 10).ok()?;
            let mut corrected: State<Equatorial, SunCenter> = state.try_into().ok()?;
            // Light-time correct by querying the SPK at the emission epoch directly.
            // This handles all objects (including the Sun at r0=0) without two-body
            // propagation, and is more accurate for objects with SPK coverage.
            let mut tau = 0.0_f64;
            for _ in 0..3 {
                let new_tau = (corrected.pos - obs.pos).norm() * C_AU_PER_DAY_INV;
                if (new_tau - tau).abs() < 1e-12 {
                    break;
                }
                tau = new_tau;
                let state = spk
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

            if (state.epoch - obs_state.epoch).elapsed.abs() < dt_limit {
                let (_, contains, _) = check_linear(fov, state);
                if let Contains::Outside(dist) = contains
                    && dist > max_dist
                {
                    return None;
                }
                let spk = LOADED_SPK.try_read().ok()?;
                let sun_state = spk.try_to_sun(state.clone()).ok()?;
                let (idx, contains, state) = check_two_body(fov, &sun_state).ok()?;
                match contains {
                    Contains::Inside => Some((idx, state.into())),
                    Contains::Outside(_) => None,
                }
            } else {
                let spk = LOADED_SPK.try_read().ok()?;
                let sun_state = spk.try_to_sun(state.clone()).ok()?;
                let (_, contains, _) = check_two_body(fov, &sun_state).ok()?;
                if let Contains::Outside(dist) = contains
                    && dist > max_dist
                {
                    return None;
                }
                let ssb_state = spk.try_to_ssb(state.clone()).ok()?;
                let (idx, contains, state) =
                    check_n_body(fov, ssb_state, include_asteroids).ok()?;
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
    use kete_core::constants::GMS_SQRT;
    use kete_core::desigs::Desig;
    use kete_core::fov::{GenericRectangle, OmniDirectional};
    use kete_core::state::State;

    use crate::propagation::SpkNBody;
    use kete_core::state::StateLike;

    #[test]
    fn test_check_rectangle_visible() {
        crate::test_data::ensure_test_spk();
        let circular = State::new(
            Desig::Empty,
            2451545.0,
            [0.0, 1., 0.0],
            [-GMS_SQRT, 0.0, 0.0],
            10,
        );
        let circular_back = State::<Equatorial>::new(
            Desig::Empty,
            2451545.0,
            [1.0, 0.0, 0.0],
            [0.0, GMS_SQRT, 0.0],
            10,
        );

        let circular_back_ssb = {
            let spk = LOADED_SPK.try_read().unwrap();
            spk.try_to_ssb(circular_back.clone()).unwrap()
        };

        for offset in [-10.0_f64, -5.0, 0.0, 5.0, 10.0] {
            let spk = LOADED_SPK.try_read().unwrap();
            let force = SpkNBody::new(&spk, false);
            let off_state = circular_back_ssb
                .clone()
                .propagate_with(&force, circular_back_ssb.epoch - offset)
                .unwrap();
            drop(spk);

            let vec = circular_back.pos - circular.pos;

            let fov = GenericRectangle::new(vec, 0.0001, 0.01, 0.01, circular.clone());
            let off_sun = {
                let spk = LOADED_SPK.try_read().unwrap();
                spk.try_to_sun(off_state.clone()).unwrap()
            };
            assert!(check_two_body(&fov, &off_sun).is_ok());
            assert!(check_n_body(&fov, off_state.clone(), false).is_ok());

            let off_dyn: State<Equatorial> = off_state.into();
            assert!(
                check_visible(&fov, &[off_dyn], 6.0, false)
                    .first()
                    .unwrap()
                    .is_some()
            );
        }
    }

    /// Test the light delay computations for the different checks
    #[test]
    fn test_check_omni_visible() {
        crate::test_data::ensure_test_spk();
        // Build an observer, and check the observability of an asteroid with different
        // offsets from the observer time.
        // this will exercise the position, velocity, and time offsets due to light delay.
        let spk = &LOADED_SPK.read().unwrap();
        let observer = State::new(
            Desig::Empty,
            2451545.0,
            [0.0, 1., 0.0],
            [-GMS_SQRT, 0.0, 0.0],
            10,
        );

        for offset in [-10.0, -5.0, 0.0, 5.0, 10.0] {
            let asteroid = spk
                .try_get_state_with_center(20000042, observer.epoch + offset, 10)
                .unwrap();

            let fov = OmniDirectional::new(observer.clone());

            // Check two body approximation calculation
            let asteroid_sun: State<_, SunCenter> = asteroid.clone().try_into().unwrap();
            let two_body = check_two_body(&fov, &asteroid_sun);
            assert!(two_body.is_ok());
            let (_, _, two_body) = two_body.unwrap();
            let dist = (two_body.pos - observer.pos).norm();
            assert!((observer.epoch.jd - two_body.epoch.jd - dist * C_AU_PER_DAY_INV).abs() < 1e-6);
            let exact = spk
                .try_get_state_with_center(20000042, two_body.epoch, 10)
                .unwrap();
            // check that we are within about 150km - not bad for 2 body
            assert!((two_body.pos - exact.pos).norm() < 1e-6);

            // Check n body approximation calculation
            let asteroid_ssb = spk.try_to_ssb(asteroid.clone()).unwrap();
            let n_body = check_n_body(&fov, asteroid_ssb, false);
            assert!(n_body.is_ok());
            let (_, _, n_body) = n_body.unwrap();
            assert!((observer.epoch.jd - n_body.epoch.jd - dist * C_AU_PER_DAY_INV).abs() < 1e-6);
            let exact = spk
                .try_get_state_with_center(20000042, n_body.epoch, 10)
                .unwrap();
            // check that we are within about 150m
            assert!((n_body.pos - exact.pos).norm() < 1e-9);

            // Check spk queries
            let spk_check = &check_spks(&fov, &[20000042])[0];
            assert!(spk_check.is_some());
            let spk_check = &spk_check.as_ref().unwrap().states[0];
            assert!(
                (observer.epoch.jd - spk_check.epoch.jd - dist * C_AU_PER_DAY_INV).abs() < 1e-6
            );
            let exact = spk
                .try_get_state_with_center(20000042, spk_check.epoch, 10)
                .unwrap();
            // check that we are within about 150 micron
            assert!((spk_check.pos - exact.pos).norm() < 1e-12);

            assert!(
                check_visible(&fov, &[asteroid], 6.0, false)
                    .first()
                    .unwrap()
                    .is_some()
            );
        }

        // Sun Fov check was previously failing due to it being co-located at itself
        let sun_fov = OmniDirectional::new(observer.clone());
        let sun_check = &check_spks(&sun_fov, &[10])[0];
        assert!(sun_check.is_some());
        let sun_state = &sun_check.as_ref().unwrap().states[0];
        // The Sun is always at the solar center.
        assert!(sun_state.pos.norm() < 1e-12);
    }
}
