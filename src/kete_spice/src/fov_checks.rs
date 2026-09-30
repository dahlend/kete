//! FOV SPICE-dependent visibility checks.
//!
//! These functions provide SPK-dependent visibility checking for FOV types.
//! The `FovLike` trait and FOV types remain in `kete_core`.

use kete_core::constants::C_AU_PER_DAY_INV;
use kete_core::desigs::Desig;
use kete_core::errors::Error;
use kete_core::forces::NonGravMask;
use kete_core::fov::{FovLike, check_linear, check_two_body};
use kete_core::frames::{Equatorial, SSB, SunCenter};
use kete_core::geometry::Contains;
use kete_core::kepler::light_time_correct;
use kete_core::prelude::{KeteResult, SimultaneousStates, State};

use crate::propagation::SpkNBody;
use crate::spk::LOADED_SPK;

use rayon::prelude::*;

/// Assuming the object undergoes n-body motion, check to see if it is within the
/// field of view.
///
/// If a non-gravitational model is provided, it is added to the gravitational
/// force model during the propagation.
///
/// # Errors
/// Errors can occur for numerous reasons, typically from numerical integration failing.
pub fn check_n_body<F: FovLike>(
    fov: &F,
    state: State<Equatorial, SSB>,
    non_grav: Option<&NonGravMask>,
    include_extended: bool,
) -> KeteResult<(usize, Contains, State<Equatorial>)> {
    let obs = fov.observer();

    let spk = LOADED_SPK.try_read()?;
    let force = SpkNBody::with_non_grav(&spk, include_extended, non_grav.cloned());
    let exact_state = state.propagate_with(&force, obs.epoch)?;
    let sun_state = spk.try_to_sun(exact_state)?;

    let final_state = light_time_correct(&sun_state, &obs.pos)?;
    let rel_pos = final_state.pos - obs.pos;

    let (idx, contains) = fov.contains(&rel_pos);

    Ok((idx, contains, final_state.into()))
}

/// Load objects from the SPKs by NAIF ID and check which are in the FOV.
///
/// The position of each object comes from the loaded SPKs at the light-time
/// corrected epoch. The result has one entry per patch of `fov`. An entry holds
/// the Sun-centered states seen in that patch, or `None` if the patch has no
/// object. An object is reported as not visible if an SPK query fails, for
/// example outside the SPK coverage.
///
/// # Errors
/// Returns [`Error::LockFailed`] if the SPK read lock cannot be taken.
///
/// # Panics
/// Panics if `fov` is inconsistent: `contains` returns a patch index of
/// `n_patches` or more, or `get_child` panics for an index below `n_patches`.
pub fn check_spks<F: FovLike>(
    fov: &F,
    obj_ids: &[i32],
) -> KeteResult<Vec<Option<SimultaneousStates>>> {
    let obs = fov.observer();
    let spk = &LOADED_SPK.try_read()?;

    let mut visible: Vec<Vec<State<_>>> = vec![Vec::new(); fov.n_patches()];

    let states: Vec<_> = obj_ids
        .into_par_iter()
        .with_min_len(100)
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

    Ok(visible
        .into_iter()
        .enumerate()
        .map(|(idx, states_patch)| {
            SimultaneousStates::new_exact(states_patch, Some(fov.get_child(idx).into_fov())).ok()
        })
        .collect())
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
/// A state is reported as not visible if a center change or a propagation
/// fails, for example outside the loaded SPK coverage.
///
/// # Errors
/// Returns [`Error::ValueError`] if `non_gravs` is not empty and does not have
/// one entry per state. Returns [`Error::LockFailed`] if the SPK read lock
/// cannot be taken.
///
/// # Panics
/// Panics if `fov` is inconsistent: `contains` returns a patch index of
/// `n_patches` or more, or `get_child` panics for an index below `n_patches`.
pub fn check_visible<F: FovLike>(
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
    let spk = LOADED_SPK.try_read()?;

    // The linear check compares positions directly, so each state moves to the
    // center of the observer. States usually share a center and an epoch. Thus
    // the offset between the two centers is kept for reuse by the next state.
    let mut center_offset: Option<(i32, f64, State<Equatorial>)> = None;

    let final_states: Vec<(usize, State<Equatorial>)> = states
        .iter()
        .enumerate()
        .filter_map(|(idx, state)| {
            let non_grav = non_gravs.get(idx).and_then(Option::as_ref);

            if non_grav.is_none() && (state.epoch - obs_state.epoch).elapsed.abs() < dt_limit {
                let offset = match &center_offset {
                    Some((center, jd, offset))
                        if *center == state.center_id() && *jd == state.epoch.jd =>
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
                        spk.try_change_center(&mut offset, obs_state.center_id())
                            .ok()?;
                        &center_offset
                            .insert((state.center_id(), state.epoch.jd, offset))
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
                let sun_state = spk.try_to_sun(state.clone()).ok()?;
                let (idx, contains, state) = check_two_body(fov, &sun_state).ok()?;
                match contains {
                    Contains::Inside => Some((idx, state.into())),
                    Contains::Outside(_) => None,
                }
            } else {
                let sun_state = spk.try_to_sun(state.clone()).ok()?;
                let max_dist = (sun_state.vel - obs_state.vel).norm() * dt_limit * 2.0;
                let (_, contains, _) = check_two_body(fov, &sun_state).ok()?;
                if let Contains::Outside(dist) = contains
                    && dist > max_dist
                {
                    return None;
                }
                let ssb_state = spk.try_to_ssb(state.clone()).ok()?;
                let (idx, contains, state) =
                    check_n_body(fov, ssb_state, non_grav, include_asteroids).ok()?;
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

#[cfg(test)]
mod tests {
    use super::*;
    use kete_core::constants::GMS_SQRT;
    use kete_core::desigs::Desig;
    use kete_core::forces::{DustNonGrav, NonGravKind, ParameterMask};
    use kete_core::fov::{GenericRectangle, OmniDirectional};
    use kete_core::state::State;

    use crate::propagation::SpkNBody;

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
            assert!(check_n_body(&fov, off_state.clone(), None, false).is_ok());

            let off_dyn: State<Equatorial> = off_state.into();
            assert!(
                check_visible(&fov, &[off_dyn], &[], 6.0, false)
                    .unwrap()
                    .first()
                    .unwrap()
                    .is_some()
            );
        }
    }

    /// A slow object near the observer passes the linear pre-filter when its
    /// state and the observer have different centers.
    #[test]
    fn linear_prefilter_matches_the_observer_center() {
        crate::test_data::ensure_test_spk();
        let observer = State::<Equatorial>::new(
            Desig::Empty,
            2451545.0,
            [0.0, 1., 0.0],
            [-GMS_SQRT, 0.0, 0.0],
            10,
        );
        // At rest relative to the observer, so the pre-filter allows no slack.
        let object = State::<Equatorial>::new(
            Desig::Empty,
            2451545.0,
            [1e-3, 1., 0.0],
            [-GMS_SQRT, 0.0, 0.0],
            10,
        );
        let object_ssb: State<Equatorial> = {
            let spk = LOADED_SPK.try_read().unwrap();
            spk.try_to_ssb(object).unwrap().into()
        };
        let fov = GenericRectangle::new([1.0, 0.0, 0.0].into(), 0.0, 0.01, 0.01, observer);
        let seen = check_visible(&fov, &[object_ssb], &[], 3.0, false).unwrap();
        assert!(seen[0].is_some());
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
            let n_body = check_n_body(&fov, asteroid_ssb, None, false);
            assert!(n_body.is_ok());
            let (_, _, n_body) = n_body.unwrap();
            assert!((observer.epoch.jd - n_body.epoch.jd - dist * C_AU_PER_DAY_INV).abs() < 1e-6);
            let exact = spk
                .try_get_state_with_center(20000042, n_body.epoch, 10)
                .unwrap();
            // check that we are within about 150m
            assert!((n_body.pos - exact.pos).norm() < 1e-9);

            // Check spk queries
            let spk_check = &check_spks(&fov, &[20000042]).unwrap()[0];
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
                check_visible(&fov, &[asteroid], &[], 6.0, false)
                    .unwrap()
                    .first()
                    .unwrap()
                    .is_some()
            );
        }

        // The Sun is co-located with itself in a Sun-centered FOV check
        let sun_fov = OmniDirectional::new(observer.clone());
        let sun_check = &check_spks(&sun_fov, &[10]).unwrap()[0];
        assert!(sun_check.is_some());
        let sun_state = &sun_check.as_ref().unwrap().states[0];
        // The Sun is always at the solar center.
        assert!(sun_state.pos.norm() < 1e-12);
    }

    /// A non-gravitational model must change the observed state, including when the
    /// state epoch is close enough to the observer that the two body check would
    /// otherwise be used.
    #[test]
    fn test_check_visible_non_grav() {
        crate::test_data::ensure_test_spk();
        let observer = State::new(
            Desig::Empty,
            2451545.0,
            [0.0, 1., 0.0],
            [-GMS_SQRT, 0.0, 0.0],
            10,
        );
        let fov = OmniDirectional::new(observer.clone());

        // One day before the observation, well within the dt_limit used below.
        let asteroid: State<Equatorial> = {
            let spk = LOADED_SPK.read().unwrap();
            spk.try_get_state_with_center(20000042, observer.epoch - 1.0, 10)
                .unwrap()
        };

        // beta = 0.5 removes half of the solar gravity, which over a day is a
        // deflection of order 1e-5 au, far above the two body vs n-body difference.
        let dust = ParameterMask::all_fixed(NonGravKind::Dust(DustNonGrav), vec![0.5]).unwrap();

        let states = [asteroid];
        let grav_only = check_visible(&fov, &states, &[], 3.0, false).unwrap();
        let with_dust = check_visible(&fov, &states, &[Some(dust)], 3.0, false).unwrap();

        let grav_pos = grav_only[0].as_ref().unwrap().states[0].pos;
        let dust_pos = with_dust[0].as_ref().unwrap().states[0].pos;
        assert!((grav_pos - dust_pos).norm() > 1e-6);

        // A non-empty non_gravs which does not cover every state is rejected.
        assert!(check_visible(&fov, &states, &[None, None], 3.0, false).is_err());
    }
}
