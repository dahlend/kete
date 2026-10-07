// SPDX-FileCopyrightText: 2026 Dar Dahlen
// SPDX-License-Identifier: BSD-3-Clause

//! Visibility checks that need body states: N-body propagation across the observer
//! epochs, or objects looked up by NAIF id, from an [`Ephemeris`].

use super::FovLike;
use crate::constants::C_AU_PER_DAY_INV;
use crate::ephemeris::Ephemeris;
use crate::errors::{Error, KeteResult};
use crate::forces::NonGravMask;
use crate::frames::{Equatorial, SunCenter, Vector};
use crate::geometry::{Contains, SkyPatch, SphericalCone};
use crate::integrators::RadauDense;
use crate::propagation::NBody;
use crate::state::{SimultaneousStates, State, propagate_state};
use crate::time::{TDB, Time};

use itertools::Itertools;
use nalgebra::Vector3;
use rayon::prelude::*;
use std::f64::consts::FRAC_PI_2;

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
            // Light-time correct by querying the ephemeris at the emission epoch.
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

/// Check which states are seen in which FOVs.
///
/// Each state is integrated once with N-body physics across the span of the FOV
/// epochs, keeping the integrator's dense output. Its position at any time in the
/// span then comes from that output without integrating again, so every check is
/// exact, including the light-time correction and any non-gravitational model.
/// `include_asteroids` adds the registered asteroid masses to the force model, and
/// body states come from `ephem`.
///
/// The FOVs are grouped into half days from the first FOV epoch, such as one night
/// of a ground based survey. As a pre-filter, each state is evaluated once for each
/// group, at the middle of it, and rejected for a FOV of the group if it is outside
/// the FOV by more than twice its speed times the time to the FOV epoch plus the
/// light time. The speed is the largest of those at the start, middle and end of
/// the group.
///
/// `non_gravs` is either empty or holds one entry per state. An empty
/// `non_gravs` means that no state has a non-gravitational model.
///
/// The result holds the FOV index, the patch index, and the Sun-centered states
/// seen in that patch at the time light left them, ordered by FOV and then by
/// patch. Patches with no state are left out. A state is not visible where its
/// integration fails, for example after an impact or outside the ephemeris
/// coverage.
///
/// # Errors
/// Returns [`Error::ValueError`] if `non_gravs` is not empty and does not have
/// one entry per state, and the error of `ephem` if the observer of a FOV cannot
/// be moved to the SSB.
///
/// # Panics
/// Panics if a FOV is inconsistent: `contains` returns a patch index of
/// `n_patches` or more, or `get_child` panics for an index below `n_patches`.
pub fn check_visible<E: Ephemeris, F: FovLike>(
    ephem: &E,
    fovs: &[F],
    states: &[State<Equatorial>],
    non_gravs: &[Option<NonGravMask>],
    include_asteroids: bool,
) -> KeteResult<Vec<(usize, usize, SimultaneousStates)>> {
    if !(non_gravs.is_empty() || non_gravs.len() == states.len()) {
        Err(Error::ValueError(format!(
            "non_gravs must be empty or have one entry per state, found {} entries for \
             {} states.",
            non_gravs.len(),
            states.len()
        )))?;
    }

    // The FOVs in epoch order. Sorting indices keeps the sort small.
    let mut order: Vec<usize> = (0..fovs.len()).collect();
    order.par_sort_unstable_by(|a, b| {
        let (a, b) = (fovs[*a].observer().epoch, fovs[*b].observer().epoch);
        a.jd().total_cmp(&b.jd())
    });
    // For each FOV in epoch order: its index, and the epoch and position of its
    // observer relative to the SSB, with a cone holding every patch of a FOV of
    // several patches.
    let observers: Vec<KeteResult<_>> = order
        .par_iter()
        .map(|&idx| {
            let fov = &fovs[idx];
            let observer = ephem.try_to_ssb(fov.observer().clone())?;
            // A cone of less than a hemisphere around the corners also holds the great
            // circle edges between them.
            let cone = if fov.n_patches() > 1 {
                fov.pointing()
                    .ok()
                    .zip(fov.corners().ok())
                    .and_then(|(center, corners)| {
                        let radius = corners
                            .iter()
                            .map(|corner| center.angle(corner))
                            .fold(0.0, f64::max);
                        (radius < FRAC_PI_2).then(|| SphericalCone::new(&center, radius))
                    })
            } else {
                None
            };
            Ok((idx, observer.epoch, observer.pos, cone))
        })
        .collect();
    drop(order);
    let observers = observers.into_iter().collect::<KeteResult<Vec<_>>>()?;
    let (Some(first), Some(last)) = (observers.first(), observers.last()) else {
        return Ok(Vec::new());
    };
    let (first_epoch, last_epoch) = (first.1, last.1);
    let max_obs_dist = observers
        .iter()
        .map(|(_, _, pos, _)| pos.norm())
        .fold(0.0, f64::max);

    // The FOVs in groups of `MAX_GROUP_SPAN` days from the first epoch, each with its
    // first, middle and last epochs. Each group is split into parallel tasks of
    // `FOV_GROUP` FOVs.
    let span_idx = |epoch: Time<TDB>| ((epoch - first_epoch).elapsed / MAX_GROUP_SPAN).floor();
    let groups: Vec<_> = observers
        .chunk_by(|a, b| span_idx(a.1) == span_idx(b.1))
        .flat_map(|group| {
            let (first, last) = (group[0].1, group[group.len() - 1].1);
            let center = first + (last - first).elapsed / 2.0;
            group
                .chunks(FOV_GROUP)
                .map(move |chunk| ([first, center, last], chunk))
        })
        .collect();

    // States are checked a block at a time: the block is integrated first, then each
    // FOV is checked against every state of the block while it is in cache.
    let mut hits: Vec<(usize, usize, usize, State<Equatorial>)> = Vec::new();
    for (block_idx, block) in states.chunks(STATE_BLOCK).enumerate() {
        let trajectories: Vec<Option<RadauDense>> = block
            .par_iter()
            .enumerate()
            .map(|(idx, state)| {
                let non_grav = non_gravs
                    .get(block_idx * STATE_BLOCK + idx)
                    .and_then(Option::as_ref);
                let force = NBody::with_non_grav(ephem, include_asteroids, non_grav.cloned());
                let at_first = ephem
                    .try_to_ssb(state.clone())
                    .ok()?
                    .propagate_with(&force, first_epoch)
                    .ok()?;
                // Light from the object reaches any observer within this many days, so
                // the trajectory starts early enough for the earliest emission time.
                let max_light_time = (at_first.pos.norm() + max_obs_dist) * C_AU_PER_DAY_INV;
                let start = at_first
                    .propagate_with(&force, first_epoch - max_light_time)
                    .ok()?;
                // A failure part way keeps the trajectory up to the failure.
                let mut trajectory = RadauDense::new();
                let _ = propagate_state(
                    &force,
                    start.pos.into(),
                    start.vel.into(),
                    &[],
                    start.epoch,
                    last_epoch,
                    Some(&mut trajectory),
                );
                Some(trajectory)
            })
            .collect();

        let block_hits: Vec<_> = groups
            .par_iter()
            .flat_map_iter(|([first, center, last], group)| {
                // Each state at the center of the group, with the largest of its speeds
                // at the first, center and last epochs of the group. A trajectory ends
                // early if its integration fails, so the times stay inside it to check
                // the FOVs before the failure.
                let at_center: Vec<_> = trajectories
                    .iter()
                    .map(|trajectory| {
                        let trajectory = trajectory.as_ref()?;
                        let end = trajectory.end()?;
                        let clamp = |time: Time<TDB>| if time > end { end } else { time };
                        let center = clamp(*center);
                        let (pos, vel) = trajectory.evaluate(center).ok()?;
                        let speed = [*first, *last]
                            .into_iter()
                            .filter_map(|time| trajectory.evaluate(clamp(time)).ok())
                            .map(|(_, vel)| Vector3::from_column_slice(&vel).norm())
                            .fold(Vector3::from_column_slice(&vel).norm(), f64::max);
                        let pos: Vector<Equatorial> = Vector3::from_column_slice(&pos).into();
                        Some((center, pos, speed))
                    })
                    .collect();

                let mut found = Vec::new();
                for (fov_idx, obs_epoch, obs_pos, cone) in *group {
                    let fov = &fovs[*fov_idx];
                    for (idx, (trajectory, at_center)) in
                        trajectories.iter().zip(&at_center).enumerate()
                    {
                        let (Some(trajectory), Some((center, pos, speed))) =
                            (trajectory, at_center)
                        else {
                            continue;
                        };

                        // Light reaching the observer left the object at the observer epoch
                        // less the light time. Between the center and then, the object
                        // moves at most its speed times the time between them; twice the
                        // largest sampled speed allows for a change of speed.
                        let obs_to_obj = *pos - *obs_pos;
                        let light_time = obs_to_obj.norm() * C_AU_PER_DAY_INV;
                        let max_dist =
                            2.0 * speed * ((*obs_epoch - *center).elapsed.abs() + light_time);
                        // The cone holds every patch, so its distance is also a lower
                        // bound, and rejecting on it skips checking each patch.
                        if let Some(Contains::Outside(dist)) =
                            cone.as_ref().map(|c| c.contains(&obs_to_obj))
                            && dist > max_dist
                        {
                            continue;
                        }
                        if let (_, Contains::Outside(dist)) = fov.contains(&obs_to_obj)
                            && dist > max_dist
                        {
                            continue;
                        }

                        let mut light_time = obs_to_obj.norm() * C_AU_PER_DAY_INV;
                        let mut emission = None;
                        for _ in 0..5 {
                            let Ok((pos, vel)) = trajectory.evaluate(*obs_epoch - light_time)
                            else {
                                break;
                            };
                            let pos: Vector<Equatorial> = Vector3::from_column_slice(&pos).into();
                            let vel: Vector<Equatorial> = Vector3::from_column_slice(&vel).into();
                            let new_light_time = (pos - *obs_pos).norm() * C_AU_PER_DAY_INV;
                            let converged = (new_light_time - light_time).abs() < 1e-12;
                            emission = Some((*obs_epoch - light_time, pos, vel));
                            if converged {
                                break;
                            }
                            light_time = new_light_time;
                        }
                        let Some((epoch, pos, vel)) = emission else {
                            continue;
                        };
                        let (patch_idx, Contains::Inside) = fov.contains(&(pos - *obs_pos)) else {
                            continue;
                        };
                        let state_idx = block_idx * STATE_BLOCK + idx;
                        let emitted = State::<Equatorial>::new(
                            states[state_idx].desig.clone(),
                            epoch,
                            pos,
                            vel,
                            0,
                        );
                        if let Ok(emitted) = ephem.try_to_sun(emitted) {
                            found.push((*fov_idx, patch_idx, state_idx, emitted.into()));
                        }
                    }
                }
                found
            })
            .collect();
        hits.extend(block_hits);
    }

    hits.sort_by_key(|(fov_idx, patch_idx, state_idx, _)| (*fov_idx, *patch_idx, *state_idx));
    let mut visible = Vec::new();
    for ((fov_idx, patch_idx), patch) in &hits
        .into_iter()
        .chunk_by(|(fov_idx, patch_idx, _, _)| (*fov_idx, *patch_idx))
    {
        let states = patch.map(|(_, _, _, state)| state).collect();
        let child = fovs[fov_idx].get_child(patch_idx).into_fov();
        visible.push((
            fov_idx,
            patch_idx,
            SimultaneousStates::new_exact(states, Some(child))?,
        ));
    }
    Ok(visible)
}

/// Number of states integrated together in [`check_visible`], which bounds the memory
/// held by their dense output.
const STATE_BLOCK: usize = 256;

/// Number of FOVs in one parallel task of [`check_visible`].
const FOV_GROUP: usize = 1024;

/// Length in days of the spans of FOV epochs that [`check_visible`] checks from one
/// state of each object.
const MAX_GROUP_SPAN: f64 = 0.5;
