use super::*;
use kete_core::errors::{Error, KeteResult};
use kete_core::forces::{FrozenNonGrav, Sum};
use kete_core::fov::{FOV, FovLike, check_statics};
use kete_core::frames::{Equatorial, SSB};
use kete_core::state::{State, StateLike};
use kete_core::time::{TDB, Time};
use kete_spice::fov_checks;
use kete_spice::propagation::{Recenter, SpkNBody};
use kete_spice::spk::LOADED_SPK;
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::{nongrav::PyNonGravModel, state::PySimultaneousStates, vector::VectorLike};

/// Given states and field of view, return only the objects which are visible to the
/// observer, adding a correction for optical light delay.
///
/// Objects are propagated using 2 body physics to the time of the FOV if time steps are
/// less than the specified `dt_limit`. Objects which have a non-gravitational model are
/// always propagated using n-body physics, since the 2 body approximation cannot
/// represent the non-gravitational acceleration.
///
/// parameters
/// ----------
/// obj_state: list[State]
///     States which do not already have a specified FOV.
/// fovs: list
///     A field of view from which to subselect objects which are visible.
/// dt_limit: float
///     Length of time in days where 2-body mechanics is a good approximation.
/// include_asteroids: bool
///     Include the additional registered gravitational masses during the computation.
/// non_gravs: list
///     A list of non-gravitational terms for each object. If provided, then every
///     object must have an associated :class:`~NonGravModel` or `None`.
#[pyfunction]
#[pyo3(name = "fov_state_check", signature = (obj_state, fovs, dt_limit=3.0,
    include_asteroids=false, non_gravs=None))]
pub fn fov_checks_py(
    py: Python<'_>,
    obj_state: PySimultaneousStates,
    mut fovs: Vec<AllowedFOV>,
    dt_limit: f64,
    include_asteroids: bool,
    non_gravs: Option<Vec<Option<PyNonGravModel>>>,
) -> PyResult<Vec<PySimultaneousStates>> {
    let pop = obj_state.0;

    let mut non_gravs: Vec<Option<FrozenNonGrav>> = match non_gravs {
        None => vec![None; pop.states.len()],
        Some(models) => {
            if models.len() != pop.states.len() {
                Err(Error::ValueError(
                    "non_gravs must be the same length as states.".into(),
                ))?;
            }
            models
                .into_iter()
                .map(|model| model.map(|model| model.to_frozen()))
                .collect()
        }
    };

    fovs.sort_by(|a, b| a.jd().jd.total_cmp(&b.jd().jd));

    // break the fovs into groups based upon the dt_limit
    let mut fov_chunks: Vec<Vec<FOV>> = Vec::new();
    let mut chunk: Vec<FOV> = Vec::new();
    for fov in fovs.into_iter() {
        let fov = fov.unwrap();
        if chunk.is_empty() {
            chunk.push(fov);
            continue;
        };
        let jd_start = chunk.first().unwrap().observer().epoch;

        // chunk is complete
        if (fov.observer().epoch - jd_start).elapsed.abs() >= 2.0 * dt_limit {
            fov_chunks.push(chunk);
            chunk = vec![fov];
        } else {
            chunk.push(fov);
        }
    }
    if !chunk.is_empty() {
        fov_chunks.push(chunk);
    }
    // Epoch that the states sit at, and the reference epoch of the last big step. Note
    // that `big_jd` is when the last big step was triggered, not the epoch the big step
    // states are at, which is `jd` at the moment that step is taken.
    let mut jd = pop.epoch().jd;
    let mut big_jd = jd;

    // The states are stepped forward often, in steps of order dt_limit. The big step
    // states are a second copy which lags behind and is only ever moved in single large
    // jumps, so they do not accumulate the error of many short integrations. They are
    // periodically swapped in to replace the small step states.
    let mut states = pop.states;
    let mut big_step_states = states.clone();
    let mut big_step_non_gravs = non_gravs.clone();
    let mut visible: Vec<PySimultaneousStates> = Vec::new();

    let spk = LOADED_SPK
        .read()
        .expect("Failed to read the loaded spice kernels.");

    // Propagate every state to the given time, dropping any which fail. Failures drop
    // the state and its non-gravitational model together, keeping the two lists aligned.
    let propagate = |states: Vec<State<Equatorial>>,
                     non_gravs: Vec<Option<FrozenNonGrav>>,
                     jd: Time<TDB>| {
        states
            .into_par_iter()
            .zip(non_gravs)
            .filter_map(|(state, non_grav)| {
                let ssb = spk.try_to_ssb(state).ok()?;
                let grav = SpkNBody::new(&spk, include_asteroids);
                let moved = match non_grav.as_ref() {
                    None => ssb.propagate_with(&grav, jd),
                    Some(non_grav) => {
                        let force = Sum::new(grav, Recenter::<SSB, _>::new(&spk, non_grav.clone()));
                        ssb.propagate_with(&force, jd)
                    }
                };
                moved.ok().map(|moved| (moved.into(), non_grav))
            })
            .unzip()
    };

    for fovs in fov_chunks {
        let jd_mean = (fovs.last().unwrap().observer().epoch.jd
            + fovs.first().unwrap().observer().epoch.jd)
            / 2.0;

        // Take large steps which are 10x the smaller steps, this helps long term numerical stability
        if (jd_mean - big_jd).abs() >= dt_limit * 50.0 {
            big_jd = jd_mean;
            (big_step_states, big_step_non_gravs) =
                propagate(big_step_states, big_step_non_gravs, jd.into());
        };
        // Take small steps based off of the large steps.
        if (jd_mean - jd).abs() >= dt_limit {
            if (jd - big_jd).abs() >= dt_limit * 25.0 {
                states.clone_from(&big_step_states);
                non_gravs.clone_from(&big_step_non_gravs);
            }
            jd = jd_mean;
            (states, non_gravs) = propagate(states, non_gravs, jd.into());
        };

        // Release the GIL during CPU-intensive parallel work so Python can
        // handle signals and other threads can proceed.
        let vis: Vec<Vec<PySimultaneousStates>> = py.detach(|| {
            fovs.par_chunks(100)
                .map(|chunk| {
                    let mut found = Vec::new();
                    for fov in chunk {
                        let seen = fov_checks::check_visible(
                            fov,
                            &states,
                            &non_gravs,
                            dt_limit,
                            include_asteroids,
                        )?;
                        found.extend(seen.into_iter().flatten().map(PySimultaneousStates::from));
                    }
                    Ok(found)
                })
                .collect::<KeteResult<Vec<_>>>()
        })?;
        visible.extend(vis.into_iter().flatten());

        py.check_signals()?;
    }
    Ok(visible)
}

/// Check if a list of loaded spice kernel objects are visible in the provided FOVs.
///
/// Returns only the objects which are visible to the  observer, adding a correction
/// for optical light delay.
///
/// Parameters
/// ----------
/// obj_ids :
///     Vector of spice kernel IDs to check.
/// fovs :
///     Collection of Field of Views to check.
#[pyfunction]
#[pyo3(name = "fov_spk_check")]
pub fn fov_spk_checks_py(
    py: Python<'_>,
    obj_ids: Vec<i32>,
    mut fovs: Vec<AllowedFOV>,
) -> Vec<PySimultaneousStates> {
    fovs.sort_by(|a, b| a.jd().jd.total_cmp(&b.jd().jd));

    py.detach(|| {
        fovs.into_par_iter()
            .filter_map(|fov| {
                let fov = fov.unwrap();
                let vis: Vec<_> = fov_checks::check_spks(&fov, &obj_ids)
                    .into_iter()
                    .filter_map(|pop| pop.map(|p| PySimultaneousStates(Box::new(p))))
                    .collect();
                match vis.is_empty() {
                    true => None,
                    false => Some(vis),
                }
            })
            .flatten()
            .collect()
    })
}

/// Check if a list of static sky positions are present in the given Field of View list.
///
/// This returns a list of tuples, where the first entry in the tuple is a vector of
/// indices, where if the input vector shows up in the specific FOV, the index
/// corresponding to that vector is returned, and the second entry is the original FOV.
///
/// An example:
/// Given a list of containing 6 vectors, and 2 FOVs ('a' and 'b'). If the first 3
/// vectors are in field 'a' and the second 3 in 'b', then the returned values will be
/// `[([0, 1, 2], fov_a), ([3, 4, 5], fov_b)`. If a third fov is provided,
/// but none of the vectors are contained within it, then nothing will be returned.
///
/// Parameters
/// ----------
/// pos :
///     Collection of Vectors defining sky positions from the point of view of the observer.
/// fovs :
///     Collection of Field of Views to check.
#[pyfunction]
#[pyo3(name = "fov_static_check")]
pub fn fov_static_checks_py(
    pos: Vec<VectorLike>,
    mut fovs: Vec<AllowedFOV>,
) -> Vec<(Vec<usize>, AllowedFOV)> {
    fovs.sort_by(|a, b| a.jd().jd.total_cmp(&b.jd().jd));
    let pos: Vec<_> = pos
        .into_iter()
        .map(|p| p.into_vector(crate::frame::PyFrames::Ecliptic))
        .collect();

    fovs.into_par_iter()
        .filter_map(|fov| {
            let fov = fov.unwrap();
            let vis: Vec<_> = check_statics(&fov, &pos)
                .into_iter()
                .filter_map(|pop| pop.map(|(p_vec, fov)| (p_vec, fov.into())))
                .collect();
            match vis.is_empty() {
                true => None,
                false => Some(vis),
            }
        })
        .flatten()
        .collect()
}
