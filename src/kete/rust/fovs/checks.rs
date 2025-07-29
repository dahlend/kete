use super::*;
use kete_core::{fov::FOV, propagation::propagate_n_body_spk};
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::{simult_states::PySimultaneousStates, vector::VectorLike};

/// Given states and field of view, return only the objects which are visible to the
/// observer, adding a correction for optical light delay.
///
/// Objects are propagated using 2 body physics to the time of the FOV if time steps are
/// less than the specified `dt`.
///
/// parameters
/// ----------
/// states: list[State]
///     States which do not already have a specified FOV.
/// fov: FOVList
///     A field of view from which to subselect objects which are visible.
/// dt: float
///     Length of time in days where 2-body mechanics is a good approximation.
/// include_asteroids: bool
///     Include the additional registered gravitational masses during the computation.
#[pyfunction]
#[pyo3(name = "fov_state_check", signature = (obj_state, fovs, dt_limit=3.0, include_asteroids=false))]
pub fn fov_checks_py(
    py: Python<'_>,
    obj_state: PySimultaneousStates,
    fovs: FOVListLike,
    dt_limit: f64,
    include_asteroids: bool,
) -> PyResult<Vec<PySimultaneousStates>> {
    let pop = obj_state.0;

    let fovs = fovs.into_sorted_vec_fov();

    // break the fovs into groups based upon the dt_limit
    let mut fov_chunks: Vec<Vec<FOV>> = Vec::new();
    let mut chunk: Vec<FOV> = Vec::new();
    for fov in fovs.into_iter() {
        if chunk.is_empty() {
            chunk.push(fov);
            continue;
        };
        let jd_start = chunk.first().unwrap().observer().jd;

        // chunk is complete
        if (fov.observer().jd - jd_start).abs() >= 2.0 * dt_limit {
            fov_chunks.push(chunk);
            chunk = vec![fov];
        } else {
            chunk.push(fov);
        }
    }
    if !chunk.is_empty() {
        fov_chunks.push(chunk);
    }
    let mut jd = pop.jd;
    let mut big_jd = jd;
    let mut states = pop.states;
    let mut big_step_states = states.clone();
    let mut visible = Vec::new();
    for fovs in fov_chunks.into_iter() {
        let jd_mean =
            (fovs.last().unwrap().observer().jd + fovs.first().unwrap().observer().jd) / 2.0;

        // Take large steps which are 10x the smaller steps, this helps long term numerical stability
        if (jd_mean - big_jd).abs() >= dt_limit * 50.0 {
            big_jd = jd_mean;
            big_step_states = big_step_states
                .into_par_iter()
                .filter_map(|state| propagate_n_body_spk(state, jd, include_asteroids, None).ok())
                .collect();
        };
        // Take small steps based off of the large steps.
        if (jd_mean - jd).abs() >= dt_limit {
            if (jd - big_jd).abs() >= dt_limit * 25.0 {
                states.clone_from(&big_step_states);
            }
            jd = jd_mean;
            states = states
                .into_par_iter()
                .filter_map(|state| propagate_n_body_spk(state, jd, include_asteroids, None).ok())
                .collect();
        };

        let vis: Vec<PySimultaneousStates> = fovs
            .par_chunks(100)
            .flat_map(|chunk| {
                chunk
                    .iter()
                    .cloned()
                    .flat_map(|fov| {
                        fov.check_visible(&states, dt_limit, include_asteroids)
                            .into_iter()
                            .filter_map(|pop| pop.map(|p| PySimultaneousStates(Box::new(p))))
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        if !vis.is_empty() {
            visible.push(vis);
        }

        py.check_signals()?;
    }
    Ok(visible.into_iter().flatten().collect())
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
pub fn fov_spk_checks_py(obj_ids: Vec<i32>, fovs: FOVListLike) -> Vec<PySimultaneousStates> {
    let fovs = fovs.into_sorted_vec_fov();

    fovs.into_par_iter()
        .filter_map(|fov| {
            let vis: Vec<_> = fov
                .check_spks(&obj_ids)
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
    fovs: FOVListLike,
) -> Vec<(Vec<usize>, AllowedFOV)> {
    let fovs = fovs.into_sorted_vec_fov();
    let pos: Vec<_> = pos
        .into_iter()
        .map(|p| p.into_vector(crate::frame::PyFrames::Ecliptic))
        .collect();

    fovs.into_par_iter()
        .filter_map(|fov| {
            let vis: Vec<_> = fov
                .check_statics(&pos)
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
