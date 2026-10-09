// SPDX-FileCopyrightText: 2026 Dar Dahlen
// SPDX-FileCopyrightText: 2025 California Institute of Technology
// SPDX-License-Identifier: BSD-3-Clause

use super::*;
use kete_core::errors::Error;
use kete_core::forces::NonGravMask;
use kete_core::fov::{FOV, check_ephemeris, check_statics, check_visible};
use kete_spice::ephemeris::SpiceEphemeris;
use pyo3::exceptions::PyDeprecationWarning;
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::{nongrav::PyNonGravModel, state::PySimultaneousStates, vector::VectorLike};

/// Given states and field of view, return only the objects which are visible to the
/// observer, adding a correction for optical light delay.
///
/// Each object is propagated once with n-body physics across the time span of the
/// FOVs, including its non-gravitational model if one is given, and checked against
/// every FOV.
///
/// The check runs to completion once started, it is not stopped by a keyboard
/// interrupt.
///
/// Parameters
/// ----------
/// obj_state: list[State]
///     States which do not already have a specified FOV.
/// fovs: list
///     A field of view from which to subselect objects which are visible.
/// dt_limit: float
///     Deprecated and unused, passing it raises a :class:`DeprecationWarning`.
/// include_asteroids: bool
///     Include the additional registered gravitational masses during the computation.
/// non_gravs: list
///     A list of non-gravitational terms for each object. If provided, then every
///     object must have an associated :class:`~kete.propagation.NonGravModel` or `None`.
#[pyfunction]
#[pyo3(name = "fov_state_check", signature = (obj_state, fovs, dt_limit=None,
    include_asteroids=false, non_gravs=None))]
pub fn fov_checks_py(
    py: Python<'_>,
    obj_state: PySimultaneousStates,
    mut fovs: Vec<AllowedFOV>,
    dt_limit: Option<f64>,
    include_asteroids: bool,
    non_gravs: Option<Vec<Option<PyNonGravModel>>>,
) -> PyResult<Vec<PySimultaneousStates>> {
    if dt_limit.is_some() {
        PyErr::warn(
            py,
            &py.get_type::<PyDeprecationWarning>(),
            c"fov_state_check: dt_limit is unused and will be removed.",
            1,
        )?;
    }
    let states = obj_state.0.states;
    let non_gravs: Vec<Option<NonGravMask>> = match non_gravs {
        None => Vec::new(),
        Some(models) => {
            if models.len() != states.len() {
                Err(Error::ValueError(
                    "non_gravs must be the same length as states.".into(),
                ))?;
            }
            models
                .into_iter()
                .map(|model| model.map(|model| model.to_fixed()))
                .collect()
        }
    };

    fovs.sort_by(|a, b| a.jd().jd().total_cmp(&b.jd().jd()));
    let fovs: Vec<FOV> = fovs.into_iter().map(AllowedFOV::unwrap).collect();

    let visible = py.detach(|| {
        let eph = SpiceEphemeris::loaded()?;
        check_visible(&eph, &fovs, &states, &non_gravs, include_asteroids)
    })?;
    Ok(visible
        .into_iter()
        .map(|(_, _, patch)| PySimultaneousStates(Box::new(patch)))
        .collect())
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
) -> PyResult<Vec<PySimultaneousStates>> {
    fovs.sort_by(|a, b| a.jd().jd().total_cmp(&b.jd().jd()));

    // The SPK read guard is taken and dropped while the GIL is released, so a thread
    // holding the GIL and waiting to load kernels cannot block on it.
    let visible: Vec<Vec<PySimultaneousStates>> = py.detach(|| {
        let eph = SpiceEphemeris::loaded()?;
        fovs.into_par_iter()
            .map(|fov| {
                let fov = fov.unwrap();
                Ok(check_ephemeris(&eph, &fov, &obj_ids)?
                    .into_iter()
                    .filter_map(|pop| pop.map(|p| PySimultaneousStates(Box::new(p))))
                    .collect())
            })
            .collect::<Result<_, Error>>()
    })?;
    Ok(visible.into_iter().flatten().collect())
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
    fovs.sort_by(|a, b| a.jd().jd().total_cmp(&b.jd().jd()));
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
