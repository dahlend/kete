//! State Transition matrix computation
use kete_core::constants::GravParams;
use kete_core::prelude::*;
use kete_core::propagation::compute_state_transition;
use kete_core::spice::LOADED_SPK;
use pyo3::{PyResult, pyfunction};

use crate::nongrav::PyNonGravModel;
use crate::state::PyState;
use crate::time::PyTime;

/// Compute full N-body state transition and parameter sensitivity matrix using the
/// Radau 15th-order integrator.
///
/// Returns `(final_state, sensitivity_matrix)` where the sensitivity matrix is a
/// list-of-lists with 6 rows and 6+N columns (N = number of free non-grav parameters).
///
/// The state must be SSB-centered internally; the function handles re-centering.
#[pyfunction]
#[pyo3(name = "compute_stm", signature = (state, jd_end, include_asteroids=false, non_grav=None))]
pub fn compute_stm_py(
    state: PyState,
    jd_end: PyTime,
    include_asteroids: bool,
    non_grav: Option<PyNonGravModel>,
) -> PyResult<(PyState, Vec<Vec<f64>>)> {
    let center = state.center_id();
    let frame = state.frame;
    let mut raw_state = state.raw;

    // Re-center to SSB (center_id = 0) as required by the Radau integrator.
    // The input state may use any center; we convert, integrate, then convert back.
    {
        let spk = &LOADED_SPK.try_read().map_err(Error::from)?;
        spk.try_change_center(&mut raw_state, 0)?;
    }

    let non_grav_model = non_grav.map(|ng| ng.0);
    let jd = jd_end.into();

    // Call with the appropriate mass list; selected_masses returns a lock guard,
    // planets returns a Vec, so we call separately to satisfy the borrow checker.
    let result = if include_asteroids {
        let masses = GravParams::selected_masses();
        compute_state_transition(&raw_state, jd, &masses, non_grav_model)?
    } else {
        let masses = GravParams::planets();
        compute_state_transition(&raw_state, jd, &masses, non_grav_model)?
    };
    let (mut final_state, sens) = result;

    // Re-center back to original center
    {
        let spk = &LOADED_SPK.try_read().map_err(Error::from)?;
        spk.try_change_center(&mut final_state, center)?;
    }

    // Convert DMatrix to Vec<Vec<f64>> for Python
    let nrows = sens.nrows();
    let ncols = sens.ncols();
    let mat: Vec<Vec<f64>> = (0..nrows)
        .map(|r| (0..ncols).map(|c| sens[(r, c)]).collect())
        .collect();

    let py_state: PyState = final_state.into();
    let py_state = py_state.change_frame(frame);
    Ok((py_state, mat))
}
