//! State Transition matrix computation
use kete_core::prelude::*;
use kete_spice::prelude::compute_state_transition;
use kete_spice::spk::LOADED_SPK;
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
/// The input state may use any center; this wrapper re-centers to SSB for the
/// underlying `compute_state_transition` (which requires SSB) and restores the
/// original center on output.
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

    let (mut final_state, sens) =
        compute_state_transition(&raw_state, jd, include_asteroids, non_grav_model)?;

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
