//! State Transition matrix computation
use kete_core::prelude::*;
use kete_core::propagation::compute_state_transition;
use pyo3::pyfunction;

use crate::time::PyTime;

/// Compute an approximate STM
#[pyfunction]
#[pyo3(name = "compute_stm")]
pub fn compute_stm_py(
    state: [f64; 6],
    jd_start: PyTime,
    jd_end: PyTime,
    central_mass: f64,
) -> ([[f64; 3]; 2], [[f64; 6]; 6]) {
    let mut state = State::new(
        Desig::Empty,
        jd_start.jd(),
        [state[0], state[1], state[2]].into(),
        [state[3], state[4], state[5]].into(),
        10,
    );

    let (final_state, stm) = compute_state_transition(&mut state, jd_end.jd(), central_mass);

    (final_state, stm.into())
}
