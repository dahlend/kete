use neospy_core::prelude::*;
use neospy_core::propagation::compute_state_transition;
use pyo3::pyfunction;

#[pyfunction]
#[pyo3(name = "compute_stm")]
pub fn compute_stm_py(
    state: [f64; 6],
    jd_start: f64,
    jd_end: f64,
    central_mass: f64,
) -> ([[f64; 3]; 2], [[f64; 6]; 6]) {
    let mut state = State::new(
        Desig::Empty,
        jd_start,
        [state[0], state[1], state[2]].into(),
        [state[3], state[4], state[5]].into(),
        Frame::Ecliptic,
        10,
    );

    let (final_state, stm) = compute_state_transition(&mut state, jd_end, central_mass);

    (final_state, stm.into())
}
