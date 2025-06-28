//! Python support for kepler orbit calculations
use itertools::Itertools;
use kete_core::frames::{Ecliptic, Equatorial, Vector};
use kete_core::state::State;
use kete_core::{constants, propagation};
use pyo3::{PyErr, PyObject, Python, exceptions};
use pyo3::{PyResult, pyfunction};
use rayon::prelude::*;

use crate::maybe_vec::{MaybeVec, maybe_vec_to_pyobj};
use crate::state::PyState;
use crate::time::PyTime;
use crate::vector::PyVector;

/// Solve kepler's equation for the Eccentric Anomaly.
///
/// Parameters
/// ----------
/// ecc :
///     Eccentricity, must be non-negative.
/// mean_anom :
///     Mean Anomaly between 0 and 2*pi.
/// peri_dist :
///     Perihelion distance in AU.
#[pyfunction]
#[pyo3(name = "compute_eccentric_anomaly")]
pub fn compute_eccentric_anomaly_py(
    ecc: Vec<f64>,
    mean_anom: Vec<f64>,
    peri_dist: Vec<f64>,
) -> PyResult<Vec<f64>> {
    if ecc.len() != mean_anom.len() || ecc.len() != peri_dist.len() {
        return Err(PyErr::new::<exceptions::PyValueError, _>(
            "Input lengths must all match.",
        ));
    }
    Ok(ecc
        .iter()
        .zip(mean_anom)
        .zip(peri_dist)
        .collect_vec()
        .par_iter()
        .map(|((e, anom), peri)| {
            propagation::compute_eccentric_anomaly(**e, *anom, *peri).unwrap_or(f64::NAN)
        })
        .collect())
}

/// Propagate the :class:`~kete.State` for all the objects to the specified time.
/// This assumes 2 body interactions.
///
/// This is a multi-core operation.
///
/// Parameters
/// ----------
/// state :
///     List of states, which are in units of AU from the Sun and velocity is in AU/Day.
/// jd :
///     Time to integrate to in JD days with TDB scaling.
/// observer_pos :
///     A vector of length 3 describing the position of an observer. If this is
///     provided then the estimated states will be returned as a result of light
///     propagation delay.
///
/// Returns
/// -------
/// State
///     Final states after propagating to the target time.
#[pyfunction]
#[pyo3(name = "propagate_two_body", signature = (states, jd, observer_pos=None))]
pub fn propagation_kepler_py(
    py: Python<'_>,
    states: MaybeVec<PyState>,
    jd: PyTime,
    observer_pos: Option<PyVector>,
) -> PyResult<PyObject> {
    let (states, was_vec): (Vec<_>, bool) = states.into();
    let jd = jd.jd();
    let states = states
        .par_iter()
        .with_min_len(10)
        .map(|state| {
            let center = state.center_id();
            let frame = state.frame();

            let Some(state) = state.change_center(crate::desigs::NaifIDLike::Int(10)).ok() else {
                let nan_state: PyState =
                    State::<Ecliptic>::new_nan(state.raw.desig.clone(), jd, center).into();
                return nan_state.change_frame(frame);
            };

            let Some(mut new_state) = propagation::propagate_two_body(&state.raw, jd).ok() else {
                let nan_state: PyState =
                    State::<Ecliptic>::new_nan(state.raw.desig.clone(), jd, center).into();
                return nan_state.change_frame(frame);
            };

            if let Some(observer_pos) = &observer_pos {
                let observer_pos: Vector<Equatorial> = observer_pos.clone().into();
                let delay = -(new_state.pos - observer_pos).norm() / constants::C_AU_PER_DAY;
                new_state = match propagation::propagate_two_body(&new_state, new_state.jd + delay)
                {
                    Ok(state) => state,
                    Err(_) => State::new_nan(state.raw.desig.clone(), jd, center),
                };
            }
            let new_pystate: PyState = new_state.into();

            new_pystate
                .change_frame(frame)
                .change_center(crate::desigs::NaifIDLike::Int(center))
                .unwrap_or(
                    State::<Ecliptic>::new_nan(state.raw.desig, jd, state.raw.center_id).into(),
                )
        })
        .collect();
    maybe_vec_to_pyobj(py, states, was_vec)
}
