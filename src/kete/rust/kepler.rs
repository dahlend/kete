//! Python support for kepler orbit calculations
use itertools::Itertools;
use kete_core::frames::{Equatorial, Vector};
use kete_core::state::State;
use kete_core::{constants, propagation};
use pyo3::{exceptions, PyErr};
use pyo3::{pyfunction, PyResult};
use rayon::prelude::*;

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
    states: Vec<PyState>,
    jd: PyTime,
    observer_pos: Option<PyVector>,
) -> Vec<PyState> {
    let jd = jd.jd();
    states
        .into_par_iter()
        .map(|state| {
            let center = state.center_id();

            let Some(state) = state.change_center(10).ok() else {
                return State::<Equatorial>::new_nan(state.0.desig.clone(), jd, center).into();
            };

            let Some(mut new_state) = propagation::propagate_two_body(&state.0, jd).ok() else {
                return State::<Equatorial>::new_nan(state.0.desig.clone(), jd, center).into();
            };

            if let Some(observer_pos) = observer_pos {
                let observer_pos: Vector<Equatorial> = observer_pos.into();
                let delay = -(new_state.pos - observer_pos).norm() / constants::C_AU_PER_DAY;

                new_state = match propagation::propagate_two_body(&new_state, new_state.jd + delay)
                {
                    Ok(state) => state,
                    Err(_) => {
                        return State::<Equatorial>::new_nan(state.0.desig.clone(), jd, center)
                            .into()
                    }
                };
            }
            PyState(new_state).change_center(center).unwrap_or(
                State::<Equatorial>::new_nan(state.0.desig.clone(), jd, state.0.center_id).into(),
            )
        })
        .collect()
}
