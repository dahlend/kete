// SPDX-FileCopyrightText: 2026 Dar Dahlen
// SPDX-FileCopyrightText: 2025 California Institute of Technology
// SPDX-License-Identifier: BSD-3-Clause

//! Python support for kepler orbit calculations
use itertools::Itertools;
use kete_core::frames::{Ecliptic, Equatorial, SunCenter, Vector};
use kete_core::kepler;
use kete_core::state::State;
use pyo3::{Py, PyAny, PyErr, Python, exceptions};
use pyo3::{PyResult, pyfunction};
use rayon::prelude::*;

use crate::maybe_vec::{MaybeVec, maybe_vec_to_pyobj};
use crate::state::PyState;
use crate::time::PyTime;
use crate::vector::PyVector;

/// Solve Kepler's equation for the eccentric anomaly.
///
/// Parameters
/// ----------
/// ecc : list of float
///   Eccentricity, must be non-negative.
/// mean_anom : list of float
///   Mean anomaly in radians.
///
/// Returns
/// -------
/// list of float
///   Eccentric anomaly in radians in ``[0, 2 pi)`` for ``ecc <= 1``, and the
///   hyperbolic anomaly for ``ecc > 1``. Invalid inputs give NaN.
#[pyfunction]
#[pyo3(name = "compute_eccentric_anomaly")]
pub fn compute_eccentric_anomaly_py(ecc: Vec<f64>, mean_anom: Vec<f64>) -> PyResult<Vec<f64>> {
    if ecc.len() != mean_anom.len() {
        return Err(PyErr::new::<exceptions::PyValueError, _>(
            "Input lengths must all match.",
        ));
    }
    Ok(ecc
        .iter()
        .zip(mean_anom)
        .collect_vec()
        .par_iter()
        .map(|(e, anom)| kepler::compute_eccentric_anomaly(**e, *anom).unwrap_or(f64::NAN))
        .collect())
}

/// Propagate the :class:`~kete.State` for all the objects to the specified time.
///
/// This assumes two-body motion about the Sun. This is a multi-core operation.
///
/// Parameters
/// ----------
/// states : State or list of State
///   States to propagate, in AU and AU/Day.
/// epoch : float or Time
///   Time to propagate to, in JD with TDB scaling.
/// observer_pos : Vector, optional
///   Sun-centered position of an observer. If it is given, the states are
///   corrected for light travel time to the observer. The delay is iterated
///   until it changes by less than 1e-12 days, at most 3 times. Defaults to
///   no correction.
///
/// Returns
/// -------
/// State or list of State
///   States after propagation. A state that fails to propagate has NaN values.
#[pyfunction]
#[pyo3(name = "propagate_two_body", signature = (states, epoch, observer_pos=None))]
pub fn propagation_kepler_py(
    py: Python<'_>,
    states: MaybeVec<PyState>,
    epoch: PyTime,
    observer_pos: Option<PyVector>,
) -> PyResult<Py<PyAny>> {
    let (states, was_vec): (Vec<_>, bool) = states.into();
    let epoch = epoch.into();
    let states = states
        .par_iter()
        .with_min_len(10)
        .map(|state| {
            let center = state.center_id();
            let frame = state.frame();

            let Some(state) = state.change_center(crate::desigs::NaifIDLike::Int(10)).ok() else {
                let nan_state: PyState =
                    State::<Ecliptic>::new_nan(state.raw.desig.clone(), epoch, center).into();
                return nan_state.change_frame(frame);
            };

            let Ok(sun_state): Result<State<Equatorial, SunCenter>, _> =
                state.raw.clone().try_into()
            else {
                let nan_state: PyState =
                    State::<Ecliptic>::new_nan(state.raw.desig.clone(), epoch, center).into();
                return nan_state.change_frame(frame);
            };

            let Some(mut new_state) = kepler::propagate_two_body(&sun_state, epoch).ok() else {
                let nan_state: PyState =
                    State::<Ecliptic>::new_nan(state.raw.desig.clone(), epoch, center).into();
                return nan_state.change_frame(frame);
            };

            if let Some(observer_pos) = &observer_pos {
                let observer_pos: Vector<Equatorial> = observer_pos.clone().into();
                new_state = match kepler::light_time_correct(&new_state, &observer_pos) {
                    Ok(state) => state,
                    Err(_) => State {
                        desig: state.raw.desig.clone(),
                        epoch,
                        pos: [f64::NAN; 3].into(),
                        vel: [f64::NAN; 3].into(),
                        center: SunCenter,
                    },
                };
            }
            let new_pystate: PyState = State::new(
                new_state.desig,
                new_state.epoch,
                new_state.pos,
                new_state.vel,
                10,
            )
            .into();

            new_pystate
                .change_frame(frame)
                .change_center(crate::desigs::NaifIDLike::Int(center))
                .unwrap_or(State::<Ecliptic>::new_nan(state.raw.desig, epoch, center).into())
        })
        .collect();
    maybe_vec_to_pyobj(py, states, was_vec)
}
