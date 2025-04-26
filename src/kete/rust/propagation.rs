//! Python support for n body propagation
use itertools::Itertools;
use kete_core::{
    errors::Error,
    propagation::{self, moid, NonGravModel},
    spice::{self, LOADED_SPK},
    state::State,
    time::{scales::TDB, Time},
};
use pyo3::{pyfunction, PyResult, Python};
use rayon::prelude::*;

use crate::state::PyState;
use crate::{nongrav::PyNonGravModel, time::PyTime};

/// Compute the MOID between the input state and an optional second state.
/// If the second state is not provided, default to Earth.
///
/// Returns the MOID in units of au.
///
/// Parameters
/// ----------
/// state_a:
///     State of the first object.
/// state_b:
///     Optional state of the second object, defaults to Earth.
#[pyfunction]
#[pyo3(name = "moid", signature = (state_a, state_b=None))]
pub fn moid_py(state_a: PyState, state_b: Option<PyState>) -> PyResult<f64> {
    let state_b =
        state_b
            .map(|x| x.0)
            .unwrap_or(LOADED_SPK.read().unwrap().try_get_state_with_center(
                399,
                state_a.0.jd,
                10,
                state_a.0.frame,
            )?);
    Ok(moid(state_a.0, state_b)?)
}

/// Propagate the provided :class:`~kete.State` using N body mechanics to the
/// specified times, no approximations are made, this can be very CPU intensive.
///
/// This does not compute light delay, however it does include corrections for general
/// relativity due to the Sun.
///
/// Parameters
/// ----------
/// states:
///     The initial states, this is a list of multiple State objects.
/// jd:
///     A JD to propagate the initial states to.
/// include_asteroids:
///     If this is true, the computation will include the largest 5 asteroids.
///     The asteroids are: Ceres, Pallas, Interamnia, Hygiea, and Vesta.
/// non_gravs:
///     A list of non-gravitational terms for each object. If provided, then every
///     object must have an associated :class:`~NonGravModel` or `None`.
/// suppress_errors:
///     If True, errors during propagation will return NaN for the relevant state
///     vectors, but propagation will continue.
/// suppress_impact_errors:
///     If True, impacts will be printed to stderr, but states will still return
///     filled with `NaN`. If False, impacts are not printed.
///
/// Returns
/// -------
/// Iterable
///     A :class:`~kete.State` at the new time.
#[pyfunction]
#[pyo3(name = "propagate_n_body", signature = (states, jd, include_asteroids=false,
    non_gravs=None, suppress_errors=true, suppress_impact_errors=false))]
pub fn propagation_n_body_spk_py(
    py: Python<'_>,
    states: Vec<PyState>,
    jd: PyTime,
    include_asteroids: bool,
    non_gravs: Option<Vec<Option<PyNonGravModel>>>,
    suppress_errors: bool,
    suppress_impact_errors: bool,
) -> PyResult<Vec<PyState>> {
    let states: Vec<State> = states.into_iter().map(|x| x.0).collect();
    let non_gravs = non_gravs.unwrap_or(vec![None; states.len()]);

    if states.len() != non_gravs.len() {
        Err(Error::ValueError(
            "non_gravs must be the same length as states.".into(),
        ))?;
    }

    let mut res: Vec<PyState> = Vec::new();
    let jd = jd.jd();

    // propagation is broken into chunks of 1000 states, every time a chunk is completed
    // python is checked for signals. This allows keyboard interrupts to be caught
    // and the process interrupted.

    for chunk in states
        .into_iter()
        .zip(non_gravs.into_iter())
        .collect_vec()
        .chunks(1000)
    {
        py.check_signals()?;

        let mut proc_chunk = chunk
            .to_owned()
            .into_par_iter()
            .map(|(state, model)| {
                let model = model.map(|x| x.0);
                let center = state.center_id;

                // if the input has a NAN in it, skip the propagation entirely and return
                // the nans.
                if !state.is_finite() {
                    if !suppress_errors {
                        Err(Error::ValueError("Input state contains NaNs.".into()))?;
                    };
                    return Ok(State::new_nan(state.desig, jd, state.frame, center).into());
                }
                let desig = state.desig.clone();
                let frame = state.frame;
                match propagation::propagate_n_body_spk(state, jd, include_asteroids, model) {
                    Ok(state) => Ok(state.into()),
                    Err(er) => {
                        if !suppress_errors {
                            Err(er)?
                        } else {
                            if let Error::Impact(id, time) = er {
                                let time_full: Time<TDB> = Time::new(time);
                                if !suppress_impact_errors {
                                    eprintln!(
                                        "Impact detected between ({:?}) <-> ({}) at time {} ({})",
                                        desig,
                                        spice::try_name_from_id(id).unwrap_or(id.to_string()),
                                        time,
                                        time_full.utc().to_iso().unwrap()
                                    );
                                }
                            };
                            Ok(State::new_nan(desig, jd, frame, center).into())
                        }
                    }
                }
            })
            .collect::<PyResult<Vec<_>>>()?;
        res.append(&mut proc_chunk);
    }

    Ok(res)
}

/// It is *STRONGLY* recommended to use `propagate_n_body` instead of this function
/// wherever possible. This function is specifically meant for kilo-year or longer
/// simulations, it is slower and less accurate than `propagate_n_body`, but that
/// function only works for as long as there are SPICE kernels for the planets
/// available.
///
/// This is designed to not require SPICE kernels to function, and is meant for long
/// term simulations on the other of less than a mega-year. It is not recommended
/// for orbits longer than this.
///
/// Propagation using this will treat the Earth and Moon as a single object for
/// performance reasons.
///
/// Propagate the provided :class:`~kete.State` using N body mechanics to the
/// specified times, very few approximations are made, this can be very CPU intensive.
///
/// This does not compute light delay, however it does include corrections for general
/// relativity due to the Sun.
///
/// This returns two lists of states:
/// - First one contains the states of the objects at the end of the integration
/// - Second contains the states of the planets at the end of the integration.
///
/// The second set of states may be used as input for continuing the integration.
///
/// Parameters
/// ----------
/// states:
///     The initial states, this is a list of multiple State objects.
/// jd:
///     A JD to propagate the initial states to.
/// planet_states:
///     Optional list of the planet's states at the same time as the provided states.
///     If this is not provided, the planets positions are loaded from the Spice kernels
///     if that information is available.
/// non_gravs:
///     A list of non-gravitational terms for each object. If provided, then every
///     object must have an associated :class:`~NonGravModel`.
/// batch_size:
///     Number of objects to propagate at once with the planets. This is used to break
///     up the simulation for multi-core support. It additionally has effects on the
///     integrator stepsize which is difficult to predict before running. This can be
///     manually tuned for increased performance, it should have no other effects than
///     performance.
///
/// Returns
/// -------
/// Iterable
///     A :class:`~kete.State` at the new time.
#[pyfunction]
#[pyo3(name = "propagate_n_body_long", signature = (states, jd_final, planet_states=None, non_gravs=None, batch_size=10))]
pub fn propagation_n_body_py(
    states: Vec<PyState>,
    jd_final: PyTime,
    planet_states: Option<Vec<PyState>>,
    non_gravs: Option<Vec<Option<PyNonGravModel>>>,
    batch_size: usize,
) -> PyResult<(Vec<PyState>, Vec<PyState>)> {
    let states: Vec<State> = states.into_iter().map(|x| x.0).collect();
    let planet_states: Option<Vec<State>> =
        planet_states.map(|s| s.into_iter().map(|x| x.0).collect());

    let non_gravs = non_gravs.unwrap_or(vec![None; states.len()]);
    let non_gravs: Vec<Option<NonGravModel>> =
        non_gravs.into_iter().map(|y| y.map(|z| z.0)).collect();

    let jd = jd_final.jd();
    let res = states
        .into_iter()
        .zip(non_gravs.into_iter())
        .collect_vec()
        .par_chunks(batch_size)
        .map(|chunk| {
            let (chunk_state, chunk_nongrav): (Vec<State>, Vec<Option<NonGravModel>>) =
                chunk.iter().cloned().unzip();

            propagation::propagate_n_body_vec(chunk_state, jd, planet_states.clone(), chunk_nongrav)
                .map(|(states, planets)| {
                    (
                        states.into_iter().map(PyState::from).collect::<Vec<_>>(),
                        planets.into_iter().map(PyState::from).collect::<Vec<_>>(),
                    )
                })
        })
        .collect::<Result<Vec<_>, _>>()?;

    let mut final_states = Vec::new();
    let mut final_planets = Vec::new();
    for (mut state, planet) in res.into_iter() {
        final_states.append(&mut state);
        final_planets = planet;
    }
    Ok((final_states, final_planets))
}
