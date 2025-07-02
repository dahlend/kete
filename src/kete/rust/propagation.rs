//! Python support for n body propagation
use itertools::Itertools;
use kete_core::{
    errors::Error,
    frames::Ecliptic,
    propagation::{self, NonGravModel, moid},
    spice::{self, LOADED_SPK},
    state::State,
    time::{TDB, Time},
};
use pyo3::{PyObject, PyResult, Python, pyfunction};
use rayon::prelude::*;

use crate::{
    maybe_vec::{MaybeVec, maybe_vec_to_pyobj},
    nongrav::PyNonGravModel,
    state::PyState,
    time::PyTime,
};

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
pub fn moid_py(
    py: Python<'_>,
    state_a: MaybeVec<PyState>,
    state_b: Option<PyState>,
) -> PyResult<PyObject> {
    let (states, was_vec): (Vec<_>, bool) = state_a.into();
    if states.is_empty() {
        Err(Error::ValueError(
            "state_a must have at least one state.".into(),
        ))?;
    }

    let state_b =
        state_b
            .map(|x| x.raw)
            .unwrap_or(LOADED_SPK.read().unwrap().try_get_state_with_center(
                399,
                states[0].raw.jd,
                10,
            )?);

    let moids: Vec<f64> = states
        .into_par_iter()
        .with_min_len(30)
        .map(|state| {
            let state = state.raw;
            moid(state, state_b.clone()).unwrap_or(f64::NAN)
        })
        .collect::<Vec<_>>();

    maybe_vec_to_pyobj(py, moids, was_vec)
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
    states: MaybeVec<PyState>,
    jd: PyTime,
    include_asteroids: bool,
    non_gravs: Option<MaybeVec<Option<PyNonGravModel>>>,
    suppress_errors: bool,
    suppress_impact_errors: bool,
) -> PyResult<PyObject> {
    let (states, was_vec): (Vec<_>, bool) = states.into();
    let non_gravs: Option<Vec<Option<PyNonGravModel>>> = non_gravs.map(|x| x.into());
    let non_gravs = non_gravs.unwrap_or(vec![None; states.len()]);

    if states.len() != non_gravs.len() {
        Err(Error::ValueError(
            "non_gravs must be the same length as states.".into(),
        ))?;
    }

    let mut res: Vec<PyState> = Vec::new();
    let jd = jd.jd();

    // propagation is broken into chunks, every time a chunk is completed
    // python is checked for signals. This allows keyboard interrupts to be caught
    // and the process interrupted.

    for chunk in states
        .into_iter()
        .zip(non_gravs.into_iter())
        .collect_vec()
        .chunks(1000)
    {
        py.check_signals()?;

        let mut proc_chunk =
            chunk
                .to_owned()
                .into_par_iter()
                .with_min_len(5)
                .map(|(state, model)| {
                    let model = model.map(|x| x.0);
                    let center = state.center_id();
                    let frame = state.frame;
                    let state = state.raw;
                    let desig = state.desig.clone();

                    // if the input has a NAN in it, skip the propagation entirely and return
                    // the nans.
                    if !state.is_finite() {
                        if !suppress_errors {
                            Err(Error::ValueError("Input state contains NaNs.".into()))?;
                        };
                        return Ok(Into::<PyState>::into(State::<Ecliptic>::new_nan(
                            desig, jd, center,
                        ))
                        .change_frame(frame));
                    }
                    match propagation::propagate_n_body_spk(state, jd, include_asteroids, model) {
                        Ok(state) => Ok(Into::<PyState>::into(state).change_frame(frame)),
                        Err(er) => {
                            if !suppress_errors {
                                Err(er)?
                            } else if let Error::Impact(id, time) = er {
                                if !suppress_impact_errors {
                                    eprintln!(
                                        "Impact detected between ({}) <-> {} at time {} ({})",
                                        desig,
                                        spice::try_name_from_id(id).unwrap_or(id.to_string()),
                                        time,
                                        Time::<TDB>::new(time).utc().to_iso().unwrap()
                                    );
                                }
                                // if we get an impact, we return a state with NaNs
                                // but put the impact time into the new state.
                                Ok(Into::<PyState>::into(State::<Ecliptic>::new_nan(
                                    desig, time, center,
                                ))
                                .change_frame(frame))
                            } else {
                                Ok(Into::<PyState>::into(State::<Ecliptic>::new_nan(
                                    desig, jd, center,
                                ))
                                .change_frame(frame))
                            }
                        }
                    }
                })
                .collect::<PyResult<Vec<_>>>()?;
        res.append(&mut proc_chunk);
    }

    maybe_vec_to_pyobj(py, res, was_vec)
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
    let states: Vec<_> = states.into_iter().map(|x| x.raw).collect();
    let planet_states: Option<Vec<_>> =
        planet_states.map(|s| s.into_iter().map(|x| x.raw).collect());

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
            let (chunk_state, chunk_nongrav): (Vec<_>, Vec<Option<NonGravModel>>) =
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
