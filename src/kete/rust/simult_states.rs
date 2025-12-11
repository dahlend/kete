//! Python support for simultaneous States.
use kete_core::errors::Error;
use kete_core::io::FileIO;
use kete_core::simult_states::SimultaneousStates;
use kete_core::spice::LOADED_SPK;
use kete_core::time::TDB;
use kete_core::time::Time;
use pyo3::exceptions;
use pyo3::prelude::*;
use pyo3::{PyResult, pyclass, pymethods};

use crate::maybe_vec::MaybeVec;
use crate::time::PyTime;
use crate::vector::PyVector;
use crate::{fovs::AllowedFOV, state::PyState};

/// Representation of a collection of [`State`] at a single point in time.
///
/// The main value in this is that also includes an optional Field of View.
/// If the FOV is provided, it is implied that the states which are present
/// in this file were objects seen by the FOV.
///
/// In the case where the FOV is provided, it is expected that the states
/// positions will include light delay, so an object which is ~1au away from
/// the FOV observer will have a JD which is offset by about 8 minutes.
///
#[pyclass(module = "kete", frozen, sequence, name = "SimultaneousStates")]
#[derive(Debug)]
pub struct PySimultaneousStates(pub Box<SimultaneousStates>);

impl From<SimultaneousStates> for PySimultaneousStates {
    fn from(value: SimultaneousStates) -> Self {
        Self(Box::new(value))
    }
}

impl<'py> FromPyObject<'py> for PySimultaneousStates {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        match ob.downcast_exact::<PySimultaneousStates>() {
            Ok(downcast) => Ok(PySimultaneousStates(downcast.get().0.clone())),
            Err(_) => {
                if let Ok(states) = ob.extract::<Vec<PyState>>() {
                    PySimultaneousStates::new(states, None)
                } else {
                    Err(Error::ValueError(
                        "Input could not be converted to a SimultaneousStates".into(),
                    ))?
                }
            }
        }
    }
}

impl From<PySimultaneousStates> for Vec<PyState> {
    fn from(value: PySimultaneousStates) -> Self {
        value.states()
    }
}

#[pymethods]
impl PySimultaneousStates {
    /// Create a new collection of States at a specific time.
    ///
    ///
    /// Parameters
    /// ----------
    /// states :
    ///     List of States to include.
    /// fov :
    ///     An optional FOV, if this is provided it is expected that the states provided
    ///     are what have been seen by this FOV. This is not checked.
    #[new]
    #[pyo3(signature = (states, fov=None))]
    pub fn new(states: Vec<PyState>, fov: Option<AllowedFOV>) -> PyResult<Self> {
        let states: Vec<_> = states.into_iter().map(|x| x.raw).collect();
        let fov = fov.map(|x| x.unwrap());
        Ok(
            SimultaneousStates::new_exact(states, fov)
                .map(|x| PySimultaneousStates(Box::new(x)))?,
        )
    }

    /// The FOV if it exists.
    #[getter]
    pub fn fov(&self) -> Option<AllowedFOV> {
        self.0.fov.clone().map(|x| x.into())
    }

    /// States contained within.
    #[getter]
    pub fn states(&self) -> Vec<PyState> {
        self.0.states.iter().map(|x| x.clone().into()).collect()
    }

    /// The time of the simultaneous states.
    #[getter]
    pub fn jd(&self) -> PyTime {
        self.0.epoch.into()
    }

    /// The reference center NAIF ID for this state.
    #[getter]
    pub fn center_id(&self) -> i32 {
        self.0.center_id
    }

    /// Load a single SimultaneousStates from a file.
    #[staticmethod]
    pub fn load(filename: String) -> PyResult<Self> {
        Ok(PySimultaneousStates(Box::new(SimultaneousStates::load(
            filename,
        )?)))
    }

    /// Save a single SimultaneousStates to a file.
    pub fn save(&self, filename: String) -> PyResult<()> {
        let _ = self.0.save(filename)?;
        Ok(())
    }

    /// Save states as a parquet file.
    ///
    /// Optionally save the times when the states were last updated.
    /// If a single value is provided, then all states are assumed to have been updated
    /// at the same time, otherwise the number of provided times must match the number
    /// of states.
    #[pyo3(signature = (filename, last_updated=None))]
    pub fn save_parquet(
        &self,
        filename: String,
        last_updated: Option<MaybeVec<PyTime>>,
    ) -> PyResult<()> {
        if self.0.fov.is_some() {
            Err(Error::IOError(
                "Cannot save a SimultaneousStates object which has a FOV as parquet. \
                Parquet can only support a basic table format. Saving metadata such \
                as a field of view is not feasible. Consider using the binary saving \
                method `SimultaneousStates.save`."
                    .into(),
            ))?;
        }
        let last_updated: Option<(Vec<_>, bool)> = last_updated.map(|v| v.into());

        if let Some((update, was_vec)) = &last_updated
            && *was_vec
            && update.len() != self.0.states.len()
        {
            Err(Error::ValueError(
                "The number of updated times provided does not match the number of \
                states."
                    .into(),
            ))?;
        };

        let last_updated: Option<Vec<Time<TDB>>> = last_updated.map(|(v, was_vec)| {
            if was_vec {
                v.into_iter().map(|t| t.into()).collect()
            } else {
                vec![v.first().unwrap().0; self.0.states.len()]
            }
        });
        kete_core::io::parquet::write_states_parquet(&self.0.states, &filename, last_updated)?;
        Ok(())
    }

    /// Load states from a parquet file.
    #[staticmethod]
    pub fn load_parquet(filename: String) -> PyResult<Self> {
        let states = kete_core::io::parquet::read_states_parquet(&filename)?.0;

        Ok(PySimultaneousStates(Box::new(
            SimultaneousStates::new_exact(states, None)?,
        )))
    }

    /// Load the last time the states were updated as saved in a parquet file.
    #[staticmethod]
    pub fn load_parquet_update_times(filename: String) -> PyResult<Vec<Option<PyTime>>> {
        let update_times = kete_core::io::parquet::read_update_times_parquet(&filename)?;
        Ok(update_times.into_iter().map(|x| x.map(PyTime)).collect())
    }

    /// Length of states
    pub fn __len__(&self) -> usize {
        self.0.states.len()
    }

    /// Get the Nth state
    pub fn __getitem__(&self, mut idx: isize) -> PyResult<PyState> {
        if idx < 0 {
            idx += self.0.states.len() as isize;
        }
        if (idx < 0) || (idx as usize >= self.__len__()) {
            return Err(PyErr::new::<exceptions::PyIndexError, _>(
                "index out of range",
            ));
        }
        Ok(self.0.states[idx as usize].clone().into())
    }

    /// If a FOV is present, calculate all vectors from the observer position to the
    /// position of the objects.
    ///
    #[getter]
    pub fn obs_vecs(&self) -> PyResult<Vec<PyVector>> {
        let fov = self
            .fov()
            .ok_or(PyErr::new::<exceptions::PyValueError, _>(
                "FOV not present, cannot compute vectors.",
            ))?
            .unwrap();
        let obs = fov.observer();
        let spk = LOADED_SPK.try_read().unwrap();

        let mut vecs = Vec::with_capacity(self.__len__());
        for state in &self.0.states {
            if state.center_id != obs.center_id {
                let mut state = state.clone();
                spk.try_change_center(&mut state, obs.center_id)?;
                let diff = state.pos - obs.pos;
                vecs.push(diff.into());
            } else {
                let diff = state.pos - obs.pos;
                vecs.push(diff.into());
            }
        }
        Ok(vecs)
    }

    /// If a FOV is present, this returns the phases of the observed objects.
    ///
    /// Specifically, this is the angle between the `center_id` and the observer as seen
    /// from the object. Typically the center is the Sun.
    #[getter]
    pub fn phase_angles(&self) -> PyResult<Vec<f64>> {
        let obs_vecs = self.obs_vecs()?;

        self.0
            .states
            .iter()
            .zip(obs_vecs)
            .map(|(state, vec)| {
                let phase = state.pos.angle(&vec.into());
                Ok(phase.to_degrees())
            })
            .collect()
    }

    /// If a FOV is present, this returns the elongation of the observed objects.
    ///
    /// If the center ID of the observer is the Sun, this is the solar elongation.
    ///
    /// Specifically, this is the angle between the `center_id` and the object as seen
    /// from the observer. Typically the center is the Sun.
    #[getter]
    pub fn elongation(&self) -> PyResult<Vec<f64>> {
        let obs_vecs = self.obs_vecs()?;
        let fov = self.fov().unwrap().unwrap();
        let observer = fov.observer();

        obs_vecs
            .into_iter()
            .map(|vec| {
                let elongation = observer.pos.angle(&vec.into());
                Ok(elongation.to_degrees())
            })
            .collect()
    }

    /// If a FOV is present, calculate the RA/Decs and their rates for all states in this object.
    /// This will automatically convert all frames to Equatorial.
    ///
    /// 4 numbers are returned for each object, [RA, DEC, RA', DEC'], where rates are provided in
    /// degrees/day.
    ///
    /// The returned RA' rate is scaled by cos(dec) so that it is equivalent to a
    /// linear projection onto the observing plane.
    ///
    #[getter]
    pub fn ra_dec_with_rates(&self) -> PyResult<Vec<[f64; 4]>> {
        Ok(self
            .0
            .ra_dec_with_rates()?
            .into_iter()
            .map(|[ra, dec, dra, ddec]| {
                [
                    ra.to_degrees(),
                    dec.to_degrees(),
                    dra.to_degrees(),
                    ddec.to_degrees(),
                ]
            })
            .collect())
    }

    fn __repr__(&self) -> String {
        let n_states = self.0.states.len();
        let fov_str = match self.fov() {
            None => "None".into(),
            Some(f) => f.__repr__(),
        };
        format!("SimultaneousStates(states=<{n_states} States>, fov={fov_str})",)
    }

    /// Save a list to a binary file.
    ///
    /// Note that this saves a list of SimultaneousStates, meaning it is a list of a list of States.
    #[staticmethod]
    #[pyo3(name = "save_list")]
    pub fn py_save_list(vec: Vec<Self>, filename: String) -> PyResult<()> {
        let vec: Vec<_> = vec.into_iter().map(|x| *x.0).collect();
        Ok(SimultaneousStates::save_vec(&vec, filename)?)
    }

    /// Load a list from a binary file.
    ///
    /// Note that this loads a list of SimultaneousStates, meaning it is a list of a list of States.
    #[staticmethod]
    #[pyo3(name = "load_list")]
    pub fn py_load_list(filename: String) -> PyResult<Vec<Self>> {
        let res = SimultaneousStates::load_vec(filename)?;
        Ok(res.into_iter().map(|x| Self(Box::new(x))).collect())
    }
}
