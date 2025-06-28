use kete_core::spice::LOADED_CK;
use pyo3::{PyResult, pyfunction};

use crate::{
    frame::PyFrames,
    time::PyTime,
    vector::{PyVector, VectorLike},
};

/// Load all specified files into the CK shared memory singleton.
#[pyfunction]
#[pyo3(name = "ck_load")]
pub fn ck_load_py(filenames: Vec<String>) -> PyResult<()> {
    let mut singleton = LOADED_CK.write().unwrap();
    for filename in filenames.iter() {
        let load = (*singleton).load_file(filename);
        if let Err(err) = load {
            eprintln!("{filename} failed to load. {err}");
        }
    }
    Ok(())
}

/// Reset the contents of the CK shared memory to the default set of CK kernels.
#[pyfunction]
#[pyo3(name = "ck_reset")]
pub fn ck_reset_py() {
    LOADED_CK.write().unwrap().reset()
}

/// List all loaded instruments in the CK singleton.
#[pyfunction]
#[pyo3(name = "ck_loaded_instruments")]
pub fn ck_loaded_instruments_py() -> Vec<i32> {
    let singleton = LOADED_CK.read().unwrap();
    singleton.loaded_instruments()
}

/// List all loaded instruments in the CK singleton.
#[pyfunction]
#[pyo3(name = "ck_loaded_instrument_info")]
pub fn ck_loaded_instrument_info_py(instrument_id: i32) -> Vec<(i32, i32, i32, f64, f64)> {
    let singleton = LOADED_CK.read().unwrap();
    singleton.available_info(instrument_id)
}

/// Convert a vector in the specified frame to equatorial coordinates.
///
/// This returns the closest time found in the CK kernels, along with the rotated
/// vector.
///
#[pyfunction]
#[pyo3(name = "instrument_frame_to_equatorial")]
pub fn ck_sc_frame_to_equatorial(
    instrument_id: i32,
    jd: PyTime,
    vec: [f64; 3],
) -> PyResult<(PyTime, PyVector)> {
    let time = jd.0;
    let cks = LOADED_CK.try_read().unwrap();
    let (time, frame) = cks.try_get_frame(time.jd, instrument_id)?;

    let (pos, _) = frame.to_equatorial(vec.into(), [0.0; 3].into())?;

    let vec = PyVector::new(pos.into(), PyFrames::Equatorial);

    Ok((time.into(), vec))
}

/// Convert a vector from the equatorial frame to an instrument frame.
///
/// This returns the closest time found in the CK kernels, along with the rotated
/// vector.
///
#[pyfunction]
#[pyo3(name = "instrument_equatorial_to_frame")]
pub fn ck_sc_equatorial_to_frame(
    instrument_id: i32,
    jd: PyTime,
    vec: VectorLike,
) -> PyResult<(PyTime, [f64; 3])> {
    let vec = vec.into_vector(PyFrames::Ecliptic);
    let time = jd.0;
    let cks = LOADED_CK.try_read().unwrap();
    let (time, frame) = cks.try_get_frame(time.jd, instrument_id)?;

    let (pos, _) = frame.from_equatorial(vec.into(), [0.0; 3].into())?;

    Ok((time.into(), pos.into()))
}
