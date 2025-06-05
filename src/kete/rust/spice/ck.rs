use kete_core::spice::LOADED_CK;
use pyo3::{pyfunction, PyResult};

use crate::{time::PyTime, vector::PyVector};

/// Load all specified files into the CK shared memory singleton.
#[pyfunction]
#[pyo3(name = "ck_load")]
pub fn ck_load_py(filenames: Vec<String>) -> PyResult<()> {
    let mut singleton = LOADED_CK.write().unwrap();
    for filename in filenames.iter() {
        let load = (*singleton).load_file(filename);
        if let Err(err) = load {
            eprintln!("{} failed to load. {}", filename, err);
        }
    }
    Ok(())
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
#[pyfunction]
#[pyo3(name = "frame_to_equatorial")]
pub fn ck_frame_to_equatorial(
    jd: PyTime,
    vec: [f64; 3],
    instrument_id: i32,
) -> PyResult<(PyTime, PyVector)> {
    let time = jd.0;
    let cks = LOADED_CK.try_read().unwrap();
    let (time, frame) = cks.try_get_frame(time.jd, instrument_id)?;

    let (pos, _) = frame.to_equatorial(vec.into(), [0.0; 3].into())?;

    let vec = PyVector::new(pos.into(), crate::frame::PyFrames::Equatorial);

    Ok((time.into(), vec))
}
