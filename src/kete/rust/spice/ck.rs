use kete_core::{frames::EquatorialNonInertial, spice::LOADED_CK};
use pyo3::{pyfunction, PyResult};

use crate::{time::PyTime, vector::PyVector};
use kete_core::frames::NonInertialFrame;

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

/// Convert a vector in the specified frame to equatorial coordinates.
#[pyfunction]
#[pyo3(name = "frame_to_equatorial")]
pub fn ck_frame_to_equatorial(
    jd: PyTime,
    vec: [f64; 3],
    naif_id: i32,
) -> PyResult<(PyTime, PyVector)> {
    let time = jd.0;
    let cks = LOADED_CK.try_read().unwrap();
    let (time, frame) = cks.try_get_frame::<EquatorialNonInertial>(time.jd, naif_id)?;

    let (pos, _) = frame.to_equatorial(vec.into(), [0.0; 3].into());

    let vec = PyVector::new(pos.into(), crate::frame::PyFrames::Equatorial);

    Ok((time.into(), vec))
}
