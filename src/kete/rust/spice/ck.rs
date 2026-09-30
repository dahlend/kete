use kete_spice::prelude::{LOADED_CK, rotations_to_equatorial_full};
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

/// Convert a vector in an instrument frame to the equatorial frame.
///
/// Where loaded CK kernels overlap, the kernel loaded last takes precedence.
/// Pointing is not extrapolated. A time inside a gap between the intervals of
/// a kernel has no pointing, unless another loaded kernel covers that time.
///
/// Parameters
/// ----------
/// instrument_id : int
///   NAIF ID of the instrument.
/// jd : :class:`~kete.Time` or float
///   Time of the request. A float is a Julian date in the TDB scale.
/// vec : list of float
///   Vector in the instrument frame, with 3 components.
///
/// Returns
/// -------
/// tuple of (:class:`~kete.Time`, :class:`~kete.Vector`)
///   Time of the pointing used, and the vector in the equatorial frame.
///
/// Raises
/// ------
/// ValueError
///   If no loaded CK holds pointing for the instrument at ``jd``, or if no
///   SCLK kernel is loaded for the spacecraft. The same error occurs for a
///   frame in the chain of reference frames of the instrument.
#[pyfunction]
#[pyo3(name = "instrument_frame_to_equatorial")]
pub fn ck_sc_frame_to_equatorial(
    instrument_id: i32,
    jd: PyTime,
    vec: [f64; 3],
) -> PyResult<(PyTime, PyVector)> {
    let (time, frame) = LOADED_CK
        .try_read()
        .unwrap()
        .try_get_frame(jd.0.jd, instrument_id)?;

    // A CK frame can be defined relative to another CK frame, such as a camera
    // relative to its spacecraft. The full chain resolves that case.
    let (rot, _) = rotations_to_equatorial_full(&frame)?;
    let pos = rot.transform_vector(&vec.into());

    let vec = PyVector::new(pos.into(), PyFrames::Equatorial);

    Ok((time.into(), vec))
}

/// Convert a vector from the equatorial frame to an instrument frame.
///
/// Where loaded CK kernels overlap, the kernel loaded last takes precedence.
/// Pointing is not extrapolated. A time inside a gap between the intervals of
/// a kernel has no pointing, unless another loaded kernel covers that time.
///
/// Parameters
/// ----------
/// instrument_id : int
///   NAIF ID of the instrument.
/// jd : :class:`~kete.Time` or float
///   Time of the request. A float is a Julian date in the TDB scale.
/// vec : :class:`~kete.Vector` or list of float
///   Vector to convert. A list is taken to be in the equatorial frame. A
///   :class:`~kete.Vector` is converted from its own frame.
///
/// Returns
/// -------
/// tuple of (:class:`~kete.Time`, list of float)
///   Time of the pointing used, and the 3 components of the vector in the
///   instrument frame.
///
/// Raises
/// ------
/// ValueError
///   If no loaded CK holds pointing for the instrument at ``jd``, or if no
///   SCLK kernel is loaded for the spacecraft. The same error occurs for a
///   frame in the chain of reference frames of the instrument.
#[pyfunction]
#[pyo3(name = "instrument_equatorial_to_frame")]
pub fn ck_sc_equatorial_to_frame(
    instrument_id: i32,
    jd: PyTime,
    vec: VectorLike,
) -> PyResult<(PyTime, [f64; 3])> {
    let vec = vec.into_vector(PyFrames::Equatorial);
    let (time, frame) = LOADED_CK
        .try_read()
        .unwrap()
        .try_get_frame(jd.0.jd, instrument_id)?;

    // A CK frame can be defined relative to another CK frame, such as a camera
    // relative to its spacecraft. The full chain resolves that case.
    let (rot, _) = rotations_to_equatorial_full(&frame)?;
    let pos = rot.inverse_transform_vector(&vec.into());

    Ok((time.into(), pos.into()))
}
