use kete_core::spice::{try_name_from_id, LOADED_SPK};
use pyo3::{pyfunction, PyResult, Python};

use crate::frame::PyFrames;
use crate::state::PyState;
use crate::time::PyTime;

/// Load all specified files into the SPK shared memory singleton.
#[pyfunction]
#[pyo3(name = "spk_load")]
pub fn spk_load_py(py: Python<'_>, filenames: Vec<String>) -> PyResult<()> {
    let mut singleton = LOADED_SPK.write().unwrap();
    if filenames.len() > 100 {
        eprintln!("Loading {} spk files...", filenames.len());
    }
    for filename in filenames.iter() {
        py.check_signals()?;
        let load = (*singleton).load_file(filename);
        if let Err(err) = load {
            eprintln!("{} failed to load. {}", filename, err);
        }
    }
    Ok(())
}

/// Return all loaded SPK info on the specified NAIF ID.
/// Loaded info contains:
/// (JD_start, JD_end, Center Naif ID, Frame, SPK Segment type ID)
#[pyfunction]
#[pyo3(name = "spk_available_info")]
pub fn spk_available_info_py(naif_id: i64) -> Vec<(f64, f64, i64, PyFrames, i32)> {
    let singleton = &LOADED_SPK.try_read().unwrap();
    singleton
        .available_info(naif_id)
        .into_iter()
        .map(|(s, e, c, frame, ty)| (s, e, c, frame.into(), ty))
        .collect()
}

/// Return a list of all NAIF IDs currently loaded in the SPK shared memory singleton.
#[pyfunction]
#[pyo3(name = "spk_loaded")]
pub fn spk_loaded_objects_py() -> Vec<i64> {
    let spk = &LOADED_SPK.try_read().unwrap();
    let loaded = spk.loaded_objects(false);
    let mut loaded: Vec<i64> = loaded.into_iter().collect();
    loaded.sort();
    loaded
}

/// Convert a NAIF ID to a string if it is contained within the dictionary of known objects.
/// If the name is not found, this just returns a string the the NAIF ID.
#[pyfunction]
#[pyo3(name = "spk_get_name_from_id")]
pub fn spk_get_name_from_id_py(id: i64) -> String {
    try_name_from_id(id).unwrap_or(id.to_string())
}

/// Reset the contents of the SPK shared memory.
#[pyfunction]
#[pyo3(name = "spk_reset")]
pub fn spk_reset_py() {
    LOADED_SPK.write().unwrap().reset()
}

/// Reload the core SPK files.
#[pyfunction]
#[pyo3(name = "spk_load_core")]
pub fn spk_load_core_py() {
    LOADED_SPK.write().unwrap().load_core().unwrap()
}

/// Reload the cache SPK files.
#[pyfunction]
#[pyo3(name = "spk_load_cache")]
pub fn spk_load_cache_py() {
    LOADED_SPK.write().unwrap().load_cache().unwrap()
}

/// Calculate the state of a given object in the target frame.
///
/// This will automatically replace the name of the object if possible.
///
/// Parameters
/// ----------
/// id : int
///     NAIF ID of the object.
/// jd : float
///     Time (JD) in TDB scaled time.
/// center : int
///     NAIF ID of the associated central point.
/// frame : Frames
///     Frame of reference for the state.
#[pyfunction]
#[pyo3(name = "spk_state")]
pub fn spk_state_py(id: i64, jd: PyTime, center: i64, frame: PyFrames) -> PyResult<PyState> {
    let jd = jd.jd();
    let spk = &LOADED_SPK.try_read().unwrap();
    let mut state = spk.try_get_state(id, jd, center, frame.into())?;
    let _ = state.try_naif_id_to_name();
    Ok(PyState(state))
}

/// Return the raw state of an object as encoded in the SPK Kernels.
///
/// This does not change frames or center points.
///
/// Parameters
/// ----------
/// id : int
///     NAIF ID of the object.
/// jd : float
///     Time (JD) in TDB scaled time.
#[pyfunction]
#[pyo3(name = "spk_raw_state")]
pub fn spk_raw_state_py(id: i64, jd: PyTime) -> PyResult<PyState> {
    let jd = jd.jd();
    let spk = &LOADED_SPK.try_read().unwrap();
    Ok(PyState(spk.try_get_raw_state(id, jd)?))
}
