use kete_core::spice::LOADED_SCLK;
use pyo3::{pyfunction, PyResult};

use crate::time::PyTime;

/// Load all specified spice clock kernels into the SCLK shared memory singleton.
#[pyfunction]
#[pyo3(name = "sclk_load")]
pub fn sclk_load_py(filenames: Vec<String>) -> PyResult<()> {
    let mut singleton = LOADED_SCLK.write().unwrap();
    for filename in filenames.iter() {
        let load = (*singleton).load_file(filename);
        if let Err(err) = load {
            eprintln!("{} failed to load. {}", filename, err);
        }
    }
    Ok(())
}

/// Convert a spacecraft clock string into a `PyTime` object.
/// This function requires that the SCLK kernels for the spacecraft have been loaded
/// into the SCLK shared memory singleton.
/// The `naif_id` is the NAIF ID of the spacecraft, and `sc_clock` is the spacecraft clock
/// string.
///
/// This conversion matches the cSPICE implementation to within a ~5 milliseconds, mostly
/// due to the fact the Kete treats TT and TDB as equivalent.
#[pyfunction]
#[pyo3(name = "sclk_time_from_string")]
pub fn sclk_time_from_str_py(naif_id: i32, sc_clock: String) -> PyResult<PyTime> {
    let singleton = LOADED_SCLK.read().unwrap();
    let time = singleton.try_get_time(naif_id, &sc_clock)?;
    Ok(time.into())
}

/// Reset the contents of the SCLK shared memory to the default set of SCLK kernels.
#[pyfunction]
#[pyo3(name = "sclk_reset")]
pub fn sclk_reset_py() {
    LOADED_SCLK.write().unwrap().reset()
}

/// Return a list of all loaded objects in the SCLK singleton.
/// This is a list of the center NAIF IDs of the segments.
#[pyfunction]
#[pyo3(name = "sclk_loaded")]
pub fn sclk_loaded_objects_py() -> Vec<i32> {
    let loaded = LOADED_SCLK.read().unwrap();
    loaded.loaded_objects()
}
