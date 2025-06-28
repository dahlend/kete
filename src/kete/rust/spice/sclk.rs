use kete_core::spice::LOADED_SCLK;
use pyo3::{PyResult, pyfunction};

use crate::time::PyTime;

/// Load all specified spice clock kernels into the SCLK shared memory singleton.
#[pyfunction]
#[pyo3(name = "sclk_load")]
pub fn sclk_load_py(filenames: Vec<String>) -> PyResult<()> {
    let mut singleton = LOADED_SCLK.write().unwrap();
    for filename in filenames.iter() {
        let load = (*singleton).load_file(filename);
        if let Err(err) = load {
            eprintln!("{filename} failed to load. {err}");
        }
    }
    Ok(())
}

/// Convert a spacecraft clock string into a :py:class:`kete.Time` object.
/// This function requires that the SCLK kernels for the spacecraft have been loaded
/// into the SCLK shared memory singleton.
///
/// Parameters
/// ----------
/// naif_id: int
///     The NAIF ID of the spacecraft.
/// sc_clock: str
///    The spacecraft clock string to convert into a time.
///
#[pyfunction]
#[pyo3(name = "sclk_time_from_string")]
pub fn sclk_str_to_time_py(naif_id: i32, sc_clock: String) -> PyResult<PyTime> {
    let singleton = LOADED_SCLK.read().unwrap();
    let time = singleton.string_get_time(naif_id, &sc_clock)?;
    Ok(time.into())
}

/// Convert a spacecraft clock tick (SCLK float) into a time.
/// This function requires that the SCLK kernels for the spacecraft have been loaded
/// into the SCLK shared memory singleton.
///
/// Parameters
/// ----------
/// naif_id: int
///     The NAIF ID of the spacecraft.
/// sc_tick: f64
///     The spacecraft clock tick to convert into a time.
#[pyfunction]
#[pyo3(name = "sclk_tick_to_time")]
pub fn sclk_tick_to_time_py(naif_id: i32, sc_tick: f64) -> PyResult<PyTime> {
    let singleton = LOADED_SCLK.read().unwrap();
    let time = singleton.try_tick_to_time(naif_id, sc_tick)?;
    Ok(time.into())
}

/// Convert a time into a clock tick (SCLK float).
/// This function requires that the SCLK kernels for the spacecraft have been loaded
/// into the SCLK shared memory singleton.
///
/// Parameters
/// ----------
/// naif_id: int
///    The NAIF ID of the spacecraft.
/// time: :py:class:`kete.Time`
///   The time to convert into a clock tick.
///
#[pyfunction]
#[pyo3(name = "sclk_time_to_tick")]
pub fn sclk_time_to_tick_py(naif_id: i32, time: PyTime) -> PyResult<f64> {
    let singleton = LOADED_SCLK.read().unwrap();
    Ok(singleton.try_time_to_tick(naif_id, time.into())?)
}

/// Reset the contents of the SCLK shared memory to the default set of SCLK kernels.
#[pyfunction]
#[pyo3(name = "sclk_reset")]
pub fn sclk_reset_py() {
    LOADED_SCLK.write().unwrap().reset()
}

/// Return a list of all loaded objects in the SCLK singleton.
/// This is a list of the NAIF IDs of the clocks.
#[pyfunction]
#[pyo3(name = "sclk_loaded")]
pub fn sclk_loaded_objects_py() -> Vec<i32> {
    let loaded = LOADED_SCLK.read().unwrap();
    loaded.loaded_objects()
}
