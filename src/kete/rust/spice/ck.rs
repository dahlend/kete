use kete_core::spice::LOADED_CK;
use pyo3::{pyfunction, PyResult};

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
