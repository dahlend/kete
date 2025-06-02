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

/// Get the closest record to the given JD for the specified instrument ID.
#[pyfunction]
#[pyo3(name = "ck_get_record")]
pub fn ck_get_record_py(time: f64, inst_id: i32) -> PyResult<(f64, [f64; 4], Option<[f64; 3]>)> {
    let singleton = LOADED_CK.read().unwrap();
    let (time, quat, accel) = singleton.get_record_at_time(time, inst_id)?;
    let time = time.jd;
    Ok((time, quat, accel))
}
