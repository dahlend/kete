use kete_core::spice::DafFile;
use pyo3::{PyResult, pyfunction};

/// Given a DAF file, return the comments contained within the header.
#[pyfunction]
#[pyo3(name = "daf_header_comments")]
pub fn daf_header_info_py(filename: &str) -> PyResult<String> {
    let mut file = std::fs::File::open(filename)?;
    let daf = DafFile::from_buffer(&mut file)?;
    Ok(daf.comments)
}
