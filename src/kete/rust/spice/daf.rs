use kete_spice::daf::convert_daf_big_to_little_endian;
use kete_spice::prelude::DafFile;
use pyo3::{PyResult, pyfunction};

/// Given a DAF file, return the comments contained within the header.
#[pyfunction]
#[pyo3(name = "daf_header_comments")]
pub fn daf_header_info_py(filename: &str) -> PyResult<String> {
    let mut file = std::fs::File::open(filename)?;
    let daf = DafFile::from_buffer(&mut file)?;
    Ok(daf.comments)
}

/// Convert a big-endian DAF/SPK file to a little-endian copy.
///
/// Some old SPK files from NAIF, for example Spitzer, are big-endian. kete
/// reads these files and converts them in memory on each load. This function
/// byte-swaps all numeric values of the input file. It writes the
/// little-endian copy to ``output_filename`` once.
///
/// Parameters
/// ----------
/// input_filename : str
///   Path to the big-endian DAF/SPK file.
/// output_filename : str
///   Path to write the converted little-endian file.
///
/// Raises
/// ------
/// ValueError
///   If the input file cannot be read, is not a valid big-endian DAF file, or
///   if the output file cannot be written.
#[pyfunction]
#[pyo3(name = "daf_convert_big_to_little_endian")]
pub fn daf_convert_be_py(input_filename: &str, output_filename: &str) -> PyResult<()> {
    convert_daf_big_to_little_endian(input_filename, output_filename)?;
    Ok(())
}
