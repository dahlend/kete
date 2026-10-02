//! Python support for reading SPICE kernels
mod ck;
mod daf;
mod frames;
mod instruments;
mod pck;
mod spk;

pub use ck::*;
pub use daf::*;
pub use frames::*;
pub use instruments::*;
use kete_core::desigs::{OBS_CODES, try_obs_code_from_name};
pub use pck::*;
pub use spk::*;

use pyo3::{PyResult, pyfunction};

/// Load kernel files of any supported type into their singletons.
///
/// The type of each file comes from its first 8 bytes: binary SPK, PCK, and
/// CK files, and text SCLK files. All headers are read before any file loads,
/// so an unsupported file loads nothing. A file of a supported type that fails
/// to load prints a message and is skipped.
///
/// Parameters
/// ----------
/// filenames : list of str
///   Paths of the kernel files.
///
/// Raises
/// ------
/// ValueError
///   If a file cannot be read, or does not have the header of a supported
///   kernel type.
#[pyfunction]
#[pyo3(name = "kernel_load")]
pub fn kernel_load_py(filenames: Vec<String>) -> PyResult<()> {
    Ok(kete_spice::load_kernels(&filenames)?)
}

/// Return a list of MPC observatory codes, along with the latitude, longitude (deg),
/// altitude (km above the WGS84 ellipsoid), and name.
#[pyfunction]
#[pyo3(name = "observatory_codes")]
pub fn obs_codes() -> Vec<(f64, f64, f64, String, String)> {
    let mut codes = Vec::new();
    for row in OBS_CODES.iter() {
        codes.push((
            row.lat,
            row.lon,
            row.altitude,
            row.name.clone(),
            row.code.to_string(),
        ))
    }
    codes
}

/// Search known observatory codes, if a single matching observatory is found, this
/// will return the [lat, lon, altitude, description, obs code] in degrees and km as
/// appropriate.
///
/// >>> kete.mpc.find_obs_code("Palomar Mountain")
/// (33.35411714, -116.86254, 1.69606, 'Palomar Mountain', '675')
///
/// Parameters
/// ----------
/// name :
///     Name of the observatory, this can be a partial name, or obs code.
#[pyfunction]
#[pyo3(name = "_find_obs_code")]
pub fn find_obs_code_py(name: &str) -> PyResult<(f64, f64, f64, String, String)> {
    let obs_codes = try_obs_code_from_name(name);

    if obs_codes.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "No observatory codes found for the given name ({name})."
        )));
    } else if obs_codes.len() > 1 {
        // if there is an exact match on name or code, return that one
        if let Some(exact_match) = obs_codes
            .iter()
            .find(|obs| obs.name == name || obs.code.to_string() == name)
        {
            return Ok((
                (exact_match.lat * 1e8).round() / 1e8 + 0.0,
                (exact_match.lon * 1e8).round() / 1e8 + 0.0,
                (exact_match.altitude * 1e5).round() / 1e5 + 0.0,
                exact_match.name.clone(),
                exact_match.code.to_string(),
            ));
        }

        let possible_matches = obs_codes
            .iter()
            .map(|obs| format!("{} - {}", obs.name.clone(), obs.code))
            .collect::<Vec<_>>()
            .join(",\n");
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Multiple observatory codes found for the given name ({name}):\n{possible_matches}",
        )));
    }
    let obs_code = obs_codes[0].clone();

    let lat = (obs_code.lat * 1e8).round() / 1e8 + 0.0;
    let lon = (obs_code.lon * 1e8).round() / 1e8 + 0.0;
    let altitude = (obs_code.altitude * 1e5).round() / 1e5 + 0.0;

    Ok((lat, lon, altitude, obs_code.name, obs_code.code.to_string()))
}
