//! Python support for reading SPICE kernels
mod ck;
mod daf;
mod pck;
mod sclk;
mod spk;

pub use ck::*;
pub use daf::*;
use kete_core::spice::try_obs_code_from_name;
pub use pck::*;
pub use sclk::*;
use sgp4::Constants;
pub use spk::*;

use pyo3::{PyResult, pyfunction};

use crate::time::PyTime;

/// Return a list of MPC observatory codes, along with the latitude, longitude (deg),
/// altitude (m above the WGS84 surface), and name.
#[pyfunction]
#[pyo3(name = "observatory_codes")]
pub fn obs_codes() -> Vec<(f64, f64, f64, String, String)> {
    let mut codes = Vec::new();
    for row in kete_core::spice::OBS_CODES.iter() {
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
        return Err(pyo3::exceptions::PyValueError::new_err(
            "No observatory codes found for the given name.",
        ));
    } else if obs_codes.len() > 1 {
        // if there is an exact match, return that one
        if let Some(exact_match) = obs_codes.iter().find(|obs| obs.name == name) {
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
            "Multiple observatory codes found for the given name:\n{possible_matches}",
        )));
    }
    let obs_code = obs_codes[0].clone();

    let lat = (obs_code.lat * 1e8).round() / 1e8 + 0.0;
    let lon = (obs_code.lon * 1e8).round() / 1e8 + 0.0;
    let altitude = (obs_code.altitude * 1e5).round() / 1e5 + 0.0;

    Ok((lat, lon, altitude, obs_code.name, obs_code.code.to_string()))
}

/// Predict the state of an object in earths orbit from the two line elements
#[pyfunction]
pub fn predict_tle(line1: String, line2: String, time: PyTime) -> PyResult<([f64; 3], [f64; 3])> {
    let elements =
        sgp4::Elements::from_tle(None, line1.as_bytes(), line2.as_bytes()).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Invalid TLE format: {}", e))
        })?;

    let orbit = sgp4::Orbit::from_kozai_elements(
        &sgp4::WGS84,
        elements.inclination * (core::f64::consts::PI / 180.0),
        elements.right_ascension * (core::f64::consts::PI / 180.0),
        elements.eccentricity,
        elements.argument_of_perigee * (core::f64::consts::PI / 180.0),
        elements.mean_anomaly * (core::f64::consts::PI / 180.0),
        elements.mean_motion * (core::f64::consts::PI / 720.0),
    )
    .map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!("Failed to create orbit from TLE: {}", e))
    })?;

    let constants = Constants::new(
        sgp4::WGS84,
        sgp4::iau_epoch_to_sidereal_time,
        elements.epoch(),
        elements.drag_term,
        orbit,
    )
    .map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!(
            "Failed to initialize SGP4 constants: {}",
            e
        ))
    })?;

    let time = time.0.utc();

    let naive_time = time
        .to_datetime()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("Failed to convert time to datetime"))?
        .naive_utc();

    let min_diff = elements
        .datetime_to_minutes_since_epoch(&naive_time)
        .map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Failed to convert time: {}", e))
        })?;

    let state = constants.propagate(min_diff).map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!("Failed to propagate TLE: {}", e))
    })?;

    Ok((state.position, state.velocity))
}
