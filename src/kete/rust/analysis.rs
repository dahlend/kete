//! Python bindings for orbital analysis functions.
use kete_core::analysis;
use pyo3::{PyResult, pyfunction};

use crate::state::PyState;

/// Compute the Hill radius of a body orbiting a more massive central body.
///
/// This is an internal function, use :func:`kete.hill_radius` for the
/// user-facing interface which accepts body names.
#[pyfunction]
#[pyo3(name = "hill_radius")]
pub fn hill_radius_py(semi_major: f64, eccentricity: f64, gm_body: f64, gm_central: f64) -> f64 {
    analysis::hill_radius(semi_major, eccentricity, gm_body, gm_central)
}

/// Compute the Laplace sphere-of-influence radius.
///
/// This is an internal function, use :func:`kete.sphere_of_influence` for the
/// user-facing interface which accepts body names.
#[pyfunction]
#[pyo3(name = "sphere_of_influence")]
pub fn sphere_of_influence_py(semi_major: f64, gm_body: f64, gm_central: f64) -> f64 {
    analysis::sphere_of_influence(semi_major, gm_body, gm_central)
}

/// Compute the specific orbital energy of a state.
///
/// .. math::
///
///     \mathcal{E} = \frac{v^2}{2} - \frac{\mu}{r}
///
/// Negative values indicate a bound orbit, positive values a hyperbolic orbit.
/// Units are AU^2/day^2.
///
/// Parameters
/// ----------
/// state :
///     Orbital state of the object.
///
/// Returns
/// -------
/// float
///     Specific orbital energy in AU^2/day^2.
#[pyfunction]
#[pyo3(name = "specific_energy")]
pub fn specific_energy_py(state: PyState) -> PyResult<f64> {
    Ok(analysis::specific_energy(&state.raw)?)
}

/// Compute B-plane parameters from a planetocentric hyperbolic state.
///
/// The state must be centered on the target body and on a hyperbolic orbit
/// (positive specific energy).
///
/// Parameters
/// ----------
/// state :
///     Planetocentric state of the incoming body.
///
/// Returns
/// -------
/// BPlane
///     B-plane encounter geometry.
///
/// Raises
/// ------
/// ValueError
///     If the orbit is bound (negative energy).
#[pyfunction]
#[pyo3(name = "compute_b_plane")]
pub fn compute_b_plane_py(state: PyState) -> PyResult<analysis::BPlane> {
    Ok(analysis::compute_b_plane(&state.raw)?)
}
