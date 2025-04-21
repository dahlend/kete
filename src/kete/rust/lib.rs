//! Core kete library code, which are wrappers over the kete_core rust package.
//! Primarily enables python interfaces

#![deny(
    bad_style,
    dead_code,
    improper_ctypes,
    non_shorthand_field_patterns,
    no_mangle_generic_items,
    overflowing_literals,
    path_statements,
    patterns_in_fns_without_body,
    unconditional_recursion,
    unused,
    while_true,
    missing_debug_implementations,
    missing_docs,
    trivial_casts,
    trivial_numeric_casts,
    unused_extern_crates,
    unused_import_braces,
    unused_qualifications,
    unused_results
)]

use pyo3::prelude::*;
use state::PyState;

pub mod covariance;
pub mod elements;
pub mod fitting;
pub mod flux;
pub mod fovs;
pub mod frame;
pub mod horizons;
pub mod kepler;
pub mod nongrav;
pub mod propagation;
pub mod simult_states;
pub mod spice;
pub mod state;
pub mod state_transition;
pub mod time;
pub mod vector;

// Due to the nature of this sort of interface, there is quite a bit of boiler-plate
// code which is difficult to avoid.
// There are many structs in this interface which are just thin wrappers over
// the underlying Rust objects. Examples include [`state::PyState`] which is a thin
// wrapper over [`kete_core::state::State`]. These allow rust objects to exist in
// the python while still being reasonably fast. These wrappers also frequently expose
// the binary save/loading features written into the kete_core code. Typically these
// wrappers are the same name as the underlying struct, with 'Py' appended on the
// front.

/// Python module which exposes all of the compiled rust functions.
#[pymodule]
fn _core(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<frame::PyFrames>()?;
    m.add_class::<PyState>()?;
    m.add_class::<vector::Vector>()?;
    m.add_class::<elements::PyCometElements>()?;
    m.add_class::<simult_states::PySimultaneousStates>()?;
    m.add_class::<nongrav::PyNonGravModel>()?;
    m.add_class::<time::PyTime>()?;

    m.add_class::<fovs::PyNeosCmos>()?;
    m.add_class::<fovs::PyNeosVisit>()?;
    m.add_class::<fovs::PyWiseCmos>()?;
    m.add_class::<fovs::PyZtfCcdQuad>()?;
    m.add_class::<fovs::PyZtfField>()?;
    m.add_class::<fovs::PyGenericRectangle>()?;
    m.add_class::<fovs::PyGenericCone>()?;
    m.add_class::<fovs::PyOmniDirectional>()?;
    m.add_class::<fovs::FOVList>()?;

    m.add_class::<flux::PyNeatmParams>()?;
    m.add_class::<flux::PyFrmParams>()?;
    m.add_class::<flux::PyModelResults>()?;

    m.add_class::<horizons::HorizonsProperties>()?;

    m.add_class::<covariance::Covariance>()?;

    m.add_function(wrap_pyfunction!(frame::wgs_lat_lon_to_ecef, m)?)?;
    m.add_function(wrap_pyfunction!(frame::ecef_to_wgs_lat_lon, m)?)?;
    m.add_function(wrap_pyfunction!(frame::calc_obliquity_py, m)?)?;
    m.add_function(wrap_pyfunction!(frame::calc_earth_precession, m)?)?;
    m.add_function(wrap_pyfunction!(frame::geodetic_lat_to_geocentric_py, m)?)?;

    m.add_function(wrap_pyfunction!(kepler::compute_eccentric_anomaly_py, m)?)?;
    m.add_function(wrap_pyfunction!(kepler::propagation_kepler_py, m)?)?;

    m.add_function(wrap_pyfunction!(propagation::propagation_n_body_spk_py, m)?)?;
    m.add_function(wrap_pyfunction!(propagation::propagation_n_body_py, m)?)?;
    m.add_function(wrap_pyfunction!(propagation::moid_py, m)?)?;

    m.add_function(wrap_pyfunction!(fovs::fov_checks_py, m)?)?;
    m.add_function(wrap_pyfunction!(fovs::fov_spk_checks_py, m)?)?;
    m.add_function(wrap_pyfunction!(fovs::fov_static_checks_py, m)?)?;

    m.add_function(wrap_pyfunction!(flux::hg_apparent_flux_py, m)?)?;
    m.add_function(wrap_pyfunction!(flux::hg_apparent_mag_py, m)?)?;
    m.add_function(wrap_pyfunction!(flux::hg_phase_curve_correction_py, m)?)?;
    m.add_function(wrap_pyfunction!(flux::sub_solar_temperature_py, m)?)?;
    m.add_function(wrap_pyfunction!(flux::black_body_flux_py, m)?)?;
    m.add_function(wrap_pyfunction!(flux::neatm_thermal_py, m)?)?;
    m.add_function(wrap_pyfunction!(flux::frm_thermal_py, m)?)?;
    m.add_function(wrap_pyfunction!(flux::neatm_facet_temperature_py, m)?)?;
    m.add_function(wrap_pyfunction!(flux::frm_facet_temperature_py, m)?)?;
    m.add_function(wrap_pyfunction!(flux::lambertian_flux_py, m)?)?;
    m.add_function(wrap_pyfunction!(flux::w1_color_correction_py, m)?)?;
    m.add_function(wrap_pyfunction!(flux::w2_color_correction_py, m)?)?;
    m.add_function(wrap_pyfunction!(flux::w3_color_correction_py, m)?)?;
    m.add_function(wrap_pyfunction!(flux::w4_color_correction_py, m)?)?;
    m.add_function(wrap_pyfunction!(flux::comet_mags_py, m)?)?;
    m.add_function(wrap_pyfunction!(flux::fib_lattice_vecs_py, m)?)?;
    m.add_function(wrap_pyfunction!(flux::solar_flux_py, m)?)?;

    m.add_function(wrap_pyfunction!(spice::spk_load_py, m)?)?;
    m.add_function(wrap_pyfunction!(spice::spk_loaded_objects_py, m)?)?;
    m.add_function(wrap_pyfunction!(spice::spk_state_py, m)?)?;
    m.add_function(wrap_pyfunction!(spice::spk_raw_state_py, m)?)?;
    m.add_function(wrap_pyfunction!(spice::spk_reset_py, m)?)?;
    m.add_function(wrap_pyfunction!(spice::spk_get_name_from_id_py, m)?)?;
    m.add_function(wrap_pyfunction!(spice::spk_available_info_py, m)?)?;
    m.add_function(wrap_pyfunction!(spice::spk_load_cache_py, m)?)?;
    m.add_function(wrap_pyfunction!(spice::spk_load_core_py, m)?)?;

    m.add_function(wrap_pyfunction!(spice::pck_reset_py, m)?)?;
    m.add_function(wrap_pyfunction!(spice::pck_load_py, m)?)?;
    m.add_function(wrap_pyfunction!(spice::pck_earth_frame_py, m)?)?;
    m.add_function(wrap_pyfunction!(spice::pck_state_to_earth, m)?)?;
    m.add_function(wrap_pyfunction!(spice::pck_loaded_objects_py, m)?)?;

    m.add_function(wrap_pyfunction!(spice::daf_header_info_py, m)?)?;
    m.add_function(wrap_pyfunction!(spice::obs_codes, m)?)?;

    m.add_function(wrap_pyfunction!(state_transition::compute_stm_py, m)?)?;

    m.add_function(wrap_pyfunction!(fitting::ks_test_py, m)?)?;
    m.add_function(wrap_pyfunction!(fitting::fit_chi2_py, m)?)?;

    m.add_function(wrap_pyfunction!(kete_core::cache::cache_path, m)?)?;

    Ok(())
}
