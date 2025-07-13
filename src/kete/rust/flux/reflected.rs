use crate::{frame::PyFrames, vector::VectorLike};
use kete_core::constants;
use kete_core::flux::HGParams;
use kete_core::flux::cometary_dust_phase_curve_correction;
use kete_core::flux::hg_phase_curve_correction;
use pyo3::pyfunction;

/// This computes the phase curve correction in the IAU format.
///
/// Parameters
/// ----------
/// g_param :
///     The G parameter of the object.
/// phase_angle :
///     The angular separation between the sun and observer as viewed from the object,
///     units of degrees.
///
/// Returns
/// -------
/// float
///     The phase curve correction.
#[pyfunction]
#[pyo3(name = "hg_phase_curve_correction")]
pub fn hg_phase_curve_correction_py(g_param: f64, phase_angle: f64) -> f64 {
    hg_phase_curve_correction(g_param, phase_angle.to_radians())
}

/// Phase correction curve for cometary dust from the following paper:
///
/// A composite phase function for cometary dust comae
/// Bertini, Ivano, et al.
/// Planetary and Space Science (2025): 106164.
///
/// This uses the fitted values from the paper for `k=0.80`, `g_f=0.944`, `g_b=-0.542`.
///
/// An additional normalization has been applied so that the value of this is 1.0 at
/// 0.0 phase angle.
///
/// Parameters
/// ----------
/// phase_angle :
///     The angular separation between the sun and observer as viewed from the dust,
///     units of degrees.
///
/// Returns
/// -------
/// float
///     The phase curve correction.
#[pyfunction]
#[pyo3(name = "comet_dust_phase_curve_correction")]
pub fn comet_dust_phase_curve_correction_py(phase_angle: f64) -> f64 {
    cometary_dust_phase_curve_correction(phase_angle.to_radians())
}

/// Calculate the reflected flux from an object using the HG asteroid model.
///
/// This assumes that the object is an ideal disk facing the sun and applies the IAU
/// correction curve to the reflected light, returning units of Jy per unit frequency.
///
/// This treats the sun as a flat black body disk, which is a good approximation as long
/// as the object is several solar radii away.
///
/// Either `h_mag` or `diameter` must be provided, but only 1 is strictly required.
/// The other will be computed if not provided.
///
/// Parameters
/// ----------
/// sun2obj :
///     A vector-like object containing the X/Y/Z coordinates pointing from the sun
///     to the object in units of AU.
/// sun2obs :
///     A vector-like object containing the X/Y/Z coordinates pointing from the sun
///     to the observer in units of AU.
/// g_param :
///     The G parameter of the object.
/// wavelength :
///     The wavelength of the object in nanometers.
/// v_albedo :
///     The geometric albedo of the object at the specified wavelength.
/// h_mag :
///     H magnitude of the object in V band.
/// diameter :
///     The diameter of the object in km.
/// c_hg :
///     The relationship constant between H, D, and pV for the bandpass. Defaults to
///     `1329.0`.
///
/// Returns
/// -------
/// float
///     Flux in Jy per unit frequency.
#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(name = "hg_apparent_flux", signature = (sun2obj, sun2obs, g_param, wavelength, v_albedo,
    h_mag=None, diameter=None, c_hg=None))]
pub fn hg_apparent_flux_py(
    sun2obj: VectorLike,
    sun2obs: VectorLike,
    g_param: f64,
    wavelength: f64,
    v_albedo: f64,
    h_mag: Option<f64>,
    diameter: Option<f64>,
    c_hg: Option<f64>,
) -> f64 {
    let c_hg = c_hg.unwrap_or(constants::C_V);
    let sun2obj = sun2obj.into_vector(PyFrames::Equatorial).into();
    let sun2obs = sun2obs.into_vector(PyFrames::Equatorial).into();
    let params = HGParams::try_new(
        "".into(),
        g_param,
        h_mag,
        Some(c_hg),
        Some(v_albedo),
        diameter,
    )
    .unwrap();
    params
        .apparent_flux(&sun2obj, &sun2obs, wavelength, v_albedo)
        .unwrap()
}

/// Compute the apparent magnitude of an object using the absolute magnitude H, G, and
/// positional information.
///
/// The HG IAU model is not technically defined above 120 degrees phase, however this
/// will continue to return values fit to the model until 160 degrees. Phases larger
/// than 160 degrees will return nan.
///
/// Parameters
/// ----------
/// sun2obj :
///     A vector-like object containing the X/Y/Z coordinates pointing from the sun
///     to the object in units of AU.
/// sun2obs :
///     A vector-like object containing the X/Y/Z coordinates pointing from the sun
///     to the observer in units of AU.
/// G :
///     The G parameter of the object.
/// H :
///     The absolute magnitude of the object.
///
/// Returns
/// -------
/// float
///     The apparent magnitude of the object.
///
#[pyfunction]
#[pyo3(name = "hg_apparent_mag")]
pub fn hg_apparent_mag_py(
    sun2obj: VectorLike,
    sun2obs: VectorLike,
    h_mag: f64,
    g_param: f64,
) -> f64 {
    let sun2obj = sun2obj.into_vector(PyFrames::Equatorial).into();
    let sun2obs = sun2obs.into_vector(PyFrames::Equatorial).into();
    let params = HGParams::new("".into(), g_param, h_mag, None);
    params.apparent_mag(&sun2obj, &sun2obs)
}
