use crate::{frame::PyFrames, vector::VectorLike};
use itertools::Itertools;
use kete_core::constants::{
    C_V, w1_color_correction, w2_color_correction, w3_color_correction, w4_color_correction,
};
use kete_core::flux::*;
use kete_core::prelude::Error;
use nalgebra::UnitVector3;
use pyo3::{PyResult, pyfunction};

/// Calculate the visible flux at the observer assuming a convex faceted object made up
/// of a collection of lambertian surfaces.
///
/// Parameters
/// ----------
/// facet_flux :
///     Flux from each of the facets in Jy / Steradian per unit freq.
/// facet_normals :
///     List of surface normal vectors.
/// obs2obj :
///     Vector from the observer to the object in AU.
/// diameter :
///     Diameter of the object in km.
/// emissivity :
///     Emissivity of the lambertian surfaces (between 0 and 1, 0.9 is a good default).
#[pyfunction]
#[pyo3(name = "lambertian_flux")]
pub fn lambertian_flux_py(
    facet_flux: Vec<f64>,
    facet_normals: Vec<VectorLike>,
    obs2obj: VectorLike,
    diameter: f64,
    emissivity: f64,
) -> f64 {
    let obs2obj = obs2obj.into_vector(PyFrames::Ecliptic);
    let facet_normals: Vec<_> = facet_normals
        .into_iter()
        .map(|norm| norm.into_vector(PyFrames::Ecliptic))
        .collect();
    let obs2obj_r = obs2obj.norm();
    let obs2obj = UnitVector3::new_normalize(obs2obj.normalize().into());
    facet_normals
        .into_iter()
        .zip(facet_flux)
        .map(|(normal, flux)| {
            let normal = UnitVector3::new_normalize(normal.into());
            lambertian_flux(&normal, &obs2obj, &flux, &obs2obj_r, &diameter, &emissivity)
        })
        .sum()
}

/// Return the Solar flux in Jy / Steradian from the 2000 ASTM Standard Extraterrestrial
/// Spectrum Reference E-490-00:
/// <https://www.nrel.gov/grid/solar-resource/spectra-astm-e490.html>
///
/// Returned values are units Janskys / steradian per unit freq.
///
/// Parameters
/// ----------
/// dist:
///     Distance the object is from the Sun in au.
/// wavelength :
///     Wavelength in nm.
#[pyfunction]
#[pyo3(name = "solar_flux")]
pub fn solar_flux_py(dist: f64, wavelength: f64) -> PyResult<f64> {
    Ok(solar_flux(dist, wavelength).ok_or(Error::ValueError(
        "Query is outside of the range of the dataset".into(),
    ))?)
}

/// Compute the subsolar surface temperature in kelvin.
///
/// Parameters
/// ----------
/// obj2sun :
///     Vector from the object to the sun in AU.
/// geom_albedo :
///     Geometric albedo.
/// g_param :
///     G Phase parameter in the HG system.
/// beaming :
///     The beaming parameter of the object.
/// emissivity :
///     Emissivity of the object, 0.9 by default.
#[pyfunction]
#[pyo3(name = "sub_solar_temperature", signature = (obj2sun, geom_albedo, g_param, beaming, emissivity=0.9))]
pub fn sub_solar_temperature_py(
    obj2sun: VectorLike,
    geom_albedo: f64,
    g_param: f64,
    beaming: f64,
    emissivity: f64,
) -> f64 {
    let obj2sun = obj2sun.into_vector(PyFrames::Ecliptic).into();
    sub_solar_temperature(&obj2sun, geom_albedo, g_param, beaming, emissivity)
}

/// Compute the black body flux at the specified temperatures and wavelength.
///
/// Flux is in units Janskys / steradian per unit freq.
/// This can be converted to being per unit wavelength by multiplying the result of this
/// by `c / wavelength**2`.
///
/// This is limited by the size of float 32s, numbers exceeding 1e-300 will be rounded
/// to 0. This should not raise any warnings, and should be relatively numerically
/// stable and correct over most domains of interest.
///
/// Parameters
/// ----------
/// temp :
///     List of temperatures, in units of Kelvin.
/// wavelength :
///     In units of nanometers.
///
/// Returns
/// -------
/// List :
///     Janskys per steradian
#[pyfunction]
#[pyo3(name = "black_body_flux")]
pub fn black_body_flux_py(temp: f64, wavelength: f64) -> f64 {
    black_body_flux(temp, wavelength)
}

/// Calculate the temperatures of each facet assuming the NEATM model.
///
/// This is a multi-core operation.
///
/// Parameters
/// ----------
/// facet_normals :
///     List of surface normal vectors.
/// subsolar_temp :
///     Temperature at the sub-solar point in kelvin.
/// obj2sun :
///     Vector from the object to the sun in AU.
#[pyfunction]
#[pyo3(name = "neatm_facet_temps")]
pub fn neatm_facet_temperature_py(
    facet_normals: Vec<VectorLike>,
    subsolar_temp: f64,
    obj2sun: VectorLike,
) -> Vec<f64> {
    let obj2sun = UnitVector3::new_normalize(obj2sun.into_vector(PyFrames::Ecliptic).into());
    let facet_normals: Vec<_> = facet_normals
        .into_iter()
        .map(|norm| norm.into_vector(PyFrames::Ecliptic))
        .collect();
    facet_normals
        .into_iter()
        .map(|normal| {
            neatm_facet_temperature(
                &UnitVector3::new_normalize(normal.into()),
                &obj2sun,
                &subsolar_temp,
            )
        })
        .collect_vec()
}

/// Calculate the temperatures of each facet assuming the FRM model.
///
/// This is a multi-core operation.
///
/// Parameters
/// ----------
/// facet_normals :
///     List of surface normal vectors.
/// subsolar_temp :
///     Temperature at the sub-solar point in kelvin.
/// obj2sun :
///     Vector from the object to the sun in AU.
#[pyfunction]
#[pyo3(name = "frm_facet_temps")]
pub fn frm_facet_temperature_py(
    facet_normals: Vec<VectorLike>,
    subsolar_temp: f64,
    obj2sun: VectorLike,
) -> Vec<f64> {
    let obj2sun = UnitVector3::new_normalize(obj2sun.into_vector(PyFrames::Ecliptic).into());
    let facet_normals: Vec<_> = facet_normals
        .into_iter()
        .map(|norm| norm.into_vector(PyFrames::Ecliptic).into())
        .collect();
    facet_normals
        .into_iter()
        .map(|normal| {
            frm_facet_temperature(&UnitVector3::new_normalize(normal), subsolar_temp, &obj2sun)
        })
        .collect_vec()
}

/// Calculate the flux in Janskys from an object using the NEATM thermal model.
///
/// See :doc:`../auto_examples/plot_thermal_model`
///
/// Parameters
/// ----------
/// sun2obj :
///     A vector-like object containing the X/Y/Z coordinates pointing from the sun
///     to the object in units of AU.
/// sun2obs :
///     A vector-like object containing the X/Y/Z coordinates pointing from the sun
///     to the observer in units of AU.
/// v_albedo :
///     The V geometric albedo of the object.
/// g_param :
///     The G parameter of the object
/// beaming :
///     The beaming parameter.
/// diameter :
///     The diameter of the object in km.
/// wavelength :
///     The wavelength of the object in nanometers.
/// emissivity :
///     The emissivity of the object, defaults to 0.9.
///
/// Returns
/// -------
/// float
///     Flux in units of Jy.
#[pyfunction]
#[pyo3(name = "neatm_flux", signature = (sun2obj, sun2obs, v_albedo, g_param, beaming, diameter, wavelength, emissivity=0.9))]
#[allow(clippy::too_many_arguments)]
pub fn neatm_thermal_py(
    sun2obj: VectorLike,
    sun2obs: VectorLike,
    v_albedo: f64,
    g_param: f64,
    beaming: f64,
    diameter: f64,
    wavelength: f64,
    emissivity: f64,
) -> f64 {
    let sun2obj = sun2obj.into_vector(PyFrames::Ecliptic).into();
    let sun2obs = sun2obs.into_vector(PyFrames::Ecliptic).into();

    let hg_params = HGParams::try_new(
        "".into(),
        g_param,
        None,
        Some(C_V),
        Some(v_albedo),
        Some(diameter),
    )
    .unwrap();
    let params = NeatmParams {
        obs_bands: ObserverBands::Generic {
            bands: vec![wavelength; 1],
            zero_mags: None,
            solar_correction: vec![1.0],
        },
        band_albedos: vec![0.0; 1],
        hg_params,
        emissivity,
        beaming,
    };
    params.apparent_thermal_flux(&sun2obj, &sun2obs).unwrap()[0]
}

/// Calculate the flux from an object using the FRM thermal model in Jansky.
///
/// This is a slightly simplified FRM model, where the pole is assumed to be in the
/// ecliptic Z direction.
///
/// See :doc:`../auto_examples/plot_thermal_model`
///
/// Parameters
/// ----------
/// sun2obj : Vector
///     A vector-like object containing the X/Y/Z coordinates pointing from the sun
///     to the object in units of AU.
/// sun2obs : Vector
///     A vector-like object containing the X/Y/Z coordinates pointing from the sun
///     to the observer in units of AU.
/// v_albedo : float
///     The V geometric albedo of the object.
/// g_param : float
///     The G parameter of the object
/// diameter : float
///     The diameter of the object in km.
/// wavelength : float
///     The wavelength of the object in nanometers.
/// emissivity : float
///     The emissivity of the object, defaults to 0.9.
///
/// Returns
/// -------
/// float
///     Flux in units of Jy.
#[pyfunction]
#[pyo3(name = "frm_flux", signature = (sun2obj, sun2obs, v_albedo, g_param, diameter, wavelength, emissivity=0.9))]
#[allow(clippy::too_many_arguments)]
pub fn frm_thermal_py(
    sun2obj: VectorLike,
    sun2obs: VectorLike,
    v_albedo: f64,
    g_param: f64,
    diameter: f64,
    wavelength: f64,
    emissivity: f64,
) -> f64 {
    let sun2obj = sun2obj.into_vector(PyFrames::Ecliptic).into();
    let sun2obs = sun2obs.into_vector(PyFrames::Ecliptic).into();
    let hg_params = HGParams::try_new(
        "".into(),
        g_param,
        None,
        Some(C_V),
        Some(v_albedo),
        Some(diameter),
    )
    .unwrap();

    let params = FrmParams {
        obs_bands: ObserverBands::Generic {
            bands: vec![wavelength; 1],
            zero_mags: None,
            solar_correction: vec![1.0],
        },
        band_albedos: vec![0.0; 1],
        hg_params,
        emissivity,
    };
    params.apparent_thermal_flux(&sun2obj, &sun2obs).unwrap()[0]
}

/// Given the M1/K1 and M2/K2 values, compute the apparent Comet visible magnitudes.
///
/// The model for apparent magnitudes are:
///
/// m1 + k1 * log10(sun2obj.r) + 5.0 * log10(obj2obs.r) + phase_mag_slope_1 * phase
/// m2 + k2 * log10(sun2obj.r) + 5.0 * log10(obj2obs.r) + phase_mag_slope_2 * phase
///
/// Where m1/k1 are related to total magnitudes and m2/k2 are nucleus magnitudes.
///
/// This model is based off of these:
/// https://ssd.jpl.nasa.gov/horizons/manual.html#obsquan  (see section 9)
/// https://en.wikipedia.org/wiki/Absolute_magnitude#Cometary_magnitudes
///
/// Note that the above model does not include a 2.5x term attached to the K1/2 terms
/// which are present in the wikipedia definition, this matches the definitions used by
/// JPL Horizons.
///
/// Note that this includes seperate phase magnitude slope for both the nucleus and
/// total fluxes.
///
/// This does a best effort to compute both magnitudes, if any values are missing this
/// will return None in the respective calculation.
///
/// Parameters
/// ----------
/// sun2obj :
///     A vector-like object containing the X/Y/Z coordinates pointing to the object
///     from the sun in units of AU.
/// sun2obs :
///     A vector-like object containing the X/Y/Z coordinates pointing from the sun
///     to the observer in units of AU.
/// mk_1 :
///     Tuple of (M_1, K_1), where M_1 is the total Absolute magnitude of the comet,
///     and K_1 magnitude slope as a function of heliocentric distance.
/// mk_2 :
///     Tuple of (M_2, K_2), where M_2 is the total Absolute magnitude of the nucleus,
///     and K_2 nucleus magnitude slope as a function of heliocentric distance.
/// phase_corr :
///     Magnitude variation of the comet as a function of observing phase, units are
///     Mag/Deg of phase, this defaults to 0.0 for the total magnitude, and 0.035
///     Mag/Deg for the nucleus.
///
/// Returns
/// -------
/// float
///     (Total apparent magnitude, Magnitude of the nucleus)
#[pyfunction]
#[pyo3(name = "comet_apparent_mags", signature = (sun2obj, sun2obs, mk_1=None, mk_2=None, phase_corr=[0.0, 0.035]))]
pub fn comet_mags_py(
    sun2obj: VectorLike,
    sun2obs: VectorLike,
    mk_1: Option<[f64; 2]>,
    mk_2: Option<[f64; 2]>,
    phase_corr: [f64; 2],
) -> (Option<f64>, Option<f64>) {
    let sun2obj = sun2obj.into_vector(PyFrames::Ecliptic).into();
    let sun2obs = sun2obs.into_vector(PyFrames::Ecliptic).into();
    let mk_params = CometMKParams::new("".into(), mk_1, mk_2, phase_corr);
    (
        mk_params.apparent_total_mag(&sun2obs, &sun2obj),
        mk_params.apparent_nuclear_mag(&sun2obs, &sun2obj),
    )
}

/// Calculate the W1 black body color correction for the given temperature in kelvin.
#[pyfunction]
#[pyo3(name = "w1_color_correction")]
pub fn w1_color_correction_py(temp: f64) -> f64 {
    w1_color_correction(temp)
}

/// Calculate the W2 black body color correction for the given temperature in kelvin.
#[pyfunction]
#[pyo3(name = "w2_color_correction")]
pub fn w2_color_correction_py(temp: f64) -> f64 {
    w2_color_correction(temp)
}

/// Calculate the W3 black body color correction for the given temperature in kelvin.
#[pyfunction]
#[pyo3(name = "w3_color_correction")]
pub fn w3_color_correction_py(temp: f64) -> f64 {
    w3_color_correction(temp)
}

/// Calculate the W4 black body color correction for the given temperature in kelvin.
#[pyfunction]
#[pyo3(name = "w4_color_correction")]
pub fn w4_color_correction_py(temp: f64) -> f64 {
    w4_color_correction(temp)
}

/// The normal vectors of the fib lattice
#[pyfunction]
#[pyo3(name = "fib_lattice_vecs")]
pub fn fib_lattice_vecs_py(n_facets: usize) -> Vec<[f64; 3]> {
    let facets = ConvexShape::new_fibonacci_lattice(n_facets).facets;
    facets
        .iter()
        .map(|f| f.normal.into_inner().into())
        .collect()
}
