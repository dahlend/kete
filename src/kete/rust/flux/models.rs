//! Model inputs and outputs for NEATM/FRM/Reflected.

use kete_flux::{BandInfo, frm_total_flux, neatm_total_flux, resolve_hg_params};
use pyo3::prelude::*;

use crate::{frame::PyFrames, vector::VectorLike};

/// Reflected/Thermal model results.
///
/// Parameters
/// ----------
/// fluxes :
///     Total fluxes per band in units of Jy / Steradian.
/// thermal_fluxes :
///     Black body specific fluxes per band in units of Jy / Steradian.
/// hg_fluxes :
///     Reflected light specific fluxes per band in units of Jy / Steradian.
/// v_band_magnitude :
///     Expected magnitude in the V-band using the HG model.
/// v_band_flux :
///     Expected flux in the V-band using the HG model.
/// magnitudes :
///     Magnitudes in the different bands if zero mags were available.
#[pyclass(frozen, module = "kete.flux", name = "ModelResults", from_py_object)]
#[derive(Clone, Debug)]
pub struct PyModelResults(pub kete_flux::ModelResults);

impl From<kete_flux::ModelResults> for PyModelResults {
    fn from(value: kete_flux::ModelResults) -> Self {
        Self(value)
    }
}

#[pymethods]
impl PyModelResults {
    #[new]
    #[pyo3(signature = (fluxes, thermal_fluxes, hg_fluxes, v_band_magnitude, v_band_flux, magnitudes=None))]
    #[allow(clippy::too_many_arguments, missing_docs)]
    pub fn new(
        fluxes: Vec<f64>,
        thermal_fluxes: Vec<f64>,
        hg_fluxes: Vec<f64>,
        v_band_magnitude: f64,
        v_band_flux: f64,
        magnitudes: Option<Vec<f64>>,
    ) -> Self {
        let magnitudes = magnitudes.unwrap_or(vec![f64::NAN; fluxes.len()]);
        kete_flux::ModelResults {
            fluxes,
            magnitudes,
            thermal_fluxes,
            hg_fluxes,
            v_band_magnitude,
            v_band_flux,
        }
        .into()
    }

    /// Total fluxes per band in units of Jy / Steradian.
    #[getter]
    pub fn fluxes(&self) -> Vec<f64> {
        self.0.fluxes.clone()
    }

    /// Magnitudes in the different bands if zero mags were available.
    #[getter]
    pub fn magnitudes(&self) -> Vec<f64> {
        self.0.magnitudes.clone()
    }

    /// Black body specific fluxes per band in units of Jy / Steradian.
    #[getter]
    pub fn thermal_fluxes(&self) -> Vec<f64> {
        self.0.thermal_fluxes.clone()
    }

    /// Reflected light specific fluxes per band in units of Jy / Steradian.
    #[getter]
    pub fn hg_fluxes(&self) -> Vec<f64> {
        self.0.hg_fluxes.clone()
    }

    /// Expected magnitude in the V-band using the HG model.
    #[getter]
    pub fn v_band_magnitude(&self) -> f64 {
        self.0.v_band_magnitude
    }

    /// Expected flux in the V-band using the HG model.
    #[getter]
    pub fn v_band_flux(&self) -> f64 {
        self.0.v_band_flux
    }

    fn __repr__(&self) -> String {
        format!(
            "ModelResults(fluxes={:?}, thermal_fluxes={:?}, hg_fluxes={:?}, v_band_magnitude={:?},\
            v_band_flux={:?}, magnitudes={:?})",
            self.fluxes(),
            self.thermal_fluxes(),
            self.hg_fluxes(),
            self.v_band_magnitude(),
            self.v_band_flux(),
            self.magnitudes(),
        )
    }
}

/// Resolve any two of (h_mag, vis_albedo, diameter) to compute the third.
///
/// Given any two of H-magnitude, visible geometric albedo, and diameter, this
/// computes the missing value using the standard C_hg relationship. If all three
/// are provided, it validates that they are consistent.
///
/// Parameters
/// ----------
/// h_mag :
///     H magnitude of the object in the HG system.
/// vis_albedo :
///     Visible geometric albedo.
/// diameter :
///     Diameter of the object in km.
/// c_hg :
///     The C_hg constant (default 1329.0 km).
///
/// Returns
/// -------
/// tuple
///     ``(h_mag, vis_albedo, diameter)`` with all three resolved.
#[pyfunction]
#[pyo3(name = "resolve_hg_params", signature = (h_mag=None, vis_albedo=None, diameter=None, c_hg=None))]
pub fn resolve_hg_params_py(
    h_mag: Option<f64>,
    vis_albedo: Option<f64>,
    diameter: Option<f64>,
    c_hg: Option<f64>,
) -> PyResult<(f64, f64, f64)> {
    Ok(resolve_hg_params(h_mag, vis_albedo, diameter, c_hg)?)
}

/// Compute NEATM thermal + reflected fluxes for a single geometry.
///
/// Evaluates the NEATM model for the given Sun-object-observer geometry,
/// computing both thermal emission and reflected solar light (HG model)
/// across multiple wavelength bands simultaneously.
///
/// Parameters
/// ----------
/// sun2obj :
///     Vector pointing from the Sun to the object (AU).
/// sun2obs :
///     Vector pointing from the Sun to the observer (AU).
/// band_albedos :
///     Albedo of the object in each band (0-1).
/// h_mag :
///     H magnitude of the object in the HG system. At least two of
///     ``h_mag``, ``diameter``, and ``vis_albedo`` must be provided.
/// diameter :
///     Diameter of the object in km.
/// vis_albedo :
///     Visible geometric albedo.
/// g_param :
///     G phase coefficient, defaults to ``0.15``.
/// beaming :
///     Beaming parameter, defaults to ``1.0``.
/// emissivity :
///     Emissivity of the object, defaults to ``0.9``.
/// band_wavelengths :
///     List of effective wavelengths in nm. Required unless ``bands`` is given.
/// bands :
///     Band preset name: ``"wise"`` or ``"neos"``. If given, ``band_wavelengths``
///     is ignored and the standard band definitions (including color corrections
///     and zero magnitudes) are used.
/// zero_mags :
///     Optional list of zero-point magnitudes for each band. Only used when
///     ``band_wavelengths`` is provided.
///
/// Returns
/// -------
/// ModelResults
///     Fluxes and magnitudes for the given geometry.
#[pyfunction]
#[pyo3(name = "neatm_model_flux", signature = (sun2obj, sun2obs, band_albedos,
    h_mag=None, diameter=None, vis_albedo=None, g_param=0.15, beaming=1.0,
    emissivity=0.9, band_wavelengths=None, bands=None, zero_mags=None))]
#[allow(clippy::too_many_arguments)]
pub fn neatm_model_flux_py(
    sun2obj: VectorLike,
    sun2obs: VectorLike,
    band_albedos: Vec<f64>,
    h_mag: Option<f64>,
    diameter: Option<f64>,
    vis_albedo: Option<f64>,
    g_param: f64,
    beaming: f64,
    emissivity: f64,
    band_wavelengths: Option<Vec<f64>>,
    bands: Option<&str>,
    zero_mags: Option<Vec<f64>>,
) -> PyResult<PyModelResults> {
    let obs_bands = resolve_bands(band_wavelengths, bands, zero_mags)?;
    let (h_mag, vis_albedo, diameter) = resolve_hg_params(h_mag, vis_albedo, diameter, None)?;
    let s2o = sun2obj.into_vector(PyFrames::Ecliptic).into();
    let s2obs = sun2obs.into_vector(PyFrames::Ecliptic).into();
    Ok(neatm_total_flux(
        &obs_bands,
        &band_albedos,
        diameter,
        vis_albedo,
        g_param,
        h_mag,
        beaming,
        emissivity,
        &s2o,
        &s2obs,
    )
    .into())
}

/// Compute FRM thermal + reflected fluxes for a single geometry.
///
/// Evaluates the FRM model for the given Sun-object-observer geometry,
/// computing both thermal emission and reflected solar light (HG model)
/// across multiple wavelength bands simultaneously.
///
/// Parameters
/// ----------
/// sun2obj :
///     Vector pointing from the Sun to the object (AU).
/// sun2obs :
///     Vector pointing from the Sun to the observer (AU).
/// band_albedos :
///     Albedo of the object in each band (0-1).
/// h_mag :
///     H magnitude of the object in the HG system. At least two of
///     ``h_mag``, ``diameter``, and ``vis_albedo`` must be provided.
/// diameter :
///     Diameter of the object in km.
/// vis_albedo :
///     Visible geometric albedo.
/// g_param :
///     G phase coefficient, defaults to ``0.15``.
/// emissivity :
///     Emissivity of the object, defaults to ``0.9``.
/// band_wavelengths :
///     List of effective wavelengths in nm. Required unless ``bands`` is given.
/// bands :
///     Band preset name: ``"wise"`` or ``"neos"``. If given, ``band_wavelengths``
///     is ignored and the standard band definitions (including color corrections
///     and zero magnitudes) are used.
/// zero_mags :
///     Optional list of zero-point magnitudes for each band. Only used when
///     ``band_wavelengths`` is provided.
///
/// Returns
/// -------
/// ModelResults
///     Fluxes and magnitudes for the given geometry.
#[pyfunction]
#[pyo3(name = "frm_model_flux", signature = (sun2obj, sun2obs, band_albedos,
    h_mag=None, diameter=None, vis_albedo=None, g_param=0.15, emissivity=0.9,
    band_wavelengths=None, bands=None, zero_mags=None))]
#[allow(clippy::too_many_arguments)]
pub fn frm_model_flux_py(
    sun2obj: VectorLike,
    sun2obs: VectorLike,
    band_albedos: Vec<f64>,
    h_mag: Option<f64>,
    diameter: Option<f64>,
    vis_albedo: Option<f64>,
    g_param: f64,
    emissivity: f64,
    band_wavelengths: Option<Vec<f64>>,
    bands: Option<&str>,
    zero_mags: Option<Vec<f64>>,
) -> PyResult<PyModelResults> {
    let obs_bands = resolve_bands(band_wavelengths, bands, zero_mags)?;
    let (h_mag, vis_albedo, diameter) = resolve_hg_params(h_mag, vis_albedo, diameter, None)?;
    let s2o = sun2obj.into_vector(PyFrames::Ecliptic).into();
    let s2obs = sun2obs.into_vector(PyFrames::Ecliptic).into();
    Ok(frm_total_flux(
        &obs_bands,
        &band_albedos,
        diameter,
        vis_albedo,
        g_param,
        h_mag,
        emissivity,
        &s2o,
        &s2obs,
    )
    .into())
}

/// Helper to build band info from either explicit wavelengths or a preset name.
fn resolve_bands(
    band_wavelengths: Option<Vec<f64>>,
    bands: Option<&str>,
    zero_mags: Option<Vec<f64>>,
) -> PyResult<Vec<BandInfo>> {
    match (bands, band_wavelengths) {
        (Some(name), _) => match name.to_lowercase().as_str() {
            "wise" => Ok(BandInfo::WISE.to_vec()),
            "neos" => Ok(BandInfo::NEOS.to_vec()),
            other => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Unknown band preset '{other}'. Use 'wise' or 'neos'."
            ))),
        },
        (None, Some(wavelengths)) => {
            let zm = zero_mags.unwrap_or(vec![f64::NAN; wavelengths.len()]);
            Ok(wavelengths
                .iter()
                .zip(zm)
                .map(|(w, z)| BandInfo::new(*w, 1.0, z, None))
                .collect())
        }
        (None, None) => Err(pyo3::exceptions::PyValueError::new_err(
            "Either 'bands' or 'band_wavelengths' must be provided.",
        )),
    }
}
