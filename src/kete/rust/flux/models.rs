//! Model inputs and outputs for NEATM/FRM/Reflected.

use kete_core::BandInfo;
use kete_flux::{
    SpinState, ThermalParams, TpmShape, frm_total_flux, gamma_from_mean_slope, neatm_total_flux,
    resolve_hg_params, rms_slope, tpm_total_flux, tpm_total_flux_rough,
};
use nalgebra::UnitVector3;
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
///     Band preset name: ``"wise"``, ``"neos"``, ``"irac"``, ``"mips"``, or
///     ``"irs_pu"``. If given, ``band_wavelengths`` is ignored and the standard
///     band definitions (including color corrections and zero magnitudes) are used.
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
///     Band preset name: ``"wise"``, ``"neos"``, ``"irac"``, ``"mips"``, or
///     ``"irs_pu"``. If given, ``band_wavelengths`` is ignored and the standard
///     band definitions (including color corrections and zero magnitudes) are used.
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

/// Compute TPM (thermophysical model) thermal + reflected fluxes for a geometry.
///
/// Unlike NEATM and FRM, the TPM solves the 1D heat conduction equation into the
/// subsurface, so the surface has a thermal memory set by the thermal inertia. The
/// night side stays warm and the temperature peak lags local noon. The model uses a
/// smooth sphere and requires a spin state (pole and rotation period).
///
/// This model is substantially more expensive than NEATM or FRM (it iterates the
/// heat solver to a periodic steady state per latitude band).
///
/// Parameters
/// ----------
/// sun2obj :
///     Vector pointing from the Sun to the object (AU).
/// sun2obs :
///     Vector pointing from the Sun to the observer (AU).
/// band_albedos :
///     Albedo of the object in each band (0-1).
/// thermal_inertia :
///     Thermal inertia ``Gamma`` in SI units (J m^-2 K^-1 s^-1/2). Must be
///     non-negative; ``0`` is the instantaneous-equilibrium limit (equivalent to
///     NEATM with a beaming of 1).
/// period :
///     Rotation period in seconds. Must be non-negative; ``0`` is the infinitely
///     fast rotator limit (equivalent to FRM).
/// pole :
///     Spin axis direction (ecliptic). Rotation is right-handed about this vector;
///     a reversed pole gives retrograde rotation, which sets the sign of the
///     thermal lag.
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
/// axis_ratios :
///     Optional ``(b/a, c/a)`` semi-axis ratios for an ellipsoidal shape, where
///     ``a`` is the long axis (body x) and ``c`` is along the pole (body z). ``None``
///     (default) is a sphere; ``(1.0, c/a)`` is an oblate spheroid; ``b/a != 1`` is a
///     triaxial ellipsoid. ``diameter`` is the effective (equal-area-sphere) diameter.
/// epoch :
///     Observation time (Julian date). Only affects non-axisymmetric (triaxial)
///     shapes, for which it sets the rotation phase.
/// phase0 :
///     Rotation phase (radians) at ``epoch0``. Defaults to ``0``.
/// epoch0 :
///     Reference epoch (Julian date) for ``phase0``. Defaults to ``0``.
/// roughness :
///     Optional surface roughness as the mean slope angle ``theta_bar`` in degrees,
///     ``(0, 57.3]`` (the cross-model roughness convention; the full-coverage
///     spherical-cap geometry tops out near 57.3 deg). ``None`` (default) is a smooth
///     surface. The rough (beaming) path solves a crater per latitude band on the fly
///     and is substantially slower than the smooth model.
/// band_wavelengths :
///     List of effective wavelengths in nm. Required unless ``bands`` is given.
/// bands :
///     Band preset name: ``"wise"``, ``"neos"``, ``"irac"``, ``"mips"``, or
///     ``"irs_pu"``. If given, ``band_wavelengths`` is ignored and the standard
///     band definitions (including color corrections and zero magnitudes) are used.
/// zero_mags :
///     Optional list of zero-point magnitudes for each band. Only used when
///     ``band_wavelengths`` is provided.
///
/// Returns
/// -------
/// ModelResults
///     Fluxes and magnitudes for the given geometry.
#[pyfunction]
#[pyo3(name = "tpm_model_flux", signature = (sun2obj, sun2obs, band_albedos,
    thermal_inertia, period, pole, h_mag=None, diameter=None, vis_albedo=None,
    g_param=0.15, emissivity=0.9, axis_ratios=None, epoch=0.0, phase0=0.0,
    epoch0=0.0, roughness=None, band_wavelengths=None, bands=None, zero_mags=None))]
#[allow(clippy::too_many_arguments)]
pub fn tpm_model_flux_py(
    sun2obj: VectorLike,
    sun2obs: VectorLike,
    band_albedos: Vec<f64>,
    thermal_inertia: f64,
    period: f64,
    pole: VectorLike,
    h_mag: Option<f64>,
    diameter: Option<f64>,
    vis_albedo: Option<f64>,
    g_param: f64,
    emissivity: f64,
    axis_ratios: Option<(f64, f64)>,
    epoch: f64,
    phase0: f64,
    epoch0: f64,
    roughness: Option<f64>,
    band_wavelengths: Option<Vec<f64>>,
    bands: Option<&str>,
    zero_mags: Option<Vec<f64>>,
) -> PyResult<PyModelResults> {
    // Zero period (infinitely fast rotator) and zero inertia (instantaneous
    // equilibrium) are valid limits; negative or non-finite values are not.
    if !(thermal_inertia.is_finite() && thermal_inertia >= 0.0) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "thermal_inertia must be finite and non-negative.",
        ));
    }
    if !(period.is_finite() && period >= 0.0) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "period must be finite and non-negative.",
        ));
    }
    let obs_bands = resolve_bands(band_wavelengths, bands, zero_mags)?;
    let (h_mag, vis_albedo, diameter) = resolve_hg_params(h_mag, vis_albedo, diameter, None)?;
    let s2o = sun2obj.into_vector(PyFrames::Ecliptic).into();
    let s2obs = sun2obs.into_vector(PyFrames::Ecliptic).into();
    let pole_vec = pole.into_vector(PyFrames::Ecliptic).into();
    let spin = SpinState {
        pole: UnitVector3::new_normalize(pole_vec),
        period,
        phase0,
        epoch0,
    };
    // axis_ratios = (b/a, c/a); a is the long axis along body-x, c is along the pole.
    let shape = match axis_ratios {
        Some((b_over_a, c_over_a)) => TpmShape::ellipsoid(1.0, b_over_a, c_over_a),
        None => TpmShape::sphere(),
    };
    let thermal = ThermalParams {
        thermal_inertia,
        emissivity,
    };

    // Roughness (mean slope angle, given in degrees) selects the slower beaming path;
    // otherwise the smooth surface is used. The mean slope angle is converted to the
    // internal crater opening half-angle that the rough solver is parameterized by.
    if let Some(roughness_deg) = roughness {
        // Mean slope angle (degrees); the full-coverage cap tops out at the hemisphere
        // limit ~57.3 deg, above which it cannot be represented. The converter clamps
        // the sub-0.01 deg rounding overshoot at the ceiling.
        if !(roughness_deg.is_finite() && roughness_deg > 0.0 && roughness_deg <= 57.3) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "roughness must be a mean slope angle in (0, 57.3] degrees.",
            ));
        }
        let roughness_angle = gamma_from_mean_slope(roughness_deg.to_radians());
        return Ok(tpm_total_flux_rough(
            &obs_bands,
            &band_albedos,
            &spin,
            &shape,
            &thermal,
            diameter,
            vis_albedo,
            g_param,
            h_mag,
            &s2o,
            &s2obs,
            epoch,
            roughness_angle,
        )?
        .into());
    }

    Ok(tpm_total_flux(
        &obs_bands,
        &band_albedos,
        &spin,
        &shape,
        &thermal,
        diameter,
        vis_albedo,
        g_param,
        h_mag,
        &s2o,
        &s2obs,
        epoch,
    )?
    .into())
}

/// Convert a surface-roughness mean slope angle to the equivalent RMS slope angle.
///
/// kete parameterizes roughness by the mean slope angle ``theta_bar`` (the
/// convention-stable, cross-model roughness number; e.g. Hapke photometric roughness).
/// Some thermal-modeling work instead quotes an RMS surface slope (e.g. the Rozitis &
/// Green ATPM and Gaussian-random-surface models). This returns the derived,
/// approximate RMS slope of the full-coverage spherical-cap geometry, for
/// cross-comparison with that literature.
///
/// Parameters
/// ----------
/// mean_slope :
///     Mean slope angle ``theta_bar`` in degrees, ``(0, 57.3]``.
///
/// Returns
/// -------
/// float
///     Approximate RMS slope angle in degrees. NOTE: the RMS slope is dominated by the
///     steepest micro-facets, so it is reliable only in the realistic regime (mean
///     slope up to ~35 deg) and rises steeply toward the hemisphere limit; treat it as
///     indicative beyond that.
#[pyfunction]
#[pyo3(name = "roughness_mean_slope_to_rms")]
pub fn roughness_mean_slope_to_rms_py(mean_slope: f64) -> PyResult<f64> {
    if !(mean_slope.is_finite() && mean_slope > 0.0 && mean_slope <= 57.3) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "mean_slope must be in (0, 57.3] degrees.",
        ));
    }
    let gamma = gamma_from_mean_slope(mean_slope.to_radians());
    Ok(rms_slope(gamma).to_degrees())
}

/// Build a band list from either explicit wavelengths or a preset group name.
///
/// ``bands`` accepts ``"wise"``, ``"neos"``, ``"irac"``, ``"mips"``, ``"irs_pu"``,
/// or a single band name recognised by [`BandInfo::from_name`].
/// ``band_wavelengths`` accepts a list of wavelengths in nm (with optional
/// ``zero_mags``); ``solar_correction`` and color correction are not set.
fn resolve_bands(
    band_wavelengths: Option<Vec<f64>>,
    bands: Option<&str>,
    zero_mags: Option<Vec<f64>>,
) -> PyResult<Vec<BandInfo>> {
    match (bands, band_wavelengths) {
        (Some(name), _) => match name.to_lowercase().as_str() {
            "wise" => Ok(BandInfo::WISE.to_vec()),
            "neos" => Ok(BandInfo::NEOS.to_vec()),
            "irac" => Ok(BandInfo::IRAC.to_vec()),
            "mips" => Ok(BandInfo::MIPS.to_vec()),
            "irs_pu" => Ok(BandInfo::IRS_PU.to_vec()),
            _ => BandInfo::from_name(name).map(|b| vec![b]).ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "Unknown band '{name}'. Use a preset group ('wise', 'neos', 'irac', \
                         'mips', 'irs_pu') or a single band name ('W1', 'V', etc.)."
                ))
            }),
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
