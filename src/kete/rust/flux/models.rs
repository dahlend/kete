//! Model inputs and outputs for NEATM/FRM/Reflected.

use rayon::prelude::*;

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

/// NEATM Model parameters.
///
/// This holds the model parameters for NEATM.
/// By definition, providing any two of the following fully define the third:
///
/// - H-magnitude
/// - Diameter
/// - Visible geometric albedo
///
/// For ease of use, this class requires only two of any of those values to be
/// provided, and the third is computed automatically. If all 3 are provided it will
/// validate that they are internally consistent, and raise an exception if not.
///
/// Parameters
/// ----------
/// band_wavelength :
///     List of effective wavelengths in nm.
/// band_albedos :
///     List of albedoes of the object for each wavelength (0-1).
/// h_mag:
///     H magnitude of the object in the HG system.
/// diameter:
///     Diameter of the object in km.
/// vis_albedo:
///     Visible geometric albedo of the object.
/// beaming :
///     Beaming parameter, defaults to `1.0`.
/// g_param :
///     G phase coefficient, defaults to `0.15`.
/// c_hg :
///     The C_hg constant used to define the relationship between diameter, albedo, and
///     H mag. This uses the default value defined in the constants, and is not
///     recommended to be changed.
/// emissivity:
///     Emissivity of the object, defaults to `0.9`.
/// zero_mags:
///     Optional - If zero mags are provided then magnitudes may be computed.
#[pyclass(frozen, module = "kete.flux", name = "NeatmParams", from_py_object)]
#[derive(Clone, Debug)]
#[allow(missing_docs)]
pub struct PyNeatmParams {
    pub obs_bands: Vec<BandInfo>,
    pub band_albedos: Vec<f64>,
    pub beaming: f64,
    pub g_param: f64,
    pub h_mag: f64,
    pub vis_albedo: f64,
    pub diameter: f64,
    pub emissivity: f64,
}

#[pymethods]
impl PyNeatmParams {
    #[new]
    #[allow(clippy::too_many_arguments, missing_docs)]
    #[pyo3(signature = (band_wavelengths, band_albedos, h_mag=None, diameter=None,
        vis_albedo=None, beaming=1.0, g_param=0.15, c_hg=None, emissivity=0.9, zero_mags=None))]
    pub fn new(
        band_wavelengths: Vec<f64>,
        band_albedos: Vec<f64>,
        h_mag: Option<f64>,
        diameter: Option<f64>,
        vis_albedo: Option<f64>,
        beaming: f64,
        g_param: f64,
        c_hg: Option<f64>,
        emissivity: f64,
        zero_mags: Option<Vec<f64>>,
    ) -> PyResult<Self> {
        let n_bands = band_wavelengths.len();
        let zero_mags = zero_mags.unwrap_or(vec![f64::NAN; n_bands]);
        let (h_mag, vis_albedo, diameter, _c_hg) =
            resolve_hg_params(h_mag, vis_albedo, diameter, c_hg)?;
        let vis_albedo = vis_albedo.ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(
                "At least two of h_mag, diameter, and vis_albedo must be provided.",
            )
        })?;
        let diameter = diameter.ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(
                "At least two of h_mag, diameter, and vis_albedo must be provided.",
            )
        })?;

        let obs_bands = band_wavelengths
            .iter()
            .zip(zero_mags)
            .map(|(wavelength, z_mag)| BandInfo::new(*wavelength, 1.0, z_mag, None))
            .collect();

        Ok(Self {
            obs_bands,
            beaming,
            band_albedos,
            g_param,
            h_mag,
            vis_albedo,
            diameter,
            emissivity,
        })
    }

    /// Create a new NeatmParams with WISE bands and zero magnitudes for 300k objects.
    /// This requires all 4 albedos to be provided.
    ///
    /// Parameters
    /// ----------
    /// band_albedos :
    ///     List of albedoes of the object for each wavelength (0-1).
    /// h_mag:
    ///     H magnitude of the object in the HG system.
    /// diameter:
    ///     Diameter of the object in km.
    /// vis_albedo:
    ///     Visible geometric albedo of the object.
    /// beaming :
    ///     Beaming parameter, defaults to `1.0`.
    /// g_param :
    ///     G phase coefficient, defaults to `0.15`.
    /// c_hg :
    ///     The C_hg constant used to define the relationship between diameter, albedo, and
    ///     H mag. This uses the default value defined in the constants, and is not
    ///     recommended to be changed.
    /// emissivity:
    ///     Emissivity of the object, defaults to `0.9`.
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (band_albedos, h_mag=None, diameter=None, vis_albedo=None,
        beaming=1.0, g_param=0.15, c_hg=None, emissivity=0.9))]
    pub fn new_wise(
        band_albedos: Vec<f64>,
        h_mag: Option<f64>,
        diameter: Option<f64>,
        vis_albedo: Option<f64>,
        beaming: f64,
        g_param: f64,
        c_hg: Option<f64>,
        emissivity: f64,
    ) -> PyResult<Self> {
        let (h_mag, vis_albedo, diameter, _c_hg) =
            resolve_hg_params(h_mag, vis_albedo, diameter, c_hg)?;
        let vis_albedo = vis_albedo.ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(
                "At least two of h_mag, diameter, and vis_albedo must be provided.",
            )
        })?;
        let diameter = diameter.ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(
                "At least two of h_mag, diameter, and vis_albedo must be provided.",
            )
        })?;

        let band_albedos: [f64; 4] = match band_albedos.try_into() {
            Ok(v) => v,
            Err(_) => Err(kete_core::errors::Error::ValueError(
                "4 Albedos must be provided, one for each WISE band.".into(),
            ))?,
        };

        let obs_bands = BandInfo::new_wise().to_vec();

        Ok(Self {
            obs_bands,
            beaming,
            band_albedos: band_albedos.to_vec(),
            g_param,
            h_mag,
            vis_albedo,
            diameter,
            emissivity,
        })
    }

    /// Create a new NeatmParams with NEOS bands and zero magnitudes.
    /// This requires 2 albedos to be provided, one for each band.
    ///
    /// Parameters
    /// ----------
    /// band_albedos :
    ///     List of albedoes of the object for each wavelength (0-1).
    /// h_mag:
    ///     H magnitude of the object in the HG system.
    /// diameter:
    ///     Diameter of the object in km.
    /// vis_albedo:
    ///     Visible geometric albedo of the object.
    /// beaming :
    ///     Beaming parameter, defaults to `1.0`.
    /// g_param :
    ///     G phase coefficient, defaults to `0.15`.
    /// c_hg :
    ///     The C_hg constant used to define the relationship between diameter, albedo, and
    ///     H mag. This uses the default value defined in the constants, and is not
    ///     recommended to be changed.
    /// emissivity:
    ///     Emissivity of the object, defaults to `0.9`.
    #[staticmethod]
    #[pyo3(signature = (band_albedos, h_mag=None, diameter=None, vis_albedo=None, beaming=1.0,
        g_param=0.15, c_hg=None, emissivity=0.9))]
    #[allow(clippy::too_many_arguments)]
    pub fn new_neos(
        band_albedos: Vec<f64>,
        h_mag: Option<f64>,
        diameter: Option<f64>,
        vis_albedo: Option<f64>,
        beaming: f64,
        g_param: f64,
        c_hg: Option<f64>,
        emissivity: f64,
    ) -> PyResult<Self> {
        let (h_mag, vis_albedo, diameter, _c_hg) =
            resolve_hg_params(h_mag, vis_albedo, diameter, c_hg)?;
        let vis_albedo = vis_albedo.ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(
                "At least two of h_mag, diameter, and vis_albedo must be provided.",
            )
        })?;
        let diameter = diameter.ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(
                "At least two of h_mag, diameter, and vis_albedo must be provided.",
            )
        })?;

        let band_albedos: [f64; 2] = match band_albedos.try_into() {
            Ok(v) => v,
            Err(_) => Err(kete_core::errors::Error::ValueError(
                "2 Albedos must be provided, one for each NEOS band.".into(),
            ))?,
        };

        let obs_bands = BandInfo::new_neos().to_vec();

        Ok(Self {
            obs_bands,
            beaming,
            band_albedos: band_albedos.to_vec(),
            g_param,
            h_mag,
            vis_albedo,
            diameter,
            emissivity,
        })
    }

    /// Evaluate the thermal model at the provided observer and object positions.
    ///
    /// This returns a list of [`ModelResults`] for the computed fluxes/magnitudes.
    /// This is a multi-core operation.
    ///
    /// Parameters
    /// ----------
    /// sun2obj_vecs :
    ///     A list of [`Vector`] like objects which define the position of the object with respect to the sun.
    /// sun2obs_vecs :
    ///     A list of [`Vector`] like objects which define the position of the observer with respect to the sun.
    pub fn evaluate(
        &self,
        sun2obj_vecs: Vec<VectorLike>,
        sun2obs_vecs: Vec<VectorLike>,
    ) -> PyResult<Vec<PyModelResults>> {
        let obs_bands = &self.obs_bands;
        let band_albedos = &self.band_albedos;
        let diameter = self.diameter;
        let vis_albedo = self.vis_albedo;
        let g_param = self.g_param;
        let h_mag = self.h_mag;
        let beaming = self.beaming;
        let emissivity = self.emissivity;
        sun2obj_vecs
            .into_par_iter()
            .zip(sun2obs_vecs)
            .map(|(sun2obj, sun2obs)| {
                let sun2obj = sun2obj.into_vector(PyFrames::Ecliptic).into();
                let sun2obs = sun2obs.into_vector(PyFrames::Ecliptic).into();

                Ok(neatm_total_flux(
                    obs_bands,
                    band_albedos,
                    diameter,
                    vis_albedo,
                    g_param,
                    h_mag,
                    beaming,
                    emissivity,
                    &sun2obj,
                    &sun2obs,
                )
                .into())
            })
            .collect()
    }

    /// List of effective wavelengths in nm.
    #[getter]
    pub fn band_wavelength(&self) -> Vec<f64> {
        self.obs_bands.iter().map(|x| x.wavelength).collect()
    }

    /// List of albedoes of the object, one for each band.
    #[getter]
    pub fn band_albedos(&self) -> Vec<f64> {
        self.band_albedos.clone()
    }

    /// H Mag for the object in the HG system.
    #[getter]
    pub fn h_mag(&self) -> f64 {
        self.h_mag
    }

    /// Diameter of the object in km.
    #[getter]
    pub fn diameter(&self) -> f64 {
        self.diameter
    }

    /// Albedo in V band.
    #[getter]
    pub fn vis_albedo(&self) -> f64 {
        self.vis_albedo
    }

    /// Beaming parameter.
    #[getter]
    pub fn beaming(&self) -> f64 {
        self.beaming
    }

    /// G Phase parameter.
    #[getter]
    pub fn g_param(&self) -> f64 {
        self.g_param
    }

    /// Emissivity of the object.
    #[getter]
    pub fn emissivity(&self) -> f64 {
        self.emissivity
    }

    /// List of the zero mags for each band if provided.
    #[getter]
    pub fn zero_mags(&self) -> Vec<f64> {
        self.obs_bands.iter().map(|x| x.zero_mag).collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "NeatmParams(band_wavelength={:?}, band_albedos={:?}, h_mag={:?}, \
                diameter={:?}, vis_albedo={:?}, beaming={:?}, g_param={:?}, emissivity={:?}, \
                zero_mags={:?})",
            self.band_wavelength(),
            self.band_albedos(),
            self.h_mag(),
            self.diameter(),
            self.vis_albedo(),
            self.beaming(),
            self.g_param(),
            self.emissivity(),
            self.zero_mags(),
        )
    }
}

/// FRM Model parameters.
///
/// This holds the model parameters for FRM.
/// By definition, providing any two of the following fully define the third:
///
/// - H-magnitude
/// - Diameter
/// - Visible geometric albedo
///
/// For ease of use, this class requires only two of any of those values to be
/// provided, and the third is computed automatically. If all 3 are provided it will
/// validate that they are internally consistent, and raise an exception if not.
///
/// Parameters
/// ----------
/// band_wavelength :
///     List of effective wavelengths in nm.
/// band_albedos :
///     List of albedoes of the object for each wavelength (0-1).
/// h_mag:
///     H magnitude of the object in the HG system.
/// diameter:
///     Diameter of the object in km.
/// vis_albedo:
///     Visible geometric albedo of the object.
/// g_param :
///     G phase coefficient, defaults to `0.15`.
/// c_hg :
///     The C_hg constant used to define the relationship between diameter, albedo, and
///     H mag. This uses the default value defined in the constants, and is not
///     recommended to be changed.
/// emissivity:
///     Emissivity of the object, defaults to `0.9`.
/// zero_mags:
///     Optional - If zero mags are provided then magnitudes may be computed.
#[pyclass(frozen, module = "kete.flux", name = "FrmParams", from_py_object)]
#[derive(Clone, Debug)]
#[allow(missing_docs)]
pub struct PyFrmParams {
    pub obs_bands: Vec<BandInfo>,
    pub band_albedos: Vec<f64>,
    pub g_param: f64,
    pub h_mag: f64,
    pub vis_albedo: f64,
    pub diameter: f64,
    pub emissivity: f64,
}

#[pymethods]
impl PyFrmParams {
    #[new]
    #[allow(clippy::too_many_arguments, missing_docs)]
    #[pyo3(signature = (band_wavelengths, band_albedos, h_mag=None, diameter=None,
        vis_albedo=None, g_param=0.15, c_hg=None, emissivity=0.9, zero_mags=None))]
    pub fn new(
        band_wavelengths: Vec<f64>,
        band_albedos: Vec<f64>,
        h_mag: Option<f64>,
        diameter: Option<f64>,
        vis_albedo: Option<f64>,
        g_param: f64,
        c_hg: Option<f64>,
        emissivity: f64,
        zero_mags: Option<Vec<f64>>,
    ) -> PyResult<Self> {
        let n_bands = band_wavelengths.len();
        let zero_mags = zero_mags.unwrap_or(vec![f64::NAN; n_bands]);
        let (h_mag, vis_albedo, diameter, _c_hg) =
            resolve_hg_params(h_mag, vis_albedo, diameter, c_hg)?;
        let vis_albedo = vis_albedo.ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(
                "At least two of h_mag, diameter, and vis_albedo must be provided.",
            )
        })?;
        let diameter = diameter.ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(
                "At least two of h_mag, diameter, and vis_albedo must be provided.",
            )
        })?;

        let obs_bands = band_wavelengths
            .iter()
            .zip(zero_mags)
            .map(|(wavelength, z_mag)| BandInfo::new(*wavelength, 1.0, z_mag, None))
            .collect();

        Ok(Self {
            obs_bands,
            band_albedos,
            g_param,
            h_mag,
            vis_albedo,
            diameter,
            emissivity,
        })
    }

    /// Create a new FrmParams with WISE bands and zero magnitudes for 300k objects.
    /// This requires all 4 albedos to be provided.
    ///
    /// Parameters
    /// ----------
    /// band_albedos :
    ///     List of albedoes of the object for each wavelength (0-1).
    /// h_mag:
    ///     H magnitude of the object in the HG system.
    /// diameter:
    ///     Diameter of the object in km.
    /// vis_albedo:
    ///     Visible geometric albedo of the object.
    /// g_param :
    ///     G phase coefficient, defaults to `0.15`.
    /// c_hg :
    ///     The C_hg constant used to define the relationship between diameter, albedo, and
    ///     H mag. This uses the default value defined in the constants, and is not
    ///     recommended to be changed.
    /// emissivity:
    ///     Emissivity of the object, defaults to `0.9`.
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (band_albedos, h_mag=None, diameter=None, vis_albedo=None, g_param=0.15,
        c_hg=None, emissivity=0.9))]
    pub fn new_wise(
        band_albedos: Vec<f64>,
        h_mag: Option<f64>,
        diameter: Option<f64>,
        vis_albedo: Option<f64>,
        g_param: f64,
        c_hg: Option<f64>,
        emissivity: f64,
    ) -> PyResult<Self> {
        let (h_mag, vis_albedo, diameter, _c_hg) =
            resolve_hg_params(h_mag, vis_albedo, diameter, c_hg)?;
        let vis_albedo = vis_albedo.ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(
                "At least two of h_mag, diameter, and vis_albedo must be provided.",
            )
        })?;
        let diameter = diameter.ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(
                "At least two of h_mag, diameter, and vis_albedo must be provided.",
            )
        })?;

        let band_albedos: [f64; 4] = match band_albedos.try_into() {
            Ok(v) => v,
            Err(_) => Err(kete_core::errors::Error::ValueError(
                "4 Albedos must be provided, one for each WISE band.".into(),
            ))?,
        };

        let obs_bands = BandInfo::new_wise().to_vec();

        Ok(Self {
            obs_bands,
            band_albedos: band_albedos.to_vec(),
            g_param,
            h_mag,
            vis_albedo,
            diameter,
            emissivity,
        })
    }

    /// Create a new FrmParams with NEOS bands and zero magnitudes.
    /// This requires 2 albedos to be provided, one for each band.
    ///
    /// Parameters
    /// ----------
    /// band_albedos :
    ///     List of albedoes of the object for each wavelength (0-1).
    /// h_mag:
    ///     H magnitude of the object in the HG system.
    /// diameter:
    ///     Diameter of the object in km.
    /// vis_albedo:
    ///     Visible geometric albedo of the object.
    /// g_param :
    ///     G phase coefficient, defaults to `0.15`.
    /// c_hg :
    ///     The C_hg constant used to define the relationship between diameter, albedo, and
    ///     H mag. This uses the default value defined in the constants, and is not
    ///     recommended to be changed.
    /// emissivity:
    ///     Emissivity of the object, defaults to `0.9`.
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (band_albedos, h_mag=None, diameter=None, vis_albedo=None, g_param=0.15,
        c_hg=None, emissivity=0.9))]
    pub fn new_neos(
        band_albedos: Vec<f64>,
        h_mag: Option<f64>,
        diameter: Option<f64>,
        vis_albedo: Option<f64>,
        g_param: f64,
        c_hg: Option<f64>,
        emissivity: f64,
    ) -> PyResult<Self> {
        let (h_mag, vis_albedo, diameter, _c_hg) =
            resolve_hg_params(h_mag, vis_albedo, diameter, c_hg)?;
        let vis_albedo = vis_albedo.ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(
                "At least two of h_mag, diameter, and vis_albedo must be provided.",
            )
        })?;
        let diameter = diameter.ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(
                "At least two of h_mag, diameter, and vis_albedo must be provided.",
            )
        })?;

        let band_albedos: [f64; 2] = match band_albedos.try_into() {
            Ok(v) => v,
            Err(_) => Err(kete_core::errors::Error::ValueError(
                "2 Albedos must be provided, one for each NEOS band.".into(),
            ))?,
        };

        let obs_bands = BandInfo::new_neos().to_vec();

        Ok(Self {
            obs_bands,
            band_albedos: band_albedos.to_vec(),
            g_param,
            h_mag,
            vis_albedo,
            diameter,
            emissivity,
        })
    }

    /// Evaluate the thermal model at the provided observer and object positions.
    ///
    /// This returns a list of [`ModelResults`] for the computed fluxes/magnitudes.
    /// This is a multi-core operation.
    ///
    /// Parameters
    /// ----------
    /// sun2obj_vecs :
    ///     A list of [`Vector`] like objects which define the position of the object with respect to the sun.
    /// sun2obs_vecs :
    ///     A list of [`Vector`] like objects which define the position of the observer with respect to the sun.
    pub fn evaluate(
        &self,
        sun2obj_vecs: Vec<VectorLike>,
        sun2obs_vecs: Vec<VectorLike>,
    ) -> PyResult<Vec<PyModelResults>> {
        let obs_bands = &self.obs_bands;
        let band_albedos = &self.band_albedos;
        let diameter = self.diameter;
        let vis_albedo = self.vis_albedo;
        let g_param = self.g_param;
        let h_mag = self.h_mag;
        let emissivity = self.emissivity;
        sun2obj_vecs
            .into_par_iter()
            .zip(sun2obs_vecs)
            .map(|(sun2obj, sun2obs)| {
                let sun2obj = sun2obj.into_vector(PyFrames::Ecliptic).into();
                let sun2obs = sun2obs.into_vector(PyFrames::Ecliptic).into();

                Ok(frm_total_flux(
                    obs_bands,
                    band_albedos,
                    diameter,
                    vis_albedo,
                    g_param,
                    h_mag,
                    emissivity,
                    &sun2obj,
                    &sun2obs,
                )
                .into())
            })
            .collect()
    }

    /// List of effective wavelengths in nm.
    #[getter]
    pub fn band_wavelength(&self) -> Vec<f64> {
        self.obs_bands.iter().map(|x| x.wavelength).collect()
    }

    /// List of albedoes of the object, one for each band.
    #[getter]
    pub fn band_albedos(&self) -> Vec<f64> {
        self.band_albedos.clone()
    }

    /// H Mag for the object in the HG system.
    #[getter]
    pub fn h_mag(&self) -> f64 {
        self.h_mag
    }

    /// Diameter of the object in km.
    #[getter]
    pub fn diameter(&self) -> f64 {
        self.diameter
    }

    /// Albedo in V band.
    #[getter]
    pub fn vis_albedo(&self) -> f64 {
        self.vis_albedo
    }

    /// G Phase parameter.
    #[getter]
    pub fn g_param(&self) -> f64 {
        self.g_param
    }

    /// Emissivity of the object.
    #[getter]
    pub fn emissivity(&self) -> f64 {
        self.emissivity
    }

    /// List of the zero mags for each band if provided.
    #[getter]
    pub fn zero_mags(&self) -> Vec<f64> {
        self.obs_bands.iter().map(|x| x.zero_mag).collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "FrmParams(band_wavelength={:?}, band_albedos={:?}, h_mag={:?}, \
                diameter={:?}, vis_albedo={:?}, g_param={:?}, emissivity={:?}, zero_mags={:?})",
            self.band_wavelength(),
            self.band_albedos(),
            self.h_mag(),
            self.diameter(),
            self.vis_albedo(),
            self.g_param(),
            self.emissivity(),
            self.zero_mags(),
        )
    }
}
