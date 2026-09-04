//! PyO3 bindings for model fitting (NEATM/FRM/HG).
//!
//! Exposes [`kete_flux::fitting`] types and functions to Python under
//! `kete.flux`.

use crate::frame::PyFrames;
use crate::stats::PyData;
use crate::vector::VectorLike;
use kete_core::BandInfo;
use kete_flux::SpinState;
use kete_flux::fitting::{self, FitResult, FluxObs, FluxPriors, Model, ParamPrior, TpmConfig};
use kete_stats::prelude::Data;
use nalgebra::UnitVector3;
use pyo3::prelude::*;

/// Resolve a band argument -- either a recognized name or a wavelength in nm.
///
/// Accepted names: ``"W1"``-``"W4"``, ``"NEOS1"``, ``"NEOS2"``, ``"V"``,
/// ``"IRAC1"``-``"IRAC4"``, ``"MIPS24"``, ``"MIPS70"``, ``"MIPS160"``,
/// ``"IRS Peak-Up Blue"``, ``"IRS Peak-Up Red"``.  If a wavelength (float)
/// is supplied the returned ``BandInfo`` has no zero-magnitude.
#[derive(Debug, FromPyObject)]
enum BandArg {
    Name(String),
    Wavelength(f64),
    WavelengthZmag(f64, f64),
}

impl BandArg {
    fn into_band_info(self) -> PyResult<BandInfo> {
        match self {
            BandArg::Name(s) => BandInfo::from_name(&s).ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "Unknown band name '{s}'. Supported: 'W1'-'W4', 'NEOS1'-'NEOS2', \
                     'IRAC1'-'IRAC4', 'MIPS24', 'MIPS70', 'MIPS160', \
                     'IRS Peak-Up Blue', 'IRS Peak-Up Red', 'V', \
                     or a wavelength in nm."
                ))
            }),
            BandArg::Wavelength(w) => Ok(BandInfo::new(w, 1.0, f64::NAN, None)),
            BandArg::WavelengthZmag(w, z) => Ok(BandInfo::new(w, 1.0, z, None)),
        }
    }
}

/// A single flux constraint at a known geometry.
///
/// A constraint on the model flux is expressed the same way a :class:`ParamPrior`
/// constrains a parameter: an optional hard ``bounds`` interval plus an optional
/// (possibly asymmetric) Gaussian point estimate. Provide ``flux``/``sigma``
/// for a point estimate, ``bounds`` for a hard interval, or both. The
/// :meth:`detection`, :meth:`upper_limit`, and :meth:`bounded` static
/// constructors cover the common cases.
///
/// Parameters
/// ----------
/// band :
///     Band identifier: a WISE name (``"W1"``-``"W4"``) or a wavelength in nm.
/// sun2obj :
///     Sun-to-object vector in AU (Ecliptic frame).
/// sun2obs :
///     Sun-to-observer vector in AU (Ecliptic frame).
/// flux :
///     Gaussian point estimate (Jy), or the upper-limit threshold when
///     ``is_upper_limit``. ``None`` for a bounds-only constraint.
/// sigma :
///     1-sigma uncertainty in Jy (lower side if asymmetric). Required when
///     ``flux`` is given, except for an upper limit, whose single noise scale
///     may be passed as either ``sigma`` or ``sigma_hi`` (not both).
/// sigma_hi :
///     Optional upper-side 1-sigma uncertainty (``sigma_plus``) in Jy. ``None``
///     means the error is symmetric and equal to ``sigma``.
/// bounds :
///     Optional hard ``(lo, hi)`` interval on the model flux (Jy): a wall, flat
///     inside, not scaled by the fitted error inflation.
/// is_upper_limit :
///     If ``True``, ``flux`` is a non-detection upper-limit threshold.
/// epoch :
///     Observation time (Julian date). Used by the TPM model to set the rotation
///     phase for non-axisymmetric shapes; ignored by NEATM/FRM/HG.
#[pyclass(frozen, module = "kete.flux", name = "FluxObs", from_py_object)]
#[derive(Clone, Debug)]
pub struct PyFluxObs(pub FluxObs);

impl PyFluxObs {
    /// Validate a hard flux interval before it reaches the core constructor
    /// (which panics on a degenerate interval).
    fn check_bounds(bounds: Option<(f64, f64)>) -> PyResult<()> {
        if let Some((lo, hi)) = bounds
            && (hi <= lo || lo.is_nan() || hi.is_nan())
        {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "`bounds` requires lo < hi, got ({lo}, {hi})"
            )));
        }
        Ok(())
    }

    /// Build the inner [`FluxObs`] from already-validated parts.
    #[allow(clippy::too_many_arguments)]
    fn build(
        flux: Option<f64>,
        sigma: Option<f64>,
        sigma_hi: Option<f64>,
        bounds: Option<(f64, f64)>,
        is_upper_limit: bool,
        band: BandArg,
        sun2obj: VectorLike,
        sun2obs: VectorLike,
        epoch: f64,
    ) -> PyResult<Self> {
        Self::check_bounds(bounds)?;
        // Resolve the point estimate as (mean, scale_lo, scale_hi). An upper
        // limit is one-sided: only the above-threshold side is constrained, so
        // its single noise scale may be given as either `sigma` or `sigma_hi`
        // (the latter matches the getters, so getter round-trips reconstruct).
        let point = match (flux, sigma, sigma_hi) {
            (Some(_), Some(_), Some(_)) if is_upper_limit => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "an upper limit takes a single noise scale: \
                     pass `sigma` or `sigma_hi`, not both",
                ));
            }
            (Some(f), Some(s), None) if is_upper_limit => Some((f, None, Some(s))),
            (Some(f), None, Some(s)) if is_upper_limit => Some((f, None, Some(s))),
            (Some(f), Some(s), sh) => Some((f, Some(s), Some(sh.unwrap_or(s)))),
            (Some(_), None, _) => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "`sigma` is required when `flux` is given",
                ));
            }
            (None, s, sh) => {
                if s.is_some() || sh.is_some() || is_upper_limit {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "`sigma`/`sigma_hi`/`is_upper_limit` require `flux` to be set",
                    ));
                }
                None
            }
        };
        if point.is_none() && bounds.is_none() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "a FluxObs needs at least one of `flux` (point estimate) or `bounds`",
            ));
        }
        Ok(Self(FluxObs::from_parts(
            bounds,
            point,
            band.into_band_info()?,
            sun2obj.into_vector(PyFrames::Ecliptic).into(),
            sun2obs.into_vector(PyFrames::Ecliptic).into(),
            epoch,
        )))
    }
}

#[pymethods]
impl PyFluxObs {
    #[new]
    #[pyo3(signature = (band, sun2obj, sun2obs, flux=None, sigma=None, sigma_hi=None, bounds=None, is_upper_limit=false, epoch=0.0))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        band: BandArg,
        sun2obj: VectorLike,
        sun2obs: VectorLike,
        flux: Option<f64>,
        sigma: Option<f64>,
        sigma_hi: Option<f64>,
        bounds: Option<(f64, f64)>,
        is_upper_limit: bool,
        epoch: f64,
    ) -> PyResult<Self> {
        Self::build(
            flux,
            sigma,
            sigma_hi,
            bounds,
            is_upper_limit,
            band,
            sun2obj,
            sun2obs,
            epoch,
        )
    }

    /// A two-sided flux detection ``flux +/- sigma`` (asymmetric if ``sigma_hi``).
    #[staticmethod]
    #[pyo3(signature = (flux, sigma, band, sun2obj, sun2obs, sigma_hi=None, epoch=0.0))]
    fn detection(
        flux: f64,
        sigma: f64,
        band: BandArg,
        sun2obj: VectorLike,
        sun2obs: VectorLike,
        sigma_hi: Option<f64>,
        epoch: f64,
    ) -> PyResult<Self> {
        Ok(Self(FluxObs::detection_asym(
            flux,
            sigma,
            sigma_hi.unwrap_or(sigma),
            band.into_band_info()?,
            sun2obj.into_vector(PyFrames::Ecliptic).into(),
            sun2obs.into_vector(PyFrames::Ecliptic).into(),
            epoch,
        )))
    }

    /// A soft photometric upper limit: a non-detection at ``threshold`` with
    /// noise scale ``sigma``.
    #[staticmethod]
    #[pyo3(signature = (threshold, sigma, band, sun2obj, sun2obs, epoch=0.0))]
    fn upper_limit(
        threshold: f64,
        sigma: f64,
        band: BandArg,
        sun2obj: VectorLike,
        sun2obs: VectorLike,
        epoch: f64,
    ) -> PyResult<Self> {
        Ok(Self(FluxObs::upper_limit(
            threshold,
            sigma,
            band.into_band_info()?,
            sun2obj.into_vector(PyFrames::Ecliptic).into(),
            sun2obs.into_vector(PyFrames::Ecliptic).into(),
            epoch,
        )))
    }

    /// A hard flux interval ``[lo, hi]`` with no point estimate.
    #[staticmethod]
    #[pyo3(signature = (lo, hi, band, sun2obj, sun2obs, epoch=0.0))]
    fn bounded(
        lo: f64,
        hi: f64,
        band: BandArg,
        sun2obj: VectorLike,
        sun2obs: VectorLike,
        epoch: f64,
    ) -> PyResult<Self> {
        Self::check_bounds(Some((lo, hi)))?;
        Ok(Self(FluxObs::bounded(
            lo,
            hi,
            band.into_band_info()?,
            sun2obj.into_vector(PyFrames::Ecliptic).into(),
            sun2obs.into_vector(PyFrames::Ecliptic).into(),
            epoch,
        )))
    }

    /// Point-estimate flux in Jy, or ``None`` for a bounds-only constraint.
    #[getter]
    fn flux(&self) -> Option<f64> {
        self.0.point_estimate()
    }

    /// Lower-side 1-sigma uncertainty in Jy. ``None`` if no point estimate, or
    /// for an upper limit (whose constrained side is ``sigma_hi``).
    #[getter]
    fn sigma(&self) -> Option<f64> {
        self.0.sigma_lo()
    }

    /// Upper-side 1-sigma uncertainty in Jy, or ``None`` if no point estimate.
    #[getter]
    fn sigma_hi(&self) -> Option<f64> {
        self.0.sigma_hi()
    }

    /// Hard ``(lo, hi)`` flux interval in Jy, or ``None`` if unbounded.
    #[getter]
    fn bounds(&self) -> Option<(f64, f64)> {
        self.0.bounds()
    }

    /// Band wavelength in nm.
    #[getter]
    fn wavelength(&self) -> f64 {
        self.0.band.wavelength
    }

    /// Whether this is a non-detection upper limit.
    #[getter]
    fn is_upper_limit(&self) -> bool {
        self.0.is_upper_limit()
    }

    /// Observation time (Julian date).
    #[getter]
    fn epoch(&self) -> f64 {
        self.0.epoch
    }

    /// Sun-to-object vector in AU (Ecliptic frame), as ``[x, y, z]``.
    #[getter]
    fn sun2obj(&self) -> [f64; 3] {
        self.0.sun2obj.into()
    }

    /// Sun-to-observer vector in AU (Ecliptic frame), as ``[x, y, z]``.
    #[getter]
    fn sun2obs(&self) -> [f64; 3] {
        self.0.sun2obs.into()
    }

    fn __repr__(&self) -> String {
        let mut parts = Vec::new();
        if let Some(mean) = self.0.point_estimate() {
            match (self.0.sigma_lo(), self.0.sigma_hi()) {
                (Some(lo), Some(hi)) if lo == hi => {
                    parts.push(format!("flux={mean:.4e}, sigma={lo:.4e}"));
                }
                (Some(lo), Some(hi)) => {
                    parts.push(format!("flux={mean:.4e}, sigma={lo:.4e}, sigma_hi={hi:.4e}"));
                }
                // Upper limit: only the above-threshold side is constrained;
                // label matches the getter that returns it.
                (None, Some(hi)) => parts.push(format!("threshold={mean:.4e}, sigma_hi={hi:.4e}")),
                (Some(lo), None) => parts.push(format!("flux={mean:.4e}, sigma_lo={lo:.4e}")),
                (None, None) => parts.push(format!("flux={mean:.4e}")),
            }
        }
        if let Some((lo, hi)) = self.0.bounds() {
            parts.push(format!("bounds=({lo:.4e}, {hi:.4e})"));
        }
        format!(
            "FluxObs({}, wavelength={:.1}, upper_limit={})",
            parts.join(", "),
            self.0.band.wavelength,
            self.0.is_upper_limit()
        )
    }
}

/// Prior configuration for model fitting.
/// Configuration for a single parameter's prior.
///
/// Parameters
/// ----------
/// bounds :
///     ``(low, high)`` logistic-barrier hard bounds.
/// gaussian :
///     Optional Gaussian centering prior. Either ``(mean, sigma)`` for a
///     symmetric prior, or ``(mean, sigma_lo, sigma_hi)`` for an asymmetric one
///     (``sigma_lo`` is the 1-sigma width below ``mean``, ``sigma_hi`` above).
///     ``None`` means flat (uniform) within the bounds.
#[pyclass(frozen, module = "kete.flux", name = "ParamPrior", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct PyParamPrior(pub ParamPrior);

impl<'a, 'py> FromPyObject<'a, 'py> for PyParamPrior {
    type Error = PyErr;
    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        // Existing ParamPrior instance.
        if let Ok(pp) = ob.cast::<PyParamPrior>() {
            return Ok(pp.get().clone());
        }
        // 5-tuple (low, high, mean, sigma_lo, sigma_hi) -> bounds + asym gaussian.
        if let Ok((low, high, mean, sigma_lo, sigma_hi)) = ob.extract::<(f64, f64, f64, f64, f64)>()
        {
            return Ok(Self(ParamPrior::with_gaussian_asym(
                low, high, mean, sigma_lo, sigma_hi,
            )));
        }
        // 4-tuple (low, high, mean, std) -> bounds + symmetric gaussian.
        if let Ok((low, high, mean, sigma)) = ob.extract::<(f64, f64, f64, f64)>() {
            return Ok(Self(ParamPrior::with_gaussian(low, high, mean, sigma)));
        }
        // 2-tuple (low, high) -> bounds only.
        if let Ok((low, high)) = ob.extract::<(f64, f64)>() {
            return Ok(Self(ParamPrior::bounds_only(low, high)));
        }
        Err(pyo3::exceptions::PyTypeError::new_err(
            "Expected a ParamPrior, a 2-tuple (low, high), a 4-tuple \
             (low, high, mean, std), or a 5-tuple (low, high, mean, sigma_lo, sigma_hi)",
        ))
    }
}

/// Gaussian centering argument: symmetric `(mean, sigma)` or asymmetric
/// `(mean, sigma_lo, sigma_hi)`.
#[derive(FromPyObject)]
enum GaussianArg {
    Sym((f64, f64)),
    Asym((f64, f64, f64)),
}

#[pymethods]
impl PyParamPrior {
    #[new]
    #[pyo3(signature = (bounds, gaussian=None))]
    fn new(bounds: (f64, f64), gaussian: Option<GaussianArg>) -> Self {
        Self(match gaussian {
            Some(GaussianArg::Sym((mean, sigma))) => {
                ParamPrior::with_gaussian(bounds.0, bounds.1, mean, sigma)
            }
            Some(GaussianArg::Asym((mean, sigma_lo, sigma_hi))) => {
                ParamPrior::with_gaussian_asym(bounds.0, bounds.1, mean, sigma_lo, sigma_hi)
            }
            None => ParamPrior::bounds_only(bounds.0, bounds.1),
        })
    }

    #[getter]
    fn bounds(&self) -> (f64, f64) {
        self.0.bounds
    }

    /// Gaussian center as ``(mean, sigma_lo, sigma_hi)``, or ``None``.
    #[getter]
    fn gaussian(&self) -> Option<(f64, f64, f64)> {
        self.0.gaussian
    }

    fn __repr__(&self) -> String {
        let (low, high) = self.0.bounds;
        match self.0.gaussian {
            // Collapse to the (mean, sigma) form when symmetric.
            Some((mean, sigma_lo, sigma_hi)) if sigma_lo == sigma_hi => {
                format!("ParamPrior(bounds=({low}, {high}), gaussian=({mean}, {sigma_lo}))")
            }
            Some((mean, sigma_lo, sigma_hi)) => format!(
                "ParamPrior(bounds=({low}, {high}), gaussian=({mean}, {sigma_lo}, {sigma_hi}))"
            ),
            None => format!("ParamPrior(bounds=({low}, {high}))"),
        }
    }
}

/// Scale a [`ParamPrior`]'s bounds and Gaussian center by `factor` (unit conversion).
fn scale_prior(p: &ParamPrior, factor: f64) -> ParamPrior {
    ParamPrior {
        bounds: (p.bounds.0 * factor, p.bounds.1 * factor),
        // (mean, sigma_lo, sigma_hi) since the asymmetric-error refactor; a unit
        // conversion scales all three.
        gaussian: p
            .gaussian
            .map(|(m, s_lo, s_hi)| (m * factor, s_lo * factor, s_hi * factor)),
    }
}

/// Convert a roughness prior specified in degrees to the internal radian units.
fn deg_prior_to_rad(p: &ParamPrior) -> ParamPrior {
    scale_prior(p, std::f64::consts::PI / 180.0)
}

/// Convert an internal radian roughness prior back to degrees for display.
fn rad_prior_to_deg(p: &ParamPrior) -> ParamPrior {
    scale_prior(p, 180.0 / std::f64::consts::PI)
}

/// Priors used in thermal model fitting, below is the list of default priors.
///
/// Priors can be specified either using the ``FluxPriors`` constructor or by directly
/// passing a tuple of the form ``(low, high)`` for bounds-only or
/// ``(low, high, mean, sigma)`` for bounds + Gaussian. Below we see the defaults
/// being specified using the tuple form.
///
/// Not all of these priors are used in every model, this is a comprehensive list for
/// all models.
///
/// If not provided, defaults are used, specifying a prior will overwrite the specified
/// priors while leaving the others at their defaults:
///
/// .. code-block:: python
///
///     kete.flux.FluxPriors(
///         diameter   = (0.001, 1000),
///         beaming    = (0.5, 3.0, 1.0, 0.3),
///         r_ir       = (0.5, 2.0, 1.6, 0.3),
///         h_mag      = (-5.0, 35.0),
///         g_param    = (-0.3, 0.7, 0.2, 0.05),
///         vis_albedo = (0.01, 1.0),
///     )
///
/// Each prior is a :class:`ParamPrior` specifying ``bounds`` (logistic
/// barrier) and an optional ``gaussian`` centering prior ``(mean, sigma)``.
/// To effectively fix a parameter, set tight bounds around the desired
/// value (e.g., ``bounds=(val - 1e-3, val + 1e-3)``).
///
/// Parameters
/// ----------
/// diameter :
///     :class:`ParamPrior` for diameter D in km.
/// beaming :
///     :class:`ParamPrior` for beaming parameter.
/// thermal_inertia :
///     :class:`ParamPrior` for thermal inertia Gamma (SI units). Used by TPM. Gamma is
///     sampled and prior'd in log space: ``bounds`` are linear Gamma but the prior is
///     log-uniform within them. An optional ``gaussian`` center is interpreted in log
///     space -- ``mean`` is a linear Gamma (the lognormal median) and the sigma is a
///     multiplicative / log-width, not a linear standard deviation.
/// r_ir :
///     :class:`ParamPrior` for IR-to-visible albedo ratio R_IR.
/// h_mag :
///     :class:`ParamPrior` for H magnitude.
/// g_param :
///     :class:`ParamPrior` for G parameter.
/// vis_albedo :
///     :class:`ParamPrior` for visible geometric albedo.
#[pyclass(frozen, module = "kete.flux", name = "FluxPriors", from_py_object)]
#[derive(Clone, Debug)]
pub struct PyFluxPriors(pub FluxPriors);

#[pymethods]
impl PyFluxPriors {
    #[new]
    #[pyo3(signature = (
        diameter=None,
        beaming=None,
        thermal_inertia=None,
        roughness=None,
        r_ir=None,
        h_mag=None,
        g_param=None,
        vis_albedo=None,
        c_a=None,
        b_a=None,
        phase0=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        diameter: Option<PyParamPrior>,
        beaming: Option<PyParamPrior>,
        thermal_inertia: Option<PyParamPrior>,
        roughness: Option<PyParamPrior>,
        r_ir: Option<PyParamPrior>,
        h_mag: Option<PyParamPrior>,
        g_param: Option<PyParamPrior>,
        vis_albedo: Option<PyParamPrior>,
        c_a: Option<PyParamPrior>,
        b_a: Option<PyParamPrior>,
        phase0: Option<PyParamPrior>,
    ) -> Self {
        let d = FluxPriors::default();
        Self(FluxPriors {
            diameter: diameter.map_or(d.diameter, |p| p.0),
            beaming: beaming.map_or(d.beaming, |p| p.0),
            thermal_inertia: thermal_inertia.map_or(d.thermal_inertia, |p| p.0),
            // The roughness prior is given in degrees but stored internally in radians.
            roughness: roughness.map_or(d.roughness, |p| deg_prior_to_rad(&p.0)),
            r_ir: r_ir.map_or(d.r_ir, |p| p.0),
            h_mag: h_mag.map_or(d.h_mag, |p| p.0),
            g_param: g_param.map_or(d.g_param, |p| p.0),
            vis_albedo: vis_albedo.map_or(d.vis_albedo, |p| p.0),
            f_sigma: d.f_sigma,
            c_a: c_a.map_or(d.c_a, |p| p.0),
            b_a: b_a.map_or(d.b_a, |p| p.0),
            // The phase0 prior is given in degrees but stored internally in radians.
            phase0: phase0.map_or(d.phase0, |p| deg_prior_to_rad(&p.0)),
        })
    }

    fn __repr__(&self) -> String {
        let p = &self.0;
        format!(
            "FluxPriors(\n  diameter={},\n  beaming={},\n  r_ir={},\n  h_mag={},\n  g_param={},\n  vis_albedo={})",
            PyParamPrior(p.diameter.clone()).__repr__(),
            PyParamPrior(p.beaming.clone()).__repr__(),
            PyParamPrior(p.r_ir.clone()).__repr__(),
            PyParamPrior(p.h_mag.clone()).__repr__(),
            PyParamPrior(p.g_param.clone()).__repr__(),
            PyParamPrior(p.vis_albedo.clone()).__repr__(),
        )
    }

    #[getter]
    fn diameter(&self) -> PyParamPrior {
        PyParamPrior(self.0.diameter.clone())
    }

    #[getter]
    fn beaming(&self) -> PyParamPrior {
        PyParamPrior(self.0.beaming.clone())
    }

    #[getter]
    fn thermal_inertia(&self) -> PyParamPrior {
        PyParamPrior(self.0.thermal_inertia.clone())
    }

    /// Roughness prior (mean slope angle), returned in degrees (stored internally in
    /// radians).
    #[getter]
    fn roughness(&self) -> PyParamPrior {
        PyParamPrior(rad_prior_to_deg(&self.0.roughness))
    }

    #[getter]
    fn r_ir(&self) -> PyParamPrior {
        PyParamPrior(self.0.r_ir.clone())
    }

    #[getter]
    fn h_mag(&self) -> PyParamPrior {
        PyParamPrior(self.0.h_mag.clone())
    }

    #[getter]
    fn g_param(&self) -> PyParamPrior {
        PyParamPrior(self.0.g_param.clone())
    }

    #[getter]
    fn vis_albedo(&self) -> PyParamPrior {
        PyParamPrior(self.0.vis_albedo.clone())
    }

    #[getter]
    fn c_a(&self) -> PyParamPrior {
        PyParamPrior(self.0.c_a.clone())
    }

    #[getter]
    fn b_a(&self) -> PyParamPrior {
        PyParamPrior(self.0.b_a.clone())
    }

    #[getter]
    fn phase0(&self) -> PyParamPrior {
        PyParamPrior(rad_prior_to_deg(&self.0.phase0))
    }
}

/// Build a [`PyData`] from a single column of non-divergent MCMC draws.
fn stats_from_column(draws: &[Vec<f64>], divergent: &[bool], col: usize) -> PyData {
    let vals: Vec<f64> = draws
        .iter()
        .zip(divergent.iter())
        .filter(|(_, div)| !**div)
        .map(|(r, _)| r[col])
        .collect();
    let data =
        Data::try_from(vals).expect("non-divergent draws should be non-empty with finite values");
    PyData(data)
}

/// Full MCMC fitting result.
#[pyclass(frozen, module = "kete.flux", name = "FitResult", from_py_object)]
#[derive(Clone, Debug)]
pub struct PyFitResult(pub FitResult);

impl PyFitResult {
    /// Posterior stats for a draw column looked up by name, or `None` if absent.
    fn column_stats(&self, name: &str) -> Option<PyData> {
        self.columns()
            .iter()
            .position(|c| *c == name)
            .map(|col| stats_from_column(&self.0.draws, &self.0.divergent, col))
    }
}

#[pymethods]
impl PyFitResult {
    /// Posterior statistics for diameter (km).  ``None`` for HG model.
    /// Only non-divergent draws are included.
    #[getter]
    fn diameter(&self) -> Option<PyData> {
        (!self.0.model.is_hg()).then(|| stats_from_column(&self.0.draws, &self.0.divergent, 0))
    }

    /// Posterior statistics for visual geometric albedo.  ``None`` for HG model.
    /// Only non-divergent draws are included.
    #[getter]
    fn vis_albedo(&self) -> Option<PyData> {
        (!self.0.model.is_hg()).then(|| stats_from_column(&self.0.draws, &self.0.divergent, 1))
    }

    /// Posterior statistics for beaming (NEATM only).  ``None`` for FRM/HG/TPM.
    /// Only non-divergent draws are included.
    #[getter]
    fn beaming(&self) -> Option<PyData> {
        self.0
            .model
            .is_neatm()
            .then(|| stats_from_column(&self.0.draws, &self.0.divergent, 2))
    }

    /// Posterior statistics for thermal inertia (TPM only).  ``None`` otherwise.
    /// Only non-divergent draws are included.
    #[getter]
    fn thermal_inertia(&self) -> Option<PyData> {
        self.0
            .model
            .is_tpm()
            .then(|| stats_from_column(&self.0.draws, &self.0.divergent, 2))
    }

    /// Posterior statistics for surface roughness as the mean slope angle in degrees.
    /// ``None`` unless the fit used ``fit_roughness=True``.
    /// Only non-divergent draws are included.
    #[getter]
    fn roughness(&self) -> Option<PyData> {
        self.0
            .model
            .fits_roughness()
            .then(|| stats_from_column(&self.0.draws, &self.0.divergent, 3))
    }

    /// Posterior statistics for the axis ratio ``c/a``.  ``None`` unless the fit fit it
    /// (``fit_oblate`` or ``fit_triaxial``).  Only non-divergent draws are included.
    #[getter]
    fn c_a(&self) -> Option<PyData> {
        self.column_stats("c_a")
    }

    /// Posterior statistics for the axis ratio ``b/a``.  ``None`` unless the fit used
    /// ``fit_triaxial=True``.  Only non-divergent draws are included.
    #[getter]
    fn b_a(&self) -> Option<PyData> {
        self.column_stats("b_a")
    }

    /// Posterior statistics for the rotation phase ``phase0`` in degrees.  ``None``
    /// unless the fit used ``fit_triaxial=True``.  Only non-divergent draws are included.
    #[getter]
    fn phase0(&self) -> Option<PyData> {
        self.column_stats("phase0")
    }

    /// Posterior statistics for H magnitude.
    /// Only non-divergent draws are included.
    #[getter]
    fn h_mag(&self) -> PyData {
        let col = match self.0.model {
            Model::TpmRough => 4,
            Model::Neatm | Model::Tpm => 3,
            Model::Frm => 2,
            Model::Hg => 0,
        };
        stats_from_column(&self.0.draws, &self.0.divergent, col)
    }

    /// Posterior statistics for G parameter.
    /// Only non-divergent draws are included.
    #[getter]
    fn g_param(&self) -> PyData {
        let col = match self.0.model {
            Model::TpmRough => 5,
            Model::Neatm | Model::Tpm => 4,
            Model::Frm => 3,
            Model::Hg => 1,
        };
        stats_from_column(&self.0.draws, &self.0.divergent, col)
    }

    /// Posterior statistics for uncertainty inflation factor.
    /// Only non-divergent draws are included.
    #[getter]
    fn f_sigma(&self) -> PyData {
        let col = match self.0.model {
            Model::TpmRough => 7,
            Model::Neatm | Model::Tpm => 6,
            Model::Frm => 5,
            Model::Hg => 2,
        };
        stats_from_column(&self.0.draws, &self.0.divergent, col)
    }

    /// Posterior statistics for IR-to-visible albedo ratio.  ``None`` for HG model.
    /// Only non-divergent draws are included.
    #[getter]
    fn ir_albedo_ratio(&self) -> Option<PyData> {
        if self.0.model.is_hg() {
            return None;
        }
        let col = match self.0.model {
            Model::TpmRough => 6,
            Model::Neatm | Model::Tpm => 5,
            // FRM (HG already returned above).
            _ => 4,
        };
        Some(stats_from_column(&self.0.draws, &self.0.divergent, col))
    }

    /// Model name (``"Neatm"``, ``"Frm"``, or ``"Hg"``).
    #[getter]
    fn model(&self) -> String {
        format!("{:?}", self.0.model)
    }

    /// Non-divergent MCMC posterior draws.  Each row is one sample; column layout
    /// depends on the model (see :attr:`columns`).
    #[getter]
    fn draws(&self) -> Vec<Vec<f64>> {
        self.0
            .draws
            .iter()
            .zip(self.0.divergent.iter())
            .filter(|(_, div)| !**div)
            .map(|(row, _)| row.clone())
            .collect()
    }

    /// All MCMC posterior draws including divergent transitions.
    /// Use :attr:`divergent` to identify which rows diverged.
    #[getter]
    fn draws_all(&self) -> Vec<Vec<f64>> {
        self.0.draws.clone()
    }

    /// Column names for each element of a draw vector.
    #[getter]
    fn columns(&self) -> Vec<&'static str> {
        let mut cols = match self.0.model {
            Model::Neatm => vec![
                "diameter",
                "vis_albedo",
                "beaming",
                "h_mag",
                "g_param",
                "ir_albedo_ratio",
                "f_sigma",
            ],
            Model::Tpm => vec![
                "diameter",
                "vis_albedo",
                "thermal_inertia",
                "h_mag",
                "g_param",
                "ir_albedo_ratio",
                "f_sigma",
            ],
            Model::TpmRough => vec![
                "diameter",
                "vis_albedo",
                "thermal_inertia",
                "roughness",
                "h_mag",
                "g_param",
                "ir_albedo_ratio",
                "f_sigma",
            ],
            Model::Frm => vec![
                "diameter",
                "vis_albedo",
                "h_mag",
                "g_param",
                "ir_albedo_ratio",
                "f_sigma",
            ],
            Model::Hg => vec!["h_mag", "g_param", "f_sigma"],
        };
        // Fitted shape/phase extras are appended last (see `Model::draw_column_names`).
        if self.0.model.is_tpm() {
            if self.0.fit_c_a {
                cols.push("c_a");
            }
            if self.0.fit_b_a {
                cols.push("b_a");
            }
            if self.0.fit_phase0 {
                cols.push("phase0");
            }
        }
        cols
    }
    /// Per-draw divergence flags from the NUTS sampler.
    #[getter]
    fn divergent(&self) -> Vec<bool> {
        self.0.divergent.clone()
    }

    /// Total number of divergent transitions.
    #[getter]
    fn n_divergent(&self) -> usize {
        self.0.n_divergent
    }

    /// Reduced chi-squared at the MAP point using inflated uncertainties.
    #[getter]
    fn chi2_best(&self) -> f64 {
        self.0.reduced_chi2
    }

    /// Number of non-upper-limit observations.
    #[getter]
    fn nobs(&self) -> usize {
        self.0.nobs
    }

    /// Model fluxes at the MAP point for each observation (Jy).
    #[getter]
    fn best_fit_fluxes(&self) -> Vec<f64> {
        self.0.best_fit_fluxes.clone()
    }

    /// Standardized residuals ``(obs - model) / (f_sigma * sigma)`` at the MAP,
    /// using the residual side's sigma. ``0.0`` where undefined: bounds-only
    /// constraints, and one-sided limits whose model flux sits on the
    /// unconstrained side (e.g. below an upper-limit threshold).
    #[getter]
    fn best_fit_residuals(&self) -> Vec<f64> {
        self.0.best_fit_residuals.clone()
    }
    /// Reflected-light fraction at the MAP point, one per observation.
    #[getter]
    fn best_fit_reflected_frac(&self) -> Vec<f64> {
        self.0.best_fit_reflected_frac.clone()
    }

    fn __repr__(&self) -> String {
        let div = &self.0.divergent;
        let n_good = div.iter().filter(|&&d| !d).count();
        if self.0.model.is_hg() {
            let h = stats_from_column(&self.0.draws, div, 0);
            let g = stats_from_column(&self.0.draws, div, 1);
            return format!(
                "FitResult(model=Hg, h_mag={}, g_param={}, n_draws={}, n_divergent={})",
                h, g, n_good, self.0.n_divergent,
            );
        }
        let d = stats_from_column(&self.0.draws, div, 0);
        let pv = stats_from_column(&self.0.draws, div, 1);
        // NEATM and TPM carry a third parameter (beaming / thermal inertia) at col 2;
        // TpmRough additionally carries roughness (degrees) at col 3.
        let (extra_str, h_col) = if self.0.model.is_neatm() {
            let b = stats_from_column(&self.0.draws, div, 2);
            (format!("\n  beaming={b},"), 3)
        } else if self.0.model.fits_roughness() {
            let ti = stats_from_column(&self.0.draws, div, 2);
            let r = stats_from_column(&self.0.draws, div, 3);
            (format!("\n  thermal_inertia={ti},\n  roughness={r},"), 4)
        } else if self.0.model.is_tpm() {
            let ti = stats_from_column(&self.0.draws, div, 2);
            (format!("\n  thermal_inertia={ti},"), 3)
        } else {
            (String::new(), 2)
        };
        let h = stats_from_column(&self.0.draws, div, h_col);
        let g = stats_from_column(&self.0.draws, div, h_col + 1);
        let rir = stats_from_column(&self.0.draws, div, h_col + 2);
        // Fitted shape/phase extras (c_a, b_a, phase0) appended in column order.
        let mut shape_str = String::new();
        for name in ["c_a", "b_a", "phase0"] {
            if let Some(stat) = self.column_stats(name) {
                shape_str.push_str(&format!("\n  {name}={stat},"));
            }
        }

        format!(
            "FitResult(model={:?},\n  diameter={d},\n  vis_albedo={pv},{extra_str}\n  h_mag={h},\n  g_param={g},\n  ir_albedo_ratio={rir},{shape_str}\n  n_draws={}, n_divergent={})",
            self.0.model, n_good, self.0.n_divergent,
        )
    }
}

fn extract_obs(obs: &[PyFluxObs]) -> Vec<FluxObs> {
    obs.iter().map(|o| o.0.clone()).collect()
}

fn parse_model(model: &str) -> PyResult<Model> {
    match model.to_lowercase().as_str() {
        "neatm" => Ok(Model::Neatm),
        "frm" => Ok(Model::Frm),
        "hg" => Ok(Model::Hg),
        "tpm" => Ok(Model::Tpm),
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Unknown model '{other}'. Use 'neatm', 'frm', 'hg', or 'tpm'."
        ))),
    }
}

/// Fit a model to observations using NUTS MCMC.
///
/// H magnitude and G parameter are fitted as free parameters.
/// If ``h_mag`` or ``g_param`` are supplied, they set the center of the
/// corresponding Gaussian prior (keeping its width from ``priors``).
/// ``emissivity`` is a fixed thermal property (not fitted).
///
/// Parameters
/// ----------
/// model :
///     Model name: ``"neatm"``, ``"frm"``, ``"hg"``, or ``"tpm"``.
/// obs :
///     List of :class:`FluxObs` observations.
/// h_mag :
///     Optional H magnitude -- sets the center of the H prior.
/// g_param :
///     Optional G parameter -- sets the center of the G prior.
/// period :
///     Rotation period in seconds. Required for the ``"tpm"`` model (spin is a
///     fixed input, not fitted); ignored otherwise.
/// pole :
///     Spin axis direction (ecliptic). Required for the ``"tpm"`` model; ignored
///     otherwise.
/// axis_ratios :
///     Optional ``(b/a, c/a)`` semi-axis ratios for an ellipsoidal TPM shape (``a``
///     is the long axis, ``c`` is along the pole). ``None`` (default) is a sphere.
///     A triaxial shape (``b/a != 1``) makes the flux depend on rotation phase, so
///     each :class:`FluxObs` should carry a meaningful ``epoch``.
/// phase0 :
///     TPM rotation phase (radians) at ``epoch0``. Defaults to ``0``.
/// epoch0 :
///     TPM reference epoch (Julian date) for ``phase0``. Defaults to ``0``.
/// roughness :
///     Optional surface roughness for the ``"tpm"`` model, as the mean slope angle
///     ``theta_bar`` in degrees ``(0, 57.3]`` (the cross-model roughness convention).
///     Fixed (not fitted) unless ``fit_roughness`` is set. Applied via the shipped
///     roughness correction table.
/// fit_roughness :
///     If ``True`` (``"tpm"`` model only), fit the surface roughness (mean slope angle)
///     as a free parameter instead of fixing it. Adds a ``roughness`` column (degrees)
///     to the posterior. Roughness is partially degenerate with thermal inertia, so
///     multi-band and a spread of solar phase angles are needed to separate them; the
///     ``roughness`` prior (degrees) configures its range.
/// fit_oblate :
///     If ``True`` (``"tpm"`` model only), fit the oblate axis ratio ``c/a`` (with
///     ``b/a = 1``) as a free parameter. Adds a ``c_a`` column to the posterior.
///     Cannot be combined with a fixed ``axis_ratios``. ``c/a`` is constrained by
///     aspect change across epochs, so a single-aspect dataset leaves it degenerate
///     with diameter; the ``c_a`` prior configures its range.
/// fit_triaxial :
///     If ``True`` (``"tpm"`` model only), fit a triaxial shape: the axis ratios
///     ``b/a`` and ``c/a`` plus the rotation phase ``phase0`` (degrees). Adds ``c_a``,
///     ``b_a``, and ``phase0`` columns. Mutually exclusive with ``fit_oblate`` and a
///     fixed ``axis_ratios``. ``b/a`` and ``phase0`` are constrained by the rotational
///     lightcurve, so observations should span rotation phases (meaningful ``epoch``);
///     ``phase0`` is restricted to ``[0, 180)`` degrees (an ellipsoid is point-symmetric).
/// emissivity :
///     Fixed thermal emissivity (default 0.9, not fitted).
/// priors :
///     Prior configuration (:class:`FluxPriors`, default if ``None``).
/// num_chains :
///     Number of MCMC chains (default 10).
/// num_tune :
///     Warmup draws per chain (default 200).
/// num_draws :
///     Posterior draws per chain (default 500).
/// c_hg :
///     HG relationship constant (default 1329.0).
///
/// Returns
/// -------
/// FitResult
///     MCMC posterior results.
///
/// Raises
/// ------
/// ValueError
///     If the fit fails to converge.
#[pyfunction]
#[pyo3(name = "fit_model", signature = (model, obs, h_mag=None, g_param=None,
    period=None, pole=None, axis_ratios=None, phase0=0.0, epoch0=0.0,
    roughness=None, fit_roughness=false, fit_oblate=false, fit_triaxial=false,
    emissivity=0.9, priors=None,
    num_chains=10, num_tune=200, num_draws=500, c_hg=None))]
#[allow(clippy::too_many_arguments)]
pub fn fit_model_py(
    model: &str,
    obs: Vec<PyFluxObs>,
    h_mag: Option<f64>,
    g_param: Option<f64>,
    period: Option<f64>,
    pole: Option<VectorLike>,
    axis_ratios: Option<(f64, f64)>,
    phase0: f64,
    epoch0: f64,
    roughness: Option<f64>,
    fit_roughness: bool,
    fit_oblate: bool,
    fit_triaxial: bool,
    emissivity: f64,
    priors: Option<PyFluxPriors>,
    num_chains: usize,
    num_tune: usize,
    num_draws: usize,
    c_hg: Option<f64>,
) -> PyResult<PyFitResult> {
    let tm = parse_model(model)?;
    // Fitting roughness only applies to the TPM model; it upgrades to the variant
    // that carries a free crater half-angle.
    if fit_roughness && !tm.is_tpm() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "fit_roughness=True is only valid for the 'tpm' model.",
        ));
    }
    // Fitting shape/phase parameters applies only to the TPM model. `fit_oblate` fits
    // c/a (b/a = 1); `fit_triaxial` fits b/a, c/a, and phase0. Neither can combine with
    // a fixed axis_ratios, and they are mutually exclusive.
    if (fit_oblate || fit_triaxial) && !tm.is_tpm() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "fit_oblate / fit_triaxial are only valid for the 'tpm' model.",
        ));
    }
    if fit_oblate && fit_triaxial {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "fit_oblate and fit_triaxial are mutually exclusive.",
        ));
    }
    if (fit_oblate || fit_triaxial) && axis_ratios.is_some() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "fit_oblate / fit_triaxial fit the axis ratios; do not also pass a fixed \
             axis_ratios.",
        ));
    }
    // Decode the shape/phase flags: oblate -> c/a only; triaxial -> b/a, c/a, phase0.
    let (fit_c_a, fit_b_a, fit_phase0) = (fit_oblate || fit_triaxial, fit_triaxial, fit_triaxial);
    let tm = if fit_roughness { Model::TpmRough } else { tm };
    let mut priors = priors.map_or_else(FluxPriors::default, |p| p.0);
    let c_hg_val = c_hg.unwrap_or(kete_core::constants::C_V);

    // Override prior centers from convenience arguments, preserving any
    // existing (possibly asymmetric) widths.
    if let Some(h) = h_mag {
        let (lo, hi) = priors.h_mag.gaussian.map_or((0.25, 0.25), |(_, lo, hi)| (lo, hi));
        priors.h_mag.gaussian = Some((h, lo, hi));
    }
    if let Some(g) = g_param {
        let (lo, hi) = priors.g_param.gaussian.map_or((0.05, 0.05), |(_, lo, hi)| (lo, hi));
        priors.g_param.gaussian = Some((g, lo, hi));
    }

    let raw_obs = extract_obs(&obs);

    // The TPM model needs a spin state, a shape, and a precomputed field grid.
    let tpm_config = if tm.is_tpm() {
        Some(build_tpm_config(
            &raw_obs,
            period,
            pole,
            axis_ratios,
            phase0,
            epoch0,
            roughness,
            tm.fits_roughness(),
            (fit_c_a, fit_b_a, fit_phase0),
            emissivity,
            &priors,
        )?)
    } else {
        None
    };

    Ok(PyFitResult(fitting::fit_mcmc(
        tm,
        &raw_obs,
        c_hg_val,
        emissivity,
        &priors,
        num_chains,
        num_tune,
        num_draws,
        tpm_config.as_ref(),
    )?))
}

/// Validate the Python-facing TPM spin and roughness arguments (units and
/// presence), then assemble the [`TpmConfig`] via [`TpmConfig::for_fit`], which
/// owns the grid sizing, shape construction, and roughness-table checks.
#[allow(
    clippy::too_many_arguments,
    reason = "TPM configuration has many inputs"
)]
fn build_tpm_config(
    obs: &[FluxObs],
    period: Option<f64>,
    pole: Option<VectorLike>,
    axis_ratios: Option<(f64, f64)>,
    phase0: f64,
    epoch0: f64,
    roughness: Option<f64>,
    fit_roughness: bool,
    shape_fit: (bool, bool, bool),
    emissivity: f64,
    priors: &FluxPriors,
) -> PyResult<TpmConfig> {
    let period = period.ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("The 'tpm' model requires a 'period' (seconds).")
    })?;
    if !(period.is_finite() && period > 0.0) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "period must be finite and positive for a TPM fit (use 'frm' for an \
             infinitely fast rotator).",
        ));
    }
    // Validate roughness up front (before the slower grid/table builds). Roughness is
    // the mean slope angle (degrees); converted to radians here and to the internal
    // crater opening half-angle at table construction.
    let mean_slope = match roughness {
        Some(deg) => {
            // The full-coverage cap tops out at the hemisphere limit ~57.3 deg; the
            // converter clamps the sub-0.01 deg rounding overshoot at the ceiling.
            if !(deg.is_finite() && deg > 0.0 && deg <= 57.3) {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "roughness must be a mean slope angle in (0, 57.3] degrees.",
                ));
            }
            Some(deg.to_radians())
        }
        None => None,
    };
    let pole = pole.ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("The 'tpm' model requires a 'pole' direction.")
    })?;
    let pole_vec = pole.into_vector(PyFrames::Ecliptic).into();
    let spin = SpinState {
        pole: UnitVector3::new_normalize(pole_vec),
        period,
        phase0,
        epoch0,
    };
    Ok(TpmConfig::for_fit(
        obs,
        spin,
        axis_ratios,
        mean_slope,
        fit_roughness,
        shape_fit,
        emissivity,
        priors,
    )?)
}
