"""
Thermal and Reflected light modeling tools.

This includes computations such as NEATM, FRM, HG, and reflection models.

Modeling is broken into categories of complexity, ranging from pure black body
calculations, through to telescope specific models. Picking the appropriate model can
save significant development time, but removes some of the control for the user.

For multi-band thermal + reflected light modeling, use :py:func:`neatm_model_flux` or
:py:func:`frm_model_flux`. These evaluate the model in parallel across multiple
geometries and return :py:class:`ModelResults` objects containing total, thermal,
and reflected fluxes.

Use :py:func:`resolve_hg_params` to compute any missing value from the
(H-mag, diameter, visible albedo) triad before calling the model functions.

If optical wavelengths are the goal, :py:func:`hg_apparent_mag` or
:py:func:`hg_apparent_flux` is probably appropriate.

There are a number of lower-level functions provided more for pedagogical reasons:

- :py:func:`lambertian_flux`
- :py:func:`frm_facet_temps`
- :py:func:`neatm_facet_temps`

"""

from ._core import (
    FitResult,
    FluxObs,
    FluxPriors,
    ModelResults,
    ParamPrior,
    SampleStats,
    black_body_flux,
    comet_apparent_mags,
    comet_dust_phase_curve_correction,
    fit_model,
    frm_facet_temps,
    frm_model_flux,
    hg_apparent_flux,
    hg_apparent_mag,
    hg_phase_curve_correction,
    lambertian_flux,
    neatm_facet_temps,
    neatm_model_flux,
    resolve_hg_params,
    solar_flux,
    sub_solar_temperature,
)

__all__ = [
    "black_body_flux",
    "comet_apparent_mags",
    "comet_dust_phase_curve_correction",
    "fit_model",
    "FitResult",
    "FluxObs",
    "FluxPriors",
    "frm_facet_temps",
    "frm_model_flux",
    "hg_apparent_flux",
    "hg_apparent_mag",
    "hg_phase_curve_correction",
    "lambertian_flux",
    "ModelResults",
    "neatm_facet_temps",
    "neatm_model_flux",
    "ParamPrior",
    "resolve_hg_params",
    "SampleStats",
    "solar_flux",
    "sub_solar_temperature",
]
