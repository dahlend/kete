"""
Thermal and Reflected light modeling tools.

This includes computations such as NEATM, FRM, and reflection models.

Modeling is broken into catagories of complexity, ranging from pure black body
calculations, through to telescope specific models. Picking the appropriate model can
save significant development time, but removes some of the control for the user.

An example of this is the :py:class:`NeatmParams` class, which defines a joint NEATM and
optical reflected light model. This can be broken up into distinct steps by the user,
however it provides convenience functions which are automatically parallelized, leading
to significant performance gains.

If you are interested in IR modeling, it is recommended to start with
:py:class:`NeatmParams` or :py:class:`FrmParams`. If optical wavelengths are the goal,
then starting with the :py:func:`hg_apparent_mag` or :py:func:`hg_apparent_flux` is
probably appropriate.

There are a number of functions provided more for pedagogical reasons, typically it is
recommended not to use these directly in most cases:
-:py:func:`frm_flux`
-:py:func:`neatm_flux`
-:py:func:`lambertian_flux`
-:py:func:`frm_facet_temps`
-:py:func:`neatm_facet_temps`

"""

from ._core import (
    FrmParams,
    ModelResults,
    NeatmParams,
    black_body_flux,
    comet_apparent_mags,
    comet_dust_phase_curve_correction,
    frm_facet_temps,
    frm_flux,
    hg_apparent_flux,
    hg_apparent_mag,
    hg_phase_curve_correction,
    lambertian_flux,
    neatm_facet_temps,
    neatm_flux,
    solar_flux,
    sub_solar_temperature,
)

__all__ = [
    "comet_apparent_mags",
    "comet_dust_phase_curve_correction",
    "frm_facet_temps",
    "frm_flux",
    "FrmParams",
    "hg_apparent_flux",
    "hg_apparent_mag",
    "hg_phase_curve_correction",
    "lambertian_flux",
    "ModelResults",
    "neatm_facet_temps",
    "neatm_flux",
    "NeatmParams",
    "solar_flux",
    "sub_solar_temperature",
    "black_body_flux",
]
