"""
Conversion functions between various physical values or representations.
"""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

from . import _core, constants
from ._core import (
    CometElements,
    compute_obliquity,
    dec_degrees_to_dms,
    dec_dms_to_degrees,
    earth_precession_rotation,
    ra_degrees_to_hms,
    ra_hms_to_degrees,
)

__all__ = [
    "bin_data",
    "compute_airmass",
    "compute_albedo",
    "compute_aphelion",
    "compute_diameter",
    "compute_earth_radius",
    "compute_eccentric_anomaly",
    "compute_h_mag",
    "compute_obliquity",
    "compute_semi_major",
    "compute_tisserand",
    "table_to_states",
    "dec_degrees_to_dms",
    "dec_dms_to_degrees",
    "earth_precession_rotation",
    "flux_to_mag",
    "mag_to_flux",
    "ra_degrees_to_hms",
    "ra_hms_to_degrees",
]


logger = logging.getLogger(__name__)


def compute_airmass(zenith_angle: float) -> float:
    """
    Compute the approximate air-mass above the observer given the zenith angle.

    Algorithm from:
    Andrew T. Young, "Air mass and refraction," Appl. Opt. 33, 1108-1110 (1994)

    Parameters
    ----------
    zenith_angle:
        True zenith angle in degrees.
    """
    cos_z = np.cos(np.radians(zenith_angle))
    num = 1.002432 * cos_z**2 + 0.148386 * cos_z + 0.0096467
    denominator = cos_z**3 + 0.149864 * cos_z**2 + 0.0102963 * cos_z + 0.000303978
    return num / denominator


def compute_h_mag(diameter: NDArray, albedo: NDArray, c_hg=constants.C_V) -> np.ndarray:
    """
    Compute the H magnitude of the object given the diameter in km and the albedo.

    Parameters
    ----------
    diameter:
        The diameter of the object in Km.
    albedo:
        The albedo of the object.
    """
    diameter = np.asarray(diameter, dtype=float)
    return -5.0 * np.log10(diameter * np.sqrt(albedo) / c_hg)


def compute_albedo(diameter: NDArray, h_mag: NDArray, c_hg=constants.C_V) -> np.ndarray:
    """
    Compute the albedo of the object given the diameter in km and H magnitude.

    Parameters
    ----------
    diameter:
        The diameter of the object in Km.
    h_mag:
        The H magnitude of the object.
    """
    return np.clip((c_hg * 10.0 ** (-0.2 * h_mag) / diameter) ** 2.0, 0, 1)


def compute_diameter(albedo: NDArray, h_mag: NDArray, c_hg=constants.C_V) -> np.ndarray:
    """
    Compute the diameter of the object in km given the albedo and H magnitude.

    Parameters
    ----------
    albedo:
        The albedo of the object.
    h_mag:
        The H magnitude of the object.
    """
    return (c_hg / np.sqrt(albedo)) * 10.0 ** (-0.2 * h_mag)


def compute_semi_major(peri_dist: NDArray, ecc: NDArray) -> np.ndarray:
    """
    Calculate semi major axis, returning nan when it cannot be computed.

    Parameters
    ----------
    peri_dist:
        Perihelion distance in units of AU.
    ecc:
        Eccentricity of the orbit.
    """
    return np.divide(
        peri_dist,
        (1.0 - ecc),
        out=np.full_like(ecc, np.nan, dtype=float),
        where=ecc < 1.0,
    )


def compute_aphelion(peri_dist: NDArray, ecc: NDArray) -> np.ndarray:
    """
    Calculate aphelion, returning nan when it cannot be computed.

    Parameters
    ----------
    peri_dist :
        Perihelion distance in units of AU.
    ecc :
        Eccentricity of the orbit.
    """
    return np.divide(
        peri_dist * (1.0 + ecc),
        (1.0 - ecc),
        out=np.full_like(ecc, np.nan, dtype=float),
        where=ecc < 1.0,
    )


def compute_earth_radius(geodetic_latitude: float) -> float:
    """
    Compute the effective Earth's radii at the specified geodetic latitude assuming that
    the Earth is an oblate spheroid.

    Returns the radii in units of AU.

    Parameters
    ----------
    geodetic_latitude:
        The geodetic latitude in degrees.
    """
    # https://en.wikipedia.org/wiki/Earth_radius#Geocentric_radius
    geodetic_latitude = np.radians(geodetic_latitude)
    a = constants.EARTH_MAJOR_AXIS_M
    b = constants.EARTH_MINOR_AXIS_M
    a_cos = a * np.cos(geodetic_latitude)
    b_sin = b * np.sin(geodetic_latitude)
    return (
        np.sqrt(((a * a_cos) ** 2 + (b * b_sin) ** 2) / (a_cos**2 + b_sin**2))
        / constants.AU_M
    )


def compute_eccentric_anomaly(
    eccentricity: NDArray, mean_anomaly: NDArray, peri_dist: NDArray
) -> np.ndarray:
    """
    Solve Kepler's equation for the eccentric anomaly.

    Parameters
    ----------
    eccentricity:
        The eccentricity of the orbit, greater than or equal to 0.
    mean_anomaly:
        The mean anomaly of the orbit in degrees.
    peri_dist:
        The perihelion distance, only required for parabolic objects. (Units of AU).
    """
    eccentricity = np.atleast_1d(eccentricity)
    mean_anomaly = np.radians(np.atleast_1d(mean_anomaly))
    peri_dist = np.atleast_1d(peri_dist)
    return np.degrees(
        _core.compute_eccentric_anomaly(eccentricity, mean_anomaly, peri_dist),
        dtype=float,
    )


def compute_tisserand(
    semi_major: float,
    ecc: float,
    inclination: float,
    perturbing_semi_major: float = 5.2038,
):
    """
    Compute Tisserand's Parameter, the default perturbing body is set to Jupiter.

    See: https://en.wikipedia.org/wiki/Tisserand%27s_parameter

    Note that if the inclination is not computed with respect to the perturbing body
    that this computation will not be correct. However, Jupiter only has an inclination
    of 1.3 degrees with respect to the Ecliptic plane, which will only introduce a very
    small error if the input inclination is w.r.t. the Ecliptic plane.

    Parameters
    ----------
    semi_major:
        Semi Major axis of small body, in units of AU.
    ecc:
        Eccentricity of the small body.
    inclination:
        Inclination of the small body with respect to the perturbing body, in units of
        Degrees.
    perturbing_semi_major:
        Semi major axis of the parent body, in units of AU.
    """
    cos_incl = np.cos(np.radians(inclination))
    return perturbing_semi_major / semi_major + 2 * cos_incl * np.sqrt(
        semi_major / perturbing_semi_major * (1 - ecc**2)
    )


def bin_data(data, bin_size=2, method="mean"):
    """
    Bin data, reducing the size of a matrix.

    Parameters
    ----------
    data:
        2 dimensional array of data to be reduced.
    bin_size:
        Binning size, binning of data is applied to both directions of the data.
        If the bin size is not a multiple of the shape of the data, the data is
        trimmed until it is.
    method:
        Which function is used to produce a single final value from the bins.
        This includes 'mean', 'median', 'min', 'max'. NANs are ignored.
    """
    funcs = {
        "mean": np.nanmean,
        "median": np.nanmedian,
        "min": np.nanmin,
        "max": np.nanmax,
    }

    # dont trust the user, check inputs
    if bin_size <= 0 or int(bin_size) != bin_size:
        raise ValueError("Bin size must be larger than 0 and an integer.")
    bin_size = int(bin_size)
    if method not in funcs:
        raise ValueError("'method' must be one of: ", list(funcs.keys()))

    data = np.asarray(data)
    if data.ndim != 2:
        raise ValueError("Array must be 2 dimensional.")

    xdim, ydim = data.shape
    if xdim % bin_size != 0:
        xdim = (xdim // bin_size) * bin_size
        logger.warning("x dimension is not a multiple of bin size, trimming to size.")
        data = data[:xdim, :]
    if ydim % bin_size != 0:
        ydim = (ydim // bin_size) * bin_size
        logger.warning("y dimension is not a multiple of bin size, trimming to size.")
        data = data[:, :ydim]

    reshaped = data.reshape(xdim // bin_size, bin_size, ydim // bin_size, bin_size)

    return funcs[method](funcs[method](reshaped, axis=1), axis=2)


def flux_to_mag(flux: float, zero_point=3631) -> float:
    """
    Convert flux in Jy to AB Magnitude, assuming it is a single frequency source.
    Note that this assumes that the band is infinitely narrow.

    See: https://en.wikipedia.org/wiki/AB_magnitude

    Parameters
    ----------
    flux:
        Flux in Jy.
    zero_point:
        Flux in Jy where the magnitude is zero.
    """
    if flux < 1e-14:
        return np.inf
    return -2.5 * np.log10(flux / zero_point)


def mag_to_flux(mag: float, zero_point=3631) -> float:
    """
    Convert AB Magnitude to flux in Jy, assuming it is a single frequency source.
    Note that this assumes that the band is infinitely narrow.

    See: https://en.wikipedia.org/wiki/AB_magnitude

    Parameters
    ----------
    mag:
        AB Magnitude.
    zero_point:
        Flux in Jy where the magnitude is zero.
    """
    return 10 ** (mag / -2.5) * zero_point


def table_to_states(orbit_dataframe):
    """
    Load :class:`kete.State` from a dataframe provided by either
    :func:`kete.mpc.fetch_known_orbit_data` or
    :func:`kete.horizons.fetch_known_orbit_data`.


    .. testcode::
        :skipif: True

        import kete

        # Load all MPC orbits
        orbits = kete.mpc.fetch_known_orbit_data()

        # Subset the table to be only NEOs
        neos = kete.population.neo(orbits.peri_dist, orbits.ecc)
        neo_subset = orbits[neos]

        # load the state object from this table
        state = kete.conversion.table_to_states(neo_subset)

    Parameters
    ----------
    orbit_dataframe:
        Pandas Dataframe as provided by the fetch_known_orbit_data function.
    """
    states = []
    for item in orbit_dataframe.itertuples():
        states.append(
            CometElements(
                str(item.desig),
                item.epoch,
                item.ecc,
                item.incl,
                item.peri_dist,
                item.peri_arg,
                item.peri_time,
                item.lon_node,
            ).state
        )
    return states
