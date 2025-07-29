from __future__ import annotations

import glob
import os
from collections import namedtuple

import numpy as np
import requests

from . import _core
from ._core import (
    approx_earth_pos_to_ecliptic,
    get_state,
    instrument_equatorial_to_frame,
    instrument_frame_to_equatorial,
    loaded_objects,
    name_lookup,
    state_to_earth_pos,
)
from .cache import cache_path, download_file
from .constants import AU_KM
from .time import Time
from .vector import State

__all__ = [
    "approx_earth_pos_to_ecliptic",
    "SpkInfo",
    "get_state",
    "name_lookup",
    "loaded_objects",
    "loaded_object_info",
    "kernel_ls",
    "kernel_fetch_from_url",
    "kernel_reload",
    "kernel_header_comments",
    "mpc_code_to_ecliptic",
    "earth_pos_to_ecliptic",
    "state_to_earth_pos",
    "moon_illumination_frac",
    "instrument_frame_to_equatorial",
    "instrument_equatorial_to_frame",
]


SpkInfo = namedtuple("SpkInfo", "name, jd_start, jd_end, center, frame, spk_type")
"""Information contained within a Spice Kernel."""
SpkInfo.name.__doc__ = "Name of the object."
SpkInfo.jd_start.__doc__ = "JD date of the start of the spice segment."
SpkInfo.jd_end.__doc__ = "JD date of the end of the spice segment."
SpkInfo.center.__doc__ = "Reference Center NAIF ID."
SpkInfo.frame.__doc__ = "Frame of reference."
SpkInfo.spk_type.__doc__ = "SPK Segment Type ID."


def loaded_object_info(desig: int | str) -> list[SpkInfo]:
    """
    Return the available SPK information for the target object.

    Parameters
    ----------
    desig :
        Name or integer id value of the object.
    """
    return [SpkInfo(*k) for k in _core._loaded_object_info(desig)]


def kernel_ls():
    """
    List all files contained within the kernels cache folder.
    """
    path = os.path.join(cache_path(), "kernels", "**")
    return glob.glob(path)


def kernel_fetch_from_url(url, force_download: bool = False):
    """
    Download the target url into the cache folder of spice kernels.
    """
    download_file(url, force_download=force_download, subfolder="kernels")


def kernel_reload(
    filenames: list[str] | None = None, include_cache=False, include_planets=True
):
    """
    Load the specified spice kernels into memory, this resets the currently loaded
    kernels.

    If `include_cache` is true, this will reload the kernels contained within the
    kete cache folder as well.

    Parameters
    ----------
    filenames :
        Paths to the specified files to load, this must be a list of filenames.
    include_cache:
        This decides if all of the files contained within the kete cache should
        be loaded in addition to the specified files.
    include_planets:
        This decides if the default planetary kernels should be loaded in
        addition. This includes the de440s, the WISE kernel, and 5 largest main
        belt asteroids. If these files are not present, they will be downloaded.
    """
    _core.spk_reset()
    _core.pck_reset()
    _core.ck_reset()

    if include_planets:
        _download_core_files()
        _core.spk_load_core()
        _core.pck_load_core()

    if include_cache:
        _core.spk_load_cache()

    if filenames:
        _core.spk_load(filenames)


def _download_core_files():
    """
    Download the core files required for the default planetary kernels.

    This includes the de440s, the WISE kernel, and 5 largest main belt asteroids.
    """
    cache_files = glob.glob(os.path.join(cache_path(), "kernels/core", "**.bsp"))

    # required files:
    de440 = "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de440s.bsp"
    wise = "https://github.com/dahlend/kete/raw/refs/heads/main/docs/data/wise.bsp"
    spherex = (
        "https://github.com/dahlend/kete/raw/refs/heads/main/docs/data/spherex.bsp"
    )

    if not any(["de440s.bsp" in file for file in cache_files]):
        # Cannot find the de440s file, so download it
        download_file(de440, subfolder="kernels/core")
    if not any(["wise.bsp" in file for file in cache_files]):
        # Cannot find the wise file, so download it
        download_file(wise, subfolder="kernels/core")
    if not any(["spherex.bsp" in file for file in cache_files]):
        # Cannot find the wise file, so download it
        download_file(spherex, subfolder="kernels/core")

    required_asteroids = [1, 2, 4, 10, 704]
    for asteroid in required_asteroids:
        expected_name = f"{asteroid}.bsp"
        if not any([expected_name in file for file in cache_files]):
            from .horizons import fetch_spice_kernel

            fetch_spice_kernel(
                asteroid,
                jd_start=Time.from_ymd(1900, 1, 1).jd,
                jd_end=Time.from_ymd(2100, 1, 1).jd,
                cache_dir="kernels/core",
            )

    # Look for PCK files
    cache_files = glob.glob(os.path.join(cache_path(), "kernels/core", "*.bpc"))
    if not any(["combined.bpc" in file for file in cache_files]):
        pck_path = "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/"

        # cannot find the combined file, so download it
        # first we download the index page and parse it for the filename
        res = requests.get(pck_path)
        res.raise_for_status()
        filename = [
            x for x in res.content.decode().split('"') if "combined.bpc" in x[-12:]
        ]
        # if parsing was successful, download the file
        if len(filename) == 0:
            raise ValueError(
                "Failed to find Earth orientation file on NAIF website, please"
                "submit a github issue! NAIF may have moved filenames or location."
            )
        pck_path = pck_path + filename[0]
        download_file(pck_path, subfolder="kernels/core")


def kernel_header_comments(filename: str):
    """
    Return the comments contained within the header of the provided DAF file, this
    includes SPK and PCK files.

    This does not load the contents of the file into memory, it only prints the
    header contents.

    Parameters
    ----------
    filename :
        Path to a DAF file.
    """
    return _core.daf_header_comments(filename).replace("\x04", "\n").strip()


def mpc_code_to_ecliptic(
    obs_code: str, jd: float | Time, center: str = "Sun", full_name=False
) -> State:
    """
    Load an MPC Observatory code as an ecliptic state.

    This only works for ground based observatories.

    Parameters
    ----------
    obs_code:
        MPC observatory code or name of observatory.
    jd:
        Julian time (TDB) of the desired state.
    center:
        The new center point, this defaults to being heliocentric.
    full_name:
        Should the final state include the full name of the observatory or just its
        code.

    Returns
    -------
    State
        Returns the equatorial state of the observatory in AU and AU/days.
    """
    from .mpc import find_obs_code

    obs = find_obs_code(obs_code)
    return earth_pos_to_ecliptic(
        jd,
        geodetic_lat=obs[0],
        geodetic_lon=obs[1],
        height_above_surface=obs[2],
        name=obs[3] if full_name else obs[4],
        center=center,
    )


def earth_pos_to_ecliptic(
    jd: float | Time,
    geodetic_lat: float,
    geodetic_lon: float,
    height_above_surface: float,
    name: str | None = None,
    center: str = "Sun",
) -> State:
    """
    Given a position in the frame of the Earth at a specific time, convert that to
    Sun centered ecliptic state.

    This uses the WGS84 model of Earth's shape to compute state. This uses Geodetic
    latitude and longitude, not geocentric.

    The frame conversion is done using a PCK file from the NAIF/JPL website.
    This is the combined PCK file containing lower accuracy historical data, high
    accuracy modern data, and the current predictions going forward.

    Parameters
    ----------
    jd:
        Julian time (TDB) of the desired state.
    geodetic_lat:
        Latitude on Earth's surface in degrees.
    geodetic_lon:
        Latitude on Earth's surface in degrees.
    height_above_surface:
        Height of the observer above the surface of the Earth in km.
    name :
        Optional name of the position on Earth.
    center:
        The new center point, this defaults to being heliocentric.

    Returns
    -------
    State
        Returns the equatorial state of the target in AU and AU/days.
    """
    if len(_core.pck_loaded()) == 0:
        kernel_reload()

    pos = _core.wgs_lat_lon_to_ecef(geodetic_lat, geodetic_lon, height_above_surface)
    pos = np.array(pos) / AU_KM
    _, center_id = name_lookup(center)
    return _core.pck_earth_frame_to_ecliptic(pos, jd, center_id, name)


def moon_illumination_frac(jd: float | Time, observer: str = "399"):
    """
    Compute the fraction of the moon which is illuminated at the specified time.

    This is a simple approximation using basic spherical geometry, and defaults to
    having the observer located at the geocenter of the Earth.

    >>> float(kete.spice.moon_illumination_frac(Time.from_ymd(2024, 2, 24)))
    0.9964936478732302

    Parameters
    ----------
    jd:
        Julian time (TDB) of the desired state.
    observer:
        NAIF ID of the observer location, defaults to Earth geocenter.

    Returns
    -------
    State
        Fraction between 0 and 1 of the moons visible surface which is illuminated.
    """

    moon2sun = -get_state("moon", jd).pos
    moon2earth = -get_state("moon", jd, center=observer).pos
    perc = 1.0 - moon2sun.angle_between(moon2earth) / 180
    return 0.5 - np.cos(np.pi * perc) / 2


# Download missing files on import
_download_core_files()
