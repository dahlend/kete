from __future__ import annotations
from typing import Union, Optional
from collections import namedtuple
import requests
import glob
import os
import numpy as np

from .time import Time
from . import _core
from .constants import AU_KM
from .cache import download_file, cache_path
from .vector import Frames, State
from ._core import state_to_earth_pos

__all__ = [
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
]


SpkInfo = namedtuple("SpkInfo", "name, jd_start, jd_end, center, frame, spk_type")
"""Information contained within a Spice Kernel."""
SpkInfo.name.__doc__ = "Name of the object."
SpkInfo.jd_start.__doc__ = "JD date of the start of the spice segment."
SpkInfo.jd_end.__doc__ = "JD date of the end of the spice segment."
SpkInfo.center.__doc__ = "Reference Center NAIF ID."
SpkInfo.frame.__doc__ = "Frame of reference."
SpkInfo.spk_type.__doc__ = "SPK Segment Type ID."


def _validate_time(time: Union[float, Time]) -> float:
    """
    Verifies that the time provided is either a `float` or
    :class:`~kete.time.Time` object.

    Parameters
    ----------
    jd:
        Julian time (TDB) of the desired record.

    Returns
    -------
    float
        The Julian time as a float.

    Raises
    ------
    TypeError
        If the input time is not a `float` or `Time`.
    """

    if isinstance(time, Time):
        return time.jd
    elif isinstance(time, float):
        return time
    try:
        return float(time)
    except Exception as exc:
        raise TypeError("Invalid jd type, use Time or float") from exc


_NAME_CACHE: dict = {}


def get_state(
    target: Union[str, int],
    jd: Union[float, Time],
    center: str = "Sun",
    frame: Frames = Frames.Ecliptic,
) -> State:
    """
    Calculates the :class:`~kete.State` of the target object at the
    specified time `jd`.

    This defaults to the ecliptic heliocentric state, though other centers may be
    chosen.

    Parameters
    ----------
    target:
        The names of the target object, this can include any object name listed in
        :meth:`~kete.spice.loaded_objects`
    jd:
        Julian time (TDB) of the desired record.
    center:
        The center point, this defaults to being heliocentric.
    frame:
        Coordinate frame of the state, defaults to ecliptic.

    Returns
    -------
    State
        Returns the ecliptic state of the target in AU and AU/days.

    Raises
    ------
    ValueError
        If the desired time is outside of the range of the source binary file.
    """
    target, ids = name_lookup(target)
    center, center_id = name_lookup(center)
    jd = _validate_time(jd)
    return _core.spk_state(ids, jd, center_id, frame)


def name_lookup(name: Union[int, str]) -> tuple[str, int]:
    """
    Given the provided partial name or integer, find the full name contained within
    the loaded SPICE kernels.

    >>> kete.spice.name_lookup("jupi")
    ('jupiter barycenter', 5)

    >>> kete.spice.name_lookup(10)
    ('sun', 10)

    If there are multiple names, but an exact match, the exact match is returned. In
    the case of ``Earth``, there is also ``Earth Barycenter``, but asking for Earth
    will return the exact match. Putting ``eart`` will raise an exception as there
    are 2 partial matches.

    >>> kete.spice.name_lookup("Earth")
    ('earth', 399)

    >>> kete.spice.name_lookup("Earth b")
    ('earth barycenter', 3)

    Parameters
    ----------
    name :
        Name, partial name, or integer id value of the object.

    Returns
    -------
    tuple :
        Two elements in the tuple, the full name and the integer id value.
    """
    if isinstance(name, str):
        name = name.lower()
    if name in _NAME_CACHE:
        return _NAME_CACHE[name]

    # barycenter of the solar system is special
    if name == 0:
        return ("ssb", 0)

    try:
        lookup_name = _core.spk_get_name_from_id(int(name))
    except ValueError:
        lookup_name = name
    lookup_name = lookup_name.lower()

    found = []
    for loaded in loaded_objects():
        loaded_lower = loaded[0].lower()
        # If it is an exact match, finish early
        if lookup_name == loaded_lower:
            _NAME_CACHE[name] = loaded
            return loaded
        if lookup_name in loaded_lower:
            found.append(loaded)
    found = list(set(found))

    if len(found) == 1:
        _NAME_CACHE[name] = found[0]
        return found[0]
    elif len(found) > 1:
        raise ValueError(f"Multiple objects match this name {found}")
    raise ValueError(f"No loaded objects which match this name ({name})")


def loaded_objects() -> list[tuple[str, int]]:
    """
    Return the name of all objects which are currently loaded in the SPICE kernels.
    """
    objects = _core.spk_loaded()
    if len(objects) == 0:
        kernel_reload()
        objects = _core.spk_loaded()
    return [(_core.spk_get_name_from_id(o), o) for o in objects]


def loaded_object_info(desig: Union[int, str]) -> list[SpkInfo]:
    """
    Return the available SPK information for the target object.

    Parameters
    ----------
    desig :
        Name or integer id value of the object.
    """
    name, naif = name_lookup(desig)
    return [SpkInfo(name, *k) for k in _core.spk_available_info(naif)]


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
    filenames: Optional[list[str]] = None, include_cache=False, include_planets=True
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

    if include_planets:
        _download_core_files()
        _core.spk_load_core()

        cache_files = glob.glob(os.path.join(cache_path(), "kernels", "*.bpc"))
        _core.pck_load(cache_files)

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
    wise = "https://github.com/Caltech-IPAC/kete/raw/refs/heads/main/src/kete_core/data/wise.bsp"

    if not any(["de440s.bsp" in file for file in cache_files]):
        # Cannot find the de440s file, so download it
        download_file(de440, subfolder="kernels/core")
    if not any(["wise.bsp" in file for file in cache_files]):
        # Cannot find the wise file, so download it
        download_file(wise, subfolder="kernels/core")

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
    cache_files = glob.glob(os.path.join(cache_path(), "kernels", "*.bpc"))
    if not any(["combined.bsp" in file for file in cache_files]):
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
        download_file(pck_path, subfolder="kernels")


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
    obs_code: str, jd: Union[float, Time], center: str = "Sun", full_name=False
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

    jd = _validate_time(jd)

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
    jd: Union[float, Time],
    geodetic_lat: float,
    geodetic_lon: float,
    height_above_surface: float,
    name: Optional[str] = None,
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

    jd = _validate_time(jd)
    pos = _core.wgs_lat_lon_to_ecef(geodetic_lat, geodetic_lon, height_above_surface)
    pos = np.array(pos) / AU_KM
    _, center_id = name_lookup(center)
    return _core.pck_earth_frame_to_ecliptic(pos, jd, center_id, name)


def moon_illumination_frac(jd: Union[float, Time], observer: str = "399"):
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
    jd = _validate_time(jd)

    moon2sun = -get_state("moon", jd).pos
    moon2earth = -get_state("moon", jd, center=observer).pos
    perc = 1.0 - moon2sun.angle_between(moon2earth) / 180
    return 0.5 - np.cos(np.pi * perc) / 2
