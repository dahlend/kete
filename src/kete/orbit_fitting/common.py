# SPDX-FileCopyrightText: 2026 Dar Dahlen
# SPDX-License-Identifier: BSD-3-Clause

"""Shared utilities for observation ingestion across data sources."""

from __future__ import annotations

import math
from collections import Counter
from functools import cache

from .._core import DebiasTable, _get_observatory_stats
from ..cache import download_file
from ..vector import State

# Default JPL distribution. EFCC18 (26 catalogs, Gaia-DR2 reference).
_DEBIAS_URL = "https://ssd.jpl.nasa.gov/ftp/ssd/debias/debias_2018.tgz"

# Lower limit (arcsec) on both principal axes of an error ellipse whose reported
# sigmas are used as given. It keeps a correlation at +-1 from giving one
# direction an unbounded weight.
_MIN_AXIS_SIGMA = 1e-4


def get_observatory_std(obs_code: str) -> tuple[float, float] | None:
    """Return (sigma_ra, sigma_dec) in arcseconds for an observatory code, or None."""
    result = _get_observatory_stats(obs_code)
    if result is None:
        return None
    return (result[0], result[1])


@cache
def _fetch_debias_table(force_download: bool = False) -> DebiasTable:
    """Load the EFCC18 debias table, downloading the tgz on first use."""
    import io
    import tarfile

    tgz_path = download_file(
        _DEBIAS_URL, force_download=force_download, subfolder="debias"
    )
    with tarfile.open(tgz_path, "r:gz") as tar:
        for member in tar.getmembers():
            if member.isfile() and member.name.endswith("bias.dat"):
                f = tar.extractfile(member)
                if f is not None:
                    text = io.TextIOWrapper(f, encoding="ascii").read()
                    return DebiasTable.from_ascii(text)
    raise RuntimeError(
        f"bias.dat not found inside {tgz_path}; archive layout may have changed"
    )


def _floor_error_ellipse(
    sigma_ra: float, sigma_dec: float, corr: float, floor: float
) -> tuple[float, float, float]:
    """Raise both principal axes of an RA/Dec error ellipse to at least ``floor``.

    ``sigma_ra`` is sky-plane, already multiplied by cos(dec). All sigmas share
    the units of ``floor``. ``corr`` is limited to [-1, 1] first. The floor acts
    on the principal axes, so the ellipse keeps its orientation. A floor on the
    RA and Dec sigmas separately would rotate a strongly correlated ellipse.
    With a positive floor, the returned correlation is strictly inside (-1, 1).

    Returns ``(sigma_ra, sigma_dec, corr)``.
    """
    corr = max(min(corr, 1.0), -1.0)
    off = corr * sigma_ra * sigma_dec
    half_sum = 0.5 * (sigma_ra**2 + sigma_dec**2)
    half_diff = 0.5 * (sigma_ra**2 - sigma_dec**2)
    radius = math.hypot(half_diff, off)
    major, minor = half_sum + radius, half_sum - radius
    floor_sq = floor * floor
    if minor >= floor_sq:
        return sigma_ra, sigma_dec, corr
    # Principal direction of the major axis, then rebuild with floored eigenvalues.
    angle = 0.5 * math.atan2(2.0 * off, sigma_ra**2 - sigma_dec**2)
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    major, minor = max(major, floor_sq), floor_sq
    var_ra = major * cos_a**2 + minor * sin_a**2
    var_dec = major * sin_a**2 + minor * cos_a**2
    cov = (major - minor) * sin_a * cos_a
    return math.sqrt(var_ra), math.sqrt(var_dec), cov / math.sqrt(var_ra * var_dec)


def _over_obs_reweight_factors(
    obs_codes: list[str],
    jds: list[float],
    spacecraft: list[bool],
    n_max: int = 4,
) -> list[float]:
    """Return per-observation sigma inflation factors for over-observed nights.

    Groups observations by (obs_code, floor(jd - 0.5)).  Spacecraft
    observations are treated as independent and always receive a factor of 1.0.
    For groups of n > n_max ground-based observations, each member is inflated
    by sqrt(n / n_max) following Veres et al. 2017.
    """
    counts: Counter = Counter()
    for code, jd, is_sc in zip(obs_codes, jds, spacecraft):
        if is_sc:
            continue
        night = int(jd - 0.5)
        counts[(code, night)] += 1

    factors = []
    for code, jd, is_sc in zip(obs_codes, jds, spacecraft):
        if is_sc:
            factors.append(1.0)
            continue
        night = int(jd - 0.5)
        n = counts[(code, night)]
        factors.append(math.sqrt(n / n_max) if n > n_max else 1.0)
    return factors


def _ground_observer(
    jd: float, geodetic_lat: float, geodetic_lon: float, height_km: float, name: str
) -> State:
    """Return the SSB-centered equatorial state of a site on the Earth at ``jd``.

    ``jd`` is TDB. The Earth orientation comes from the loaded PCK kernels. Where
    they do not cover ``jd``, as before 1962 for the default kernel, it comes
    from :func:`~kete.spice.approx_earth_pos_to_ecliptic`. That model has no
    polar motion and uses a model of Delta T, so it is less accurate than the
    kernels.

    Raises
    ------
    ValueError
        If the loaded planetary ephemeris does not cover ``jd``.
    """
    from .. import spice

    try:
        state = spice.earth_pos_to_ecliptic(
            jd, geodetic_lat, geodetic_lon, height_km, name=name, center=0
        )
    except ValueError:
        state = spice.approx_earth_pos_to_ecliptic(
            jd, geodetic_lat, geodetic_lon, height_km, name=name
        ).change_center(0)
    return state.as_equatorial


def _time_sigma_for_obs(note2: str, year: float) -> tuple[float, float]:
    """Return 1-sigma for timing and default astrometry uncertainty in seconds and
    arcseconds for an MPC observation.
    """

    # These are rough fallbacks if there is no submitted astrometry/timing, and the
    # uncertainty is not available from the lookup table. The observing mode sets
    # only the timing uncertainty; the astrometric default follows the epoch.
    astrometric = _epoch_sigmas(year)[1]
    if note2 in ("S", "s"):
        # internal clocks on spacecraft often drift.
        return 1.0, astrometric
    if note2 == "n":
        # video observations with accurate timestamps
        return 0.5, astrometric
    return _epoch_sigmas(year)


def _epoch_sigmas(year: float) -> tuple[float, float]:
    """Default timing (s) and astrometric (arcsec) uncertainty for an epoch."""
    if year >= 2010:
        return 0.5, 0.5
    if year >= 2000:
        return 1.0, 0.5
    if year >= 1993:
        return 2.0, 1.0
    if year >= 1970:
        return 5.0, 2.0
    return 60.0, 3.0  # old stuff
