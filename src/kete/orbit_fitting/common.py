"""Shared utilities for observation ingestion across data sources."""

from __future__ import annotations

import math
from collections import Counter
from functools import cache

from .._core import DebiasTable, _get_observatory_stats
from ..cache import download_file

# Default JPL distribution. EFCC18 (26 catalogs, Gaia-DR2 reference).
_DEBIAS_URL = "https://ssd.jpl.nasa.gov/ftp/ssd/debias/debias_2018.tgz"


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


def get_observatory_std(obs_code: str) -> tuple[float, float] | None:
    """Return (sigma_ra, sigma_dec) in arcseconds for an observatory code, or None."""
    result = _get_observatory_stats(obs_code)
    if result is None:
        return None
    return (result[0], result[1])


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


def _time_sigma_for_obs(note2: str, year: float) -> tuple[float, float]:
    """Return 1-sigma for timing and default astrometry uncertainty in seconds and
    arcseconds for an MPC observation.
    """

    # These are rough fallbacks if there is no submitted astrometry/timing, and the
    # uncertainty is not available from the lookup table.
    if note2 in ("S", "s"):
        # internal clocks on spacecraft often drift.
        return 1.0, 0.1
    if note2 == "n":
        # video observations with accurate timestamps
        return 0.5, 0.01
    if year >= 2010:
        return 0.5, 0.5
    if year >= 2000:
        return 1.0, 0.5
    if year >= 1993:
        return 2.0, 1.0
    if year >= 1970:
        return 5.0, 2.0
    return 60.0, 3.0  # old stuff
