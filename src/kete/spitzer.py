"""
Spitzer Space Telescope functions.

Spitzer operated from August 2003 to January 2020. This module provides access to
Spitzer IRAC, MIPS, and IRS Peak-Up Basic Calibrated Data (BCD) frames via the IRSA
CAOM TAP service.

IRAC observed simultaneously at 3.6, 4.5, 5.8, and 8.0 um (channels 1-4) during the
cryogenic phase; cryogen was exhausted in 2009, after which only channels 1 and 2 were
available (the "warm mission").

MIPS observed at 24, 70 and 160 um during the cryogenic phase (2003-2009).

IRS Peak-Up imaging observed at 13.3-18.7 um (Blue) and 18.5-26.0 um (Red) during the
cryogenic phase (2003-2009).

Frame FOV metadata is queried from the IRSA ``caom.plane`` table (collection
``spitzer_sha``) and cached locally as parquet files. Each FOV record includes the IRSA
IBE artifact URI, so frames can be downloaded directly without additional network
queries.
"""

from __future__ import annotations

import logging
import os
import struct
from functools import cache

import pandas as pd
from astropy.io import fits

from . import spice
from .cache import cache_path, download_file
from .fov import SpitzerFrame
from .tap import query_tap
from .time import Time
from .vector import Vector

__all__ = [
    "IRAC_BAND_WAVELENGTHS",
    "IRAC_ZERO_MAGS",
    "IRS_PU_BAND_WAVELENGTHS",
    "IRS_PU_ZERO_MAGS",
    "MIPS_BAND_WAVELENGTHS",
    "MIPS_ZERO_MAGS",
    "fetch_fovs",
    "fetch_frame",
    "parse_poly",
    "resolve_artifact_url",
]

logger = logging.getLogger(__name__)

IRAC_BAND_WAVELENGTHS: list[float] = [3550.0, 4493.0, 5731.0, 7872.0]
"""
Effective channel wavelengths in nm for IRAC channels 1-4 (3.6, 4.5, 5.8, 8.0 um).
Source: IRAC Instrument Handbook v2.1, Table 4.1.
"""

IRAC_ZERO_MAGS: list[float] = [280.9, 179.7, 115.0, 64.13]
"""
Vega-system zero-magnitude flux densities in Jy for IRAC channels 1-4.
Magnitude is ``-2.5 * log10(flux_jy / zero_mag)``.
Source: IRAC Instrument Handbook v2.1, Table 4.1.
"""

MIPS_BAND_WAVELENGTHS: list[float] = [23680.0, 71420.0, 155900.0]
"""
Effective wavelengths in nm for MIPS 24, 70, and 160 um bands.
Source: MIPS Instrument Handbook (Rieke et al. 2004).
"""

MIPS_ZERO_MAGS: list[float] = [7.17, 0.778, 0.159]
"""
Vega-system zero-magnitude flux densities in Jy for MIPS 24, 70, and 160 um.
Sources: Engelbracht et al. 2007 (24um), Gordon et al. 2007 (70um),
Stansberry et al. 2007 (160um).
"""

IRS_PU_BAND_WAVELENGTHS: list[float] = [15800.0, 22300.0]
"""
Effective wavelengths in nm for IRS Peak-Up Blue and Red.
Source: IRS Instrument Handbook v5.0.
"""

IRS_PU_ZERO_MAGS: list[float] = [15.6, 7.80]
"""
Vega-system zero-magnitude flux densities in Jy for IRS Peak-Up Blue and Red.
Derived from the Cohen (1999) Vega spectral model at the effective wavelengths.
"""

_VALID_BANDS = frozenset(
    [
        "IRAC1",
        "IRAC2",
        "IRAC3",
        "IRAC4",
        "MIPS24",
        "MIPS70",
        "MIPS160",
        "IRS Peak-Up Blue",
        "IRS Peak-Up Red",
    ]
)


def parse_poly(poly_hex: str) -> list[Vector]:
    """
    Parse an EWKB (Extended Well-Known Binary) polygon from the CAOM ``poly`` column.

    The IRSA CAOM tables return polygon footprints as hex-encoded EWKB with SRID 4326.
    Longitude is in the range [-180, 180] and is normalized to [0, 360] before
    constructing vectors.

    The polygon has 5 vertices (4 corners plus a closing repeat of the first); only
    the first 4 are returned.

    Parameters
    ----------
    poly_hex :
        Hex-encoded EWKB polygon string.

    Returns
    -------
    list[Vector]
        Four corner vectors in ICRS (equatorial) coordinates.
    """
    data = bytes.fromhex(poly_hex.replace(" ", ""))
    byte_order = "<" if data[0] == 1 else ">"
    offset = 1  # skip byte-order byte
    offset += 4  # WKB type
    offset += 4  # SRID
    nrings = struct.unpack_from(f"{byte_order}I", data, offset)[0]
    offset += 4
    if nrings < 1:
        raise ValueError("EWKB polygon has no rings")
    npts = struct.unpack_from(f"{byte_order}I", data, offset)[0]
    offset += 4
    if npts < 5:
        raise ValueError(f"Expected at least 5 polygon vertices, got {npts}")
    corners = []
    for _ in range(4):
        lon = struct.unpack_from(f"{byte_order}d", data, offset)[0]
        offset += 8
        lat = struct.unpack_from(f"{byte_order}d", data, offset)[0]
        offset += 8
        corners.append(Vector.from_ra_dec(lon % 360.0, lat))
    return corners


def fetch_fovs(year: int, band: str | None = None) -> list[SpitzerFrame]:
    """
    Load all Spitzer IRAC and MIPS BCD FOVs for the specified mission year.

    FOV metadata is queried from the IRSA ``caom.plane`` table (science observations
    only) and cached locally as a parquet file. The Spitzer spacecraft SPICE kernel
    must be loaded for the observer state to be computed.

    Available bands:

    - ``'IRAC1'`` (3.6 um), ``'IRAC2'`` (4.5 um), ``'IRAC3'`` (5.8 um),
      ``'IRAC4'`` (8.0 um) -- cryogenic 2003-2009, warm IRAC1/2 through 2020.
    - ``'MIPS24'`` (24 um), ``'MIPS70'`` (70 um), ``'MIPS160'`` (160 um) --
      cryogenic phase only (2003-2009).
    - ``'IRS Peak-Up Blue'`` (13.3-18.7 um), ``'IRS Peak-Up Red'`` (18.5-26.0 um) --
      cryogenic phase only (2003-2009).

    Parameters
    ----------
    year :
        Calendar year of observations, 2003 to 2020.
    band :
        If provided, only return FOVs for that band (e.g. ``'IRAC1'``, ``'MIPS24'``).
        If None, all available bands are returned for the given year.

    Returns
    -------
    list[SpitzerFrame]
        FOVs sorted by observation time.
    """
    if band is not None and band not in _VALID_BANDS:
        raise ValueError(
            f"Unknown band {band!r}. Valid bands: {', '.join(sorted(_VALID_BANDS))}"
        )
    fovs = _fetch_fovs_cached(int(year))
    if band is not None:
        return [f for f in fovs if f.band == band]
    return list(fovs)


@cache
def _fetch_fovs_cached(year: int) -> list[SpitzerFrame]:
    """Build the full FOV list for a year (cached by year only)."""
    if year < 2003 or year > 2020:
        raise ValueError("Spitzer data available from 2003 to 2020.")

    cache_dir = cache_path()
    dir_path = os.path.join(cache_dir, "fovs")
    filename = os.path.join(dir_path, f"spitzer_{year}.parquet")

    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)

    if os.path.isfile(filename):
        data = pd.read_parquet(filename)
    else:
        mjd_start = Time.from_ymd(year, 1, 1).utc_mjd
        mjd_end = Time.from_ymd(year + 1, 1, 1).utc_mjd

        data = query_tap(
            "SELECT plane.obs_publisher_did, plane.energy_bandpassname, "
            "plane.time_bounds_lower, plane.time_bounds_upper, plane.poly, "
            # MIN collapses the rare case where a plane has multiple matching
            # BCD artifacts to a single deterministic URI (lexicographic first).
            "MIN(art.uri) AS artifact_uri "
            "FROM caom.plane AS plane "
            "JOIN caom.observation AS obs ON plane.obsid = obs.obsid "
            "JOIN caom.artifact AS art ON art.planeid = plane.planeid "
            "WHERE obs.collection = 'spitzer_sha' "
            "AND obs.intent = 'science' "
            "AND plane.calibrationlevel = 2 "
            "AND plane.dataproducttype = 'image' "
            "AND (plane.energy_bandpassname LIKE 'IRAC%' "
            "     OR plane.energy_bandpassname IN ('MIPS24', 'MIPS70', 'MIPS160', "
            "         'IRS Peak-Up Blue', 'IRS Peak-Up Red')) "
            f"AND plane.time_bounds_lower BETWEEN {mjd_start} AND {mjd_end} "
            "AND (art.uri LIKE '%_bcd.fits' OR art.uri LIKE '%_bcdb.fits' "
            "     OR art.uri LIKE '%_bcdr.fits') "
            "AND art.uri NOT LIKE '%_cbcd.fits' "
            "AND art.uri NOT LIKE '%_ebcd.fits' "
            "AND art.uri NOT LIKE '%_fbcd.fits' "
            "GROUP BY plane.obs_publisher_did, plane.energy_bandpassname, "
            "plane.time_bounds_lower, plane.time_bounds_upper, plane.poly",
            cache=False,
            verbose=True,
        )

        data.to_parquet(filename, index=False)

    fovs = []
    for row in data.itertuples():
        # time_bounds_lower/upper are MJD UTC; use the midpoint
        mjd_mid = (float(row.time_bounds_lower) + float(row.time_bounds_upper)) / 2.0
        jd = Time.from_mjd(mjd_mid, scaling="utc").jd

        try:
            # The kernel uses the SIRTF name (Spitzer's pre-launch NAIF designation,
            # NAIF ID -79).
            state = spice.get_state("sirtf", jd)
        except Exception as exc:
            raise RuntimeError(
                "Failed to get Spitzer state from SPICE. "
                "Ensure kernels are loaded by calling kete.spice.kernel_reload()."
            ) from exc

        try:
            corners = parse_poly(row.poly)
        except (ValueError, TypeError):
            logger.warning(
                "Could not parse polygon for obs %s, skipping.", row.obs_publisher_did
            )
            continue

        obs_id = str(row.obs_publisher_did)
        band_name = str(row.energy_bandpassname)
        artifact_uri = str(row.artifact_uri)
        # CAOM time window (not detector integration time) in seconds.
        duration = (
            float(row.time_bounds_upper) - float(row.time_bounds_lower)
        ) * 86400.0

        # Reverse corners: CAOM polygons are CW, FOV constructor expects CCW.
        fov = SpitzerFrame(
            corners[::-1], state, obs_id, band_name, artifact_uri, duration
        )
        fovs.append(fov)

    fovs.sort(key=lambda x: x.jd)
    return fovs


_IBE_DATA_BASE = "https://irsa.ipac.caltech.edu/ibe/data/"
# CAOM artifact URIs use a legacy HTTPS path; the public IBE download path differs
_CAOM_DATA_PREFIX = "https://irsa.ipac.caltech.edu/data/SPITZER/SHA/archive/proc/"
_IBE_PROC_PREFIX = "https://irsa.ipac.caltech.edu/ibe/data/spitzer/sha/archive/proc/"


def resolve_artifact_url(uri: str, file_type: str = "bcd") -> str:
    """
    Convert a CAOM artifact URI to an IRSA IBE download URL.

    Parameters
    ----------
    uri :
        Artifact URI from the CAOM ``artifact`` table (CAOM HTTPS, ``ibe://data/``,
        or plain ``https://``).
    file_type :
        BCD product suffix. ``'bcd'`` for the flux-calibrated image,
        ``'bunc'`` for the per-pixel uncertainty image.

    Returns
    -------
    str
        Public HTTPS download URL.
    """
    if uri.startswith(_CAOM_DATA_PREFIX):
        url = _IBE_PROC_PREFIX + uri[len(_CAOM_DATA_PREFIX) :]
    elif uri.startswith("ibe://data/"):
        url = _IBE_DATA_BASE + uri[len("ibe://data/") :]
    elif uri.startswith("https://") or uri.startswith("http://"):
        url = uri
    else:
        raise ValueError(
            f"Unrecognised artifact URI scheme in {uri!r}. "
            "Expected CAOM HTTPS, 'ibe://data/', or 'https://'."
        )

    # Substitute product type suffix when not requesting the primary BCD.
    # IRAC/MIPS BCDs end with _bcd.fits; IRS Peak-Up Blue/Red end with
    # _bcdb.fits / _bcdr.fits respectively, with matching suffixes for other
    # products (e.g. _uncb.fits / _uncr.fits for uncertainty).
    if file_type != "bcd":
        if url.endswith("_bcd.fits"):
            url = url[: -len("_bcd.fits")] + f"_{file_type}.fits"
        elif url.endswith("_bcdb.fits"):
            url = url[: -len("_bcdb.fits")] + f"_{file_type}b.fits"
        elif url.endswith("_bcdr.fits"):
            url = url[: -len("_bcdr.fits")] + f"_{file_type}r.fits"
        else:
            raise ValueError(
                f"Cannot derive {file_type!r} URL from artifact URI {uri!r}."
            )

    return url


def fetch_frame(
    fov: SpitzerFrame,
    as_fits: bool = True,
    file_type: str = "bcd",
) -> fits.ImageHDU | str:
    """
    Fetch a Spitzer FITS frame from the IRSA IBE archive.

    The artifact URI stored on the FOV (populated by :py:func:`fetch_fovs`) is
    translated to an HTTPS download URL and the file is retrieved and cached
    locally. No additional network queries are made beyond the download itself.

    To retrieve a product other than the primary BCD (e.g. uncertainty ``'bunc'``),
    the URI suffix ``_bcd.fits`` is substituted with ``_{file_type}.fits`` -- the
    Spitzer naming convention guarantees all BCD-level products for the same frame
    share the same directory and base filename.

    Parameters
    ----------
    fov :
        A :py:class:`~kete.SpitzerFrame` obtained from :py:func:`fetch_fovs`.
    as_fits :
        If True (default), return an Astropy FITS HDU. Otherwise return the path
        to the cached file.
    file_type :
        BCD product suffix to retrieve. Use ``'bcd'`` for the flux-calibrated
        image or ``'bunc'`` for the per-pixel uncertainty image.

    Returns
    -------
    astropy.io.fits.ImageHDU or str
        The FITS image HDU, or the local file path if ``as_fits=False``.
    """
    uri = fov.artifact_uri
    if not uri:
        raise ValueError(
            "This FOV has no artifact_uri. Re-run fetch_fovs to refresh the cache."
        )

    url = resolve_artifact_url(uri, file_type)

    subfolder = os.path.join("spitzer_frames", fov.band.lower().replace(" ", "_"))
    file_path = download_file(url, auto_zip=False, subfolder=subfolder)

    if as_fits:
        with fits.open(file_path) as hdul:
            return hdul[0].copy()
    return file_path
