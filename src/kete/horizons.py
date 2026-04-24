"""
Interface functions and classes to JPL Horizons web services.
"""

import base64
import contextlib
import gzip
import json
import logging
import os
from functools import lru_cache
from typing import Optional, Union

import pandas as pd
import requests

from . import constants, spice
from ._core import HorizonsProperties
from .cache import cache_path
from .fitting import Observation
from .mpc import unpack_designation
from .time import Time
from .vector import Frames, State

logger = logging.getLogger(__name__)

__all__ = [
    "HorizonsProperties",
    "fetch_spice_kernel",
    "fetch_known_orbit_data",
    "fetch_radar_table",
    "fetch_radar_observations",
    "fetch",
]


fetch = HorizonsProperties.fetch


def fetch_spice_kernel(
    name,
    jd_start: Union[Time, float],
    jd_end: Union[Time, float],
    exact_name: bool = False,
    update_cache: bool = False,
    apparition_year: Optional[int] = None,
    cache_dir: str = "kernels",
):
    """
    Download a SPICE kernel from JPL Horizons and save it directly into the Cache.

    .. code-block:: python

        from kete import horizons, Time
        jd_start = Time.from_ymd(1900, 1, 1)
        jd_end = Time.from_ymd(2100, 1, 1)
        horizons.fetch_spice_kernel("10p", jd_start, jd_end)

    Parameters
    ----------
    name :
        Name or integer id value of the object.
    jd_start:
        Start date of the SPICE kernel to download.
    jd_end:
        End date of the SPICE kernel to download.
    exact_name:
        If the specified name is the exact name in Horizons, this can help for
        comet fragments.
    update_cache:
        If the current state of the cache should be ignored and the file
        re-downloaded.
    apparition_year:
        If the object is a comet, retrieve the orbit fit which is previous to this
        specified year. If this is not provided, then default to the most recent
        epoch of orbit fit. Ex: `apparition_year=1980` will return the closest
        epoch before 1980.
    cache_dir:
        The directory in the cache where the file will be saved. This is used to
        separate the planet SPICE kernels from the other cache files.
    """

    if not isinstance(jd_start, Time):
        jd_start = Time(jd_start)
    if not isinstance(jd_end, Time):
        jd_end = Time(jd_end)

    if isinstance(name, str):
        with contextlib.suppress(ValueError):
            name = unpack_designation(name)

    query = "des" if exact_name else "sstr"
    # Name resolution using the sbdb database
    name_dat = requests.get(
        f"https://ssd-api.jpl.nasa.gov/sbdb.api?{query}={name}",
        timeout=30,
    )
    if "object" not in name_dat.json():
        raise ValueError("Failed to find object: ", str(name_dat.json()))
    comet = "c" in name_dat.json()["object"]["kind"].lower()

    if comet and apparition_year is None:
        apparition_year = jd_end.ymd[0]

    spk_id = int(name_dat.json()["object"]["spkid"])

    dir_path = os.path.join(cache_path(), cache_dir)

    if apparition_year is not None:
        filename = os.path.join(dir_path, f"{spk_id}_epoch_{apparition_year}.bsp")
    else:
        filename = os.path.join(dir_path, f"{spk_id}.bsp")

    if os.path.isfile(filename) and not update_cache:
        return

    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)

    jd_s_str = "{:d}-{:02d}-{:0.0f}".format(*jd_start.ymd)
    jd_e_str = "{:d}-{:02d}-{:0.0f}".format(*jd_end.ymd)
    cap = f"CAP<{apparition_year}%3B" if comet else ""
    response = requests.get(
        f"https://ssd.jpl.nasa.gov/api/horizons.api?COMMAND='DES={spk_id}%3B{cap}'"
        f"&EPHEM_TYPE=SPK&START_TIME='{jd_s_str}'&STOP_TIME='{jd_e_str}'&CENTER=0",
        timeout=30,
    )

    if response.status_code == 300:
        names = [
            des["pdes"] for des in response.json()["list"] if "-" not in des["pdes"]
        ]
        if len(names) == 1:
            fetch_spice_kernel(names[0], jd_start, jd_end, exact_name=True)

    if response.status_code != 200:
        raise OSError(f"Error from Horizons: {response.json()}")

    if "spk" not in response.json():
        raise ValueError("Failed to fetch file\n:", response.json())

    with open(filename, "wb") as f:
        f.write(base64.b64decode(response.json()["spk"]))


@lru_cache
def fetch_known_orbit_data(update_cache=False):
    """
    Fetch the known orbit data from JPL Horizons for all known asteroids and comets.

    This gets loaded as a pandas table, and if the file already exists in cache, then
    the contents of this file are returned by default.

    The constructed pandas table may be turned into states using the
    :func:`~kete.conversion.table_to_states` function.

    Parameters
    ==========
    update_cache :
        Force download a new file from JPL Horizons, this can be used to update the
        orbit fits which are currently saved.
    """
    filename = os.path.join(cache_path(), "horizons_orbits.json.gz")
    if update_cache or not os.path.isfile(filename):
        res = requests.get(
            (
                "https://ssd-api.jpl.nasa.gov/sbdb_query.api?fields="
                "pdes,name,spkid,orbit_id,rms,H,G,diameter,spec_T,spec_B,epoch,"
                "e,i,q,w,tp,om,A1,A2,A3,DT,M1,M2,K1,K2,PC,rot_per,H_sigma,soln_date"
                "&full-prec=1&sb-xfrag=1"
            ),
            timeout=240,
        )
        res.raise_for_status()
        with gzip.open(filename, "wb") as f:
            f.write(res.content)
        file_contents = res.json()
    else:
        with gzip.open(filename, "rb") as f:
            file_contents = json.load(f)
    columns = file_contents["fields"]

    # relabel some of the columns so that they match the contents of the MPC orbit file
    # this allows user to reuse the table_to_state function in mpc.py
    lookup = {
        "e": "ecc",
        "i": "incl",
        "q": "peri_dist",
        "w": "peri_arg",
        "tp": "peri_time",
        "om": "lon_node",
        "pdes": "desig",
    }
    columns = [lookup.get(c, c) for c in columns]
    table = pd.DataFrame.from_records(file_contents["data"], columns=columns)
    # dont coerce numerics for these columns
    others = table.columns.difference(
        ["desig", "name", "spkid", "orbit_id", "spec_T", "spec_B", "soln_date"]
    )
    table[others] = table[others].apply(pd.to_numeric, errors="coerce")
    return table


@lru_cache
def fetch_radar_table(desig: Optional[str] = None, update_cache: bool = False):
    """
    Fetch JPL Small-Body radar astrometry data.

    The complete JPL Small-Body Radar Astrometry data set is downloaded
    once and cached locally as a parquet file. Subsequent calls reuse
    the cached file unless ``update_cache`` is set. If ``desig`` is
    supplied, only the records whose ``des`` field matches that
    designation are returned.

    The returned table includes the default API fields, the ``observer``
    list for each measurement, and geodetic station coordinates joined
    onto each record for both the receiver (``rcvr_*``) and transmitter
    (``xmit_*``) stations.

    See https://ssd-api.jpl.nasa.gov/doc/sb_radar.html for field
    definitions.

    Parameters
    ==========
    desig :
        Object designation as it appears in the JPL ``des`` field (e.g.
        ``"99942"``, ``"1998 KY26"``, ``"45P"``). If ``None``, every
        record in the data set is returned.
    update_cache :
        Force download a new file from JPL, refreshing the cached parquet.
    """
    filename = os.path.join(cache_path(), "horizons_radar.parquet")
    if update_cache or not os.path.isfile(filename):
        res = requests.get(
            "https://ssd-api.jpl.nasa.gov/sb_radar.api?observer=1&coords=1",
            timeout=240,
        )
        res.raise_for_status()
        payload = res.json()

        columns = payload.get("fields", [])
        table = pd.DataFrame.from_records(payload.get("data", []), columns=columns)

        # observer is returned as a list per record; leave it as-is.
        string_cols = {"des", "epoch", "units", "rcvr", "xmit", "bp", "observer"}
        numeric_cols = [c for c in table.columns if c not in string_cols]
        if numeric_cols:
            table[numeric_cols] = table[numeric_cols].apply(
                pd.to_numeric, errors="coerce"
            )
        if "epoch" in table.columns:
            table["epoch"] = pd.to_datetime(table["epoch"], utc=True, errors="coerce")

        # Join station coordinates (geodetic) onto each record for receiver
        # and transmitter. Station codes are negative integers stored as
        # strings.
        coords = payload.get("coords", {})
        if coords and not table.empty:
            coord_rows = []
            for code, info in coords.items():
                coord_rows.append(
                    {
                        "code": str(code),
                        "name": info.get("name"),
                        "longitude": pd.to_numeric(
                            info.get("longitude"), errors="coerce"
                        ),
                        "latitude": pd.to_numeric(
                            info.get("latitude"), errors="coerce"
                        ),
                        "altitude": pd.to_numeric(
                            info.get("altitude"), errors="coerce"
                        ),
                        "alt_units": info.get("alt_units"),
                    }
                )
            coord_df = pd.DataFrame.from_records(coord_rows)
            for prefix in ("rcvr", "xmit"):
                renamed = coord_df.rename(
                    columns={
                        c: f"{prefix}_{c}" for c in coord_df.columns if c != "code"
                    }
                )
                table = table.merge(
                    renamed, how="left", left_on=prefix, right_on="code"
                ).drop(columns=["code"])

        table.to_parquet(filename, index=False)
    else:
        table = pd.read_parquet(filename)

    if desig is not None:
        table = table[table["des"] == desig].reset_index(drop=True)
    return table


def _build_radar_observer(jd: float, rec: dict, prefix: str):
    """Build an equatorial Earth-centered observer for one radar station.

    ``prefix`` is either ``"rcvr"`` or ``"xmit"`` and selects which set of
    joined coordinate columns to read from ``rec``.
    """
    lat = rec.get(f"{prefix}_latitude")
    lon = rec.get(f"{prefix}_longitude")
    alt = rec.get(f"{prefix}_altitude")
    alt_units = rec.get(f"{prefix}_alt_units") or "km"
    code = rec.get(prefix)
    try:
        lat = float(lat)  # type: ignore
        lon = float(lon)  # type: ignore
        alt = float(alt)  # type: ignore
    except ValueError:
        logger.debug(
            "Invalid coordinates for stn %s: lat=%s lon=%s alt=%s", code, lat, lon, alt
        )
        return None

    if alt_units == "m":
        alt_km = alt / 1000.0
    elif alt_units == "km":
        alt_km = alt
    else:
        logger.debug("Unsupported altitude units '%s' for stn %s", alt_units, code)
        return None
    try:
        return spice.earth_pos_to_ecliptic(
            jd, lat, lon, alt_km, name=str(code), center=10
        ).as_equatorial
    except Exception as exc:
        logger.debug("Failed to build observer for stn %s: %s", code, exc)
        return None


def fetch_radar_observations(desig: str, update_cache: bool = False):
    """
    Fetch JPL radar observations for an object as fitting Observations.

    Wraps :func:`fetch_radar_obs` and converts each record into either a
    :py:meth:`Observation.radar_range` or :py:meth:`Observation.radar_rate`
    instance ready for orbit fitting.

    Each record is converted using the station as the observer state.
    For bistatic measurements (transmitter != receiver) the observer is
    placed at the midpoint of the two stations; this approximates the
    bistatic round-trip path to second order in (baseline / range) and
    is accurate to a few meters for Earth-baseline NEO radar. Records
    with missing coordinates or unrecognized units are skipped.

    Conversions assumed:

    - Delay (``us``): round-trip light time. One-way range in AU is
      ``c * (delay_us * 1e-6) / 2 / AU_M``.
    - Doppler (``Hz``): two-way Doppler shift at carrier frequency
      ``freq`` (MHz). Range-rate in AU/day is
      ``-c * doppler / (2 * freq_Hz)`` converted from m/s.
      JPL convention: positive Doppler means approaching; kete
      convention: positive range-rate means receding, so the sign is
      flipped.

    Parameters
    ----------
    desig :
        Object designation as it appears in the JPL ``des`` field
        (e.g. ``"99942"``, ``"1998 KY26"``).
    update_cache :
        If ``True``, refresh the cached master parquet before filtering.

    Returns
    -------
    list[Observation]
        One ``Observation.radar_range`` or ``Observation.radar_rate`` per
        valid monostatic record.
    """
    table = fetch_radar_table(desig=desig, update_cache=update_cache)
    if table.empty:
        return []

    c_m_s = constants.SPEED_OF_LIGHT
    au_m = constants.AU_M
    sec_per_day = 86400.0

    observations = []
    for rec in table.to_dict(orient="records"):
        rcvr = rec.get("rcvr")
        xmit = rec.get("xmit")
        if rcvr is None or xmit is None:
            continue

        epoch = rec.get("epoch")
        if epoch is None or pd.isna(epoch):
            continue
        jd = Time.from_iso(pd.Timestamp(epoch).isoformat()).jd

        rcvr_state = _build_radar_observer(jd, rec, "rcvr")
        if rcvr_state is None:
            continue

        if xmit == rcvr:
            observer = rcvr_state
        else:
            xmit_state = _build_radar_observer(jd, rec, "xmit")
            if xmit_state is None:
                continue
            # Midpoint approximation for bistatic geometry: place the
            # observer at (xmit + rcvr) / 2. The round-trip path
            # |xmit -> tgt -> rcvr| equals 2 * |midpoint -> tgt| to
            # second order in (baseline / range), so reusing the
            # monostatic predictor incurs only a baseline^2 / (8*range)
            # error (a few meters for typical Earth baselines and NEO
            # ranges).
            mid_pos = (rcvr_state.pos + xmit_state.pos) * 0.5
            mid_vel = (rcvr_state.vel + xmit_state.vel) * 0.5
            observer = State(
                f"{rcvr}+{xmit}",
                jd,
                mid_pos,
                mid_vel,
                Frames.Equatorial,
                center_id=rcvr_state.center_id,
            )

        value = rec.get("value")
        sigma = rec.get("sigma")
        units = rec.get("units")
        if value is None or pd.isna(value) or units is None:
            continue

        if units == "us":
            # Round-trip delay in microseconds -> one-way range in AU.
            range_au = c_m_s * (float(value) * 1e-6) / 2.0 / au_m
            sigma_au = c_m_s * (float(sigma) * 1e-6) / 2.0 / au_m
            if not (sigma_au > 0):
                continue
            observations.append(
                Observation.radar_range(
                    observer=observer, range=range_au, sigma_range=sigma_au
                )
            )
        elif units == "Hz":
            freq_mhz = rec.get("freq")
            if freq_mhz is None or pd.isna(freq_mhz) or freq_mhz <= 0:
                continue
            # Two-way Doppler: range_rate_m_s = -c * doppler / (2 * f_carrier).
            # Sign flip converts JPL "approach positive" to kete "recede positive".
            f_hz = float(freq_mhz) * 1e6
            rr_m_s = -c_m_s * float(value) / (2.0 * f_hz)
            sigma_m_s = c_m_s * float(sigma) / (2.0 * f_hz)
            # Convert m/s -> AU/day
            rr_au_day = rr_m_s * sec_per_day / au_m
            sigma_au_day = sigma_m_s * sec_per_day / au_m
            if not (sigma_au_day > 0):
                continue
            observations.append(
                Observation.radar_rate(
                    observer=observer,
                    range_rate=rr_au_day,
                    sigma_range_rate=sigma_au_day,
                )
            )
        else:
            logger.debug("Unsupported radar units '%s'", units)
            continue

    return observations
