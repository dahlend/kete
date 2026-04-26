"""JPL Small-Body radar astrometry: fetching and conversion to Observations."""

from __future__ import annotations

import logging
import os

import pandas as pd
import requests

from .. import constants
from .._core import Observation
from ..cache import cache_path
from ..time import Time

logger = logging.getLogger(__name__)


def fetch_radar_table(desig: str | None = None, update_cache: bool = False):
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
    ----------
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

        string_cols = {"des", "epoch", "units", "rcvr", "xmit", "bp", "observer"}
        numeric_cols = [c for c in table.columns if c not in string_cols]
        if numeric_cols:
            table[numeric_cols] = table[numeric_cols].apply(
                pd.to_numeric, errors="coerce"
            )
        if "epoch" in table.columns:
            table["epoch"] = pd.to_datetime(table["epoch"], utc=True, errors="coerce")

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


def _station_coords(rec: dict, prefix: str) -> tuple[float, float, float] | None:
    """Extract WGS84 geodetic (lat_deg, lon_deg, height_km) for a station.

    ``prefix`` is either ``"rcvr"`` or ``"xmit"``.  Returns None when any
    coordinate is missing or the altitude units are unsupported.
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
    except (TypeError, ValueError):
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
    return (lat, lon, alt_km)


def fetch_radar_observations(
    desig: str, update_cache: bool = False
) -> list[Observation]:
    """
    Fetch JPL radar observations for an object as fitting Observations.

    Wraps :func:`fetch_radar_table` and converts each record into either a
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
        valid record.
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
        # JPL publishes radar epochs as the receive time (UTC).
        jd_rx = Time.from_iso(pd.Timestamp(epoch).isoformat()).jd

        value = rec.get("value")
        sigma = rec.get("sigma")
        units = rec.get("units")
        if value is None or pd.isna(value) or units is None:
            continue

        # Extract the WGS84 geodetic coordinates for both stations.  The
        # Rust residual computes their inertial states at the appropriate
        # epochs (rcvr at t_rx, xmit at iteratively-refined t_tx) via
        # PCK kernel lookups.
        rcvr_coords = _station_coords(rec, "rcvr")
        xmit_coords = _station_coords(rec, "xmit")
        if rcvr_coords is None or xmit_coords is None:
            continue
        rcvr_lat, rcvr_lon, rcvr_height = rcvr_coords
        xmit_lat, xmit_lon, xmit_height = xmit_coords

        if units == "us":
            range_au = c_m_s * (float(value) * 1e-6) / 2.0 / au_m
            sigma_au = c_m_s * (float(sigma) * 1e-6) / 2.0 / au_m
            if not (sigma_au > 0):
                continue
            observations.append(
                Observation.radar_range(
                    xmit_lat=xmit_lat,
                    xmit_lon=xmit_lon,
                    xmit_height=xmit_height,
                    rcvr_lat=rcvr_lat,
                    rcvr_lon=rcvr_lon,
                    rcvr_height=rcvr_height,
                    epoch=Time(jd_rx),
                    range=range_au,
                    sigma_range=sigma_au,
                )
            )
        elif units == "Hz":
            freq_mhz = rec.get("freq")
            if freq_mhz is None or pd.isna(freq_mhz) or freq_mhz <= 0:
                continue
            f_hz = float(freq_mhz) * 1e6
            rr_m_s = -c_m_s * float(value) / (2.0 * f_hz)
            sigma_m_s = c_m_s * float(sigma) / (2.0 * f_hz)
            rr_au_day = rr_m_s * sec_per_day / au_m
            sigma_au_day = sigma_m_s * sec_per_day / au_m
            if not (sigma_au_day > 0):
                continue
            observations.append(
                Observation.radar_rate(
                    xmit_lat=xmit_lat,
                    xmit_lon=xmit_lon,
                    xmit_height=xmit_height,
                    rcvr_lat=rcvr_lat,
                    rcvr_lon=rcvr_lon,
                    rcvr_height=rcvr_height,
                    epoch=Time(jd_rx),
                    range_rate=rr_au_day,
                    sigma_range_rate=sigma_au_day,
                )
            )
        else:
            logger.debug("Unsupported radar units '%s'", units)
            continue

    return observations
