"""MPC ADES API: fetch observations and convert to fitting Observations."""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
import os

import numpy as np
import requests

from ..cache import cache_path
from ..fitting import Observation
from ..time import Time
from ..vector import Frames, State
from ._common import (
    _fetch_debias_table,
    _over_obs_reweight_factors,
    get_observatory_std,
)

logger = logging.getLogger(__name__)

# ADES ``astCat`` name -> single-character MPC catalog code.
# Source: EFCC18 ``bias.dat`` header cross-referenced with
# https://www.minorplanetcenter.net/iau/info/CatalogueCodes.html
# Add new entries cautiously; a misassignment silently applies the wrong correction.
_ADES_TO_MPC_CODE: dict[str, str] = {
    "USNOA1": "a",
    "USNOSA1": "b",
    "USNOA2": "c",
    "USNOSA2": "d",
    "UCAC1": "e",
    "Tyc2": "g",
    "GSC1.1": "i",
    "GSC1.2": "j",
    "ACT": "l",
    "GSCACT": "m",
    "SDSS8": "n",
    "USNOB1": "o",
    "PPM": "p",
    "UCAC4": "q",
    "UCAC2": "r",
    "PPMXL": "t",
    "UCAC3": "u",
    "NOMAD": "v",
    "CMC14": "w",
    "2MASS": "L",
    "SDSS7": "N",
    "CMC15": "Q",
    "SSTRC4": "R",
    "URAT1": "S",
    "GAIA1": "U",
    "UCAC5": "Y",
}


def _build_observer(stn: str, jd: float, rec: dict):
    """Return an SSB-centered equatorial observer State from an ADES record, or None."""
    from .. import constants, spice
    from ..vector import Vector

    # Ground-station lookup.
    try:
        obs = spice.mpc_code_to_ecliptic(stn, jd, center=0).as_equatorial
        if obs.is_finite:
            return obs
    except Exception:
        pass

    # Fallback: pos1/pos2/pos3 from ADES record (satellite/roving observers).
    pos1, pos2, pos3 = rec.get("pos1"), rec.get("pos2"), rec.get("pos3")
    if pos1 is None or pos2 is None or pos3 is None:
        return None
    try:
        sys = rec.get("sys", "").upper()
        ctr = int(float(rec.get("ctr", 399)))

        pos_km = np.array([float(pos1), float(pos2), float(pos3)])
        if sys == "ICRF_AU":
            pos_au = pos_km
        elif sys == "ICRF_KM":
            pos_au = pos_km / constants.AU_KM
        elif sys == "WGS84":
            lon, lat, alt = float(pos1), float(pos2), float(pos3)
            return spice.earth_pos_to_ecliptic(
                jd, lat, lon, alt, name=stn, center=10
            ).as_equatorial
        else:
            logger.warning(
                "Unsupported ADES coordinate system '%s' for stn %s", sys, stn
            )
            return None

        center_state = spice.get_state(ctr, jd, center=10).as_equatorial
        ssb_pos = center_state.pos + Vector(list(pos_au), Frames.Equatorial)
        return State(
            stn,
            jd,
            ssb_pos,
            center_state.vel,
            Frames.Equatorial,
            center_id=center_state.center_id,
        )
    except Exception:
        return None


def fetch_mpc_observations(
    desig: str,
    use_observatory_residuals: bool = True,
    debias: bool = True,
    apply_over_obs_reweight: bool = True,
    update_cache: bool = False,
) -> list[Observation]:
    """
    Fetch observations from the MPC API and convert to fitting Observations.

    Queries ``https://data.minorplanetcenter.net/api/get-obs`` for the
    given object designation and returns optical observations ready for
    orbit fitting.  Only optical records with valid RA/Dec are included;
    radar and other types are silently skipped.

    Results are cached under ``~/.kete/observations/`` so that repeated
    queries for the same designation do not hit the network.

    Uncertainties are first taken from a pre-computed table of residual error by
    observatory code, if available.  If not, the MPC-provided ``rmsra`` and
    ``rmsdec`` fields are used, defaulting to 1 arcsecond if not provided.

    When ``debias`` is True, the EFCC18 star-catalog bias correction is applied
    to each observation's (RA, Dec) using the ADES ``astCat`` field.
    Observations with unknown or missing catalog codes are passed through
    unchanged.

    When ``apply_over_obs_reweight`` is True, sigma is inflated by
    sqrt(n/4) for groups of more than 4 observations from the same
    observatory on the same night, following Veres et al. 2017.

    Parameters
    ----------
    desig :
        Object designation recognized by the MPC (e.g. ``"Apophis"``,
        ``"101955"``, ``"1999 RQ36"``).
    use_observatory_residuals :
        If ``True``, apply per-observatory sigmas from the pre-computed
        residual table when available; otherwise, use the MPC-provided
        ``rmsra`` and ``rmsdec`` fields, defaulting to 1 arcsecond.
    debias :
        If ``True``, apply the EFCC18 star-catalog debiasing correction.
        Requires the JPL ``debias_2018.tgz`` archive, downloaded on first use.
    apply_over_obs_reweight :
        When True (default), inflate sigma for over-observed nights following
        Veres et al. 2017.
    update_cache :
        If ``True``, discard any cached result and re-query the MPC.

    Returns
    -------
    list[Observation]
        One ``Observation.optical`` per valid optical record.

    Raises
    ------
    RuntimeError
        If the MPC API request fails.

    Examples
    --------
    .. testcode::
        :skipif: True

        import kete

        observations = kete.observations.fetch_mpc_observations("Apophis")
    """
    _hash = hashlib.md5(desig.encode()).hexdigest()[:16]
    obs_dir = os.path.join(cache_path(sub_path="observations"), _hash[:3])
    os.makedirs(obs_dir, exist_ok=True)
    cached_path = os.path.join(obs_dir, f"{_hash}.json.gz")

    if os.path.isfile(cached_path) and not update_cache:
        logger.debug("Loading cached MPC observations for '%s'", desig)
        with gzip.open(cached_path, "rb") as f:
            records = json.loads(f.read().decode())
    else:
        response = requests.get(
            "https://data.minorplanetcenter.net/api/get-obs",
            json={"desigs": [desig], "output_format": ["ADES_DF"]},
            timeout=120,
        )
        response.raise_for_status()
        records = response.json()
        if not records:
            return []
        records = records[0]
        with gzip.open(cached_path, "wb") as f:
            f.write(json.dumps(records).encode())

    if not records:
        return []

    debias_table = _fetch_debias_table() if debias else None

    # validate and collect all fields needed for reweighting.
    valid = []
    for rec in records["ADES_DF"]:
        if rec.get("Obstype") != "optical":
            continue
        ra_str = rec.get("ra")
        dec_str = rec.get("dec")
        if ra_str is None or dec_str is None:
            continue
        ra_deg = float(ra_str)
        dec_deg = float(dec_str)

        obstime = rec.get("obstime")
        if obstime is None:
            continue
        try:
            jd = Time.from_iso(obstime).jd
        except ValueError:
            continue

        stn = rec.get("stn")
        if stn is None:
            continue

        observer = _build_observer(stn, jd, rec)
        if observer is None:
            continue

        obs_errors = get_observatory_std(stn)
        if obs_errors is not None and use_observatory_residuals:
            s_ra, s_dec = obs_errors
        else:
            s_ra = float(rec.get("rmsra") or 1.0)
            s_dec = float(rec.get("rmsdec") or 1.0)

        if debias_table is not None:
            astcat = rec.get("astcat")
            if astcat:
                mpc_code = _ADES_TO_MPC_CODE.get(astcat.upper())
                if mpc_code:
                    shift = debias_table.lookup(mpc_code, ra_deg, dec_deg, jd)
                    if shift is not None:
                        ra_deg -= shift[0] / 3600.0
                        dec_deg -= shift[1] / 3600.0

        band = rec.get("band") or "V"
        try:
            mag = float(rec.get("mag"))
        except (TypeError, ValueError):
            mag = float("nan")

        valid.append(
            {
                "stn": stn,
                "jd": jd,
                "ra": ra_deg,
                "dec": dec_deg,
                "s_ra": s_ra,
                "s_dec": s_dec,
                "band": band,
                "mag": mag,
                "observer": observer,
            }
        )

    if not valid:
        return []

    # per-night sigma reweighting.
    if apply_over_obs_reweight:
        factors = _over_obs_reweight_factors(
            [r["stn"] for r in valid],
            [r["jd"] for r in valid],
            [False] * len(valid),  # ADES format does not distinguish spacecraft
        )
    else:
        factors = [1.0] * len(valid)

    # build Observation objects.
    observations = []
    for r, factor in zip(valid, factors):
        observations.append(
            Observation.optical(
                observer=r["observer"],
                ra=r["ra"],
                dec=r["dec"],
                sigma_ra=float(r["s_ra"]) * factor,
                sigma_dec=float(r["s_dec"]) * factor,
                band=r["band"],
                mag=r["mag"],
            )
        )

    return observations
