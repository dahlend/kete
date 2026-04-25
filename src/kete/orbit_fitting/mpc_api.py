"""MPC ADES API: fetch observations and convert to fitting Observations."""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
import os

import numpy as np
import requests

from .._core import Observation
from ..cache import cache_path
from ..time import Time
from ..vector import Frames, State
from .common import (
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

    # Pass 1: validate and collect all fields needed for reweighting.
    valid = []
    for rec in records["ADES_DF"]:
        obstype = rec.get("Obstype")

        if obstype == "optical":
            ra_str = rec.get("ra")
            dec_str = rec.get("dec")
            if ra_str is None or dec_str is None:
                continue
            ra_deg = float(ra_str)
            dec_deg = float(dec_str)
            is_occultation = False

        elif obstype == "occultation":
            # Occultation: the asteroid occulted a background star.  The
            # reported star position (rastar, decstar) is the asteroid's sky
            # location, offset by (deltara, deltadec) in arcseconds.  The
            # per-observation rmsra/rmsdec/rmscorr uncertainties come directly
            # from the Gaia-referenced star catalog -- typically sub-mas --
            # so the per-observatory sigma table is not used.
            rastar = rec.get("rastar")
            decstar = rec.get("decstar")
            if rastar is None or decstar is None:
                continue
            dec_deg = float(decstar) + float(rec.get("deltadec") or 0.0) / 3600.0
            cos_dec = np.cos(np.radians(dec_deg))
            ra_deg = float(rastar) + float(rec.get("deltara") or 0.0) / (
                3600.0 * max(cos_dec, 1e-6)
            )
            is_occultation = True

        else:
            continue

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

        # Observation.optical expects sky-plane sigma_ra (sigma_ra * cos(dec))
        # to match the convention of MPC ADES, Gaia DR3, and other astrometric
        # data formats.
        #
        # Sigma resolution priority (matches JPL Horizons practice):
        #   1. Per-observation ADES rmsra/rmsdec when present (modern submitters
        #      report measurement-specific uncertainties that reflect actual
        #      conditions on the night).
        #   2. Per-observatory residual table from get_observatory_std().
        #   3. Hardcoded 1.0 arcsec fallback.
        # Occultation records always come through path 1; the per-observatory
        # table reflects optical astrometry residuals and does not apply.
        ades_rmsra = rec.get("rmsra")
        ades_rmsdec = rec.get("rmsdec")
        used_table = False
        if ades_rmsra is not None and ades_rmsdec is not None:
            s_ra = float(ades_rmsra)
            s_dec = float(ades_rmsdec)
        elif (
            not is_occultation
            and use_observatory_residuals
            and (obs_errors := get_observatory_std(stn)) is not None
        ):
            # Table ra_std is computed from raw RA coordinate residuals;
            # multiply by cos(dec) to match the sky-plane input convention.
            s_ra = obs_errors[0] * np.cos(np.radians(dec_deg))
            s_dec = obs_errors[1]
            used_table = True
        else:
            s_ra = 1.0
            s_dec = 1.0

        # ADES ``rmscorr`` is the RA/Dec correlation coefficient in [-1, 1].
        # Occultation records routinely have significant correlation (the
        # timing uncertainty stretches the ellipse along-track).  When falling
        # back to per-observatory sigmas, the table carries no correlation, so
        # we leave it at zero.  Clamp strictly inside (-1, 1).
        sigma_corr = 0.0
        if not used_table:
            rmscorr = rec.get("rmscorr")
            if rmscorr is not None:
                try:
                    rmscorr = float(rmscorr)
                    if np.isfinite(rmscorr):
                        sigma_corr = max(min(rmscorr, 0.999), -0.999)
                except (TypeError, ValueError):
                    pass

        if debias_table is not None and not is_occultation:
            # Occultation positions are tied to the Gaia reference frame
            # directly; the EFCC18 catalog-bias correction does not apply.
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
                "sigma_corr": sigma_corr,
                "band": band,
                "mag": mag,
                "observer": observer,
                "is_occultation": is_occultation,
            }
        )

    if not valid:
        return []

    # Pass 2: per-night sigma reweighting (occultations are always independent).
    if apply_over_obs_reweight:
        factors = _over_obs_reweight_factors(
            [r["stn"] for r in valid],
            [r["jd"] for r in valid],
            [r["is_occultation"] for r in valid],
        )
    else:
        factors = [1.0] * len(valid)

    # Pass 3: build Observation objects.
    observations = []
    for r, factor in zip(valid, factors):
        # Occultation positional uncertainties already encode the timing
        # constraint (rmsra/rmsdec are derived from timing error projected
        # onto the sky).  Set time_sigma=0 to avoid double-counting.
        time_sigma = 0.0 if r["is_occultation"] else 0.1
        observations.append(
            Observation.optical(
                observer=r["observer"],
                ra=r["ra"],
                dec=r["dec"],
                sigma_ra=float(r["s_ra"]) * factor,
                sigma_dec=float(r["s_dec"]) * factor,
                sigma_corr=r["sigma_corr"],
                band=r["band"],
                mag=r["mag"],
                time_sigma=time_sigma,
                is_occultation=r["is_occultation"],
            )
        )

    return observations
