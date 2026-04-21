from __future__ import annotations

import gzip
import hashlib
import json
import logging
import os
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd
import requests

from . import constants, conversion, spice
from ._core import _find_obs_code, pack_designation, unpack_designation
from .cache import cache_path, download_json
from .fitting import Observation
from .time import Time
from .vector import Frames, State, Vector

__all__ = [
    "unpack_designation",
    "pack_designation",
    "fetch_known_designations",
    "fetch_known_orbit_data",
    "fetch_known_comet_orbit_data",
    "fetch_mpc_observations",
    "find_obs_code",
]

logger = logging.getLogger(__name__)


@lru_cache
def find_obs_code(site):
    return _find_obs_code(site)


find_obs_code.__doc__ = _find_obs_code.__doc__


@lru_cache
def fetch_known_designations(force_download=False):
    """
    Download the most recent copy of the MPCs known ID mappings in their unpacked
    format.

    This download only occurs the first time this function is called.

    This then returns a dictionary of all known unpacked IDs to a single ID which is the
    one that the MPC specifies as their default.

    For example, here are the first two objects which are returned:

    {'1': '1',
    'A801 AA': '1',
    'A899 OF': '1',
    '1943 XB': '1',
    '2': '2',
    'A802 FA': '2',
    ...}

    Ceres has 4 entries, which all map to '1'.
    """
    # download the data from the MPC
    known_ids = download_json(
        "https://minorplanetcenter.net/Extended_Files/mpc_ids.json.gz",
        force_download,
    )

    # The data which is in the format {'#####"; ['#####', ...], ...}
    # where the keys of the dictionary are the MPC default name, and the values are the
    # other possible names.
    # Reshape the MPC dictionary to be flat, with every possible name mapping to the
    # MPC default name.
    desig_map = {}
    for name, others in known_ids.items():
        desig_map[name] = name
        for other in others:
            desig_map[other] = name
    return desig_map


@lru_cache
def fetch_known_packed_to_full_names(force_download=False):
    """
    Download the most recent copy of the MPCs known ID mappings in their packed format.

    This download only occurs the first time this function is called.

    This then returns a dictionary of all known packed IDs to a full unpacked name if it
    exists.

    For example, here are the first two objects which are returned:

    {'00001': 'Ceres',
    'I01A00A': 'Ceres',
    'I99O00F': 'Ceres',
    'J43X00B': 'Ceres',
    '00002': 'Pallas',
    'I02F00A': 'Pallas',
    ...}

    Ceres has 4 entries, since it has 4 unique packed designations.
    """
    orb = fetch_known_orbit_data(force_download=force_download)
    packed_ids = download_json(
        "https://minorplanetcenter.net/Extended_Files/mpc_ids_packed.json.gz",
        force_download,
    )
    lookup = {}
    for row in orb.itertuples():
        lookup[row.desig] = row.name
        if row.desig in packed_ids:
            for other in packed_ids[row.desig]:
                lookup[other] = row.name
    return lookup


@lru_cache
def fetch_known_orbit_data(url=None, force_download=False):
    """
    Download the orbital elements from the MPC at the specified URL.

    Object names are set to the packed normalized MPC representation.

    This loads the ``*.json.gz`` files located in the ``Orbits`` category located at
    https://minorplanetcenter.net/data

    This doesn't work with the comet file on the MPC website as they have a different
    file format, see the function ``fetch_known_comet_orbit_data``.

    Example URLS:

        | Full MPCORB data for all asteroids in the MPC database
        | https://minorplanetcenter.net/Extended_Files/mpcorb_extended.json.gz
        | NEAs
        | https://minorplanetcenter.net/Extended_Files/nea_extended.json.gz
        | PHAs
        | https://minorplanetcenter.net/Extended_Files/pha_extended.json.gz
        | Latest DOU MPEC
        | https://minorplanetcenter.net/Extended_Files/daily_extended.json.gz
        | Orbits for TNOs, Centaurs, and SDOs
        | https://minorplanetcenter.net/Extended_Files/distant_extended.json.gz
        | Orbits for asteroids with e> 0.5 and q > 6 AU
        | https://minorplanetcenter.net/Extended_Files/unusual_extended.json.gz

    """
    if url is None:
        url = "https://minorplanetcenter.net/Extended_Files/mpcorb_extended.json.gz"
    objs = download_json(url, force_download)
    objects = []
    for obj in objs:
        # "Principal_design" is always a preliminary designation
        # Number is defined if it has a permanent designation, so look for that first
        if "Number" in obj:
            desig = int(obj["Number"].replace("(", "").replace(")", ""))
        else:
            desig = obj["Principal_desig"]

        arc_len = obj.get("Arc_length", None)
        if arc_len is None and "Arc_years" in obj:
            t0, t1 = obj["Arc_years"].split("-")
            arc_len = (float(t1) - float(t0)) * 365.25

        props = dict(
            desig=desig,
            g_phase=obj.get("G", None),
            h_mag=obj.get("H", None),
            group_name=obj.get("Orbit_type", None),
            peri_dist=obj["Perihelion_dist"],
            ecc=obj["e"],
            incl=obj["i"],
            lon_node=obj["Node"],
            peri_arg=obj["Peri"],
            peri_time=Time(obj["Tp"], scaling="utc").jd,
            epoch=Time(obj["Epoch"], scaling="utc").jd,
            arc_len=arc_len,
            name=obj.get("Name", None),
        )
        objects.append(props)
    return pd.DataFrame.from_records(objects)


@lru_cache
def fetch_known_comet_orbit_data(force_download=False):
    """
    Download the orbital elements for comets from the MPC at the specified URL.

    This returns a list of :class:`~dict`, one for each orbital element fetched from the
    MPC. Object names are set to the packed normalized MPC representation.
    """
    url = "https://minorplanetcenter.net/Extended_Files/cometels.json.gz"
    objs = download_json(url, force_download)
    objects = []
    for comet in objs:
        name = comet.get("Designation_and_name").split("(")[0]
        peri_time = (
            comet["Year_of_perihelion"],
            comet["Month_of_perihelion"],
            comet["Day_of_perihelion"],
        )
        epoch_time = peri_time
        if "Epoch_year" in comet:
            epoch_time = (comet["Epoch_year"], comet["Epoch_month"], comet["Epoch_day"])

        obj = dict(
            desig=name,
            group_name=f"Comet {comet['Orbit_type']}",
            peri_dist=comet["Perihelion_dist"],
            ecc=comet["e"],
            incl=comet["i"],
            lon_node=comet["Node"],
            peri_arg=comet["Peri"],
            peri_time=Time.from_ymd(*peri_time).jd,
            epoch=Time.from_ymd(*epoch_time).jd,
        )
        objects.append(obj)
    return pd.DataFrame.from_records(objects)


@dataclass
class MPCObservation:
    """
    Representation of an observation in the MPC observation files.

    .. testcode::
        :skipif: True

        import kete
        import gzip

        # Comet Observations
        # url = "https://www.minorplanetcenter.net/iau/ECS/MPCAT-OBS/CmtObs.txt.gz"

        # Download the database of unnumbered observations from the MPC
        url = "https://www.minorplanetcenter.net/iau/ECS/MPCAT-OBS/UnnObs.txt.gz"
        path = kete.data.download_file(url)

        # Fetch all lines from the file which contain C51 (WISE) observatory code.
        obs_code = "C51".encode()
        with gzip.open(path) as f:
            lines = [line.decode() for line in f if obs_code == line[77:80]]

        # Parse lines into a list of MPCObservations
        observations = kete.mpc.MPCObservation.from_lines(lines)

    """

    desig: str
    prov_desig: str
    discovery: bool
    note1: str
    note2: str
    jd: float
    ra: float
    dec: float
    mag_band: str
    obs_code: str
    sun2sc: list[float]

    _UNSUPPORTED = set("WwQqVvRrXxTt")

    def __post_init__(self):
        if self.sun2sc is None:
            self.sun2sc = [np.nan, np.nan, np.nan]
        self.sun2sc = list(self.sun2sc)

    @classmethod
    def from_lines(cls, lines, load_sc_pos=True):
        """
        Create a list of MPCObservations from a list of single 80 char lines.
        """
        found = []
        idx = 0
        while True:
            if idx >= len(lines):
                break
            line = cls._read_first_line(lines[idx])
            idx += 1
            if line is None:
                continue
            if line["note2"] == "s":
                logger.warning("Second line of spacecraft observation found alone")
                continue
            elif line["note2"] == "S":
                if idx >= len(lines):
                    logger.warning("Missing second line of spacecraft observation.")
                    break
                pos_line = lines[idx]
                idx += 1
                if load_sc_pos:
                    line["sun2sc"] = cls._read_second_line(pos_line, line["jd"])
            found.append(cls(**line))
        return found

    @staticmethod
    def _read_first_line(line):
        if line[14] in MPCObservation._UNSUPPORTED:
            # unsupported or deprecated observation types
            return None

        mag_band = line[65:71].strip()

        year, month, day = line[15:32].strip().split()
        jd = Time.from_ymd(int(year), int(month), float(day)).jd
        if len(mag_band) > 0:
            mag_band = mag_band.split(maxsplit=1)[0]

        ra = conversion.ra_hms_to_degrees(line[32:44].strip())
        dec = conversion.dec_dms_to_degrees(line[44:55].strip())

        try:
            desig = unpack_designation(line[:5])
        except ValueError:
            desig = line[:5].strip()
        try:
            prov_desig = unpack_designation(line[5:12].strip())
        except ValueError:
            prov_desig = line[5:12].strip()

        contents = dict(
            desig=desig,
            prov_desig=prov_desig,
            discovery=line[12] == "*",
            note1=line[13].strip(),
            note2=line[14].strip(),
            ra=ra,
            dec=dec,
            mag_band=mag_band,
            obs_code=line[77:80],
            sun2sc=None,
            jd=jd,
        )
        return contents

    @staticmethod
    def _read_second_line(line, jd):
        from . import spice

        if line[14] != "s":
            raise SyntaxError("No second line of spacecraft observation found.")

        x = float(line[34:45].replace(" ", "")) / constants.AU_KM
        y = float(line[46:57].replace(" ", "")) / constants.AU_KM
        z = float(line[58:69].replace(" ", "")) / constants.AU_KM
        earth2sc = Vector([x, y, z], Frames.Equatorial).as_ecliptic
        sun2earth = spice.get_state("Earth", jd).pos
        sun2sc = sun2earth + earth2sc
        return list(sun2sc)

    @property
    def sc2obj(self):
        return Vector.from_ra_dec(self.ra, self.dec).as_ecliptic


def _parse_sigma(value, default: float) -> float:
    """Return a finite positive sigma, or *default* if *value* is unusable."""
    if value is None:
        return default
    try:
        v = float(value)
    except (ValueError, TypeError):
        return default
    if not np.isfinite(v) or v <= 0:
        return default
    return v


# Per-station 1-sigma astrometric weights (arcseconds, per component) used
# as a fallback when the MPC ADES record does not carry rmsRA/rmsDec.
#
# Values are taken directly from Veres, Farnocchia, Chesley, Chamberlin
# 2017, "Statistical analysis of the astrometric errors for the most
# productive asteroid surveys" (Icarus 296, 139-149), Tables 2-4.  Only
# the per-station/per-catalog/per-mode weight is applied here; catalog-bias
# debiasing (Farnocchia et al. 2015 / Eggl et al. 2020) is a separate
# step and is not performed by this function.

# Table 2: epoch-dependent station weights (cutoff JD, sigma_before, sigma_after).
# JD 2456658.5 = 2014-01-01, 2452640.5 = 2003-01-01, 2452883.5 = 2003-09-01.
_VERES_TABLE2: dict[str, tuple[float, float, float]] = {
    "703": (2456658.5, 1.0, 0.8),
    "691": (2452640.5, 0.6, 0.5),
    "644": (2452883.5, 0.6, 0.4),
}

# Table 3: epoch-independent station weights for the most active CCD surveys.
_VERES_TABLE3: dict[str, float] = {
    "704": 1.0,
    "G96": 0.5,
    "F51": 0.2,
    "G45": 0.6,
    "699": 0.8,
    "D29": 0.75,
    "C51": 1.0,
    "E12": 0.75,
    "608": 0.6,
    "J75": 1.0,
}

# Table 4: NEO follow-up observers (catalog-dependent in some cases).
# Mapping of MPC code to either:
#   - a single float (catalog-independent), or
#   - a dict {catalog_code: sigma, ...} with key "_default" used as
#     fallback when the catalog is not listed.
# LCO codes share a single 0.4" weight (table caption lists them).
_LCO_CODES = {
    "K92",
    "K93",
    "Q63",
    "Q64",
    "V37",
    "W84",
    "W85",
    "W86",
    "W87",
    "K91",
    "E10",
    "F65",
}
_VERES_TABLE4: dict[str, float | dict[str, float]] = {
    "645": 0.3,
    "673": 0.3,
    "689": 0.5,
    "950": 0.5,
    "H01": 0.3,
    "J04": 0.4,
    "W84": 0.5,
    "Y28": {"PPMXL": 0.3, "Gaia1": 0.3, "Gaia2": 0.3, "Gaia3": 0.3, "_default": 0.3},
    "568": {
        "USNO-B1.0": 0.5,
        "USNO-B2.0": 0.5,
        "Gaia1": 0.1,
        "Gaia2": 0.1,
        "Gaia3": 0.1,
        "PPMXL": 0.2,
        "_default": 0.5,
    },
    "T09": {"Gaia1": 0.1, "Gaia2": 0.1, "Gaia3": 0.1, "_default": 0.5},
    "T12": {"Gaia1": 0.1, "Gaia2": 0.1, "Gaia3": 0.1, "_default": 0.5},
    "T14": {"Gaia1": 0.1, "Gaia2": 0.1, "Gaia3": 0.1, "_default": 0.5},
    "G83": {
        "UCAC4": 0.3,
        "PPMXL": 0.3,
        "Gaia1": 0.2,
        "Gaia2": 0.2,
        "Gaia3": 0.2,
        "_default": 0.3,
    },
    "309": {
        "UCAC4": 0.3,
        "PPMXL": 0.3,
        "Gaia1": 0.2,
        "Gaia2": 0.2,
        "Gaia3": 0.2,
        "_default": 0.3,
    },
}
for _c in _LCO_CODES:
    _VERES_TABLE4.setdefault(_c, 0.4)

# Table 5: non-CCD modes / pre-CCD photographic plates (sigma in arcsec).
# Photographic uses three epoch buckets (cutoff JDs in TDB):
#   2411368.5 = 1890-01-01, 2433282.5 = 1950-01-01.
_VERES_TABLE5_NON_CCD: dict[str, float] = {
    "OCC": 0.2,  # occultation
    "HIP": 0.2,  # Hipparcos
    "MER": 0.5,  # transit circle / meridian
    "ENC": 0.75,  # encoder
    "MIC": 2.0,  # micrometer
    "SAT": 1.5,  # satellite
    "NOR": 1.0,  # normal place
}


def _veres_2017_sigma(
    stn: str | None,
    mode: str | None,
    astcat: str | None,
    jd: float,
) -> float:
    """1-sigma astrometric uncertainty (arcsec) per Veres et al. 2017.

    Implements the weighting tables in Section 3 of the paper.  Returns a
    single value used for both RA (cos(dec)-corrected) and Dec.
    """
    mode_u = (mode or "").upper()

    # Non-CCD modes are dispatched first (Table 5).
    if mode_u in _VERES_TABLE5_NON_CCD:
        return _VERES_TABLE5_NON_CCD[mode_u]
    if mode_u in ("PHA", "PHO", "PH", "A", "N"):
        # Photographic: epoch-binned weights.
        if jd < 2411368.5:
            return 10.0
        if jd < 2433282.5:
            return 5.0
        return 2.5

    # Table 2: epoch-dependent surveys.
    if stn is not None and stn in _VERES_TABLE2:
        cutoff, sig_before, sig_after = _VERES_TABLE2[stn]
        return sig_before if jd < cutoff else sig_after

    # Table 4: follow-up observers (potentially catalog-dependent).
    if stn is not None and stn in _VERES_TABLE4:
        entry = _VERES_TABLE4[stn]
        if isinstance(entry, dict):
            cat = (astcat or "").strip()
            return entry.get(cat, entry.get("_default", 1.0))
        return entry

    # Table 3: epoch-independent CCD surveys.
    if stn is not None and stn in _VERES_TABLE3:
        return _VERES_TABLE3[stn]

    # Other CCD: 1.0" if catalog known, 1.5" if unknown (Section 3 text).
    if astcat:
        return 1.0
    return 1.5


def _build_observer(stn: str, jd: float, rec: dict):
    """Return an SSB-centered equatorial observer State, or None."""
    # Ground-station lookup.
    try:
        obs = spice.mpc_code_to_ecliptic(stn, jd, center=0).as_equatorial
        if obs.is_finite:
            return obs
    except Exception:
        pass

    # Fallback: pos1/pos2/pos3 from ADES record (satellite/roving observers).
    # The record includes 'sys' (coordinate system) and 'ctr' (center body
    # NAIF ID).  We require ICRF_KM or ICRF_AU; other systems are not yet
    # supported.
    pos1, pos2, pos3 = rec.get("pos1"), rec.get("pos2"), rec.get("pos3")
    if pos1 is None or pos2 is None or pos3 is None:
        return None
    try:
        sys = rec.get("sys", "").upper()
        ctr = int(float(rec.get("ctr", 399)))

        pos_km = np.array([float(pos1), float(pos2), float(pos3)])
        if sys == "ICRF_AU":
            # already AU despite the variable name
            pos_au = pos_km
        elif sys == "ICRF_KM":
            pos_au = pos_km / constants.AU_KM
        elif sys == "WGS84":
            # pos1=lon, pos2=lat, pos3=altitude (degrees, degrees, km).
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


def fetch_mpc_observations(desig: str, update_cache: bool = False):
    """
    Fetch observations from the MPC API and convert to fitting Observations.

    Queries ``https://data.minorplanetcenter.net/api/get-obs`` for the
    given object designation and returns optical observations ready for
    orbit fitting.  Only optical records with valid RA/Dec are included;
    radar and other types are silently skipped.

    Results are cached under ``~/.kete/observations/`` so that repeated
    queries for the same designation do not hit the network.

    Per-observation uncertainties are taken from the ADES ``rmsRA`` /
    ``rmsDec`` fields (1-sigma astrometric uncertainty in arcseconds,
    with ``rmsRA`` already including the ``cos(dec)`` factor) when
    available.  Otherwise, a default of 1.0 arcsecond is used.  The
    ``precRA`` / ``precDec`` fields are coordinate publication precision,
    not uncertainty, and are intentionally not used.

    Parameters
    ----------
    desig :
        Object designation recognized by the MPC (e.g. ``"Apophis"``,
        ``"101955"``, ``"1999 RQ36"``).
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

        observations = kete.mpc.fetch_mpc_observations("Apophis")
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
        # The API returns a list with one element per designation; take the
        # first (and only) entry which is itself a list of observation dicts.
        records = records[0]
        with gzip.open(cached_path, "wb") as f:
            f.write(json.dumps(records).encode())

    if not records:
        return []

    observations = []
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

        # Per-observation sigma: prefer the ADES rmsRA/rmsDec fields when
        # present (the reported astrometric 1-sigma uncertainty, with
        # rmsRA already including the cos(dec) factor).  Fall back to the
        # Veres et al. 2017 weighting tables (per-station, with mode and
        # catalog dependence) when rmsRA/rmsDec are absent.  Do NOT use
        # precRA/precDec -- those are the publication precision (number
        # of decimal places retained when rounding the position) and are
        # typically far smaller than the actual measurement uncertainty.
        stn = rec.get("stn", None)
        if stn is None:
            continue
        veres = _veres_2017_sigma(stn, rec.get("mode"), rec.get("astcat"), jd)
        s_ra = _parse_sigma(rec.get("rmsra"), veres)
        s_dec = _parse_sigma(rec.get("rmsdec"), veres)

        observer = _build_observer(stn, jd, rec)
        if observer is None:
            continue

        band = rec.get("band") or "V"
        try:
            mag = float(rec.get("mag"))
        except (TypeError, ValueError):
            mag = float("nan")

        observations.append(
            Observation.optical(
                observer=observer,
                ra=ra_deg,
                dec=dec_deg,
                sigma_ra=s_ra,
                sigma_dec=s_dec,
                band=band,
                mag=mag,
            )
        )

    return observations
