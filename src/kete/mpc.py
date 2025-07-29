from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd

from . import constants, conversion, deprecation
from ._core import find_obs_code, pack_designation, unpack_designation
from .cache import download_json
from .conversion import table_to_states
from .time import Time
from .vector import Frames, Vector

__all__ = [
    "unpack_designation",
    "pack_designation",
    "fetch_known_designations",
    "fetch_known_orbit_data",
    "fetch_known_comet_orbit_data",
    "find_obs_code",
]

table_to_states = deprecation.rename(
    table_to_states,
    "2.0.0",
    old_name="table_to_states",
    additional_msg="Use `kete.conversion.table_to_states` instead.",
)

logger = logging.getLogger(__name__)


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

    _UNSUPPORTED = set("WwQqVvRrXx")

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
            if line["note2"] == "s" or line["note2"] == "t":
                logger.warning("Second line of spacecraft observation found alone")
                continue
            elif line["note2"] == "S" or line["note2"] == "T":
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
