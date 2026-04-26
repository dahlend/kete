from __future__ import annotations

import logging
from functools import lru_cache

import pandas as pd

from ._core import (
    _find_obs_code,
    pack_designation,
    unpack_designation,
)
from .cache import download_json
from .time import Time

__all__ = [
    "unpack_designation",
    "pack_designation",
    "fetch_known_designations",
    "fetch_known_orbit_data",
    "fetch_known_comet_orbit_data",
    "find_obs_code",
]

logger = logging.getLogger(__name__)


@lru_cache
def find_obs_code(site):
    return _find_obs_code(site)


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
    known_ids = download_json(
        "https://minorplanetcenter.net/Extended_Files/mpc_ids.json.gz",
        force_download,
    )

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
