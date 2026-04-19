"""
Interface functions and classes to JPL Horizons web services.
"""

import base64
import contextlib
import gzip
import json
import os
from functools import lru_cache
from typing import Optional, Union

import pandas as pd
import requests

from ._core import HorizonsProperties
from .cache import cache_path
from .mpc import unpack_designation
from .time import Time

__all__ = [
    "HorizonsProperties",
    "fetch_spice_kernel",
    "fetch_known_orbit_data",
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
