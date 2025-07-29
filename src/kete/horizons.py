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
from warnings import warn

import numpy as np
import pandas as pd
import requests

from ._core import CometElements, Covariance, HorizonsProperties, NonGravModel
from .cache import cache_path
from .covariance import generate_sample_from_cov
from .mpc import pack_designation, unpack_designation
from .time import Time

__all__ = [
    "HorizonsProperties",
    "fetch_spice_kernel",
    "fetch_known_orbit_data",
    "fetch",
]


_PARAM_MAP = {
    "a1": "a1",
    "a2": "a2",
    "a3": "a3",
    "aln": "alpha",
    "nm": "m",
    "r0": "r_0",
    "nk": "k",
    "nn": "n",
    "dt": "dt",
    "e": "eccentricity",
    "q": "peri_dist",
    "tp": "peri_time",
    "node": "lon_of_ascending",
    "peri": "peri_arg",
    "i": "inclination",
}


def fetch(name, update_name=True, cache=True, update_cache=False, exact_name=False):
    """
    Fetch object properties from JPL Horizons service.

    This does a best effort to fill in the values from horizons, but some values
    may be missing. The keyword arguments in this function allow to fill in
    defaults when the specified values cannot be found from horizons.


    Parameters
    ----------
    name :
        Name of the object to fetch.
    update_name :
        If this is true, replace name with the name that horizons uses for the
        object.
    cache :
        If this is true, the results are cached, and no query to horizons takes
        place if the object is already in the local cache.
    update_cache :
        If this is true, the current cache contents are ignored, and new values
        are saved after querying horizons.
    exact_name :
        If this is true, it is assumed that an exact designation in the format of
        horizons has been provided. This is particularly useful for comets which
        have fragments, as these objects are difficult to query since there are
        several names which have matches on substrings.
    """
    props = _fetch_json(
        name,
        update_cache=update_cache,
        update_name=update_name,
        cache=cache,
        exact_name=exact_name,
    )
    if update_name:
        horizons_name = props.get("object", {})
        if "des" in horizons_name:
            name = horizons_name["des"]
        elif "fullname" in horizons_name:
            name = horizons_name["fullname"].split("(")[-1].replace(")", "")
    phys = {
        "desig": name,
        "inclination": None,
        "eccentricity": None,
        "lon_of_ascending": None,
        "h_mag": None,
        "g_phase": None,
        "diameter": None,
        "vis_albedo": None,
        "moid": None,
        "peri_dist": None,
        "peri_arg": None,
        "peri_time": None,
        "arc_len": None,
        "epoch": None,
        "covariance": None,
    }
    if "orbit" in props:
        lookup = {
            "eccentricity": "e",
            "inclination": "i",
            "lon_of_ascending": "node",
            "peri_arg": "peri",
            "peri_dist": "q",
            "peri_time": "tp",
        }
        lookup_rev = dict(x[::-1] for x in lookup.items())

        for kete_v, jpl_v in lookup.items():
            val = [
                float(p["value"])
                for p in props["orbit"]["elements"]
                if p["label"] == jpl_v
            ]
            if len(val) == 0:
                continue
            phys[kete_v] = val[0]

        phys["epoch"] = float(props["orbit"]["epoch"])
        if "moid" in props["orbit"] and props["orbit"]["moid"] is not None:
            phys["moid"] = float(props["orbit"]["moid"])
        if "data_arc" in props["orbit"] and props["orbit"]["data_arc"] is not None:
            phys["arc_len"] = float(props["orbit"]["data_arc"])
        if props["orbit"]["covariance"] is not None:
            cov_epoch = float(props["orbit"]["covariance"]["epoch"])
            mat = np.nan_to_num(
                np.array(props["orbit"]["covariance"]["data"], dtype=float)
            )
            labels = props["orbit"]["covariance"]["labels"]
            if "elements" in props["orbit"]["covariance"]:
                elements = {
                    x["label"]: float(x["value"])
                    for x in props["orbit"]["covariance"]["elements"]
                }
            else:
                elements = {
                    lab: phys[lookup_rev[lab]] for lab in labels if lab in lookup_rev
                }
            if "model_pars" in props["orbit"]:
                for param in props["orbit"]["model_pars"]:
                    elements[param["name"]] = float(param["value"])
            params = [
                (lookup_rev.get(x, x.lower()), elements.get(x, np.nan)) for x in labels
            ]
            phys["covariance"] = Covariance(name, cov_epoch, params, mat)
    else:
        raise ValueError(
            f"Horizons did not return orbit information for this object:\n{props}"
        )

    if "phys_par" in props:
        lookup = {
            "h_mag": "H",
            "g_phase": "G",
            "diameter": "diameter",
            "vis_albedo": "albedo",
        }
        for kete_v, jpl_v in lookup.items():
            val = [p["value"] for p in props["phys_par"] if p["name"] == jpl_v]
            if len(val) == 0:
                continue
            phys[kete_v] = float(val[0])

    phys["group"] = props.get("object", {}).get("orbit_class", {}).get("name", None)
    return HorizonsProperties(**phys)


def _fetch_json(
    name, update_name=True, cache=True, update_cache=False, exact_name=False
):
    """
    Fetch object properties from JPL Horizons service.

    This does a best effort to fill in the values from horizons, but some values
    may be missing. The keyword arguments in this function allow to fill in
    defaults when the specified values cannot be found from horizons.


    Parameters
    ----------
    name :
        Name of the object to fetch.
    update_name :
        If this is true, replace name with the name that horizons uses for the
        object.
    cache :
        If this is true, the results are cached, and no query to horizons takes
        place if the object is already in the local cache.
    update_cache :
        If this is true, the current cache contents are ignored, and new values
        are saved after querying horizons.
    exact_name :
        If this is true, it is assumed that an exact designation in the format of
        horizons has been provided. This is particularly useful for comets which
        have fragments, as these objects are difficult to query since there are
        several names which have matches on substrings.
    """

    dir_path = os.path.join(cache_path(), "horizons_props")
    try:
        filename = os.path.join(dir_path, f"{pack_designation(name)}.json")
    except ValueError:
        filename = os.path.join(dir_path, f"{name.replace('/', '_')}.json")

    if not os.path.isdir(dir_path) and cache:
        os.makedirs(dir_path)

    if os.path.isfile(filename) and cache and not update_cache:
        try:
            with open(filename, encoding="utf8") as f:
                return json.load(f)
        except Exception:  # pylint: disable=broad-except
            pass

    if isinstance(name, str):
        with contextlib.suppress(ValueError):
            name = unpack_designation(name)
    else:
        name = str(name)
    query = "des" if exact_name else "sstr"
    response = requests.get(
        f"https://ssd-api.jpl.nasa.gov/sbdb.api?{query}={name}"
        "&phys-par=true&full-prec=true&cov=mat"
        "&alt-des=true&alt-spk=true&alt-orbits=true&discovery=true",
        timeout=30,
    )

    if response.status_code == 300:
        names = [
            des["pdes"] for des in response.json()["list"] if "-" not in des["pdes"]
        ]
        if len(names) == 1:
            return _fetch_json(
                names[0], update_name, cache, update_cache, exact_name=True
            )

    response.raise_for_status()

    response_json = response.json()

    if cache:
        with open(filename, "w", encoding="utf8") as f:
            json.dump(response_json, f)
    return response_json


@property  # type: ignore
def _json(self) -> dict:
    """
    JSON Representation of the object as queried from horizons.

    This will by default come directly from the cached results, but will query if the
    object is not found.
    """
    return _fetch_json(self.desig, update_cache=False)


def _default_nongrav_params() -> dict:
    """default parameters used by jpl horizons for their non-grav models."""
    # This exists so that we dont have to worry about a global variable being
    # updated by accident.
    params = {
        "a1": 0.0,
        "a2": 0.0,
        "a3": 0.0,
        "alpha": 0.1112620426,
        "r_0": 2.808,
        "m": 2.15,
        "n": 5.093,
        "k": 4.6142,
        "dt": 0.0,
    }
    return params


@property  # type: ignore
def _nongrav(self) -> NonGravModel:
    """
    The non-gravitational forces model from the values returned from horizons.
    """
    orbit = self.json["orbit"]
    if "model_pars" not in orbit:
        return None
    model_pars_json = orbit["model_pars"]

    params = _default_nongrav_params()
    for vals in model_pars_json:
        param_name = vals["name"].lower()
        if param_name not in _PARAM_MAP:
            warn(f"Unknown non-grav values: {vals}", stacklevel=2)
            continue
        params[_PARAM_MAP[param_name]] = float(vals["value"])

    return NonGravModel.new_comet(**params)


def _sample(self, n_samples):
    """
    Sample the covariance matrix for this object, returning `n_samples` of states and
    non-gravitational models, which may be used to propagate the orbit in time.

    This uses the full covariance matrix in order to correctly sample the object's orbit
    with all correlations, including correlations with the non-gravitational forces.

    Parameters
    ----------
    n_samples :
        The number of samples to take of the covariance.
    """
    if self.covariance is None:
        raise ValueError(
            "This object does not have a covariance matrix, cannot sample from it."
        )
    matrix = self.covariance.cov_matrix
    epoch = Time(self.covariance.epoch, scaling="utc").jd
    samples = generate_sample_from_cov(n_samples, matrix)

    elem_keywords = [
        "eccentricity",
        "inclination",
        "lon_of_ascending",
        "peri_arg",
        "peri_dist",
        "peri_time",
    ]

    orbit = self.json["orbit"]
    has_nongrav = "model_pars" in orbit
    has_warned = False

    states = []
    non_gravs = []
    for sample in samples:
        names, vals = zip(*self.covariance.params)
        sample_params = dict(zip(names, np.array(vals) + sample))
        elem_params = {x: sample_params.pop(x) for x in elem_keywords}
        state = CometElements(self.desig, epoch, **elem_params).state
        if has_nongrav:
            params = _default_nongrav_params()
            for key, value in sample_params.items():
                if key in _PARAM_MAP:
                    params[_PARAM_MAP[key]] = value
                elif not has_warned:
                    warn(
                        f"Unknown non-grav parameter {key} in sample, "
                        "this may cause issues with the non-grav model.",
                        stacklevel=2,
                    )
                    has_warned = True
            non_grav = NonGravModel.new_comet(**params)
        else:
            non_grav = None
        states.append(state)
        non_gravs.append(non_grav)

    return states, non_gravs


@property  # type: ignore
def _desigs(self) -> list[str]:
    """
    List of alternate designations for this object.

    First designation in this list is the Horizons preferred designation.
    """
    obj_dict = self.json["object"]
    desigs = [obj_dict["des"]]
    for alt in obj_dict.get("des_alt", []):
        # dont trust horizons to always provide lower case keys
        alt = {k.lower(): v for k, v in alt.items()}
        if "des" in alt:
            # some objects are prepended with the full designation plus a /
            # for example: 10P/1878 O1
            # This line strips that '10P/' from the designation.
            des = alt["des"].replace(desigs[0] + "/", "")
            desigs.append(des)
    return desigs


HorizonsProperties.fetch = fetch
HorizonsProperties.json = _json
HorizonsProperties.non_grav = _nongrav
HorizonsProperties.sample = _sample
HorizonsProperties.desigs = _desigs


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
                "e,i,q,w,tp,om,A1,A2,A3,DT,M1,M2,K1,K2,PC,rot_per,H_sigma"
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
        ["desig", "name", "spkid", "orbit_id", "spec_T", "spec_B"]
    )
    table[others] = table[others].apply(pd.to_numeric, errors="coerce")
    return table
