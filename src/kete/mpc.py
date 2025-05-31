from __future__ import annotations
import logging
from functools import lru_cache
from dataclasses import dataclass
import pandas as pd
import numpy as np


from . import conversion, constants, deprecation
from .time import Time
from .vector import Vector, Frames
from .cache import download_json
from .conversion import table_to_states

from . import _core


__all__ = [
    "unpack_designation",
    "pack_designation",
    "fetch_known_packed_designations",
    "fetch_known_designations",
    "fetch_known_orbit_data",
    "fetch_known_comet_orbit_data",
    "find_obs_code",
    "unpack_permanent_designation",
    "pack_permanent_designation",
    "unpack_provisional_designation",
    "pack_provisional_designation",
    "unpack_satellite_designation",
    "pack_satellite_designation",
    "unpack_comet_designation",
    "pack_comet_designation",
    "normalize_names",
]

table_to_states = deprecation.rename(
    table_to_states,
    "1.2.0",
    old_name="table_to_states",
    additional_msg="Use `kete.conversion.table_to_states` instead.",
)

_mpc_hex = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

logger = logging.getLogger(__name__)


@lru_cache
def find_obs_code(name: str):
    """
    Search known observatory codes, if a single matching observatory is found, this will
    return the [lat, lon, altitude, description, obs code] in degrees and km as
    appropriate.

    >>> kete.mpc.find_obs_code("Palomar Mountain")
    (33.35411714208074, -116.86253999999998, 1.6960628760407417, 'Palomar Mountain', '675')

    Parameters
    ----------
    name :
        Name of the observatory, this can be a partial name, or obs code.
    """
    codes = _core.observatory_codes()
    found = []
    name_lower = name.lower().strip()
    for obs in codes:
        if name_lower in obs[3].lower() or name_lower in obs[4].lower():
            # If an exact match, return early.
            if name == obs[3]:
                return obs
            found.append(obs)
    if len(found) == 1:
        return found[0]
    elif len(found) == 0:
        raise ValueError("Failed to find matching observatory code.")
    else:
        found_str = "\n".join([f"{x[3]} - {x[4]}" for x in found])
        raise ValueError(f"Found multiple codes: \n{found_str}")


def num_to_base62(n: int, base_len=6) -> str:
    """
    Convert an integer to a base 62 MPC string of the specified length.

    >>> kete.mpc.num_to_base62(63, 3)
    '011'

    """
    base = len(_mpc_hex)
    buf = []
    while n > 0:
        buf.append(_mpc_hex[n % base])
        n //= base
    return "".join(reversed(buf)).zfill(base_len)


def base62_to_num(n: str) -> int:
    """
    Convert a base 62 MPC string into an integer.

    >>> kete.mpc.base62_to_num('011')
    63

    """
    val = 0
    for i, c in enumerate(n[::-1]):
        val += _mpc_hex.index(c) * 62**i
    return val


def unpack_mpc_dates(packed: str) -> float:
    """
    Convert a packed MPC date into JD time.

    See https://www.minorplanetcenter.net/iau/info/PackedDates.html for more details.

    Parameters
    ----------
    packed:
        The packed MPC date.
    """
    year = _mpc_hex.index(packed[0]) * 100 + int(packed[1:3])
    month = _mpc_hex.index(packed[3])
    day = float(_mpc_hex.index(packed[4]))
    day += float("." + packed[5:]) if len(packed) > 5 else 0.0
    return Time.from_ymd(year, month, day).jd


def unpack_permanent_designation(designation):
    """
    Convert an MPC 5 character permanent designation into an unpacked representation.

    >>> kete.mpc.unpack_permanent_designation("~AZaz")
    '3140113'
    >>> kete.mpc.unpack_permanent_designation("C3456")
    '123456'
    >>> kete.mpc.unpack_permanent_designation("0021P")
    '21P'
    >>> kete.mpc.unpack_permanent_designation("J005S")
    'Jupiter V'

    Parameters
    ----------
    designation : str
        A 5 character permanent designation to convert to unpack.
    """
    if len(designation) != 5:
        raise ValueError(f"({designation}) is not a permanent designation.")
    if designation[0] == "~":
        # Numbered object at or above 620000
        dig1 = _mpc_hex.index(designation[1]) * (62**3)
        dig2 = _mpc_hex.index(designation[2]) * (62**2)
        dig3 = _mpc_hex.index(designation[3]) * 62
        dig4 = _mpc_hex.index(designation[4])
        return str(620000 + dig1 + dig2 + dig3 + dig4)
    elif designation[-1] == "S":
        # planetary satellites
        return unpack_satellite_designation(designation)
    elif designation[-1] in "APDXI":
        # numbered comets and interstellar objects
        return unpack_comet_designation(designation)
    return str(_mpc_hex.index(designation[0]) * 10000 + int(designation[1:]))


def pack_permanent_designation(unpacked):
    """
    Convert an MPC asteroid number into an MPC 5 character packed designation.

    >>> kete.mpc.pack_permanent_designation('3140113')
    '~AZaz'
    >>> kete.mpc.pack_permanent_designation('21P')
    '0021P'
    >>> kete.mpc.pack_permanent_designation('Saturn V')
    'S005S'

    Parameters
    ----------
    unpacked : str
        A string to convert to an MPC permanent designation.
    """

    unpacked = str(unpacked).strip()

    if unpacked.isnumeric():
        num = int(unpacked)

        if num < 620_000:
            idx = int(num / 10_000)
            rem = str(num % 10_000).zfill(4)
            return _mpc_hex[idx] + rem

        num -= 620_000
        if num >= 62**4:
            raise ValueError("No designation possible")
        return "~" + num_to_base62(num, base_len=4)

    if " " in unpacked:
        if unpacked.split()[0] in [
            "Earth",
            "Mars",
            "Jupiter",
            "Saturn",
            "Uranus",
            "Neptune",
        ]:
            # satellite names like Jupiter V
            return pack_satellite_designation(unpacked)
        else:
            raise ValueError("This is not a permanent designation")

    return pack_comet_designation(unpacked)


def unpack_provisional_designation(packed: str):
    """
    Accepts a packed MPC provisional designation and returns an unpacked provisional
    designation.

    See https://www.minorplanetcenter.net/iau/info/PackedDes.html for more details.

    >>> kete.mpc.unpack_provisional_designation("J98SA8Q")
    '1998 SQ108'

    Parameters
    ----------
    packed :
        A packed 7 character provisional MPC designation of an object.
    """
    if len(packed) == 12:
        packed = packed[6:]
    if len(packed) == 8:
        return unpack_comet_designation(packed)
    if len(packed) != 7:
        raise ValueError("Packed designation is not correctly formatted.")
    if packed[:3] in ["PLS", "T1S", "T2S", "T3S"]:
        return packed[3:] + " " + packed[0] + "-" + packed[1]
    year = str(_mpc_hex.index(packed[0]) * 100 + int(packed[1:3]))
    if int(year) < 1925:
        year = "A" + year[1:]
    loop = _mpc_hex.index(packed[4]) * 10 + int(packed[5])
    loop_str = "" if loop == 0 else str(loop)
    order = packed[6]
    if order.isnumeric() or order.islower():
        # it's a comet
        return unpack_comet_designation(packed)
    return year + " " + packed[3] + order + loop_str


def pack_provisional_designation(unpacked: str):
    """
    Accepts an unpacked MPC provisional designation and returns a packed provisional
    designation.

    See https://www.minorplanetcenter.net/iau/info/PackedDes.html for more details.

    >>> kete.mpc.pack_provisional_designation("1998 SQ108")
    'J98SA8Q'

    Parameters
    ----------
    unpacked :
        An unpacked provisional MPC designation of an object.
    """
    year, designation = unpacked.split()
    if designation[:3] in ["P-L", "T-1", "T-2", "T-3"]:
        return designation[0] + designation[2] + "S" + year

    order = designation[1]
    if order.isnumeric() or "/" in unpacked:
        # its a comet
        return pack_comet_designation(unpacked)
    else:
        num = 0 if len(designation) == 2 else int(designation[2:])
    loop = _mpc_hex[int(num / 10)]
    subloop = str(int(num % 10))
    year_lookup = {"18": "I", "19": "J", "20": "K", "A9": "J", "A8": "I"}
    century = year_lookup[year[:2]]
    decade = year[2:]
    half_month = designation[0]
    return century + decade + half_month + loop + subloop + order


def pack_satellite_designation(unpacked: str):
    """
    Accepts an unpacked MPC planetary satellite designation and returns a packed
    designation.

    See https://www.minorplanetcenter.net/iau/info/PackedDes.html for more details.

    >>> kete.mpc.pack_satellite_designation("Jupiter XIII")
    'J013S'

    Parameters
    ----------
    unpacked :
        An unpacked satellite MPC designation of an object.
    """
    planet, satnum = unpacked.split()
    pout = planet[0]

    if pout not in "EMJSUN":
        raise ValueError("planet name not known")

    digout = roman_to_int(satnum)

    return f"{pout:1s}{digout:03d}S"


def unpack_satellite_designation(packed: str):
    """
    Accepts a packed MPC satellite designation and returns an unpacked
    designation.

    See https://www.minorplanetcenter.net/iau/info/PackedDes.html for more details.

    >>> kete.mpc.unpack_satellite_designation("J013S")
    'Jupiter XIII'

    Parameters
    ----------
    packed :
        A packed 5 character satellite MPC designation of an object.
    """

    if packed[4] != "S":
        raise ValueError("this is not a packed satellite designation")

    planets = {
        "E": "Earth",
        "M": "Mars",
        "J": "Jupiter",
        "S": "Saturn",
        "U": "Uranus",
        "N": "Neptune",
    }

    if packed[0] not in planets:
        raise ValueError("Planet character not known")

    return planets[packed[0]] + " " + int_to_roman(int(packed[1:4]))


def pack_comet_designation(unpacked: str):
    """
    Accepts an unpacked MPC provisional designation and returns a packed provisional
    designation.

    See https://www.minorplanetcenter.net/iau/info/PackedDes.html for more details.

    >>> kete.mpc.pack_comet_designation("C/2020 F3")
    'CK20F030'

    >>> kete.mpc.pack_comet_designation("1P/Halley")
    '0001P'

    Parameters
    ----------
    unpacked :
        An unpacked MPC comet designation of an object.
    """

    unpacked = unpacked.strip()

    if "/" in unpacked and unpacked[0].isdigit():
        unpacked = unpacked.split("/")[0].upper()

    frag = None
    if "-" in unpacked:
        # fragments
        unpacked, frag = unpacked.split("-")
        frag = frag.lower()

    if " " not in unpacked:
        comet_type = unpacked[-1]
        comet_number = int(unpacked[:-1])
        packed = f"{comet_number:04d}{comet_type:1s}"
        if frag is None:
            return packed
        else:
            return packed + "{:>7s}".format(frag)

    else:
        if "/" in unpacked:
            comet_type = unpacked[0]
            unpacked = unpacked[2:]
        else:
            comet_type = None

        year, designation = unpacked.split()
        if designation[1] in "0123456789":
            # comet-like designation
            num = int(designation[1:])
            if num > 99:
                outnum = _mpc_hex[int(num / 10)] + str(num % 10)
            else:
                outnum = f"{num:02d}"
            packed = _mpc_hex[int(year[0:2])] + year[2:] + designation[0] + outnum
            if comet_type is not None:
                packed = comet_type + packed

            if frag is None:
                return packed + "0"
            else:
                return packed + frag
        else:
            # asteroid-like designation
            if comet_type is None:
                return pack_provisional_designation(unpacked)
            else:
                return comet_type + pack_provisional_designation(unpacked)


def unpack_comet_designation(packed: str):
    """
    Accepts a packed MPC comet designation and returns an unpacked provisional
    designation.

    See https://www.minorplanetcenter.net/iau/info/PackedDes.html for more details.

    >>> kete.mpc.unpack_comet_designation("CI70Q010")
    'C/1870 Q1'

    Parameters
    ----------
    packed :
        A packed 5,7, or 8 character provisional MPC designation of an object.
    """
    if len(packed) == 5:
        return str(int(packed[0:4])) + packed[4]

    comet_add = ""
    if len(packed) == 8 and packed[0].upper() in "APCDXI":
        comet_add = packed[0] + "/"
        packed = packed[1:]

    year = str(_mpc_hex.index(packed[0]) * 100 + int(packed[1:3]))
    comet_num = _mpc_hex.index(packed[4]) * 10 + int(packed[5])
    half_month = packed[3]
    frag = packed[6]
    if frag == "0":
        return comet_add + year + " " + half_month + str(comet_num)
    elif frag.islower():
        return comet_add + year + " " + half_month + str(comet_num) + "-" + frag.upper()
    else:
        return comet_add + unpack_provisional_designation(packed)


def unpack_designation(packed: str):
    """
    Accepts either a packed provisional designation or permanent designation and returns
    the unpacked representation.

    >>> kete.mpc.unpack_designation("J98SA8Q")
    '1998 SQ108'

    >>> kete.mpc.unpack_designation("~AZaz")
    '3140113'

    Parameters
    ----------
    packed :
        A packed 5, 7, or 8 character MPC designation of an object.
    """
    packed = packed.strip()
    if len(packed) == 5:
        return unpack_permanent_designation(packed)
    elif len(packed) == 7:
        return unpack_provisional_designation(packed)
    elif len(packed) == 8:
        return unpack_comet_designation(packed)

    raise SyntaxError(f"This designation could not be unpacked '{packed}'")


def pack_designation(unpacked: str):
    """
    Accepts either a unpacked provisional designation or permanent designation and
    returns the packed representation.

    >>> kete.mpc.pack_designation("1998 SQ108")
    'J98SA8Q'

    >>> kete.mpc.pack_designation("3140113")
    '~AZaz'

    Parameters
    ----------
    unpacked :
        An unpacked designation to be packed into either a permanent or provisional
        designation.
    """

    try:
        return pack_permanent_designation(unpacked)
    except ValueError:
        return pack_provisional_designation(unpacked)


@lru_cache()
def fetch_known_packed_designations(force_download=False):
    """
    Download the most recent copy of the MPCs known ID mappings in their packed format.

    This download only occurs the first time this function is called.

    This then returns a dictionary of all known packed IDs to a single ID which is the
    one that the MPC specifies as their default.

    For example, here are the first two objects which are returned:

    {'00001': '00001',
    'I01A00A': '00001',
    'I99O00F': '00001',
    'J43X00B': '00001',
    '00002': '00002',
    'I02F00A': '00002',
    ...}

    Ceres has 4 entries, which all map to '00001'.
    """
    # download the data from the MPC
    packed_ids = download_json(
        "https://minorplanetcenter.net/Extended_Files/mpc_ids_packed.json.gz",
        force_download,
    )

    # The data which is in the format {'#####"; ['#####', ...], ...}
    # where the keys of the dictionary are the MPC default name, and the values are the
    # other possible names.
    # Reshape the MPC dictionary to be flat, with every possible name mapping to the
    # MPC default name.
    packed_map = {}
    for name, others in packed_ids.items():
        packed_map[name] = name
        for other in others:
            packed_map[other] = name
    return packed_map


@lru_cache()
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


def normalize_names(dataset, col: str = "MPC_packed_name", name_lookup=None):
    """
    Given a Pandas Dataframe containing packed MPC names, alter the names to be the up
    to date MPC designation.


    Parameters
    ----------
    dataset :
        A pandas dataframe which contains a column of packed MPC names.
    col :
        The column of the dataset which contains the packed MPC names.
    name_lookup :
        Dictionary mapping old names to current names, if None is provided, this will
        use :py:func:`fetch_known_packed_designations`.
    """
    if name_lookup is None:
        name_lookup = fetch_known_packed_designations()
    dataset = dataset.copy()
    new_names = []
    for item in dataset[col]:
        item = item.strip()
        new_names.append(name_lookup.get(item, item))
    dataset[col] = new_names
    return dataset


@lru_cache()
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


@lru_cache()
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
        name = pack_comet_designation(comet.get("Designation_and_name").split("(")[0])
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


def mpc_known_orbit_filtered(filt):
    """
    Return all objects in the MPC database which pass the selected filter function.

    Parameters
    ----------
    filt:
        Filter function which defines which group to select. The filter function must
        accept 3 parameters, `peri_dist, eccentricity, h_mag`, and return a bool.
        See `kete.population` for a collection of filter functions which
        are used to generation model populations.
    """
    orbs = fetch_known_orbit_data()
    orbs.fillna(value=np.nan)
    return orbs[filt(orbs.peri_dist, orbs.ecc, orbs.h_mag)]


def int_to_roman(num: int):
    """Convert an integer to a Roman numeral."""
    if not isinstance(num, int):
        raise TypeError("Input needs to be an integer")
    if not 0 < num < 4000:
        raise ValueError("Argument must be between 1 and 3999")

    ints = (1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1)
    nums = ("M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV", "I")
    result = []
    for val, dig in zip(ints, nums):
        count = int(num / val)
        result.append(dig * count)
        num -= val * count
    return "".join(result)


def roman_to_int(num: str):
    """Convert a Roman numeral to an integer."""

    if not isinstance(num, str):
        raise TypeError("input must be a string")
    num = num.upper()

    if any(v not in "MDCLXVI" for v in num):
        raise ValueError("input is not a valid roman numeral")

    nums = {"M": 1000, "D": 500, "C": 100, "L": 50, "X": 10, "V": 5, "I": 1}

    # convert the string representation to a list of values
    values = [nums[v] for v in num]

    val = 0
    for digit, next_digit in zip(values, values[1:]):
        # If the next place holds a larger number, this value is negative
        if next_digit > digit:
            val -= digit
        else:
            val += digit

    # Add the last digit
    val += values[-1]

    # easiest test for validity...
    if int_to_roman(val) == num:
        return val
    else:
        raise ValueError("input is not a valid Roman numeral")


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

    name: str
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

        contents = dict(
            name=line[:12].strip(),
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
            raise ValueError("No second line of spacecraft observation found.")

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
