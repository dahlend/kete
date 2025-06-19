"""
ZTF Related Functions and Data.
ZTF data is available from 2018 to present.

The imaging sensor is made up of a 4x4 grid of CCDs, which are further subdivided into
2x2 "Quads", meaning each image that is captured is returned as 64 frames.
"""

import logging
import os
from collections import defaultdict
from functools import lru_cache

import pandas as pd
from astropy.io import fits

from . import spice
from .cache import cache_path, download_file
from .deprecation import rename
from .fov import FOVList, ZtfCcdQuad, ZtfField
from .mpc import find_obs_code
from .tap import query_tap
from .time import Time
from .vector import State, Vector

__all__ = ["fetch_fovs", "fetch_frame", "fetch_file"]

SURVEY_START_JD = Time.from_ymd(2018, 3, 20).jd
"""First image in ZTF dataset."""

logger = logging.getLogger(__name__)


@lru_cache(maxsize=2)
def fetch_fovs(year: int):
    """
    Load all FOVs taken during the specified mission year of ZTF.

    This will download and cache all FOV information for the given year from IRSA.

    This can take about 20 minutes per year of survey, each year is 2-3 GB of data.

    Parameters
    ----------
    year :
        Which year of ZTF, 2018 through 2024.
    """
    year = int(year)
    if year not in range(2018, 2025):
        raise ValueError("Year must only be in the range 2018-2024")
    cache_dir = cache_path()
    dir_path = os.path.join(cache_dir, "fovs")
    filename = os.path.join(dir_path, f"ztf_fields_{year}.parquet")

    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
    if os.path.isfile(filename):
        irsa_query = pd.read_parquet(filename)
    else:
        table = "ztf.ztf_current_meta_sci"
        cols = [
            "field",
            "filefracday",
            "ccdid",
            "filtercode",
            "imgtypecode",
            "qid",
            "obsdate",
            "maglimit",
            "fid",
            "ra",
            "dec",
            "ra1",
            "dec1",
            "ra2",
            "dec2",
            "ra3",
            "dec3",
            "ra4",
            "dec4",
        ]
        jd_start = Time.from_ymd(year, 1, 1).jd
        jd_end = Time.from_ymd(year + 1, 1, 1).jd

        irsa_query = query_tap(
            f"SELECT {', '.join(cols)} FROM {table} "
            f"WHERE obsjd between {jd_start} and {jd_end}",
            verbose=True,
            cache=False,
        )
        irsa_query.to_parquet(filename, index=False)

    offset = 15 / 24 / 60 / 60
    jds = [Time.from_iso(x + ":00").jd + offset for x in irsa_query["obsdate"]]
    obs_info = find_obs_code("ZTF")

    # ZTF fields are made up of up to 64 individual CCD quads, here we first construct
    # the individual CCD quad information.
    fovs = []
    for jd, row in zip(jds, irsa_query.itertuples()):
        corners = []
        for i in range(4):
            ra = getattr(row, f"ra{i + 1}")
            dec = getattr(row, f"dec{i + 1}")
            corners.append(Vector.from_ra_dec(ra, dec))
        observer = spice.earth_pos_to_ecliptic(jd, *obs_info[:-1])
        observer = State("ZTF", observer.jd, observer.pos, observer.vel)

        fov = ZtfCcdQuad(
            corners,
            observer,
            row.field,
            row.filefracday,
            row.ccdid,
            row.filtercode,
            row.imgtypecode,
            row.qid,
            row.maglimit,
            row.fid,
        )
        fovs.append(fov)

    # Now group the quad information into full 64 size Fields
    grouped = defaultdict(list)
    for fov in fovs:
        key = (fov.filefracday, fov.fid, fov.filtercode)
        grouped[key].append(fov)

    # Sort the quads by ccdid and qid and make ZTF Fields
    final_fovs = []
    for value in grouped.values():
        value = sorted(value, key=lambda x: (x.ccdid, x.qid))
        fov = ZtfField(value)
        final_fovs.append(fov)

    # finally save and return the result
    fov_list = FOVList(final_fovs)
    fov_list.sort()
    return fov_list


fetch_ztf_fovs = rename(
    fetch_fovs,
    "2.0.0",
    old_name="fetch_ztf_fovs",
)


def file_frac_day_split(filefracday):
    """
    Given the file frac day value from a field of view, return the year, month, and
    fraction of day.

    Note that the fraction of a day is in approximately JD UTC time, however there is
    about a 0.4 second offset in the ZTF time conversions for JD time.
    """
    filefracday = str(filefracday)
    year = int(filefracday[:4])
    month = int(filefracday[4:6])
    day = int(filefracday[6:8])
    frac_day = int(filefracday[8:])
    return (year, month, day, frac_day)


def fetch_frame(
    fov: ZtfCcdQuad,
    products="sci",
    im_type="sciimg.fits",
    force_download=False,
    retry=2,
):
    """
    Given a ztf FOV, return the FITs file associated with it.

    This downloads the fits file into the cache.

    Parameters
    ----------
    fov :
        A single CCD Quad FOV.
    products :
        Which data product to fetch.
    im_type :
        Image extension, this must match the products variable.
    force_download :
        Optionally force a re-download if the file already exists in the cache.
    """
    file = fetch_file(
        fov.field,
        fov.filefracday,
        fov.filtercode,
        fov.ccdid,
        fov.imgtypecode,
        fov.qid,
        products,
        im_type,
        force_download,
    )
    try:
        return fits.open(file)[0]
    except OSError as exc:
        if retry == 0:
            raise ValueError("Failed to fetch ZTF frame.") from exc
        logger.info("ZTF file appears corrupted, attempting to fetch again.")
        os.remove(file)
        return fetch_frame(fov, products, im_type, force_download, retry - 1)


def fetch_file(
    field,
    filefracday,
    filtercode,
    ccdid,
    imgtypecode,
    qid,
    products="sci",
    im_type="sciimg.fits",
    force_download=False,
):
    """
    Fetch a ZTF file directly from the IPAC server, returning the path to where it was
    saved.

    Parameters
    ----------
    field :
        Field identifier number, integer between 1 and ~2200.
    filefracday :
        String describing the fraction of a day, a record keeping string based on time.
    filtercode :
        Which filter was used for the exposure.
    ccdid :
        The CCD identified. (1-16)
    imgtypecode:
        Image type code.
    qid :
        Which quad of the ccd was used. (1-4)
    products :
        Exposure products, "sci" for science images.
    im_type :
        Image extension, this must match the products variable.
    force_download :
        Optionally force a re-download if the file already exists in the cache.

    """

    ztf_base = f"https://irsa.ipac.caltech.edu/ibe/data/ztf/products/{products}/"
    filefracday = str(filefracday)
    year = filefracday[:4]
    month_day = filefracday[4:8]
    frac_day = filefracday[8:]
    field = str(field).zfill(6)
    ccdid = str(ccdid).zfill(2)

    path = f"{year}/{month_day}/{frac_day}/"
    file = (
        f"ztf_{filefracday}_{field}_"
        f"{filtercode}_c{ccdid}_{imgtypecode}_q{qid}_{im_type}"
    )

    url = ztf_base + path + file

    return download_file(
        url, force_download=force_download, auto_zip=True, subfolder="ztf_frames"
    )
