"""
PTF Observatory, the predecessor to ZTF which operated from 2009 to 2018, however
the last 3 years of data are not public.
"""

import logging
import os
from collections import defaultdict
from functools import lru_cache

from astropy.io import fits

from . import spice
from .cache import download_file
from .fov import FOVList, PtfCcd, PtfField
from .mpc import find_obs_code
from .tap import query_tap
from .time import Time
from .vector import State, Vector

__all__ = ["fetch_fovs", "fetch_frame"]


logger = logging.getLogger(__name__)

PTF_TABLE = "ptf.ptf_procimg"


@lru_cache(maxsize=2)
def fetch_fovs(year: int):
    """
    Load all FOVs taken during the specified mission year of PTF.

    PTF data is currently available from March 2009 to Feb 2015.

    This will download and cache all FOV information for the given year from IRSA.

    This can take up to about 20 minutes per year of survey.

    Parameters
    ----------
    year :
        Which year of PTF.
    """
    year = int(year)
    if year < 2009 or year > 2016:
        raise ValueError("PTF Data available from 2009 to 2015.")

    cols = [
        "fieldid",
        "obsjd",
        "ccdid",
        "filter",
        "pfilename",
        "infobits",
        "seeing",
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
        f"SELECT {', '.join(cols)} FROM {PTF_TABLE} "
        f"WHERE obsjd between {jd_start} and {jd_end}",
        verbose=True,
    )

    # Exposures are 30 seconds
    jds = [Time(x, scaling="utc") for x in irsa_query["obsjd"]]
    obs_info = find_obs_code("ZTF")

    # PTF fields are made up of up to 11 individual CCDs, here we first construct
    # the individual CCD information.
    fovs = []
    for jd, row in zip(jds, irsa_query.itertuples()):
        corners = []
        for i in range(4):
            ra = getattr(row, f"ra{i + 1}")
            dec = getattr(row, f"dec{i + 1}")
            corners.append(Vector.from_ra_dec(ra, dec))
        observer = spice.earth_pos_to_ecliptic(jd, *obs_info[:-1])
        observer = State("PTF", observer.jd, observer.pos, observer.vel)

        try:
            fov = PtfCcd(
                corners,
                observer,
                row.fieldid,
                row.ccdid,
                row.filter,
                row.pfilename,
                row.infobits,
                row.seeing,
            )
        except Exception:
            print(
                corners,
                observer,
                row.fieldid,
                row.ccdid,
                row.filter,
                row.pfilename,
                row.infobits,
                row.seeing,
            )
        fovs.append(fov)

    # Now group the quad information into full 64 size Fields
    grouped = defaultdict(list)
    for fov in fovs:
        grouped[fov.field].append(fov)

    # Sort the fovs by ccdid and make PTF Fields
    final_fovs = []
    for value in grouped.values():
        value = sorted(value, key=lambda x: (x.ccdid))
        fov = PtfField(value)
        final_fovs.append(fov)

    # return the result
    fov_list = FOVList(final_fovs)
    return fov_list


def fetch_frame(
    fov: PtfCcd,
    force_download=False,
    retry=2,
):
    """
    Given a PTF FOV, return the FITs file associated with it.

    This downloads the fits file into the cache.

    Parameters
    ----------
    fov :
        A single CCD FOV.
    force_download :
        Optionally force a re-download if the file already exists in the cache.
    """
    ptf_base = (
        f"https://irsa.ipac.caltech.edu/ibe/data/ptf/images/level1/{fov.filename}"
    )
    file = download_file(
        ptf_base, force_download=force_download, auto_zip=True, subfolder="ptf_frames"
    )

    try:
        return fits.open(file)[0]
    except OSError as exc:
        if retry == 0:
            raise ValueError("Failed to fetch PTF frame.") from exc
        logger.info("PTF file appears corrupted, attempting to fetch again.")
        os.remove(file)
        return fetch_frame(fov, force_download, retry - 1)
