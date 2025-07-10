"""
Spherex Related Functions and Data.
"""

from collections import defaultdict

import numpy as np
import pandas as pd
from astropy.io import fits
from scipy.interpolate import RegularGridInterpolator

from .cache import download_file
from .fov import FOVList, SpherexCmos, SpherexField
from .spice import get_state
from .tap import query_tap
from .time import Time
from .vector import Vector

__all__ = [
    "fetch_fovs",
    "fetch_observation_table",
    "fetch_spectral_image",
    "SpherexCmos",
    "SpherexField",
]


def fetch_frame(fov, index=1):
    """
    Download the level 2 frame from IRSA.

    SPHEREx level 2 frames are made up of 7 sections:
    0 - Metadata.
    1 - Actual Image (MJy/sr)
    2 - Pixel bitwise flags.
    3 - Per pixel variance (MJy/sr)^2.
    4 - Model estimated zodiacal light (MJy/sr).
    5 - Over-sampled PSF, 101x101 pixels
    6 - Binary Table containing the WCS-WAVE Spectral coordinate system.

    By default this function will return the image itself.

    """
    url = f"https://irsa.ipac.caltech.edu/{fov.uri}"
    frame = download_file(url, auto_zip=True, subfolder="spherex_frames")
    return fits.open(frame)[index]


def fetch_spectral_image(fov):
    """
    Construct an two simulated images, where the values of each pixel corresponds
    to the central wavelength and the bandwidth for the linear variable filter.

    This will return 2 matrices, first is the central wavelength, the second is
    the bandwidth.
    """
    frame = fetch_frame(fov, index=6)
    x = frame.data[0][0]
    y = frame.data[0][1]
    fq = frame.data[0][2]

    interp = RegularGridInterpolator([x, y], fq)

    # +1 is because spice is 1 indexed
    freqs, width = interp(
        np.indices(frame[1].data.shape).transpose(1, 2, 0) + 1
    ).transpose(2, 0, 1)

    return freqs, width


def fetch_fovs(update_cache=False):
    """
    Download every public Spherex Field of View from IRSA.

    Currently the position of Spherex is not publicly available, so
    kete uses a custom SPICE kernel which has been reconstructed from publicly
    available data. This SPICE kernel is NOT official, and matches JPL Horizons
    values to within about 30km.
    """
    table = fetch_observation_table(update_cache=update_cache)
    table = table[[s is not None for s in table["s_region"]]]
    fields = defaultdict(list)

    for row in table.itertuples():
        region = _parse_s_region(row.s_region)
        if (row.obs_id, row.obsid) in fields:
            observer = fields[(row.obs_id, row.obsid)][0].observer
        else:
            time = (row.time_bounds_lower + row.time_bounds_lower) / 2
            jd = Time.from_mjd(time, scaling="UTC").jd
            observer = get_state("spherex", jd)
        cmos = SpherexCmos(region, observer, row.uri, row.planeid)
        fields[(row.obs_id, row.obsid)].append(cmos)
    fields = dict(fields)

    full_fields = []
    for (obs_id, observerid), frames in fields.items():
        full_fields.append(SpherexField(frames, obs_id, observerid))
    full_fields = FOVList(full_fields)
    full_fields.sort()
    return full_fields


def fetch_observation_table(update_cache=False):
    """
    Fetch the information required to define the SPHEREx raw images and their
    location on the sky.

    This data is merged together into a single table which is easy to use.

    This does not include observer location, as those fields in the IRSA dataset are
    NaN.
    """
    # Download all rows, but a subset of columns of all 4 IRSA tables for SPHEREx
    # This could be done with clever SQL tricks, but in practice, since we are
    # downloading the entire table, it is easier and faster to just do it locally.
    plane_columns = """obsid, planeid, time_bounds_lower, time_bounds_upper,
        energy_bounds_lower, energy_bounds_upper, energy_bandpassname"""

    plane_table = query_tap(
        f"""select {plane_columns} from spherex.plane""", update_cache=update_cache
    )

    obscore_table = query_tap(
        "select obs_id, s_region, energy_bandpassname from spherex.obscore",
        update_cache=update_cache,
    )

    # observationid in this table is obs_id in obscore, note: obsid is different
    observation_table = query_tap(
        "select obsid, observationid as obs_id from spherex.observation",
        update_cache=update_cache,
    )

    artifact_table = query_tap(
        """select planeid, uri from spherex.artifact""",
        update_cache=update_cache,
    )

    observation = pd.merge(obscore_table, observation_table, on="obs_id", how="outer")
    planes = pd.merge(
        observation, plane_table, on=["obsid", "energy_bandpassname"], how="outer"
    )
    return pd.merge(planes, artifact_table, on="planeid", how="outer")


def _parse_s_region(s_region):
    parts = s_region.split()
    if parts[0] != "POLYGON":
        raise ValueError("Can only parse 'POLYGON' sky regions.")
    if parts[1] != "ICRS":
        raise ValueError("Can only parse 'ICRS' frames.")
    values = [float(x) for x in parts[2:]]
    ras = values[::2]
    decs = values[1::2]
    if len(ras) != 5:
        raise ValueError("Can only handle rectangular regions of the sky")
    vecs = []
    for ra, dec in zip(ras, decs):
        vecs.append(Vector.from_ra_dec(ra, dec))
    return vecs[:-1]
