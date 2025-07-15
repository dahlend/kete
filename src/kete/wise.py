from __future__ import annotations

import logging
import os
from collections import namedtuple
from functools import lru_cache

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import fits

from . import spice
from ._core import (
    w1_color_correction,
    w2_color_correction,
    w3_color_correction,
    w4_color_correction,
)
from .cache import cache_path, download_file
from .deprecation import rename
from .fov import FOVList, WiseCmos
from .plot import annotate_plot, plot_fits_image, zoom_plot
from .tap import IRSA_URL, query_tap
from .time import Time
from .vector import Vector

__all__ = [
    "fetch_frame",
    "plot_frames",
    "MISSION_PHASES",
    "mission_phase_from_jd",
    "mission_phase_from_scan",
    "MissionPhase",
    "w1_color_correction",
    "w2_color_correction",
    "w3_color_correction",
    "w4_color_correction",
]

logger = logging.getLogger(__name__)

# All constants below are indexed as follows W1 = [0], W2 = [1]. W3 = [2], W4 = [3]

_COLOR_CORR = np.array(
    [  # Tbb  K_W1     K_W2    K_W3     K_W4
        [100, 17.2062, 3.9096, 2.6588, 1.0032],
        [110, 9.3213, 3.1120, 2.1424, 0.9955],
        [120, 6.5668, 2.5973, 1.8071, 0.9905],
        [130, 5.0963, 2.2458, 1.5776, 0.9873],
        [140, 4.1630, 1.9949, 1.4139, 0.9853],
        [150, 3.5203, 1.8096, 1.2935, 0.9841],
        [160, 3.0554, 1.6687, 1.2027, 0.9834],
        [170, 2.7070, 1.5591, 1.1328, 0.9830],
        [180, 2.4387, 1.4722, 1.0782, 0.9830],
        [190, 2.2273, 1.4021, 1.0350, 0.9831],
        [200, 2.0577, 1.3448, 1.0006, 0.9833],
        [210, 1.9194, 1.2975, 0.9728, 0.9836],
        [220, 1.8051, 1.2579, 0.9503, 0.9839],
        [230, 1.7095, 1.2245, 0.9321, 0.9843],
        [240, 1.6286, 1.1961, 0.9172, 0.9847],
        [250, 1.5596, 1.1717, 0.9051, 0.9851],
        [260, 1.5003, 1.1508, 0.8952, 0.9855],
        [270, 1.4489, 1.1326, 0.8872, 0.9860],
        [280, 1.4040, 1.1167, 0.8808, 0.9864],
        [290, 1.3647, 1.1029, 0.8755, 0.9868],
        [300, 1.3300, 1.0907, 0.8714, 0.9871],
        [310, 1.2993, 1.0799, 0.8682, 0.9875],
        [320, 1.2719, 1.0704, 0.8657, 0.9879],
        [330, 1.2474, 1.0619, 0.8639, 0.9882],
        [340, 1.2255, 1.0543, 0.8626, 0.9886],
        [350, 1.2057, 1.0476, 0.8617, 0.9889],
        [360, 1.1879, 1.0416, 0.8612, 0.9892],
        [370, 1.1718, 1.0361, 0.8611, 0.9895],
        [380, 1.1571, 1.0313, 0.8613, 0.9898],
        [390, 1.1438, 1.0269, 0.8616, 0.9901],
        [400, 1.1316, 1.0229, 0.8622, 0.9903],
    ]
)
"""
This table contains calibrated flux correction values for the 4 different WISE bands.

This is provided as reference, however kete has built in interpolation methods for
these black-body correction terms. See :func:`w1_color_correction` or similarly named
functions for other bands.
"""
# The data in the color correction table above is relatively well fit to a 1 / f(x)
# where f is a 4th order polynomial. It is the least accurate for the final band, but
# W4 is almost a constant.
# These fits perform much better than linear interpolation except near 100k.
#
# The values above were fit to polynomial equations, the results of which were
# hard-coded into the rust backend to allow for fast computation.


SUN_COLOR_CORRECTION: list[float] = [1.0049, 1.0193, 1.0024, 1.0012]
"""
Flux in the reflected light model should be scaled by these values.

This corrects for the Sun's spectral difference to the calibrator.
"""

ZERO_MAGS: list[float] = [306.681, 170.663, 29.0448, 8.2839]
"""
Non-color corrected values for zero mag corrections in Janskys.
"""

ZERO_MAGS_COLOR_CORRECTED: list[float] = [
    ZERO_MAGS[0],
    ZERO_MAGS[1],
    1.08 * ZERO_MAGS[2],
    0.96 * ZERO_MAGS[3],
]
"""Color Corrected Zero Mags in units of Jy."""

BAND_WAVELENGTHS: list[float] = [3352.6, 4602.8, 11560.8, 22088.3]
"""Non-color corrected values for the effective central wavelength of the bands (nm)"""

BAND_WAVELENGTHS_COLOR_CORRECTED: list[float] = [
    BAND_WAVELENGTHS[0],
    BAND_WAVELENGTHS[1],
    0.96 * BAND_WAVELENGTHS[2],
    1.025 * BAND_WAVELENGTHS[3],
]
"""Color Corrected Band Wavelengths in units of nm"""

FOV_WIDTH: float = 47 / 60
"""
Approximate width of a WISE chip FOV, this slightly over-estimates the true FOV.
This is 47 arc-minutes.
"""

DN_TO_JY = [1.9350e-06, 2.7048e-06, 2.9045e-06, 5.2269e-05]
"""
Convert directly from DN to Jy using this Jy/DN conversion factor.

These values came from:
https://wise2.ipac.caltech.edu/docs/release/prelim/expsup/sec2_3f.html
"""


MissionPhase = namedtuple(
    "MissionPhase",
    [
        "name",
        "jd_start",
        "jd_end",
        "bands",
        "frame_url",
        "frame_meta_table",
        "source_table",
    ],
)
MissionPhase.__doc__ = (
    "Information about a specific mission phase. The canonical set of these is stored"
    " in the :py:class:`MISSION_PHASES` constant."
)
MissionPhase.name.__doc__ = "Name of the mission phase."
MissionPhase.jd_start.__doc__ = "JD date of the start of the mission phase."
MissionPhase.jd_end.__doc__ = "JD date of the end of the mission phase."
MissionPhase.bands.__doc__ = "WISE wavelength bands available during the phase."
MissionPhase.frame_url.__doc__ = "URL of where the frames are stored on IRSA servers."
MissionPhase.frame_meta_table.__doc__ = (
    "SQL Table on IRSA where the metadata for the frames are stored."
)
MissionPhase.source_table.__doc__ = (
    "SQL Table on IRSA where source information is stored."
)

MISSION_PHASES = {
    "Cryo": MissionPhase(
        name="Cryo",
        jd_start=Time.from_ymd(2009, 12, 14).jd,
        jd_end=2455414.941783008,
        bands=(1, 2, 3, 4),
        frame_url=IRSA_URL + "/ibe/data/wise/allsky/4band_p1bm_frm/",
        frame_meta_table="allsky_4band_p1bs_frm",
        source_table="allsky_4band_p1bs_psd",
    ),
    "3-Band": MissionPhase(
        name="3-Band",
        jd_start=2455414.9417830084,
        jd_end=2455469.278276,
        bands=(1, 2, 3),
        frame_url=IRSA_URL + "/ibe/data/wise/cryo_3band/3band_p1bm_frm/",
        frame_meta_table="allsky_3band_p1bs_frm",
        source_table="allsky_3band_p1bs_psd",
    ),
    "Post-Cryo": MissionPhase(
        name="Post-Cryo",
        jd_start=2455469.278277,
        jd_end=2455593.96119803,
        bands=(1, 2),
        frame_url=IRSA_URL + "/ibe/data/wise/postcryo/2band_p1bm_frm/",
        frame_meta_table="allsky_2band_p1bs_frm",
        source_table="allsky_2band_p1bs_psd",
    ),
    "Reactivation_2014": MissionPhase(
        name="Reactivation_2014",
        jd_start=Time.from_ymd(2013, 12, 13).jd,
        jd_end=Time.from_ymd(2015, 1, 1).jd,
        bands=(1, 2),
        frame_url=IRSA_URL + "/ibe/data/wise/neowiser/p1bm_frm/",
        frame_meta_table="neowiser_p1bs_frm",
        source_table="neowiser_p1bs_psd",
    ),
    "Reactivation_2024": MissionPhase(
        name="Reactivation_2024",
        jd_start=Time.from_ymd(2024, 1, 1).jd,
        jd_end=Time.from_ymd(2024, 8, 1.291525).jd,
        bands=(1, 2),
        frame_url=IRSA_URL + "/ibe/data/wise/neowiser/p1bm_frm/",
        frame_meta_table="neowiser_p1bs_frm",
        source_table="neowiser_p1bs_psd",
    ),
}
"""Public released mission phases of WISE."""

for year in range(2015, 2024):
    MISSION_PHASES[f"Reactivation_{year}"] = MissionPhase(
        name=f"Reactivation_{year}",
        jd_start=Time.from_ymd(year, 1, 1).jd,
        jd_end=Time.from_ymd(year + 1, 1, 1).jd,
        bands=(1, 2),
        frame_url=IRSA_URL + "/ibe/data/wise/neowiser/p1bm_frm/",
        frame_meta_table="neowiser_p1bs_frm",
        source_table="neowiser_p1bs_psd",
    )
MISSION_PHASES = dict(sorted(MISSION_PHASES.items(), key=lambda x: x[1].jd_start))


def mission_phase_from_jd(jd: float) -> None | MissionPhase:
    """
    Return which mission phase is associated with the provided JD. Returns None if no
    matching mission phase.

    The WISE mission has different phases based on the cryogen on board and the
    hibernation period, resulting in different useable bands and different locations in
    the IRSA archive.

    Parameters
    ----------
    jd :
        The Julian Date in TDB scaled time, this is fully sufficient to determine the
        mission phase.
    """
    for mission in MISSION_PHASES.values():
        if mission.jd_start <= jd and jd < mission.jd_end:
            return mission
    return None


def mission_phase_from_scan(scan_id: str) -> None | MissionPhase:
    """
    Return the mission phase for WISE from the provided scan id.

    During reactivation, scan ids do not end on exact dates at the end of the year, and
    as a result this function may return the wrong reactivation year for scan IDs near
    the end of a year. However the Cryo/3-band/post-cryo phases will all be exact.

    The WISE mission has different phases based on the cryogen on board and the
    hibernation period, resulting in different useable bands and different locations in
    the IRSA archive.

    Parameters
    ----------
    scan_id :
        The scan id to be converted, this is sufficient to determine the mission phase.
    """

    scan_num = int(scan_id[0:5])
    letter = scan_id[-1]

    if letter == "j":
        return MISSION_PHASES["Cryo"]
    if letter == "d":
        return MISSION_PHASES["Post-Cryo"]
    if letter == "c":
        if scan_num <= 6217:
            return MISSION_PHASES["Cryo"]
        elif scan_num <= 8101:
            return MISSION_PHASES["3-Band"]
        return None
    elif letter == "a":
        if scan_num <= 7101:
            return MISSION_PHASES["Cryo"]
        elif scan_num <= 8744:
            return MISSION_PHASES["3-Band"]
        elif scan_num <= 12514:
            return MISSION_PHASES["Post-Cryo"]
        elif scan_num <= 55858:
            return MISSION_PHASES["Reactivation_2014"]
        elif scan_num <= 66982:
            return MISSION_PHASES["Reactivation_2015"]
        elif scan_num <= 78154:
            return MISSION_PHASES["Reactivation_2016"]
        elif scan_num <= 89305:
            return MISSION_PHASES["Reactivation_2017"]
        elif scan_num <= 99799:
            return MISSION_PHASES["Reactivation_2018"]
        return None
    elif letter == "b":
        if scan_num <= 7097:
            return MISSION_PHASES["Cryo"]
        elif scan_num <= 8741:
            return MISSION_PHASES["3-Band"]
        elif scan_num <= 12514:
            return MISSION_PHASES["Post-Cryo"]
        elif scan_num <= 55857:
            return MISSION_PHASES["Reactivation_2014"]
        elif scan_num <= 66981:
            return MISSION_PHASES["Reactivation_2015"]
        elif scan_num <= 78153:
            return MISSION_PHASES["Reactivation_2016"]
        elif scan_num <= 89301:
            return MISSION_PHASES["Reactivation_2017"]
        elif scan_num <= 99105:
            return MISSION_PHASES["Reactivation_2018"]
        return None

    # 2018 or later
    elif letter == "r":
        if scan_num <= 1660:
            return MISSION_PHASES["Reactivation_2018"]
        elif scan_num <= 12819:
            return MISSION_PHASES["Reactivation_2019"]
        elif scan_num <= 24012:
            return MISSION_PHASES["Reactivation_2020"]
        elif scan_num <= 35181:
            return MISSION_PHASES["Reactivation_2021"]
        elif scan_num <= 46370:
            return MISSION_PHASES["Reactivation_2022"]
        elif scan_num <= 57041 or scan_num <= 57626:
            return MISSION_PHASES["Reactivation_2023"]
        elif scan_num <= 64272:
            return MISSION_PHASES["Reactivation_2024"]
        return None
    elif letter == "s":
        if scan_num <= 1615:
            return MISSION_PHASES["Reactivation_2018"]
        elif scan_num <= 12037:
            return MISSION_PHASES["Reactivation_2019"]
        elif scan_num <= 23769:
            return MISSION_PHASES["Reactivation_2020"]
        elif scan_num <= 34831:
            return MISSION_PHASES["Reactivation_2021"]
        elif scan_num <= 46369:
            return MISSION_PHASES["Reactivation_2022"]
        elif scan_num <= 57519:
            return MISSION_PHASES["Reactivation_2023"]
        elif scan_num <= 64267:
            return MISSION_PHASES["Reactivation_2024"]
        return None
    return None


def _scan_frame(scan_id, frame_num=None, band=None):
    """
    Convert input from either a WiseCmos object or (scan id, frame num) into
    the scan id and frame number. This is a convenience function so that either of
    these objects can be passed as input.
    """
    if isinstance(scan_id, WiseCmos):
        if band is None and frame_num in [1, 2, 3, 4]:
            band = frame_num
        frame_num = scan_id.frame_num
        scan_id = scan_id.scan_id
    elif frame_num is None:
        raise ValueError(
            "Frame Number must be provided if the first arg is not a FOV object."
        )
    return scan_id, frame_num, band


def fetch_frame(
    scan_id, frame_num=None, band=None, as_fits=True, im_type="int", retry=2
):
    """
    Fetch the WISE FITs frame, if it is not present in the cache, download it first.

    This can return either an Astropy FITs file object, or the path to the downloaded
    file.

    Parameters
    ----------
    scan_id :
        The scan id of the desired frames, or a WISE FOV object, if this is
        provided, then frame number does not have to be provided.
    frame_num :
        The frame number of the desired frames.
    band :
        Band number of the target frame/scan combination.
    as_fits :
        Should the file path or an Astropy FITs object be returned.
    im_type :
        Which image type should be returned,
        `int` for intensity is typical, `mask` for the mask file.
    """
    scan_id, frame_num, band = _scan_frame(scan_id, frame_num, band)

    scan_id = str(scan_id)
    frame_num = int(frame_num)
    phase = mission_phase_from_scan(scan_id)

    # if no band is provided, assume 3 if possible, 2 otherwise
    if band is None:
        band = 3 if 3 in phase.bands else 2
    if band not in phase.bands:
        raise ValueError(f"Band {band} not present in this mission phase")

    # format the url
    frame_num = f"{frame_num:03d}"
    scan_group = str(scan_id[-2:])
    band_str = f"w{band:1d}"
    ext = "fits" if im_type == "int" else "fits.gz"
    filename = f"{scan_id}{frame_num}-{band_str}-{im_type}-1b.{ext}"
    url = f"{phase.frame_url}{scan_group}/{scan_id}/{frame_num}/{filename}"

    subfolder = os.path.join("wise_frames", scan_group)

    file_path = download_file(url, auto_zip=True, subfolder=subfolder)
    if as_fits:
        try:
            return fits.open(file_path)[0]
        except OSError as exc:
            if retry == 0:
                raise ValueError("Failed to fetch WISE frame.") from exc
            logger.info("WISE file appears corrupted, attempting to fetch again.")
            os.remove(file_path)
            return fetch_frame(scan_id, frame_num, band, as_fits, im_type, retry - 1)
    else:
        return file_path


def plot_frames(
    scan_id,
    frame_num=None,
    ra: None | float | list[float] = None,
    dec: None | float | list[float] = None,
    zoom: bool | float = True,
    bands: None | list[int] = None,
    cmap="gray_r",
    annotate=True,
):
    """
    Plot up to a 2x2 grid of images showing the W1, W2, W3, and W4 data for the
    (scan, frame) pair, centered on the ra, dec if provided.

    More than one RA/DEC pair may also be provided, but zooming will not work
    in that case.

    Parameters
    ----------
    scan_id :
        The scan id of the desired frames, or a WISE FOV object, if this is
        provided, then frame number does not have to be provided.
    frame_num :
        The frame number of the desired frames.
    ra :
        The RA position of where to zoom/center the frames.
    dec :
        The RA position of where to zoom/center the frames.
    zoom :
        If the image should be centered and zoomed in on the provided RA/DEC.
        This can also be a number, which will change the zoom level.
    bands :
        Bands of WISE to plot, if not provided this will plot all bands for the
        given mission phase.
    annotate :
        If ra/dec are provided, then the plot may be optionally annotated as well.
        This may be any style which is accepted by the annotate function.
    """
    scan_id, frame_num, band = _scan_frame(scan_id, frame_num)

    if zoom is True:
        zoom = 50

    if bands is None:
        phase = mission_phase_from_scan(scan_id)
        if phase is None:
            raise ValueError("Cannot identify the mission phase for this scan id")
        bands = phase.bands

    plt.figure(dpi=120, figsize=(8, 8), facecolor="w")
    plt.suptitle(f"Scan: {scan_id}  Frame: {frame_num}")
    plt.subplots_adjust(
        wspace=0.3, hspace=0.3, left=0.1, right=0.9, top=0.9, bottom=0.1
    )

    if len(bands) == 4:
        x_ind = [3, 4]
    elif len(bands) == 2:
        x_ind = [1, 2]
    else:
        x_ind = [2, 3]
    subplot_y = 1 if len(bands) <= 2 else 2

    for band in bands:
        frame = fetch_frame(scan_id, frame_num, band)

        plt.subplot(2, subplot_y, band)
        wcs = plot_fits_image(frame, percentiles=None, power_stretch=1, cmap=cmap)
        if band not in [1, 3]:
            plt.ylabel(" ")
            plt.tick_params(axis="y", labelbottom=False)
        if band not in x_ind:
            plt.xlabel(" ")
            plt.tick_params(axis="x", labelbottom=False)
        if ra is not None:
            actual_zoom = zoom / 2 if band == 4 else zoom
            zoom_plot(wcs, ra, dec, zoom=actual_zoom)
            if annotate:
                style = "+" if annotate is True else annotate
                annotate_plot(
                    wcs,
                    ra,
                    dec,
                    px_gap=actual_zoom / 4,
                    length=actual_zoom / 4,
                    lw=1.5,
                    style=style,
                )
        plt.title(f"W{band:1d}")


@lru_cache(maxsize=2)
def fetch_fovs(phase):
    """
    Load all FOVs taken during the specified mission phase of WISE.

    This will download and cache all FOV information for the given mission phase from
    IRSA.

    Parameters
    ----------
    phase :
        A mission phase object.
    """
    phase = MISSION_PHASES[phase]
    dir_path = os.path.join(cache_path(), "fovs")
    filename = os.path.join(dir_path, f"wise_{phase.name}_fovs.parquet")

    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
    if os.path.isfile(filename):
        res = pd.read_parquet(filename)
    else:
        table = phase.frame_meta_table
        cols = [
            "scan_id",
            "frame_num",
            "qual_frame",
            "mjd",
            "w1ra1",
            "w1ra2",
            "w1ra3",
            "w1ra4",
            "w1dec1",
            "w1dec2",
            "w1dec3",
            "w1dec4",
            "crota",
        ]

        if "Reactivation_" in phase.name:
            mjd_start = Time(phase.jd_start).mjd
            mjd_end = Time(phase.jd_end).mjd
            mjd_limits = f"WHERE mjd between {mjd_start} and {mjd_end}"
        else:
            mjd_limits = ""

        res = query_tap(
            f"SELECT {', '.join(cols)} FROM {table} {mjd_limits}",
            cache=False,
            verbose=True,
        )

        res.to_parquet(filename, index=False)

    jd = [Time.from_mjd(mjd, scaling="utc").jd for mjd in list(res.mjd)]
    res["jd"] = jd

    fovs = []
    for row in res.itertuples():
        state = spice.get_state("WISE", row.jd)

        # Each band has a slightly different size on sky.
        # Kete represents all bands simultaneously, so this information is not kept
        # track of. In order to deal with this, here we load the W1 ra/dec, and then
        # push the corners out by 1 arc-minute to act as a bit of a buffer region.
        # this 1 arc-minute offset is in the rust constructor for WISECmos objects.

        corners = []
        for i in range(4):
            corners.append(
                Vector.from_ra_dec(
                    getattr(row, f"w1ra{i + 1}"),
                    getattr(row, f"w1dec{i + 1}"),
                )
            )

        fov = WiseCmos(
            corners[::-1],
            state,
            row.frame_num,
            row.scan_id,
        )

        fovs.append(fov)
    fovs = FOVList(fovs)
    fovs.sort()
    return fovs


fetch_WISE_fovs = rename(
    fetch_fovs,
    "2.0.0",
    old_name="fetch_WISE_fovs",
)
