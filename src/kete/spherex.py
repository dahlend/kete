"""
Spherex Related Functions and Data.
"""

from collections import defaultdict

import pandas as pd

from ._core import SpherexCmos, SpherexField
from .spice import get_state
from .tap import query_tap
from .time import Time
from .vector import Vector

__all__ = ["fetch_fovs", "fetch_observation_table", "SpherexCmos", "SpherexField"]


def fetch_fovs(update_cache=False):
    """Download every Spherex Field of View from IRSA."""
    table = fetch_observation_table(update_cache=update_cache)
    fields = defaultdict(list)

    for row in table.itertuples():
        region = _parse_s_region(row.s_region)
        if (row.obs_id, row.obsid) in fields:
            observer = fields[(row.obs_id, row.obsid)][0].observer
        else:
            time = (row.time_bounds_lower + row.time_bounds_lower) / 2
            jd = Time.from_mjd(time, scaling="UTC").jd
            observer = get_state("Earth", jd)
        cmos = SpherexCmos(region, observer, row.uri, row.planeid)
        fields[(row.obs_id, row.obsid)].append(cmos)
    fields = dict(fields)

    full_fields = []
    for (obs_id, observerid), frames in fields.items():
        full_fields.append(SpherexField(frames, obs_id, observerid))


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
