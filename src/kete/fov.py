"""
Field of view definitions, along with tests for checks to see if objects are within the
FOVs.
"""

from . import spice
from ._core import (
    ConeFOV,
    FOVList,
    NeosCmos,
    NeosVisit,
    OmniDirectionalFOV,
    PtfCcd,
    PtfField,
    RectangleFOV,
    SpherexCmos,
    SpherexField,
    WiseCmos,
    ZtfCcdQuad,
    ZtfField,
    fov_state_check,
    fov_static_check,
)
from ._core import (
    fov_spk_check as _fov_spk_check,
)
from .vector import State, Vector

__all__ = [
    "NeosCmos",
    "NeosVisit",
    "WiseCmos",
    "ZtfCcdQuad",
    "ZtfField",
    "PtfCcd",
    "PtfField",
    "RectangleFOV",
    "ConeFOV",
    "OmniDirectionalFOV",
    "FOVList",
    "SpherexCmos",
    "SpherexField",
    "fov_static_check",
    "fov_state_check",
    "fov_spice_check",
]


def fov_spice_check(desigs: list[str], fovs) -> list[State]:
    """
    Given a list of object names and field of views, return only the objects which are
    visible to the observer, adding a correction for optical light delay.

    Objects are queried from the loaded SPK files. This does a best effort lookup and
    may silently not return states if an object is not loaded or doesn't have data for
    the specified epochs.

    parameters
    ----------
    desigs:
        Designations to lookup.
    fov:
        A list of field of views from which to subselect objects which are visible.
    """
    obj_ids = [spice.name_lookup(n)[1] for n in desigs]
    return _fov_spk_check(obj_ids, fovs)


def _from_wcs(wcs, obs: State) -> RectangleFOV:
    """
    Construct a RectangleFOV from an astropy WCS along with an observer state.

    parameters
    ----------
    wcs:
        An astropy WCS, this must include the shape of the array.
    obs:
        The observer position
    """
    types = wcs.axis_type_names
    if types == ["RA", "DEC"]:
        x, y = wcs.array_shape
        corners = wcs.pixel_to_world_values([0, 0, x, x], [0, y, y, 0])
        vecs = [Vector.from_ra_dec(ra, dec) for ra, dec in zip(*corners)]
    else:
        raise NotImplementedError(
            f"Support for WCS with frame {types} is not currently supported."
        )
    return RectangleFOV.from_corners(vecs, obs)


# Monkey patch the rust class with a new constructor.
RectangleFOV.from_wcs = staticmethod(_from_wcs)
