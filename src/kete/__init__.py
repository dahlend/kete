import importlib
import logging

from . import (
    analysis,
    cache,
    constants,
    covariance,
    fitting,
    flux,
    fov,
    mpc,
    neos,
    plot,
    population,
    ptf,
    shape,
    spherex,
    spice,
    state_transition,
    tap,
    wise,
    ztf,
)
from ._core import Data
from .analysis import (
    BPlane,
    compute_b_plane,
    specific_energy,
)
from .conversion import (
    compute_albedo,
    compute_aphelion,
    compute_diameter,
    compute_h_mag,
    compute_semi_major,
    flux_to_mag,
    hill_radius,
    mag_to_flux,
    sphere_of_influence,
)
from .fov import (
    ConeFOV,
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
    fov_spice_check,
    fov_state_check,
    fov_static_check,
)
from .horizons import HorizonsProperties
from .propagation import (
    closest_approach,
    moid,
    propagate_n_body,
    propagate_two_body,
)
from .time import Time
from .vector import (
    CometElements,
    Frames,
    SimultaneousStates,
    State,
    Vector,
)

__all__ = [
    "analysis",
    "BPlane",
    "cache",
    "closest_approach",
    "CometElements",
    "compute_albedo",
    "compute_aphelion",
    "compute_b_plane",
    "compute_diameter",
    "compute_h_mag",
    "compute_semi_major",
    "ConeFOV",
    "constants",
    "covariance",
    "Data",
    "fitting",
    "flux",
    "flux_to_mag",
    "fov",
    "fov_spice_check",
    "fov_state_check",
    "fov_static_check",
    "Frames",
    "hill_radius",
    "HorizonsProperties",

    "mag_to_flux",
    "moid",
    "mpc",
    "neos",
    "NeosCmos",
    "NeosVisit",
    "OmniDirectionalFOV",
    "plot",
    "population",
    "propagate_n_body",
    "propagate_two_body",
    "ptf",
    "PtfCcd",
    "PtfField",
    "RectangleFOV",
    "set_logging",
    "shape",
    "SimultaneousStates",
    "specific_energy",
    "sphere_of_influence",
    "spherex",
    "SpherexCmos",
    "SpherexField",
    "spice",
    "State",
    "state_transition",
    "tap",
    "Time",
    "Vector",
    "wise",
    "WiseCmos",
    "ZtfCcdQuad",
    "ZtfField",
    "ztf",
]


def set_logging(level=logging.INFO, fmt="%(asctime)s - %(message)s"):
    """
    Output logging information to the console.

    Parameters
    ----------
    level:
        The logging level to output, if this is set to 0 logging is disabled.
    fmt:
        Format of the logging messages, see the ``logging`` package for format string
        details. Here is a more verbose output example:
        "%(asctime)s %(name)s:%(lineno)s - %(message)s"
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(level)

    # If there is already a handler in the logger, dont add another
    logger.handlers.clear()

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(logging.Formatter(fmt))
    logger.addHandler(ch)
    return logger


set_logging()
__version__ = importlib.metadata.version(__name__)
