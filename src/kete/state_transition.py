import numpy as np
from numpy.typing import NDArray

from . import _core
from .vector import State


def compute_stm(state: State, jd_end: float) -> NDArray:
    """
    Compute the 6x6 State Transition Matrix (STM) for the provided state to the target
    JD.

    The STM is computed using 2-body mechanics.

    This matrix may be used to compute the final state at the specified JD by
    multiplying the vectorized version of the state against the matrix. Note that
    this is far less efficient than using the two-body propagation code.

    There are two main practical uses for the STM:
        - To propagate covariance matrices which represent uncertainty to a new epoch.
        - To turn the orbit determination problem into a least squares optimization
          problem.

    Parameters
    ----------
    state:
        State of a single object.
    jd_end:
        Julian time (TDB) of the desired final state.

    Returns
    -------
    np.ndarray
        Returns the 6x6 state transition matrix.
    """
    return np.array(_core.compute_stm(np.array(state).ravel(), state.jd, jd_end)[1])


def propagate_covariance(state: State, covariance: NDArray, jd_end: float) -> NDArray:
    """
    Given a 6x6 covariance matrix which represents uncertainty in [X, Y, Z, Vx, Vy, Vz],
    compute the covariance matrix at a future time defined by `jd_end`.

    Parameters
    ----------
    state:
        State of a single object.
    covariance:
        A 6x6 covariance matrix with units of AU for distance and AU/Day for time.
    jd_end:
        Julian time (TDB) of the desired final state.

    Returns
    -------
    np.ndarray
        Returns the new 6x6 covariance matrix.
    """
    stm = compute_stm(state, jd_end)
    return stm @ covariance @ stm.T
