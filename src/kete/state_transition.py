import numpy as np
from numpy.typing import NDArray

from . import _core
from .vector import State


def compute_stm(
    state: State,
    jd_end: float,
    include_asteroids: bool = False,
    non_grav=None,
) -> tuple[State, NDArray]:
    """
    Compute the state transition and parameter sensitivity matrix using the Radau
    15th-order integrator with full N-body physics.

    Returns the propagated state and a 6x(6+N) sensitivity matrix where N is the
    number of free non-gravitational parameters (0, 1, or 3 depending on the model).

    When no non-gravitational model is provided, the result is a standard 6x6 STM.
    When a ``NonGravModel`` is provided, additional columns give the partial
    derivatives of the final state with respect to the non-grav parameters:
        - ``NonGravModel.new_comet``: 3 extra columns for A1, A2, A3.
        - ``NonGravModel.new_dust``: 1 extra column for beta.

    Parameters
    ----------
    state:
        State of a single object.
    jd_end:
        Julian time (TDB) of the desired final state.
    include_asteroids:
        If True, include perturbations from selected massive asteroids.
    non_grav:
        Optional non-gravitational force model (``NonGravModel``).

    Returns
    -------
    tuple[State, np.ndarray]
        A tuple of (final_state, sensitivity_matrix).
    """
    final_state, mat = _core.compute_stm(state, jd_end, include_asteroids, non_grav)
    return final_state, np.array(mat)


def propagate_covariance(state: State, covariance: NDArray, jd_end: float) -> NDArray:
    """
    Given a 6x6 covariance matrix which represents uncertainty in [X, Y, Z, Vx, Vy, Vz],
    compute the covariance matrix at a future time defined by `jd_end`.

    Uses the Radau 15th-order integrator with full N-body physics. Units are AU for
    position and AU/day for velocity, matching the state convention throughout kete.

    Parameters
    ----------
    state:
        State of a single object.
    covariance:
        A 6x6 covariance matrix. Position components in AU^2, velocity components in
        (AU/day)^2, and cross terms in AU * AU/day.
    jd_end:
        Julian time (TDB) of the desired final state.

    Returns
    -------
    np.ndarray
        The propagated 6x6 covariance matrix in the same units.
    """
    _, stm = compute_stm(state, jd_end)
    return stm @ covariance @ stm.T
