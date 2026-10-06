# SPDX-FileCopyrightText: 2026 Dar Dahlen
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest

from kete import State, propagate_n_body
from kete.state_transition import compute_stm, propagate_covariance


def _finite_difference_stm(state, jd_end):
    """Central difference STM of propagate_n_body, in the frame of `state`."""
    steps = [1e-7] * 3 + [1e-9] * 3
    stm = np.zeros((6, 6))
    for idx, step in enumerate(steps):
        ends = []
        for sign in (1.0, -1.0):
            pos = np.array(state.pos, dtype=float)
            vel = np.array(state.vel, dtype=float)
            if idx < 3:
                pos[idx] += sign * step
            else:
                vel[idx - 3] += sign * step
            perturbed = State("test", state.jd, pos, vel, frame=state.frame)
            moved = propagate_n_body(perturbed, jd_end)
            ends.append(np.concatenate([moved.pos, moved.vel]))
        stm[:, idx] = (ends[0] - ends[1]) / (2 * step)
    return stm


@pytest.mark.parametrize("frame", ["ecliptic", "equatorial"])
def test_stm_is_in_the_state_frame(frame):
    state = State("test", 2460000.5, [2.0, 0.5, 0.1], [-0.003, 0.011, 0.001])
    if frame == "equatorial":
        state = state.as_equatorial
    jd_end = state.jd + 60

    _, stm = compute_stm(state, jd_end)
    expected = _finite_difference_stm(state, jd_end)
    assert np.allclose(stm, expected, rtol=1e-4, atol=1e-6)

    cov = np.diag([1e-8] * 3 + [1e-12] * 3)
    assert np.allclose(propagate_covariance(state, cov, jd_end), stm @ cov @ stm.T)
