# SPDX-FileCopyrightText: 2026 Dar Dahlen
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest

import kete
from kete.propagation import NonGravModel


@pytest.fixture
def observer():
    return kete.spice.get_state("Earth", 2461161.5)


@pytest.fixture
def ceres(observer):
    """Ceres, one day before the observation, well inside the default dt_limit."""
    return kete.spice.get_state("Ceres", observer.jd - 1.0)


def test_fov_state_check_non_gravs(observer, ceres):
    """
    A non-gravitational model must change the observed position, even when the
    state epoch is close enough to the observer that the two body check would
    otherwise be used.
    """
    fov = kete.OmniDirectionalFOV(observer)

    # beta = 0.5 removes half of the solar gravity from the object.
    dust = NonGravModel.new_dust(0.5)

    grav_only = kete.fov_state_check([ceres], [fov])[0]
    with_dust = kete.fov_state_check([ceres], [fov], non_gravs=[dust])[0]

    offset = np.linalg.norm(np.array(grav_only[0].pos) - np.array(with_dust[0].pos))
    assert offset > 1e-6


def test_fov_state_check_non_gravs_none(observer, ceres):
    """A list of `None` must match passing no models at all."""
    fov = kete.OmniDirectionalFOV(observer)

    expected = kete.fov_state_check([ceres], [fov])[0]
    calc = kete.fov_state_check([ceres], [fov], non_gravs=[None])[0]

    assert np.allclose(expected[0].pos, calc[0].pos)


def test_fov_state_check_non_gravs_length(observer, ceres):
    fov = kete.OmniDirectionalFOV(observer)
    with pytest.raises(ValueError, match="same length"):
        kete.fov_state_check([ceres, ceres], [fov], non_gravs=[None])


def test_fov_state_check_dt_limit_deprecated(observer, ceres):
    """Passing dt_limit warns and does not change the result."""
    fov = kete.OmniDirectionalFOV(observer)
    expected = kete.fov_state_check([ceres], [fov])[0]
    with pytest.warns(DeprecationWarning, match="dt_limit"):
        calc = kete.fov_state_check([ceres], [fov], dt_limit=0.1)[0]
    assert np.allclose(expected[0].pos, calc[0].pos)


def test_rectangle_at_pole_raises():
    """A rectangle pointed at a celestial pole has no defined rotation."""
    observer = kete.State("obs", 2460000.5, [1.0, 0.0, 0.0], [0.0, 0.0172, 0.0])
    pole = kete.Vector.from_ra_dec(0.0, 90.0)
    with pytest.raises(ValueError):
        kete.RectangleFOV(pole, 0.0, observer, 1.0, 1.0)
    near = kete.Vector.from_ra_dec(0.0, 89.9)
    fov = kete.RectangleFOV(near, 0.0, observer, 1.0, 1.0)
    assert np.isfinite(fov.pointing.ra)
