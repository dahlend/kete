# SPDX-FileCopyrightText: 2026 Dar Dahlen
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest

import kete  # noqa: F401  (used by eval of repr)
from kete import State, propagate_n_body
from kete.orbit_fitting import Observation, fit_orbit
from kete.propagation import NonGravModel

JD = 2460000.5
SIGMA = 0.1  # arcsec
A2 = 1e-8


def _truth():
    return State("p", JD, [1.5, 0.3, 0.1], [-0.005, 0.012, 0.001])


def _observations(model):
    """Noise-free RA/Dec of the truth under model, from Earth, with light time."""
    out = []
    for jd in JD + np.arange(15) * 20.0:
        earth = kete.spice.get_state(
            "earth", jd, center=0, frame=kete.Frames.Equatorial
        )
        t = jd
        for _ in range(3):
            obj = propagate_n_body([_truth()], t, non_gravs=[model])[0]
            obj = obj.change_center(0).as_equatorial
            dist = np.linalg.norm(np.array(obj.pos) - np.array(earth.pos))
            t = jd - dist * kete.constants.AU_KM / 299792.458 / 86400
        d = np.array(obj.pos) - np.array(earth.pos)
        d /= np.linalg.norm(d)
        ra = np.degrees(np.arctan2(d[1], d[0])) % 360
        dec = np.degrees(np.arcsin(d[2]))
        out.append(Observation.optical(earth, ra, dec, SIGMA, SIGMA, time_sigma=0.0))
    return out


class TestWithFree:
    def test_free_parameters_and_repr(self):
        model = NonGravModel.new_comet(1e-8, 2e-9, 0.0)
        assert model.free_parameters == []
        freed = model.with_free("a1", "a3")
        assert freed.free_parameters == ["a1", "a3"]
        assert freed.items == model.items
        assert model.with_free().free_parameters == ["a1", "a2", "a3"]
        again = eval(repr(freed))
        assert again.free_parameters == freed.free_parameters
        assert again.items == freed.items
        nan_free = NonGravModel.new_comet(a2=0.0, a3=0.0)
        assert nan_free.free_parameters == ["a1"]
        assert "with_free" not in repr(nan_free)

    def test_rejects_unknown_name(self):
        with pytest.raises(ValueError, match="fittable"):
            NonGravModel.new_comet(0.0, 0.0, 0.0).with_free("beta")

    def test_propagation_uses_the_values(self):
        model = NonGravModel.new_comet(0.0, A2, 0.0)
        got = propagate_n_body([_truth()], JD + 30, non_gravs=[model.with_free()])[0]
        want = propagate_n_body([_truth()], JD + 30, non_gravs=[model])[0]
        assert np.allclose(got.pos, want.pos, atol=0, rtol=0)


class TestFitWarmStart:
    def test_warm_start_from_the_truth(self):
        """Started at the true state and non-grav value, the fit reaches the solution a
        cold start finds, near the truth."""
        truth = NonGravModel.new_comet(0.0, A2, 0.0)
        obs = _observations(truth)
        free = NonGravModel.new_comet(a1=0.0, a3=0.0)
        cold = fit_orbit(_truth(), obs, non_grav=free, max_reject_passes=0)
        warm = fit_orbit(
            _truth(), obs, non_grav=truth.with_free("a2"), max_reject_passes=0
        )
        assert cold.converged and warm.converged
        assert abs(warm.non_grav.items["a2"] - A2) < 0.02 * A2
        assert abs(warm.non_grav.items["a2"] - cold.non_grav.items["a2"]) < 1e-3 * A2

    def test_continue_an_earlier_fit(self):
        """A fit continued from its own result stays there."""
        obs = _observations(NonGravModel.new_comet(0.0, A2, 0.0))
        free = NonGravModel.new_comet(a1=0.0, a3=0.0)
        cold = fit_orbit(_truth(), obs, non_grav=free, max_reject_passes=0)
        warm = fit_orbit(
            cold.state,
            obs,
            non_grav=cold.non_grav.with_free("a2"),
            max_reject_passes=0,
        )
        assert warm.converged
        assert abs(warm.non_grav.items["a2"] - cold.non_grav.items["a2"]) < 1e-4 * A2
        assert abs(warm.rms - cold.rms) < 1e-6
