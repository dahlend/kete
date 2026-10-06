# SPDX-FileCopyrightText: 2026 Dar Dahlen
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest

import kete  # noqa: F401  (used by eval of repr)
from kete import State, propagate_n_body
from kete.propagation import NonGravModel

JD = 2460000.5


def _state():
    return State("p", JD, [1.5, 0.3, 0.1], [-0.005, 0.012, 0.001])


class TestRampedThrust:
    def test_construction_and_items(self):
        model = NonGravModel.new_ramped_thrust(JD)
        items = model.items
        assert items["t0"] == JD
        for key in ("a1", "a2", "a3", "rate"):
            assert np.isnan(items[key])
        fixed = NonGravModel.new_ramped_thrust(JD, 1e-8, 0.0, -2e-9, 0.1)
        assert eval(repr(fixed)).items == fixed.items

    def test_rejects_bad_epoch(self):
        with pytest.raises(ValueError):
            NonGravModel.new_ramped_thrust(float("nan"))

    def test_zero_rate_matches_comet_model(self):
        """With rate 0 the propagation matches new_comet with g(r) = 1."""
        a = (3e-8, -1e-8, 5e-9)
        ramped = NonGravModel.new_ramped_thrust(JD, *a, rate=0.0)
        comet = NonGravModel.new_comet(*a, alpha=1.0, r_0=1.0, m=0.0, n=0.0, k=0.0)
        got = propagate_n_body([_state()], JD + 20, non_gravs=[ramped])[0]
        want = propagate_n_body([_state()], JD + 20, non_gravs=[comet])[0]
        assert np.allclose(got.pos, want.pos, atol=1e-13, rtol=0)

    def test_thrust_off_before_the_ramp_starts(self):
        """rate 0.1 with t0 at JD + 20 switches the thrust on at JD + 10; up to then
        the path matches gravity alone."""
        start = _state()
        on = NonGravModel.new_ramped_thrust(JD + 20, 3e-8, 0.0, 0.0, rate=0.1)
        got = propagate_n_body([start], JD + 10, non_gravs=[on])[0]
        free = propagate_n_body([start], JD + 10)[0]
        assert np.allclose(got.pos, free.pos, atol=1e-14, rtol=0)

    def test_turning_part_items_and_repr(self):
        model = NonGravModel.new_ramped_thrust(JD, period=0.5)
        items = model.items
        assert items["period"] == 0.5
        for key in ("a1", "a2", "a3", "rate", "b1", "b2", "b3", "c1", "c2", "c3"):
            assert np.isnan(items[key])
        fixed = NonGravModel.new_ramped_thrust(
            JD,
            1e-8,
            0.0,
            -2e-9,
            0.1,
            period=0.5,
            b1=1e-9,
            b2=0.0,
            b3=0.0,
            c1=0.0,
            c2=2e-9,
            c3=0.0,
        )
        assert eval(repr(fixed)).items == fixed.items
        assert "period" not in NonGravModel.new_ramped_thrust(JD).items

    def test_turning_part_needs_a_period(self):
        with pytest.raises(ValueError, match="period"):
            NonGravModel.new_ramped_thrust(JD, b1=1e-9)
        for bad in (0.0, -1.0, float("nan")):
            with pytest.raises(ValueError):
                NonGravModel.new_ramped_thrust(JD, period=bad)

    def test_zero_turning_part_matches_steady_thrust(self):
        a = (3e-8, -1e-8, 5e-9)
        steady = NonGravModel.new_ramped_thrust(JD, *a, rate=0.05)
        turning = NonGravModel.new_ramped_thrust(
            JD,
            *a,
            rate=0.05,
            period=0.5,
            b1=0.0,
            b2=0.0,
            b3=0.0,
            c1=0.0,
            c2=0.0,
            c3=0.0,
        )
        got = propagate_n_body([_state()], JD + 5, non_gravs=[turning])[0]
        want = propagate_n_body([_state()], JD + 5, non_gravs=[steady])[0]
        assert np.allclose(got.pos, want.pos, atol=1e-14, rtol=0)

    def test_turning_part_averages_out_over_whole_periods(self):
        """A fast turning part moves the body by ~b / omega^2, far less than a
        steady thrust of the same size over the same time."""
        b = 3e-8
        turning = NonGravModel.new_ramped_thrust(
            JD,
            0.0,
            0.0,
            0.0,
            0.0,
            period=0.1,
            b1=b,
            b2=0.0,
            b3=0.0,
            c1=0.0,
            c2=0.0,
            c3=0.0,
        )
        steady = NonGravModel.new_ramped_thrust(JD, b, 0.0, 0.0, 0.0)
        free = propagate_n_body([_state()], JD + 5)[0]
        moved = propagate_n_body([_state()], JD + 5, non_gravs=[turning])[0]
        pushed = propagate_n_body([_state()], JD + 5, non_gravs=[steady])[0]
        d_turning = np.linalg.norm(np.array(moved.pos) - np.array(free.pos))
        d_steady = np.linalg.norm(np.array(pushed.pos) - np.array(free.pos))
        assert d_turning < 0.02 * d_steady
