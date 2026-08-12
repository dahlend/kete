import numpy as np
import pytest

from kete import EquinoctialElements, State, UncertainState
from kete.vector import CometElements


class TestEquinoctialElements:
    def test_from_state_round_trips_to_state(self):
        st = State(
            "t", 2460000.5, pos=(2.0, 0.3, 0.1), vel=(-0.002, 0.011, 0.001)
        )
        el = EquinoctialElements.from_state(st)
        back = el.state
        assert np.allclose(np.array(back.pos), np.array(st.pos))
        assert np.allclose(np.array(back.vel), np.array(st.vel))

    def test_derived_quantities_agree_with_comet_elements(self):
        """Round-tripping a known CometElements orbit through a State and back
        into EquinoctialElements must reproduce the same classical elements -
        the two representations describe the same physical orbit."""
        comet = CometElements(
            epoch=123456,
            desig="test",
            eccentricity=0.35,
            inclination=12.0,
            peri_dist=1.7,
            peri_arg=80.0,
            lon_of_ascending=210.0,
            peri_time=123450.0,
        )
        eq = EquinoctialElements.from_state(comet.state)
        assert np.isclose(eq.eccentricity, comet.eccentricity)
        assert np.isclose(eq.inclination, comet.inclination)
        assert np.isclose(eq.lon_of_ascending, comet.lon_of_ascending)
        assert np.isclose(eq.peri_arg, comet.peri_arg)
        assert np.isclose(eq.peri_dist, comet.peri_dist)
        assert np.isclose(eq.semi_major, comet.semi_major)
        assert np.isclose(eq.aphelion, comet.aphelion)
        assert np.isclose(eq.orbital_period, comet.orbital_period)

    def test_direct_construction(self):
        el = EquinoctialElements(
            desig="manual",
            epoch=2460000.5,
            semi_latus=1.5,
            ecc_f=0.1,
            ecc_g=-0.05,
            pole_h=0.02,
            pole_k=0.01,
            true_lon=40.0,
        )
        assert el.semi_latus == 1.5
        assert el.ecc_f == 0.1
        # Degrees in, degrees out, through the core's radians.
        assert np.isclose(el.true_lon, 40.0)
        assert el.center_id == 10
        # a real orbit, not just stored floats
        assert np.isfinite(el.state.pos.x)

    def test_displaced_by_and_offset_to_are_inverses(self):
        el = EquinoctialElements.from_state(
            State("t", 2460000.5, pos=(1.6, -0.4, 0.2), vel=(0.004, 0.013, 0.001))
        )
        # The last entry is the true longitude, in degrees.
        delta = [1e-6, -2e-7, 3e-7, 0.0, 1e-8, 0.05]
        moved = el.displaced_by(delta)
        recovered = el.offset_to(moved)
        assert np.allclose(recovered, delta, rtol=1e-12, atol=1e-15)

    def test_offset_to_reduces_true_longitude_wrap(self):
        """Two orbits a whole turn apart in true longitude must read as
        coincident, not as separated by 2*pi."""
        el = EquinoctialElements.from_state(
            State("t", 2460000.5, pos=(1.6, -0.4, 0.2), vel=(0.004, 0.013, 0.001))
        )
        wrapped = EquinoctialElements(
            "t", 2460000.5, el.semi_latus, el.ecc_f, el.ecc_g,
            el.pole_h, el.pole_k, el.true_lon + 360.0,
        )
        delta = el.offset_to(wrapped)
        assert abs(delta[5]) < 1e-9

    def test_displaced_by_rejects_wrong_length(self):
        el = EquinoctialElements.from_state(
            State("t", 2460000.5, pos=(1.6, -0.4, 0.2), vel=(0.004, 0.013, 0.001))
        )
        with pytest.raises(ValueError, match="length 6"):
            el.displaced_by([1.0, 2.0])

    def test_uncertain_state_elements_matches_cov_matrix_basis(self):
        """UncertainState.elements is the mean the cov_matrix covariance is
        centered on - both must describe the same orbit."""
        st = State("t", 2460000.5, pos=(2.0, 0.3, 0.1), vel=(-0.002, 0.011, 0.001))
        us = UncertainState.from_state(st, pos_sigma=1e-8, vel_sigma=1e-6)
        el = us.elements
        back = el.state
        # the mean state must match UncertainState.state (same reconstruction)
        assert np.allclose(np.array(back.pos), np.array(us.state.pos))
        assert np.allclose(np.array(back.vel), np.array(us.state.vel))
