import numpy as np
import pytest

from kete.vector import CometElements, State


class TestlState:
    def test_init(self):
        pos = np.array([1, 2, 3])
        vel = np.array([7, 8, 9])
        vs = State("a", 1, pos, vel)
        assert vs.jd == 1
        assert np.allclose(np.array(vs.pos), pos)
        assert np.allclose(np.array(vs.vel), vel)

    def test_from_orbital_elements_specific(self):
        vs = CometElements(
            epoch=123456,
            desig="test",
            eccentricity=0.1,
            inclination=0,
            peri_dist=0.45,
            peri_arg=0,
            lon_of_ascending=0,
            peri_time=123456,
        ).state
        assert vs.jd == 123456
        elem = vs.elements
        assert np.isclose(elem.eccentricity, 0.1)
        assert np.isclose(elem.inclination, 0)
        assert np.isclose(elem.peri_arg, 0)
        assert np.isclose(elem.lon_of_ascending, 0)
        assert np.isclose(elem.peri_time, 123456)
        assert np.isclose(elem.peri_dist, 0.45)
        assert np.isclose(elem.semi_major, 0.5)

    @pytest.mark.parametrize("ecc", [0.1, 0.5, 1.0, 1.3])
    @pytest.mark.parametrize("inc", [0.1, 2])
    @pytest.mark.parametrize("peri_dist", [0.1, 1, 5])
    @pytest.mark.parametrize("peri_arg", [0.1, 3, 6])
    @pytest.mark.parametrize("lon", [0.1, 2.7, 5.7])
    @pytest.mark.parametrize("time", [-10, 0, 10])
    def test_from_orbital_elements(self, ecc, inc, peri_dist, peri_arg, lon, time):
        state = CometElements(
            epoch=123456,
            desig="test",
            eccentricity=ecc,
            inclination=inc,
            peri_dist=peri_dist,
            peri_arg=peri_arg,
            lon_of_ascending=lon,
            peri_time=123456 + time,
        ).state
        assert state.jd == 123456
        elements = state.elements
        assert np.isclose(elements.eccentricity, ecc)
        assert np.isclose(elements.inclination, inc)
        assert np.isclose(elements.peri_arg, peri_arg)
        assert np.isclose(elements.lon_of_ascending, lon)
        assert np.isclose(elements.peri_dist, peri_dist)
