import numpy as np
import pytest

from kete import spice, State, Time, Frames
from kete.mpc import find_obs_code
from kete.spice import SpkInfo

# Turn off flake8 checker for this file.
# flake8: noqa

# fmt: off
# Here are expected interpolation equatorial state values as calculated by Horizon.
# [target, jd, pos.x, pos.y, pos.z,
#              vel.x, vel.y, vel.z]
EXPECTED_EQUITORIAL = [
    ["Earth" , 2505035.5,   -0.143102566296,    -0.922493350579,    -0.399580836275,
                             0.016758306839,    -0.002280675804,    -0.000989096637],
    ["Moon"  , 2505035.5,-0.145246016114196,-0.9209739949551414,-0.3990390962430936,
                          0.016436330755163,-0.0027213958479751,-0.0011310662345952],
    ["earth ba", 2485075.5, 0.890691383427617, 0.4087920914165625, 0.177113518664760,
                         -0.007976016046565, 0.0140578719519273, 0.0060915119230729],
    ["Mars"  , 2435040.5, 1.357638349280660,-0.2305928688109060,-0.1426082027542960,
                          0.003255868695805, 0.0135970078786732, 0.0061478371780576],
]
# fmt: on


class TestSpice:
    @pytest.mark.parametrize("expected", EXPECTED_EQUITORIAL)
    def test_ecliptic_state(self, expected):
        e_pos = expected[2:5]
        e_vel = expected[5:8]
        state = spice.get_state(expected[0], expected[1])
        assert isinstance(state, State)
        assert np.allclose(state.pos.as_equatorial, e_pos)
        assert np.allclose(state.vel.as_equatorial, e_vel)

        # Ensure that the same values are calculated if using an astropy time
        jd = Time(expected[1])
        state = spice.get_state(expected[0], jd)
        assert isinstance(state, State)
        assert np.allclose(state.pos.as_equatorial, e_pos)
        assert np.allclose(state.vel.as_equatorial, e_vel)

    def test_loaded_objects(self):
        assert len(spice.loaded_objects()) > 8

    def test_earth_to_ecliptic(self):
        state = spice.earth_pos_to_ecliptic(2454832.5, 0, 0, 0, center="399")
        assert np.allclose(
            list(-state.pos),
            [7.686364937857906e-06, -3.847842319488030e-05, 1.667609317250255e-05],
        )
        assert np.allclose(
            -state.vel,
            [2.642169198371849e-04, 4.433586059319487e-05, -1.948260054980404e-05],
        )

    def test_name_lookup(self):
        assert spice.name_lookup("jupiter b") == ("jupiter barycenter", 5)
        assert spice.name_lookup(0) == ("solar system barycenter", 0)

        with pytest.raises(ValueError, match="Multiple NAIF"):
            spice.name_lookup("ear")

        with pytest.raises(ValueError, match="No NAIF ID found for"):
            spice.name_lookup("banana")

    def test_loaded_info(self):
        info = spice.loaded_object_info("ceres")
        expected = SpkInfo(
            name="ceres",
            jd_start=Time(2415020.5),
            jd_end=Time(2488069.5),
            center=10,
            frame=1,
            spk_type=21,
        )
        assert info == [expected]

    def test_cache(self):
        spice.kernel_ls()

    def test_reload(self):
        spice.kernel_reload([], include_cache=True)

    def test_mpc_code(self):
        jd = Time.j2000().jd
        state0 = spice.mpc_code_to_ecliptic("Palomar Mountain", jd)

        code = find_obs_code("Palomar Mountain")

        state1 = spice.earth_pos_to_ecliptic(jd, *code[:-1])
        assert np.isclose((state0.pos - state1.pos).r, 0.0)
        assert np.isclose((state0.vel - state1.vel).r, 0.0)

    def test_moon_frac(self):
        jd = Time.j2000().jd
        frac = spice.moon_illumination_frac(jd)
        # matches JPL Horizons to 4 decimal places
        # Horizons reports: 0.230064
        assert np.isclose(frac, 0.23018685933)

        # Horizons reports: 0.200455
        assert np.isclose(spice.moon_illumination_frac(2451555), 0.2003318)
