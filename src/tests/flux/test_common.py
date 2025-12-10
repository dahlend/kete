import numpy as np
import pytest

from kete import Vector, constants, conversion
from kete.flux import (
    black_body_flux,
    hg_apparent_flux,
    hg_apparent_mag,
    hg_phase_curve_correction,
    sub_solar_temperature,
)


def test_black_body_radiation():
    # A black body at 1000 Kelvin, at wavelength 20757:
    # At this point, the 1/expm1 portion of plancks law == 1.
    wavelength = 20757.16266505257  # nm
    temp = 1000  # kelvin
    rad = black_body_flux(temp=temp, wavelength=wavelength)
    # 1474 is 2 * h * c^3 * GHz in Jy / steradian
    # (see the wikipedia entry for plancks law)
    rad /= 1474.49946
    assert abs((constants.SPEED_OF_LIGHT / rad ** (1 / 3)) - wavelength) < 0.5

    # This temp will result in the output to be scaled by a factor of 2
    temp = np.log(2) * 1000 / np.log(3)
    rad = black_body_flux(temp=temp, wavelength=wavelength)
    rad /= 1474.49946 / 2
    assert abs((constants.SPEED_OF_LIGHT / rad ** (1 / 3)) - wavelength) < 0.5

    assert np.allclose(black_body_flux(0, wavelength), 0)
    assert np.allclose(black_body_flux(1000, 0), 0)
    assert np.allclose(black_body_flux(1e-5, 1e-5), 0)


def test_subsolarpoint_temp():
    vec = Vector([1, 0, 0])
    # albedo, G set to make bond_albedo == 1
    assert (
        sub_solar_temperature(
            vec.r,
            geom_albedo=1 / 0.29,
            g_param=0,
            emissivity=1,
            beaming=1,
        )
        == 0
    )

    for r in range(1, 10):
        vec = Vector([r, 0, 0])
        temp = sub_solar_temperature(
            r, geom_albedo=0, g_param=0, emissivity=1, beaming=1
        )
        temp = temp**4
        expected = constants.SOLAR_FLUX / r**2 / constants.STEFAN_BOLTZMANN
        assert np.allclose(temp, expected)


@pytest.mark.parametrize("obs_pos", np.arange(-0.5, 1, 0.1))
@pytest.mark.parametrize("h", np.arange(5, 15, 0.5))
def test_reflected(obs_pos, h):
    albedo = 0.1
    g = 0.1

    app_mag = hg_apparent_mag([2, 0, 0], [0, obs_pos, 0], h, g)
    diam = conversion.compute_diameter(albedo, h)

    flux = hg_apparent_flux(
        sun2obj=[2, 0, 0],
        sun2obs=[0, obs_pos, 0],
        g_param=g,
        wavelength=562.25,
        v_albedo=albedo,
        h_mag=h,
        diameter=diam,
    )
    converted_mag = conversion.flux_to_mag(flux, 3620)

    assert np.isclose(app_mag, converted_mag, 1e-3)


@pytest.mark.parametrize("G", np.arange(-0.5, 1, 0.1))
def test_hg_phase_curve_correction(G):
    """
    The only thing we know for certain is that 0 phase should always result in a
    correction of 1 regardless of G.  Other values are based on fits that could change,
    so keep the test to just this for the time being.
    """
    assert np.isclose(hg_phase_curve_correction(G, 0.0), 1.0)
