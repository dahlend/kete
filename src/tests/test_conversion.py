import numpy as np
import pytest
from kete import constants
from kete.conversion import (
    compute_albedo,
    compute_diameter,
    compute_earth_radius,
    compute_h_mag,
    compute_eccentric_anomaly,
    compute_tisserand,
    ra_degrees_to_hms,
    ra_hms_to_degrees,
    dec_degrees_to_dms,
    dec_dms_to_degrees,
    flux_to_mag,
    mag_to_flux,
)


# eccentricity, mean_anomaly, eccentric_anomaly
ECC_ANOM_VALUES = [
    [0.0, 90, 90],
    [0.5, 35.190199, 60],
    [1.0, 66.8450757, 57.2957795],
    [1.5, 40.9451300, 55.1428098],
]


@pytest.mark.parametrize(
    "d_pv_Hv",
    [
        [1, 0.01, 20.6176],
        [1, 0.05, 18.8702],
        [1, 0.1, 18.1176],
        [1, 0.3, 16.9248],
        [1, 0.6, 16.1722],
    ],
)
def test_compute_h_d_pv(d_pv_Hv):
    """
    For a 1 km object with the listed range of albedos, confirm the
    computed H matches the manual values given in the input params. Also ensure
    that all three conversions between the params work.
    """
    d, pv, hv = d_pv_Hv
    c_hg = 1329.0  # default for V band
    assert np.isclose(compute_h_mag(d, pv, c_hg), hv, atol=0.0001)
    assert np.isclose(compute_albedo(d, hv, c_hg), pv, atol=0.0001)
    assert np.isclose(compute_diameter(pv, hv, c_hg), d, atol=0.0001)


def test_ra_deg_hms():
    """
    Step through the full range of RA values, doing a back-and-forth check.
    """
    ra_in = np.arange(0, 360, 0.1357)
    ra_out = [ra_hms_to_degrees(ra_degrees_to_hms(r)) for r in ra_in]
    assert np.allclose(ra_in, ra_out)

    with pytest.raises(ValueError):
        ra_hms_to_degrees("0 1 2 3 4")


def test_dec_deg_dms():
    """
    Step through the full range of Dec values, doing a back-and-forth check.
    """
    dec_in = np.append(np.arange(-90, 90, 0.1357), 90)
    dec_out = [dec_dms_to_degrees(dec_degrees_to_dms(d)) for d in dec_in]

    for deg in dec_in:
        dec_out = dec_dms_to_degrees(dec_degrees_to_dms(deg))
        assert np.allclose(deg, dec_out)

    with pytest.raises(ValueError, match="format"):
        dec_dms_to_degrees("+0 1 2 3 4")
    with pytest.raises(ValueError, match="between"):
        dec_degrees_to_dms(95)
    with pytest.raises(ValueError, match="between"):
        dec_degrees_to_dms(-95)


@pytest.mark.parametrize("ecc, mean_anom, expected_ecc_anom", ECC_ANOM_VALUES)
def test_eccentric_anomaly(ecc, mean_anom, expected_ecc_anom):
    assert np.isclose(compute_eccentric_anomaly(ecc, mean_anom, 1), expected_ecc_anom)
    assert np.isclose(
        compute_eccentric_anomaly(np.array(ecc), np.array(mean_anom), 1),
        expected_ecc_anom,
    )


def test_eccentric_anomaly_vec():
    vals = np.array(ECC_ANOM_VALUES).T
    assert np.allclose(
        compute_eccentric_anomaly(vals[0], vals[1], [1 for _ in vals[0]]), vals[2]
    )


def test_mag_to_flux():
    assert np.isclose(mag_to_flux(100, 100), 0.0)
    assert np.isclose(flux_to_mag(100, 100), 0.0)
    assert np.isclose(flux_to_mag(0, 100), np.inf)


def test_earth_radius():
    assert np.isclose(
        constants.EARTH_MINOR_AXIS_M / constants.AU_M, compute_earth_radius(90)
    )
    assert np.isclose(
        constants.EARTH_MAJOR_AXIS_M / constants.AU_M, compute_earth_radius(0)
    )


def test_tisserand():
    val = compute_tisserand(2, 0.1, 15)
    assert np.isclose(val, 3.793542737489037)

    val = compute_tisserand(2, 0, 0, 2)
    assert np.isclose(val, 3)
