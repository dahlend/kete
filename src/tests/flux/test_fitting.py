"""Tests for the model-fitting bindings."""

import pytest

import kete


def test_fluxobs_positional_order():
    # The leading positional arguments are (flux, sigma, band, sun2obj, sun2obs,
    # is_upper_limit), so existing positional calls keep their meaning.
    sun2obj = [2.0, 0.0, 0.0]
    sun2obs = [2.5, 0.3, 0.0]
    det = kete.flux.FluxObs(1.0e-3, 1.0e-4, "W3", sun2obj, sun2obs)
    assert det.flux == 1.0e-3
    assert det.sigma == 1.0e-4
    assert not det.is_upper_limit
    assert list(det.sun2obj) == sun2obj

    lim = kete.flux.FluxObs(2.0e-3, 3.0e-4, "W3", sun2obj, sun2obs, True)
    assert lim.is_upper_limit
    assert lim.flux == 2.0e-3

    bounded = kete.flux.FluxObs(None, None, "W3", sun2obj, sun2obs, bounds=(0.1, 0.2))
    assert bounded.flux is None
    assert bounded.bounds == (0.1, 0.2)

    with pytest.raises(ValueError, match="at least one"):
        kete.flux.FluxObs(None, None, "W3", sun2obj, sun2obs)
