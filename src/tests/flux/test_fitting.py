# SPDX-FileCopyrightText: 2026 Dar Dahlen
# SPDX-License-Identifier: BSD-3-Clause

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

    assert lim.sigma == 3.0e-4


def test_param_prior_gaussian():
    assert kete.flux.ParamPrior((0.5, 3.0), (1.0, 0.3)).gaussian == (1.0, 0.3)
    asym = kete.flux.ParamPrior((0.5, 3.0), (1.0, 0.2, 0.4))
    assert asym.gaussian == (1.0, 0.2, 0.4)
    assert kete.flux.ParamPrior((0.5, 3.0)).gaussian is None


@pytest.mark.parametrize("bounds", [(5.0, 5.0), (10.0, 5.0), (float("nan"), 10.0)])
def test_unordered_prior_bounds_raise(bounds):
    sun2obj = [2.0, 0.0, 0.0]
    sun2obs = [2.5, 0.3, 0.0]
    obs = [
        kete.flux.FluxObs(f, f * 0.05, band, sun2obj, sun2obs)
        for f, band in [(1e-3, "W3"), (3e-3, "W4"), (1e-5, "W2")]
    ]
    priors = kete.flux.FluxPriors(diameter=kete.flux.ParamPrior(bounds))
    with pytest.raises(ValueError, match="diameter prior bounds"):
        kete.flux.fit_model("neatm", obs, h_mag=15.0, priors=priors)
