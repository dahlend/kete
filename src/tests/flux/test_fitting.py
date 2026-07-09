"""Tests for the model-fitting bindings, focused on the TPM model."""

import numpy as np
import pytest

import kete

PERIOD = 6 * 3600.0
POLE = [0, 0, 1]


def _synthetic_tpm_obs():
    """Multi-phase WISE observations of a 10 km, Gamma=200 object at 2 AU."""
    sun2obj = np.array([2.0, 0.0, 0.0])
    obj2sun_hat = -sun2obj / np.linalg.norm(sun2obj)
    obs = []
    for phase_deg in (0.0, 45.0, 90.0, 135.0):
        phi = np.radians(phase_deg)
        sun2obs = sun2obj + np.array([obj2sun_hat[0] * np.cos(phi), -np.sin(phi), 0.0])
        result = kete.flux.tpm_model_flux(
            sun2obj.tolist(),
            sun2obs.tolist(),
            band_albedos=[0.05] * 4,
            thermal_inertia=200.0,
            period=PERIOD,
            pole=POLE,
            diameter=10.0,
            vis_albedo=0.05,
            bands="wise",
        )
        for band, flux in zip(["W1", "W2", "W3", "W4"], result.fluxes):
            obs.append(
                kete.flux.FluxObs(
                    flux, flux * 0.05, band, sun2obj.tolist(), sun2obs.tolist()
                )
            )
    return obs


def test_tpm_fit_requires_spin():
    obs = _synthetic_tpm_obs()
    # period and pole are mandatory for the TPM model
    with pytest.raises(ValueError):
        kete.flux.fit_model(
            "tpm", obs, pole=POLE, num_chains=1, num_tune=5, num_draws=5
        )
    with pytest.raises(ValueError):
        kete.flux.fit_model(
            "tpm", obs, period=PERIOD, num_chains=1, num_tune=5, num_draws=5
        )
    # a zero period is the FRM limit, not a valid TPM fit
    with pytest.raises(ValueError):
        kete.flux.fit_model(
            "tpm", obs, period=0.0, pole=POLE, num_chains=1, num_tune=5, num_draws=5
        )


def test_tpm_fit_runs():
    obs = _synthetic_tpm_obs()
    res = kete.flux.fit_model(
        "tpm",
        obs,
        period=PERIOD,
        pole=POLE,
        h_mag=18.0,
        num_chains=1,
        num_tune=40,
        num_draws=40,
    )
    assert res.model == "Tpm"
    assert res.columns[2] == "thermal_inertia"
    assert res.thermal_inertia is not None
    assert res.beaming is None  # beaming is NEATM-only
    # diameter is well constrained; thermal inertia to the right order of magnitude
    assert abs(res.diameter.median - 10.0) / 10.0 < 0.4
    assert 50.0 < res.thermal_inertia.median < 800.0


def test_tpm_fit_fixed_roughness_runs():
    # A fixed roughness given as a mean slope angle (degrees) is converted to the
    # internal crater angle and applied via the shipped correction table; the fit must
    # still run and recover the diameter. An angle above the ~57.3 deg ceiling is
    # rejected up front.
    obs = _synthetic_tpm_obs()
    with pytest.raises(ValueError):
        kete.flux.fit_model(
            "tpm", obs, period=PERIOD, pole=POLE, roughness=70.0,
            num_chains=1, num_tune=5, num_draws=5,
        )
    res = kete.flux.fit_model(
        "tpm",
        obs,
        period=PERIOD,
        pole=POLE,
        h_mag=18.0,
        roughness=30.0,
        num_chains=1,
        num_tune=40,
        num_draws=40,
    )
    assert res.model == "Tpm"
    assert abs(res.diameter.median - 10.0) / 10.0 < 0.4


def test_tpm_roughness_rejects_uncovered_band():
    # The shipped roughness table covers only the WISE bands. Fitting roughness on data
    # at a far-off wavelength (here ~MIPS 70 um) must fail loudly at setup rather than
    # silently snapping the correction to the nearest WISE band.
    sun2obj = [2.0, 0.0, 0.0]
    sun2obs = [2.5, 0.3, 0.0]
    obs = [
        kete.flux.FluxObs(1.0e-3, 1.0e-4, 70000.0, sun2obj, sun2obs) for _ in range(4)
    ]
    with pytest.raises(ValueError, match="band"):
        kete.flux.fit_model(
            "tpm", obs, period=PERIOD, pole=POLE, h_mag=18.0, roughness=30.0,
            num_chains=1, num_tune=5, num_draws=5,
        )
