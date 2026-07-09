import numpy as np
import pytest

import kete

SUN2OBS = [0, 1, 0]
SUN2OBJ = [1, 1, 0]


def test_neos_neatm_model():
    output = kete.flux.neatm_model_flux(
        SUN2OBJ,
        SUN2OBS,
        band_albedos=[0.3, 0.3],
        vis_albedo=0.3,
        diameter=1.0,
        g_param=0.15,
        beaming=1.4,
        bands="neos",
    )
    assert output.fluxes[0] >= 0.0
    assert output.fluxes[1] >= 0.0


def test_neos_frm_model():
    output = kete.flux.frm_model_flux(
        SUN2OBJ,
        SUN2OBS,
        band_albedos=[0.3, 0.3],
        vis_albedo=0.3,
        diameter=1.0,
        g_param=0.15,
        bands="neos",
    )
    assert output.fluxes[0] >= 0.0
    assert output.fluxes[1] >= 0.0


def test_wise_neatm_model():
    output = kete.flux.neatm_model_flux(
        SUN2OBJ,
        SUN2OBS,
        band_albedos=[0.3, 0.3, 0.3, 0.3],
        vis_albedo=0.3,
        diameter=1.0,
        g_param=0.15,
        beaming=1.4,
        bands="wise",
    )
    assert np.isclose(output.fluxes[0], 3.04049074097550e-05)
    assert np.isclose(output.fluxes[1], 1.55783572941838e-04)
    assert np.isclose(output.fluxes[2], 5.34077839291434e-03)
    assert np.isclose(output.fluxes[3], 8.67403681666771e-03)


def test_wise_tpm_model():
    output = kete.flux.tpm_model_flux(
        SUN2OBJ,
        SUN2OBS,
        band_albedos=[0.3, 0.3, 0.3, 0.3],
        thermal_inertia=200.0,
        period=6 * 3600.0,
        pole=[0, 0, 1],
        vis_albedo=0.3,
        diameter=1.0,
        g_param=0.15,
        bands="wise",
    )
    assert len(output.fluxes) == 4
    assert np.all(np.isfinite(output.fluxes))
    assert np.all(np.array(output.fluxes) > 0.0)
    # thermal emission dominates in the longer WISE bands
    assert output.thermal_fluxes[2] / output.fluxes[2] > 0.5
    assert output.thermal_fluxes[3] / output.fluxes[3] > 0.5


def test_tpm_prograde_retrograde_differ():
    # at a nonzero phase angle the thermal lag has a sign, so reversing the spin
    # (prograde vs retrograde) changes the observed thermal flux.
    kwargs = dict(
        sun2obj=[1.5, 0, 0],
        sun2obs=[0.5, 0.4, 0],
        band_albedos=[0.05] * 4,
        thermal_inertia=200.0,
        period=6 * 3600.0,
        vis_albedo=0.05,
        diameter=1.0,
        bands="wise",
    )
    prograde = kete.flux.tpm_model_flux(pole=[0, 0, 1], **kwargs)
    retrograde = kete.flux.tpm_model_flux(pole=[0, 0, -1], **kwargs)
    assert not np.isclose(prograde.fluxes[2], retrograde.fluxes[2])


def test_tpm_limits_match_neatm_and_frm():
    # Zero inertia is instantaneous equilibrium (NEATM with beaming 1); zero period
    # is an infinitely fast rotator (FRM). Both are valid limits, not errors.
    common = dict(
        sun2obj=[1.5, 0, 0],
        sun2obs=[0.5, 0.3, 0],
        band_albedos=[0.05] * 4,
        vis_albedo=0.05,
        diameter=1.0,
        bands="wise",
    )
    zero_inertia = kete.flux.tpm_model_flux(
        thermal_inertia=0.0, period=3600.0, pole=[0, 0, 1], **common
    )
    neatm = kete.flux.neatm_model_flux(beaming=1.0, **common)
    assert np.allclose(zero_inertia.thermal_fluxes, neatm.thermal_fluxes, rtol=1e-2)

    zero_period = kete.flux.tpm_model_flux(
        thermal_inertia=200.0, period=0.0, pole=[0, 0, 1], **common
    )
    frm = kete.flux.frm_model_flux(**common)
    assert np.allclose(zero_period.thermal_fluxes, frm.thermal_fluxes, rtol=5e-2)


def test_tpm_invalid_inputs_raise():
    common = dict(
        sun2obj=[1.5, 0, 0],
        sun2obs=[0.5, 0.3, 0],
        band_albedos=[0.05] * 4,
        vis_albedo=0.05,
        diameter=1.0,
        pole=[0, 0, 1],
        bands="wise",
    )
    with pytest.raises(ValueError):
        kete.flux.tpm_model_flux(thermal_inertia=-1.0, period=3600.0, **common)
    with pytest.raises(ValueError):
        kete.flux.tpm_model_flux(thermal_inertia=200.0, period=-5.0, **common)


def test_tpm_oblate_phase_independent():
    # an oblate spheroid (a = b) is rotationally symmetric, so the flux does not
    # depend on the rotation phase / epoch
    base = dict(
        sun2obj=[1.5, 0, 0],
        sun2obs=[0.5, 0.4, 0],
        band_albedos=[0.05] * 4,
        thermal_inertia=150.0,
        period=6 * 3600.0,
        pole=[0, 0, 1],
        diameter=1.0,
        vis_albedo=0.05,
        bands="wise",
        axis_ratios=(1.0, 0.6),
    )
    a = kete.flux.tpm_model_flux(epoch=0.0, **base)
    b = kete.flux.tpm_model_flux(epoch=1.234, **base)
    assert np.allclose(a.fluxes, b.fluxes, rtol=1e-9)


def test_tpm_triaxial_phase_dependent():
    # a triaxial ellipsoid (a != b) produces a rotational thermal lightcurve
    base = dict(
        sun2obj=[1.5, 0, 0],
        sun2obs=[0.5, 0.4, 0],
        band_albedos=[0.05] * 4,
        thermal_inertia=150.0,
        period=6 * 3600.0,
        pole=[0, 0, 1],
        diameter=1.0,
        vis_albedo=0.05,
        bands="wise",
        axis_ratios=(2.0, 0.8),
    )
    quarter_turn_days = (6 * 3600.0 / 4) / 86400.0
    a = kete.flux.tpm_model_flux(epoch=0.0, **base)
    b = kete.flux.tpm_model_flux(epoch=quarter_turn_days, **base)
    assert abs(a.fluxes[3] - b.fluxes[3]) / a.fluxes[3] > 0.01


def test_tpm_roughness_runs():
    # the rough (beaming) path solves a crater per latitude band and is slow; verify
    # it runs and returns finite positive flux, and that an invalid angle raises.
    # `roughness` is the mean slope angle in degrees, capped at the ~57.3 deg
    # full-coverage-cap (hemisphere) limit.
    common = dict(
        sun2obj=[1.5, 0, 0],
        sun2obs=[0.5, 0.3, 0],
        band_albedos=[0.05] * 4,
        thermal_inertia=150.0,
        period=6 * 3600.0,
        pole=[0, 0, 1],
        diameter=1.0,
        vis_albedo=0.05,
        bands="wise",
    )
    # a mean slope angle above the ~57.3 deg ceiling is rejected before any solve (fast)
    with pytest.raises(ValueError):
        kete.flux.tpm_model_flux(roughness=70.0, **common)
    res = kete.flux.tpm_model_flux(roughness=40.0, **common)
    assert np.all(np.isfinite(res.fluxes))
    assert np.all(np.array(res.fluxes) > 0.0)


def test_roughness_mean_slope_to_rms():
    # The mean-slope -> RMS-slope cross-walk is monotone, returns an RMS slope at least
    # as large as the mean slope, and rejects values outside (0, 57.3] degrees.
    prev = 0.0
    for ms in [10.0, 20.0, 30.0, 45.0]:
        rms = kete.flux.roughness_mean_slope_to_rms(ms)
        assert rms >= ms  # RMS slope >= mean slope
        assert rms > prev  # monotone increasing
        prev = rms
    for bad in [0.0, -5.0, 60.0]:
        with pytest.raises(ValueError):
            kete.flux.roughness_mean_slope_to_rms(bad)


def test_tpm_night_side_low_inertia_finite():
    # a ~180 degree phase geometry views the night side; NEATM would give zero
    # thermal flux there, but the TPM emits stored heat. Even at very low (stiff)
    # inertia the result must be finite and non-negative.
    for gamma in [0.5, 1.0, 200.0]:
        output = kete.flux.tpm_model_flux(
            sun2obj=[1.5, 0, 0],
            sun2obs=[2.5, 0, 0],
            band_albedos=[0.1] * 4,
            thermal_inertia=gamma,
            period=6 * 3600.0,
            pole=[0, 0, 1],
            vis_albedo=0.1,
            diameter=1.0,
            bands="wise",
        )
        assert np.all(np.isfinite(output.thermal_fluxes))
        assert np.all(np.array(output.thermal_fluxes) >= 0.0)
