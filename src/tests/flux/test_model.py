import numpy as np
import kete

SUN2OBS = [0, 1, 0]
SUN2OBJ = [1, 1, 0]


def test_neos_neatm_model():
    model = kete.flux.NeatmParams.new_neos(
        "visible",
        vis_albedo=0.3,
        band_albedos=[0.3, 0.3],
        g_param=0.15,
        beaming=1.4,
        diam=1.0,
    )
    output = model.evaluate([SUN2OBJ], [SUN2OBS])[0]
    assert model.desig == "visible"
    assert output.fluxes[0] >= 0.0
    assert output.fluxes[1] >= 0.0


def test_neos_frm_model():
    model = kete.flux.FrmParams.new_neos(
        "visible",
        vis_albedo=0.3,
        band_albedos=[0.3, 0.3],
        g_param=0.15,
        diam=1.0,
    )
    output = model.evaluate([SUN2OBJ], [SUN2OBS])[0]
    assert model.desig == "visible"
    assert output.fluxes[0] >= 0.0
    assert output.fluxes[1] >= 0.0


def test_wise_neatm_model():
    model = kete.flux.NeatmParams.new_wise(
        "visible",
        vis_albedo=0.3,
        band_albedos=[0.3, 0.3, 0.3, 0.3],
        g_param=0.15,
        beaming=1.4,
        diam=1.0,
    )
    output = model.evaluate([SUN2OBJ], [SUN2OBS])[0]
    assert model.desig == "visible"
    assert np.isclose(output.fluxes[0], 3.04038361370317e-05)
    assert np.isclose(output.fluxes[1], 0.000155764430378102)
    assert np.isclose(output.fluxes[2], 0.005340059443086914)
    assert np.isclose(output.fluxes[3], 0.008673191020104592)
