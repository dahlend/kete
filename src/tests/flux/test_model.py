import numpy as np

import kete

SUN2OBS = [0, 1, 0]
SUN2OBJ = [1, 1, 0]


def test_neos_neatm_model():
    model = kete.flux.NeatmParams.new_neos(
        vis_albedo=0.3,
        band_albedos=[0.3, 0.3],
        g_param=0.15,
        beaming=1.4,
        diameter=1.0,
    )
    output = model.evaluate([SUN2OBJ], [SUN2OBS])[0]
    assert output.fluxes[0] >= 0.0
    assert output.fluxes[1] >= 0.0


def test_neos_frm_model():
    model = kete.flux.FrmParams.new_neos(
        vis_albedo=0.3,
        band_albedos=[0.3, 0.3],
        g_param=0.15,
        diameter=1.0,
    )
    output = model.evaluate([SUN2OBJ], [SUN2OBS])[0]
    assert output.fluxes[0] >= 0.0
    assert output.fluxes[1] >= 0.0


def test_wise_neatm_model():
    model = kete.flux.NeatmParams.new_wise(
        vis_albedo=0.3,
        band_albedos=[0.3, 0.3, 0.3, 0.3],
        g_param=0.15,
        beaming=1.4,
        diameter=1.0,
    )
    output = model.evaluate([SUN2OBJ], [SUN2OBS])[0]
    assert np.isclose(output.fluxes[0], 3.04049074097550e-05)
    assert np.isclose(output.fluxes[1], 1.55783572941838e-04)
    assert np.isclose(output.fluxes[2], 5.34077839291434e-03)
    assert np.isclose(output.fluxes[3], 8.67403681666771e-03)
