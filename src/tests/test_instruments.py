import numpy as np
import pytest

import kete

IK = """KPL/IK
\\begindata
INS399903_FOV_FRAME = 'J2000'
INS399903_FOV_SHAPE = 'RECTANGLE'
INS399903_BORESIGHT = ( 1 0 0 )
INS399903_FOV_CLASS_SPEC = 'ANGLES'
INS399903_FOV_REF_VECTOR = ( 0 1 0 )
INS399903_FOV_REF_ANGLE = 2.0
INS399903_FOV_CROSS_ANGLE = 1.0
INS399903_FOV_ANGLE_UNITS = 'DEGREES'
INS399903_PIXEL_SIZE = ( 13.5 13.5 )
INS399904_FOV_FRAME = 'J2000'
INS399904_FOV_SHAPE = 'ELLIPSE'
INS399904_BORESIGHT = ( 1 0 0 )
INS399904_FOV_BOUNDARY_CORNERS = ( 1 0.1 0  1 0 0.05 )
NAIF_BODY_NAME += 'PYTEST CAMERA'
NAIF_BODY_CODE += 399903
\\begintext
"""


@pytest.fixture
def ik(tmp_path):
    path = tmp_path / "test.ti"
    path.write_text(IK)
    kete.spice.kernel_reload([str(path)])
    yield
    kete.spice.kernel_reload()


def test_instrument_fov(ik):
    jd = kete.Time(2460000.5)
    fov = kete.spice.instrument_fov("pytest camera", jd)
    assert isinstance(fov, kete.RectangleFOV)
    earth = kete.spice.get_state(399, jd, center=10)
    assert np.allclose(fov.observer.pos, earth.pos)
    # The boresight is +x of J2000; the reference angle is along +y.
    pointing = np.array(list(fov.pointing.as_equatorial))
    assert np.allclose(pointing, [1, 0, 0], atol=1e-12)
    assert sorted([fov.lon_width, fov.lat_width]) == pytest.approx([2.0, 4.0])


def test_instrument_fov_definition(ik):
    fov = kete.spice.instrument_fov_definition(399903)
    shape, frame, boresight, bounds = fov
    assert (shape, frame, boresight, len(bounds)) == (
        "RECTANGLE",
        "J2000",
        [1, 0, 0],
        4,
    )
    assert fov.shape == "RECTANGLE"
    assert fov.bounds == bounds
    with pytest.raises(ValueError, match="elliptical"):
        kete.spice.instrument_fov(399904, kete.Time(2460000.5))
    with pytest.raises(ValueError):
        kete.spice.instrument_fov("NO SUCH CAMERA", kete.Time(2460000.5))


def test_kernel_variable(ik):
    assert kete.spice.kernel_variable("INS399903_PIXEL_SIZE") == [13.5, 13.5]
    assert kete.spice.kernel_variable("INS399903_FOV_FRAME") == ["J2000"]
    assert kete.spice.kernel_variable("NOT_A_VARIABLE") is None


def test_polygon_fov_non_convex():
    obs = kete.State("obs", kete.Time(2460000.5), [1, 0, 0], [0, 0, 0])
    corners = [
        [1, x, y]
        for x, y in [
            (-0.05, -0.03),
            (0.05, -0.03),
            (0.05, 0.03),
            (0.01, 0.03),
            (0.01, 0.0),
            (-0.01, 0.0),
            (-0.01, 0.03),
            (-0.05, 0.03),
        ]
    ]
    fov = kete.PolygonFOV(corners, obs)
    assert len(fov.corners) == 8
    with pytest.raises(ValueError):
        kete.PolygonFOV([[1, 0, 0], [1, 0.1, 0]], obs)
