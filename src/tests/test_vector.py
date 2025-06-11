import numpy as np
import pytest

from kete.vector import Frames, Vector


@pytest.fixture(scope="function")
def random_vecs():
    rng = np.random.default_rng(12345)
    return rng.random([10, 3])


class TestVector:
    @pytest.mark.parametrize("vals", [(0, 1, 2), (5, 5, 5), (3, -1, 5)])
    def test_init(self, vals):
        v = Vector(vals)
        assert np.allclose(v, vals)
        assert np.allclose(v.x, vals[0])
        assert np.allclose(v.y, vals[1])
        assert np.allclose(v.z, vals[2])
        assert np.allclose(v.r, np.linalg.norm(vals, axis=0))
        assert all(np.array([v.r]).reshape(-1) > 0)
        assert v == v

        v0 = Vector.from_el_az(v.el, v.az, v.r, Frames.Equatorial)
        assert np.allclose(v, v0)

    def test_rotation(self):
        x = Vector([1, 0, 0])
        y = Vector([0, 1, 0])
        z = Vector([0, 0, 1])

        assert x.rotate_around(x, 90) == x
        assert y.rotate_around(y, 90) == y
        assert z.rotate_around(z, 90) == z
        assert x.rotate_around(y, 90) == -z
        assert x.rotate_around(y, -90) == z
        assert y.rotate_around(z, 90) == -x
        assert y.rotate_around(z, -90) == x
        assert z.rotate_around(x, 90) == -y
        assert z.rotate_around(x, -90) == y
        assert x.rotate_around(y, 180) == -x

        for theta in np.linspace(0, 180, 10):
            assert x.rotate_around(z, theta).rotate_around(z, -theta) == x
            assert x.rotate_around(z, theta) == x.rotate_around(-z, -theta)

    @pytest.mark.parametrize("el", np.linspace(0.001, 90 - 0.001, 4))
    @pytest.mark.parametrize("az", np.linspace(0.001, 180 - 0.001, 4))
    def test_el_az_range(self, el, az):
        vec = Vector.from_el_az(el, az, 1, Frames.Ecliptic)
        assert np.isclose(vec.el, el)
        assert np.isclose(vec.az, az)

    def test_angle_between(self):
        # Calculate the haversine distance from big ben and the statue of liberty
        p0 = Vector.from_lat_lon(51.5007, 0.1246)
        p1 = Vector.from_lat_lon(40.6892, 74.0445)
        assert np.allclose(np.radians(p0.angle_between(p1)) * 6371, 5574.84, 2)

    @pytest.mark.parametrize("lat", np.linspace(-90 + 0.001, 90 - 0.001, 3))
    @pytest.mark.parametrize("lon", np.linspace(0.001, 180 - 0.001, 7))
    def test_lat_lon_range(self, lat, lon):
        vec = Vector.from_lat_lon(lat, lon)
        assert np.isclose(vec.lat, lat)
        assert np.isclose(vec.lon, lon)

    @pytest.mark.parametrize("lat", np.linspace(0.1, 90 - 0.1, 3))
    @pytest.mark.parametrize("lon", np.linspace(0.1, 90 - 0.1, 3))
    def test_lat_lon(self, lat, lon):
        vec = Vector.from_lat_lon(lat, lon)
        assert np.isclose(vec.lat, lat)
        assert np.isclose(vec.lon, lon)

        vec2 = vec.as_equatorial.as_ecliptic
        assert np.isclose(vec.lat, vec2.lat)
        assert np.isclose(vec.lon, vec2.lon)
