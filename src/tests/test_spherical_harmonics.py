# SPDX-FileCopyrightText: 2026 Dar Dahlen
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest

import kete

CUBE_FACES = np.array(
    [
        [0, 2, 1],
        [1, 2, 3],
        [4, 5, 6],
        [5, 7, 6],
        [0, 1, 4],
        [1, 5, 4],
        [2, 6, 3],
        [3, 6, 7],
        [0, 4, 2],
        [2, 4, 6],
        [1, 3, 5],
        [3, 7, 5],
    ]
)

CERES = 20000001


def box(size, offset=(0.0, 0.0, 0.0)):
    """A 3:2:1 box, centered on `offset`."""
    corners = np.array(
        [[(i & 1), (i >> 1) & 1, (i >> 2) & 1] for i in range(8)], dtype=float
    )
    return (corners - 0.5) * np.array([3.0, 2.0, 1.0]) * size + np.array(offset)


@pytest.fixture
def restore_ceres():
    yield
    kete.propagation.register_mass(CERES)


def directions():
    return [
        np.array([0.3, -0.8, 0.5]),
        np.array([-1.0, 0.2, 0.1]),
        np.array([0.1, 0.1, -1.0]),
    ]


class TestSphericalHarmonics:
    def test_point_mass_and_accessors(self):
        field = kete.shape.SphericalHarmonics(2.0, 1.5, [[1.0]], [[0.0]], 1.2)
        p = np.array([1.0, -2.0, 0.5])
        r = np.linalg.norm(p)
        assert np.isclose(field.potential(p), 2.0 / r, rtol=1e-14)
        assert np.allclose(field.field(p), -2.0 * p / r**3, rtol=1e-14)
        accel, grad = field.field_and_gradient(p)
        assert np.allclose(accel, -2.0 * p / r**3, rtol=1e-14)
        expected = 2.0 * (3 * np.outer(p, p) - r**2 * np.eye(3)) / r**5
        assert np.allclose(grad, expected, rtol=1e-13)
        assert field.degree == 0
        assert field.radius == 1.5
        assert field.min_radius == 1.2
        assert field.c == [[1.0]]
        assert field.s == [[0.0]]
        with pytest.raises(ValueError, match="minimum radius"):
            field.field([0.5, 0.0, 0.0])

    def test_rejects_bad_coefficients(self):
        with pytest.raises(ValueError, match="S_10"):
            kete.shape.SphericalHarmonics(
                1.0, 1.0, [[1.0], [0.0, 0.0]], [[0.0], [0.1, 0.0]]
            )
        with pytest.raises(ValueError, match="degree 1"):
            kete.shape.SphericalHarmonics(1.0, 1.0, [[1.0], [0.0]], [[0.0], [0.0]])

    def test_from_polyhedron_matches_closed_form(self):
        poly = kete.shape.Polyhedron(box(1.0, (0.2, -0.1, 0.05)), CUBE_FACES, 1.0)
        exact = poly.without_far_field()
        field = kete.shape.SphericalHarmonics.from_polyhedron(exact, 20)
        assert np.isclose(field.min_radius, poly.bounding_radius)
        for d in directions():
            p = d / np.linalg.norm(d) * 3 * poly.bounding_radius
            expected = np.array(exact.field(p)[0])
            got = np.array(field.field(p))
            assert np.linalg.norm(got - expected) < 1e-9 * np.linalg.norm(expected)
        assert field.truncated(4).degree == 4

    def test_polyhedron_far_field(self):
        poly = kete.shape.Polyhedron(box(1.0, (0.2, -0.1, 0.05)), CUBE_FACES, 1.0)
        exact = poly.without_far_field()
        assert np.isclose(poly.far_field_radius, 3 * poly.bounding_radius)
        far = poly.far_field()
        assert far is not None
        assert 0 < far.degree <= 40
        assert exact.far_field_radius is None
        assert exact.far_field() is None
        for d in directions():
            p = d / np.linalg.norm(d) * 1.5 * poly.far_field_radius
            accel, omega = poly.field(p)
            expected = np.array(exact.field(p)[0])
            assert omega == 0.0
            assert np.linalg.norm(np.array(accel) - expected) < 1e-10 * np.linalg.norm(
                expected
            )


def near_ceres(jd):
    """A satellite 2000 km from Ceres on a near circular orbit."""
    ceres = kete.spice.get_state("ceres", jd, center=0, frame=kete.Frames.Equatorial)
    au_km = kete.constants.AU_KM
    return kete.State(
        "near",
        jd,
        np.array(ceres.pos) + np.array([2000.0, 0.0, 0.0]) / au_km,
        np.array(ceres.vel) + np.array([0.0, 0.17, 0.03]) * 86400 / au_km,
        frame=kete.Frames.Equatorial,
        center_id=0,
    )


def propagate(state, jd):
    out = kete.propagate_n_body(
        [state], jd, include_asteroids=True, suppress_errors=False
    )
    return np.array(out[0].change_center(0).as_equatorial.pos)


class TestRegisterSphericalHarmonics:
    def test_argument_checks(self, restore_ceres):
        reg = kete.propagation.register_spherical_harmonics
        c, s = [[1.0]], [[0.0]]
        with pytest.raises(ValueError, match="Exactly one"):
            reg(CERES, c, s, 500.0, 500.0, 5000.0)
        with pytest.raises(ValueError, match="at least min_radius"):
            reg(CERES, c, s, 500.0, 500.0, 400.0, rotation=np.eye(3))
        with pytest.raises(ValueError, match="C_00"):
            reg(CERES, [[0.9]], s, 500.0, 500.0, 5000.0, rotation=np.eye(3))

    def test_degree_zero_is_the_point_mass(self, restore_ceres):
        jd = kete.Time.from_ymd(2024, 1, 1).jd
        state = near_ceres(jd)
        point = propagate(state, jd + 0.25)
        kete.propagation.register_spherical_harmonics(
            CERES, [[1.0]], [[0.0]], 500.0, 500.0, 50_000.0, rotation=np.eye(3)
        )
        harmonics = propagate(state, jd + 0.25)
        # Heliocentric positions near 2.6 AU carry roundoff of a few mm.
        assert np.linalg.norm(harmonics - point) * kete.constants.AU_KM < 1e-4

    def test_matches_a_registered_polyhedron(self, restore_ceres):
        """A box registered as a polyhedron and as its own exact expansion moves a
        satellite outside the Brillouin sphere the same way."""
        jd = kete.Time.from_ymd(2024, 1, 1).jd
        state = near_ceres(jd)
        verts = box(400.0)
        kete.propagation.register_polyhedron(
            CERES, verts, CUBE_FACES, 50_000.0, rotation=np.eye(3)
        )
        with_polyhedron = propagate(state, jd + 0.25)
        field = kete.shape.SphericalHarmonics.from_polyhedron(
            kete.shape.Polyhedron(verts, CUBE_FACES, 1.0).without_far_field(), 24
        )
        kete.propagation.register_spherical_harmonics(
            CERES,
            field.c,
            field.s,
            field.radius,
            field.min_radius,
            50_000.0,
            rotation=np.eye(3),
        )
        with_harmonics = propagate(state, jd + 0.25)
        shift = np.linalg.norm(with_harmonics - with_polyhedron)
        shift_km = shift * kete.constants.AU_KM
        assert shift_km < 1e-3

    def test_inside_minimum_radius_raises(self, restore_ceres):
        jd = kete.Time.from_ymd(2024, 1, 1).jd
        kete.propagation.register_spherical_harmonics(
            CERES, [[1.0]], [[0.0]], 500.0, 3000.0, 50_000.0, rotation=np.eye(3)
        )
        with pytest.raises(ValueError, match="impact"):
            propagate(near_ceres(jd), jd + 0.25)
