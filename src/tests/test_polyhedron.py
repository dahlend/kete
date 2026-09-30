import numpy as np
import pytest

import kete

CUBE_VERTS = np.array(
    [[(i & 1), (i >> 1) & 1, (i >> 2) & 1] for i in range(8)], dtype=float
)
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


def box(size_km):
    """A 3:2:1 box, in km, centered on the origin."""
    return (
        CUBE_VERTS * np.array([3.0, 2.0, 1.0]) * size_km
        - np.array([1.5, 1.0, 0.5]) * size_km
    )


@pytest.fixture
def restore_ceres():
    yield
    kete.propagation.register_mass(CERES)


class TestReadObj:
    def test_round_trip(self, tmp_path):
        path = tmp_path / "cube.obj"
        lines = ["# cube", "vn 0 0 1"]
        lines += [f"v {x} {y} {z}" for x, y, z in CUBE_VERTS]
        # mix plain, slashed and negative (relative) indices
        for k, (a, b, c) in enumerate(CUBE_FACES):
            if k % 3 == 0:
                lines.append(f"f {a + 1} {b + 1} {c + 1}")
            elif k % 3 == 1:
                lines.append(f"f {a + 1}/1/1 {b + 1}//1 {c + 1}/2")
            else:
                lines.append(f"f {a - 8} {b - 8} {c - 8}")
        path.write_text("\n".join(lines) + "\n")
        verts, faces = kete.shape.read_obj(path)
        assert np.array_equal(verts, CUBE_VERTS)
        assert np.array_equal(faces, CUBE_FACES)

    def test_rejects_quads(self, tmp_path):
        path = tmp_path / "quad.obj"
        path.write_text("v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nf 1 2 3 4\n")
        with pytest.raises(ValueError, match="triangular"):
            kete.shape.read_obj(path)


class TestPolyhedron:
    def test_cube_properties(self):
        poly = kete.shape.Polyhedron(CUBE_VERTS, CUBE_FACES, 2.0)
        assert np.isclose(poly.volume, 1.0)
        assert np.allclose(poly.centroid, 0.5)
        assert np.allclose(poly.vertices, CUBE_VERTS)
        assert np.isclose(poly.bounding_radius, np.sqrt(3))
        assert poly.gm == 2.0

    def test_field(self):
        poly = kete.shape.Polyhedron(box(1.0), CUBE_FACES, 1.0)
        accel, omega = poly.field([0.1, 0.2, -0.1])
        assert np.isclose(omega, 4 * np.pi)
        _, omega = poly.field([3.0, 0.0, 0.0])
        assert abs(omega) < 1e-12
        # far away, a point mass pointing at the body
        p = np.array([300.0, -200.0, 100.0])
        accel, _ = poly.field(p)
        point = -p / np.linalg.norm(p) ** 3
        assert np.linalg.norm(accel - point) < 1e-4 * np.linalg.norm(point)
        # the acceleration is the gradient of the potential
        p = np.array([2.3, 1.1, -0.9])
        h = 1e-5
        fd = [
            (poly.potential(p + h * e) - poly.potential(p - h * e)) / (2 * h)
            for e in np.eye(3)
        ]
        assert np.allclose(poly.field(p)[0], fd, rtol=1e-7)
        _, grad, _ = poly.field_and_gradient(p)
        assert np.allclose(grad, np.transpose(grad), atol=1e-14)

    def test_rejects_bad_meshes(self):
        with pytest.raises(ValueError):
            kete.shape.Polyhedron(CUBE_VERTS, CUBE_FACES[:-1], 1.0)
        with pytest.raises(ValueError):
            kete.shape.Polyhedron(CUBE_VERTS, CUBE_FACES[:, ::-1], 1.0)
        with pytest.raises(ValueError):
            kete.shape.Polyhedron(CUBE_VERTS, CUBE_FACES, -1.0)


class TestRegisterPolyhedron:
    def test_argument_checks(self, restore_ceres):
        reg = kete.propagation.register_polyhedron
        verts = box(100.0)
        with pytest.raises(ValueError, match="Exactly one"):
            reg(CERES, verts, CUBE_FACES, 5000.0)
        with pytest.raises(ValueError, match="Exactly one"):
            reg(CERES, verts, CUBE_FACES, 5000.0, frame_id=-1, rotation=np.eye(3))
        with pytest.raises(ValueError, match="rotation"):
            reg(CERES, verts, CUBE_FACES, 5000.0, rotation=np.diag([1, 1, -1.0]))
        with pytest.raises(ValueError, match="larger than the shape"):
            reg(CERES, verts, CUBE_FACES, 50.0, rotation=np.eye(3))
        with pytest.raises(ValueError, match="no known mass"):
            reg(-12345, verts, CUBE_FACES, 5000.0, rotation=np.eye(3))

    def test_units_and_mass(self, restore_ceres):
        poly = kete.propagation.register_polyhedron(
            CERES, box(100.0), CUBE_FACES, 5000.0, rotation=np.eye(3)
        )
        au_km = kete.constants.AU_KM
        assert np.isclose(poly.volume, 6e6 / au_km**3)
        # gm is the table mass (a fraction of the Sun's) times the Sun's GM in
        # AU^3 / day^2
        known = {m[1]: m[2] for m in kete._core.known_masses()}
        assert np.isclose(poly.gm / known[CERES], 2.9591220828559115e-4, rtol=1e-9)

    def test_propagation(self, restore_ceres):
        """A satellite well inside the switch radius feels the box; an orbit that
        never comes near Ceres is unchanged."""
        jd = kete.Time.from_ymd(2024, 1, 1).jd
        ceres = kete.spice.get_state(
            "ceres", jd, center=0, frame=kete.Frames.Equatorial
        )
        au_km = kete.constants.AU_KM
        near = kete.State(
            "near",
            jd,
            np.array(ceres.pos) + np.array([2000.0, 0.0, 0.0]) / au_km,
            # near circular at 2000 km (Ceres GM ~63 km^3/s^2), clear of the box
            np.array(ceres.vel) + np.array([0.0, 0.17, 0.03]) * 86400 / au_km,
            frame=kete.Frames.Equatorial,
            center_id=0,
        )
        far = kete.State(
            "far",
            jd,
            np.array(ceres.pos) + np.array([0.3, 0.0, 0.0]),
            ceres.vel,
            frame=kete.Frames.Equatorial,
            center_id=0,
        )

        def run():
            out = kete.propagate_n_body([near, far], jd + 0.25, include_asteroids=True)
            return [np.array(s.change_center(0).as_equatorial.pos) for s in out]

        point_near, point_far = run()
        # a 1200 x 800 x 400 km box, the long axis along x, toward the satellite
        kete.propagation.register_polyhedron(
            CERES, box(400.0), CUBE_FACES, 50_000.0, rotation=np.eye(3)
        )
        box_near, box_far = run()
        assert np.array_equal(box_far, point_far)
        shift_km = np.linalg.norm(box_near - point_near) * au_km
        # the elongation adds a quadrupole of a few percent of the point-mass pull at
        # 2000 km; over a quarter of the ~20 h orbit that moves the satellite by
        # tens to hundreds of km
        assert 10.0 < shift_km < 1000.0
