import numpy as np

import kete


def test_triangle_ellipsoid():
    geom = kete.shape.TriangleEllipsoid(6)

    norms = geom.normals
    assert len(norms) > 100
    assert np.allclose(np.linalg.norm(norms, axis=1), 1.0)
    assert len(norms) == len(geom.areas)
