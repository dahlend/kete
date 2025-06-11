"""
Definitions of geometric objects which are used by the thermal and reflected light
models.
"""

from __future__ import annotations

from functools import lru_cache as cache

import numpy as np
from numpy.typing import ArrayLike

from .vector import Vector


class Geometry:
    """
    Generic representation of the geometry of an object made up of planar sections with
    fixed area.

    Parameters
    ----------
    normals:
        A list of vectors which define the normals for each planar surface. These
        are automatically normalized to unit length. This can be a numpy array of
        shape `(n, 3)`
    areas:
        A list of the surface areas associated with the normals, this must be the
        same length as the normals list.
    normalize_area : bool
        Whether or not to normalize all of the areas so that the total is 1.
    """

    def __init__(
        self, normals: ArrayLike, areas: ArrayLike, normalize_area: bool = True
    ):
        self._normals = np.array(normals, dtype=float)
        self._normals /= np.linalg.norm(self._normals, axis=1)[:, np.newaxis]
        self._areas = np.array(areas, dtype=float)
        if len(self._normals) != len(self._areas):
            raise ValueError("The number of normals do not match the number of areas.")
        if normalize_area:
            self._areas /= np.sum(self._areas)

    @property
    def normals(self) -> np.ndarray:
        """
        The normal vectors associated with the faces of this geometry.
        This must be the same length as the areas of this geometry.
        """
        return self._normals

    @property
    def areas(self) -> np.ndarray:
        """
        The area of each of the faces of this geometry, this must be the same length as
        the normals.
        """
        return self._areas


class TriangleFaceted(Geometry):
    """
    Constructed object from a list of triangular facets.

    Parameters
    ----------
    facets:
        A list of facets, where a facet is a length 3 collection of 3 vectors,
        equivalently this can be a numpy array of shape `(n, 3, 3)`. These 3 vectors
        must define a triangle which make up the facet.
    normalize_area:
        This determines of the total surface area should be normalized to 1.
    """

    def __init__(self, facets: ArrayLike, normalize_area: bool = True):
        self._facets = np.array(facets)
        super().__init__(
            [self._normal(*v) for v in self.facets],
            [self._area(*v) for v in self.facets],
            normalize_area,
        )

    @staticmethod
    def _normal(vec1, vec2, vec3):
        """
        Given 3 vectors which define a triangle, compute the normal vector of the
        surface of that triangle. Where the normal vector is defined away from the
        center of the coordinate system if possible. The returned vector has length 1.
        """
        normal = np.cross(vec2 - vec1, vec3 - vec1)
        if np.dot(vec1, normal) < 0.0:
            normal *= -1.0
        return normal / np.linalg.norm(normal)

    @staticmethod
    def _area(vec1, vec2, vec3):
        """
        Given 3 vectors which define a triangle, compute the area of the triangle.
        """
        return np.linalg.norm(np.cross(vec2 - vec1, vec3 - vec1)) / 2

    @property
    def facets(self) -> np.ndarray:
        """A ``(n, 3, 3)`` array of all of the facet corners in this geometry."""
        return self._facets


class TriangleEllipsoid(TriangleFaceted):
    """
    A simple ellipsoid/sphere made up of triangle facets.

    This is a subclass of :class:`~kete.shape.TriangleFaceted` which
    constructs an approximate Ellipsoid geometry out of triangle facets.

    See https://arxiv.org/abs/1502.04816

    Parameters
    ----------
    n_div:
        The number of divisions.
    x_scale:
        Scaling parameter along the x-axis, all x-values are multiplied by this.
    y_scale:
        Scaling parameter along the y-axis, all y-values are multiplied by this.
    z_scale:
        Scaling parameter along the z-axis, all z-values are multiplied by this.

    Examples
    --------

    .. plot::
        :context: close-figs

            >>> import kete
            >>> import matplotlib.pyplot as plt
            >>> from mpl_toolkits.mplot3d.art3d import Poly3DCollection
            >>> geom = kete.shape.TriangleEllipsoid(6)
            >>> # Plot the results
            >>> plt.figure(dpi=150, figsize=(4, 4))
            >>> plt.subplot(111, projection="3d")
            >>> polygons = Poly3DCollection(geom.facets, edgecolor="black", lw=0.2)
            >>> plt.gca().add_collection3d(polygons)
            >>> plt.xlim(-1.1, 1.1)
            >>> plt.ylim(-1.1, 1.1)
            >>> plt.gca().set_zlim(-1.1, 1.1)

    Here is a triangle ellipsoid with ``n_div=6``.

    """

    @cache(128)  # type: ignore
    def __new__(
        cls, n_div: int = 6, x_scale: float = 1, y_scale: float = 1, z_scale: float = 1
    ):
        # This __new__ exists for the sole purpose of allowing cache to work
        return super().__new__(cls)

    def __init__(
        self, n_div: int = 6, x_scale: float = 1, y_scale: float = 1, z_scale: float = 1
    ):
        dth = np.pi / (2.0 * float(n_div))
        points = [[0.0, 0.0]]

        for i in range(1, n_div + 1):
            dphi = np.pi / (2.0 * float(i))
            for j in range(0, 4 * i):
                points.append([float(i) * dth, float(j) * dphi])
        for i in range(n_div - 1, 0, -1):
            dphi = np.pi / (2.0 * float(i))
            for j in range(0, 4 * i):
                points.append([np.pi - float(i) * dth, j * dphi])
        points.append([np.pi, 0.0])
        points = np.degrees(points).tolist()  # type: ignore

        vecs = []
        for point in points:
            vecs.append(np.array(Vector.from_lat_lon(90 - point[0], point[1])))

        # Create the vertix matrix
        matrix = []
        for i in range(0, n_div + 1):
            row = []
            for _ in range(0, 4 * i + 1):
                row.append(0)
            matrix.append(row)
        for i in range(n_div - 1, -1, -1):
            row = []
            for _ in range(0, 4 * i + 1):
                row.append(0)
            matrix.append(row)

        count = 0
        for i in range(1, n_div + 1):
            for j in range(0, 4 * i):
                count += 1
                matrix[i][j] = count
                if j == 0:
                    matrix[i][4 * i] = count
        for i in range(n_div - 1, 0, -1):
            for j in range(0, 4 * i):
                count += 1
                matrix[2 * n_div - i][j] = count
                if j == 0:
                    matrix[2 * n_div - i][4 * i] = count
        matrix[2 * n_div][0] = count + 1

        # Associates vertices to form facets
        facets = []
        # Creating facets for the northern hemisphere
        for j1 in range(1, n_div + 1):
            for j3 in range(1, 5):
                j0 = (j3 - 1) * j1
                vec1 = vecs[matrix[j1 - 1][j0 - (j3 - 1)]]
                vec2 = vecs[matrix[j1][j0]]
                vec3 = vecs[matrix[j1][j0 + 1]]
                facets.append([vec1, vec2, vec3])

                for j2 in range(j0 + 1, j0 + j1):
                    vec1 = vecs[matrix[j1][j2]]
                    vec2 = vecs[matrix[j1 - 1][j2 - (j3 - 1)]]
                    vec3 = vecs[matrix[j1 - 1][(j2 - 1) - (j3 - 1)]]
                    facets.append([vec1, vec2, vec3])

                    vec1 = vecs[matrix[j1][j2]]
                    vec2 = vecs[matrix[j1 - 1][j2 - (j3 - 1)]]
                    vec3 = vecs[matrix[j1][j2 + 1]]
                    facets.append([vec1, vec2, vec3])

        # Creating facets for the southern hemisphere
        for j1 in range(n_div + 1, 2 * n_div + 1):
            for j3 in range(1, 5):
                j0 = (j3 - 1) * (2 * n_div - j1)
                vec1 = vecs[matrix[j1][j0]]
                vec2 = vecs[matrix[j1 - 1][(j0 + 1) + (j3 - 1)]]
                vec3 = vecs[matrix[j1 - 1][j0 + (j3 - 1)]]
                facets.append([vec1, vec2, vec3])

                for j2 in range(j0 + 1, j0 + 2 * n_div - j1 + 1):
                    vec1 = vecs[matrix[j1][j2]]
                    vec2 = vecs[matrix[j1 - 1][j2 + (j3 - 1)]]
                    vec3 = vecs[matrix[j1][j2 - 1]]
                    facets.append([vec1, vec2, vec3])

                    vec1 = vecs[matrix[j1][j2]]
                    vec2 = vecs[matrix[j1 - 1][(j2 + 1) + (j3 - 1)]]
                    vec3 = vecs[matrix[j1 - 1][j2 + (j3 - 1)]]
                    facets.append([vec1, vec2, vec3])
        facets_arr = np.array(facets)
        facets_arr[:, :, 0] *= x_scale
        facets_arr[:, :, 1] *= y_scale
        facets_arr[:, :, 2] *= z_scale
        super().__init__(facets_arr)
