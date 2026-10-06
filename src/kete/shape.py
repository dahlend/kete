# SPDX-FileCopyrightText: 2026 Dar Dahlen
# SPDX-License-Identifier: BSD-3-Clause

"""
Definitions of geometric objects: triangle shapes used by the thermal and reflected
light models, and the polyhedron and spherical harmonic fields used for gravity.
"""

from __future__ import annotations

import os

import numpy as np

from ._core import Polyhedron, SphericalHarmonics, TriangleEllipsoid

__all__ = ["Polyhedron", "SphericalHarmonics", "TriangleEllipsoid", "read_obj"]


def read_obj(path: str | os.PathLike) -> tuple[np.ndarray, np.ndarray]:
    """
    Read the vertices and triangular faces of a Wavefront OBJ file.

    Only ``v`` and ``f`` records are read; normals, texture coordinates and other
    records are ignored. Face indices may carry ``/``-separated texture and normal
    indices, which are dropped, and may be negative (relative to the end of the
    vertex list), as the format allows. Faces are returned 0-based, in the order
    and winding of the file.

    Parameters
    ----------
    path :
        Path to the OBJ file.

    Returns
    -------
    vertices :
        Vertex positions, shape ``(n, 3)``, in the units of the file.
    faces :
        Faces as 0-based vertex indices, shape ``(m, 3)``.

    Raises
    ------
    ValueError
        If a face is not a triangle or an index is out of range.
    """
    vert_rows: list[list[float]] = []
    face_rows: list[list[int]] = []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "v":
                vert_rows.append([float(x) for x in parts[1:4]])
            elif parts[0] == "f":
                idx = [int(p.split("/")[0]) for p in parts[1:]]
                if len(idx) != 3:
                    raise ValueError(
                        f"Only triangular faces are supported, found {len(idx)} "
                        f"vertices in: {line.strip()}"
                    )
                n = len(vert_rows)
                face_rows.append([i - 1 if i > 0 else n + i for i in idx])
    vertices = np.array(vert_rows, dtype=float).reshape(-1, 3)
    faces = np.array(face_rows, dtype=np.int64).reshape(-1, 3)
    if faces.size and (faces.min() < 0 or faces.max() >= len(vertices)):
        raise ValueError("A face index is out of range of the vertex list.")
    return vertices, faces
