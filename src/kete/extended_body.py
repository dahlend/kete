"""
Extended-gravity (polyhedron) body machinery.

This module exposes :class:`Polyhedron`, :class:`RotationModel`,
:class:`ExtendedBody` and the body-centric integrator wrapper
:func:`propagate_near_body` for working with non-point-mass gravity in
the immediate vicinity of small bodies (asteroids, comets, satellites
of larger bodies).

Units follow the kete convention: positions in AU, gravitational
parameters in AU^3 / day^2, angles in degrees, and time as a
:class:`kete.Time` (TDB-scaled).  Body-natural internal units are
handled inside the bindings.
"""

from ._core import (
    ExtendedBody,
    Polyhedron,
    RotationModel,
    body_acceleration,
    is_inside_proximity,
    propagate_near_body,
)

__all__ = [
    "ExtendedBody",
    "Polyhedron",
    "RotationModel",
    "body_acceleration",
    "is_inside_proximity",
    "propagate_near_body",
]
