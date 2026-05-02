Extended Body Gravity
=====================

The :mod:`kete.extended_body` submodule provides polyhedron gravity
models and a body-centric integrator for working with non-point-mass
gravity in the immediate vicinity of small bodies (asteroids, comets,
satellites of larger planets).  It is intended for trajectory studies
where the spacecraft is close enough to the central body that the
deviation from a point-mass field matters - typically within a few
body radii.

.. note::

   The dominant uncertainty in any extended-gravity calculation is
   nearly always the underlying shape and density model, not the
   gravity solver.  See "Accuracy and limitations" below for guidance
   on which distance regimes give predictive answers and which give
   only sensitivity studies.

Units
-----

The Python API follows the kete convention:

- Positions in AU (astronomical units).
- Gravitational parameters in :math:`\text{AU}^3 / \text{day}^2`.
- Times as :class:`kete.Time` (TDB-scaled internally) or JD floats.
- Angles in degrees, angular rates in degrees per day.

The body-natural unit system used internally by the integrator
(length scale equal to the body's bounding radius, GM scale equal
to the aggregate :math:`G M`) is constructed automatically and never
exposed to the user.

Gravity model
-------------

The closed-form Werner-Scheeres expressions for polyhedron gravity
are used.  These are mathematically exact for any closed orientable
triangular mesh - convex or non-convex - so contact binaries with
self-consistent mesh winding are supported in principle.  See
:meth:`Polyhedron.contains` for the Laplacian-sign interior check
that is also exact for non-convex meshes.

Accuracy and limitations
------------------------

- More than one body radius above the surface, the polyhedron field
  is reliable at the few-percent level, with the residual dominated
  by total-mass uncertainty.
- Between 0.1 and 1 body radius, the field shape is qualitatively
  correct but quantitative accuracy depends on the assumed density
  distribution.
- Below 0.1 body radius (e.g. inside a contact-binary neck region),
  results are scenario-dependent and should be treated as
  sensitivity studies, not predictions.  Use multi-component
  :class:`ExtendedBody` with per-lobe densities to bracket the
  uncertainty.
- Near or on the surface this model is outside its intended use;
  consider dedicated discrete-element or N-body codes such as
  ``pkdgrav`` or Project Chrono.

Numerics notes
--------------

Internally the body-centric integrator works in a body-natural unit
system that is constructed automatically from the ``length_au`` and
aggregate ``gm`` you pass to :class:`ExtendedBody`:

- Lengths are scaled by ``length_au`` (typically the bounding radius
  of the largest mesh component).
- Time is left in TDB days, which keeps the integrator's step-size
  controller (Radau, MIN_STEP = 5e-4 days) consistent across
  proximity-regime and heliocentric segments.
- Accelerations are evaluated as a sum of two pieces:

  1. The direct body-fixed Werner-Scheeres acceleration from each
     polyhedron component, evaluated at the body-fixed field point
     and rotated back into the inertial frame using the
     :class:`RotationModel`.
  2. The *tidal* acceleration from each perturber, written as

     .. math::

        \\mathbf{a}_{\\text{tidal}}
        = \\sum_p G M_p \\left(
          \\frac{\\mathbf{r}_{p \\rightarrow s} - \\mathbf{r}}
                {|\\mathbf{r}_{p \\rightarrow s} - \\mathbf{r}|^{3}}
        - \\frac{\\mathbf{r}_{p \\rightarrow s}}
                {|\\mathbf{r}_{p \\rightarrow s}|^{3}}
        \\right)

     so that the leading point-mass term cancels (the body's own
     SPK ephemeris already accounts for the perturber's pull on
     the body).  Only the differential (tidal) part enters the
     body-centric equations of motion.

The body's own SPK ephemeris (queried via
:func:`propagate_near_body`'s ``body_naif_id`` argument) provides
the inertial position of the body at every integrator evaluation.
This means the integrator never has to model the body's
heliocentric trajectory itself; it is consumed as a forced
external state.

Non-gravitational forces
------------------------

:func:`propagate_near_body` accepts an optional ``non_grav``
argument of type :class:`kete.propagation.NonGravModel`.  This is
the same model family used by the heliocentric N-body propagator
and includes the JPL comet ``A1``/``A2``/``A3`` parameterisation,
dust SRP plus Poynting-Robertson drag, and the Farnocchia 2025
physical radiation model.  Internally the body-fixed particle
state is rotated into the inertial Sun-relative frame at every
integrator evaluation (with the rigid-body ``omega x r`` velocity
correction applied), the model is queried in its native
heliocentric units, and the resulting acceleration is rotated
back into body-fixed coordinates and rescaled into body-natural
units.

Reference
---------

.. automodule:: kete.extended_body
   :members:
   :inherited-members:
