# SPDX-FileCopyrightText: 2026 Dar Dahlen
# SPDX-License-Identifier: BSD-3-Clause

"""
Link detections of unknown moving objects into candidate objects, and confirm
them with orbit fits.

Two detections of one object give its direction on the sky at two times, but not
the two numbers that place it along the lines of sight. A hypothesis asserts
those two numbers for every object at once, as the depth of the object along a
fixed direction and its rate at a reference time. The directions of hypotheses
are the six Equatorial coordinate axes, both ways. Under a hypothesis a pair of
detections becomes one two-body orbit about the Sun, a run of several close in
time becomes the orbit that fits them best, and the orbits of one object agree.

Nothing is assumed about the observer, the observing geometry, or the cadence:
there are no nights, tracklets, or fields. Only bound orbits are found, and the
span of the detections has to be short against the orbital period of the objects
searched for. All detections of a unit must point within about 63 degrees of the
same one of the six Equatorial coordinate axes, a cosine of
:data:`MIN_ALIGNMENT`.

The observations are :class:`~kete.orbit_fitting.Observation` objects, such as
from :func:`~kete.orbit_fitting.mpc_obs_to_observations`. Pass it
``apply_over_obs_reweight=False``: its default inflates the uncertainties of
an observatory's observations on a night by ``sqrt(n / 4)`` when it made
``n > 4`` of them, counted over the whole list. That suits fitting the
observations of one object, but over the detections of a survey it inflates them
many times over, which loosens linking and confirmation and the scaling of the
uncertainties of each station. It also skips records it cannot convert, so labels of the
observations, such as the ``stations`` of :func:`search`, have to be built from
the observations it returns rather than from the records.

The steps:

- :class:`Sights` holds the detections as lines of sight, built from
  observations with :meth:`Sights.from_observations`, or from arrays.
- :func:`form_units` finds the units of linking, runs of detections close in time
  and pairs across gaps, which could be one object, as :class:`Units`.
- :func:`hypothesis_grid` lists the hypotheses of a search about one origin, as
  :class:`Hypotheses`.
- :func:`link` groups the units into :class:`Candidate` objects.
- :func:`confirm` fits each candidate with
  :func:`~kete.orbit_fitting.fit_orbit` and keeps the ones that fit, without
  overlaps.
- :func:`search` runs all of these from observations, in passes over a coarse
  and then a finer grid, each with its own rate limit and each over the
  observations no earlier pass confirmed.
- :func:`search_windows` runs the passes of :func:`search` one tile of the sky
  at a time, cut by :func:`tiles`, optionally in windows of time.
- :func:`merge_pieces` merges confirmed objects which are pieces of one object,
  which both searches do last.

Use :func:`search` when all the observations fit in memory at once. Use
:func:`search_windows` for a whole survey, and give it ``window`` for a long
span, so that each tile stays small. Observers far apart at the same time, such
as sites an Earth diameter apart, see one object in directions which differ by a
parallax. For :func:`search_windows` pass
``parallax_margin=parallax_margin(separation, min_depth)``, with the largest
separation of two observers at one time and the smallest depth searched. In
:func:`search` the ``max_rate`` of each pass has to cover that parallax.

Detections are referred to by their row index everywhere. Only
:meth:`Sights.from_observations`, :func:`hypothesis_grid`, :func:`confirm`,
:func:`merge_pieces`, :func:`search` and :func:`search_windows` use the loaded
SPICE kernels.

Units and frames: times are JD in the TDB scale, also taken as :class:`~kete.Time`,
positions are in AU with respect to the Sun, rates of motion on the sky are in
degrees per day, and astrometric uncertainties and the grid allowance of
:func:`link` are in arcseconds. The arrays of :class:`Sights` are in the frame it
was given, Ecliptic by default. The directions of hypotheses and the states of
hypotheses, candidates and fits are kete :class:`~kete.Vector` and
:class:`~kete.State` objects, which display in kete's default frame, Ecliptic,
as other kete vectors and states do; ``as_equatorial`` and the like give them in
another frame.

.. code-block:: python

    import kete
    from kete import linking

    lines = [...]  # 80-column MPC observation lines
    records = kete.orbit_fitting.MPCObservation.from_lines(lines)
    obs = kete.orbit_fitting.mpc_obs_to_observations(
        records, apply_over_obs_reweight=False
    )

    found = linking.search(obs)
    for obj in found:
        rows = obj.sights  # row indices into obs
        state = obj.fit.state  # the fit orbit, at the epoch of the fit
        print(len(rows), obj.fit.rms, state.jd)

    # With the observatory code of each observation, confirmation scales the
    # uncertainties of each observatory to the scatter of its confirmed objects.
    # mpc_obs_to_observations names the observer of each observation after its
    # code, which keeps the labels in line with the observations it returns:
    stations = [o.observer.desig for o in obs]
    found = linking.search(obs, stations=stations)

    # A whole survey, one tile of the sky at a time, in windows of 14 days
    # starting every 7, with sites up to an Earth diameter apart:
    margin = linking.parallax_margin(2 * kete.constants.EARTH_RADIUS_AU, 0.5)
    found = linking.search_windows(
        obs, stations=stations, window=7.0, parallax_margin=margin
    )

    # The steps of one pass of search, by hand. The reference time is the
    # epoch of the origin, and must lie within the span of the detections:
    sights = linking.Sights.from_observations(obs)
    units = linking.form_units(
        sights, min_gap=0.5, max_gap=1.5, max_rate=1.5, run_span=0.125
    )
    times = sorted(sights.time)
    reference_jd = times[len(times) // 2]
    origin = kete.spice.get_state("Earth", reference_jd)
    grid = linking.hypothesis_grid(origin, min_depth=0.8, max_depth=4.0)
    candidates = linking.link(sights, units, grid)
    found = linking.confirm(
        obs, candidates, min_sights=4, max_rms=2.0, group_span=0.1, max_gap=1.5
    )
"""

from . import _core
from ._core import (
    Candidate,
    Confirmed,
    Hypotheses,
    Hypothesis,
    Sights,
    Units,
    confirm,
    form_units,
    hypothesis_grid,
    link,
    merge_pieces,
    parallax_margin,
    search,
    search_windows,
    tiles,
)

DEPTH_STEP: float = _core.DEPTH_STEP
"""Default step of :func:`hypothesis_grid` in the natural log of the depth."""

RATE_STEP: float = _core.RATE_STEP
"""Default step of :func:`hypothesis_grid` in depth rate over depth, per day."""

MAX_HYPOTHESES: int = _core.MAX_HYPOTHESES
"""Most hypotheses :func:`hypothesis_grid` builds."""

MIN_ALIGNMENT: float = _core.MIN_ALIGNMENT
"""Smallest cosine between a detection and the direction of a hypothesis it is
solved under, an angle of about 63 degrees."""

__all__ = [
    "DEPTH_STEP",
    "MAX_HYPOTHESES",
    "MIN_ALIGNMENT",
    "RATE_STEP",
    "Candidate",
    "Confirmed",
    "Hypotheses",
    "Hypothesis",
    "Sights",
    "Units",
    "confirm",
    "form_units",
    "hypothesis_grid",
    "link",
    "merge_pieces",
    "parallax_margin",
    "search",
    "search_windows",
    "tiles",
]
