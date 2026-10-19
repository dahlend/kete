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
detections becomes one two-body orbit about the Sun, a tracklet of several close
in time becomes the orbit that fits them best, and the orbits of one object
agree.

Nothing is assumed about the observer, the observing geometry, or the cadence:
there are no nights or fields, and tracklets are found from the detections
alone. Only bound orbits are found, and the span of the detections has to be
short against the orbital period of the objects searched for. All detections of
a unit must point within about 63 degrees of the same one of the six Equatorial
coordinate axes, a cosine of :data:`MIN_ALIGNMENT`.

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
- :func:`form_units` finds the units of linking, tracklets of detections close in
  time and pairs across gaps, which could be one object, as :class:`Units`.
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
- :func:`inject` makes detections of synthetic objects in fields of view, and
  :func:`score` tells which of them a search found.

Use :func:`search` when all the observations fit in memory at once. Use
:func:`search_windows` for a whole survey, and give it ``window`` for a long
span, so that each tile stays small. Observers far apart at the same time, such
as a site on the ground and a spacecraft, see one object in directions which
differ by a parallax. Pass ``parallax_margin=parallax_margin(separation,
min_depth)`` to :func:`search` or :func:`search_windows`, with the largest
separation of two observers at one time and the smallest depth searched; it
widens the pairs of detections and the halos of the tiles.

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

Measuring completeness: :func:`inject` makes the detections synthetic objects
would leave in the fields of view of a survey, with noise and a chance of being
missed. Searched together with the real observations, :func:`score` tells, for
each synthetic object, whether a confirmed object holds only its detections.
The fraction found over the objects deemed findable is the completeness of the
search on that survey. Clones of known objects, their orbits with the
longitude of the node turned by a few degrees, move as real objects near the
fields do without sharing their detections.

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
    # starting every 7, with observers up to an Earth diameter apart:
    margin = linking.parallax_margin(2 * kete.constants.EARTH_RADIUS_AU, 0.5)
    found = linking.search_windows(
        obs, stations=stations, window=7.0, parallax_margin=margin
    )

    # The steps of one pass of search, by hand. The reference time is the
    # epoch of the origin, and must lie within the span of the detections:
    sights = linking.Sights.from_observations(obs)
    units = linking.form_units(
        sights, max_gap=1.5, max_rate=1.5, tracklet_span=0.125
    )
    times = sorted(sights.time)
    reference_jd = times[len(times) // 2]
    origin = kete.spice.get_state("Earth", reference_jd)
    grid = linking.hypothesis_grid(origin, min_depth=0.8, max_depth=4.0)
    candidates = linking.link(sights, units, grid)
    found = linking.confirm(
        obs, candidates, min_sights=4, max_rms=2.0, group_span=0.1, max_gap=1.5
    )

    # Measuring completeness, with clones of known objects:
    import numpy as np

    known = [...]  # kete States of known objects
    fields = [...]  # the fields of view of the survey, such as kete.fov.ConeFOV
    rng = np.random.default_rng(1)
    clones = []
    for k, state in enumerate(known):
        e = kete.CometElements.from_state(state)
        turn = rng.uniform(1.0, 3.0) * rng.choice([-1, 1])
        clones.append(
            kete.CometElements(
                f"clone{k}",
                e.epoch,
                e.eccentricity,
                e.inclination,
                e.peri_dist,
                e.peri_arg,
                e.peri_time,
                (e.lon_of_ascending + turn) % 360.0,
            ).state
        )
    injected, labels = linking.inject(clones, fields, sigma=0.2, detect=0.9, seed=2)
    everything = obs + injected
    labels = np.concatenate([np.full(len(obs), -1), labels])
    found = linking.search(everything)
    table = linking.score(everything, found, labels)
    # Completeness over the clones seen six times or more over two days or more:
    findable = table[(table.detections >= 6) & (table.span >= 2.0)]
    print(findable.found.mean(), table.attrs["mixed"])
"""

from collections import defaultdict

import numpy as np
import pandas as pd

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
from .fov import fov_state_check, fov_static_check
from .orbit_fitting import Observation
from .vector import State

DEPTH_STEP: float = _core.DEPTH_STEP
"""Default step of :func:`hypothesis_grid` in the natural log of the depth."""

RATE_STEP: float = _core.RATE_STEP
"""Default step of :func:`hypothesis_grid` in depth rate over depth, per day."""

MAX_HYPOTHESES: int = _core.MAX_HYPOTHESES
"""Most hypotheses :func:`hypothesis_grid` builds."""

MAX_UNITS: int = _core.MAX_UNITS
"""Most units :func:`link` and each pass of :func:`search` and
:func:`search_windows` take. Each thread of :func:`link` holds every unit
solved under its hypothesis at once, several hundred bytes each, so this keeps
the memory of linking to a few gigabytes on a machine of eight to sixteen
threads. More units than this mostly come from pairs of false detections; the
remedies are fewer false detections, a smaller ``max_rate``, a larger
``min_tracklet`` (see :func:`form_units`), or smaller tiles with
:func:`search_windows`, keeping ``lone_pairs=False``."""

MIN_ALIGNMENT: float = _core.MIN_ALIGNMENT
"""Smallest cosine between a detection and the direction of a hypothesis it is
solved under, an angle of about 63 degrees."""


def inject(states, fovs, sigma, detect=1.0, seed=None, non_gravs=None):
    """
    Detections of synthetic objects in fields of view, with noise.

    :func:`~kete.fov_state_check` finds the fields each object falls in, with
    n-body propagation and light time, and each object makes one detection in
    each field it falls in. Fields of one observer at one time, such as the
    detectors of one exposure, give an object one detection if they overlap,
    from the first of them in ``fovs`` which holds it. The detection is the
    direction from the observer of the field to the object, with Gaussian noise
    added on the sky, and is kept with probability ``detect``.

    Parameters
    ----------
    states :
        List of :class:`~kete.State`, the synthetic objects, all at one epoch
        and about one center, such as states propagated together by
        :func:`~kete.propagate_n_body`.
    fovs :
        List of fields of view, of any type :func:`~kete.fov_state_check` takes.
        Each holds its observer, which is the observer of its detections.
    sigma :
        Astrometric uncertainty in arcseconds, one value, or one per field of
        view. It is the uncertainty on the sky along both RA and Dec, as the
        ``sigma_ra`` and ``sigma_dec`` of
        :meth:`~kete.orbit_fitting.Observation.optical`, which states the RA
        uncertainty on the sky: the RA coordinate itself is moved by
        ``sigma / cos(dec)``. Noise of that size is added along each of the two
        directions, and the observations carry it as their uncertainty.
    detect :
        Probability that an object in a field is detected, a number from 0 to 1,
        or a function ``detect(fov, state)`` returning that probability, such as
        from the brightness of the object. ``fov`` is the element of ``fovs``,
        and ``state`` is the Sun-centered state of the object at the time the
        light seen left it, as :func:`~kete.fov_state_check` returns it, with
        the designation of the element of ``states``.
    seed :
        Seed of the :func:`numpy.random.default_rng` which draws the detections
        and noise. The same seed gives the same observations.
    non_gravs :
        Optional list of one :class:`~kete.propagation.NonGravModel` or ``None``
        per state, passed to :func:`~kete.fov_state_check`.

    Returns
    -------
    observations :
        List of optical :class:`~kete.orbit_fitting.Observation`, ordered by
        field, then by object. Their times are exact, with a ``time_sigma`` of
        zero.
    labels :
        Numpy integer array, the index into ``states`` of the object of each
        observation.
    """
    states, fovs = list(states), list(fovs)
    if len({(s.jd, s.center_id) for s in states}) > 1:
        raise ValueError(
            "states must all be at one epoch and about one center, propagate them "
            "to one epoch first."
        )
    if np.ndim(sigma) == 0:
        sigmas = np.full(len(fovs), float(sigma))
    else:
        sigmas = np.asarray(sigma, dtype=float)
        if sigmas.shape != (len(fovs),):
            raise ValueError(
                f"sigma has {sigmas.size} values for {len(fovs)} fields of view."
            )
    bad = ~(np.isfinite(sigmas) & (sigmas > 0))
    if bad.any():
        raise ValueError(
            f"sigma must be positive and finite, got {sigmas[bad][0]} for field "
            f"{np.flatnonzero(bad)[0]}."
        )
    if not callable(detect):
        detect = float(detect)
        if not 0.0 <= detect <= 1.0:
            raise ValueError(f"detect must be from 0 to 1, got {detect}.")
    rng = np.random.default_rng(seed)

    # Each state is named by its index, so each state seen tells its object, and
    # each field is found by its observer.
    named = [
        State(str(k), s.jd, list(s.pos), list(s.vel), s.frame, s.center_id)
        for k, s in enumerate(states)
    ]

    def observer_key(fov):
        observer = fov.observer
        return observer.jd, observer.center_id, tuple(observer.pos)

    fields = defaultdict(list)
    for i, fov in enumerate(fovs):
        fields[observer_key(fov)].append(i)
    fields = dict(fields)

    # The direction and state of each object in each field it falls in. Fields
    # with the same observer at the same time are told apart by their shape.
    hits = {}
    visible = fov_state_check(named, fovs, non_gravs=non_gravs) if named else []
    for seen in visible:
        vecs = seen.obs_vecs
        same = fields[observer_key(seen.fov)]
        for i in same:
            inside = range(len(vecs))
            if len(same) > 1:
                inside = {
                    k for idx, _ in fov_static_check(vecs, [fovs[i]]) for k in idx
                }
            for k in inside:
                state = seen.states[k]
                hits.setdefault((i, int(state.desig)), (vecs[k], state))
    # One detection of an object per observer and time, in the first field.
    first = {}
    for i, obj in sorted(hits):
        first.setdefault((observer_key(fovs[i]), obj), i)
    order = sorted((i, obj) for (_, obj), i in first.items())

    if callable(detect):
        chance = []
        for i, obj in order:
            s = hits[i, obj][1]
            state = State(
                states[obj].desig, s.jd, list(s.pos), list(s.vel), s.frame, s.center_id
            )
            chance.append(float(detect(fovs[i], state)))
        chance = np.array(chance)
        bad = ~((chance >= 0.0) & (chance <= 1.0))
        if bad.any():
            raise ValueError(
                f"detect must return a probability from 0 to 1, got {chance[bad][0]}."
            )
    else:
        chance = np.full(len(order), detect)
    kept = rng.random(len(order)) < chance
    noise = rng.normal(size=(len(order), 2))

    observations, labels = [], []
    for (i, obj), keep, (n_ra, n_dec) in zip(order, kept, noise):
        if not keep:
            continue
        u = np.array(list(hits[i, obj][0].as_equatorial))
        u /= np.linalg.norm(u)
        ra, dec = np.arctan2(u[1], u[0]), np.arcsin(np.clip(u[2], -1.0, 1.0))
        # The direction moved along the unit vectors of RA and Dec.
        east = np.array([-np.sin(ra), np.cos(ra), 0.0])
        north = np.array(
            [-np.sin(dec) * np.cos(ra), -np.sin(dec) * np.sin(ra), np.cos(dec)]
        )
        u += np.radians(sigmas[i] / 3600.0) * (n_ra * east + n_dec * north)
        noisy_ra = np.degrees(np.arctan2(u[1], u[0])) % 360.0
        noisy_dec = np.degrees(np.arctan2(u[2], np.hypot(u[0], u[1])))
        observations.append(
            Observation.optical(
                fovs[i].observer,
                noisy_ra,
                noisy_dec,
                sigmas[i],
                sigmas[i],
                time_sigma=0.0,
            )
        )
        labels.append(obj)
    return observations, np.array(labels, dtype=np.int64)


def score(observations, found, labels):
    """
    Which synthetic objects a search found, from the labels of the observations.

    An object is found when some confirmed object holds only its detections.
    Its detections may also be split over several such pieces, or sit inside
    confirmed objects which mix them with other detections.

    Completeness is left to the caller, over the objects deemed findable, such
    as ``table[table.detections >= 6].found.mean()``.

    Parameters
    ----------
    observations :
        The observations given to the search.
    found :
        The :class:`Confirmed` objects of :func:`search`,
        :func:`search_windows` or :func:`confirm` over ``observations``.
    labels :
        One integer per observation: the index of its synthetic object, such as
        the labels of :func:`inject`, or -1 for every other observation, real
        or unknown. Any negative label is another observation.

    Returns
    -------
    pandas.DataFrame
        One row per synthetic object with detections, in order of label, with
        columns:

        - ``object``: its label.
        - ``detections``: its number of detections.
        - ``span``: days from its first detection to its last.
        - ``found``: whether a confirmed object holds only its detections.
        - ``pieces``: the number of confirmed objects which hold only its
          detections.
        - ``largest``: the detections of the largest of them, 0 if none.
        - ``in_mixed``: its detections in confirmed objects which also hold
          detections of other labels, -1 included.

        ``attrs["mixed"]`` is the number of confirmed objects which mix labels
        and hold a detection of a synthetic object.
    """
    labels = np.asarray(labels)
    if labels.ndim != 1 or len(labels) != len(observations):
        raise ValueError(
            f"labels must hold one label per observation, got shape {labels.shape} "
            f"for {len(observations)} observations."
        )
    if labels.size == 0:
        labels = labels.astype(np.int64)
    if labels.dtype.kind not in "iu":
        raise ValueError(f"labels must be integers, got {labels.dtype}.")
    times = np.array([o.epoch.jd for o in observations], dtype=float)
    synthetic = labels >= 0
    objects, row, detections = np.unique(
        labels[synthetic], return_inverse=True, return_counts=True
    )
    first = np.full(len(objects), np.inf)
    last = np.full(len(objects), -np.inf)
    np.minimum.at(first, row, times[synthetic])
    np.maximum.at(last, row, times[synthetic])
    index = {int(o): k for k, o in enumerate(objects)}

    pieces = np.zeros(len(objects), dtype=np.int64)
    largest = np.zeros(len(objects), dtype=np.int64)
    in_mixed = np.zeros(len(objects), dtype=np.int64)
    mixed = 0
    for k, confirmed in enumerate(found):
        rows = np.asarray(confirmed.sights, dtype=np.int64)
        if rows.size and (rows.min() < 0 or rows.max() >= len(labels)):
            raise ValueError(f"confirmed object {k} indexes past the observations.")
        own = labels[rows]
        kinds = np.unique(own)
        if len(kinds) == 1 and kinds[0] >= 0:
            j = index[int(kinds[0])]
            pieces[j] += 1
            largest[j] = max(largest[j], len(rows))
        elif len(kinds) > 1 and kinds[-1] >= 0:
            mixed += 1
            for label, count in zip(*np.unique(own[own >= 0], return_counts=True)):
                in_mixed[index[int(label)]] += count

    table = pd.DataFrame(
        {
            "object": objects.astype(np.int64),
            "detections": detections.astype(np.int64),
            "span": last - first,
            "found": pieces > 0,
            "pieces": pieces,
            "largest": largest,
            "in_mixed": in_mixed,
        }
    )
    table.attrs["mixed"] = mixed
    return table


__all__ = [
    "DEPTH_STEP",
    "MAX_HYPOTHESES",
    "MAX_UNITS",
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
    "inject",
    "link",
    "merge_pieces",
    "parallax_margin",
    "score",
    "search",
    "search_windows",
    "tiles",
]
