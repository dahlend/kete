"""
Query tools for Gaia DR3 solar system object observations.

Provides :func:`fetch_gaia_observations` to retrieve optical astrometry from
the Gaia DR3 ``sso_observation`` table via TAP and return the results as
:class:`~kete.fitting.Observation` objects ready for orbit fitting.
"""

from __future__ import annotations

import logging

import numpy as np

from .fitting import Observation
from .tap import query_tap
from .time import Time
from .vector import Frames, State

__all__ = ["fetch_gaia_observations"]

logger = logging.getLogger(__name__)

# Gaia DR3 sso_observation table identifier.
_GAIA_TABLE = "gaiadr3.sso_observation"

# The epoch column stores JD_TCB(Gaia) - J2010.0 in days.
# J2010.0 = JD 2455197.5 (TCB).  Adding this offset converts to JD_TCB.
_J2010_JD = 2455197.5

# Columns retrieved from the table.
_COLUMNS = (
    "epoch",
    "ra",
    "dec",
    "ra_error_random",
    "dec_error_random",
    "ra_error_systematic",
    "dec_error_systematic",
    "x_gaia",
    "y_gaia",
    "z_gaia",
    "vx_gaia",
    "vy_gaia",
    "vz_gaia",
    "g_mag",
    "astrometric_outcome_transit",
)


def fetch_gaia_observations(
    desig: str,
    update_cache: bool = False,
) -> list[Observation]:
    """
    Fetch Gaia DR3 solar system object observations and convert to
    fitting Observations.

    Queries the ``gaiadr3.sso_observation`` table via the Gaia TAP service
    for the given object and returns one :class:`~kete.fitting.Observation`
    per accepted astrometric transit.  Only transits with
    ``astrometric_outcome_transit == 1`` (good positions) are returned.

    The Gaia spacecraft position and velocity are taken directly from the
    table (barycentric ICRS, SSB-centered) so no SPICE lookup is needed.

    Observation epoch is the Gaia-centric TCB epoch stored in the table,
    converted to TDB via :class:`~kete.Time` with ``scaling='tcb'``.

    Positional uncertainties are the quadrature sum of the random and
    systematic components from the table (in mas, already multiplied by
    cos(dec) for the RA component).

    Results are cached via :func:`~kete.tap.query_tap`; pass
    ``update_cache=True`` to force a fresh query.

    Parameters
    ----------
    desig :
        Object identifier as recognised by Gaia DR3.  If the string parses
        as an integer it is matched against the ``number_mp`` column
        (recommended for numbered minor planets); otherwise it is matched
        against the ``denomination`` column (e.g. ``"Apophis"``,
        ``"1999 RQ36"``).
    update_cache :
        If ``True``, discard any cached result and re-query the TAP service.

    Returns
    -------
    list[Observation]
        One ``Observation.optical`` per accepted transit observation.

    Examples
    --------
    .. testcode::
        :skipif: True

        import kete

        observations = kete.gaia.fetch_gaia_observations("Apophis")
        fit = kete.fitting.fit_orbit(initial_state, observations)
    """
    cols = ", ".join(_COLUMNS)

    # Build the WHERE clause.  Use number_mp for numeric designations to
    # avoid ambiguity with denomination strings.
    try:
        number_mp = int(desig.strip())
        where = f"number_mp = {number_mp}"
    except ValueError:
        # Sanitize: escape any single-quotes in the name.
        safe_desig = desig.replace("'", "''")
        where = f"denomination = '{safe_desig}'"

    query = f"SELECT {cols} FROM {_GAIA_TABLE} WHERE {where}"

    df = query_tap(query, service="GAIA", update_cache=update_cache)

    if df is None or len(df) == 0:
        return []

    observations = []
    for _, row in df.iterrows():
        # Only keep transits with a clean astrometric solution.
        outcome = row.get("astrometric_outcome_transit")
        if outcome is None or int(outcome) != 1:
            continue

        epoch = row.get("epoch")
        if epoch is None or (isinstance(epoch, float) and np.isnan(epoch)):
            continue
        jd = Time(float(epoch) + _J2010_JD, scaling="tcb").jd

        ra = row.get("ra")
        dec = row.get("dec")
        if ra is None or dec is None:
            continue
        ra = float(ra)
        dec = float(dec)

        # Combine random and systematic errors in quadrature.
        # The ra_error_* columns are in mas and already multiplied by
        # cos(dec); dec_error_* columns are in mas (no cos factor).
        ra_rand = float(row.get("ra_error_random") or 0.0)
        ra_sys = float(row.get("ra_error_systematic") or 0.0)
        dec_rand = float(row.get("dec_error_random") or 0.0)
        dec_sys = float(row.get("dec_error_systematic") or 0.0)

        # Total sky-projected RA uncertainty in arcseconds (includes cos(dec)).
        sigma_ra_sky = np.hypot(ra_rand, ra_sys) / 1000.0
        sigma_dec = np.hypot(dec_rand, dec_sys) / 1000.0

        if sigma_ra_sky <= 0.0 or sigma_dec <= 0.0:
            continue

        # Observation.optical expects sigma_ra in arcseconds with the
        # cos(dec) factor removed (i.e. uncertainty in the RA coordinate).
        cos_dec = np.cos(np.radians(dec))
        sigma_ra = sigma_ra_sky / max(cos_dec, 1e-6)

        # Gaia position and velocity: barycentric ICRS (Equatorial J2000),
        # SSB-centered, in AU and AU/day.
        try:
            x = float(row["x_gaia"])
            y = float(row["y_gaia"])
            z = float(row["z_gaia"])
            vx = float(row["vx_gaia"])
            vy = float(row["vy_gaia"])
            vz = float(row["vz_gaia"])
        except (KeyError, TypeError, ValueError):
            continue

        if any(np.isnan(v) for v in (x, y, z, vx, vy, vz)):
            continue

        observer = State(
            "Gaia",
            jd,
            [x, y, z],
            [vx, vy, vz],
            Frames.Equatorial,
            center_id=0,
        )

        try:
            mag = float(row.get("g_mag") or float("nan"))
        except (TypeError, ValueError):
            mag = float("nan")

        observations.append(
            Observation.optical(
                observer=observer,
                ra=ra,
                dec=dec,
                sigma_ra=sigma_ra,
                sigma_dec=sigma_dec,
                band="G",
                mag=mag,
            )
        )

    return observations
