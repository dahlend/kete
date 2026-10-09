# SPDX-FileCopyrightText: 2026 Dar Dahlen
# SPDX-License-Identifier: BSD-3-Clause

"""
Query tools for Gaia DR3 solar system object observations.

Provides :func:`fetch_gaia_observations` to retrieve optical astrometry from
the Gaia DR3 ``sso_observation`` table via TAP and return the results as
:class:`~kete.fitting.Observation` objects ready for orbit fitting.
"""

from __future__ import annotations

import logging

import numpy as np

from .. import spice
from .._core import Observation
from ..constants import SPEED_OF_LIGHT_AUDAY, SUN_GM
from ..tap import query_tap
from ..time import Time
from ..vector import Frames, State
from .common import _MIN_AXIS_SIGMA, _floor_error_ellipse

__all__ = ["fetch_gaia_observations"]

logger = logging.getLogger(__name__)


_GAIA_TABLE = "gaiadr3.sso_observation"

# The epoch column stores JD_TCB(Gaia) - J2010.0 in days.
# J2010.0 = JD 2455197.5 (TCB).  Adding this offset converts to JD_TCB.
_J2010_JD = 2455197.5

# Schwarzschild radius of the Sun, 2 GM / c^2, in AU.
_SUN_SCHWARZSCHILD_AU = 2.0 * SUN_GM / SPEED_OF_LIGHT_AUDAY**2

_COLUMNS = (
    "epoch",
    "epoch_err",
    "ra",
    "dec",
    "ra_error_random",
    "dec_error_random",
    "ra_error_systematic",
    "dec_error_systematic",
    "ra_dec_correlation_random",
    "ra_dec_correlation_systematic",
    "x_gaia_geocentric",
    "y_gaia_geocentric",
    "z_gaia_geocentric",
    "vx_gaia_geocentric",
    "vy_gaia_geocentric",
    "vz_gaia_geocentric",
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

    The Gaia spacecraft state is the table's geocentric position and velocity
    added to the Earth's state from the loaded SPICE kernels. The table's
    barycentric vectors are not used: their barycenter differs from that of
    the loaded planetary ephemeris by an amount that is large at Gaia's
    precision.

    Gaia positions are corrected for aberration but not for the solar deflection
    of light. Each is shifted by minus the deflection of a star in the same
    direction, which puts it in the convention of positions reduced against
    background stars, the convention the fit models.

    Observation epoch is the Gaia-centric TCB epoch stored in the table,
    converted to TDB via :class:`~kete.Time` with ``scaling='tcb'``.

    Positional uncertainties are the quadrature sum of the random and
    systematic components from the table (in mas, already multiplied by
    cos(dec) for the RA component), with both principal axes of the error
    ellipse limited to at least 0.1 mas.

    Results are cached via :func:`~kete.tap.query_tap`; pass
    ``update_cache=True`` to force a fresh query.

    Parameters
    ----------
    desig :
        Object identifier as recognized by Gaia DR3.  If the string parses
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

        observations = kete.observations.fetch_gaia_observations("Apophis")
        fit = kete.fitting.fit_orbit(initial_state, observations)
    """
    cols = ", ".join(_COLUMNS)

    try:
        number_mp = int(desig.strip())
        where = f"number_mp = {number_mp}"
    except ValueError:
        safe_desig = desig.replace("'", "''")
        where = f"denomination = '{safe_desig}'"

    query = f"SELECT {cols} FROM {_GAIA_TABLE} WHERE {where}"

    df = query_tap(query, service="GAIA", update_cache=update_cache)

    if df is None or len(df) == 0:
        return []

    observations = []
    for _, row in df.iterrows():
        outcome = row.get("astrometric_outcome_transit")
        if outcome is None or int(outcome) != 1:
            continue

        epoch = row.get("epoch")
        if epoch is None or (isinstance(epoch, float) and np.isnan(epoch)):
            continue
        jd = Time(float(epoch) + _J2010_JD, scaling="tcb").jd

        jd_err = row.get("epoch_err", 0.5 / 24 / 60 / 60) * 24 * 60 * 60

        ra = row.get("ra")
        dec = row.get("dec")
        if ra is None or dec is None:
            continue
        ra = float(ra)
        dec = float(dec)

        ra_rand = float(row.get("ra_error_random") or 0.0)
        ra_sys = float(row.get("ra_error_systematic") or 0.0)
        dec_rand = float(row.get("dec_error_random") or 0.0)
        dec_sys = float(row.get("dec_error_systematic") or 0.0)
        # Gaia DR3 exposes per-transit correlation between RA and Dec for
        # both the random and systematic error components.  Absent in older
        # schemas; fall back to zero.
        corr_rand = row.get("ra_dec_correlation_random")
        corr_sys = row.get("ra_dec_correlation_systematic")
        corr_rand = (
            float(corr_rand)
            if corr_rand is not None and np.isfinite(corr_rand)
            else 0.0
        )
        corr_sys = (
            float(corr_sys) if corr_sys is not None and np.isfinite(corr_sys) else 0.0
        )

        # Sum random and systematic covariance matrices (mas^2, sky
        # projection -- the random/systematic RA components already include
        # cos(dec)).  Extract effective sigmas and correlation from the
        # summed matrix.
        c_ra2 = ra_rand * ra_rand + ra_sys * ra_sys
        c_dec2 = dec_rand * dec_rand + dec_sys * dec_sys
        c_rd = corr_rand * ra_rand * dec_rand + corr_sys * ra_sys * dec_sys
        if c_ra2 <= 0.0 or c_dec2 <= 0.0:
            continue
        # Gaia DR3 ra_error_* fields are sky-plane (already include cos(dec)),
        # which matches the input convention of Observation.optical.
        sigma_ra = np.sqrt(c_ra2) / 1000.0  # mas -> arcsec
        sigma_dec = np.sqrt(c_dec2) / 1000.0
        # Effective correlation from the summed covariance: in mas units,
        # dividing numerator and denominator by 1e6 cancels the scaling.
        sigma_corr_eff = c_rd / np.sqrt(c_ra2 * c_dec2)
        sigma_ra, sigma_dec, sigma_corr_eff = _floor_error_ellipse(
            sigma_ra, sigma_dec, sigma_corr_eff, _MIN_AXIS_SIGMA
        )

        try:
            geo_pos = np.array([float(row[f"{c}_gaia_geocentric"]) for c in "xyz"])
            geo_vel = np.array([float(row[f"v{c}_gaia_geocentric"]) for c in "xyz"])
        except (KeyError, TypeError, ValueError):
            continue

        if not (np.all(np.isfinite(geo_pos)) and np.all(np.isfinite(geo_vel))):
            continue

        earth = spice.get_state("Earth", jd, center=0).as_equatorial
        gaia_pos = np.array(list(earth.pos)) + geo_pos
        observer = State(
            "Gaia",
            jd,
            list(gaia_pos),
            list(np.array(list(earth.vel)) + geo_vel),
            Frames.Equatorial,
            center_id=0,
        )
        sun_pos = np.array(list(spice.get_state("Sun", jd, center=0).as_equatorial.pos))
        ra, dec = _remove_star_deflection(ra, dec, gaia_pos - sun_pos)

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
                sigma_corr=sigma_corr_eff,
                band="G",
                mag=mag,
                time_sigma=jd_err,
            )
        )

    return observations


def _remove_star_deflection(
    ra: float, dec: float, observer_helio: np.ndarray
) -> tuple[float, float]:
    """Shift a direction by minus the solar deflection of a star at infinity.

    Gaia DR3 positions carry the full solar deflection of the light from the
    object. The fit models positions reduced against background stars. Those
    carry the deflection of the object less that of a star in the same
    direction. Removing the deflection of the star converts the first convention
    into the second. For a unit direction ``p`` and an observer at distance
    ``|e|`` from the Sun along ``e_hat``, the deflection of the star is
    ``2 GM / (c^2 |e|) * (e_hat - (p . e_hat) p) / (1 + p . e_hat)``.
    """
    ra_r, dec_r = np.radians(ra), np.radians(dec)
    p = np.array(
        [np.cos(dec_r) * np.cos(ra_r), np.cos(dec_r) * np.sin(ra_r), np.sin(dec_r)]
    )
    dist = np.linalg.norm(observer_helio)
    e = observer_helio / dist
    p_e = p @ e
    delta = _SUN_SCHWARZSCHILD_AU / dist * (e - p_e * p) / max(1.0 + p_e, 1e-9)
    q = p - delta
    q /= np.linalg.norm(q)
    return float(np.degrees(np.arctan2(q[1], q[0])) % 360.0), float(
        np.degrees(np.arcsin(q[2]))
    )
