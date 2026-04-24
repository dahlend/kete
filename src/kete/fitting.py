"""
Orbit fitting and uncertainty estimation from observations.

This module provides tools for determining and refining orbits from
astronomical observations:

- **Initial Orbit Determination (IOD)** (:func:`initial_orbit_determination`) --
  find candidate orbits from a small number of observations using
  statistical ranging.  Returns scored candidates sorted best-first.
- **Orbit Fitting** (:func:`fit_orbit`) -- refine an orbit guess to best
  match the data, with automatic outlier rejection.  Produces a best-fit
  state and Gaussian uncertainty estimate (covariance).
- **MCMC Uncertainty Estimation** (:func:`fit_orbit_mcmc`) -- for short
  arcs where the Gaussian approximation is unreliable, sample the full
  range of plausible orbits consistent with the data.
"""

from __future__ import annotations

import numpy as np

from ._core import (
    Observation,
    OrbitFit,
    OrbitSamples,
    UncertainState,
    fit_orbit,
    fit_orbit_mcmc,
    initial_orbit_determination,
    lambert,
)

__all__ = [
    "Observation",
    "OrbitFit",
    "OrbitSamples",
    "UncertainState",
    "fit_orbit",
    "fit_orbit_mcmc",
    "initial_orbit_determination",
    "lambert",
    "mpc_obs_to_observations",
]


def mpc_obs_to_observations(
    mpc_obs: list,
    sigma_ra: float = 1.0,
    sigma_dec: float = 1.0,
) -> list[Observation]:
    """
    Convert a list of MPCObservation objects to fitting Observations.

    Only optical (RA/Dec) observations are supported. Each MPCObservation is
    converted to an ``Observation.optical`` with the observer state computed
    from the MPC observatory code (ground-based) or from the stored spacecraft
    position.

    Per-observatory bias corrections and uncertainties are applied when
    available from the precomputed residual table (see
    :func:`kete.mpc._parse_residuals`).  The median RA and Dec residuals are
    subtracted from the observed position and the per-axis standard deviations
    are used as sigmas.  When no table entry exists for an observatory code,
    the caller-supplied ``sigma_ra`` and ``sigma_dec`` are used instead.

    Parameters
    ----------
    mpc_obs :
        List of ``MPCObservation`` objects (see :mod:`kete.mpc`).
    sigma_ra :
        Fallback 1-sigma RA uncertainty in arcseconds when no per-observatory
        data is available.  Defaults to 1.
    sigma_dec :
        Fallback 1-sigma Dec uncertainty in arcseconds when no per-observatory
        data is available.  Defaults to 1.

    Returns
    -------
    list[Observation]
        One ``Observation.optical`` per input observation.

    Raises
    ------
    ValueError
        If an observation has a spacecraft flag but no valid position.

    Examples
    --------
    .. testcode::
        :skipif: True

        import kete

        lines = [...]  # 80-char MPC observation lines
        mpc_obs = kete.mpc.MPCObservation.from_lines(lines)
        observations = kete.fitting.mpc_obs_to_observations(mpc_obs)
        fit = kete.fitting.fit_orbit(initial_state, observations)
    """
    from . import spice
    from .mpc import _parse_residuals
    from .vector import Frames, State

    observations = []
    for obs in mpc_obs:
        ra = obs.ra
        dec = obs.dec

        # Apply per-observatory bias correction and use per-observatory sigmas
        # when available; otherwise fall back to the caller-supplied values.
        obs_errors = _parse_residuals(obs.obs_code)
        if obs_errors is not None:
            ra_med, s_ra, dec_med, s_dec = obs_errors
            ra -= ra_med / 3600.0
            dec -= dec_med / 3600.0
        else:
            s_ra = sigma_ra
            s_dec = sigma_dec

        # Apply cos(dec) factor to RA sigma.
        cos_dec = np.cos(np.radians(dec))
        s_ra = s_ra / max(cos_dec, 1e-6)

        # Determine observer state (SSB-centered, Equatorial).
        if obs.note2 in ("S", "T") and not any(np.isnan(obs.sun2sc)):
            # Spacecraft observation: sun2sc is the heliocentric position
            # in ecliptic coordinates. Build a State and re-center to SSB.
            sun_pos = spice.get_state("Sun", obs.jd, center=0).pos
            pos_ssb = np.array(obs.sun2sc) + np.array(list(sun_pos))
            observer = State(
                desig=obs.obs_code,
                jd=obs.jd,
                pos=pos_ssb,
                vel=[0.0, 0.0, 0.0],
                frame=Frames.Ecliptic,
                center_id=0,
            ).as_equatorial
        else:
            # Ground-based: look up from obs code, SSB-centered.
            observer = spice.mpc_code_to_ecliptic(
                obs.obs_code, obs.jd, center=0
            ).as_equatorial

        observations.append(
            Observation.optical(
                observer=observer,
                ra=ra,
                dec=dec,
                sigma_ra=s_ra,
                sigma_dec=s_dec,
            )
        )

    return observations
