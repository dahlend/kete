"""
Orbit fitting and initial orbit determination.

This module provides tools for determining and refining orbits from
astronomical observations:

- **Initial Orbit Determination (IOD)** -- statistical ranging to produce
  candidate orbits from a handful of optical observations.
- **Differential Correction** -- batch least-squares with
  Levenberg--Marquardt damping and optional outlier rejection.
- **NUTS MCMC Sampling** -- No-U-Turn posterior sampling for short-arc
  orbit characterization where a Gaussian approximation is insufficient.
- **Lambert Solver** -- single-revolution Keplerian transfer between two
  position vectors.
"""

from __future__ import annotations

import numpy as np

from ._core import (
    Observation,
    OrbitFit,
    OrbitSamples,
    UncertainState,
    differential_correction,
    initial_orbit_determination,
    lambert,
    nuts_sample,
)

__all__ = [
    "Observation",
    "OrbitFit",
    "OrbitSamples",
    "UncertainState",
    "differential_correction",
    "initial_orbit_determination",
    "lambert",
    "mpc_obs_to_observations",
    "nuts_sample",
]


def mpc_obs_to_observations(
    mpc_obs: list,
    sigma_ra: float = 0.1,
    sigma_dec: float = 0.1,
) -> list[Observation]:
    """
    Convert a list of MPCObservation objects to fitting Observations.

    Only optical (RA/Dec) observations are supported. Each MPCObservation is
    converted to an ``Observation.optical`` with the observer state computed
    from the MPC observatory code (ground-based) or from the stored spacecraft
    position.

    Parameters
    ----------
    mpc_obs :
        List of ``MPCObservation`` objects (see :mod:`kete.mpc`).
    sigma_ra :
        Default 1-sigma RA uncertainty in arcseconds. The cos(dec) factor
        is applied automatically.
    sigma_dec :
        Default 1-sigma Dec uncertainty in arcseconds.

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
        fit = kete.fitting.differential_correction(initial_state, observations)
    """
    from . import spice
    from .vector import Frames, State

    observations = []
    for obs in mpc_obs:
        # Per-observation sigma in arcseconds (apply cos(dec) to RA).
        cos_dec = np.cos(np.radians(obs.dec))
        sig_ra = sigma_ra / max(cos_dec, 1e-6)
        sig_dec = sigma_dec

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
                ra=obs.ra,
                dec=obs.dec,
                sigma_ra=sig_ra,
                sigma_dec=sig_dec,
            )
        )

    return observations
