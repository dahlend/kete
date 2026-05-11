"""
Orbit fitting and observation ingestion.

Fitting tools:

- **Initial Orbit Determination (IOD)** (:func:`initial_orbit_determination`) --
  find candidate orbits from a small number of observations using
  statistical ranging.  Returns scored candidates sorted best-first.
- **Orbit Fitting** (:func:`fit_orbit`) -- refine an orbit guess to best
  match the data, with automatic outlier rejection.  Produces a best-fit
  state and Gaussian uncertainty estimate (covariance).
- **MCMC Uncertainty Estimation** (:func:`fit_orbit_mcmc`) -- for short
  arcs where the Gaussian approximation is unreliable, sample the full
  range of plausible orbits consistent with the data.

Observation ingestion:

- **MPC legacy format** (:func:`mpc_obs_to_observations`, :class:`MPCObservation`) --
  parse the MPC 80-character fixed-width format and convert to Observations.
- **MPC ADES API** (:func:`fetch_mpc_observations`) --
  query the MPC web API and apply EFCC18 catalog debiasing.
- **Gaia DR3** (:func:`fetch_gaia_observations`) --
  retrieve optical astrometry from the Gaia TAP service.
- **JPL radar** (:func:`fetch_radar_observations`, :func:`fetch_radar_table`) --
  fetch delay and Doppler measurements from the JPL Small-Body Radar database.
"""

from .._core import (
    DiffuseState,
    Observation,
    OrbitFit,
    OrbitSamples,
    RangingSamples,
    UncertainState,
    fit_orbit,
    fit_orbit_mcmc,
    fit_orbit_ranging,
    initial_orbit_determination,
    lambert,
)
from .gaia import fetch_gaia_observations
from .mpc_api import fetch_mpc_observations
from .mpc_legacy import MPCObservation, mpc_obs_to_observations
from .radar import fetch_radar_observations, fetch_radar_table

__all__ = [
    # fitting
    "Observation",
    "OrbitFit",
    "OrbitSamples",
    "DiffuseState",
    "RangingSamples",
    "UncertainState",
    "fit_orbit",
    "fit_orbit_mcmc",
    "fit_orbit_ranging",
    "initial_orbit_determination",
    "lambert",
    # observation ingestion
    "fetch_observations",
    "fetch_gaia_observations",
    "fetch_mpc_observations",
    "fetch_radar_observations",
    "fetch_radar_table",
    "MPCObservation",
    "mpc_obs_to_observations",
]


def fetch_observations(desig: str, update_cache: bool = False) -> list[Observation]:
    """
    Fetch all available observations for an object as fitting Observations.

    This is a convenience wrapper around the various specific fetchers that
    tries to get "everything" for a given object.  Currently this means
    MPC ADES optical, JPL radar, and Gaia DR3 measurements.

    MPC observations have the EFCC18 debiasing correction applied and per-observatory
    sigmas when available.

    Parameters
    ----------
    desig :
        Object designation (e.g. ``"Apophis"``, ``"101955"``, ``"1999 RQ36"``).
    update_cache :
        If ``True``, refresh any cached data before filtering.

    Returns
    -------
    list[Observation]
        All available observations for the object.
    """
    observations = []

    # Fetch MPC observations
    observations.extend(fetch_mpc_observations(desig, update_cache=update_cache))

    # Fetch Gaia observations
    observations.extend(fetch_gaia_observations(desig, update_cache=update_cache))

    # Fetch JPL radar observations
    observations.extend(fetch_radar_observations(desig, update_cache=update_cache))

    return observations
