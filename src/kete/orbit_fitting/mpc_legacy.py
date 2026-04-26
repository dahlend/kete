"""MPC 80-character observation format: parsing and conversion to Observations."""

from __future__ import annotations

import logging

import numpy as np

from .._core import Observation
from .common import (
    _fetch_debias_table,
    _over_obs_reweight_factors,
    _time_sigma_for_obs,
    get_observatory_std,
)

logger = logging.getLogger(__name__)


class MPCObservation:
    """
    Representation of an 80-character format MPC observation.

    .. testcode::
        :skipif: True

        import kete
        import gzip

        # Comet Observations
        # url = "https://www.minorplanetcenter.net/iau/ECS/MPCAT-OBS/CmtObs.txt.gz"

        # Download the database of unnumbered observations from the MPC
        url = "https://www.minorplanetcenter.net/iau/ECS/MPCAT-OBS/UnnObs.txt.gz"
        url = "https://www.minorplanetcenter.net/iau/ECS/MPCAT-OBS/NumObs.txt.gz"
        path = kete.data.download_file(url)

        # Fetch all lines from the file which contain C51 (WISE) observatory code.
        obs_code = "C51".encode()
        with gzip.open(path) as f:
            lines = [line.decode() for line in f if obs_code == line[77:80]]

        # Parse lines into a list of MPCObservations
        observations = kete.observations.MPCObservation.from_lines(lines)

    """

    _UNSUPPORTED = set("WwQqVvRrXxTt")

    def __init__(
        self,
        desig: str,
        prov_desig: str,
        discovery: bool,
        note1: str,
        note2: str,
        jd: float,
        ra: float,
        dec: float,
        mag_band: str,
        catalog_code: str,
        obs_code: str,
        sun2sc: list[float],
    ):
        self.desig = desig
        self.prov_desig = prov_desig
        self.discovery = discovery
        self.note1 = note1
        self.note2 = note2
        self.jd = jd
        self.ra = ra
        self.dec = dec
        self.mag_band = mag_band
        self.catalog_code = catalog_code
        self.obs_code = obs_code
        self.sun2sc = list(sun2sc) if sun2sc is not None else [np.nan, np.nan, np.nan]

    @classmethod
    def from_lines(cls, lines, load_sc_pos=True):
        """
        Create a list of MPCObservations from a list of single 80 char lines.
        """
        from .. import conversion
        from ..time import Time

        found = []
        idx = 0
        while True:
            if idx >= len(lines):
                break
            line = cls._read_first_line(lines[idx], conversion, Time)
            idx += 1
            if line is None:
                continue
            if line["note2"] == "s":
                logger.warning("Second line of spacecraft observation found alone")
                continue
            elif line["note2"] == "S":
                if idx >= len(lines):
                    logger.warning("Missing second line of spacecraft observation.")
                    break
                pos_line = lines[idx]
                idx += 1
                if load_sc_pos:
                    line["sun2sc"] = cls._read_second_line(pos_line, line["jd"])
            found.append(cls(**line))
        return found

    @staticmethod
    def _read_first_line(line, conversion, Time):
        from .._core import unpack_designation as _unpack

        if line[14] in MPCObservation._UNSUPPORTED:
            return None

        mag_band = line[65:71].strip()
        year, month, day = line[15:32].strip().split()
        jd = Time.from_ymd(int(year), int(month), float(day)).jd
        if len(mag_band) > 0:
            mag_band = mag_band.split(maxsplit=1)[0]

        ra = conversion.ra_hms_to_degrees(line[32:44].strip())
        dec = conversion.dec_dms_to_degrees(line[44:55].strip())

        try:
            desig = _unpack(line[:5])
        except ValueError:
            desig = line[:5].strip()
        try:
            prov_desig = _unpack(line[5:12].strip())
        except ValueError:
            prov_desig = line[5:12].strip()

        # Column 72 (0-indexed 71): single-character MPC star catalog code.
        catalog_code = line[71].strip() if len(line) > 71 else ""

        return dict(
            desig=desig,
            prov_desig=prov_desig,
            discovery=line[12] == "*",
            note1=line[13].strip(),
            note2=line[14].strip(),
            ra=ra,
            dec=dec,
            mag_band=mag_band,
            catalog_code=catalog_code,
            obs_code=line[77:80],
            sun2sc=None,
            jd=jd,
        )

    @staticmethod
    def _read_second_line(line, jd):
        from .. import constants, spice
        from ..vector import Frames, Vector

        if line[14] != "s":
            raise SyntaxError("No second line of spacecraft observation found.")

        x = float(line[34:45].replace(" ", "")) / constants.AU_KM
        y = float(line[46:57].replace(" ", "")) / constants.AU_KM
        z = float(line[58:69].replace(" ", "")) / constants.AU_KM
        earth2sc = Vector([x, y, z], Frames.Equatorial).as_ecliptic
        sun2earth = spice.get_state("Earth", jd).pos
        sun2sc = sun2earth + earth2sc
        return list(sun2sc)

    @property
    def sc2obj(self):
        from ..vector import Vector

        return Vector.from_ra_dec(self.ra, self.dec).as_ecliptic


def mpc_obs_to_observations(
    mpc_obs: list,
    apply_over_obs_reweight: bool = True,
    debias: bool = True,
) -> list[Observation]:
    """
    Convert a list of MPCObservation objects to fitting Observations.

    Only optical (RA/Dec) observations are supported. Each MPCObservation is
    converted to an ``Observation.optical`` with the observer state computed
    from the MPC observatory code (ground-based) or from the stored spacecraft
    position.

    Per-observatory uncertainties are applied when available from the
    pre-computed residual table.  When no table entry exists for an observatory
    code, an epoch- and observation-type-based fallback is used (see
    :func:`~kete.orbit_fitting.common._time_sigma_for_obs`).

    When ``debias`` is True, the EFCC18 star-catalog bias correction is applied
    using the catalog code stored on each observation (column 72 of the 80-char
    format).  Observations with an unknown or blank catalog code are passed
    through unchanged.

    Parameters
    ----------
    mpc_obs :
        List of ``MPCObservation`` objects (see :mod:`kete.observations`).
    apply_over_obs_reweight :
        When True (default), inflate sigma by sqrt(n/4) for groups of more
        than 4 observations from the same observatory on the same night,
        following Veres et al. 2017.  Spacecraft observations are exempt.
    debias :
        When True (default), apply the EFCC18 star-catalog bias correction.
        Requires the JPL ``debias_2018.tgz`` archive, downloaded on first use.

    Returns
    -------
    list[Observation]
        One ``Observation.optical`` per input observation.

    Examples
    --------
    .. testcode::
        :skipif: True

        import kete

        lines = [...]  # 80-char MPC observation lines
        mpc_obs = kete.observations.MPCObservation.from_lines(lines)
        observations = kete.observations.mpc_obs_to_observations(mpc_obs)
        fit = kete.fitting.fit_orbit(initial_state, observations)
    """
    from .. import spice
    from ..time import Time as _Time
    from ..vector import Frames, State

    debias_table = _fetch_debias_table() if debias else None

    spacecraft = [obs.note2 in ("S", "s") for obs in mpc_obs]
    if apply_over_obs_reweight:
        factors = _over_obs_reweight_factors(
            [obs.obs_code for obs in mpc_obs],
            [obs.jd for obs in mpc_obs],
            spacecraft,
        )
    else:
        factors = [1.0] * len(mpc_obs)

    observations = []
    for obs, factor, is_sc in zip(mpc_obs, factors, spacecraft):
        ra = obs.ra
        dec = obs.dec

        # Observation.optical expects sky-plane sigma_ra (sigma_ra * cos(dec)).
        # Priority: per-observatory table, then epoch/type-based fallback.
        time_std, std = _time_sigma_for_obs(obs.note2, _Time(obs.jd).year_float)
        obs_errors = get_observatory_std(obs.obs_code)
        if obs_errors is not None:
            # Table ra_std is computed from raw RA coordinate residuals;
            # multiply by cos(dec) to match the sky-plane input convention.
            s_ra = obs_errors[0] * np.cos(np.radians(dec))
            s_dec = obs_errors[1]
        else:
            s_ra = std * np.cos(np.radians(dec))
            s_dec = std

        if debias_table is not None and obs.catalog_code:
            shift = debias_table.lookup(obs.catalog_code, ra, dec, obs.jd)
            if shift is not None:
                ra -= shift[0] / 3600.0
                dec -= shift[1] / 3600.0

        if is_sc and not any(np.isnan(obs.sun2sc)):
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
            observer = spice.mpc_code_to_ecliptic(
                obs.obs_code, obs.jd, center=0
            ).as_equatorial

        observations.append(
            Observation.optical(
                observer=observer,
                ra=ra,
                dec=dec,
                sigma_ra=s_ra * factor,
                sigma_dec=s_dec * factor,
                time_sigma=time_std,
            )
        )

    return observations
