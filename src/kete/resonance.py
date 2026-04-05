"""
Mean-motion resonance detection via orbital element time series.

Provides tools to identify whether an asteroid is in (or near) a mean-motion
resonance with a planet, by propagating the orbit and tracking the resonant
critical angle over time.
"""

import numpy as np
from numpy.typing import NDArray

from . import _core, spice
from .vector import CometElements, State

__all__ = [
    "nearest_resonance",
    "identify_resonance",
]

# Semi-major axes of the planets (AU), used for resonance location computation.
_PLANET_SEMI_MAJOR: dict[str, float] = {
    "Mercury Barycenter": 0.3871,
    "Venus Barycenter": 0.7233,
    "Earth Barycenter": 1.0000,
    "Mars Barycenter": 1.5237,
    "Jupiter Barycenter": 5.2038,
    "Saturn Barycenter": 9.5826,
    "Uranus Barycenter": 19.2184,
    "Neptune Barycenter": 30.1104,
}

# Planet-to-Sun mass ratios, used to weight resonance significance.
_PLANET_MASS_RATIO: dict[str, float] = {
    "Mercury Barycenter": 1.660e-7,
    "Venus Barycenter": 2.448e-6,
    "Earth Barycenter": 3.003e-6,
    "Mars Barycenter": 3.227e-7,
    "Jupiter Barycenter": 9.546e-4,
    "Saturn Barycenter": 2.858e-4,
    "Uranus Barycenter": 4.366e-5,
    "Neptune Barycenter": 5.150e-5,
}


def nearest_resonance(
    semi_major: float,
    planet: str = "Jupiter Barycenter",
    max_order: int = 4,
    max_p: int = 12,
) -> tuple[tuple[int, int], float, float]:
    """
    Find the nearest mean-motion resonance to a given semi-major axis.

    This is an instantaneous calculation from the semi-major axis alone —
    no propagation is performed. It identifies which ``p:q`` resonance with
    the specified planet has its nominal location closest to the given ``a``.

    The resonance location is:

    .. math::

        a_{p:q} = a_{\\text{planet}} \\cdot (q / p)^{2/3}

    Parameters
    ----------
    semi_major:
        Semi-major axis of the small body in AU.
    planet:
        Name of the perturbing planet.
    max_order:
        Maximum resonance order ``|p - q|`` to consider.
    max_p:
        Maximum value of ``p`` to consider.

    Returns
    -------
    tuple[tuple[int, int], float, float]
        A tuple of ((p, q), a_resonance, delta_a) where:
        - ``(p, q)`` is the resonance ratio (asteroid completes p orbits
          while the planet completes q).
        - ``a_resonance`` is the nominal resonance semi-major axis in AU.
        - ``delta_a`` is ``semi_major - a_resonance`` in AU.
    """
    a_planet = _PLANET_SEMI_MAJOR[planet]
    candidates = _find_candidate_resonances(
        semi_major,
        max_order,
        max_p,
        n_candidates=1,
        planets={planet: a_planet},
    )
    if not candidates:
        return (0, 0), 0.0, np.inf
    pl, p, q, delta = candidates[0]
    a_res = a_planet * (q / p) ** (2.0 / 3.0)
    return (p, q), a_res, semi_major - a_res


def _orbital_angles(state: State) -> tuple[float, float, float]:
    """Return (mean_longitude, varpi, Omega) in degrees for a State."""
    elem = CometElements.from_state(state)
    lam = (elem.mean_anomaly + elem.peri_arg + elem.lon_of_ascending) % 360.0
    varpi = (elem.peri_arg + elem.lon_of_ascending) % 360.0
    omega = elem.lon_of_ascending % 360.0
    return lam, varpi, omega


def _resonant_arguments(order: int) -> list[tuple[int, int, int, int]]:
    """
    Enumerate all valid D'Alembert resonant argument coefficients for a given
    resonance order ``|p - q|``.

    For a ``p:q`` resonance the general critical angle is:

    .. math::

        \\phi = p \\lambda_{\\text{planet}} - q \\lambda_{\\text{ast}}
               - r_1 \\varpi_{\\text{ast}} - r_2 \\varpi_{\\text{planet}}
               - r_3 \\Omega_{\\text{ast}} - r_4 \\Omega_{\\text{planet}}

    subject to:
    - ``r_1 + r_2 + r_3 + r_4 = p - q``
    - ``r_3 + r_4`` must be even (D'Alembert rule)

    Parameters
    ----------
    order:
        The resonance order ``|p - q|``.

    Returns
    -------
    list[tuple[int, int, int, int]]
        Each tuple is ``(r1, r2, r3, r4)``.
    """
    args = []
    for r1 in range(order + 1):
        for r2 in range(order - r1 + 1):
            for r3 in range(order - r1 - r2 + 1):
                r4 = order - r1 - r2 - r3
                if (r3 + r4) % 2 == 0:
                    args.append((r1, r2, r3, r4))
    return args


def _find_candidate_resonances(
    semi_major: float,
    max_order: int = 4,
    max_p: int = 12,
    n_candidates: int = 10,
    planets: dict[str, float] | None = None,
) -> list[tuple[str, int, int, float]]:
    """
    Find candidate resonances sorted by proximity to a semi-major axis.

    Parameters
    ----------
    semi_major:
        Semi-major axis in AU.
    max_order:
        Maximum resonance order ``|p - q|``.
    max_p:
        Maximum value of ``p``.
    n_candidates:
        Number of closest candidates to return.
    planets:
        Dict of planet name to semi-major axis (AU). Defaults to all planets.

    Returns a list of (planet, p, q, delta_a) tuples.
    """
    if planets is None:
        planets = _PLANET_SEMI_MAJOR
    candidates = []
    for planet, a_planet in planets.items():
        for p in range(1, max_p + 1):
            for q in range(max(1, p - max_order), p + max_order + 1):
                if q < 1:
                    continue
                a_res = a_planet * (q / p) ** (2.0 / 3.0)
                delta = abs(semi_major - a_res)
                candidates.append((planet, p, q, delta))

    candidates.sort(key=lambda x: x[3])
    return candidates[:n_candidates]


def _classify_angle(phi: NDArray) -> dict:
    """Compute circular statistics for an angle time series."""
    phi_rad = np.radians(phi)
    mean_sin = np.mean(np.sin(phi_rad))
    mean_cos = np.mean(np.cos(phi_rad))
    R = np.sqrt(mean_sin**2 + mean_cos**2)
    center = np.degrees(np.arctan2(mean_sin, mean_cos))
    deviations = (phi - center + 180.0) % 360.0 - 180.0
    amplitude = float(np.max(np.abs(deviations)))
    librating = R > 0.3 and amplitude < 170.0
    return {
        "R": float(R),
        "center": float(center),
        "amplitude": amplitude,
        "librating": librating,
    }


def _windowed_libration_fraction(phi: NDArray, n_windows: int = 5) -> float:
    """Fraction of time windows in which the angle librates.

    Splits the angle series into ``n_windows`` equal segments and applies
    ``_classify_angle`` to each. Returns the fraction of windows that are
    individually classified as librating (0.0 to 1.0).

    A value near 1.0 means stable libration. A value between ~0.2 and ~0.8
    indicates intermittent libration — the hallmark of a transitional object
    near a resonance boundary (e.g. Kirkwood gap edge).
    """
    segments = np.array_split(phi, n_windows)
    librating_count = sum(1 for seg in segments if _classify_angle(seg)["librating"])
    return librating_count / n_windows


def identify_resonance(
    state: State,
    n_steps: int = 500,
    step_size: float = 365.25,
    max_order: int = 4,
    max_p: int = 12,
    n_candidates: int = 10,
) -> dict:
    """
    Automatically identify the best mean-motion resonance for an object.

    Propagates the orbit once and checks all candidate resonances with all
    planets, returning the best match. Candidates are selected by proximity
    of the object's semi-major axis to the nominal resonance locations.

    The orbit is propagated a single time. At each step the asteroid's
    orbital angles and the angles of all relevant planets are recorded.
    All candidate resonant angles (including all D'Alembert combinations
    for inclined orbits) are then computed from the stored data without
    re-propagation.

    The returned ``"status"`` field classifies the object's relationship
    to its nearest resonance:

    - ``"librating"`` — the critical angle librates throughout the
      integration, indicating a stable mean-motion resonance.
    - ``"transitional"`` — the critical angle alternates between libration
      and circulation over the integration. This is characteristic of
      objects near resonance boundaries (e.g. Kirkwood gap edges) that are
      undergoing or about to undergo orbital transitions.
    - ``"circulating"`` — the critical angle circulates throughout. The
      object is near a resonance in semi-major axis but is not currently
      captured.

    Parameters
    ----------
    state:
        Initial state of the object (SSB-centered, Equatorial).
    n_steps:
        Number of propagation steps.
    step_size:
        Duration of each step in days.
    max_order:
        Maximum resonance order ``|p - q|`` to consider.
    max_p:
        Maximum value of ``p`` to scan.
    n_candidates:
        Number of closest resonance candidates to evaluate.

    Returns
    -------
    dict
        A dictionary containing:

        - ``"planet"``: str, name of the perturbing planet (e.g.
          ``"Jupiter Barycenter"``).
        - ``"p"``: int, asteroid mean-motion multiplier in the ``p:q`` ratio.
          The asteroid completes ``p`` orbits for every ``q`` orbits of the
          planet.
        - ``"q"``: int, planet mean-motion multiplier.
        - ``"resonance"``: str, human-readable label (e.g. ``"3:1"``).
        - ``"status"``: str, one of ``"librating"``, ``"transitional"``, or
          ``"circulating"`` (see above).
        - ``"d_alembert_coefficients"``: tuple of ``(r1, r2, r3, r4)``,
          the D'Alembert rule coefficients of the best critical angle:
          ``phi = p*lam_planet - q*lam_ast - r1*varpi_ast - r2*varpi_planet
          - r3*Omega_ast - r4*Omega_planet``.
        - ``"libration_center_deg"``: float, circular mean of the critical
          angle in degrees (−180, 180].
        - ``"libration_amplitude_deg"``: float, maximum deviation from the
          center in degrees.
        - ``"concentration"``: float in [0, 1], the mean resultant length of
          the critical angle distribution. Values near 1 indicate tight
          libration; values near 0 indicate full circulation.
        - ``"libration_fraction"``: float in [0, 1], the fraction of time
          windows in which the critical angle librates. A value of 1.0
          indicates continuous libration. Values between ~0.2 and ~0.8
          indicate intermittent libration (transitional behavior).
        - ``"delta_a_au"``: float, offset of the object's initial semi-major
          axis from the nominal resonance location in AU.
    """
    # Get initial semi-major axis to find candidates.
    elem = CometElements.from_state(state)
    a0 = elem.semi_major
    candidates = _find_candidate_resonances(a0, max_order, max_p, n_candidates)

    # Determine which planets we need to track.
    planets_needed = sorted({c[0] for c in candidates})

    # Propagate once, storing all angles.
    n = n_steps
    jd_times = np.empty(n)
    a_series = np.empty(n)
    ast_lam = np.empty(n)
    ast_varpi = np.empty(n)
    ast_omega = np.empty(n)
    planet_angles: dict[str, tuple[NDArray, NDArray, NDArray]] = {
        pl: (np.empty(n), np.empty(n), np.empty(n)) for pl in planets_needed
    }

    current_state = state
    for i in range(n):
        jd_end = current_state.jd + step_size
        current_state = _core.propagate_n_body(current_state, jd_end)

        lam, varpi, omega = _orbital_angles(current_state)
        ast_lam[i] = lam
        ast_varpi[i] = varpi
        ast_omega[i] = omega
        jd_times[i] = current_state.jd

        el = CometElements.from_state(current_state)
        a_series[i] = el.semi_major

        for pl in planets_needed:
            pl_state = spice.get_state(pl, current_state.jd)
            pl_lam, pl_varpi, pl_omega = _orbital_angles(pl_state)
            planet_angles[pl][0][i] = pl_lam
            planet_angles[pl][1][i] = pl_varpi
            planet_angles[pl][2][i] = pl_omega

    # Evaluate each candidate resonance from stored data.
    # Score = R * mu / (delta_a^2 + eps) so that massive, nearby planets
    # are strongly preferred when all angles are noise-level.
    best_result = None
    best_score = -1.0
    eps = 1e-12  # avoid division by zero for exact resonance location

    for planet, p, q, delta_a in candidates:
        order = abs(p - q)
        args = _resonant_arguments(order)
        mu = _PLANET_MASS_RATIO[planet]

        pl_lam = planet_angles[planet][0]
        pl_varpi = planet_angles[planet][1]
        pl_omega = planet_angles[planet][2]

        base = p * pl_lam - q * ast_lam

        for r1, r2, r3, r4 in args:
            phi = base - r1 * ast_varpi - r2 * pl_varpi - r3 * ast_omega - r4 * pl_omega
            phi = (phi + 180.0) % 360.0 - 180.0
            stats = _classify_angle(phi)

            score = stats["R"] * mu / (delta_a**2 + eps)
            if score > best_score:
                best_score = score
                best_result = {
                    "planet": planet,
                    "p": p,
                    "q": q,
                    "delta_a": float(delta_a),
                    "d_alembert": (r1, r2, r3, r4),
                    "stats": stats,
                    "phi": phi,
                }

    # Compute windowed libration fraction for the best angle and classify.
    planet = best_result["planet"]
    p = best_result["p"]
    q = best_result["q"]
    stats = best_result["stats"]
    phi = best_result["phi"]
    lib_frac = _windowed_libration_fraction(phi)

    if stats["librating"] and lib_frac >= 0.8:
        status = "librating"
    elif lib_frac > 0.0:
        status = "transitional"
    else:
        status = "circulating"

    return {
        "planet": planet,
        "p": p,
        "q": q,
        "resonance": f"{p}:{q}",
        "status": status,
        "d_alembert_coefficients": best_result["d_alembert"],
        "libration_center_deg": stats["center"],
        "libration_amplitude_deg": stats["amplitude"],
        "concentration": stats["R"],
        "libration_fraction": lib_frac,
        "delta_a_au": best_result["delta_a"],
    }
