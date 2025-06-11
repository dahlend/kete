"""
Definitions for population groups.

Group definitions are computed strictly from perihelion distance and eccentricity.
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from .conversion import compute_aphelion, compute_semi_major

MAX_ECCENTRICITY = 0.995
"""Maximum allowed eccentricity to be considered non-hyperbolic."""


def mba_inner(
    peri_dist: NDArray[np.floating], ecc: NDArray[np.floating], *_
) -> np.ndarray:
    """
    Orbital element filter to select inner main belt objects.

    Returns `True` if the object is in the inner main belt.

    Parameters
    ----------
    peri_dist:
        Perihelion distance in units of AU.
    ecc:
        Eccentricity of the orbit.
    """
    a = compute_semi_major(peri_dist, ecc)
    return (peri_dist > 1.3) & (a <= 2.5) & (ecc < MAX_ECCENTRICITY) & (ecc >= 0.0)


def mba_middle(
    peri_dist: NDArray[np.floating], ecc: NDArray[np.floating], *_
) -> np.ndarray:
    """
    Orbital element filter to select middle main belt objects.

    Returns `True` if the object is in the middle main belt.

    Parameters
    ----------
    peri_dist:
        Perihelion distance in units of AU.
    ecc:
        Eccentricity of the orbit.
    """
    a = compute_semi_major(peri_dist, ecc)
    return (
        (peri_dist > 1.3)
        & (a > 2.5)
        & (a <= 2.82)
        & (ecc < MAX_ECCENTRICITY)
        & (ecc >= 0.0)
    )


def mba_outer(
    peri_dist: NDArray[np.floating], ecc: NDArray[np.floating], *_
) -> np.ndarray:
    """
    Orbital element filter to select outer main belt objects.

    Returns `True` if the object is in the outer main belt.

    Parameters
    ----------
    peri_dist:
        Perihelion distance in units of AU.
    ecc:
        Eccentricity of the orbit.
    """
    a = compute_semi_major(peri_dist, ecc)
    return (
        (peri_dist > 1.3)
        & (a > 2.82)
        & (a <= 3.6)
        & (ecc < MAX_ECCENTRICITY)
        & (ecc >= 0.0)
    )


def neo_atira(
    peri_dist: NDArray[np.floating], ecc: NDArray[np.floating], *_
) -> np.ndarray:
    """
    Orbital element filter to select Atira NEO objects.

    Returns `True` if the object is in the Atiras.

    Parameters
    ----------
    peri_dist:
        Perihelion distance in units of AU.
    ecc:
        Eccentricity of the orbit.
    """
    a = compute_semi_major(peri_dist, ecc)
    aphelion = compute_aphelion(peri_dist, ecc)
    return (
        (peri_dist <= 1.3)
        & (peri_dist >= 0.0)
        & (a < 1.0)
        & (aphelion < 0.983)
        & (ecc < MAX_ECCENTRICITY)
        & (ecc >= 0.0)
    )


def neo_aten(
    peri_dist: NDArray[np.floating], ecc: NDArray[np.floating], *_
) -> np.ndarray:
    """
    Orbital element filter to select Aten NEO objects.

    Returns `True` if the object is in the Atens.

    Parameters
    ----------
    peri_dist:
        Perihelion distance in units of AU.
    ecc:
        Eccentricity of the orbit.
    """
    a = compute_semi_major(peri_dist, ecc)
    aphelion = compute_aphelion(peri_dist, ecc)
    return (
        (peri_dist <= 1.3)
        & (peri_dist >= 0.0)
        & (a < 1.0)
        & (aphelion >= 0.983)
        & (ecc < MAX_ECCENTRICITY)
        & (ecc >= 0.0)
    )


def neo_apollo(
    peri_dist: NDArray[np.floating], ecc: NDArray[np.floating], *_
) -> np.ndarray:
    """
    Orbital element filter to select Apollo NEO objects.

    Returns `True` if the object is in the Apollo.

    Parameters
    ----------
    peri_dist:
        Perihelion distance in units of AU.
    ecc:
        Eccentricity of the orbit.
    """
    a = compute_semi_major(peri_dist, ecc)
    return (peri_dist <= 1.017) & (a >= 1.0) & (ecc < MAX_ECCENTRICITY) & (ecc >= 0.0)


def neo_amor(
    peri_dist: NDArray[np.floating], ecc: NDArray[np.floating], *_
) -> np.ndarray:
    """
    Orbital element filter to select Amor NEO objects.

    Returns `True` if the object is in the Amors.

    Parameters
    ----------
    peri_dist:
        Perihelion distance in units of AU.
    ecc:
        Eccentricity of the orbit.
    """
    a = compute_semi_major(peri_dist, ecc)
    return (
        (peri_dist <= 1.3)
        & (a > 1.0)
        & (peri_dist > 1.017)
        & (ecc < MAX_ECCENTRICITY)
        & (ecc >= 0.0)
    )


def jup_trojan(
    peri_dist: NDArray[np.floating], ecc: NDArray[np.floating], *_
) -> np.ndarray:
    """
    Orbital element filter to select objects which appear to be jupiter trojans.

    Returns `True` if the object is not one of the above.

    Parameters
    ----------
    peri_dist:
        Perihelion distance in units of AU.
    ecc:
        Eccentricity of the orbit.
    """
    a = compute_semi_major(peri_dist, ecc)
    return (a < 5.5) & (a > 4.6) & (ecc < 0.3)


def centaur(
    peri_dist: NDArray[np.floating], ecc: NDArray[np.floating], *_
) -> np.ndarray:
    """
    Orbital element filter to select a centaur orbit.

    Returns `True` if the object is a centaur.

    Parameters
    ----------
    peri_dist:
        Perihelion distance in units of AU.
    ecc:
        Eccentricity of the orbit.
    """
    a = compute_semi_major(peri_dist, ecc)
    return (a >= 5.5) & (a < 30.1)


def trans_neptunian(
    peri_dist: NDArray[np.floating], ecc: NDArray[np.floating], *_
) -> np.ndarray:
    """
    Orbital element filter to select a trans-neptunian orbit.

    Returns `True` if the object is a TNO.

    Parameters
    ----------
    peri_dist:
        Perihelion distance in units of AU.
    ecc:
        Eccentricity of the orbit.
    """
    return compute_semi_major(peri_dist, ecc) >= 30.1


def parabolic(
    peri_dist: NDArray[np.floating], ecc: NDArray[np.floating], *_
) -> np.ndarray:
    """
    Orbital element filter to select a parabolic orbit.

    Returns `True` if the object is parabolic.

    Parameters
    ----------
    peri_dist:
        Perihelion distance in units of AU.
    ecc:
        Eccentricity of the orbit.
    """
    return np.abs(ecc - 1.0) < 0.005


def hyperbolic(
    peri_dist: NDArray[np.floating], ecc: NDArray[np.floating], *_
) -> np.ndarray:
    """
    Orbital element filter to select a hyperbolic orbit.

    Returns `True` if the object is hyperbolic.

    Parameters
    ----------
    peri_dist:
        Perihelion distance in units of AU.
    ecc:
        Eccentricity of the orbit.
    """
    return ecc > 1.005


def distant(
    peri_dist: NDArray[np.floating], ecc: NDArray[np.floating], *_
) -> np.ndarray:
    """
    Orbital element filter to select distant objects outside the main belt.

    Returns `True` if the object is not one of the above.

    Parameters
    ----------
    peri_dist:
        Perihelion distance in units of AU.
    ecc:
        Eccentricity of the orbit.
    """
    a = compute_semi_major(peri_dist, ecc)
    return (
        (peri_dist > 1.3)
        & (a > 3.6)
        & (ecc < MAX_ECCENTRICITY)
        & (ecc >= 0.0)
        & ~jup_trojan(peri_dist, ecc)
        & ~trans_neptunian(peri_dist, ecc)
        & ~centaur(peri_dist, ecc)
        & ~parabolic(peri_dist, ecc)
        & ~hyperbolic(peri_dist, ecc)
    )


# Functions below use the orbital elements definitions above.


def which_group(
    peri_dist: NDArray[np.floating], ecc: NDArray[np.floating], *_
) -> list[str]:
    """
    Return a label for the object orbital elements provided.

    Parameters
    ----------
    peri_dist:
        Perihelion distance in units of AU.
    ecc:
        Eccentricity of the orbit.
    """
    groups = []
    for peri, e in zip(np.atleast_1d(peri_dist), np.atleast_1d(ecc)):
        for func in [
            mba_inner,
            mba_middle,
            mba_outer,
            neo_apollo,
            neo_aten,
            neo_amor,
            neo_atira,
            distant,
            jup_trojan,
            hyperbolic,
            parabolic,
            trans_neptunian,
            centaur,
        ]:
            if func(peri, e):
                groups.append(func.__name__)
                break
        else:
            groups.append("Other")
    return groups


def mba(peri_dist, ecc, *_):
    """
    Orbital element filter to select all main belt objects.

    Returns `True` if the object is in the main belt.

    Parameters
    ----------
    peri_dist:
        Perihelion distance in units of AU.
    ecc:
        Eccentricity of the orbit.
    """
    return (
        mba_inner(peri_dist, ecc)
        | mba_middle(peri_dist, ecc)
        | mba_outer(peri_dist, ecc)
    )


def neo(peri_dist, ecc, *_):
    """
    Orbital element filter to select all NEO objects.

    Returns `True` if the object is an NEO.

    Parameters
    ----------
    peri_dist:
        Perihelion distance in units of AU.
    ecc:
        Eccentricity of the orbit.
    """
    return (
        neo_atira(peri_dist, ecc)
        | neo_aten(peri_dist, ecc)
        | neo_apollo(peri_dist, ecc)
        | neo_amor(peri_dist, ecc)
    )


def plot_groups():
    """
    Plot orbital element groups defined above.


    .. plot::
        :context: close-figs

        import kete
        kete.population.plot_groups();

    """

    ecc = np.linspace(0.01, 1.1, 501)
    peri = np.linspace(0.01, 5, 501)

    eccs, peris = np.meshgrid(ecc, peri)

    for func in [
        mba_inner,
        mba_middle,
        mba_outer,
        neo_apollo,
        neo_aten,
        neo_amor,
        neo_atira,
        distant,
        jup_trojan,
        hyperbolic,
        parabolic,
        trans_neptunian,
        centaur,
    ]:
        plt.scatter(
            peris[func(peris, eccs)], eccs[func(peris, eccs)], s=1, label=func.__name__
        )
    plt.xlabel("Perihelion Dist (AU)")
    plt.ylabel("Eccentricity")
    plt.legend(loc=1)
