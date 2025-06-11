import numpy as np

__all__ = ["sunshield_rotation", "FOV_WIDTH", "FOV_HEIGHT"]


BANDS: list[float] = [4700.0, 8000.0]
"""Effective wavelength of the NC1 and NC2 bands in nm."""

FOV_WIDTH: float = 7.10
"""Expected effective field of view width in degrees. Approximate Value."""

FOV_HEIGHT: float = 1.68
"""Expected effective field of view height in degrees. Approximate Value."""

FOV_CHIP_GAP: float = 0.11
"""Expected effective gap between individual chips in degrees. Approximate Value."""

ZERO_MAGS: list[float] = [170.662, 64.13]
"""Zero point magnitude for nc1 and nc2"""

COLOR_CORR = [
    # Tbb    nc1      nc2
    [200.0, 1.50001, 1.17925],
    [220.0, 1.38320, 1.10277],
    [240.0, 1.29964, 1.05175],
    [260.0, 1.23789, 1.01713],
    [280.0, 1.19106, 0.99348],
    [300.0, 1.15477, 0.97740],
    [400.0, 1.05708, 0.95284],
    [500.0, 1.01897, 0.96149],
]
"""Expected color correction required for black body sources at 300k"""


def sunshield_rotation(sun2obs, pointing):
    """
    Calculate the angle the field of view must be rotated around the pointing
    vector by in order to place the sun shield directly between the telescope
    and sun.

    The rotation is defined as the angle needed to move the sun shield from
    the Z-axis down to the angle required to place it between the
    sun and telescope. The angle is defined by the right hand rule applied
    along the pointing vector.

    Pointing vector must be at least about a half an arcsecond from the poles
    in order to be computable.

    Note that no coordinate frames are specified here, this provides the
    rotation in the current frame from the frame's Z-axis. Provided inputs
    are assumed to be from the matching frame.

    Parameters
    ----------
    sun2obs :
        The vector from the sun to the telescope, units are arbitrary.
    pointing :
        The vector along which the telescope is pointing, the spacecraft's
        Z-axis.

    Returns
    -------
    float
        Angle in degrees around the pointing vector the spacecraft must be
        rotated to place the sun shield between the telescope and the sun.
    """
    obs2sun = -np.array(sun2obs)
    obs2sun /= np.linalg.norm(obs2sun)

    # normalize for safety
    pointing = np.array(pointing) / np.linalg.norm(pointing)

    # NEOS has a restriction of no closer than 45 degrees during operation.
    # This allows down to 44 for convenience
    if np.degrees(np.arccos(np.dot(obs2sun, pointing))) < 44:
        raise ValueError("Pointing vector is aiming too close to the Sun.")

    # The normal vector for the plane defined by the sun and pointing vectors.
    sun_plane_normal = np.cross(obs2sun, pointing)
    sun_plane_normal /= np.linalg.norm(sun_plane_normal)

    # Assuming that the spacecraft body z is along the pointing vector and
    # body y-axis (sun shield) is facing directly up in the current frame.
    # This means the body x-axis is defined by:
    # sc_x = cross((0, 0, 1), pointing) = (-pointing.y, pointing.x, 0)
    # This vector is the normal through the plane defined by the pointing and
    # the Z-axis of the plane.
    obs_body_x = np.array([-pointing[1], pointing[0], 0])

    r = np.linalg.norm(obs_body_x)
    if r < 1e-6:
        # Safety check, this definition is bad near poles.
        raise ValueError(
            "Nearly pointing at the pole, cannot compute within about half "
            "an arcsecond of the pole. It is strongly recommended to not "
            "use this angular definition so close to poles, quaternions are "
            "the better choice."
        )
    obs_body_x /= r

    # The great circle distance from the body_x to the sun_plane_normal is
    # an absolute rotation, so if the sun is on the "left" vs "right"
    # is not captured, and an additional calculation is needed to find
    # the sign of the rotation.
    sign = np.sign(np.dot(obs_body_x, obs2sun))
    return sign * np.degrees(np.arccos(np.dot(sun_plane_normal, obs_body_x)))
