import numpy as np
from scipy.stats import chi2

import kete


def generate_sample_from_cov(n_samples, cov, seed=None):
    """
    Generate normal samples from a covariance matrix.
    """
    rng = np.random.default_rng(seed=seed)
    return rng.multivariate_normal(np.zeros(len(cov)), cov, n_samples)


def ci_along_vec(cov, vec, ci=0.95):
    """
    Given a covariance matrix, and a unit length vector, calculate the confidence
    interval in the direction of that vector.

    NOTE: This is explicitly the confidence interval along the vector, not along the
    projection of the vector onto a 2d plane. What this means in practice is that it
    cannot be used to calculate on sky visible confidence intervals, as those are
    projections onto a surface.

    This returns a scalar value for each vector, indicating the bounds of the CI in that
    direction. This can be used to plot the CI ellipsoid for any given covariance
    matrix.

    This calculates the CI using the mahalanobis distance, so the chi squared
    distribution is used.

    Input vec can be a (3, n) array, which is recommended if computation will be
    repeated many times for a given covariance matrix.
    """
    cov_inv = np.linalg.inv(cov)

    # This calculation is based off of
    # https://en.wikipedia.org/wiki/Hotelling%27s_T-squared_distribution
    # the sqrt, chi2 ppf call is to calculated the confidence interval
    return np.sqrt(
        chi2.ppf(ci, len(cov)) / np.einsum("ji, jk, ki->i", vec, cov_inv, vec)
    )


def plane_basis(normal):
    """
    Construct a random pair of orthonormal vectors in the plane defined by the normal
    vector.
    """
    normal = normal / np.linalg.norm(normal)
    x = normal + np.random.randn(3)
    x = x - (x @ normal) * normal
    x = x / np.linalg.norm(x)

    y = np.cross(normal, x)
    y = y / np.linalg.norm(y)
    return np.array([x, y])


def vec_circle(normal, n_steps=1000):
    """
    Construct unit length vectors in a uniform circle in the plane defined by the
    provided normal vector.

    Returns [3, n_steps]
    """

    # Construct a vector in the plane defined by the normal
    vec = plane_basis(normal)[0]
    vec = kete.Vector(vec)

    angles = np.linspace(0.0, 360, n_steps)
    vecs = []
    for angle in angles:
        vecs.append(vec.rotate_around(normal, angle))
    return np.array(vecs).T


def ci_on_plane(cov, normal, ci=0.95, n_steps=1000):
    """
    Given a covariance matrix, and a plane normal, construct vectors which define the
    confidence interval projected into that plane.

    This can be used as an approximation method for calculating on sky uncertainties. If
    it is used in this way, note that the approximation is only valid when the CI is
    small enough that the plane is a good approximation for the celestial sphere, so
    maybe a few degrees at most.
    """
    normal = np.array(normal) / np.linalg.norm(normal)
    basis = plane_basis(normal)
    vecs = vec_circle(normal, n_steps=n_steps)
    proj_vecs = basis @ vecs
    proj_cov = basis @ cov @ basis.T
    return ci_along_vec(proj_cov, proj_vecs, ci=ci) * vecs


def ci_on_sphere(cov, obs2obj, ci=0.95, n_steps=300, n_iter=10):
    """
    Construct unit vectors which define the confidence interval on a sphere.

    This is a more accurate computation than the planar projection, but it significantly
    more expensive computationally.
    """
    normal = np.array(obs2obj) / np.linalg.norm(obs2obj)
    basis = plane_basis(normal)
    vecs = vec_circle(normal, n_steps=n_steps)
    proj_vecs = basis @ vecs
    proj_cov = basis @ cov @ basis.T

    # First, assume that we can project the covariance into the plane of the observation
    approx_vecs = ci_along_vec(proj_cov, proj_vecs, ci=ci) * vecs

    # Using these approximation vectors, project them onto the surface of the celestial
    # sphere, and using that as the definition for the new projection plane, re-project
    # the covariance matrix onto that and recompute the CI, repeat
    for _ in range(n_iter):
        for idx in range(approx_vecs.shape[1]):
            t_vec = approx_vecs[:, idx]
            t_norm = obs2obj + t_vec
            t_norm = t_norm / np.linalg.norm(t_norm)
            t_basis = plane_basis(t_norm)
            proj_vec = t_basis @ t_vec
            proj_cov = t_basis @ cov @ t_basis.T
            approx_vecs[:, idx] = (
                ci_along_vec(proj_cov, proj_vec[:, np.newaxis], ci=ci)[0] * t_vec
            )
    surface = approx_vecs + obs2obj[:, np.newaxis]
    surface = surface / np.linalg.norm(surface, axis=0)
    return surface


def projection_matrix(plane_normal):
    """
    Build a projection matrix, which will take a vector in 3d an project it onto the
    plane as defined by the provided normal vector.

    Example, projecting a vector into the y/z plane:
    >>> normal = np.array([1, 0, 0])
    >>> vec = np.array([1, 2, 0])
    >>> projection_matrix(normal) @ vec
    [0, 2, 0]
    """
    M = plane_basis(plane_normal).T
    return M @ np.linalg.inv(M.T @ M) @ M.T


def prob_obs_matches(obs2obj, obs2obj_mean, cov):
    """
    Given observation vectors, along with the fitted average position of an object and
    the covariance matrix, return the probabilities that the observations are
    sampled from the covariance matrix.
    """
    obs2obj = obs2obj / np.linalg.norm(obs2obj, axis=0)

    probs = []
    for vec in obs2obj:
        basis = plane_basis(vec)

        proj_vec = basis @ (vec - obs2obj_mean)
        proj_cov = basis @ cov @ basis.T

        cov_inv = np.linalg.inv(proj_cov)
        probs.append(
            chi2.cdf(np.einsum("i, ij, j->", proj_vec, cov_inv, proj_vec), df=2)
        )
    return np.array(probs)


def prob_obs_in_cloud(obs2obj, cov):
    """
    Calculate the probability that the observer is inside of the gaussian uncertainty of
    the object.
    """
    cov_inv = np.linalg.inv(cov)
    return 1 - chi2.cdf(np.einsum("i, ij, j->", obs2obj, cov_inv, obs2obj), df=3)
