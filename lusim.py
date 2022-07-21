import numpy as np
import scipy.spatial as sps


def covar(t, d, r):
    """covariance calculation for standard models"""
    h = d / r
    if t == 1:  # Spherical
        c = 1 - h * (1.5 - 0.5 * h ** 2)
        c[h > 1] = 0
    elif t == 2:  # Exponential
        c = np.exp(-3 * h)
    elif t == 3:  # Gaussian
        c = np.exp(-3 * h ** 2)
    return c


def pairwisecova_1D(points, ranges, vtypes, c0, cc):
    """"pairwise covariance matrix between 1D array `points`"""
    EPS = 1e-5
    if len(ranges) != len(vtypes):
        raise ValueError("len(ranges) must equal len(vtype)")
    nx = len(points)
    nst = len(vtypes)
    cmax = c0 + np.sum(cc)
    cova = np.zeros((nx, nx))
    for i in range(nst):
        Q = points.copy()
        Q[:, 0] = Q[:, 0] / ranges[i]
        d = sps.distance_matrix(Q, Q)
        cova = cova + cc[i] * covar(vtypes[i], d, r=1)
    cova[d < EPS] = cmax
    return cova


def pairwisecova_2D(points, ranges_x, ranges_y, vtypes, c0, cc):
    """"pairwise covariance matrix between 2D array `points`"""
    EPS = 1e-5
    if len(ranges_x) != len(vtypes):
        raise ValueError("len(ranges) must equal len(vtype)")
    nxy = points.shape[0]
    nst = len(vtypes)
    cmax = c0 + np.sum(cc)
    cova = np.zeros((nxy, nxy))
    for i in range(nst):
        Q = points.copy()
        Q[:, 0] = Q[:, 0] / ranges_x[i]
        Q[:, 1] = Q[:, 1] / ranges_y[i]
        d = sps.distance_matrix(Q, Q)
        cova = cova + cc[i] * covar(vtypes[i], d, r=1)
    cova[d < EPS] = cmax
    return cova


def pairwisecova_3D(points, ranges_x, ranges_y, ranges_z, vtypes, c0, cc):
    """"pairwise covariance matrix between 3D array `points`"""
    EPS = 1e-5
    if len(ranges_x) != len(vtypes):
        raise ValueError("len(ranges) must equal len(vtype)")
    nxy = points.shape[0]
    nst = len(vtypes)
    cmax = c0 + np.sum(cc)
    cova = np.zeros((nxy, nxy))
    for i in range(nst):
        Q = points.copy()
        Q[:, 0] = Q[:, 0] / ranges_x[i]
        Q[:, 1] = Q[:, 1] / ranges_y[i]
        Q[:, 2] = Q[:, 2] / ranges_z[i]
        d = sps.distance_matrix(Q, Q)
        cova = cova + cc[i] * covar(vtypes[i], d, r=1)
    cova[d < EPS] = cmax
    return cova


def lusim(cova, nr, seed):
    """
    LU matrix simulation for an isotropic variogram
    cova: covariance matrix
    nr: number of realizations
    seed: random number seed
    """
    rng = np.random.default_rng(seed)
    L = np.linalg.cholesky(cova)

    # Draw normal random numbers
    if nr == 1:
        x = rng.normal(0, 1, cova.shape[0])
    else:
        x = rng.normal(0, 1, [cova.shape[0], nr])

    # Correlate with cholesky
    x = L @ x

    return x
