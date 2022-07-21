import numpy as np
from numba import njit

"""
Variogram related functions
"""


def inv_dist(bins, power):
    if not bins:
        return
    wts = 1 / np.array(bins) ** power
    return wts / np.sum(wts)


def indicator_transform(z, zc):
    """indicator transform of z based on cutoffs zc"""
    z = z.flatten()
    zi = np.zeros((len(z), len(zc)))
    ivars = []
    for j, c in enumerate(zc):
        zi[:, j] = np.where(z <= c, 1, 0)
        iprop = np.mean(zi[:, j])
        ivars.append(iprop * (1 - iprop))
    return zi, ivars


def update_variogram(values, lag_dict):
    """updated current experimental variogram with `values`
    TODO: account for lags with no pairs
    """
    if not lag_dict:
        return
    nlags = len(lag_dict)
    expvario = np.full(nlags, np.nan)
    for n in range(nlags):
        tail = values[lag_dict[n][:, 0]]
        head = values[lag_dict[n][:, 1]]
        if len(tail) == 0 or len(head) == 0:
            #             print(f"No pairs in lag {n}!")
            continue
        expvario[n] = 1 / (2 * len(tail)) * np.sum((tail - head) ** 2)
    # set missing values to sill
    expvario[np.isnan(expvario)] = np.var(values)
    return expvario


def get_vario_model_points(model_pts, lags):
    """get variogram model values at corresponding experimental lag dists"""
    xvals = np.arange(len(model_pts))
    return np.interp(lags, xvals, model_pts)


def expvario_loss(exp_pts, model_pts, lags, lag_wts, sill):
    """squared error between experimental points and variogram model"""
    if not np.array(exp_pts).any():
        return 0.0
    exp_pts /= sill
    model_at_lags = get_vario_model_points(model_pts, lags)
    return np.sum(((exp_pts - model_at_lags) ** 2) * lag_wts)  # / len(exp_pts)


@njit
def assemble_pairs(
    x,
    y,
    z,
    nd,
    uvxazm,
    uvyazm,
    csatol,
    csdtol,
    uvzdec,
    uvhdec,
    lagdis,
    lagtol,
    nlags,
    bandh,
    bandv,
    maxdis,
):
    """return data indices and lag index for valid paris"""

    EPS = 1e-6
    pairs = []

    # loop over pairs
    for i in range(nd):

        for j in range(i, nd):

            # lag for current pair
            dx = x[j] - x[i]
            dy = y[j] - y[i]
            dz = z[j] - z[i]
            dxs = dx * dx
            dys = dy * dy
            dzs = dz * dz
            hs = dxs + dys + dzs

            if hs < EPS:
                continue
            if hs > maxdis:
                continue

            h = np.sqrt(max(hs, 0.0))

            # check acceptable azimuth angle
            dxy = np.sqrt(max((dxs + dys), 0.0))

            if dxy < EPS:
                dcazm = 1.0
            else:
                dcazm = (dx * uvxazm + dy * uvyazm) / dxy

            if np.abs(dcazm) < csatol:
                continue

            # check the horizontal bandwidth
            band = uvxazm * dy - uvyazm * dx

            if np.abs(band) > bandh:
                continue

            # check the dip angle
            if dcazm < 0.0:
                dxy = -dxy
            if dxy < EPS:
                dcdec = 0.0
            else:
                dcdec = (dxy * uvhdec + dz * uvzdec) / h
                if np.abs(dcdec) < csdtol:
                    continue

            # check the vertical bandwidth
            band = uvhdec * dz - uvzdec * dxy
            if np.abs(band) > bandv:
                continue

            # and check lag tolerance
            # iterate over all bins as they may overlap
            for n in range(nlags):

                if (h > n * lagdis - lagtol) and (h < n * lagdis + lagtol):

                    # if we made it this far the pair is acceptable
                    pairs.append([i, j, n, h])

    return np.array(pairs)


def variogram_pairs(
    data, xcol, ycol, zcol, azm, atol, bandh, dip, dtol, bandv, nlags, lagdis, lagtol,
):
    """calculate directional variogram pair indices by lag"""

    if not any([xcol, ycol, zcol]):
        raise ValueError("coordinate columns must be provided!")

    nd = len(data)

    if ycol is None and zcol is None:  # 1d
        x = data[xcol].values
        y = np.zeros(nd)
        z = np.zeros(nd)

    elif zcol is None:  # 2d
        x = data[xcol].values
        y = data[ycol].values
        z = np.zeros(nd)

    else:  # 3d
        x = data[xcol].values
        y = data[ycol].values
        z = data[zcol].values

    # calculate distances and tolerances
    azimuth = (90.0 - azm) * np.pi / 180.0
    uvxazm = np.cos(azimuth)
    uvyazm = np.sin(azimuth)

    if atol < 0.0:
        csatol = np.cos(45.0 * np.pi / 180.0)
    else:
        csatol = np.cos(atol * np.pi / 180.0)

    declin = (90.0 - dip) * np.pi / 180.0
    uvzdec = np.cos(declin)
    uvhdec = np.sin(declin)

    if dtol < 0.0:
        csdtol = np.cos(45.0 * np.pi / 180.0)
    else:
        csdtol = np.cos(dtol * np.pi / 180.0)

    maxdis = (nlags * lagdis) ** 2

    if lagtol is None:
        lagtol = lagdis * 0.5

    pairs = assemble_pairs(
        x,
        y,
        z,
        nd,
        uvxazm,
        uvyazm,
        csatol,
        csdtol,
        uvzdec,
        uvhdec,
        lagdis,
        lagtol,
        nlags,
        bandh,
        bandv,
        maxdis,
    )

    lag_dict = {}
    bins = []
    for n in range(nlags):
        lag_pairs = pairs[pairs[:, 2] == n]
        bins.append(np.mean(lag_pairs[:, -1]))
        lag_dict[n] = (lag_pairs[:, :-1]).astype(int)

    return lag_dict, bins
