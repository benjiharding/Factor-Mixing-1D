import numpy as np
import network as nn


def check_twod(z):
    twod = False
    ndim = z.shape
    if len(ndim) > 1:
        if ndim[1] > 1:
            twod = True
    return twod


#
# Variograms
#


def griddedgam(z):
    """gridded variogram values"""
    if check_twod(z):
        gx = griddedgam_x(z)
        gy = griddedgam_y(z)
        return gx, gy
    else:
        return griddedgam_x(z)


def griddedgam_x(z):
    """gridded variogram values along x axis"""
    nx = z.shape[0]
    gx = np.zeros(nx - 1)
    for i in range(1, nx - 1):
        z0 = z[0 : nx - i]
        z1 = z[i:nx]
        dz = (z1 - z0) ** 2
        gx[i] = np.sum(dz) / (2 * (nx - i))
    return gx


def griddedgam_y(z):
    """gridded variogram values along y axis"""
    ny = z.shape[1]
    gy = np.zeros(ny - 1)
    for i in range(1, ny - 1):
        z0 = z[0 : ny - i, :]
        z1 = z[i:ny, :]
        dz = (z1 - z0) ** 2
        gy[i] = np.sum(dz) / (2 * (ny - i))
    return gy


def objective_vario(true, pred, i, j, scale):
    """objective function value for continuous variograms"""
    loss = 0.0
    if check_twod(true):
        tx, ty = true[:, 0], true[:, 1]
        gx, gy = pred[:, 0], pred[:, 1]
        loss = vario_loss(tx, gx, i, j)
        loss = loss + vario_loss(ty, gy, i, j)
    else:
        loss = vario_loss(true, pred, i, j)
    return loss * scale


def vario_loss(true, pred, i, j):
    """squared error between predicted and target variogram"""
    return np.sum((pred[i:j] - true[i:j]) ** 2)


#
# Indicator Variograms
#


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


def objective_ivario(true, pred, nx, ny, i, j, quantiles, ivars, scale):
    """objective function value for indivator variograms
    true: dict of fitted variograms where quantiles are keys
    pred: (nx*ny, len(quantiles)) array of indicator transformed values
    """
    twod = False
    if ny > 1:
        twod = True
    loss = 0.0
    if twod:
        for k, q in enumerate(quantiles):
            tx, ty = true[f"{q}_x"], true[f"{q}_y"]
            gx, gy = griddedgam(pred[:, k].reshape(ny, nx))
            loss = vario_loss(tx, gx / ivars[k], i, j)
            loss = loss + vario_loss(ty, gy / ivars[k], i, j)
    else:
        for k, q in enumerate(quantiles):
            gx = griddedgam(pred[:, k])
            loss = vario_loss(true[f"{q}_x"], gx / ivars[k], i, j)
    return loss * scale


#
# Runs
#


def binary_runs(x, runs_above):
    """Calcualte runs and cumulative runs in binary array x"""
    runs_data = {}
    x = np.asarray(x)
    first_run = x[0]  # 1 or 0
    runstart = np.nonzero(np.diff(np.r_[[-np.inf], x, [np.inf]]))[0]
    runs = np.diff(runstart)
    runs = _check_runs_above(first_run, runs, runs_above)
    cum_runs = []
    for run in runs:
        for i in range(run):
            sub_run_length = run - i
            num_sub_runs = i + 1
            cum_runs.append([*[sub_run_length] * num_sub_runs])

    runs_data["runs"] = runs
    runs_data["cum_runs"] = np.array([a for b in cum_runs for a in b])
    runs_data["run_idxs"] = runstart
    runs_data["n_runs"] = len(runs)

    try:  # catch situation where all runs are below/above?
        runs_data["cum_runs_freqs"] = np.bincount(runs_data["cum_runs"])[1:]
        runs_data["runs_freqs"] = np.bincount(runs)[1:]
    except:
        runs_data["cum_runs_freqs"] = np.array([])
        runs_data["runs_freqs"] = np.array([])

    return runs_data


def _check_runs_above(first_run, runs, runs_above=True):
    if runs_above:
        if first_run:
            runs = runs[1::2]
        else:
            runs = runs[0::2]
        return runs
    else:
        return runs


#
# Total Runs
#


def objective_totruns(x, scale):
    """
    input: sequences
    output: combined histogram of total runs
    """
    pass


#
# Run Length Frequencies
#


def objective_runfreqs(x, scale):
    """
    input: sequences
    output: combined histogram of cumulative runs
    """

    pass


#
# n-Point Connectivity
#


def objective_npoint(x, scale):
    pass


#
# Component Weight Calibration
#


#
# Objective Function
#


def network_lmr_i_objective_2D(x, *args):
    """objective function for network lmr with indicator loss"""

    # params, connections, Y, nx, ny, yc, quantiles,
    # ivarios, xranges_it, yranges_it = args

    L = len(parameters) // 2
    num_wts = np.cumsum([0] + connections)

    # reshape 1D vector into appropriate matrices
    for ell in range(1, L + 1):
        shape = parameters["W" + str(ell)].shape
        parameters["W" + str(ell)] = x[num_wts[ell - 1] : num_wts[ell]].reshape(shape)

    # caluclate the forward pass
    AL = nn.linear_forward_2D(Y, parameters, relu)

    # indicator transform
    AL_i, ivars = indicator_transform(AL, yc)

    # calculate the gridded varaiograms and loss
    loss = 0.0
    for j, q in enumerate(quantiles):
        gx, gy = griddedgam_2D(AL_i[:, j].reshape(ny, nx))
        loss = loss + calculate_loss(
            y_true=[ivarios[f"{q}_x"], ivarios[f"{q}_y"]],
            y_pred=[gx / ivars[j], gy / ivars[j]],
            i=0,
            j=max(max(xranges_it[j], yranges_it[j])[1:]),
        )

    return loss


def network_lmr_objective_2D(x, *args):
    """objective function for network lmr"""

    # params, connections, Y, nx, ny, tx,
    # ty, sr, lr, wt_sr, wt_lr = args

    L = len(parameters) // 2
    num_wts = np.cumsum([0] + connections)

    # reshape 1D vector into appropriate matrices
    for ell in range(1, L + 1):
        shape = parameters["W" + str(ell)].shape
        parameters["W" + str(ell)] = x[num_wts[ell - 1] : num_wts[ell]].reshape(shape)
    # caluclate the forward pass
    AL = nn.linear_forward(Y, parameters, relu)

    # calculate the gridded varaiograms
    gx, gy = griddedgam_2D(AL.reshape(ny, nx))

    # calculate sum of squares
    loss_sr = calculate_loss_2D(
        y_true=[target_x, target_y], y_pred=[gx, gy], i=sr, j=nx - 1
    )
    loss_lr = calculate_loss_2D(
        y_true=[target_x, target_y], y_pred=[gx, gy], i=lr, j=nx - 1
    )

    return loss_sr * wt_sr + loss_lr * wt_lr


def network_lmr_objective_1D(
    x,
    parameters,
    connections,
    Y,
    nx,
    ny,
    target_x,
    ivarios,
    thresholds,
    afunc,
    max_range,
):
    """objective function for network lmr"""

    L = len(parameters) // 2
    num_wts = np.cumsum([0] + connections)

    # reshape 1D vector into appropriate matrices
    for ell in range(1, L + 1):
        shape = parameters["W" + str(ell)].shape
        parameters["W" + str(ell)] = x[num_wts[ell - 1] : num_wts[ell]].reshape(shape)

    # caluclate the forward pass
    AL = nn.linear_forward(Y, parameters, afunc)
    AL_i, ivars = indicator_transform(AL, thresholds.values())

    # calculate the gridded varaiograms
    gx = griddedgam(AL)

    # variogram objective - should probably scale here rather than in obj
    loss_vario = objective_vario(true=target_x, pred=gx, i=0, j=max_range, scale=1.0)
    loss_ivario = objective_ivario(
        ivarios,
        AL_i,
        nx,
        ny,
        i=0,
        j=max_range,
        quantiles=thresholds.keys(),
        ivars=ivars,
        scale=1.0,
    )

    return loss_vario + loss_ivario

