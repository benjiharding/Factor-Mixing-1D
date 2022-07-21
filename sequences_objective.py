import numpy as np
from numba import njit

"""
Sqeuence related functions
"""


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


# @njit
# def n_pt_conn(X, nstep):
#     """Global n-point connectivity fucntion of binary matrix X
#     X.shape = (len(dh), ndh)
#     """
#     X = np.asarray(X)
#     nx = X.shape[0]
#     ndh = X.shape[1]
#     phi_n = np.zeros(nstep, dtype=np.float64)
#     for n in range(1, nstep + 1):
#         prod = []
#         for k in range(ndh):
#             x = X[:, k]
#             for i in range(nx - n + 1):
#                 idxs = [i] + [j + i for j in range(n)]
#                 a = [x[int(idx)] for idx in idxs]
#                 p = 1
#                 for i in a:
#                     p *= i
#                 prod.append(p)
#         phi_n[n - 1] = np.mean(np.array(prod, dtype=np.float64))
#     return np.asarray(phi_n)


@njit
def n_pt_conn(X, nstep):
    """Global n-point connectivity fucntion of binary matrix X
    X.shape = (len(dh), ndh)
    """
    X = np.asarray(X)
    xmax = X.shape[0]
    ndh = X.shape[1]
    phi_n = np.zeros(nstep, dtype=np.float64)

    for n in range(1, nstep + 1):

        prod = np.zeros((xmax - (n - 1)) * ndh, dtype=np.float64)
        temp_idx = np.array([0] + [j for j in range(n)])

        for k in range(ndh):

            x = X[:, k]
            x = x[x > -1]
            nx = len(x)

            for i in range(nx - n + 1):

                idxs = temp_idx + i
                arr = x[idxs]

                p = 1
                for a in arr:
                    p *= a

                ii = i + (xmax - (n - 1)) * k

                prod[ii] = p

        phi_n[n - 1] = np.mean(prod)

    return phi_n
