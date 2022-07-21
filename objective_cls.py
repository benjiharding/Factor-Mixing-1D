import numpy as np

from network import vector_to_matrices, linear_forward
from sequences_objective import binary_runs, n_pt_conn
from variogram_objective import (
    indicator_transform,
    update_variogram,
    get_vario_model_points,
    expvario_loss,
    inv_dist,
)

"""
Objective function class

__call__(self, x, args) method is the function to be minimized
"""


class Objective(object):
    def __init__(self, Y):
        # pool of Gaussian factors to be mixed
        self.Y = Y

    def __call__(self, x, args):  # starargs for scipy
        # returns scalar objective
        return self.network_lmr_objective(x, *args)

    def objective_vario(self, AL, targets, lag_dicts, lags, scale):
        """continuous variogram objective function"""
        tx, ty, tz = targets
        xlag, ylag, zlag = lag_dicts
        lx, ly, lz = lags
        xvario = update_variogram(AL, xlag)
        yvario = update_variogram(AL, ylag)
        zvario = update_variogram(AL, zlag)
        xlag_wts = inv_dist(lx, 1.0)
        ylag_wts = inv_dist(ly, 1.0)
        zlag_wts = inv_dist(lz, 1.0)
        sill = np.var(AL)
        objv = 0.0
        objv += expvario_loss(xvario, tx, lx, xlag_wts, sill)
        objv += expvario_loss(yvario, ty, ly, ylag_wts, sill)
        objv += expvario_loss(zvario, tz, lz, zlag_wts, sill)
        return objv * scale

    def objective_ivario(self, AL, targets, lag_dicts, lags, thresholds, scale):
        """continuous variogram objective function"""
        xlag, ylag, zlag = lag_dicts
        lx, ly, lz = lags
        xlag_wts = inv_dist(lx, 1.0)
        ylag_wts = inv_dist(ly, 1.0)
        zlag_wts = inv_dist(lz, 1.0)
        AL_i, ivars = indicator_transform(AL, thresholds.values())
        nq = len(thresholds)
        objv = 0.0
        for j, q in enumerate(thresholds):
            tx, ty, tz = targets[q]
            sill = ivars[j]
            xivario = update_variogram(AL_i[:, j], xlag)
            yivario = update_variogram(AL_i[:, j], ylag)
            zivario = update_variogram(AL_i[:, j], zlag)
            objv += expvario_loss(xivario, tx, lx, xlag_wts, sill)
            objv += expvario_loss(yivario, ty, ly, ylag_wts, sill)
            objv += expvario_loss(zivario, tz, lz, zlag_wts, sill)
        return objv * scale

    def objective_histogram(self, hist, target, scale):
        """relative mse between histogram bins"""
        objv = np.sum(((hist - target)) ** 2) / len(hist)
        return objv * scale

    def objective_runs(
        self, AL, target, thresholds, runs_above, maxrun, dhids, dh_lens, scale
    ):
        """cumulative run frequency objective fucntion"""
        AL_i, ivars = indicator_transform(AL, thresholds.values())
        nq = len(thresholds)
        objv = 0.0
        for j, q in enumerate(thresholds):
            # X = AL_i[:, j].reshape(ncomps, ndh, order="F")
            X = self._reshape_vector(AL_i[:, j], dhids, dh_lens, -1)
            temp_runs = np.zeros(maxrun)
            ndh = len(set(dhids))
            for n in range(ndh):
                x = X[:, n]
                x = x[x > -1]
                run_freqs = binary_runs(x, runs_above)["cum_runs_freqs"][:maxrun]
                temp_runs += np.pad(run_freqs, (0, maxrun - len(run_freqs)))
            objv += self.objective_histogram(temp_runs, target[q], scale)  # * (1 / nq)
        return objv

    def objective_npoint(self, AL, target, thresholds, nstep, dhids, dh_lens, scale):
        """n-point connectivity objective fucntion"""
        AL_i, ivars = indicator_transform(AL, thresholds.values())
        objv = 0.0
        for j, q in enumerate(thresholds):
            X = self._reshape_vector(AL_i[:, j], dhids, dh_lens, -1)
            npoint = n_pt_conn(X, nstep)
            objv += self.objective_histogram(npoint, target[q], scale)
        return objv

    def network_lmr_objective(
        self,
        x,
        parameters,
        connections,
        target_vario,
        target_ivario,
        target_runs,
        target_npoint,
        lag_dicts,
        lags,
        maxrun,
        nstep,
        dhids,
        dh_lens,
        runs_above,
        thresholds,
        afunc,
        objscale,
        vario,
        ivario,
        runs,
        npoint,
    ):
        """objective function for network lmr"""

        # caluclate the forward pass
        params = vector_to_matrices(parameters, connections, x)
        AL = linear_forward(self.Y, params, afunc)

        # initialize objective value
        objv = 0.0

        # continuous variogram
        if vario:
            objv += self.objective_vario(AL, target_vario, lag_dicts, lags, objscale[0])

        # indicator variogram
        if ivario:
            objv += self.objective_ivario(
                AL, target_ivario, lag_dicts, lags, thresholds, objscale[1]
            )

        # cumulative runs
        if runs:
            objv += self.objective_runs(
                AL,
                target_runs,
                thresholds,
                runs_above,
                maxrun,
                dhids,
                dh_lens,
                objscale[2],
            )

        # npoint connectivity runs
        if npoint:
            objv += self.objective_npoint(
                AL, target_npoint, thresholds, nstep, dhids, dh_lens, objscale[3]
            )

        return objv

    def init_vario(
        self, targets, lag_dicts, lags, connections, parameters, bounds, afunc, rng,
    ):
        """initialize continuous variogram objective function"""
        AL = self._random_forward_pass(parameters, afunc, bounds, connections, rng)
        return self.objective_vario(AL, targets, lag_dicts, lags, scale=1.0)

    def init_ivario(
        self,
        targets,
        lag_dicts,
        lags,
        thresholds,
        connections,
        parameters,
        bounds,
        afunc,
        rng,
    ):
        """initialize continuous variogram objective function"""
        AL = self._random_forward_pass(parameters, afunc, bounds, connections, rng)
        return self.objective_ivario(
            AL, targets, lag_dicts, lags, thresholds, scale=1.0
        )

    def init_runs(
        self,
        target,
        maxrun,
        dhids,
        dh_lens,
        thresholds,
        runs_above,
        connections,
        parameters,
        bounds,
        afunc,
        rng,
    ):
        """initialize runs objective"""
        AL = self._random_forward_pass(parameters, afunc, bounds, connections, rng)
        return self.objective_runs(
            AL, target, thresholds, runs_above, maxrun, dhids, dh_lens, scale=1.0
        )

    def init_npoint(
        self,
        target,
        thresholds,
        nstep,
        dhids,
        dh_lens,
        connections,
        parameters,
        bounds,
        afunc,
        rng,
    ):
        """initialize n-point connectivity objective"""
        AL = self._random_forward_pass(parameters, afunc, bounds, connections, rng)
        return self.objective_npoint(
            AL, target, thresholds, nstep, dhids, dh_lens, scale=1.0
        )

    def objective_scaling(
        self,
        connections,
        parameters,
        bounds,
        lag_dicts,
        lags,
        thresholds,
        runs_above,
        maxrun,
        nstep,
        dhids,
        dh_lens,
        afunc,
        maxpert,
        seed,
        vario=True,
        vario_target=None,
        ivario=True,
        ivario_target=None,
        runs=True,
        runs_target=None,
        npoint=True,
        npoint_target=None,
    ):
        """scale objective function components"""

        rng = np.random.default_rng(seed)

        objinit = np.zeros(4)
        objscale = np.ones(4)
        objdelta = np.zeros(4)

        # initalize objective values
        if vario:
            objinit[0] = self.init_vario(
                vario_target,
                lag_dicts,
                lags,
                connections,
                parameters,
                bounds,
                afunc,
                rng,
            )
        if ivario:
            objinit[1] = self.init_ivario(
                ivario_target,
                lag_dicts,
                lags,
                thresholds,
                connections,
                parameters,
                bounds,
                afunc,
                rng,
            )
        if runs:
            objinit[2] = self.init_runs(
                runs_target,
                maxrun,
                dhids,
                dh_lens,
                thresholds,
                runs_above,
                connections,
                parameters,
                bounds,
                afunc,
                rng,
            )
        if npoint:
            objinit[3] = self.init_npoint(
                npoint_target,
                thresholds,
                nstep,
                dhids,
                dh_lens,
                connections,
                parameters,
                bounds,
                afunc,
                rng,
            )

        for m in range(maxpert):

            # caluclate the forward pass
            AL = self._random_forward_pass(parameters, afunc, bounds, connections, rng)

            # approximate objective contributions
            if vario:
                temp_obj_vario = self.objective_vario(
                    AL, vario_target, lag_dicts, lags, scale=objscale[0]
                )
                if temp_obj_vario < 0.0:
                    temp_obj_vario = objinit[0]
                objdelta[0] += np.abs(objinit[0] - temp_obj_vario)

            if ivario:
                temp_obj_ivario = self.objective_ivario(
                    AL, ivario_target, lag_dicts, lags, thresholds, scale=objscale[1],
                )
                if temp_obj_ivario < 0.0:
                    temp_obj_ivario = objinit[1]
                objdelta[1] += np.abs(objinit[1] - temp_obj_ivario)

            if runs:
                temp_obj_runs = self.objective_runs(
                    AL,
                    runs_target,
                    thresholds,
                    runs_above,
                    maxrun,
                    dhids,
                    dh_lens,
                    scale=objscale[2],
                )
                if temp_obj_runs < 0.0:
                    temp_obj_runs = objinit[2]
                objdelta[2] += np.abs(objinit[2] - temp_obj_runs)

            if npoint:
                temp_obj_npt = self.objective_npoint(
                    AL,
                    npoint_target,
                    thresholds,
                    nstep,
                    dhids,
                    dh_lens,
                    scale=objscale[3],
                )
                if temp_obj_npt < 0.0:
                    temp_obj_npt = objinit[3]
                objdelta[3] += np.abs(objinit[3] - temp_obj_npt)

        # scale objective components
        if vario:
            objscale[0] = maxpert / objdelta[0]
        if ivario:
            objscale[1] = maxpert / objdelta[1]
        if runs:
            objscale[2] = maxpert / objdelta[2]
        if npoint:
            objscale[3] = maxpert / objdelta[3]

        # rescale factor
        rescale = 0.0
        for objv, scl in zip(objinit, objscale):
            rescale += scl * objv
        rescale = 1 / max(rescale, 1e-10)
        objscale *= rescale

        return objscale

    def _random_forward_pass(self, parameters, afunc, bounds, connections, rng):
        """random pass through network for initialization"""
        x = rng.uniform(bounds[0], bounds[1], size=np.sum(connections))
        params = vector_to_matrices(parameters, connections, x)
        return linear_forward(self.Y, params, afunc)

    def _reshape_vector(self, vector, dhids, dh_lens, fill_value):
        """reshape mixture vector into dh matrix"""

        uids = set(dhids)
        ndh = len(uids)
        maxdhlen = max(dh_lens)

        X = np.full((maxdhlen, ndh), fill_value)

        for i, uid in enumerate(uids):

            idxs = dhids == uid
            X[: dh_lens[i], i] = vector[idxs]

        return X
