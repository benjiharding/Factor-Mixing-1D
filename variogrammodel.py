import numpy as np
import pandas as pd
from scipy.spatial import distance
from rotation import rotmat, rot_from_matrix, pt_rot_from_matrix


class VariogramModel(object):
    def __init__(
        self,
        vargstr=None,
        nst=None,
        c0=None,
        it=None,
        cc=None,
        ang1=None,
        ang2=None,
        ang3=None,
        ahmax=None,
        ahmin=None,
        avert=None,
        parsestring=True,
    ):
        self.nst = nst
        self.c0 = c0
        self.it = it
        self.cc = cc
        self.ang1 = ang1
        self.ang2 = ang2
        self.ang3 = ang3
        self.ahmax = ahmax
        self.ahmin = ahmin
        self.avert = avert
        self.vargstr = vargstr
        self.rotmat = None
        self.cmax = None
        if self.vargstr is not None and parsestring:
            self.parsestr()

    def parsestr(self, vargstr=None):
        """
        Parses a GSLIB-style variogram model string

        .. codeauthor:: Jared Deutsch - 2015-02-04
        """
        if vargstr is None:
            vargstr = self.vargstr
        # Split into separate lines
        varglines = vargstr.splitlines()
        # Read nugget effect + nst line
        self.nst = int(varglines[0].split()[0])
        self.c0 = float(varglines[0].split()[1])
        # Read structures
        self.it = []
        self.cc = []
        self.ang1 = []
        self.ang2 = []
        self.ang3 = []
        self.ahmax = []
        self.ahmin = []
        self.avert = []
        for st in range(self.nst):
            self.it.append(int(varglines[1 + st * 2].split()[0]))
            self.cc.append(float(varglines[1 + st * 2].split()[1]))
            self.ang1.append(float(varglines[1 + st * 2].split()[2]))
            self.ang2.append(float(varglines[1 + st * 2].split()[3]))
            self.ang3.append(float(varglines[1 + st * 2].split()[4]))
            self.ahmax.append(float(varglines[2 + st * 2].split()[0]))
            self.ahmin.append(float(varglines[2 + st * 2].split()[1]))
            self.avert.append(float(varglines[2 + st * 2].split()[2]))

        self.cmax = self.c0 + np.sum(self.cc)
        self.setcova()

    def setcova(self):
        # initialize the rotation matrix
        self.rotmat = np.ones((self.nst, 3, 3))

        # determine anisotropies
        self.anis1 = [0.0] * self.nst
        self.anis2 = [0.0] * self.nst

        for st in range(self.nst):
            self.anis1[st] = self.ahmin[st] / self.ahmax[st]
            self.anis2[st] = self.avert[st] / self.ahmax[st]

        # determine the rotation matrix for each structure
        for st in range(self.nst):
            self.rotmat[st, :, :] = rotmat(
                self.ang1[st],
                self.ang2[st],
                self.ang3[st],
                self.anis1[st],
                self.anis2[st],
            )

    def pairwisecova(self, points):
        """pairwise covariance between ``points``"""

        # nst is the number of nested structures
        # it is a vector of variogram model types
        # cc is a vector of variance contributions where c0 + sum(cc) = sill
        # aa is a vector of ranges for each nested structure
        # rotmat is 3D where the frame index is nst

        EPS = 1e-5

        assert isinstance(points, np.ndarray), "`points` must be a `np.ndarray`"
        n = points.shape[0]
        pwcova = np.zeros((n, n))
        dmat = distance.cdist(points, points)

        for i in range(self.nst):

            # anisotropic distance matrix for current structure
            rot_xyz = rot_from_matrix(
                points[:, 0], points[:, 1], points[:, 2], self.rotmat[i],
            )

            # spherical
            if self.it[i] == 1:
                h = dmat / self.ahmax[i]
                pwcova = pwcova + self.cc[i] * np.where(
                    h < 1, (1 - h * (1.5 - 0.5 * h ** 2)), 0
                )

            # exponential
            if self.it[i] == 2:
                h = dmat / self.ahmax[i]
                pwcova = pwcova + self.cc[i] * np.exp(-3 * h)

            # Gaussian
            if self.it[i] == 3:
                h = dmat / self.ahmax[i]
                pwcova = pwcova + self.cc[i] * np.exp(-3 * h ** 2)

        pwcova[dmat < EPS] = self.cmax

        return pwcova

    def getcova(self, x1, y1, z1, x2, y2, z2, variogram=False):
        """point covariance between (x1, y1, z1) and (x2, y2, z2)"""

        EPS = 1e-5
        cova = 0.0

        for i in range(self.nst):

            rx1, ry1, rz1 = pt_rot_from_matrix(x1, y1, z1, self.rotmat[i])
            rx2, ry2, rz2 = pt_rot_from_matrix(x2, y2, z2, self.rotmat[i])

            h = np.sqrt((rx2 - rx1) ** 2 + (ry2 - ry1) ** 2 + (rz2 - rz1) ** 2)

            if h < EPS:
                return 0.0 if variogram else self.cmax

            # spherical
            if self.it[i] == 1:
                hr = h / self.ahmax[i]
                cova = cova + self.cc[i] * np.where(
                    hr < 1, (1 - hr * (1.5 - 0.5 * hr ** 2)), 0
                )

            # exponential
            if self.it[i] == 2:
                hr = h / self.ahmax[i]
                cova = cova + self.cc[i] * np.exp(-3 * hr)

            # Gaussian
            if self.it[i] == 3:
                hr = h / self.ahmax[i]
                cova = cova + self.cc[i] * np.exp(-3 * hr ** 2)

        if variogram:
            return self.cmax - cova
        return cova

    def calcpoints(
        self, azm, dip, lags=None, lagdist=None, nlags=None, epslon=1.0e-5,
    ):
        """
        Calculates variogram model points along the vector specified by the
        azimuth (azm) and dip"""

        # If list of lags not provided, use nlags and lagdist or make a guess!
        if nlags is None and lags is None and lagdist is None:
            nlags = 100
            lagdist = np.max([self.ahmax, self.ahmin, self.avert]) / float(nlags)

        if nlags is not None and lagdist is not None:
            lags = [0.0, epslon]
            lags.extend(
                list(
                    np.arange(lagdist, (nlags + 1) * lagdist, step=lagdist, dtype=float)
                )
            )

        # Always recalculate the rotation matrix in case anything has changed
        self.setcova()

        # Switch to pandas DataFrame
        vmodel = pd.DataFrame({"Distance": lags})

        # Get the offsets
        azm_rad = np.radians(azm)
        dip_rad = np.radians(dip)

        def offset(lag):
            "Return xoff, yoff, zoff values given a lag along azm, dip"
            xoff = np.sin(azm_rad) * np.cos(dip_rad) * lag
            yoff = np.cos(azm_rad) * np.cos(dip_rad) * lag
            zoff = np.sin(dip_rad) * lag
            return (xoff, yoff, zoff)

        # Calculate offset locations
        vmodel["xoff"], vmodel["yoff"], vmodel["zoff"] = zip(
            *vmodel["Distance"].map(offset)
        )

        def runcova(series):
            return self.getcova(
                0.0, 0.0, 0.0, series["xoff"], series["yoff"], series["zoff"]
            )

        vmodel["Cova"] = vmodel.apply(runcova, axis=1)

        # Get the variogram values
        vmodel["Variogram"] = self.cmax - vmodel["Cova"]

        return vmodel[["Distance", "Variogram"]]

