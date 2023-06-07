from abc import ABC, abstractproperty

import numpy as np

from lifters.state_lifter import StateLifter
from poly_matrix.poly_matrix import PolyMatrix
from utils import get_rot_matrix


def get_C_r_from_theta(theta, d):
    r = theta[:d]
    alpha = theta[d:]
    C = get_rot_matrix(alpha)
    return C, r


def get_C_r_from_xtheta(xtheta, d):
    C = xtheta[: d**2].reshape((d, d))
    r = xtheta[-d:]
    return C, r


def get_T(xtheta, d):
    C, r = get_C_r_from_xtheta(xtheta, d)
    T = np.zeros((d + 1, d + 1))
    T[:d, :d] = C
    T[:d, d] = r
    T[-1, -1] = 1.0
    return T


def get_xtheta_from_theta(theta, d):
    pos = theta[:d]
    alpha = theta[d:]
    C = get_rot_matrix(alpha)
    c = C.flatten("C")  # row-wise flatten
    theta = np.r_[c, pos]
    return theta


def get_xtheta_from_T(T):
    # T is either 4x4 or 3x3 matrix.
    C = T[:-1, :-1]
    r = T[:-1, -1]
    return np.r_[C.flatten("C"), r]  # row-wise


class StereoLifter(StateLifter, ABC):
    """General lifter for stereo localization problem.

    Naming convention:
    - theta is the actual number of unknowns, so 6 in 3d or 3 in 2d.
    - xtheta is the vector [position, vec(C)], with C the rotation matrix
    """

    LEVELS = [
        "no",
        "r@r",  # x**2 + y**2
        "r2",  # x**2, y**2
        "rrT",  # x**2, y**2, xy
        "u@u",  # ...
        "u2",
        "u@r",
        "uuT",
        "urT",
        "urT-diag",
        "urT-off",
    ]

    def __init__(self, n_landmarks, d, level="no"):
        assert level in self.LEVELS, f"level ({level}) not in {self.LEVELS}"
        self.d = d
        self.level = level
        self.n_landmarks = n_landmarks

        M = self.n_landmarks * self.d  # u, v, z (3d) or u, z (2d)
        L = self.get_level_dims(n=self.n_landmarks)[level]

        self.theta_ = self.generate_random_theta()
        self.var_dict_ = None

        super().__init__(theta_shape=(self.d**2 + self.d,), M=M, L=L)

    def get_level_dims(self, n=1):
        """
        :param n: number of landmarks to consider
        """
        return {
            "no": 0,
            "r@r": 1,  # x**2 + y**2
            "r2": self.d,  # x**2, y**2
            "rrT": self.d**2,  # x**2, y**2, xy
            "u@u": n,  # ...
            "u2": n * self.d,
            "u@r": n,
            "uuT": n * self.d**2,
            "urT": n * self.d**2,
            "urT-diag": n * self.d,
            "urT-off": n * int(self.d * (self.d - 1) / 2),
        }

    @abstractproperty
    def M_matrix(self):
        return

    @property
    def theta(self):
        return self.theta_

    @property
    def base_var_dict(self):
        var_dict = {}
        var_dict.update({f"c_{i}": self.d for i in range(self.d)})
        var_dict.update({"t": self.d})
        return var_dict

    @property
    def sub_var_dict(self):
        var_dict = {f"z_{k}": self.d for k in range(self.n_landmarks)}
        level_dim = self.get_level_dims()[self.level]
        if "u" in self.level:
            var_dict.update({f"y_{i}": level_dim for i in range(self.n_landmarks)})
        elif level_dim > 0:
            var_dict.update({f"y": level_dim})
        return var_dict

    @property
    def var_dict(self):
        if self.var_dict_ is None:
            self.var_dict_ = {"l": 1}
            self.var_dict_.update(self.base_var_dict)
            self.var_dict_.update(self.sub_var_dict)
        return self.var_dict_

    def get_inits(self, n_inits):
        n_angles = self.d * (self.d - 1) / 2
        return np.c_[
            np.random.rand(n_inits, self.d),
            2 * np.pi * np.random.rand(n_inits, n_angles),
        ]

    def generate_random_setup(self):
        # important!! we can only resample x, the landmarks have to stay the same
        # [(x, y, alpha), (landmarks)]
        self.landmarks = np.random.rand(self.n_landmarks, self.d)

    def generate_random_theta(self):
        n_angles = int(self.d * (self.d - 1) / 2)
        return np.r_[np.random.rand(self.d), np.random.rand(n_angles) * 2 * np.pi]

    def get_x(self, theta=None, var_subset=None):
        """
        :param var_subset: list of variables to include in x vector. Set to None for all.
        """
        if theta is None:
            theta = self.theta
        if var_subset is None:
            var_subset = self.var_dict.keys()

        # TODO(FD) below is a bit hacky, these two variables should not both be called theta.
        # theta is either (x, y, alpha) or (x, y, z, a1, a2, a3)
        if len(theta) in [3, 6]:
            C, r = get_C_r_from_theta(theta, self.d)
        # theta is (x, y, z, C.flatten()), technically this should be called xtheta!
        elif len(theta) == 12:
            C, r = get_C_r_from_xtheta(theta, self.d)

        x_data = []
        for key in var_subset:
            if key == "l":
                x_data.append(1.0)
            elif key == "t":
                x_data += list(r)
            elif "c" in key:
                # c = C.flatten("C")
                d = int(key.split("_")[-1])
                x_data += list(C[d, :])

            elif key == "y":
                if self.level == "r2":
                    x_data += list(r**2)
                if self.level == "r@r":
                    x_data += [r @ r]
                elif self.level == "rrT":
                    x_data += list(np.outer(r, r).flatten())
            elif "z" in key:
                j = int(key.split("_")[-1])
                pj = self.landmarks[j, :]
                zj = C[self.d - 1, :] @ pj + r[self.d - 1]
                u = 1 / zj * np.r_[C[: self.d - 1, :] @ pj + r[: self.d - 1], 1]
                x_data += list(u)

            elif "y" in key:
                j = int(key.split("_")[-1])
                pj = self.landmarks[j]
                zj = C[self.d - 1, :] @ pj + r[self.d - 1]
                u = 1 / zj * np.r_[C[: self.d - 1, :] @ pj + r[: self.d - 1], 1]
                if self.level == "u2":
                    x_data += list(u**2)
                elif self.level == "u@u":
                    x_data += [u @ u]
                elif self.level == "u@r":
                    x_data += [u @ r]
                elif self.level == "uuT":
                    x_data += list(np.outer(u, u).flatten())
                elif self.level == "urT":
                    # this works
                    x_data += list(np.outer(u, r).flatten())
                elif self.level == "urT-off":
                    # this works
                    x_data += list(np.outer(u, r)[np.triu_indices(self.d - 1)])
                elif self.level == "urT-diag":
                    # this works
                    x_data += list(np.diag(np.outer(u, r)))

        dim_x = self.get_dim(var_subset=var_subset)
        assert len(x_data) == dim_x
        return np.array(x_data)

    def get_A_known(self):
        # C = [ c1
        #       c2
        #       c3 ]
        # [xj]   [c1 @ pj]
        # [yj] = [c2 @ pj]
        # [zj]   [c3 @ pj]
        # enforce that u_xj = 1/zj * xj -> zj*u_xj = c3 @ pj * u_xj = c1 @ pj
        # enforce that u_yj = 1/zj * yj -> zj*u_yj = c3 @ pj * u_yj = c2 @ pj
        A_known = []
        for k in range(self.n_landmarks):
            for j in range(self.d):
                A = PolyMatrix()
                # x contains: [c1, c2, c3, t]
                fill_mat = np.zeros((self.d, self.d))
                fill_mat[:, j] = self.landmarks[k]
                A[f"c_{self.d-1}", f"z_{k}"] = fill_mat

                fill_mat = np.zeros((self.d, self.d))
                fill_mat[-1, j] = 1.0
                A[f"t", f"z_{k}"] = fill_mat

                if j < self.d - 1:  # u, (v)
                    A["l", f"c_{j}"] = -self.landmarks[k].reshape((1, -1))
                    fill_mat = -np.eye(self.d)[j]
                    A["l", "t"] = fill_mat.reshape((1, -1))
                elif j == self.d - 1:  # z
                    A["l", "l"] = -2.0
                A_known.append(A.get_matrix(self.var_dict))
        return A_known

    def sample_theta(self):
        return self.generate_random_theta().flatten()

    def get_Q(self, noise: float = 1e-3) -> tuple:
        return self._get_Q(noise=noise)

    def _get_Q(self, noise: float = 1e-3) -> tuple:
        from poly_matrix.least_squares_problem import LeastSquaresProblem

        xtheta = get_xtheta_from_theta(self.theta, self.d)
        T = get_T(xtheta, self.d)

        y = []
        for j in range(self.n_landmarks):
            y_gt = T @ np.r_[self.landmarks[j], 1.0]

            # in 2d: y_gt[1]
            # in 3d: y_gt[2]
            y_gt /= y_gt[self.d - 1]
            y_gt = self.M_matrix @ y_gt
            y.append(y_gt + np.random.normal(loc=0, scale=noise))

        # in 2d: M[:, [0, 2]]
        # in 3d: M[:, [0, 1, 3]]
        M_tilde = self.M_matrix[:, list(range(self.d - 1)) + [self.d]]

        # in 2d: M[:, 1]
        # in 3d: M[:, 2]
        m = self.M_matrix[:, self.d - 1]

        ls_problem = LeastSquaresProblem()
        for j in range(len(y)):
            ls_problem.add_residual({"l": y[j] - m, f"z_{j}": -M_tilde})
        return ls_problem.get_Q().get_matrix(self.var_dict), y

    def get_grad(self, t, y):
        raise NotImplementedError("get_grad not implement yet")

    def __repr__(self):
        level_str = str(self.level).replace(".", "-")
        return f"stereo{self.d}d_{level_str}"
