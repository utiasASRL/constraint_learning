from abc import ABC, abstractproperty

import numpy as np

from lifters.state_lifter import StateLifter
from poly_matrix.poly_matrix import PolyMatrix
from utils.common import get_rot_matrix


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
        "u@u",  # ...
        "u2",
        "u@r",
        "uuT",
        "urT",
    ]
    PARAM_LEVELS = ["no", "p", "ppT"]
    MAX_VARS = 4
    # no tightness
    #VARIABLE_LIST = [
    #    ["l", "x"] + [f"z_{i}" for i in range(j)] for j in range(MAX_VARS + 1)
    #]
    # weak tightness
    #VARIABLE_LIST = [
        #["l"] + [f"z_{i}" for i in range(j)] for j in range(1, MAX_VARS + 1)
    #]
    LEVEL_NAMES = {
        "no": "$\\boldsymbol{u}_n$",
        "urT": "$\\boldsymbol{u}\\boldsymbol{t}^\\top_n$",
    }
    VARIABLE_LIST = [
        ["l", "z_0"], 
        ["l", "z_0", "z_1"],
        ["l", "z_0", "z_1", "z_2"],
        ["l", "x"], 
        ["l", "x", "z_0"],
        ["l", "x", "z_0", "z_1"]
    ]
    def __init__(self, n_landmarks, d, level="no", param_level="no", variable_list=None):
        self.d = d
        self.n_landmarks = n_landmarks
        if variable_list is not None:
            self.variable_list = variable_list
        else:
            self.variable_list = self.VARIABLE_LIST
        super().__init__(level=level, param_level=param_level)

    def get_level_dims(self, n=1):
        """
        :param n: number of landmarks to consider
        """
        return {
            "no": 0,
            "u@u": n,  # ...
            "u2": n * self.d,
            "u@r": n,
            "uuT": n * self.d**2,
            "urT": n * self.d**2,
        }

    @abstractproperty
    def M_matrix(self):
        return

    def get_inits(self, n_inits):
        n_angles = self.d * (self.d - 1) / 2
        return np.c_[
            np.random.rand(n_inits, self.d),
            2 * np.pi * np.random.rand(n_inits, n_angles),
        ]

    def generate_random_setup(self):
        self.landmarks = np.random.rand(self.n_landmarks, self.d)
        self.parameters = np.r_[1.0, self.landmarks.flatten()]

    def generate_random_theta(self, factor=1.0):
        n_angles = 1 if self.d == 2 else 3
        return np.r_[
            np.random.rand(self.d) * factor, np.random.rand(n_angles) * 2 * np.pi
        ]

    def get_parameters(self, var_subset=None):
        if var_subset is None:
            var_subset = self.var_dict

        landmarks = self.get_variable_indices(var_subset)
        if self.param_level == "no":
            return [1.0]
        else:
            # row-wise flatten: l_0x, l_0y, l_1x, l_1y, ...
            parameters = self.landmarks[landmarks, :].flatten()
            return np.r_[1.0, parameters]

    def get_x(self, theta=None, parameters=None, var_subset=None):
        """
        :param var_subset: list of variables to include in x vector. Set to None for all.
        """
        if theta is None:
            theta = self.theta
        if parameters is None:
            parameters = self.parameters
        if var_subset is None:
            var_subset = self.var_dict.keys()

        # TODO(FD) below is a bit hacky, these two variables should not both be called theta.
        # theta is either (x, y, alpha) or (x, y, z, a1, a2, a3)
        if len(theta) in [3, 6]:
            C, r = get_C_r_from_theta(theta, self.d)
        # theta is (x, y, z, C.flatten()), technically this should be called xtheta!
        elif len(theta) == 12:
            C, r = get_C_r_from_xtheta(theta, self.d)

        if self.param_level != "no":
            landmarks = np.array(parameters[1:]).reshape((self.n_landmarks, self.d))
        else:
            landmarks = self.landmarks

        x_data = []
        for key in var_subset:
            if key == "l":
                x_data.append(1.0)
            elif key == "x":
                x_data += list(r) + list(C.flatten("F"))  # column-wise flatten
            elif "z" in key:
                j = int(key.split("_")[-1])

                pj = landmarks[j, :]

                zj = C[self.d - 1, :] @ pj + r[self.d - 1]
                u = 1 / zj * np.r_[C[: self.d - 1, :] @ pj + r[: self.d - 1], 1]
                x_data += list(u)

                if self.level == "no":
                    continue
                elif self.level == "u2":
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

        dim_x = self.get_dim_x(var_subset=var_subset)
        assert len(x_data) == dim_x
        return np.array(x_data)

    def get_A_known(self, add_known_redundant=False):
        # C = [ c1
        #       c2
        #       c3 ]
        # [xj]   [c1 @ pj]
        # [yj] = [c2 @ pj]
        # [zj]   [c3 @ pj]
        # enforce that u_xj = 1/zj * xj -> zj*u_xj = c3 @ pj * u_xj = c1 @ pj
        # enforce that u_yj = 1/zj * yj -> zj*u_yj = c3 @ pj * u_yj = c2 @ pj
        print("Warning: get_A_known not adapted to new variables yet.")
        return []

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

    def sample_theta(self, factor=1.0):
        return self.generate_random_theta(factor=factor).flatten()

    def sample_parameters(self):
        if self.param_level == "no":
            return [1.0]
        else:
            parameters = np.random.rand(self.n_landmarks, self.d).flatten()
            return np.r_[1.0, parameters]

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
        M_tilde = np.zeros((self.var_dict["z_0"], self.d))
        M_tilde[: self.M_matrix.shape[0], :] = self.M_matrix[
            :, list(range(self.d - 1)) + [self.d]
        ]

        # in 2d: M[:, 1]
        # in 3d: M[:, 2]
        m = self.M_matrix[:, self.d - 1]

        ls_problem = LeastSquaresProblem()
        for j in range(len(y)):
            ls_problem.add_residual({"l": y[j] - m, f"z_{j}": -M_tilde.T})
        Q = ls_problem.get_Q().get_matrix(self.var_dict)

        # sanity check
        x = self.get_x()
        t = self.theta
        cost_raw = self.get_cost(t, y)
        cost_Q = x.T @ Q.toarray() @ x
        assert abs(cost_raw - cost_Q) < 1e-8, (cost_raw, cost_Q)

        return Q, y

    def __repr__(self):
        level_str = str(self.level).replace(".", "-")
        return f"stereo{self.d}d_{level_str}_{self.param_level}"
