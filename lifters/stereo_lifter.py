from abc import ABC, abstractproperty

import numpy as np

from lifters.state_lifter import StateLifter
from utils import get_rot_matrix


def get_C_r_from_xtheta(xtheta, d):
    C = xtheta[: d**2].reshape((d, d)).T
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
    c = C.flatten("F")  # column-wise flatten
    theta = np.r_[c, pos]
    return theta


def get_theta_from_T(T):
    # T is either 4x4 or 3x3 matrix.
    C = T[:-1, :-1]
    r = T[:-1, -1]
    return np.r_[C.flatten("F"), r]


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
    ]

    def __init__(self, n_landmarks, d, level="no"):
        assert level in self.LEVELS, f"level ({level}) not in {self.LEVELS}"
        self.d = d
        self.level = level
        self.n_landmarks = n_landmarks

        M = self.n_landmarks * self.d
        L = self.get_level_dims(n=self.n_landmarks)[level]

        self.theta_ = self.generate_random_theta()

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
        }

    @abstractproperty
    def M_matrix(self):
        return

    @property
    def theta(self):
        return self.theta_

    @property
    def var_dict(self):
        var_dict_ = {"l": 1}
        var_dict_["x"] = self.theta_shape[0]
        var_dict_.update({f"z{i}": self.d for i in range(self.n_landmarks)})

        level_dim = self.get_level_dims()[self.level]
        if "u" in self.level:
            var_dict_.update({f"y{i}": level_dim for i in range(self.n_landmarks)})
        else:
            if level_dim > 0:
                var_dict_.update({f"y": level_dim})
        return var_dict_

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

    def get_x(self, theta=None):
        if theta is None:
            theta = self.theta
        xtheta = get_xtheta_from_theta(theta, self.d)
        C, r = get_C_r_from_xtheta(xtheta, self.d)

        x_data = [1] + list(xtheta)

        higher_data = []
        if self.level == "r2":
            higher_data += list(r**2)
        if self.level == "r@r":
            higher_data += [r @ r]
        elif self.level == "rrT":
            higher_data += list(np.outer(r, r).flatten())

        for j in range(self.n_landmarks):
            pj = self.landmarks[j, :]
            zj = C[self.d - 1, :] @ pj + r[self.d - 1]
            u = 1 / zj * np.r_[C[: self.d - 1, :] @ pj + r[: self.d - 1], 1]
            x_data += list(u)

            if self.level == "u2":
                higher_data += list(u**2)
            if self.level == "u@u":
                higher_data += [u @ u]
            elif self.level == "u@r":
                higher_data += [u @ r]
            elif self.level == "uuT":
                higher_data += list(np.outer(u, u).flatten())
            elif self.level == "urT":
                # this works
                higher_data += list(np.outer(u, r).flatten())
        x_data += higher_data
        assert len(x_data) == self.dim_x
        return np.array(x_data)

    def sample_feasible(self):
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
            ls_problem.add_residual({"l": y[j] - m, f"z{j}": -M_tilde})
        return ls_problem.get_Q().get_matrix(self.var_dict), y

    def __repr__(self):
        level_str = str(self.level).replace(".", "-")
        return f"stereo{self.d}d_{level_str}"
