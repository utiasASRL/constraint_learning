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


class StereoLifter(StateLifter):
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

    def get_level_dims(self, n):
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

    def __init__(self, n_landmarks, d, level="no"):
        assert level in self.LEVELS, f"level ({level}) not in {self.LEVELS}"
        self.d = d
        self.level = level
        self.n_landmarks = n_landmarks

        M = self.n_landmarks * self.d
        level_dims = self.get_level_dims(n=self.n_landmarks)
        L = level_dims[level]

        self.theta_ = self.generate_random_theta()

        super().__init__(theta_shape=(self.d**2 + self.d,), M=M, L=L)

    @property
    def theta(self):
        return self.theta_

    def generate_random_setup(self):
        # important!! we can only resample x, the landmarks have to stay the same
        # [(x, y, alpha), (landmarks)]
        self.landmarks = np.random.rand(self.n_landmarks, self.d)

    def generate_random_theta(self, n=1):
        n_angles = int(self.d * (self.d - 1) / 2)
        return np.r_[
            np.random.rand(n * self.d), np.random.rand(n * n_angles) * 2 * np.pi
        ]

    def sample_feasible(self):
        return self.generate_random_theta(n=1).flatten()

    def get_C_r_from_theta(self, theta=None):
        if theta is None:
            theta = self.theta
        xtheta = self.get_xtheta_from_theta(theta)
        return get_C_r_from_xtheta(xtheta, self.d)

    def get_xtheta_from_theta(self, theta=None):
        if theta is None:
            theta = self.theta
        return get_xtheta_from_theta(theta, self.d)

    def get_xtheta(self):
        xtheta = self.get_xtheta_from_theta(theta=None)
        assert len(xtheta) == self.theta_shape[0]
        return xtheta

    def get_x(self, theta=None):
        if theta is None:
            theta = self.theta
        C, r = self.get_C_r_from_theta(theta)

        xtheta = self.get_xtheta_from_theta(theta)
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

    def get_vec_around_gt(self, delta):
        """
        param delta_gt:
        - float: sample from gt + std(delta_gt) (set to 0 to start from gt.)
        """
        t_gt = self.theta
        t_0 = t_gt + np.random.normal(scale=delta, loc=0, size=len(t_gt))
        return t_0

    @property
    def var_dict(self):
        var_dict_ = {"l": 1}
        var_dict_["x"] = self.theta_shape[0]
        var_dict_.update({f"z{i}": self.d for i in range(self.n_landmarks)})

        level_dim = self.get_level_dims(n=1)[self.level]
        if "u" in self.level:
            var_dict_.update({f"y{i}": level_dim for i in range(self.n_landmarks)})
        else:
            if level_dim > 0:
                var_dict_.update({f"y": level_dim})
        return var_dict_

    def get_T(self, theta=None):
        if theta is None:
            theta = self.theta
        xtheta = self.get_xtheta_from_theta(theta)
        return get_T(d=self.d, xtheta=xtheta)

    def _get_Q(self, M: np.ndarray, noise: float = 1e-3) -> tuple:
        from poly_matrix.poly_matrix import PolyMatrix

        T = self.get_T()

        y = []
        for j in range(self.n_landmarks):
            y_gt = T @ np.r_[self.landmarks[j], 1.0]

            # in 2d: y_gt[1]
            # in 3d: y_gt[2]
            y_gt /= y_gt[self.d - 1]
            y_gt = M @ y_gt
            y.append(y_gt + np.random.normal(loc=0, scale=noise))

        # in 2d: M[:, [0, 2]]
        # in 3d: M[:, [0, 1, 3]]
        M_tilde = M[:, list(range(self.d - 1)) + [self.d]]

        # in 2d: M[:, 1]
        # in 3d: M[:, 2]
        m = M[:, self.d - 1]

        Q = PolyMatrix()
        M_tilde_sq = M_tilde.T @ M_tilde
        for j in range(len(y)):
            Q["l", "l"] += np.linalg.norm(y[j] - m) ** 2
            Q[f"z{j}", "l"] += -(y[j] - m).T @ M_tilde
            Q[f"z{j}", f"z{j}"] += M_tilde_sq
            # Q[f"y{j}", "l"] = 0.5 * np.diag(M_tilde_sq)
        return Q.get_matrix(self.var_dict), y

    def __repr__(self):
        level_str = str(self.level).replace(".", "-")
        return f"stereo{self.d}d_{level_str}"
