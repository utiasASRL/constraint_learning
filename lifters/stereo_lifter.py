import numpy as np

from lifters.state_lifter import StateLifter
from utils import get_rot_matrix


class StereoLifter(StateLifter):
    def __init__(self, n_landmarks, d, level=0):
        self.d = d
        self.level = level
        self.n_landmarks = n_landmarks
        M = self.n_landmarks * self.d
        if level == 1:
            M += self.n_landmarks * self.d
        elif level > 2:
            M += self.n_landmarks * self.d**2
        super().__init__(theta_shape=(self.d**2 + self.d,), M=M)

    def generate_random_setup(self):
        # important!! we can only resample x, the landmarks have to stay the same
        # [(x, y, alpha), (landmarks)]
        self.landmarks = np.random.rand(self.n_landmarks, self.d)

    def generate_random_unknowns(self):
        n_angles = int(self.d * (self.d - 1) / 2)
        self.unknowns = np.r_[
            np.random.rand(self.d), np.random.rand(n_angles) * 2 * np.pi
        ]

    def get_C_r_from_theta(self, theta=None):
        if theta is None:
            theta = self.get_theta()
        C = theta[: self.d**2].reshape((self.d, self.d)).T
        r = theta[-self.d :]
        return C, r

    def get_theta_from_unknowns(self, unknowns=None):
        if unknowns is None:
            unknowns = self.unknowns
        pos = self.unknowns[: self.d]
        alpha = self.unknowns[self.d :]
        C = get_rot_matrix(alpha)
        c = C.flatten("F")  # column-wise flatten
        theta = np.r_[c, pos]
        return theta

    def get_theta(self):
        theta = self.get_theta_from_unknowns()
        assert len(theta) == self.theta_shape[0]
        return theta

    def get_x(self, theta=None):
        Ctest = None
        if theta is None:
            theta = self.get_theta()
            Ctest = get_rot_matrix(self.unknowns[2])
        elif len(theta) < self.theta_shape[0]:
            theta = self.get_theta_from_unknowns(unknowns=theta)

        C, r = self.get_C_r_from_theta(theta)
        if Ctest is not None:
            np.testing.assert_allclose(Ctest, C)
        x_data = [1] + list(theta)

        higher_data = []
        for j in range(self.n_landmarks):
            pj = self.landmarks[j, :]
            zj = C[self.d - 1, :] @ pj + r[self.d - 1]
            u = 1 / zj * np.r_[C[: self.d - 1, :] @ pj + r[: self.d - 1], 1]
            x_data += list(u)

            if self.level == 1:
                # this doesn't work
                # higher_data += list(r * u)
                # higher_data += list(u**2)
                higher_data += list(np.outer(u, r)[:, 0])
            elif self.level == 2:
                # this doesn't work
                higher_data += list(np.outer(u, u).flatten())
            elif self.level == 3:
                # this works
                higher_data += list(np.outer(u, r).flatten())
        x_data += higher_data
        assert len(x_data) == self.dim_X()
        return np.array(x_data)

    def get_var_dict(self):
        var_dict = {"l": 1}
        var_dict["x"] = self.theta_shape[0]
        var_dict.update({f"z{i}": self.d for i in range(self.n_landmarks)})
        if self.level == 1:
            var_dict.update({f"y{i}": self.d for i in range(self.n_landmarks)})
        elif self.level > 2:
            var_dict.update({f"y{i}": self.d**2 for i in range(self.n_landmarks)})
        return var_dict

    def get_T(self):
        C, r = self.get_C_r_from_theta()
        T = np.zeros((self.d + 1, self.d + 1))
        T[: self.d, : self.d] = C
        T[: self.d, self.d] = r
        T[-1, -1] = 1.0
        return T
