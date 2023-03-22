import numpy as np

from lifters.state_lifter import StateLifter


class RangeOnlyLifter(StateLifter):
    def __init__(self, n_positions, d):
        self.n_positions = n_positions
        self.d = d
        super().__init__(theta_shape=(self.n_positions, d), M=n_positions)

    def generate_random_unknowns(self):
        self.unknowns = np.random.rand(self.n_positions, self.d)

    def get_theta(self):
        return self.unknowns

    def get_x(self, theta=None):
        if theta is None:
            theta = self.get_theta()

        norms = np.linalg.norm(theta, axis=1) ** 2
        t = theta.flatten("C")  # generates t1, t2, ... with t1=[x1,y1]
        np.testing.assert_allclose(t[:2], theta[0, :])

        x = np.r_[1, t, norms]
        assert len(x) == self.N + self.M + 1
        return x


class Poly4Lifter(StateLifter):
    def __init__(self):
        super().__init__(theta_shape=(1,), M=1)

    def generate_random_unknowns(self):
        self.unknowns = np.random.rand(1)

    def get_theta(self):
        return np.r_[self.unknowns]

    def get_x(self, theta=None):
        if theta is None:
            theta = self.get_theta()
        return np.r_[1, theta, theta**2]


class Poly6Lifter(StateLifter):
    def __init__(self):
        super().__init__(theta_shape=(1,), M=2)

    def generate_random_unknowns(self):
        self.unknowns = np.random.rand(1)

    def get_theta(self):
        return np.r_[self.unknowns]

    def get_x(self, theta=None):
        if theta is None:
            theta = self.get_theta()
        return np.r_[1, theta, theta**2, theta**3]
