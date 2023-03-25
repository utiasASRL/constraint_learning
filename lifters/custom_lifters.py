import numpy as np

from lifters.state_lifter import StateLifter


class RangeOnlyLifter(StateLifter):
    def __init__(self, n_landmarks, d):
        self.n_landmarks = n_landmarks
        self.d = d
        super().__init__(theta_shape=(self.n_landmarks, d), M=n_landmarks)

    def generate_random_unknowns(self, replace=True):
        unknowns = np.random.rand(self.n_landmarks, self.d)
        if replace:
            self.unknowns = unknowns
        return unknowns

    def get_theta(self):
        return self.unknowns

    def get_x(self, theta=None):
        if theta is None:
            theta = self.get_theta()

        norms = np.linalg.norm(theta, axis=1) ** 2
        t = theta.flatten("C")  # generates t1, t2, ... with t1=[x1,y1]
        np.testing.assert_allclose(t[:self.d], theta[0, :])

        x = np.r_[1, t, norms]
        assert len(x) == self.N + self.M + 1
        return x

    def __repr__(self):
        return f"rangeonly{self.d}d"


class Poly4Lifter(StateLifter):
    def __init__(self):
        super().__init__(theta_shape=(1,), M=1)

    def generate_random_unknowns(self, replace=True):
        unknowns = np.random.rand(1)
        if replace:
            self.unknowns = unknowns
        return unknowns

    def get_theta(self):
        return np.r_[self.unknowns]

    def get_x(self, theta=None):
        if theta is None:
            theta = self.get_theta()
        return np.r_[1, theta, theta**2]

    def __repr__(self):
        return "poly4"


class Poly6Lifter(StateLifter):
    def __init__(self):
        super().__init__(theta_shape=(1,), M=2)

    def generate_random_unknowns(self, replace=True):
        unknowns = np.random.rand(1)
        if replace:
            self.unknowns = unknowns
        return unknowns

    def get_theta(self):
        return np.r_[self.unknowns]

    def get_x(self, theta=None):
        if theta is None:
            theta = self.get_theta()
        return np.r_[1, theta, theta**2, theta**3]

    def __repr__(self):
        return "poly6"
