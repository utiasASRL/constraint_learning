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
        np.testing.assert_allclose(t[: self.d], theta[0, :])

        x = np.r_[1, t, norms]
        assert len(x) == self.N + self.M + 1
        return x

    def __repr__(self):
        return f"rangeonly{self.d}d"


def get_x_poly4(t):
    return np.r_[1, t, t**2]


def get_Q_poly4():
    Q = np.r_[np.c_[2, 1, 0], np.c_[1, -1 / 2, -1 / 3], np.c_[0, -1 / 3, 1 / 4]]
    return Q


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
        return get_x_poly4(theta)

    def get_Q(self, noise=None):
        Q = get_Q_poly4()
        y = None
        return Q, y

    def get_x(self, theta=None):
        if theta is None:
            theta = self.get_theta()
        return get_x_poly4(theta)

    def get_cost(self, t, *args, **kwargs):
        Q = get_Q_poly4()
        x = get_x_poly4(t)
        return x.T @ Q @ x

    def local_solver(self, t0, *args, **kwargs):
        from scipy.optimize import minimize

        sol = minimize(self.get_cost, t0)
        return sol.x[0], sol.success, sol.fun

    def __repr__(self):
        return "poly4"


def get_x_poly6(t):
    return np.r_[1, t, t**2, t**3]


def get_Q_poly6():
    Q = (
        np.r_[
            np.c_[25, 12, 0, 0],
            np.c_[12, -13, -5 / 2, 0],
            np.c_[0, -5 / 2, 25 / 4, -9 / 10],
            np.c_[0, 0, -9 / 10, 1 / 6],
        ]
        / 10
    )
    return Q


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
        return get_x_poly6(theta)

    def get_Q(self, noise=None):
        Q = get_Q_poly6()
        y = None
        return Q, y

    def get_cost(self, t, *args, **kwargs):
        Q = get_Q_poly6()
        x = get_x_poly6(t)
        return x.T @ Q @ x

    def local_solver(self, t0, *args, **kwargs):
        from scipy.optimize import minimize

        sol = minimize(self.get_cost, t0)
        return sol.x[0], sol.success, sol.fun

    def __repr__(self):
        return "poly6"
