from abc import abstractmethod

import numpy as np

from lifters.state_lifter import StateLifter


class PolyLifter(StateLifter):
    def __init__(self, degree):
        self.degree = degree
        super().__init__(theta_shape=(1,), M=self.M)

    @property
    def M(self):
        return self.degree // 2 - 1

    @property
    def theta(self):
        return self.theta_

    @abstractmethod
    def get_Q_mat(self):
        return

    def sample_feasible(self):
        return np.random.rand(1)

    def get_x(self, t=None):
        if t is None:
            t = self.theta
        return np.array([t**i for i in range(self.degree // 2 + 1)])

    def get_Q(self, noise=1e-3):
        Q = self.get_Q_mat()
        return Q, None

    def get_cost(self, t, *args, **kwargs):
        Q = self.get_Q_mat()
        x = self.get_x(t)
        return x.T @ Q @ x

    def local_solver(self, t0, *args, **kwargs):
        from scipy.optimize import minimize

        sol = minimize(self.get_cost, t0)
        return sol.x[0], sol.success, sol.fun

    def __repr__(self):
        return f"poly{self.degree}"


class Poly4Lifter(PolyLifter):
    def __init__(self):
        # actual minimum
        super().__init__(degree=4)

    def get_Q_mat(self):
        Q = np.r_[np.c_[2, 1, 0], np.c_[1, -1 / 2, -1 / 3], np.c_[0, -1 / 3, 1 / 4]]
        return Q

    def generate_random_setup(self):
        self.theta_ = np.array([-1])


class Poly6Lifter(PolyLifter):
    def __init__(self):
        super().__init__(degree=6)

    def get_Q_mat(self):
        return (
            np.r_[
                np.c_[25, 12, 0, 0],
                np.c_[12, -13, -5 / 2, 0],
                np.c_[0, -5 / 2, 25 / 4, -9 / 10],
                np.c_[0, 0, -9 / 10, 1 / 6],
            ]
            / 10
        )

    def generate_random_setup(self):
        self.theta_ = np.array([-1])
