from abc import abstractmethod

import numpy as np

from .state_lifter import StateLifter


class PolyLifter(StateLifter):
    def __init__(self, degree, param_level="no"):
        self.degree = degree
        super().__init__(d=1, param_level=param_level)

    @property
    def var_dict(self):
        if self.var_dict_ is None:
            self.var_dict_ = {self.HOM: 1, "t": 1}
            self.var_dict_.update({f"z{i}": 1 for i in range(self.M)})
        return self.var_dict_

    @property
    def M(self):
        return self.degree // 2 - 1

    def sample_theta(self):
        return np.random.rand(1)

    @abstractmethod
    def get_Q_mat(self):
        return

    def get_error(self, t):
        return {"MAE": float(abs(self.theta - t)), "error": float(abs(self.theta - t))}

    def get_x(self, theta=None, parameters=None, var_subset=None):
        if theta is None:
            theta = self.theta
        return np.array([theta**i for i in range(self.degree // 2 + 1)])

    def get_Q(self, noise=1e-3):
        Q = self.get_Q_mat()
        return Q, None

    def get_cost(self, theta, *args, **kwargs):
        Q = self.get_Q_mat()
        x = self.get_x(theta)
        return x.T @ Q @ x

    def get_hess(self, *args, **kwargs):
        raise NotImplementedError

    def local_solver(self, t0, *args, **kwargs):
        from scipy.optimize import minimize

        sol = minimize(self.get_cost, t0)
        info = {"success": sol.success}
        return sol.x, info, sol.fun

    def __repr__(self):
        return f"poly{self.degree}"
