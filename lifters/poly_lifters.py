from abc import abstractmethod

import numpy as np

from lifters.state_lifter import StateLifter


class PolyLifter(StateLifter):
    def __init__(self, degree):
        self.degree = degree
        super().__init__(d=1)
        self.parameters = [1.0]

        # TODO (FD): remove requirement for this variable
        self.n_landmarks = 0

    @property
    def var_dict(self):
        if self.var_dict_ is None:
            self.var_dict_ = {"h": 1, "t": 1}
            self.var_dict_.update({f"z{i}": 1 for i in range(self.M)})
        return self.var_dict_

    @property
    def M(self):
        return self.degree // 2 - 1

    def sample_theta(self):
        return np.random.rand(1)

    @property
    def theta(self):
        if self.theta_ is None:
            self.theta_ = self.sample_theta()
        return self.theta_

    @abstractmethod
    def get_Q_mat(self):
        return

    def get_error(self, t):
        return {"MAE": float(abs(self.theta - t))}

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


class Poly4Lifter(PolyLifter):
    VARIABLE_LIST = [["h", "t", "z0"]]

    def __init__(self):
        # actual minimum
        super().__init__(degree=4)

    def get_Q_mat(self):
        Q = np.r_[np.c_[2, 1, 0], np.c_[1, -1 / 2, -1 / 3], np.c_[0, -1 / 3, 1 / 4]]
        return Q

    def get_A_known(self, target_dict=None, output_poly=False):
        from poly_matrix import PolyMatrix

        # z_0 = t^2
        A_1 = PolyMatrix(symmetric=True)
        A_1["h", "z0"] = -1
        A_1["t", "t"] = 2
        if output_poly:
            return [A_1]
        else:
            return [A_1.get_matrix(self.var_dict)]

    def generate_random_setup(self):
        self.theta_ = np.array([-1])

    def get_D(self, that):
        # TODO(FD) generalize and move to PolyLifter
        D = np.array(
            [
                [1.0, 0.0, 0.0],
                [that, 1.0, 0.0],
                [that**2, 2 * that, 1.0],
            ]
        )
        return D


class Poly6Lifter(PolyLifter):
    VARIABLE_LIST = [["h", "t", "z0", "z1"]]

    def __init__(self, poly_type="A"):
        assert poly_type in ["A", "B"]
        self.poly_type = poly_type
        super().__init__(degree=6)

    def get_Q_mat(self):
        if self.poly_type == "A":
            return 0.1 * np.array(
                [
                    [25, 12, 0, 0],
                    [12, -13, -2.5, 0],
                    [0, -2.5, 6.25, -0.9],
                    [0, 0, -0.9, 1 / 6],
                ]
            )
        elif self.poly_type == "B":
            return np.array(
                [
                    [5.0000, 1.3167, -1.4481, 0],
                    [1.3167, -1.4481, 0, 0.2685],
                    [-1.4481, 0, 0.2685, -0.0667],
                    [0, 0.2685, -0.0667, 0.0389],
                ]
            )

    def get_A_known(self, target_dict=None, output_poly=False):
        from poly_matrix import PolyMatrix

        # z_0 = t^2
        A_1 = PolyMatrix(symmetric=True)
        A_1["h", "z0"] = -1
        A_1["t", "t"] = 2

        # z_1 = t^3 = t z_0
        A_2 = PolyMatrix(symmetric=True)
        A_2["h", "z1"] = -1
        A_2["t", "z0"] = 1

        # t^4 = z_1 t = z_0 z_0
        B_0 = PolyMatrix(symmetric=True)
        B_0["z0", "z0"] = 2
        B_0["t", "z1"] = -1
        if output_poly:
            return [A_1, A_2, B_0]
        else:
            return [A_i.get_matrix(self.var_dict) for A_i in [A_1, A_2, B_0]]

    def get_D(self, that):
        # TODO(FD) generalize and move to PolyLifter
        D = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [that, 1.0, 0.0, 0.0],
                [that**2, 2 * that, 1.0, 0.0],
                [that**3, 3 * that**2, 3 * that, 1.0],
            ]
        )
        return D

    def generate_random_setup(self):
        self.theta_ = np.array([-1])
