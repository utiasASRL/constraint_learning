import itertools
from abc import abstractmethod

import numpy as np

from lifters.state_lifter import StateLifter


class RangeOnlyLifter(StateLifter):
    def __init__(self, n_positions, n_landmarks, d, edges=None):
        self.n_positions = n_positions
        self.n_landmarks = n_landmarks
        self.d = d
        # TODO(FD) replace edges with W
        if edges is None:
            self.edges = list(
                itertools.product(range(n_positions), range(n_landmarks), repeat=1)
            )
        else:
            # TODO(FD) add tests
            self.edges = edges
        super().__init__(theta_shape=(self.get_N(), d), M=self.get_M())

    def generate_random_setup(self):
        self.landmarks = np.random.rand(self.n_landmarks, self.d)

    def generate_random_unknowns(self, replace=True):
        self.positions = np.random.rand(self.n_positions, self.d)
        self.unknowns = self.positions

    def get_positions_and_landmarks(self, t):
        t = t.reshape((-1, self.d))
        return t, self.landmarks

    def get_theta(self):
        return self.unknowns

    def get_Q(self, noise: float = 1e-3) -> tuple:
        # N x K matrix
        y_gt = (
            np.linalg.norm(
                self.landmarks[None, :, :] - self.positions[:, None, :], axis=2
            )
            ** 2
        )
        y = y_gt + np.random.normal(loc=0, scale=noise)
        Q = self.get_Q_from_y(y)
        return Q, y

    def get_grad(self, t, y):
        """get gradient"""
        x = self.get_x(theta=t)

        J = np.empty((x.shape[0], t.size))
        J[: t.size, : t.size] = np.eye(t.size)
        J[t.shape[0] :, :] = self.get_J_lifting(t)
        Q = self.get_Q_from_y(y)
        return -2 * J.T @ Q @ x

    def get_cost(self, t, y):
        """
        get cost for given positions, landmarks and noise.

        :param t: (positions, landmarks) tuple
        """
        if type(t) == tuple:
            positions, landmarks = t
        else:
            positions, landmarks = self.get_positions_and_landmarks(t)

        y_current = (
            np.linalg.norm(landmarks[None, :, :] - positions[:, None, :], axis=2) ** 2
        )
        cost = 0
        for n, k in self.edges:
            cost += (y[n, k] - y_current[n, k]) ** 2
        return cost

    def local_solver(self, t_init, y, verbose=False):
        """
        :param t_init: (positions, landmarks) tuple
        """
        from scipy.optimize import minimize

        t0 = t_init.flatten("C")

        # for testing only
        t_init_test = t0.reshape((-1, self.d))
        np.testing.assert_allclose(t_init_test, t_init)

        sol = minimize(self.get_cost, x0=t0, args=y, jac=self.get_grad)
        that = sol.x.reshape((-1, self.d))
        msg = sol.message
        cost = sol.fun
        return that, msg, cost

    @abstractmethod
    def get_x(self):
        pass

    @abstractmethod
    def get_var_dict(self):
        pass

    @abstractmethod
    def get_Q_from_y(self, y):
        pass

    @abstractmethod
    def get_J_lifting(self, t):
        pass

    @abstractmethod
    def get_M(self):
        pass

    @abstractmethod
    def get_N(self):
        pass


class RangeOnlySLAM1Lifter(RangeOnlyLifter):
    """Range-only SLAM, version 1

    Uses substitution tau_i=||t_i||^2, alpha_k=||a_k||^2, e_ik = a_k @ t_i
    """

    def __init__(self, n_positions, n_landmarks, d, edges=None):
        super().__init__(n_positions, n_landmarks, d, edges=edges)

    def get_M(self):
        return self.n_positions + self.n_landmarks + len(self.edges)

    def get_N(self):
        return self.n_positions + self.n_landmarks

    def get_var_dict(self):
        var_dict = {"l": 1}
        var_dict.update({f"t:{n}": self.d for n in range(self.n_positions)})
        var_dict.update({f"a:{k}": self.d for k in range(self.n_landmarks)})
        var_dict.update({f"tau:{n}": 1 for n in range(self.n_positions)})
        var_dict.update({f"alpha:{k}": 1 for k in range(self.n_landmarks)})
        var_dict.update({f"e:{n}{k}": 1 for n, k in self.edges})
        return var_dict

    def get_positions_and_landmarks(self, t):
        t = t.reshape((-1, self.d))
        positions = t[: self.n_positions, :]
        landmarks = t[self.n_positions :, :]
        assert positions.shape[0] == self.n_positions
        assert landmarks.shape[0] == self.n_landmarks
        return positions, landmarks

    def get_x(self, theta=None):
        if theta is None:
            positions, landmarks = self.get_theta()
        elif type(theta) == tuple:
            positions, landmarks = theta
        else:
            positions = theta[: self.n_positions, :]
            landmarks = theta[self.n_positions :, :]

        x_data = [[1]]
        x_data += [p for p in positions]
        x_data += [a for a in landmarks]
        x_data += [[np.linalg.norm(p) ** 2] for p in positions]
        x_data += [[np.linalg.norm(a) ** 2] for a in landmarks]
        x_data += [[landmarks[k] @ positions[n]] for n, k in self.edges]

        x = np.concatenate(x_data, axis=0)
        assert len(x) == self.N + self.M + 1
        return x

    def get_Q_from_y(self, y):
        from poly_matrix.cost_matrix import CostMatrix

        Q = CostMatrix()
        for n, k in self.edges:
            Q.add_from_residual(
                {"l": y[n, k], f"tau:{n}": -1, f"alpha:{k}": -1, f"e:{n}{k}": 2}
            )
        return Q.get_matrix(self.get_var_dict())

    def generate_random_unknowns(self, replace=True):
        positions = np.random.rand(self.n_positions, self.d)
        unknowns = (positions, self.landmarks)
        if replace:
            self.positions = positions
            self.unknowns = unknowns
        return unknowns

    def get_J_lifting(self, t):
        t = t.reshape((-1, self.d))
        positions = t[: self.n_positions, :]
        landmarks = t[self.n_positions :, :]

        J_lifting = np.zeros((self.M, self.N))
        counter = 0
        for n in range(self.n_positions):
            J_lifting[counter, counter * self.d : (counter + 1) * self.d] = (
                2 * positions[n]
            )
            counter += 1
        for k in range(self.n_landmarks):
            J_lifting[counter, counter * self.d : (counter + 1) * self.d] = (
                2 * landmarks[k]
            )
            counter += 1
        for n, k in self.edges:
            J_lifting[counter, n * self.d : (n + 1) * self.d] = 2 * landmarks[k]
            J_lifting[
                counter,
                (self.n_positions + k) * self.d : (self.n_positions + k + 1) * self.d,
            ] = (
                2 * positions[n]
            )
            counter += 1
        return J_lifting

    def __repr__(self):
        return f"rangeonlyslam1{self.d}d"


class RangeOnlyLocLifter(RangeOnlyLifter):
    """Range-only localization

    Uses substitution z_i=||x_i||^2
    """

    def __init__(self, n_positions, n_landmarks, d, edges=None):
        super().__init__(n_positions, n_landmarks, d, edges=edges)

    def get_M(self):
        return self.n_positions

    def get_x(self, theta=None):
        if theta is None:
            theta = self.get_theta()
        if theta.ndim == 1:
            theta = theta.reshape((-1, self.d))

        norms = np.linalg.norm(theta, axis=1) ** 2
        t = theta.flatten("C")  # generates t1, t2, ... with t1=[x1,y1]
        np.testing.assert_allclose(t[: self.d], theta[0, :])

        x = np.r_[1, t, norms]
        assert len(x) == self.N + self.M + 1
        return x

    def __repr__(self):
        return f"rangeonly{self.d}d"

    def get_Q_from_y(self, y):
        from poly_matrix.cost_matrix import CostMatrix

        Q = CostMatrix()
        for n, k in self.edges:
            ak = self.landmarks[k]
            Q.add_from_residual(
                {
                    "l": y[n, k] - np.linalg.norm(ak) ** 2,
                    f"x{n}": 2 * ak.reshape((-1, 1)),
                    f"z{n}": -1,
                }
            )
        return Q.get_matrix(self.get_var_dict())

    def get_J_lifting(self, t):
        # theta
        t = t.reshape((-1, self.d))
        J_lifting = np.zeros((self.M, self.N))
        for n in range(self.n_positions):
            J_lifting[n, n * self.d : (n + 1) * self.d] = -2 * t[n]
        return J_lifting

    def get_var_dict(self):
        var_dict = {"l": 1}
        var_dict.update({f"x{n}": self.d for n in range(self.n_positions)})
        var_dict.update({f"z{n}": 1 for n in range(self.n_positions)})
        return var_dict

    def get_N(self):
        return self.n_positions

    def __repr__(self):
        return f"rangeonlyloc{self.d}d"
