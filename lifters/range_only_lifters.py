import itertools
from abc import abstractmethod

import numpy as np

from lifters.state_lifter import StateLifter
from poly_matrix.least_squares_problem import LeastSquaresProblem


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
        Q = self.get_Q_from_y(y).toarray()
        return Q, y

    def get_J(self, t, y):
        t = t.reshape((-1, self.d))
        J = np.empty((self.dim_X(), self.N))
        J[0, :] = 0
        J[1 : self.N + 1, : self.N + 1] = np.eye(self.N)
        J[self.N + 1 :, :] = self.get_J_lifting(t)
        return J

    def get_hess(self, t, y):
        """get Hessian"""
        x = self.get_x(t)
        Q = self.get_Q_from_y(y)
        J = self.get_J(t, y)
        hess = 2 * J.T @ Q @ J

        hessians = self.get_hess_lifting(t)
        B = self.ls_problem.get_B_matrix(self.get_var_dict(only_lifters=False))
        residuals = B @ x
        for m, h in enumerate(hessians):
            bm_tilde = B[:, -self.M + m]
            factor = bm_tilde.T @ residuals
            hess += 2 * factor * h
        return hess

    def get_grad(self, t, y):
        """get gradient"""
        J = self.get_J(t, y)
        x = self.get_x(t)
        Q = self.get_Q_from_y(y)
        return 2 * J.T @ Q @ x

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
        np.testing.assert_allclose(t0[: self.d], t_init[0, :])
        t_init_test = t0.reshape((-1, self.d))
        np.testing.assert_allclose(t_init_test, t_init)

        sol = minimize(self.get_cost, x0=t0, args=y, jac=self.get_grad)
        that = sol.x.reshape((-1, self.d))

        rel_error = self.get_cost(that, y) - self.get_cost(sol.x, y)
        assert abs(rel_error) < 1e-10, rel_error
        msg = sol.message + f" in {sol.nit} iterations"
        cost = sol.fun
        return that, msg, cost

    @abstractmethod
    def get_x(self):
        pass

    @abstractmethod
    def get_var_dict(self, only_lifters=False):
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

    def get_var_dict(self, only_lifters=False):
        if not only_lifters:
            var_dict = {"l": 1}
            var_dict.update({f"x{n}": self.d for n in range(self.n_positions)})
            var_dict.update({f"a{k}": self.d for k in range(self.n_landmarks)})
        else:
            var_dict = {}
        var_dict.update({f"tau{n}": 1 for n in range(self.n_positions)})
        var_dict.update({f"alpha{k}": 1 for k in range(self.n_landmarks)})
        var_dict.update({f"e{n}{k}": 1 for n, k in self.edges})
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
            if np.ndim(theta) == 1:
                theta = theta.reshape((-1, self.d))
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
        self.ls_problem = LeastSquaresProblem()
        for n, k in self.edges:
            self.ls_problem.add_residual(
                # d_nk**2 - ||t_n||**2 + 2t_n@a_k - ||a_k||**2
                #   l         tau_n        e_nk        alpha_k
                {"l": y[n, k], f"tau{n}": -1, f"alpha{k}": -1, f"e{n}{k}": 2}
            )
        return self.ls_problem.get_Q().get_matrix(self.get_var_dict())

    def generate_random_unknowns(self, replace=True):
        positions = np.random.rand(self.n_positions, self.d)
        unknowns = (positions, self.landmarks)
        if replace:
            self.positions = positions
            self.unknowns = unknowns
        return unknowns

    def get_J_lifting(self, t):
        positions, landmarks = self.get_positions_and_landmarks(t)

        J_lifting = np.zeros((self.M, self.N))
        counter = 0
        for n in range(self.n_positions):
            i = n * self.d
            J_lifting[counter, i : i + self.d] = 2 * positions[n]
            counter += 1
        for k in range(self.n_landmarks):
            i = (self.n_positions + k) * self.d
            J_lifting[counter, i : i + self.d] = 2 * landmarks[k]
            counter += 1
        for n, k in self.edges:
            i = n * self.d
            J_lifting[counter, i : i + self.d] = landmarks[k]
            i = (self.n_positions + k) * self.d
            J_lifting[counter, i : i + self.d] = positions[n]
            counter += 1
        return J_lifting

    def get_hess_lifting(self, t):
        """return list of the hessians of the M lifting functions."""
        hessians = []
        for j in range(self.M):
            hessian = np.zeros((self.N, self.N))
            i = j * self.d
            if j < self.n_positions:
                hessian[range(i, i + self.d), range(i, i + self.d)] = 2
            elif j < self.n_landmarks + self.n_positions:
                hessian[range(i, i + self.d), range(i, i + self.d)] = 2
            else:
                n, k = self.edges[j - self.n_landmarks - self.n_positions]
                i = n * self.d
                j = (self.n_positions + k) * self.d
                hessian[range(i, i + self.d), range(j, j + self.d)] = 2
                hessian[range(j, j + self.d), range(i, i + self.d)] = 2
            hessians.append(hessian)
        return hessians

    def __repr__(self):
        return f"rangeonlyslam1-{self.d}d"


class RangeOnlySLAM2Lifter(RangeOnlySLAM1Lifter):
    """Range-only SLAM, version 1

    Uses substitutions e_ik = ||a_k - t_i||
    """

    def __init__(self, n_positions, n_landmarks, d, edges=None):
        super().__init__(n_positions, n_landmarks, d, edges=edges)

    def get_M(self):
        return len(self.edges)

    def get_N(self):
        return self.n_positions + self.n_landmarks

    def get_var_dict(self, only_lifters=False):
        if not only_lifters:
            var_dict = {"l": 1}
            var_dict.update({f"x{n}": self.d for n in range(self.n_positions)})
            var_dict.update({f"a{k}": self.d for k in range(self.n_landmarks)})
        else:
            var_dict = {}
        var_dict.update({f"e{n}{k}": 1 for n, k in self.edges})
        return var_dict

    def get_x(self, theta=None):
        if theta is None:
            positions, landmarks = self.get_theta()
        elif type(theta) == tuple:
            positions, landmarks = theta
        else:
            if np.ndim(theta) == 1:
                theta = theta.reshape((-1, self.d))
            positions = theta[: self.n_positions, :]
            landmarks = theta[self.n_positions :, :]

        x_data = [[1]]
        x_data += [p for p in positions]
        x_data += [a for a in landmarks]
        x_data += [
            [np.linalg.norm(landmarks[k] - positions[n]) ** 2] for n, k in self.edges
        ]
        x = np.concatenate(x_data, axis=0)
        assert len(x) == self.N + self.M + 1
        return x

    def get_Q_from_y(self, y):
        self.ls_problem = LeastSquaresProblem()
        for n, k in self.edges:
            self.ls_problem.add_residual({"l": y[n, k], f"e{n}{k}": -1})
        return self.ls_problem.get_Q().get_matrix(self.get_var_dict())

    def get_J_lifting(self, t):
        positions, landmarks = self.get_positions_and_landmarks(t)

        J_lifting = np.zeros((self.M, self.N))
        for i, (n, k) in enumerate(self.edges):
            delta = landmarks[k] - positions[n]
            J_lifting[i, n * self.d : (n + 1) * self.d] = -2 * delta
            start = self.n_positions * self.d + k * self.d
            J_lifting[i, start : start + self.d] = 2 * delta
        return J_lifting

    def get_hess_lifting(self, t):
        """return list of the hessians of the M lifting functions."""
        hessians = []
        for j, (n, k) in enumerate(self.edges):
            hessian = np.zeros((self.N, self.N))
            i = n * self.d
            j = (self.n_positions + k) * self.d
            hessian[range(i, i + self.d), range(i, i + self.d)] = 2
            hessian[range(j, j + self.d), range(j, j + self.d)] = 2
            hessian[range(i, i + self.d), range(j, j + self.d)] = 2
            hessian[range(j, j + self.d), range(i, i + self.d)] = 2
            hessians.append(hessian)
        return hessians

    def __repr__(self):
        return f"rangeonlyslam2-{self.d}d"


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

    def get_Q_from_y(self, y):
        self.ls_problem = LeastSquaresProblem()
        for n, k in self.edges:
            ak = self.landmarks[k]
            self.ls_problem.add_residual(
                {
                    "l": y[n, k] - np.linalg.norm(ak) ** 2,
                    f"x{n}": 2 * ak.reshape((1, -1)),
                    f"z{n}": -1,
                }
            )
        return self.ls_problem.get_Q().get_matrix(self.get_var_dict())

    def get_J_lifting(self, t):
        # theta
        t = t.reshape((-1, self.d))
        J_lifting = np.zeros((self.M, self.N))
        for n in range(self.n_positions):
            J_lifting[n, n * self.d : (n + 1) * self.d] = 2 * t[n]
        return J_lifting

    def get_hess_lifting(self, t):
        """return list of the hessians of the M lifting functions."""
        hessians = []
        for n in range(self.M):
            hessian = np.zeros((self.N, self.N))
            i = n * self.d
            hessian[range(i, i + self.d), range(i, i + self.d)] = 2
            hessians.append(hessian)
        return hessians

    def get_var_dict(self, only_lifters=False):
        if not only_lifters:
            var_dict = {"l": 1}
            var_dict.update({f"x{n}": self.d for n in range(self.n_positions)})
        else:
            var_dict = {}
        var_dict.update({f"z{n}": 1 for n in range(self.n_positions)})
        return var_dict

    def get_N(self):
        return self.n_positions

    def __repr__(self):
        return f"rangeonlyloc{self.d}d"
