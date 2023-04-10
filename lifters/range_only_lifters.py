import itertools
from abc import ABC, abstractmethod

import numpy as np

from lifters.state_lifter import StateLifter
from poly_matrix.least_squares_problem import LeastSquaresProblem

# How to deal with Gauge freedom.
# - None: do not remove it
# - "hard": remove variables alltogether
# - "cost": add cost on a0 (attempting to make it 0)
# recommended is "hard"
# REMOVE_GAUGE = None
REMOVE_GAUGE = "hard"
# REMOVE_GAUGE = "cost"


class RangeOnlyLifter(StateLifter):
    """Lifters for range-only localization and SLAM problems.

    Some clarifications on names:
    - generate_random_landmarks: always fix
        - first one at 0, (a0 = [a0_x, a0_y, a0_z] = [0, 0, 0])
        - second one along x, (a1_y = 0, a1_z = 0)
        - third along positive y (a2_z = 0, a2_y > 0) } fix the "flip" in 2d, rotation in 3d
        - fourth with positive z axis  (a3_z > 0)     } fix rotation in 3d
        --> in 2d, will optimize for landmarks_theta = [a1_x, a2_x, a2_y, ...]
        --> in 3d, will optimize for landmarks_theta = [a1_x, a2_x, a2_y, a3_x, a3_y, a3_z, ...]
        saved in self.landmarks
    - generate_random_positions:
        return randomly generated positions (motion model to be implemented)

    - sample_feasible:
        - for localization problem, this can regenerate both landmarks and positions
        - for SLAM problem, regenerate only positions (to be tested)

    theta: vector of unknowns [positions, (landmarks_theta for SLAM, see above)]
    x: [1, theta, lifting_functions]
        where lifting_functions contains:
        - ||t_n||^2 for RangeOnlyLoc
        - ||t_n||^2, ||a_k||^2, t_n.T@a_k for RangeOnlySLAM1
        - ||a_k - t_n||^2 for RangeOnlySLAM2
    """

    def __init__(
        self, n_positions, n_landmarks, d, edges=None, remove_gauge=REMOVE_GAUGE
    ):
        self.remove_gauge = remove_gauge
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
        super().__init__(theta_shape=(self.N, d), M=self.M)

        np.random.seed(1)
        self.generate_random_landmarks()
        self.generate_random_positions()

    def generate_random_landmarks(self):
        self.landmarks = np.random.rand(self.n_landmarks, self.d)

        if self.remove_gauge is not None:
            self.landmarks[0, :] = 0.0
            self.landmarks[1, 1] = 0  # set a1_y = 0
            if self.d == 3:
                self.landmarks[1, 2] = 0  # set a1_z = 0
                self.landmarks[2, 2] = 0  # set a2_z = 0

            # TODO(FD) figure out if below is necessary (it shouldn't hurt anyways)
            # make sure to also fix the flip
            if self.d == 2:
                # make sure a2_y > 0.
                if self.landmarks[2, 1] < 0:  #
                    self.landmarks[:, 1] = -self.landmarks[:, 1]
            elif self.d == 3:
                # landmarks 0, 1, 2 now live in the x-y plane.
                if self.landmarks[3, 2] < 0:
                    self.landmarks[:, 2] = -self.landmarks[:, 2]

    def generate_random_positions(self):
        # TODO(FD) implement motion model?
        self.positions = np.random.rand(self.n_positions, self.d)

    def get_positions_and_landmarks(self, theta):
        """
        --> in 2d, will optimize for landmarks_theta = [a1_x, a2_x, a2_y, ...]
        --> in 3d, will optimize for landmarks_theta = [a1_x, a2_x, a2_y, a3_x, a3_y, a3_z, ...]
        """
        N = self.n_positions * self.d
        positions = theta[:N].reshape((-1, self.d))

        if len(theta) > N:
            if self.remove_gauge == "hard":
                # range-only SLAM
                landmarks = np.empty((self.n_landmarks, self.d))
                landmarks[0, :] = 0.0
                landmarks[1, 0] = theta[N]
                landmarks[1, 1] = 0.0
                if self.d == 2:
                    landmarks[2:, :] = theta[N + 1 :].reshape(
                        (self.n_landmarks - 2, self.d)
                    )
                elif self.d == 3:
                    landmarks[1, 2] = 0
                    landmarks[2, 0] = theta[N + 1]
                    landmarks[2, 1] = theta[N + 2]
                    landmarks[3:, :] = theta[N + 3 :].reshape(
                        (self.n_landmarks - 2, self.d)
                    )
            else:
                landmarks = theta[N:].reshape((-1, self.d))
            return positions, landmarks
        else:
            # range-only localization
            return positions, self.landmarks

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

    def get_J(self, t, y):
        J = np.empty((self.dim_x, self.N))
        J[0, :] = 0  # corresponds to homogenization variable
        J[1 : self.N + 1, : self.N + 1] = np.eye(self.N)  # corresponds to theta
        J[self.N + 1 :, :] = self.get_J_lifting(t)  # corresponds to lifting
        return J

    def get_hess(self, t, y):
        """get Hessian"""
        x = self.get_x(t)
        Q = self.get_Q_from_y(y)
        J = self.get_J(t, y)
        hess = 2 * J.T @ Q @ J

        hessians = self.get_hess_lifting(t)
        B = self.ls_problem.get_B_matrix(self.var_dict)
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
        positions, landmarks = self.get_positions_and_landmarks(t)

        y_current = (
            np.linalg.norm(landmarks[None, :, :] - positions[:, None, :], axis=2) ** 2
        )
        cost = 0
        for n, k in self.edges:
            cost += (y[n, k] - y_current[n, k]) ** 2
        return cost

    def local_solver(self, t_init, y, tol=1e-10, verbose=False):
        """
        :param t_init: (positions, landmarks) tuple
        """
        from scipy.optimize import minimize

        sol = minimize(
            self.get_cost,
            x0=t_init,
            args=y,
            jac=self.get_grad,
            method="Newton-CG",
            tol=tol,
            options={"disp": verbose, "xtol": tol},
        )
        that = sol.x
        rel_error = self.get_cost(that, y) - self.get_cost(sol.x, y)
        assert abs(rel_error) < 1e-10, rel_error
        msg = sol.message + f" in {sol.nit} iterations"
        cost = sol.fun
        return that, msg, cost

    @abstractmethod
    def sample_feasible(self):
        pass

    @abstractmethod
    def get_Q_from_y(self, y):
        pass

    @abstractmethod
    def get_J_lifting(self, t):
        pass

    @abstractmethod
    def get_x(self, theta=None):
        return

    @property
    def theta(self):
        return self.positions.flatten("C")

    @property
    def theta(self):
        """
        for SLAM:
        --> in 2d, will optimize for landmarks_theta = [a1_x, a2_x, a2_y, ...]
        --> in 3d, will optimize for landmarks_theta = [a1_x, a2_x, a2_y, a3_x, a3_y, a3_z, ...]
        """
        theta = list(self.positions.flatten("C"))
        if self.remove_gauge == "hard":
            theta.append(self.landmarks[1, 0])
            if self.d == 2:
                theta += list(self.landmarks[2:, :].flatten("C"))
            elif self.d == 3:
                theta.append(self.landmarks[2, 0])
                theta.append(self.landmarks[2, 1])
                theta += list(self.landmarks[3:, :].flatten("C"))
        else:
            theta += list(self.landmarks.flatten("C"))
        return np.array(theta)

    @property
    def var_dict(self):
        pass


class RangeOnlySLAM1Lifter(RangeOnlyLifter):
    """Range-only SLAM, version 1

    Uses substitution tau_i=||t_i||^2, alpha_k=||a_k||^2, e_ik = a_k @ t_i
    """

    def __init__(self, n_positions, n_landmarks, d, edges=None):
        super().__init__(n_positions, n_landmarks, d, edges=edges)

    @property
    def M(self):
        return self.n_positions + self.n_landmarks + len(self.edges)

    @property
    def N(self):
        if self.remove_gauge == "hard":
            if self.d == 2:
                # a1_x, a2, a3, ...
                return self.n_positions * self.d + (self.n_landmarks - 2) * self.d + 1
            else:
                # a1_x, a2_x, a2_y, a3, a4, ...
                return self.n_positions * self.d + (self.n_landmarks - 3) * self.d + 3
        else:
            return (self.n_positions + self.n_landmarks) * self.d

    def get_base_var_dict(self):
        var_dict = {"l": 1}
        var_dict.update({f"x{n}": self.d for n in range(self.n_positions)})
        if self.remove_gauge == "hard":
            if self.d == 2:
                # a1_x, a2, a3, ...
                var_dict.update({"a1": 1})
                var_dict.update({f"a{k}": self.d for k in range(2, self.n_landmarks)})
            else:
                # a1_x, a2_x, a2_y, a3, a4, ...
                var_dict.update({"a1": 1})
                var_dict.update({"a2": 2})
                var_dict.update({f"a{k}": 1 for k in range(3, self.n_landmarks)})
        else:
            var_dict.update({f"a{k}": self.d for k in range(self.n_landmarks)})
        return var_dict

    @property
    def var_dict(self):
        var_dict = self.get_base_var_dict()
        var_dict.update({f"tau{n}": 1 for n in range(self.n_positions)})
        var_dict.update({f"alpha{k}": 1 for k in range(self.n_landmarks)})
        var_dict.update({f"e{n}{k}": 1 for n, k in self.edges})
        return var_dict

    def get_x(self, theta=None):
        if theta is None:
            theta = self.theta
        positions, landmarks = self.get_positions_and_landmarks(theta)

        x_data = [[1]]
        x_data += [list(theta)]
        x_data += [[np.linalg.norm(p) ** 2] for p in positions]
        x_data += [[np.linalg.norm(a) ** 2] for a in landmarks]
        x_data += [[landmarks[k] @ positions[n]] for n, k in self.edges]

        x = np.concatenate(x_data, axis=0)
        assert len(x) == self.N + self.M + 1
        return x

    def generate_random_setup(self):
        self.generate_random_landmarks()

    def sample_feasible(self):
        self.generate_random_positions()

    def get_Q_from_y(self, y):
        self.ls_problem = LeastSquaresProblem()
        for n, k in self.edges:
            self.ls_problem.add_residual(
                # d_nk**2 - ||t_n||**2 + 2t_n@a_k - ||a_k||**2
                #   l         tau_n        e_nk        alpha_k
                {"l": y[n, k], f"tau{n}": -1, f"alpha{k}": -1, f"e{n}{k}": 2}
            )
        # fix Gauge freedom
        if self.remove_gauge == "cost":
            I = np.eye(self.d)
            for d in range(self.d):
                self.ls_problem.add_residual({"a0": I[d].reshape((1, -1))})
        return self.ls_problem.get_Q().get_matrix(self.var_dict)

    def get_cost(self, t, y):
        # fix Gauge freedom
        cost = super().get_cost(t, y)
        if self.remove_gauge == "cost":
            cost += np.linalg.norm(self.landmarks[0]) ** 2
        return cost

    def sample_feasible(self):
        self.generate_random_positions()

    def fill_depending_on_k(self, J_lifting, counter, k, vec):
        """Because of Gauge freedom removal,
        the first columns of the landmark-based part of J
        are incomplete.
        """
        if k == 1:
            i = self.n_positions * self.d
            J_lifting[counter, i] = vec[0]
        elif k == 2:
            i = self.n_positions * self.d + 1
            J_lifting[counter, i : i + 2] = vec[:2]
        elif k == 3:
            i = self.n_positions * self.d + 3
            J_lifting[counter, i : i + self.d] = vec
        else:
            i = self.n_positions * self.d + 3 + (k - 4) * self.d
            J_lifting[counter, i : i + self.d] = vec

    def get_J_lifting(self, t):
        positions, landmarks = self.get_positions_and_landmarks(t)

        J_lifting = np.zeros((self.M, self.N))
        counter = 0
        for n in range(self.n_positions):
            i = n * self.d
            J_lifting[counter, i : i + self.d] = 2 * positions[n]
            counter += 1
        for k in range(self.n_landmarks):
            if self.remove_gauge == "hard":
                self.fill_depending_on_k(J_lifting, counter, k, 2 * landmarks[k, :])
            else:
                i = (self.n_positions + k) * self.d
                J_lifting[counter, i : i + self.d] = 2 * landmarks[k]
            counter += 1
        for n, k in self.edges:
            i = n * self.d
            J_lifting[counter, i : i + self.d] = landmarks[k]
            if self.remove_gauge == "hard":
                self.fill_depending_on_k(J_lifting, counter, k, positions[n])
            else:
                i = (self.n_positions + k) * self.d
                J_lifting[counter, i : i + self.d] = positions[n]
            counter += 1
        return J_lifting

    def get_hess_lifting(self, t):
        """return list of the hessians of the M lifting functions."""
        if self.remove_gauge == "hard":
            raise NotImplementedError(
                "get_hess_lifting without Gauge freedom is not implemented yet"
            )

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

    @property
    def M(self):
        return len(self.edges)

    @property
    def var_dict(self):
        var_dict = self.get_base_var_dict()
        var_dict.update({f"e{n}{k}": 1 for n, k in self.edges})
        return var_dict

    def get_x(self, theta=None):
        if theta is None:
            theta = self.theta
        positions, landmarks = self.get_positions_and_landmarks(theta)

        x_data = [[1]]
        x_data += [list(theta)]
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
        # fix Gauge freedom
        if self.remove_gauge == "cost":
            I = np.eye(self.d)
            for d in range(self.d):
                self.ls_problem.add_residual({"a0": I[d].reshape((1, -1))})
        return self.ls_problem.get_Q().get_matrix(self.var_dict)

    def get_J_lifting(self, t):
        positions, landmarks = self.get_positions_and_landmarks(t)

        J_lifting = np.zeros((self.M, self.N))
        for i, (n, k) in enumerate(self.edges):
            delta = landmarks[k] - positions[n]
            J_lifting[i, n * self.d : (n + 1) * self.d] = -2 * delta

            if self.remove_gauge == "hard":
                self.fill_depending_on_k(J_lifting, i, k, delta)
            else:
                start = self.n_positions * self.d + k * self.d
                J_lifting[i, start : start + self.d] = 2 * delta
        return J_lifting

    def get_hess_lifting(self, t):
        if self.remove_gauge == "hard":
            raise NotImplementedError(
                "get_hess_lifting without Gauge freedom is not implemented yet"
            )

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
        # there is no Gauge freedom in range-only localization!
        super().__init__(n_positions, n_landmarks, d, edges=edges, remove_gauge=None)

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
        return self.ls_problem.get_Q().get_matrix(self.var_dict)

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

    def sample_feasible(self):
        self.generate_random_positions()
        self.generate_random_landmarks()

    def generate_random_setup(self):
        print("nothing to setup.")

    def get_x(self, theta=None):
        if theta is None:
            theta = self.theta
        positions, landmarks = self.get_positions_and_landmarks(theta)
        norms = np.linalg.norm(positions, axis=1) ** 2

        x = np.r_[1, theta, norms]
        assert len(x) == self.N + self.M + 1
        return x

    @property
    def var_dict(self):
        var_dict = {"l": 1}
        var_dict.update({f"x{n}": self.d for n in range(self.n_positions)})
        var_dict.update({f"z{n}": 1 for n in range(self.n_positions)})
        return var_dict

    @property
    def theta(self):
        return self.positions.flatten("C")

    @property
    def N(self):
        return self.n_positions * self.d

    @property
    def M(self):
        return self.n_positions

    def __repr__(self):
        return f"rangeonlyloc{self.d}d"


if __name__ == "__main__":
    lifter = RangeOnlyLocLifter(n_positions=3, n_landmarks=4, d=2)
    # lifter = RangeOnlySLAM1Lifter(n_positions=3, n_landmarks=4, d=2)
    lifter.run()
