import itertools
from abc import abstractmethod

import numpy as np

from lifters.state_lifter import StateLifter

# How to deal with Gauge freedom.
# - None: do not remove it
# - "hard": remove variables alltogether
# - "cost": add cost on a0 (attempting to make it 0)
# recommended is "hard"
# REMOVE_GAUGE = None
REMOVE_GAUGE = "hard"
# REMOVE_GAUGE = "cost"

SOLVER_KWARGS = dict(
    # method="Nelder-Mead",
    # method="BFGS"  # the only one that almost always converges
    method="Powell"
)


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

    - sample_theta:
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
        super().__init__()

    @property
    def theta(self):
        if self.theta_ is None:
            self.theta_ = self.get_theta(self.landmarks, self.positions)
        return self.theta_

    def sample_random_positions(self):
        return np.random.rand(self.n_positions, self.d)

    def generate_random_setup(self):
        self.generate_random_theta()
        self.landmarks = self.sample_random_landmarks()
        self.parameters = np.r_[1.0, self.landmarks.flatten()]

    def generate_random_theta(self):
        self.positions = self.sample_random_positions()

    def get_parameters(self, var_subset=None):
        if var_subset is None:
            var_subset = self.var_dict

        landmarks = self.get_variable_indices(var_subset)
        if self.param_level == "no":
            return [1.0]
        else:
            # row-wise flatten: l_0x, l_0y, l_1x, l_1y, ...
            parameters = self.landmarks[landmarks, :].flatten()
            return np.r_[1.0, parameters]

    def sample_parameters(self):
        if self.param_level == "no":
            return [1.0]
        else:
            parameters = np.random.rand(self.n_landmarks, self.d).flatten()
            return np.r_[1.0, parameters]

    def get_theta(self, landmarks=None, positions=None):
        if landmarks is None:
            landmarks = self.sample_random_landmarks()
        if positions is None:
            positions = self.sample_random_positions()
        theta = list(positions.flatten("C"))
        if self.remove_gauge == "hard":
            theta.append(landmarks[1, 0])
            if self.d == 2:
                theta += list(landmarks[2:, :].flatten("C"))
            elif self.d == 3:
                theta.append(landmarks[2, 0])
                theta.append(landmarks[2, 1])
                theta += list(landmarks[3:, :].flatten("C"))
        else:
            theta += list(landmarks.flatten("C"))
        return np.array(theta)

    def sample_random_landmarks(self):
        landmarks = np.random.rand(self.n_landmarks, self.d)
        if self.remove_gauge is not None:
            landmarks[0, :] = 0.0
            landmarks[1, 1] = 0  # set a1_y = 0
            if self.d == 3:
                landmarks[1, 2] = 0  # set a1_z = 0
                landmarks[2, 2] = 0  # set a2_z = 0

            # TODO(FD) figure out if below is necessary (it shouldn't hurt anyways)
            # make sure to also fix the flip
            if self.d == 2:
                # make sure a2_y > 0.
                if landmarks[2, 1] < 0:  #
                    landmarks[:, 1] = -landmarks[:, 1]
            elif self.d == 3:
                # landmarks 0, 1, 2 now live in the x-y plane.
                if landmarks[3, 2] < 0:
                    landmarks[:, 2] = -landmarks[:, 2]
        return landmarks

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

                # TODO(FD): figure out if it's easier to set to zero or to the actual landmarks.
                landmarks = np.zeros((self.n_landmarks, self.d))
                # landmarks = deepcopy(self.landmarks)

                # landmarks[0, :] = 0.0  # self.landmarks[0, :]
                landmarks[1, 0] = theta[N]
                # landmarks[1, 1] = 0.0  # self.landmarks[1, 1]
                if self.d == 2:
                    landmarks[2:, :] = theta[N + 1 :].reshape(
                        (self.n_landmarks - 2, self.d)
                    )
                elif self.d == 3:
                    # landmarks[1, 2] = 0.0 #self.landmarks[1, 2]
                    landmarks[2, 0] = theta[N + 1]
                    landmarks[2, 1] = theta[N + 2]
                    landmarks[3:, :] = theta[N + 3 :].reshape(
                        (self.n_landmarks - 3, self.d)
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
        y = y_gt + np.random.normal(loc=0, scale=noise, size=y_gt.shape)
        Q = self.get_Q_from_y(y)

        # DEBUGGING
        x = self.get_x()
        cost1 = x.T @ Q @ x
        cost2 = np.sum((y - y_gt) ** 2)
        cost3 = self.get_cost(self.theta, y)
        assert abs(cost1 - cost2) < 1e-10
        assert abs(cost1 - cost3) < 1e-10
        return Q, y

    def get_J(self, t, y):
        import scipy.sparse as sp

        N = self.n_positions * self.n_landmarks

        J = sp.csr_array(
            (np.ones(self.N), (range(1, self.N + 1), range(self.N))),
            shape=(self.dim_x, self.N),
        )
        J[self.N + 1 :, :] = self.get_J_lifting(t)
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
            factor = float(bm_tilde.T @ residuals)
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

    def local_solver(
        self, t_init, y, tol=1e-8, verbose=False, solver_kwargs=SOLVER_KWARGS
    ):
        """
        :param t_init: (positions, landmarks) tuple
        """
        from scipy.optimize import minimize

        sol = minimize(
            self.get_cost,
            x0=t_init,
            args=y,
            jac=self.get_grad,
            # hess=self.get_hess, not used by any solvers.
            **solver_kwargs,
            tol=tol,
            options={"disp": verbose},  # j, "maxfun": 100},
        )
        if sol.success:
            that = sol.x
            rel_error = self.get_cost(that, y) - self.get_cost(sol.x, y)
            assert abs(rel_error) < 1e-10, rel_error
            cost = sol.fun
        else:
            that = cost = None
        msg = sol.message + f"(# iterations: {sol.nit})"
        return that, msg, cost

    @abstractmethod
    def sample_theta(self):
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

    def plot_setup(self, title="setup"):
        import matplotlib.pylab as plt

        if self.d == 3:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
        else:
            fig, ax = plt.subplots()
        ax.scatter(*self.landmarks.T, label="landmarks", marker="x")
        ax.scatter(*self.positions.T, label="positions", marker="x")
        ax.legend()
        ax.axis("equal")
        ax.grid()
        ax.set_title(title)
        plt.show()
        return fig, ax

    def plot_nullvector(self, vec, ax, **kwargs):
        j = 0
        for n in range(self.n_positions):
            pos = self.positions[n]
            line = np.c_[pos, pos + vec[j : j + self.d]]  # 2 x self.d
            ax.plot(*line, **kwargs)
            j += self.d
        for n in range(self.n_landmarks):
            if n == 1:
                pos = self.landmarks[n]
                e = np.zeros(self.d)
                e[0] = vec[j]
                line = np.c_[pos, pos + e]  # 2 x self.d
                ax.plot(*line, **kwargs)
                j += 1
            elif n == 2:
                pos = self.landmarks[n]
                e = np.zeros(self.d)
                e[:2] = vec[j : j + 2]
                line = np.c_[pos, pos + e]  # 2 x self.d
                ax.plot(*line, **kwargs)
                j += 2
            elif n > 2:
                pos = self.landmarks[n]
                e = vec[j : j + self.d]
                line = np.c_[pos, pos + e]  # 2 x self.d
                ax.plot(*line, **kwargs)
                j += self.d
