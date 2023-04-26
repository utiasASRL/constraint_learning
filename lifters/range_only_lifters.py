import itertools
from abc import abstractmethod
from copy import deepcopy

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
        super().__init__(theta_shape=(self.N, d), M=self.M)
        self.generate_random_landmarks()
        self.generate_random_positions()

    def generate_random_landmarks(self):
        landmarks = self.sample_random_landmarks()
        self.landmarks = landmarks

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

    def sample_random_positions(self):
        # TODO(FD) implement motion model?
        return np.random.rand(self.n_positions, self.d)

    def generate_random_positions(self):
        self.positions = self.sample_random_positions()

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
        return self.get_theta(self.positions, self.landmarks)

    def get_theta(self, positions, landmarks):
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

    @property
    def var_dict(self):
        pass

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

    def get_A_known(self):
        from poly_matrix.poly_matrix import PolyMatrix

        A_list = []
        for n in range(self.n_positions):
            A = PolyMatrix()
            A[f"x{n}", f"x{n}"] = np.eye(self.d)
            A["l", f"z{n}"] = -0.5
            A_list.append(A.get_matrix(self.var_dict))
        return A_list

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
    lifter.run()