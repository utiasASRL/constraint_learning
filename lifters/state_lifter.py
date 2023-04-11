from abc import ABC, abstractmethod

import numpy as np

EPS = 1e-10  # threshold for nullspace (eigenvalues)
METHOD = "qr"  # basis pursuit method
NORMALIZE = True  # normalize learned Ai or not


class StateLifter(ABC):
    def __init__(self, theta_shape, M, L=0):
        self.theta_shape = theta_shape
        if len(theta_shape) > 1:
            self.N_ = np.multiply(*theta_shape)
        else:
            self.N_ = theta_shape[0]
        self.M_ = M
        self.L = L
        self.setup = None

        # fixing seed for testing purposes
        np.random.seed(1)
        self.generate_random_setup()

    # the property decorator creates by default read-only attributes.
    # can create a @dim_x.setter function to make it also writeable.
    @property
    def dim_x(self):
        return 1 + self.N + self.M + self.L

    @property
    def d(self):
        return self.d_

    @d.setter
    def d(self, var):
        assert var in [1, 2, 3]
        self.d_ = var

    @property
    def N(self):
        return self.N_

    @N.setter
    def N(self, var):
        self.N_ = var

    @property
    def M(self):
        return self.M_

    def get_Q(self, noise=1e-3):
        print("Warning: get_Q not implemented")
        return None, None

    @abstractmethod
    def generate_random_setup(self):
        return

    @abstractmethod
    def sample_feasible(self):
        return

    @abstractmethod
    def get_x(self, theta) -> np.ndarray:
        return

    def get_A_known(self) -> list:
        return []

    def get_A_learned(self) -> list:
        Y = self.generate_Y()
        basis, _ = self.get_basis(Y, eps=EPS, method=METHOD)
        return self.generate_matrices(basis, normalize=NORMALIZE)

    def get_vec_around_gt(self, delta: float = 0):
        """Sample around groudn truth.

        :param delta: sample from gt + std(delta) (set to 0 to start from gt.)
        """
        return self.theta + np.random.normal(size=self.theta.shape, scale=delta)

    def _get_vec(self, mat):
        return mat[np.triu_indices(n=self.dim_x)].flatten()

    def generate_Y(self, factor=3, ax=None):
        dim_Y = int(self.dim_x * (self.dim_x + 1) / 2)

        # need at least dim_Y different random setups
        n_seeds = dim_Y * factor
        Y = np.empty((n_seeds, dim_Y))
        self.generate_random_setup()
        for seed in range(n_seeds):
            np.random.seed(seed)

            theta = self.sample_feasible()
            if seed < 10 and ax is not None:
                if np.ndim(self.theta) == 1:
                    ax.scatter(np.arange(len(theta)), theta)
                else:
                    ax.scatter(*theta[:, :2].T)

            x = self.get_x(theta)
            X = np.outer(x, x)

            Y[seed, :] = self._get_vec(X)
        return Y

    def get_basis(self, Y, eps=EPS, method=METHOD):
        """
        generate basis from lifted state matrix Y
        """
        U, S, Vh = np.linalg.svd(
            Y
        )  # nullspace of Y is in last columns of V / last rows of Vh
        rank = np.sum(np.abs(S) > eps)
        if method == "svd":
            basis = Vh[rank:, :]

            # test that it is indeed a null space
            np.testing.assert_allclose(Y @ basis.T, 0.0, atol=1e-5)
        elif method == "qr":
            Q, R = np.linalg.qr(Y.T)
            basis = Q[:, rank:].T

            # TODO(FD) below is pretty high. figure out if that's a problem
            # print("max QR basis error:", np.max(np.abs(Y @ basis.T)))
        else:
            raise ValueError(method)

        # test that all columns are orthonormal
        np.testing.assert_allclose(basis @ basis.T, np.eye(basis.shape[0]), atol=1e-10)
        return basis, S

    def generate_matrices(self, basis, normalize=NORMALIZE):
        """
        generate matrices from vectors
        """
        triu = np.triu_indices(n=self.dim_x)

        A_list = []
        for i in range(basis.shape[0]):
            Ai = np.zeros((self.dim_x, self.dim_x))
            Ai[triu] = basis[i, :]
            Ai += Ai.T
            Ai /= 2

            if normalize:
                Ai /= np.max(Ai)
            A_list.append(Ai)
        return A_list

    def run(self):
        """Convenience function to quickly test and debug lifter"""
        import matplotlib.pylab as plt

        from lifters.plotting_tools import plot_matrices, plot_singular_values

        Y = self.generate_Y()
        basis, S = self.get_basis(Y, eps=EPS)
        A_list = self.generate_matrices(basis)

        plot_singular_values(S, eps=EPS)

        # check that learned constraints meat tolerance
        tol = 1e-5
        for A in A_list[:10]:
            self.sample_feasible()
            x = self.get_x()

            constraint_violation = x.T @ A @ x
            if constraint_violation > tol:
                print("Warning: big violation")

        plot_matrices(A_list, n_matrices=5, start_idx=0)
        plt.show()

        from solvers.common import find_local_minimum, solve_dual

        noise = 1e-3
        Q, y = self.get_Q(noise=noise)

        print("solve dual problems...")
        dual_costs = []

        n = 4
        if len(A_list) > n:
            use_constraints = [0, (len(A_list) - n) // 2] + list(
                range(len(A_list) - n + 2, len(A_list) + 1)
            )
        else:
            use_constraints = range(len(A_list))
        for i in use_constraints:
            A_current = A_list[:i]
            cost, H, status = solve_dual(Q, A_current)
            dual_costs.append(cost)
            print(f"added {i}")

        print("find local minimum...")
        xhat, local_cost = find_local_minimum(self, y, delta=noise, n_inits=1)

        plt.figure()
        plt.axhline(local_cost, label=f"QCQP cost {local_cost:.2e}")
        plt.scatter(range(len(use_constraints)), dual_costs, label="dual costs")
        plt.xticks(range(len(use_constraints)), use_constraints)
        plt.xlabel("added constraints up to")
        plt.ylabel("cost")
        plt.legend()
        plt.show()
        plt.pause(1)
