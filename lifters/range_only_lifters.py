from lifters.state_lifter import StateLifter
from poly_matrix.least_squares_problem import LeastSquaresProblem

import scipy.sparse as sp
from scipy.optimize import minimize
import numpy as np

SOLVER_KWARGS = dict(
    # method="Nelder-Mead",
    method="BFGS"  # the only one that almost always converges
    #method="Powell"
)

class RangeOnlyLocLifter(StateLifter):
    """Range-only localization

    - level "no" uses substitution z_i=||p_i||^2=x_i^2 + y_i^2
    - level "quad" uses substitution z_i=[x_i^2, y_i^2, x_iy_i]
    """
    LEVELS = ["no", "quad"]
    def __init__(self, n_positions, n_landmarks, d, W=None, level="no"):
        # there is no Gauge freedom in range-only localization!
        self.n_positions = n_positions
        self.n_landmarks = n_landmarks
        self.d = d
        if W is not None:
            assert W.shape == (n_landmarks, n_positions)
            self.W = W
        else:
            self.W = np.ones((n_landmarks, n_positions))
        super().__init__(level=level)

    def get_Q_from_y(self, y):
        import itertools

        self.ls_problem = LeastSquaresProblem()

        if self.level == "quad":
            from utils.common import diag_indices
            diag_idx = diag_indices(self.n_positions)

        for n, k in itertools.product(range(self.n_positions), range(self.n_landmarks)):
            if self.W[n, k] > 0:
                ak = self.landmarks[k]
                if self.level == "no":
                    self.ls_problem.add_residual(
                        {
                            "l": y[n, k] - np.linalg.norm(ak) ** 2,
                            f"x{n}": 2 * ak.reshape((1, -1)),
                            f"z{n}": -1,
                        }
                    )
                elif self.level == "quad":
                    res_dict = {
                            "l": y[n, k] - np.linalg.norm(ak) ** 2,
                            f"x{n}": 2 * ak.reshape((1, -1)),
                    }
                    res_dict.update({f"z{n}:{i}": -1 for i in diag_idx})
                    self.ls_problem.add_residual(res_dict)
        return self.ls_problem.get_Q().get_matrix(self.var_dict)

    def generate_random_setup(self):
        self.landmarks = np.random.rand(self.n_landmarks, self.d)
        self.parameters = np.r_[1.0, self.landmarks.flatten()]

    def generate_random_theta(self):
        return np.random.rand(self.n_positions, self.d).flatten()

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

    def sample_theta(self):
        return self.generate_random_theta()

    def get_A_known(self):
        from poly_matrix.poly_matrix import PolyMatrix

        if self.level == "quad":
            from utils.common import diag_indices
            diag_idx = diag_indices(self.n_positions)

        A_list = []
        for n in range(self.n_positions):
            A = PolyMatrix()
            A[f"x{n}", f"x{n}"] = np.eye(self.d)
            if self.level == "no":
                A["l", f"z{n}"] = -0.5
            elif self.level == "quad":
                for i in diag_idx:
                    A["l", f"z{n}:{i}"] = -0.5
            A_list.append(A.get_matrix(self.var_dict))
        return A_list

    def get_x(self, theta=None, parameters=None, var_subset=None):
        from utils.common import upper_triangular
        if var_subset is None:
            var_subset = self.var_dict
        if theta is None:
            theta = self.theta
        if parameters is None:
            parameters = self.parameters

        positions = theta.reshape(self.n_positions, -1)

        x_data = [1.0]
        x_data += list(theta)

        for n in range(self.n_positions):
            if self.level == "no":
                x_data.append(np.linalg.norm(positions[n])**2)
            elif self.level == "quad":
                x_data += list(upper_triangular(positions[n]))
        assert len(x_data) == self.N + self.M + 1
        return np.array(x_data)


    def get_J_lifting(self, t):
        pos = t.reshape((-1, self.d))
        ii = []
        jj = []
        data = []

        for n in range(self.n_positions):
            if self.level == "no":
                ii += [n] * self.d
                jj += list(range(n * self.d, (n + 1) * self.d))
                data += list(2 * pos[n])
            elif self.level == "quad":
                ii += list(range(n * self.size_z, (n + 1) * self.size_z))
                # it seemed easier to do this manually that programtically
                if self.d == 3:
                    x, y, z = pos[n]
                    jj += [0, 0, 1, 0, 2, 1, 1, 2, 2]
                    data += [2*x, y, x, z, x, 2*y, z, y, 2*z]
                elif self.d == 2:
                    x, y = pos[n]
                    jj += [0, 0, 1, 1]
                    data += [2*x, y, x, 2*y]
        J_lifting = sp.csr_array(
            (data, (ii, jj)),
            shape=(self.M, self.N),
        )
        return J_lifting

    def get_hess_lifting(self, t):
        """return list of the hessians of the M lifting functions."""
        hessians = []
        for n in range(self.M):
            if self.level == "no":
                idx = range(n*self.d, (n + 1)*self.d)
                hessian = sp.csr_array(
                    ([2] * self.d, (idx, idx)),
                    shape=(self.N, self.N),
                )
                hessians.append(hessian)
            elif self.level == "quad":
                hessians += self.fixed_hessian_list
        return hessians
    
    @property
    def fixed_hessian_list(self):
        if self.d == 2:
            return [
                np.array(
                    [[2, 0], 
                     [0, 0]]),
                np.array(
                    [[0, 1], 
                     [1, 0]]),
                np.array(
                    [[0, 0], 
                     [0, 2]])
            ]
        elif self.d == 3:
            return [
                np.array(
                    [[2, 0, 0],
                     [0, 0, 0]
                     [0, 0, 0]]
                ),
                np.array(
                    [[0, 1, 0],
                     [1, 0, 0]
                     [0, 0, 0]]
                ),
                np.array(
                    [[0, 0, 1],
                     [0, 0, 0]
                     [1, 0, 0]]
                ),
                np.array(
                    [[0, 0, 0],
                     [0, 2, 0]
                     [0, 0, 0]]
                ),
                np.array(
                    [[0, 0, 0],
                     [0, 0, 1]
                     [0, 1, 0]]
                ),
                np.array(
                    [[0, 0, 0],
                     [0, 0, 0]
                     [0, 0, 2]]
                ),
            ]

    def get_cost(self, t, y):
        """
        get cost for given positions, landmarks and noise.

        :param t: (positions, landmarks) tuple
        """
        
        positions = t.reshape(self.n_positions, -1)

        y_current = (
            np.linalg.norm(self.landmarks[None, :, :] - positions[:, None, :], axis=2) ** 2
        )
        cost = np.sum(self.W * (y - y_current) ** 2)
        return cost


    def get_grad(self, t, y):
        """get gradient"""
        J = self.get_J(t, y)
        x = self.get_x(t)
        Q = self.get_Q_from_y(y)
        return 2 * J.T @ Q @ x

    def get_J(self, t, y):
        J = sp.csr_array(
            (np.ones(self.N), (range(1, self.N + 1), range(self.N))),
            shape=(self.N + 1, self.N),
        )
        J_lift = self.get_J_lifting(t)
        J = sp.vstack([J, J_lift])
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

    def get_Q(self, noise: float = 1e-3) -> tuple:
        # N x K matrix
        positions = self.theta.reshape(self.n_positions, -1)
        y_gt = (
            np.linalg.norm(
                self.landmarks[None, :, :] - positions[:, None, :], axis=2
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

    def local_solver(
        self, t_init, y, tol=1e-8, verbose=False, solver_kwargs=SOLVER_KWARGS
    ):
        """
        :param t_init: (positions, landmarks) tuple
        """

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


    @property
    def var_dict(self):
        var_dict = {"l": 1}
        var_dict.update({f"x{n}": self.d for n in range(self.n_positions)})
        if self.level == "no":
            var_dict.update({f"z{n}": 1 for n in range(self.n_positions)})
        elif self.level == "quad":
            # x^2, xy, y^2, 
            for n in range(self.n_positions):
                var_dict.update({f"z{n}:{d}": 1 for n in range(self.n_positions) for d in range(self.size_z)})
        return var_dict

    @property
    def size_z(self):
        return int(self.d * (self.d + 1) / 2)

    @property
    def N(self):
        return self.n_positions * self.d

    @property
    def M(self):
        if self.level == "no":
            return self.n_positions
        elif self.level == "quad":
            return self.n_positions * self.d * (self.d+1)/2

    def __repr__(self):
        return f"rangeonlyloc{self.d}d_{self.level}"


if __name__ == "__main__":
    lifter = RangeOnlyLocLifter(n_positions=3, n_landmarks=4, d=2)