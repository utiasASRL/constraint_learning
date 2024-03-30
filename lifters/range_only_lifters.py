import matplotlib.pylab as plt
import numpy as np
import scipy.sparse as sp
from scipy.optimize import minimize

from lifters.state_lifter import StateLifter
from poly_matrix.least_squares_problem import LeastSquaresProblem

plt.ion()

NOISE = 1e-2  # std deviation of distance noise

METHOD = "BFGS"
NORMALIZE = True

# TODO(FD): parameters below are not all equivalent.
SOLVER_KWARGS = {
    "BFGS": dict(gtol=1e-6, xrtol=1e-10),  # relative step size
    "Nelder-Mead": dict(xatol=1e-10),  # absolute step size
    "Powell": dict(ftol=1e-6, xtol=1e-10),
    "TNC": dict(gtol=1e-6, xtol=1e-10),
}

class RangeOnlyLocLifter(StateLifter):
    """Range-only localization

    - level "no" uses substitution z_i=||p_i||^2=x_i^2 + y_i^2
    - level "quad" uses substitution z_i=[x_i^2, x_iy_i, y_i^2]
    """

    TIGHTNESS = "rank"
    LEVELS = ["no", "quad"]
    LEVEL_NAMES = {
        "no": "$z_n$",
        "quad": "$\\boldsymbol{y}_n$",
    }

    def get_vec_around_gt(self, delta: float = 0):
        """Sample around ground truth.
        :param delta: sample from gt + std(delta) (set to 0 to start from gt.)
        """
        if delta == 0:
            return self.theta
        else:
            bbox_max = np.max(self.landmarks, axis=0) * 2
            bbox_min = np.min(self.landmarks, axis=0) * 2
            pos = (
                np.random.rand(self.n_positions, self.d)
                * (bbox_max - bbox_min)[None, :]
                + bbox_min[None, :]
            )
            return pos.flatten()

    def __init__(
        self, n_positions, n_landmarks, d, W=None, level="no", variable_list=None
    ):
        # there is no Gauge freedom in range-only localization!
        self.n_positions = n_positions
        self.n_landmarks = n_landmarks
        if W is not None:
            assert W.shape == (n_landmarks, n_positions)
            self.W = W
        else:
            self.W = np.ones((n_positions, n_landmarks))
        self.y_ = None

        if variable_list == "all":
            variable_list = self.get_all_variables()
        super().__init__(level=level, d=d, variable_list=variable_list)

    @property
    def VARIABLE_LIST(self):
        return [
            ["h", "x_0"],
            ["h", "x_0", "z_0"],
            ["h", "x_0", "z_0", "z_1"],
            ["h", "x_0", "x_1", "z_0", "z_1"],
        ]

    def get_all_variables(self):
        vars = ["h"]
        vars += [f"x_{i}" for i in range(self.n_positions)]
        vars += [f"z_{i}" for i in range(self.n_positions)]
        return [vars]

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

    def sample_parameters(self, *args, **kwargs):
        if self.param_level == "no":
            return [1.0]
        else:
            parameters = np.random.rand(self.n_landmarks, self.d).flatten()
            return np.r_[1.0, parameters]

    def sample_theta(self):
        return self.generate_random_theta()

    def get_A_known(self, var_dict=None, output_poly=False):
        from poly_matrix.poly_matrix import PolyMatrix

        if var_dict is None:
            var_dict = self.var_dict
        positions = self.get_variable_indices(var_dict)

        if self.level == "quad":
            from utils.common import diag_indices

            diag_idx = diag_indices(self.d)

        A_list = []
        for n in positions:
            if self.level == "no":
                A = PolyMatrix(symmetric=True)
                A[f"x_{n}", f"x_{n}"] = np.eye(self.d)
                A["h", f"z_{n}"] = -0.5
                if output_poly:
                    A_list.append(A)
                else:
                    A_list.append(A.get_matrix(self.var_dict))

            elif self.level == "quad":
                count = 0
                for i in range(self.d):
                    for j in range(i, self.d):
                        A = PolyMatrix(symmetric=True)
                        mat_x = np.zeros((self.d, self.d))
                        mat_z = np.zeros((1, self.size_z))
                        if i == j:
                            mat_x[i, i] = 1.0
                        else:
                            mat_x[i, j] = 0.5
                            mat_x[j, i] = 0.5
                        mat_z[0, count] = -0.5
                        A[f"x_{n}", f"x_{n}"] = mat_x
                        A["h", f"z_{n}"] = mat_z
                        count += 1
                        if output_poly:
                            A_list.append(A)
                        else:
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

        x_data = []
        for key in var_subset:
            if key == "h":
                x_data.append(1.0)
            elif "x" in key:
                n = int(key.split("_")[-1])
                x_data += list(positions[n])
            elif "z" in key:
                n = int(key.split("_")[-1])
                if self.level == "no":
                    x_data.append(np.linalg.norm(positions[n]) ** 2)
                elif self.level == "quad":
                    x_data += list(upper_triangular(positions[n]))
        assert len(x_data) == self.get_dim_x(var_subset)
        return np.array(x_data)

    def get_J_lifting(self, t):
        pos = t.reshape((-1, self.d))
        ii = []
        jj = []
        data = []

        idx = 0
        for n in range(self.n_positions):
            if self.level == "no":
                ii += [n] * self.d
                jj += list(range(n * self.d, (n + 1) * self.d))
                data += list(2 * pos[n])
            elif self.level == "quad":
                # it seemed easier to do this manually that programtically
                if self.d == 3:
                    x, y, z = pos[n]
                    jj += [n * self.d + j for j in [0, 0, 1, 0, 2, 1, 1, 2, 2]]
                    data += [2 * x, y, x, z, x, 2 * y, z, y, 2 * z]
                    ii += [idx + i for i in [0, 1, 1, 2, 2, 3, 4, 4, 5]]
                elif self.d == 2:
                    x, y = pos[n]
                    jj += [n * self.d + j for j in [0, 0, 1, 1]]
                    data += [2 * x, y, x, 2 * y]
                    ii += [idx + i for i in [0, 1, 1, 2]]
                idx += self.size_z
        J_lifting = sp.csr_array(
            (data, (ii, jj)),
            shape=(self.M, self.N),
        )
        return J_lifting

    def get_hess_lifting(self, t):
        """return list of the hessians of the M lifting functions."""
        hessians = []
        for n in range(self.n_positions):
            idx = range(n * self.d, (n + 1) * self.d)
            if self.level == "no":
                hessian = sp.csr_array(
                    ([2] * self.d, (idx, idx)),
                    shape=(self.N, self.N),
                )
                hessians.append(hessian)
            elif self.level == "quad":
                for h in self.fixed_hessian_list:
                    ii, jj = np.meshgrid(idx, idx)
                    hessian = sp.csr_array(
                        (h.flatten(), (ii.flatten(), jj.flatten())),
                        shape=(self.N, self.N),
                    )
                    hessians.append(hessian)
        return hessians

    @property
    def fixed_hessian_list(self):
        if self.d == 2:
            return [
                np.array([[2, 0], [0, 0]]),
                np.array([[0, 1], [1, 0]]),
                np.array([[0, 0], [0, 2]]),
            ]
        elif self.d == 3:
            return [
                np.array([[2, 0, 0], [0, 0, 0], [0, 0, 0]]),
                np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]]),
                np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]]),
                np.array([[0, 0, 0], [0, 2, 0], [0, 0, 0]]),
                np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]]),
                np.array([[0, 0, 0], [0, 0, 0], [0, 0, 2]]),
            ]

    def get_residuals(self, t, y):
        positions = t.reshape((-1, self.d))
        y_current = (
            np.linalg.norm(self.landmarks[None, :, :] - positions[:, None, :], axis=2)
            ** 2
        )
        return self.W * (y - y_current)

    def get_cost(self, t, y, sub_idx=None):
        """
        get cost for given positions, landmarks and noise.

        :param t: flattened positions of length Nd
        :param y: N x K distance measurements
        """
        residuals = self.get_residuals(t, y)
        if sub_idx is None:
            cost = np.sum(residuals**2)
        else:
            cost = np.sum(residuals[sub_idx] ** 2)
        if NORMALIZE:
            return cost / np.sum(self.W > 0)
        return cost

    def get_grad(self, t, y, sub_idx=None):
        """get gradient"""
        J = self.get_J(t, y)
        x = self.get_x(t)
        Q = self.get_Q_from_y(y)
        if sub_idx is None:
            return 2 * J.T @ Q @ x
        else:
            sub_idx_x = self.get_sub_idx_x(sub_idx)
            return 2 * J.T[:, sub_idx_x] @ Q[sub_idx_x, :][:, sub_idx_x] @ x[sub_idx_x]

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

    def get_Q_from_y(self, y):
        import itertools

        self.ls_problem = LeastSquaresProblem()

        if self.level == "quad":
            from utils.common import diag_indices

            diag_idx = diag_indices(self.d)

        for n, k in itertools.product(range(self.n_positions), range(self.n_landmarks)):
            if self.W[n, k] > 0:
                ak = self.landmarks[k]
                if self.level == "no":
                    self.ls_problem.add_residual(
                        {
                            "h": y[n, k] - np.linalg.norm(ak) ** 2,
                            f"x_{n}": 2 * ak.reshape((1, -1)),
                            f"z_{n}": -1,
                        }
                    )
                elif self.level == "quad":
                    mat = np.zeros((1, self.size_z))
                    mat[0, diag_idx] = -1
                    res_dict = {
                        "h": y[n, k] - np.linalg.norm(ak) ** 2,
                        f"x_{n}": 2 * ak.reshape((1, -1)),
                        f"z_{n}": mat,
                    }
                    self.ls_problem.add_residual(res_dict)
        Q = self.ls_problem.get_Q().get_matrix(self.var_dict)
        if NORMALIZE:
            return Q / np.sum(self.W > 0)
        return Q

    def simulate_y(self, noise: float = None):
        # N x K matrix
        if noise is None:
            noise = NOISE
        positions = self.theta.reshape(self.n_positions, -1)
        y_gt = (
            np.linalg.norm(self.landmarks[None, :, :] - positions[:, None, :], axis=2)
            ** 2
        )
        return y_gt + np.random.normal(loc=0, scale=noise, size=y_gt.shape)

    def get_Q(self, noise: float = None) -> tuple:
        if self.y_ is None:
            self.y_ = self.simulate_y(noise=noise)
        Q = self.get_Q_from_y(self.y_)

        # DEBUGGING
        x = self.get_x()
        cost1 = x.T @ Q @ x
        cost3 = self.get_cost(self.theta, self.y_)
        assert abs(cost1 - cost3) < 1e-10
        return Q, self.y_

    def get_D(self, that):
        D = np.eye(1 + self.n_positions * self.d + self.size_z)
        x = self.get_x(theta=that)
        J = self.get_J_lifting(t=that)

        D = sp.lil_array((len(x), len(x)))
        D[range(len(x)), range(len(x))] = 1.0
        D[:, 0] = x
        D[-J.shape[0] :, 1 : 1 + J.shape[1]] = J
        return D.tocsc()

    def get_sub_idx_x(self, sub_idx, add_z=True):
        sub_idx_x = [0]
        for idx in sub_idx:
            sub_idx_x += [1 + idx * self.d + d for d in range(self.d)]
        if not add_z:
            return sub_idx_x
        for idx in sub_idx:
            sub_idx_x += [
                1 + self.n_positions * self.d + idx * self.size_z + d
                for d in range(self.size_z)
            ]
        return sub_idx_x

    def get_theta(self, x):
        return x[1 : self.d + 1]

    def get_position(self, theta=None):
        if theta is not None:
            return theta.reshape(self.n_positions, self.d)

    def get_error(self, that):
        err = np.sqrt(np.mean((self.theta - that) ** 2))
        return {"total error": err, "error": err}

    def local_solver(
        self,
        t_init,
        y,
        verbose=False,
        method="BFGS",
        solver_kwargs=SOLVER_KWARGS,
    ):
        """
        :param t_init: (positions, landmarks) tuple
        """

        # TODO(FD): split problem into individual points.
        options = solver_kwargs[method]
        options["disp"] = verbose
        sol = minimize(
            self.get_cost,
            x0=t_init,
            args=y,
            jac=self.get_grad,
            # hess=self.get_hess, not used by any solvers.
            method=method,
            options=options,
        )

        info = {}
        info["success"] = sol.success
        info["msg"] = sol.message + f"(# iterations: {sol.nit})"
        if sol.success:
            print("RangeOnly local solver:", sol.nit)
            that = sol.x
            rel_error = self.get_cost(that, y) - self.get_cost(sol.x, y)
            assert abs(rel_error) < 1e-10, rel_error
            residuals = self.get_residuals(that, y)
            cost = sol.fun
            info["max res"] = np.max(np.abs(residuals))
            hess = self.get_hess(that, y)
            eigs = np.linalg.eigvalsh(hess.toarray())
            info["cond Hess"] = eigs[-1] / eigs[0]
        else:
            that = cost = None
            info["max res"] = None
            info["cond Hess"] = None
        return that, info, cost

    @property
    def var_dict(self):
        var_dict = {"h": 1}
        var_dict.update({f"x_{n}": self.d for n in range(self.n_positions)})
        if self.level == "no":
            var_dict.update({f"z_{n}": 1 for n in range(self.n_positions)})
        elif self.level == "quad":
            var_dict.update({f"z_{n}": self.size_z for n in range(self.n_positions)})
        return var_dict

    @property
    def size_z(self):
        if self.level == "no":
            return self.d
        else:
            return int(self.d * (self.d + 1) / 2)

    @property
    def N(self):
        return self.n_positions * self.d

    @property
    def M(self):
        if self.level == "no":
            return self.n_positions
        elif self.level == "quad":
            return int(self.n_positions * self.d * (self.d + 1) / 2)

    def __repr__(self):
        return f"rangeonlyloc{self.d}d_{self.level}"


if __name__ == "__main__":
    lifter = RangeOnlyLocLifter(n_positions=3, n_landmarks=4, d=2)
