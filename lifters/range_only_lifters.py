import itertools
from copy import deepcopy

import matplotlib.pylab as plt
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl
from scipy.optimize import minimize

from lifters.state_lifter import StateLifter
from poly_matrix import PolyMatrix
from ro_certs.gauss_newton import gauss_newton
from ro_certs.problem import Problem, Reg, generate_random_trajectory
from utils.common import diag_indices

plt.ion()

NOISE = 1e-2  # std deviation of distance noise

# METHOD = "BFGS"
METHOD = "GN"
NORMALIZE = True

# TODO(FD): parameters below are not all equivalent.
SOLVER_KWARGS = {
    "BFGS": dict(gtol=1e-6, maxiter=200),  # xtol=1e-10),  # relative step size
    "Nelder-Mead": dict(xatol=1e-10, maxiter=200),  # absolute step size
    "Powell": dict(ftol=1e-6, xtol=1e-10, maxiter=200),
    "TNC": dict(gtol=1e-6, xtol=1e-10, maxiter=200),
    "GN": dict(tol=1e-6, gtol=1e-6, maxiter=1000),
}


class RangeOnlyLocLifter(StateLifter):
    """Range-only localization

    - level "no" uses substitution z_i=||p_i||^2=x_i^2 + y_i^2
    - level "quad" uses substitution z_i=[x_i^2, x_iy_i, y_i^2]
    """

    VARIABLE_LIST = [
        ["h", "x_0"],
        ["h", "x_0", "z_0"],
        ["h", "x_0", "z_0", "z_1"],
        ["h", "x_0", "x_1", "z_0", "z_1"],
    ]

    TIGHTNESS = "rank"
    LEVELS = ["no", "dir", "quad"]
    LEVEL_NAMES = {
        "no": "$z_n$",
        "quad": "$\\boldsymbol{y}_n$",
        "dir": "$\\boldsymbol{n}_n$",
    }
    PRIOR_NOISE = 0.2

    ADMM_OPTIONS = dict(use_fusion=True, maxiter=20, early_stop=False, rho_start=1e2)
    ADMM_INIT_XHAT = False

    def get_vec_around_gt(self, delta: float = 0):
        """Sample around ground truth."""
        if delta == 0:
            return self.theta.flatten()
        else:
            bbox_max = np.max(self.landmarks, axis=0) * 2
            bbox_min = np.min(self.landmarks, axis=0) * 2
            pos = (
                np.random.rand(self.n_positions, self.d)
                * (bbox_max - bbox_min)[None, :]
                + bbox_min[None, :]
            )
            theta = np.ones((self.n_positions, self.k))
            theta[:, : self.d] = pos
            return theta.flatten()

    def __init__(
        self,
        n_positions,
        n_landmarks,
        d,
        level="no",
        variable_list=None,
        reg=Reg.NONE,
    ):
        self.reg = reg
        self.Q_fixed = None
        self.times = None
        self.d = d
        self.n_positions = n_positions
        self.n_landmarks = n_landmarks
        self.n_cliques = n_positions - 1
        self.y_ = None

        if variable_list == "all":
            variable_list = self.get_all_variables()

        super().__init__(level=level, d=d, variable_list=variable_list)
        # dimension of primary variables
        self.N = self.n_positions * self.k

        # dimension of one substitution variable
        self.size_z = 1 if self.level == "no" else int(self.d * (self.d + 1) / 2)

        # dimension of substitution variables
        self.M = int(self.n_positions * self.size_z)

    def get_all_variables(self):
        vars = ["h"]
        vars += [f"x_{i}" for i in range(self.n_positions)]
        vars += [f"z_{i}" for i in range(self.n_positions)]
        return [vars]

    def generate_random_setup(self):
        # self.prob.landmarks
        # self.prob = Problem(K=n_landmarks, N=n_positions, d=d, regularization=reg)
        self.prob = Problem.generate_prob(
            N=self.n_positions, K=self.n_landmarks, d=self.d, linear_anchors=True
        )
        self.k = self.prob.get_dim()
        self.theta_ = deepcopy(self.prob.theta)
        self.landmarks = deepcopy(self.prob.anchors)
        self.trajectory = deepcopy(self.prob.trajectory)
        # trajectory = self.theta.reshape(-1, self.k)
        # self.prob.generate_random_anchors(trajectory=trajectory[:, : self.d])
        # self.landmarks = self.prob.anchors
        self.parameters = np.r_[1.0, self.landmarks.flatten()]
        self.times = deepcopy(self.prob.times)
        self.y_ = deepcopy(self.prob.D_noisy_sq)

    def generate_random_theta(self):
        times = sorted(
            np.random.uniform(low=0, high=self.n_positions - 1, size=self.n_positions)
        )
        trajectory, velocities = generate_random_trajectory(
            self.n_positions,
            self.d,
            times=times,
            v_sigma=self.PRIOR_NOISE,
            return_velocities=True,
            fix_x0=False,
            fix_v0=False,
        )
        theta = np.hstack([trajectory, velocities])
        return theta[:, : self.k].flatten()

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
            parameters = self.prob.generate_random_anchors()
            return np.r_[1.0, parameters]

    def sample_theta(self):
        return self.generate_random_theta()

    def get_A_known(self, var_dict=None, output_poly=False, add_redundant=False):
        if var_dict is None:
            var_dict = self.var_dict
        positions = self.get_variable_indices(var_dict)

        A_list = []
        for n in positions:
            if self.level == "no":
                A = PolyMatrix(symmetric=True)
                mat = np.zeros((self.k, self.k))
                mat[range(self.d), range(self.d)] = 1.0
                A[f"x_{n}", f"x_{n}"] = mat
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
                        mat_x = np.zeros((self.k, self.k))
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

                        if add_redundant:
                            raise ValueError("redundant not implemented yet for quad")
        return A_list

    def get_x(self, theta=None, parameters=None, var_subset=None):
        from utils.common import upper_triangular

        if var_subset is None:
            var_subset = self.var_dict
        if theta is None:
            theta = self.theta

        positions = theta.reshape(self.n_positions, self.k)

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
                    x_data.append(np.linalg.norm(positions[n, : self.d]) ** 2)
                elif self.level == "quad":
                    x_data += list(upper_triangular(positions[n, : self.d]))
        assert len(x_data) == self.get_dim_x(var_subset)
        return np.array(x_data)

    def get_J_lifting(self, t):
        pos = t.reshape((-1, self.k))
        ii = []
        jj = []
        data = []

        idx = 0
        for n in range(self.n_positions):
            if self.level == "no":
                ii += [n] * self.d
                jj += list(range(n * self.k, n * self.k + self.d))
                data += list(2 * pos[n, : self.d])
            elif self.level == "quad":
                # it was easier to do this manually that programtically
                if self.d == 3:
                    x, y, z = pos[n, : self.d]
                    jj += [n * self.k + j for j in [0, 0, 1, 0, 2, 1, 1, 2, 2]]
                    data += [2 * x, y, x, z, x, 2 * y, z, y, 2 * z]
                    ii += [idx + i for i in [0, 1, 1, 2, 2, 3, 4, 4, 5]]
                elif self.d == 2:
                    x, y = pos[n, : self.d]
                    jj += [n * self.k + j for j in [0, 0, 1, 1]]
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
            idx = range(n * self.k, n * self.k + self.d)
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

    def get_B_matrix(self, y):
        Q = self.get_Q_from_y(y)
        S, U = spl.eigsh(Q)
        return np.diag(np.sqrt(S)) @ U.T

    def get_residuals(self, t, y):
        x = self.get_x(theta=t)
        B = self.get_B_matrix(y)
        return B @ x

    def get_residuals_prior(self):
        assert self.prob.v.nnz == 0, "non-zero v not supported currently."
        return self.prob.A_inv @ self.theta.flatten()

    def get_cost(self, t, y, sub_idx=None):
        """
        get cost for given positions, landmarks and noise.

        :param t: flattened positions of length Nk
        :param y: K x N distance measurements
        """
        x = self.get_x(theta=t)
        if self.Q_fixed is None:
            self.get_Q_from_y(y, save=True)

        Q = self.Q_fixed
        if sub_idx is not None:
            sub_idx_x = self.get_sub_idx_x(sub_idx)
            x = x[sub_idx_x]
            Q = Q[sub_idx_x, :][:, sub_idx]
            return x.T @ Q @ x
        return x.T @ Q @ x

        # old implementation -- removed to not duplicate code.
        residuals = self.get_residuals(t, y)
        if sub_idx is None:
            cost = np.sum(residuals**2)
        else:
            cost = np.sum(residuals[sub_idx] ** 2)
        if NORMALIZE:
            cost /= np.sum(self.prob.W > 0)

        if self.REG != Reg.NONE:
            prior_cost = np.sum(self.get_residuals_prior())
            cost += prior_cost
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
        """Jacobian of derivative of x w.r.t. theta."""
        J = sp.csr_array(
            (np.ones(self.N), (range(1, self.N + 1), range(self.N))),
            shape=(self.N + 1, self.N),
        )
        J_lift = self.get_J_lifting(t)
        J = sp.vstack([J, J_lift])
        return J

    def get_hess(self, t, y):
        """Hessian of cost w.r.t. theta"""

        x = self.get_x(t)
        Q = self.get_Q_from_y(y)
        J = self.get_J(t, y)
        hess = 2 * J.T @ Q @ J

        # TODO(FD) currently this is not consistent with the new added regularization.
        # but we are not using the Hessian anyways so it doesn't matter for now.
        raise NotImplementedError("hessian not adjusted yet")
        hessians = self.get_hess_lifting(t)
        B = self.get_B_matrix(y)
        residuals = B @ x
        for m, h in enumerate(hessians):
            bm_tilde = B[:, -self.M + m]
            factor = float(bm_tilde.T @ residuals)
            hess += 2 * factor * h
        return hess

    def get_Q_from_y(
        self, y, use_cliques=None, output_poly=False, overlap_type=0, save=False
    ):
        if use_cliques is None:
            use_cliques = range(self.n_positions)

        if self.level == "quad":
            diag_idx = diag_indices(self.d)

        Q_poly = PolyMatrix()
        for n in use_cliques:
            if (use_cliques is not None) and (n not in use_cliques):
                continue
            nnz = np.where(self.prob.W[:, n] > 0)[0]

            # Create the cost terms corresponding to residual:
            #              [h  ]
            # [bn An fn] @ [x_n]
            #              [z_n]
            # which can be easily formed from (d_ij^2 - ||ai - xj||^2)
            Q_sub = PolyMatrix()
            An = np.zeros((len(nnz), self.k))
            An[:, : self.d] = 2 * self.landmarks[nnz]
            if self.level == "no":
                fn = -np.ones((len(nnz), 1))
            elif self.level == "quad":
                fn = np.zeros((len(nnz), self.size_z))
                fn[:, diag_idx] = -1
            Sig_inv_n = self.prob.Sig_inv[nnz, :][:, nnz]
            bn = y[nnz, n] - np.linalg.norm(self.landmarks[nnz], axis=1) ** 2
            bn = bn.reshape(len(nnz), 1)
            vars = ["h", f"x_{n}", f"z_{n}"]
            residual = [bn, An, fn]
            for i, j in itertools.combinations_with_replacement(range(3), 2):
                Q_sub[vars[i], vars[j]] = residual[i].T @ Sig_inv_n @ residual[j]

            if (overlap_type == 1) and (n > 0) and (n < self.n_positions - 1):
                Q_poly += 0.5 * Q_sub
            elif (
                (overlap_type == 2)
                and (n == use_cliques[1])
                and (n != self.n_positions - 1)
            ):
                pass
            elif (overlap_type == 3) and (n == use_cliques[0]) and (n > 0):
                pass
            else:
                Q_poly += Q_sub

        Q_poly /= np.sum(self.prob.W > 0)

        if self.reg != Reg.NONE:
            for n in use_cliques:
                Q_sub = self.prob.get_R_nn(n)
                if (overlap_type == 1) and (n > 0) and (n < self.n_positions - 1):
                    Q_poly[f"x_{n}", f"x_{n}"] += Q_sub * 0.5
                elif (
                    (overlap_type == 2)
                    and (n == use_cliques[1])
                    and (n != self.n_positions - 1)
                ):
                    pass
                elif (overlap_type == 3) and (n == use_cliques[0]) and (n > 0):
                    pass
                else:
                    Q_poly[f"x_{n}", f"x_{n}"] += Q_sub

            for n in range(1, self.n_positions):
                if (n not in use_cliques) or (n - 1 not in use_cliques):
                    continue
                Q_poly[f"x_{n-1}", f"x_{n}"] += self.prob.get_R_nm(n)

        if save:
            self.Q_fixed = Q_poly.get_matrix(self.var_dict)

        if output_poly:
            return Q_poly
        return Q_poly.get_matrix(self.var_dict)

    def simulate_y(self, noise: float = None, sparsity: float = 1.0):
        assert isinstance(self.prob, Problem)
        self.prob.generate_distances(sigma_dist_real=noise)
        self.y_ = deepcopy(self.prob.D_noisy_sq)
        if sparsity == 1.0:
            self.prob.W = np.ones((self.n_landmarks, self.n_positions))
            self.prob.W = self.prob.W
        else:
            num_total = self.n_landmarks * self.n_positions
            num_keep = int(sparsity * num_total)
            num_min = self.n_positions * (self.d + 1)
            assert num_keep >= num_min

            self.prob.W = np.zeros((self.n_landmarks, self.n_positions))

            # first, make sure we see enough landmarks per position.
            for i in range(self.n_positions):
                chosen_landmarks = np.random.choice(
                    range(self.n_landmarks), self.d + 1, replace=False
                )
                self.prob.W[chosen_landmarks, i] = 1.0

            # then, fill remaining ones.
            left_i, left_j = np.where(self.prob.W == 0)
            chosen_idx = np.random.choice(
                range(len(left_i)), num_keep - num_min, replace=False
            )
            self.prob.W[left_i[chosen_idx], left_j[chosen_idx]] = 1.0
            assert np.sum(self.prob.W) == num_keep

    def get_Q(self, noise: float = None, sparsity: float = 1.0) -> tuple:
        if noise is not None:
            self.simulate_y(
                noise=noise, sparsity=sparsity
            )  # defines self.y_ and self.prob.W
            self.Q_fixed = None

        if self.Q_fixed is None:
            self.get_Q_from_y(self.y_, save=True)  # saves Q_fixed

        # DEBUGGING
        x = self.get_x(theta=self.theta)
        cost1 = x.T @ self.Q_fixed @ x
        cost3 = self.get_cost(t=self.theta, y=self.y_)
        assert abs(cost1 - cost3) < 1e-10, (cost1, cost3)

        from ro_certs.gauss_newton import get_grad_hess_cost_f

        cost2 = get_grad_hess_cost_f(
            self.theta,
            self.prob,
            return_cost=True,
            return_grad=False,
            return_hess=False,
        )[0]
        if abs(cost1 - cost2) > 1e-10:
            # print(f"Warning: costs not the same {cost1:.4f}, {cost2:.4f}")
            pass
        return self.Q_fixed, self.y_

    def get_sub_idx_x(self, sub_idx, add_z=True):
        sub_idx_x = [0]
        for idx in sub_idx:
            sub_idx_x += [1 + idx * self.k + d for d in range(self.d)]
        if not add_z:
            return sub_idx_x
        for idx in sub_idx:
            sub_idx_x += [
                1 + self.n_positions * self.k + idx * self.size_z + i
                for i in range(self.size_z)
            ]
        return sub_idx_x

    def get_position(self, theta=None, xtheta=None):
        if theta is not None:
            return theta.reshape(self.n_positions, self.d)
        elif xtheta is not None:
            return xtheta.reshape(self.n_positions, self.d)

    def get_error(self, that):
        err = np.sqrt(np.mean((self.theta - that) ** 2))
        return {"total error": err, "error": err}

    def local_solver(
        self,
        t_init,
        y,
        verbose=False,
        method=METHOD,
        solver_kwargs=SOLVER_KWARGS,
    ):
        """
        :param t_init: (positions, landmarks) tuple
        """
        if method == "GN":
            t_init_mat = t_init.reshape(-1, self.prob.get_dim())
            that, info = gauss_newton(t_init_mat, self.prob, **solver_kwargs[method])
            cost = info["cost"]
            success = info["success"]
            msg = info["status"]

        else:
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
            that = sol.x
            cost = sol.fun
            success = sol.success
            msg = sol.message + f" (# iterations: {sol.nit})"
        residuals = self.get_residuals(that, y)
        info = {
            "msg": msg,
            "cost": cost,
            "success": success,
            "max res": np.max(np.abs(residuals)),
        }
        # hess = self.get_hess(that, y)
        # eigs = np.linalg.eigvalsh(hess.toarray())
        # info["cond Hess"] = eigs[-1] / eigs[0]
        if not success:
            print("Warning: local solver finished with", msg)
        return that, info, cost

    @property
    def var_dict(self):
        var_dict = {"h": 1}
        var_dict.update({f"x_{n}": self.k for n in range(self.n_positions)})
        if self.level == "no":
            var_dict.update({f"z_{n}": 1 for n in range(self.n_positions)})
        elif self.level == "quad":
            var_dict.update({f"z_{n}": self.size_z for n in range(self.n_positions)})
        return var_dict

    # clique stuff
    def base_size(self):
        return self.var_dict["h"]

    def node_size(self):
        return self.var_dict["x_0"] + self.var_dict["z_0"]

    def get_clique_vars_ij(self, *args):
        var_dict = {"h": self.var_dict["h"]}
        for i in args:
            var_dict.update(
                {
                    f"x_{i}": self.var_dict[f"x_{i}"],
                    f"z_{i}": self.var_dict[f"z_{i}"],
                }
            )
        return var_dict

    def get_clique_vars(self, i, n_overlap=0):
        used_landmarks = list(range(i, min(i + n_overlap + 1, self.n_poses)))
        vars = {"h": self.var_dict["h"]}
        for j in used_landmarks:
            vars.update(
                {
                    f"x_{j}": self.var_dict[f"x_{j}"],
                    f"z_{j}": self.var_dict[f"z_{j}"],
                }
            )
        return vars

    def get_clique_cost(self, i):
        """
        All elements with an * need to be halved because they appear in
        multiple cliques.

           h  | x1 z1 | x2 z2 | x3 z3 | ...
        h       q1    | q2*   | q3*
        x1      Q_11  | Q_12  | ...
        z1
        x2              Q_22* | Q_23  |
        z2
        x3                      Q_33* | Q_34
        """
        # returns clique x1, x2
        Q = self.get_Q_from_y(
            self.y_, use_cliques=[i, i + 1], overlap_type=1, output_poly=True
        )
        return Q

    def __repr__(self):
        if self.reg is not Reg.NONE:
            if self.reg == Reg.CONSTANT_VELOCITY:
                return f"rangeonlyloc{self.d}d_{self.level}_const-vel"
            else:
                return f"rangeonlyloc{self.d}d_{self.level}_zero-vel"
        return f"rangeonlyloc{self.d}d_{self.level}"


if __name__ == "__main__":
    lifter = RangeOnlyLocLifter(n_positions=3, n_landmarks=4, d=2)
