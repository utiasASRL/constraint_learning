from abc import ABC

import autograd.numpy as np

from lifters.state_lifter import StateLifter
from poly_matrix.poly_matrix import PolyMatrix
from utils.geometry import (
    get_C_r_from_theta,
    get_noisy_pose,
    get_pose_errors_from_theta,
    get_T,
    get_theta_from_C_r,
)

NOISE = 1.0  #

NORMALIZE = True

SOLVER_KWARGS = dict(
    min_gradient_norm=1e-6, max_iterations=10000, min_step_size=1e-10, verbosity=1
)


class StereoLifter(StateLifter, ABC):
    """General lifter for stereo localization problem."""

    LEVELS = [
        "no",
        "u@u",  # ...
        "u2",
        "u@r",
        "uuT",
        "urT",
        "uxT",
    ]
    PARAM_LEVELS = ["no", "p", "ppT"]
    LEVEL_NAMES = {
        "no": "$\\boldsymbol{u}_n$",
        "urT": "$\\boldsymbol{u}\\boldsymbol{t}^\\top_n$",
        "uxT": "$\\boldsymbol{u}\\boldsymbol{x}^\\top_n$",
    }
    VARIABLE_LIST = [
        ["h", "x"],
        ["h", "z_0"],
        ["h", "x", "z_0"],
        ["h", "z_0", "z_1"],  # should achieve tightness here
    ]

    def __init__(
        self, n_landmarks, d, level="no", param_level="no", variable_list=None
    ):
        self.y_ = None
        self.n_landmarks = n_landmarks
        assert self.M_matrix is not None, "Inheriting class must initialize M_matrix."
        super().__init__(
            d=d, level=level, param_level=param_level, variable_list=variable_list
        )

    def get_all_variables(self):
        return [["h", "x"] + [f"z_{i}" for i in range(self.n_landmarks)]]

    def get_level_dims(self, n=1):
        """
        :param n: number of landmarks to consider
        """
        return {
            "no": 0,
            "u@u": n,  # ...
            "u2": n * self.d,
            "u@r": n,
            "uuT": n * self.d**2,
            "urT": n * self.d**2,
            "uxT": n * (self.d * (self.d + self.d**2)),
        }

    def generate_random_landmarks(self, theta=None):
        if theta is not None:
            C, r = get_C_r_from_theta(theta, self.d)
            if self.d == 3:
                # sample left u, v coordinates in left image, and compute landmark coordinates from that.
                fu, cu, b = self.M_matrix[0, [0, 2, 3]]
                fv, cv = self.M_matrix[1, [1, 2]]
                u = np.random.uniform(0, cu * 2, self.n_landmarks)
                v = np.random.uniform(0, cv * 2, self.n_landmarks)
                z = np.random.uniform(0, 5, self.n_landmarks)
                x = 1 / fu * (z * (u - cu) - b)
                y = 1 / fv * z * (v - cv)
                points_cam = np.c_[x, y, z]  # N x 3
            else:
                # sample left u in left image, and compute landmark coordinates from that.
                fu, cu, b = self.M_matrix[0, :]
                u = np.random.uniform(0, cu * 2, self.n_landmarks)
                y = np.random.uniform(1, 5, self.n_landmarks)
                x = 1 / fu * (y * (u - cu) - b)
                points_cam = np.c_[x, y]
            # transform points from  camera to world
            return (C.T @ (points_cam.T - r[:, None])).T
        else:
            return np.random.rand(self.n_landmarks, self.d)

    def generate_random_setup(self):
        self.landmarks = self.generate_random_landmarks(theta=self.theta)
        self.parameters = np.r_[1.0, self.landmarks.flatten()]

    def generate_random_theta(self):
        from utils.geometry import generate_random_pose

        return generate_random_pose(d=self.d)

    def get_parameters(self, var_subset=None):
        return self.extract_parameters(var_subset, self.landmarks)

    def get_x(self, theta=None, parameters=None, var_subset=None):
        """
        :param var_subset: list of variables to include in x vector. Set to None for all.
        """
        if theta is None:
            theta = self.theta
        if parameters is None:
            parameters = self.parameters
        if var_subset is None:
            var_subset = self.var_dict.keys()

        # TODO(FD) below is a bit hacky, these two variables should not both be called theta.
        # theta is either (x, y, alpha) or (x, y, z, a1, a2, a3)
        C, r = get_C_r_from_theta(theta, self.d)
        if (self.param_level != "no") and (len(parameters) > 1):
            landmarks = np.array(parameters[1:]).reshape((self.n_landmarks, self.d))
        else:
            landmarks = self.landmarks

        x_data = []
        for key in var_subset:
            if key == "h":
                x_data.append(1.0)
            elif key == "x":
                x_data += list(r) + list(C.flatten("C"))  # row-wise flatten
            elif "z" in key:
                j = int(key.split("_")[-1])

                pj = landmarks[j, :]

                zj = C[self.d - 1, :] @ pj + r[self.d - 1]
                u = 1 / zj * np.r_[C[: self.d - 1, :] @ pj + r[: self.d - 1], 1]
                x_data += list(u)

                if self.level == "no":
                    continue
                elif self.level == "u2":
                    x_data += list(u**2)
                elif self.level == "u@u":
                    x_data += [u @ u]
                elif self.level == "u@r":
                    x_data += [u @ r]
                elif self.level == "uuT":
                    x_data += list(np.outer(u, u).flatten())
                elif self.level == "urT":
                    # this works
                    x_data += list(np.outer(u, r).flatten())
                elif self.level == "uxT":
                    x = np.r_[r, C.flatten("C")]
                    x_data += list(np.outer(u, x).flatten())
        dim_x = self.get_dim_x(var_subset=var_subset)
        assert len(x_data) == dim_x
        return np.array(x_data)

    def get_A_known(self, var_dict=None, output_poly=False):
        """
        T = |   cx'   tx |
            |   cy'   ty |
            |   cz'   tz |
            | 0  0  0  1 |
        Let pj be the j-th landmark coordinate.
         [xj]   [cx @ pj + tx]
         [yj] = [cy @ pj + ty]
         [zj]   [cz @ pj + tz]

        Let u be the substitution variable, which has d-1 elements.
        Then we want to enforce that:
        u_xj = 1/zj * xj -> u_xj * zj = xj  ->  (cz @ pj + tz) * u_xj - (cx @ pj + tx) = 0
        u_yj = 1/zj * yj -> u_yj * zj = yj  -> same as above
        u_zj = 1/zj -> u_zj * zj = 1  -> u_zj * (cz @ pj + tz) -1 = 0
        Writing things as homogeneous constraints:
        a1) cz @ pj * u_xj + tz*u_xj - cx @ pj - h * tx = 0
        a2) -----1x-------   --2x---   -- 3 --   --4---
        a3) cz @ pj * u_zj + tz*u_zj  - h*h = 0
           ------1z-------   --2z---
        """
        # x contains: [c1, c2, c3, t]
        # z contains: [u_xj, u_yj, u_zj, H.O.T.]
        if self.d == 2:
            x = self.get_x()
            _, tx, tz, cx1, cx2, cz1, cz2, u_xj, u_zj, *_ = x
            cz = np.array([cz1, cz2])
            cx = np.array([cx1, cx2])
            pj = self.landmarks[0]
            assert abs(cz @ pj * u_xj + tz * u_xj - cx @ pj - tx) < 1e-10
            assert abs(u_zj * cz @ pj + u_zj * tz - 1) < 1e-10
        elif self.d == 3:
            x = self.get_x()
            # fmt: off
            (_, tx, ty, tz, cx1, cx2, cx3, cy1, cy2, cy3, cz1, cz2, cz3, u_x1, u_y1, u_z1, *_) = x
            # fmt: on
            p1 = self.landmarks[0]
            assert (
                abs(u_z1 * (cx1 * p1[0] + cx2 * p1[1] + cx3 * p1[2]) + u_z1 * tx - u_x1)
                < 1e-10
            )
            assert (
                abs(u_z1 * (cy1 * p1[0] + cy2 * p1[1] + cy3 * p1[2]) + u_z1 * ty - u_y1)
                < 1e-10
            )
            assert (
                abs(u_z1 * (cz1 * p1[0] + cz2 * p1[1] + cz3 * p1[2]) + u_z1 * tz - 1)
                < 1e-10
            )

        if var_dict is None:
            var_dict = self.var_dict

        print("Not using known stereo templates because they depend on the landmarks.")
        return []

        A_known = []
        z_dim = self.get_level_dims()[self.level]

        if "x" not in var_dict or "h" not in var_dict:
            return A_known
        landmarks = [j for j in range(self.n_landmarks) if f"z_{j}" in var_dict]
        for j in landmarks:
            # one complete constraint has x, z_j and h.
            pj = self.landmarks[j]
            for i in range(self.d):
                A = PolyMatrix()
                #     -----1i-------   --2i---   -- 3 --   --4---
                # a1) cz @ pj * u_xj + tz*u_xj - cx @ pj - h * tx = 0
                # a2) cz @ pj * u_yj + tz*u_yj - cy @ pj - h * ty = 0
                # a3) cz @ pj * u_zj + tz*u_zj  - h*h = 0
                #     ------1i-------   --2i---
                # --- 1i ---
                fill_mat = np.zeros((self.d + self.d**2, self.d + z_dim))
                # chooses cz of x, and u_xj, u_yj or u_zj of z
                fill_mat[-self.d :, i] = pj

                # --- 2 --- u_zj * tx
                # chooses tz of x, and u_ij of z
                fill_mat[self.d - 1, i] = 1.0
                A[f"x", f"z_{j}"] = fill_mat

                if i < self.d - 1:  # u, (v)
                    fill_mat = np.zeros((self.d + self.d**2, 1))
                    # chooses ci of x
                    fill_mat[(i + 1) * self.d : (i + 2) * self.d, 0] = -pj

                    # chooses ti of x
                    fill_mat[i, 0] = -1
                    A["x", "h"] = fill_mat
                elif i == self.d - 1:  # z
                    A["h", "h"] = -0  # 2.0
                if output_poly:
                    A_known.append(A)
                else:
                    A_known.append(A.get_matrix(var_dict))
        self.test_constraints(A_known)
        return A_known

    def sample_theta(self):
        return self.generate_random_theta().flatten()

    def sample_parameters(self, theta=None):
        if self.param_level == "no":
            return [1.0]
        else:
            parameters = self.generate_random_landmarks(theta=theta).flatten()
            return np.r_[1.0, parameters]

    def simulate_y(self, noise: float = None):
        if noise is None:
            noise = NOISE

        T = get_T(theta=self.theta, d=self.d)

        y_sim = np.zeros((self.n_landmarks, self.M_matrix.shape[0]))
        for j in range(self.n_landmarks):
            y_gt = T @ np.r_[self.landmarks[j], 1.0]

            # in 2d: y_gt[1]
            # in 3d: y_gt[2]
            y_gt /= y_gt[self.d - 1]
            y_gt = self.M_matrix @ y_gt
            y_sim[j, :] = y_gt + np.random.normal(loc=0, scale=noise, size=len(y_gt))
        return y_sim

    def get_Q(
        self, noise: float = None, output_poly: bool = False, use_cliques: list = []
    ) -> tuple:
        if self.y_ is None:
            if noise is None:
                noise = NOISE
            self.y_ = self.simulate_y(noise=noise)

        Q = self.get_Q_from_y(self.y_, output_poly=output_poly, use_cliques=use_cliques)
        return Q, self.y_

    def get_Q_from_y(self, y, output_poly=False, use_cliques=[]):
        """
        The least squares problem reads
        min_T sum_{n=0}^{N-1} || y - Mtilde@z ||
        where the first d elements of z correspond to u, and Mtilde contains the first d-1 and last element of M
        Mtilde is thus of shape d*2 by dim_z, where dim_z=d+dL (the additional Lasserre variables)
        y is of length d*2, corresponding to the measured pixel values in left and right image.
        """
        from poly_matrix.least_squares_problem import LeastSquaresProblem

        if len(use_cliques):
            js = use_cliques
        else:
            js = range(y.shape[0])

        # in 2d: M_tilde is 2 by 6, with first 2 columns: M[:, [0, 2]]
        # in 3d: M_tilde is 4 by 12, with first 3 columns: M[:, [0, 1, 3]]
        M_tilde = np.zeros((len(y[0]), self.var_dict["z_0"]))  # 4 by dim_z
        M_tilde[:, : self.d] = self.M_matrix[:, list(range(self.d - 1)) + [self.d]]

        # in 2d: M[:, 1]
        # in 3d: M[:, 2]
        m = self.M_matrix[:, self.d - 1]

        ls_problem = LeastSquaresProblem()
        for j in js:
            ls_problem.add_residual({"h": (y[j] - m), f"z_{j}": -M_tilde})

        if output_poly:
            Q = ls_problem.get_Q()
        else:
            Q = ls_problem.get_Q().get_matrix(self.var_dict)
        if NORMALIZE:
            Q /= self.n_landmarks * self.d

        # sanity check
        x = self.get_x()

        # sanity checks. Below is the best conditioned because we don't have to compute B.T @ B, which
        # can contain very large values.
        B = ls_problem.get_B_matrix(self.var_dict)
        errors = B @ x
        cost_test = errors.T @ errors
        if NORMALIZE:
            cost_test /= self.n_landmarks * self.d

        if output_poly:
            cost_Q = x.T @ Q.get_matrix(self.var_dict) @ x
        else:
            cost_Q = x.T @ Q @ x
        assert abs(cost_test - cost_Q) < 1e-6, (cost_test, cost_Q)
        if not len(use_cliques):
            cost_raw = self.get_cost(self.theta, y)
            assert abs(cost_test - cost_raw) < 1e-6, (cost_test, cost_raw)
        return Q

    def get_theta(self, x):
        return x[1 : 1 + self.d + self.d**2]

    def get_vec_around_gt(self, delta: float = 0):
        if delta == 0:
            return self.theta

        C, r = get_C_r_from_theta(self.theta, self.d)
        if self.d == 2:
            return super().get_vec_around_gt(delta=delta)
        else:
            return get_noisy_pose(C, r, delta)

    def get_C_cw(self, theta=None):
        C_cw, __ = get_C_r_from_theta(theta, self.d)
        return C_cw

    def get_position(self, theta=None):
        C_cw, r_wc_c = get_C_r_from_theta(theta, self.d)
        return (-C_cw.T @ r_wc_c)[None, :]

    def get_error(self, theta_hat):
        return get_pose_errors_from_theta(theta_hat, self.theta, self.d)

    def local_solver_manopt(self, t0, y, W=None, verbose=False, method="CG", **kwargs):
        import pymanopt
        from pymanopt.manifolds import Euclidean, Product, SpecialOrthogonalGroup

        if method == "CG":
            from pymanopt.optimizers import ConjugateGradient as Optimizer  # fastest
        elif method == "SD":
            from pymanopt.optimizers import SteepestDescent as Optimizer  # slow
        elif method == "TR":
            from pymanopt.optimizers import TrustRegions as Optimizer  # okay
        else:
            raise ValueError(method)

        solver_kwargs = SOLVER_KWARGS
        solver_kwargs.update(kwargs)

        if verbose:
            solver_kwargs["verbosity"] = 2
        else:
            solver_kwargs["verbosity"] = 1

        manifold = Product((SpecialOrthogonalGroup(self.d, k=1), Euclidean(self.d)))

        if W is None:
            W = np.eye(4) if self.d == 3 else np.eye(2)

        @pymanopt.function.autograd(manifold)
        def cost(R, t):
            cost = 0
            for i in range(self.n_landmarks):
                pi_cam = np.concatenate([R @ self.landmarks[i] + t, [1]], axis=0)
                y_gt = self.M_matrix @ (pi_cam / pi_cam[self.d - 1])
                residual = y[i] - y_gt
                cost += residual.T @ W @ residual
            if NORMALIZE:
                return cost / (self.n_landmarks * self.d)
            return cost

        euclidean_gradient = None  # set to None
        problem = pymanopt.Problem(
            manifold, cost, euclidean_gradient=euclidean_gradient  #
        )
        optimizer = Optimizer(**solver_kwargs)

        R_0, t_0 = get_C_r_from_theta(t0[: self.d + self.d**2], self.d)
        res = optimizer.run(problem, initial_point=(R_0, t_0))
        R, t = res.point

        theta_hat = get_theta_from_C_r(R, t)
        return theta_hat, res.stopping_criterion, res.cost

    def __repr__(self):
        level_str = str(self.level).replace(".", "-")
        return f"stereo{self.d}d_{level_str}_{self.param_level}"
