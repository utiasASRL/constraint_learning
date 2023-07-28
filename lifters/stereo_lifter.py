from abc import ABC, abstractproperty

import numpy as np

from lifters.state_lifter import StateLifter
from poly_matrix.poly_matrix import PolyMatrix
from utils.geometry import get_C_r_from_theta, get_C_r_from_xtheta, get_T, get_xtheta_from_theta

NOISE = 0.5 # 

class StereoLifter(StateLifter, ABC):
    """General lifter for stereo localization problem.

    Naming convention:
    - theta is the actual number of unknowns, so 6 in 3d or 3 in 2d.
    - xtheta is the vector [position, vec(C)], with C the rotation matrix
    """

    LEVELS = [
        "no",
        "u@u",  # ...
        "u2",
        "u@r",
        "uuT",
        "urT",
        "u1u2rT"
    ]
    PARAM_LEVELS = ["no", "p", "ppT"]
    LEVEL_NAMES = {
        "no": "$\\boldsymbol{u}_n$",
        "urT": "$\\boldsymbol{u}\\boldsymbol{t}^\\top_n$",
        "u1u2rT": "$\\boldsymbol{u}_1\\boldsymbol{u}_2\\boldsymbol{t}^\\top_n$",
    }
    VARIABLE_LIST = [
        ["l", "x"], 
        ["l", "z_0"], 
        ["l", "x", "z_0"],
        ["l", "z_0", "z_1"],
        ["l", "x", "z_0", "z_1"],
        ["l", "z_0", "z_1", "z_2"],
    ]
    def __init__(self, n_landmarks, d, level="no", param_level="no", variable_list=None):
        self.d = d
        self.n_landmarks = n_landmarks
        super().__init__(level=level, param_level=param_level, variable_list=variable_list)

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
            "u1u2rT": n * 2 * self.d**2,
        }

    @abstractproperty
    def M_matrix(self):
        return

    def get_inits(self, n_inits):
        n_angles = self.d * (self.d - 1) / 2
        return np.c_[
            np.random.rand(n_inits, self.d),
            2 * np.pi * np.random.rand(n_inits, n_angles),
        ]

    def generate_random_landmarks(self, theta=None):
        if theta is not None:
            C, r = get_C_r_from_theta(theta, self.d)
            if self.d == 3:
                # sample left u, v coordinates in left image, and compute landmark coordinates from that.
                fu, cu, b = self.M_matrix[0, [0, 2, 3]]
                fv, cv = self.M_matrix[1, [1, 2]]
                u = np.random.uniform(0,cu*2,self.n_landmarks)
                v = np.random.uniform(0,cv*2,self.n_landmarks)
                z = np.random.uniform(0, 5, self.n_landmarks) 
                x = 1/fu*(z*(u - cu) - b)
                y = 1/fv*z*(v - cv)
                points_cam = np.c_[x, y, z] # N x 3
            else:
                # sample left u in left image, and compute landmark coordinates from that.
                fu, cu, b = self.M_matrix[0, :]
                u = np.random.uniform(0,cu*2,self.n_landmarks)
                y = np.random.uniform(1, 5, self.n_landmarks) 
                x = 1/fu*(y*(u - cu) - b)
                points_cam = np.c_[x, y]
            # transform points from  camera to world
            return (C.T @ (points_cam.T - r[:, None])).T
        else:
            return np.random.rand(self.n_landmarks, self.d)

    def generate_random_setup(self):
        self.landmarks = self.generate_random_landmarks(theta=self.theta)
        self.parameters = np.r_[1.0, self.landmarks.flatten()]

    def generate_random_theta(self):
        from utils.common import generate_random_pose
        return generate_random_pose(d=self.d)

    def get_parameters(self, var_subset=None):
        return self.extract_parameters(self, var_subset, self.landmarks)

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
        if len(theta) in [3, 6]:
            C, r = get_C_r_from_theta(theta, self.d)
        # theta is (x, y, z, C.flatten()), technically this should be called xtheta!
        elif len(theta) == 12:
            C, r = get_C_r_from_xtheta(theta, self.d)

        if self.param_level != "no":
            landmarks = np.array(parameters[1:]).reshape((self.n_landmarks, self.d))
        else:
            landmarks = self.landmarks

        x_data = []
        for key in var_subset:
            if key == "l":
                x_data.append(1.0)
            elif key == "x":
                x_data += list(r) + list(C.flatten("F"))  # column-wise flatten
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
                elif self.level == "u1u2rT":
                    x_data += list(np.outer(u, r).flatten())
                    x_data += list(np.outer(u**2, r).flatten())

        dim_x = self.get_dim_x(var_subset=var_subset)
        assert len(x_data) == dim_x
        return np.array(x_data)

    def get_A_known(self, add_known_redundant=False):
        # C = [ c1
        #       c2
        #       c3 ]
        # [xj]   [c1 @ pj]
        # [yj] = [c2 @ pj]
        # [zj]   [c3 @ pj]
        # enforce that u_xj = 1/zj * xj -> zj*u_xj = c3 @ pj * u_xj = c1 @ pj
        # enforce that u_yj = 1/zj * yj -> zj*u_yj = c3 @ pj * u_yj = c2 @ pj
        print("Warning: get_A_known not adapted to new variables yet.")
        return []

        A_known = []
        for k in range(self.n_landmarks):
            for j in range(self.d):
                A = PolyMatrix()
                # x contains: [c1, c2, c3, t]
                fill_mat = np.zeros((self.d, self.d))
                fill_mat[:, j] = self.landmarks[k]
                A[f"c_{self.d-1}", f"z_{k}"] = fill_mat

                fill_mat = np.zeros((self.d, self.d))
                fill_mat[-1, j] = 1.0
                A[f"t", f"z_{k}"] = fill_mat

                if j < self.d - 1:  # u, (v)
                    A["l", f"c_{j}"] = -self.landmarks[k].reshape((1, -1))
                    fill_mat = -np.eye(self.d)[j]
                    A["l", "t"] = fill_mat.reshape((1, -1))
                elif j == self.d - 1:  # z
                    A["l", "l"] = -2.0
                A_known.append(A.get_matrix(self.var_dict))
        return A_known

    def sample_theta(self, factor=1.0):
        return self.generate_random_theta(factor=factor).flatten()

    def sample_parameters(self, theta=None):
        if self.param_level == "no":
            return [1.0]
        else:
            parameters = self.generate_random_landmarks(theta=theta).flatten()
            return np.r_[1.0, parameters]

    def get_Q(self, noise: float = None) -> tuple:
        if noise is None:
            noise = NOISE
        xtheta = get_xtheta_from_theta(self.theta, self.d)
        T = get_T(xtheta, self.d)

        y = []

        for j in range(self.n_landmarks):
            y_gt = T @ np.r_[self.landmarks[j], 1.0]

            # in 2d: y_gt[1]
            # in 3d: y_gt[2]
            y_gt /= y_gt[self.d - 1]
            y_gt = self.M_matrix @ y_gt
            y.append(y_gt + np.random.normal(loc=0, scale=noise, size=len(y_gt)))
        Q = self.get_Q_from_y(y)
        return Q, y

    def get_Q_from_y(self, y):
        """
        The least squares problem reads
        min_T \sum_{n=0}^{N-1} || y - Mtilde@z || 
        where the first d elements of z correspond to u, and Mtilde contains the first d-1 and last element of M
        Mtilde is thus of shape d*2 by dim_z, where dim_z=d+dL (the additional Lasserre variables)
        y is of length d*2, corresponding to the measured pixel values in left and right image.
        """
        from poly_matrix.least_squares_problem import LeastSquaresProblem

        # in 2d: M_tilde is 2 by 6, with first 2 columns: M[:, [0, 2]]
        # in 3d: M_tilde is 4 by 12, with first 3 columns: M[:, [0, 1, 3]]
        M_tilde = np.zeros((len(y[0]), self.var_dict["z_0"])) # 4 by dim_z
        M_tilde[:, :self.d] = self.M_matrix[:, list(range(self.d - 1)) + [self.d]]

        # in 2d: M[:, 1]
        # in 3d: M[:, 2]
        m = self.M_matrix[:, self.d - 1]

        ls_problem = LeastSquaresProblem()
        for j in range(len(y)):
            ls_problem.add_residual({"l": (y[j] - m), f"z_{j}": -M_tilde})

        Q = ls_problem.get_Q().get_matrix(self.var_dict)
        # there is precision loss because Q is 

        # sanity check
        x = self.get_x()

        # sanity checks. Below is the best conditioned because we don't have to compute B.T @ B, which 
        # can contain very large values. 
        B = ls_problem.get_B_matrix(self.var_dict)
        errors = B @ x
        cost_test = errors.T @ errors

        t = self.theta
        cost_raw = self.get_cost(t, y)
        cost_Q = x.T @ Q.toarray() @ x
        assert abs(cost_raw - cost_Q) < 1e-8, (cost_raw, cost_Q)
        assert abs(cost_raw - cost_test) < 1e-8, (cost_raw, cost_test)
        return Q

    def __repr__(self):
        level_str = str(self.level).replace(".", "-")
        return f"stereo{self.d}d_{level_str}_{self.param_level}"
