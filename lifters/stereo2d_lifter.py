import numpy as np

from utils import get_T, get_rot_matrix
from lifters.state_lifter import StateLifter

def change_dimensions(a, y, x):
    p_w = np.concatenate([a, np.ones((a.shape[0], 1))], axis=1)
    y_mat = np.c_[[*y]]  # N x 2
    return p_w[:, :, None], y_mat[:, :, None], x[:, None]

class Stereo2DLifter(StateLifter):
    def __init__(self, n_landmarks, level=0):
        self.level = level
        self.n_landmarks = n_landmarks
        self.d = 2
        M = self.n_landmarks * self.d
        if level == 1:
            M += self.n_landmarks * self.d
        elif level > 2:
            M += self.n_landmarks * self.d**2
        super().__init__(theta_shape=(6,), M=M)

    def generate_random_setup(self):
        # important!! we can only resample x, the landmarks have to stay the same
        # [(x, y, alpha), (landmarks)]
        self.landmarks = np.random.rand(self.n_landmarks, self.d)

    def generate_random_unknowns(self):
        self.unknowns = np.r_[np.random.rand(2), np.random.rand() * 2 * np.pi]

    def get_theta(self):
        x, y, alpha = self.unknowns
        C = get_rot_matrix(alpha)
        C = get_rot_matrix(alpha)
        return np.r_[C[:, 0], C[:, 1], x, y]

    def get_x(self, theta=None):
        if theta is None:
            theta = self.get_theta()
        elif len(theta) == 3:  # theta actually contains unknowns
            x, y, alpha = theta
            C = get_rot_matrix(alpha)
            r = np.r_[x, y]
            theta = np.r_[C[:, 0], C[:, 1], x, y]

        C = np.c_[theta[:2], theta[2:4]]
        r = theta[4:]
        x_data = [1] + list(theta)

        higher_data = []
        for j in range(self.n_landmarks):
            pj = self.landmarks[j, :]
            zj = C[1, :] @ pj + r[1]
            u = 1 / zj * np.r_[C[0, :] @ pj + r[0], 1]
            x_data += list(u)

            if self.level == 1:
                # this doesn't work
                #higher_data += list(r * u)
                #higher_data += list(u**2)
                higher_data += list(np.outer(u, r)[:, 0])
            elif self.level == 2:
                # this doesn't work
                higher_data += list(np.outer(u, u).flatten())
            elif self.level == 3:
                # this works
                higher_data += list(np.outer(u, r).flatten())
        x_data += higher_data
        return np.array(x_data)

    def get_var_dict(self):
        var_dict = {"l": 1}
        var_dict["x"] = 6
        var_dict.update({f"z{i}": 2 for i in range(self.n_landmarks)})
        if self.level == 1:
            var_dict.update({f"y{i}": 2 for i in range(self.n_landmarks)})
        elif self.level > 2:
            var_dict.update({f"y{i}": 4 for i in range(self.n_landmarks)})
        
        return var_dict

    def get_Q(self, noise: float = 1e-3) -> tuple:
        from poly_matrix.poly_matrix import PolyMatrix
        from lifters.stereo2d_problem import M as M_mat

        T = get_T(self.get_theta())

        y = []
        for j in range(self.n_landmarks):
            y_gt = T @ np.r_[self.landmarks[j, :], 1.0]
            y_gt /= y_gt[1]
            y_gt = M_mat @ y_gt
            y.append(y_gt + np.random.normal(loc=0, scale=noise))

        M_tilde = M_mat[:, [0, 2]]

        Q = PolyMatrix()
        M_tilde_sq = M_tilde.T @ M_tilde
        for j in range(len(y)):
            Q["l", "l"] += np.linalg.norm(y[j] - M_mat[:, 1]) ** 2
            Q[f"z{j}", "l"] += -(y[j] - M_mat[:, 1]).T @ M_tilde
            Q[f"z{j}", f"z{j}"] += M_tilde_sq
            # Q[f"y{j}", "l"] = 0.5 * np.diag(M_tilde_sq)
        return Q.toarray(self.get_var_dict()), y

    @staticmethod
    def get_inits(n_inits):
        return np.c_[
            np.random.rand(n_inits),
            np.random.rand(n_inits),
            2 * np.pi * np.random.rand(n_inits),
        ]

    @staticmethod
    def get_cost(a, y, x):
        from lifters.stereo2d_problem import _cost

        p_w, y, phi = change_dimensions(a, y, x)
        cost = _cost(p_w=p_w, y=y, phi=phi, W=None)[0, 0]
        return cost

    @staticmethod
    def local_solver(a, y, x_init, verbose=False):
        from lifters.stereo2d_problem import local_solver

        p_w, y, init_phi = change_dimensions(a, y, x_init)
        success, phi_hat, cost = local_solver(
            p_w=p_w, y=y, W=None, init_phi=init_phi, log=verbose
        )
        if success:
            return phi_hat.flatten(), "converged"
        else:
            return None, "didn't converge"