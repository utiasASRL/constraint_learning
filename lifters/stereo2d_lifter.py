import numpy as np

from lifters.stereo_lifter import StereoLifter


def change_dimensions(a, y, x):
    p_w = np.concatenate([a, np.ones((a.shape[0], 1))], axis=1)
    y_mat = np.c_[[*y]]  # N x 2
    return p_w[:, :, None], y_mat[:, :, None], x[:, None]


class Stereo2DLifter(StereoLifter):
    def __init__(self, n_landmarks, level=0):
        super().__init__(n_landmarks=n_landmarks, d=2, level=level)

    def _get_Q(self, M: np.ndarray, noise: float = 1e-3) -> tuple:
        from poly_matrix.poly_matrix import PolyMatrix

        T = self.get_T()

        y = []
        for j in range(self.n_landmarks):
            y_gt = T @ np.r_[self.landmarks[j], 1.0]

            # in 2d: y_gt[1]
            # in 3d: y_gt[2]
            y_gt /= y_gt[self.d - 1]
            y_gt = M @ y_gt
            y.append(y_gt + np.random.normal(loc=0, scale=noise))

        # in 2d: M[:, [0, 2]]
        # in 3d: M[:, [0, 1, 3]]
        M_tilde = M[:, list(range(self.d - 1)) + [self.d]]

        # in 2d: M[:, 1]
        # in 3d: M[:, 2]
        m = M[:, self.d - 1]

        Q = PolyMatrix()
        M_tilde_sq = M_tilde.T @ M_tilde
        for j in range(len(y)):
            Q["l", "l"] += np.linalg.norm(y[j] - m) ** 2
            Q[f"z{j}", "l"] += -(y[j] - m).T @ M_tilde
            Q[f"z{j}", f"z{j}"] += M_tilde_sq
            # Q[f"y{j}", "l"] = 0.5 * np.diag(M_tilde_sq)
        return Q.toarray(self.get_var_dict()), y

    def get_Q(self, noise: float = 1e-3) -> tuple:
        from lifters.stereo2d_problem import M as M_mat

        return self._get_Q(noise=noise, M=M_mat)

    def get_Q_old(self, noise: float = 1e-3) -> tuple:
        assert self.d == 2

        from poly_matrix.poly_matrix import PolyMatrix
        from lifters.stereo2d_problem import M as M_mat

        T = self.get_T()

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
    def get_cost(a, y, t):
        from lifters.stereo2d_problem import _cost

        p_w, y, phi = change_dimensions(a, y, t)
        cost = _cost(p_w=p_w, y=y, phi=phi, W=None)[0, 0]
        return cost

    @staticmethod
    def local_solver(a, y, t_init, verbose=False):
        from lifters.stereo2d_problem import local_solver

        p_w, y, init_phi = change_dimensions(a, y, t_init)
        success, phi_hat, cost = local_solver(
            p_w=p_w, y=y, W=None, init_phi=init_phi, log=verbose
        )
        if success:
            return phi_hat.flatten(), "converged"
        else:
            return None, "didn't converge"
