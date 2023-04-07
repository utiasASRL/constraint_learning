import numpy as np

from lifters.stereo2d_problem import M
from lifters.stereo_lifter import StereoLifter


def change_dimensions(a, y, x):
    p_w = np.concatenate([a, np.ones((a.shape[0], 1))], axis=1)
    y_mat = np.c_[[*y]]  # N x 2
    return p_w[:, :, None], y_mat[:, :, None], x[:, None]


class Stereo2DLifter(StereoLifter):
    def __init__(self, n_landmarks, level="no"):
        # TODO(FD) W is inconsistent with 3D model
        self.W = np.stack([np.eye(2)] * n_landmarks)
        super().__init__(n_landmarks=n_landmarks, d=2, level=level)

    def get_Q_old(self, noise: float = 1e-3) -> tuple:
        assert self.d == 2
        from poly_matrix.poly_matrix import PolyMatrix

        T = self.get_T()

        y = []
        for j in range(self.n_landmarks):
            y_gt = T @ np.r_[self.landmarks[j, :], 1.0]
            y_gt /= y_gt[1]
            y_gt = M @ y_gt
            y.append(y_gt + np.random.normal(loc=0, scale=noise))

        M_tilde = M[:, [0, 2]]

        Q = PolyMatrix()
        M_tilde_sq = M_tilde.T @ M_tilde
        for j in range(len(y)):
            Q["l", "l"] += np.linalg.norm(y[j] - M[:, 1]) ** 2
            Q[f"z{j}", "l"] += -(y[j] - M[:, 1]).T @ M_tilde
            Q[f"z{j}", f"z{j}"] += M_tilde_sq
            # Q[f"y{j}", "l"] = 0.5 * np.diag(M_tilde_sq)
        return Q.get_matrix(self.var_dict), y

    def get_Q(self, noise: float = 1e-3) -> tuple:
        return self._get_Q(noise=noise, M=M)

    @staticmethod
    def get_inits(n_inits):
        return np.c_[
            np.random.rand(n_inits),
            np.random.rand(n_inits),
            2 * np.pi * np.random.rand(n_inits),
        ]

    def get_cost(self, t, y, W=None):
        from lifters.stereo2d_problem import _cost

        a = self.landmarks

        p_w, y, phi = change_dimensions(a, y, t)
        cost = _cost(p_w=p_w, y=y, phi=phi, W=W)[0, 0]
        return cost

    def local_solver(self, t_init, y, W=None, verbose=False):
        from lifters.stereo2d_problem import local_solver

        a = self.landmarks

        p_w, y, init_phi = change_dimensions(a, y, t_init)
        success, phi_hat, cost = local_solver(
            p_w=p_w, y=y, W=W, init_phi=init_phi, log=verbose
        )
        if success:
            return phi_hat.flatten(), "converged", cost
        else:
            return None, "didn't converge", cost


if __name__ == "__main__":
    lifter = Stereo2DLifter(n_landmarks=3)
    lifter.run()
