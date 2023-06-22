import numpy as np

from lifters.stereo_lifter import StereoLifter


def change_dimensions(a, y, x):
    p_w = np.concatenate([a, np.ones((a.shape[0], 1))], axis=1)
    y_mat = np.c_[[*y]]  # N x 2
    return p_w[:, :, None], y_mat[:, :, None], x[:, None]


class Stereo2DLifter(StereoLifter):
    def __init__(self, n_landmarks, level="no", param_level="no"):
        self.W = np.stack([np.eye(2)] * n_landmarks)
        self.M_matrix_ = None
        super().__init__(
            n_landmarks=n_landmarks, level=level, param_level=param_level, d=2
        )

    @property
    def M_matrix(self):
        if self.M_matrix_ is None:
            from lifters.stereo2d_problem import M as M_matrix

            self.M_matrix_ = M_matrix
        return self.M_matrix_

    def get_cost(self, t, y, W=None):
        from lifters.stereo2d_problem import _cost

        if W is None:
            W = self.W
        a = self.landmarks

        p_w, y, phi = change_dimensions(a, y, t)
        cost = _cost(phi, p_w, y, W, self.M_matrix)
        return cost

    def local_solver(self, t_init, y, W=None, verbose=False, **kwargs):
        from lifters.stereo2d_problem import local_solver

        if W is None:
            W = self.W
        a = self.landmarks

        p_w, y, init_phi = change_dimensions(a, y, t_init)
        success, phi_hat, cost = local_solver(
            p_w=p_w, y=y, W=W, init_phi=t_init, log=verbose
        )
        if success:
            return phi_hat.flatten(), "converged", cost
        else:
            return None, "didn't converge", cost


if __name__ == "__main__":
    lifter = Stereo2DLifter(n_landmarks=3)
    lifter.run()
