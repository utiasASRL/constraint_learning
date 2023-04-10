import numpy as np

from lifters.stereo_lifter import StereoLifter


def change_dimensions(a, y, x):
    p_w = np.concatenate([a, np.ones((a.shape[0], 1))], axis=1)
    y_mat = np.c_[[*y]]  # N x 2
    return p_w[:, :, None], y_mat[:, :, None], x[:, None]


class Stereo2DLifter(StereoLifter):
    def __init__(self, n_landmarks, level="no"):
        # TODO(FD) W is inconsistent with 3D model
        self.W = np.stack([np.eye(2)] * n_landmarks)
        self.M_matrix_ = None
        super().__init__(n_landmarks=n_landmarks, d=2, level=level)

    @property
    def M_matrix(self):
        if self.M_matrix_ is None:
            from lifters.stereo2d_problem import M as M_matrix

            self.M_matrix_ = M_matrix
        return self.M_matrix_

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
            p_w=p_w, y=y, W=W, init_phi=t_init, log=verbose
        )
        if success:
            return phi_hat.flatten(), "converged", cost
        else:
            return None, "didn't converge", cost


if __name__ == "__main__":
    lifter = Stereo2DLifter(n_landmarks=3)
    lifter.run()
