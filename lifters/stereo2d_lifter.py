import autograd.numpy as np

from lifters.stereo_lifter import StereoLifter, NORMALIZE
from utils.geometry import get_xtheta_from_theta


def change_dimensions(a, y, x):
    p_w = np.concatenate([a, np.ones((a.shape[0], 1))], axis=1)
    y_mat = np.c_[[*y]]  # N x 2
    return p_w[:, :, None], y_mat[:, :, None], x[:, None]


class Stereo2DLifter(StereoLifter):
    def __init__(self, n_landmarks, level="no", param_level="no", variable_list=None):
        self.W = np.stack([np.eye(2)] * n_landmarks)

        f_u = 484.5
        c_u = 322
        b = 0.24
        self.M_matrix = np.array([[f_u, c_u, f_u * b / 2], [f_u, c_u, -f_u * b / 2]])

        super().__init__(
            n_landmarks=n_landmarks,
            level=level,
            param_level=param_level,
            d=2,
            variable_list=variable_list,
        )

    def get_cost(self, t, y, W=None):
        from lifters.stereo2d_problem import _cost

        if W is None:
            W = self.W
        a = self.landmarks

        p_w, y, phi = change_dimensions(a, y, t)
        cost = _cost(phi, p_w, y, W, self.M_matrix)
        if NORMALIZE:
            return cost / (self.n_landmarks * self.d)
        else:
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
        xtheta_hat = get_xtheta_from_theta(phi_hat, self.d)
        # cost /= self.n_landmarks * self.d
        info = {"success": success, "msg": "converged"}
        if success:
            return xtheta_hat, info, cost
        else:
            return None, info, cost


if __name__ == "__main__":
    lifter = Stereo2DLifter(n_landmarks=3)
