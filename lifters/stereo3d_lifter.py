import autograd.numpy as np

from lifters.stereo_lifter import StereoLifter
from utils.geometry import (get_T,
    get_xtheta_from_T,
    get_xtheta_from_theta,
)


def change_dimensions(a, y):
    p_w = np.concatenate([a, np.ones((a.shape[0], 1))], axis=1)
    y_mat = np.c_[[*y]]  # N x 2
    return p_w[:, :, None], y_mat[:, :, None]


class Stereo3DLifter(StereoLifter):
    def __init__(self, n_landmarks, level="no", param_level="no", variable_list=None):
        self.W = np.stack([np.eye(4)] * n_landmarks)

        f_u = 484.5
        f_v = 484.5
        c_u = 322
        c_v = 247
        b = 0.24
        self.M_matrix = np.array(
            [
                [f_u, 0, c_u, f_u * b / 2],
                [0, f_v, c_v, 0],
                [f_u, 0, c_u, -f_u * b / 2],
                [0, f_v, c_v, 0],
            ]
        )
        super().__init__(
            n_landmarks=n_landmarks, level=level, param_level=param_level, d=3, variable_list=variable_list
        )


    def get_vec_around_gt(self, delta):
        t0 = super().get_vec_around_gt(delta)
        return get_xtheta_from_theta(t0, 3)

    def get_cost(self, t, y, W=None):
        """
        :param t: can be either
        - x, y, z, yaw, pitch roll: vector of unknowns, or
        - [c1, c2, c3, x, y, z], the theta vector (flattened C and x, y, z)
        """
        from lifters.stereo3d_problem import _cost

        if W is None:
            W = self.W
        a = self.landmarks

        p_w, y = change_dimensions(a, y)

        if len(t) == 6:
            xtheta = get_xtheta_from_theta(t, 3)
        else:
            xtheta = t
        T = get_T(xtheta, 3)

        cost = _cost(p_w=p_w, y=y, T=T, M=self.M_matrix, W=W)
        return cost / (self.n_landmarks * self.d)

    def local_solver(self, t_init, y, W=None, verbose=False, **kwargs):
        """
        :param t_init: same options  asfor t in cost.
        """
        from lifters.stereo3d_problem import local_solver

        if W is None:
            W = self.W

        a = self.landmarks
        p_w, y = change_dimensions(a, y)
        if len(t_init) == 6:
            xtheta_init = get_xtheta_from_theta(t_init, 3)
        else:
            xtheta_init = t_init
        T_init = get_T(xtheta_init, 3)

        success, T_hat, cost = local_solver(
            T_init=T_init, y=y, p_w=p_w, W=W, M=self.M_matrix, log=verbose, min_update_norm=1e-10
        )
        x_hat = get_xtheta_from_T(T_hat)

        x = self.get_x(theta=x_hat)
        Q = self.get_Q_from_y(y[:, :, 0])
        cost_Q = x.T @ Q @ x
        if abs(cost) > 1e-10:
            assert abs(cost_Q - cost)/cost < 1e-8

        if success:
            return x_hat, "converged", cost
        else:
            return None, "didn't converge", cost


if __name__ == "__main__":
    lifter = Stereo3DLifter(n_landmarks=4)
    lifter.run()
