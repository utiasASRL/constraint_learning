import numpy as np

from lifters.stereo_lifter import (
    StereoLifter,
    get_T,
    get_xtheta_from_T,
    get_xtheta_from_theta,
)


def change_dimensions(a, y):
    p_w = np.concatenate([a, np.ones((a.shape[0], 1))], axis=1)
    y_mat = np.c_[[*y]]  # N x 2
    return p_w[:, :, None], y_mat[:, :, None]


class Stereo3DLifter(StereoLifter):
    def __init__(self, n_landmarks, level="no"):
        self.W = np.eye(4)
        self.M_matrix_ = None
        super().__init__(n_landmarks=n_landmarks, level=level, d=3)

    @property
    def M_matrix(self):
        if self.M_matrix_ is None:
            from lifters.stereo3d_problem import M as M_matrix

            self.M_matrix_ = M_matrix
        return self.M_matrix_

    def get_vec_around_gt(self, delta):
        t0 = super().get_vec_around_gt(delta)
        return get_xtheta_from_theta(t0, 3)

    def get_cost(self, t, y, W=None):
        """
        :param t: can be either
        - x, y, z, yaw, pitch roll: vector of unknowns, or
        - [c1, c2, c3, x, y, z], the theta vector (unrolled C and x, y, z)
        """
        from lifters.stereo3d_problem import projection_error

        if W is None:
            W = self.W
        a = self.landmarks
        p_w, y = change_dimensions(a, y)

        if len(t) == 6:
            theta = get_xtheta_from_theta(t, 3)
        else:
            theta = t
        T = get_T(theta, 3)

        cost = projection_error(p_w=p_w, y=y, T=T, M=self.M_matrix, W=W)
        return cost

    def local_solver(self, t_init, y, W=None, verbose=False, **kwargs):
        """
        :param t_init: same options  asfor t in cost.
        """
        from lifters.stereo3d_problem import stereo_localization_gauss_newton

        if W is None:
            W = self.W

        a = self.landmarks
        p_w, y = change_dimensions(a, y)
        if len(t_init) == 6:
            theta_init = get_xtheta_from_theta(t_init, 3)
        else:
            theta_init = t_init
        T_init = get_T(theta_init, 3)

        success, T_hat, cost = stereo_localization_gauss_newton(
            T_init=T_init, y=y, p_w=p_w, W=W, M=self.M_matrix, log=verbose
        )
        x_hat = get_xtheta_from_T(T_hat)

        if success:
            return x_hat, "converged", cost
        else:
            return None, "didn't converge", cost


if __name__ == "__main__":
    lifter = Stereo3DLifter(n_landmarks=4)
    lifter.run()
