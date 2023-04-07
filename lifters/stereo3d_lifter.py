import numpy as np

from lifters.stereo3d_problem import M
from lifters.stereo_lifter import StereoLifter
from lifters.stereo_lifter import get_T, get_theta_from_T, get_xtheta_from_theta

def change_dimensions(a, y):
    p_w = np.concatenate([a, np.ones((a.shape[0], 1))], axis=1)
    y_mat = np.c_[[*y]]  # N x 2
    return p_w[:, :, None], y_mat[:, :, None]


class Stereo3DLifter(StereoLifter):
    def __init__(self, n_landmarks, level="no"):
        self.W = np.eye(4)
        super().__init__(n_landmarks=n_landmarks, level=level, d=3)

    def get_Q(self, noise: float = 1e-3) -> tuple:
        return self._get_Q(noise=noise, M=M)

    def get_vec_around_gt(self, delta):
        t_gt = self.theta
        t_0 = t_gt + np.random.normal(scale=delta, loc=0, size=len(t_gt))
        theta_0 = self.get_xtheta_from_theta(t_0)
        return theta_0

    @staticmethod
    def get_inits(n_inits):
        return np.c_[np.random.rand(n_inits, 3), 2 * np.pi * np.random.rand(n_inits, 3)]

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

        cost = projection_error(p_w=p_w, y=y, T=T, M=M, W=W)
        return cost

    def local_solver(self, t_init, y, W=None, verbose=False):
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
            T_init=T_init, y=y, p_w=p_w, W=W, M=M, log=verbose
        )
        x_hat = get_theta_from_T(T_hat)

        if success:
            return x_hat, "converged", cost
        else:
            return None, "didn't converge", cost


if __name__ == "__main__":
    lifter = Stereo3DLifter(n_landmarks=4)
    lifter.run()
