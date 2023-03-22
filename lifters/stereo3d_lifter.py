from thesis.solvers import local_solver

import numpy as np

from utils import get_rot_matrix
from lifters.stereo_lifter import StereoLifter
from lifters.stereo3d_problem import M as M_mat


def change_dimensions(a, y, x):
    p_w = np.concatenate([a, np.ones((a.shape[0], 1))], axis=1)
    y_mat = np.c_[[*y]]  # N x 2
    return p_w[:, :, None], y_mat[:, :, None], x[:, None]


class Stereo3DLifter(StereoLifter):
    def __init__(self, n_landmarks, level=0):
        self.W = np.eye(4)
        super().__init__(n_landmarks=n_landmarks, level=level, d=3)

    def get_Q(self, noise: float = 1e-3) -> tuple:
        return self._get_Q(noise=noise, M=M_mat)

    @staticmethod
    def get_inits(n_inits):
        return np.c_[np.random.rand(n_inits, 3), 2 * np.pi * np.random.rand(n_inits, 3)]

    @staticmethod
    def get_cost(a, y, t, W):
        from lifters.stereo_lifter import get_T, get_theta_from_unknowns
        from lifters.stereo3d_problem import M as M_mat
        from thesis.solvers.local_solver import projection_error

        p_w, y, __ = change_dimensions(a, y, t)

        theta = get_theta_from_unknowns(t, 3)
        T = get_T(theta, 3)
        cost = projection_error(p_w=p_w, y=y, T=T, M=M_mat, W=W)
        return cost

    @staticmethod
    def local_solver(a, y, t_init, W, verbose=False):
        from lifters.stereo_lifter import get_T, get_theta_from_unknowns
        from lifters.stereo3d_problem import M as M_mat
        from thesis.solvers.local_solver import _stereo_localization_gauss_newton

        p_w, y, __ = change_dimensions(a, y, t_init)
        theta_init = get_theta_from_unknowns(t_init, 3)
        T_init = get_T(theta_init, 3)

        solution = _stereo_localization_gauss_newton(
            T_init=T_init, y=y, p_w=p_w, W=W, M=M_mat, log=verbose
        )
        success = solution.solved
        T_hat = solution.T_cw
        # cost = solution.cost

        # should have the same dimension as self.unknowns
        from pylgmath.se3.operations import tran2vec

        t_hat = tran2vec(T_hat)

        if success:
            return t_hat.flatten(), "converged"
        else:
            return None, "didn't converge"
