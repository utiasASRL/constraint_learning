import pickle

import autograd.numpy as np

from lifters.stereo3d_problem import _cost, local_solver
from lifters.stereo_lifter import NORMALIZE, StereoLifter
from utils.geometry import get_T, get_theta_from_T


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
            n_landmarks=n_landmarks,
            level=level,
            param_level=param_level,
            d=3,
            variable_list=variable_list,
        )

    @staticmethod
    def from_file(fname):
        with open(fname, "rb") as f:
            y_ = pickle.load(f)
            landmarks = pickle.load(f)
            theta = pickle.load(f)

            level = pickle.load(f)
            param_level = pickle.load(f)
            variable_list = pickle.load(f)
        lifter = Stereo3DLifter(
            n_landmarks=landmarks.shape[0],
            level=level,
            param_level=param_level,
            variable_list=variable_list,
        )
        lifter.y_ = y_
        lifter.landmarks = landmarks
        lifter.parameters = np.r_[1, landmarks.flatten()]
        lifter.theta = theta
        return lifter

    def to_file(self, fname):
        with open(fname, "wb") as f:
            pickle.dump(self.y_, f)
            pickle.dump(self.landmarks, f)

            pickle.dump(self.theta, f)
            pickle.dump(self.level, f)
            pickle.dump(self.param_level, f)
            pickle.dump(self.variable_list, f)

    def get_cost(self, t, y, W=None):
        """
        :param t: can be either
        - x, y, z, yaw, pitch roll: vector of unknowns, or
        - [c1, c2, c3, x, y, z], the theta vector (flattened C and x, y, z)
        """

        if W is None:
            W = self.W
        a = self.landmarks

        p_w, y = change_dimensions(a, y)

        T = get_T(theta=t, d=3)

        cost = _cost(p_w=p_w, y=y, T=T, M=self.M_matrix, W=W)
        if NORMALIZE:
            return cost / (self.n_landmarks * self.d)
        else:
            return cost

    def local_solver(self, t_init, y, W=None, verbose=False, **kwargs):
        """
        :param t_init: same options  asfor t in cost.
        """

        if W is None:
            W = self.W

        a = self.landmarks
        p_w, y = change_dimensions(a, y)
        T_init = get_T(theta=t_init, d=3)

        info, T_hat, cost = local_solver(
            T_init=T_init,
            y=y,
            p_w=p_w,
            W=W,
            M=self.M_matrix,
            log=False,
            min_update_norm=-1,  # disabled this stopping criterion
        )

        if verbose:
            print("Stereo3D local solver:", info["msg"])

        if NORMALIZE:
            cost /= self.n_landmarks * self.d

        x_hat = get_theta_from_T(T_hat)
        x = self.get_x(theta=x_hat)
        Q = self.get_Q_from_y(y[:, :, 0])
        cost_Q = x.T @ Q @ x
        if abs(cost) > 1e-10:
            if not (abs(cost_Q - cost) / cost < 1e-8):
                print(f"Warning, cost not equal {cost_Q:.2e} {cost:.2e}")

        if info["success"]:
            return x_hat, info, cost
        else:
            return None, info, cost


if __name__ == "__main__":
    lifter = Stereo3DLifter(n_landmarks=4)
