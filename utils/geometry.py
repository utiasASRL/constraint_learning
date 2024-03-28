""" Tools related to geometry.

Naming conventions:
- C and r are rotation matrix and translation. T is the transformation matrix.
- theta contains the translation and vectorzied rotation (i.e., r and vec(C))
"""

import numpy as np
from scipy.linalg import expm
from scipy.spatial.transform import Rotation as R


def generate_random_pose(d=2, size=1, use_euler=True):
    r = np.random.rand(d) * size - size / 2
    if d == 2:
        rot = np.random.rand() * 2 * np.pi
        C = R.from_euler("z", rot).as_matrix()[:2, :2]
    elif d == 3:
        # TODO(FD) sampling rotations as below does not ensure uniform sampling. However, it leads
        # to better results than actually sampling uniformly (using R.random() for example). Better
        # results here means that fewer of the resulting learned constraints are required for tightness.
        if use_euler:
            rot = np.random.rand(3) * 2 * np.pi
            C = R.from_euler("xyz", rot).as_matrix()
        else:
            C = R.random().as_matrix()
    else:
        raise ValueError("d has to be 2 or 3.")
    return get_theta_from_C_r(C, r)


def get_C_r_from_theta(theta, d):
    r = theta[:d]
    C = theta[d:].reshape((d, d)).T
    return C, r


def get_theta_from_C_r(C, r):
    # column-wise flatten
    return np.r_[r, C.flatten("F")]


def get_T(theta=None, d=None):
    C_cw, r_wc_c = get_C_r_from_theta(theta, d)
    T = np.zeros((d + 1, d + 1))
    T[:d, :d] = C_cw
    T[:d, d] = r_wc_c
    T[-1, -1] = 1.0
    return T


def get_theta_from_T(T):
    # T is either 4x4 or 3x3 matrix.
    C = T[:-1, :-1]
    r = T[:-1, -1]
    return get_theta_from_C_r(C, r)


def get_theta_from_C_r(C, r):
    return np.r_[r, C.flatten("F")]


def get_pose_errors_from_theta(theta_hat, theta_gt, d):
    C_hat, r_hat = get_C_r_from_theta(theta_hat, d)
    C_gt, r_gt = get_C_r_from_theta(theta_gt, d)
    r_error = np.linalg.norm(r_hat - r_gt)
    C_error = np.linalg.norm(C_gt.T @ C_hat - np.eye(d))
    return {
        "error": r_error + C_error,
        "r error": r_error,
        "C error": C_error,
        "total error": r_error + C_error,
    }


def get_noisy_pose(C, r, delta):
    def skew(x):
        return np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]])

    d = len(r)
    rot_eps = np.random.normal(scale=delta, loc=0, size=d)
    C_eps = expm(skew(rot_eps)) @ C
    r_eps = r + np.random.normal(scale=delta, loc=0, size=d)
    return get_theta_from_C_r(C_eps, r_eps)
