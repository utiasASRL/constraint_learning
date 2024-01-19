import numpy as np
from scipy.spatial.transform import Rotation as R


def generate_random_pose(d=2, size=1):
    if d == 2:
        n_angles = 1
    elif d == 3:
        n_angles = 3
    else:
        raise ValueError("d has to be 2 or 3.")
    trans = np.random.rand(d) * size - size / 2
    rot = np.random.rand(n_angles) * 2 * np.pi
    return np.r_[trans, rot]


def get_euler(C):
    r = R.from_matrix(C)
    return r.as_euler("xyz")


def get_rot_matrix(rot):
    """return the desired parameterization"""
    if np.ndim(rot) == 0 or len(rot) == 1:
        if np.ndim(rot) == 1:
            rot = float(rot[0])
        r = R.from_euler("z", rot)
        return r.as_matrix()[:2, :2]
    elif len(rot) == 3:
        r = R.from_euler("xyz", rot)
        return r.as_matrix()
    else:
        raise ValueError(rot)


def get_C_r_from_theta(theta, d):
    r = theta[:d]
    alpha = theta[d:]
    C = get_rot_matrix(alpha)
    return C, r


def get_C_r_from_xtheta(xtheta, d):
    r = xtheta[:d]
    C = xtheta[d:].reshape((d, d))
    return C, r


def get_xtheta_from_C_r(C, r):
    # row-wise flatten
    return np.r_[r, C.flatten("C")]


def get_T(xtheta=None, d=None, theta=None):
    if theta is not None:
        C_cw, r_wc_c = get_C_r_from_theta(theta, d)
    else:
        C_cw, r_wc_c = get_C_r_from_xtheta(xtheta, d)
    T = np.zeros((d + 1, d + 1))
    T[:d, :d] = C_cw
    T[:d, d] = r_wc_c
    T[-1, -1] = 1.0
    return T


def get_xtheta_from_theta(theta, d):
    n_rot = d * (d - 1) // 2
    pos = theta[:d]
    alpha = theta[d : d + n_rot]
    C = get_rot_matrix(alpha)
    c = C.flatten("C")  # row-wise flatten
    xtheta = np.r_[pos, c]
    return xtheta


def get_theta_from_xtheta(xtheta, d):
    from utils.geometry import get_euler

    pos = xtheta[1 : d + 1]
    c = xtheta[d + 1 : d + 1 + d**2]
    C = c.reshape(d, d)

    alpha = get_euler(C)
    theta = np.r_[pos, alpha]
    return theta


def get_xtheta_from_T(T):
    # T is either 4x4 or 3x3 matrix.
    C = T[:-1, :-1]
    r = T[:-1, -1]
    return np.r_[r, C.flatten("C")]


def get_pose_errors_from_xtheta(xtheta_hat, xtheta_gt, d):
    C_hat, r_hat = get_C_r_from_xtheta(xtheta_hat, d)
    C_gt, r_gt = get_C_r_from_xtheta(xtheta_gt, d)
    r_error = np.linalg.norm(r_hat - r_gt)
    C_error = np.linalg.norm(C_gt.T @ C_hat - np.eye(d))
    return {
        "error": r_error + C_error,
        "r error": r_error,
        "C error": C_error,
        "total error": r_error + C_error,
    }
