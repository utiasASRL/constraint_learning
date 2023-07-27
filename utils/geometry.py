import numpy as np


def generate_random_pose(d=2):
    if d == 2:
        n_angles = 1
    elif d == 3:
        n_angles = 3
    else:
        raise ValueError("d has to be 2 or 3.")
    return np.r_[np.random.rand(d), np.random.rand(n_angles) * 2 * np.pi]


def get_rot_matrix(rot):
    from scipy.spatial.transform import Rotation as R

    """ return the desired parameterization """
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
    C = xtheta[d:].reshape((d, d)).T
    return C, r


def get_xtheta_from_C_r(C, r):
    # column-wise flatten
    return np.r_[r, C.flatten("F")]


def get_T(xtheta=None, d=None, theta=None):
    if theta is not None:
        C, r = get_C_r_from_theta(theta, d)
    else:
        C, r = get_C_r_from_xtheta(xtheta, d)
    T = np.zeros((d + 1, d + 1))
    T[:d, :d] = C
    T[:d, d] = r
    T[-1, -1] = 1.0
    return T


def get_xtheta_from_theta(theta, d):
    n_rot = d * (d - 1) // 2
    pos = theta[:d]
    alpha = theta[d : d + n_rot]
    C = get_rot_matrix(alpha)
    c = C.flatten("F")  # col-wise flatten
    theta = np.r_[pos, c]
    return theta


def get_xtheta_from_T(T):
    # T is either 4x4 or 3x3 matrix.
    C = T[:-1, :-1]
    r = T[:-1, -1]
    return np.r_[r, C.flatten("F")]
